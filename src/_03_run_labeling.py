from __future__ import annotations

import os
import sys
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed

# add project root to sys.path (so `from src...` works when running from repo root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src._02_llm_client import LabelingAgent
from src._01_config import SYSTEM_PROMPT, LABEL_COLS, COL_TO_CODE, CODE_TO_COL


def build_tfidf_retriever(promise_df: pd.DataFrame):
    texts = promise_df["RequirementText"].astype(str).tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
    X = vec.fit_transform(texts)
    return vec, X, texts


def get_example_label_dict(row: pd.Series) -> Dict[str, bool]:
    """Return labels in the SAME schema style as the model output: code -> bool."""
    out: Dict[str, bool] = {code: False for code in CODE_TO_COL.keys()}
    for col in LABEL_COLS:
        v = row.get(col, 0)
        try:
            v = int(v)
        except Exception:
            v = 0
        if v == 1:
            out[COL_TO_CODE[col]] = True
    return out


def retrieve_examples(
    requirement_text: str,
    promise_df: pd.DataFrame,
    vec,
    X_promise,
    promise_texts: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    q = vec.transform([requirement_text])
    sims = cosine_similarity(q, X_promise).ravel()
    idxs = sims.argsort()[::-1][:top_k]
    examples: List[Dict[str, Any]] = []
    for i in idxs:
        row = promise_df.iloc[int(i)]
        examples.append(
            {
                "text": str(promise_texts[int(i)])[:600],
                "labels": get_example_label_dict(row),
            }
        )
    return examples


def safe_write_csv(df: pd.DataFrame, path: Path) -> bool:
    """
    Write CSV safely on Windows where Excel/VSCode may lock the file.
    Returns True if write succeeded, False otherwise.
    """
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return True
    except PermissionError:
        print(
            f"⚠️ Permission denied while writing: {path} (file is locked). "
            f"Close Excel/VSCode preview and continue..."
        )
        return False


def _is_empty_error(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", default="data/Dataset_Full_EN.csv", help="Input .csv or .xlsx")
    ap.add_argument("--promise", default="data/PROMISE-relabeled-NICE.csv", help="PROMISE csv for optional TF-IDF examples")
    ap.add_argument("--output", default="outputs/Dataset_Full_EN_labeled.csv", help="Output CSV")

    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--reasoning_effort", default="low", choices=["none", "low", "medium", "high"])

    ap.add_argument("--overwrite", action="store_true", help="If set, re-label ALL rows (overwrite existing labels).")
    ap.add_argument("--top_k", type=int, default=0, help="TF-IDF retrieve K PROMISE examples per row (0 = off).")
    ap.add_argument("--save_every", type=int, default=25, help="Checkpoint frequency (#completed rows).")

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Label only first N rows from the top (0 = all). Example: --limit 5",
    )

    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of parallel requests (default=8). Too high may cause rate limits/timeouts.",
    )

    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(".progress.csv")

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # read input
    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)

    if "RequirementText" not in df.columns:
        raise ValueError("Input must contain column 'RequirementText'")

    if "ID" not in df.columns:
        raise ValueError("Input must contain column 'ID' to run in ID-mode.")

    # Ensure stable order; ID-mode still uses file order as canonical row order for applying results
    df = df.reset_index(drop=True)

    # Strong checks for ID
    if df["ID"].isna().any():
        raise ValueError("Column 'ID' contains NaN. ID must be non-empty.")
    if df["ID"].duplicated().any():
        raise ValueError("Column 'ID' has duplicates. ID must be unique.")

    # Normalize ID dtype to string to avoid resume mismatches (1 vs 1.0, etc.)
    df["ID"] = df["ID"].astype(str)

    # ensure label columns exist
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # resume state from progress (keyed by ID)
    done: Dict[str, Dict[str, Any]] = {}
    if progress_path.exists():
        prog = pd.read_csv(progress_path)
        if "ID" in prog.columns:
            prog["ID"] = prog["ID"].astype(str)
            for _, r in prog.iterrows():
                key = str(r["ID"])
                if _is_empty_error(key):
                    continue
                done[key] = r.to_dict()
        print(f"🔄 Resume: loaded {len(done)} rows from {progress_path}")

    # Decide row range to label (limit)
    max_row = len(df) if args.limit <= 0 else min(args.limit, len(df))
    range_info = f"0..{max_row-1}" if max_row > 0 else "empty"

    # Decide rows to process (ONLY within 0..max_row-1)
    to_process: List[int] = []
    for row_idx in range(max_row):
        key = str(df.at[row_idx, "ID"])

        if args.overwrite:
            to_process.append(row_idx)
            continue

        # If already labeled (no error) in progress -> skip
        if key in done:
            err_val = done[key].get("error", None)
            if _is_empty_error(err_val):
                # additionally: if df already has all labels filled, we can skip safely
                # (even if df was loaded without those labels, progress already has it)
                continue

        # If any label is missing on df -> schedule
        missing = False
        for col in LABEL_COLS:
            if pd.isna(df.at[row_idx, col]):
                missing = True
                break

        # Retry if missing labels or if previous attempt had error
        prev_has_error = (key in done) and (not _is_empty_error(done[key].get("error", None)))
        if missing or prev_has_error:
            to_process.append(row_idx)

    print(
        f"🧾 Total rows: {len(df)} | limit={args.limit} -> range={range_info} | "
        f"to label now: {len(to_process)} | overwrite={args.overwrite} | concurrency={args.concurrency}"
    )

    # optional TF-IDF examples
    promise_df = None
    vec = X_promise = promise_texts = None
    if args.top_k > 0:
        promise_path = Path(args.promise)
        if not promise_path.exists():
            raise FileNotFoundError(f"PROMISE file not found: {promise_path}")
        promise_df = pd.read_csv(promise_path)
        if "RequirementText" not in promise_df.columns:
            raise ValueError("PROMISE csv must contain 'RequirementText'")

        vec, X_promise, promise_texts = build_tfidf_retriever(promise_df)
        print(f"🔎 TF-IDF retriever ready (top_k={args.top_k}).")

    # Thread-local agent so each thread has its own client wrapper (safe for concurrency)
    local = threading.local()

    def get_agent() -> LabelingAgent:
        if getattr(local, "agent", None) is None:
            local.agent = LabelingAgent(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )
        return local.agent

    def label_job(row_id: int) -> Tuple[int, Dict[str, Any], bool]:
        key = str(df.at[row_id, "ID"])
        text = str(df.at[row_id, "RequirementText"])

        examples: Optional[List[Dict[str, Any]]] = None
        if args.top_k > 0 and promise_df is not None:
            examples = retrieve_examples(text, promise_df, vec, X_promise, promise_texts, args.top_k)

        try:
            agent = get_agent()
            out = agent.classify(text, SYSTEM_PROMPT, examples=examples)
            rec = out.model_dump()
            rec["ID"] = key
            rec["row_id"] = row_id  # optional debug
            rec["model"] = args.model
            return row_id, rec, True
        except Exception as e:
            return row_id, {"ID": key, "row_id": row_id, "model": args.model, "error": str(e)}, False

    # Run labeling (parallel)
    success_count = 0
    fail_count = 0
    completed = 0

    if len(to_process) == 0:
        print("✅ Nothing to label in the selected range.")
    else:
        workers = max(1, int(args.concurrency))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(label_job, rid) for rid in to_process]

            pbar = tqdm(total=len(futures), desc=f"Labeling (concurrency={workers})")
            for fut in as_completed(futures):
                row_idx, rec, ok = fut.result()

                key = rec.get("ID")
                if key is None or _is_empty_error(key):
                    # fallback: read from df to avoid losing record
                    key = str(df.at[row_idx, "ID"])
                    rec["ID"] = key

                done[str(key)] = rec

                completed += 1
                if ok:
                    success_count += 1
                else:
                    fail_count += 1

                pbar.update(1)

                # checkpoint
                if completed % args.save_every == 0:
                    ok_write = safe_write_csv(pd.DataFrame(done.values()), progress_path)
                    print(
                        f"📌 Checkpoint: completed={completed} | success={success_count} | "
                        f"fail={fail_count} | progress_saved={ok_write}"
                    )

            pbar.close()

        # final save progress
        safe_write_csv(pd.DataFrame(done.values()), progress_path)

    # merge predictions -> final output
    df_out = df.copy()

    apply_max_row = max_row  # only apply in selected range when limit is set
    id_to_idx: Dict[str, int] = {str(df.at[i, "ID"]): i for i in range(len(df))}

    for key, rec in done.items():
        if "error" in rec and (not _is_empty_error(rec.get("error", None))):
            continue
        if key not in id_to_idx:
            continue

        rid = id_to_idx[key]
        if rid < 0 or rid >= apply_max_row:
            continue

        # map code fields -> label columns
        for code, col in CODE_TO_COL.items():
            if code in rec:
                try:
                    df_out.at[rid, col] = int(rec[code])
                except Exception:
                    df_out.at[rid, col] = pd.NA

    # meta columns
    df_out["AI_Model"] = args.model

    # optional: store confidence/rationale if present (keyed by ID)
    conf = [None] * len(df_out)
    rat = [None] * len(df_out)

    for key, rec in done.items():
        if "error" in rec and (not _is_empty_error(rec.get("error", None))):
            continue
        if key not in id_to_idx:
            continue
        rid = id_to_idx[key]
        if rid < 0 or rid >= apply_max_row:
            continue

        conf[rid] = rec.get("confidence")
        rat[rid] = rec.get("rationale")

    df_out["AI_Confidence"] = conf
    df_out["AI_Rationale"] = rat

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ✅ Summary
    total_target = len(to_process)
    rate = (success_count / total_target) * 100.0 if total_target > 0 else 100.0

    print(f"✅ Done. Output: {out_path}")
    print(f"🧩 Progress file (resume): {progress_path}")
    print(
        f"📊 Summary (this run): target={total_target} | success={success_count} | "
        f"fail={fail_count} | success_rate={rate:.1f}%"
    )


if __name__ == "__main__":
    main()
