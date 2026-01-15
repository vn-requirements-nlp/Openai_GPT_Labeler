# src/run_labeling.py
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

from src.llm_client import LabelingAgent
from src.config import SYSTEM_PROMPT, LABEL_COLS, COL_TO_CODE, CODE_TO_COL


def build_tfidf_retriever(promise_df: pd.DataFrame):
    texts = promise_df["RequirementText"].astype(str).tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
    X = vec.fit_transform(texts)
    return vec, X, texts


def get_example_labels(row: pd.Series) -> str:
    codes = []
    for col in LABEL_COLS:
        v = row.get(col, 0)
        try:
            v = int(v)
        except Exception:
            v = 0
        if v == 1:
            codes.append(COL_TO_CODE[col])
    return ",".join(codes) if codes else ""


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
                "labels": get_example_labels(row),
            }
        )
    return examples


def safe_write_csv(df: pd.DataFrame, path: Path) -> bool:
    """
    Write CSV safely on Windows where Excel/VSCode may lock the file.
    Returns True if write succeeded, False otherwise.
    """
    try:
        df.to_csv(path, index=False)
        return True
    except PermissionError:
        print(f"⚠️ Permission denied while writing: {path} (file is locked). Close Excel/VSCode preview and continue...")
        return False


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

    # limit first N rows from top
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Label only first N rows from the top (0 = all). Example: --limit 5",
    )

    # ✅ NEW: concurrency
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

    # Ensure row_id follows file order (top-to-bottom)
    df = df.reset_index(drop=True)

    # ensure label columns exist
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # resume state from progress
    done: Dict[int, Dict[str, Any]] = {}
    if progress_path.exists():
        prog = pd.read_csv(progress_path)
        if "row_id" in prog.columns:
            for _, r in prog.iterrows():
                try:
                    rid = int(r["row_id"])
                except Exception:
                    continue
                done[rid] = r.to_dict()
        print(f"🔄 Resume: loaded {len(done)} rows from {progress_path}")

    # Decide row range to label (limit)
    max_row = len(df) if args.limit <= 0 else min(args.limit, len(df))

    # Decide rows to process (ONLY within 0..max_row-1)
    to_process: List[int] = []
    for row_id in range(max_row):
        row = df.loc[row_id]

        if args.overwrite:
            # overwrite means ALWAYS re-label within range
            to_process.append(row_id)
            continue

        # not overwrite: skip already-success rows
        if row_id in done:
            err_val = done[row_id].get("error", None)
            # Nếu err_val là NaN/None/"" => coi như SUCCESS và skip
            if err_val is None or (isinstance(err_val, float) and pd.isna(err_val)) or str(err_val).strip() == "":
                continue

        # label rows with any missing label col (or retry rows that had error)
        missing = False
        for col in LABEL_COLS:
            v = row.get(col, pd.NA)
            if pd.isna(v):
                missing = True
                break
        if missing or (row_id in done and "error" in done[row_id]):
            to_process.append(row_id)

    range_info = f"0..{max_row-1}" if max_row > 0 else "empty"
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
        text = str(df.at[row_id, "RequirementText"])

        examples: Optional[List[Dict[str, Any]]] = None
        if args.top_k > 0 and promise_df is not None:
            examples = retrieve_examples(text, promise_df, vec, X_promise, promise_texts, args.top_k)

        try:
            agent = get_agent()
            out = agent.classify(text, SYSTEM_PROMPT, examples=examples)
            rec = out.model_dump()
            rec["row_id"] = row_id
            rec["model"] = args.model
            # success
            return row_id, rec, True
        except Exception as e:
            # fail
            return row_id, {"row_id": row_id, "model": args.model, "error": str(e)}, False

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
                row_id, rec, ok = fut.result()
                done[int(row_id)] = rec

                completed += 1
                if ok:
                    success_count += 1
                else:
                    fail_count += 1

                pbar.update(1)

                # checkpoint
                if completed % args.save_every == 0:
                    ok_write = safe_write_csv(pd.DataFrame(done.values()), progress_path)
                    # print stats at checkpoint
                    print(f"📌 Checkpoint: completed={completed} | success={success_count} | fail={fail_count} | progress_saved={ok_write}")

            pbar.close()

        # final save progress
        safe_write_csv(pd.DataFrame(done.values()), progress_path)

    # merge predictions -> final output
    df_out = df.copy()

    apply_max_row = max_row  # only apply in selected range when limit is set
    for row_id, rec in done.items():
        rid = int(row_id)
        if rid < 0 or rid >= apply_max_row:
            continue
        if "error" in rec:
            continue

        # map code fields -> label columns
        for code, col in CODE_TO_COL.items():
            if code in rec:
                df_out.at[rid, col] = int(rec[code])

    # meta columns
    df_out["AI_Model"] = args.model

    # optional: store confidence/rationale if present
    conf = [None] * len(df_out)
    rat = [None] * len(df_out)
    for row_id, rec in done.items():
        rid = int(row_id)
        if rid < 0 or rid >= apply_max_row:
            continue
        if "error" in rec:
            continue
        conf[rid] = rec.get("confidence")
        rat[rid] = rec.get("rationale")
    df_out["AI_Confidence"] = conf
    df_out["AI_Rationale"] = rat

    df_out.to_csv(out_path, index=False, encoding="utf-8")

    # ✅ Summary
    total_target = len(to_process)
    if total_target > 0:
        rate = (success_count / total_target) * 100.0
    else:
        rate = 100.0

    print(f"✅ Done. Output: {out_path}")
    print(f"🧩 Progress file (resume): {progress_path}")
    print(f"📊 Summary (this run): target={total_target} | success={success_count} | fail={fail_count} | success_rate={rate:.1f}%")


if __name__ == "__main__":
    main()
