import argparse
import pandas as pd

LABEL_COLS = [
    "Functional (F)", "Availability (A)", "Fault Tolerance (FT)", "Legal (L)",
    "Look & Feel (LF)", "Maintainability (MN)", "Operability (O)", "Performance (PE)",
    "Portability (PO)", "Scalability (SC)", "Security (SE)", "Usability (US)",
]
CODES = ["F","A","FT","L","LF","MN","O","PE","PO","SC","SE","US"]
CODE_TO_LABEL = dict(zip(CODES, LABEL_COLS))

# Progress columns (thường gặp) -> Output columns (trong labeled)
EXTRA_MAP = {
    "model": "AI_Model",
    "confidence": "AI_Confidence",
    "rationale": "AI_Rationale",
}

def nonempty(x) -> bool:
    if x is None:
        return False
    s = str(x).strip()
    return s != "" and s.lower() != "nan"

def to_int01(v):
    # handle bool/int/str safely
    if pd.isna(v):
        return 0
    if isinstance(v, bool):
        return int(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return 1
    if s in ("0", "false", "no"):
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0

def to_float(v):
    if pd.isna(v):
        return pd.NA
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return pd.NA
    try:
        return float(s)
    except Exception:
        return pd.NA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input dataset CSV")
    ap.add_argument("--progress", required=True, help="Progress CSV (row_id + labels + confidence/rationale)")
    ap.add_argument("--output", required=True, help="Output labeled CSV")
    ap.add_argument("--id_col", default="", help="Optional: use stable ID column instead of row_id if both files have it")
    args = ap.parse_args()

    df = pd.read_csv(args.input).reset_index(drop=True)
    prog = pd.read_csv(args.progress)

    # accept either RequirementText or requirementtext
    if ("RequirementText" not in df.columns) and ("requirementtext" not in df.columns):
        raise ValueError("Input must contain column RequirementText (or requirementtext).")

    # ensure label columns exist in output
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # ensure extra columns exist in output
    for dst in EXTRA_MAP.values():
        if dst not in df.columns:
            df[dst] = pd.NA

    # determine label source format in progress
    has_codes = all(c in prog.columns for c in CODES)
    has_labelcols = all(c in prog.columns for c in LABEL_COLS)

    if not has_codes and not has_labelcols:
        raise ValueError(
            "Progress file does NOT contain label columns. "
            "Need either codes columns (F,A,FT,...,US) or full label columns."
        )

    # choose join key
    use_id = args.id_col and (args.id_col in df.columns) and (args.id_col in prog.columns)
    if use_id:
        prog_key = args.id_col
        prog = prog.dropna(subset=[prog_key])
    else:
        if "row_id" not in prog.columns:
            raise ValueError("Progress must contain row_id column if id_col is not used.")
        prog = prog.dropna(subset=["row_id"])
        prog["row_id"] = prog["row_id"].astype(int)

    updated = 0
    skipped_err = 0

    for _, r in prog.iterrows():
        # skip failed rows
        if "error" in prog.columns and nonempty(r.get("error", "")):
            skipped_err += 1
            continue

        if use_id:
            key = r[args.id_col]
            idxs = df.index[df[args.id_col] == key].tolist()
            if not idxs:
                continue
            rid = idxs[0]
        else:
            rid = int(r["row_id"])
            if rid < 0 or rid >= len(df):
                continue

        # 1) recover labels
        if has_codes:
            for code, col in CODE_TO_LABEL.items():
                df.at[rid, col] = to_int01(r.get(code, 0))
        else:
            for col in LABEL_COLS:
                df.at[rid, col] = to_int01(r.get(col, 0))

        # 2) recover extras: model / confidence / rationale
        # model
        if "model" in prog.columns and nonempty(r.get("model")):
            df.at[rid, "AI_Model"] = str(r.get("model")).strip()

        # confidence
        if "confidence" in prog.columns and nonempty(r.get("confidence")):
            df.at[rid, "AI_Confidence"] = to_float(r.get("confidence"))

        # rationale
        if "rationale" in prog.columns and nonempty(r.get("rationale")):
            df.at[rid, "AI_Rationale"] = str(r.get("rationale"))

        updated += 1

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"✅ Rebuilt: {args.output}")
    print(f"   Updated rows: {updated}")
    print(f"   Skipped error rows: {skipped_err}")

if __name__ == "__main__":
    main()
