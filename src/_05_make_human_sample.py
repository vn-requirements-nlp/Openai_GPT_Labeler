import pandas as pd
from pathlib import Path

LABEL_COLS = [
    "Functional (F)", "Availability (A)", "Fault Tolerance (FT)", "Legal (L)",
    "Look & Feel (LF)", "Maintainability (MN)", "Operability (O)", "Performance (PE)",
    "Portability (PO)", "Scalability (SC)", "Security (SE)", "Usability (US)",
]

IN_PATH = "output/Dataset_Full_EN_labeled.csv"          # file đã có nhãn AI
OUT_CSV = "output/human_labels_sample_500.csv"          # file để human label (blind)
KEY_AI  = "output/ai_labels_sample_500.csv"             # file “đối chiếu” AI cho đúng 500 dòng

SEED = 42
N = 500

# đảm bảo thư mục output tồn tại
Path("output").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_PATH)

# nếu chưa có ID thì tạo ID = index
if "ID" not in df.columns:
    df = df.copy()
    df.insert(0, "ID", range(len(df)))

# dò cột text (RequirementText vs requirementtext)
if "RequirementText" in df.columns:
    TEXT_COL = "RequirementText"
elif "requirementtext" in df.columns:
    TEXT_COL = "requirementtext"
else:
    raise ValueError("Không tìm thấy cột requirement text. Cần có 'RequirementText' hoặc 'requirementtext'.")

# lấy 500 dòng ngẫu nhiên
sample = df.sample(n=N, random_state=SEED).reset_index(drop=True)

# file cho human label: chỉ ID + text + cột human trống
out = sample[["ID", TEXT_COL]].copy()
for col in LABEL_COLS:
    out[f"Human_{col}"] = ""   # human điền 0/1

# lưu CSV (utf-8-sig để mở bằng Excel đỡ lỗi font)
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# file key AI (để tính kappa): giữ ID + text + nhãn AI tương ứng
sample[["ID", TEXT_COL] + LABEL_COLS].to_csv(KEY_AI, index=False, encoding="utf-8-sig")

print("Saved:", OUT_CSV)
print("Saved:", KEY_AI)
print("TEXT_COL used:", TEXT_COL)
