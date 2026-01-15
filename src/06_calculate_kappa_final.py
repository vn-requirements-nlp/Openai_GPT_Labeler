import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import os

# --- CẤU HÌNH ---
HUMAN_FILE = "output/human_labels_sample_500.csv"  # File human
AI_FILE = "output/ai_labels_sample_500.csv"        # File AI
OUTPUT_REPORT = "output/Kappa_Reliability_Report.csv"
# ----------------

def main():
    # 1. Đọc dữ liệu
    if not os.path.exists(HUMAN_FILE) or not os.path.exists(AI_FILE):
        print("❌ Lỗi: Không tìm thấy file input.")
        return

    print("reading files...")
    df_h = pd.read_csv(HUMAN_FILE)
    df_a = pd.read_csv(AI_FILE)

    # 2. Merge 2 file dựa trên ID để đảm bảo so sánh đúng dòng
    # (Dù em có sort file human kiểu gì thì merge theo ID vẫn đúng)
    df_merged = pd.merge(df_h, df_a, on="ID", suffixes=('_human', '_ai'))
    
    print(f"✅ Đã ghép đôi thành công {len(df_merged)} dòng dữ liệu.")

    # 3. Danh sách các nhãn cần tính
    # Mapping: Tên cột Human -> Tên cột AI (Check kỹ tên cột trong file CSV của em)
    # Dựa trên file em gửi, Human có prefix "Human_", AI thì không.
    labels_map = [
        ("Human_Functional (F)",       "Functional (F)"),
        ("Human_Availability (A)",     "Availability (A)"),
        ("Human_Fault Tolerance (FT)", "Fault Tolerance (FT)"),
        ("Human_Legal (L)",            "Legal (L)"),
        ("Human_Look & Feel (LF)",     "Look & Feel (LF)"),
        ("Human_Maintainability (MN)", "Maintainability (MN)"),
        ("Human_Operability (O)",      "Operability (O)"),
        ("Human_Performance (PE)",     "Performance (PE)"),
        ("Human_Portability (PO)",     "Portability (PO)"),
        ("Human_Scalability (SC)",     "Scalability (SC)"),
        ("Human_Security (SE)",        "Security (SE)"),
        ("Human_Usability (US)",       "Usability (US)")
    ]

    report = []
    scores = []

    print("\n" + "="*50)
    print(f"{'LABEL':<25} | {'KAPPA SCORE':<12} | {'QUALITY':<15}")
    print("="*50)

    for col_h, col_a in labels_map:
        # Kiểm tra xem cột có tồn tại không
        if col_h not in df_merged.columns or col_a not in df_merged.columns:
            print(f"⚠️ Cảnh báo: Không tìm thấy cột {col_h} hoặc {col_a}")
            continue

        # Lấy dữ liệu 2 cột
        y_human = df_merged[col_h].fillna(0).astype(int)
        y_ai = df_merged[col_a].fillna(0).astype(int)

        # Tính Kappa
        # Lưu ý: Nếu một nhãn cả Human và AI đều không gán lần nào (toàn số 0), Kappa sẽ là NaN (đặt là 1.0 tuyệt đối)
        if y_human.sum() == 0 and y_ai.sum() == 0:
            kappa = 1.0
        else:
            kappa = cohen_kappa_score(y_human, y_ai)
            if np.isnan(kappa): kappa = 0 # Trường hợp lỗi khác

        scores.append(kappa)

        # Đánh giá chất lượng
        quality = ""
        if kappa >= 0.8: quality = "Excellent 🌟"
        elif kappa >= 0.6: quality = "Good ✅"
        elif kappa >= 0.4: quality = "Moderate ⚠️"
        else: quality = "Poor ❌"

        print(f"{col_a.split('(')[0]:<25} | {kappa:.4f}       | {quality}")
        
        report.append({
            "Label": col_a,
            "Kappa": kappa,
            "Quality": quality,
            "Human_Count": y_human.sum(),
            "AI_Count": y_ai.sum()
        })

    # 4. Tính trung bình
    avg_kappa = np.mean(scores)
    print("="*50)
    print(f"{'AVERAGE (MACRO)':<25} | {avg_kappa:.4f}       | {'PASSED' if avg_kappa > 0.6 else 'REVIEW NEEDED'}")
    print("="*50)

    # 5. Lưu báo cáo
    os.makedirs("output", exist_ok=True)
    pd.DataFrame(report).to_csv(OUTPUT_REPORT, index=False)
    print(f"\n📄 Đã lưu báo cáo chi tiết tại: {OUTPUT_REPORT}")

    # Lời khuyên của giảng viên
    if avg_kappa > 0.75:
        print("\n🎉 CHÚC MỪNG! Bộ dữ liệu đạt chuẩn 'High Quality Silver Standard'.")
    elif avg_kappa > 0.6:
        print("\n✅ Tốt! Dữ liệu chấp nhận được cho nghiên cứu khoa học.")
    else:
        print("\n⚠️ Cần xem lại: Có vẻ Human và AI đang hiểu sai ý nhau ở các nhãn điểm thấp.")

if __name__ == "__main__":
    main()