# GPT Labeler (PROMISE NFR)

Công cụ gán nhãn yêu cầu phần mềm theo 12 nhãn PROMISE NFR bằng OpenAI. Repo hiện có dữ liệu mẫu trong `data/` và các kết quả đã sinh trong `output/`.

## Cài đặt
PowerShell:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Cấu hình API
Tạo file `.env` từ `.env.example` và đặt khóa:
```
OPENAI_API_KEY=your_key_here
```

## Dữ liệu đầu vào
- `data/Dataset_Full_EN.csv`: file cần gán nhãn, bắt buộc có cột `RequirementText`.
- `data/PROMISE-relabeled-NICE.csv`: dữ liệu PROMISE đã gán nhãn để làm ví dụ tham chiếu khi bật `--top_k`.

## Chạy gán nhãn (03_run_labeling.py)
Ví dụ (đồng bộ với các script hậu xử lý dùng thư mục `output/`):
```powershell
python src/_03_run_labeling.py `
  --input data/Dataset_Full_EN.csv `
  --promise data/PROMISE-relabeled-NICE.csv `
  --output output/Dataset_Full_EN_labeled.csv `
  --model gpt-5-mini `
  --reasoning_effort low `
  --top_k 4 `
  --save_every 25 `
  --concurrency 8 `
  --limit 0
```

Ghi chú:
- Mặc định code đặt `--output` là `outputs/Dataset_Full_EN_labeled.csv`. Repo này đang dùng `output/`, nên ví dụ ở trên có `--output output/...` để thống nhất.
- File progress sẽ là `output/Dataset_Full_EN_labeled.progress.csv` (cùng tên với output nhưng thêm `.progress.csv`).
- `--limit 0` = gán nhãn toàn bộ; thêm `--overwrite` nếu muốn ghi đè các nhãn đã có.

Tham số chính:
- `--input`: CSV/XLSX có cột `RequirementText`.
- `--promise`: file PROMISE (chỉ dùng khi `--top_k > 0`).
- `--output`: file CSV kết quả.
- `--model`: ví dụ `gpt-5-mini`.
- `--reasoning_effort`: `none|low|medium|high`.
- `--top_k`: số ví dụ PROMISE dùng TF-IDF (0 = tắt).
- `--save_every`: tần suất checkpoint (mặc định 25).
- `--concurrency`: số request chạy song song (mặc định 8).
- `--limit`: chỉ chạy N dòng đầu (0 = tất cả).

## Khôi phục output từ progress (04_recover_labeled_from_progress.py)
```powershell
python src/_04_recover_labeled_from_progress.py `
  --input data/Dataset_Full_EN.csv `
  --progress output/Dataset_Full_EN_labeled.progress.csv `
  --output output/Dataset_Full_EN_labeled_REBUILT.csv `
  --id_col ID
```

## Tạo mẫu cho human labeling (05_make_human_sample.py)
Script dùng các đường dẫn cố định trong file:
```powershell
python src/_05_make_human_sample.py
```
Sinh ra:
- `output/human_labels_sample_500.csv`
- `output/ai_labels_sample_500.csv`

## Tính kappa giữa human và AI (06_calculate_kappa_final.py)
```powershell
python src/_06_calculate_kappa_final.py
```
Sinh report: `output/Kappa_Reliability_Report.csv`.
