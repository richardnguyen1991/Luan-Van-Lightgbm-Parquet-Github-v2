# LightGBM baseline for CIC-DDoS2019 Parquet

Pipeline huấn luyện baseline LightGBM đa lớp trên mẫu xác định đúng 10.000.000 dòng của bộ dữ liệu
`dungnguyen28101991/cicddos2019-parquet`. Mẫu được phân bổ theo tỷ lệ số dòng vật lý của từng
file rồi lấy ngẫu nhiên không hoàn lại với seed 2026, vì vậy không lấy phần đầu file, không tái cân
bằng lớp và tái lập được. Pipeline chạy trên Kaggle CPU,
lưu checkpoint S3 mỗi 10 boosting iterations, tự tiếp tục sau khi session Kaggle kết thúc
và chỉ hoàn tất khi model đạt chính xác iteration 100 cùng bộ báo cáo cuối.

Khi dữ liệu thô không thể materialize an toàn trong RAM, pipeline giải mã từng Parquet row-group
thành NumPy `float32` trong bộ đệm LRU dùng chung. Production chỉ giữ đúng một row-group,
không memory-map cả file Parquet, đóng handle ngay sau mỗi lần đọc, tái sử dụng cùng một
buffer NumPy và trả các trang heap đã giải phóng về Linux. Nhờ vậy quá trình sampling
của `lgb.Dataset` không tích lũy RSS sau hàng trăm row-group và không tạo bản sao `.npy`
làm đầy ổ Kaggle. Chỉ train và validation được dựng thành LightGBM Dataset;
test vẫn lazy.

Dữ liệu CIC-DDoS2019 có rất nhiều giá trị 0 và hàng trăm triệu hàng. Chế độ sparse tự động
của LightGBM tích lũy hơn 20 GiB native memory trong khi `Dataset` được dựng. Vì vậy
pipeline đặt `is_enable_sparse=false` để lưu các bin dạng dense có giới hạn. Đây chỉ
thay đổi biểu diễn bộ nhớ nội bộ; không bỏ hàng, bỏ feature, tái cân bằng hay đổi
100 boosting iterations và các siêu tham số học.

## Hợp đồng thí nghiệm

- `lightgbm.train` dùng đúng một `lgb.Dataset` cho train và validation; test được đọc lazy khi báo cáo.
- `objective=multiclass`, `learning_rate=0.001`, `num_boost_round=100` chính xác.
- Không early stopping, tuning, feature selection, class/sample weight hoặc tái cân bằng.
- Toàn bộ train split của mẫu 10 triệu dòng được dùng ở cả 100 vòng; validation chỉ theo dõi,
  test chỉ đánh giá cuối.
- CPU bắt buộc, seed cố định, deterministic và `force_col_wise=true`.
- Checkpoint mỗi 10 vòng gồm Booster `.txt`, `training_state.json` và history append-only.
- Model cuối luôn là `final_model_round_100.txt`; importance không thay đổi baseline này.

Các tham số đầy đủ nằm trong `config/*.json`, không hard-code rải rác trong mã nguồn.

## Kiến trúc

```text
data.py                         đọc Parquet, profile RAM, split và chống rò rỉ
model.py                        dựng lgb.Dataset, custom Macro-F1 và callback
train.py                        huấn luyện/resume đúng 100 boosting iterations
checkpoint.py                   checkpoint local/S3 và xác minh SHA-256
viz.py                          toàn bộ hàm vẽ, không gọi plt.show()
make_report.py                  đánh giá cuối và tái tạo báo cáo từ artifact
kaggle_notebook.ipynb           notebook production tự chứa
kaggle_smoke_test.ipynb         notebook smoke tự chứa
config/data.json                cấu hình mẫu tỷ lệ xác định 10 triệu dòng
config/train.json               cấu hình baseline production
config/report.json              cấu hình metric/figure/importance
config/orchestration.json       watchdog GitHub Actions/Kaggle
scripts/                        build notebook, Kaggle API và orchestration
.github/workflows/run-kaggle.yml
```

Luồng production:

```text
Kaggle Parquet -> data.py -> outputs/data/{train,validation,test}
                              |
                              v
                         train.py / LightGBM
                              |
                    checkpoint mỗi 10 vòng
                              |
                 S3 checkpoints + history + hình nhẹ
                              |
             GitHub Actions kiểm tra và mở session kế tiếp
                              |
                    final iteration = 100
                              |
                make_report.py -> metrics/figures/importance
```

## GitHub Secrets

Repository cần các secret sau:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION` (`us-east-1` cho cấu hình hiện tại)
- `S3_BUCKET` (`my-thesis-checkpoints` cho cấu hình hiện tại)
- `S3_PREFIX` (`Luan-Van-Lightgbm-Parquet-Github-v2` cho cấu hình hiện tại)
- `KAGGLE_API_TOKEN`; workflow hỗ trợ token OAuth `KGAT_...` và legacy API key.
- `KAGGLE_USERNAME` nếu dùng legacy API key.
- `KAGGLE_KERNEL` là tùy chọn; mặc định lấy từ `config/orchestration.json`.

Workflow chỉ dùng AWS credential trong GitHub Actions để tạo presigned URL ngắn hạn,
giới hạn theo từng object. Notebook Kaggle không chứa khóa AWS dài hạn và không ghi secret
vào log, checkpoint hoặc artifact.

## Chạy trên Kaggle qua GitHub Actions

1. Xác nhận dataset đã được gắn theo `kernel-metadata.json` và accelerator là CPU.
2. Mở **Actions -> Run Kaggle LightGBM sessions -> Run workflow**.
3. Chọn `smoke` để kiểm tra nhanh hoặc `production` để chạy mẫu khoa học 10 triệu dòng.
4. Chỉ bật `force_push` khi cần bỏ qua quyết định chờ của watchdog.
5. Workflow chạy định kỳ phút 07 và 37. Nó không mở session trùng khi kernel đang chạy,
   không chạy thêm sau trạng thái `complete`, và tiếp tục các trạng thái `paused`,
   `cancelled` hoặc `ready_for_report`.

`maximum_session_attempts` trong `config/orchestration.json` giới hạn số lần mở session
liên tiếp không tạo thêm iteration hoặc file preprocessing bền vững. Bộ đếm tự reset khi
watchdog quan sát thấy tiến triển mới, nên một run dài không bị khóa vĩnh viễn chỉ vì tổng
số session đã vượt ngưỡng.

Notebook production tự thực hiện:

```bash
python data.py --config config/data.json --output-dir outputs/data
python train.py --config config/train.json --output-dir outputs/runs
```

Exit code `75` nghĩa là session đã dừng an toàn sau checkpoint; watchdog sẽ mở session mới.
Exit code `0` chỉ được xem là hoàn tất khi model đạt iteration 100 và báo cáo cuối thành công.

## Chạy cục bộ

Yêu cầu Python 3.10+ và các gói: `lightgbm>=4,<5`, `numpy`, `pandas`, `pyarrow`,
`scikit-learn`, `matplotlib`, `seaborn`, `psutil`, `boto3` và `requests`.

Chuẩn bị mẫu production 10 triệu dòng:

```bash
python data.py \
  --config config/data.json \
  --data-dir /path/to/cicddos2019-parquet \
  --output-dir outputs/data
```

Chạy sampled/smoke:

```bash
python data.py \
  --config config/data.smoke.json \
  --data-dir /path/to/cicddos2019-parquet \
  --output-dir outputs/data-smoke \
  --samples-per-file 2000

python train.py \
  --config config/train.smoke.json \
  --prepared-data-dir outputs/data-smoke \
  --output-dir outputs/runs-smoke \
  --max-rounds-this-session 20
```

Lệnh train trên kết thúc với code `75` sau vòng 20, tức đã hoàn tất hai checkpoint block.
Chạy lại cùng lệnh nhưng bỏ `--max-rounds-this-session`; `active_run.json` và checkpoint
sẽ làm pipeline tiếp tục từ vòng 21, không lặp vòng 1-20:

```bash
python train.py \
  --config config/train.smoke.json \
  --prepared-data-dir outputs/data-smoke \
  --output-dir outputs/runs-smoke
```

Production với S3 dùng `config/train.json`; đặt biến môi trường tương ứng GitHub Secrets
hoặc truyền `--upload-checkpoints-to-s3`. Không đưa credential vào câu lệnh hay file cấu hình.

## Tái tạo báo cáo

Từ artifact local:

```bash
python make_report.py --run-dir outputs/runs/<run_id> --no-upload-to-s3
```

Từ artifact S3 mà không huấn luyện lại:

```bash
python make_report.py \
  --run-dir s3://my-thesis-checkpoints/Luan-Van-Lightgbm-Parquet-Github-v2/<run_id> \
  --upload-to-s3
```

Báo cáo cuối là idempotent. ROC, PR, confusion matrix, permutation importance và SHAP chỉ
được sinh sau vòng 100. Mỗi nhóm hình có PNG 300 dpi, PDF vector và CSV dữ liệu tương ứng.

## Artifact và cấu trúc S3

```text
s3://<bucket>/<prefix>/<run_id>/
  checkpoints/    last_model.txt, final_model_round_100.txt, training_state.json
  metrics/        history.*, test_metrics.json, summary_metrics.csv,
                  per_class_metrics.csv, confusion_matrix*.csv,
                  roc_curves.csv, pr_curves.csv, feature_importance*.csv
  figures/        *.png, *.pdf và CSV dữ liệu hình
  raw/            y_true.npy, y_prob.npy, explain_sample.parquet và manifest
  explainability/ gain, split, permutation và SHAP ở định dạng CSV/PNG/PDF
  config/         run_config.json, model_params.json, preprocessing.json,
                  sample_manifest.json, label_mapping.json
```

`sample_manifest.json` là bằng chứng chống rò rỉ: sample ID và group không giao nhau giữa
train/validation/test. `data_profile.json` ghi số hàng, cột, dtype, RAM ước lượng và điều
kiện an toàn trước khi nạp mẫu đã chọn.

## Nghiệm thu

Chạy test suite:

```bash
python -m unittest discover -s tests -v
```

Sau production run, kiểm tra:

1. `history.json` chứa đúng các iteration 1..100, không thiếu hoặc trùng.
2. Có ít nhất hai `session_id`; learning curves có vạch `Resume` tại điểm đổi session.
3. `final_model_round_100.txt` nạp độc lập và `current_iteration()==100`.
4. `sample_manifest.json` báo kiểm tra giao split đạt.
5. Có đủ 13 nhóm hình, CSV đi kèm và bốn nhóm importance gain/split/permutation/SHAP.
6. Tên/thứ tự feature của Booster khớp tuyệt đối với `preprocessing.json`.

## Diễn giải feature importance

Gain và split là importance phụ thuộc cấu trúc model. SHAP biểu diễn đóng góp vào dự đoán,
không chứng minh quan hệ nhân quả. Các thuộc tính tương quan có thể chia sẻ importance;
không gộp bốn thước đo thành một điểm duy nhất và không dùng chúng để sửa baseline này.
