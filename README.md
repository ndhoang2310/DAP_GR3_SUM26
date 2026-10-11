# BlinkGuard — Eye Blink & Drowsiness Detection

BlinkGuard là hệ thống phát hiện chớp mắt và trạng thái nhắm mắt kéo dài từ webcam. Dự án bao phủ toàn bộ vòng đời machine learning: thu thập và quản lý dữ liệu, tiền xử lý, huấn luyện và benchmark hai hướng mô hình, chạy realtime bằng Python, và triển khai web 100% client-side.

## Trải nghiệm trực tuyến

> **[Mở BlinkGuard Web Demo](https://orange-credit-f323.ndhoang2310.workers.dev/)**

Web demo chạy trực tiếp trên trình duyệt. Toàn bộ quá trình xử lý webcam và suy luận diễn ra trên thiết bị của người dùng; hình ảnh camera không được gửi tới backend.

Để trải nghiệm tốt nhất:

- Mở liên kết bằng Chrome, Edge hoặc Safari phiên bản mới.
- Cho phép trình duyệt sử dụng camera.
- Nhìn thẳng vào camera trong vài giây đầu để hệ thống hiệu chuẩn EAR.
- Sử dụng nơi có ánh sáng ổn định và khuôn mặt không bị che khuất.

## Tổng quan

Dự án giải quyết ba bài toán liên kết với nhau:

1. **Thu thập và chuẩn hóa dữ liệu mắt:** ghi hình webcam, định vị khuôn mặt bằng MediaPipe, trích xuất vùng mắt và gán nhãn.
2. **Nghiên cứu mô hình:** so sánh pipeline Image Model và TimeSeries EAR trên các tập dữ liệu tách theo `video_id` để hạn chế data leakage.
3. **Ứng dụng realtime:** theo dõi chớp mắt, nhận diện nhắm mắt kéo dài và cảnh báo người dùng trên desktop hoặc ngay trong trình duyệt.

### Luồng hệ thống

```text
Webcam / video
      ↓
MediaPipe Face Landmarks
      ↓
Vùng mắt 24×24 hoặc chuỗi EAR
      ↓
Tiền xử lý + hiệu chuẩn động
      ↓
Image models / TimeSeries models
      ↓
Temporal smoothing + blink counting
      ↓
Cảnh báo và giao diện realtime
```

## Hai hướng mô hình

| Pipeline | Đầu vào | Mô hình chính | Kết quả nổi bật | Vai trò |
| --- | --- | --- | ---: | --- |
| Image Model | Ảnh xám vùng mắt 24×24 | CNN | Accuracy **90.86%**, Macro F1 **0.8603** | Baseline thị giác và desktop demo |
| TimeSeries EAR | Cửa sổ 7 frame, 12 đặc trưng | Linear SVM | Accuracy **93.34%**, Macro F1 **0.9049** | Pipeline chính và web deployment |

Các tập train, validation và test được chia theo nhóm `video_id`. Frame từ cùng một video không xuất hiện đồng thời ở nhiều tập, giúp benchmark phản ánh tốt hơn khả năng tổng quát hóa sang người dùng mới.

### Image Model

Pipeline ảnh so sánh ba cấu hình:

- **SVM HOG-only:** mô hình ablation dùng đặc trưng HOG.
- **SVM HOG + EAR:** baseline nhẹ, độ trễ khoảng 0.75 ms/ảnh.
- **CNN:** mô hình phân loại ảnh tốt nhất, dùng threshold 0.45 và temporal smoothing trong realtime demo.

| Model | Accuracy | AUC | Macro F1 |
| --- | ---: | ---: | ---: |
| SVM HOG-only | 0.8888 | **0.9471** | 0.8339 |
| SVM HOG + EAR | 0.8911 | **0.9483** | 0.8366 |
| CNN, threshold 0.45 | **0.9086** | 0.9313 | **0.8603** |

Xem báo cáo chi tiết tại [reports/report_image.md](reports/report_image.md).

### TimeSeries EAR

Pipeline TimeSeries lấy mẫu ở khoảng 15 FPS và sử dụng cửa sổ trượt 7 giá trị EAR. Mỗi cửa sổ được biểu diễn bằng 12 đặc trưng: 7 EAR đã chuẩn hóa, min, max, standard deviation, ratio center và kurtosis.

Các thành phần chính:

- Hiệu chuẩn động từ 15 frame đầu và cập nhật `expanding_max` trong khi chạy.
- Linear SVM phân loại `no-blink`, `blink` và `long-closure`.
- Hybrid heuristic filter bỏ qua suy luận ML khi mắt mở rõ ràng, giảm khoảng 54% số lần gọi model.
- Temporal smoothing và gom cụm dự đoán để hạn chế cảnh báo giả hoặc đếm một cú chớp mắt nhiều lần.

| Model | Accuracy | Macro F1 | Ghi chú |
| --- | ---: | ---: | --- |
| Linear SVM, 12 features | **93.34%** | **0.9049** | Mô hình tốt nhất |
| Hybrid Filter + SVC | 93.09% | 0.8952 | Giảm khoảng 54.5% lượt suy luận |
| RBF SVM, 7 EAR | 93.09% | 0.8937 | Baseline chuỗi thô |
| LSTM | 76.44% | 0.5131 | Overfit trên tập dữ liệu nhỏ |

Xem benchmark chi tiết tại [src/models/TimeSeries/report_model_benchmark.md](src/models/TimeSeries/report_model_benchmark.md).

## Web app client-side

Phiên bản BlinkGuard trên web không cần backend inference:

```text
Camera trong trình duyệt
      ↓
MediaPipe FaceLandmarker (WASM)
      ↓
EAR + dynamic calibration
      ↓
Sliding window 7 frame
      ↓
Hybrid filter + Linear SVM JavaScript
      ↓
Blink metrics + cảnh báo người dùng
```

Model `StandardScaler + Linear SVM` được xuất từ Python sang JavaScript thuần. Nhờ đó, video không rời khỏi thiết bị, độ trễ thấp và máy chủ chỉ cần phục vụ các file tĩnh.

Mã nguồn web nằm trong [`deployment/scripts`](deployment/scripts). Tài liệu kiến trúc và triển khai:

- [Báo cáo web client-side](reports/web_client_side_deployment_report.md)
- [Kiến trúc deployment](reports/client_side_deployment_architecture.md)
- [Project context summary](reports/PROJECT_CONTEXT_SUMMARY.md)

## Cài đặt

### Yêu cầu

- Python 3.10 được khuyến nghị
- Webcam nếu chạy các demo realtime
- Git và DVC nếu cần tải dataset/model
- Windows, macOS hoặc Linux

### Thiết lập môi trường

```bash
git clone https://github.com/ndhoang2310/DAP_GR3_SUM26.git
cd DAP_GR3_SUM26

python -m venv .venv
```

Kích hoạt môi trường trên macOS/Linux:

```bash
source .venv/bin/activate
```

Kích hoạt môi trường trên Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Cài đặt dependency và tải dữ liệu được quản lý bằng DVC:

```bash
pip install -r requirements.txt
dvc pull
```

> `dvc pull` cần quyền truy cập remote dữ liệu của dự án. Người chỉ muốn thử sản phẩm có thể dùng [web demo](https://orange-credit-f323.ndhoang2310.workers.dev/) mà không cần cài đặt.

## Chạy nhanh

### Web app trên máy local

```bash
cd deployment/scripts
python -m http.server 8000
```

Mở `http://localhost:8000`. Quyền camera trên thiết bị khác trong mạng LAN thường yêu cầu HTTPS; URL triển khai chính thức đã đáp ứng yêu cầu này.

### TimeSeries benchmark và realtime demo

```bash
python src/preprocessing/sequence_preprocessing.py
python src/models/TimeSeries/main_benchmark.py
python src/models/TimeSeries/realtime_preview.py
```

Chạy realtime với video có sẵn:

```bash
python src/models/TimeSeries/realtime_preview.py --input path/to/video.mp4
```

### Image Model pipeline

```bash
# Tạo shared split theo video_id
python src/preprocessing/create_image_split.py

# Tiền xử lý
python src/preprocessing/image_preprocessing.py
python src/preprocessing/image_preprocessing_hog_only.py
python src/preprocessing/preprocess_cnn.py

# Huấn luyện
python src/models/Image/train_ml.py
python src/models/Image/train_ml_hog_only.py
python src/models/Image/train_dl.py

# Đánh giá
python src/models/Image/tune_cnn_threshold.py
python src/models/Image/image_benchmark.py
python src/models/Image/measure_latency.py

# Webcam demo
python src/models/Image/realtime_cnn_webcam.py
```

### Thu thập và gán nhãn dữ liệu

```bash
python src/data_collection/collect_video.py
python src/data_collection/extract_eyes.py
python src/data_collection/label_tool.py --mode auto
python src/data_collection/label_tool.py --mode review
```

Quy trình đầy đủ, quy ước đặt tên và hướng dẫn đồng bộ DVC nằm trong [src/data_collection/workflow_guide.md](src/data_collection/workflow_guide.md).

## Cấu trúc repository

```text
DAP_GR3_SUM26/
├── deployment/
│   ├── assets/                  # Model/asset phục vụ deployment
│   └── scripts/                 # BlinkGuard HTML, CSS, JS và SVM đã export
├── notebooks/                   # EDA và thử nghiệm mô hình
├── reports/                     # Báo cáo, kiến trúc và biểu đồ đánh giá
├── src/
│   ├── data_collection/         # Thu thập, trích xuất, gán nhãn và DVC sync
│   ├── preprocessing/           # Image split và sequence preprocessing
│   └── models/
│       ├── Image/               # SVM/CNN image pipeline
│       └── TimeSeries/          # EAR models, benchmark và realtime preview
├── dataset_master.dvc           # Dataset chính được quản lý bằng DVC
├── processed_image.dvc          # Artifact tiền xử lý ảnh
├── config.py                    # Cấu hình chung
├── requirements.txt             # Python dependencies
└── README.md
```

## Tài liệu chính

| Nội dung | Tài liệu |
| --- | --- |
| Thu thập và quản lý dữ liệu | [workflow_guide.md](src/data_collection/workflow_guide.md) |
| Tiền xử lý chuỗi EAR | [report_sequence_preprocessing.md](src/preprocessing/report_sequence_preprocessing.md) |
| Benchmark Image Model | [report_image.md](reports/report_image.md) |
| Benchmark TimeSeries | [report_model_benchmark.md](src/models/TimeSeries/report_model_benchmark.md) |
| Web deployment | [web_client_side_deployment_report.md](reports/web_client_side_deployment_report.md) |
| Kiến trúc web | [client_side_deployment_architecture.md](reports/client_side_deployment_architecture.md) |

## Lưu ý về quyền riêng tư và phạm vi sử dụng

- Web app xử lý camera ở phía client và không chủ động tải frame khuôn mặt lên máy chủ.
- Đây là dự án nghiên cứu/học thuật, không phải thiết bị y tế hoặc hệ thống an toàn đã được chứng nhận.
- Không nên sử dụng BlinkGuard làm cơ chế cảnh báo duy nhất trong các tình huống có rủi ro cao như lái xe hoặc vận hành máy móc.

---

**Live demo:** [orange-credit-f323.ndhoang2310.workers.dev](https://orange-credit-f323.ndhoang2310.workers.dev/)
