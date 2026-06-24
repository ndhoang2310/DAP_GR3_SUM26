
# 👁️ Eye Blink Data Collection Pipeline

> Hệ thống thu thập, trích xuất và gán nhãn dữ liệu mắt (Mở/Nhắm) phục vụ cho bài toán nhận diện nháy mắt (Blink Detection).
> **Dự án**: Cảnh báo sức khỏe người dùng qua tần suất nháy mắt.
> **Trạng thái hiện tại**: Giai đoạn thu thập và tiền xử lý dữ liệu (Data Collection & Labeling).

---

## 📋 Mục Lục

- [Tổng Quan Dự Án](#tổng-quan-dự-án)
- [Kiến Trúc &amp; Tính Năng Chức Năng](#kiến-trúc--tính-năng)
- [Cài Đặt Hệ Thống](#cài-đặt-hệ-thống)
- [Hướng Dẫn Quy Trình Thực Hiện](#hướng-dẫn-quy-trình-thực-hiện)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)

---

## Tổng Quan Dự Án

Đây là nền tảng khởi đầu cho dự án **Cảnh báo sức khỏe người dùng**. Chức năng chính của phần này là thu thập và chuẩn bị một bộ dữ liệu (dataset) chất lượng cao về hình ảnh mắt của người dùng.

Hệ thống sử dụng **MediaPipe Face Mesh** để quét khuôn mặt qua webcam, tự động định vị và cắt chính xác khu vực mắt trái/phải thành các hình ảnh kích thước 24x24 pixel. Những hình ảnh này, kết hợp với nhãn (Nhắm/Mở) được gán thủ công hoặc tự động qua các công cụ tích hợp, sẽ đóng vai trò cốt lõi trong việc huấn luyện các mô hình Machine Learning sau này (như SVM, Random Forest).

---

## Kiến Trúc & Tính Năng

**1. Ghi hình Webcam (Video Collection):** Thu thập các luồng video thực tế của người dùng với nhiều điều kiện ánh sáng và góc độ khác nhau.
**2. Trích xuất tự động (Eye Extraction):** Bỏ qua các phương pháp truyền thống chậm chạp, hệ thống sử dụng sức mạnh của MediaPipe để tracking khuôn mặt với độ trễ cực thấp.
**3. Công cụ Gán nhãn (Labeling Tool):** Bộ công cụ console tiện dụng cho phép người dùng gán nhãn hàng loạt tự động bằng công thức EAR và duyệt lại (review) bằng phím tắt một cách trực quan.
**4. Đồng bộ Dữ liệu tự động (DVC):** Quản lý phiên bản dữ liệu nặng bằng DVC, tự động hóa hoàn toàn quy trình tải lên/tải xuống qua Google Drive giữa các thành viên và Leader.

---

## Cài Đặt Hệ Thống

### 1. Yêu cầu môi trường

- **Python 3.8+** (Khuyên dùng Python 3.10)
- Webcam (camera laptop hoặc webcam rời)
- Hệ điều hành: Windows / macOS / Linux

### 2. Thiết lập dự án

```bash
# Khởi tạo virtual environment
python -m venv venv

# Kích hoạt venv (trên Windows)
venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

> **Lưu ý**: Nhờ sử dụng thư viện hiện đại MediaPipe, việc cài đặt diễn ra cực kỳ nhanh chóng mà không cần các thao tác build phức tạp từ CMake như trước.

---

## Hướng Dẫn Quy Trình Thực Hiện

Để giữ cho tài liệu README tổng quan và dễ đọc, toàn bộ **hướng dẫn chi tiết từng bước** về cách thu thập video, cách trích xuất, và các phím tắt dùng để gán nhãn dữ liệu đã được tách riêng vào thư mục `processing`.

👉 **Vui lòng xem chi tiết tại:** [Hướng Dẫn Quy Trình Xử Lý Dữ Liệu](src/data_collection/workflow_guide.md)

*(Trong file này sẽ chứa toàn bộ các dòng lệnh cần thiết để chạy pipeline của bạn một cách trơn tru)*

---

## Cấu Trúc Thư Mục

```text
eye-blink-detection/
├── data/
│   └── raw_videos/               # Chứa video gốc (quay từ người dùng)
├── dataset/
│   ├── raw_eyes/                 # Ảnh mắt thô & metadata được tạo ra tự động
│   ├── train/                    # Bộ dữ liệu 80% để huấn luyện
│   └── test/                     # Bộ dữ liệu 20% để kiểm thử
├── src/
│   └── data_collection/          # Nơi chứa mã nguồn chính của pipeline
│       ├── collect_video.py    
│       ├── extract_eyes.py     
│       ├── label_tool.py
│       ├── dvc_sync.py           # Script tự động đẩy dữ liệu lên DVC & Github (Dành cho Member)
│       ├── dvc_pull_and_merge.py # Script tự động tải và gộp dữ liệu từ DVC (Dành cho Leader)
│       ├── merge_datasets.py   
│       └── workflow_guide.md     # 📖 TÀI LIỆU HƯỚNG DẪN QUY TRÌNH CHI TIẾT
├── config.py                     # File cấu hình (kích thước, EAR threshold,...)
├── requirements.txt              # Danh sách thư viện cần thiết
└── README.md                     # Tài liệu tổng quan dự án (File này)
```




## Image Model Pipeline

Image Model là một hướng tiếp cận trong hệ thống **Blink Detection**, sử dụng ảnh crop vùng mắt để phân loại hai trạng thái:

```text
open
closed
```

Trong phần Image Model, nhóm triển khai và so sánh ba mô hình:

| Model        | Input                      | Mục đích                        |
| ------------ | -------------------------- | ---------------------------------- |
| SVM HOG-only | HOG features               | Ablation study                     |
| SVM HOG+EAR  | HOG + EAR average          | Baseline ML nhẹ                   |
| CNN          | Grayscale eye image 24×24 | Mô hình chính cho realtime demo |

Pipeline Image Model sử dụng **shared split theo `video_id`** để giảm data leakage. Cách chia này đảm bảo ảnh từ cùng một video không xuất hiện đồng thời trong train, validation hoặc test set. Nhờ đó, benchmark giữa SVM và CNN công bằng hơn vì các mô hình được đánh giá trên cùng một test split.

---

### Dataset Split

Tổng số sample hợp lệ:

| Class  | Samples |
| ------ | ------: |
| Open   |    6634 |
| Closed |    1658 |
| Total  |    8292 |

Kết quả chia dữ liệu theo `video_id`:

| Split      | Samples | Open | Closed | Unique Videos |
| ---------- | ------: | ---: | -----: | ------------: |
| Train      |    5124 | 4150 |    974 |            16 |
| Validation |    1396 | 1135 |    261 |             4 |
| Test       |    1772 | 1349 |    423 |             5 |

Leakage check:

```text
train_video_ids ∩ val_video_ids = 0
train_video_ids ∩ test_video_ids = 0
val_video_ids ∩ test_video_ids = 0
```

---

### Run Image Model Pipeline

Tạo shared split:

```bash
py .\src\preprocessing\create_image_split.py
```

Preprocess dữ liệu cho SVM và CNN:

```bash
py .\src\preprocessing\image_preprocessing.py
py .\src\preprocessing\image_preprocessing_hog_only.py
py .\src\preprocessing\preprocess_cnn.py
```

Train các mô hình:

```bash
py .\src\models\Image\train_ml.py
py .\src\models\Image\train_ml_hog_only.py
py .\src\models\Image\train_dl.py
```

Tune threshold, benchmark và đo latency:

```bash
py .\src\models\Image\tune_cnn_threshold.py
py .\src\models\Image\image_benchmark.py
py .\src\models\Image\measure_latency.py
```

Chạy realtime webcam demo:

```bash
py .\src\models\Image\realtime_cnn_webcam.py
```

Nhấn `q` để thoát webcam demo.

---

### Model Outputs

Các model sau khi train được lưu tại:

```text
models/svm_model.pkl
models/svm_hog_only_model.pkl
models/best_cnn.keras
models/cnn_blink_model.keras
```

Trong đó:

```text
best_cnn.keras = checkpoint tốt nhất trên validation set
cnn_blink_model.keras = model CNN cuối cùng dùng cho benchmark và realtime demo
```

Threshold tốt nhất của CNN được lưu tại:

```text
processed_image/best_cnn_threshold.txt
```

---

### Current Benchmark Results

Các mô hình được đánh giá trên cùng test split.

| Model              |         Accuracy |    AUC |          Open F1 |        Closed F1 |         Macro F1 |
| ------------------ | ---------------: | -----: | ---------------: | ---------------: | ---------------: |
| SVM HOG-only       |           0.8888 | 0.9471 |           0.9294 |           0.7384 |           0.8339 |
| SVM HOG+EAR        |           0.8911 | 0.9483 |           0.9309 |           0.7423 |           0.8366 |
| CNN threshold=0.45 | **0.9086** | 0.9313 | **0.9424** | **0.7781** | **0.8603** |

Confusion matrices được lưu tại:

```text
reports/figures/
```

Detailed Image Model report:

```text
reports/report_image.md
```

---

### Latency Result

Latency được đo trên 100 samples.

| Model       |         Latency |
| ----------- | --------------: |
| SVM HOG+EAR | 0.7502 ms/image |
| CNN         | 8.3654 ms/image |

SVM nhanh hơn CNN khoảng 11 lần. Tuy nhiên, CNN vẫn đủ nhanh để chạy realtime webcam trong demo.

---

### Realtime Webcam Demo

Realtime demo sử dụng:

```text
Model: cnn_blink_model.keras
Raw threshold: 0.45
Smoothing: hysteresis temporal smoothing
Smooth window: 7
Close threshold: 0.55
Open threshold: 0.35
Alert duration: 1.0 second
```

Realtime pipeline:

```text
webcam frame
    ↓
MediaPipe FaceMesh
    ↓
crop vùng mắt
    ↓
CNN prediction
    ↓
hysteresis temporal smoothing
    ↓
time-based closed-eye alert
```

Kết quả test realtime:

```text
Chớp mắt nhanh không kích hoạt cảnh báo.
Nhắm mắt khoảng 1 giây sẽ kích hoạt cảnh báo.
FaceMesh crop mắt ổn định.
FPS khi chạy bình thường khoảng 21–30 FPS.
Khi che mặt hoặc không detect được mặt, FPS tăng vì hệ thống bỏ qua bước CNN prediction.
```

---

### Final Image Model Choice

| Mục tiêu              | Model                    |
| ----------------------- | ------------------------ |
| Phân loại tốt nhất  | CNN threshold=0.45       |
| Baseline nhẹ và nhanh | SVM HOG+EAR              |
| Ablation study          | SVM HOG-only             |
| Realtime demo           | CNN + temporal smoothing |

Final selected realtime model:

```text
CNN with threshold 0.45, hysteresis temporal smoothing and time-based closed-eye alert.
```
