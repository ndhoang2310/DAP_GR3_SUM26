# Báo cáo Image Model – Blink Detection

## 1. Mục tiêu

Image Model được xây dựng để phân loại ảnh crop vùng mắt thành hai lớp:

```text
open
closed
```

Mô hình này là một phần của hệ thống Blink Detection. Mục tiêu cuối cùng là hỗ trợ nhận diện trạng thái mắt theo thời gian thực qua webcam và cảnh báo khi người dùng nhắm mắt quá lâu.

Trong phần Image Model, nhóm triển khai ba hướng thử nghiệm chính:

| Mô hình    | Đầu vào            | Mục đích           |
| ------------ | --------------------- | --------------------- |
| SVM HOG-only | Đặc trưng HOG      | Thử nghiệm ablation |
| SVM HOG+EAR  | HOG + EAR trung bình | Baseline ML nhẹ      |
| CNN          | Ảnh mắt grayscale   | Hướng Deep Learning |

Mô hình được dùng cho realtime demo cuối cùng là  **CNN kết hợp threshold tuning và temporal smoothing** .

---

## 2. Tổng quan dữ liệu

Dataset được nhóm tự thu thập từ video/webcam. Mỗi sample là một ảnh crop vùng mắt và có nhãn cuối cùng.

Chỉ các sample hợp lệ được sử dụng:

```text
status == "success"
final_label thuộc {"open", "closed"}
image_path tồn tại
video_id không bị null
```

Tổng số dữ liệu hợp lệ:

| Nhãn  | Số lượng |
| ------ | ----------: |
| Open   |        6634 |
| Closed |        1658 |
| Tổng  |        8292 |

Dataset bị mất cân bằng vì trạng thái mắt mở xuất hiện nhiều hơn trạng thái mắt nhắm.

---

## 3. Chia dữ liệu theo video_id

Để giảm data leakage, dữ liệu được chia theo `video_id` thay vì chia ngẫu nhiên theo từng ảnh.

Điều này có nghĩa là các ảnh thuộc cùng một video sẽ không xuất hiện đồng thời ở train, validation hoặc test.

Kết quả chia dữ liệu:

| Tập dữ liệu | Số mẫu | Open | Closed | Số video |
| -------------- | -------: | ---: | -----: | --------: |
| Train          |     5124 | 4150 |    974 |        16 |
| Validation     |     1396 | 1135 |    261 |         4 |
| Test           |     1772 | 1349 |    423 |         5 |

Kiểm tra leakage:

```text
train_video_ids ∩ val_video_ids = 0
train_video_ids ∩ test_video_ids = 0
val_video_ids ∩ test_video_ids = 0
```

Kết quả trên cho thấy không có video nào xuất hiện đồng thời ở các tập train, validation và test.

Việc dùng chung một split cũng giúp việc so sánh giữa SVM và CNN công bằng hơn, vì cả hai mô hình đều được đánh giá trên cùng tập test.

---

## 4. Pipeline SVM HOG+EAR

Mô hình SVM HOG+EAR sử dụng đặc trưng thủ công.

Với mỗi ảnh mắt:

```text
image_path → đọc ảnh grayscale
ảnh → resize / normalize
ảnh → trích xuất đặc trưng HOG
ear_avg → thêm vào vector đặc trưng
feature vector = HOG + EAR
```

Số chiều đặc trưng:

```text
900 đặc trưng HOG + 1 đặc trưng EAR = 901 features
```

Tiền xử lý tập train:

```text
StandardScaler fit trên train
SMOTE chỉ áp dụng trên train
```

Tiền xử lý tập test:

```text
StandardScaler transform trên test
Không áp dụng SMOTE trên test
```

Cấu hình mô hình:

```text
SVC(kernel="rbf", C=10, gamma="scale", probability=True)
```

Model được lưu tại:

```text
models/svm_model.pkl
```

---

## 5. Pipeline SVM HOG-only

Mô hình HOG-only được triển khai như một ablation experiment.

Mục đích là kiểm tra xem đặc trưng EAR có thật sự giúp cải thiện kết quả của SVM hay không.

Với mỗi ảnh mắt:

```text
image_path → đọc ảnh grayscale
ảnh → resize / normalize
ảnh → trích xuất đặc trưng HOG
feature vector = HOG only
```

Số chiều đặc trưng:

```text
900 HOG features
```

Model được lưu tại:

```text
models/svm_hog_only_model.pkl
```

---

## 6. Pipeline CNN

Mô hình CNN học đặc trưng trực tiếp từ ảnh mắt grayscale.

Tiền xử lý ảnh:

```text
image_path → đọc ảnh grayscale
resize về 24 × 24
normalize về [0, 1]
expand dimension thành shape (24, 24, 1)
```

Kiến trúc CNN:

```text
Input 24 × 24 × 1
Data Augmentation
Conv2D 32
BatchNormalization
MaxPooling
Conv2D 64
BatchNormalization
MaxPooling
Flatten
Dense 64
Dropout 0.3
Dense 1 sigmoid
```

Cấu hình huấn luyện:

```text
Optimizer: Adam
Learning rate: 0.0003
Loss: binary_crossentropy
Metrics: accuracy, precision, recall, AUC
EarlyStopping monitor: val_auc
ModelCheckpoint monitor: val_auc
```

Ở lần chạy cuối cùng, CNN không dùng SMOTE và không dùng class weight. Trước đó, khi dùng balanced class weight, mô hình có xu hướng dự đoán quá nhiều về lớp `closed`. Vì vậy, pipeline cuối cùng giữ nguyên phân phối dữ liệu gốc và điều chỉnh decision boundary bằng threshold tuning.

Model được lưu tại:

```text
models/best_cnn.keras
models/cnn_blink_model.keras
```

Trong đó:

```text
best_cnn.keras = checkpoint tốt nhất trên validation set
cnn_blink_model.keras = model CNN cuối cùng dùng cho benchmark và realtime demo
```

---

## 7. Tối ưu threshold cho CNN

CNN trả về xác suất ảnh mắt thuộc lớp `closed`:

```text
prob_closed
```

Thay vì chỉ dùng threshold mặc định `0.50`, nhóm thử nhiều threshold khác nhau trên validation set.

Threshold tốt nhất theo Macro F1:

```text
0.45
```

Threshold này được lưu tại:

```text
processed_image/best_cnn_threshold.txt
```

Trong benchmark cuối cùng, CNN sử dụng:

```text
threshold = 0.45
```

---

## 8. Kết quả benchmark

Tất cả mô hình được đánh giá trên cùng một test split.

| Mô hình          |         Accuracy |    AUC |          Open F1 |        Closed F1 |         Macro F1 |      Weighted F1 |
| ------------------ | ---------------: | -----: | ---------------: | ---------------: | ---------------: | ---------------: |
| SVM HOG-only       |           0.8888 | 0.9471 |           0.9294 |           0.7384 |           0.8339 |           0.8838 |
| SVM HOG+EAR        |           0.8911 | 0.9483 |           0.9309 |           0.7423 |           0.8366 |           0.8859 |
| CNN threshold=0.45 | **0.9086** | 0.9313 | **0.9424** | **0.7781** | **0.8603** | **0.9032** |

Confusion matrix:

### SVM HOG+EAR

| True / Pred | Open | Closed |
| ----------- | ---: | -----: |
| Open        | 1301 |     48 |
| Closed      |  145 |    278 |

### SVM HOG-only

| True / Pred | Open | Closed |
| ----------- | ---: | -----: |
| Open        | 1297 |     52 |
| Closed      |  145 |    278 |

### CNN threshold=0.45

| True / Pred | Open | Closed |
| ----------- | ---: | -----: |
| Open        | 1326 |     23 |
| Closed      |  139 |    284 |

Benchmark script cũng kiểm tra và xác nhận:

```text
Shared test label check: PASSED
```

Điều này nghĩa là SVM và CNN đang được đánh giá trên cùng danh sách label test.

---

## 9. Phân tích kết quả benchmark

CNN đạt hiệu năng phân loại tổng thể tốt nhất.

So với SVM HOG+EAR:

| Metric    | SVM HOG+EAR |    CNN | Chênh lệch |
| --------- | ----------: | -----: | -----------: |
| Accuracy  |      0.8911 | 0.9086 |      +0.0175 |
| Closed F1 |      0.7423 | 0.7781 |      +0.0358 |
| Macro F1  |      0.8366 | 0.8603 |      +0.0237 |

CNN cũng giảm cả hai loại lỗi chính:

| Loại lỗi                        | SVM HOG+EAR | CNN |
| --------------------------------- | ----------: | --: |
| Open bị dự đoán thành Closed |          48 |  23 |
| Closed bị dự đoán thành Open |         145 | 139 |

Tuy nhiên, SVM HOG+EAR có AUC cao nhất:

```text
SVM HOG+EAR AUC = 0.9483
CNN AUC = 0.9313
```

Điều này cho thấy SVM vẫn có khả năng ranking tốt, dù CNN cho kết quả tốt hơn tại threshold đã chọn.

Thử nghiệm HOG-only cho thấy EAR chỉ cải thiện nhẹ:

| Mô hình    | Accuracy | Closed F1 | Macro F1 |
| ------------ | -------: | --------: | -------: |
| SVM HOG-only |   0.8888 |    0.7384 |   0.8339 |
| SVM HOG+EAR  |   0.8911 |    0.7423 |   0.8366 |

Điều này cho thấy HOG đã nắm được phần lớn thông tin hình dạng quan trọng, còn EAR chỉ bổ sung thêm một lượng thông tin nhỏ.

---

## 10. Kết quả latency

Latency được đo trên 100 samples.

| Mô hình   |                   Latency |
| ----------- | ------------------------: |
| SVM HOG+EAR | **0.7502 ms/image** |
| CNN         |           8.3654 ms/image |

SVM nhanh hơn CNN đáng kể.

Tỷ lệ tốc độ xấp xỉ:

```text
8.3654 / 0.7502 ≈ 11.15
```

Nghĩa là SVM nhanh hơn CNN khoảng 11 lần.

Tuy nhiên, latency của CNN vẫn chấp nhận được cho realtime. Ví dụ, nếu webcam chạy ở 15 FPS, mỗi frame có khoảng:

```text
1000 / 15 ≈ 66.7 ms/frame
```

CNN mất khoảng 8.37 ms cho mỗi ảnh, vẫn nằm trong giới hạn realtime. Tuy vậy, tổng latency thực tế còn phụ thuộc vào nhiều bước khác như đọc webcam, FaceMesh detection, crop mắt, preprocessing, smoothing và hiển thị giao diện.

---

## 11. Thử nghiệm realtime webcam

Sau khi benchmark và đo latency, CNN được thử nghiệm trong realtime webcam demo.

Pipeline realtime:

```text
Webcam frame
    ↓
MediaPipe FaceMesh
    ↓
Crop vùng mắt
    ↓
CNN prediction
    ↓
Temporal smoothing
    ↓
Time-based closed-eye alert
```

Realtime demo sử dụng:

```text
Model: cnn_blink_model.keras
Raw threshold: 0.45
Smoothing: Hysteresis moving average
Smooth window: 7
Close threshold: 0.55
Open threshold: 0.35
Alert duration: 1.0 second
```

Kết quả quan sát realtime:

| Trường hợp test            | Kết quả                                        |
| ----------------------------- | ------------------------------------------------ |
| Chớp mắt nhanh              | Không cảnh báo sai                            |
| Nhắm mắt khoảng 1 giây    | Cảnh báo đúng                                |
| Mắt mở bình thường       | Có một vài dự đoán nhầm closed nhưng ít |
| Mắt đảo liên tục         | Raw prediction khá ổn định                   |
| FaceMesh crop mắt            | Ổn định                                       |
| FPS bình thường            | Khoảng 21–30 FPS                               |
| Khi che mặt / không predict | FPS tăng do bỏ qua bước CNN inference        |

Kết quả realtime cho thấy CNN có thể chạy trong điều kiện webcam thực tế. Temporal smoothing giúp dự đoán ổn định hơn và hạn chế việc cảnh báo do lỗi đơn frame.

---

## 12. Temporal smoothing

Temporal smoothing chỉ được sử dụng ở giai đoạn realtime inference. Nó không phải là một phần của training.

Phiên bản ban đầu dùng moving average đơn giản. Tuy nhiên, output sau smoothing vẫn có thể bị nhảy khi xác suất nằm gần threshold.

Phiên bản cuối cùng sử dụng hysteresis moving average:

```text
Nếu trạng thái hiện tại là OPEN:
    chỉ chuyển sang CLOSED khi avg_prob >= 0.55

Nếu trạng thái hiện tại là CLOSED:
    chỉ chuyển về OPEN khi avg_prob <= 0.35
```

Cách này giúp giảm việc trạng thái bị nhảy liên tục giữa `open` và `closed`.

Logic cảnh báo cũng được đổi từ đếm số frame sang dựa trên thời gian thật:

```text
Nếu trạng thái sau smoothing là CLOSED liên tục ít nhất 1.0 giây:
    kích hoạt cảnh báo
```

Cách này ổn định hơn vì FPS webcam có thể thay đổi trong lúc chạy.

---

## 13. Phân tích lỗi

Lỗi còn lại chính là bỏ sót một số ảnh mắt nhắm.

Từ confusion matrix của CNN:

```text
True Closed predicted as Open = 139
```

Điều này nghĩa là vẫn có một số sample mắt nhắm bị dự đoán thành mắt mở.

Một số nguyên nhân có thể gồm:

```text
eye crop chưa hoàn toàn thẳng hoặc chưa đồng nhất
hình dạng mắt nhắm khác nhau giữa các người dùng
điều kiện ánh sáng thay đổi
một số ảnh closed giống trạng thái mắt hơi mở
dataset bị mất cân bằng về phía lớp open
```

Ngoài ra, trong realtime vẫn có một số dự đoán nhầm `closed` khi mắt đang mở. Tuy nhiên, các lỗi này xuất hiện không nhiều và thường không kích hoạt cảnh báo vì đã có temporal smoothing và time-based alert để lọc các lỗi ngắn.

---

## 14. Đề xuất lựa chọn mô hình

| Mục tiêu                           | Mô hình đề xuất     |
| ------------------------------------ | ------------------------ |
| Hiệu năng phân loại tốt nhất   | CNN threshold=0.45       |
| Baseline nhẹ, chạy nhanh trên CPU | SVM HOG+EAR              |
| So sánh ablation                    | SVM HOG-only             |
| Realtime demo                        | CNN + temporal smoothing |

Mô hình được chọn cho realtime demo:

```text
CNN với threshold 0.45, hysteresis temporal smoothing và time-based closed-eye alert.
```

CNN đạt kết quả tốt nhất về phân loại:

```text
Accuracy = 0.9086
Closed F1 = 0.7781
Macro F1 = 0.8603
Latency = 8.3654 ms/image
```

Dù SVM nhanh hơn đáng kể, CNN cho chất lượng phân loại tốt hơn và vẫn đủ nhanh để chạy realtime webcam.

---

## 15. Kết luận

Pipeline Image Model sau khi cải tiến đáng tin cậy hơn so với cách chia random ban đầu vì đã sử dụng shared video-level split để giảm leakage. Benchmark cũng công bằng hơn vì SVM và CNN được đánh giá trên cùng tập test.

Kết quả cuối cùng cho thấy CNN có hiệu năng tổng thể tốt nhất, trong khi SVM HOG+EAR vẫn là một baseline nhẹ và rất nhanh. Thử nghiệm realtime webcam xác nhận rằng CNN có thể chạy thực tế với FaceMesh eye cropping, temporal smoothing và closed-eye alert logic.

Vì vậy, CNN được chọn làm mô hình chính cho realtime demo, còn SVM được giữ lại như một baseline nhanh để so sánh.
