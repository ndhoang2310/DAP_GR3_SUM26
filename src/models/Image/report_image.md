# Báo Cáo Đánh Giá & Thử Nghiệm Mô Hình: Image-based Blink Detection

## 1. Hướng dẫn triển khai

> **💡 Lưu ý:** Do sự khác biệt về đặc trưng đầu vào (HOG cho SVM vs Raw Pixels cho CNN), quy trình chuẩn bị dữ liệu cần được thực hiện riêng biệt.

1. **Tiền xử lý:**
   - Chạy `src/preprocessing/image_preprocessing.py` (Trích xuất HOG cho SVM).
   - Chạy `src/preprocessing/preprocess_cnn.py` (Chuẩn bị Tensor 24x24 cho CNN).
2. **Huấn luyện:**
   - Chạy `src/models/Image/train_ml.py` (SVM).
   - Chạy `src/models/Image/train_dl.py` (CNN).
3. **Benchmark:** Chạy `src/models/Image/measure_latency.py` để lấy kết quả thời gian thực.

## 2. Thông số thực nghiệm (Phân tách theo mô hình)

### 🔹 Mô hình Machine Learning (SVM)

* **Dữ liệu:** Sử dụng 8,292 mẫu HOG features.
* **Phân bổ:** Train/Test (80/20) - *Do SVM sử dụng Cross-validation hoặc tập test độc lập, không cần tập Validation riêng biệt như Deep Learning.*

### 🔹 Mô hình Deep Learning (CNN)

* **Dữ liệu:** 8,292 ảnh xám 24x24.
* **Phân bổ:** Train (64%) - Val (16%) - Test (20%).
* *Validation set được sử dụng để điều chỉnh Learning Rate và thực hiện Early Stopping.*

## 3. Kiến trúc mô hình & Kỹ thuật tối ưu hóa

### 🔹 Hướng 1: Machine Learning (SVM + HOG)

Sử dụng bộ trích xuất đặc trưng **HOG (Histogram of Oriented Gradients)** kết hợp với chỉ số EAR trung bình.

* **Kỹ thuật tối ưu (SMOTE):** Vì dữ liệu mắt mở (Open) chiếm ưu thế, chúng tôi áp dụng **SMOTE (Synthetic Minority Over-sampling Technique)** để tạo các mẫu giả lập cho lớp mắt nhắm (Closed). Điều này giúp SVM xây dựng ranh giới quyết định (decision boundary) cân bằng hơn, tránh việc mô hình bị bias vào lớp đa số.

### 🔹 Hướng 2: Deep Learning (Custom CNN)

Kiến trúc CNN được thiết kế để tự học đặc trưng hình thái mí mắt.

* **Kỹ thuật tối ưu (Data Augmentation):** Áp dụng chuỗi biến đổi hình ảnh: `RandomRotation` (xoay), `RandomZoom` (phóng đại), `RandomTranslation` (dịch chuyển).
* **Ý nghĩa:** Giúp mô hình trở nên bất biến (invariant) với các điều kiện thay đổi của góc chụp và vị trí mắt, tăng khả năng tổng quát hóa trong môi trường thực tế.
* 

## 4. Kết quả Benchmark & Đánh giá hiệu năng

| Mô hình              |     Accuracy     |     Macro F1     |      Latency      | Đánh giá sơ bộ                         |
| :--------------------- | :--------------: | :--------------: | :---------------: | :------------------------------------------ |
| **SVM (RBF)**    | **96.50%** |      0.9432      | **0.86 ms** | Cực nhanh, ổn định trên CPU.           |
| **CNN (Custom)** |      96.32%      | **0.9436** |     46.18 ms     | Khả năng tự học đặc trưng tốt hơn. |

### Nhận xét & Đánh giá chuyên sâu:

1. **Về độ chính xác (Accuracy):** Cả hai mô hình đều đạt ngưỡng ~96.5%, chứng minh rằng vùng mắt 24x24 chứa đủ thông tin để phân biệt nhị phân (nhắm/mở).
2. **Về khả năng cân bằng (Macro F1):** CNN nhỉnh hơn không đáng kể (0.9436 so với 0.9432). Điều này cho thấy với kích thước ảnh nhỏ, các đặc trưng "thủ công" (HOG) vẫn rất mạnh mẽ, không thua kém nhiều so với việc tự học đặc trưng của CNN.
3. **Về hiệu năng thời gian (Latency):** Đây là điểm phân hóa lớn nhất. SVM nhanh hơn gấp 50 lần. Nếu ứng dụng của bạn cần chạy song song với các tác vụ nặng (như trình duyệt, code IDE), SVM là lựa chọn mang lại trải nghiệm mượt mà nhất.

## 5. Phân tích ma trận nhầm lẫn (Confusion Matrix)

Dưới đây là so sánh trực diện khả năng phân loại giữa hai mô hình:

| **Mô hình** | **True Open (Dự đoán sai)** | **True Closed (Dự đoán sai)** |
| ------------------- | ------------------------------------ | -------------------------------------- |
| **SVM (HOG)** | 12                                   | **46**                           |
| **CNN (Raw)** | **39**                         | 22                                     |

* **SVM:** Hoạt động rất chính xác khi mắt mở (chỉ bỏ sót 12 trường hợp), nhưng gặp khó khăn hơn khi phân loại trạng thái mắt nhắm.
* **CNN:** Có xu hướng báo nhầm mắt mở thành mắt nhắm nhiều hơn (39 trường hợp), nhưng lại vượt trội trong việc nhận diện chính xác các trạng thái mắt nhắm thực sự.

## 6. Kết luận & Định hướng

Sau quá trình thử nghiệm, chúng tôi kết luận như sau:

1. **Khả năng ứng dụng:** Cả hai mô hình đều đạt độ chính xác ấn tượng trên 96%. Sự khác biệt nằm ở triết lý triển khai:
   * **Ưu tiên hiệu năng hệ thống (Real-time Efficiency):** Chọn  **SVM + HOG** . Đây là giải pháp phù hợp nhất cho các thiết bị văn phòng, máy tính cấu hình yếu nhờ độ trễ gần như bằng không.
   * **Ưu tiên độ bền vững (Robustness):** Chọn  **CNN** . Với khả năng tự trích xuất đặc trưng và Data Augmentation, CNN là nền tảng tốt để mở rộng cho các bài toán phức tạp hơn (ví dụ: cảnh báo ngủ gật, theo dõi mệt mỏi lâu dài).
2. **Khắc phục lỗi:** Để triệt tiêu các sai số trong bảng ma trận trên, chúng tôi khuyến nghị áp dụng kỹ thuật **Temporal Smoothing** (Làm mịn theo thời gian) trong giai đoạn hậu xử lý. Bằng cách lấy kết quả bầu chọn trên cửa sổ trượt 3-5 frame, hệ thống sẽ loại bỏ hoàn toàn các lỗi nhiễu ngẫu nhiên, nâng cao độ tin cậy của cảnh báo lên mức tối đa.

*Người thực hiện: Nhã Trân*
