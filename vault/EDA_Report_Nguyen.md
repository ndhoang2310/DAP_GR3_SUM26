# Báo Cáo Phân Tích Dữ Liệu Hình Ảnh Sâu (Deep Image EDA)
**Người thực hiện:** Nguyên (M05)
**Ngày thực hiện:** 08/06/2026
**Mục tiêu:** Phân tích đặc trưng thị giác của các lớp Open/Closed bằng Average Image Analysis và Pixel Intensity Histogram, đánh giá sự phân bố dữ liệu giữa các thành viên, tìm ra bộ tham số trích xuất đặc trưng tối ưu (HOG), và bàn giao Interface Contract cho bước Modeling.

---

## 1. Giai Đoạn 1: Đánh Giá Dữ Liệu Gốc (Raw Dataset Analysis)

### Mục tiêu & Cách thức

![alt text](image-2.png)
Thực hiện phân tích trên bộ dữ liệu gốc (16.461 ảnh) nhằm tìm hiểu "bức tranh thực tế" của tập dữ liệu sau thu thập. Tập dữ liệu được chia làm 4 nhóm để so sánh: `M05-Open`, `M05-Closed`, `Others-Open`, `Others-Closed`.
Phương pháp phân tích bao gồm:
*   **Average Image Analysis:** Tính toán Ảnh trung bình (Mean Image) và Ảnh phương sai (Variance Image) đã được resize về 24x24 pixel.
*   **Pixel Intensity Histogram:** Phân tích phân bổ cường độ sáng pixel để đánh giá mức độ chênh lệch ánh sáng giữa các thành viên và các lớp.

### Nhận định & Vấn đề phát hiện
*   **Về Hình dáng (Mean Image):** Mắt nhắm và mắt mở vẫn có khác biệt cấu trúc, tuy nhiên mức độ ổn định giữa các nhóm không đồng đều do số lượng ảnh và điều kiện chụp khác nhau.
*   **Vấn đề Mất cân bằng (Bias):** Dữ liệu của M05 chiếm đa số nhẹ với **8.628/16.461 ảnh (52,41%)**, trong đó `M05-Open` có **7.596 ảnh** so với `M05-Closed` chỉ **1.032 ảnh**. Kết quả thật cho thấy bias ánh sáng không phải là M05 sáng hơn Others: độ sáng trung bình lần lượt là `M05-Open` **67,20**, `M05-Closed` **59,61**, `Others-Open` **86,92**, và `Others-Closed` **89,00**. Nhóm Others sáng hơn và phân tán mạnh hơn M05.
*   **Hệ quả:** Nếu sử dụng toàn bộ dữ liệu gốc để huấn luyện mà không xử lý, mô hình có nguy cơ học lệch theo contributor và điều kiện ánh sáng, đặc biệt là sự chênh lệch lớn giữa `M05-Open` và các nhóm còn lại, thay vì chỉ học đặc trưng mắt mở/nhắm.

![alt text](image.png)

![alt text](image-1.png)
---

## 2. Giai Đoạn 2: Thực Nghiệm Tiêu Chuẩn Vàng (Golden Standard)

### Mục tiêu & Cách thức
Để loại bỏ Bias từ M05 và tìm ra "Khuôn mẫu chung nhất" (Universal Pattern) của toàn bộ nhóm, một tập dữ liệu "Tiêu chuẩn vàng" đã được xây dựng.
*   **Sampling Logic:** Lấy ngẫu nhiên và công bằng tuyệt đối đúng **125 ảnh mỗi lớp cho MỖI thành viên (M01-M06)**.
*   **Quy mô:** Tập `Gold-Open` (750 ảnh) và `Gold-Closed` (750 ảnh).
*   **Phân tích:** Tính lại Mean/Variance và vẽ biểu đồ phân phối độ sáng (Pixel Intensity Histogram/KDE) cho 2 tập này.

### Phân tích Số liệu & Kết luận
Sau khi lấy mẫu cân bằng, đỉnh phân phối độ sáng (Peak Intensity) đã dịch chuyển về vùng dung hòa:
*   `Gold-Closed`: Đỉnh ở mức **51.27**
*   `Gold-Open`: Đỉnh ở mức **79.43**

**Kết luận chuyên môn:**
1.  Ảnh hưởng của Bias ở Giai đoạn 1 là có thật và rất lớn.
2.  Tập Golden Standard thể hiện rõ sự phân tách giữa hai lớp (Mắt nhắm tối hơn mắt mở).
3.  Do tổng thể dữ liệu vẫn có xu hướng thiếu sáng và chênh lệch lớn giữa các cá nhân, bắt buộc phải sử dụng đặc trưng hình khối (HOG) kết hợp với thuật toán Chuẩn hóa độ sáng (StandardScaler) cho Modeling.

![Mean_Variance Grid](Mean_Variance_Grid.png)

![alt text](Pixel_Intensity_Histogram.png)

---

## 3. Phân Tích Phụ: Individual Bias Analysis (Độ lệch Cá nhân)

Để minh chứng rõ hơn cho quyết định ở Giai đoạn 2, một phân tích Boxplot đã được thực hiện để so sánh trực diện dải phân phối cường độ sáng của 6 thành viên (M01-M06), mỗi người lấy mẫu 200 ảnh.

![alt text](Individual_Bias_Boxplot.png)

---

## 4. Giai Đoạn 3: Tối Ưu Hóa Tham Số HOG (HOG Parameter Tuning)

### Vấn đề
Mô hình cần một bộ tham số trích xuất HOG chung cho cả 2 lớp (Open/Closed) sao cho nó bất biến với ánh sáng nhưng vẫn bắt được nét cong của con ngươi và đường ngang của mí mắt.

### Thuật toán Đánh giá (Machine Learning Approach)
Thay vì đánh giá bằng mắt thường (dễ mang tính chủ quan), một mô hình **Linear SVM** kết hợp **Cross-Validation (5-Fold)** đã được sử dụng trực tiếp trên tập dữ liệu Golden Standard (1500 ảnh) để "chấm điểm" khả năng phân tách của 3 bộ tham số:
*   **Set 1 (8, 4x4, 1x1) - 288 features:** Kém nhất (81.87%). Thiếu chuẩn hóa khối khiến dữ liệu dễ nhiễu.
*   **Set 2 (9, 8x8, 2x2) - 144 features:** Khá (87.47%). Chạy nhanh nhưng ô lưới 8x8 quá to trên ảnh 24x24 làm mất chi tiết mí mắt mảnh.
*   **Set 3 (9, 4x4, 2x2) - 900 features:** **Tốt nhất (89.20%)**. Cân bằng hoàn hảo giữa việc giữ chi tiết nét mí mắt và chuẩn hóa bóng đổ.

![alt text](OPEN_HOG_Visualization_Comparison.png)

![alt text](CLOSE_HOG_Visualization_Comparison.png)

---

## 5. Interface Contract (Giao Ước Bàn Giao)

Dựa trên toàn bộ quá trình EDA, dưới đây là Giao ước bàn giao cho bộ phận Tiền xử lý & Huấn luyện mô hình (Kiệt).

```python
# ============================================================
# INTERFACE CONTRACT (EDA TO PIPELINE)
# ============================================================
from skimage.feature import hog

HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (4, 4),
    'cells_per_block': (2, 2),
    'feature_vector': True
}

def extract_hog_features(img_gray):
    """
    Hàm trích xuất đặc trưng HOG đã được tối ưu hóa.
    Lưu ý từ EDA: 
    1. Ảnh ĐẦU VÀO bắt buộc phải là Grayscale.
    2. Kích thước bắt buộc phải là 24x24.
    3. Cần chuẩn hóa độ sáng (StandardScaler) trên feature vector HOG để giảm thiểu Bias cá nhân.
    """
    features = hog(img_gray, **HOG_PARAMS)
    return features
```