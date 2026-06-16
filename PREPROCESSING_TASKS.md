# 🎯 Hướng Dẫn & Giao Việc Tiền Xử Lý Dữ Liệu (Preprocessing Tasks)

Tài liệu này dùng để giao việc chi tiết cho **Kiệt** (Tiền xử lý Ảnh - Hướng 1) và **Bảo** (Tiền xử lý Chuỗi EAR - Hướng 2). Master Dataset đã được gộp xong và sẵn sàng trên DVC. 

Dựa trên các phân tích quan trọng từ bộ phận EDA (Trân, Nguyên, Hiền), các bước tiền xử lý dưới đây đã được tinh chỉnh để giải quyết triệt để các vấn đề của dữ liệu.

---

## 📥 1. Hướng Dẫn Tải Dataset (Dành cho cả 2 bạn)

Hiện tại, toàn bộ 8,292 ảnh và file `metadata_master.csv` đã được tự động gộp và đẩy lên hệ thống. Các bạn chạy tuần tự các lệnh sau trong terminal (nhớ mở `venv` trước):

```bash
# 1. Kéo cập nhật mới nhất từ Github (chứa file dataset_master.dvc)
git pull origin main

# 2. Kéo toàn bộ ảnh và file csv từ DVC Google Drive về máy
dvc pull
```
> **Kết quả:** Thư mục `dataset_master/` chứa `raw_eyes/` và `metadata_master.csv`.

---

## 👤 2. Nhiệm Vụ Của Kiệt (Tiền Xử Lý Hình Ảnh - Hướng 1)

**Mục tiêu:** Xây dựng pipeline biến đổi ảnh thành các vector đặc trưng phục vụ cho mô hình (SVM, CNN). 
👉 **File làm việc:** `src/preprocessing/image_preprocessing.py`

### Các bước yêu cầu (Dựa trên Phân tích EDA của Trân & Nguyên):
1. **Lọc dữ liệu hợp lệ (Từ General EDA của Trân)**: 
   - **Bắt buộc**: Không lặp mù quáng qua thư mục ảnh. Phải đọc `metadata_master.csv`, chỉ lấy các dòng có `status == 'success'` và có nhãn hợp lệ (`open` hoặc `closed`). Bỏ qua các dòng `skipped` hoặc `deleted`.
2. **Tiền xử lý cơ bản**: 
   - Load ảnh từ `raw_eyes/` dựa trên tên file trong CSV.
   - Chuyển sang Grayscale và chuẩn hóa pixel về dải `[0, 1]` (chia 255.0).
3. **Trích xuất đặc trưng HOG (Từ Deep Image EDA của Nguyên)**: 
   - Áp dụng bộ tham số tối ưu mà Nguyên đã tìm ra để chống lại sự chênh lệch ánh sáng (Bias) giữa các thành viên:
     * `orientations = 9`
     * `pixels_per_cell = (4, 4)`
     * `cells_per_block = (2, 2)`
     * `feature_vector = True`
4. **Ghép đặc trưng & Chuẩn hóa (Feature Scaling)**: 
   - Nối vector HOG với chỉ số `ear_avg` tương ứng.
   - **Bắt buộc**: Áp dụng `StandardScaler` lên toàn bộ ma trận đặc trưng cuối cùng. Nguyên đã chứng minh độ sáng trung bình của M05 và nhóm Others chênh lệch rất lớn, nếu không Scale, mô hình SVM sẽ bị nhiễu.
5. **Xử lý mất cân bằng (Từ General EDA của Trân)**:
   - Dữ liệu bị mất cân bằng lớp (thiên về `open`). Cắt tập Train/Test (80/20) và áp dụng **SMOTE** *chỉ trên tập Train* để cân bằng số lượng mẫu.
6. **Lưu kết quả**: Xuất file Numpy (`X_train_img.npy`, `y_train_img.npy`, `X_test_img.npy`, `y_test_img.npy`).

---

## 👤 3. Nhiệm Vụ Của Bảo (Tiền Xử Lý Chuỗi EAR - Hướng 2)

**Mục tiêu:** Xây dựng pipeline xử lý chuỗi thời gian EAR liên tục thành các cửa sổ trượt (sliding windows) để train mô hình phát hiện nháy mắt (LSTM, 1D-CNN).
👉 **File làm việc:** `src/preprocessing/sequence_preprocessing.py`

### Các bước yêu cầu (Dựa trên Phân tích EDA của Hiền & Trân):
1. **Đọc và Phân rã dữ liệu theo Video/Người dùng**: 
   - Đọc `metadata_master.csv`. Phải `groupby` dữ liệu theo từng video hoặc từng lần thu thập để đảm bảo chuỗi thời gian không bị nối chéo giữa các video khác nhau.
2. **Nội suy phục hồi (Từ General EDA của Trân)**: 
   - Trong CSV có các frame bị lỗi (status = `deleted` hoặc `skipped`). Việc mất frame này làm đứt gãy chuỗi thời gian.
   - **Bắt buộc**: Dựa vào `frame_index` hoặc `timestamp`, dùng thuật toán Nội suy tuyến tính (Linear Interpolation) để điền bù giá trị `ear_avg` bị thiếu trước khi cắt cửa sổ.
3. **Chuẩn hóa EAR Cá Nhân (Từ TimeSeries EDA của Hiền)**: 
   - Hiền đã chứng minh cấu trúc mắt tự nhiên của mỗi người là khác nhau (dải EAR khác nhau).
   - **Bắt buộc**: Áp dụng `MinMaxScaler` để chuẩn hóa cột `ear_avg` về dải `[0, 1]`, nhưng phải **làm riêng cho từng contributor**, tuyệt đối không Scale chung toàn bộ dataset.
4. **Cắt Cửa sổ trượt (Sliding Window)**:
   - Sử dụng kết quả phân tích động lực học nháy mắt của Hiền: Cắt chuỗi thành các cửa sổ có kích thước **$N = 7$ frames**, bước trượt (step) = 1 frame.
5. **Gán nhãn Nháy mắt (V-Shape Labeling)**:
   - Sử dụng **Static Threshold = 0.1050** do Hiền cung cấp từ biểu đồ ROC để làm mốc xác định điểm đáy (trạng thái nhắm).
   - **Quy luật dán nhãn chuyển động (Motion-based Labeling)**: Lưu ý quan trọng, dữ liệu gốc ở các frame chỉ gán nhãn tĩnh là `open` (mở) hoặc `closed` (nhắm). Nhưng bài toán của bạn là phát hiện **hành động chớp mắt (Blink)**. Hành động này được tính bắt đầu ngay từ lúc mí mắt bắt đầu di chuyển khép lại, chạm đáy, và sau đó mở lên, chứ không phải đợi đến khi mắt nhắm tịt mới gọi là chớp.
   - Do đó, một cửa sổ 7 frames được gán nhãn `Blink` (1) khi nó bao gồm trọn vẹn chuỗi động lực học hình chữ V: **Bắt đầu từ frame có EAR cao (mắt đang mở) $\to$ EAR giảm dần (chuyển động khép mí) $\to$ Điểm đáy vượt qua ngưỡng 0.1050 (nhắm hoàn toàn) $\to$ EAR tăng trở lại (mở ra)**.
   - Trái lại, nếu cửa sổ chỉ chứa các frame `closed` liên tục (EAR $< 0.1050$ trong toàn bộ 7 frames, ví dụ lúc ngủ gật), thì không có chuyển động nháy, bắt buộc phải dán nhãn `No-Blink` (0) hoặc `Long-Closure` (2) để mô hình không đếm sai.
6. **Lưu kết quả**: Xuất file Numpy 3D (`X_train_seq.npy`, `y_train_seq.npy`, `X_test_seq.npy`, `y_test_seq.npy`).
