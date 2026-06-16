# 🎯 Hướng Dẫn & Giao Việc Tiền Xử Lý Dữ Liệu (Preprocessing Tasks)

Tài liệu này dùng để giao việc chi tiết cho **Kiệt** (Tiền xử lý Ảnh - Hướng 1) và **Bảo** (Tiền xử lý Chuỗi EAR - Hướng 2). Master Dataset đã được gộp xong và sẵn sàng trên DVC. Các bạn vui lòng làm theo hướng dẫn bên dưới.

---

## 📥 1. Hướng Dẫn Tải Dataset (Dành cho cả 2 bạn)

Hiện tại, toàn bộ 8,292 ảnh và file `metadata_master.csv` đã được tự động gộp và đẩy lên hệ thống. Các bạn không cần phải tự chạy script gộp nữa. Để lấy dữ liệu về máy để code, các bạn chạy tuần tự các lệnh sau trong terminal (nhớ mở `venv` trước):

```bash
# 1. Kéo cập nhật mới nhất từ Github (chứa file dataset_master.dvc)
git pull origin main

# 2. Kéo toàn bộ ảnh và file csv từ DVC Google Drive về máy
dvc pull
```
> **Kết quả mong đợi:** Sau khi chạy xong, trong thư mục gốc của dự án sẽ xuất hiện thư mục `dataset_master/` chứa thư mục con `raw_eyes/` (chứa toàn bộ ảnh) và file `metadata_master.csv`.

---

## 👤 2. Nhiệm Vụ Của Kiệt (Tiền Xử Lý Hình Ảnh - Hướng 1)

**Mục tiêu:** Xây dựng pipeline xử lý dữ liệu ảnh thành các vector đặc trưng phục vụ cho mô hình phân loại (SVM, CNN). Viết code vào file `src/preprocessing/image_preprocessing.py`.

### Các bước yêu cầu (Dựa trên Giao ước 1 từ Nguyên):
1. **Đọc dữ liệu**: Duyệt qua toàn bộ ảnh trong `dataset_master/raw_eyes/`.
2. **Tiền xử lý cơ bản**: 
   - Chuyển đổi ảnh sang Grayscale (ảnh xám).
   - Đảm bảo kích thước ảnh là 24x24 pixel.
   - Chuẩn hóa giá trị pixel bằng cách chia cho `255.0` (đưa về dải `[0, 1]`).
3. **Trích xuất đặc trưng HOG**: Bắt buộc sử dụng bộ tham số tối ưu sau để trích xuất:
   - `orientations = 9`
   - `pixels_per_cell = (4, 4)`
   - `cells_per_block = (2, 2)`
   - `feature_vector = True`
4. **Ghép đặc trưng & Chuẩn hóa**: 
   - Nối vector đặc trưng HOG vừa trích xuất với chỉ số EAR cá nhân tương ứng (lấy từ `metadata_master.csv`).
   - Bắt buộc chạy `StandardScaler` trên toàn bộ ma trận đặc trưng sau khi ghép để cân bằng độ lớn của HOG và EAR.
5. **Xử lý mất cân bằng lớp (SMOTE)**:
   - Cắt tập dữ liệu thành Train/Test (tỷ lệ 80/20).
   - Chỉ áp dụng SMOTE trên tập **Train** (sau khi đã trích xuất HOG) để cân bằng số lượng mẫu Nhắm và Mở.
6. **Lưu kết quả đầu ra**: Lưu các ma trận hoàn chỉnh thành file Numpy (`X_train_img.npy`, `y_train_img.npy`, `X_test_img.npy`, `y_test_img.npy`).

---

## 👤 3. Nhiệm Vụ Của Bảo (Tiền Xử Lý Chuỗi EAR - Hướng 2)

**Mục tiêu:** Xây dựng pipeline xử lý chuỗi thời gian EAR liên tục thành các cửa sổ trượt (sliding windows) để train mô hình phát hiện nháy mắt (LSTM, 1D-CNN). Viết code vào file `src/preprocessing/sequence_preprocessing.py`.

### Các bước yêu cầu (Dựa trên Giao ước 2 từ Hiền):
1. **Đọc dữ liệu**: Tải file `dataset_master/metadata_master.csv`.
2. **Nội suy phục hồi (Linear Interpolation)**: 
   - Trong quá trình thu thập, một số frame bị nhiễu đã bị xóa (discard). Việc này làm chuỗi thời gian bị đứt đoạn.
   - Bắt buộc phải kiểm tra cột `frame_index` hoặc `timestamp_sec` và dùng Nội suy tuyến tính (Linear Interpolation) để bù các giá trị `ear_avg` bị thiếu trước khi cắt cửa sổ.
3. **Chuẩn hóa EAR Cá Nhân**: 
   - Sử dụng `MinMaxScaler` hoặc tính toán Min-Max thủ công **riêng cho từng người (từng contributor)** để đưa cột `ear_avg` về dải `[0, 1]`. Không chuẩn hóa gộp toàn bộ dataset.
4. **Cắt Cửa sổ trượt (Sliding Window)**:
   - Tham số tối ưu: Kích thước cửa sổ **$N = 7$ frames**, bước trượt (step) = 1 frame.
5. **Gán nhãn Nháy mắt (V-Shape Labeling)**:
   - Dùng **Static Threshold = 0.1050** để làm mốc xác định mắt nhắm/mở.
   - Định nghĩa nhãn `Blink` (1) chỉ khi cửa sổ đó có đủ hình thái chữ V (Mở $\to$ Nhắm $\to$ Mở qua threshold). 
   - Nếu toàn bộ 7 frames đều nằm dưới threshold (nhắm tịt mắt), đánh nhãn là `No-Blink` (0) hoặc `Long-Closure` (2).
6. **Lưu kết quả đầu ra**: Lưu các ma trận chuỗi 3D (samples, time_steps, features) thành file Numpy (`X_train_seq.npy`, `y_train_seq.npy`, `X_test_seq.npy`, `y_test_seq.npy`).
