# Báo Cáo Tiền Xử Lý Chuỗi EAR (Sequence Preprocessing)

**Tác giả:** Bảo (Pipeline Engineer - Hướng 2)
**Đầu vào:** `dataset_master/metadata_master.csv`
**Đầu ra:** `dataset_master/processed_seq/` (Chứa 4 file numpy)

---

## 1. Mục tiêu thuật toán
Mục tiêu của module này là chuẩn bị dữ liệu chuỗi thời gian (time-series) cho mô hình Deep Learning (LSTM / 1D-CNN) phát hiện hành vi nháy mắt và ngủ gật. 
Khác với xử lý ảnh tĩnh, module này đánh giá trạng thái mắt dựa trên một "chuỗi chuyển động" gồm nhiều khung hình liên tiếp.

## 2. Quy trình xử lý (Pipeline)
1. **Lọc dữ liệu:** Loại bỏ các frame lỗi (status != success) và các dòng trùng lặp thời gian.
2. **Chuẩn hóa Cá nhân (Min-Max Scaler):** Chỉ số EAR (Eye Aspect Ratio) của mỗi người (Contributor) được chuẩn hóa riêng biệt về khoảng `[0, 1]` để triệt tiêu sự khác biệt về kích thước mắt tự nhiên giữa các thành viên.
3. **Nội suy phục hồi (Linear Interpolation):** Điền khuyết các frame bị rớt/mất trong quá trình quay video bằng phương pháp nội suy tuyến tính, đảm bảo tính liên tục của chuỗi thời gian.
4. **Cửa sổ trượt (Sliding Window):** Áp dụng cửa sổ kích thước $N = 7$ frames, bước trượt $step = 1$.
5. **Gán nhãn V-Shape:** 
   - Sử dụng Threshold tĩnh: `0.1050`.
   - **Nhãn 1 (Blink):** Khung hình nhắm nằm giữa 2 khung hình mở (tạo hình chữ V).
   - **Nhãn 2 (Long-Closure):** Nhắm mắt quá 4 frames liên tục trong cửa sổ (Ngủ gật).
   - **Nhãn 0 (No-Blink):** Các trường hợp mở mắt bình thường hoặc đang chuyển trạng thái.
6. **Chia tập Train/Test:** Cắt tập dữ liệu với tỷ lệ 80/20, sử dụng `stratify` để duy trì tỷ lệ nhãn đồng đều giữa 2 tập.

## 3. Thống kê Dữ liệu Đầu ra
*(Cập nhật theo dữ liệu từ `dataset_master`)*

- **Tổng số cửa sổ hợp lệ:** 3,996 mẫu
- **Nhãn 0 (No-Blink):** 3,355 mẫu
- **Nhãn 1 (Blink):** 287 mẫu
- **Nhãn 2 (Long-Closure):** 354 mẫu

### Phân bổ Train/Test (80/20)
- **Tập Train (Huấn luyện):** 3,196 mẫu (`X_train_seq.npy`, `y_train_seq.npy`)
- **Tập Test (Kiểm thử):** 800 mẫu (`X_test_seq.npy`, `y_test_seq.npy`)

## 4. Hướng dẫn sử dụng
Để tự tạo lại các file `.npy` trên máy tính cá nhân, các thành viên trong nhóm chỉ cần chạy lệnh sau từ thư mục gốc của dự án:

```bash
python src/preprocessing/sequence_preprocessing.py
```
*(Yêu cầu đã cài đặt `pandas`, `numpy`, `scikit-learn` và đã kéo `dataset_master` về qua DVC).*

# Báo Cáo Tiền Xử Lý Chuỗi EAR (Sequence Preprocessing - Fixed Leakage)

**Tác giả:** Bảo (Pipeline Engineer - Hướng 2)
**Đầu vào:** `dataset_master/metadata_master.csv`
**Đầu ra:** `dataset_master/processed_seq/` (Chứa 4 file numpy)

---

## 1. Mục tiêu thuật toán
Mục tiêu của module này là chuẩn bị dữ liệu chuỗi thời gian (time-series) cho mô hình Deep Learning (LSTM / 1D-CNN) phát hiện hành vi nháy mắt và ngủ gật. 

## 2. Quy trình xử lý Cải Tiến (Zero-Leakage Pipeline)
1. **Lọc dữ liệu:** Loại bỏ các frame lỗi và các dòng trùng lặp thời gian.
2. **Chuẩn hóa Động (Dynamic/Causal Normalization):** Khắc phục lỗi nhìn trộm tương lai (Future Leakage). Ở mỗi khung hình, EAR chỉ được chuẩn hóa Min-Max dựa trên lịch sử dữ liệu thu thập được từ đầu video cho đến thời điểm đó (Rolling Window). Phù hợp tuyệt đối với môi trường chạy Real-time qua Webcam.
3. **Nội suy phục hồi (Linear Interpolation):** Điền khuyết các frame bị rớt bằng nội suy tuyến tính cho EAR, và sao chép nhãn (Forward Fill) cho cột `final_label`.
4. **Chia tập Train/Test theo Video (Chống Overlap Leakage):** Thay vì cắt nhỏ thành cửa sổ rồi mới chia (gây rò rỉ dữ liệu do các cửa sổ trùng nhau), thuật toán tiến hành chia 25 videos thành 20 videos cho Train và 5 videos cho Test **TRƯỚC KHI** cắt cửa sổ. Mô hình sẽ không bao giờ nhìn thấy người ở tập Test lúc Huấn luyện!
5. **Cửa sổ trượt (Sliding Window):** Cắt tập dữ liệu thành các cửa sổ $N = 7$ frames, bước trượt $step = 1$.
6. **Gán nhãn V-Shape (Ground-truth):** 
   - Sử dụng trực tiếp nhãn `final_label` có sẵn trong tập dữ liệu.
   - **Nhãn 1 (Blink):** Khung hình nhắm nằm giữa 2 khung hình mở.
   - **Nhãn 2 (Long-Closure):** Nhắm mắt quá 4 frames liên tục.
   - **Nhãn 0 (No-Blink):** Các trường hợp còn lại.

## 3. Thống kê Dữ liệu Đầu ra (Clean Split)

- **Tổng số Video:** 25 videos (Tập Train: 20, Tập Test: 5)
- **Tổng số cửa sổ hợp lệ:** 3,996 mẫu

### Phân bổ Tập Train (Huấn luyện) - 3,140 mẫu
- **Nhãn 0 (No-Blink):** 2,557 mẫu
- **Nhãn 1 (Blink):** 311 mẫu
- **Nhãn 2 (Long-Closure):** 272 mẫu

### Phân bổ Tập Test (Kiểm thử) - 856 mẫu
- **Nhãn 0 (No-Blink):** 663 mẫu
- **Nhãn 1 (Blink):** 37 mẫu
- **Nhãn 2 (Long-Closure):** 156 mẫu

## 4. Hướng dẫn sử dụng
Để tự tạo lại các file `.npy` chuẩn không rò rỉ dữ liệu, chạy lệnh sau:
```bash
python src/preprocessing/sequence_preprocessing.py
```
