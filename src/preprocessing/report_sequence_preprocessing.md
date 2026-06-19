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
