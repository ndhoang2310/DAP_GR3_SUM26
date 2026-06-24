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
