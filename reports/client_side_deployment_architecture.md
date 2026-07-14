# Hướng Dẫn Triển Khai 100% Client-Side (Trình Duyệt) cho Mô Hình Drowsiness Detection

Tài liệu này giải thích cách nhóm có thể đưa toàn bộ mô hình **TimeSeries EAR (Linear SVM)** lên chạy trực tiếp trên trình duyệt của người dùng (điện thoại/máy tính) mà **không cần server xử lý (Zero Server Cost)**.

---

## 1. Tại Sao Lại Chạy Client-Side?

- **Siêu nhẹ và siêu nhanh:** Model Linear SVM tốt nhất của chúng ta cực kỳ nhẹ. Khi chuyển đổi sang file JavaScript thuần túy, dung lượng giảm từ 120KB (file `.pkl`) xuống chỉ còn khoảng **vài KB**.
- **Không tốn tiền Server GPU/CPU:** Toàn bộ quá trình tính toán (từ nhận diện khuôn mặt đến chạy mô hình) dùng chính CPU của thiết bị người dùng.
- **Bảo mật tuyệt đối (Privacy):** Hình ảnh khuôn mặt của người dùng không bao giờ bị gửi qua mạng.
- **Độ trễ bằng 0 (Zero Latency):** Có thể chạy mượt mà ở tốc độ 30-60 FPS.

---

## 2. Kiến Trúc Hệ Thống Trên Web (Javascript)

Để chạy được giống hệt file `realtime_preview.py` trên Python, kiến trúc Web sẽ bao gồm các bước sau:

### Bước 1: Trích xuất tọa độ mắt (MediaPipe JS)
- Thay vì dùng Python, ta sẽ sử dụng thư viện `@mediapipe/tasks-vision` chính thức của Google dành cho Javascript. Thư viện này hỗ trợ vẽ Face Mesh trực tiếp từ thẻ `<video>` của HTML5 thông qua camera thiết bị.

### Bước 2: Tính toán EAR và Chuẩn hóa (State Variables)
Trên Web, ta sẽ mô phỏng lại hoàn toàn logic của file `sequence_preprocessing.py`:
- Tính EAR thô bằng công thức khoảng cách Euclid.
- Duy trì mảng 15 frames khởi động để tính `init_max`.
- Giữ logic **Expanding Max** (cập nhật max liên tục) và công thức **Hybrid Calibration**: `ear_norm = ear / expanding_max`.
- Dùng một mảng tĩnh (array) kích thước 7 làm cửa sổ trượt (Sliding Window).

### Bước 3: Trích xuất 12 Advanced Features
- Các hàm `np.min`, `np.max`, `np.std`, `kurtosis` từ thư viện Numpy trong Python có thể dễ dàng được viết lại bằng Javascript thuần túy chỉ với vài vòng lặp `for`.

### Bước 4: Chạy Mô Hình SVM (Bằng mã JS Thuần)
- Nhóm sử dụng công cụ **`m2cgen`** (Model-to-Code Generator) để dịch nguyên khối file `best_traditional_model.pkl` thành một hàm Javascript tên là `score(features)`.
- *Tin vui:* Bởi vì mô hình sử dụng `Pipeline` kết hợp `StandardScaler` và `LinearSVC`, `m2cgen` sẽ tự động dịch luôn cả thuật toán chuẩn hóa dữ liệu vào trong JS. Việc của chúng ta chỉ là đưa 12 đặc trưng vào và nhận kết quả đầu ra.

### Bước 5: Hậu xử lý & Làm Mịn (Temporal Smoothing)
- Sử dụng thuật toán lọc Hybrid Heuristic: Nếu `min(window) >= 0.5`, lập tức dự đoán nhãn 0 (Tiết kiệm CPU).
- Sử dụng biến đếm để đếm số lượng frame dự đoán nhãn 2 (Sleep). Nếu ngủ gật quá $0.67$ giây thì mới bật âm thanh cảnh báo bằng hàm `audio.play()`.

---

## 3. Quá Trình Convert Model Đã Thực Hiện

Nhóm đã tạo sẵn script convert trong hệ thống:
File: `src/models/TimeSeries/export_svm_to_js.py`

Khi chạy file này:
```bash
python src/models/TimeSeries/export_svm_to_js.py
```
Nó sẽ đọc model tốt nhất và sinh ra một file JavaScript siêu nhẹ (khoảng ~2KB) tại:
`src/deployment/web_demo/svm_model.js`

File này chứa hàm `score(input)` chứa toàn bộ trọng số (weights) và thuật toán của Linear SVM kèm theo logic StandardScaler.

---

## 4. Tóm Lược 
Kiến trúc dự án (Đặc trưng 12 chiều, Linear SVM, Bộ lọc Hybrid, Tự động hiệu chuẩn) là một thiết kế **hoàn hảo** cho môi trường Mobile/Web. Việc đưa model này lên Client-Side sẽ mở ra tiềm năng tạo thành một PWA (Tiến trình ứng dụng Web) cực kỳ xịn xò mà không tốn kém chi phí hạ tầng.
