# 🧠 Tổng Hợp Context Dự Án: Eye Blink & Drowsiness Detection

Tài liệu này tổng hợp toàn bộ bối cảnh (context), kiến trúc, các luồng xử lý (pipelines) và bước triển khai Web Client-Side của dự án. File này có thể dùng làm tài liệu bàn giao, báo cáo tổng kết hoặc làm context mồi cho các hệ thống AI/Agent khác khi tham gia vào dự án.

---

## 1. Tổng Quan Dự Án (Project Overview)
- **Mục tiêu chính:** Phát hiện trạng thái Mắt (Mở/Nháy) và cảnh báo Buồn ngủ (Drowsiness) dựa trên dữ liệu camera người dùng.
- **Tập dữ liệu (Dataset):**
  - Thu thập qua Webcam, quản lý version qua DVC.
  - Sử dụng MediaPipe Face Mesh để crop ảnh mắt (24x24) và trích xuất tọa độ mắt tính EAR (Eye Aspect Ratio).
  - Tách tập Train/Val/Test theo `video_id` (Group-based Split) để **chống hiện tượng Rò rỉ dữ liệu (Data Leakage)**, đảm bảo model học được đặc trưng tổng quát thay vì ghi nhớ khuôn mặt.

---

## 2. Các Hướng Tiếp Cận (Pipelines)

Dự án phát triển song song 2 hướng tiếp cận để so sánh hiệu năng và độ trễ:

### Hướng 1: Image Model Pipeline (Xử lý Ảnh)
- **Đầu vào:** Ảnh xám vùng mắt kích thước 24x24 pixel.
- **Mô hình thử nghiệm:** 
  - SVM HOG-only (Ablation study).
  - SVM HOG + EAR (Baseline ML nhẹ, độ trễ 0.75ms).
  - **CNN (Convolutional Neural Network)**: Mô hình tốt nhất phân loại Mở/Nhắm (Accuracy 90.86%, Macro F1 0.8603, độ trễ 8.36ms).
- **Ứng dụng:** Dùng làm phương pháp tham chiếu (baseline độ chính xác cao) nhưng nặng về tính toán.

### Hướng 2: TimeSeries EAR Pipeline (Chuỗi Thời Gian) -> Hướng Triển Khai Chính
- **Đầu vào:** Chuỗi 7 giá trị EAR liên tiếp (Sliding Window N=7, ở tốc độ 15 FPS).
- **Kỹ thuật Tiền xử lý (Đỉnh cao của dự án):**
  - **Dynamic Normalize (Hiệu chuẩn lai):** Lấy giá trị lớn nhất của 15 frame đầu làm mốc `init_max` (áp dụng chặn dưới 0.23 để chống lỗi nhắm mắt khi khởi động). Liên tục cập nhật `expanding_max` nếu gặp EAR lớn hơn để cá nhân hóa dải EAR cho từng người dùng.
- **Trích xuất Đặc trưng (Feature Extraction):**
  - Tinh giản từ 28 đặc trưng cũ xuống **12 đặc trưng động học phi tuyến tính** (loại bỏ cộng tuyến): 7 EAR thô + Min + Max + Std + Kurtosis + Ratio_center.
- **Mô hình thử nghiệm:**
  - Lật tẩy điểm yếu của Deep Learning: Mô hình LSTM bị overfit nghiêm trọng khi fix lỗi rò rỉ dữ liệu (Macro F1 tụt xuống 0.51).
  - **Linear SVM (C=1.0)**: Lên ngôi mạnh mẽ với **Accuracy 93.34%, Macro F1 0.9049**, tốc độ suy luận siêu nhanh 0.008ms.

---

## 3. Các Cơ Chế Tối Ưu Hệ Thống (System Optimizations)

Để đưa mô hình ra thực tế không bị báo động giả và không tốn pin, nhóm áp dụng 3 kỹ thuật cốt lõi:
1. **Hybrid Heuristic Filter (Lọc nhanh):** Nếu giá trị nhỏ nhất trong cửa sổ 7 frame `min(w) >= 0.5` -> Dự đoán ngay là Mở Mắt (Nhãn 0). Giúp bỏ qua bước chạy Machine Learning, **tiết kiệm 54% CPU**.
2. **Temporal Smoothing (Làm mịn thời gian):** Nhắm mắt đơn thuần có thể là chớp mắt chậm. Còi cảnh báo chỉ kêu khi hệ thống dự đoán "Sleep" liên tục trên 5-10 khung hình (> 0.67 giây).
3. **NMS (Non-Maximum Suppression):** Gom cụm các khung hình nháy mắt liên tiếp thành 1 lần chớp mắt duy nhất.

---

## 4. Triển Khai 100% Web Client-Side (Zero Server Cost)

Dựa trên lợi thế siêu nhẹ của Linear SVM, toàn bộ hệ thống đã được port (chuyển đổi) sang Web App chạy bằng trình duyệt của người dùng. Máy chủ chỉ đóng vai trò host file tĩnh (HTML/JS).

**Kiến trúc Web App:**
- **Webcam Input:** Giao tiếp qua `navigator.mediaDevices.getUserMedia` (Yêu cầu HTTPS hoặc Localhost).
- **Tracking:** Sử dụng thư viện `@mediapipe/tasks-vision` bản Javascript (chạy WebAssembly cực nhẹ).
- **Core Logic (`app.js`):** Mô phỏng lại 100% thuật toán Python: Quản lý hàng đợi Sliding Window 7 phần tử, hàm tính 12 đặc trưng (std, kurtosis), và bộ lọc Hybrid Filter, Temporal Smoothing.
- **Model Inference (`svm_model.js`):**
  - Sử dụng Python tool `m2cgen` để đọc file `best_traditional_model.pkl` (120KB) và biên dịch trực tiếp thuật toán Linear SVM cùng hàm StandardScaler thành code Javascript thuần túy (chỉ nặng ~3KB).
  - Kết quả là một hàm `scoreSVM()` không dùng bất kỳ thư viện bên thứ 3 nào, có thể chạy hàng ngàn lần mỗi giây trên chip điện thoại mà không hề giật lag.

**Hướng dẫn Test Local LAN:**
1. Khởi động Web Server tại folder chứa code Web: `python -m http.server 8000`
2. Truy cập qua IP Local: `http://<IP_MAY_TINH>:8000`
3. *Lưu ý khi mở Webcam thật:* Trình duyệt điện thoại (iOS Safari/Android Chrome) yêu cầu bảo mật HTTPS. Cần sử dụng công cụ như **ngrok** (`ngrok http 8000`) để tạo đường dẫn HTTPS an toàn khi test bằng camera điện thoại.
