# Đề Xuất Giải Pháp: Cải Tiến Mô Hình EAR Bằng HMM (Hidden Markov Model)

Tài liệu này lưu trữ ý tưởng chuyển đổi cấu trúc phân tích dữ liệu TimeSeries EAR từ cách tiếp cận Sliding Windows hiện tại sang phương pháp Học máy Thống kê (Statistical Machine Learning) sử dụng Lọc Bayes (Bayesian Filtering), cụ thể là mô hình HMM với Gaussian Emissions.

## 1. Vấn đề của Pipeline hiện tại (Sliding Windows + Machine Learning)

*   **Data Leakage (Rò rỉ dữ liệu):** Việc chia tập Train/Test ngẫu nhiên trên các Sliding Windows đan xen nhau (stride = 1) khiến tập Test chứa rất nhiều frame đã có trong tập Train. Điều này dẫn đến kết quả Benchmark ảo, độ chính xác cao nhưng không hoạt động tốt với dữ liệu mới. (Giải pháp khắc phục tạm thời: Dùng `GroupKFold` theo `member_id`).
*   **Normalization Mismatch (Lệch chuẩn hóa):** Quá trình tiền xử lý đang dùng global Min/Max của cả video để chuẩn hóa EAR. Khi chạy thực tế qua Webcam, do không có dữ liệu tương lai nên phải dùng kỹ thuật "Dynamic Calibration". Sự khác biệt này làm phân phối dữ liệu đưa vào mô hình (Domain) bị lệch.
*   **Độ trễ hệ thống (Latency/Non-causal Design):** Để nhận dạng 1 hành vi chớp mắt (V-Shape), hệ thống bắt buộc phải chờ thu thập đủ các frame tương lai để lấp đầy cửa sổ (N=7). Ở tốc độ 15 FPS, điều này tạo ra độ trễ.

## 2. Ý tưởng Cải tiến: Áp dụng HMM (Hidden Markov Model)

HMM giải quyết triệt để vấn đề độ trễ thông qua việc Cập nhật Niềm tin liên tục (Bayesian Updating) ở mỗi frame. Nó hoạt động như sau:

*   **Trạng thái ẩn (Hidden States):** `[Mở mắt, Đang chớp mắt, Buồn ngủ]`. Đây là trạng thái ý thức thật sự của người dùng, ta không đo trực tiếp được.
*   **Quan sát (Observations):** Chỉ số EAR đo được từ hệ thống camera ở mỗi frame.
*   **Nguyên tắc "Niềm tin":** HMM giữ một "niềm tin" (xác suất Prior) về trạng thái của mắt. Mỗi khi frame mới đến, nó dùng định lý Bayes kết hợp niềm tin cũ và dữ liệu EAR mới để tạo ra niềm tin mới (Posterior). Khi Posterior của trạng thái `Buồn ngủ` > 80%, hệ thống lập tức phát cảnh báo.

### Ưu điểm vượt trội:
1.  **Zero-Latency (Causal):** Mỗi frame mới vào là update niềm tin và ra quyết định ngay lập tức dựa trên lịch sử trước đó, không cần đợi lấp đầy bất kỳ cửa sổ tương lai nào.
2.  **Kháng nhiễu cực tốt:** Một vài frame EAR đo sai lệch (nhiễu MediaPipe) sẽ không thể thay đổi toàn bộ kết luận nếu xác suất Prior từ quá khứ đang rất vững chắc.

## 3. Khái niệm Gaussian Emissions

Chỉ số EAR là dữ liệu liên tục (Continuous), vì vậy ta kết hợp HMM với **Gaussian Emissions (Phát xạ Gaussian)**.

*   **Tư duy:** Khi người dùng đang ở trạng thái `Mở mắt`, giá trị EAR sinh ra không cố định một số, mà phân bổ theo **Phân phối chuẩn hình quả chuông (Gaussian)** (ví dụ: tập trung cao quanh Mean = 0.32, dao động sai số Std = 0.03).
*   **Lợi ích:** Mang lại khả năng dung sai mềm mại. Máy tính sẽ đánh giá xác suất của 1 giá trị EAR bằng cách "ướm" xem nó nằm ở phần đỉnh hay rìa của quả chuông tương ứng với mỗi trạng thái, thay vì áp dụng một mốc cắt cứng nhắc (Threshold < 0.2).

## 4. Hướng triển khai thực tế
Sử dụng thư viện `hmmlearn` của Python.
*   **Thuật toán:** `GaussianHMM`
*   **Features đầu vào mỗi bước:** `[EAR_t, Delta_EAR_t]` (giá trị EAR và mức độ biến thiên so với frame trước đó).
*   **Huấn luyện (Training):** Dùng thuật toán Baum-Welch để tự động học ra các Ma trận Chuyển trạng thái (Transition Matrix) và thông số của quả chuông (Gaussian Mean, Variance) cho mỗi trạng thái.
*   **Dự đoán (Inference):** Dùng Viterbi Algorithm hoặc Forward algorithm để giải mã trạng thái thực sự tại frame t.
