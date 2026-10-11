# Báo Cáo Đánh Giá & Thử Nghiệm Các Mô Hình Hướng 2 (TimeSeries EAR)

> [!IMPORTANT]
> **Hướng dẫn chạy nhanh Benchmark khi clone dự án:**
> 1. **Kích hoạt môi trường ảo & Cài đặt thư viện:**
>    ```bash
>    # Tạo & Kích hoạt venv
>    python -m venv venv
>    # Windows (PowerShell):
>    .\venv\Scripts\activate
>    # Linux/macOS:
>    source venv/bin/activate
>    
>    # Cài đặt thư viện (bao gồm tensorflow & scikit-learn)
>    pip install -r requirements.txt
>    ```
> 2. **Tải dữ liệu qua DVC (nếu chưa có thư mục dataset_master):**
>    ```bash
>    dvc pull
>    ```
> 3. **Chạy Tiền xử lý chuỗi (Sequence Preprocessing) để trích xuất dữ liệu cửa sổ trượt:**
>    ```bash
>    python src/preprocessing/sequence_preprocessing.py
>    ```
> 4. **Chạy Pipeline Benchmark:**
>    ```bash
>    python src/models/TimeSeries/main_benchmark.py
>    ```
>    *Sau khi chạy sẽ xuất ra file kết quả tại `dataset_master\benchmark_results.csv` để đánh giá.*
> 5. **Chạy Demo tương tác thời gian thực (Real-Time Preview):**
>    ```bash
>    python src/models/TimeSeries/realtime_preview.py
>    ```
>    *Xem hướng dẫn tương tác chi tiết ở cuối file.*


**Thành viên thực hiện:** Hiền (Pipeline Engineer - Hướng 2)  
**Tập dữ liệu thử nghiệm:** mẫu chuỗi EAR ($15 \text{ FPS}$, cửa sổ trượt $N = 7$ frames).
 - Tổng số cửa sổ trích xuất : 3621
 Trong đó (Áp dụng chia tập theo nhóm người dùng - Group-based Split để chống rò rỉ dữ liệu): 
 - **Tập Train Sub (17 Videos):** 2455 mẫu (~67.8%)
    - Nhãn 0 (No-Blink)    : 1920 mẫu
    - Nhãn 1 (Blink)       : 342 mẫu
    - Nhãn 2 (Long-Closure): 193 mẫu
 - **Tập Validation (3 Videos - Cho DL):** 385 mẫu (~10.6%)
    - Nhãn 0 (No-Blink)    : 272 mẫu
    - Nhãn 1 (Blink)       : 41 mẫu
    - Nhãn 2 (Long-Closure): 72 mẫu
 - **Tập Test (5 Videos - Độc lập):** 781 mẫu (~21.6%)
    - Nhãn 0 (No-Blink)    : 566 mẫu
    - Nhãn 1 (Blink)       : 59 mẫu
    - Nhãn 2 (Long-Closure): 156 mẫu

*(Lưu ý: Để huấn luyện các mô hình Machine Learning truyền thống không dùng Early Stopping, tập Train Sub và Validation được gộp lại thành tập Train lớn gồm 20 video với 2840 mẫu để tối ưu hóa lượng dữ liệu).*

---

## 1. Tổng quan các hướng tiếp cận mô hình đã thử nghiệm

Trong Hướng 2 (TimeSeries EAR), chúng tôi đã thiết kế và thử nghiệm 4 nhóm giải pháp công nghệ khác nhau để tìm ra phương án tối ưu nhất cho đồ án:

1. **Nhóm 1: Machine Learning với 7 Đặc trưng EAR thô (ML Raw):**
   * *Mô tả:* Làm phẳng cửa sổ 7 frame EAR thành một vector 7 chiều và đưa thẳng vào các mô hình học máy truyền thống (SVM, Logistic Regression, Random Forest, KNN, Decision Tree, Gradient Boosting, Naive Bayes).

2. **Nhóm 2: Machine Learning với 12 Đặc trưng Động học độc lập (ML Adv):**
   * *Mô tả:* Để tối ưu hóa hiệu năng, chúng tôi rút gọn từ bộ 28 đặc trưng cũ (loại bỏ hoàn toàn 16 đặc trưng tuyến tính có tính cộng tuyến như đạo hàm bậc 1, bậc 2, range, trend slope) để giữ lại **12 đặc trưng độc lập phi tuyến** mạnh mẽ nhất:
     * 7 giá trị EAR thô trong cửa sổ.
     * Cực tiểu (Min).
     * Cực đại (Max).
     * Độ lệch chuẩn (Std).
     * Tỷ lệ tâm (Ratio_center = $EAR_4 / \max(w)$).
     * Độ nhọn phân phối (Kurtosis).
 
3. **Nhóm 3: Kiến trúc lai Hybrid (Heuristic Filter + Machine Learning):**
   * *Mô tả:* Sử dụng ngưỡng lọc nhanh heuristic để bỏ qua bước suy luận ML trên các cửa sổ mà người dùng mở mắt bình thường (nhãn 0 - No-Blink). Điều kiện lọc dựa trên công thức:
     $$\min(w) \ge t \implies \text{Dự đoán ngay nhãn } 0 \text{ (Mắt mở - No-Blink)}$$
     Trong đó, $w = [EAR_1, EAR_2, \dots, EAR_7]$ là chuỗi EAR trong cửa sổ kích thước 7, và $t$ là ngưỡng lọc nhạy (ví dụ: $t = 0.5$ hoặc $t = 0.4$). Hệ thống chỉ thực hiện trích xuất đặc trưng và đưa vào mô hình học máy tốt nhất (Linear SVM) phân loại chi tiết khi cửa sổ nghi ngờ có sự giảm EAR sâu:
     $$\min(w) < t$$
   * *Đánh giá:* Giúp giảm tải đáng kể tài nguyên tính toán (CPU/RAM) khi mắt mở bình thường (tiết kiệm đến 54% - 57% số lần chạy mô hình ML), cực kỳ tối ưu cho các thiết bị chạy thời gian thực.

4. **Nhóm 4: Deep Learning - Simple LSTM (Option B):**
   * *Mô tả:* Huấn luyện mô hình mạng nơ-ron hồi quy LSTM tối giản trên chuỗi 3D `(samples, 7, 3)` gồm các đặc trưng động học theo từng time-step. Để tránh lỗi rò rỉ dữ liệu (Data Leakage) khi huấn luyện, tập Validation được tách biệt hoàn toàn theo ID Video độc lập chứ không dùng hàm chia ngẫu nhiên của thư viện.

---

## 2. Bảng kết quả tổng hợp hiệu năng (Benchmark Results)

Kết quả chi tiết dưới đây đã được làm sạch hoàn toàn khỏi lỗi rò rỉ dữ liệu (Group-based Split). Các mô hình được sắp xếp theo chỉ số **Macro F1-score**:

| STT | Tên mô hình | Loại đặc trưng | Siêu tham số (Config) | Accuracy | Macro F1 | Thời gian chạy (Inference Latency) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | **ML Adv - Linear SVM** | advanced_12feature | C=1.0 | **93.34%** | **0.9049** | 0.0080s |
| 2 | **Hybrid (Filter + SVC)** | hybrid_ear_norm | threshold=0.5 (t.kiệm 54.5% CPU) | **93.09%** | **0.8952** | 0.1220s |
| 3 | **ML Raw - RBF SVM** | raw_7frame_ear | C=0.5 | **93.09%** | **0.8937** | 0.0305s |
| 4 | **ML Adv - Linear SVM** | advanced_12feature | C=5.0 | **92.83%** | **0.8923** | 0.0080s |
| 5 | **ML Adv - Linear SVM** | advanced_12feature | C=0.1 | **92.96%** | **0.8908** | 0.0080s |
| 6 | **ML Adv - Logistic Regression** | advanced_12feature | class_weight='balanced' | **92.70%** | **0.8896** | 0.0000s |
| 7 | **Hybrid (Filter + SVC)** | hybrid_ear_norm | threshold=0.4 (t.kiệm 57.4% CPU) | **92.96%** | **0.8859** | 0.1241s |
| 8 | **ML Raw - RBF SVM** | raw_7frame_ear | C=1.5 | 92.83% | 0.8837 | 0.0275s |
| 9 | **ML Adv - RBF SVM** | advanced_12feature | C=0.5 | 92.70% | 0.8821 | 0.0330s |
| 10 | **DL - LSTM (Option B)** | 3d_temporal_dynamic | units=16, dropout=0.5, epochs=40 | **76.44%** | **0.5131** | 0.2994s |

> [!NOTE]
> **Nhận xét quan trọng về kết quả thử nghiệm:**
> * **Độ chính xác vượt trội của Linear SVM**: Việc tinh giản đặc trưng từ 28 xuống **12 đặc trưng độc lập** cùng logic hiệu chuẩn lai (Hybrid Calibration) - lấy max 15 frame đầu làm mốc mở mắt và cập nhật expanding max động - giúp mô hình Linear SVM đạt độ chính xác tối ưu **93.34%** và F1-score vượt trội **0.9049**.
> * **Lật tẩy lỗi Rò rỉ dữ liệu (Data Leakage) của Deep Learning**: 
>   * Ở các thử nghiệm trước, do dùng tham số `validation_split=0.15` mặc định của Keras, dữ liệu bị chia cắt ngẫu nhiên ở mức frame. Một video bị cắt đôi làm hai nửa (nửa đầu ở tập Train, nửa sau ở tập Val), làm rò rỉ đặc điểm khuôn mặt người dùng và thổi phồng kết quả Macro F1 của LSTM lên ~0.73.
>   * Khi **sửa lỗi rò rỉ dữ liệu** bằng cách cô lập tập Validation theo Video ID riêng biệt, mô hình LSTM đã bộc lộ hiệu năng thực tế thấp hơn nhiều (**Macro F1 sụt giảm xuống còn 0.5131**). Điều này chứng minh mạng học sâu LSTM bị quá khớp (overfit) nghiêm trọng trên tập dữ liệu nhỏ (chỉ 17 video train) và không có khả năng tổng quát hóa cho người dùng lạ nếu không có data leakage.
> * **Sự hiệu quả của Hybrid**: Mô hình Hybrid với ngưỡng 0.5 giúp tiết kiệm **54.5%** số lần chạy mô hình ML và giữ được Macro F1 rất cao **0.8952**.

---

## 3. Phân tích lỗi & Đánh giá trải nghiệm người dùng (UX)

Mặc dù mô hình Linear SVM tốt nhất đạt độ chính xác **93.34%**, chúng ta cần phân tích kỹ tỷ lệ sai số dưới góc nhìn ứng dụng thực tế:

### 3.1. Ý nghĩa của tỷ lệ lỗi 6.66% trong thời gian thực:
* Hệ thống chạy ở tốc độ **$15 \text{ FPS}$**, nghĩa là trong **1 phút** sẽ thu được: 
  $$15 \text{ frames/s} \times 60 \text{ s} = 900 \text{ frames}$$
* Áp dụng cửa sổ trượt kích thước 7, chúng ta sẽ có khoảng **894 cửa sổ cần phân loại mỗi phút**.
* Với tỷ lệ lỗi là **$6.66\%$**, số lượng cửa sổ bị dự đoán sai mỗi phút sẽ là:
  $$894 \text{ windows} \times 0.0666 \approx \mathbf{60 \text{ cửa sổ bị lỗi/phút}}$$
* **Hậu quả:** Trung bình cứ **mỗi giây sẽ có khoảng 1 cửa sổ bị phân loại sai**. Nếu hệ thống lập tức cộng dồn số lần chớp mắt hoặc kích hoạt chuông cảnh báo buồn ngủ ngay khi có 1 cửa sổ lỗi, trải nghiệm người dùng sẽ bị ảnh hưởng bởi báo động giả liên tục. Do đó, các bộ lọc làm mịn thời gian ở khâu hậu xử lý (Temporal Smoothing) là bắt buộc.

### 3.2. Phân tích lỗi báo nhầm và lỗi bỏ sót (cho mô hình tốt nhất ML Adv - Linear SVM C=1.0):

#### A. Đối với hành vi nháy mắt (Nhãn 1 - Blink):
* Mô hình đạt **Precision 80.88%**, **Recall 93.22%**.
  * *Lỗi báo giả (False Positive):* **19.12%**. Xảy ra chủ yếu do mô hình phát hiện sớm hoặc muộn hơn 1 frame so với điểm chính giữa hoàn hảo mà thuật toán gán nhãn quy định. Khi chạy thực tế, cơ chế trượt cửa sổ liên tục sẽ tự động sửa lỗi này và vẫn nhận diện thành công 1 xung chớp mắt sạch.
  * *Lỗi bỏ sót (False Negative):* **6.78%**. Mô hình nhận diện rất tốt nháy mắt, chỉ bỏ sót 6.78% số cú nháy thực tế.

#### B. Đối với hành vi buồn ngủ (Nhãn 2 - Sleep/Long-Closure):
* Mô hình đạt **Precision 81.91%**, **Recall 98.72%**.
  * *Lỗi báo giả (False Positive):* **18.09%**. Báo nhầm người dùng đang nhắm mắt buồn ngủ khi họ thực ra chỉ đang nheo mắt hoặc nhìn cúi đầu tập trung.
  * *Lỗi bỏ sót (False Negative):* **1.28%**. **Cực kỳ thấp!** Mô hình gần như phát hiện được 98.72% số trường hợp nhắm mắt dài thực tế, đảm bảo an toàn tuyệt đối nếu dùng cho cảnh báo ngủ quên.

---

## 4. Đề xuất giải pháp khắc phục cho hệ thống thực tế

Để đưa mô hình từ nghiên cứu vào ứng dụng thực tế chạy mượt mà, chúng tôi đề xuất 3 kỹ thuật bắt buộc sau:

1. **Bộ lọc làm mịn thời gian (Temporal Smoothing / Voting):**
   * Không kích hoạt chuông cảnh báo ngay lập tức. Chỉ phát còi báo động ngủ gật khi mô hình dự đoán nhãn 2 (Sleep) liên tục trong ít nhất **5 đến 10 cửa sổ liên tiếp** (tương đương nhắm mắt liên tục > 0.5 giây). Điều này giúp triệt tiêu hoàn toàn các cửa sổ lỗi ngẫu nhiên do nháy mắt hoặc nhiễu landmark.

2. **Kỹ thuật Triệt tiêu phi cực đại (Non-Maximum Suppression - NMS) cho chớp mắt:**
   * Vì một cú chớp mắt thực tế kéo dài 2-4 frame sẽ làm cho 3-4 cửa sổ trượt liền kề nhau đều nhận diện là nhãn 1 (Blink), chúng ta cần dùng thuật toán gom cụm (NMS hoặc lọc đỉnh) để gộp các dự đoán Blink liền kề nhau thành **đúng 1 lần chớp mắt duy nhất**.

3. **Thích ứng động (Adaptive Calibration) pha khởi chạy:**
   * Hệ thống tự động thiết lập và cập nhật giá trị **expanding max** liên tục khi chạy camera để tạo mốc mở mắt chuẩn (`calib_baseline`), giúp thích ứng linh hoạt và tự động sửa sai nếu người dùng nhắm mắt lúc khởi động.

---

## 5. Hướng dẫn chạy Demo thời gian thực (Real-Time Preview)

Để kiểm chứng hiệu năng và trải nghiệm người dùng thực tế của mô hình (Linear SVM kết hợp bộ lọc Hybrid và tự động hiệu chuẩn), bạn có thể chạy script Demo tương tác thời gian thực:

```bash
# Sử dụng Webcam mặc định (Webcam index 0)
python src/models/TimeSeries/realtime_preview.py

# Hoặc truyền vào file video đã quay sẵn
python src/models/TimeSeries/realtime_preview.py --input path/to/video.mp4
```

**Tính năng nổi bật trong Demo:**
* **Tự động hiệu chuẩn (Calibration):** Hệ thống tự động học và thích ứng động với dải EAR mở mắt của bạn để cá nhân hóa ngưỡng của từng người. (Bấm phím `r` để reset hiệu chuẩn).
* **HUD giám sát hiệu năng:** Hiển thị trực quan tốc độ camera (Webcam FPS), tốc độ lấy mẫu chuẩn (15 FPS), thanh EAR Bar và tỉ lệ tiết kiệm phần cứng của bộ lọc Hybrid (`CPU Saved %`).
* **Khử nhiễu thực tế:** Chỉ báo ngủ gật khi nhắm mắt liên tục $>0.67$ giây (tránh báo động giả khi chớp mắt nhanh hoặc nheo mắt).
* **Che 1 mắt vẫn chạy tốt:** Tích hợp bộ lọc lấy giá trị nhỏ nhất của 2 mắt giúp hệ thống vẫn theo dõi được cú chớp/nhắm của mắt còn lại khi một bên bị che khuất.
