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
 - Tổng số cửa sổ trích xuất : 3996
 - Số lượng nhãn 0 (No-Blink)    : 3220
 - Số lượng nhãn 1 (Blink)       : 348
 - Số lượng nhãn 2 (Long-Closure): 428
Trong đó: 
 - Tập Train: 3196 mẫu
 - Tập Test : 800 mẫu

---

## 1. Tổng quan các hướng tiếp cận mô hình đã thử nghiệm

Trong Hướng 2 (TimeSeries EAR), chúng tôi đã thiết kế và thử nghiệm 4 nhóm giải pháp công nghệ khác nhau để tìm ra phương án tối ưu nhất cho đồ án:

1. **Nhóm 1: Baseline Ngưỡng tĩnh Heuristic (Static Threshold):**
   * *Mô tả:* Sử dụng mốc EAR tĩnh `0.1050` để phân loại nhắm/mở từng frame đơn lẻ, tạo chuỗi nhị phân trạng thái rồi dùng thuật toán V-Shape để gán nhãn cho cả cửa sổ.
   * *Đánh giá:* Chạy nhanh nhất, nhưng không có khả năng tự tổng quát hóa (generalization) cho người dùng mới ngoài thực tế.

2. **Nhóm 2: Machine Learning với 7 Đặc trưng EAR thô (ML Raw):**
   * *Mô tả:* Làm phẳng cửa sổ 7 frame EAR thành một vector 7 chiều và đưa thẳng vào các mô hình học máy truyền thống (SVM, Random Forest, KNN, Decision Tree, Gradient Boosting).

3. **Nhóm 3: Machine Learning với 26 Đặc trưng Động học cải tiến (ML Adv):**
   * *Mô tả:* Thực hiện trích xuất thêm các đặc trưng động lực học của mí mắt gồm: Thống kê mô tả (Min, Max, Mean, Std, Range), Vận tốc biến thiên (Đạo hàm bậc 1 - Delta), Gia tốc biến thiên (Đạo hàm bậc 2 - Delta-Delta) và Độ dốc hai bên hình thể chữ V. Tổng cộng 26 đặc trưng.

4. **Nhóm 4: Kiến trúc lai Hybrid (Heuristic Filter + Machine Learning):**
   * *Mô tả:* Sử dụng ngưỡng lọc nhanh heuristic để bỏ qua bước suy luận ML trên các cửa sổ mà người dùng mở mắt bình thường (nhãn 0 - No-Blink). Điều kiện lọc dựa trên công thức:
     $$\min(w) \ge t \implies \text{Dự đoán ngay nhãn } 0 \text{ (Mắt mở - No-Blink)}$$
     Trong đó, $w = [EAR_1, EAR_2, \dots, EAR_7]$ là chuỗi EAR trong cửa sổ kích thước 7, và $t$ là ngưỡng lọc nhạy (ví dụ: $t = 0.2$). Hệ thống chỉ thực hiện trích xuất 26 đặc trưng nâng cao và đưa vào mô hình Random Forest phân loại chi tiết (Nhãn 0, 1 hoặc 2) khi cửa sổ nghi ngờ có sự suy giảm EAR sâu:
     $$\min(w) < t$$
   * *Đánh giá:* Giúp giảm tải đáng kể tài nguyên tính toán (CPU/RAM) khi mắt mở bình thường, cực kỳ tối ưu cho các thiết bị chạy thời gian thực.

5. **Nhóm 5: Deep Learning - Simple LSTM:**
   * *Mô tả:* Huấn luyện mô hình mạng nơ-ron hồi quy LSTM tối giản trên chuỗi 3D `(samples, 7, 3)` gồm các đặc trưng động học theo từng time-step.

---

## 2. Bảng kết quả tổng hợp hiệu năng (Benchmark Results)

Kết quả chi tiết được sắp xếp theo chỉ số **Macro F1-score** (độ đo cân bằng giữa Precision và Recall trên cả 3 lớp):

| STT | Tên mô hình | Loại đặc trưng | Siêu tham số (Config) | Accuracy | Macro F1 | Thời gian chạy |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | **Gradient Boosting (Adv)** | advanced_26feature | Tree=150, LR=0.10 | **93.50%** | **0.8618** | 0.0051s |
| 2 | **Hybrid (Filter + RF)** | hybrid_ear_norm | threshold=0.2 (t.kiệm 62.5% CPU) | **92.00%** | **0.8405** | 1.9424s |
| 3 | **Random Forest (Adv)** | advanced_26feature | Tree=200, Depth=10 | 0.9150 | 0.8380 | 0.0187s |
| 4 | **Gradient Boosting (Raw)** | raw_7frame_ear | Tree=150, LR=0.10 | 0.9200 | 0.8310 | 0.0056s |
| 5 | **Baseline Static (Normalized)**| ear_norm | threshold=0.1050_normalized_per_member | 0.9163 | 0.8122 | 0.0010s |
| 6 | **Baseline Static (Raw)** | raw_ear_avg | threshold=0.1050_static_global | 0.9163 | 0.8122 | 0.0010s |
| 7 | **Simple LSTM** | 3d_temporal_dynamic| LSTM_units=16_dropout=0.5_epochs=40 | 0.9000 | 0.7909 | 0.1871s |
| 8 | **RBF SVM (Adv)** | advanced_26feature | C=5.0, Scaled | 0.8975 | 0.8069 | 0.0364s |
| 9 | **KNN (Raw)** | raw_7frame_ear | K=5, Scaled | 0.9113 | 0.8108 | 0.0061s |
| 10| **Decision Tree (Adv)** | advanced_26feature | Depth=8, class_weight=balanced | 0.8900 | 0.7869 | 0.0010s |

---

## 3. Phân tích lỗi & Đánh giá trải nghiệm người dùng (UX)

Mặc dù mô hình tốt nhất (**Gradient Boosting trên 26 đặc trưng**) đạt độ chính xác rất cao là **93.50%**, chúng ta cần phân tích kỹ tỷ lệ sai số dưới góc nhìn ứng dụng thực tế:

### 3.1. Ý nghĩa của tỷ lệ lỗi 6.5% trong thời gian thực:
* Hệ thống chạy ở tốc độ **$15 \text{ FPS}$**, nghĩa là trong **1 phút** sẽ thu được: 
  $$15 \text{ frames/s} \times 60 \text{ s} = 900 \text{ frames}$$
* Áp dụng cửa sổ trượt kích thước 7, chúng ta sẽ có khoảng **894 cửa sổ cần phân loại mỗi phút**.
* Với tỷ lệ lỗi là **$6.5\%$**, số lượng cửa sổ bị dự đoán sai mỗi phút sẽ là:
  $$894 \text{ windows} \times 0.065 \approx \mathbf{58 \text{ cửa sổ bị lỗi/phút}}$$
* **Hậu quả:** Trung bình cứ **mỗi giây sẽ có gần 1 cửa sổ bị phân loại sai**. Nếu hệ thống lập tức cộng dồn số lần chớp mắt hoặc kích hoạt chuông cảnh báo buồn ngủ ngay khi có 1 cửa sổ lỗi, trải nghiệm người dùng sẽ cực kỳ tồi tệ (báo động giả liên tục).

### 3.2. Phân tích lỗi báo nhầm và lỗi bỏ sót:

#### A. Đối với hành vi nháy mắt (Nhãn 1 - Blink):
* **Gradient Boosting (Adv):** Đạt Precision 79%, Recall 74%.
  * *Lỗi báo giả (FPR - False Positive):* **21%**. Nghĩa là cứ 10 lần mô hình báo người dùng nháy mắt thì có 2 lần là báo nhầm (do người dùng nhìn xuống, ti hí mắt hoặc nhiễu landmark).
  * *Lỗi bỏ sót (FNR - False Negative):* **26%**. Cứ 10 lần người dùng chớp mắt thực tế thì mô hình bỏ sót khoảng 2.6 lần.

#### B. Đối với hành vi buồn ngủ (Nhãn 2 - Sleep/Long-Closure):
* **Gradient Boosting (Adv):** Đạt Precision 84% (báo nhầm 16%), Recall 87% (bỏ sót 13%).
* **Random Forest (Adv):** Đạt Precision 76% (báo nhầm 24%), nhưng **Recall lên tới 91% (chỉ bỏ sót 9%)**.
* *Nhận xét quan trọng (Phân tích theo hướng ứng dụng thực tế):* Vì mục tiêu thực tế của nhóm không phải là hệ thống cảnh báo an toàn lái xe (yêu cầu Recall cao bằng mọi giá), mà tập trung vào **hỗ trợ học tập và làm việc văn phòng**, việc lựa chọn mô hình sẽ phụ thuộc vào định hướng cụ thể của sản phẩm:
  * **Nếu phát triển theo hướng đếm tần suất nháy mắt & cảnh báo mỏi mắt khi học tập/làm việc:** Yếu tố **Precision** (độ chính xác dự báo) cực kỳ quan trọng. Nếu hệ thống liên tục báo nhầm (False Positive) là người dùng đang nhắm mắt/buồn ngủ khi họ thực chất chỉ đang nheo mắt tập trung nhìn màn hình, điều này sẽ gây khó chịu và gián đoạn công việc. Trong kịch bản này, **Gradient Boosting (Adv)** hoặc giải pháp **Hybrid** là lựa chọn tối ưu nhờ tỷ lệ báo nhầm thấp nhất (chỉ 16%).
  * **Nếu phát triển theo hướng cảnh báo ngủ gật/ngủ quên (Drowsiness Warning):** Yếu tố **Recall** (độ nhạy) lại được ưu tiên hàng đầu để đảm bảo người dùng được đánh thức kịp thời khi thực sự chìm vào giấc ngủ gật. Trong kịch bản này, **Random Forest (Adv)** là lựa chọn phù hợp hơn nhờ khả năng phát hiện nhạy bén, giảm tỷ lệ bỏ sót xuống chỉ còn 9%.

---

## 4. Đề xuất giải pháp khắc phục cho hệ thống thực tế

Để đưa mô hình từ nghiên cứu vào ứng dụng thực tế chạy mượt mà, chúng tôi đề xuất 2 kỹ thuật bắt buộc sau:

1. **Bộ lọc làm mịn thời gian (Temporal Smoothing / Voting):**
   * Không kích hoạt chuông cảnh báo ngay lập tức. Thay vào đó, chỉ phát còi báo động ngủ gật khi mô hình dự đoán nhãn 2 (Sleep) liên tục trong ít nhất **5 đến 10 cửa sổ liên tiếp** (tương đương nhắm mắt liên tục > 0.5 giây). Điều này giúp triệt tiêu hoàn toàn 58 cửa sổ lỗi ngẫu nhiên mỗi phút.

2. **Kỹ thuật Triệt tiêu phi cực đại (Non-Maximum Suppression - NMS) cho chớp mắt:**
   * Vì một cú chớp mắt thực tế kéo dài 2-4 frame sẽ làm cho 3-4 cửa sổ trượt liên tiếp đều nhận diện là nhãn 1 (Blink), chúng ta cần dùng thuật toán gom cụm (NMS hoặc lọc đỉnh) để gộp các dự đoán Blink liền kề nhau thành **đúng 1 lần chớp mắt duy nhất**. Kỹ thuật này sẽ giúp số lần chớp mắt được đếm chính xác tuyệt đối.

3. **Kiến trúc Hybrid để tối ưu phần cứng:**
   * Triển khai giải pháp **Hybrid với ngưỡng $t = 0.2$**. Bằng cách áp dụng công thức sàng lọc heuristic $\min(w) \ge 0.2$, hệ thống có thể bỏ qua toàn bộ bước trích xuất đặc trưng động lực học phức tạp và suy luận mô hình Random Forest cho **62.5%** số cửa sổ mở mắt bình thường. Điều này giúp tối ưu hóa phần cứng hiệu quả, chạy mát hơn, tiết kiệm pin và giảm độ trễ xử lý trong thời gian thực.

---

## 5. Hướng dẫn chạy Demo thời gian thực (Real-Time Preview)

Để kiểm chứng hiệu năng và trải nghiệm người dùng thực tế của mô hình (Gradient Boosting Adv kết hợp bộ lọc Hybrid và hậu xử lý thời gian), bạn có thể chạy script Demo tương tác thời gian thực:

```bash
# Sử dụng Webcam mặc định (Webcam index 0)
python src/models/TimeSeries/realtime_preview.py

# Hoặc truyền vào file video đã quay sẵn
python src/models/TimeSeries/realtime_preview.py --input path/to/video.mp4
```

**Tính năng nổi bật trong Demo:**
* **Tự động hiệu chuẩn (Calibration):** 100 khung hình đầu tiên sẽ tự học dải EAR mở/nhắm của mắt bạn để cá nhân hóa ngưỡng của từng người. (Bấm phím `r` để reset hiệu chuẩn).
* **HUD giám sát hiệu năng:** Hiển thị trực quan tốc độ camera (Webcam FPS), tốc độ lấy mẫu chuẩn (15 FPS), thanh EAR Bar và tỉ lệ tiết kiệm phần cứng của bộ lọc Hybrid (`CPU Saved %`).
* **Khử nhiễu thực tế:** Chỉ báo ngủ gật khi nhắm mắt liên tục $>0.67$ giây (tránh báo động giả khi chớp mắt nhanh hoặc nheo mắt).
* **Che 1 mắt vẫn chạy tốt:** Tích hợp bộ lọc lấy giá trị nhỏ nhất của 2 mắt giúp hệ thống vẫn theo dõi được cú chớp/nhắm của mắt còn lại khi một bên bị che khuất.

