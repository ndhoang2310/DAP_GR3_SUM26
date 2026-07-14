# Báo Cáo Triển Khai Web Client-Side Cho BlinkGuard

## 1. Mục tiêu triển khai

Giai đoạn deployment của dự án chuyển hệ thống phát hiện trạng thái mắt từ môi trường Python nghiên cứu sang một web app chạy trực tiếp trên trình duyệt. Mục tiêu không chỉ là demo mô hình, mà còn biến pipeline thành một ứng dụng thực tế có thể theo dõi tần suất chớp mắt và cảnh báo người dùng khi họ tập trung quá lâu dẫn đến quên chớp mắt.

Tên web app hiện tại: **BlinkGuard**.

Các mục tiêu chính:

- Chạy 100% client-side, không cần server xử lý video.
- Không gửi hình ảnh khuôn mặt người dùng ra backend.
- Dùng webcam của trình duyệt để tracking mắt theo thời gian thực.
- Dùng mô hình Linear SVM đã huấn luyện để phân loại trạng thái cửa sổ EAR.
- Theo dõi số lần chớp mắt trong 60 giây gần nhất.
- Cảnh báo khi người dùng không chớp mắt quá lâu.
- Gửi system notification ngoài trang web khi có cảnh báo quan trọng.
- Có thể deploy như static site trên Cloudflare Pages.

## 2. Lý do chọn triển khai client-side

Pipeline TimeSeries EAR có mô hình tốt nhất là Linear SVM với 12 đặc trưng động học. Đây là mô hình rất nhẹ, tốc độ suy luận nhanh và không phụ thuộc GPU. Vì vậy, thay vì dựng backend API để nhận frame hoặc nhận feature từ client, toàn bộ pipeline được đưa vào trình duyệt.

Lợi ích:

- **Zero server inference cost:** server chỉ host file tĩnh.
- **Low latency:** suy luận diễn ra ngay trên thiết bị người dùng.
- **Privacy-friendly:** frame webcam không rời khỏi trình duyệt.
- **Dễ deploy:** chỉ cần `index.html`, `app.js`, `svm_model.js`, `favicon.svg`.
- **Phù hợp Cloudflare Pages:** project không cần build step.

## 3. Model được dùng để deploy

Model triển khai là model tốt nhất của pipeline TimeSeries EAR:

- Model gốc: `dataset_master/models/best_traditional_model.pkl`
- Loại model: `Pipeline(StandardScaler + Linear SVC)`
- Input: 12 features
- Classes:
  - `0`: mắt mở / no-blink
  - `1`: blink
  - `2`: sleep / long-closure

12 features đầu vào:

```text
7 giá trị EAR chuẩn hóa trong sliding window
min(window)
max(window)
std(window)
ratio_center = (w[0] + w[6]) / (2 * w[3] + 1e-5)
kurtosis(window)
```

Điểm quan trọng: web app phải giữ công thức feature extraction giống lúc train/export model. Nếu đổi công thức feature mà không train lại model, kết quả suy luận sẽ lệch.

## 4. Chuyển model Python sang JavaScript

Script chuyển model:

```text
src/models/TimeSeries/export_svm_to_js.py
```

Ý tưởng của bước convert:

1. Load file `best_traditional_model.pkl` bằng `joblib`.
2. Tách `StandardScaler` và `SVC` trong pipeline.
3. Xuất các tham số của scaler:
   - `mean_`
   - `scale_`
4. Xuất trọng số Linear SVM:
   - `coef_`
   - `intercept_`
   - `classes_`
5. Sinh code JavaScript thuần cho:
   - `scaleFeatures(features)`
   - `scoreSVM(scaledFeatures)`
   - `predictPipeline(features)`

File model sau khi convert:

```text
deployment/scripts/svm_model.js
```

File này không cần thư viện ML bên thứ ba. Khi web app có đủ 12 features, nó chỉ gọi:

```javascript
const prediction = predictPipeline(features);
```

## 5. Cấu trúc deployment hiện tại

Các file web app nằm tại:

```text
deployment/scripts/
├── index.html
├── app.js
├── svm_model.js
└── favicon.svg
```

Vai trò từng file:

| File | Vai trò |
| --- | --- |
| `index.html` | Giao diện BlinkGuard, camera view, HUD chỉ số, nút bật notification |
| `app.js` | Logic runtime: MediaPipe, EAR, dynamic calibration, SVM inference, blink metrics, notification |
| `svm_model.js` | Model Linear SVM đã convert sang JavaScript |
| `favicon.svg` | Icon khiên xanh nước đậm với con mắt ở giữa |

## 6. Kiến trúc runtime của web app

Luồng xử lý chính:

```text
Webcam
  ↓
MediaPipe FaceLandmarker JS
  ↓
Face landmarks
  ↓
Tính EAR trái/phải
  ↓
rawEAR = min(leftEAR, rightEAR)
  ↓
Dynamic calibration + expanding max
  ↓
Sliding window 7 frame
  ↓
Hybrid heuristic filter
  ↓
12-feature extraction
  ↓
Linear SVM JavaScript inference
  ↓
Blink / Drowsy / Eye-health logic
  ↓
HUD + browser notification
```

## 7. Tracking mắt bằng MediaPipe JS

Web app dùng:

```javascript
@mediapipe/tasks-vision FaceLandmarker
```

Model FaceLandmarker được load từ CDN:

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

WASM runtime được load từ:

```text
https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm
```

Trong MVP, cách này giúp deploy nhanh vì không cần bundler, không cần npm, không cần build step. Khi cần bản production ổn định hơn, có thể self-host các asset MediaPipe để tránh phụ thuộc CDN.

## 8. Tính EAR trên trình duyệt

Landmark indices dùng để tính EAR:

```javascript
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
```

Công thức EAR:

```text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Web app lấy:

```javascript
rawEAR = Math.min(leftEAR, rightEAR)
```

Lý do dùng `min` thay vì average: nếu một mắt bị che hoặc MediaPipe ước lượng sai một bên mắt, lấy giá trị nhỏ hơn giúp hệ thống nhạy hơn với trạng thái nhắm/chớp thực tế.

## 9. Dynamic calibration

Ở giai đoạn đầu, app lấy 15 sample đầu tiên để xác định baseline mở mắt:

```text
currentMax = max(15 raw EAR đầu)
```

Nếu baseline quá thấp, app dùng ngưỡng fallback:

```text
0.23
```

Sau đó, trong quá trình chạy, nếu gặp EAR lớn hơn baseline hiện tại, app cập nhật:

```javascript
if (rawEAR > currentMax) {
  currentMax = rawEAR;
}
```

EAR chuẩn hóa:

```javascript
earNorm = rawEAR / currentMax
```

Giá trị được clamp về `[0, 1]`.

## 10. Sampling rate và sliding window

Model được train theo chuỗi EAR ở khoảng 15 FPS, nên web app không suy luận theo mọi frame của `requestAnimationFrame`. Thay vào đó, app downsample bằng:

```javascript
SAMPLE_INTERVAL_MS = 1000 / 15
```

Sliding window:

```javascript
WINDOW_SIZE = 7
```

Chỉ khi đủ 7 giá trị EAR chuẩn hóa, app mới chạy bước filter/model.

## 11. Hybrid heuristic filter

Để giảm số lần gọi model, web app dùng filter:

```javascript
if (Math.min(slidingWindow) >= 0.5) {
  prediction = 0;
}
```

Ý nghĩa: nếu toàn bộ cửa sổ EAR đều cao, mắt đang mở bình thường, không cần chạy SVM.

Nếu cửa sổ có dấu hiệu EAR giảm:

```javascript
features = extractAdvancedFeatures(slidingWindow)
prediction = predictPipeline(features)
```

HUD có hiển thị:

- `CPU Saved`
- `CPU Mode`: `Filter` hoặc `SVM`

## 12. Chức năng BlinkGuard hiện tại

BlinkGuard hiện có các chức năng chính sau:

### 12.1. Start / Stop / Reset camera

Người dùng có thể:

- Start Camera
- Stop
- Reset calibration/session

Webcam yêu cầu chạy trên:

- `localhost`, hoặc
- HTTPS domain, ví dụ Cloudflare Pages

### 12.2. Realtime HUD

Các chỉ số hiển thị:

| Chỉ số | Ý nghĩa |
| --- | --- |
| `Blinks / 60s` | Số blink trong 60 giây gần nhất |
| `Target` | Mục tiêu blink rate mặc định, hiện là 12/min |
| `3-min Trend` | Tốc độ blink trung bình theo trend tối đa 3 phút |
| `No-blink Streak` | Thời gian liên tục chưa blink |
| `Session Time` | Thời lượng phiên hiện tại |
| `Raw EAR` | EAR thô từ MediaPipe |
| `Norm EAR` | EAR đã chuẩn hóa theo calibration |
| `Prediction` | Class model dự đoán |
| `Calibration` | Tiến độ calibration 15 sample |
| `CPU Saved` | Tỷ lệ cửa sổ được filter bỏ qua SVM |
| `CPU Mode` | Đang dùng filter hay SVM |
| `Blink Count` | Tổng số blink trong phiên |
| `Drowsy Events` | Số lần phát hiện trạng thái buồn ngủ |
| `Notifications` | Trạng thái quyền system notification |

### 12.3. Cảnh báo không chớp mắt quá lâu

Ngưỡng hiện tại:

```javascript
LONG_NO_BLINK_MS = 15000
```

Nếu người dùng không chớp mắt quá 15 giây, app tạo warning:

```text
Long no-blink streak
```

Warning này hiển thị trong web và có thể gửi system notification nếu người dùng đã bật quyền.

### 12.4. Cảnh báo blink rate thấp

Sau khi phiên chạy đủ 60 giây, app kiểm tra số blink trong 60 giây gần nhất.

Ngưỡng hiện tại:

```javascript
TARGET_BLINKS_PER_MIN = 12
VERY_LOW_BLINKS_PER_MIN = 7
```

Các trạng thái:

- `< 12/min`: Low blink rate
- `< 7/min`: Very low blink rate

### 12.5. Cảnh báo buồn ngủ

App giữ logic drowsiness từ pipeline cũ:

```javascript
DROWSY_HISTORY_SIZE = 15
DROWSY_VOTES_REQUIRED = 10
```

Nếu trong history 15 prediction gần nhất có ít nhất 10 prediction class `2`, app xem là drowsy warning.

## 13. System notification ngoài web page

Để app không chỉ cảnh báo trong trang web, BlinkGuard dùng Browser Notification API.

Người dùng bấm:

```text
Enable Notifications
```

Sau đó browser sẽ xin quyền notification.

Nếu permission là `granted`, app có thể gửi notification khi:

- Không chớp mắt quá lâu.
- Blink rate quá thấp.
- Có dấu hiệu buồn ngủ.

Ví dụ notification cho no-blink streak:

```text
BlinkGuard: chớp mắt một chút nhé
Bạn đã không chớp mắt X giây. Chớp mắt vài lần và nhìn xa khỏi màn hình.
```

Để tránh spam, app có cooldown:

```javascript
NOTIFICATION_COOLDOWN_MS = 60000
```

Tức cùng một loại warning chỉ gửi tối đa 1 lần/phút.

Nếu thiết bị hỗ trợ, app gọi thêm:

```javascript
navigator.vibrate([160, 80, 160])
```

## 14. Favicon / nhận diện app

Icon hiện tại:

```text
deployment/scripts/favicon.svg
```

Thiết kế:

- Khiên màu xanh nước đậm.
- Con mắt ở giữa.
- Ý nghĩa: bảo vệ mắt, đúng với tên BlinkGuard.

File được link trong `index.html`:

```html
<link rel="icon" type="image/svg+xml" href="favicon.svg">
```

## 15. Hướng dẫn chạy local

Tại thư mục:

```bash
cd deployment/scripts
```

Chạy static server:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Mở:

```text
http://127.0.0.1:8000
```

Lưu ý:

- Webcam chạy được trên localhost.
- MediaPipe CDN cần internet.
- Notification cũng hoạt động trên localhost nếu browser cho phép.

## 16. Hướng dẫn deploy Cloudflare Pages

Vì app là static site, có thể deploy bằng Cloudflare Pages.

Cách nhanh nhất:

1. Vào Cloudflare Dashboard.
2. Chọn Workers & Pages.
3. Tạo Pages project.
4. Chọn Direct Upload.
5. Upload toàn bộ file trong:

```text
deployment/scripts/
```

Các file cần có:

```text
index.html
app.js
svm_model.js
favicon.svg
```

Nếu deploy qua Git:

```text
Build command: để trống
Build output directory: deployment/scripts
```

Cloudflare Pages cấp HTTPS mặc định, phù hợp với webcam và notification.

## 17. Quyền riêng tư và bảo mật

Thiết kế hiện tại bảo vệ privacy tốt hơn mô hình client-server:

- Webcam frame chỉ xử lý trong trình duyệt.
- Không upload ảnh/video lên server.
- SVM inference chạy bằng JavaScript local.
- Server/CDN chỉ phục vụ file tĩnh.

Các dữ liệu runtime như blink count, EAR, warning hiện không được lưu lên cloud.

## 18. Hạn chế hiện tại

Một số hạn chế cần ghi nhận:

1. **Phụ thuộc CDN MediaPipe**
   Nếu mạng chặn `cdn.jsdelivr.net` hoặc `storage.googleapis.com`, app không load được FaceLandmarker.

2. **Notification phụ thuộc browser**
   Desktop browser hỗ trợ tốt hơn. Mobile browser, đặc biệt iOS, có thể yêu cầu cài app dưới dạng PWA mới nhận notification ổn định.

3. **Blink count phụ thuộc chất lượng landmark**
   Ánh sáng yếu, mặt nghiêng, kính phản quang hoặc camera quá xa có thể làm EAR nhiễu.

4. **Ngưỡng blink/min chưa cá nhân hóa**
   Hiện target mặc định là `12/min`. Sau này có thể cho người dùng chỉnh target hoặc tự hiệu chuẩn theo baseline cá nhân.

5. **Chưa có âm thanh cảnh báo**
   MVP hiện dùng web alert và system notification. Có thể thêm âm thanh nhẹ nếu cần demo rõ hơn.

6. **Chưa lưu lịch sử phiên**
   App chưa có chart theo thời gian hoặc export session summary.

## 19. Hướng phát triển tiếp theo

Các nâng cấp đề xuất:

- Cho người dùng chỉnh `Target blinks/min`.
- Thêm âm thanh cảnh báo optional.
- Thêm PWA manifest để cài như app.
- Self-host MediaPipe WASM/model trong repo.
- Thêm biểu đồ blink/min theo thời gian.
- Lưu session summary vào `localStorage`.
- Thêm chế độ 20-20-20 reminder.
- Thêm calibration UX rõ hơn: countdown 15 sample.
- Tối ưu giao diện mobile.

## 20. Kết luận

Giai đoạn deployment đã chuyển thành công mô hình TimeSeries EAR từ Python sang web app client-side. Linear SVM được convert thành JavaScript thuần, giữ cả StandardScaler và logic multiclass SVM. Web app dùng MediaPipe FaceLandmarker để tính EAR theo thời gian thực, chạy dynamic calibration, sliding window, hybrid filter và SVM inference trực tiếp trên trình duyệt.

Từ MVP ban đầu phát hiện buồn ngủ, app đã được mở rộng thành BlinkGuard: một công cụ theo dõi sức khỏe mắt khi dùng màn hình, có blink/min, no-blink streak, drowsiness warning và system notification ngoài trang web. Kiến trúc hiện tại phù hợp để deploy miễn phí hoặc chi phí thấp trên Cloudflare Pages, đồng thời giữ được quyền riêng tư vì toàn bộ xử lý hình ảnh diễn ra trên thiết bị người dùng.
