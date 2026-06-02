# Hướng Dẫn Quy Trình Xử Lý Dữ Liệu (Data Processing Workflow)

Quy trình xử lý dữ liệu của dự án bao gồm 3 bước chính: Thu thập video, Trích xuất ảnh mắt, và Gán nhãn dữ liệu. Bạn hãy thực hiện lần lượt các bước dưới đây.

## Bước 1: Thu thập Video (Video Collection)

Sử dụng camera để quay các đoạn video ngắn (khoảng 12 giây) người dùng đang chớp mắt hoặc nhắm mắt.
- **Lệnh thực thi**:
  ```bash
  python src/data_collection/collect_video.py
  ```
- **Kết quả**: Video sẽ được lưu dưới định dạng `.mp4` vào thư mục `data/raw_videos/`.

## Bước 2: Trích xuất hình ảnh mắt (Eye Extraction)

Hệ thống sẽ đọc các video thô, sử dụng **MediaPipe Face Mesh** để nhận diện khuôn mặt và cắt ra các vùng mắt (trái/phải) với kích thước 24x24 pixel.
- **Lệnh thực thi (Chạy ngầm)**:
  ```bash
  python src/data_collection/extract_eyes.py
  ```
- **Lệnh thực thi (Có giao diện xem trước)**:
  ```bash
  python src/data_collection/extract_eyes.py --preview
  ```
- **Kết quả**: 
  - Các hình ảnh mắt 24x24 được lưu vào thư mục `dataset/raw_eyes/`.
  - Một file `metadata.csv` được tạo ra chứa các chỉ số EAR (Eye Aspect Ratio) của từng khung hình.

## Bước 3: Gán nhãn dữ liệu (Data Labeling)

Dữ liệu hình ảnh cần được gán nhãn `Open` (Mở mắt) hoặc `Closed` (Nhắm mắt) để phục vụ cho việc huấn luyện mô hình sau này. Có 3 chế độ gán nhãn:

**1. Gán nhãn tự động (Auto Labeling):**
Dựa vào chỉ số EAR và một ngưỡng (threshold) để tự động phân loại.
```bash
python src/data_collection/label_tool.py --mode auto 
```

**2. Đánh giá và chỉnh sửa thủ công (Manual Review):**
Sau khi gán tự động, bạn nên kiểm tra lại bằng mắt thường. Giao diện review sẽ hiện ra từng ảnh mắt.
```bash
python src/data_collection/label_tool.py --mode review
```
*Các phím tắt hỗ trợ trong quá trình review:*
- `o`: Đổi nhãn thành **Open** (Mở mắt).
- `c`: Đổi nhãn thành **Closed** (Nhắm mắt).
- `d` hoặc `x`: **Xóa bỏ (Discard)** ảnh lỗi (ảnh mờ, cắt sai vị trí).
- `SPACE`: Bỏ qua (Giữ nguyên nhãn hiện tại).
- `q`: Lưu tiến độ và thoát.

**3. Chia tập Train/Test (Split Dataset):**
Sau khi hoàn tất gán nhãn, chia bộ dữ liệu vào các thư mục `train` (80%) và `test` (20%).
```bash
python src/data_collection/label_tool.py --mode split
```
- **Kết quả**: Dữ liệu sẽ được tự động phân bổ vào `dataset/train/` và `dataset/test/` sẵn sàng cho các bước Machine Learning.

---

## Bước 4: Làm việc nhóm và Gộp dữ liệu (Dành cho Leader/ML Team)

Khi làm việc nhóm, mỗi người tự thực hiện từ Bước 1 đến Bước 3 trên máy cá nhân. Sau đó, nhóm cần gom dữ liệu lại để có một Master Dataset.

**4.1. Nộp dữ liệu (Upload):**
1. Tạo một thư mục chung trên Google Drive (VD: `EyeBlink_Contributions`).
2. Bên trong, tạo các thư mục mang tên từng thành viên (VD: `Hoa/`, `Nam/`).
3. Mỗi thành viên **nén (zip)** thư mục `dataset/` của mình lại thành `dataset.zip` và tải lên thư mục tên mình trên Drive.

**4.2. Gộp dữ liệu tự động (Merge):**
Người phụ trách tổng hợp dữ liệu sẽ tải tất cả các file `dataset.zip` về máy, giải nén và đặt vào thư mục `data/contributions/` trong dự án theo cấu trúc:
```text
data/contributions/
├── Hoa/
│   └── dataset/  <-- Folder dataset của Hoa sau khi giải nén
├── Nam/
│   └── dataset/  <-- Folder dataset của Nam sau khi giải nén
```
Sau đó, chạy công cụ gộp dữ liệu:
```bash
python src/data_collection/merge_datasets.py
```
- **Kết quả**: 
  - Tạo ra một thư mục `dataset_master/` chứa toàn bộ ảnh của cả nhóm đã được trộn chung vào các thư mục `train` và `test`.
  - Tạo ra một file `metadata_master.csv` tổng hợp tất cả các dòng dữ liệu của cả nhóm, đồng thời thêm cột `contributor` để biết ảnh nào do ai đóng góp.
  - Bạn có thể lấy thư mục `dataset_master/` này để tiến hành train model mà không sợ bị xung đột file!
