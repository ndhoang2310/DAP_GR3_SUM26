# Hướng Dẫn Quy Trình Xử Lý Dữ Liệu (Data Processing Workflow)

Quy trình xử lý dữ liệu của dự án bao gồm 3 bước chính: Thu thập video, Trích xuất ảnh mắt, và Gán nhãn dữ liệu. Bạn hãy thực hiện lần lượt các bước dưới đây.

## Bước 1: Thu thập Video (Video Collection)

Sử dụng camera để quay các đoạn video ngắn (khoảng 12 giây) người dùng đang chớp mắt hoặc nhắm mắt. Video sẽ luôn được tự động xuất ra với chuẩn **15 FPS** dù phần cứng camera của bạn đang quay ở tốc độ cao hơn (như 30 FPS) hay chậm hơn.
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

Dữ liệu hình ảnh cần được gán nhãn `Open` (Mở mắt) hoặc `Closed` (Nhắm mắt) để phục vụ cho việc huấn luyện mô hình sau này. Có 2 chế độ gán nhãn:

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

---

## Bước 4: Làm việc nhóm và Gộp dữ liệu (Dành cho Leader/ML Team)

Khi làm việc nhóm, mỗi người tự thực hiện từ Bước 1 đến Bước 3 trên máy cá nhân. Sau đó, nhóm cần gom dữ liệu lại để có một Master Dataset.



**4.1. Giải thích về DVC (Dành cho toàn bộ thành viên):**
Dự án sử dụng **DVC (Data Version Control)** để tự động hóa việc đồng bộ dữ liệu nặng qua Google Drive:
- **Tiết kiệm & Thông minh**: Khi đẩy lên, DVC **chỉ tải thư mục `data/raw_videos/` và `dataset/`**. Khi bạn chạy script nhiều lần, DVC tự động nhận diện và chỉ tải lên các video/ảnh mới, bỏ qua file cũ giúp tiết kiệm tối đa thời gian.
- **Tại sao thư mục Drive lại lộn xộn?** DVC biến Google Drive thành một "kho chứa ngầm" (database). Các file được băm thành mã MD5 và lưu vào các thư mục tên 2 chữ cái để chống trùng lặp. Vì vậy, **bạn không thể và không cần xem ảnh trực tiếp trên trình duyệt web Drive**. Việc khôi phục lại cấu trúc thư mục đẹp đẽ sẽ được hệ thống của Leader tự lo.

**4.2. Cấu hình DVC (Đã được Leader hoàn tất):**
Leader đã khởi tạo DVC, liên kết Google Drive và nạp mã xác thực (OAuth). Cấu hình này đã được đẩy lên Github nên các thành viên không cần thiết lập lại.

**4.3. Dành cho các thành viên (Upload Dữ Liệu Tự Động):**
Mọi thành viên sau khi quay video và gán nhãn xong, chỉ cần thao tác:
1. Đảm bảo đã cài DVC: `pip install -r requirements.txt`
2. Chạy script đồng bộ siêu tốc:
```bash
python src/data_collection/dvc_sync.py
```
3. **Xác thực Google (Chỉ lần đầu tiên)**: Khi cửa sổ trình duyệt hiện ra, bạn đăng nhập bằng email Google của mình. Nếu Google hiện cảnh báo an toàn *(Google hasn't verified this app)*, bạn cứ nhấn **Advanced (Nâng cao)** -> Chọn **Go to dap-dvc (unsafe)** -> Chọn **Continue (Tiếp tục)** là xong.
- Script sẽ tự lo toàn bộ phần việc còn lại: Tạo nhánh `data/M0X`, đẩy dữ liệu lên Drive, và push nhánh lên Github. Bạn chỉ việc ngồi uống nước chờ báo `[DONE]`.

**4.4. Dành cho Leader (Tổng hợp Dữ Liệu Tự Động):**
Leader không cần lên web tải từng file zip nữa. Chỉ việc chạy lệnh:
```bash
python src/data_collection/dvc_pull_and_merge.py
```
- Khi được hỏi, Leader nhập mã các thành viên cần gộp (VD: `M01, M02`). DVC sẽ tự động kết nối với kho Drive, tải chính xác hình ảnh của những người đó về máy và khôi phục lại thành cấu trúc thư mục người đọc được.
- Script sau đó sẽ tự kích hoạt cơ chế gộp, tạo ra thư mục `dataset_master/` và file `metadata_master.csv` hoàn chỉnh cuối cùng.
