# Hướng Dẫn Quy Trình Xử Lý Dữ Liệu (Data Processing Workflow)

Quy trình xử lý dữ liệu của dự án bao gồm 3 bước chính: Thu thập video, Trích xuất ảnh mắt, và Gán nhãn dữ liệu. Bạn hãy thực hiện lần lượt các bước dưới đây.

> ⚠️ **QUAN TRỌNG – ĐỌC TRƯỚC KHI BẮT ĐẦU:**  
> Nếu bạn đang **thu thập dữ liệu lại (làm lại dataset)**, hãy đọc kỹ mục [Bước 0: Dọn Dẹp Dữ Liệu Cũ](#bước-0-dọn-dẹp-dữ-liệu-cũ-bắt-buộc-khi-thu-thập-lại) và [Checklist Kiểm Tra Trước Khi Upload](#-checklist-bắt-buộc-trước-khi-chạy-dvc_syncpy) trước khi thực hiện bất kỳ thao tác nào.

---

## Bước 0: Dọn Dẹp Dữ Liệu Cũ (BẮT BUỘC Khi Thu Thập Lại)

> 🚨 **BÀI HỌC RÚT RA:** Trong đợt thu thập trước, một số thành viên quên xóa dữ liệu cũ trước khi quay mới rồi đẩy lên DVC, khiến **dữ liệu cũ và mới bị trộn lẫn**, gây loạn toàn bộ dataset của nhóm. Đây là lỗi nghiêm trọng nhất và mất rất nhiều thời gian để sửa.

Khi Leader thông báo **"Làm lại dataset"**, **TOÀN BỘ thành viên** phải thực hiện các bước sau **TRƯỚC KHI** quay video mới:

**Bước 0.1: Xóa sạch video cũ và ảnh mắt cũ trên máy cá nhân:**
```powershell
# Trên Windows PowerShell
Remove-Item -Path "data\raw_videos\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "dataset\raw_eyes\*" -Recurse -Force -ErrorAction SilentlyContinue
```
```bash
# Trên macOS / Linux
rm -rf data/raw_videos/*
rm -rf dataset/raw_eyes/*
```

**Bước 0.2: Kiểm tra lại thư mục đã trống:**
```powershell
# Phải không còn file nào trong 2 thư mục này
dir data\raw_videos
dir dataset\raw_eyes
```

**Bước 0.3: TUYỆT ĐỐI KHÔNG chạy `dvc pull`:**
- Lệnh `dvc pull` sẽ kéo toàn bộ dữ liệu cũ từ Google Drive về lại máy của bạn, gây lẫn lộn dữ liệu.
- Chỉ chạy `dvc pull` khi Leader yêu cầu cụ thể.

---

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

## 📋 CHECKLIST BẮT BUỘC TRƯỚC KHI CHẠY `dvc_sync.py`

> 🚨 **Đây là bước kiểm tra cuối cùng mà MỌI thành viên PHẢI thực hiện trước khi đẩy dữ liệu lên.** Nếu dữ liệu của bạn có lỗi, nó sẽ ảnh hưởng đến toàn bộ nhóm.

### ✅ 1. Kiểm tra không còn dữ liệu cũ lẫn vào
- Đảm bảo thư mục `data/raw_videos/` **chỉ chứa các video MỚI** của đợt thu thập này.
- Đảm bảo thư mục `dataset/raw_eyes/` **chỉ chứa ảnh mắt trích xuất từ video mới**.
- Nếu bạn thấy file có tên video từ đợt trước (ví dụ: kịch bản khác, ngày khác), hãy xóa chúng đi.

### ✅ 2. Kiểm tra `metadata.csv` có đầy đủ dữ liệu
Mở file `dataset/raw_eyes/metadata.csv` và kiểm tra:
- **Cột `ear_left` và `ear_right`:** Không được có hàng nào bị trống (empty) ở các dòng có `status = success`. Nếu có dòng bị trống EAR, đó là dấu hiệu MediaPipe không detect được mắt đúng cách → cần quay lại video đó.
- **Cột `final_label`:** Sau khi chạy `label_tool.py --mode auto`, mọi dòng có `status = success` đều phải có nhãn (`open` hoặc `closed`). Nếu có dòng bị thiếu nhãn (missing label), hãy chạy lại lệnh auto-label.
- **Số lượng ảnh:** Đếm số file `.png` trong thư mục `dataset/raw_eyes/` phải khớp với số dòng `status = success` trong `metadata.csv`.

```powershell
# Kiểm tra nhanh trên PowerShell
# Đếm số ảnh
(Get-ChildItem dataset\raw_eyes\*.png).Count

# Kiểm tra metadata có dòng nào bị thiếu EAR không
python -c "import csv; rows=list(csv.DictReader(open('dataset/raw_eyes/metadata.csv'))); missing=[r for r in rows if r['status']=='success' and (not r['ear_left'] or not r['ear_right'])]; print(f'Missing EAR: {len(missing)} rows') if missing else print('EAR OK')"

# Kiểm tra metadata có dòng nào bị thiếu label không
python -c "import csv; rows=list(csv.DictReader(open('dataset/raw_eyes/metadata.csv'))); missing=[r for r in rows if r['status']=='success' and not r['final_label']]; print(f'Missing labels: {len(missing)} rows') if missing else print('Labels OK')"
```

### ✅ 3. Kiểm tra cấu trúc thư mục ĐÚNG CHUẨN

> 🚨 **BÀI HỌC RÚT RA:** Trong đợt trước, một số thành viên tự ý di chuyển file hoặc chạy script từ thư mục sai, dẫn đến cấu trúc folder bị khác nhau giữa các thành viên, khiến script gộp dữ liệu của Leader bị lỗi.

Cấu trúc **BẮT BUỘC** trên máy của mỗi thành viên phải đúng y như sau:
```text
DAP_GR3_SUM26/              ← Thư mục gốc của dự án
├── data/
│   └── raw_videos/         ← Chứa file .mp4 (video quay trực tiếp)
│       ├── M0X_SC1_E01_G0_D1.mp4
│       └── ...
├── dataset/
│   └── raw_eyes/           ← Chứa ảnh mắt 24x24 + metadata
│       ├── metadata.csv
│       ├── labels.csv
│       ├── ear_values.csv
│       ├── M0X_SC1_E01_G0_D1_frame00001_left.png
│       └── ...
├── config.py
├── dataset.dvc
└── ...
```

**Các lỗi thường gặp về cấu trúc:**
| Sai | Đúng |
|-----|-------|
| Đặt ảnh mắt trực tiếp vào `dataset/` | Ảnh mắt phải nằm trong `dataset/raw_eyes/` |
| Đặt `metadata.csv` ở thư mục gốc | `metadata.csv` phải nằm trong `dataset/raw_eyes/` |
| Tạo thêm thư mục con bên trong `raw_eyes/` | Tất cả ảnh `.png` phải nằm trực tiếp trong `raw_eyes/`, không có subfolder |
| Chạy script từ thư mục `src/` | **Luôn chạy mọi lệnh từ thư mục gốc `DAP_GR3_SUM26/`** |

### ✅ 4. Đảm bảo đã chạy ĐẦY ĐỦ 3 bước theo đúng thứ tự
1. `collect_video.py` → Quay video
2. `extract_eyes.py` → Trích xuất ảnh mắt + tạo `metadata.csv`
3. `label_tool.py --mode auto` → Gán nhãn tự động (tạo `labels.csv`)
4. *(Khuyến khích)* `label_tool.py --mode review` → Duyệt lại bằng mắt thường

**Nếu bạn bỏ qua bước 3**, file `metadata.csv` sẽ có các cột `auto_label`, `final_label` bị trống rỗng → Leader gộp dữ liệu sẽ bị thiếu nhãn (missing labels).

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
1. Đảm bảo cập nhật code và cấu hình DVC mới nhất từ Leader: `git pull origin main`
2. Đảm bảo đã cài DVC: `pip install -r requirements.txt`
3. **Hoàn thành CHECKLIST ở trên** ← ⚠️ ĐỪNG BỎ QUA
4. Chạy script đồng bộ siêu tốc:
```bash
python src/data_collection/dvc_sync.py
```
5. **Xác thực Google (Chỉ lần đầu tiên)**: Khi cửa sổ trình duyệt hiện ra, bạn đăng nhập bằng email Google của mình. Nếu Google hiện cảnh báo an toàn *(Google hasn't verified this app)*, bạn cứ nhấn **Advanced (Nâng cao)** -> Chọn **Go to dap-dvc (unsafe)** -> Chọn **Continue (Tiếp tục)** là xong.
- Script sẽ tự lo toàn bộ phần việc còn lại: Tạo nhánh `data/M0X`, đẩy dữ liệu lên Drive, và push nhánh lên Github. Bạn chỉ việc ngồi uống nước chờ báo `[DONE]`.

**4.4. Dành cho Leader (Tổng hợp Dữ Liệu Tự Động):**
Leader không cần lên web tải từng file zip nữa. Chỉ việc chạy lệnh:
```bash
python src/data_collection/dvc_pull_and_merge.py
```
- Khi được hỏi, Leader nhập mã các thành viên cần gộp (VD: `M01, M02`). DVC sẽ tự động kết nối với kho Drive, tải chính xác hình ảnh của những người đó về máy và khôi phục lại thành cấu trúc thư mục người đọc được.
- Script sau đó sẽ tự kích hoạt cơ chế gộp, tạo ra thư mục `dataset_master/` và file `metadata_master.csv` hoàn chỉnh cuối cùng.

---

## ⚠️ CÁC LỖI THƯỜNG GẶP & CÁCH PHÒNG TRÁNH (Lessons Learned)

### Lỗi 1: Dữ liệu cũ bị trộn lẫn với dữ liệu mới
- **Nguyên nhân:** Thành viên quên xóa video/ảnh cũ trước khi quay mới rồi chạy `dvc_sync.py`.
- **Hậu quả:** Dataset bị loạn, chứa cả dữ liệu cũ lẫn mới, phải làm lại từ đầu.
- **Cách phòng tránh:** Luôn thực hiện **Bước 0** trước khi bắt đầu thu thập mới.

### Lỗi 2: Thiếu nhãn (Missing Labels)
- **Nguyên nhân:** Thành viên chỉ chạy `extract_eyes.py` mà **quên chạy** `label_tool.py --mode auto` trước khi upload.
- **Hậu quả:** File `metadata.csv` có các cột `auto_label` và `final_label` bị trống. Khi gộp, dataset sẽ thiếu nhãn, không train model được.
- **Cách phòng tránh:** Luôn chạy đủ 3 bước theo thứ tự. Dùng lệnh kiểm tra nhanh trong Checklist ở trên.

### Lỗi 3: Thiếu giá trị EAR (Missing `ear_left` hoặc `ear_right`)
- **Nguyên nhân có thể:**
  - Mắt bị che khuất một phần (tóc, tay, gọng kính dày) khiến MediaPipe detect sai landmark.
  - Ánh sáng quá yếu hoặc quá chói khiến camera không bắt được chi tiết khuôn mặt.
  - Góc nghiêng đầu quá lớn (trên 30°) khiến một bên mắt bị méo dạng.
  - Ngồi quá xa camera (Distance = D2) khiến mắt quá nhỏ để detect chính xác.
- **Cách phòng tránh:** 
  - Quay video ở khoảng cách vừa phải (D1: 50-60cm).
  - Đảm bảo ánh sáng đủ và đồng đều trên khuôn mặt.
  - Không để tóc che mắt, nếu đeo kính hãy đảm bảo kính sạch không bị lóa.
  - Sau khi trích xuất xong, **kiểm tra `metadata.csv`** xem có dòng nào bị trống EAR không (dùng lệnh trong Checklist).

### Lỗi 4: Cấu trúc thư mục sai khác giữa các thành viên
- **Nguyên nhân:** Thành viên chạy script từ thư mục sai (ví dụ: chạy từ bên trong `src/` thay vì thư mục gốc), hoặc tự ý di chuyển/đổi tên file.
- **Hậu quả:** Script gộp dữ liệu của Leader không tìm thấy file đúng vị trí, bị bỏ sót hoặc báo lỗi.
- **Cách phòng tránh:**
  - **LUÔN** mở terminal/PowerShell tại thư mục gốc `DAP_GR3_SUM26/` trước khi chạy bất kỳ lệnh nào.
  - **KHÔNG TỰ Ý** di chuyển, đổi tên, hoặc tạo thêm thư mục con bên trong `dataset/raw_eyes/`.
  - Kiểm tra cấu trúc thư mục đúng chuẩn theo bảng trong Checklist.

### Lỗi 5: Chạy `extract_eyes.py` nhiều lần gây trùng lặp ảnh
- **Nguyên nhân:** Thành viên chạy lại `extract_eyes.py` mà quên rằng lần chạy trước đã tạo ảnh trong `dataset/raw_eyes/`.
- **Hậu quả:** Nếu video gốc vẫn còn, script sẽ ghi đè ảnh cùng tên (không sao). Nhưng nếu bạn đã xóa một video cũ và thêm video mới, ảnh từ video cũ vẫn nằm trong thư mục → dataset bị lẫn.
- **Cách phòng tránh:** Nếu muốn chạy lại `extract_eyes.py`, hãy xóa sạch `dataset/raw_eyes/` trước rồi mới chạy lại.

### Lỗi 6: Mất kết nối mạng giữa chừng khi `dvc push`
- **Nguyên nhân:** Wifi không ổn định, hoặc upload quá lâu do dữ liệu nặng.
- **Hậu quả:** Dữ liệu chỉ được upload một phần lên Google Drive. Khi Leader gộp sẽ bị thiếu file.
- **Cách phòng tránh:**
  - Đảm bảo kết nối mạng ổn định trước khi chạy `dvc_sync.py`.
  - Nếu bị ngắt giữa chừng, **chạy lại** `dvc_sync.py` là được. DVC sẽ tự tiếp tục upload các file còn thiếu mà không upload lại từ đầu.

### Lỗi 7: Quên kích hoạt Virtual Environment (venv)
- **Nguyên nhân:** Thành viên mở terminal mới mà quên chạy `venv\Scripts\activate`.
- **Hậu quả:** Lệnh `dvc` hoặc `python` không tìm thấy các thư viện đã cài → báo lỗi `ModuleNotFoundError` hoặc `dvc is not recognized`.
- **Cách phòng tránh:** Luôn chạy lệnh sau mỗi khi mở terminal mới:
  ```powershell
  venv\Scripts\activate
  ```

### Lỗi 8: Camera không mở được hoặc chọn sai camera
- **Nguyên nhân:** Máy có nhiều camera (webcam + camera ngoài), hoặc ứng dụng khác (Zoom, Teams) đang chiếm camera.
- **Cách phòng tránh:**
  - Tắt hết các ứng dụng đang dùng camera trước khi quay.
  - Nếu webcam mặc định không phải camera bạn muốn, sửa giá trị `CAMERA_INDEX` trong `config.py` (thử `0`, `1`, `2`...).

---

## 📝 LƯU Ý BỔ SUNG CHO CHẤT LƯỢNG DỮ LIỆU

1. **Kiểm tra bằng `--preview` trước khi quay chính thức:** Chạy `extract_eyes.py --preview` với 1 video thử nghiệm ngắn để đảm bảo MediaPipe detect được mặt và mắt của bạn ổn định. Nếu preview thấy ảnh mắt bị cắt sai hoặc detect lung tung, hãy thay đổi góc ngồi/ánh sáng.

2. **Không dùng phần mềm filter/beauty khi quay:** Một số laptop có phần mềm camera tích hợp bộ lọc làm đẹp (beauty filter, HDR filter). Các bộ lọc này có thể làm biến dạng vùng mắt và ảnh hưởng đến chỉ số EAR → phải tắt hết.

3. **Mỗi thành viên nên quay ĐỦ các kịch bản (Scenario) đã phân công:** Không nên chỉ quay 1-2 kịch bản rồi upload. Dataset cần đa dạng để mô hình học được nhiều trường hợp.

4. **Sau khi review bằng `label_tool.py --mode review`, nhớ nhấn `q` để lưu:** Nếu tắt cửa sổ bằng nút X mà không nhấn `q`, tiến độ review có thể bị mất.

5. **Báo cáo lỗi cho Leader ngay lập tức:** Nếu bạn gặp bất kỳ lỗi nào bất thường (script crash, ảnh mắt bị đen, EAR = 0 toàn bộ...), hãy chụp màn hình và báo Leader TRƯỚC KHI upload để tránh đẩy dữ liệu lỗi lên hệ thống chung.
