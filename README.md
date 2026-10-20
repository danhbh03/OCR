# OCR tiếng Việt chạy local

Ứng dụng web chạy trên Windows để OCR ảnh/PDF tiếng Việt, đối chiếu kết quả với ảnh gốc, sửa thủ công và lưu mẫu đã xác nhận.

## Cách hoạt động

- Kiến trúc tổng thể, luồng xử lý, API và cấu trúc dữ liệu được mô tả tại [ARCHITECTURE.md](./ARCHITECTURE.md).
- **Tesseract** là bộ duy nhất nhận dạng chữ. Chế độ mặc định dùng DocLayout-YOLO để phân tích bố cục rồi tự chọn cách xử lý: với trang chủ yếu là các khối văn bản lớn, Tesseract nhận dạng trực tiếp từng khối; với bố cục nhiều cột có khoảng cách cột rõ ràng, PP-OCR phát hiện dòng trong các khối DocLayout rồi Tesseract nhận dạng. Số lượng khung dòng đơn lẻ không tự kích hoạt nhánh nhiều cột; các khung trùng gần như hoàn toàn được loại bỏ.
- Với khung được DocLayout nhận diện là bảng, ứng dụng ưu tiên dò đường kẻ để cắt từng ô; Tesseract đọc theo thứ tự từng hàng từ trái sang phải. Các cột có phần lớn nội dung là số được nhận dạng lại bằng Tesseract với whitelist chữ số để giảm nhầm lẫn ký tự. Nếu bảng không có lưới rõ, PP-OCR tìm dòng chữ để Tesseract nhận dạng và sắp theo hàng; nếu không phát hiện được dòng thì Tesseract đọc toàn vùng bảng.
- Có thể chọn riêng từng nhánh: DocLayout + Tesseract nhận dạng trực tiếp, hoặc DocLayout + PP-OCR phát hiện dòng + Tesseract nhận dạng. PP-OCR chỉ phát hiện dòng, không nhận dạng chữ.
- PDF có lớp văn bản được dùng để thay đoạn OCR khớp nhất, tránh ghép thêm nội dung lân cận; PDF scan được xử lý như ảnh.
- Ảnh hỗ trợ JPEG/JPG (bao gồm MPO từ một số điện thoại), PNG, WEBP, BMP và TIFF; với MPO, OCR dùng ảnh chính.
- Không có tự động sửa chính tả. Người dùng tự sửa nội dung trước khi lưu.
- Ô kết quả máy ghép các dòng ảnh thành câu và chỉ xuống dòng sau dấu kết câu; tiêu đề, danh sách, và dòng kết thúc bằng dấu hai chấm được giữ riêng. Ô chỉnh sửa vẫn giữ từng dòng OCR để việc đối chiếu với khung ảnh và lưu mẫu huấn luyện chính xác.
nde
## Tiền xử lý ảnh xoay và giấy cong

Trước khi phát hiện bố cục và OCR, ảnh được chuẩn hóa theo thứ tự:

1. Đọc hướng EXIF nếu ảnh có metadata.
2. Dùng Tesseract OSD tìm hướng chữ. Chỉ tự xoay 90°, 180° hoặc 270° khi độ tin cậy OSD đạt ít nhất `5.0`; nếu ảnh quá ít chữ hoặc OSD không chắc chắn, giữ nguyên hướng để tránh xoay nhầm.
3. Tìm tứ giác trang và hiệu chỉnh phối cảnh nếu nhận diện được viền trang; khi viền khó thấy, thử tách nền tối và vùng giấy sáng trước khi bỏ qua hiệu chỉnh.
4. Nếu không tìm được viền trang, dùng Hough lines để deskew góc từ `0.4°` đến `80°`. Điều này hỗ trợ ảnh nghiêng đáng kể; các góc xoay vuông được xử lý riêng bằng OSD.
5. Thử dewarp nhẹ toàn trang trước khi phát hiện bố cục; nếu không đủ tín hiệu cong, ảnh được giữ nguyên. Các khung DocLayout và dòng OCR vẫn lấy trên ảnh đã căn chỉnh này.
6. DocLayout phát hiện khung nội dung và mở rộng theo chiều dọc 3 px. Nếu mức này làm các khung riêng biệt chồng lấn, lần lượt thử 2 px rồi 1 px; luôn giới hạn trong biên ảnh.
7. Trong từng khung, thử dewarp thêm chỉ khi còn đủ tín hiệu cong sau bước toàn trang. Nếu khung không có đủ dòng hoặc không phát hiện cong đáng kể, giữ nguyên khung đó. Nếu không tìm được khung, luồng dự phòng chạy trên ảnh đã dewarp toàn trang (nếu có).
8. Khi cắt từng dòng PP-OCR cho Tesseract, chừa tối đa 3 px theo chiều dọc để giữ dấu tiếng Việt nằm sát mép polygon. Nếu có dòng khác gần đó, tự giảm khoảng chừa để crop không lấn sang dòng bên cạnh; không thay đổi khung DocLayout. Trên từng vùng Tesseract, tăng tương phản cục bộ khi cần, phóng to chữ nhỏ và bỏ qua vùng gần như trống.

### Tham số dewarping

Các giá trị dưới đây là mặc định trong `app/ocr.py`; thay đổi chúng sẽ ảnh hưởng mức độ nhạy và độ mạnh của hiệu chỉnh:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `DEWARP_MIN_PAGE_DIMENSION` | 160 px | Không thử dewarp nếu cạnh ngắn của ảnh/khung nhỏ hơn giá trị này; vùng nhỏ thường không đủ mẫu hình học ổn định. |
| `DEWARP_THRESHOLD_BLOCK_SIZE` | 31 px | Kích thước lân cận cho adaptive threshold. Tự giảm cho ảnh nhỏ và luôn giữ số lẻ; lớn hơn xét vùng rộng hơn nhưng kém thích ứng với bóng cục bộ. |
| `DEWARP_THRESHOLD_C` | 15 | Độ lệch ngưỡng adaptive threshold; tăng giá trị làm mask ít nhạy hơn với nét mực nhạt và nền nhiễu, giảm giá trị giữ lại nhiều nét hơn nhưng dễ bắt bóng/giấy bẩn. |
| `DEWARP_PROJECTION_KERNEL_SIZE` | 3 hàng | Làm mượt phép chiếu mực theo trục dọc để tìm các dải dòng; lớn hơn ổn định hơn nhưng có thể nhập các dòng sát nhau. |
| `DEWARP_MIN_INK_WIDTH_RATIO` | 0.003 | Ngưỡng sàn của mật độ mực theo chiều rộng khung khi xác định hàng có chữ. |
| `DEWARP_PEAK_PROJECTION_RATIO` | 0.08 | Ngưỡng tương đối so với hàng có mật độ mực lớn nhất; giúp thích ứng với độ đậm/nhạt khác nhau. |
| `DEWARP_MIN_LINE_BAND_HEIGHT` | 3 px | Bỏ qua các dải thấp hơn mức này vì thường là nhiễu. |
| `DEWARP_MAX_LINE_BAND_HEIGHT_RATIO` | 0.1 | Bỏ qua dải cao hơn 10% chiều cao khung; dải quá cao thường là vùng đồ họa hoặc nhiều dòng bị nhập làm một. |
| `DEWARP_STRIP_COUNT` | 32 | Chia khung thành 32 dải dọc để đo độ lệch dòng cục bộ. Tăng số dải cho phép mô tả cong chi tiết hơn nhưng nhạy nhiễu hơn. |
| `DEWARP_STRIP_INK_RATIO` | 0.015 | Mỗi dải phải có đủ mực trong một vùng dòng mới được dùng để đo độ lệch. |
| `DEWARP_MIN_TEXT_LINES` | 4 | Cần ít nhất 4 dải dòng trong ảnh hoặc khung; mỗi dải dọc cũng cần tín hiệu từ ít nhất một nửa con số này trước khi đóng góp. |
| `DEWARP_MIN_VALID_STRIP_RATIO` | 0.5 | Ít nhất một nửa các dải dọc phải có phép đo hợp lệ để nội suy; nếu không, bỏ qua dewarp. |
| `DEWARP_MIN_VALID_STRIPS` | 8 | Số dải dọc hợp lệ tối thiểu, ngay cả khi thay đổi tổng số dải. |
| `DEWARP_SMOOTHING_RATIO` | 0.08 | Tỷ lệ tạo cửa sổ Gaussian làm mượt độ lệch giữa các dải; mặc định tương đương khoảng 3 dải. |
| `DEWARP_MIN_SMOOTHING_WIDTH` | 3 dải | Cửa sổ làm mượt tối thiểu; cửa sổ thực tế được làm thành số lẻ. |
| `DEWARP_MIN_DISPLACEMENT_RATIO` | 0.002 | Nếu biên độ độ cong đo được nhỏ hơn 0.2% chiều cao khung thì không warp, tránh nội suy không cần thiết. |
| `DEWARP_MAX_DISPLACEMENT_RATIO` | 0.025 | Giới hạn hiệu chỉnh mỗi vị trí ở ±2.5% chiều cao khung để tránh biến dạng quá mạnh khi đo sai. |
| `DEWARP_MIN_THRESHOLD_BLOCK_SIZE` | 3 px | Giới hạn dưới cho kích thước adaptive threshold sau khi điều chỉnh theo ảnh nhỏ. |
| `DEWARP_MIN_MAX_LINE_BAND_HEIGHT` | 8 px | Sàn cho giới hạn kích thước dải chữ, để ảnh độ phân giải thấp vẫn tìm được dòng. |

Đây là dewarping cổ điển dựa trên hình học, không phải mô hình khôi phục tài liệu đã huấn luyện. Hiệu chỉnh toàn trang được giới hạn và chỉ áp dụng khi đo thấy độ cong đủ rõ; sau đó từng khung chỉ nhận hiệu chỉnh bổ sung nếu còn cong. Nên so sánh ảnh gốc và kết quả trên tài liệu thực; giấy nhàu mạnh hoặc nếp gấp cục bộ có thể cần mô hình dewarping chuyên dụng.

## Yêu cầu

- Windows 64-bit, Python 3.10 và Conda (Miniconda hoặc Anaconda).
- Tesseract có dữ liệu ngôn ngữ tiếng Việt (`vie`).
- Kết nối Internet lần đầu chạy để tải model OCR/bố cục.
- GPU là tùy chọn; chế độ GPU cần NVIDIA GPU tương thích với PaddlePaddle CUDA 11.8.

## Cài đặt

Mở Miniconda Prompt tại thư mục project và tạo **một** trong hai môi trường:

### CPU

```powershell
conda create -n ocr-local python=3.10 pip -y
conda activate ocr-local
conda install -c conda-forge tesseract -y
python -m pip install -r requirements.txt
```

### GPU NVIDIA

```powershell
conda create -n ocr-gpu python=3.10 pip -y
conda activate ocr-gpu
conda install -c conda-forge tesseract -y
python -m pip install -r requirements-gpu.txt
```

Không cài chồng hai bộ requirements vào cùng một môi trường. Tesseract cần có ngôn ngữ `vie`; có thể kiểm tra bằng:

```powershell
tesseract --list-langs
```

## Chạy ứng dụng

Nhấp đúp file tương ứng:

- `Chay_OCR_CPU.bat` — dùng môi trường `ocr-local`.
- `Chay_OCR_GPU.bat` — dùng môi trường `ocr-gpu`.

Launcher tự mở <http://127.0.0.1:8000>. Đóng cửa sổ server để dừng ứng dụng. Nếu vừa cập nhật mã nguồn, dừng server cũ rồi khởi động lại. Tài liệu API: <http://127.0.0.1:8000/docs>.

Có thể chạy thủ công khi môi trường đã được kích hoạt:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Dữ liệu và giới hạn

Launcher lưu ảnh tải lên, cơ sở dữ liệu, mẫu huấn luyện và cache model tại `%LOCALAPPDATA%\OCR-Vietnamese\data`. Nếu chạy Uvicorn thủ công, mặc định dữ liệu nằm trong thư mục `data` của project; có thể đặt biến môi trường `OCR_DATA_DIR` để đổi vị trí.

Có thể chọn và tải lên nhiều ảnh/PDF trong một lần; ứng dụng nhận dạng tuần tự từng tệp và báo riêng các tệp lỗi. Mỗi ảnh/PDF giới hạn 15 MB; PDF tối đa 25 trang. Thư mục dữ liệu local không thuộc mã nguồn và được Git bỏ qua.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

## Lưu ý giấy phép

DocLayout-YOLO được công bố theo AGPL-3.0; checkpoint được dùng có thẻ Apache-2.0. Hãy rà soát giấy phép của thư viện, model và dữ liệu trước khi phân phối ứng dụng. Thông tin này không thay thế tư vấn pháp lý.
