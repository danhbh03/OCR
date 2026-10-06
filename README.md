# OCR tiếng Việt local (MVP)

Ứng dụng mẫu dùng FastAPI để nhận ảnh, chạy OCR cục bộ, hiển thị ảnh gốc cạnh kết quả và lưu bản người dùng chỉnh sửa vào SQLite. Ảnh và cơ sở dữ liệu nằm trong thư mục `data/` mặc định.

## Yêu cầu và cài đặt trên Windows

- **Python 3.10.x 64-bit** là phiên bản khuyến nghị và được dùng để kiểm thử; không dùng Python 3.12+ cho môi trường này.
- Cài Miniconda/Anaconda hoặc Python 3.10 trước khi tiếp tục.
- Chọn đúng một file requirements cho mỗi môi trường: `requirements-gpu.txt` cho NVIDIA GPU hoặc `requirements.txt` (CPU) nếu máy không có GPU, CUDA không tương thích, hay bản GPU không cài/chạy được.
- Bản Windows GPU hiện ghim PaddlePaddle 2.6.1 CUDA 12.0/cuDNN 8.9.1 và PyTorch 2.5.1 CUDA 12.1; dùng Python 3.10 x64 và NVIDIA driver tương thích CUDA 12. Lệnh Conda bên dưới cài CUDA runtime/cuDNN cần cho Paddle; PyTorch wheel mang runtime riêng.

### Dùng Miniconda/Anaconda

Mở PowerShell tại thư mục project:

```powershell
cd D:\MCP_AI\OCR
conda create -n ocr-local python=3.10 pip -y
conda activate ocr-local
conda install -c conda-forge cuda-version=12.0 cudnn=8.9.1.23 -y
python -m pip install -r requirements-gpu.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`requirements-gpu.txt` cài wheel Paddle GPU CUDA 12.0 và PyTorch CUDA 12.1; lệnh Conda phía trên cung cấp runtime CUDA/cuDNN 8 mà PaddleOCR 2.6 cần trên Windows.

Nếu CUDA/cuDNN không tương thích hoặc kiểm tra GPU thất bại, tạo môi trường sạch và dùng bản CPU:

```powershell
conda create -n ocr-local-cpu python=3.10 pip -y
conda activate ocr-local-cpu
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Nếu `conda` chưa có trong PATH, gọi Python của environment trực tiếp:

```powershell
& "D:\Miniconda\Scripts\conda.exe" run -n ocr-local python -m pip install -r requirements-gpu.txt
& "D:\Miniconda\Scripts\conda.exe" run -n ocr-local python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Dùng Python 3.10 không cần Conda

Chọn bản CPU nếu không muốn cài CUDA/cuDNN. Nếu cần GPU, dùng Miniconda theo mục trên vì PaddleOCR GPU trên Windows cần runtime CUDA/cuDNN trong environment.

```powershell
cd D:\MCP_AI\OCR
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Nếu máy không có NVIDIA GPU hoặc cài CUDA thất bại, thay `requirements-gpu.txt` bằng `requirements.txt` trong lệnh cài và dùng một môi trường Python sạch. Không cài hai file requirements vào cùng một environment vì chúng chứa các bản Paddle/PyTorch CPU và GPU loại trừ nhau.

Mở <http://127.0.0.1:8000>. Lần đầu chạy, PaddleOCR/VietOCR có thể tải trọng số mô hình; các lần sau dùng cache trên máy. Ứng dụng tự chọn CUDA khi framework báo GPU khả dụng, nếu không thì chạy CPU. PyTorch CPU được ghim trong `requirements.txt` làm bản dự phòng; NumPy dưới phiên bản 2 để tương thích PaddleOCR 2.x. `setuptools<81` được ghim vì VietOCR/gdown cần `pkg_resources`.

Trước OCR, ứng dụng thử tìm viền tờ giấy bằng cạnh ảnh và tứ giác lớn nhất có hình dạng phù hợp. Khi phát hiện đủ tin cậy, trang được nắn phối cảnh thành hình chữ nhật; nếu không, ứng dụng ước lượng góc nghiêng nhỏ từ các đoạn thẳng dài rồi xoay ảnh, mở rộng nền để không cắt nội dung. Nếu không ước lượng được góc đáng tin cậy thì giữ nguyên ảnh. Sau đó áp dụng CLAHE nhẹ trên kênh sáng để hỗ trợ ảnh thiếu tương phản; bước này không thể khôi phục ảnh mất nét. Ảnh gốc không bị thay đổi, và tọa độ vùng chữ được ánh xạ về hệ tọa độ ảnh gốc.

Ảnh chữ được căn phối cảnh theo tứ giác PaddleOCR phát hiện trước khi đưa từng dòng sang VietOCR. Tự động tìm giấy phụ thuộc vào độ tương phản giữa giấy và nền; ảnh giấy trắng trên nền trắng, viền bị che khuất hoặc ảnh cắt sát mép có thể khiến bước này bỏ qua hoặc chọn sai viền. Hãy kiểm tra ảnh đầu vào nếu kết quả bị cắt hoặc nắn sai.

VietOCR trong mẫu này dùng PaddleOCR phát hiện vùng chữ, sau đó nhận dạng từng dòng bằng VietOCR và là lựa chọn mặc định cho tiếng Việt in. `lang="vi"` của PaddleOCR 2.x dùng bộ nhận dạng đa ngôn ngữ Latin, không phải mô hình chuyên biệt tiếng Việt. CLAHE chỉ hỗ trợ chất lượng ảnh, không thể khắc phục giới hạn của mô hình hoặc thay thế fine-tune trên dữ liệu phù hợp. Kết quả được chuẩn hóa Unicode NFC để dấu tổ hợp được lưu nhất quán; bước này không tự sửa ký tự bị nhận dạng nhầm. Trọng số VietOCR có thể được tải về ở lần chạy đầu tiên. Mẫu đặt thiết bị VietOCR là CPU để hoạt động được trên máy không có GPU.

Các chứng từ chuyển khoản có tên người, số tài khoản, mã giao dịch và số tiền cần đối chiếu với ảnh. Không tự động sửa các giá trị này theo từ điển hay ngữ cảnh vì OCR có thể biến một số hợp lệ thành một số khác. Bản người dùng chỉnh sửa vẫn là nguồn chính xác để kiểm duyệt.

## API

- `POST /api/documents`: multipart form với `file` và `engine` (`paddle` hoặc `vietocr`).
- `GET /api/documents`: danh sách tài liệu và kết quả đã lưu.
- `GET /api/documents/{id}/image`: ảnh gốc.
- `POST /api/documents/{id}/regions`: JSON `{ "x": 100, "y": 200, "width": 300, "height": 80, "edited_text": "..." }` để OCR vùng bị bỏ sót; có thể gọi nhiều lần, kết quả được chèn theo thứ tự đọc trên ảnh (từ trên xuống, cùng dòng từ trái sang phải). `edited_text` giúp giữ các sửa đổi chưa lưu trên giao diện.
- `POST /api/documents/{id}/detections/adjust`: nhận `{ "detection_index": 2, "edited_line_index": 2, "x": 100, "y": 200, "width": 300, "height": 80, "edited_text": "..." }` để OCR lại khung và thay đúng dòng được chọn, đồng thời cập nhật bounding box.
- `PUT /api/documents/{id}`: JSON `{ "edited_text": "..." }` để lưu bản sửa.
- `GET /api/training-samples`: danh sách crop dòng đã thay đổi và nhãn sửa; mỗi mẫu có URL lấy ảnh crop.
- `GET /api/training-samples?engine=vietocr` hoặc `?engine=paddle`: chỉ lấy mẫu nhận dạng thuộc đúng engine.
- `GET /api/detection-samples`: mẫu vùng phát hiện PaddleOCR đã được người dùng nới/xác nhận.
- `GET /api/training-data/paddle/detection`: nhãn vùng theo dạng points/transcription để chuẩn bị tập fine-tune detector.
- `GET /api/health`: kiểm tra ứng dụng.

Trong giao diện, đặt con trỏ hoặc chọn một hay nhiều dòng trong ô văn bản đã sửa để tô sáng vùng OCR tương ứng trên ảnh gốc. Bấm **Chọn vùng bị bỏ sót** rồi kéo trên ảnh để OCR vùng đó; có thể lặp lại thao tác để thêm nhiều vùng. Kết quả được chèn theo tọa độ từ trên xuống, và các vùng cùng dòng được xếp từ trái sang phải; nếu OCR không nhận ra vùng, nội dung hiện tại vẫn được giữ để bạn thử lại bằng khung rộng hơn. Để sửa vùng phát hiện quá nhỏ, đặt con trỏ trong đúng một dòng, bấm **Điều chỉnh vùng dòng đang chọn**, rồi kéo một khung rộng hơn quanh dòng. App OCR lại vùng đó, thay nội dung đúng dòng đang chọn (kể cả các sửa đổi hiện có ở dòng đó), cập nhật tọa độ và lưu crop/đa giác vào tập `paddle-det`.

## Phạm vi và bước tiếp theo

- Đây là bản nền để thử luồng sản phẩm, chưa phải mô hình chuyên biệt cho chữ viết tay. Chất lượng cần được đo trên tài liệu thật; chữ viết tay thường cần dữ liệu gán nhãn và fine-tune riêng.
- Mỗi tài liệu lưu cả văn bản OCR gốc và bản chỉnh sửa. Chỉ dùng các bản đã được kiểm duyệt để tạo dữ liệu huấn luyện; hiện tại ứng dụng chưa tự fine-tune mô hình.
- Dữ liệu fine-tune được tách theo mục tiêu: crop + transcript trong `data/training_samples/` mang `engine=paddle` hoặc `engine=vietocr` cho model nhận dạng tương ứng; crop vùng và đa giác trong `data/detection_samples/` mang `target_engine=paddle-det` cho detector PaddleOCR. Dedupe recognition dùng cặp engine + pixel fingerprint để crop giống nhau không bị gộp giữa hai engine. API trả về từng nhóm riêng. Chỉnh vùng OCR vừa tạo detector label; sau khi người dùng kiểm tra/sửa transcript rồi lưu, phần nhận dạng được thêm riêng cho engine của tài liệu. Chưa có lệnh tự fine-tune; dữ liệu vẫn cần kiểm duyệt, xuất theo định dạng huấn luyện đúng model và chia train/validation trước khi chạy PaddleOCR training. Muốn huấn luyện cả detector và recognizer cần tập ảnh/nhãn đủ lớn, sạch và tách tập kiểm tra; chỉnh vài bounding box chỉ tạo nhãn mẫu, không tự làm mô hình tốt lên ngay.
- SQLite phù hợp một máy và ít người dùng. Khi cần tách máy xử lý, có thể giữ API/UI ở máy điều phối rồi chuyển tác vụ tới worker riêng.
- Nếu chưa cài engine OCR, API báo lỗi rõ ràng thay vì trả kết quả giả.
- Có thể đổi vị trí dữ liệu bằng biến môi trường `OCR_DATA_DIR`; giới hạn ảnh mặc định 15 MB và có thể đổi bằng `OCR_MAX_UPLOAD_BYTES`.
