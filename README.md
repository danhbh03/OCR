# OCR tiếng Việt chạy local

Ứng dụng web chạy trên Windows để OCR ảnh/PDF tiếng Việt, đối chiếu kết quả với ảnh gốc, sửa thủ công và lưu mẫu đã xác nhận.

## Cách hoạt động

- **PP-OCRv6 Medium** phát hiện dòng chữ; **DocLayout-YOLO** nhận diện bố cục và sắp xếp nội dung.
- **Tesseract** là bộ duy nhất nhận dạng chữ. Có thể chọn Tesseract + DocLayout-YOLO để đối chiếu, không dùng PP-OCRv6.
- PDF có lớp văn bản được dùng để thay nội dung OCR khi khớp; PDF scan được xử lý như ảnh.
- Không có tự động sửa chính tả. Người dùng tự sửa nội dung trước khi lưu.

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

Ảnh/PDF tải lên giới hạn 15 MB; PDF tối đa 25 trang. Thư mục dữ liệu local không thuộc mã nguồn và được Git bỏ qua.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

## Lưu ý giấy phép

DocLayout-YOLO được công bố theo AGPL-3.0; checkpoint được dùng có thẻ Apache-2.0. Hãy rà soát giấy phép của thư viện, model và dữ liệu trước khi phân phối ứng dụng. Thông tin này không thay thế tư vấn pháp lý.
