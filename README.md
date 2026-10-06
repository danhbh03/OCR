# OCR tiếng Việt chạy local

Ứng dụng dùng FastAPI, hiển thị ảnh gốc và kết quả OCR để người dùng chỉnh sửa, lưu mẫu đã xác nhận. Chạy trên Windows với **Python 3.10.x 64-bit**.

## Cài đặt lần đầu

Cài Miniconda/Anaconda, mở PowerShell tại thư mục project và chọn **một** cấu hình:

**GPU NVIDIA:**

```powershell
conda create -n ocr-gpu python=3.10 pip -y
conda activate ocr-gpu
conda install -c conda-forge cuda-version=12.0 cudnn=8.9.1.23 -y
python -m pip install -r requirements-gpu.txt
```

**CPU:**

```powershell
conda create -n ocr-local python=3.10 pip -y
conda activate ocr-local
python -m pip install -r requirements.txt
```

Không cài hai file requirements vào cùng một môi trường. Bản GPU yêu cầu driver NVIDIA tương thích CUDA 12; nếu không dùng được GPU, cài bản CPU.

## Chạy ứng dụng

Sau khi cài đặt, nhấp đúp một trong hai file:

- `Chay_OCR_GPU.bat` — chạy bằng môi trường `ocr-gpu`.
- `Chay_OCR_CPU.bat` — chạy bằng môi trường `ocr-local` (cũng nhận `ocr-local-cpu` hoặc `.venv`).

Giữ `Chay_OCR_Launcher.bat` cùng thư mục với hai file trên. Trình duyệt mở tại <http://127.0.0.1:8000>. Đóng cửa sổ lệnh để dừng ứng dụng; dừng server hiện tại trước khi chuyển chế độ CPU/GPU.

Ảnh, cơ sở dữ liệu và mẫu được lưu tại `%LOCALAPPDATA%\OCR-Vietnamese\data`, không lưu trong thư mục project. Lần đầu chạy OCR có thể cần tải trọng số mô hình. Tài liệu API có tại <http://127.0.0.1:8000/docs>.
