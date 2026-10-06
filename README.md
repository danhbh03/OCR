# OCR tiếng Việt chạy local

Ứng dụng dùng FastAPI, hiển thị ảnh gốc và kết quả OCR để người dùng chỉnh sửa, lưu mẫu đã xác nhận. Chạy trên Windows với **Python 3.10.x 64-bit**.

## Cài GPU NVIDIA trên Windows

### Điều kiện

- Máy có GPU NVIDIA và driver NVIDIA hỗ trợ CUDA 12; kiểm tra driver bằng `nvidia-smi` trong PowerShell.
- Cài Miniconda/Anaconda. Dùng Python **3.10 x64** trong môi trường Conda bên dưới.
- Không cần cài CUDA Toolkit toàn hệ thống. Môi trường Conda sẽ cài CUDA runtime và cuDNN mà PaddleOCR cần.

### Tạo môi trường GPU

Mở **Miniconda Prompt** hoặc PowerShell đã dùng được lệnh `conda`, chuyển tới thư mục project, rồi chạy:

```powershell
cd D:\MCP_AI\OCR
conda create -n ocr-gpu python=3.10 pip -y
conda activate ocr-gpu
conda install -c conda-forge cuda-version=12.0 cudnn=8.9.1.23 -y
python -m pip install -r requirements-gpu.txt
```

Các lệnh tạo môi trường và cài thư viện chỉ cần chạy một lần. Nếu đã có môi trường `ocr-gpu`, bỏ qua lệnh `conda create` và kích hoạt môi trường đó bằng `conda activate ocr-gpu`.

File GPU cài PaddlePaddle GPU `2.6.1.post120` (CUDA 12.0) và PyTorch `2.5.1+cu121` (CUDA 12.1). Đây là hai runtime đi kèm riêng theo từng framework; không đổi phiên bản tùy ý và không cài `requirements.txt` CPU vào môi trường GPU.

### Kiểm tra GPU

Khi vẫn đang kích hoạt `ocr-gpu`, chạy:

```powershell
python -c "import paddle; print('Paddle CUDA:', paddle.is_compiled_with_cuda()); paddle.set_device('gpu:0'); print('Paddle device:', paddle.to_tensor([1]).place)"
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'không khả dụng')"
```

Kết quả mong đợi: Paddle CUDA là `True`, thiết bị Paddle có dạng `Place(gpu:0)`, PyTorch CUDA là `True` và tên GPU NVIDIA được in ra.

### Chạy ứng dụng bằng GPU

Sau khi cài xong, nhấp đúp `Chay_OCR_GPU.bat`. Ứng dụng dùng môi trường `ocr-gpu` và tự mở trình duyệt tại <http://127.0.0.1:8000>. Giữ `Chay_OCR_Launcher.bat` cùng thư mục với file chạy. Đóng cửa sổ lệnh để dừng server.

Nếu không khởi động được GPU, kiểm tra driver bằng `nvidia-smi`, xác nhận đã cài đúng requirements trong `ocr-gpu`, rồi chạy lại hai lệnh kiểm tra ở trên. Nếu Paddle báo thiếu DLL hoặc CUDA/cuDNN, kích hoạt `ocr-gpu` và cài lại runtime Conda:

```powershell
conda activate ocr-gpu
conda install -c conda-forge cuda-version=12.0 cudnn=8.9.1.23 -y
```

Nếu GPU không khả dụng trên máy đó, dùng môi trường CPU riêng bên dưới, không cài chồng hai bộ requirements.

## Cài đặt CPU

Nếu không có GPU NVIDIA hoặc không muốn cài CUDA, dùng bản CPU trong một môi trường riêng:

```powershell
conda create -n ocr-local python=3.10 pip -y
conda activate ocr-local
python -m pip install -r requirements.txt
```

## Chạy ứng dụng

Sau khi cài môi trường tương ứng, nhấp đúp một trong hai file:

- `Chay_OCR_GPU.bat` — chạy bằng môi trường `ocr-gpu`.
- `Chay_OCR_CPU.bat` — chạy bằng môi trường `ocr-local` (cũng nhận `ocr-local-cpu` hoặc `.venv`).

Giữ `Chay_OCR_Launcher.bat` cùng thư mục với hai file trên. Cả hai chế độ dùng cổng 8000; dừng server hiện tại trước khi chuyển CPU/GPU.

Ảnh, cơ sở dữ liệu và mẫu được lưu tại `%LOCALAPPDATA%\OCR-Vietnamese\data`, không lưu trong thư mục project. Lần đầu chạy có thể cần kết nối Internet để tải trọng số mô hình. Tài liệu API có tại <http://127.0.0.1:8000/docs>.
