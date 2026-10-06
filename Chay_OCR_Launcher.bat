@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="gpu" (
    set "ENV_NAME=ocr-gpu"
    set "MODE_NAME=GPU"
) else if /i "%~1"=="cpu" (
    set "ENV_NAME=ocr-local"
    set "MODE_NAME=CPU"
) else (
    echo Khong xac dinh duoc che do CPU/GPU.
    exit /b 1
)

set "APP_URL=http://127.0.0.1:8000"
curl.exe --silent --fail "%APP_URL%/api/health" >nul 2>&1
if not errorlevel 1 (
    echo OCR server da chay tren cong 8000. Hay dung server hien tai truoc khi doi CPU/GPU.
    start "" "%APP_URL%"
    pause
    exit /b 0
)

if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "OCR_DATA_DIR=%LOCALAPPDATA%\OCR-Vietnamese\data"
if not exist "%OCR_DATA_DIR%" mkdir "%OCR_DATA_DIR%"
if not exist "%OCR_DATA_DIR%" (
    echo Khong the tao thu muc luu du lieu OCR: %OCR_DATA_DIR%
    pause
    exit /b 1
)

if defined CONDA_EXE if not exist "%CONDA_EXE%" set "CONDA_EXE="
if not defined CONDA_EXE if exist "D:\Miniconda\Scripts\conda.exe" set "CONDA_EXE=D:\Miniconda\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE for /f "delims=" %%I in ('where.exe conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%~fI"

set "ENV_PATH="
set "LAUNCH_WITH_CONDA="
if defined CONDA_EXE for %%I in ("%CONDA_EXE%\..\..") do set "CONDA_ROOT=%%~fI"
if defined CONDA_ROOT if exist "%CONDA_ROOT%\envs\%ENV_NAME%\python.exe" (
    set "ENV_PATH=%CONDA_ROOT%\envs\%ENV_NAME%"
    set "LAUNCH_WITH_CONDA=1"
)

if /i "%~1"=="cpu" if not defined ENV_PATH if exist "%CONDA_ROOT%\envs\ocr-local-cpu\python.exe" (
    set "ENV_PATH=%CONDA_ROOT%\envs\ocr-local-cpu"
    set "LAUNCH_WITH_CONDA=1"
)
if /i "%~1"=="cpu" if not defined ENV_PATH if exist "%~dp0.venv\Scripts\python.exe" set "ENV_PATH=%~dp0.venv"

if not defined ENV_PATH (
    echo Khong tim thay moi truong %MODE_NAME%.
    echo Hay cai theo huong dan trong README.md:
    echo   GPU: Conda environment ocr-gpu va requirements-gpu.txt
    echo   CPU: Conda environment ocr-local va requirements.txt
    pause
    exit /b 1
)

echo Dang khoi dong OCR %MODE_NAME% tu %ENV_PATH%
start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$deadline=(Get-Date).AddSeconds(60); while((Get-Date) -lt $deadline){try{$r=Invoke-WebRequest -UseBasicParsing '%APP_URL%/api/health' -TimeoutSec 2;if($r.StatusCode -eq 200){Start-Process '%APP_URL%';exit}}catch{};Start-Sleep -Seconds 1};Write-Error 'Khong the ket noi OCR server sau 60 giay.'"

if defined LAUNCH_WITH_CONDA (
    call "%CONDA_EXE%" run --no-capture-output -p "%ENV_PATH%" python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
) else (
    "%ENV_PATH%\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
)

if errorlevel 1 (
    echo OCR server da dung do co loi. Kiem tra thong bao loi o tren.
    pause
)
