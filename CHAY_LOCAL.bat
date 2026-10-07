@echo off
cd /d "%~dp0"
echo ===============================================
echo QuanLyThiTrucTuyen V5.27.9 HYBRID SMART - CHAY LOCAL
echo ===============================================
python --version
if errorlevel 1 (
  echo KHONG TIM THAY PYTHON. Hay cai Python hoac them Python vao PATH.
  pause
  exit /b 1
)
echo.
echo Dang chay app.py ...
python -u app.py
echo.
echo Chuong trinh da dung. Neu co loi, hay chup TOAN BO thong bao phia tren.
pause
