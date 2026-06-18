@echo off
title PD Detect - Backend API Server
color 0A
echo.
echo  ========================================
echo    PD Detect ^| Backend API Server
echo    http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo  ========================================
echo.

cd /d "%~dp0backend"

if not exist venv\Scripts\python.exe (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 ( echo [ERROR] Python not found. Install Python 3.10+. & pause & exit )
)

echo [INFO] Installing/verifying dependencies...
venv\Scripts\pip install -q -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit )

if not exist models\ensemble.pkl (
    echo.
    echo [INFO] Trained models not found. Running model training...
    echo [INFO] This downloads the UCI dataset and trains RF+XGBoost+SVM (~2 min)...
    echo.
    venv\Scripts\python model_trainer.py
    if errorlevel 1 ( echo [ERROR] Model training failed. & pause & exit )
    echo.
    echo [INFO] Models trained and saved successfully!
    echo.
)

echo [INFO] Starting FastAPI server on port 8000...
echo [INFO] Press Ctrl+C to stop.
echo.
venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
