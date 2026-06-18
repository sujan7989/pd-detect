@echo off
title PD Detect - Frontend (React)
color 0B
echo.
echo  ========================================
echo    PD Detect ^| Frontend React App
echo    http://localhost:5173
echo  ========================================
echo.

cd /d "%~dp0frontend"

if not exist node_modules (
    echo [INFO] Installing npm packages (first time ~1-2 min)...
    call npm install
    if errorlevel 1 ( echo [ERROR] npm install failed. Install Node.js 18+. & pause & exit )
)

echo [INFO] Starting Vite dev server...
echo [INFO] Opening http://localhost:5173 in your browser...
echo [INFO] Press Ctrl+C to stop.
echo.
call npm run dev
pause
