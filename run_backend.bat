@echo off
setlocal enabledelayedexpansion
title PathSnap Backend Server (Laptop 1)

echo ==============================================================================
echo                      PathSnap Backend Server Launcher
echo                     (FastAPI + ML Model + HMM Decoder)
echo ==============================================================================

:: Change directory to repo root
cd /d "%~dp0"

:: 1. Check or create virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment in .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [!] ERROR: Python 3 not found on PATH. Please install Python 3.10+ and add to PATH.
        pause
        exit /b 1
    )
    echo [*] Installing dependencies from requirements.txt...
    .venv\Scripts\pip install -r requirements.txt
    if errorlevel 1 (
        echo [!] ERROR: Failed to install requirements.
        pause
        exit /b 1
    )
)

:: 2. Display Local IP Address for Laptop 2
echo.
echo ==============================================================================
echo [INFO] Detected Local Network IP Addresses on this Laptop:
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike '*Loopback*' -and $_.IPAddress -notlike '169.254*' } | ForEach-Object { Write-Host ('   -> IP: ' + $_.IPAddress + ' (' + $_.InterfaceAlias + ')') -ForegroundColor Cyan }"
echo.
echo [ACTION FOR LAPTOP 2]:
echo    In the Streamlit Dashboard on Laptop 2, enter:
echo    Backend URL: http://^<YOUR_IP_ABOVE^>:8000
echo ==============================================================================
echo.
echo [*] Starting FastAPI server on 0.0.0.0:8000...
echo [*] Press Ctrl+C to stop the server.
echo.

.venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 8000
pause
