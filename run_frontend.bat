@echo off
setlocal enabledelayedexpansion
title PathSnap Evaluator Dashboard (Laptop 2)

echo ==============================================================================
echo                   PathSnap Evaluator Dashboard Launcher
echo                         (Streamlit + Folium UI)
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

:: 2. Launch Streamlit
echo.
echo [*] Starting Streamlit dashboard on port 8501...
echo [*] Open your browser at: http://localhost:8501
echo [*] Press Ctrl+C to stop the dashboard.
echo.

.venv\Scripts\streamlit run frontend/app.py --server.port 8501
pause
