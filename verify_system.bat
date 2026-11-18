@echo off
setlocal enabledelayedexpansion
title PathSnap System Verification

echo ==============================================================================
echo                      PathSnap System Verification
echo               (Unit Tests, Contract Tests, & 10-Trip E2E)
echo ==============================================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment .venv not found. Please run run_backend.bat or setup .venv first.
    pause
    exit /b 1
)

echo.
echo [1/2] Running Unit & Contract Tests (pytest)...
.venv\Scripts\python -m pytest -q
if errorlevel 1 (
    echo [!] Tests failed!
    pause
    exit /b 1
)

echo.
echo [2/2] Running End-to-End Verification Across All 10 Student Test Trips...
.venv\Scripts\python frontend/verify_evaluation_trips.py
if errorlevel 1 (
    echo [!] E2E verification failed!
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo [SUCCESS] All unit, contract, and E2E evaluation tests passed!
echo ==============================================================================
pause
