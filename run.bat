@echo off
title Waste Detection AI System
echo ============================================================
echo Starting Waste Detection AI System...
echo ============================================================
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    echo Using Virtual Environment Python: .\venv\Scripts\python.exe
    venv\Scripts\python.exe app.py
) else (
    echo Virtual environment not found, trying system python...
    python app.py
)
pause

