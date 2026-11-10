@echo off
rem Double-click to start the ALL study app. The first run installs everything.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Install Python 3.11, 3.12 or 3.13 from https://www.python.org/downloads/
    echo During installation tick "Add python.exe to PATH", then double-click this file again.
    pause
    exit /b 1
)

if not exist ".venv\setup-done.txt" (
    echo First start: installing the app. This downloads about 1 GB and can take 5 to 15 minutes.
    python -m venv .venv
    if errorlevel 1 goto failed
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
    echo done > ".venv\setup-done.txt"
)

echo Starting the app. It opens in your browser; keep this window open while you use it.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
pause
exit /b 0

:failed
echo.
echo Installation failed. Check your internet connection, delete the ".venv" folder and try again.
echo If it still fails, send a screenshot of this window to the person who shared the app.
pause
exit /b 1
