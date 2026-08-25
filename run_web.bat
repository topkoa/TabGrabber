@echo off
REM Launch the TabGrabber web UI, preferring a local virtual environment.
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    ".venv\Scripts\python.exe" run_web.pyw %*
) else (
    python run_web.pyw %*
)
