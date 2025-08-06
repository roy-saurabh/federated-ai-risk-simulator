@echo off
echo 🤖 Federated AI Risk Simulator Launcher
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install dependencies if needed
echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

echo 🚀 Starting Federated AI Risk Simulator...
echo The application will open in your default web browser.
echo If it doesn't open automatically, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo.

REM Run the Streamlit app
streamlit run federated_ai_risk_simulator.py

echo.
echo 👋 Simulator stopped. Thanks for using the Federated AI Risk Simulator!
pause 