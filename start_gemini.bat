@echo off
REM Gemini-powered Fake News Detector Launcher
REM This script helps you set up and run the detector with Gemini AI

echo ============================================================
echo FAKE NEWS DETECTOR - GEMINI AI BACKEND
echo ============================================================
echo.

REM Check if GEMINI_API_KEY is set
if not defined GEMINI_API_KEY (
    echo WARNING: GEMINI_API_KEY is not set!
    echo.
    echo To use Gemini AI for predictions, you need to set your API key.
    echo.
    echo Option 1: Set it now temporarily:
    echo   set GEMINI_API_KEY=your-api-key-here
    echo.
    echo Option 2: Get your API key from:
    echo   https://makersuite.google.com/app/apikey
    echo.
    echo Press any key to continue without Gemini (will use ML model)...
    pause > nul
    echo.
    echo Continuing without Gemini backend...
    echo.
) else (
    echo Gemini API Key detected: %GEMINI_API_KEY:~0,10%...
    echo.
)

REM Check if google-generativeai is installed
echo Checking for google-generativeai package...
python -c "import google.generativeai" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: google-generativeai is not installed!
    echo Installing it now...
    pip install google-generativeai
    echo.
)

echo ============================================================
echo Starting Fake News Detector Web Server...
echo ============================================================
echo.
echo The web interface will be available at:
echo   http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python web_ui.py

pause
