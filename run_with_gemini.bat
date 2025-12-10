@echo off
REM Start Fake News Detector with Gemini Backend

echo ============================================================
echo STARTING FAKE NEWS DETECTOR WITH GEMINI AI
echo ============================================================
echo.

REM Set the API key
set GEMINI_API_KEY=AIzaSyBOAtbyGB2fw-QRtPjx8o2hXeJSLR1pfl4

REM Navigate to the directory
cd /d D:\probaliti\fake_news_detector\fake_news_detector

echo Starting server...
echo.
echo Web UI will be available at: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the server
D:\probaliti\fake_news_detector\.venv\Scripts\python.exe web_ui.py

pause
