# Start Fake News Detector with Gemini Backend
# This script sets up the environment and starts the server

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "FAKE NEWS DETECTOR - GEMINI AI BACKEND" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Set API key
$env:GEMINI_API_KEY = "AIzaSyBOAtbyGB2fw-QRtPjx8o2hXeJSLR1pfl4"
Write-Host "✅ Gemini API Key configured" -ForegroundColor Green
Write-Host ""

# Navigate to directory
Set-Location -Path "D:\probaliti\fake_news_detector\fake_news_detector"

# Check if google-generativeai is installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$pythonExe = "D:\probaliti\fake_news_detector\.venv\Scripts\python.exe"

$checkPackage = & $pythonExe -c "import google.generativeai; print('OK')" 2>&1
if ($checkPackage -like "*OK*") {
    Write-Host "✅ google-generativeai is installed" -ForegroundColor Green
} else {
    Write-Host "⚠️  Installing google-generativeai..." -ForegroundColor Yellow
    & "$pythonExe" -m pip install google-generativeai --quiet
    Write-Host "✅ google-generativeai installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting server..." -ForegroundColor Yellow
Write-Host "Web UI will be available at: http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Run the server
& $pythonExe web_ui.py
