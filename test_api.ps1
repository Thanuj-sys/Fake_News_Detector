# Test the Fake News Detector API
# Make sure web_ui.py is running first!

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host " Testing Fake News Detector API" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

# Check if server is running
Write-Host "[1/6] Checking if server is running..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:5000/health" -Method GET
    Write-Host "[OK] Server is running!" -ForegroundColor Green
    Write-Host "  Status: $($health.status)" -ForegroundColor Gray
    Write-Host "  Detector Ready: $($health.detector_ready)`n" -ForegroundColor Gray
} catch {
    Write-Host "[ERROR] Server is not running!" -ForegroundColor Red
    Write-Host "  Please start it with: python web_ui.py`n" -ForegroundColor Red
    exit 1
}

# Test 1: Real news from Reuters
Write-Host "[2/6] Testing REAL news from reliable source (Reuters)..." -ForegroundColor Yellow
$test1 = @{
    title = "Federal Reserve Announces Interest Rate Decision"
    text = "The Federal Reserve announced today that it will maintain current interest rates following a two-day policy meeting. The decision comes as inflation continues to moderate and the labor market remains stable."
    source = "reuters.com"
} | ConvertTo-Json

$result1 = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Body $test1 -ContentType "application/json"
Write-Host "  Prediction: $($result1.label)" -ForegroundColor $(if ($result1.label -eq "REAL") {"Green"} else {"Red"})
Write-Host "  Combined Probability: $($result1.combined_prob)" -ForegroundColor Gray
Write-Host "  Web Score: $($result1.web_score)`n" -ForegroundColor Gray

# Test 2: Fake news with sensational language
Write-Host "[3/6] Testing FAKE news with sensational language..." -ForegroundColor Yellow
$test2 = @{
    title = "SHOCKING: You Won't Believe This!!!"
    text = "BREAKING NEWS!!! Scientists discover AMAZING secret! You won't BELIEVE what they found! Click NOW before this gets DELETED! This will CHANGE EVERYTHING!"
    source = "clickbait-news.fake"
} | ConvertTo-Json

$result2 = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Body $test2 -ContentType "application/json"
Write-Host "  Prediction: $($result2.label)" -ForegroundColor $(if ($result2.label -eq "FAKE") {"Green"} else {"Red"})
Write-Host "  Combined Probability: $($result2.combined_prob)" -ForegroundColor Gray
Write-Host "  Web Score: $($result2.web_score)`n" -ForegroundColor Gray

# Test 3: Real news from BBC
Write-Host "[4/6] Testing REAL news from BBC..." -ForegroundColor Yellow
$test3 = @{
    title = "New Climate Study Published in Nature"
    text = "Researchers at Oxford University published findings examining climate change effects on ocean temperatures. The peer-reviewed study analyzed two decades of data from monitoring stations worldwide."
    source = "bbc.com"
} | ConvertTo-Json

$result3 = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Body $test3 -ContentType "application/json"
Write-Host "  Prediction: $($result3.label)" -ForegroundColor $(if ($result3.label -eq "REAL") {"Green"} else {"Red"})
Write-Host "  Combined Probability: $($result3.combined_prob)" -ForegroundColor Gray
Write-Host "  Web Score: $($result3.web_score)`n" -ForegroundColor Gray

# Test 4: Conspiracy theory
Write-Host "[5/6] Testing FAKE conspiracy theory..." -ForegroundColor Yellow
$test4 = @{
    title = "EXPOSED: Secret Government Plot!"
    text = "Anonymous sources reveal shocking conspiracy! The deep state is hiding the TRUTH! Wake up sheeple! They don't want you to know this! Share before they censor it!"
    source = "conspiracy-truth.blog"
} | ConvertTo-Json

$result4 = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Body $test4 -ContentType "application/json"
Write-Host "  Prediction: $($result4.label)" -ForegroundColor $(if ($result4.label -eq "FAKE") {"Green"} else {"Red"})
Write-Host "  Combined Probability: $($result4.combined_prob)" -ForegroundColor Gray
Write-Host "  Web Score: $($result4.web_score)`n" -ForegroundColor Gray

# Test 5: Real tech news
Write-Host "[6/6] Testing REAL tech news..." -ForegroundColor Yellow
$test5 = @{
    title = "Apple Announces New Product Release"
    text = "Apple Inc. confirmed its latest iPhone model will be available for pre-order next Friday. The device features improved camera and battery life. The announcement was made at the company's annual event in Cupertino."
    source = "techcrunch.com"
} | ConvertTo-Json

$result5 = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Body $test5 -ContentType "application/json"
Write-Host "  Prediction: $($result5.label)" -ForegroundColor $(if ($result5.label -eq "REAL") {"Green"} else {"Red"})
Write-Host "  Combined Probability: $($result5.combined_prob)" -ForegroundColor Gray
Write-Host "  Web Score: $($result5.web_score)`n" -ForegroundColor Gray

# Summary
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " Test Summary" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

$tests = @(
    @{Name="Reuters (Real)"; Expected="REAL"; Actual=$result1.label},
    @{Name="Clickbait (Fake)"; Expected="FAKE"; Actual=$result2.label},
    @{Name="BBC (Real)"; Expected="REAL"; Actual=$result3.label},
    @{Name="Conspiracy (Fake)"; Expected="FAKE"; Actual=$result4.label},
    @{Name="TechCrunch (Real)"; Expected="REAL"; Actual=$result5.label}
)

$correct = 0
foreach ($test in $tests) {
    $match = $test.Expected -eq $test.Actual
    if ($match) { $correct++ }
    $symbol = if ($match) {"[PASS]"} else {"[FAIL]"}
    $color = if ($match) {"Green"} else {"Red"}
    Write-Host "  $symbol $($test.Name): Expected $($test.Expected), Got $($test.Actual)" -ForegroundColor $color
}

Write-Host "`n  Accuracy: $correct / $($tests.Count) ($([math]::Round($correct/$tests.Count*100, 1))%)" -ForegroundColor $(if ($correct -eq $tests.Count) {"Green"} else {"Yellow"})
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host " Testing Complete!" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan
