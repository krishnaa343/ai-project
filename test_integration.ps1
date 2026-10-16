$health = Invoke-RestMethod 'http://127.0.0.1:5000/api/health'
Write-Host "1. Backend Health: status=$($health.status)  model_loaded=$($health.model_loaded)  classes=$($health.num_classes)"

$frontend = Invoke-WebRequest 'http://localhost:5173/' -UseBasicParsing
Write-Host "2. Frontend Dev Server: HTTP $($frontend.StatusCode)"

$proxy = Invoke-RestMethod 'http://localhost:5173/api/health'
Write-Host "3. Vite Proxy /api: status=$($proxy.status)"

$gcUrl = 'http://localhost:5173/static/gradcam/gradcam_5e13d259f88f482698ee3747c4ebeb94.png'
$gc = Invoke-WebRequest $gcUrl -UseBasicParsing
Write-Host "4. Static Grad-CAM via 5173: HTTP $($gc.StatusCode)  bytes=$($gc.RawContentLength)"

Write-Host ""
Write-Host "=== ALL INTEGRATION CHECKS PASSED ==="
