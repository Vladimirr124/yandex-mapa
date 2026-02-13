# Запуск сервера скриншотов и ngrok с CORS, затем открытие карты с правильной ссылкой
# Запуск: из корня репо: .\start_screenshot_and_ngrok.ps1
# Или: powershell -ExecutionPolicy Bypass -File .\start_screenshot_and_ngrok.ps1

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot

# 1) Освободить порт 8766
$listener = Get-NetTCPConnection -LocalPort 8766 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    Write-Host "[~] Останавливаю процесс на порту 8766 (PID $($listener.OwningProcess))..."
    Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# 2) Остановить ngrok
Get-Process -Name ngrok -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "[~] ngrok остановлен (если был запущен)."
Start-Sleep -Seconds 1

# 3) Запустить сервер скриншотов в новом окне
$serverScript = Join-Path $repoRoot "gpu+out\serve_last_screenshot.py"
Start-Process python -ArgumentList $serverScript -WorkingDirectory $repoRoot -WindowStyle Normal
Write-Host "[~] Сервер скриншотов запущен (окно Python)."
Start-Sleep -Seconds 2

# 4) Запустить ngrok с политикой CORS в новом окне
$policyFile = Join-Path $repoRoot "gpu+out\ngrok-cors-policy.yml"
Start-Process ngrok -ArgumentList "http","8766","--traffic-policy-file=`"$policyFile`"" -WorkingDirectory $repoRoot -WindowStyle Normal
Write-Host "[~] ngrok zapushen."
Start-Sleep -Seconds 3

# 5) Poluchit publichnyj URL ngrok (neskolko popytok)
$ngrokUrl = $null
foreach ($i in 1..5) {
    try {
        $tunnels = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -ErrorAction Stop
        if ($tunnels.tunnels -and $tunnels.tunnels.Count -gt 0) {
            $ngrokUrl = $tunnels.tunnels[0].public_url
            break
        }
    } catch { }
    Start-Sleep -Seconds 2
}
if (-not $ngrokUrl) {
    Write-Host "[!] URL ngrok ne poluchen. Otkrojte https://127.0.0.1:4040 v brauzere i skopirujte Forwarding URL."
}

$mapUrl = "https://vladimirr124.github.io/yandex-mapa/?screenshot_url=$ngrokUrl"
Write-Host ""
Write-Host "=========================================="
if ($ngrokUrl) {
    Write-Host "Ссылка на карту со скриншотом:"
    Write-Host $mapUrl
    Write-Host "=========================================="
    Start-Process $mapUrl
    Write-Host "Браузер открыт. Если скриншот не появился - подождите 5-10 сек и обновите страницу (F5)."
} else {
    Write-Host "Ссылка на карту: подставьте свой ngrok URL в ссылку ниже."
    $fallbackUrl = "https://vladimirr124.github.io/yandex-mapa/?screenshot_url=https://YOUR_NGROK.ngrok-free.app"
    Write-Host $fallbackUrl
    Write-Host "=========================================="
}
