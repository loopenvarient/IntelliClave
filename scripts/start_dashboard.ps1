param(
    [string]$EnvName = "IntelliClave"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

function New-DashboardWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    $fullCommand = "cd /d `"$Root`"; title $Title; $Command; pause"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $fullCommand
}

New-DashboardWindow `
    -Title "IntelliClave Dashboard Backend" `
    -Command "cd dashboard/backend && conda run -n $EnvName uvicorn main:app --host 0.0.0.0 --port 8001 --reload"

New-DashboardWindow `
    -Title "IntelliClave Dashboard Frontend" `
    -Command "cd dashboard/frontend/intelliclave-ui && npm run dev"

Write-Host "Dashboard started in separate terminals." -ForegroundColor Green
Write-Host "Backend:  http://localhost:8001"
Write-Host "Frontend: http://localhost:5173"
