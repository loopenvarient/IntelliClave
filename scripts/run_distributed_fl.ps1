param(
    [string]$EnvName = "IntelliClave",
    [int]$Rounds = 5,
    [int]$Clients = 3,
    [double]$Epsilon = 10.0,
    [switch]$Dp,
    [switch]$Crypto,
    [switch]$Attest,
    [string]$ServerAddress = "127.0.0.1:8080"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function New-PipelineWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    $fullCommand = "cd /d `"$Root`"; title $Title; $Command; pause"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $fullCommand
}

$serverFlags = @("fl/run_server.py", "--rounds", "$Rounds", "--min-clients", "$Clients")
if ($Crypto) { $serverFlags += "--crypto" }
if ($Attest) { $serverFlags += "--attest" }
$serverCommand = "conda run -n $EnvName python " + ($serverFlags -join " ")

New-PipelineWindow -Title "IntelliClave FL Server" -Command $serverCommand
Start-Sleep -Seconds 8

for ($clientId = 1; $clientId -le $Clients; $clientId++) {
    $clientFlags = @(
        "fl/run_client.py",
        "--id", "$clientId",
        "--server", $ServerAddress,
        "--rounds", "$Rounds"
    )
    if ($Dp) { $clientFlags += @("--dp", "--epsilon", "$Epsilon") }
    if ($Crypto) { $clientFlags += "--crypto" }
    if ($Attest) { $clientFlags += "--attest" }

    $clientCommand = "conda run -n $EnvName python " + ($clientFlags -join " ")
    New-PipelineWindow -Title "IntelliClave Client $clientId" -Command $clientCommand
}

Write-Host "Started distributed FL in separate terminals." -ForegroundColor Green
Write-Host "After training finishes, run: conda run -n $EnvName python fl/evaluate_global_model.py"
