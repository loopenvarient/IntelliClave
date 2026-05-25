param(
    [string]$PythonExe = "C:\Users\DELL\miniconda3\envs\IntelliClave\python.exe",
    [string]$BugsRoot = "C:\Users\DELL\OneDrive\Documents\IntelliClave\IntelliClave",
    [string]$MeenRoot = "C:\Users\DELL\OneDrive\Documents\IntelliClave-meen",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-CompareRun {
    param(
        [string]$Root,
        [string]$Label,
        [string]$Checkpoint
    )

    $compareDir = Join-Path $Root "results\compare\$Label"
    New-Item -ItemType Directory -Force -Path $compareDir | Out-Null

    $evalOut = Join-Path $compareDir "eval.json"
    $invOut = Join-Path $compareDir "model_inversion.json"

    $evalArgs = @(
        "fl\evaluate_global_model.py",
        "--checkpoint", $Checkpoint,
        "--csv", "data\processed\client1.csv",
        "--output", $evalOut
    )

    $invArgs = @(
        "security\attacks\model_inversion.py",
        "--model-path", $Checkpoint,
        "--out", $invOut,
        "--steps", "200"
    )

    if ($DryRun) {
        Write-Host "[$Label] $PythonExe $($evalArgs -join ' ')"
        Write-Host "[$Label] $PythonExe $($invArgs -join ' ')"
        return
    }

    Push-Location $Root
    try {
        & $PythonExe @evalArgs
        & $PythonExe @invArgs
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$bugsCheckpoint = Join-Path $BugsRoot "results\fl_rounds\run_dp_hardened\global_model_latest.pth"
if (-not (Test-Path $bugsCheckpoint)) {
    throw "Bugs checkpoint not found: $bugsCheckpoint"
}

$meenCheckpoint = Join-Path $MeenRoot "results\fl_rounds\global_model_latest.pth"
if (-not (Test-Path $meenCheckpoint)) {
    throw "Meen checkpoint not found: $meenCheckpoint"
}

Invoke-CompareRun -Root $BugsRoot -Label "bugs" -Checkpoint $bugsCheckpoint
Invoke-CompareRun -Root $MeenRoot -Label "meen" -Checkpoint $meenCheckpoint

if ($DryRun) {
    exit 0
}

Write-Host ""
Write-Host "Summary command:"
Write-Host "$PythonExe security\attacks\summarize_attack_results.py --files `"$BugsRoot\results\compare\bugs\model_inversion.json`" `"$MeenRoot\results\compare\meen\model_inversion.json`""