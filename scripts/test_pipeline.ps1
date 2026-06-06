param(
    [string]$EnvName = "IntelliClave",
    [switch]$SkipSlow,
    [string]$ResultsDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $ResultsDir) {
    $ResultsDir = Join-Path $Root "results\pipeline_tests\run_$Timestamp"
}
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$Script:Results = @()
$Script:FailedSteps = 0

function Get-SafeName {
    param([string]$Name)
    return ($Name.ToLowerInvariant() -replace "[^a-z0-9]+", "_").Trim("_")
}

function Save-Summary {
    $summaryPath = Join-Path $ResultsDir "summary.json"
    $Script:Results | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $summaryPath
    return $summaryPath
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    $started = Get-Date
    $logPath = Join-Path $ResultsDir ("{0}.log" -f (Get-SafeName $Name))
    New-Item -ItemType File -Force -Path $logPath | Out-Null
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & conda run -n $EnvName python @CommandArgs 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    } catch {
        $_ | Out-String | Tee-Object -FilePath $logPath -Append
        $exitCode = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } else { 1 }
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    $finished = Get-Date
    $Script:Results += [ordered]@{
        name = $Name
        status = $(if ($exitCode -eq 0) { "passed" } else { "failed" })
        exit_code = $exitCode
        started_at = $started.ToString("o")
        finished_at = $finished.ToString("o")
        duration_seconds = [Math]::Round(($finished - $started).TotalSeconds, 2)
        log = $logPath
        command = "conda run -n $EnvName python $($CommandArgs -join ' ')"
    }
    Save-Summary | Out-Null
    if ($exitCode -ne 0) {
        $Script:FailedSteps += 1
        Write-Host "$Name failed with exit code $exitCode. Log: $logPath" -ForegroundColor Red
    }
}

Run-Step "Data/schema check" @("data/datascripts/check_data.py")
Run-Step "Crypto tests" @("crypto/certs/test_crypto.py")
Run-Step "Attestation integration" @("tee/attestation/attestation_integration.py")
Run-Step "Sealed storage" @("tee/sealed_storage/sealed_storage.py")
Run-Step "Dashboard backend E2E" @("dashboard/backend/test_e2e.py")
Run-Step "Client import smoke test" @("-c", "import sys; sys.path.insert(0, 'fl'); import data_utils, fl_client; print('classes=', data_utils.infer_default_num_classes()); print('fl_client import ok')")

if (-not $SkipSlow) {
    Run-Step "Opacus smoke test" @("privacy/opacus_smoke_test.py")
    Run-Step "Single-client local training smoke test" @("fl/train_local.py", "--csv", "data/processed/client1.csv", "--epochs", "1")
}

Write-Host ""
if ($Script:FailedSteps -eq 0) {
    Write-Host "All requested pipeline tests completed." -ForegroundColor Green
} else {
    Write-Host "$Script:FailedSteps pipeline test step(s) failed." -ForegroundColor Red
}
Write-Host "Results saved to $ResultsDir" -ForegroundColor Green
if ($Script:FailedSteps -gt 0) {
    exit 1
}
