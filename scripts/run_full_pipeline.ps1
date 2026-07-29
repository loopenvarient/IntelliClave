param(
    [string]$EnvName = "IntelliClave",
    [int]$Rounds = 5,
    [int]$Clients = 3,
    [double]$Epsilon = 10.0,
    [double]$DirichletAlpha = 0.3,
    [switch]$RepartitionData,
    [switch]$SkipDp,
    [switch]$SkipExperiments,
    [switch]$SkipAttacks,
    [string]$ResultsDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $ResultsDir) {
    $ResultsDir = Join-Path $Root "results\pipeline_runs\run_$Timestamp"
}
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$Script:Results = @()

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
        throw "$Name failed with exit code $exitCode. Log: $logPath"
    }
}

function Run-PowerShellStep {
    param(
        [string]$Name,
        [string]$ScriptPath
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    $started = Get-Date
    $stepDir = Join-Path $ResultsDir (Get-SafeName $Name)
    New-Item -ItemType Directory -Force -Path $stepDir | Out-Null
    $logPath = Join-Path $ResultsDir ("{0}.log" -f (Get-SafeName $Name))
    New-Item -ItemType File -Force -Path $logPath | Out-Null
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & powershell -ExecutionPolicy Bypass -File $ScriptPath -EnvName $EnvName -SkipSlow -ResultsDir $stepDir 2>&1 | Tee-Object -FilePath $logPath
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
        nested_results = $stepDir
        command = "powershell -ExecutionPolicy Bypass -File $ScriptPath -EnvName $EnvName -SkipSlow -ResultsDir $stepDir"
    }
    Save-Summary | Out-Null
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. Log: $logPath"
    }
}

if ($RepartitionData) {
    Run-Step "Dirichlet non-IID data partition" @(
        "data/datascripts/pipeline.py",
        "--mode", "textfiles",
        "--partition", "dirichlet",
        "--dirichlet-alpha", "$DirichletAlpha",
        "--n-clients", "$Clients"
    )
    Run-Step "Class weights" @("data/datascripts/weights.py")
}

Run-PowerShellStep "Pipeline tests" "scripts/test_pipeline.ps1"

Run-Step "Baseline FL simulation" @(
    "fl/run_fl_simulation.py",
    "--rounds", "$Rounds",
    "--clients", "$Clients",
    "--save-dir", "results/fl_rounds/baseline_sim"
)

$evalCheckpoint = "results/fl_rounds/baseline_sim/global_model_latest.pth"
$privacyJson = $null

if (-not $SkipDp) {
    Run-Step "DP FL simulation" @(
        "fl/run_fl_simulation.py",
        "--rounds", "$Rounds",
        "--clients", "$Clients",
        "--dp",
        "--epsilon", "$Epsilon",
        "--save-dir", "results/fl_rounds/dp_sim"
    )
    $evalCheckpoint = "results/fl_rounds/dp_sim/global_model_latest.pth"
    $privacyJson = "results/fl_rounds/dp_sim/fl_privacy.json"
}

Run-Step "Evaluate global model" @(
    "fl/evaluate_global_model.py",
    "--checkpoint", $evalCheckpoint
)

if (-not $SkipExperiments) {
    if ($privacyJson) {
        Run-Step "Privacy budget monitor" @(
            "privacy/run_budget_monitor.py",
            "--max-epsilon", "$Epsilon",
            "--privacy-json", $privacyJson
        )
    }
    Run-Step "Cross-validation" @("evaluation/cross_validation.py", "--folds", "5", "--epochs", "10")
    Run-Step "Epsilon sweep" @("privacy/epsilon_sweep.py", "--epsilons", "1", "2", "5", "10", "20")
    Run-Step "Generate final graph" @("evaluation/generate_graph6.py")
}

if (-not $SkipAttacks) {
    Run-Step "Model inversion attack" @("security/attacks/model_inversion.py")
    Run-Step "Membership inference attack" @("security/attacks/membership_inference.py")
    Run-Step "Gradient poisoning attack" @("security/attacks/gradient_poisoning.py", "--fl-rounds", "$Rounds")
    Run-Step "Summarize attack results" @("security/attacks/summarize_attack_results.py")
}

Write-Host ""
Write-Host "Full IntelliClave pipeline completed." -ForegroundColor Green
Write-Host "Results saved to $ResultsDir" -ForegroundColor Green
