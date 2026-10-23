$ErrorActionPreference = 'Stop'
$taskWorkspace = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$taskPython = Join-Path (Split-Path -Parent $taskWorkspace) '05_水下小目标检测_工程\检测工程\.venv\Scripts\python.exe'
$taskRunner = Join-Path $taskWorkspace 'scripts\run_gate1_float_baseline.py'
$taskContract = Join-Path $PSScriptRoot 'training_contract_v2'
$taskOutput = Join-Path $PSScriptRoot 'float_training_v1'
$taskLaunchDirectory = Join-Path $PSScriptRoot 'float_training_launch_v1'
$taskDry = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'training_dry_validation_v2\attempt_receipt.json') -Raw | ConvertFrom-Json
$taskAudit = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'split_plan_v1_independent_audit\verification.json') -Raw | ConvertFrom-Json
$taskFrozen = Get-Content -LiteralPath (Join-Path $taskContract 'contract.json') -Raw | ConvertFrom-Json
if ($taskDry.status -ne 'PASS_DRY_VALIDATION_NO_COPY_NO_TRAINING' -or $taskAudit.status -ne 'PASS_INDEPENDENT_SAVED_SPLIT_PLAN_VERIFICATION') { throw 'Required pretraining validation has not passed.' }
if ((Get-FileHash -LiteralPath $taskRunner -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskFrozen.runner_sha256) { throw 'Runner changed after freeze.' }
if ((Get-FileHash -LiteralPath (Join-Path $taskContract 'contract.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskDry.contract_sha256) { throw 'Dry validation refers to a different contract.' }
if ((Test-Path -LiteralPath $taskOutput) -or (Test-Path -LiteralPath $taskLaunchDirectory)) { throw 'Refusing to repeat a training attempt.' }
$taskOtherTraining = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_gate1_float_baseline.py* train *' })
if ($taskOtherTraining.Count -gt 0) { throw 'Another Gate1 training runner is already active.' }
New-Item -ItemType Directory -Path $taskLaunchDirectory | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$taskArguments = '-B -u "{0}" train --contract "{1}" --output "{2}"' -f $taskRunner, $taskContract, $taskOutput
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskWorkspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLaunchDirectory 'stdout.log') -RedirectStandardError (Join-Path $taskLaunchDirectory 'stderr.log') -PassThru
$taskLaunch = [ordered]@{
    schema = 'rampq.gate1.float_training_launch/v1'
    launched_at_utc = [DateTime]::UtcNow.ToString('o')
    process_id = $taskProcess.Id
    executable = $taskPython
    runner = $taskRunner
    runner_sha256 = $taskFrozen.runner_sha256
    contract = $taskContract
    contract_sha256 = $taskDry.contract_sha256
    output = $taskOutput
    external_restarts = 0
    window_style = 'Hidden'
    selection_detector_inference = $false
    holdout_accessed = $false
}
$taskLaunch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLaunchDirectory 'launch.json') -Encoding UTF8
$taskLaunch | ConvertTo-Json -Depth 5
