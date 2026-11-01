$ErrorActionPreference = 'Stop'
$taskWorkspace = Split-Path -Parent $PSScriptRoot
$taskBase = Join-Path $taskWorkspace 'artifacts\gate1'
$taskAttempt = Join-Path $taskBase '20261022_split_refinement\float_training_v2'
$taskContract = Join-Path $taskBase '20261022_split_refinement\training_contract_v3'
$taskReceiptPath = Join-Path $taskAttempt 'attempt_receipt.json'
$taskReceipt = Get-Content -LiteralPath $taskReceiptPath -Raw | ConvertFrom-Json
if ($taskReceipt.status -ne 'COMPLETE_FLOAT_TRAINING_REQUIRES_SCIENTIFIC_REVIEW' -or @($taskReceipt.epoch_history).Count -ne 30) {
    throw 'The frozen 30-epoch training is not complete; no model check has been started.'
}
$taskPython = if ($env:RAMPQ_PYTHON) { $env:RAMPQ_PYTHON } else { "python" }
$taskOutput = Join-Path $taskBase '20261022_posttraining_checks_v1'
if (Test-Path -LiteralPath $taskOutput) { throw 'Refusing an existing post-training output; preserve previous attempts.' }
$taskAuditor = Join-Path $PSScriptRoot 'audit_gate1_completed_training.py'
$taskQualifier = Join-Path $PSScriptRoot 'qualify_gate1_float_baseline.py'
$taskOutputAuditor = Join-Path $PSScriptRoot 'audit_gate1_float_qualification.py'
foreach ($taskScript in @($taskAuditor, $taskQualifier, $taskOutputAuditor)) {
    if (-not (Test-Path -LiteralPath $taskScript -PathType Leaf)) { throw "Missing prepared tool: $taskScript" }
}
$taskQualificationPreparation = Get-Content -LiteralPath (Join-Path $taskBase '20261022_float_qualification_preparation\preparation_receipt.json') -Raw | ConvertFrom-Json
$taskAuditPreparation = Get-Content -LiteralPath (Join-Path $taskBase '20261022_training_completion_audit_preparation\preparation.json') -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $taskQualifier -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskQualificationPreparation.runner.sha256) { throw 'Qualification runner changed after preparation.' }
if ((Get-FileHash -LiteralPath $taskAuditor -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskAuditPreparation.script_sha256) { throw 'Training auditor changed after preparation.' }
New-Item -ItemType Directory -Path $taskOutput | Out-Null
Set-Location -LiteralPath $taskWorkspace
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$taskAuditOutput = Join-Path $taskOutput '01_training_audit'
$taskQualificationOutput = Join-Path $taskOutput '02_float_qualification'
$taskOutputAuditOutput = Join-Path $taskOutput '03_qualification_output_audit'
& $taskPython -B -u $taskAuditor --attempt $taskAttempt --contract $taskContract --output $taskAuditOutput 2>&1 | Tee-Object -FilePath (Join-Path $taskOutput '01_training_audit_console.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -B -u $taskQualifier --final-receipt $taskReceiptPath --contract-dir $taskContract --output $taskQualificationOutput 2>&1 | Tee-Object -FilePath (Join-Path $taskOutput '02_float_qualification_console.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -B -u $taskOutputAuditor --qualification $taskQualificationOutput --output $taskOutputAuditOutput 2>&1 | Tee-Object -FilePath (Join-Path $taskOutput '03_qualification_output_audit_console.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
