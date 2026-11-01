$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Gate0Config = Join-Path $ProjectRoot "config\gate0.local.json"
if (-not (Test-Path -LiteralPath $Gate0Config)) { $Gate0Config = Join-Path $ProjectRoot "config\gate0.json" }
$PythonExe = if ($env:RAMPQ_PYTHON) { $env:RAMPQ_PYTHON } else { "python" }

& $PythonExe (Join-Path $ProjectRoot "src\create_gate0_split.py") `
    --config $Gate0Config `
    --predictions (Join-Path $ProjectRoot "artifacts\predictions\screen.csv") `
    --project-root $ProjectRoot `
    --selection-ratio 0.70

exit $LASTEXITCODE
