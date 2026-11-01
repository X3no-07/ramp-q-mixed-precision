$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Gate0Config = Join-Path $ProjectRoot "config\gate0.local.json"
if (-not (Test-Path -LiteralPath $Gate0Config)) { $Gate0Config = Join-Path $ProjectRoot "config\gate0.json" }
$PythonExe = if ($env:RAMPQ_PYTHON) { $env:RAMPQ_PYTHON } else { "python" }

$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"

& $PythonExe (Join-Path $ProjectRoot "src\compute_gate0_output_reconstruction.py") `
    --config $Gate0Config `
    --project-root $ProjectRoot `
    --calibration (Join-Path $ProjectRoot "artifacts\calibration\screen.json") `
    --control-preregistration (Join-Path $ProjectRoot "artifacts\tables\gate0_control_preregistration.json")

exit $LASTEXITCODE
