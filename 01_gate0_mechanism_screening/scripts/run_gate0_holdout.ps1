param(
    [ValidateSet("q_only:G1+G5", "r_only:G1+G5", "q_only:G1", "r_only:G1")]
    [string]$Configuration = "q_only:G1+G5"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$HoldoutImageList = Join-Path $ProjectRoot "splits\gate0_holdout_images.local.txt"
if (-not (Test-Path -LiteralPath $HoldoutImageList)) { $HoldoutImageList = Join-Path $ProjectRoot "splits\gate0_holdout_images.txt" }
$Gate0Config = Join-Path $ProjectRoot "config\gate0.local.json"
if (-not (Test-Path -LiteralPath $Gate0Config)) { $Gate0Config = Join-Path $ProjectRoot "config\gate0.json" }
$PythonExe = if ($env:RAMPQ_PYTHON) { $env:RAMPQ_PYTHON } else { "python" }

$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"

& $PythonExe (Join-Path $ProjectRoot "src\run_gate0_selection.py") `
    --config $Gate0Config `
    --project-root $ProjectRoot `
    --configuration $Configuration `
    --subset holdout `
    --image-list $HoldoutImageList `
    --baseline (Join-Path $ProjectRoot "artifacts\predictions\screen.csv") `
    --calibration (Join-Path $ProjectRoot "artifacts\calibration\screen.json") `
    --split-manifest (Join-Path $ProjectRoot "artifacts\manifests\gate0_split.json") `
    --candidate-lock (Join-Path $ProjectRoot "artifacts\tables\selection_candidate_lock.json")

exit $LASTEXITCODE
