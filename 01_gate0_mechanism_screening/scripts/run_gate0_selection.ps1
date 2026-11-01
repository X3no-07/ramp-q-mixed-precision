param(
    [ValidatePattern("^(q_only|r_only):G[1-8](\+G[1-8])*$")]
    [string]$Configuration = "q_only:G1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SelectionImageList = Join-Path $ProjectRoot "splits\gate0_selection_images.local.txt"
if (-not (Test-Path -LiteralPath $SelectionImageList)) { $SelectionImageList = Join-Path $ProjectRoot "splits\gate0_selection_images.txt" }
$Gate0Config = Join-Path $ProjectRoot "config\gate0.local.json"
if (-not (Test-Path -LiteralPath $Gate0Config)) { $Gate0Config = Join-Path $ProjectRoot "config\gate0.json" }
$PythonExe = if ($env:RAMPQ_PYTHON) { $env:RAMPQ_PYTHON } else { "python" }

$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"

& $PythonExe (Join-Path $ProjectRoot "src\run_gate0_selection.py") `
    --config $Gate0Config `
    --project-root $ProjectRoot `
    --configuration $Configuration `
    --image-list $SelectionImageList `
    --baseline (Join-Path $ProjectRoot "artifacts\predictions\screen.csv") `
    --calibration (Join-Path $ProjectRoot "artifacts\calibration\screen.json") `
    --split-manifest (Join-Path $ProjectRoot "artifacts\manifests\gate0_split.json")

exit $LASTEXITCODE
