param(
    [ValidatePattern("^(q_only|r_only):G[1-8](\+G[1-8])*$")]
    [string]$Configuration = "q_only:G1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$LegacySitePackages = "C:\Users\lenovo\Desktop\视觉边缘研究\05_水下小目标检测_工程\检测工程\.venv\Lib\site-packages"

$env:PYTHONPATH = $LegacySitePackages
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"

& $PythonExe (Join-Path $ProjectRoot "src\run_gate0_selection.py") `
    --config (Join-Path $ProjectRoot "config\gate0.json") `
    --project-root $ProjectRoot `
    --configuration $Configuration `
    --image-list (Join-Path $ProjectRoot "splits\gate0_selection_images.txt") `
    --baseline (Join-Path $ProjectRoot "artifacts\predictions\screen.csv") `
    --calibration (Join-Path $ProjectRoot "artifacts\calibration\screen.json") `
    --split-manifest (Join-Path $ProjectRoot "artifacts\manifests\gate0_split.json")

exit $LASTEXITCODE
