$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$LegacySitePackages = "C:\Users\lenovo\Desktop\视觉边缘研究\05_水下小目标检测_工程\检测工程\.venv\Lib\site-packages"

$env:PYTHONPATH = $LegacySitePackages
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"

& $PythonExe (Join-Path $ProjectRoot "src\compute_gate0_output_reconstruction.py") `
    --config (Join-Path $ProjectRoot "config\gate0.json") `
    --project-root $ProjectRoot `
    --calibration (Join-Path $ProjectRoot "artifacts\calibration\screen.json") `
    --control-preregistration (Join-Path $ProjectRoot "artifacts\tables\gate0_control_preregistration.json")

exit $LASTEXITCODE
