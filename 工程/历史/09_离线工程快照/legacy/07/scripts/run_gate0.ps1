param(
    [ValidateSet("inspect", "smoke", "screen")]
    [string]$Phase = "smoke"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$LegacySitePackages = "C:\Users\lenovo\Desktop\视觉边缘研究\05_水下小目标检测_工程\检测工程\.venv\Lib\site-packages"

$env:PYTHONPATH = $LegacySitePackages
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"

New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs") | Out-Null

& $PythonExe (Join-Path $ProjectRoot "src\gate0.py") `
    $Phase `
    --config (Join-Path $ProjectRoot "config\gate0.json") `
    --project-root $ProjectRoot

exit $LASTEXITCODE
