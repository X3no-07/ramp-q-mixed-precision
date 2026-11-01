param(
    [ValidateSet("inspect", "smoke", "screen")]
    [string]$Phase = "smoke"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Gate0Config = Join-Path $ProjectRoot "config\gate0.local.json"
if (-not (Test-Path -LiteralPath $Gate0Config)) { $Gate0Config = Join-Path $ProjectRoot "config\gate0.json" }
$PythonExe = if ($env:RAMPQ_PYTHON) { $env:RAMPQ_PYTHON } else { "python" }

$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"

New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs") | Out-Null

& $PythonExe (Join-Path $ProjectRoot "src\gate0.py") `
    $Phase `
    --config $Gate0Config `
    --project-root $ProjectRoot

exit $LASTEXITCODE
