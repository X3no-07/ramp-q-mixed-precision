$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $PythonExe (Join-Path $ProjectRoot "src\create_gate0_split.py") `
    --config (Join-Path $ProjectRoot "config\gate0.json") `
    --predictions (Join-Path $ProjectRoot "artifacts\predictions\screen.csv") `
    --project-root $ProjectRoot `
    --selection-ratio 0.70

exit $LASTEXITCODE
