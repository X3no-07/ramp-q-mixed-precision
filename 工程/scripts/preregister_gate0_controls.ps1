$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $PythonExe (Join-Path $ProjectRoot "src\preregister_gate0_controls.py") `
    --project-root $ProjectRoot

exit $LASTEXITCODE
