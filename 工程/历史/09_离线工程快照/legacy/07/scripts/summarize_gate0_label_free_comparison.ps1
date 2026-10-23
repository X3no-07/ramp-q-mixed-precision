$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $PythonExe (Join-Path $ProjectRoot "src\summarize_gate0_label_free_comparison.py") `
    --project-root $ProjectRoot

exit $LASTEXITCODE
