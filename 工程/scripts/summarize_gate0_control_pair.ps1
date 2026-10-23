param(
    [ValidatePattern("^G[1-8]\+G[1-8]$")]
    [Parameter(Mandatory = $true)]
    [string]$Pair
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $PythonExe (Join-Path $ProjectRoot "src\summarize_gate0_control_pair.py") `
    --project-root $ProjectRoot `
    --pair $Pair

exit $LASTEXITCODE
