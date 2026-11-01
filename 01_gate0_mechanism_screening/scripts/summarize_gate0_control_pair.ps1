param(
    [ValidatePattern("^G[1-8]\+G[1-8]$")]
    [Parameter(Mandatory = $true)]
    [string]$Pair
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Gate0Config = Join-Path $ProjectRoot "config\gate0.local.json"
if (-not (Test-Path -LiteralPath $Gate0Config)) { $Gate0Config = Join-Path $ProjectRoot "config\gate0.json" }
$PythonExe = if ($env:RAMPQ_PYTHON) { $env:RAMPQ_PYTHON } else { "python" }

& $PythonExe (Join-Path $ProjectRoot "src\summarize_gate0_control_pair.py") `
    --project-root $ProjectRoot `
    --pair $Pair

exit $LASTEXITCODE
