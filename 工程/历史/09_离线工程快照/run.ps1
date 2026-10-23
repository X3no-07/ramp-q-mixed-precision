param(
    [ValidateSet('verify','status','compare','plan')][string]$Task = 'status',
    [string]$Python = 'python',
    [string]$Output = '',
    [string]$SourceRoot = ''
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$rampqArgs = @((Join-Path $PSScriptRoot 'src\stage2.py'), $Task)
if ($Output) { $rampqArgs += @('--output', $Output) }
if ($SourceRoot) { $rampqArgs += @('--source-root', $SourceRoot) }
& $Python @rampqArgs
exit $LASTEXITCODE
