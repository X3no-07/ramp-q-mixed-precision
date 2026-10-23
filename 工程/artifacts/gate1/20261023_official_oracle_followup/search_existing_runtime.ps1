$ErrorActionPreference = 'Stop'
$output = Join-Path $PSScriptRoot 'runtime_search_receipt.json'
if (Test-Path -LiteralPath $output) { throw 'Refusing to overwrite existing search receipt' }
$commands = foreach ($name in @('matlab', 'octave', 'octave-cli')) {
    $items = @(Get-Command -Name $name -ErrorAction SilentlyContinue)
    [ordered]@{ name = $name; found = $items.Count -gt 0; paths = @($items | ForEach-Object Source) }
}
$registryRoots = @(
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths',
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths',
    'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths',
    'HKCU:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths'
)
$appPaths = foreach ($registryRoot in $registryRoots) {
    $exists = Test-Path -LiteralPath $registryRoot
    $matches = @()
    $scanErrors = @()
    if ($exists) {
        $matches = @(Get-ChildItem -LiteralPath $registryRoot -ErrorAction SilentlyContinue -ErrorVariable scanErrors | Where-Object PSChildName -Match 'matlab|octave' | ForEach-Object {
            $item = Get-ItemProperty -LiteralPath $_.PSPath
            [ordered]@{ key = $_.PSPath; executable = $item.'(default)'; path = $item.Path }
        })
    }
    [ordered]@{ root = $registryRoot; exists = $exists; matches = $matches; access_errors = @($scanErrors | ForEach-Object { $_.Exception.Message }) }
}
$uninstallRoots = @(
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall',
    'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
    'HKCU:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall'
)
$uninstall = foreach ($registryRoot in $uninstallRoots) {
    $exists = Test-Path -LiteralPath $registryRoot
    $matches = @()
    $scanErrors = @()
    if ($exists) {
        $matches = @(Get-ItemProperty -Path (Join-Path $registryRoot '*') -ErrorAction SilentlyContinue -ErrorVariable scanErrors |
            Where-Object { $_.DisplayName -match 'MATLAB|Octave|MathWorks' } | ForEach-Object {
                [ordered]@{ display_name = $_.DisplayName; install_location = $_.InstallLocation; display_version = $_.DisplayVersion }
            })
    }
    [ordered]@{ root = $registryRoot; exists = $exists; matches = $matches; access_errors = @($scanErrors | ForEach-Object { $_.Exception.Message }) }
}
$vendorRegistry = foreach ($key in @('HKLM:\SOFTWARE\MathWorks', 'HKLM:\SOFTWARE\WOW6432Node\MathWorks', 'HKCU:\SOFTWARE\MathWorks')) {
    [ordered]@{ key = $key; exists = (Test-Path -LiteralPath $key) }
}
$directoryRoots = @(
    'C:\', 'C:\Program Files', 'C:\Program Files (x86)',
    'C:\Users\lenovo\AppData\Local\Programs', 'C:\Users\lenovo\scoop\apps',
    'C:\ProgramData\chocolatey\bin', 'C:\tools', 'C:\Software', 'C:\Apps'
)
$directories = foreach ($directoryRoot in $directoryRoots) {
    $exists = Test-Path -LiteralPath $directoryRoot -PathType Container
    $candidates = @()
    if ($exists) {
        $candidates = @(Get-ChildItem -LiteralPath $directoryRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object Name -Match 'MATLAB|Octave|MathWorks' | ForEach-Object FullName)
    }
    [ordered]@{ root = $directoryRoot; exists = $exists; matching_child_directories = $candidates }
}
$knownRoots = foreach ($path in @(
    'C:\Program Files\MATLAB', 'C:\Program Files\GNU Octave', 'C:\Program Files\Octave',
    'C:\Program Files (x86)\MATLAB', 'C:\Program Files (x86)\GNU Octave', 'C:\Program Files (x86)\Octave',
    'C:\MATLAB', 'C:\Octave', 'C:\GNU Octave',
    'C:\Users\lenovo\AppData\Local\Programs\MATLAB', 'C:\Users\lenovo\AppData\Local\Programs\Octave'
)) { [ordered]@{ path = $path; exists = (Test-Path -LiteralPath $path -PathType Container) } }
$nativeFound = @($commands | Where-Object found).Count -gt 0
$registryCandidate = @($appPaths | Where-Object { $_.matches.Count -gt 0 }).Count -gt 0 -or @($uninstall | Where-Object { $_.matches.Count -gt 0 }).Count -gt 0
$folderCandidate = @($directories | Where-Object { $_.matching_child_directories.Count -gt 0 }).Count -gt 0 -or @($knownRoots | Where-Object exists).Count -gt 0
$receipt = [ordered]@{
    schema = 'rampq.gate1.existing_official_runtime_search/v1'
    client_date = '2026-10-23'
    recorded_at_utc = [DateTime]::UtcNow.ToString('o')
    status = $(if ($nativeFound -or $registryCandidate -or $folderCandidate) { 'CANDIDATE_REQUIRES_FOLLOWUP' } else { 'NO_RUNTIME_DISCOVERED_IN_BOUNDED_SEARCH' })
    commands = @($commands); app_paths = @($appPaths); uninstall = @($uninstall)
    vendor_registry = @($vendorRegistry); directory_roots = @($directories); known_install_roots = @($knownRoots)
    exhaustive_search = $false
    scope_limitations = @('No exhaustive disk search', 'No secondary drive enumeration or probing', 'No broad user content scan', 'No remote or network filesystem search')
    installation_performed = $false; download_performed = $false; registry_modified = $false
    official_oracle_executed = $false; official_AP_equivalence_certified = $false
    real_data_read = $false; model_executed = $false; holdout_accessed = $false
    next_action = 'Native execution remains pending a usable existing MATLAB or Octave runtime with mean2 and JSON support. No runtime absence is claimed outside the bounded search.'
}
$receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $output -Encoding utf8NoBOM
Write-Output $receipt.status
