# Run this once in a visible Windows PowerShell terminal.
# The SSH password is handled by ssh.exe, never read or stored by this script.
$ErrorActionPreference = 'Stop'
$rampqKeyPath = 'C:\Users\lenovo\.ssh\codex_jetson_01a143ab_20261016'
$rampqSsh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$rampqUsbAddress = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -eq '192.168.55.100' })
if ($rampqUsbAddress.Count -eq 0) {
    throw 'Jetson USB network 192.168.55.100 is missing. Power on the board and connect its USB data cable to this computer first.'
}
$rampqPublicKey = (Get-Content -LiteralPath ($rampqKeyPath + '.pub') -Raw).Trim()
if ($rampqPublicKey -notmatch '^ssh-ed25519 [A-Za-z0-9+/=]+ codex-jetson-01a143ab-20261016$') {
    throw 'Unexpected task public key. Nothing has been changed on the Jetson.'
}
$rampqPublicB64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($rampqPublicKey))
$rampqRemotePython = @'
import base64
import datetime
import os
import shutil
import tempfile
from pathlib import Path

public_key = base64.b64decode('__PUBLIC_KEY_B64__').decode('ascii')
ssh_dir = Path.home() / '.ssh'
ssh_dir.mkdir(mode=0o700, exist_ok=True)
os.chmod(str(ssh_dir), 0o700)
authorized = ssh_dir / 'authorized_keys'
if authorized.is_symlink():
    raise RuntimeError('authorized_keys is a symlink; leaving it unchanged for inspection')
existing = authorized.read_bytes() if authorized.exists() else b''
key_blob = public_key.split()[1]
retained = b''.join(line for line in existing.splitlines(keepends=True)
                    if key_blob not in line.decode('utf-8', errors='replace').split())
stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
if authorized.exists():
    backup = ssh_dir / ('authorized_keys.bak-codex-01a143ab-' + stamp)
    shutil.copy2(str(authorized), str(backup))
    os.chmod(str(backup), 0o600)
    print('BACKUP:', backup)
expiry = (datetime.datetime.now() + datetime.timedelta(hours=2)).strftime('%Y%m%d%H%M%S')
options = 'restrict,from="192.168.55.100",expiry-time="' + expiry + '" '
if retained and not retained.endswith(b'\n'):
    retained += b'\n'
updated = retained + (options + public_key + '\n').encode('ascii')
fd, temporary_name = tempfile.mkstemp(prefix='.authorized_keys.codex-', dir=str(ssh_dir))
try:
    with os.fdopen(fd, 'wb') as handle:
        handle.write(updated)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary_name, 0o600)
    os.replace(temporary_name, str(authorized))
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
print('KEY_EXPIRES_AT_BOARD_LOCAL_TIME:', expiry)
print('TEMPORARY_KEY_INSTALLED')
'@
$rampqRemotePython = $rampqRemotePython.Replace('__PUBLIC_KEY_B64__', $rampqPublicB64)
$rampqPayload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($rampqRemotePython))
$rampqRemoteCommand = 'printf %s ' + $rampqPayload + ' | base64 -d | python3 -'

Write-Host 'Enter the Jetson password only at the SSH prompt below.'
Write-Host 'This task key is limited to this USB-network IP, expires in 2 hours, and disables forwarding and PTY.'
& $rampqSsh -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o PubkeyAuthentication=no -o PreferredAuthentications=password nvidia@192.168.55.1 $rampqRemoteCommand
if ($LASTEXITCODE -ne 0) { throw 'Temporary SSH access setup failed. Keep the output for diagnosis.' }

& $rampqSsh -i $rampqKeyPath -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 nvidia@192.168.55.1 'uname -m'
if ($LASTEXITCODE -ne 0) { throw 'The task key was installed but automatic SSH verification failed.' }
Write-Host 'CODEX_ACCESS_READY - return to the chat. No password needs to be sent there.'
