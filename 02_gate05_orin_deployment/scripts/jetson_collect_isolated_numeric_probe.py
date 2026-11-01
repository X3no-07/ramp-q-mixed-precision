"""Collect an ended, private diagnostic directory with exact file receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--end-status-name', default='end_board_status.json')
    args = parser.parse_args()
    base = Path('/home/nvidia/rampq').resolve()
    root, destination = args.root.resolve(), args.archive.resolve()
    assert base in root.parents and destination.parent == base and root not in destination.parents
    assert not destination.exists()
    assert Path(args.end_status_name).name == args.end_status_name and args.end_status_name.endswith('.json')
    status = json.loads((root / 'monitor' / 'status.json').read_text())
    assert status['status'] != 'RUNNING'
    process_checks = {}
    for field in ('monitor_pid', 'build_pid'):
        if field in status:
            p = subprocess.run(['ps', '-o', 'stat=', '-p', str(status[field])], capture_output=True, text=True)
            assert p.returncode in (0, 1)
            assert not p.stdout.strip() or p.stdout.strip().startswith('Z'), field + ' remains live'
            process_checks[field] = {'pid': status[field], 'stat': p.stdout.strip()}
    p = subprocess.run(['pgrep', '-a', '-x', 'trtexec'], capture_output=True, text=True)
    assert p.returncode == 1, 'trtexec still running or process check failed'
    end = {'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'terminal_monitor': status, 'owned_process_checks': process_checks, 'trtexec_stdout': p.stdout}
    for name, command in (('fan', ['systemctl', 'is-active', 'nvfancontrol.service']),
                          ('jtop', ['systemctl', 'show', 'jtop.service', '-p', 'ActiveState', '-p', 'NRestarts']),
                          ('memory', ['free', '-m'])):
        p = subprocess.run(command, capture_output=True, text=True, timeout=15)
        end[name] = {'command': command, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    with (root / args.end_status_name).open('x') as handle:
        handle.write(json.dumps(end, indent=2) + '\n')
    files = {}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink(), 'No symlinks permitted'
        if path.is_file():
            assert root in path.resolve().parents
            files[path.relative_to(root).as_posix()] = path
    receipt = {'schema': 'rampq.remote_file_receipt/v1',
               'files': {name: {'bytes': path.stat().st_size, 'sha256': digest(path)} for name, path in files.items()}}
    with zipfile.ZipFile(str(destination), 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(str(path), name)
            assert path.stat().st_size == receipt['files'][name]['bytes'] and digest(path) == receipt['files'][name]['sha256'], 'File changed during collection: ' + name
        archive.writestr('remote_receipt.json', json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'path': str(destination), 'bytes': destination.stat().st_size,
                      'sha256': digest(destination), 'files': len(files)}))


if __name__ == '__main__':
    main()
