"""Collect one terminal DFL diagnostic, preserving failed native observations."""
import datetime
import json
from pathlib import Path
import subprocess

root = Path('/home/nvidia/rampq/gate05_dfl_context_20261020/gate05_dfl_context_probe_20261020')
read = lambda p: json.loads(p.read_text())
monitor, guard, launch = [read(root / name) for name in ('monitor/status.json', 'supplemental_group_guard.json', 'launch.json')]
assert monitor['status'] != 'RUNNING'
assert guard['status'] == 'COMPLETE' and guard['owned_group_fully_empty'] and not guard['final_group_members']
pids = [launch['monitor_pid'], launch['private_group'], launch['guard_pid']]
if (root / 'latest_probe.json').exists():
    run = Path(read(root / 'latest_probe.json')['run'])
    for name in ('code_clip_none', 'code_clip_obey'):
        path = run / name / 'child_cleanup.json'
        if path.exists():
            child = read(path)
            assert child['direct_python_api_child_terminated'] and isinstance(child['returncode'], int)
            pids.append(child['pid'])
checks = {}
for pid in pids:
    p = Path('/proc') / str(pid)
    exists, state = p.exists(), None
    if exists:
        raw = (p / 'stat').read_text()
        state = raw[raw.rfind(')') + 2:].split()[0]
    checks[str(pid)] = dict(exists=exists, state=state, live_non_zombie=exists and state != 'Z')
assert not any(p['live_non_zombie'] for p in checks.values()), checks
with (root / 'all_child_exit_verification.json').open('x') as stream:
    json.dump(dict(board_time=datetime.datetime.now().astimezone().isoformat(),
                   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   known_processes=checks, scope='Fresh root check of recorded DFL native task PIDs before exact collection'), stream, indent=2)
archive = Path('/home/nvidia/rampq/gate05_dfl_context_evidence_20261020.zip')
result = subprocess.run(['python3', str(root / 'jetson_collect_isolated_numeric_probe.py'), '--root', str(root), '--archive', str(archive)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
assert result.returncode == 0, result.stdout + result.stderr
print(result.stdout, end='')
