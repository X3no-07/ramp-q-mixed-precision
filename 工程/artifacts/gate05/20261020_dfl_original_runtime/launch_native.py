"""Transport-side one-shot launch, stdin on board; all inputs pinned by argv."""
import atexit
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import zipfile

archive_sha, spec_sha, manifest_sha, review_sha = sys.argv[1:]
assert all(re.fullmatch('[0-9a-f]{64}', value) for value in sys.argv[1:])
base = Path('/home/nvidia/rampq')
staging = base / 'gate05_dfl_original_20261020'
root = staging / 'gate05_dfl_original_probe_20261020'
archive = base / 'gate05_dfl_original_probe_20261020.zip'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
assert not root.exists() and staging.is_dir() and sha(archive) == archive_sha
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert name.startswith(root.name + '/') and root in (staging / name).resolve().parents
    z.extractall(staging)
assert sha(root / 'spec.json') == spec_sha and sha(root / 'manifest.json') == manifest_sha
assert sha(root / 'protocol.json') == '81200cd4900a35f0533c45c7542f6dacc353203e8338a09b47cd24465cc0f6ed'
assert sha(staging / 'pre_native_review.json') == review_sha
review = read(staging / 'pre_native_review.json')
assert not review.get('launch_blockers')
helpers = {'jetson_monitored_build.py': '71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1',
           'jetson_owned_group_guard.py': '4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4',
           'jetson_collect_isolated_numeric_probe.py': 'af9b7f823e4f889813fb88b6e8a529a6b1a6423b0c7ead0d5704513b517775eb'}
for name, expected in helpers.items():
    assert sha(staging / name) == expected
    shutil.copy2(staging / name, root / name)
shutil.copy2(staging / 'pre_native_review.json', root / 'pre_native_review.json')
check = subprocess.run([sys.executable, str(root / 'jetson_dfl_original_probe.py'), '--root', str(root), '--check-input-only'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
assert check.returncode == 0, check.stdout + check.stderr
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
tasks = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) == os.getpid():
        continue
    try:
        cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        if any(s in cmd for s in ('jetson_build_', 'jetson_dfl_original_probe.py', 'jetson_dfl_context_probe.py', 'jetson_twoconv_context_probe.py', 'jetson_prefix_flag_probe.py', 'trtexec')):
            tasks.append(dict(pid=int(p.name), cmdline=cmd))
    except OSError:
        pass
assert not tasks, tasks
identity = dict(schema='rampq.gate05.dfl_original_native_launch/v1',
                board_time=datetime.datetime.now().astimezone().isoformat(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                root=str(root), archive_sha256=archive_sha, spec_sha256=spec_sha, manifest_sha256=manifest_sha,
                pre_native_review_sha256=review_sha, helper_identities=helpers, pure_stdlib_input_check_stdout=check.stdout,
                case_order=['original_none', 'original_obey'], max_seconds=900, per_case_seconds=300)
with (root / 'native_input_receipt.json').open('x') as f:
    json.dump(identity, f, indent=2)
console = (root / 'monitor_console.log').open('wb', buffering=0)
monitor, guard, owned, guard_module = None, None, None, None
launch_ok = False


def cleanup_unconfirmed_launch():
    if launch_ok:
        return
    failure = dict(status='LAUNCH_UNCONFIRMED_CLEANUP', termination_actions=[])
    failure_path = root / 'launch_failure.json'

    def save_failure():
        failure_path.write_text(json.dumps(failure, indent=2) + '\n')

    try:
        if owned is not None:
            # The pinned guard module authenticates boot, leader start ticks,
            # source argv and the private PGID/SID before every signal.
            remaining = guard_module.terminate_owned(owned, failure, save_failure)
            failure['owned_group_final_members'] = remaining
            failure['owned_group_fully_empty'] = not remaining
        if monitor is not None and monitor.poll() is None:
            monitor.send_signal(signal.SIGINT)
            try:
                monitor.wait(timeout=40)
            except subprocess.TimeoutExpired:
                failure['monitor_cleanup_timed_out'] = True
        if guard is not None and guard.poll() is None:
            try:
                guard.wait(timeout=8)
            except subprocess.TimeoutExpired:
                guard.send_signal(signal.SIGINT)
                guard.wait(timeout=8)
        failure['monitor_returncode'] = monitor.poll() if monitor is not None else None
        failure['guard_returncode'] = guard.poll() if guard is not None else None
    except BaseException as error:
        failure['cleanup_error'] = repr(error)
    finally:
        save_failure()


atexit.register(cleanup_unconfirmed_launch)
monitor = subprocess.Popen([sys.executable, '-u', str(root / 'jetson_monitored_build.py'), '--root', str(root),
                            '--run', str(root / 'monitor'), '--max-seconds', '900',
                            '--build-script', 'jetson_dfl_original_probe.py'], cwd=str(root),
                           stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
status_path = root / 'monitor/status.json'
deadline = time.monotonic() + 12
while not status_path.exists() and time.monotonic() < deadline and monitor.poll() is None:
    time.sleep(0.25)
assert status_path.exists(), 'Monitor failed before status: ' + (root / 'monitor_console.log').read_text(errors='replace')
status = read(status_path)
assert status['monitor_pid'] == monitor.pid and status['root'] == str(root) and status['max_seconds'] == 900
sys.path.insert(0, str(root))
import jetson_owned_group_guard as guard_module
owned = guard_module.OwnedGroup(status['build_pid'], identity['boot_id'], root / 'jetson_dfl_original_probe.py')
guard_console = (root / 'supplemental_guard_console.log').open('wb', buffering=0)
guard = subprocess.Popen([sys.executable, '-u', str(root / 'jetson_owned_group_guard.py'), '--root', str(root),
                          '--expected-monitor-pid', str(monitor.pid)], cwd=str(root), stdin=subprocess.DEVNULL,
                         stdout=guard_console, stderr=subprocess.STDOUT, start_new_session=True)
guard_path = root / 'supplemental_group_guard.json'
deadline = time.monotonic() + 8
while not guard_path.exists() and time.monotonic() < deadline and guard.poll() is None:
    time.sleep(0.25)
assert guard_path.exists(), 'Guard startup failed; inspect private owned monitor immediately'
guard_status = read(guard_path)
for field, expected in [('root', str(root)), ('boot_id', identity['boot_id']), ('guard_pid', guard.pid), ('owned_pgid', status['build_pid'])]:
    assert guard_status[field] == expected, 'Guard identity mismatch: ' + field
for field in ('root', 'monitor_pid', 'build_pid', 'max_seconds', 'build_script'):
    assert guard_status['monitor_identity'][field] == status[field], 'Guard monitor mismatch: ' + field
assert guard_status['status'] == 'WATCHING' or (guard_status['status'] == 'COMPLETE' and guard_status['owned_group_fully_empty'])
identity.update(status='DFL_ORIGINAL_CASES_LAUNCHED_UNDER_MONITOR', monitor_pid=monitor.pid,
                private_group=status['build_pid'], guard_pid=guard.pid, guard_initial_status=guard_status['status'])
with (root / 'launch.json').open('x') as f:
    json.dump(identity, f, indent=2)
launch_ok = True
print(json.dumps(identity), flush=True)

