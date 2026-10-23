"""Run three unchanged prefer recovery bundles once with external 7200 s limits.

No GPU library is imported here. Original manifest/spec/builder/monitor bytes
remain frozen; the inherited spec budget is 1800 s, while the unchanged monitor
is explicitly invoked with --max-seconds 7200. This discrepancy is intentional
and recorded, not silently rewritten as a change to the scientific protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import subprocess
import sys
import time


BASE = PurePosixPath('/home/nvidia/rampq')
SCHEMA = 'rampq.gate05.extended_restore_queue_plan/v1'
ORDER = ('r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2')
GROUPS = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'}
MASKS = {'r_only_G1': GROUPS - {'G1'}, 'r_only_G1_G5': GROUPS - {'G1', 'G5'},
         'r_only_G1_G2': GROUPS - {'G1', 'G2'}}
COUNTS = {'r_only_G1': (82, 6), 'r_only_G1_G5': (74, 14), 'r_only_G1_G2': (77, 11)}
TIMEOUT_REASON = 'Build supervision time limit reached'
STOP_MARKERS = ('controller error detected', 'HC died;', 'unregister_netdevice: waiting',
                'Out of memory:', 'oom-kill:', 'Kernel panic')
SAFE_BOOT = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')
SHA = re.compile(r'[0-9a-f]{64}')
INPUTS_NAME = 'gate05_restore_extended_inputs_20261019'
RUN_NAME = 'gate05_restore_extended_queue_20261019'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def board_path(value):
    require(isinstance(value, str), 'Expected an absolute board path')
    path = PurePosixPath(value)
    require(path.is_absolute() and path.parent == BASE and '..' not in path.parts and str(path) == value,
            'Expected a canonical direct child of /home/nvidia/rampq: ' + value)
    return path


def basename(configuration):
    return 'gate05_precision_' + configuration + '_20261019_v3_extended_7200'


def verify_files(root, files):
    require(root.is_dir() and not root.is_symlink(), 'Missing or redirected frozen input directory')
    resolved = root.resolve()
    for name, info in files.items():
        require(isinstance(name, str) and Path(name).name == name and '/' not in name and '\\' not in name,
                'Frozen file must be a direct child')
        path = root / name
        require(not path.is_symlink() and path.resolve().parent == resolved, 'Frozen file escapes input root')
        require(type(info['bytes']) is int and info['bytes'] >= 0 and SHA.fullmatch(info['sha256']),
                'Invalid frozen file receipt')
        require(path.is_file() and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                'Frozen input size/hash differs: ' + str(path))


def validate_plan(plan):
    require(plan['schema'] == SCHEMA, 'Unsupported extended queue plan')
    require(plan['configuration_order'] == list(ORDER), 'Frozen recovery order differs')
    require(plan['maximum_attempts_per_configuration'] == 1, 'Only one extended attempt is permitted')
    require(plan['inherited_spec_max_seconds'] == 1800 and plan['effective_monitor_max_seconds'] == 7200,
            'Only the documented external 1800-to-7200 budget revision is permitted')
    require(plan['max_monitor_and_cleanup_seconds'] == 7380, 'Unexpected outer monitor/cleanup bound')
    require(plan['expected_boot_id_from_required_launch_argument'] is True, 'An explicit live boot ID is required')
    require(plan['scientific_criteria_changed'] is False and plan['dataset_evaluation_run'] is False,
            'Scope must remain frozen engineering diagnostics')
    require(board_path(plan['inputs_root']).name == INPUTS_NAME, 'Unexpected inputs root')
    require(board_path(plan['queue_run']).name == RUN_NAME, 'Unexpected fresh queue root')
    require(SHA.fullmatch(plan['queue_script_sha256']) and SHA.fullmatch(plan['budget_revision_sha256']),
            'Queue script and external budget revision hashes are required')
    require([row['configuration'] for row in plan['configurations']] == list(ORDER), 'Exactly three recoveries required')
    paths = [plan['inputs_root'], plan['queue_run']]
    for row in plan['configurations']:
        name = row['configuration']
        require(board_path(row['bundle_root']).name == basename(name), 'Unexpected fresh recovery root')
        require(row['precisionConstraints'] == 'prefer' and row['inherited_spec_max_seconds'] == 1800 and
                row['effective_monitor_max_seconds'] == 7200 and row['maximum_attempts'] == 1,
                'Frozen request/budget policy differs')
        require(SHA.fullmatch(row['bundle_manifest_sha256']), 'Original manifest SHA256 required')
        paths.append(row['bundle_root'])
    require(len(paths) == len(set(paths)), 'Queue, input and bundle roots must differ')


def verify_bundle(item, root):
    name = item['configuration']
    verify_files(root, item['all_original_files'])
    require(sha(root / 'manifest.json') == item['bundle_manifest_sha256'], 'Original manifest hash differs')
    manifest = read(root / 'manifest.json')
    require(manifest['schema'] == 'rampq.gate05.precision_runtime_bundle/v1' and
            manifest['configuration'] == name, 'Manifest identity differs')
    require(set(item['all_original_files']) == set(manifest['files']) | {'manifest.json', 'SHA256SUMS'},
            'Expected every original bundle file, with no replacement metadata inside the bundle')
    verify_files(root, manifest['files'])
    sums = (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    require(set(sums) == {info['sha256'] + '  ' + filename for filename, info in manifest['files'].items()},
            'Original SHA256SUMS differs from original manifest')
    spec = read(root / 'spec.json')
    require(spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1' and
            spec['configuration'] == name and spec['diagnostic_only'] is True, 'Build spec identity/scope differs')
    require(spec['precisionConstraints'] == 'prefer' and spec['bundle_revision'] == 'v2_prefer' and
            spec['precision_request_is_not_actual_adoption'] is True and
            spec['actual_precision_audit_criteria_changed'] is False, 'Original prefer precision policy differs')
    require(spec['max_build_seconds'] == 1800 and spec['workspace_mib'] == 1024 and spec['avg_timing'] == 1,
            'Original scientific/build inputs changed; override belongs only to the external launch')
    requests = spec['requests']
    require(len(requests) == len({row['onnx_node'] for row in requests}) == 88, 'Expected 88 unique Conv requests')
    require({row['group'] for row in requests} == GROUPS, 'Frozen group inventory differs')
    require(all(row['precision'] == ('INT8' if row['group'] in MASKS[name] else 'FP16') for row in requests),
            'Frozen precision mask differs')
    counts = (sum(row['precision'] == 'INT8' for row in requests), sum(row['precision'] == 'FP16' for row in requests))
    require(counts == COUNTS[name] == (spec['quantized_conv_count'], spec['fp16_conv_count']),
            'Frozen precision counts differ')
    for filename, field in [('model.onnx', 'graph_sha256'), ('graph_audit.json', 'graph_audit_sha256'),
                            ('module_map.json', 'module_map_sha256'), ('preflight.json', 'failed_preflight_sha256'),
                            ('diagnostic_scope.json', 'diagnostic_scope_sha256')]:
        require(sha(root / filename) == spec[field], 'Frozen spec input hash differs: ' + filename)
    scope = read(root / 'diagnostic_scope.json')
    require(spec['original_local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED' and
            read(root / 'preflight.json')['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED',
            'Original full-model numerical failure must remain failed')
    require(scope['status'] == 'DIAGNOSTIC_MATRIX_BUILD_AND_FIXED_IMAGE_ONLY' and
            scope['original_tolerances_changed'] is False and scope['dataset_evaluation_run'] is False and
            scope['actual_precision_audit_criteria_changed'] is False, 'Diagnostic scope differs')
    return {'configuration': name, 'bundle_root': item['bundle_root'], 'verified_at': str(root),
            'manifest_sha256': sha(root / 'manifest.json'), 'spec_sha256': sha(root / 'spec.json'),
            'graph_sha256': sha(root / 'model.onnx'), 'original_file_count': len(item['all_original_files']),
            'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200,
            'original_numerical_failure': 'FAILED_UNCHANGED', 'actual_precision_audit': 'NOT_RUN'}


def proc_info(pid):
    try:
        raw = Path('/proc/{}/stat'.format(pid)).read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'sid': int(fields[3]), 'start_ticks': int(fields[19]),
                'argv': [part.decode(errors='replace') for part in
                         Path('/proc/{}/cmdline'.format(pid)).read_bytes().split(b'\0') if part]}
    except (OSError, ValueError, IndexError):
        return None


def live(info):
    return info is not None and info['state'] not in ('Z', 'X', 'x')


def processes():
    result = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            info = proc_info(int(path.name))
            if live(info):
                result[info['pid']] = info
    return result


def same_identity(left, right):
    return live(left) and live(right) and left['pid'] == right['pid'] and left['start_ticks'] == right['start_ticks']


def track_owned(monitor, tracked, groups):
    snapshot = processes()
    observed = snapshot.get(monitor['pid'])
    require(observed is None or same_identity(monitor, observed), 'Owned monitor PID was reused')
    parents = {pid for pid, old in tracked.items() if same_identity(old, snapshot.get(pid))}
    if observed:
        parents.add(monitor['pid'])
        tracked[monitor['pid']] = observed
    # A private session/PGID remains reserved while any of its processes exists.
    # Keep captured leader start_ticks to reject a reused numerical group ID.
    for pgid, group in groups.items():
        leader = snapshot.get(pgid)
        require(leader is None or leader['start_ticks'] == group['leader_start_ticks'],
                'An owned private process group number was reused')
        parents.update(pid for pid, info in snapshot.items() if info['pgid'] == pgid and info['sid'] == group['sid'])
    while True:
        added = {pid for pid, info in snapshot.items() if info['ppid'] in parents} - parents
        if not added:
            break
        parents.update(added)
    for pid in parents:
        if pid not in snapshot:
            continue
        info = snapshot[pid]
        tracked[pid] = info
        if info['pgid'] == pid:
            require(info['sid'] == pid, 'Owned group must have a private session')
            groups[pid] = {'leader_start_ticks': info['start_ticks'], 'sid': info['sid']}
    return snapshot


def owned_live(snapshot, tracked, groups):
    result = []
    for pid, info in snapshot.items():
        if same_identity(tracked.get(pid), info):
            result.append(info)
        elif info['pgid'] in groups and info['sid'] == groups[info['pgid']]['sid']:
            leader = snapshot.get(info['pgid'])
            require(leader is None or leader['start_ticks'] == groups[info['pgid']]['leader_start_ticks'],
                    'Owned PGID reused; do not signal it')
            result.append(info)
    return result


def gpu_work(info):
    names = {PurePosixPath(value).name for value in info['argv']}
    return ('trtexec' in names or 'jetson_monitored_build.py' in names or
            any(value.startswith('jetson_gate05_') and 'queue' in value for value in names) or
            any(value.startswith(('jetson_', 'run_')) and ('smoke' in value or 'build_' in value) and
                value.endswith('.py') for value in names) or
            ('nsys' in names and 'profile' in info['argv']))


def require_no_foreign_gpu_work(snapshot, owned=None):
    allowed = {os.getpid()} | {info['pid'] for info in (owned or [])}
    conflicts = [info for info in snapshot.values() if info['pid'] not in allowed and gpu_work(info)]
    require(not conflicts, 'Other GPU build/monitor/queue/smoke work exists: ' + json.dumps(conflicts))


def require_boot(expected):
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == expected,
            'Boot ID changed; entire queue must stop')


def check_fan():
    result = subprocess.run(['systemctl', 'is-active', '--quiet', 'nvfancontrol.service'], timeout=10)
    require(result.returncode == 0, 'Fan service is not active; entire queue must stop')


def resources():
    result = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts = line.split()
        if parts[0].rstrip(':') in ('MemAvailable', 'SwapFree', 'SwapTotal'):
            result[parts[0].rstrip(':') + '_kib'] = int(parts[1])
    temperatures = {}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            temperatures[(zone / 'type').read_text().strip()] = int((zone / 'temp').read_text()) / 1000.0
        except (OSError, ValueError):
            pass
    require(temperatures and 'MemAvailable_kib' in result, 'Required thermal/memory readings are unavailable')
    result['temperatures_c'] = temperatures
    return result


def check_ready(boot):
    require_boot(boot)
    check_fan()
    snapshot = processes()
    require_no_foreign_gpu_work(snapshot)
    require(not any('trtexec' in {PurePosixPath(value).name for value in info['argv']}
                    for info in snapshot.values()), 'A trtexec remains; no next build permitted')
    result = resources()
    require(result['MemAvailable_kib'] >= 3 * 1024 * 1024, 'Less than 3 GiB available before the next build')
    require(max(result['temperatures_c'].values()) < 85, 'Temperature reached 85 C')
    return {'resources': result, 'global_trtexec_empty': True, 'other_gpu_work_empty': True}


def terminal_status(run, monitor, root, boot):
    path = run / 'status.json'
    require(path.is_file(), 'Monitor ended without terminal status')
    status = read(path)
    require(status.get('monitor_pid') == monitor['pid'] and status.get('root') == str(root) and
            status.get('boot_id') == boot and status.get('max_seconds') == 7200,
            'Terminal monitor identity or effective budget differs')
    require(status.get('status') != 'RUNNING', 'Monitor process ended but retained status is RUNNING')
    kernel = (run / 'kernel.log').read_text(errors='replace') if (run / 'kernel.log').is_file() else ''
    require(not any(marker in kernel for marker in STOP_MARKERS), 'Monitor recorded a kernel safety event')
    normal = status.get('status') in ('BUILD_SCRIPT_OK', 'BUILD_SCRIPT_FAILED') and status.get('reason') is None
    timeout = status.get('status') == 'STOPPED_BY_MONITOR' and status.get('reason') == TIMEOUT_REASON
    require(normal or timeout, 'Monitor safety/collector/unknown failure; stop queue: ' + str(status.get('reason')))
    return status


def wait_owned_exit(monitor, tracked, groups, boot, update, limit=40):
    started = time.monotonic()
    while True:
        require_boot(boot)
        snapshot = track_owned(monitor, tracked, groups)
        remaining = owned_live(snapshot, tracked, groups)
        require_no_foreign_gpu_work(snapshot, remaining)
        if not remaining:
            require(not any('trtexec' in {PurePosixPath(value).name for value in info['argv']}
                            for info in snapshot.values()), 'Global trtexec is not empty after monitor termination')
            return {'owned_processes_empty': True, 'global_trtexec_empty': True,
                    'tracked_process_identities': list(tracked.values()), 'owned_private_groups': groups}
        update('WAITING_FOR_OWNED_DESCENDANTS', remaining_processes=remaining)
        require(time.monotonic() - started <= limit, 'Owned descendants remain; never start the next build')
        time.sleep(1)


def stop_owned(process, monitor, tracked, groups, boot):
    """Clean up only recorded same-boot process identities/private sessions."""
    require_boot(boot)
    if process.poll() is None:
        info = proc_info(process.pid)
        require(same_identity(monitor, info), 'Do not signal a reused monitor PID')
        track_owned(monitor, tracked, groups)
        os.kill(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        require_boot(boot)
        snapshot = track_owned(monitor, tracked, groups)
        remaining = owned_live(snapshot, tracked, groups)
        if not remaining:
            return
        # Signal identities individually so a numerical PGID is never blindly targeted.
        for info in remaining:
            current = proc_info(info['pid'])
            if same_identity(info, current):
                try:
                    os.kill(info['pid'], sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            process.poll()
            snapshot = track_owned(monitor, tracked, groups)
            if not owned_live(snapshot, tracked, groups):
                return
            time.sleep(1)
    raise RuntimeError('Owned processes survived cleanup; queue is stopped and requires inspection')


def build_result(root):
    latest = root / 'latest_build.json'
    if not latest.is_file():
        return {'status': 'NO_BUILD_RESULT_RECORD'}
    value = read(latest)
    result = {'latest_build_sha256': sha(latest), 'latest_build': value}
    if 'run_dir' in value:
        run = Path(value['run_dir'])
        require(run.resolve().parent == (root / 'results').resolve(), 'Build result path escapes bundle')
        path = run / 'build_result.json'
        if path.is_file():
            result.update(build_result_path=str(path), build_result_sha256=sha(path), build_result=read(path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path(__file__).resolve().parent / 'plan.json')
    parser.add_argument('--expected-boot-id', help='Exact boot UUID obtained from the current live board; required to run')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--local-archive-root', type=Path,
                        help='Check-only path mapping for the flat extracted archive, without reading any board state')
    args = parser.parse_args()
    require(args.check_only or args.local_archive_root is None, 'Local path mapping is check-only')
    if args.expected_boot_id is not None:
        require(SAFE_BOOT.fullmatch(args.expected_boot_id), 'Expected a valid boot UUID')
    plan = read(args.plan)
    validate_plan(plan)
    inputs = args.plan.resolve().parent
    require(sha(Path(__file__)) == plan['queue_script_sha256'], 'Queue script differs from frozen plan')
    require(sha(inputs / 'budget_revision.json') == plan['budget_revision_sha256'], 'Budget revision differs')
    revision = read(inputs / 'budget_revision.json')
    require(revision['inherited_spec_max_seconds'] == 1800 and revision['effective_monitor_max_seconds'] == 7200 and
            revision['maximum_attempts_per_configuration'] == 1 and revision['scientific_criteria_changed'] is False and
            revision['original_inputs_byte_identical'] is True, 'External budget revision policy differs')
    mapped = lambda row: ((args.local_archive_root.resolve() / PurePosixPath(row['bundle_root']).name)
                          if args.local_archive_root else Path(row['bundle_root']))
    bundles = [verify_bundle(row, mapped(row)) for row in plan['configurations']]
    if args.check_only:
        print(json.dumps({'status': 'EXTENDED_RESTORE_INPUTS_VERIFIED_ONLY', 'bundles': bundles,
                          'plan_sha256': sha(args.plan), 'queue_script_sha256': sha(Path(__file__)),
                          'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200,
                          'native_state_checked': False, 'inference_run': False, 'dataset_evaluation_run': False}, indent=2),
              flush=True)
        return
    require(args.expected_boot_id is not None, '--expected-boot-id from the current live board is required')
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Jetson Linux aarch64 required')
    require(str(inputs) == plan['inputs_root'] and not args.plan.is_symlink(), 'Native plan must be in the exact inputs root')
    require_boot(args.expected_boot_id)
    import fcntl
    lock = Path(str(BASE / '.gate05_frozen_build_queue.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = Path(plan['queue_run'])
    require(not run.exists() and str(run.resolve()) == str(run), 'Fresh canonical queue output is required')
    for row in plan['configurations']:
        root = Path(row['bundle_root'])
        require(str(root.resolve()) == str(root) and not root.is_symlink(), 'Bundle root must be canonical')
        require(not (root / 'latest_build.json').exists() and not (root / 'results').exists() and
                not (root / '.monitored_build.lock').exists(), 'An extended attempt already exists; never retry/overwrite')
    initial_ready = check_ready(args.expected_boot_id)
    run.mkdir(parents=False, exist_ok=False)
    save(run / 'plan.json', plan)
    request = {'schema': 'rampq.gate05.extended_restore_queue_request/v1', 'queue_pid': os.getpid(),
               'queue_process': proc_info(os.getpid()), 'queue_script_sha256': sha(Path(__file__)),
               'original_plan_sha256': sha(args.plan), 'budget_revision_sha256': sha(inputs / 'budget_revision.json'),
               'boot_id': args.expected_boot_id, 'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'bundles': bundles, 'initial_ready': initial_ready,
               'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200,
               'inference_run': False, 'dataset_evaluation_run': False, 'native_precision_pass_claim': False}
    save(run / 'request.json', request)
    results = []
    queue_started = time.monotonic()
    current = None
    monitor, tracked, groups = None, {}, {}

    def update(state, **extra):
        value = dict(request, status=state, elapsed_seconds=round(time.monotonic() - queue_started, 2),
                     completed_builds=results, **extra)
        save(run / 'status.json', value)
        with (run / 'progress.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def terminate(_signum, _frame):
        raise KeyboardInterrupt('Queue termination requested')

    signal.signal(signal.SIGTERM, terminate)
    try:
        update('VALIDATED_EXTENDED_RESTORE_QUEUE_STARTING')
        for item in plan['configurations']:
            name, root = item['configuration'], Path(item['bundle_root'])
            verify_bundle(item, root)
            require(not (root / 'latest_build.json').exists() and not (root / 'results').exists() and
                    not (root / '.monitored_build.lock').exists(), 'An attempt already exists: ' + name)
            initial = check_ready(args.expected_boot_id)
            monitor_run = run / ('monitor_' + name)
            command = [sys.executable, '-u', str(root / 'jetson_monitored_build.py'), '--root', str(root),
                       '--run', str(monitor_run), '--max-seconds', '7200',
                       '--build-script', 'jetson_build_precision_candidate.py']
            tracked, groups = {}, {}
            with (run / ('console_' + name + '.log')).open('wb', buffering=0) as handle:
                current = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                monitor = proc_info(current.pid)
                require(live(monitor) and monitor['pgid'] == current.pid and monitor['sid'] == current.pid,
                        'Cannot capture the owned private monitor identity')
                tracked[current.pid] = monitor
                groups[current.pid] = {'leader_start_ticks': monitor['start_ticks'], 'sid': monitor['sid']}
                save(run / ('launch_' + name + '.json'), {'configuration': name, 'command': command,
                     'monitor_process': monitor, 'boot_id': args.expected_boot_id,
                     'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200})
                started, low_memory = time.monotonic(), 0
                while current.poll() is None:
                    require_boot(args.expected_boot_id)
                    check_fan()
                    snapshot = track_owned(monitor, tracked, groups)
                    require_no_foreign_gpu_work(snapshot, owned_live(snapshot, tracked, groups))
                    sample = resources()
                    low_memory = low_memory + 1 if sample['MemAvailable_kib'] < 384 * 1024 else 0
                    require(max(sample['temperatures_c'].values()) < 85, 'Queue temperature reached 85 C')
                    require(low_memory < 3, 'Queue memory stayed below 384 MiB for three consecutive samples')
                    kernel = (monitor_run / 'kernel.log').read_text(errors='replace') if (monitor_run / 'kernel.log').is_file() else ''
                    require(not any(marker in kernel for marker in STOP_MARKERS), 'New kernel USB/OOM/panic safety event')
                    elapsed = round(time.monotonic() - started, 2)
                    update('BUILD_MONITOR_RUNNING', configuration=name, monitor_pid=current.pid,
                           monitor_process=monitor, monitor_run=str(monitor_run), build_seconds=elapsed,
                           latest_resources=sample, low_memory_consecutive_samples=low_memory)
                    require(elapsed <= 7380, 'Owned monitor exceeded 7200 seconds plus bounded cleanup')
                    time.sleep(2)
            terminal = terminal_status(monitor_run, monitor, root, args.expected_boot_id)
            terminated = wait_owned_exit(monitor, tracked, groups, args.expected_boot_id, update)
            ready = check_ready(args.expected_boot_id)
            result = {'configuration': name, 'monitor_pid': current.pid, 'monitor_process': monitor,
                      'monitor_returncode': current.returncode, 'monitor_run': str(monitor_run),
                      'terminal_monitor_status': terminal, 'terminal_monitor_status_sha256': sha(monitor_run / 'status.json'),
                      'initial_ready': initial, 'after_termination_ready': ready, 'process_termination': terminated,
                      'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200,
                      'native_precision_audit': 'NOT_RUN', 'fixed_image_execution': 'NOT_RUN'}
            result.update(build_result(root))
            results.append(result)
            save(run / ('result_' + name + '.json'), result)
            current, monitor, tracked, groups = None, None, {}, {}
            update('ONE_EXTENDED_RESTORE_BUILD_FINISHED', configuration=name)
        update('EXTENDED_RESTORE_QUEUE_FINISHED_AUDIT_AND_SMOKE_PENDING')
        print('EXTENDED_RESTORE_QUEUE_FINISHED_AUDIT_AND_SMOKE_PENDING', flush=True)
    except BaseException as error:
        cleanup = {'status': 'NO_OWNED_MONITOR_TO_CLEAN'}
        if current is not None and monitor is not None:
            try:
                stop_owned(current, monitor, tracked, groups, args.expected_boot_id)
                cleanup = wait_owned_exit(monitor, tracked, groups, args.expected_boot_id, update)
            except BaseException as cleanup_error:
                cleanup = {'status': 'CLEANUP_REQUIRES_INSPECTION',
                           'error': type(cleanup_error).__name__ + ': ' + str(cleanup_error)}
        update('QUEUE_STOPPED_REQUIRES_INSPECTION', error=type(error).__name__ + ': ' + str(error), cleanup=cleanup)
        print('QUEUE_STOPPED_REQUIRES_INSPECTION: ' + str(error), flush=True)
        raise


if __name__ == '__main__':
    main()
