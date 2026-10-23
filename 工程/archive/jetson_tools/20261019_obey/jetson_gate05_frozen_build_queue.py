"""Build the remaining frozen Gate 0.5 candidates once, sequentially, on Jetson.

This is an immediate, bounded build queue, not a scheduler. It never imports a
GPU library, runs inference, evaluates data, or declares a precision pass. Each
candidate still uses the frozen 1800-second monitored builder. Invoke with
--plan PLAN.json --run /home/nvidia/rampq/NEW_QUEUE_DIRECTORY.
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


SCHEMA = 'rampq.gate05.frozen_build_queue_plan/v1'
BASE = PurePosixPath('/home/nvidia/rampq')
ORDER = ('r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1')
GROUPS = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'}
MASKS = {'fp16_control': set(), 'r_only_G1': GROUPS - {'G1'},
         'r_only_G1_G5': GROUPS - {'G1', 'G5'},
         'r_only_G1_G2': GROUPS - {'G1', 'G2'}, 'q_only_G1': {'G1'}}
STOP_MARKERS = ('controller error detected', 'HC died;', 'unregister_netdevice: waiting',
                'Out of memory:', 'oom-kill:', 'Kernel panic')
TIMEOUT_REASON = 'Build supervision time limit reached'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def board_path(value):
    require(isinstance(value, str), 'Expected an absolute board path')
    path = PurePosixPath(value)
    require(path.is_absolute() and path.parent == BASE and '..' not in path.parts,
            'Expected a direct child directory under /home/nvidia/rampq: ' + value)
    require(str(path) == value, 'Board path must be canonical: ' + value)
    return path


def validate_plan_shape(plan, run):
    require(plan['schema'] == SCHEMA, 'Unsupported queue plan schema')
    require(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', plan['expected_boot_id']),
            'Expected an exact boot UUID')
    require(plan['max_build_seconds'] == 1800, 'Frozen build limit must be 1800 seconds')
    wait = plan['wait_for']
    require(type(wait['pid']) is int and wait['pid'] > 1, 'Invalid external monitor PID')
    require(type(wait['proc_start_ticks']) is int and wait['proc_start_ticks'] > 0,
            'External monitor /proc start ticks required to reject PID reuse')
    require(board_path(wait['bundle_root']).name == 'gate05_precision_fp16_control_20261019',
            'Only the current matched FP16 monitor may be awaited')
    board_path(wait['monitor_run'])
    require(wait['bundle_root'] != wait['monitor_run'], 'Monitor output must differ from bundle')
    items = plan['configurations']
    names = [item['configuration'] for item in items]
    require(names in [list(ORDER[:3]), list(ORDER)], 'Expected three frozen recovery configurations, then optional q_only_G1')
    paths = [str(board_path(str(run))), wait['bundle_root'], wait['monitor_run']]
    for item in [wait] + items:
        require(re.fullmatch('[0-9a-f]{64}', item['bundle_manifest_sha256']), 'Expected frozen manifest SHA256')
    for item in items:
        expected_name = 'gate05_precision_' + item['configuration'] + '_20261019'
        require(board_path(item['bundle_root']).name == expected_name, 'Unexpected frozen bundle root')
        paths.append(item['bundle_root'])
    require(len(paths) == len(set(paths)), 'Queue, monitor, and bundle directories must differ')


def verify_bundle(item, expected_configuration):
    root = Path(item['bundle_root'])
    require(not root.is_symlink() and str(root.resolve()) == str(root), 'Bundle directory must not be redirected')
    manifest_path = root / 'manifest.json'
    require(sha(manifest_path) == item['bundle_manifest_sha256'], 'Frozen bundle manifest SHA256 mismatch')
    manifest = read(manifest_path)
    require(manifest['schema'] == 'rampq.gate05.precision_runtime_bundle/v1' and
            manifest['configuration'] == expected_configuration, 'Bundle manifest identity differs')
    required = {'model.onnx', 'spec.json', 'graph_audit.json', 'module_map.json', 'preflight.json',
                'diagnostic_scope.json', 'jetson_build_precision_candidate.py', 'jetson_monitored_build.py'}
    require(required <= set(manifest['files']), 'Bundle manifest is incomplete')
    for name, info in manifest['files'].items():
        path = root / name
        require(Path(name).name == name and path.resolve().parent == root, 'Bundle file escapes frozen directory')
        require(path.is_file() and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                'Bundle input hash/size differs: ' + name)
    spec = read(root / 'spec.json')
    require(spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1' and
            spec['configuration'] == expected_configuration and spec['diagnostic_only'] is True,
            'Build spec identity/scope differs')
    require(spec['max_build_seconds'] == 1800 and spec['workspace_mib'] == 1024 and spec['avg_timing'] == 1,
            'Frozen build resource settings differ')
    requests = spec['requests']
    require(len(requests) == len({row['onnx_node'] for row in requests}) == 88,
            'Expected 88 unique source Conv requests')
    require({row['group'] for row in requests} == GROUPS, 'Group inventory differs')
    require(all(row['precision'] == ('INT8' if row['group'] in MASKS[expected_configuration] else 'FP16')
                for row in requests), 'Frozen requested precision mask differs')
    quantized = sum(row['precision'] == 'INT8' for row in requests)
    require((quantized, 88 - quantized) == (spec['quantized_conv_count'], spec['fp16_conv_count']),
            'Requested Conv count differs')
    require(spec['original_local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED' and
            read(root / 'preflight.json')['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED',
            'Original numerical parity failure must remain recorded')
    for filename, key in [('model.onnx', 'graph_sha256'), ('graph_audit.json', 'graph_audit_sha256'),
                          ('module_map.json', 'module_map_sha256'), ('preflight.json', 'failed_preflight_sha256'),
                          ('diagnostic_scope.json', 'diagnostic_scope_sha256')]:
        require(sha(root / filename) == spec[key], 'Spec input hash differs: ' + filename)
    scope = read(root / 'diagnostic_scope.json')
    require(scope['status'] == 'DIAGNOSTIC_MATRIX_BUILD_AND_FIXED_IMAGE_ONLY' and
            scope['original_tolerances_changed'] is False and scope['dataset_evaluation_run'] is False,
            'Diagnostic scope differs')
    return {'configuration': expected_configuration, 'bundle_root': str(root),
            'manifest_sha256': sha(manifest_path), 'spec_sha256': sha(root / 'spec.json')}


def proc_info(pid):
    """Return a live process identity; zombies are terminal, PID reuse is distinct."""
    try:
        text = Path('/proc/{}/stat'.format(pid)).read_text()
        fields = text[text.rfind(')') + 2:].split()
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'start_ticks': int(fields[19]),
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


def track_children(pid, tracked):
    snapshot = processes()
    parents = {pid} | {number for number, old in tracked.items()
                          if number in snapshot and old['start_ticks'] == snapshot[number]['start_ticks']}
    while True:
        added = {number for number, info in snapshot.items() if info['ppid'] in parents} - parents
        if not added:
            break
        parents.update(added)
    for number in parents - {pid}:
        if number in snapshot:
            tracked[number] = snapshot[number]


def remaining_children(tracked):
    snapshot = processes()
    groups = {info['pgid'] for info in tracked.values()}
    return [info for pid, info in snapshot.items() if
            (pid in tracked and info['start_ticks'] == tracked[pid]['start_ticks']) or info['pgid'] in groups]


def require_boot(expected):
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == expected,
            'Board boot ID changed; stop and inspect before any further build')


def resource_snapshot():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        fields = line.split()
        if fields[0] == 'MemAvailable:':
            memory['available_kib'] = int(fields[1])
    temperatures = {}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            temperatures[(zone / 'type').read_text().strip()] = int((zone / 'temp').read_text()) / 1000.0
        except (OSError, ValueError):
            pass
    require(temperatures, 'No thermal measurements available')
    memory['temperatures_c'] = temperatures
    return memory


def check_ready(expected_boot):
    require_boot(expected_boot)
    active = subprocess.run(['pgrep', '-x', 'trtexec'], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True, timeout=10)
    require(active.returncode == 1, 'A trtexec process remains or process check failed: ' + active.stdout)
    fan = subprocess.run(['systemctl', 'is-active', '--quiet', 'nvfancontrol.service'], timeout=10)
    require(fan.returncode == 0, 'Fan service is not active')
    snapshot = resource_snapshot()
    require(snapshot['available_kib'] >= 3 * 1024 * 1024, 'Less than 3 GiB available; do not start another build')
    require(max(snapshot['temperatures_c'].values()) < 85, 'Temperature reached the 85 C stop limit')
    return snapshot


def check_fan():
    result = subprocess.run(['systemctl', 'is-active', '--quiet', 'nvfancontrol.service'], timeout=10)
    require(result.returncode == 0, 'Fan service stopped; queue cannot continue')


def validate_terminal(run, expected_pid, bundle_root, expected_boot):
    require((run / 'status.json').is_file(), 'Monitor process ended without a terminal status record')
    status = read(run / 'status.json')
    require(status.get('monitor_pid') == expected_pid and status.get('root') == bundle_root and
            status.get('boot_id') == expected_boot, 'Monitor terminal identity differs')
    require(status.get('status') != 'RUNNING', 'External monitor lost: PID ended but status remains RUNNING')
    kernel = (run / 'kernel.log').read_text(errors='replace') if (run / 'kernel.log').is_file() else ''
    require(not any(marker in kernel for marker in STOP_MARKERS), 'Monitor recorded a kernel safety event')
    safe = (status.get('status') in ('BUILD_SCRIPT_OK', 'BUILD_SCRIPT_FAILED') and status.get('reason') is None)
    safe = safe or (status.get('status') == 'STOPPED_BY_MONITOR' and status.get('reason') == TIMEOUT_REASON)
    require(safe, 'Monitor safety/collector/unknown failure: ' + str(status.get('reason')))
    return status


def await_external(plan, update):
    wait, boot = plan['wait_for'], plan['expected_boot_id']
    run = Path(wait['monitor_run'])
    tracked = {}
    started = time.monotonic()
    while True:
        require_boot(boot)
        check_fan()
        info = proc_info(wait['pid'])
        if not live(info):
            break
        require(info['start_ticks'] == wait['proc_start_ticks'], 'External monitor PID was reused')
        argv = info['argv']
        script = str(Path(wait['bundle_root']) / 'jetson_monitored_build.py')
        require(script in argv and '--root' in argv and '--run' in argv, 'External monitor command identity differs')
        require(argv[argv.index('--root') + 1] == wait['bundle_root'] and
                argv[argv.index('--run') + 1] == wait['monitor_run'], 'External monitor root/run identity differs')
        track_children(wait['pid'], tracked)
        update('WAITING_FOR_CURRENT_FP16_MONITOR', external_monitor_pid=wait['pid'],
               observed_process_state=info['state'], elapsed_wait_seconds=round(time.monotonic() - started, 2))
        require(time.monotonic() - started <= 2100, 'External monitor exceeded bounded wait; inspect it before continuing')
        time.sleep(2)
    status = validate_terminal(run, wait['pid'], wait['bundle_root'], boot)
    wait_for_children(tracked, boot, update)
    check_ready(boot)
    return status


def wait_for_children(tracked, boot, update):
    started = time.monotonic()
    while True:
        require_boot(boot)
        remaining = remaining_children(tracked)
        if not remaining:
            return
        update('WAITING_FOR_MONITOR_DESCENDANTS_TO_END', remaining_processes=remaining)
        require(time.monotonic() - started <= 40, 'Monitor descendants/process groups still live; no next build started')
        time.sleep(2)


def stop_owned_monitor(process, tracked):
    """Interrupt only this queue's child monitor, allowing its own cleanup first."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        track_children(process.pid, tracked)
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=35)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def collect_build_result(root):
    latest = root / 'latest_build.json'
    if not latest.is_file():
        return {'status': 'NO_BUILD_RESULT_RECORD'}
    value = read(latest)
    result = {'latest_build_sha256': sha(latest), 'latest_build': value}
    if 'run_dir' in value:
        run = Path(value['run_dir'])
        require(run.resolve().parent == (root / 'results').resolve(), 'Latest build result path escapes bundle')
        path = run / 'build_result.json'
        if path.is_file():
            result['build_result_path'] = str(path)
            result['build_result_sha256'] = sha(path)
            result['build_result'] = read(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true', help='Validate frozen files/plan only; start no process')
    args = parser.parse_args()
    plan = read(args.plan)
    validate_plan_shape(plan, args.run)
    bundles = [verify_bundle(plan['wait_for'], 'fp16_control')]
    bundles += [verify_bundle(item, item['configuration']) for item in plan['configurations']]
    if args.check_only:
        print(json.dumps({'status': 'PLAN_AND_FROZEN_BUNDLES_VERIFIED_ONLY', 'bundles': bundles,
                          'inference_run': False, 'dataset_evaluation_run': False}, indent=2), flush=True)
        return
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Jetson Linux aarch64 required')

    def terminate(_signum, _frame):
        raise KeyboardInterrupt('Queue termination requested')

    signal.signal(signal.SIGTERM, terminate)
    import fcntl
    lock = Path(str(BASE / '.gate05_frozen_build_queue.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = args.run
    require(not run.exists() and str(run.resolve()) == str(run), 'Choose a fresh canonical queue output directory')
    run.mkdir(parents=False, exist_ok=False)
    save(run / 'plan.json', plan)
    request = {'schema': 'rampq.gate05.frozen_build_queue_request/v1', 'queue_pid': os.getpid(),
               'queue_script_sha256': sha(Path(__file__)), 'original_plan_sha256': sha(args.plan),
               'boot_id': plan['expected_boot_id'], 'bundles': bundles,
               'inference_run': False, 'dataset_evaluation_run': False, 'native_precision_pass_claim': False}
    save(run / 'request.json', request)
    results, current = [], None
    queue_started = time.monotonic()

    def update(state, **extra):
        value = dict(request, status=state, elapsed_seconds=round(time.monotonic() - queue_started, 2),
                     completed_builds=results, **extra)
        save(run / 'status.json', value)
        with (run / 'progress.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    try:
        update('VALIDATED_WAITING_FOR_CURRENT_FP16')
        external = await_external(plan, update)
        save(run / 'external_fp16_terminal.json', external)
        for item in plan['configurations']:
            name, root = item['configuration'], Path(item['bundle_root'])
            verify_bundle(item, name)
            require(not (root / 'latest_build.json').exists() and
                    (not (root / 'results').exists() or not any((root / 'results').iterdir())),
                    'Candidate already has a build attempt; preserve it and inspect before any retry: ' + name)
            initial = check_ready(plan['expected_boot_id'])
            monitor_run = run / ('monitor_' + name)
            command = [sys.executable, '-u', str(root / 'jetson_monitored_build.py'), '--root', str(root),
                       '--run', str(monitor_run), '--max-seconds', '1800',
                       '--build-script', 'jetson_build_precision_candidate.py']
            tracked = {}
            with (run / ('console_' + name + '.log')).open('wb', buffering=0) as handle:
                current = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                started = time.monotonic()
                try:
                    while current.poll() is None:
                        require_boot(plan['expected_boot_id'])
                        check_fan()
                        track_children(current.pid, tracked)
                        update('BUILD_MONITOR_RUNNING', configuration=name, monitor_pid=current.pid,
                               monitor_run=str(monitor_run), build_seconds=round(time.monotonic() - started, 2))
                        require(time.monotonic() - started <= 1920,
                                'Owned monitor exceeded build plus cleanup bound')
                        time.sleep(2)
                except BaseException:
                    stop_owned_monitor(current, tracked)
                    raise
            terminal = validate_terminal(monitor_run, current.pid, str(root), plan['expected_boot_id'])
            wait_for_children(tracked, plan['expected_boot_id'], update)
            final_resources = resource_snapshot()
            result = {'configuration': name, 'monitor_pid': current.pid, 'monitor_returncode': current.returncode,
                      'monitor_run': str(monitor_run), 'terminal_monitor_status': terminal,
                      'initial_resources': initial, 'final_resources': final_resources,
                      'native_precision_audit': 'NOT_RUN', 'fixed_image_execution': 'NOT_RUN'}
            result.update(collect_build_result(root))
            results.append(result)
            save(run / ('result_' + name + '.json'), result)
            current = None
            update('ONE_FROZEN_BUILD_FINISHED', configuration=name)
            check_ready(plan['expected_boot_id'])
        update('FROZEN_BUILD_QUEUE_FINISHED_AUDIT_AND_SMOKE_PENDING')
        print('FROZEN_BUILD_QUEUE_FINISHED_AUDIT_AND_SMOKE_PENDING', flush=True)
    except BaseException as error:
        update('QUEUE_STOPPED_REQUIRES_INSPECTION', error=type(error).__name__ + ': ' + str(error))
        print('QUEUE_STOPPED_REQUIRES_INSPECTION: ' + str(error), flush=True)
        raise


if __name__ == '__main__':
    main()
