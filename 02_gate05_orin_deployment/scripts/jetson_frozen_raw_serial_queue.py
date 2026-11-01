"""Run six frozen raw-output jobs serially; default is input checks only.

Each immutable monitor/guard owns one private wrapper/runner process group.
No graph build, warmup, retry, NMS, score computation or GPU import exists here.
The plan and every work input are pinned before --execute can claim the queue.
"""
import argparse
import atexit
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ORDER = ['fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1']
RUNNER = 'run_gate05_frozen_raw_dataset_v2.py'
RUNNER_SHA = 'ef7e2780203c04442ee3378b7cea31093dafa40eb29c4cea9c012cdb0cd4d26b'
WRAPPER = 'jetson_run_frozen_raw_work.py'
HELPERS = {
    'jetson_monitored_build.py': '71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1',
    'jetson_owned_group_guard.py': '4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4',
    'jetson_collect_isolated_numeric_probe.py': 'af9b7f823e4f889813fb88b6e8a529a6b1a6423b0c7ead0d5704513b517775eb',
}
BAD_KERNEL = ('controller error detected', 'HC died;', 'unregister_netdevice: waiting',
              'Out of memory:', 'oom-kill:', 'Kernel panic')
IDENTITY_KEYS = ('root', 'monitor_pid', 'build_pid', 'max_seconds', 'build_script', 'boot_id')
MONITOR_EXIT_GRACE = 90


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    require(not path.is_symlink(), 'Output became symlink: ' + str(path))
    temporary = path.with_name(path.name + '.queue-' + str(os.getpid()) + '.tmp')
    require(not temporary.is_symlink(), 'Output temporary became symlink')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def contained(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe work path: ' + name)
    path = root / relative
    require(root in path.resolve().parents and not path.is_symlink(), 'Work file escapes root: ' + name)
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), 'Work file has symlink parent: ' + name)
    return path


def runner_arguments(root, spec, contract):
    cfg, backend = spec['configuration'], spec['backend']
    policy = contract['configurations'][cfg][backend]
    args = ['--contract', str(root / 'raw_contract.json'), '--contract-sha256', spec['contract_sha256'],
            '--configuration', cfg, '--backend', backend, '--inputs', spec['inputs_root'],
            '--graph', spec['graph_file'], '--state-root', policy['attempt_state_root'],
            '--output', str(root / 'raw_outputs'), '--evidence-root', str(root)]
    if backend == 'trt':
        args += ['--engine', spec['engine_file']]
    return args


def verify_manifest_inputs(root, manifest):
    require(isinstance(manifest.get('files'), dict) and manifest['files'], 'Missing work input inventory')
    for name, info in manifest['files'].items():
        path = contained(root, name)
        require(path.is_file() and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                'Manifest input changed: ' + name)


def verify_member(item, plan, run_input_check=True):
    root, base = Path(item['root']), Path(plan['base'])
    require(root.is_absolute() and root.parent == base and root.resolve() == root and
            root.is_dir() and not root.is_symlink(), 'Noncanonical work root')
    for name, field in (('manifest.json', 'manifest_sha256'), ('execution_spec.json', 'execution_spec_sha256'),
                        ('raw_contract.json', 'contract_sha256')):
        require(sha(root / name) == item[field], 'Frozen work identity changed: ' + name)
    require(item['contract_sha256'] == plan['contract_sha256'], 'Members must share one frozen contract')
    manifest, spec, contract = read(root / 'manifest.json'), read(root / 'execution_spec.json'), read(root / 'raw_contract.json')
    verify_manifest_inputs(root, manifest)
    required = {'execution_spec.json', 'raw_contract.json', RUNNER, WRAPPER} | set(HELPERS)
    require(required <= set(manifest['files']), 'Manifest omits a required execution input')
    require(sha(root / RUNNER) == plan['runner_sha256'] == RUNNER_SHA and
            sha(root / WRAPPER) == plan['wrapper_sha256'], 'Runner/wrapper identity differs')
    for name, digest in HELPERS.items():
        require(sha(root / name) == plan['helper_sha256'][name] == digest, 'Immutable helper changed: ' + name)
    require(spec['schema'] == 'rampq.gate05.raw_work_execution/v1' and
            spec['wrapper_sha256'] == plan['wrapper_sha256'] and spec['contract_sha256'] == plan['contract_sha256'] and
            spec['configuration'] == item['configuration'] and spec['backend'] == plan['backend'] == 'trt' and
            spec['max_seconds'] == 1800 and spec['extra_model_calls'] == 0, 'Execution spec scope differs')
    require(contract['schema'] == 'rampq.gate05.frozen_raw_dataset_contract/v1' and
            contract['runner_sha256'] == RUNNER_SHA and
            contract['maximum_attempts_per_backend_configuration'] == 1 and
            contract['warmup_calls'] == contract['extra_inference_calls'] == contract['new_builds'] == 0 and
            contract['no_NMS'] is True and contract['dataset_accuracy_evaluation_by_runner'] is False,
            'Contract scope differs')
    require(contract['configurations'][item['configuration']]['trt']['authorized'] is True and
            contract['configurations'][item['configuration']]['trt']['calls'] == 150, 'Member is not authorized')
    for name in ('monitor', 'raw_outputs', 'pure_input_check.json', 'raw_runner_exit.json',
                 'monitor_console.log', 'supplemental_group_guard.json', 'queue_input_receipt.json',
                 'queue_terminal_receipt.json', 'queue_exception_cleanup.json', 'end_board_status.json'):
        require(not (root / name).exists(), 'Member has already started or contains runtime evidence: ' + name)
    archive = Path(item['archive'])
    require(archive.is_absolute() and archive.resolve() == archive and archive.parent == Path('/home/nvidia/rampq') and
            archive.suffix == '.zip' and root not in archive.parents and not archive.exists(), 'Archive must be fresh outside work')
    receipt = {'configuration': item['configuration'], 'root': str(root), 'manifest_sha256': item['manifest_sha256'],
               'execution_spec_sha256': item['execution_spec_sha256'], 'contract_sha256': item['contract_sha256'],
               'raw_runner_check_executed': False, 'native_runtime_imported': False, 'model_calls': 0}
    if run_input_check:
        command = [sys.executable, '-B', str(root / RUNNER)] + runner_arguments(root, spec, contract) + ['--check-input-only']
        result = subprocess.run(command, cwd=str(root), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        require(result.returncode == 0, 'Pure raw input check failed: ' + result.stdout + result.stderr)
        checked = json.loads(result.stdout)
        require(checked['status'] == 'FROZEN150_RAW_DATASET_INPUTS_VERIFIED_NO_MODEL_EXECUTION' and
                checked['native_runtime_imported'] is False and checked['model_calls'] == 0 and
                checked['planned_calls'] == 150 and checked['new_builds'] == 0 and checked['NMS_run'] is False and
                checked['lineage']['configuration'] == item['configuration'] and checked['lineage']['backend'] == 'trt' and
                checked['lineage']['contract_sha256'] == plan['contract_sha256'], 'Input check returned wrong scope')
        receipt.update(raw_runner_check_executed=True, input_check_argv=command,
                       input_check_returncode=result.returncode, input_check_stdout=result.stdout, input_check_stderr=result.stderr)
    return receipt


def validate_plan(args):
    require(sha(args.plan) == args.plan_sha256, 'Plan SHA256 differs')
    plan = read(args.plan)
    require(plan['schema'] == 'rampq.gate05.frozen_raw_serial_queue_plan/v1' and
            plan['status'] == 'FROZEN_BEFORE_NATIVE_RAW_DATASET_EXECUTION' and
            plan['queue_script_sha256'] == sha(Path(__file__)) and plan['configuration_order'] == ORDER and
            [row['configuration'] for row in plan['members']] == ORDER, 'Wrong frozen queue plan/order/tool')
    require(plan['backend'] == 'trt' and plan['model_calls_per_member'] == 150 and plan['maximum_model_calls'] == 900 and
            plan['warmup_calls'] == plan['extra_inference_calls'] == plan['new_builds'] == 0 and
            plan['maximum_attempts'] == 1 and plan['max_seconds_per_member'] == 1800 and
            plan['collection_timeout_seconds'] == 240 and plan['runner_sha256'] == RUNNER_SHA and
            plan['helper_sha256'] == HELPERS, 'Frozen queue calls/budget/helpers differ')
    base = Path(plan['base'])
    require(base.is_absolute() and base.resolve() == base and base.parent == Path('/home/nvidia/rampq') and
            base.is_dir() and not base.is_symlink(), 'Unsafe queue stage')
    require(len({row['root'] for row in plan['members']}) == len({row['archive'] for row in plan['members']}) == 6,
            'Duplicate work root or archive')
    require(not (base / 'raw_queue_attempt.json').exists() and not (base / 'raw_queue_status.json').exists(),
            'Queue has already been claimed; no replay or resume')
    require(sys.platform.startswith('linux') and boot() == plan['expected_boot_id'], 'Expected Linux board boot changed')
    receipts = [verify_member(item, plan) for item in plan['members']]
    return plan, receipts


def load_group_module(root):
    path = root / 'jetson_owned_group_guard.py'
    require(sha(path) == HELPERS[path.name], 'Guard helper drifted before import')
    spec = importlib.util.spec_from_file_location('raw_queue_guard_' + hashlib.sha256(str(root).encode()).hexdigest()[:12], path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def wait_complete_json(path, child, seconds, required):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            value = read(path)
            # The immutable supplemental guard first publishes {}. A filename
            # alone cannot authenticate or acknowledge its ready state.
            if isinstance(value, dict) and all(name in value for name in required):
                return value
        if child.poll() is not None:
            break
        time.sleep(.2)
    raise RuntimeError('Missing complete child receipt: ' + str(path))


def verify_monitor(value, root, monitor_pid, expected_boot):
    expected = {'root': str(root), 'monitor_pid': monitor_pid, 'max_seconds': 1800,
                'build_script': str(root / WRAPPER), 'boot_id': expected_boot}
    require(all(value.get(k) == v for k, v in expected.items()) and type(value.get('build_pid')) is int and
            value['build_pid'] > 1, 'Monitor identity/budget changed')


def verify_guard(value, monitor, guard_pid, owned):
    require(value['guard_pid'] == guard_pid and value['root'] == monitor['root'] and value['boot_id'] == monitor['boot_id'] and
            value['owned_pgid'] == monitor['build_pid'] and value['leader_start_ticks'] == owned.leader_start_ticks and
            all(value['monitor_identity'][k] == monitor[k] for k in IDENTITY_KEYS if k != 'boot_id'),
            'Supplemental guard identity changed')
    require(value['status'] in ('WATCHING', 'COMPLETE') and not value.get('intervention_performed', False),
            'Guard failed or intervened; stop queue')
    require(not any(row.get('dispatch') == 'SENT' for row in value['termination_actions']), 'Guard signalled a group')
    if value['status'] == 'COMPLETE':
        require(value['owned_group_fully_empty'] is True and not owned.snapshot(), 'Completed guard group not physically empty')


def verify_raw_result(root, item, plan):
    state = read(root / 'raw_outputs/run_state.json')
    raw = read(root / 'raw_outputs/raw_manifest.json')
    contract = read(root / 'raw_contract.json')
    require(state['status'] == raw['status'] == 'ALL150_RAW_OUTPUTS_SERIALIZED_NO_SCORING' and
            state['phase'] == 'COMPLETE' and
            state['model_calls_attempted'] == state['model_calls_completed'] == state['serialized_images'] == len(raw['images']) == 150,
            'Incomplete or failed raw dataset job')
    require(state['all_outputs_finite'] is True and not state['nonfinite_images'] and not raw['nonfinite_images'],
            'Nonfinite outputs preserved; stop before next configuration')
    require(state['NMS_run'] is False and state['accuracy_pass_claim'] is False and state['performance_pass_claim'] is False and
            raw['NMS_run'] is False and raw['target_matching_run'] is False and raw['dataset_accuracy_evaluated'] is False,
            'Unexpected scoring/postprocessing occurred')
    for lineage in (state['lineage'], raw['lineage']):
        require(lineage['contract_sha256'] == plan['contract_sha256'] and lineage['configuration'] == item['configuration'] and
                lineage['backend'] == 'trt' and lineage['runner_sha256'] == RUNNER_SHA and
                lineage['graph_sha256'] == contract['configurations'][item['configuration']]['graph_sha256'] and
                lineage['engine_sha256'] == contract['configurations'][item['configuration']]['trt']['engine_sha256'],
                'Raw source/engine lineage differs')
    require([r['image_index'] for r in raw['images']] == list(range(150)) and
            [r['image'] for r in raw['images']] == contract['input_manifest']['ordered_ids'], 'Raw image order differs')
    for row in raw['images']:
        path = contained(root / 'raw_outputs', row['raw_file'])
        require(path.is_file() and path.stat().st_size == row['raw_bytes'] and sha(path) == row['raw_sha256'] and
                row['all_outputs_finite'] is True, 'Raw image missing, nonfinite or changed')
    require(read(root / 'raw_runner_exit.json')['returncode'] == 0, 'Raw wrapper recorded nonzero runner exit')
    return {'model_calls_attempted': 150, 'model_calls_completed': 150, 'serialized_images': 150,
            'raw_manifest_sha256': sha(root / 'raw_outputs/raw_manifest.json'),
            'run_state_sha256': sha(root / 'raw_outputs/run_state.json'), 'all_outputs_finite': True,
            'NMS_calls': 0, 'accuracy_claim': False}


def busy_jobs():
    result = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            names = [Path(word.decode(errors='replace')).name for word in (folder / 'cmdline').read_bytes().split(b'\0') if word]
        except OSError:
            continue
        if any(name.startswith(('jetson_build_', 'jetson_matched_fixed_input_adapter', 'jetson_matched_profile_child_guard',
                                'run_gate05_frozen_raw_dataset', 'jetson_frozen_raw_serial_queue', 'trtexec')) or
               name in ('run_fixed.py', WRAPPER, 'nsys', 'nsys-launcher', 'nsys-agent') for name in names):
            result.append({'pid': int(folder.name), 'argv_basenames': names})
    return result


def interrupt(_sig, _frame):
    raise KeyboardInterrupt('Frozen raw queue interrupted; clean authenticated current group')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-input-only', action='store_true')
    args = parser.parse_args()
    plan, receipts = validate_plan(args)
    if not args.execute:
        print(json.dumps({'schema': 'rampq.gate05.frozen_raw_queue_input_review/v1',
              'status': 'SIX_RAW_WORKS_VERIFIED_NO_MODEL_EXECUTION', 'members': receipts,
              'native_runtime_imported': False, 'model_calls': 0, 'new_builds': 0, 'NMS_calls': 0,
              'markers_created': 0, 'child_groups_started': 0}, indent=2), flush=True)
        return
    require(os.getpid() == os.getpgrp() == os.getsid(0), 'Launch queue in its own private session')
    require(not busy_jobs(), 'Another native job is active: ' + repr(busy_jobs()))
    base = Path(plan['base'])
    write_new(base / 'raw_queue_attempt.json', {'queue_script_sha256': sha(Path(__file__)),
              'plan_sha256': args.plan_sha256, 'pid': os.getpid(), 'boot_id': boot(),
              'started_UTC': datetime.now(timezone.utc).isoformat(), 'maximum_model_calls': 900,
              'maximum_attempts': 1, 'new_builds': 0})
    state = {'schema': 'rampq.gate05.frozen_raw_serial_queue_status/v1', 'status': 'RUNNING',
             'pid': os.getpid(), 'pgid': os.getpgrp(), 'sid': os.getsid(0), 'boot_id': boot(),
             'plan_sha256': args.plan_sha256, 'queue_script_sha256': sha(Path(__file__)),
             'members': [], 'maximum_model_calls': 900, 'completed_model_calls': 0,
             'new_builds': 0, 'NMS_calls': 0, 'automatic_retry': False, 'gate05_status': 'INCOMPLETE',
             'original_complete_model_numerical_guard': 'FAILED_UNCHANGED'}
    status_path = base / 'raw_queue_status.json'
    save(status_path, state)
    active = {'root': None, 'monitor': None, 'guard': None, 'owned': None, 'module': None,
              'collector': None, 'collector_owned': None, 'cleaned': False}

    def cleanup():
        if active['root'] is None or active['cleaned']:
            return
        root, module, monitor = active['root'], active['module'], active['monitor']
        evidence = {'status': 'RAW_QUEUE_EXCEPTION_AUTHENTICATED_CLEANUP', 'termination_actions': []}
        persist = lambda: save(root / 'queue_exception_cleanup.json', evidence)
        owned = active['owned']
        try:
            if owned is None and monitor is not None and module is not None and boot() == state['boot_id']:
                p = root / 'monitor/status.json'
                if p.is_file():
                    observed = read(p)
                    verify_monitor(observed, root, monitor.pid, state['boot_id'])
                    owned = module.OwnedGroup(observed['build_pid'], state['boot_id'], root / WRAPPER)
                else:
                    candidates = []
                    for folder in Path('/proc').iterdir():
                        if not folder.name.isdigit():
                            continue
                        child = module.process_info(int(folder.name))
                        if child is not None and child['ppid'] == monitor.pid and child['pid'] == child['pgid'] == child['sid'] and str(root / WRAPPER) in child['argv']:
                            candidates.append(child)
                    require(len(candidates) <= 1, 'Ambiguous early wrapper; refuse guessed signals')
                    if candidates:
                        evidence['early_wrapper_identity'] = candidates[0]
                        owned = module.OwnedGroup(candidates[0]['pid'], state['boot_id'], root / WRAPPER)
                active['owned'] = owned
            if owned is not None:
                evidence['wrapper_final_members'] = module.terminate_owned(owned, evidence, persist)
        except BaseException as error:
            evidence['wrapper_cleanup_error'] = repr(error)
        try:
            if active['collector_owned'] is None and active['collector'] is not None and module is not None:
                current = module.process_info(active['collector'].pid)
                if current is not None and current['state'] not in ('Z', 'X'):
                    require(current['pid'] == current['pgid'] == current['sid'] and current['ppid'] == os.getpid() and
                            str(root / 'jetson_collect_isolated_numeric_probe.py') in current['argv'],
                            'Cannot authenticate early collector private group')
                    active['collector_owned'] = module.OwnedGroup(current['pid'], state['boot_id'], root / 'jetson_collect_isolated_numeric_probe.py')
            if active['collector_owned'] is not None:
                evidence['collector_final_members'] = module.terminate_owned(active['collector_owned'], evidence, persist)
        except BaseException as error:
            evidence['collector_group_cleanup_error'] = repr(error)
        for name, grace in (('monitor', 45), ('guard', 12), ('collector', 8)):
            child = active[name]
            if child is None:
                continue
            try:
                if child.poll() is None:
                    if name != 'guard':
                        child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=grace)
                    except subprocess.TimeoutExpired:
                        evidence[name + '_grace_timeout'] = True
                        child.send_signal(signal.SIGINT)
                        try:
                            child.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait(timeout=5)
                evidence[name + '_returncode'] = child.poll()
            except BaseException as error:
                evidence[name + '_cleanup_error'] = repr(error)
        if owned is not None:
            try:
                remaining = owned.snapshot()
                evidence.update(wrapper_final_members=remaining, wrapper_group_physically_empty=not remaining)
            except BaseException as error:
                evidence['final_wrapper_snapshot_error'] = repr(error)
        active['cleaned'] = True
        persist()

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, interrupt); signal.signal(signal.SIGTERM, interrupt)
    try:
        for item, receipt in zip(plan['members'], receipts):
            require(boot() == state['boot_id'], 'Boot changed between members')
            require(sha(args.plan) == args.plan_sha256 and sha(Path(__file__)) == plan['queue_script_sha256'], 'Plan or queue source drifted')
            verify_member(item, plan, run_input_check=False)
            require(not busy_jobs(), 'Other workload active before next member')
            root = Path(item['root'])
            module = load_group_module(root)
            active.update(root=root, module=module, monitor=None, guard=None, owned=None,
                          collector=None, collector_owned=None, cleaned=False)
            write_new(root / 'queue_input_receipt.json', dict(receipt, plan_sha256=args.plan_sha256))
            row = {'configuration': item['configuration'], 'root': str(root), 'status': 'STARTING_MONITOR'}
            state['members'].append(row); state['current_configuration'] = item['configuration']; save(status_path, state)
            started = time.monotonic()
            with (root / 'monitor_console.log').open('xb', buffering=0) as console:
                active['monitor'] = subprocess.Popen([sys.executable, '-u', str(root / 'jetson_monitored_build.py'),
                      '--root', str(root), '--run', str(root / 'monitor'), '--max-seconds', '1800',
                      '--build-script', WRAPPER], cwd=str(root), stdin=subprocess.DEVNULL,
                      stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
            monitor = active['monitor']; row['monitor_pid'] = monitor.pid; save(status_path, state)
            monitor_process = module.process_info(monitor.pid)
            require(monitor_process is not None and monitor_process['pid'] == monitor_process['pgid'] == monitor_process['sid'] and
                    str(root / 'jetson_monitored_build.py') in monitor_process['argv'], 'Monitor is not our private session')
            ms = wait_complete_json(root / 'monitor/status.json', monitor, 12, IDENTITY_KEYS)
            verify_monitor(ms, root, monitor.pid, state['boot_id'])
            owned = module.OwnedGroup(ms['build_pid'], state['boot_id'], root / WRAPPER)
            active['owned'] = owned
            row.update(private_group=ms['build_pid'], wrapper_leader_start_ticks=owned.leader_start_ticks,
                       initial_wrapper_group_members=owned.initial_members, monitor_process_identity=monitor_process)
            save(status_path, state)
            with (root / 'supplemental_guard_console.log').open('xb', buffering=0) as console:
                active['guard'] = subprocess.Popen([sys.executable, '-u', str(root / 'jetson_owned_group_guard.py'),
                      '--root', str(root), '--expected-monitor-pid', str(monitor.pid)], cwd=str(root),
                      stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
            guard = active['guard']; row['guard_pid'] = guard.pid; save(status_path, state)
            gs = wait_complete_json(root / 'supplemental_group_guard.json', guard, 8,
                                    ('status', 'guard_pid', 'root', 'boot_id', 'owned_pgid', 'monitor_identity'))
            verify_guard(gs, ms, guard.pid, owned)
            guard_process = module.process_info(guard.pid)
            require(guard.poll() is None or (gs['status'] == 'COMPLETE' and not owned.snapshot()), 'Guard exited unexpectedly')
            if guard.poll() is None:
                require(guard_process is not None and guard_process['pid'] == guard_process['pgid'] == guard_process['sid'] and
                        str(root / 'jetson_owned_group_guard.py') in guard_process['argv'], 'Guard is not our private session')
            row.update(status='RUNNING', guard_process_identity=guard_process); save(status_path, state)
            while monitor.poll() is None:
                require(boot() == state['boot_id'], 'Boot changed during raw job')
                require(time.monotonic() - started <= 1800 + MONITOR_EXIT_GRACE, 'Monitor exceeded raw budget plus bounded exit grace')
                latest = read(root / 'monitor/status.json')
                verify_monitor(latest, root, monitor.pid, state['boot_id'])
                require(all(latest[k] == ms[k] for k in IDENTITY_KEYS), 'Monitor identity changed')
                guard_latest = read(root / 'supplemental_group_guard.json')
                verify_guard(guard_latest, ms, guard.pid, owned)
                require(guard.poll() is None or (guard_latest['status'] == 'COMPLETE' and not owned.snapshot()), 'Guard died before terminal')
                row['latest_monitor'] = latest; save(status_path, state); time.sleep(2)
            guard.wait(timeout=45)
            terminal, guard_terminal = read(root / 'monitor/status.json'), read(root / 'supplemental_group_guard.json')
            verify_monitor(terminal, root, monitor.pid, state['boot_id']); verify_guard(guard_terminal, ms, guard.pid, owned)
            require(monitor.returncode == 0 and terminal['status'] == 'BUILD_SCRIPT_OK' and terminal['build_returncode'] == 0 and
                    terminal.get('reason') is None and guard.returncode == 0 and guard_terminal['status'] == 'COMPLETE' and
                    guard_terminal['owned_group_fully_empty'] and not owned.snapshot(), 'Raw job or terminal group requires inspection')
            for pid in (monitor.pid, guard.pid):
                observed = module.process_info(pid)
                require(observed is None or observed['state'] in ('Z', 'X'), 'Supervisor remains live')
            require(not any(marker in (root / 'monitor/kernel.log').read_text(errors='replace') for marker in BAD_KERNEL), 'Kernel fault during raw execution')
            raw_result = verify_raw_result(root, item, plan)
            require(sha(args.plan) == args.plan_sha256 and sha(Path(__file__)) == plan['queue_script_sha256'] and
                    sha(root / 'manifest.json') == item['manifest_sha256'], 'Queue/plan/work identity drifted during execution')
            verify_manifest_inputs(root, read(root / 'manifest.json'))
            row.update(status='RAW150_COMPLETE_PENDING_COLLECTION', terminal_monitor=terminal,
                       guard_complete_no_intervention=True, final_wrapper_group_members=[], raw_result=raw_result)
            row.pop('latest_monitor', None); save(status_path, state)
            write_new(root / 'queue_terminal_receipt.json', {'configuration': item['configuration'],
                      'boot_id': state['boot_id'], 'plan_sha256': args.plan_sha256, 'monitor': terminal,
                      'guard': guard_terminal, 'physically_empty_wrapper_group': True, 'raw_result': raw_result})
            archive = Path(item['archive']); require(not archive.exists(), 'Archive appeared; no overwrite or collection retry')
            command = [sys.executable, '-B', str(root / 'jetson_collect_isolated_numeric_probe.py'),
                       '--root', str(root), '--archive', str(archive)]
            with (base / (item['configuration'] + '_collection_console.log')).open('xb', buffering=0) as console:
                active['collector'] = subprocess.Popen(command, cwd=str(root), stdin=subprocess.DEVNULL,
                      stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
                collector = active['collector']; row['collector_pid'] = collector.pid; save(status_path, state)
                collector_process = module.process_info(collector.pid)
                if collector_process is not None and collector_process['state'] not in ('Z', 'X'):
                    active['collector_owned'] = module.OwnedGroup(collector.pid, state['boot_id'], root / 'jetson_collect_isolated_numeric_probe.py')
                collector.wait(timeout=240)
            require(collector.returncode == 0, 'Collection failed or timed out; preserve evidence and stop')
            output = read(base / (item['configuration'] + '_collection_console.log'))
            require(output['path'] == str(archive) and output['sha256'] == sha(archive) and output['bytes'] == archive.stat().st_size and
                    output['files'] > 0 and boot() == state['boot_id'], 'Collected archive identity differs')
            if active['collector_owned'] is not None:
                require(not active['collector_owned'].snapshot(), 'Collector private group remains nonempty')
            require(not owned.snapshot(), 'Raw wrapper group reappeared before next configuration')
            row.update(status='SUCCESS_RAW150_TERMINAL_EMPTY_AND_COLLECTED', archive=output)
            state['completed_model_calls'] += 150; save(status_path, state)
            active.update(root=None, module=None, monitor=None, guard=None, owned=None,
                          collector=None, collector_owned=None, cleaned=False)
        require(state['completed_model_calls'] == 900 and boot() == state['boot_id'], 'Queue complete count or boot differs')
        state['status'] = 'ALL_SIX_RAW150_ATTEMPTS_TERMINAL_EMPTY_AND_COLLECTED_NO_SCORING'
        save(status_path, state)
    except BaseException as error:
        cleanup()
        state.update(status='STOPPED_REQUIRES_INSPECTION_NO_AUTO_RETRY', error_type=type(error).__name__,
                     error=str(error), traceback=traceback.format_exc())
        save(status_path, state)
        raise
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
