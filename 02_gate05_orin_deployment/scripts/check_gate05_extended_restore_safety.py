"""Offline negative checks for the new long-running queue, with no Linux/GPU access."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / 'scripts/jetson_gate05_extended_restore_queue.py'
    spec = importlib.util.spec_from_file_location('extended_restore_queue_checks', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = json.loads((args.inputs / module.INPUTS_NAME / 'plan.json').read_text(encoding='utf-8'))
    passed = []

    def ok(name, function):
        function()
        passed.append(name)

    def rejects(name, function):
        try:
            function()
        except RuntimeError:
            passed.append(name)
            return
        raise AssertionError('Unsafe case was not rejected: ' + name)

    ok('frozen_plan_accepted', lambda: module.validate_plan(plan))
    mutations = [('wrong_effective_budget_rejected', 'effective_monitor_max_seconds', 1800),
                 ('modified_inherited_spec_budget_rejected', 'inherited_spec_max_seconds', 7200),
                 ('automatic_retry_rejected', 'maximum_attempts_per_configuration', 2),
                 ('scientific_criteria_change_rejected', 'scientific_criteria_changed', True),
                 ('dataset_scope_change_rejected', 'dataset_evaluation_run', True),
                 ('missing_live_boot_requirement_rejected', 'expected_boot_id_from_required_launch_argument', False),
                 ('unbounded_monitor_cleanup_rejected', 'max_monitor_and_cleanup_seconds', 999999)]
    for name, field, value in mutations:
        altered = copy.deepcopy(plan)
        altered[field] = value
        rejects(name, lambda row=altered: module.validate_plan(row))
    altered = copy.deepcopy(plan)
    altered['configurations'][0]['bundle_root'] = '/home/nvidia/rampq/gate05_precision_r_only_G1_20261019_v2_prefer'
    rejects('old_attempt_root_rejected', lambda: module.validate_plan(altered))
    altered = copy.deepcopy(plan)
    altered['configurations'] = list(reversed(altered['configurations']))
    rejects('reordered_configurations_rejected', lambda: module.validate_plan(altered))
    for item in plan['configurations']:
        # On Windows, PurePosixPath is needed to map the frozen Linux basename.
        root = args.inputs / module.PurePosixPath(item['bundle_root']).name
        ok('unchanged_bundle_verified_' + item['configuration'], lambda row=item, path=root: module.verify_bundle(row, path))
        altered = copy.deepcopy(item)
        altered['all_original_files']['model.onnx']['bytes'] += 1
        rejects('tampered_input_receipt_rejected_' + item['configuration'],
                lambda row=altered, path=root: module.verify_bundle(row, path))

    boot = '11111111-2222-3333-4444-555555555555'
    monitor = {'pid': 100, 'state': 'S', 'ppid': 1, 'pgid': 100, 'sid': 100,
               'start_ticks': 1000, 'argv': ['python3', 'jetson_monitored_build.py']}
    bundle = Path('/home/nvidia/rampq/test_frozen_root')
    base_status = {'monitor_pid': 100, 'root': str(bundle), 'boot_id': boot, 'max_seconds': 7200,
                   'status': 'BUILD_SCRIPT_OK', 'reason': None}
    with tempfile.TemporaryDirectory(prefix='rampq_extended_queue_check_') as temporary:
        run = Path(temporary).resolve()
        status_path = run / 'status.json'
        kernel_path = run / 'kernel.log'

        def terminal(status, kernel=''):
            status_path.write_text(json.dumps(status), encoding='utf-8')
            kernel_path.write_text(kernel, encoding='utf-8')
            return module.terminal_status(run, monitor, bundle, boot)

        ok('ordinary_build_success_accepted', lambda: terminal(base_status))
        failed = dict(base_status, status='BUILD_SCRIPT_FAILED')
        ok('ordinary_build_failure_may_advance', lambda: terminal(failed))
        timed_out = dict(base_status, status='STOPPED_BY_MONITOR', reason=module.TIMEOUT_REASON)
        ok('pure_monitor_timeout_may_advance', lambda: terminal(timed_out))
        for name, altered in [('thermal_stop_rejected', dict(timed_out, reason='Preventive temperature limit reached (85 C)')),
                              ('collector_failure_rejected', dict(timed_out, reason='A diagnostic collector exited')),
                              ('unknown_stop_rejected', dict(timed_out, reason='Unknown failure')),
                              ('stale_running_status_rejected', dict(base_status, status='RUNNING')),
                              ('terminal_pid_mismatch_rejected', dict(base_status, monitor_pid=999)),
                              ('terminal_boot_mismatch_rejected', dict(base_status, boot_id='different')),
                              ('terminal_root_mismatch_rejected', dict(base_status, root='/tmp/other')),
                              ('terminal_budget_mismatch_rejected', dict(base_status, max_seconds=1800))]:
            rejects(name, lambda status=altered: terminal(status))
        rejects('kernel_usb_event_rejected', lambda: terminal(base_status, 'unregister_netdevice: waiting'))
        rejects('kernel_oom_event_rejected', lambda: terminal(base_status, 'Out of memory:'))

    collector = dict(monitor, pid=101, ppid=100, pgid=101, sid=101, start_ticks=1001, argv=['journalctl'])
    builder = dict(monitor, pid=102, ppid=100, pgid=102, sid=102, start_ticks=1002,
                   argv=['python3', 'jetson_build_precision_candidate.py'])
    trtexec = dict(monitor, pid=103, ppid=102, pgid=102, sid=102, start_ticks=1003, argv=['/usr/src/tensorrt/bin/trtexec'])
    process_table = {row['pid']: row for row in (monitor, collector, builder, trtexec)}
    tracked, groups = {}, {}
    with patch.object(module, 'processes', return_value=process_table):
        module.track_owned(monitor, tracked, groups)
    assert set(tracked) == {100, 101, 102, 103} and set(groups) == {100, 101, 102}
    passed.append('all_owned_private_sessions_and_descendants_captured')
    ok('owned_gpu_work_permitted', lambda: module.require_no_foreign_gpu_work(
        process_table, module.owned_live(process_table, tracked, groups)))
    orphan = dict(trtexec, ppid=1)
    unseen_orphan = dict(trtexec, pid=104, ppid=1, start_ticks=1004)
    orphan_table = {103: orphan, 104: unseen_orphan}
    with patch.object(module, 'processes', return_value=orphan_table):
        module.track_owned(monitor, tracked, groups)
    assert {row['pid'] for row in module.owned_live(orphan_table, tracked, groups)} == {103, 104}
    passed.append('reparented_and_new_private_group_descendants_remain_owned')
    foreign = dict(trtexec, pid=900, ppid=1, pgid=900, sid=900, start_ticks=9000)
    rejects('foreign_trtexec_blocks_next_build', lambda: module.require_no_foreign_gpu_work({900: foreign}))
    foreign_monitor = dict(monitor, pid=901, pgid=901, sid=901, start_ticks=9001)
    rejects('foreign_monitor_blocks_next_build', lambda: module.require_no_foreign_gpu_work({901: foreign_monitor}))
    foreign_smoke = dict(foreign_monitor, argv=['python3', 'jetson_precision_candidate_smoke.py'])
    rejects('foreign_smoke_blocks_next_build', lambda: module.require_no_foreign_gpu_work({901: foreign_smoke}))
    reused_monitor = dict(monitor, start_ticks=999999)
    with patch.object(module, 'processes', return_value={100: reused_monitor}):
        rejects('monitor_pid_reuse_blocks_ownership', lambda: module.track_owned(monitor, tracked, groups))
    reused_group = dict(builder, start_ticks=999999)
    with patch.object(module, 'processes', return_value={102: reused_group}):
        rejects('private_group_leader_reuse_blocks_ownership', lambda: module.track_owned(monitor, tracked, groups))
    with patch.object(module, 'require_boot'), patch.object(module, 'processes', return_value={}):
        result = module.wait_owned_exit(monitor, tracked, groups, boot, lambda *args, **kwargs: None)
    assert result['owned_processes_empty'] and result['global_trtexec_empty']
    passed.append('fully_empty_owned_and_global_state_allows_completion')
    ticks = iter([0, 41])
    with patch.object(module, 'require_boot'), patch.object(module, 'processes', return_value=orphan_table), \
            patch.object(module.time, 'monotonic', side_effect=lambda: next(ticks)):
        rejects('lingering_owned_descendants_block_progress', lambda: module.wait_owned_exit(
            monitor, tracked, groups, boot, lambda *args, **kwargs: None))
    healthy = {'MemAvailable_kib': 4 * 1024 * 1024, 'temperatures_c': {'cpu': 45.0}}
    with patch.object(module, 'require_boot'), patch.object(module, 'check_fan'), \
            patch.object(module, 'processes', return_value={}), patch.object(module, 'resources', return_value=healthy):
        ready = module.check_ready(boot)
        assert ready['global_trtexec_empty'] and ready['other_gpu_work_empty']
        passed.append('healthy_idle_readiness_accepted')
        with patch.object(module, 'resources', return_value=dict(healthy, MemAvailable_kib=3 * 1024 * 1024 - 1)):
            rejects('prebuild_less_than_3gib_rejected', lambda: module.check_ready(boot))
        with patch.object(module, 'resources', return_value=dict(healthy, temperatures_c={'cpu': 85.0})):
            rejects('prebuild_85c_rejected', lambda: module.check_ready(boot))
    result = {'status': 'OFFLINE_SAFETY_BRANCH_CHECKS_PASSED', 'queue_source': str(source),
              'queue_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'check_count': len(passed), 'checks': passed,
              'native_board_access': False, 'gpu_or_subprocess_started': False,
              'limits': 'Deterministic mocked orchestration and frozen-file negative checks only; '
                        'not proof that OS/hardware failures can always be recovered.'}
    assert not args.output.exists(), 'Preserve prior verification results'
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
