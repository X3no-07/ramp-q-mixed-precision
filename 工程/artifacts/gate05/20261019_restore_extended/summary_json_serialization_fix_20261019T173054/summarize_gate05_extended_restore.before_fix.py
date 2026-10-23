"""Verify collected v3 extended-restore evidence locally; never contact the board.

The input spec remains byte-identical to v2_prefer with an inherited 1800 s
budget. The external v3 plan, launch and monitor must independently establish
the effective 7200 s budget. Missing evidence is PENDING; contradictions require
review. A complete engineering checkpoint still leaves Gate 0.5 INCOMPLETE.
"""
import argparse
from datetime import datetime
import math
from pathlib import Path, PurePosixPath
import re

from jetson_gate05_extended_restore_queue import (COUNTS, ORDER, validate_plan,
                                                  verify_bundle as verify_extended_bundle)
from jetson_precision_candidate_smoke import binding_contract
from summarize_gate05_prefer_matrix import (PendingEvidence, precision_state, read,
                                            receipt_state, require, required_file,
                                            resource_summary, sha, smoke_state)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'rampq.gate05.extended_restore_checkpoint/v1'
INPUTS_NAME = 'gate05_restore_extended_inputs_20261019'
SCRIPT_NAME = 'jetson_gate05_extended_restore_queue.py'
QUEUE_ROOT = '/home/nvidia/rampq/gate05_restore_extended_queue_20261019'
KERNEL_MARKERS = ('controller error detected', 'HC died;', 'unregister_netdevice: waiting',
                  'Out of memory:', 'oom-kill:', 'Kernel panic')
ERRORS = (OSError, ValueError, KeyError, TypeError, AssertionError, RuntimeError)
BOOT = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')


def original_sources(root, bundle_root, prepared_inputs):
    """Pin collected originals to the already frozen client v3 input package."""
    result = {'status': 'PENDING', 'pending': [], 'errors': [], 'source_files': {},
              'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200}
    plan, specs = None, {}
    try:
        plan = read(prepared_inputs / 'plan.json')
        validate_plan(plan)
        require(sha(prepared_inputs / SCRIPT_NAME) == plan['queue_script_sha256'], 'Prepared v3 queue script differs')
        require(sha(prepared_inputs / 'budget_revision.json') == plan['budget_revision_sha256'], 'Prepared v3 budget differs')
        revision = read(prepared_inputs / 'budget_revision.json')
        require(revision['inherited_spec_max_seconds'] == 1800 and revision['effective_monitor_max_seconds'] == 7200 and
                revision['maximum_attempts_per_configuration'] == 1 and revision['original_inputs_byte_identical'] is True and
                revision['scientific_criteria_changed'] is False and revision['actual_precision_audit_criteria_changed'] is False and
                revision['original_numerical_guard'] == 'FAILED_UNCHANGED' and revision['dataset_evaluation_run'] is False,
                'Prepared v3 budget/scope policy differs')
        result['frozen_bundles'] = []
        for item in plan['configurations']:
            bundle = bundle_root / item['configuration']
            result['frozen_bundles'].append(verify_extended_bundle(item, bundle))
            specs[item['configuration']] = read(bundle / 'spec.json')
        for name in ('plan.json', 'budget_revision.json', SCRIPT_NAME):
            expected = prepared_inputs / name
            collected = root / 'original_inputs' / name
            result['source_files'][name] = {'prepared_sha256': sha(expected),
                                           'collected_path': str(collected), 'status': 'PENDING'}
            if not collected.is_file():
                result['pending'].append('original_inputs/' + name)
                continue
            require(not collected.is_symlink() and sha(collected) == sha(expected),
                    'Collected v3 original differs from frozen input package: ' + name)
            result['source_files'][name].update(status='VERIFIED_IDENTICAL', collected_sha256=sha(collected))
        result.update(plan_sha256=sha(prepared_inputs / 'plan.json'),
                      queue_script_sha256=plan['queue_script_sha256'], budget_revision_sha256=plan['budget_revision_sha256'],
                      original_numerical_guard='FAILED_UNCHANGED', scientific_criteria_changed=False)
        result['status'] = 'PENDING' if result['pending'] else 'V3_ORIGINAL_INPUTS_AND_FROZEN_BUNDLES_VERIFIED'
    except ERRORS as error:
        result.update(status='INVALID_OR_INCONSISTENT_EVIDENCE')
        result['errors'].append(type(error).__name__ + ': ' + str(error))
    return result, plan, specs


def readiness(value, label):
    require(value['global_trtexec_empty'] is True and value['other_gpu_work_empty'] is True,
            label + ' did not confirm no other GPU work')
    sample = value['resources']
    require(sample['MemAvailable_kib'] >= 3 * 1024 * 1024, label + ' had less than 3 GiB available')
    require(sample['temperatures_c'] and max(sample['temperatures_c'].values()) < 85,
            label + ' has absent/unsafe temperature measurements')


def monitor_command(root, config):
    return ['-u', root + '/jetson_monitored_build.py', '--root', root,
            '--run', QUEUE_ROOT + '/monitor_' + config, '--max-seconds', '7200',
            '--build-script', 'jetson_build_precision_candidate.py']


def verify_monitor_process(value, pid, command=None):
    require(value['pid'] == pid and type(value['start_ticks']) is int and value['start_ticks'] > 0 and
            value['pgid'] == pid and value['sid'] == pid, 'Captured monitor private process identity differs')
    if command is not None:
        require(value['argv'] == command, 'Captured monitor argv differs from recorded launch')


def check_terminal_monitor(status, item, boot, pid):
    root = item['bundle_root']
    require(status['monitor_pid'] == pid and status['root'] == root and status['boot_id'] == boot and
            status['max_seconds'] == 7200 and status['build_script'] == root + '/jetson_build_precision_candidate.py',
            'v3 monitor identity, root, build script or effective 7200 s budget differs')
    require(status['stop_temperature_c'] == 85 and
            status['low_memory_stop'] == 'MemAvailable < 384 MiB for 3 consecutive 2-second samples',
            'Original monitor resource guards differ')
    state = status['status']
    require(state in ('RUNNING', 'BUILD_SCRIPT_OK', 'BUILD_SCRIPT_FAILED', 'STOPPED_BY_MONITOR', 'FAILED'),
            'Unknown monitor state')
    if state == 'RUNNING':
        return 'PENDING'
    require(math.isfinite(status['elapsed_seconds']) and 0 <= status['elapsed_seconds'] <= 7380,
            'Monitor duration is outside the external bounded attempt')
    if state in ('BUILD_SCRIPT_OK', 'BUILD_SCRIPT_FAILED'):
        require(status.get('reason') is None, 'Ordinary build outcome carries a monitor failure reason')
        require((status['build_returncode'] == 0) is (state == 'BUILD_SCRIPT_OK'),
                'Monitor state contradicts builder return code')
        return 'BUILT' if state == 'BUILD_SCRIPT_OK' else 'BUILD_FAILED'
    if state == 'STOPPED_BY_MONITOR' and status.get('reason') == 'Build supervision time limit reached':
        require(status['elapsed_seconds'] >= 7200, 'Timeout record predates the effective 7200 s budget')
        return 'TIMEOUT_INCONCLUSIVE'
    return 'MONITOR_SAFETY_OR_UNKNOWN_FAILURE_REQUIRES_REVIEW'


def queue_state(root, sources, plan, bundle_root):
    folder = root / 'queue_evidence'
    receipt = receipt_state(folder)
    result = {'status': 'PENDING', 'receipt': receipt, 'pending': [], 'errors': [],
              'terminal_snapshot_verified': False, 'completed_build_identities': {}}
    if receipt['status'] != 'VERIFIED':
        result['status'] = receipt['status']
        if receipt['status'] == 'INVALID_EVIDENCE':
            result['errors'].append(receipt['reason'])
        return result
    try:
        require(plan is not None, 'No verified prepared v3 plan available')
        recorded_plan_path = required_file(folder, receipt, 'queue/plan.json')
        require(sha(recorded_plan_path) == sources['plan_sha256'] and read(recorded_plan_path) == plan,
                'Collected queue plan differs from the frozen external v3 plan')
        request_path = required_file(folder, receipt, 'queue/request.json')
        request = read(request_path)
        require(request['schema'] == 'rampq.gate05.extended_restore_queue_request/v1', 'Queue request schema differs')
        require(request['original_plan_sha256'] == sources['plan_sha256'] and
                request['queue_script_sha256'] == sources['queue_script_sha256'] and
                request['budget_revision_sha256'] == sources['budget_revision_sha256'], 'Queue execution source lineage differs')
        require(request['inherited_spec_max_seconds'] == 1800 and request['effective_monitor_max_seconds'] == 7200 and
                request['inference_run'] is False and request['dataset_evaluation_run'] is False and
                request['native_precision_pass_claim'] is False, 'Queue budget/scope differs')
        boot, pid = request['boot_id'], request['queue_pid']
        require(BOOT.fullmatch(boot) and type(pid) is int and pid > 1, 'Queue boot/PID identity is invalid')
        process = request['queue_process']
        require(process['pid'] == pid and type(process['start_ticks']) is int and process['start_ticks'] > 0,
                'Queue process identity differs')
        argv = process['argv']
        require(plan['inputs_root'] + '/' + SCRIPT_NAME in argv and '--expected-boot-id' in argv and
                argv[argv.index('--expected-boot-id') + 1] == boot, 'Queue was not launched with its recorded live boot UUID')
        readiness(request['initial_ready'], 'Queue initial readiness')
        require([row['configuration'] for row in request['bundles']] == list(ORDER), 'Queue bundle order differs')
        for item, observed in zip(plan['configurations'], request['bundles']):
            bundle = bundle_root / item['configuration']
            require(observed['bundle_root'] == item['bundle_root'] and observed['verified_at'] == item['bundle_root'] and
                    observed['manifest_sha256'] == item['bundle_manifest_sha256'] and
                    observed['spec_sha256'] == sha(bundle / 'spec.json') and observed['graph_sha256'] == sha(bundle / 'model.onnx') and
                    observed['original_file_count'] == len(item['all_original_files']) and
                    observed['inherited_spec_max_seconds'] == 1800 and observed['effective_monitor_max_seconds'] == 7200 and
                    observed['original_numerical_failure'] == 'FAILED_UNCHANGED' and observed['actual_precision_audit'] == 'NOT_RUN',
                    'Queue frozen bundle verification metadata differs')
        state = read(required_file(folder, receipt, 'queue/status.json'))
        require(all(state[key] == value for key, value in request.items()), 'Queue snapshot changes its original request metadata')
        completed = state['completed_builds']
        names = [row['configuration'] for row in completed]
        require(names == list(ORDER[:len(names)]) and len(names) <= 3, 'Completed queue results are not an ordered unique prefix')
        result.update(queue_pid=pid, queue_process_start_ticks=process['start_ticks'], boot_id=boot,
                      queue_request_sha256=sha(request_path), snapshot_status=state['status'],
                      completed_configurations=names, native_live_pid_observed_by_this_script=False,
                      inherited_spec_max_seconds=1800, effective_monitor_max_seconds=7200)
        for item in plan['configurations']:
            config = item['configuration']
            launch_name = 'queue/launch_' + config + '.json'
            if launch_name not in receipt['verified_files']:
                if config in names:
                    raise PendingEvidence('Completed queue result lacks a receipt-covered launch: ' + config)
                continue
            launch = read(required_file(folder, receipt, launch_name))
            command = launch['command']
            require(command[1:] == monitor_command(item['bundle_root'], config), 'v3 monitor launch command differs')
            require(launch['configuration'] == config and launch['boot_id'] == boot and
                    launch['inherited_spec_max_seconds'] == 1800 and launch['effective_monitor_max_seconds'] == 7200,
                    'v3 launch identity/budget differs')
            monitor_process = launch['monitor_process']
            verify_monitor_process(monitor_process, monitor_process['pid'], command)
            monitor_path = required_file(folder, receipt, 'queue/monitor_' + config + '/status.json')
            terminal = read(monitor_path)
            outcome = check_terminal_monitor(terminal, item, boot, monitor_process['pid'])
            if config not in names:
                require(state['status'] != 'EXTENDED_RESTORE_QUEUE_FINISHED_AUDIT_AND_SMOKE_PENDING',
                        'Finished queue omits a launched configuration')
                result['pending'].append(config + ' has no completed queue result; monitor outcome ' + outcome)
                continue
            result_name = 'queue/result_' + config + '.json'
            recorded = read(required_file(folder, receipt, result_name))
            embedded = completed[names.index(config)]
            require(recorded == embedded, 'Per-item result differs from the queue terminal snapshot')
            require(recorded['monitor_pid'] == monitor_process['pid'] and recorded['monitor_process'] == monitor_process and
                    recorded['monitor_returncode'] == 0 and recorded['monitor_run'] == QUEUE_ROOT + '/monitor_' + config and
                    recorded['terminal_monitor_status'] == terminal and
                    recorded['terminal_monitor_status_sha256'] == sha(monitor_path) and
                    recorded['inherited_spec_max_seconds'] == 1800 and recorded['effective_monitor_max_seconds'] == 7200,
                    'Per-item queue/launch/terminal identities differ')
            require(outcome in ('BUILT', 'BUILD_FAILED', 'TIMEOUT_INCONCLUSIVE'),
                    'Queue incorrectly advanced after a safety/collector/unknown failure')
            termination = recorded['process_termination']
            require(termination['owned_processes_empty'] is True and termination['global_trtexec_empty'] is True,
                    'Queue item did not confirm owned descendant/global trtexec termination')
            tracked = termination['tracked_process_identities']
            require(any(row['pid'] == monitor_process['pid'] and row['start_ticks'] == monitor_process['start_ticks']
                        for row in tracked), 'Termination record lacks the same monitor PID/start_ticks')
            readiness(recorded['initial_ready'], config + ' initial readiness')
            readiness(recorded['after_termination_ready'], config + ' after-termination readiness')
            require(recorded['native_precision_audit'] == 'NOT_RUN' and recorded['fixed_image_execution'] == 'NOT_RUN',
                    'Build queue claims an audit/inference it does not execute')
            identity = {'monitor_pid': monitor_process['pid'], 'native_root': item['bundle_root'], 'boot_id': boot,
                        'monitor_process_start_ticks': monitor_process['start_ticks'],
                        'monitor_status_sha256': sha(monitor_path), 'monitor_outcome': outcome,
                        'build_result_sha256': recorded.get('build_result_sha256'),
                        'terminal_monitor_status': terminal, 'owned_and_global_processes_reported_ended': True}
            result['completed_build_identities'][config] = identity
        if state['status'] == 'EXTENDED_RESTORE_QUEUE_FINISHED_AUDIT_AND_SMOKE_PENDING':
            require(names == list(ORDER) and len(result['completed_build_identities']) == 3,
                    'Terminal queue is missing a complete validated result')
            result.update(status='TERMINAL_V3_QUEUE_SNAPSHOT_VERIFIED', terminal_snapshot_verified=True)
        elif state['status'] == 'QUEUE_STOPPED_REQUIRES_INSPECTION':
            result.update(status='QUEUE_STOP_REQUIRES_REVIEW', terminal_snapshot_verified=True, reason=state.get('error'))
        else:
            result['pending'].append('Queue snapshot is not terminal; offline data does not prove a live process')
    except PendingEvidence as error:
        result['pending'].append(str(error))
    except ERRORS as error:
        result.update(status='INVALID_OR_INCONSISTENT_EVIDENCE')
        result['errors'].append(type(error).__name__ + ': ' + str(error))
    return result


def expected_build_command(root, run, spec):
    command = ['/usr/src/tensorrt/bin/trtexec', '--onnx=' + root + '/model.onnx',
               '--saveEngine=' + run + '/model.engine', '--fp16', '--noTF32', '--memPoolSize=workspace:1024',
               '--profilingVerbosity=detailed', '--buildOnly', '--avgTiming=1', '--precisionConstraints=prefer',
               '--inputIOFormats=fp32:chw', '--outputIOFormats=fp32:chw,fp32:chw', '--int8']
    constraints = ','.join(row['onnx_node'] + ':fp16' for row in spec['requests'] if row['precision'] == 'FP16')
    return command + ['--layerPrecisions=' + constraints, '--layerOutputTypes=' + constraints]


def check_build(folder, receipt, bundle, spec, item):
    monitor_path = required_file(folder, receipt, 'monitor/status.json')
    monitor = read(monitor_path)
    boot, pid = monitor['boot_id'], monitor['monitor_pid']
    require(BOOT.fullmatch(boot) and type(pid) is int and pid > 1, 'Monitor boot/PID is invalid')
    outcome = check_terminal_monitor(monitor, item, boot, pid)
    details = {'status': 'PENDING', 'terminal_snapshot_verified': outcome != 'PENDING',
               'monitor_status': monitor['status'], 'monitor_reason': monitor.get('reason'),
               'monitor_elapsed_seconds': monitor.get('elapsed_seconds'), 'native_root': monitor['root'],
               'boot_id': boot, 'monitor_pid': pid, 'monitor_sha256': sha(monitor_path),
               'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200,
               'native_live_pid_observed_by_this_script': False}
    if outcome == 'PENDING':
        details['reason'] = 'Retained monitor snapshot remains RUNNING; not live process proof'
        return details, None
    launch = read(required_file(folder, receipt, 'monitor/request.json'))
    for key in ('monitor_pid', 'boot_id', 'root', 'max_seconds', 'stop_temperature_c', 'build_script', 'low_memory_stop'):
        require(launch[key] == monitor[key], 'Monitor initial request and terminal status differ: ' + key)
    require(launch['initial_resources']['MemAvailable_kib'] >= 3 * 1024 * 1024 and
            launch['initial_resources']['temperatures_c'] and
            max(launch['initial_resources']['temperatures_c'].values()) < 85,
            'Monitor initial resources were not ready')
    manifest = read(bundle / 'manifest.json')
    for name in ['manifest.json'] + [name for name in manifest['files'] if Path(name).suffix in ('.json', '.py')]:
        require(sha(required_file(folder, receipt, 'inputs/' + name)) == sha(bundle / name),
                'Collected source input differs from the original prefer bundle: ' + name)
    request_path = required_file(folder, receipt, 'build/build_request.json')
    request = read(request_path)
    command = request['command']
    outputs = [value.split('=', 1)[1] for value in command if value.startswith('--saveEngine=')]
    require(len(outputs) == 1, 'Native command lacks a unique engine output path')
    engine_path = PurePosixPath(outputs[0])
    run = engine_path.parent
    require(engine_path.name == 'model.engine' and run.parent == PurePosixPath(item['bundle_root']) / 'results' and
            re.fullmatch(r'build_[0-9]{8}T[0-9]{6}_[0-9]{6}Z', run.name), 'Native build result escaped the fresh v3 root')
    require(command == expected_build_command(item['bundle_root'], str(run), spec), 'Exact frozen trtexec command differs')
    lineage = {'configuration': spec['configuration'], 'onnx_sha256': spec['graph_sha256'],
               'spec_sha256': sha(bundle / 'spec.json'), 'manifest_sha256': sha(bundle / 'manifest.json'),
               'graph_audit_sha256': spec['graph_audit_sha256'], 'module_map_sha256': spec['module_map_sha256'],
               'requested_precision_constraints_mode': 'prefer', 'diagnostic_only': True,
               'local_preflight_status': 'LOCAL_QDQ_PREFLIGHT_FAILED', 'primary_output_name': spec['primary_output_name']}
    require(all(request[key] == expected for key, expected in lineage.items()), 'Build request/source lineage differs')
    require(binding_contract(request['expected_binding_contract']) == binding_contract(spec['expected_binding_contract']) and
            request['restored_fp16_compute_and_output_constraints'] ==
            [row['onnx_node'] for row in spec['requests'] if row['precision'] == 'FP16'] and
            request['precision_request_is_not_actual_adoption'] is True, 'Build request binding/precision scope differs')
    environment = read(required_file(folder, receipt, 'build/environment.json'))
    require(environment['boot_id'] == boot, 'Build environment and monitor boot IDs differ')
    details.update(build_request_sha256=sha(request_path), native_build_run=str(run))
    kernel = required_file(folder, receipt, 'monitor/kernel.log')
    details['kernel_safety_event_lines'] = [line for line in kernel.read_text(errors='replace').splitlines()
                                           if any(marker in line for marker in KERNEL_MARKERS)]
    details['resources'] = resource_summary(required_file(folder, receipt, 'monitor/resources.jsonl'))
    require(details['resources']['samples'] > 0, 'No monitor resource samples were retained')
    build_path = folder / 'build/build_result.json'
    build = read(required_file(folder, receipt, 'build/build_result.json')) if build_path.is_file() else None
    if build is not None:
        require(all(build[key] == expected for key, expected in lineage.items()), 'Build result/source lineage differs')
        details.update(build_result_sha256=sha(build_path), build_result_status=build['status'])
    if details['kernel_safety_event_lines'] or outcome == 'MONITOR_SAFETY_OR_UNKNOWN_FAILURE_REQUIRES_REVIEW':
        details['status'] = 'MONITOR_SAFETY_OR_KERNEL_EVENT_REQUIRES_REVIEW'
    elif outcome == 'TIMEOUT_INCONCLUSIVE':
        require(build is None or build['status'] != 'BUILT_AND_DESERIALIZED', 'Timeout monitor contradicts completed builder result')
        details.update(status='TIMEOUT_INCONCLUSIVE', stale_running_build_metadata_is_not_live_process_proof=True)
    elif outcome == 'BUILD_FAILED':
        require(build is None or build['status'] != 'BUILT_AND_DESERIALIZED', 'Failed monitor contradicts completed builder result')
        details['status'] = 'BUILD_FAILED' if build is not None else 'BUILD_FAILED_RESULT_PENDING'
    else:
        if build is None:
            details['reason'] = 'Successful monitor lacks a receipt-covered build_result.json'
            return details, None
        require(build['status'] == 'BUILT_AND_DESERIALIZED', 'Successful monitor contradicts builder outcome')
        engine = required_file(folder, receipt, 'build/model.engine')
        layers = required_file(folder, receipt, 'build/engine_layers.json')
        require(sha(engine) == build['engine_sha256'] and engine.stat().st_size == build['engine_bytes'], 'Engine/result integrity differs')
        require(binding_contract(build['bindings']) == binding_contract(spec['expected_binding_contract']) and
                binding_contract(build['expected_binding_contract']) == binding_contract(spec['expected_binding_contract']) and
                all(row.get('format') == 'TensorFormat.LINEAR' for row in build['bindings']), 'Engine external named bindings differ')
        parsed = read(required_file(folder, receipt, 'build/parsed_network_validation.json'))
        require(parsed['status'] == 'ALL_REQUESTED_CONV_NAMES_EXIST_IN_PARSED_NETWORK' and parsed['source_conv_count'] == 88 and
                len(parsed['requests']) == 88 and {row['onnx_node'] for row in parsed['requests']} ==
                {row['onnx_node'] for row in spec['requests']}, 'Imported Conv addressing inventory differs')
        for row in parsed['requests']:
            original = next(request for request in spec['requests'] if request['onnx_node'] == row['onnx_node'])
            require(all(row[key] == value for key, value in original.items()), 'Imported source precision request differs')
        details.update(status='BUILT_AND_DESERIALIZED_INTEGRITY_VERIFIED', engine_sha256=sha(engine),
                       engine_bytes=engine.stat().st_size, layer_file_sha256=sha(layers))
    details['engine_file_present'] = (folder / 'build/model.engine').is_file()
    return details, build


def extended_smoke(folder, receipt, bundle, spec, build, precision):
    require(precision['requested_conv_formats_complete'] is True, 'Fixed-image execution requires complete actual formats')
    result = smoke_state(folder, receipt, bundle, spec, build, precision)
    if result['status'] == 'PENDING':
        return result
    report = read(folder / 'smoke/smoke_result.json')
    lineage = {'onnx_sha256': spec['graph_sha256'], 'bundle_manifest_sha256': sha(bundle / 'manifest.json'),
               'graph_audit_sha256': spec['graph_audit_sha256'], 'module_map_sha256': spec['module_map_sha256'],
               'failed_preflight_sha256': spec['failed_preflight_sha256'], 'diagnostic_scope_sha256': spec['diagnostic_scope_sha256']}
    require(all(report['lineage'][key] == expected for key, expected in lineage.items()), 'Fixed-image full source lineage differs')
    require(report['all_requested_conv_formats_verified'] is True and report['diagnostic_only'] is True and
            report['precision_audit_status'] == precision['status'] and report['strict_int8_status'] == precision['strict_int8_status'],
            'Smoke actual-format/strict-tactic labels differ from the validated audit')
    require(report['preprocess'] == {'imgsz': 640, 'batch': 1, 'rect': False, 'bgr_to_rgb': True,
                                     'scale': 'float32 / 255.0', 'letterbox_auto': False, 'stride': 32} and
            report['image'] == '000020.jpg' and report['class_count'] == 10 and
            report['primary_output_name'] == spec['primary_output_name'] and report['primary_output_npz_key'] == 'tensorrt',
            'Fixed-image input/preprocessing/output contract differs')
    require(report['numerical_migration_status'] == 'ORIGINAL_LOCAL_FULL_MODEL_PYTORCH_ORT_QDQ_PARITY_FAILED_RETAINED',
            'Original numerical migration failure was not retained')
    result.update(boot_id_of_execution=report['boot_id'], board_time_of_execution=report['board_time'],
                  inherited_spec_max_seconds=1800, effective_build_monitor_max_seconds=7200)
    return result


def precision_metadata(audit):
    """Cross-check counters/status against the unchanged auditor's per-Conv rows."""
    quantized = [row for row in audit['mapping'] if row['precision'] == 'INT8']
    floating = [row for row in audit['mapping'] if row['precision'] == 'FP16']
    require(all(type(row['requested_formats_verified']) is bool and type(row['strict_int8_tactic_verified']) is bool
                for row in audit['mapping']), 'Per-Conv format/tactic evidence must be explicit booleans')
    complete = not audit['errors'] and all(row['requested_formats_verified'] for row in audit['mapping'])
    strict = complete and all(row['strict_int8_tactic_verified'] for row in quantized)
    require(audit['requested_conv_formats_complete'] is complete and
            audit['verified_int8_format_convs'] == sum(row['requested_formats_verified'] for row in quantized) and
            audit['verified_fp16_format_convs'] == sum(row['requested_formats_verified'] for row in floating) and
            audit['named_integer_tactic_source_convs'] == sum(row['strict_int8_tactic_verified'] for row in quantized) and
            audit['source_convs_without_tactic_name'] ==
            sum(row.get('integer_tactic_evidence') == 'TACTIC_NAME_NOT_EXPOSED' for row in quantized),
            'Precision format/tactic counters contradict their per-Conv evidence')
    expected = 'ALL_REQUESTED_INT8_TACTICS_VERIFIED' if strict else 'INTEGER_TACTIC_REVIEW_REQUIRED'
    require(audit['strict_int8_status'] == expected, 'Strict INT8 label contradicts the frozen per-Conv criteria')
    require(audit['original_numerical_failure_retained'] is True, 'Precision audit lost the original numerical failure')


def config_state(root, bundle_root, config, spec, item):
    row = {'configuration': config, 'expected_int8': COUNTS[config][0], 'expected_fp16': COUNTS[config][1],
           'status': 'PENDING', 'pending': [], 'errors': [], 'checkpoint_evidence_complete': False,
           'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200}
    try:
        if spec is None or item is None:
            raise PendingEvidence('A verified frozen v3 plan and original prefer spec are required')
        bundle = bundle_root / config
        row['bundle_manifest_sha256'] = sha(bundle / 'manifest.json')
        first_folder, post_folder = root / config / 'build_evidence', root / config / 'post_smoke_evidence'
        first, post = receipt_state(first_folder), receipt_state(post_folder)
        row.update(build_receipt=first, post_smoke_receipt=post)
        require(first['status'] != 'INVALID_EVIDENCE' and post['status'] != 'INVALID_EVIDENCE',
                'Invalid collected receipt: ' + str(first.get('reason') or post.get('reason')))
        if first['status'] != 'VERIFIED':
            raise PendingEvidence('Verified build_evidence receipt')
        details, build = check_build(first_folder, first, bundle, spec, item)
        row['build'] = details
        if details['status'] == 'PENDING' or details['status'] == 'BUILD_FAILED_RESULT_PENDING':
            raise PendingEvidence(details.get('reason', 'Receipt-covered terminal builder result'))
        if details['status'] != 'BUILT_AND_DESERIALIZED_INTEGRITY_VERIFIED':
            require('smoke/smoke_result.json' not in first['verified_files'] and
                    'smoke/smoke_result.json' not in post['verified_files'], 'Smoke exists without a verified complete engine')
            row.update(status=details['status'], precision={'status': 'NOT_ESTABLISHED_NO_VERIFIED_COMPLETE_ENGINE'},
                       smoke={'status': 'NOT_RUN_NO_VERIFIED_COMPLETE_ENGINE'})
            row['checkpoint_evidence_complete'] = details['status'] in ('BUILD_FAILED', 'TIMEOUT_INCONCLUSIVE')
            return row
        folder, receipt = (post_folder, post) if post['status'] == 'VERIFIED' else (first_folder, first)
        if folder == post_folder:
            later, later_build = check_build(folder, receipt, bundle, spec, item)
            require(later['status'] == details['status'] and later['monitor_sha256'] == details['monitor_sha256'] and
                    later['build_result_sha256'] == details['build_result_sha256'] and
                    later_build['engine_sha256'] == build['engine_sha256'], 'Post-smoke build differs from the first collected build')
            build = later_build
        precision, audit = precision_state(folder, receipt, bundle, spec, build)
        row['precision'] = precision
        if precision['status'] == 'PENDING':
            raise PendingEvidence(precision['reason'])
        precision_metadata(audit)
        if not precision['requested_conv_formats_complete']:
            require('smoke/smoke_result.json' not in first['verified_files'] and
                    'smoke/smoke_result.json' not in post['verified_files'], 'Smoke performed while requested actual formats remain incomplete')
            row.update(status='REQUESTED_FORMATS_NOT_ESTABLISHED',
                       smoke={'status': 'NOT_PERMITTED_REQUESTED_FORMATS_INCOMPLETE'}, checkpoint_evidence_complete=True)
            return row
        smoke = extended_smoke(folder, receipt, bundle, spec, build, precision)
        row['smoke'] = smoke
        if smoke['status'] == 'PENDING':
            raise PendingEvidence(smoke['reason'])
        row.update(status=('STRICT_INT8_TACTICS_AND_FIXED_IMAGE_EVIDENCE_VERIFIED' if precision['strict_int8_tactics_verified'] else
                           'FORMATS_AND_DIAGNOSTIC_FIXED_IMAGE_VERIFIED_INTEGER_TACTICS_UNRESOLVED'), checkpoint_evidence_complete=True)
    except PendingEvidence as error:
        row['pending'].append(str(error))
    except ERRORS as error:
        row.update(status='INVALID_OR_INCONSISTENT_EVIDENCE', checkpoint_evidence_complete=False)
        row['errors'].append(type(error).__name__ + ': ' + str(error))
    return row


def summarize(root, bundle_root, prepared_inputs):
    sources, plan, specs = original_sources(root, bundle_root, prepared_inputs)
    queue = queue_state(root, sources, plan, bundle_root)
    items = {row['configuration']: row for row in plan['configurations']} if plan else {}
    rows = [config_state(root, bundle_root, config, specs.get(config), items.get(config)) for config in ORDER]
    for row in rows:
        build = row.get('build')
        if not build or build['status'] == 'PENDING':
            continue
        identity = queue['completed_build_identities'].get(row['configuration'])
        if identity is None:
            row['pending'].append('Receipt-covered completed queue result and monitor launch cross-check')
            row['checkpoint_evidence_complete'] = False
            continue
        mismatches = [key for key in ('monitor_pid', 'native_root', 'boot_id') if build[key] != identity[key]]
        if build['monitor_sha256'] != identity['monitor_status_sha256']:
            mismatches.append('monitor_status_sha256')
        if identity.get('build_result_sha256') is not None and build.get('build_result_sha256') != identity['build_result_sha256']:
            mismatches.append('build_result_sha256')
        if mismatches:
            row.update(status='INVALID_OR_INCONSISTENT_EVIDENCE', checkpoint_evidence_complete=False)
            row['errors'].append('Per-config collected identity differs from v3 queue result: ' + ', '.join(mismatches))
    invalid = bool(sources['errors'] or queue['errors'] or any(row['errors'] or 'REQUIRES_REVIEW' in row['status'] for row in rows))
    invalid = invalid or queue['status'] == 'QUEUE_STOP_REQUIRES_REVIEW'
    complete = (not invalid and sources['status'] == 'V3_ORIGINAL_INPUTS_AND_FROZEN_BUNDLES_VERIFIED' and
                queue['status'] == 'TERMINAL_V3_QUEUE_SNAPSHOT_VERIFIED' and all(row['checkpoint_evidence_complete'] for row in rows))
    return {'schema': SCHEMA, 'client_at': datetime.now().astimezone().isoformat(),
            'status': ('EXTENDED_RESTORE_EVIDENCE_REVIEW_REQUIRED' if invalid else
                       'EXTENDED_RESTORE_LOCAL_CHECKPOINT_COMPLETE' if complete else 'EXTENDED_RESTORE_EVIDENCE_PENDING'),
            'gate05_status': 'INCOMPLETE', 'root': str(root.resolve()), 'bundle_root': str(bundle_root.resolve()),
            'prepared_inputs': str(prepared_inputs.resolve()), 'original_sources': sources, 'queue': queue,
            'configurations': rows, 'local_checkpoint_evidence_complete': complete,
            'inherited_spec_max_seconds': 1800, 'effective_monitor_max_seconds': 7200,
            'all_three_terminal_build_snapshots_verified': all(row.get('build', {}).get('terminal_snapshot_verified') is True for row in rows),
            'format_complete_configurations': [row['configuration'] for row in rows if row.get('precision', {}).get('requested_conv_formats_complete')],
            'strict_int8_tactics_verified_configurations': [row['configuration'] for row in rows if row.get('precision', {}).get('strict_int8_tactics_verified')],
            'formats_only_configurations': [row['configuration'] for row in rows if row.get('precision', {}).get('formats_only')],
            'verified_saved_fixed_image_executions': sum(row.get('smoke', {}).get('status') == 'VERIFIED_FIXED_IMAGE_EXECUTION_ONLY' for row in rows),
            'original_q_all_numerical_guard': 'FAILED_UNCHANGED', 'dataset_acceptance_rules_status': 'NOT_ASSESSED_BY_THIS_SCRIPT',
            'native_process_live_state_observed_by_this_script': False, 'inference_performed_by_this_script': False,
            'native_layer_audit_recomputed_by_this_script': False, 'dataset_evaluation_performed_by_this_script': False,
            'causal_or_target_rescue_claim_established': False,
            'verification_helper_sources': {name: sha(ROOT / 'scripts' / name) for name in
                                            ('summarize_gate05_prefer_matrix.py', 'jetson_precision_candidate_smoke.py',
                                             'jetson_gate05_extended_restore_queue.py', 'verify_gate05_evidence_archive.py')},
            'scope': 'Three existing frozen restore configurations with one externally supervised 7200-second attempt each. '
                     'The original spec remains 1800 seconds. Collected identities, formats and strict integer tactics are '
                     'separate from build/inference success. This checkpoint does not establish numerical migration, '
                     'Gate 0.5, dataset accuracy, target rescue, causal attribution, performance or current live process state.'}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'artifacts/gate05/20261019_restore_extended')
    parser.add_argument('--bundle-root', type=Path, default=ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles_prefer')
    parser.add_argument('--prepared-inputs', type=Path,
                        default=ROOT / 'artifacts/gate05/20261019_restore_extended_input_v3' / INPUTS_NAME)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root / 'checkpoint.json'
    if output.exists():
        require(read(output).get('schema') == SCHEMA, 'Refuse to overwrite unrelated/original evidence')
    report = summarize(args.root, args.bundle_root, args.prepared_inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({key: value for key, value in report.items() if key not in ('original_sources', 'queue', 'configurations')}, indent=2))


if __name__ == '__main__':
    main()
