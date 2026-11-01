"""Summarize verified, already collected prefer evidence; never contact a board.

Missing evidence remains PENDING. Native monitor records are terminal snapshots,
not a claim that this offline script has observed live PIDs. Formats-only evidence
never becomes strict integer-tactic verification. The old obey summary is untouched.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from jetson_precision_candidate_smoke import (FIXED_IMAGE_SHA256, NMS, SCHEMA,
                                               binding_contract, verify_requested_precision)
from verify_gate05_evidence_archive import safe_name, unique_object


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'artifacts/gate05/20261019_precision_native'
CONFIGS = ('r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1')
COUNTS = {'r_only_G1': (82, 6), 'r_only_G1_G5': (74, 14),
          'r_only_G1_G2': (77, 11), 'q_only_G1': (6, 82)}
SCHEMA_OUT = 'rampq.gate05.prefer_matrix_checkpoint/v1'
KERNEL_MARKERS = ('controller error detected', 'HC died;', 'unregister_netdevice: waiting',
                  'Out of memory:', 'oom-kill:', 'Kernel panic')


class PendingEvidence(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_object)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def receipt_state(folder):
    path = folder / 'remote_receipt.json'
    if not path.is_file():
        return {'status': 'PENDING', 'reason': 'remote_receipt.json is absent', 'verified_files': []}
    try:
        receipt = read(path)
        require(receipt.get('schema') in (None, 'rampq.remote_file_receipt/v1'), 'Unsupported receipt schema')
        require(isinstance(receipt.get('files'), dict) and bool(receipt['files']), 'Empty receipt inventory')
        verified = []
        folded = set()
        for name, info in receipt['files'].items():
            safe_name(name)
            require(name.casefold() not in folded, 'Case-colliding receipt paths')
            folded.add(name.casefold())
            target = folder / name
            require(folder.resolve() in target.resolve().parents and not target.is_symlink(), 'Receipt path escaped directory')
            require(type(info.get('bytes')) is int and info['bytes'] >= 0 and
                    re.fullmatch('[0-9a-f]{64}', info.get('sha256', '')), 'Invalid receipt file metadata: ' + name)
            require(target.is_file() and target.stat().st_size == info['bytes'] and sha(target) == info['sha256'],
                    'Receipt size/SHA256 mismatch or missing file: ' + name)
            verified.append(name)
        return {'status': 'VERIFIED', 'receipt_sha256': sha(path), 'verified_file_count': len(verified),
                'verified_files': sorted(verified), 'receipt_schema': receipt.get('schema'),
                'limits': 'Extracted receipt integrity only; wire ZIP hash and native PID termination are separate checks'}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {'status': 'INVALID_EVIDENCE', 'reason': str(error), 'verified_files': []}


def required_file(folder, receipt, name):
    if name not in receipt['verified_files']:
        raise PendingEvidence('Required file is not yet covered by the verified receipt: ' + name)
    return folder / name


def resource_summary(path):
    count, available, swap_used, maximum_temperature = 0, None, 0.0, None
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            current = row['MemAvailable_kib'] / 1024.0
            available = current if available is None else min(available, current)
            swap_used = max(swap_used, (row['SwapTotal_kib'] - row['SwapFree_kib']) / 1024.0)
            values = list(row['temperatures_c'].values())
            if values:
                current = max(values)
                maximum_temperature = current if maximum_temperature is None else max(maximum_temperature, current)
    return {'samples': count, 'min_available_mib': available, 'max_swap_used_mib': swap_used,
            'max_temperature_c': maximum_temperature}


def verify_bundle(folder, config):
    manifest_path = folder / 'manifest.json'
    manifest = read(manifest_path)
    require(manifest['schema'] == 'rampq.gate05.precision_runtime_bundle/v1' and
            manifest['configuration'] == config, 'Local frozen bundle identity differs')
    for name, info in manifest['files'].items():
        safe_name(name)
        path = folder / name
        require(path.resolve().parent == folder.resolve() and path.is_file() and
                path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Frozen bundle file differs: ' + name)
    spec = read(folder / 'spec.json')
    require(spec['configuration'] == config and spec['bundle_revision'] == 'v2_prefer' and
            spec['precisionConstraints'] == 'prefer' and spec['diagnostic_only'] is True and
            spec['actual_precision_audit_criteria_changed'] is False, 'Prefer request/scope differs')
    require(spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1', 'Unsupported build schema')
    require((spec['quantized_conv_count'], spec['fp16_conv_count']) == COUNTS[config], 'Requested Conv counts differ')
    require(spec['max_build_seconds'] == 1800 and spec['workspace_mib'] == 1024 and spec['avg_timing'] == 1,
            'Prefer build resource settings differ')
    for name, field in [('model.onnx', 'graph_sha256'), ('module_map.json', 'module_map_sha256'),
                        ('graph_audit.json', 'graph_audit_sha256'), ('preflight.json', 'failed_preflight_sha256'),
                        ('diagnostic_scope.json', 'diagnostic_scope_sha256')]:
        require(sha(folder / name) == spec[field], 'Spec input lineage differs: ' + name)
    require(read(folder / 'preflight.json')['status'] == spec['original_local_preflight_status'] ==
            'LOCAL_QDQ_PREFLIGHT_FAILED', 'Original numerical failure was not retained')
    return spec


def check_build(folder, receipt, bundle, spec):
    monitor = read(required_file(folder, receipt, 'monitor/status.json'))
    require(monitor['root'] == '/home/nvidia/rampq/gate05_precision_' + spec['configuration'] + '_20261019_v2_prefer',
            'Native monitor targeted a different bundle')
    require(monitor['max_seconds'] == 1800 and monitor['build_script'] ==
            monitor['root'] + '/jetson_build_precision_candidate.py', 'Native monitor launch differs')
    state = monitor['status']
    if state == 'RUNNING':
        return {'status': 'PENDING', 'reason': 'Monitor snapshot remains RUNNING; offline data does not prove a live PID',
                'terminal_snapshot_verified': False}, None
    require(state in ('BUILD_SCRIPT_OK', 'BUILD_SCRIPT_FAILED', 'STOPPED_BY_MONITOR', 'FAILED'),
            'Unsupported native monitor state')
    for name in ('spec.json', 'manifest.json', 'jetson_build_precision_candidate.py'):
        native = required_file(folder, receipt, 'inputs/' + name)
        require(sha(native) == sha(bundle / name), 'Collected frozen input differs from selected bundle: ' + name)
    request = read(required_file(folder, receipt, 'build/build_request.json'))
    command = request['command']
    for flag in ('--precisionConstraints=prefer', '--int8', '--fp16', '--noTF32', '--buildOnly',
                 '--avgTiming=1', '--memPoolSize=workspace:1024', '--inputIOFormats=fp32:chw',
                 '--outputIOFormats=fp32:chw,fp32:chw'):
        require(flag in command, 'Native build command differs: ' + flag)
    restored = [row['onnx_node'] for row in spec['requests'] if row['precision'] == 'FP16']
    constraints = ','.join(name + ':fp16' for name in restored)
    require('--layerPrecisions=' + constraints in command and '--layerOutputTypes=' + constraints in command,
            'Native request omits a restored Conv compute/output constraint')
    require(request['requested_precision_constraints_mode'] == 'prefer' and
            request['spec_sha256'] == sha(bundle / 'spec.json') and
            request['manifest_sha256'] == sha(bundle / 'manifest.json') and
            request['onnx_sha256'] == spec['graph_sha256'], 'Native build request lineage differs')
    details = {'terminal_snapshot_verified': True, 'monitor_status': state, 'monitor_reason': monitor.get('reason'),
               'monitor_elapsed_seconds': monitor.get('elapsed_seconds'), 'boot_id': monitor['boot_id'],
               'monitor_pid': monitor['monitor_pid'], 'native_root': monitor['root'],
               'monitor_sha256': sha(folder / 'monitor/status.json'), 'native_live_pid_observed_by_this_script': False,
               'build_request_sha256': sha(folder / 'build/build_request.json')}
    kernel = required_file(folder, receipt, 'monitor/kernel.log')
    details['kernel_safety_event_lines'] = [line for line in kernel.read_text(errors='replace').splitlines()
                                           if any(marker in line for marker in KERNEL_MARKERS)]
    details['resources'] = resource_summary(required_file(folder, receipt, 'monitor/resources.jsonl'))
    build_path = folder / 'build/build_result.json'
    build = read(required_file(folder, receipt, 'build/build_result.json')) if build_path.is_file() else None
    if build is not None:
        require(build['configuration'] == spec['configuration'] and build['spec_sha256'] == sha(bundle / 'spec.json') and
                build['manifest_sha256'] == sha(bundle / 'manifest.json') and build['onnx_sha256'] == spec['graph_sha256'],
                'Build result/source lineage differs')
        details.update(build_result_status=build['status'], build_result_sha256=sha(build_path))
    if state == 'STOPPED_BY_MONITOR':
        details['status'] = ('TIMEOUT_INCONCLUSIVE' if monitor.get('reason') == 'Build supervision time limit reached'
                             else 'SAFETY_OR_COLLECTOR_STOP_REQUIRES_INSPECTION')
        details['stale_running_build_metadata_is_not_live_process_proof'] = True
    elif state == 'FAILED':
        details['status'] = 'MONITOR_FAILURE_REQUIRES_INSPECTION'
    elif state == 'BUILD_SCRIPT_FAILED':
        require(build is None or build['status'] != 'BUILT_AND_DESERIALIZED', 'Monitor/result success contradiction')
        log = required_file(folder, receipt, 'build/trtexec_build.log') if (folder / 'build/trtexec_build.log').is_file() else required_file(folder, receipt, 'monitor/build_console.log')
        details.update(status='BUILD_FAILED' if build is not None else 'BUILD_SCRIPT_FAILED_WITHOUT_RESULT',
                       error_log_sha256=sha(log), error_lines=[line for line in log.read_text(errors='replace').splitlines()
                                                               if '[E]' in line or 'Error' in line][-12:])
    else:
        if build is None:
            details.update(status='PENDING', reason='Successful monitor lacks a receipt-covered build_result.json')
            return details, None
        require(build['status'] == 'BUILT_AND_DESERIALIZED', 'Monitor success contradicts build result')
        engine = required_file(folder, receipt, 'build/model.engine')
        layers = required_file(folder, receipt, 'build/engine_layers.json')
        require(sha(engine) == build['engine_sha256'], 'Engine/result SHA256 differs')
        require(build['diagnostic_only'] is True and build['local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED',
                'Native build scope differs')
        require(binding_contract(build['bindings']) == binding_contract(spec['expected_binding_contract']) and
                all(row.get('format') == 'TensorFormat.LINEAR' for row in build['bindings']), 'Native binding contract differs')
        details.update(status='BUILT_AND_DESERIALIZED_INTEGRITY_VERIFIED', engine_sha256=sha(engine),
                       engine_bytes=engine.stat().st_size, layer_file_sha256=sha(layers))
    details['engine_file_present'] = (folder / 'build/model.engine').is_file()
    return details, build


def precision_state(folder, receipt, bundle, spec, build):
    name = 'build/precision_audit.json'
    if name not in receipt['verified_files']:
        return {'status': 'PENDING', 'reason': 'No receipt-covered precision audit'}, None
    path = required_file(folder, receipt, name)
    audit = read(path)
    for key, expected in [('engine_sha256', build['engine_sha256']), ('onnx_sha256', spec['graph_sha256']),
                          ('spec_sha256', sha(bundle / 'spec.json')), ('module_map_sha256', spec['module_map_sha256']),
                          ('layer_file_sha256', sha(folder / 'build/engine_layers.json')),
                          ('build_result_sha256', sha(folder / 'build/build_result.json'))]:
        require(audit[key] == expected, 'Precision audit lineage differs: ' + key)
    require(audit['configuration'] == spec['configuration'] and audit['source_convs'] == 88 and
            (audit['quantized_conv_count'], audit['fp16_conv_count']) == COUNTS[spec['configuration']],
            'Precision inventory differs')
    require(audit['schema'] == 'rampq.gate05.precision_candidate_audit/v1', 'Unsupported precision audit schema')
    mapping = audit['mapping']
    require(len(mapping) == len({row['onnx_node'] for row in mapping}) == 88, 'Precision source mapping incomplete')
    observed = {row['onnx_node']: row for row in mapping}
    require(set(observed) == {row['onnx_node'] for row in spec['requests']}, 'Precision source identities differ')
    for request in spec['requests']:
        require(all(observed[request['onnx_node']][key] == value for key, value in request.items()),
                'Precision request mapping differs')
    formats = audit['requested_conv_formats_complete'] is True
    strict = False
    if formats:
        strict = verify_requested_precision(audit, spec, diagnostic_formats_only=True)
    else:
        require(audit['status'] == 'PRECISION_REVIEW_REQUIRED', 'Incomplete formats mislabeled as passed')
    return {'status': audit['status'], 'audit_sha256': sha(path), 'requested_conv_formats_complete': formats,
            'verified_int8_format_convs': audit['verified_int8_format_convs'],
            'verified_fp16_format_convs': audit['verified_fp16_format_convs'],
            'strict_int8_status': audit['strict_int8_status'], 'strict_int8_tactics_verified': strict,
            'formats_only': formats and not strict,
            'named_integer_tactic_source_convs': audit['named_integer_tactic_source_convs'],
            'source_convs_without_tactic_name': audit['source_convs_without_tactic_name'],
            'errors': audit['errors'], 'unmapped_engine_convs': audit['unmapped_engine_convs']}, audit


def smoke_state(folder, receipt, bundle, spec, build, precision):
    if 'smoke/smoke_result.json' not in receipt['verified_files']:
        return {'status': 'PENDING', 'reason': 'No receipt-covered fixed-image smoke result'}
    report = read(required_file(folder, receipt, 'smoke/smoke_result.json'))
    raw_path = required_file(folder, receipt, 'smoke/raw_outputs.npz')
    require(report['schema'] == SCHEMA and report['configuration'] == spec['configuration'] and
            report['status'] == 'DIAGNOSTIC_FIXED_IMAGE_EXECUTION_OK' and report['execution_success'] is True,
            'Fixed-image execution is not successful')
    require(report['engine_sha256'] == build['engine_sha256'] and report['image_sha256'] == FIXED_IMAGE_SHA256 and
            report['nms'] == NMS and report['raw_outputs_sha256'] == sha(raw_path), 'Smoke input/engine/raw lineage differs')
    require(binding_contract(report['bindings']) == binding_contract(spec['expected_binding_contract']) and
            binding_contract(report['expected_binding_contract']) == binding_contract(spec['expected_binding_contract']),
            'Smoke named binding contract differs')
    require(report['lineage']['precision_audit_sha256'] == precision['audit_sha256'] and
            report['lineage']['build_result_sha256'] == sha(folder / 'build/build_result.json') and
            report['lineage']['spec_sha256'] == sha(bundle / 'spec.json'), 'Smoke audit/build lineage differs')
    require(report['strict_int8_tactics_verified'] is precision['strict_int8_tactics_verified'],
            'Smoke incorrectly labels integer tactic evidence')
    require(precision['strict_int8_tactics_verified'] or report['diagnostic_formats_only_opt_in'] is True,
            'Formats-only execution lacks explicit diagnostic opt-in')
    require(report['dataset_evaluation_run'] is False and report['accuracy_pass_claim'] is False and
            report['performance_pass_claim'] is False and report['causal_attribution_allowed'] is False,
            'Fixed-image diagnostic claims exceed the frozen scope')
    outputs = {row['name']: row['shape'] for row in spec['expected_binding_contract'] if not row['input']}
    require(set(report['all_outputs_finite']) == set(outputs) and all(report['all_outputs_finite'].values()),
            'Finite-output report is incomplete')
    try:
        import numpy as np
    except ImportError:
        return {'status': 'PENDING', 'reason': 'NumPy unavailable for independent saved-output finite/shape check'}
    with np.load(str(raw_path), allow_pickle=False) as raw:
        for name, shape in outputs.items():
            key = 'tensorrt' if name == spec['primary_output_name'] else report['auxiliary_output_npz_keys'][name]
            require(key in raw and list(raw[key].shape) == shape and bool(np.isfinite(raw[key]).all()),
                    'Saved primary/auxiliary output has wrong shape or nonfinite values')
    for row in report['detections']:
        require(len(row) == 6 and all(math.isfinite(value) for value in row) and
                row[5] == int(row[5]) and 0 <= int(row[5]) < 10 and 0.25 <= row[4] <= 1,
                'Invalid saved detection row')
    return {'status': 'VERIFIED_FIXED_IMAGE_EXECUTION_ONLY', 'report_sha256': sha(folder / 'smoke/smoke_result.json'),
            'raw_outputs_sha256': sha(raw_path), 'all_saved_outputs_finite': True,
            'detections': report['detections'], 'strict_int8_tactics_verified': precision['strict_int8_tactics_verified'],
            'diagnostic_formats_only_opt_in': report['diagnostic_formats_only_opt_in'],
            'causal_or_accuracy_claim': False}


def config_state(root, bundle_root, config):
    row = {'configuration': config, 'expected_int8': COUNTS[config][0], 'expected_fp16': COUNTS[config][1],
           'status': 'PENDING', 'pending': [], 'errors': [], 'checkpoint_evidence_complete': False}
    bundle = bundle_root / config
    try:
        spec = verify_bundle(bundle, config)
        row['bundle_manifest_sha256'] = sha(bundle / 'manifest.json')
        build_folder = root / config / 'build_evidence'
        post_folder = root / config / 'post_smoke_evidence'
        first, post = receipt_state(build_folder), receipt_state(post_folder)
        row['build_receipt'], row['post_smoke_receipt'] = first, post
        require(first['status'] != 'INVALID_EVIDENCE' and post['status'] != 'INVALID_EVIDENCE',
                'A collected receipt is invalid: ' + str(first.get('reason') or post.get('reason')))
        if first['status'] != 'VERIFIED':
            row['pending'].append('Verified build_evidence receipt')
            return row
        details, build = check_build(build_folder, first, bundle, spec)
        row['build'] = details
        if details['status'] == 'PENDING':
            row['pending'].append(details['reason'])
            return row
        if details['kernel_safety_event_lines']:
            row['status'] = 'KERNEL_SAFETY_EVENT_REQUIRES_INSPECTION'
            return row
        if details['status'] != 'BUILT_AND_DESERIALIZED_INTEGRITY_VERIFIED':
            row['status'] = details['status']
            row['precision'] = {'status': 'NOT_ESTABLISHED_NO_VERIFIED_COMPLETE_ENGINE'}
            row['smoke'] = {'status': 'NOT_RUN_NO_VERIFIED_COMPLETE_ENGINE'}
            row['checkpoint_evidence_complete'] = details['status'] in ('BUILD_FAILED', 'BUILD_SCRIPT_FAILED_WITHOUT_RESULT', 'TIMEOUT_INCONCLUSIVE')
            return row
        chosen, chosen_receipt = (post_folder, post) if post['status'] == 'VERIFIED' else (build_folder, first)
        if chosen == post_folder:
            after, after_build = check_build(chosen, chosen_receipt, bundle, spec)
            require(after['status'] == 'BUILT_AND_DESERIALIZED_INTEGRITY_VERIFIED' and
                    after_build['engine_sha256'] == build['engine_sha256'], 'Post-smoke build differs from original build')
            build = after_build
        precision, audit = precision_state(chosen, chosen_receipt, bundle, spec, build)
        row['precision'] = precision
        if precision['status'] == 'PENDING':
            row['pending'].append(precision['reason'])
            return row
        if not precision['requested_conv_formats_complete']:
            row['status'] = 'REQUESTED_FORMATS_NOT_ESTABLISHED'
            row['smoke'] = {'status': 'NOT_PERMITTED_REQUESTED_FORMATS_INCOMPLETE'}
            row['checkpoint_evidence_complete'] = True
            return row
        smoke = smoke_state(chosen, chosen_receipt, bundle, spec, build, precision)
        row['smoke'] = smoke
        if smoke['status'] == 'PENDING':
            row['pending'].append(smoke['reason'])
            return row
        row['status'] = ('STRICT_INT8_TACTICS_AND_FIXED_IMAGE_EVIDENCE_VERIFIED' if precision['strict_int8_tactics_verified']
                         else 'FORMATS_AND_DIAGNOSTIC_FIXED_IMAGE_VERIFIED_INTEGER_TACTICS_UNRESOLVED')
        row['checkpoint_evidence_complete'] = True
    except PendingEvidence as error:
        row['status'] = 'PENDING'
        row['pending'].append(str(error))
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as error:
        row['status'] = 'INVALID_OR_INCONSISTENT_EVIDENCE'
        row['errors'].append(type(error).__name__ + ': ' + str(error))
    return row


def queue_state(root, bundle_root):
    folder = root / 'queue_evidence'
    receipt = receipt_state(folder)
    result = {'status': 'PENDING', 'receipt': receipt, 'terminal_snapshot_verified': False}
    if receipt['status'] != 'VERIFIED':
        result['status'] = receipt['status']
        return result
    try:
        plan = read(required_file(folder, receipt, 'queue/plan.json'))
        state = read(required_file(folder, receipt, 'queue/status.json'))
        require(plan['schema'] == 'rampq.gate05.frozen_build_queue_plan/v1' and
                plan['bundle_revision'] == 'v2_prefer' and plan['max_build_seconds'] == 1800,
                'Queue frozen plan differs')
        require([item['configuration'] for item in plan['configurations']] == list(CONFIGS), 'Queue plan inventory differs')
        require(state['boot_id'] == plan['expected_boot_id'], 'Queue plan and snapshot boot identities differ')
        for item in plan['configurations']:
            require(item['precisionConstraints'] == 'prefer' and item['bundle_root'].endswith(
                    '/gate05_precision_' + item['configuration'] + '_20261019_v2_prefer') and
                    item['bundle_manifest_sha256'] == sha(bundle_root / item['configuration'] / 'manifest.json'),
                    'Queue selected a different frozen bundle')
        result.update(snapshot_status=state['status'], completed_configurations=[row['configuration'] for row in state['completed_builds']],
                      boot_id=state['boot_id'], native_live_pid_observed_by_this_script=False)
        result['completed_build_identities'] = {row['configuration']: {
            'monitor_pid': row['monitor_pid'],
            'native_root': row['terminal_monitor_status']['root'],
            'boot_id': row['terminal_monitor_status']['boot_id'],
            'build_result_sha256': row.get('build_result_sha256')}
            for row in state['completed_builds']}
        if state['status'] == 'FROZEN_BUILD_QUEUE_FINISHED_AUDIT_AND_SMOKE_PENDING':
            require(result['completed_configurations'] == list(CONFIGS), 'Terminal queue omitted a configuration')
            result.update(status='TERMINAL_QUEUE_SNAPSHOT_VERIFIED', terminal_snapshot_verified=True)
        elif state['status'] == 'QUEUE_STOPPED_REQUIRES_INSPECTION':
            result.update(status='TERMINAL_QUEUE_STOP_REQUIRES_INSPECTION', terminal_snapshot_verified=True,
                          reason=state.get('error'))
        else:
            result.update(reason='Queue snapshot is not terminal; offline data alone does not prove a live PID')
    except PendingEvidence as error:
        result.update(status='PENDING', reason=str(error))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status='INVALID_OR_INCONSISTENT_EVIDENCE', error=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefer-root', type=Path, default=NATIVE / 'prefer')
    parser.add_argument('--bundle-root', type=Path, default=ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles_prefer')
    parser.add_argument('--output', type=Path, default=NATIVE / 'prefer_matrix_summary.json')
    args = parser.parse_args()
    require(args.output.resolve() != (NATIVE / 'matrix_summary.json').resolve(), 'Never overwrite the old obey summary')
    if args.output.exists():
        require(read(args.output).get('schema') == SCHEMA_OUT, 'Refuse to overwrite an unrelated evidence file')
    rows = [config_state(args.prefer_root, args.bundle_root, config) for config in CONFIGS]
    queue = queue_state(args.prefer_root, args.bundle_root)
    for row in rows:
        build = row.get('build')
        identity = queue.get('completed_build_identities', {}).get(row['configuration'])
        if build and identity:
            mismatch = any(build.get(key) != identity[key] for key in ('monitor_pid', 'native_root', 'boot_id'))
            mismatch = mismatch or (identity.get('build_result_sha256') is not None and
                                     build.get('build_result_sha256') != identity['build_result_sha256'])
            if mismatch:
                row.update(status='INVALID_OR_INCONSISTENT_EVIDENCE', checkpoint_evidence_complete=False)
                row['errors'].append('Per-configuration native identity differs from collected queue result')
    complete = all(row['checkpoint_evidence_complete'] for row in rows) and queue['status'] == 'TERMINAL_QUEUE_SNAPSHOT_VERIFIED'
    invalid = any(row['errors'] for row in rows) or queue['status'].startswith('INVALID')
    status = ('PREFER_MATRIX_EVIDENCE_REVIEW_REQUIRED' if invalid else
              'PREFER_MATRIX_LOCAL_CHECKPOINT_COMPLETE' if complete else 'PREFER_MATRIX_EVIDENCE_PENDING')
    report = {'schema': SCHEMA_OUT, 'client_at': datetime.now().astimezone().isoformat(), 'status': status,
              'gate05_status': 'INCOMPLETE', 'prefer_root': str(args.prefer_root.resolve()),
              'bundle_root': str(args.bundle_root.resolve()), 'configurations': rows, 'queue': queue,
              'all_four_terminal_build_snapshots_verified': all(row.get('build', {}).get('terminal_snapshot_verified') is True for row in rows),
              'local_checkpoint_evidence_complete': complete,
              'format_complete_configurations': [row['configuration'] for row in rows if row.get('precision', {}).get('requested_conv_formats_complete')],
              'strict_int8_tactics_verified_configurations': [row['configuration'] for row in rows if row.get('precision', {}).get('strict_int8_tactics_verified')],
              'formats_only_configurations': [row['configuration'] for row in rows if row.get('precision', {}).get('formats_only')],
              'verified_saved_fixed_image_executions': sum(row.get('smoke', {}).get('status') == 'VERIFIED_FIXED_IMAGE_EXECUTION_ONLY' for row in rows),
              'original_q_all_numerical_guard': 'FAILED_UNCHANGED', 'dataset_acceptance_rules_status': 'NOT_ASSESSED_BY_THIS_SCRIPT',
              'native_process_live_state_observed_by_this_script': False, 'inference_performed_by_this_script': False,
              'native_layer_audit_recomputed_by_this_script': False,
              'dataset_evaluation_performed_by_this_script': False, 'causal_or_target_rescue_claim_established': False,
              'scope': 'Four existing mixed configurations under prefer requests only. FP16 control and extended q_all are separate '
                       'evidence. A complete local checkpoint does not establish Gate 0.5, numerical migration, '
                       'dataset accuracy, target rescue, performance, or current native process state.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in ('configurations', 'queue')}, indent=2))


if __name__ == '__main__':
    main()
