"""Read saved restore traces/arrays and a repeated FP16 control; no execution.

Exact NVTX/thread/API correlation identifies runtime kernels only. Integer-like
function names never replace missing TacticName fields or arithmetic instruction
evidence. This is the final local diagnostic of these already saved executions.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ('r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2')
CAPTURE = 'RAMPQ_QDQ_FIXED_IMAGE'
FIXED_IMAGE_SHA256 = '494bbd9d09de29f3a992d434da03c252b4f6cf1f28099fd5a443d0c797f5ec52'
NMS = {'conf_thres': 0.25, 'iou_thres': 0.7, 'max_det': 1000, 'nc': 10}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def verified_receipt(folder):
    receipt = read(folder / 'remote_receipt.json')
    files = [folder / 'remote_receipt.json']
    for name, info in receipt['files'].items():
        path = folder / name
        require(folder.resolve() in path.resolve().parents and path.is_file() and
                path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Receipt mismatch: ' + name)
        files.append(path)
    return files


def hashes(paths):
    return {str(path.relative_to(ROOT)).replace('\\', '/'): {'bytes': path.stat().st_size, 'sha256': sha(path)}
            for path in sorted(set(paths), key=str)}


def correlate(database, layers, audit):
    targets = [row for row in layers if row.get('LayerType') == 'CudaDepthwiseConvolution']
    missing = [row for row in audit['mapping'] if row.get('integer_tactic_evidence') == 'TACTIC_NAME_NOT_EXPOSED']
    require(len(targets) == len(missing) == 7 and {row['Name'] for row in targets} ==
            {row['engine_layer'] for row in missing}, 'Expected exactly the seven missing-tactic depthwise layers')
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA query_only=ON')
    connection.row_factory = sqlite3.Row
    try:
        strings = dict(connection.execute('SELECT id,value FROM StringIds'))
        nvtx = []
        for item in connection.execute('SELECT rowid AS sqlite_rowid,* FROM NVTX_EVENTS'):
            row = dict(item)
            row['resolved_text'] = row['text'] if row['text'] is not None else strings.get(row['textId'])
            row['resolved_json'] = row['jsonText'] if row['jsonText'] is not None else strings.get(row['jsonTextId'])
            nvtx.append(row)
        captures = [row for row in nvtx if row['resolved_text'] == CAPTURE and row['end'] is not None]
        require(len(captures) == 1, 'Expected one completed fixed-image capture range')
        capture = captures[0]
        apis = [dict(row) for row in connection.execute('SELECT rowid AS sqlite_rowid,* FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
        kernels = [dict(row) for row in connection.execute('SELECT rowid AS sqlite_rowid,* FROM CUPTI_ACTIVITY_KIND_KERNEL')]
        api_correlations, kernel_correlations = {}, {}
        for row in apis:
            api_correlations.setdefault(row['correlationId'], []).append(row)
        for row in kernels:
            kernel_correlations.setdefault(row['correlationId'], []).append(row)
        names = {row['Name'] for row in layers}
        layer_ranges = [row for row in nvtx if row['resolved_text'] in names and row['end'] is not None]
        observations = []
        for layer in targets:
            row = {'native_layer_name': layer['Name'], 'native_layer_type': layer['LayerType'],
                   'native_weights': layer.get('Weights'), 'native_tactic_name': layer.get('TacticName'),
                   'native_input_formats': [value.get('Format/Datatype') for value in layer.get('Inputs', [])],
                   'native_output_formats': [value.get('Format/Datatype') for value in layer.get('Outputs', [])],
                   'runtime_identity_observed': False, 'integer_instruction_proof': False,
                   'fills_missing_named_tactic': False, 'errors': [], 'correlations': []}
            matches = [event for event in layer_ranges if event['resolved_text'] == layer['Name'] and
                       event['globalTid'] == capture['globalTid'] and capture['start'] <= event['start'] and
                       event['end'] <= capture['end']]
            row['completed_exact_name_range_count'] = len(matches)
            if len(matches) != 1:
                row['errors'].append('Missing or ambiguous exact layer name/range')
                observations.append(row)
                continue
            event = matches[0]
            row['nvtx'] = event
            row['registered_nvtx_json_equals_native_inspector'] = (
                event['resolved_json'] is not None and json.loads(event['resolved_json']) == layer)
            if not row['registered_nvtx_json_equals_native_inspector']:
                row['errors'].append('NVTX registered JSON differs from the inspector layer')
            if event['endGlobalTid'] not in (None, event['globalTid']):
                row['errors'].append('Cross-thread NVTX end is unsupported')
            enclosed = [api for api in apis if api['globalTid'] == event['globalTid'] and
                        event['start'] <= api['start'] and api['end'] <= event['end']]
            for api in enclosed:
                gpu = kernel_correlations.get(api['correlationId'], [])
                if not gpu:
                    continue
                containers = [candidate for candidate in layer_ranges if candidate['globalTid'] == api['globalTid'] and
                              candidate['start'] <= api['start'] and api['end'] <= candidate['end']]
                unique = len(api_correlations.get(api['correlationId'], [])) == 1
                one_layer = len(containers) == 1 and containers[0]['sqlite_rowid'] == event['sqlite_rowid']
                chain = {'runtime_api': dict(api, resolved_name=strings.get(api['nameId'])),
                         'same_thread_and_fully_inside_exact_layer_cpu_interval': True,
                         'unique_runtime_correlation_id': unique,
                         'unique_enclosing_native_layer': one_layer, 'kernels': []}
                if not unique or not one_layer or api['returnValue'] != 0:
                    row['errors'].append('Ambiguous or unsuccessful runtime launch correlation')
                for kernel in gpu:
                    recorded = dict(kernel)
                    recorded.update(demangled_name_text=strings.get(kernel['demangledName']),
                                    short_name_text=strings.get(kernel['shortName']),
                                    mangled_name_text=strings.get(kernel['mangledName']))
                    chain['kernels'].append(recorded)
                row['correlations'].append(chain)
            row['exact_correlated_kernel_count'] = sum(len(chain['kernels']) for chain in row['correlations'])
            if not row['exact_correlated_kernel_count']:
                row['errors'].append('No exact API correlation reaches a GPU kernel')
            row['runtime_identity_observed'] = not row['errors']
            observations.append(row)
        return {'status': 'ALL_SEVEN_RUNTIME_IDENTITIES_OBSERVED' if all(row['runtime_identity_observed'] for row in observations)
                else 'RUNTIME_IDENTITY_EVIDENCE_INCOMPLETE', 'sqlite_sha256': sha(database),
                'capture_range': capture, 'runtime_api_rows': len(apis), 'gpu_kernel_rows': len(kernels),
                'observed_layer_count': sum(row['runtime_identity_observed'] for row in observations),
                'layers': observations, 'strict_tactic_status_changed': False,
                'integer_instruction_proof_established': False,
                'limits': 'Layer CPU ranges contain launch APIs; asynchronous GPU events are joined only by unique correlationId. '
                          'Function names/templates do not prove executed arithmetic instructions or supply absent TacticName fields.'}
    finally:
        connection.close()


def saved_outputs(folder, np):
    report = read(folder / 'smoke/smoke_result.json')
    require(report['status'] == 'DIAGNOSTIC_FIXED_IMAGE_EXECUTION_OK' and report['execution_success'] is True and
            report['image_sha256'] == FIXED_IMAGE_SHA256 and report['nms'] == NMS, 'Saved smoke settings/status differ')
    path = folder / 'smoke/raw_outputs.npz'
    require(sha(path) == report['raw_outputs_sha256'], 'Saved output hash differs')
    expected = {row['name']: row['shape'] for row in report['expected_binding_contract'] if not row['input']}
    require(set(report['all_outputs_finite']) == set(expected) and all(report['all_outputs_finite'].values()),
            'Finite report does not cover all expected output names')
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    for name, shape in expected.items():
        key = 'tensorrt' if name == report['primary_output_name'] else report['auxiliary_output_npz_keys'][name]
        require(key in arrays and list(arrays[key].shape) == shape and bool(np.isfinite(arrays[key]).all()),
                'Saved primary or auxiliary output is nonfinite or has the wrong shape')
    primary = arrays['tensorrt']
    scores = primary[:, 4:]
    summary = {'report_sha256': sha(folder / 'smoke/smoke_result.json'), 'raw_outputs_sha256': sha(path),
               'engine_sha256': report['engine_sha256'], 'boot_id': report['boot_id'],
               'configuration': report['configuration'], 'raw_score_min': float(scores.min()),
               'raw_score_max': float(scores.max()), 'anchors_above_frozen_conf': int((scores.max(axis=1) > 0.25).sum()),
               'all_expected_arrays_finite_and_shape_verified': True, 'saved_detection_count': len(report['detections']),
               'saved_detections': report['detections'], 'strict_int8_status': report['strict_int8_status'],
               'diagnostic_formats_only_opt_in': report['diagnostic_formats_only_opt_in']}
    return summary, arrays, report


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-root', type=Path, default=ROOT / 'artifacts/gate05/20261019_restore_extended')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/gate05/20261019_restore_saved_execution_diagnostic')
    args = parser.parse_args()
    original_fp16 = ROOT / 'artifacts/gate05/20261019_precision_native/fp16_control/post_smoke_evidence'
    ending_fp16 = args.restore_root / 'fp16_end_control/post_smoke_evidence'
    folders = [args.restore_root / config / 'post_smoke_evidence' for config in CONFIGS] + [ending_fp16, original_fp16]
    paths, receipt_counts = [Path(__file__).resolve()], {}
    for folder in folders:
        files = verified_receipt(folder)
        receipt_counts[str(folder.relative_to(ROOT)).replace('\\', '/')] = len(files) - 1
        paths.extend(files)
    before = hashes(paths)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'source_hashes_before.json', before)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    rows = []
    for config, folder in zip(CONFIGS, folders):
        audit = read(folder / 'build/precision_audit.json')
        layers_path, build_path = folder / 'build/engine_layers.json', folder / 'build/build_result.json'
        layers, build = read(layers_path)['Layers'], read(build_path)
        require(audit['requested_conv_formats_complete'] is True and audit['strict_int8_status'] ==
                'INTEGER_TACTIC_REVIEW_REQUIRED' and audit['source_convs_without_tactic_name'] == 7,
                'Original strict audit scope differs')
        require(audit['layer_file_sha256'] == sha(layers_path) and audit['build_result_sha256'] == sha(build_path),
                'Audit layer/build lineage differs')
        summary, _arrays, smoke = saved_outputs(folder, np)
        require(summary['configuration'] == config and summary['saved_detection_count'] == 0 and
                summary['raw_score_max'] < 0.25 and summary['anchors_above_frozen_conf'] == 0,
                'Zero saved detections are not consistent with all saved scores below the frozen threshold')
        require(summary['engine_sha256'] == build['engine_sha256'] == audit['engine_sha256'] and
                summary['diagnostic_formats_only_opt_in'] is True and smoke['strict_int8_tactics_verified'] is False,
                'Engine or diagnostic-only precision boundary differs')
        trace = correlate(folder / 'smoke/profile/fixed_image.sqlite', layers, audit)
        rows.append({'configuration': config, 'saved_execution': summary,
                     'precision_formats': {'int8_convs': audit['verified_int8_format_convs'],
                                           'fp16_convs': audit['verified_fp16_format_convs'],
                                           'named_integer_tactics': audit['named_integer_tactic_source_convs'],
                                           'missing_tactic_names': 7, 'strict_status': audit['strict_int8_status']},
                     'runtime_kernel_trace': trace})
    old_summary, old_arrays, old_report = saved_outputs(original_fp16, np)
    end_summary, end_arrays, end_report = saved_outputs(ending_fp16, np)
    require(old_summary['configuration'] == end_summary['configuration'] == 'fp16_control' and
            old_summary['engine_sha256'] == end_summary['engine_sha256'], 'FP16 control engine differs')
    require(set(old_arrays) == set(end_arrays), 'FP16 saved output array keys differ')
    equality = {}
    for name in old_arrays:
        first, second = old_arrays[name], end_arrays[name]
        equality[name] = {'same_dtype': first.dtype == second.dtype, 'same_shape': first.shape == second.shape,
                          'both_finite': bool(np.isfinite(first).all() and np.isfinite(second).all()),
                          'same_raw_array_bytes': first.tobytes(order='C') == second.tobytes(order='C'),
                          'max_absolute_difference': float(np.abs(first - second).max())}
    require(all(all(value[key] for key in ('same_dtype', 'same_shape', 'both_finite', 'same_raw_array_bytes'))
                for value in equality.values()), 'FP16 repeated saved arrays are not bit-exact and finite')
    require(old_summary['saved_detection_count'] == end_summary['saved_detection_count'] == 2 and
            old_report['detections'] == end_report['detections'], 'Saved FP16 predictions differ')
    after = hashes(paths)
    require(before == after, 'A source file changed during read-only diagnostics')
    write(output / 'source_hashes_after.json', after)
    report = {'schema': 'rampq.gate05.restore_saved_execution_diagnostic/v1',
              'client_at': datetime.now().astimezone().isoformat(),
              'status': 'SAVED_RESTORE_EXECUTIONS_AND_REPEATED_FP16_CONTROL_CHECKED',
              'original_receipt_file_counts': receipt_counts, 'all_source_files_unchanged': True,
              'restores': rows,
              'repeated_fp16_control': {'original': old_summary, 'ending': end_summary,
                                        'saved_array_comparison': equality, 'all_primary_and_auxiliary_arrays_bit_exact': True,
                                        'saved_prediction_rows_exactly_equal': True,
                                        'cross_boot_recorded': old_summary['boot_id'] != end_summary['boot_id'],
                                        'limits': 'One saved repeated FP16 execution confirms identical finite outputs for this fixed image; no stability or performance generalization.'},
              'integer_instruction_evidence_added': False, 'strict_precision_status_changed': False,
              'new_engine_execution_performed': False, 'board_access': False,
              'dataset_evaluation_run': False, 'causal_or_rescue_claim': False,
              'scope': 'Last local diagnostic of existing original traces/arrays. No new experiments, thresholds, graphs, criteria or inference.'}
    write(output / 'analysis.json', report)
    files = {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(output.iterdir()) if path.is_file()}
    write(output / 'manifest.json', {'schema': 'rampq.gate05.restore_saved_diagnostic_files/v1', 'files': files})
    (output / 'SHA256SUMS').write_text(''.join(info['sha256'] + '  ' + name + '\n' for name, info in files.items()), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'output': str(output),
                      'restores': [{'configuration': row['configuration'],
                                    'max_score': row['saved_execution']['raw_score_max'],
                                    'saved_detections': row['saved_execution']['saved_detection_count'],
                                    'runtime_layers_exactly_correlated': row['runtime_kernel_trace']['observed_layer_count'],
                                    'strict_status': row['precision_formats']['strict_status']} for row in rows],
                      'fp16_control_all_arrays_bit_exact': True, 'fp16_saved_detection_count': 2,
                      'all_source_files_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()
