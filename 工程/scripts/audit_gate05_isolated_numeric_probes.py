"""Recompute isolated diagnostic observations from receipt-verified native bytes.

Read-only native evidence; writes a fresh audit only. No CUDA, model build,
dataset inference, tolerance changes, or complete-model pass claims.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from jetson_firstconv_numeric_probe import compare, require, sha, stats


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_recorded_statistics(recomputed, recorded):
    # Different NumPy/libm implementations can differ at the final FP64 bit in
    # reductions/sqrt. This checks reporting arithmetic only, not model parity.
    require(set(recomputed) == set(recorded), 'Metric fields differ')
    differences = {}
    reduction_fields = {'mean_abs', 'rms_difference', 'abs_percentiles_50_90_99_999'}
    for key, actual in recomputed.items():
        original = recorded[key]
        if actual == original:
            continue
        require(key in reduction_fields, 'Exact metric mismatch: ' + key)
        first, second = np.asarray(actual, dtype=np.float64), np.asarray(original, dtype=np.float64)
        require(first.shape == second.shape and bool(np.isfinite(first).all()) and bool(np.isfinite(second).all()), 'Invalid metric')
        bound = 8 * np.maximum(np.abs(np.spacing(first)), np.abs(np.spacing(second)))
        require(bool((np.abs(first - second) <= bound).all()), 'Recorded reduction differs beyond FP64 rounding: ' + key)
        differences[key] = {'recorded': original, 'recomputed': actual,
                            'scope': 'Reporting reduction rounding only; original model tolerances unchanged'}
    return differences


def audit(root, receipt_verification):
    verification = read(receipt_verification)
    require(verification['status'] == 'ARCHIVE_AND_EXACT_RECEIPT_FILE_INVENTORY_VERIFIED', 'Unverified archive')
    require(Path(verification['extracted_to']).resolve() == root.resolve(), 'Receipt directory mismatch')
    for entry in verification['verified_files']:
        path = root / entry['path']
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Evidence bytes changed')
    spec, launch, monitor = (read(root / name) for name in ('spec.json', 'launch_receipt.json', 'monitor/status.json'))
    require(monitor['status'] == 'BUILD_SCRIPT_OK' and monitor['build_returncode'] == 0, 'Native task not successful')
    require(launch['boot_id'] == monitor['boot_id'] == read(root / 'end_board_status.json')['boot_id'], 'Boot changed')
    require(sha(root / 'spec.json') == launch['spec_sha256'], 'Launch spec hash')
    require(spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED', 'Original guard changed')
    if 'tool_hashes' in launch:
        for name, digest in launch['tool_hashes'].items():
            require(sha(root / name) == digest, 'Executed tool bytes changed')
    else:
        require(sha(root / 'jetson_firstconv_numeric_probe.py') == launch['runner_sha256'], 'Runner changed')
        require(sha(root / 'jetson_monitored_build.py') == launch['monitor_sha256'], 'Monitor changed')
    image = np.load(root / spec['input_file'], allow_pickle=False)
    require(sha(root / spec['input_file']) == spec['input_file_sha256'], 'Frozen input mismatch')
    expected_path = root / spec['cpu_expected_outputs_file']
    require(sha(expected_path) == spec['cpu_expected_outputs_sha256'], 'CPU expected bytes mismatch')
    with np.load(expected_path, allow_pickle=False) as data:
        expected = {key: data[key].copy() for key in data.files}
    cases, raw = {}, {}
    for result_path in sorted(root.glob('results/*/*/result.json')):
        report = read(result_path)
        directory = result_path.parent
        case = report['case']
        request = read(directory / 'request.json')
        require(sha(directory / 'model.engine') == report['engine_sha256'], 'Engine hash mismatch')
        require(sha(directory / 'raw_outputs.npz') == report['raw_outputs_sha256'], 'Raw hash mismatch')
        variant_name = request.get('variant', case)
        variant = spec['variants'][variant_name]
        require(sha(root / variant['model_file']) == variant['model_sha256'] == request['onnx_sha256'], 'Model request lineage')
        require(request['spec_sha256'] == sha(root / 'spec.json'), 'Case spec mismatch')
        require(report['boot_id'] == monitor['boot_id'], 'Case boot mismatch')
        contract = [spec['input_binding']] + spec['output_bindings_in_order']
        observed = report['bindings']
        fields = ('name', 'input', 'shape', 'dtype')
        require(sorted(({k: row[k] for k in fields} for row in contract), key=lambda r: r['name']) ==
                sorted(({k: row[k] for k in fields} for row in observed), key=lambda r: r['name']), 'Named binding mismatch')
        with np.load(directory / 'raw_outputs.npz', allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        outputs, raw[case] = {}, {}
        for binding in spec['output_bindings_in_order']:
            name = binding['name']
            reference_key = variant['expected_output_npz_keys'][name]
            first_key = 'first_' + reference_key if 'outputs' in report else 'first'
            repeat_key = 'repeat_' + reference_key if 'outputs' in report else 'repeat'
            array = values[first_key]
            require(array.dtype == np.float32 and list(array.shape) == binding['shape'], 'Raw dtype/shape mismatch')
            observed_metrics = compare(array, expected[reference_key], np)
            recorded_metrics = report['outputs'][name]['comparison_to_cpu_ort'] if 'outputs' in report else report['comparison_to_cpu_ort']
            metric_rounding = verify_recorded_statistics(observed_metrics, recorded_metrics)
            require(compare(values[repeat_key], array, np)['bit_exact'], 'Repeated execution differs')
            outputs[name] = {'stats': stats(array, np), 'comparison_to_cpu_ort': observed_metrics, 'repeat_bit_exact': True,
                             'native_vs_host_statistics_rounding': metric_rounding}
            raw[case][name] = array
        layers = read(directory / 'engine_layers.json')['Layers']
        convs = [row for row in layers if 'Convolution' in row.get('LayerType', '')]
        if 'selected_conv_name' in spec:
            require(len(convs) == 1 and spec['selected_conv_name'] in convs[0]['Name'], 'Conv mapping incomplete')
            conv = convs[0]
            expected_type = 'Int8' if variant_name not in ('fp32',) else ('Half' if case.startswith('fp16') else 'Float')
            require(conv['Weights']['Type'] == expected_type, 'Actual Conv weight precision differs')
            input_index = max(conv.get('ConvXAsActInputIdx', -1), 0)
            input_format = conv['Inputs'][input_index]['Format/Datatype'].upper()
            require(('INT8' if expected_type == 'Int8' else 'FP16' if expected_type == 'Half' else 'FP32') in input_format,
                    'Actual Conv activation precision differs')
        cases[case] = {'outputs': outputs, 'convolutions': convs,
                       'pointwise_clip_layers': [row for row in layers if 'Clip' in row.get('Name', '')],
                       'engine_sha256': report['engine_sha256'], 'onnx_sha256': request['onnx_sha256']}
    require(bool(cases), 'No executed cases')
    samples = [json.loads(line) for line in (root / 'monitor/resources.jsonl').read_text().splitlines()]
    kernel = (root / 'monitor/kernel.log').read_text(errors='replace')
    markers = ('controller error detected', 'HC died;', 'unregister_netdevice: waiting', 'Out of memory:', 'oom-kill:', 'Kernel panic')
    require(not any(marker in kernel for marker in markers), 'Kernel safety record present')
    result = {'native_root': str(root.resolve()), 'receipt_verified_files': len(verification['verified_files']),
              'archive_sha256': verification['archive_sha256'], 'spec_sha256': sha(root / 'spec.json'),
              'input_raw_sha256': hashlib.sha256(image.tobytes()).hexdigest(), 'cases': cases,
              'boot_id': monitor['boot_id'], 'monitor_seconds': monitor['elapsed_seconds'],
              'maximum_temperature_c': max(max(row['temperatures_c'].values()) for row in samples),
              'minimum_available_memory_mib': min(row['MemAvailable_kib'] for row in samples) / 1024,
              'native_processes_terminal_at_collection': True, 'no_kernel_safety_marker': True}
    return result, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', action='append', type=Path, required=True)
    parser.add_argument('--verification', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(len(args.evidence) == len(args.verification), 'Expected paired paths')
    require(not args.output.exists(), 'Audit output must be fresh')
    audits, raw = [], {}
    for path, verification in zip(args.evidence, args.verification):
        result, arrays = audit(path, verification)
        audits.append(result)
        raw[path.name] = arrays
    cross = {}
    firstconv = raw.get('20261019_firstconv_numeric_native_evidence')
    boundary = raw.get('20261019_first_activation_boundary_native_evidence')
    constraint = raw.get('20261019_activation_fp32_clip_constraint_native_evidence')
    intervention = raw.get('20261019_firstconv_clip_intervention_native_evidence')
    code_corrected = raw.get('20261019_firstconv_code_clip_corrected_native_evidence')
    if firstconv:
        cross['qdq_global_fp16_on_vs_off'] = {
            name: compare(array, firstconv['qdq_fp32_no_tf32'][name], np)
            for name, array in firstconv['qdq_fp16_no_tf32'].items()}
        require(all(value['bit_exact'] for value in cross['qdq_global_fp16_on_vs_off'].values()), 'QDQ control results differ')
    if boundary:
        name = 'rampq_model_0_conv_a_dq'
        clip, no_clip = boundary['clip'][name], boundary['no_clip'][name]
        cross['activation_boundary'] = {'clip_unique_float32_values': np.unique(clip).tolist(),
                                        'no_clip_unique_float32_value_count': int(np.unique(no_clip).size),
                                        'clip_vs_no_clip': compare(clip, no_clip, np)}
        require(np.unique(clip).size == 1, 'Clip output is not constant')
    if boundary and constraint:
        name = 'rampq_model_0_conv_a_dq'
        cross['fp32_clip_constraint_vs_original_clip'] = compare(constraint['clip'][name], boundary['clip'][name], np)
        require(cross['fp32_clip_constraint_vs_original_clip']['bit_exact'], 'Constraint result changed')
    if firstconv and intervention:
        cross['repeated_original_clip_firstconv'] = {
            name: compare(array, firstconv['qdq_fp16_no_tf32'][name], np)
            for name, array in intervention['clip_fp16'].items()}
        require(all(value['bit_exact'] for value in cross['repeated_original_clip_firstconv'].values()), 'Repeated Clip baseline changed')
        cross['firstconv_clip_intervention'] = {
            name: compare(array, intervention['no_clip_fp16'][name], np)
            for name, array in intervention['clip_fp16'].items()}
    if intervention and code_corrected:
        cross['code_clip_corrected_vs_fixed_input_no_clip'] = {
            name: compare(array, intervention['no_clip_fp16'][name], np)
            for name, array in code_corrected['code_clip_importer_corrected'].items()}
        require(all(value['bit_exact'] for value in cross['code_clip_corrected_vs_fixed_input_no_clip'].values()),
                'Corrected code Clip differs from no-Clip fixed input control')
    observation = {'schema': 'rampq.gate05.isolated_numeric_root_audit/v1',
                   'status': 'RECEIPT_LINEAGE_NATIVE_PRECISION_AND_RAW_METRICS_RECOMPUTED',
                   'client_at': datetime.now().astimezone().isoformat(), 'native_evidence_audits': audits,
                   'cross_native_observations': cross,
                   'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
                   'dataset_evaluation_run': False, 'complete_detector_rebuilt_or_executed': False,
                   'new_numerical_pass_criterion_defined': False,
                   'limits': 'These isolated numeric diagnostics do not establish complete-model numerical migration, actual full-engine root cause, strict integer instructions, detector accuracy, target rescue, or performance.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(observation, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': observation['status'], 'audited_cases': sum(len(r['cases']) for r in audits),
                      'output_sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
