"""Independent saved-only DFL review: never imports TRT, torch or ORT."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np


PINS = {
    'protocol.json': 'b2723549e6acee51c86aeba3b72c8451601ba4add6088f46f7443536a1ab74ee',
    'spec.json': '934e075aa58dde1829b600d3d011f5ad51c4f1da1af78f7791944f492781b14d',
    'manifest.json': '3836e82ed53afb14fe16d3b9431864c0c4a45fd0d3418c7c06a4f6fcd46ed1d0',
    'source_model.onnx': '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee',
    'parent_spec.json': 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb',
    'parent_manifest.json': 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f',
    'failure_inventory.json': 'a2780d83f1de24ab166553a8a3d341f04cf3c6f74ae41cabd5885124bc81d112',
    'model_dfl_code_clip.onnx': 'a8a30c71a20d6b6dd3d031d23ce8dc8672cc62abe6c3eb8725b324fd2e600092',
    'dfl_transpose_fixture.npy': 'cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab',
    'cpu_expected_outputs.npz': 'fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511',
    'jetson_dfl_context_probe.py': '5f4c14dd6b4989f5d4011502b3caa647c785d355cdb8f54ec7d1f296d4159b65',
    'jetson_build_dfl_context_probe.py': '3651f7deec68cb4e604022999ba0c42f20f66eaf961b35feea0fc94d62a50ffd',
}
BOOT = '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
SOURCE = '/model.23/dfl/conv/Conv'
CASES = ('code_clip_none', 'code_clip_obey')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def fingerprint(root):
    return {p.relative_to(root).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in root.rglob('*') if p.is_file()}


def compare(left, right):
    assert left.dtype == right.dtype == np.float32 and left.shape == right.shape
    finite = np.isfinite(left) & np.isfinite(right)
    differences = np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64))
    return {
        'all_pairs_finite': bool(finite.all()), 'nonfinite_pair_count': int((~finite).sum()),
        'numeric_equal': bool(np.array_equal(left, right)), 'byte_identical': left.tobytes() == right.tobytes(),
        'different_elements': int(np.count_nonzero(left != right)),
        'max_abs': float(differences.max()) if differences.size else None,
        'mean_abs': float(differences.mean()) if differences.size else None,
        'rms_difference': float(np.sqrt(np.mean(differences ** 2))) if differences.size else None,
        'descriptive_only_no_new_pass_criterion': True,
    }


def normalized_constraints(rows):
    result = []
    for row in rows:
        selected = {k: row[k] for k in ('index', 'name', 'precision_is_set', 'precision', 'precision_value_error') if k in row}
        selected['outputs'] = [{k: output[k] for k in ('index', 'output_type_is_set', 'requested_output_type',
                              'requested_output_type_error') if k in output} for output in row['output_types']]
        result.append(selected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.output.resolve()
    assert raw.is_dir() and output.parent.is_dir() and not output.exists() and raw not in output.parents
    before = fingerprint(raw)
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)

    receipt = read(raw / 'remote_receipt.json')
    check('exact_saved_receipt_inventory', set(before) == set(receipt['files']) | {'remote_receipt.json'})
    for rel, expected in receipt['files'].items():
        check('receipt_member:' + rel, before[rel] == expected)
    for rel, expected in read(raw / 'manifest.json')['files'].items():
        check('sealed_bundle_member:' + rel, before[rel] == expected)
    for rel, digest in PINS.items():
        check('fixed_identity:' + rel, before[rel]['sha256'] == digest)
    spec, protocol = read(raw / 'spec.json'), read(raw / 'protocol.json')
    check('frozen_two_case_order', tuple(c['id'] for c in spec['cases']) == CASES and spec['cases'] == protocol['cases'])
    check('sole_DFL_source_and_boundary', spec['conv_names'] == [SOURCE] and spec['clip_names'] == ['rampq_model_23_dfl_conv_Clip'])
    check('fixture_not_images_or_boxes', spec['input_file'] == 'dfl_transpose_fixture.npy' and
          spec['input_binding']['name'] == '/model.23/dfl/Transpose_output_0' and
          spec['input_binding']['shape'] == [1, 16, 4, 8400] and
          spec['output_bindings_in_order'][0]['name'] == '/model.23/dfl/Reshape_1_output_0' and
          spec['output_bindings_in_order'][0]['shape'] == [1, 4, 8400])
    check('frozen_build_execution_limits', spec['max_seconds'] == 900 and spec['max_build_seconds'] == 300 and
          spec['maximum_native_builds'] == 2 and spec['maximum_native_inferences'] == 4 and
          spec['executions_per_successful_case'] == 2)
    lineage = read(raw / 'fixture_lineage.json')
    fixture = np.load(raw / spec['input_file'], allow_pickle=False)
    check('fixture_saved_finite_FLOAT_shape', fixture.dtype == np.float32 and fixture.shape == (1, 16, 4, 8400) and np.isfinite(fixture).all())
    check('fixture_saved_array_byte_identity', hashlib.sha256(fixture.tobytes()).hexdigest() == lineage['fixture']['raw_array_bytes_sha256'])
    check('fixture_ancestor_once_not_full_detector', lineage['ancestor_nodes'] == 638 and lineage['ancestor_initializers'] == 576 and
          lineage['ancestor_conv_count'] == 72 and lineage['ancestor_cpu_executions'] == 1 and
          lineage['complete_detector_output_executions'] == 0 and lineage['provider'] == 'CPUExecutionProvider' and
          lineage['graph_optimization'] == 'ORT_DISABLE_ALL')
    with np.load(raw / spec['cpu_expected_outputs_file'], allow_pickle=False) as archive:
        expected = archive['code_clip_output'].copy()
        original = archive['original_output'].copy()
    check('saved_CPU_tail_original_corrected_byte_equal', expected.dtype == original.dtype == np.float32 and
          expected.shape == original.shape == (1, 4, 8400) and expected.tobytes() == original.tobytes() and np.isfinite(expected).all())
    runs = list((raw / 'results').glob('probe_*'))
    check('single_native_frozen_batch', len(runs) == 1)
    run = runs[0]
    summary = read(run / 'summary.json')
    check('saved_batch_case_order', tuple(summary['requested_case_order']) == CASES)
    arrays, case_reports, request_records, constraint_records = {}, {}, {}, {}
    reduction_differences = {}

    def verify_native_comparison(name, actual, reported):
        check(name + ':field_set', set(actual) == set(reported))
        reduction_differences[name] = {}
        for field in actual:
            if field in ('mean_abs', 'rms_difference'):
                reduction_differences[name][field] = {'host': actual[field], 'native': reported[field],
                    'host_minus_native': actual[field] - reported[field] if actual[field] is not None else None,
                    'descriptive_only_no_numeric_acceptance_rule': True}
            else:
                check(name + ':' + field, actual[field] == reported[field])

    for name in CASES:
        folder = run / name
        if not folder.is_dir():
            case_reports[name] = {'status': 'NOT_RUN_AFTER_SAFE_ABORT'}
            continue
        cleanup = read(folder / 'child_cleanup.json')
        check(name + ':direct_builder_terminated', cleanup['direct_python_api_child_terminated'] and cleanup['returncode'] is not None)
        report = {'saved_status': summary['cases'][name]['status'], 'child_cleanup': cleanup}
        if (folder / 'parsed_network_request.json').is_file():
            request = read(folder / 'parsed_network_request.json')
            request_records[name] = request
            check(name + ':request_frozen_case', request['case'] == next(c for c in spec['cases'] if c['id'] == name))
            check(name + ':request_hashes', request['spec_sha256'] == PINS['spec.json'] and request['model_sha256'] == PINS['model_dfl_code_clip.onnx'] and
                  request['script_sha256'] == PINS['jetson_build_dfl_context_probe.py'] and request['runner_script_sha256'] == PINS['jetson_dfl_context_probe.py'])
            flags = request['flags']['get_flag']
            for flag, enabled in (('INT8', True), ('FP16', True), ('TF32', False), ('PREFER_PRECISION_CONSTRAINTS', False),
                                  ('OBEY_PRECISION_CONSTRAINTS', name.endswith('_obey'))):
                check(name + ':actual_flag:' + flag, flags[flag] == {'supported': True, 'enabled': enabled})
            check(name + ':native_common_settings', request['workspace_bytes_actual'] == 1024 ** 3 and request['avg_timing_iterations_actual'] == 1 and
                  request['profiling_verbosity_actual'] == 'ProfilingVerbosity.DETAILED' and request['no_layer_precision_or_output_type_setters_or_resets'])
            clips = request['clip_attributes']
            check(name + ':single_integer_Clip_correction', len(clips) == 1 and clips[0]['name'] == spec['clip_names'][0] and
                  clips[0]['after'] == {'alpha': -127.0, 'beta': 127.0} and clips[0]['explicit_importer_correction'])
            earlier, later = read(folder / 'parsed_constraints_before.json'), read(folder / 'parsed_constraints_after.json')
            constraint_records[name] = normalized_constraints(earlier)
            check(name + ':all_parser_constraints_retained', normalized_constraints(earlier) == normalized_constraints(later))
            errors = []
            requested_precision, requested_outputs = [], []
            for layer in earlier:
                check(name + ':parser_precision_query:' + layer['name'], layer['precision_is_set'].get('supported') is True)
                if layer['precision_is_set'].get('value'):
                    requested_precision.append({'name': layer['name'], 'precision': layer.get('precision')})
                    if 'precision' not in layer:
                        errors.append({'name': layer['name'], 'precision_error': layer.get('precision_value_error')})
                else:
                    check(name + ':unset_precision_value_absent:' + layer['name'], 'precision' not in layer)
                for value in layer['output_types']:
                    check(name + ':parser_output_query:' + layer['name'] + ':' + str(value['index']), value['output_type_is_set'].get('supported') is True)
                    if value['output_type_is_set'].get('value'):
                        requested_outputs.append({'name': layer['name'], 'index': value['index'], 'type': value.get('requested_output_type')})
                        if 'requested_output_type' not in value:
                            errors.append({'name': layer['name'], 'output_error': value.get('requested_output_type_error')})
                    else:
                        check(name + ':unset_output_value_absent:' + layer['name'] + ':' + str(value['index']), 'requested_output_type' not in value)
            check(name + ':no_parser_query_value_errors', not errors)
            report['parser_constraint_inventory'] = {'layers': len(earlier), 'requested_precision': requested_precision,
                'requested_output_types': requested_outputs, 'query_value_errors': errors, 'before_after_equal': True}
            report['actual_flags'] = flags
        build = read(folder / 'build_result.json')
        report['build'] = build
        if (folder / 'raw_outputs.npz').is_file():
            result = read(folder / 'result.json')
            check(name + ':successful_native_build', build['status'] == 'DFL_CONTEXT_ENGINE_SAVED' and cleanup['returncode'] == 0)
            check(name + ':engine_raw_hashes', sha(folder / 'model.engine') == build['engine_sha256'] == result['engine_sha256'] and
                  sha(folder / 'raw_outputs.npz') == result['raw_outputs_sha256'])
            inspector = read(folder / 'engine_layers.json')
            check(name + ':two_inspector_snapshots_equal', inspector == read(folder / 'runner_engine_layers.json'))
            convs = [r for r in inspector['Layers'] if 'Convolution' in r.get('LayerType', '')]
            check(name + ':raw_Conv_rows_retained', convs == read(folder / 'conv_rows.json'))
            indices = [i for i, row in enumerate(convs) if SOURCE in row.get('Name', '')]
            unique = len(convs) == 1 and len(indices) == 1
            saved_mapping = read(folder / 'conv_source_mapping.json')
            check(name + ':mapping_report_honest', saved_mapping['one_dfl_actual_convolution_uniquely_mapped'] == unique and
                  result['one_dfl_actual_convolution_uniquely_mapped'] == unique and saved_mapping['int8_output_required'] is False)
            mapping = {'actual_row_count': len(convs), 'matched_source_indices': indices, 'one_unique_source_row': unique,
                       'int8_output_required': False, 'all_actual_rows': convs}
            if unique:
                conv = convs[indices[0]]
                mapping.update({'input_formats': [r.get('Format/Datatype') for r in conv.get('Inputs', [])],
                    'weight_type': conv.get('Weights', {}).get('Type'),
                    'output_formats': [r.get('Format/Datatype') for r in conv.get('Outputs', [])],
                    'named_integer_tactic_present': bool(re.search(r'i8|s8|int8|imma', conv.get('TacticName', ''), re.I)),
                    'actual_input_INT8': bool(conv.get('Inputs')) and all('int8' in r.get('Format/Datatype', '').lower() for r in conv['Inputs']),
                    'actual_weight_INT8': conv.get('Weights', {}).get('Type', '').lower() == 'int8',
                    'actual_output_INT8_descriptive': bool(conv.get('Outputs')) and all('int8' in r.get('Format/Datatype', '').lower() for r in conv['Outputs']),
                    'tactic_name': conv.get('TacticName'), 'activation': conv.get('Activation'),
                    'does_not_prove_full_qall_adoption_or_Gate_pass': True})
            report['actual_DFL_convolution'] = mapping
            bindings = read(folder / 'runtime_bindings.json')
            desired = {r['name']: r for r in [spec['input_binding']] + spec['output_bindings_in_order']}
            check(name + ':runtime_binding_count', len(bindings) == len(desired) == 2 and len({r['name'] for r in bindings}) == 2)
            for binding in bindings:
                target = desired[binding['name']]
                check(name + ':runtime_FLOAT_LINEAR:' + binding['name'], binding['dtype'] == 'DataType.FLOAT' and binding['format'] == 'TensorFormat.LINEAR' and
                      binding['input'] == target['input'] and binding['shape'] == target['shape'])
            with np.load(folder / 'raw_outputs.npz', allow_pickle=False) as data:
                check(name + ':exact_two_raw_keys', set(data.files) == {'first', 'repeat'})
                first, repeat = data['first'].copy(), data['repeat'].copy()
            check(name + ':raw_DFL_distance_contract', first.dtype == repeat.dtype == np.float32 and
                  first.shape == repeat.shape == (1, 4, 8400) and np.isfinite(first).all() and np.isfinite(repeat).all())
            report['repeat_comparison'] = compare(first, repeat)
            report['CPU_difference_descriptive'] = compare(first, expected)
            verify_native_comparison(name + ':repeat', report['repeat_comparison'], result['repeat_comparison'])
            verify_native_comparison(name + ':CPU', report['CPU_difference_descriptive'], result['comparison_to_frozen_cpu_ort'])
            arrays[name] = first
        else:
            report['inference_run'] = False
            report['failure'] = read(folder / 'build_failure.json') if (folder / 'build_failure.json').is_file() else None
            check(name + ':failed_case_not_executed', not (folder / 'result.json').exists())
        case_reports[name] = report
    if len(request_records) == 2:
        a, b = request_records[CASES[0]], request_records[CASES[1]]
        changed = [k for k in a['flags']['get_flag'] if a['flags']['get_flag'][k] != b['flags']['get_flag'][k]]
        check('only_actual_OBEY_flag_differs', changed == ['OBEY_PRECISION_CONSTRAINTS'])
        check('same_model_and_parser_constraints_across_flags', a['model_sha256'] == b['model_sha256'] and
              constraint_records[CASES[0]] == constraint_records[CASES[1]])
    pair = compare(arrays[CASES[0]], arrays[CASES[1]]) if len(arrays) == 2 else {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'}
    verify_native_comparison('NONE_vs_OBEY', pair, summary['cross_native_comparisons']['code_clip_none_vs_code_clip_obey'])
    monitor, guard = read(raw / 'monitor/status.json'), read(raw / 'supplemental_group_guard.json')
    launch, exit_checks = read(raw / 'launch.json'), read(raw / 'all_child_exit_verification.json')
    check('one_native_boot_and_terminal_monitor', monitor['boot_id'] == guard['boot_id'] == launch['boot_id'] == exit_checks['boot_id'] == BOOT and monitor['status'] != 'RUNNING')
    check('saved_guard_group_empty', guard['status'] == 'COMPLETE' and guard['owned_group_fully_empty'] and not guard['final_group_members'])
    check('launch_monitor_private_group_identity', launch['monitor_pid'] == monitor['monitor_pid'] and launch['private_group'] == monitor['build_pid'] == guard['owned_pgid'] and launch['guard_pid'] == guard['guard_pid'])
    check('all_recorded_task_PIDs_not_live', all(not row['live_non_zombie'] for row in exit_checks['known_processes'].values()))
    check('saved_original_complete_guard_failed_unchanged', spec['original_complete_model_numerical_guard'] == summary['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED')
    check('all_raw_evidence_bytes_unchanged_by_review', fingerprint(raw) == before)
    report = {'schema': 'rampq.gate05.dfl_context_independent_native_review/v1',
        'reviewed_client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'SAVED_DFL_CONTEXT_OUTCOMES_INDEPENDENTLY_VERIFIED',
        'raw_root': str(raw), 'check_count': len(checks), 'checks': checks,
        'receipt_files': len(receipt['files']), 'remote_receipt_sha256': sha(raw / 'remote_receipt.json'),
        'cases': case_reports, 'cross_native_NONE_vs_OBEY': pair,
        'descriptive_reduction_platform_differences': reduction_differences,
        'fixture_lineage': lineage, 'successful_output_cases': len(arrays),
        'monitor_status': monitor['status'], 'monitor_seconds': monitor['elapsed_seconds'],
        'guard_intervention_performed': guard['intervention_performed'], 'all_owned_groups_empty': True,
        'saved_evidence_unchanged': True, 'new_CPU_or_native_inference_run_by_review': False,
        'output_is_DFL_distance_not_detection_boxes': True, 'actual_Conv_INT8_output_required': False,
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
        'full_qall_failure_resolved': False, 'dataset_evaluation_run': False, 'new_numeric_tolerance': False,
        'limitations': ['These saved contexts cannot identify a full-model failing node or prove OBEY causality elsewhere.',
                        'Corrected CPU upstream fixture is an internal diagnostic reference, not native data.',
                        'Successful batch completion or local raw comparison is not Gate, detection accuracy or performance proof.']}
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'check_count': len(checks), 'successful_output_cases': len(arrays),
                      'cases': {k: v['saved_status'] for k, v in case_reports.items()}, 'cross_native': pair,
                      'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
