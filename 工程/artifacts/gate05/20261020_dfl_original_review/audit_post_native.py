"""Independent saved-only DFL review: never imports TRT, torch or ORT."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import onnx

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_gate05_evidence_archive import verify_archive

import numpy as np


PINS = {
    'protocol.json': '81200cd4900a35f0533c45c7542f6dacc353203e8338a09b47cd24465cc0f6ed',
    'spec.json': '99d7b3179d9c540c06331a2135de9d645b2421722edb20ff5268e53c7b56f1ae',
    'manifest.json': 'aa8aa071beb59e40e48f3c0022fcf187f430285b1651df200d7c554623f1d064',
    'source_original_full_model.onnx': '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d',
    'parent_dfl_spec.json': '934e075aa58dde1829b600d3d011f5ad51c4f1da1af78f7791944f492781b14d',
    'parent_dfl_manifest.json': '3836e82ed53afb14fe16d3b9431864c0c4a45fd0d3418c7c06a4f6fcd46ed1d0',
    'parent_dfl_checkpoint.json': '28edf81c0be1f72065a8952f84dc31330ba95709d50ed13aafc16a5bb56c602b',
    'model_dfl_original.onnx': 'b60be2c2d196ed1fb4e962446f374d9e53d066476172cfa1db415194b2306d90',
    'dfl_transpose_fixture.npy': 'cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab',
    'cpu_expected_outputs.npz': 'fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511',
    'jetson_dfl_original_probe.py': '91fc095c0e923bf45cacd8f32f0838e17d9ce39406a27628bccb93e1438e40e1',
    'jetson_build_dfl_original_probe.py': '2787202b71cc058cce7d00c3ea4c2490aa7dcdf2831088e3e01aea2b17e28fc1',
}
BOOT = '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
SOURCE = '/model.23/dfl/conv/Conv'
CASES = ('original_none', 'original_obey')


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


def named_explicit_constraints(rows):
    """Only requests shared across topology derivatives; indices may change."""
    result = []
    for row in rows:
        if not row['precision_is_set'].get('value') and not any(
                value['output_type_is_set'].get('value') for value in row['output_types']):
            continue
        assert row['precision_is_set']['supported'] and 'precision_value_error' not in row
        outputs = []
        for value in row['output_types']:
            assert value['output_type_is_set']['supported'] and 'requested_output_type_error' not in value
            outputs.append({'index': value['index'], 'is_set': value['output_type_is_set']['value'],
                            'type': value.get('requested_output_type') if value['output_type_is_set']['value'] else None})
        result.append({'name': row['name'], 'precision_is_set': row['precision_is_set']['value'],
                       'precision': row.get('precision') if row['precision_is_set']['value'] else None,
                       'outputs': outputs})
    return sorted(result, key=lambda row: row['name'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-bytes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.output.resolve()
    assert raw.is_dir() and output.parent.is_dir() and not output.exists() and raw not in output.parents
    archive_verification = verify_archive(args.archive, args.expected_sha256, args.expected_bytes)
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
        check('original_sealed_payload_bytes:' + rel, (raw / rel).read_bytes() ==
              (ROOT / 'artifacts/gate05/20261020_dfl_original_probe' / rel).read_bytes())
    for rel, digest in PINS.items():
        check('fixed_identity:' + rel, before[rel]['sha256'] == digest)
    spec, protocol = read(raw / 'spec.json'), read(raw / 'protocol.json')
    check('zero_new_CPU_execution', spec['new_CPU_execution_runs_this_stage'] ==
          protocol['CPU_execution_runs_this_stage'] == 0)
    check('signed_pre_native_review_identity', sha(Path(__file__).with_name('pre_native_review.json')) ==
          '62158ccefba1f732f88904bce8cf92c24f0992ec9dc706988770dfc78482a5ed')
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
        expected = archive['original_output'].copy()
        original = archive['original_output'].copy()
    check('saved_CPU_tail_original_corrected_byte_equal', expected.dtype == original.dtype == np.float32 and
          expected.shape == original.shape == (1, 4, 8400) and expected.tobytes() == original.tobytes() and np.isfinite(expected).all())
    runs = list((raw / 'results').glob('probe_*'))
    check('single_native_frozen_batch', len(runs) == 1)
    run = runs[0]
    summary = read(run / 'summary.json')
    check('saved_batch_case_order', tuple(summary['requested_case_order']) == CASES)
    arrays, case_reports, request_records, constraint_records, raw_constraint_records = {}, {}, {}, {}, {}
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
            check(name + ':request_hashes', request['spec_sha256'] == PINS['spec.json'] and request['model_sha256'] == PINS['model_dfl_original.onnx'] and
                  request['script_sha256'] == PINS['jetson_build_dfl_original_probe.py'] and request['runner_script_sha256'] == PINS['jetson_dfl_original_probe.py'])
            flags = request['flags']['get_flag']
            for flag, enabled in (('INT8', True), ('FP16', True), ('TF32', False), ('PREFER_PRECISION_CONSTRAINTS', False),
                                  ('OBEY_PRECISION_CONSTRAINTS', name.endswith('_obey'))):
                check(name + ':actual_flag:' + flag, flags[flag] == {'supported': True, 'enabled': enabled})
            check(name + ':native_common_settings', request['workspace_bytes_actual'] == 1024 ** 3 and request['avg_timing_iterations_actual'] == 1 and
                  request['profiling_verbosity_actual'] == 'ProfilingVerbosity.DETAILED' and request['no_layer_precision_or_output_type_setters_or_resets'])
            clips = request['clip_attributes']
            check(name + ':single_original_Clip_no_mutation', len(clips) == 1 and clips[0]['name'] == spec['clip_names'][0] and
                  clips[0]['after'] == clips[0]['before'] and clips[0]['explicit_importer_correction'] is False and
                  request['no_clip_attribute_setter_calls'] is True)
            earlier, later = read(folder / 'parsed_constraints_before.json'), read(folder / 'parsed_constraints_after.json')
            raw_constraint_records[name] = earlier
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
            check(name + ':successful_native_build', build['status'] == 'DFL_ORIGINAL_ENGINE_SAVED' and cleanup['returncode'] == 0)
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
            report['raw_output_stats'] = result['stats']
            report['raw_array_sha256_recomputed'] = hashlib.sha256(first.tobytes()).hexdigest()
            check(name + ':saved_raw_stats_correct', result['stats']['raw_bytes_sha256'] ==
                  report['raw_array_sha256_recomputed'] and result['stats']['min'] == float(first.min()) and
                  result['stats']['max'] == float(first.max()) and result['stats']['all_finite'] is True)
        else:
            report['inference_run'] = False
            report['failure'] = read(folder / 'build_failure.json') if (folder / 'build_failure.json').is_file() else None
            check(name + ':failed_case_not_executed', not (folder / 'result.json').exists())
            report['failed_case_engine_absent'] = not (folder / 'model.engine').exists()
            report['original_builder_error_lines'] = [line for line in
                (folder / 'builder.log').read_text(encoding='utf-8', errors='replace').splitlines()
                if '[E]' in line or 'RuntimeError:' in line]
        case_reports[name] = report
    if len(request_records) == 2:
        a, b = request_records[CASES[0]], request_records[CASES[1]]
        changed = [k for k in a['flags']['get_flag'] if a['flags']['get_flag'][k] != b['flags']['get_flag'][k]]
        check('only_actual_OBEY_flag_differs', changed == ['OBEY_PRECISION_CONSTRAINTS'])
        check('same_model_and_parser_constraints_across_flags', a['model_sha256'] == b['model_sha256'] and
              constraint_records[CASES[0]] == constraint_records[CASES[1]])
    pair = compare(arrays[CASES[0]], arrays[CASES[1]]) if len(arrays) == 2 else {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'}
    verify_native_comparison('NONE_vs_OBEY', pair, summary['cross_native_comparisons']['original_none_vs_original_obey'])
    # Cross-stage comparisons use old receipts and fixed identities; no old
    # engine is rerun, and topology plus importer correction is one factor.
    old_base = ROOT / 'artifacts/gate05/20261020_dfl_context_native'
    old = old_base / 'native_evidence'
    old_before = fingerprint(old)
    old_zip_verification = verify_archive(old_base / 'native_evidence.zip',
                                         protocol['parent_dfl_native_archive_sha256'])
    old_receipt = read(old / 'remote_receipt.json')
    check('old_receipt_exact_inventory', set(old_before) == set(old_receipt['files']) | {'remote_receipt.json'})
    for name, info in old_receipt['files'].items():
        check('old_receipt_member:' + name, old_before[name] == info)
    old_spec = read(old / 'spec.json')
    for key in ['input_file_sha256', 'cpu_expected_outputs_sha256', 'input_binding', 'output_bindings_in_order', 'conv_names']:
        check('shared_cross_stage_contract:' + key, spec[key] == old_spec[key])
    for name in ['dfl_transpose_fixture.npy', 'cpu_expected_outputs.npz', 'fixture_lineage.json']:
        check('shared_cross_stage_original_bytes:' + name, (raw / name).read_bytes() == (old / name).read_bytes())
    old_model = onnx.load(old / old_spec['variants']['code_clip']['model_file'])
    new_model = onnx.load(raw / spec['variants']['original']['model_file'])
    old_nodes, new_nodes = {n.name: n for n in old_model.graph.node}, {n.name: n for n in new_model.graph.node}
    old_init, new_init = {i.name: i for i in old_model.graph.initializer}, {i.name: i for i in new_model.graph.initializer}
    changed_node_names = {'rampq_model_23_dfl_conv_ActivationQ', 'rampq_model_23_dfl_conv_Clip'}
    changed_init_names = {'rampq_model_23_dfl_conv_clip_min', 'rampq_model_23_dfl_conv_clip_max'}
    check('cross_stage_graph_names_equal', set(old_nodes) == set(new_nodes) and set(old_init) == set(new_init))
    for name in set(old_nodes) - changed_node_names:
        check('cross_stage_shared_node_bytes:' + name, old_nodes[name].SerializeToString() == new_nodes[name].SerializeToString())
    for name in set(old_init) - changed_init_names:
        check('cross_stage_shared_weight_scale_zero_bytes:' + name,
              old_init[name].SerializeToString() == new_init[name].SerializeToString())
    old_run = next((old / 'results').glob('probe_*'))
    four_cells = {name: {'stage': 'new_original', 'outcome': summary['cases'][name]['status'],
                         'engine_present': (run / name / 'model.engine').exists(),
                         'raw_present': name in arrays,
                         'Clip_attributes': request_records[name]['clip_attributes']}
                  for name in CASES}
    all_explicit = {name: named_explicit_constraints(rows) for name, rows in raw_constraint_records.items()}
    all_requests = dict(request_records)
    old_summary = read(old_run / 'summary.json')
    old_arrays = {}
    for name in ['code_clip_none', 'code_clip_obey']:
        folder = old_run / name
        request = read(folder / 'parsed_network_request.json')
        all_requests[name] = request
        all_explicit[name] = named_explicit_constraints(read(folder / 'parsed_constraints_before.json'))
        row = {'stage': 'saved_corrected', 'outcome': old_summary['cases'][name]['status'],
               'engine_present': (folder / 'model.engine').exists(), 'raw_present': (folder / 'raw_outputs.npz').exists(),
               'Clip_attributes': request['clip_attributes'], 'build_result': read(folder / 'build_result.json')}
        if row['raw_present']:
            with np.load(folder / 'raw_outputs.npz', allow_pickle=False) as data:
                old_arrays[name] = data['first'].copy()
            row['CPU_descriptive_difference_recomputed'] = compare(old_arrays[name], expected)
        else:
            row['original_builder_error_lines'] = [line for line in
                (folder / 'builder.log').read_text(encoding='utf-8', errors='replace').splitlines() if '[E]' in line]
            check('old_failed_case_not_executed:' + name, not (folder / 'result.json').exists())
        four_cells[name] = row
    base_flags = all_requests['code_clip_none']['flags']['get_flag']
    for name, request in all_requests.items():
        diffs = [key for key in base_flags if base_flags[key] != request['flags']['get_flag'][key]]
        check('all_four_flags_only_OBEY_differs:' + name,
              diffs == (['OBEY_PRECISION_CONSTRAINTS'] if name.endswith('_obey') else []))
        for key in ['workspace_bytes_actual', 'avg_timing_iterations_actual', 'profiling_verbosity_actual', 'bindings']:
            check('all_four_common_request:' + name + ':' + key, request[key] == all_requests['code_clip_none'][key])
        check('all_four_named_QDQ_requests_equal:' + name, all_explicit[name] == all_explicit['code_clip_none'])
    cross_stage_raw = {}
    for first, second in [('original_none', 'code_clip_none'), ('original_obey', 'code_clip_obey')]:
        cross_stage_raw[first + '_vs_' + second] = (compare(arrays[first], old_arrays[second])
            if first in arrays and second in old_arrays else {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'})
    check('old_raw_evidence_unchanged', fingerprint(old) == old_before)
    monitor, guard = read(raw / 'monitor/status.json'), read(raw / 'supplemental_group_guard.json')
    launch, exit_checks = read(raw / 'launch.json'), read(raw / 'all_child_exit_verification.json')
    check('one_native_boot_and_terminal_monitor', monitor['boot_id'] == guard['boot_id'] == launch['boot_id'] == exit_checks['boot_id'] == BOOT and monitor['status'] != 'RUNNING')
    check('saved_guard_group_empty', guard['status'] == 'COMPLETE' and guard['owned_group_fully_empty'] and not guard['final_group_members'])
    check('launch_monitor_private_group_identity', launch['monitor_pid'] == monitor['monitor_pid'] and launch['private_group'] == monitor['build_pid'] == guard['owned_pgid'] and launch['guard_pid'] == guard['guard_pid'])
    check('all_recorded_task_PIDs_not_live', all(not row['live_non_zombie'] for row in exit_checks['known_processes'].values()))
    check('native_task_PIDs_frozen', monitor['monitor_pid'] == 33046 and monitor['build_pid'] == 33051 and guard['guard_pid'] == 33057)
    check('saved_original_complete_guard_failed_unchanged', spec['original_complete_model_numerical_guard'] == summary['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED')
    check('all_raw_evidence_bytes_unchanged_by_review', fingerprint(raw) == before)
    report = {'schema': 'rampq.gate05.dfl_original_independent_native_review/v1',
        'reviewed_client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'SAVED_ORIGINAL_DFL_OUTCOMES_INDEPENDENTLY_VERIFIED',
        'raw_root': str(raw), 'archive_verification': archive_verification, 'check_count': len(checks), 'checks': checks,
        'receipt_files': len(receipt['files']), 'remote_receipt_sha256': sha(raw / 'remote_receipt.json'),
        'cases': case_reports, 'cross_native_NONE_vs_OBEY': pair,
        'four_saved_local_cells': four_cells, 'cross_stage_raw_comparisons': cross_stage_raw,
        'cross_stage_named_explicit_QDQ_requests': all_explicit,
        'cross_stage_old_archive_verification': old_zip_verification,
        'cross_stage_factor_limit': 'Original topology versus code-Clip plus importer correction is a combined derivative factor. Four single observations in two sequential bounded stages, not randomized/repeated trials; no unique full-model cause or flag workaround is proven.',
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
