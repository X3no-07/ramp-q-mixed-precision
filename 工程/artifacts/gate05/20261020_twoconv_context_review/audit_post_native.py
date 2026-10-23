"""Independent saved-file audit of the bounded two-Conv context outcomes."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_gate05_evidence_archive import verify_archive

PRE_SHA = '6308333eba0c6b7c2f3753943e7ec8eb78b1546462fde1a63d1e89b34494f382'
REMOTE = '/home/nvidia/rampq/gate05_twoconv_context_20261020/gate05_twoconv_context_probe_20261020'
BOOT = '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
CASES = ['code_clip_none', 'code_clip_obey']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def difference(first, second):
    assert first.shape == second.shape and first.dtype == second.dtype == np.float32
    valid = np.isfinite(first) & np.isfinite(second)
    delta = np.abs(first[valid].astype(np.float64) - second[valid].astype(np.float64))
    return {'all_pairs_finite': bool(valid.all()), 'nonfinite_pair_count': int((~valid).sum()),
            'numeric_equal': bool(np.array_equal(first, second)),
            'byte_identical': first.tobytes() == second.tobytes(),
            'different_elements': int(np.count_nonzero(first != second)),
            'max_abs': float(delta.max()) if delta.size else None,
            'mean_abs': float(delta.mean()) if delta.size else None,
            'rms_difference': float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
            'descriptive_only_no_new_pass_criterion': True}


def constraint_signature(rows):
    selected = []
    for row in rows:
        value = {k: row[k] for k in ['index', 'name', 'precision_is_set', 'precision', 'precision_value_error'] if k in row}
        value['outputs'] = [{k: o[k] for k in ['index', 'output_type_is_set', 'requested_output_type',
                                             'requested_output_type_error'] if k in o} for o in row['output_types']]
        selected.append(value)
    return selected


def constraint_summary(rows):
    precisions, outputs, errors = Counter(), Counter(), []
    for row in rows:
        state = row['precision_is_set']
        if not state['supported'] or 'precision_value_error' in row:
            errors.append(row)
        elif state['value']:
            precisions[row['precision']] += 1
        else:
            assert 'precision' not in row
        for output in row['output_types']:
            state = output['output_type_is_set']
            if not state['supported'] or 'requested_output_type_error' in output or 'tensor_query_error' in output:
                errors.append(output)
            elif state['value']:
                outputs[output['requested_output_type']] += 1
            else:
                assert 'requested_output_type' not in output
    return {'parsed_layers': len(rows), 'precision_counts': dict(precisions),
            'output_type_counts': dict(outputs), 'metadata_errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-bytes', type=int, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.output.resolve()
    assert raw.is_dir() and not output.exists() and raw not in output.parents
    checks, reductions = [], []

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
        if not value:
            raise AssertionError(name)

    def saved_comparison(name, computed, saved):
        check(name + ':keys', set(computed) == set(saved))
        for key in set(computed) - {'mean_abs', 'rms_difference'}:
            check(name + ':' + key, computed[key] == saved[key])
        for key in ['mean_abs', 'rms_difference']:
            if computed[key] != saved[key]:
                reductions.append({'comparison': name, 'field': key, 'native_saved': saved[key],
                                   'host_recomputed': computed[key],
                                   'host_minus_native': computed[key] - saved[key],
                                   'new_tolerance': False, 'raw_changed': False})

    verification = verify_archive(args.archive, args.expected_sha256, args.expected_bytes)
    inventory = {r['path']: r for r in verification['verified_files']}
    inventory[verification['receipt_path']] = {'sha256': verification['receipt_sha256']}
    before = {p.relative_to(raw).as_posix(): sha(p) for p in raw.rglob('*') if p.is_file()}
    check('raw_exact_receipt_inventory', set(before) == set(inventory))
    for name, info in inventory.items():
        check('raw_receipt_sha:' + name, before[name] == info['sha256'])
        if 'bytes' in info:
            check('raw_receipt_bytes:' + name, (raw / name).stat().st_size == info['bytes'])
    pre_path = Path(__file__).with_name('pre_native_review.json')
    check('signed_pre_native_review', sha(pre_path) == PRE_SHA)
    pre = read(pre_path)
    bundle = ROOT / 'artifacts/gate05/20261020_twoconv_context_probe'
    for name, digest in pre['identities'].items():
        if name.endswith('.json') or name.endswith('.py'):
            check('signed_raw_identity:' + name, sha(raw / name) == digest)
    for name, info in read(raw / 'manifest.json')['files'].items():
        check('sealed_payload_identity:' + name,
              (raw / name).stat().st_size == info['bytes'] and sha(raw / name) == info['sha256'])
        check('sealed_payload_original_bytes:' + name, (raw / name).read_bytes() == (bundle / name).read_bytes())
    spec, protocol = read(raw / 'spec.json'), read(raw / 'protocol.json')
    check('cases_frozen', spec['cases'] == protocol['cases'] and spec['case_order'] == CASES)
    check('guard_dataset_scope', spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
          spec['dataset_evaluation_run'] is False and spec['complete_detector_executions'] == 0)
    with np.load(raw / spec['cpu_expected_outputs_file'], allow_pickle=False) as data:
        expected = {key: data[key].copy() for key in data.files}
    check('saved_CPU_reference_exact', expected['original_output'].tobytes() == expected['code_clip_output'].tobytes())
    latest = read(raw / 'latest_probe.json')
    relative = PurePosixPath(latest['run']).relative_to(REMOTE)
    run = raw.joinpath(*relative.parts)
    check('one_frozen_result_run', [p.resolve() for p in (raw / 'results').iterdir() if p.is_dir()] == [run.resolve()])
    summary = read(run / 'summary.json')
    check('summary_frozen_order', summary['requested_case_order'] == CASES)
    check('summary_guard_scope', summary['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
          summary['dataset_evaluation_run'] is False and summary['batch_completion_is_not_numerical_pass'] is True)
    cases, arrays, requests, constraints, child_exits = {}, {}, {}, {}, {}
    for case in spec['cases']:
        name = case['id']; folder = run / name
        if name not in summary['cases']:
            check('unattempted_only_safe_abort:' + name, summary['status'] == 'TWOCONV_CONTEXT_BATCH_ABORTED')
            cases[name] = {'status': 'UNATTEMPTED_AFTER_SAFE_ABORT'}
            continue
        request = read(folder / 'request.json')
        check('case_request:' + name, request['case'] == case and request['retries'] == 0 and
              request['max_child_seconds'] == 300 and request['spec_sha256'] == sha(raw / 'spec.json'))
        cleanup = read(folder / 'child_cleanup.json')
        child_exits[name] = cleanup
        check('builder_child_exited:' + name, cleanup['direct_python_api_child_terminated'] is True and cleanup['returncode'] is not None)
        parsed_file = folder / 'parsed_network_request.json'
        if parsed_file.exists():
            req = read(parsed_file); requests[name] = req
            check('parsed_request_identity:' + name, req['case'] == case and req['model_sha256'] ==
                  spec['variants']['code_clip']['model_sha256'] and req['spec_sha256'] == sha(raw / 'spec.json'))
            flags = req['flags']['get_flag']
            for flag, expected_value in [('INT8', True), ('FP16', True), ('TF32', False),
                                         ('PREFER_PRECISION_CONSTRAINTS', False),
                                         ('OBEY_PRECISION_CONSTRAINTS', name.endswith('_obey'))]:
                check('actual_flag:' + name + ':' + flag, flags[flag] == {'supported': True, 'enabled': expected_value})
            check('workspace_avg_actual:' + name, req['workspace_bytes_actual'] == 1024 * 1024 * 1024 and
                  req['avg_timing_iterations_actual'] == 1 and 'DETAILED' in req['profiling_verbosity_actual'])
            first, last = read(folder / 'parsed_constraints_before.json'), read(folder / 'parsed_constraints_after.json')
            check('parser_constraint_semantics_preserved:' + name, constraint_signature(first) == constraint_signature(last) and
                  req['no_layer_precision_or_output_type_setters_or_resets'] is True)
            constraints[name] = first
            check('three_exact_Clip_attributes:' + name, [c['name'] for c in req['clip_attributes']] == protocol['clip_names'])
            for clip in req['clip_attributes']:
                check('three_Clip_bounds_corrected:' + name + ':' + clip['name'], clip['explicit_importer_correction'] is True and
                      clip['after'] == {'alpha': -127.0, 'beta': 127.0})
        observed = summary['cases'][name]
        if observed['status'] != 'TWOCONV_CONTEXT_FIXED_INPUT_OUTPUTS_SAVED':
            check('failed_case_no_raw_execution:' + name, not (folder / 'raw_outputs.npz').exists() and not (folder / 'result.json').exists())
            cases[name] = {'status': observed['status'], 'observation': observed,
                           'build_result': read(folder / 'build_result.json') if (folder / 'build_result.json').exists() else None,
                           'compiler_log': (folder / 'builder.log').read_text(encoding='utf-8', errors='replace'),
                           'inference_run': False}
            continue
        result, build = read(folder / 'result.json'), read(folder / 'build_result.json')
        check('successful_case_original_result:' + name, result == observed and build['status'] == 'TWOCONV_CONTEXT_ENGINE_SAVED')
        check('engine_identity:' + name, sha(folder / 'model.engine') == result['engine_sha256'] == build['engine_sha256'])
        check('raw_output_identity:' + name, sha(folder / 'raw_outputs.npz') == result['raw_outputs_sha256'])
        check('result_same_boot:' + name, result['boot_id'] == BOOT)
        desired = {v['name']: v for v in [spec['input_binding']] + spec['output_bindings_in_order']}
        bindings = read(folder / 'runtime_bindings.json')
        check('runtime_two_exact_binding_names:' + name, len(bindings) == 2 and {v['name'] for v in bindings} == set(desired))
        for binding in bindings:
            target = desired[binding['name']]
            check('runtime_binding:' + name + ':' + binding['name'], binding['input'] == target['input'] and
                  binding['shape'] == target['shape'] and binding['dtype'] == 'DataType.FLOAT' and binding['format'] == 'TensorFormat.LINEAR')
        layers = read(folder / 'engine_layers.json')
        check('original_inspector_rows_repeated:' + name, layers == read(folder / 'runner_engine_layers.json'))
        convs = [v for v in layers['Layers'] if 'Convolution' in v.get('LayerType', '')]
        check('original_actual_conv_rows_preserved:' + name, convs == read(folder / 'conv_rows.json'))
        mapping = read(folder / 'conv_source_mapping.json')
        actual_convs = []
        for source in protocol['source_conv_names']:
            indices = [i for i, v in enumerate(convs) if source in v.get('Name', '')]
            check('unique_actual_source_mapping:' + name + ':' + source, len(indices) == 1)
            conv = convs[indices[0]]
            saved_mapping = next(v for v in mapping['mapping'] if v['source_conv_name'] == source)
            check('saved_source_mapping_correct:' + name + ':' + source, saved_mapping['actual_row_indices'] == indices and
                  saved_mapping['status'] == 'UNIQUE_SOURCE_MAPPING')
            inputs = [v.get('Format/Datatype', '') for v in conv.get('Inputs', [])]
            outputs = [v.get('Format/Datatype', '') for v in conv.get('Outputs', [])]
            weight_int8 = conv.get('Weights', {}).get('Type', '').lower() == 'int8'
            adopted_int8 = bool(inputs) and bool(outputs) and weight_int8 and all('int8' in v.lower() for v in inputs + outputs)
            integer_tactic = bool(re.search(r'i8|s8|int8|imma', conv.get('TacticName', ''), re.I))
            actual_convs.append({'source_conv': source, 'actual_row_index': indices[0], 'original_row': conv,
                                 'input_weight_output_int8': adopted_int8, 'named_integer_tactic': integer_tactic,
                                 'strict_integer_tactic_evidence': adopted_int8 and integer_tactic,
                                 'activation': conv.get('Activation'), 'tactic_name': conv.get('TacticName'),
                                 'clip_names_fused_in_this_conv': [clip for clip in protocol['clip_names'] if clip in conv.get('Name', '')]})
        check('two_distinct_actual_convolutions:' + name, len(convs) == 2 and
              len({v['actual_row_index'] for v in actual_convs}) == 2 and mapping['two_distinct_actual_convolutions_verified'] is True)
        with np.load(folder / 'raw_outputs.npz', allow_pickle=False) as data:
            check('exact_two_raw_repeats:' + name, set(data.files) == {'first', 'repeat'})
            first, repeat = data['first'].copy(), data['repeat'].copy()
        check('raw_float32_static_shape_finite:' + name, first.shape == repeat.shape == (1, 32, 160, 160) and
              first.dtype == repeat.dtype == np.float32 and np.isfinite(first).all() and np.isfinite(repeat).all())
        raw_stats = result['stats']
        check('saved_raw_stats_identity:' + name, raw_stats['shape'] == list(first.shape) and
              raw_stats['dtype'] == 'float32' and raw_stats['all_finite'] is True and raw_stats['nonfinite_count'] == 0 and
              raw_stats['raw_bytes_sha256'] == hashlib.sha256(first.tobytes()).hexdigest() and
              raw_stats['min'] == float(first.min()) and raw_stats['max'] == float(first.max()))
        cpu_delta, repeat_delta = difference(first, expected['code_clip_output']), difference(repeat, first)
        saved_comparison('CPU_descriptive_delta:' + name, cpu_delta, result['comparison_to_frozen_cpu_ort'])
        saved_comparison('repeat_delta:' + name, repeat_delta, result['repeat_comparison'])
        third_clip = protocol['clip_names'][2]
        third_rows = [v for v in layers['Layers'] if third_clip in v.get('Name', '')]
        cases[name] = {'status': result['status'], 'engine_sha256': sha(folder / 'model.engine'),
                       'raw_outputs_sha256': sha(folder / 'raw_outputs.npz'), 'raw_array_sha256': hashlib.sha256(first.tobytes()).hexdigest(),
                       'parser_constraints': constraint_summary(constraints[name]), 'actual_convolutions': actual_convs,
                       'third_Clip_actual_rows': third_rows,
                       'third_Clip_fused_with_second_actual_Conv': third_clip in actual_convs[1]['clip_names_fused_in_this_conv'],
                       'CPU_descriptive_delta': cpu_delta, 'repeat_delta': repeat_delta, 'inference_run': True}
        arrays[name] = first
    paired = {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'}
    if len(arrays) == 2:
        paired = difference(arrays[CASES[0]], arrays[CASES[1]])
        saved_comparison('NONE_OBEY_raw', paired, summary['cross_native_comparisons']['code_clip_none_vs_code_clip_obey'])
        check('paired_parser_constraint_semantics_same', constraint_signature(constraints[CASES[0]]) == constraint_signature(constraints[CASES[1]]))
        lf, rf = dict(requests[CASES[0]]['flags']['get_flag']), dict(requests[CASES[1]]['flags']['get_flag'])
        lf.pop('OBEY_PRECISION_CONSTRAINTS'); rf.pop('OBEY_PRECISION_CONSTRAINTS')
        check('only_OBEY_flag_differs', lf == rf)
        for field in ['model_sha256', 'bindings', 'workspace_bytes_actual', 'avg_timing_iterations_actual', 'profiling_verbosity_actual']:
            check('same_common_native_request:' + field, requests[CASES[0]][field] == requests[CASES[1]][field])
    monitor, guard = read(raw / 'monitor/status.json'), read(raw / 'supplemental_group_guard.json')
    check('monitor_owned_identity', monitor['monitor_pid'] == 31158 and monitor['build_pid'] == 31163 and
          monitor['root'] == REMOTE and monitor['boot_id'] == BOOT and monitor['max_seconds'] == 900)
    check('monitor_terminal', monitor['status'] != 'RUNNING')
    check('guard_owned_terminal_empty', guard['guard_pid'] == 31169 and guard['root'] == REMOTE and guard['boot_id'] == BOOT and
          guard['owned_pgid'] == guard['owned_sid'] == 31163 and guard['status'] == 'COMPLETE' and guard['owned_group_fully_empty'] is True)
    end_files = list(raw.glob('*end_board_status.json'))
    check('one_end_status', len(end_files) == 1)
    end = read(end_files[0])
    check('end_same_boot', end['boot_id'] == BOOT)
    check('monitor_runner_absent_at_collection', all(not v['stat'] for v in end['owned_process_checks'].values()))
    check('no_trtexec_at_collection', end['trtexec_stdout'] == '')
    if summary['status'] == 'TWOCONV_CONTEXT_BATCH_COMPLETED':
        check('two_completed_case_outcomes', summary['observed_case_count'] == 2 and list(summary['cases']) == CASES and summary['aborted'] is None)
        check('completed_monitor_success', monitor['status'] == 'BUILD_SCRIPT_OK' and monitor['build_returncode'] == 0 and monitor['reason'] is None)
    observations = [json.loads(line) for line in (raw / 'monitor/resources.jsonl').read_text(encoding='utf-8').splitlines() if line]
    resources = {'samples': len(observations), 'maximum_temperature_c': max(max(v['temperatures_c'].values()) for v in observations),
                 'minimum_available_memory_mib': min(v['MemAvailable_kib'] / 1024 for v in observations),
                 'monitor_elapsed_seconds_not_benchmark': monitor['elapsed_seconds'], 'reason': monitor.get('reason'),
                 'kernel_log': (raw / 'monitor/kernel.log').read_text(encoding='utf-8', errors='replace')}
    after = {p.relative_to(raw).as_posix(): sha(p) for p in raw.rglob('*') if p.is_file()}
    check('all_raw_bytes_unchanged', before == after)
    report = {'schema': 'rampq.gate05.twoconv_context_post_native_independent_review/v1',
              'reviewed_client_at': datetime.now().astimezone().isoformat(), 'status': 'SAVED_TWOCONV_CONTEXT_OUTCOMES_INDEPENDENTLY_VERIFIED',
              'check_count': len(checks), 'checks': checks, 'archive_sha256': verification['archive_sha256'],
              'receipt_sha256': verification['receipt_sha256'], 'receipt_files': verification['verified_file_count'],
              'pre_native_review_sha256': PRE_SHA, 'cases': cases, 'NONE_OBEY_raw_comparison': paired,
              'all_two_cases_succeeded': len(arrays) == 2,
              'full_error_reproduction_status': 'NOT_REPRODUCED_IN_THIS_CONTEXT' if len(arrays) == 2 else
                  'SEE_EXACT_RECORDED_ERROR_OR_ABORT_DO_NOT_INFER_FROM_MISSING_OUTPUT',
              'third_Clip_planned_fusion': read(raw / 'failure_inventory.json')['boundaries'][2]['optimized_clip_log'],
              'derived_FP64_reduction_metadata_differences': reductions,
              'resources': resources, 'monitor': monitor, 'guard': guard,
              'builder_child_exits': child_exits, 'raw_tree_unchanged': True,
              'reviewer_native_execution': False, 'reviewer_new_CPU_inference': False, 'reviewer_SSH': False,
              'original_complete_model_guard': 'FAILED_UNCHANGED', 'full_model_retry_run': False,
              'dataset_evaluation_run': False, 'gate05_status': 'INCOMPLETE',
              'limits': 'Saved two-Conv context only. Success does not validate full q_all; failure does not identify a new node as cause. No monotonic-prefix assumption, complete model retry, new tolerance, causal recovery, dataset or performance PASS.'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'sha256': sha(output),
                      'check_count': len(checks), 'cases': {k: v['status'] for k, v in cases.items()},
                      'NONE_OBEY_raw': paired}, indent=2))


if __name__ == '__main__':
    main()
