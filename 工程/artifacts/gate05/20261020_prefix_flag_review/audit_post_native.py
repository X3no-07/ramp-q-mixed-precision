"""Independently verify a collected prefix flag batch using saved files only.

Never imports TensorRT, torch, or ONNX Runtime; never connects to the board.
The expected ZIP identity must be provided from the recorded native collector.
Only a fresh derived report outside the raw evidence tree is written.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / 'scripts'))
from verify_gate05_evidence_archive import verify_archive

PRE_REVIEW_SHA = 'bade9ebf0401c0b208bf1b2d9f871915bf41482bf1f42a287e6af959c0b0d10d'
REMOTE_ROOT = '/home/nvidia/rampq/gate05_prefix_flag_20261020/gate05_prefix_flag_probe_20261020'
BOOT = '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
CASE_IDS = ['original_none', 'original_obey', 'code_clip_none', 'code_clip_obey']
PAIRS = [(CASE_IDS[0], CASE_IDS[1]), (CASE_IDS[2], CASE_IDS[3]),
         (CASE_IDS[0], CASE_IDS[2]), (CASE_IDS[1], CASE_IDS[3])]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def stats(array):
    values = array.astype(np.float64)
    finite = np.isfinite(values)
    selected = values[finite]
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'all_finite': bool(finite.all()), 'nonfinite_count': int((~finite).sum()),
            'min': float(selected.min()) if selected.size else None,
            'max': float(selected.max()) if selected.size else None,
            'raw_bytes_sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def compare(actual, expected):
    assert actual.shape == expected.shape
    valid = np.isfinite(actual) & np.isfinite(expected)
    delta = np.abs(actual[valid].astype(np.float64) - expected[valid].astype(np.float64))
    return {'all_pairs_finite': bool(valid.all()), 'nonfinite_pair_count': int((~valid).sum()),
            'numeric_equal': bool(np.array_equal(actual, expected)),
            'byte_identical': bool(actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()),
            'different_elements': int(np.count_nonzero(actual != expected)),
            'max_abs': float(delta.max()) if delta.size else None,
            'mean_abs': float(delta.mean()) if delta.size else None,
            'rms_difference': float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
            'descriptive_only_no_new_pass_criterion': True}


def constraints_summary(rows):
    precisions, outputs = Counter(), Counter()
    errors = []
    for row in rows:
        state = row['precision_is_set']
        if not state['supported']:
            errors.append({'layer': row['name'], 'query': 'precision_is_set', 'state': state})
        elif state['value']:
            if 'precision' in row:
                precisions[row['precision']] += 1
            else:
                errors.append({'layer': row['name'], 'query': 'precision',
                               'state': row.get('precision_value_error')})
        else:
            assert 'precision' not in row
        for output in row['output_types']:
            state = output['output_type_is_set']
            if not state['supported']:
                errors.append({'layer': row['name'], 'output': output['index'], 'state': state})
            elif state['value']:
                if 'requested_output_type' in output:
                    outputs[output['requested_output_type']] += 1
                else:
                    errors.append({'layer': row['name'], 'output': output['index'],
                                   'requested_output_type_error': output.get('requested_output_type_error')})
            else:
                assert 'requested_output_type' not in output
            if 'tensor_query_error' in output:
                errors.append({'layer': row['name'], 'output': output['index'],
                               'tensor_query_error': output['tensor_query_error']})
    return {'layer_count': len(rows), 'explicit_layer_precision_counts': dict(precisions),
            'explicit_output_type_counts': dict(outputs), 'query_errors': errors}


def constraint_semantics(rows):
    """Compare actual requests, excluding wrapper-dependent .type metadata."""
    result = []
    for row in rows:
        values = {k: row[k] for k in ['index', 'name', 'precision_is_set',
                                      'precision', 'precision_value_error'] if k in row}
        values['outputs'] = [{k: item[k] for k in ['index', 'output_type_is_set',
                             'requested_output_type', 'requested_output_type_error'] if k in item}
                            for item in row['output_types']]
        result.append(values)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--expected-sha256', required=True)
    ap.add_argument('--expected-bytes', type=int, required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    evidence, output = args.evidence.resolve(), args.output.resolve()
    assert evidence.is_dir() and not output.exists() and output.parent.is_dir()
    assert evidence not in output.parents
    checks = []
    derived_reduction_differences = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)

    def check_saved_comparison(name, recomputed, saved):
        # Raw arrays, finite masks, counts, equality and extremal errors are
        # exact evidence. ARM/x86 NumPy FP64 mean/RMS reductions can differ in
        # their last arithmetic bit; retain both instead of inventing a new
        # numerical acceptance tolerance for this descriptive metadata.
        check(name + ':field_inventory', set(recomputed) == set(saved))
        for key in set(recomputed) - {'mean_abs', 'rms_difference'}:
            check(name + ':' + key, recomputed[key] == saved[key])
        for key in ['mean_abs', 'rms_difference']:
            if recomputed[key] != saved[key]:
                derived_reduction_differences.append({
                    'comparison': name, 'field': key, 'native_saved': saved[key],
                    'host_recomputed': recomputed[key],
                    'host_minus_native': recomputed[key] - saved[key],
                    'raw_arrays_modified': False,
                    'new_acceptance_tolerance_defined': False})

    verification = verify_archive(args.archive, args.expected_sha256, args.expected_bytes)
    check('archive_exact_receipt_verified', verification['status'] ==
          'ARCHIVE_AND_EXACT_RECEIPT_FILE_INVENTORY_VERIFIED')
    expected_inventory = {row['path']: row for row in verification['verified_files']}
    expected_inventory[verification['receipt_path']] = {
        'sha256': verification['receipt_sha256']}
    before = {p.relative_to(evidence).as_posix(): sha(p)
              for p in evidence.rglob('*') if p.is_file()}
    check('raw_tree_exact_receipt_inventory', set(before) == set(expected_inventory))
    for name, entry in expected_inventory.items():
        check('extracted_sha:' + name, before[name] == entry['sha256'])
        if 'bytes' in entry:
            check('extracted_bytes:' + name, (evidence / name).stat().st_size == entry['bytes'])

    review_path = Path(__file__).with_name('pre_native_review.json')
    check('pre_native_review_identity', sha(review_path) == PRE_REVIEW_SHA)
    pre = read(review_path)
    check('pre_native_no_launch_blockers', pre['launch_blockers'] == [])
    source_bundle = PROJECT / pre['bundle']
    for name, info in read(evidence / 'manifest.json')['files'].items():
        check('sealed_manifest:' + name,
              (evidence / name).stat().st_size == info['bytes'] and sha(evidence / name) == info['sha256'])
        check('source_bundle_byte_exact:' + name,
              (evidence / name).read_bytes() == (source_bundle / name).read_bytes())
    spec, protocol = read(evidence / 'spec.json'), read(evidence / 'protocol.json')
    for name, key in [('spec.json', 'spec_sha256'), ('manifest.json', 'manifest_sha256'),
                      ('protocol.json', 'protocol_sha256')]:
        check('signed_identity:' + name, sha(evidence / name) == pre['identities'][key])
    check('exact_frozen_cases', spec['cases'] == protocol['cases'])
    check('exact_case_order', spec['case_order'] == CASE_IDS)
    check('full_guard_failed_unchanged', spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED')
    check('no_dataset_or_complete_model', spec['dataset_evaluation_run'] is False and
          spec['complete_detector_executions'] == 0)
    for name, digest in pre['identities']['scripts'].items():
        check('signed_script:' + name, sha(evidence / name) == digest)
    input_array = np.load(evidence / spec['input_file'], allow_pickle=False)
    check('fixed_input_metadata', input_array.dtype == np.float32 and
          input_array.shape == (1, 3, 640, 640) and np.isfinite(input_array).all())
    check('fixed_input_hash', sha(evidence / spec['input_file']) == spec['input_file_sha256'])
    with np.load(evidence / spec['cpu_expected_outputs_file'], allow_pickle=False) as z:
        expected = {key: z[key].copy() for key in z.files}
    check('existing_expected_equal', expected['original_output'].tobytes() ==
          expected['code_clip_output'].tobytes())
    check('no_new_cpu_inference', spec['new_cpu_ort_or_torch_executions'] == 0)

    latest = read(evidence / 'latest_probe.json')
    relative_run = PurePosixPath(latest['run']).relative_to(PurePosixPath(REMOTE_ROOT))
    run = evidence.joinpath(*relative_run.parts)
    check('one_result_run', [p.resolve() for p in (evidence / 'results').iterdir() if p.is_dir()] == [run.resolve()])
    summary = read(run / 'summary.json')
    check('summary_frozen_case_order', summary['requested_case_order'] == CASE_IDS)
    check('summary_protected_guard', summary['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
          summary['dataset_evaluation_run'] is False)
    check('summary_no_new_numeric_pass', summary['batch_completion_is_not_numerical_pass'] is True)
    check('explicit_terminal_batch', summary['status'] in
          ('PREFIX_FLAG_BATCH_COMPLETED', 'PREFIX_FLAG_BATCH_ABORTED'))
    if summary['status'] == 'PREFIX_FLAG_BATCH_COMPLETED':
        check('completed_has_four_observations', summary['observed_case_count'] == 4 and
              list(summary['cases']) == CASE_IDS and summary['aborted'] is None)
    else:
        check('aborted_reason_recorded', isinstance(summary['aborted'], dict) and bool(summary['aborted']))

    cases, arrays, requests, constraints, child_exits = {}, {}, {}, {}, {}
    wrapper_metadata_differences = {}
    for case in spec['cases']:
        cid = case['id']
        out = run / cid
        if cid not in summary['cases']:
            check('unattempted_only_after_abort:' + cid, summary['status'] == 'PREFIX_FLAG_BATCH_ABORTED')
            check('unattempted_no_output:' + cid, not (out / 'raw_outputs.npz').exists())
            cases[cid] = {'status': 'UNATTEMPTED_AFTER_EXPLICIT_BATCH_ABORT'}
            continue
        request = read(out / 'request.json')
        check('case_request_frozen:' + cid, request['case'] == case and request['retries'] == 0 and
              request['max_child_seconds'] == 300 and request['spec_sha256'] == sha(evidence / 'spec.json'))
        command = request['command']
        check('case_command_one_owned_python_builder:' + cid,
              command[1:] == ['-u', REMOTE_ROOT + '/jetson_build_prefix_flag_probe.py',
                              '--root', REMOTE_ROOT, '--case', cid, '--output',
                              latest['run'] + '/' + cid + '/model.engine'])
        if (out / 'child_cleanup.json').exists():
            cleanup = read(out / 'child_cleanup.json')
            child_exits[cid] = cleanup
            check('child_terminated:' + cid, cleanup['direct_python_api_child_terminated'] is True and
                  cleanup['returncode'] is not None)
        parsed_path = out / 'parsed_network_request.json'
        if parsed_path.exists():
            req = read(parsed_path)
            requests[cid] = req
            check('parsed_request_case:' + cid, req['case'] == case and
                  req['model_sha256'] == spec['variants'][case['variant']]['model_sha256'])
            flag_values = req['flags']['get_flag']
            for flag, value in [('INT8', True), ('FP16', True), ('TF32', False),
                                ('PREFER_PRECISION_CONSTRAINTS', False),
                                ('OBEY_PRECISION_CONSTRAINTS', case['precision_constraint_flag'] == 'obey')]:
                check('actual_flag:' + cid + ':' + flag,
                      flag_values[flag] == {'supported': True, 'enabled': value})
            check('workspace_avg_actual:' + cid, req['workspace_bytes_actual'] == 1024 * 1024 * 1024 and
                  req['avg_timing_iterations_actual'] == 1 and 'DETAILED' in req['profiling_verbosity_actual'])
            before_constraints = read(out / 'parsed_constraints_before.json')
            after_constraints = read(out / 'parsed_constraints_after.json')
            check('parser_constraints_untouched:' + cid,
                  constraint_semantics(before_constraints) == constraint_semantics(after_constraints) and
                  req['parsed_constraints_before_after_identical'] is True and
                  req['no_layer_precision_or_output_type_setters_or_resets'] is True)
            wrapper_metadata_differences[cid] = [
                {'before': first, 'after': second}
                for first, second in zip(before_constraints, after_constraints) if first != second]
            check('constraint_layer_names_unique:' + cid,
                  len({row['name'] for row in before_constraints}) == len(before_constraints))
            constraints[cid] = before_constraints
            check('exact_named_clip_attributes:' + cid,
                  [row['name'] for row in req['clip_attributes']] == spec['clip_names'])
            for clip in req['clip_attributes']:
                corrected = case['apply_native_clip_attribute_correction']
                check('clip_correction_flag:' + cid + ':' + clip['name'],
                      clip['explicit_importer_correction'] is corrected)
                check('clip_values:' + cid + ':' + clip['name'],
                      clip['after'] == ({'alpha': -127.0, 'beta': 127.0} if corrected else clip['before']))

        observed = summary['cases'][cid]
        if observed['status'] != 'PREFIX_FLAG_FIXED_INPUT_OUTPUTS_SAVED':
            check('failed_case_no_raw:' + cid, not (out / 'raw_outputs.npz').exists())
            check('failed_case_no_success_result:' + cid, not (out / 'result.json').exists())
            if observed['status'] == 'BUILD_FAILED_OBSERVATION':
                failure = read(out / 'runner_build_failure.json')
                builder_failure = read(out / 'build_failure.json') if (out / 'build_failure.json').exists() else None
                check('compiler_failure_observation_exact:' + cid, failure == observed and
                      failure['returncode'] > 0 and failure['inference_run'] is False)
                check('failed_engine_not_executed:' + cid, not (out / 'model.engine').exists())
                cases[cid] = {'status': observed['status'], 'returncode': failure['returncode'],
                              'builder_failure': builder_failure,
                              'builder_log': (out / 'builder.log').read_text(errors='replace'),
                              'parser_constraints': constraints_summary(constraints[cid]) if cid in constraints else None,
                              'inference_run': False}
            else:
                cases[cid] = {'status': observed['status'], 'observation': observed,
                              'inference_run': False}
            continue

        result = read(out / 'result.json')
        build = read(out / 'build_result.json')
        check('successful_case_summary_matches:' + cid, result == observed)
        check('successful_build_result:' + cid, build['status'] == 'PREFIX_FLAG_ENGINE_SAVED' and
              build['phase'] == 'COMPLETE')
        check('successful_engine_identity:' + cid,
              sha(out / 'model.engine') == result['engine_sha256'] == build['engine_sha256'])
        check('successful_raw_identity:' + cid, sha(out / 'raw_outputs.npz') == result['raw_outputs_sha256'])
        check('successful_result_guard:' + cid, result['original_complete_model_numerical_guard'] ==
              'FAILED_UNCHANGED' and result['dataset_evaluation_run'] is False)
        check('successful_result_boot:' + cid, result['boot_id'] == BOOT)
        bindings = read(out / 'runtime_bindings.json')
        check('exact_runtime_binding_count:' + cid, len(bindings) == 2 and len({b['name'] for b in bindings}) == 2)
        target_bindings = {row['name']: row for row in [spec['input_binding']] + spec['output_bindings_in_order']}
        for binding in bindings:
            target = target_bindings[binding['name']]
            check('runtime_binding:' + cid + ':' + binding['name'], binding['input'] == target['input'] and
                  binding['shape'] == target['shape'] and 'FLOAT' in binding['dtype'].upper() and
                  'LINEAR' in binding['format'].upper())
        layers = read(out / 'engine_layers.json')
        check('builder_runner_inspectors_equal:' + cid, layers == read(out / 'runner_engine_layers.json'))
        convs = [row for row in layers.get('Layers', []) if 'Convolution' in row.get('LayerType', '')]
        check('one_actual_source_conv:' + cid,
              len(convs) == 1 and spec['conv_name'] in convs[0]['Name'])
        check('sole_conv_original_rows_retained:' + cid, convs == read(out / 'sole_convolution_rows.json'))
        conv = convs[0]
        input_formats = [row.get('Format/Datatype', '') for row in conv.get('Inputs', [])]
        output_formats = [row.get('Format/Datatype', '') for row in conv.get('Outputs', [])]
        weights = conv.get('Weights', {})
        formats_int8 = (bool(input_formats) and bool(output_formats) and
                        all('int8' in value.lower() for value in input_formats + output_formats) and
                        str(weights.get('Type', '')).lower() == 'int8')
        tactic = conv.get('TacticName', '')
        named_integer = bool(re.search(r'i8|s8|int8|imma', tactic, re.IGNORECASE))
        with np.load(out / 'raw_outputs.npz', allow_pickle=False) as z:
            check('exact_two_raw_keys:' + cid, set(z.files) == {'first', 'repeat'})
            first, repeat = z['first'].copy(), z['repeat'].copy()
        check('raw_shape_dtype:' + cid, first.shape == repeat.shape == (1, 16, 320, 320) and
              first.dtype == repeat.dtype == np.float32)
        expected_key = spec['variants'][case['variant']]['expected_output_npz_keys']['rampq_model_1_conv_a_dq']
        actual_stats, cpu_delta, repeat_delta = stats(first), compare(first, expected[expected_key]), compare(repeat, first)
        check('saved_stats_recomputed:' + cid, actual_stats == result['stats'])
        check_saved_comparison('saved_cpu_delta_recomputed:' + cid,
                               cpu_delta, result['comparison_to_frozen_cpu_ort'])
        check_saved_comparison('saved_repeat_delta_recomputed:' + cid,
                               repeat_delta, result['repeat_comparison'])
        arrays[cid] = first
        cases[cid] = {'status': result['status'], 'engine_sha256': sha(out / 'model.engine'),
                      'raw_outputs_sha256': sha(out / 'raw_outputs.npz'), 'raw_stats': actual_stats,
                      'cpu_descriptive_delta': cpu_delta, 'repeat_descriptive_delta': repeat_delta,
                      'parser_constraints': constraints_summary(constraints[cid]),
                      'actual_conv': {'original_row': conv, 'input_formats': input_formats,
                                      'weight_type': weights.get('Type'), 'output_formats': output_formats,
                                      'actual_input_weight_output_int8': formats_int8,
                                      'named_integer_tactic': named_integer,
                                      'strict_int8_evidence_for_this_sole_conv': formats_int8 and named_integer,
                                      'activation': conv.get('Activation'), 'tactic_name': tactic,
                                      'tactic_value': conv.get('TacticValue')},
                      'clip_rows': [row for row in layers.get('Layers', []) if
                                    any(name in row.get('Name', '') for name in spec['clip_names'])],
                      'inference_run': True}

    paired = {}
    for first, second in PAIRS:
        key = first + '_vs_' + second
        if first in arrays and second in arrays:
            comparison = compare(arrays[first], arrays[second])
            check_saved_comparison('cross_comparison_recomputed:' + key,
                                   comparison, summary['cross_native_comparisons'][key])
            paired[key] = comparison
        else:
            check('cross_comparison_unavailable:' + key, summary['cross_native_comparisons'][key] ==
                  {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'})
            paired[key] = {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'}
    for first, second in PAIRS[:2]:
        if first in requests and second in requests:
            left, right = requests[first], requests[second]
            lf, rf = dict(left['flags']['get_flag']), dict(right['flags']['get_flag'])
            lf.pop('OBEY_PRECISION_CONSTRAINTS'); rf.pop('OBEY_PRECISION_CONSTRAINTS')
            check('paired_flags_only_OBEY_differs:' + first, lf == rf)
            check('paired_parser_constraints_same:' + first,
                  constraint_semantics(constraints[first]) == constraint_semantics(constraints[second]))
            # Uncorrected INT8 Clip importer garbage can be nondeterministic;
            # compare effective corrected attributes, retain both raw befores.
            normalized = lambda rows: [{key: row[key] for key in
                                       ['name', 'after', 'explicit_importer_correction']} for row in rows]
            check('paired_effective_clip_attributes_same:' + first,
                  normalized(left['clip_attributes']) == normalized(right['clip_attributes']))
            for field in ['bindings', 'workspace_bytes_actual',
                          'avg_timing_iterations_actual', 'profiling_verbosity_actual', 'model_sha256']:
                check('paired_common_request:' + first + ':' + field, left[field] == right[field])

    monitor = read(evidence / 'monitor/status.json')
    guard = read(evidence / 'supplemental_group_guard.json')
    check('expected_monitor_identity', monitor['monitor_pid'] == 30071 and monitor['build_pid'] == 30076 and
          monitor['boot_id'] == BOOT and monitor['root'] == REMOTE_ROOT and monitor['max_seconds'] == 1500)
    check('terminal_monitor', monitor['status'] in ('BUILD_SCRIPT_OK', 'BUILD_SCRIPT_FAILED', 'STOPPED'))
    check('guard_identity_and_terminal', guard['guard_pid'] == 30078 and guard['boot_id'] == BOOT and
          guard['owned_pgid'] == guard['owned_sid'] == 30076 and guard['root'] == REMOTE_ROOT and
          guard['status'] == 'COMPLETE' and guard['owned_group_fully_empty'] is True)
    end_files = list(evidence.glob('*end_board_status.json'))
    check('one_end_board_status', len(end_files) == 1)
    end = read(end_files[0])
    check('end_boot_unchanged', end['boot_id'] == BOOT)
    check('owned_processes_gone_at_collection', all(not row['stat'] for row in end['owned_process_checks'].values()))
    check('no_trtexec_at_collection', end['trtexec_stdout'] == '')
    resource_rows = [json.loads(line) for line in (evidence / 'monitor/resources.jsonl').read_text().splitlines() if line]
    resources = {'sample_count': len(resource_rows),
                 'maximum_temperature_c': max((max(row['temperatures_c'].values()) for row in resource_rows), default=None),
                 'minimum_available_memory_mib': min((row['MemAvailable_kib'] / 1024 for row in resource_rows), default=None),
                 'monitor_reason': monitor.get('reason'), 'monitor_status': monitor['status'],
                 'elapsed_seconds_not_benchmark': monitor['elapsed_seconds'],
                 'guard_intervention_performed': guard['intervention_performed'],
                 'kernel_log': (evidence / 'monitor/kernel.log').read_text(errors='replace')}
    if summary['status'] == 'PREFIX_FLAG_BATCH_COMPLETED':
        check('completed_outer_monitor_ok', monitor['status'] == 'BUILD_SCRIPT_OK' and
              monitor['build_returncode'] == 0 and monitor['reason'] is None)
        check('all_four_python_build_children_exited', len(child_exits) == 4 and
              all(row['returncode'] == 0 for row in child_exits.values()))
    historical = {}
    old_root = PROJECT / 'artifacts/gate05/20261019_fused_prefix_native_evidence'
    old_zip = PROJECT / 'artifacts/gate05/20261019_fused_prefix_native_evidence.zip'
    if old_root.is_dir() and old_zip.is_file():
        old_verification = verify_archive(old_zip,
            'fc1ca5dbf68ad6214803d93ed716c2aa6d3ee62421a71509ac6f463ff1063a46', 2123956)
        check('historical_archive_exact_receipt', old_verification['status'] ==
              'ARCHIVE_AND_EXACT_RECEIPT_FILE_INVENTORY_VERIFIED')
        old_inventory = {row['path']: row for row in old_verification['verified_files']}
        for variant in ['original', 'code_clip']:
            candidates = list((old_root / 'results').glob('probe_*/' + variant + '/raw_outputs.npz'))
            check('one_historical_raw:' + variant, len(candidates) == 1)
            path = candidates[0]
            rel = path.relative_to(old_root).as_posix()
            check('historical_raw_receipt_identity:' + variant, sha(path) == old_inventory[rel]['sha256'])
            with np.load(path, allow_pickle=False) as z:
                old_first = z['first'].copy()
            for suffix in ['none', 'obey']:
                cid = variant + '_' + suffix
                if cid in arrays:
                    historical[cid] = compare(arrays[cid], old_first)
    after = {p.relative_to(evidence).as_posix(): sha(p) for p in evidence.rglob('*') if p.is_file()}
    check('all_raw_evidence_bytes_unchanged', before == after)
    all_success = len(arrays) == 4
    report = {'schema': 'rampq.gate05.prefix_flag_post_native_independent_review/v1',
              'reviewed_client_at': datetime.now().astimezone().isoformat(),
              'status': 'SAVED_PREFIX_FLAG_OUTCOMES_INDEPENDENTLY_VERIFIED',
              'evidence': str(evidence), 'archive_sha256': verification['archive_sha256'],
              'receipt_sha256': verification['receipt_sha256'],
              'verified_receipt_file_count': verification['verified_file_count'],
              'pre_native_review_sha256': PRE_REVIEW_SHA, 'check_count': len(checks), 'checks': checks,
              'cases': cases, 'cross_native_comparisons': paired, 'all_four_cases_succeeded': all_success,
              'historical_native_prefix_descriptive_comparisons': historical,
              'derived_reduction_metadata_differences': derived_reduction_differences,
              'constraint_wrapper_metadata_differences': wrapper_metadata_differences,
              'parser_constraint_equality_scope': 'All precision_is_set/value and output_type_is_set/value requests; wrapper layer_type strings are separate retained metadata, not precision constraints.',
              'pre_native_wording_clarification': 'The signed pre-native manual note before==after refers to precision/output request semantics. Complete snapshot metadata may differ because Clip wrapper casting changes Python-visible layer_type.',
              'small_context_failure_reproduction': 'NOT_REPRODUCED_IN_THIS_ONE_CONV_PREFIX' if all_success else
                  'SEE_EXACT_FAILED_OR_ABORTED_CASES_NOT_A_FULL_MODEL_CAUSE',
              'interpretation': 'Four small engines succeeding would show OBEY alone is insufficient to reproduce the full q_all error in this prefix. A local failure likewise does not alone identify the complete graph cause.',
              'resources': resources, 'monitor': monitor, 'guard_summary': {
                  'status': guard['status'], 'intervention_performed': guard['intervention_performed'],
                  'owned_group_fully_empty': guard['owned_group_fully_empty']},
              'four_builder_child_exits': child_exits,
              'terminal_process_evidence': 'Collector shows monitor and runner absent; all four direct builders have exit records; supplemental guard is COMPLETE with owned private group fully empty. Seven owned task PIDs are thus accounted for without a new live board query by this reviewer.',
              'raw_tree_unchanged': True, 'native_execution_run_by_reviewer': False,
              'new_cpu_inference_run_by_reviewer': False, 'dataset_evaluation_run': False,
              'original_complete_model_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
              'limits': 'Saved-file audit only, one source Conv and one fixed image. No full detector retry or new numerical tolerance; no complete model, causal recovery, dataset accuracy or performance PASS.'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'sha256': sha(output),
                      'check_count': len(checks), 'all_four_cases_succeeded': all_success,
                      'case_statuses': {name: row['status'] for name, row in cases.items()},
                      'cross_native_comparisons': paired}, indent=2))


if __name__ == '__main__':
    main()
