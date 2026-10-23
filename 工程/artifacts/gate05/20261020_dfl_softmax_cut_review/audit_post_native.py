"""Independent saved-only audit of Softmax cut; no ORT/TRT/torch imports."""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[3]
REVIEW = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_native'
RAW = BASE / 'native_evidence'
OLD_BASE = ROOT / 'artifacts/gate05/20261020_dfl_context_native'
OLD = OLD_BASE / 'native_evidence'
BUNDLE = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_probe'
CASES = ('code_clip_none', 'code_clip_obey')
SOURCE = '/model.23/dfl/conv/Conv'
BOOT = '692c7544-b8bd-415b-939c-b11cdbd9f7f0'
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_gate05_evidence_archive import verify_archive


def load_helpers(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pre = load_helpers('softmax_cut_pre_helpers', REVIEW / 'audit_pre_native.py')
prior = load_helpers('original_dfl_saved_helpers',
                     ROOT / 'artifacts/gate05/20261020_dfl_original_review/audit_post_native.py')
sha, read = pre.sha, pre.read
fingerprint, compare = prior.fingerprint, prior.compare
normalized_constraints = prior.normalized_constraints
named_constraints = prior.named_explicit_constraints


def main():
    output = BASE / 'independent_native_review.json'
    assert not output.exists() and RAW not in output.parents
    checks, reduction_notes = [], {}

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)

    before, old_before = fingerprint(RAW), fingerprint(OLD)
    archive = verify_archive(BASE / 'native_evidence.zip',
        '1b531ae8079a7563432e9702071ca8d384695f2a2624d80ab3da7798fa7215cc', 13823424)
    old_archive = verify_archive(OLD_BASE / 'native_evidence.zip',
        'a53750401ccdbb7c008c14e75cadec80a586a05f2baa32f5e985abdff527bd24', 21481431)
    for label, raw, snap in [('new', RAW, before), ('old', OLD, old_before)]:
        receipt = read(raw / 'remote_receipt.json')
        check(label + ':exact_receipt_inventory', set(snap) == set(receipt['files']) | {'remote_receipt.json'})
        for name, info in receipt['files'].items():
            check(label + ':receipt:' + name, snap[name] == info)
    receipt = read(RAW / 'remote_receipt.json')
    check('new_receipt_count_and_identity', len(receipt['files']) == 89 and sha(RAW / 'remote_receipt.json') ==
          '5538c67b45e6ba020bc095b4ad4eca8920e0cb8fdaf60b67d90a90e4b37d44b8')
    for name, digest in pre.PINS.items():
        check('hardpin:' + name, sha(RAW / name) == digest)
    for name, info in read(RAW / 'manifest.json')['files'].items():
        check('sealed_payload:' + name, before[name] == info)
        check('original_bundle_bytes:' + name, (RAW / name).read_bytes() == (BUNDLE / name).read_bytes())
    check('signed_pre_native_report_identity', sha(RAW / 'pre_native_review.json') ==
          sha(REVIEW / 'pre_native_review.json') == '7f31020bf9ba1ce2b66a61cf96dfb59ba6b4094b75a496b156d238e9dcf4cd86')
    spec, protocol = read(RAW / 'spec.json'), read(RAW / 'protocol.json')
    check('frozen_case_order', tuple(c['id'] for c in spec['cases']) == CASES and spec['cases'] == protocol['cases'])
    check('fixed_execution_budgets', spec['max_seconds'] == 900 and spec['max_build_seconds'] == 300 and
          spec['maximum_attempts_per_case'] == 1 and spec['maximum_native_builds'] == 2 and
          spec['maximum_native_inferences'] == 4 and spec['executions_per_successful_case'] == 2)
    check('only_preparation_two_CPU_runs_no_upstream_rerun', spec['new_CPU_execution_runs_this_stage'] == 2 and
          spec['new_72Conv_upstream_runs'] == 0 and spec['complete_detector_executions'] == 0)
    launch_receipt = read(RAW / 'native_input_receipt.json')
    for key, filename in [('spec_sha256', 'spec.json'), ('manifest_sha256', 'manifest.json'),
                          ('pre_native_review_sha256', 'pre_native_review.json')]:
        check('launch_receipt_identity:' + key, launch_receipt[key] == sha(RAW / filename))
    for name, digest in launch_receipt['helper_identities'].items():
        check('launch_helper_identity:' + name, sha(RAW / name) == digest)
    check('launch_input_archive_pin', launch_receipt['archive_sha256'] ==
          'aeff82c52ff3b2ec1d83b4b06c21f014b30699a1e428a0c5fb37bcd3f6cb25dd')
    for first, last in [('source_hashes_before.json', 'source_hashes_after.json'),
                         ('seal_native_source_hashes_before.json', 'seal_native_source_hashes_after.json')]:
        earlier, later = read(RAW / first), read(RAW / last)
        check('source_snapshots_equal:' + first, earlier == later)
        for name, info in earlier.items():
            check('source_still_exact:' + first + ':' + name,
                  (ROOT / name).stat().st_size == info['bytes'] and sha(ROOT / name) == info['sha256'])
    old_spec = read(OLD / 'spec.json')
    model = onnx.load(RAW / spec['variants']['code_clip']['model_file'])
    old_model = onnx.load(OLD / old_spec['variants']['code_clip']['model_file'])
    generator = onnx.load(RAW / 'model_dfl_source_softmax.onnx')
    for graph in [model, old_model, generator]:
        onnx.checker.check_model(graph, full_check=True)
    new_nodes = {n.name: n for n in model.graph.node}
    old_nodes = {n.name: n for n in old_model.graph.node}
    check('cut_exact_8nodes_7init_1conv_no_Softmax', len(new_nodes) == 8 and len(model.graph.initializer) == 7 and
          sum(n.op_type == 'Conv' for n in model.graph.node) == 1 and not any(n.op_type == 'Softmax' for n in model.graph.node))
    check('generator_exact_one_Softmax_zero_init', len(generator.graph.node) == 1 and
          generator.graph.node[0].op_type == 'Softmax' and len(generator.graph.initializer) == 0 and
          generator.graph.node[0].SerializeToString() == old_nodes['/model.23/dfl/Softmax'].SerializeToString())
    check('cut_exact_old_tail_node_inventory', set(new_nodes) == set(old_nodes) - {'/model.23/dfl/Softmax'})
    for name, node in new_nodes.items():
        check('cut_node_unchanged_from_parent:' + name, node.SerializeToString() == old_nodes[name].SerializeToString())
        check('cut_serialized_node_pin:' + name, hashlib.sha256(node.SerializeToString()).hexdigest() ==
              spec['variants']['code_clip']['selected_node_serialized_sha256'][name])
    old_initializers = {i.name: i for i in old_model.graph.initializer}
    check('cut_exact_initializer_inventory', {i.name for i in model.graph.initializer} == set(old_initializers))
    for init in model.graph.initializer:
        check('cut_initializer_unchanged:' + init.name, init.SerializeToString() == old_initializers[init.name].SerializeToString())
    check('cut_input_is_old_Softmax_output', model.graph.input[0].name == generator.graph.output[0].name ==
          old_nodes['/model.23/dfl/Softmax'].output[0] == '/model.23/dfl/Softmax_output_0')
    lineage = read(RAW / 'fixture_lineage.json')
    probability = np.load(RAW / 'dfl_softmax_probabilities.npy', allow_pickle=False)
    logits = np.load(RAW / 'saved_dfl_logits.npy', allow_pickle=False)
    check('probability_FLOAT_contract', probability.dtype == np.float32 and probability.shape == (1, 16, 4, 8400) and
          np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)))
    check('probability_raw_identity', hashlib.sha256(probability.tobytes()).hexdigest() ==
          lineage['fixture']['raw_array_bytes_sha256'] == 'd4c4b21e1d1a2e025d5b3eabaade7e99f3093c0cf7511795f9477a4ffd069ae7')
    check('new_input_different_from_old_logits', probability.tobytes() != logits.tobytes() and
          (RAW / 'saved_dfl_logits.npy').read_bytes() == (OLD / 'dfl_transpose_fixture.npy').read_bytes())
    check('CPU_generator_once_not_native_probabilities', lineage['new_generator_CPU_runs'] == 1 and
          lineage['new_72Conv_or_old_DFL_or_full_detector_runs'] == 0 and lineage['provider'] == 'CPUExecutionProvider' and
          lineage['graph_optimization'] == 'ORT_DISABLE_ALL' and lineage['source_Softmax_axis'] == 1)
    check('same_frozen_CPU_npz_bytes', (RAW / 'cpu_expected_outputs.npz').read_bytes() ==
          (OLD / 'cpu_expected_outputs.npz').read_bytes())
    with np.load(RAW / 'cpu_expected_outputs.npz', allow_pickle=False) as data:
        cpu = data['code_clip_output'].copy()
        check('old_original_and_corrected_CPU_raw_equal', data['original_output'].tobytes() == cpu.tobytes())
    check('saved_new_composed_CPU_equal_old_CPU_reference', np.load(RAW / 'cpu_actual_code_clip.npy', allow_pickle=False).tobytes() ==
          np.load(RAW / 'saved_cpu_reference.npy', allow_pickle=False).tobytes() == cpu.tobytes())

    def verify_comparison(name, actual, reported):
        check(name + ':field_set', set(actual) == set(reported))
        reduction_notes[name] = {}
        for key, value in actual.items():
            if key in ('mean_abs', 'rms_difference'):
                reduction_notes[name][key] = {'host': value, 'native': reported[key],
                    'host_minus_native': value - reported[key] if value is not None else None,
                    'descriptive_only_no_new_tolerance': True}
            else:
                check(name + ':' + key, value == reported[key])

    runs, old_runs = list((RAW / 'results').glob('probe_*')), list((OLD / 'results').glob('probe_*'))
    check('one_new_and_one_saved_parent_batch', len(runs) == len(old_runs) == 1)
    run, old_run = runs[0], old_runs[0]
    summary = read(run / 'summary.json')
    check('batch_completed_two_observations_only', summary['status'] == 'DFL_SOFTMAX_CUT_BATCH_COMPLETED' and
          summary['requested_case_order'] == list(CASES) and summary['observed_case_count'] == 2 and
          summary['successful_output_case_count'] == 1 and summary['aborted'] is None)
    cases, requests, explicit, constraints = {}, {}, {}, {}
    arrays, success_rows = {}, {}
    signature_fragments = ['optimizer.cpp::filterQDQFormats::4421',
        'Assertion !n->candidateRequirements.empty() failed. All of the candidates were removed',
        'builder.cpp::buildSerializedNetwork::751', 'Assertion engine != nullptr failed.']
    for name in CASES:
        folder = run / name
        request = read(folder / 'parsed_network_request.json')
        requests[name] = request
        check(name + ':case_frozen', request['case'] == next(c for c in spec['cases'] if c['id'] == name))
        for key, filename in [('spec_sha256', 'spec.json'), ('model_sha256', 'model_dfl_softmax_cut_code_clip.onnx'),
                              ('script_sha256', 'jetson_build_dfl_softmax_cut_probe.py'),
                              ('runner_script_sha256', 'jetson_dfl_softmax_cut_probe.py'),
                              ('fixture_sha256', 'dfl_softmax_probabilities.npy')]:
            check(name + ':request_identity:' + key, request[key] == sha(RAW / filename))
        check(name + ':no_native_Softmax', request['native_softmax_layer_count'] == 0 and
              request['input_role'] == 'CPU_SOFTMAX_PROBABILITIES_NOT_LOGITS')
        for flag, enabled in [('INT8', True), ('FP16', True), ('TF32', False), ('PREFER_PRECISION_CONSTRAINTS', False),
                              ('OBEY_PRECISION_CONSTRAINTS', name.endswith('_obey'))]:
            check(name + ':actual_flag:' + flag, request['flags']['get_flag'][flag] == {'supported': True, 'enabled': enabled})
        check(name + ':actual_common_settings', request['workspace_bytes_actual'] == 1024 ** 3 and
              request['avg_timing_iterations_actual'] == 1 and request['profiling_verbosity_actual'] == 'ProfilingVerbosity.DETAILED' and
              request['no_layer_precision_or_output_type_setters_or_resets'])
        clips = request['clip_attributes']
        check(name + ':single_code_Clip_explicit_correction', len(clips) == 1 and clips[0]['name'] == spec['clip_names'][0] and
              clips[0]['explicit_importer_correction'] is True and clips[0]['after'] == {'alpha': -127.0, 'beta': 127.0})
        earlier, later = [read(folder / ('parsed_constraints_' + suffix + '.json')) for suffix in ('before', 'after')]
        constraints[name] = normalized_constraints(earlier)
        explicit[name] = named_constraints(earlier)
        check(name + ':all_parser_requests_retained', constraints[name] == normalized_constraints(later))
        check(name + ':parser12layers_noSoftmax', len(earlier) == 12 and not any('SOFTMAX' in r['layer_type'].upper() for r in earlier))
        precision, outputs = [], []
        for row in earlier:
            check(name + ':precision_query_supported:' + row['name'], row['precision_is_set'].get('supported') is True)
            check(name + ':precision_value_iff_set:' + row['name'], ('precision' in row) == row['precision_is_set'].get('value') and 'precision_value_error' not in row)
            if row['precision_is_set']['value']:
                precision.append((row['name'], row['precision']))
            for value in row['output_types']:
                label = name + ':' + row['name'] + ':' + str(value['index'])
                check(label + ':output_query_supported', value['output_type_is_set'].get('supported') is True)
                check(label + ':output_value_iff_set', ('requested_output_type' in value) == value['output_type_is_set'].get('value') and 'requested_output_type_error' not in value)
                if value['output_type_is_set']['value']:
                    outputs.append((row['name'], value['requested_output_type']))
        check(name + ':exact_two_DQ_precision_INT8', sorted(precision) == sorted([
            ('rampq_model_23_dfl_conv_ActivationDQ', 'DataType.INT8'), ('rampq_model_23_dfl_conv_WeightDQ', 'DataType.INT8')]))
        check(name + ':exact_two_Q_output_INT8', sorted(outputs) == sorted([
            ('rampq_model_23_dfl_conv_ActivationQ', 'DataType.INT8'), ('rampq_model_23_dfl_conv_WeightQ', 'DataType.INT8')]))
        cleanup, build = read(folder / 'child_cleanup.json'), read(folder / 'build_result.json')
        check(name + ':direct_child_terminal', cleanup['direct_python_api_child_terminated'] and cleanup['returncode'] is not None)
        report = {'summary_status': summary['cases'][name]['status'], 'build': build, 'cleanup': cleanup,
            'actual_flags': request['flags']['get_flag'], 'clip_attributes': clips,
            'parser_constraint_inventory': {'layers': len(earlier), 'precision_requests': precision,
                                             'output_requests': outputs, 'before_after_equal': True}}
        if name == 'code_clip_none':
            result = read(folder / 'result.json')
            check(name + ':success_engine_saved', build['status'] == 'DFL_SOFTMAX_CUT_ENGINE_SAVED' and cleanup['returncode'] == 0)
            check(name + ':success_engine_and_raw_identity', sha(folder / 'model.engine') == build['engine_sha256'] == result['engine_sha256'] and
                  sha(folder / 'raw_outputs.npz') == result['raw_outputs_sha256'])
            inspector = read(folder / 'engine_layers.json')
            check(name + ':inspector_snapshots_equal', inspector == read(folder / 'runner_engine_layers.json'))
            check(name + ':inspector_has_noSoftmax', not any('softmax' in row.get('LayerType', '').lower() for row in inspector['Layers']))
            convs = [r for r in inspector['Layers'] if 'Convolution' in r.get('LayerType', '')]
            check(name + ':actual_sole_conv_rows_exact', len(convs) == 1 and SOURCE in convs[0]['Name'] and convs == read(folder / 'conv_rows.json'))
            conv = convs[0]
            mapping = read(folder / 'conv_source_mapping.json')
            check(name + ':actual_source_mapping_exact', mapping['mapping'] == [{'source_conv_name': SOURCE, 'actual_row_indices': [0], 'status': 'UNIQUE_SOURCE_MAPPING'}] and
                  mapping['one_dfl_actual_convolution_uniquely_mapped'] and result['one_dfl_actual_convolution_uniquely_mapped'] and
                  mapping['int8_output_required'] is False and mapping['output_qdq_present'] is False)
            report['actual_Conv'] = conv
            report['actual_Conv_descriptive'] = {'input_formats': [r['Format/Datatype'] for r in conv['Inputs']],
                'weight_type': conv['Weights']['Type'], 'output_formats': [r['Format/Datatype'] for r in conv['Outputs']],
                'named_integer_tactic_present': bool(re.search('int8|imma|i8|s8', conv.get('TacticName', ''), re.I)),
                'int8_output_required': False}
            success_rows['new'] = conv
            bindings = read(folder / 'runtime_bindings.json')
            check(name + ':binding_inventory', len(bindings) == 2 and len({b['name'] for b in bindings}) == 2)
            for binding, desired in zip(bindings, [spec['input_binding']] + spec['output_bindings_in_order']):
                check(name + ':FLOAT_LINEAR:' + binding['name'], binding['name'] == desired['name'] and binding['input'] == desired['input'] and
                      binding['shape'] == desired['shape'] and binding['dtype'] == 'DataType.FLOAT' and binding['format'] == 'TensorFormat.LINEAR')
            with np.load(folder / 'raw_outputs.npz', allow_pickle=False) as data:
                check(name + ':exact_raw_keys', set(data.files) == {'first', 'repeat'})
                first, repeat = data['first'].copy(), data['repeat'].copy()
            check(name + ':raw_FLOAT_DFL_contract', first.dtype == repeat.dtype == np.float32 and first.shape == repeat.shape == (1, 4, 8400) and
                  np.isfinite(first).all() and np.isfinite(repeat).all())
            repeat_diff, cpu_diff = compare(first, repeat), compare(first, cpu)
            verify_comparison(name + ':repeat', repeat_diff, result['repeat_comparison'])
            verify_comparison(name + ':CPU', cpu_diff, result['comparison_to_frozen_cpu_ort'])
            check(name + ':repeat_byte_exact', repeat_diff['byte_identical'] and repeat_diff['different_elements'] == 0)
            check(name + ':raw_stats_exact', result['stats']['raw_bytes_sha256'] == hashlib.sha256(first.tobytes()).hexdigest() and
                  result['stats']['all_finite'] and result['stats']['nonfinite_count'] == 0 and
                  result['stats']['min'] == float(first.min()) and result['stats']['max'] == float(first.max()))
            report.update({'repeat_comparison': repeat_diff, 'CPU_comparison_descriptive': cpu_diff, 'raw_stats': result['stats']})
            arrays['new'] = first
        else:
            text = (folder / 'builder.log').read_text(encoding='utf-8', errors='strict')
            for fragment in signature_fragments:
                check(name + ':failure_signature:' + fragment, fragment in text)
            check(name + ':compiler_failure_not_inference', build['status'] == 'DFL_SOFTMAX_CUT_BUILD_FAILED' and
                  build['phase'] == 'BUILDING' and build['error'] == 'TensorRT returned no serialized engine' and
                  build['inference_run'] is False and cleanup['returncode'] == 1)
            for filename in ['model.engine', 'raw_outputs.npz', 'result.json', 'engine_layers.json', 'conv_rows.json', 'runtime_bindings.json']:
                check(name + ':absent_failed_artifact:' + filename, not (folder / filename).exists())
            report['failure_error_lines'] = [line for line in text.splitlines() if '[E]' in line or 'RuntimeError:' in line]
            report['inference_run'] = False
        cases[name] = report
    check('new_only_OBEY_flag_differs', [key for key in requests[CASES[0]]['flags']['get_flag'] if requests[CASES[0]]['flags']['get_flag'][key] !=
          requests[CASES[1]]['flags']['get_flag'][key]] == ['OBEY_PRECISION_CONSTRAINTS'])
    check('new_requests_same_named_QDQ_and_all_constraints', constraints[CASES[0]] == constraints[CASES[1]] and explicit[CASES[0]] == explicit[CASES[1]])
    check('new_crossflag_raw_unavailable', summary['cross_native_comparisons']['code_clip_none_vs_code_clip_obey'] ==
          {'status': 'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'})

    old_cells = {}
    for name in CASES:
        folder = old_run / name
        request = read(folder / 'parsed_network_request.json')
        check('old_shared_named_QDQ:' + name, named_constraints(read(folder / 'parsed_constraints_before.json')) == explicit[name])
        for key in ['flags', 'workspace_bytes_actual', 'avg_timing_iterations_actual', 'profiling_verbosity_actual', 'no_layer_precision_or_output_type_setters_or_resets']:
            check('old_common_native_request:' + name + ':' + key, request[key] == requests[name][key])
        check('old_shared_corrected_Clip_after:' + name, request['clip_attributes'][0]['after'] == requests[name]['clip_attributes'][0]['after'])
        check('old_input_is_different_named_boundary:' + name, request['bindings'][0]['name'] == '/model.23/dfl/Transpose_output_0' and
              requests[name]['bindings'][0]['name'] == '/model.23/dfl/Softmax_output_0')
        row = {'build': read(folder / 'build_result.json'), 'native_input_file': old_spec['input_file'],
               'native_input_sha256': old_spec['input_file_sha256'], 'named_QDQ_requests': named_constraints(read(folder / 'parsed_constraints_before.json'))}
        if name == 'code_clip_none':
            with np.load(folder / 'raw_outputs.npz', allow_pickle=False) as data:
                arrays['old'] = data['first'].copy()
                row['repeat_comparison_recomputed'] = compare(data['first'], data['repeat'])
            row['CPU_comparison_recomputed'] = compare(arrays['old'], cpu)
            old_convs = read(folder / 'conv_rows.json')
            check('old_one_Conv_source_mapping', len(old_convs) == 1 and SOURCE in old_convs[0]['Name'])
            success_rows['old'] = old_convs[0]
            row['actual_Conv'] = old_convs[0]
        else:
            text = (folder / 'builder.log').read_text(encoding='utf-8', errors='strict')
            for fragment in signature_fragments:
                check('old_same_OBEY_failure_signature:' + fragment, fragment in text)
            for filename in ['model.engine', 'raw_outputs.npz', 'result.json']:
                check('old_failed_artifact_absent:' + filename, not (folder / filename).exists())
            row['failure_error_lines'] = [line for line in text.splitlines() if '[E]' in line]
        old_cells[name] = row
    cross = compare(arrays['new'], arrays['old'])
    check('new_NONE_vs_old_corrected_NONE_raw_byte_exact', cross['byte_identical'] and cross['different_elements'] == 0 and cross['max_abs'] == 0)
    for key in ['Inputs', 'Outputs', 'Weights', 'TacticName', 'Activation', 'LayerType']:
        check('same_actual_Conv_descriptor:' + key, success_rows['new'][key] == success_rows['old'][key])
    for label, conv in success_rows.items():
        check(label + ':actual_INT8_input_weight_FP32_output', len(conv['Inputs']) == len(conv['Outputs']) == 1 and
              'int8' in conv['Inputs'][0]['Format/Datatype'].lower() and conv['Weights']['Type'] == 'Int8' and
              'FP32' in conv['Outputs'][0]['Format/Datatype'] and conv['Weights']['Count'] == 16 and conv['Activation'] == 'NONE')
    monitor, guard, launch, exit_record = [read(RAW / name) for name in
        ['monitor/status.json', 'supplemental_group_guard.json', 'launch.json', 'all_child_exit_verification.json']]
    check('same_boot_terminal_records', monitor['boot_id'] == guard['boot_id'] == launch['boot_id'] == exit_record['boot_id'] == BOOT)
    check('monitor_terminal_batch_not_inference_pass', monitor['status'] == 'BUILD_SCRIPT_OK' and monitor['build_returncode'] == 0)
    check('guard_no_intervention_group_empty', guard['status'] == 'COMPLETE' and not guard['intervention_performed'] and
          guard['owned_group_fully_empty'] and not guard['final_group_members'] and not guard['termination_actions'])
    check('task_monitor_private_guard_ids', monitor['monitor_pid'] == launch['monitor_pid'] == 37496 and
          monitor['build_pid'] == launch['private_group'] == guard['owned_pgid'] == 37501 and guard['guard_pid'] == launch['guard_pid'] == 37507)
    check('all5recordedPIDs_absent', set(exit_record['known_processes']) == {'37496', '37501', '37507', '37514', '37550'} and
          all(not value['exists'] and not value['live_non_zombie'] for value in exit_record['known_processes'].values()))
    final_path = BASE / 'final_board_observation.json'
    final = read(final_path)
    check('saved_final_board_sameboot_no_tasks', final['boot_id'] == BOOT and not final['task_processes'] and
          all(not value['exists'] for value in final['known_processes'].values()))
    check('saved_final_remote89hashes_match', final['all_remote_receipt_files_still_match'] and
          final['receipt_file_count'] == 89 and len(final['remote_checks']) == 89 and all(final['remote_checks'].values()) and
          final['receipt_sha256'] == sha(RAW / 'remote_receipt.json'))
    check('saved_final_old_engine_preserved_no_qall_engine', final['earlier_corrected_DFL_none_engine_sha256'] ==
          'a0b23d8fce8a25bbf6ca5de9bf638dabbc9be11db23e56ad16cb2efc3ce2a33e' and
          not final['earlier_corrected_DFL_obey_engine_present'] and not final['corrected_qall_engine_paths'] and
          final['qonly_preserved_engine']['sha256'] == '486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476')
    check('saved_final_fan_jtop_active', final['fan']['stdout'].strip() == 'active' and
          'NRestarts=0' in final['jtop']['stdout'] and 'ActiveState=active' in final['jtop']['stdout'])
    check('original_guard_unchanged', spec['original_complete_model_numerical_guard'] == summary['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED')
    check('new_raw_all_bytes_unchanged_by_review', fingerprint(RAW) == before)
    check('old_raw_all_bytes_unchanged_by_review', fingerprint(OLD) == old_before)
    report = {'schema': 'rampq.gate05.dfl_softmax_cut_independent_native_review/v1',
        'reviewed_client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'SAVED_SOFTMAX_CUT_OUTCOMES_INDEPENDENTLY_VERIFIED', 'check_count': len(checks), 'checks': checks,
        'raw_root': str(RAW), 'archive_verification': archive, 'receipt_files': 89,
        'remote_receipt_sha256': sha(RAW / 'remote_receipt.json'), 'cases': cases,
        'saved_parent_corrected_DFL_cases': old_cells, 'old_archive_verification': old_archive,
        'new_NONE_vs_old_corrected_NONE': cross, 'same_CPU_reference': True, 'same_native_external_inputs': False,
        'same_supported_common_flags_and_named_QDQ_requests': True,
        'derived_FP64_reduction_differences': reduction_notes,
        'successful_output_cases': 1, 'monitor_status': monitor['status'], 'monitor_seconds': monitor['elapsed_seconds'],
        'monitor_completion_means_batch_terminal_only': True, 'guard_intervention': False, 'all_owned_groups_empty': True,
        'saved_final_board_observation_sha256': sha(final_path), 'saved_final_board_observation': final,
        'old_new_saved_raw_unchanged': True, 'new_CPU_OR_native_inference_by_reviewer': False, 'SSH_by_reviewer': False,
        'actual_INT8_output_required': False, 'output_is_DFL_distance_not_decoded_boxes': True,
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'full_qall_failure_resolved': False,
        'gate05_status': 'INCOMPLETE', 'dataset_evaluation_run': False, 'new_numerical_tolerance': False,
        'conclusion': 'The Softmax-free 8-node context still fails under OBEY with the saved 4421/751 signature. NONE raw is byte-identical to the saved corrected 9-node DFL NONE raw for this one composed CPU fixture.',
        'limitations': ['Removing native Softmax also changes the external input boundary and fusion context and substitutes CPU-generated probabilities. This is not same-native-input equivalence or general native Softmax equivalence.',
                        'One attempt per case in sequential bounded stages is not randomized replication. No full-model offending node or unique root cause follows.',
                        'No full model, old native engine or CPU graph was rerun by the reviewer. Saved near-CPU output and successful NONE build are not a Gate pass, dataset accuracy or performance result.']}
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': sha(output),
                      'cross_native_saved_raw': cross, 'CPU_difference': cases['code_clip_none']['CPU_comparison_descriptive']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
