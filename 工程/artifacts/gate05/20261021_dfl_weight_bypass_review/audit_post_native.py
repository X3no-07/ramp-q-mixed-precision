"""Independent external saved-evidence reviewer, not the probe tool author."""
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/gate05/20261021_dfl_weight_bypass_native'
RAW = BASE / 'native_evidence'
OLD_BASE = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_native'
OLD = OLD_BASE / 'native_evidence'
BUNDLE = ROOT / 'artifacts/gate05/20261021_dfl_weight_bypass_probe'
REVIEW = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_gate05_evidence_archive import verify_archive


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


helpers = module('saved_original_DFL_helpers', ROOT / 'artifacts/gate05/20261020_dfl_original_review/audit_post_native.py')
pre = module('saved_weightbypass_pre_helpers', REVIEW / 'audit_pre_native.py')
sha, read = pre.sha, pre.read
compare, fingerprint = helpers.compare, helpers.fingerprint
normal = helpers.normalized_constraints
CASES = ['weight_bypass_none', 'weight_bypass_obey']
SOURCE = '/model.23/dfl/conv/Conv'


def main():
    output = BASE / 'reviewer_independent_native_review.json'
    assert not output.exists()
    checks, reductions = [], {}

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)

    before, old_before = fingerprint(RAW), fingerprint(OLD)
    archive = verify_archive(BASE / 'native_evidence.zip', '5353ea2f980c5b9e0a50f63cb947c737551e5bd52978970f2f8ab282a0444c0c', 12003589)
    old_archive = verify_archive(OLD_BASE / 'native_evidence.zip', '07e583b3aac44cebd4008a6c0f1fda83000a9a71c4e940ae1c6bb63f3baa63cd', 12152044)
    for label, folder, snap in [('new', RAW, before), ('old', OLD, old_before)]:
        receipt = read(folder / 'remote_receipt.json')
        check(label + ':exact_receipt_inventory', set(snap) == set(receipt['files']) | {'remote_receipt.json'})
        for name, info in receipt['files'].items():
            check(label + ':receipt:' + name, snap[name] == info)
    receipt = read(RAW / 'remote_receipt.json')
    check('new93receipt_identity', len(receipt['files']) == 93 and sha(RAW / 'remote_receipt.json') == 'b490e6ed56b928d430e43076da2f6a2d53d84b3204f0483fd147246a7a019875')
    pins = dict(pre.PINS, **{'spec.json': 'df7de4468cec5cb0672f89c61e63ce26decbb1d6bc52a521e453e02aad9c5b4f',
        'manifest.json': 'af7e0bb5db03cd228d05df912cf4298f19ad2715a476f7c1757200fbaf1b8d3a'})
    for name, digest in pins.items():
        check('hardpin:' + name, sha(RAW / name) == digest)
    for name, info in read(RAW / 'manifest.json')['files'].items():
        check('sealedpayload:' + name, before[name] == info and (RAW / name).read_bytes() == (BUNDLE / name).read_bytes())
    check('signed_pre_native_review', sha(RAW / 'pre_native_review.json') == sha(REVIEW / 'pre_native_review.json') == '4f314e7457fa3a4ad572323988db23f875baebe38573d815b1b9f43e4b41c623')
    for first, last in [('source_hashes_before.json', 'source_hashes_after.json'), ('seal_native_source_hashes_before.json', 'seal_native_source_hashes_after.json')]:
        a, b = read(RAW / first), read(RAW / last)
        check('sourcepreservation:' + first, a == b)
        for name, info in a.items():
            check('sourcecurrentlyexact:' + first + ':' + name, sha(ROOT / name) == info['sha256'] and (ROOT / name).stat().st_size == info['bytes'])
    spec, protocol = read(RAW / 'spec.json'), read(RAW / 'protocol.json')
    check('own_reference_semantics', spec['own_candidate_CPU_reference_required'] and spec['semantic_change'] == protocol['semantic_change'] and
          spec['weight_quantization_path_removed'] and spec['original_weight_initializer_values_changed'] is False)
    model, parent = onnx.load(RAW / 'model_dfl_weight_bypass.onnx'), onnx.load(RAW / 'parent_conv_output_model.onnx')
    source_nodes = {n.name: n for n in parent.graph.node}
    check('4nodes5init_exact_removal', len(model.graph.node) == 4 and len(model.graph.initializer) == 5 and
          set(source_nodes) - {n.name for n in model.graph.node} == {'rampq_model_23_dfl_conv_WeightQ', 'rampq_model_23_dfl_conv_WeightDQ'})
    for node in model.graph.node:
        restored = copy.deepcopy(node)
        if node.name == SOURCE:
            check('single_Conv_input1_rewire', node.input[1] == 'model.23.dfl.conv.weight')
            restored.input[1] = 'rampq_model_23_dfl_conv_w_dq'
        check('source_exact_except_declared_rewire:' + node.name, restored.SerializeToString() == source_nodes[node.name].SerializeToString())
    old_init = {i.name: i for i in parent.graph.initializer}
    for item in model.graph.initializer:
        check('retained_original_initializer:' + item.name, item.SerializeToString() == old_init[item.name].SerializeToString())
    onnx.checker.check_model(model, full_check=True)
    check('same_parent_probability_file', (RAW / 'dfl_softmax_probabilities.npy').read_bytes() == (OLD / 'dfl_softmax_probabilities.npy').read_bytes())
    with np.load(RAW / 'cpu_expected_outputs.npz', allow_pickle=False) as data:
        check('own_expected_only_key', set(data.files) == {'weight_bypass_output'})
        cpu = data['weight_bypass_output'].copy()
    actual_cpu = np.load(RAW / 'cpu_actual_weight_bypass.npy', allow_pickle=False)
    old_cpu = np.load(RAW / 'saved_parent_cpu_reference_4d.npy', allow_pickle=False)
    check('own_expected_exact_saved_actual', cpu.shape == actual_cpu.shape == (1, 1, 4, 8400) and cpu.dtype == actual_cpu.dtype == np.float32 and
          cpu.tobytes() == actual_cpu.tobytes() and np.isfinite(cpu).all())
    check('own_expected_not_parent_reference', cpu.tobytes() != old_cpu.tobytes())
    cpu_semantic_delta = compare(cpu, old_cpu)
    runs = list((RAW / 'results').glob('probe_*')); old_runs = list((OLD / 'results').glob('probe_*'))
    check('one_new_and_parent_batch', len(runs) == len(old_runs) == 1)
    run, old_run = runs[0], old_runs[0]
    summary = read(run / 'summary.json')
    check('frozen_case_order_and_success_counts', summary['requested_case_order'] == CASES and summary['successful_output_case_count'] == 2 and
          summary['observed_case_count'] == 2 and summary['aborted'] is None)
    cases, arrays, requests, constraints = {}, {}, {}, {}

    def compare_reported(label, computed, native):
        check(label + ':field_set', set(computed) == set(native))
        reductions[label] = {}
        for key, value in computed.items():
            if key in ('mean_abs', 'rms_difference'):
                reductions[label][key] = {'host': value, 'native': native[key], 'host_minus_native': value - native[key], 'descriptive_only': True}
            else:
                check(label + ':' + key, value == native[key])

    for name in CASES:
        folder = run / name
        request, result, build, cleanup = [read(folder / f) for f in ('parsed_network_request.json', 'result.json', 'build_result.json', 'child_cleanup.json')]
        requests[name] = request
        check(name + ':frozencase', request['case'] == next(c for c in spec['cases'] if c['id'] == name))
        for key, file in [('spec_sha256', 'spec.json'), ('model_sha256', 'model_dfl_weight_bypass.onnx'),
                          ('script_sha256', 'jetson_build_dfl_weight_bypass_probe.py'), ('runner_script_sha256', 'jetson_dfl_weight_bypass_probe.py')]:
            check(name + ':identity:' + key, request[key] == sha(RAW / file))
        for flag, enabled in [('INT8', True), ('FP16', True), ('TF32', False), ('PREFER_PRECISION_CONSTRAINTS', False), ('OBEY_PRECISION_CONSTRAINTS', name.endswith('_obey'))]:
            check(name + ':actualflag:' + flag, request['flags']['get_flag'][flag] == {'supported': True, 'enabled': enabled})
        check(name + ':commonsettings_noreset', request['workspace_bytes_actual'] == 1024 ** 3 and request['avg_timing_iterations_actual'] == 1 and
              request['profiling_verbosity_actual'] == 'ProfilingVerbosity.DETAILED' and request['no_layer_precision_or_output_type_setters_or_resets'])
        clip = request['clip_attributes']
        check(name + ':onlyClipcorrected', len(clip) == 1 and clip[0]['after'] == {'alpha': -127.0, 'beta': 127.0} and clip[0]['explicit_importer_correction'])
        earlier, later = read(folder / 'parsed_constraints_before.json'), read(folder / 'parsed_constraints_after.json')
        constraints[name] = normal(earlier)
        check(name + ':allconstraints_retained', normal(earlier) == normal(later))
        check(name + ':actual6parserlayers_removed_weightQDQ', len(earlier) == 6 and not any(r['name'] in ('rampq_model_23_dfl_conv_WeightQ', 'rampq_model_23_dfl_conv_WeightDQ') for r in earlier))
        precision, outputs = [], []
        for row in earlier:
            check(name + ':precisionquery:' + row['name'], row['precision_is_set']['supported'] and 'precision_value_error' not in row and
                  ('precision' in row) == row['precision_is_set']['value'])
            if row['precision_is_set']['value']:
                precision.append((row['name'], row['precision']))
            for value in row['output_types']:
                check(name + ':outputquery:' + row['name'] + ':' + str(value['index']), value['output_type_is_set']['supported'] and
                      'requested_output_type_error' not in value and 'tensor_query_error' not in value and
                      ('requested_output_type' in value) == value['output_type_is_set']['value'])
                if value['output_type_is_set']['value']:
                    outputs.append((row['name'], value['requested_output_type']))
        check(name + ':actualoneDQprecision_oneQoutput', precision == [('rampq_model_23_dfl_conv_ActivationDQ', 'DataType.INT8')] and
              outputs == [('rampq_model_23_dfl_conv_ActivationQ', 'DataType.INT8')])
        check(name + ':successfulterminal_build_nochildleft', build['status'] == 'DFL_WEIGHT_BYPASS_ENGINE_SAVED' and cleanup['returncode'] == 0 and
              cleanup['direct_python_api_child_terminated'] and not cleanup['signals_sent'])
        check(name + ':engineandrawhash', sha(folder / 'model.engine') == result['engine_sha256'] == build['engine_sha256'] and sha(folder / 'raw_outputs.npz') == result['raw_outputs_sha256'])
        inspector = read(folder / 'engine_layers.json')
        check(name + ':inspector_snapshots', inspector == read(folder / 'runner_engine_layers.json'))
        convs = [r for r in inspector['Layers'] if 'Convolution' in r.get('LayerType', '')]
        check(name + ':actualoneConvsource', len(convs) == 1 and SOURCE in convs[0]['Name'] and convs == read(folder / 'conv_rows.json'))
        conv = convs[0]
        mapping = read(folder / 'conv_source_mapping.json')
        check(name + ':rawmapping_unique', mapping['one_dfl_actual_convolution_uniquely_mapped'] and mapping['mapping'][0]['actual_row_indices'] == [0])
        check(name + ':actual_FP16_input_weight_output_observed', conv['Weights']['Type'] == 'Half' and conv['Weights']['Count'] == 16 and
              all('FP16' in r['Format/Datatype'] for r in conv['Inputs'] + conv['Outputs']))
        bindings = read(folder / 'runtime_bindings.json')
        desired = {r['name']: r for r in [spec['input_binding']] + spec['output_bindings_in_order']}
        check(name + ':two_FLOATLINEARbindings', len(bindings) == 2 and len({b['name'] for b in bindings}) == 2)
        for binding in bindings:
            target = desired[binding['name']]
            check(name + ':bindingcontract:' + binding['name'], binding['shape'] == target['shape'] and binding['input'] == target['input'] and
                  binding['dtype'] == 'DataType.FLOAT' and binding['format'] == 'TensorFormat.LINEAR')
        with np.load(folder / 'raw_outputs.npz', allow_pickle=False) as data:
            check(name + ':raw_two_keys', set(data.files) == {'first', 'repeat'})
            first, repeat = data['first'].copy(), data['repeat'].copy()
        check(name + ':raw_FLOAT4Dfinite', first.dtype == repeat.dtype == np.float32 and first.shape == repeat.shape == (1, 1, 4, 8400) and np.isfinite(first).all() and np.isfinite(repeat).all())
        repeat_delta, cpu_delta = compare(first, repeat), compare(first, cpu)
        compare_reported(name + ':repeat', repeat_delta, result['repeat_comparison'])
        compare_reported(name + ':ownCPU', cpu_delta, result['comparison_to_frozen_cpu_ort'])
        check(name + ':repeat_byteexact', repeat_delta['byte_identical'])
        check(name + ':rawstatshash', hashlib.sha256(first.tobytes()).hexdigest() == result['stats']['raw_bytes_sha256'] and
              result['stats']['min'] == float(first.min()) and result['stats']['max'] == float(first.max()))
        arrays[name] = first
        cases[name] = {'build': build, 'actual_Conv': conv, 'parser_layers': len(earlier), 'precision_requests': precision,
            'output_requests': outputs, 'actual_flags': request['flags'], 'repeat': repeat_delta, 'ownCPU_descriptive': cpu_delta,
            'actual_conv_uses_INT8_input_weight': False, 'actual_FP16_all3_formats': True, 'raw_stats': result['stats']}
    check('only_OBEYflag_changes', [k for k in requests[CASES[0]]['flags']['get_flag'] if requests[CASES[0]]['flags']['get_flag'][k] != requests[CASES[1]]['flags']['get_flag'][k]] == ['OBEY_PRECISION_CONSTRAINTS'])
    check('same_parser_requests', constraints[CASES[0]] == constraints[CASES[1]])
    pair = compare(arrays[CASES[0]], arrays[CASES[1]])
    compare_reported('NONE_vs_OBEY', pair, summary['cross_native_comparisons']['weight_bypass_none_vs_weight_bypass_obey'])
    with np.load(old_run / 'code_clip_none/raw_outputs.npz', allow_pickle=False) as data:
        old_raw = data['first'].copy()
    old_Conv = read(old_run / 'code_clip_none/conv_rows.json')[0]
    parent_comparisons = {name: compare(value, old_raw) for name, value in arrays.items()}
    old_failure_log = (old_run / 'code_clip_obey/builder.log').read_text(encoding='utf-8')
    check('parent_OBEY4421_failed_noengine', 'filterQDQFormats::4421' in old_failure_log and not (old_run / 'code_clip_obey/model.engine').exists())
    monitor, guard, launch, terminal = [read(RAW / n) for n in ('monitor/status.json', 'supplemental_group_guard.json', 'launch.json', 'all_child_exit_verification.json')]
    check('sameboot', len({r['boot_id'] for r in (monitor, guard, launch, terminal)}) == 1 and monitor['boot_id'] == '692c7544-b8bd-415b-939c-b11cdbd9f7f0')
    check('guardempty_nointervention', guard['status'] == 'COMPLETE' and guard['owned_group_fully_empty'] and not guard['final_group_members'] and not guard['intervention_performed'])
    check('all5PIDs_absent', set(terminal['known_processes']) == {'40168', '40173', '40179', '40186', '40222'} and all(not r['exists'] and not r['live_non_zombie'] for r in terminal['known_processes'].values()))
    final = read(BASE / 'final_board_observation.json')
    check('final93remotehashesmatch', final['all_remote_receipt_files_still_match'] and final['receipt_file_count'] == len(final['remote_checks']) == 93 and
          all(final['remote_checks'].values()) and final['receipt_sha256'] == sha(RAW / 'remote_receipt.json'))
    check('finalsameboot5absent', final['boot_id'] == monitor['boot_id'] and not final['task_processes'] and all(not r['exists'] for r in final['known_processes'].values()))
    check('fan_jtopactive_zero_restarts', final['fan']['stdout'].strip() == 'active' and 'ActiveState=active' in final['jtop']['stdout'] and 'NRestarts=0' in final['jtop']['stdout'])
    check('oldnewraw_notmodified', fingerprint(RAW) == before and fingerprint(OLD) == old_before)
    report = {'schema': 'rampq.gate05.weight_bypass_external_independent_review/v1',
        'client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'status': 'SAVED_WEIGHT_BYPASS_OUTCOMES_EXTERNALLY_VERIFIED',
        'reviewer': 'prefix_flag_review (not preparer/native tool author)', 'checks': checks, 'check_count': len(checks),
        'archive_verification': archive, 'old_archive_verification': old_archive, 'receipt_files': 93,
        'cases': cases, 'NONE_vs_OBEY_raw_comparison': pair, 'ownCPU_vs_parentCPU_semantic_change_descriptive': cpu_semantic_delta,
        'new_native_vs_parent_NATIVE_descriptive': parent_comparisons, 'parent_actual_Conv': old_Conv,
        'derived_mean_RMS_platform_differences': reductions,
        'interpretation': 'Both builds succeed after explicit weightQDQ removal, but actual sole Conv is FP16 input/Halfweight/FP16output in both. Weight mathematics, parser constraints and selected arithmetic differ; this is not a W8A8 repair or proof of unique failure cause.',
        'newCPU_expected_correctly_own_reference': True, 'source_FP32_initializer_is_actual_FP32_arithmetic': False,
        'sameflag_outputs_byteidentical': pair['byte_identical'], 'raw_input_trees_unchanged': True,
        'new_CPU_native_SSH_by_reviewer': False, 'dataset_evaluation_run': False,
        'original_complete_model_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
        'final_board_observation_sha256': sha(BASE / 'final_board_observation.json'),
        'limits': ['Native flagpair outputs differ despite repeated-within-case bit identity; no equivalenceclaim betweenNONE/OBEY.',
                   'Comparison to parent has different quantized weight mathematics and actualConv arithmetic, not an isolated equivalent correction.',
                   'This is independent of the same-author519 review; it preserves that existing file untouched.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': sha(output), 'cross': pair}))


if __name__ == '__main__':
    main()
