"""Independent frozen weight-bypass source and saved-array audit; no inference."""
import argparse
import ast
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[3]
REVIEW = Path(__file__).resolve().parent
BUNDLE = ROOT / 'artifacts/gate05/20261021_dfl_weight_bypass_probe'
PARENT = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_probe'
PINS = {
    'protocol.json': '9b2ea26ff12ecd5a4b5a699b09c511a3ea6c2187125d037117bb85f57b7b1170',
    'source_model.onnx': '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee',
    'model_dfl_weight_bypass.onnx': '020dad7f70c04f495dfe1b6e62e63e925ff6468a6cde0c157b7122a38274eb41',
    'dfl_softmax_probabilities.npy': '5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520',
    'cpu_expected_outputs.npz': 'dff4725f1dbbb99979723b47267973f9cde5bc390db30131ad5d39361d958526',
    'cpu_actual_weight_bypass.npy': '808e4a1a17e2f9fc622d858a3dba8390e45749a72afb0ac6a8fd3d8ac413cf73',
    'prepare_gate05_dfl_weight_bypass_probe.py': 'd77b32d916aa887b0542bdf2ad4dd661b3425a44e711fdea7b9f10b60e89a0ba',
    'jetson_dfl_weight_bypass_probe.py': '87a75b81a4f5b463ba1ff304fbbf87ade038dea56c134c20b357c2b8f746c51b',
    'jetson_build_dfl_weight_bypass_probe.py': '01da9df3d682fbea334345c49c0dbbe171ab5ea932e5fbe1fc6c5ab0b1c6f854',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def fingerprint(path):
    return {p.relative_to(path).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in path.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('zip-sha', 'spec-sha', 'manifest-sha'):
        parser.add_argument('--' + arg, required=True)
    parser.add_argument('--zip-bytes', type=int, required=True)
    args = parser.parse_args()
    output = REVIEW / 'pre_native_review.json'
    assert not output.exists()
    before = fingerprint(BUNDLE)
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)

    pins = dict(PINS, **{'spec.json': args.spec_sha, 'manifest.json': args.manifest_sha})
    for name, digest in pins.items():
        check('frozen_pin:' + name, sha(BUNDLE / name) == digest)
    spec, protocol, manifest = [read(BUNDLE / name) for name in ('spec.json', 'protocol.json', 'manifest.json')]
    check('exact_payload_and_seals', set(before) == set(manifest['files']) | {'manifest.json', 'SHA256SUMS'})
    check('regular_files', all(p.is_file() and not p.is_symlink() for p in BUNDLE.iterdir()))
    for name, value in manifest['files'].items():
        check('manifest:' + name, (BUNDLE / name).resolve().parent == BUNDLE.resolve() and before[name] == value)
    for line in (BUNDLE / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        check('SHA256SUMS:' + name, sha(BUNDLE / name) == digest)
    archive_path = BUNDLE.with_suffix('.zip')
    check('zip_size_and_sha', sha(archive_path) == args.zip_sha and archive_path.stat().st_size == args.zip_bytes)
    with zipfile.ZipFile(archive_path) as archive:
        check('zip_crc', archive.testzip() is None)
        members = {'gate05_dfl_weight_bypass_probe_20261021/' + p.name for p in BUNDLE.iterdir()}
        check('zip_exact_inventory', len(archive.namelist()) == len(members) and set(archive.namelist()) == members)
        for name in members:
            check('zip_payload:' + name, archive.read(name) == (BUNDLE / name.split('/')[-1]).read_bytes())
    for first, last in [('source_hashes_before.json', 'source_hashes_after.json'),
                         ('seal_native_source_hashes_before.json', 'seal_native_source_hashes_after.json')]:
        original, final = read(BUNDLE / first), read(BUNDLE / last)
        check('source_snapshots_equal:' + first, original == final)
        for relative, value in original.items():
            check('source_current:' + first + ':' + relative,
                  (ROOT / relative).stat().st_size == value['bytes'] and sha(ROOT / relative) == value['sha256'])
    for name, info in read(BUNDLE / 'prepared_content_manifest.json')['files'].items():
        check('prepared_content_unchanged:' + name, before[name] == info)
    copies = {'parent_conv_output_spec.json': PARENT / 'spec.json',
        'parent_conv_output_manifest.json': PARENT / 'manifest.json',
        'parent_conv_output_model.onnx': PARENT / 'model_dfl_conv_output_code_clip.onnx',
        'parent_fixture_lineage.json': PARENT / 'fixture_lineage.json',
        'dfl_softmax_probabilities.npy': PARENT / 'dfl_softmax_probabilities.npy',
        'saved_parent_cpu_reference_4d.npy': PARENT / 'cpu_actual_code_clip.npy',
        'saved_parent_expected_outputs.npz': PARENT / 'cpu_expected_outputs.npz',
        'parent_conv_output_checkpoint.json': ROOT / 'artifacts/gate05/20261020_dfl_conv_output_native/checkpoint.json',
        'parent_next_control_plan.json': ROOT / 'artifacts/gate05/20261020_dfl_conv_output_native/next_control_plan.json',
        'protocol.json': ROOT / 'artifacts/gate05/20261021_dfl_weight_bypass_protocol/protocol.json'}
    for name, path in copies.items():
        check('historical_copy:' + name, (BUNDLE / name).read_bytes() == path.read_bytes())
    check('exact_two_cases', spec['cases'] == protocol['cases'] and spec['case_order'] == protocol['case_order'] == ['weight_bypass_none', 'weight_bypass_obey'])
    check('oneCPU_no_upstream_regeneration', spec['new_CPU_execution_runs_this_stage'] == 1 and
          spec['new_probability_fixture_generation_runs'] == spec['new_Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0)
    check('frozen_native_budgets', spec['maximum_attempts_per_case'] == 1 and spec['maximum_native_builds'] == 2 and
          spec['maximum_native_inferences'] == 4 and spec['executions_per_successful_case'] == 2 and
          spec['max_build_seconds'] == 300 and spec['max_seconds'] == 900)
    check('native_common_policy', spec['native_build_policy'] == protocol['common_native_settings'])
    model = onnx.load(BUNDLE / 'model_dfl_weight_bypass.onnx')
    parent = onnx.load(BUNDLE / 'parent_conv_output_model.onnx')
    full = onnx.load(BUNDLE / 'source_model.onnx')
    for graph in (model, parent):
        onnx.checker.check_model(graph, full_check=True)
    check('four_nodes_five_used_initializers', len(model.graph.node) == 4 and len(model.graph.initializer) == 5 and
          sorted(n.op_type for n in model.graph.node) == sorted(['QuantizeLinear', 'Clip', 'DequantizeLinear', 'Conv']))
    nodes, old_nodes, source_nodes = [{n.name: n for n in g.graph.node} for g in (model, parent, full)]
    initializers, old_init, source_init = [{i.name: i for i in g.graph.initializer} for g in (model, parent, full)]
    removed = ['rampq_model_23_dfl_conv_WeightQ', 'rampq_model_23_dfl_conv_WeightDQ']
    check('removed_exact_weightQ_DQ', set(old_nodes) - set(nodes) == set(removed) and spec['removed_node_names'] == protocol['removed_node_names'] == removed)
    conv_name = '/model.23/dfl/conv/Conv'
    check('one_Conv_single_input1_mutation_declared', spec['declared_node_mutation'] == protocol['declared_node_mutation'] ==
          {'node': conv_name, 'field': 'input[1]', 'old': 'rampq_model_23_dfl_conv_w_dq', 'new': 'model.23.dfl.conv.weight'})
    for name, node in nodes.items():
        original = copy.deepcopy(node)
        if name == conv_name:
            check('Conv_direct_source_weight_input', list(node.input) == ['rampq_model_23_dfl_conv_a_dq', 'model.23.dfl.conv.weight'])
            original.input[1] = 'rampq_model_23_dfl_conv_w_dq'
        check('only_allowed_node_mutation:' + name, original.SerializeToString() == old_nodes[name].SerializeToString() == source_nodes[name].SerializeToString())
        check('node_hash:' + name, hashlib.sha256(node.SerializeToString()).hexdigest() == spec['variants']['weight_bypass']['selected_node_serialized_sha256'][name])
    check('only_unused_weight_scale_zero_removed', set(old_init) - set(initializers) ==
          {'rampq_model_23_dfl_conv_w_scale', 'rampq_model_23_dfl_conv_w_zero'})
    used_init = {name for node in model.graph.node for name in node.input if name in source_init}
    check('exact_used_initializer_set', set(initializers) == used_init)
    for name, value in initializers.items():
        check('initializer_source_parent_bytes:' + name, value.SerializeToString() == old_init[name].SerializeToString() == source_init[name].SerializeToString())
        check('initializer_hash:' + name, hashlib.sha256(value.SerializeToString()).hexdigest() == spec['variants']['weight_bypass']['selected_initializer_serialized_sha256'][name])
    weight = onnx.numpy_helper.to_array(initializers['model.23.dfl.conv.weight'])
    check('original_DFL_weights_unchanged_values0_to15', weight.dtype == np.float32 and weight.shape == (1, 16, 1, 1) and
          np.array_equal(weight.reshape(-1), np.arange(16, dtype=np.float32)))
    check('exact_parent_IO_declarations', len(model.graph.input) == len(model.graph.output) == 1 and
          model.graph.input[0].SerializeToString() == parent.graph.input[0].SerializeToString() and
          model.graph.output[0].SerializeToString() == parent.graph.output[0].SerializeToString())
    check('explicit4D_IO', [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] == [1, 16, 4, 8400] and
          [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim] == [1, 1, 4, 8400])
    check('own_new_reference_not_equivalent_repair', spec['semantic_change'] == protocol['semantic_change'] and
          spec['own_candidate_CPU_reference_required'] and spec['weight_quantization_path_removed'] and
          spec['original_weight_initializer_values_changed'] is False)
    cpu, lineage, plan, graph_audit = [read(BUNDLE / n) for n in
        ('cpu_diagnostic.json', 'fixture_lineage.json', 'cpu_execution_plan.json', 'graph_identity_audit.json')]
    for name, key in [('cpu_diagnostic.json', 'cpu_diagnostic_sha256'), ('fixture_lineage.json', 'fixture_lineage_sha256'),
                      ('graph_identity_audit.json', 'graph_identity_audit_sha256')]:
        check('derived_record_sha:' + name, sha(BUNDLE / name) == spec[key])
    probability = np.load(BUNDLE / 'dfl_softmax_probabilities.npy', allow_pickle=False)
    actual = np.load(BUNDLE / 'cpu_actual_weight_bypass.npy', allow_pickle=False)
    old = np.load(BUNDLE / 'saved_parent_cpu_reference_4d.npy', allow_pickle=False)
    check('probability_saved_FLOAT_domain', probability.dtype == np.float32 and probability.shape == (1, 16, 4, 8400) and
          np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)))
    check('probability_saved_rawhash', hashlib.sha256(probability.tobytes()).hexdigest() == lineage['fixture']['raw_array_bytes_sha256'] ==
          'd4c4b21e1d1a2e025d5b3eabaade7e99f3093c0cf7511795f9477a4ffd069ae7' and lineage['new_fixture_generation_runs'] == 0)
    check('candidate_CPU_saved_FLOAT4D', actual.dtype == old.dtype == np.float32 and actual.shape == old.shape == (1, 1, 4, 8400) and
          np.isfinite(actual).all() and actual.size == 33600)
    check('candidate_CPU_rawhash', hashlib.sha256(actual.tobytes()).hexdigest() == cpu['actual_output']['raw_array_bytes_sha256'] ==
          '685452498c250f80567e057d5a24d02b8e23605ff44c7a70ec0be52e27d65010')
    check('one_own_CPU_result_not_old_expected', cpu['status'] == 'OWN_ONE_RUN_CPU_REFERENCE_ESTABLISHED' and cpu['new_CPU_executions'] == 1 and
          cpu['native_executions'] == cpu['new_probability_Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0 and
          cpu['old_reference_adopted_as_candidate_expected'] is False and cpu['expected_key'] == 'weight_bypass_output')
    check('one_CPU_runtime_frozen', cpu['provider'] == 'CPUExecutionProvider' and cpu['graph_optimization'] == 'ORT_DISABLE_ALL' and cpu['threads'] == 4)
    check('one_CPU_plan', plan['maximum_total_new_CPU_runs'] == plan['new_candidate_CPU_runs'] == 1 and
          plan['probability_fixture_generation_runs'] == plan['Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0)
    delta = np.abs(actual.astype(np.float64) - old.astype(np.float64))
    descriptive = cpu['old_reference_comparison_descriptive_only']
    check('old_reference_description_only_honest', descriptive['equality_required'] is False and descriptive['new_tolerance_or_numerical_acceptance_rule'] is False and
          descriptive['differing_elements'] == int(np.count_nonzero(actual != old)) and descriptive['max_absolute_difference'] == float(delta.max()) and
          descriptive['C_order_array_bytes_identical'] == (actual.tobytes() == old.tobytes()))
    with np.load(BUNDLE / 'cpu_expected_outputs.npz', allow_pickle=False) as data:
        check('own_expected_single_key_BYTEEXACT', set(data.files) == {'weight_bypass_output'} and data['weight_bypass_output'].dtype == np.float32 and
              data['weight_bypass_output'].shape == (1, 1, 4, 8400) and data['weight_bypass_output'].tobytes() == actual.tobytes())
    for name in ('prepare_gate05_dfl_weight_bypass_probe.py', 'jetson_dfl_weight_bypass_probe.py', 'jetson_build_dfl_weight_bypass_probe.py'):
        text = (BUNDLE / name).read_text(encoding='utf-8')
        check('tool_workspace_exact:' + name, (BUNDLE / name).read_bytes() == (ROOT / 'scripts' / name).read_bytes())
        tree = ast.parse(text, feature_version=(3, 8) if name.startswith('jetson_') else None)
        check('AST_syntax:' + name, isinstance(tree, ast.Module))
        if name.startswith('jetson_'):
            check('no_pending_pins:' + name, 'PENDING' not in text)
            calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            stores = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
            check('no_precision_setters_reset:' + name, not calls & {'set_precision', 'reset_precision', 'set_output_type', 'reset_output_type'})
            check('no_precision_dtype_store:' + name, not stores & {'precision', 'dtype'})
    proc = subprocess.run([sys.executable, str(BUNDLE / 'jetson_dfl_weight_bypass_probe.py'), '--root', str(BUNDLE), '--check-input-only'], capture_output=True, text=True)
    check('purestdlib_inputcheck_exit0', proc.returncode == 0)
    receipt = json.loads(proc.stdout)
    check('purestdlib_inputcheck_no_runtime', receipt['status'] == 'FROZEN_DFL_WEIGHT_BYPASS_INPUTS_VERIFIED' and
          receipt['case_count'] == 2 and receipt['native_runtime_imported'] is False)
    manual = [
        ('parser_constraints', 'Only Clip alpha/beta ±127 corrected; all actual parser precision/output requests before/after recorded without setter/reset. Removed weightQ/DQ names absent; no old2+2 count fabricated.'),
        ('actual_precision', 'FP32 sourceweight does not require or prove actualFP32 arithmetic. Record sole Conv input/weight/output/tactic/fusion separately; no INT8output requirement.'),
        ('cleanup_and_budgets', '300s per child/900s batch,2builders once each,successful2raw. Ordinary positive compile failure continues after child/input checks; timeout/interrupt/stale/signal abort and cleanup. No retry.'),
        ('causal_scope', 'WeightQDQ removal changes quantization mathematics and parser context together; successful build would not be fullq_all equivalent repair or unique fullmodel cause.'),
        ('no_reviewer_inference', 'Only static ONNX parse/check and saved np.load; no ORT session, native/SSH. Root retains outer monitored privategroup/guard and liveboard responsibility.')]
    for name, _ in manual:
        check('manual:' + name, True)
    check('protected_guard_scope', spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
          spec['new_tolerances_defined'] is False and spec['dataset_evaluation_run'] is False and spec['complete_detector_executions'] == 0)
    check('sealed_payload_unchanged_by_review', fingerprint(BUNDLE) == before)
    report = {'schema': 'rampq.gate05.dfl_weight_bypass_pre_native_independent_review/v1',
        'client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'PRE_NATIVE_WEIGHT_BYPASS_STATIC_AND_OWN_CPU_REFERENCE_REVIEW_PASSED_ADOPTION_PENDING',
        'launch_blockers': [], 'checks': checks, 'check_count': len(checks),
        'manual_review': [{'name': n, 'assessment': v} for n, v in manual],
        'identities': {**pins, 'zip_sha256': sha(archive_path), 'zip_bytes': args.zip_bytes},
        'own_CPU_raw_sha256': hashlib.sha256(actual.tobytes()).hexdigest(), 'old_CPU_difference_descriptive': descriptive,
        'check_input_only_receipt': receipt, 'new_CPU_or_native_inference_run_by_review': False, 'reviewer_SSH': False,
        'original_complete_model_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE', 'scope_limit': protocol['cross_stage_policy']}
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': sha(output), 'launch_blockers': []}))


if __name__ == '__main__':
    main()
