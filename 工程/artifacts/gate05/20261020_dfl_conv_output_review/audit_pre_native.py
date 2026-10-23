"""Independent static/saved-array sealed review. No ORT/TRT/torch imports."""
import argparse
import ast
import copy
from datetime import datetime, timedelta, timezone
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
BUNDLE = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_probe'
PARENT = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_probe'
PINS = {
    'protocol.json': '8f6a55f640915b84d7300d1595ebb895092ccf0374e40a27b6aaf812f406e09b',
    'model_dfl_conv_output_code_clip.onnx': '805abb78ed4e4d5abc8041b2916dadc83370a14ef1776fcc0a03f163c16bcf81',
    'dfl_softmax_probabilities.npy': '5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520',
    'cpu_expected_outputs.npz': '711fb41fa18bef712bd069c88a552e1bc918d1d1034b6f6d32d802cbd769d0c6',
    'cpu_actual_code_clip.npy': 'd2f4327fcc792b864e420f1b1554a5b7b52b88fd27a8f01f6c07a0159c3e723b',
    'saved_cpu_reference_3d.npy': 'fd4bcdcee24cd53651fdd7a508423c802f3b432c759647204c7b748f22abd50d',
    'saved_expected_outputs_3d.npz': 'fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511',
    'source_model.onnx': '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee',
    'prepare_gate05_dfl_conv_output_probe.py': 'f550ba88f24106d666a7da381c43de8d0c148b0a6839e7168ee62bf094667f44',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def fingerprint(folder):
    return {p.relative_to(folder).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in folder.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['zip-sha', 'spec-sha', 'manifest-sha', 'runner-sha', 'builder-sha']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--zip-bytes', required=True, type=int)
    args = parser.parse_args()
    output = REVIEW / 'pre_native_review.json'
    assert not output.exists()
    pins = dict(PINS, **{'spec.json': args.spec_sha, 'manifest.json': args.manifest_sha,
        'jetson_dfl_conv_output_probe.py': args.runner_sha, 'jetson_build_dfl_conv_output_probe.py': args.builder_sha})
    checks = []

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
        if not value:
            raise AssertionError(name)

    before = fingerprint(BUNDLE)
    for name, digest in pins.items():
        check('pinned_identity:' + name, sha(BUNDLE / name) == digest)
    check('static_source_review_signed', sha(REVIEW / 'source_readonly_review.json') ==
          '6b8b3b45eb12cd1ccefe02eec6d4c05b6ae029a648c388067e91340195a62efb')
    spec, protocol, manifest = [read(BUNDLE / name) for name in ['spec.json', 'protocol.json', 'manifest.json']]
    for name, info in manifest['files'].items():
        check('sealed_member_path:' + name, (BUNDLE / name).resolve().parent == BUNDLE.resolve())
        check('sealed_member_bytes:' + name, before[name] == info)
    check('exact_payload_plus_seals', set(before) == set(manifest['files']) | {'manifest.json', 'SHA256SUMS'})
    check('regular_files_no_symlinks', all(p.is_file() and not p.is_symlink() for p in BUNDLE.iterdir()))
    for line in (BUNDLE / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        check('sha256sums:' + name, sha(BUNDLE / name) == digest)
    archive_path = BUNDLE.with_suffix('.zip')
    check('zip_identity_and_size', sha(archive_path) == args.zip_sha and archive_path.stat().st_size == args.zip_bytes)
    with zipfile.ZipFile(archive_path) as archive:
        check('zip_crc', archive.testzip() is None)
        members = {'gate05_dfl_conv_output_probe_20261020/' + p.name for p in BUNDLE.iterdir()}
        check('zip_exact_inventory', set(archive.namelist()) == members and len(archive.namelist()) == len(members))
        for name in members:
            check('zip_original_payload:' + name, archive.read(name) == (BUNDLE / name.split('/')[-1]).read_bytes())
    copies = {'protocol.json': ROOT / 'artifacts/gate05/20261020_dfl_conv_output_protocol/protocol.json',
        'source_model.onnx': ROOT / protocol['source_graph'],
        'parent_spec.json': ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate/spec.json',
        'parent_manifest.json': ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate/manifest.json',
        'parent_softmax_cut_spec.json': PARENT / 'spec.json', 'parent_softmax_cut_manifest.json': PARENT / 'manifest.json',
        'parent_softmax_cut_checkpoint.json': ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_native/checkpoint.json',
        'parent_next_control_plan.json': ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_native/next_control_plan.json',
        'parent_dfl_spec.json': PARENT / 'parent_dfl_spec.json', 'parent_dfl_manifest.json': PARENT / 'parent_dfl_manifest.json',
        'parent_fixture_lineage.json': PARENT / 'fixture_lineage.json',
        'dfl_softmax_probabilities.npy': PARENT / 'dfl_softmax_probabilities.npy',
        'saved_cpu_reference_3d.npy': PARENT / 'cpu_actual_code_clip.npy',
        'saved_expected_outputs_3d.npz': PARENT / 'cpu_expected_outputs.npz'}
    for name, source in copies.items():
        check('historical_copy_exact:' + name, (BUNDLE / name).read_bytes() == source.read_bytes())
    for first, last in [('source_hashes_before.json', 'source_hashes_after.json'),
                         ('seal_native_source_hashes_before.json', 'seal_native_source_hashes_after.json')]:
        earlier, later = read(BUNDLE / first), read(BUNDLE / last)
        check('source_snapshots_equal:' + first, earlier == later)
        for name, info in earlier.items():
            check('source_fresh:' + first + ':' + name, (ROOT / name).stat().st_size == info['bytes'] and sha(ROOT / name) == info['sha256'])
    for name, info in read(BUNDLE / 'prepared_content_manifest.json')['files'].items():
        check('prepared_content_unchanged:' + name, before[name] == info)
    check('frozen_cases', spec['cases'] == protocol['cases'] and spec['case_order'] == protocol['case_order'] == ['code_clip_none', 'code_clip_obey'])
    check('frozen_budgets', spec['maximum_attempts_per_case'] == 1 and spec['maximum_native_builds'] == 2 and
          spec['maximum_native_inferences'] == 4 and spec['executions_per_successful_case'] == 2 and
          spec['max_seconds'] == 900 and spec['max_build_seconds'] == 300)
    check('exact_common_native_policy', spec['native_build_policy'] == protocol['common_native_settings'])
    check('CPU_once_no_old_reruns', spec['new_CPU_execution_runs_this_stage'] == 1 and
          spec['new_probability_fixture_generation_runs'] == spec['new_Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0)
    model = onnx.load(BUNDLE / 'model_dfl_conv_output_code_clip.onnx')
    parent = onnx.load(PARENT / 'model_dfl_softmax_cut_code_clip.onnx')
    full = onnx.load(BUNDLE / 'source_model.onnx')
    full_nodes, old_nodes = {n.name: n for n in full.graph.node}, {n.name: n for n in parent.graph.node}
    full_init, old_init = {i.name: i for i in full.graph.initializer}, {i.name: i for i in parent.graph.initializer}
    new_nodes = {n.name: n for n in model.graph.node}
    identity = read(BUNDLE / 'graph_identity_audit.json')
    check('six_nodes_seven_init_one_Conv', len(model.graph.node) == 6 and len(model.graph.initializer) == 7 and
          [n.name for n in model.graph.node if n.op_type == 'Conv'] == ['/model.23/dfl/conv/Conv'])
    check('removed_exact_two_nodes_no_Softmax', set(old_nodes) - set(new_nodes) == set(protocol['removed_node_names']) ==
          {'/model.23/dfl/Constant_1', '/model.23/dfl/Reshape_1'} and not any(n.op_type in ('Softmax', 'Reshape', 'Constant') for n in model.graph.node))
    producer = {value: node for node in full.graph.node for value in node.output}
    needed, selected = set(), set()

    def visit(name):
        needed.add(name)
        if name == protocol['input_name']:
            return
        node = producer.get(name)
        if node is None or node.name in selected:
            return
        selected.add(node.name)
        for value in node.input:
            if value:
                visit(value)

    visit(protocol['output_name'])
    check('exact_full_source_closure', [n.name for n in model.graph.node] == [n.name for n in full.graph.node if n.name in selected] and
          [i.name for i in model.graph.initializer] == [i.name for i in full.graph.initializer if i.name in needed])
    for node in model.graph.node:
        check('source_and_parent_node_exact:' + node.name, node.SerializeToString() == full_nodes[node.name].SerializeToString() == old_nodes[node.name].SerializeToString())
        check('serialized_node_hash:' + node.name, hashlib.sha256(node.SerializeToString()).hexdigest() ==
              identity['node_serialized_sha256'][node.name] == spec['variants']['code_clip']['selected_node_serialized_sha256'][node.name])
    check('all_parent_initializers_retained', {i.name for i in model.graph.initializer} == set(old_init))
    for value in model.graph.initializer:
        check('source_and_parent_initializer_exact:' + value.name, value.SerializeToString() == full_init[value.name].SerializeToString() == old_init[value.name].SerializeToString())
        check('serialized_initializer_hash:' + value.name, hashlib.sha256(value.SerializeToString()).hexdigest() ==
              identity['initializer_serialized_sha256'][value.name] == spec['variants']['code_clip']['selected_initializer_serialized_sha256'][value.name])
    check('one_input_one_output', len(model.graph.input) == len(model.graph.output) == 1)
    declarations = {v.name: v for v in list(full.graph.input) + list(full.graph.value_info) + list(full.graph.output)}
    for value, name, shape in [(model.graph.input[0], protocol['input_name'], [1, 16, 4, 8400]),
                               (model.graph.output[0], protocol['output_name'], [1, 1, 4, 8400])]:
        check('exact_source_io_declaration:' + name, value.name == name and value.SerializeToString() == declarations[name].SerializeToString() and
              value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and [d.dim_value for d in value.type.tensor_type.shape.dim] == shape)
    first, last = copy.deepcopy(model), copy.deepcopy(full)
    first.ClearField('graph'); last.ClearField('graph')
    check('source_model_metadata_unchanged', first.SerializeToString() == last.SerializeToString())
    onnx.checker.check_model(model, full_check=True)
    check('static_ONNX_check', True)
    check('new_4D_binding_contract', spec['input_binding']['name'] == protocol['input_name'] and spec['input_binding']['shape'] == [1, 16, 4, 8400] and
          len(spec['output_bindings_in_order']) == 1 and spec['output_bindings_in_order'][0]['name'] == protocol['output_name'] and
          spec['output_bindings_in_order'][0]['shape'] == [1, 1, 4, 8400])
    lineage, cpu, plan, shape_lineage = [read(BUNDLE / name) for name in
        ['fixture_lineage.json', 'cpu_diagnostic.json', 'cpu_execution_plan.json', 'reference_shape_lineage.json']]
    for name, key in [('fixture_lineage.json', 'fixture_lineage_sha256'), ('cpu_diagnostic.json', 'cpu_diagnostic_sha256'),
                      ('reference_shape_lineage.json', 'reference_shape_lineage_sha256'), ('graph_identity_audit.json', 'graph_identity_audit_sha256')]:
        check('derived_record_identity:' + name, sha(BUNDLE / name) == spec[key])
    probability = np.load(BUNDLE / spec['input_file'], allow_pickle=False)
    check('same_saved_probability_domain_FLOAT_shape', probability.dtype == np.float32 and probability.shape == (1, 16, 4, 8400) and
          probability.flags.c_contiguous and np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)))
    check('same_saved_probability_raw_hash', hashlib.sha256(probability.tobytes(order='C')).hexdigest() ==
          lineage['fixture']['raw_array_bytes_sha256'] == 'd4c4b21e1d1a2e025d5b3eabaade7e99f3093c0cf7511795f9477a4ffd069ae7')
    check('no_new_probability_generation', lineage['new_fixture_generation_runs'] == 0 and
          lineage['fixture_sha256'] == PINS['dfl_softmax_probabilities.npy'])
    check('frozen_one_CPU_plan', plan['status'] == 'FROZEN_BEFORE_ONLY_CPU_EXECUTION' and
          plan['maximum_total_new_CPU_runs'] == plan['new_six_node_CPU_runs'] == 1 and
          plan['probability_fixture_generation_runs'] == plan['Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0)
    check('one_CPU_result_metadata', cpu['status'] == 'ONE_CPU_CONV_OUTPUT_RAW_BYTE_EQUIVALENCE_PASSED' and cpu['new_CPU_executions'] == 1 and
          cpu['new_probability_Softmax_72Conv_old_DFL_or_full_detector_runs'] == cpu['native_executions'] == 0 and
          cpu['provider'] == 'CPUExecutionProvider' and cpu['graph_optimization'] == 'ORT_DISABLE_ALL' and cpu['threads'] == 4)
    actual = np.load(BUNDLE / cpu['actual_output_file'], allow_pickle=False)
    old = np.load(BUNDLE / 'saved_cpu_reference_3d.npy', allow_pickle=False)
    remapped = np.load(BUNDLE / 'reshaped_saved_cpu_reference_4d.npy', allow_pickle=False)
    check('3D_4D_dtype_size_and_declared_shapes', actual.dtype == old.dtype == remapped.dtype == np.float32 and
          actual.shape == remapped.shape == (1, 1, 4, 8400) and old.shape == (1, 4, 8400) and
          actual.size == old.size == remapped.size == 33600 and np.isfinite(actual).all())
    check('3D_4D_C_order_raw_array_bytes_exact', actual.tobytes(order='C') == old.tobytes(order='C') == remapped.tobytes(order='C') and
          np.array_equal(actual, old.reshape(1, 1, 4, 8400)))
    check('NPY_shape_header_difference_not_rejected', sha(BUNDLE / 'cpu_actual_code_clip.npy') != sha(BUNDLE / 'saved_cpu_reference_3d.npy') and
          shape_lineage['NPY_whole_file_hash_equality_required'] is False)
    check('shape_lineage_authenticates_new_files', shape_lineage['old_3d_file_sha256'] == sha(BUNDLE / 'saved_cpu_reference_3d.npy') and
          shape_lineage['reshaped_4d_file_sha256'] == sha(BUNDLE / 'reshaped_saved_cpu_reference_4d.npy') and
          shape_lineage['new_expected_NPZ_sha256'] == sha(BUNDLE / 'cpu_expected_outputs.npz') and
          shape_lineage['old_and_reshaped_C_order_array_bytes_identical'] and shape_lineage['new_CPU_executions_for_reshape'] == 0)
    check('CPU_byteeq_zero_difference_record_honest', cpu['C_order_array_bytes_identical_saved_reference'] and cpu['both_outputs_byte_exact'] and
          cpu['differing_elements'] == 0 and cpu['max_absolute_difference'] == 0 and cpu['actual_output_file_sha256'] == sha(BUNDLE / cpu['actual_output_file']))
    with np.load(BUNDLE / spec['cpu_expected_outputs_file'], allow_pickle=False) as data:
        check('new_expected_NPZ_only_4D_code_clip', set(data.files) == {'code_clip_output'} and data['code_clip_output'].dtype == np.float32 and
              data['code_clip_output'].shape == (1, 1, 4, 8400) and data['code_clip_output'].tobytes(order='C') == actual.tobytes(order='C'))
    for name in ['prepare_gate05_dfl_conv_output_probe.py', 'jetson_dfl_conv_output_probe.py', 'jetson_build_dfl_conv_output_probe.py']:
        text = (BUNDLE / name).read_text(encoding='utf-8')
        check('workspace_tool_exact:' + name, text == (ROOT / 'scripts' / name).read_text(encoding='utf-8'))
        check('no_pending_native_hardpins:' + name, not name.startswith('jetson_') or 'PENDING' not in text)
        tree = ast.parse(text, feature_version=(3, 8) if name.startswith('jetson_') else None)
        check('tool_AST_syntax:' + name, isinstance(tree, ast.Module))
        if name.startswith('jetson_'):
            calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            stores = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)]
            check('no_layer_setter_or_reset:' + name, not set(calls) & {'set_precision', 'reset_precision', 'set_output_type', 'reset_output_type'})
            check('no_direct_precision_or_dtype_store:' + name, not set(stores) & {'precision', 'dtype'})
    proc = subprocess.run([sys.executable, str(BUNDLE / 'jetson_dfl_conv_output_probe.py'), '--root', str(BUNDLE), '--check-input-only'],
                          capture_output=True, text=True)
    check('pure_stdlib_check_exit0', proc.returncode == 0)
    receipt = json.loads(proc.stdout)
    check('pure_stdlib_check_no_native_import', receipt['status'] == 'FROZEN_DFL_CONV_OUTPUT_INPUTS_VERIFIED' and
          receipt['case_count'] == 2 and receipt['native_runtime_imported'] is False)
    check('original_guard_and_scope_retained', spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
          spec['new_tolerances_defined'] is False and spec['weights_changed'] is False and spec['calibration_recomputed'] is False and
          spec['dataset_evaluation_run'] is False and spec['complete_detector_executions'] == 0)
    manual = [
        ('same_input_weights_scales_QDQ', 'Six source nodes and seven initializers exact. Only terminal Constant/Reshape removed; same probability NPY reused without generation. All parser requests retained, Clip alpha/beta alone corrected -127/+127.'),
        ('explicit_shape_correspondence', 'Saved old3D and new4D CPU arrays have equal dtype/count and C-order raw bytes; expectedNPZ authenticates new4D contract. Different NPY shape headers do not imply numerical difference.'),
        ('bounded_native_execution', 'NONE/OBEY one builder each,300s each/batch900s,successful engine2raw repetitions. Ordinary compiler failure continues only after child exit/input revalidation; timeout/signal/interrupt/stale input abort batch. No resets/retry.'),
        ('actual_precision_pending', 'Actual sole Conv input/weight/output/tactic each need saved inspector verification. No INT8 output requirement; external output FLOAT_LINEAR4D.'),
        ('output_boundary_context_limit', 'Removing Reshape also changes external boundary and format selection. Compatible/incompatible results cannot uniquely identify removed node or fullmodel root cause.'),
        ('no_new_reviewer_execution', 'Only ONNX parse/staticcheck and savednp.load used; stdlib input check does not import native runtime. No CPUgraph/native inference/SSH.')]
    for name, _ in manual:
        check('manual:' + name, True)
    check('sealed_payload_unchanged_by_review', fingerprint(BUNDLE) == before)
    report = {'schema': 'rampq.gate05.dfl_conv_output_pre_native_independent_review/v1',
        'client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'PRE_NATIVE_CONV_OUTPUT_STATIC_AND_SAVED_CPU_REVIEW_PASSED_ADOPTION_PENDING',
        'launch_blockers': [], 'check_count': len(checks), 'checks': checks,
        'manual_review': [{'name': name, 'assessment': value} for name, value in manual],
        'identities': {**{name: sha(BUNDLE / name) for name in pins}, 'zip_sha256': sha(archive_path), 'zip_bytes': args.zip_bytes},
        'check_input_only_receipt': receipt, 'new_CPU_or_native_inference_run_by_review': False, 'reviewer_SSH': False,
        'original_complete_model_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
        'limit': protocol['cross_stage_policy']}
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'check_count': len(checks), 'sha256': sha(output), 'launch_blockers': []}))


if __name__ == '__main__':
    main()
