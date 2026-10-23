"""Independently audit original DFL local controls without new inference."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / 'artifacts/gate05/20261020_dfl_original_probe'
PARENT = ROOT / 'artifacts/gate05/20261020_dfl_context_probe'
REVIEW = Path(__file__).resolve().parent
PINS = {
    'protocol.json': '81200cd4900a35f0533c45c7542f6dacc353203e8338a09b47cd24465cc0f6ed',
    'spec.json': '99d7b3179d9c540c06331a2135de9d645b2421722edb20ff5268e53c7b56f1ae',
    'manifest.json': 'aa8aa071beb59e40e48f3c0022fcf187f430285b1651df200d7c554623f1d064',
    'jetson_dfl_original_probe.py': '91fc095c0e923bf45cacd8f32f0838e17d9ce39406a27628bccb93e1438e40e1',
    'jetson_build_dfl_original_probe.py': '2787202b71cc058cce7d00c3ea4c2490aa7dcdf2831088e3e01aea2b17e28fc1',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    output = REVIEW / 'pre_native_review.json'
    assert not output.exists()
    checks = []

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
        if not value:
            raise AssertionError(name)

    bundle = BUNDLE
    spec, protocol, manifest = [read(bundle / n) for n in ['spec.json', 'protocol.json', 'manifest.json']]
    for name, digest in PINS.items():
        check('pinned_identity:' + name, sha(bundle / name) == digest)
    check('manifest_schema', manifest['schema'] == 'rampq.gate05.dfl_original_probe_bundle/v1')
    for name, info in manifest['files'].items():
        path = bundle / name
        check('manifest_containment:' + name, path.resolve().parent == bundle.resolve())
        check('manifest_size:' + name, path.stat().st_size == info['bytes'])
        check('manifest_sha:' + name, sha(path) == info['sha256'])
    check('payload_exact_plus_seals', {p.name for p in bundle.iterdir()} == set(manifest['files']) | {'manifest.json', 'SHA256SUMS'})
    check('all_regular_no_links', all(p.is_file() and not p.is_symlink() for p in bundle.iterdir()))
    for line in (bundle / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        check('SHA256SUMS:' + name, sha(bundle / name) == digest)
    archive_path = bundle.with_suffix('.zip')
    check('zip_identity', sha(archive_path) == '6ea5c52c4697339692d3d5e4efac01a46d033e34855a3ed8f60e4aa1d1a22709')
    check('zip_size', archive_path.stat().st_size == 11545669)
    with zipfile.ZipFile(archive_path) as archive:
        check('zip_crc', archive.testzip() is None)
        members = {'gate05_dfl_original_probe_20261020/' + p.name for p in bundle.iterdir()}
        check('zip_exact_members', set(archive.namelist()) == members and len(archive.namelist()) == len(members))
        for name in members:
            check('zip_payload_bytes:' + name, archive.read(name) == (bundle / name.split('/')[-1]).read_bytes())
    copies = {
        'model_dfl_original.onnx': PARENT / 'model_dfl_original.onnx',
        'dfl_transpose_fixture.npy': PARENT / 'dfl_transpose_fixture.npy',
        'cpu_expected_outputs.npz': PARENT / 'cpu_expected_outputs.npz',
        'cpu_actual_original.npy': PARENT / 'cpu_actual_original.npy',
        'cpu_actual_code_clip.npy': PARENT / 'cpu_actual_code_clip.npy',
        'parent_cpu_diagnostic.json': PARENT / 'cpu_diagnostic.json',
        'fixture_lineage.json': PARENT / 'fixture_lineage.json',
        'parent_dfl_spec.json': PARENT / 'spec.json',
        'parent_dfl_manifest.json': PARENT / 'manifest.json',
        'parent_dfl_protocol.json': PARENT / 'protocol.json',
        'parent_dfl_checkpoint.json': ROOT / 'artifacts/gate05/20261020_dfl_context_native/checkpoint.json',
        'source_original_full_model.onnx': ROOT / protocol['source_graph'],
        'protocol.json': ROOT / 'artifacts/gate05/20261020_dfl_original_protocol/protocol.json',
    }
    for name, source in copies.items():
        check('historical_source_exact:' + name, (bundle / name).read_bytes() == source.read_bytes())
    check('earlier_source_review', sha(REVIEW / 'source_readonly_review.json') ==
          '802eb684b0c6b91866222e12239e76c005d4268bbe34f068d193a2a9f3899ca4')
    for before, after in [('source_hashes_before.json', 'source_hashes_after.json'),
                          ('seal_native_source_hashes_before.json', 'seal_native_source_hashes_after.json')]:
        first, second = read(bundle / before), read(bundle / after)
        check('source_snapshots_equal:' + before, first == second)
        for name, info in first.items():
            check('source_current_fresh:' + before + ':' + name,
                  (ROOT / name).stat().st_size == info['bytes'] and sha(ROOT / name) == info['sha256'])
    for name, info in read(bundle / 'prepared_content_manifest.json')['files'].items():
        check('prepared_payload_unmodified:' + name,
              (bundle / name).stat().st_size == info['bytes'] and sha(bundle / name) == info['sha256'])
    check('frozen_case_order', spec['cases'] == protocol['cases'] and spec['case_order'] == ['original_none', 'original_obey'])
    check('no_Clip_mutation_permitted', spec['native_clip_attribute_mutation_permitted'] is False and
          spec['requires_explicit_native_importer_clip_attribute_correction'] is False and
          all(v['apply_native_clip_attribute_correction'] is False for v in spec['cases']))
    check('zero_new_CPU_execution', spec['new_CPU_execution_runs_this_stage'] == protocol['CPU_execution_runs_this_stage'] == 0)
    check('budgets_counts', spec['max_seconds'] == 900 and spec['max_build_seconds'] == 300 and
          spec['maximum_attempts_per_case'] == 1 and spec['maximum_native_builds'] == 2 and
          spec['maximum_native_inferences'] == 4 and spec['executions_per_successful_case'] == 2)
    check('exact_common_native_settings', spec['native_build_policy'] == protocol['common_native_settings'])
    entry = spec['variants']['original']
    check('only_original_variant', set(spec['variants']) == {'original'})
    model_path = bundle / entry['model_file']
    model = onnx.load(model_path)
    source = onnx.load(bundle / 'source_original_full_model.onnx')
    check('original_tail_identity', sha(model_path) == entry['model_sha256'] == protocol['original_tail_sha256'])
    check('original_full_source_identity', sha(bundle / 'source_original_full_model.onnx') == protocol['source_graph_sha256'])
    check('tail_9nodes7init1Conv', len(model.graph.node) == 9 and len(model.graph.initializer) == 7 and
          [n.name for n in model.graph.node if n.op_type == 'Conv'] == protocol['source_conv_names'])
    producers = {name: node for node in source.graph.node for name in node.output}
    needed, selected = set(), set()

    def visit(name):
        needed.add(name)
        if name == protocol['input_name']:
            return
        node = producers.get(name)
        if node is None or node.name in selected:
            return
        selected.add(node.name)
        for value in node.input:
            if value:
                visit(value)

    visit(protocol['output_name'])
    source_nodes = [n for n in source.graph.node if n.name in selected]
    source_initializers = [i for i in source.graph.initializer if i.name in needed]
    check('exact_dependency_closure_and_order', [n.name for n in model.graph.node] == [n.name for n in source_nodes] and
          [i.name for i in model.graph.initializer] == [i.name for i in source_initializers])
    for node, original in zip(model.graph.node, source_nodes):
        check('source_node_bytes:' + node.name, node.SerializeToString() == original.SerializeToString())
        check('selected_node_hash:' + node.name, hashlib.sha256(node.SerializeToString()).hexdigest() == entry['selected_node_serialized_sha256'][node.name])
    for tensor, original in zip(model.graph.initializer, source_initializers):
        check('source_initializer_bytes:' + tensor.name, tensor.SerializeToString() == original.SerializeToString())
        check('selected_initializer_hash:' + tensor.name, hashlib.sha256(tensor.SerializeToString()).hexdigest() == entry['selected_initializer_serialized_sha256'][tensor.name])
    declarations = {v.name: v for v in list(source.graph.input) + list(source.graph.value_info) + list(source.graph.output)}
    check('tail_one_input_one_output', len(model.graph.input) == len(model.graph.output) == 1 and
          model.graph.input[0].name == protocol['input_name'] and model.graph.output[0].name == protocol['output_name'])
    for value in list(model.graph.input) + list(model.graph.output):
        check('source_io_declaration:' + value.name, value.SerializeToString() == declarations[value.name].SerializeToString())
    first, second = copy.deepcopy(model), copy.deepcopy(source)
    first.ClearField('graph'); second.ClearField('graph')
    check('model_metadata_unchanged', first.SerializeToString() == second.SerializeToString())
    onnx.checker.check_model(model, full_check=True)
    check('ONNX_static_check', True)
    nodes, initializers = {n.name: n for n in model.graph.node}, {i.name: i for i in model.graph.initializer}
    clip, quantizer, dequantizer = [nodes['rampq_model_23_dfl_conv' + suffix] for suffix in ['_Clip', '_ActivationQ', '_ActivationDQ']]
    check('FLOAT_Clip_before_Q_DQ', clip.input[0] == '/model.23/dfl/Softmax_output_0' and
          quantizer.input[0] == clip.output[0] and dequantizer.input[0] == quantizer.output[0])
    scale = onnx.numpy_helper.to_array(initializers[quantizer.input[1]])
    for name, code in zip(clip.input[1:], [-127, 127]):
        array = onnx.numpy_helper.to_array(initializers[name])
        check('original_FLOAT_Clip_bound:' + name, array.dtype == np.float32 and array.shape == () and
              array.tobytes() == np.asarray(np.float32(code) * scale, dtype=np.float32).tobytes())
    fixture = np.load(bundle / spec['input_file'], allow_pickle=False)
    check('reused_fixture_identity', sha(bundle / spec['input_file']) == protocol['fixture_sha256'] == spec['input_file_sha256'])
    check('reused_fixture_shape_dtype_finite', fixture.shape == (1, 16, 4, 8400) and fixture.dtype == np.float32 and np.isfinite(fixture).all())
    check('reused_fixture_raw_sha', hashlib.sha256(fixture.tobytes()).hexdigest() == '799c5f7875edd7f9a3e46c8beba833b69b56baf04f76f060015dab2132cbb52a')
    with np.load(bundle / spec['cpu_expected_outputs_file'], allow_pickle=False) as saved:
        check('CPU_saved_two_keys', set(saved.files) == {'original_output', 'code_clip_output'})
        arrays = {name: saved[name].copy() for name in saved.files}
    check('reused_CPU_outputs_byte_equal', arrays['original_output'].tobytes() == arrays['code_clip_output'].tobytes())
    for variant in ['original', 'code_clip']:
        saved = np.load(bundle / ('cpu_actual_' + variant + '.npy'), allow_pickle=False)
        check('reused_saved_raw_NPY_metadata:' + variant, saved.shape == (1, 4, 8400) and saved.dtype == np.float32 and np.isfinite(saved).all())
        check('reused_saved_raw_matches_NPZ:' + variant, saved.tobytes() == arrays[variant + '_output'].tobytes())
        check('reused_saved_raw_array_sha:' + variant, hashlib.sha256(saved.tobytes()).hexdigest() ==
              '3e4280d6a97b12d4e0c33eeb1e0a462ab3d457cef8eca9ce2ac5554816f45436')
    reuse = read(bundle / 'static_reuse_audit.json')
    check('reuse_audit_identity', sha(bundle / 'static_reuse_audit.json') == spec['static_reuse_audit_sha256'])
    for key in ['new_CPU_executions', 'new_ORT_executions', 'new_Torch_executions', 'new_upstream_executions', 'new_native_executions']:
        check('reuse_no_new_execution:' + key, reuse[key] == 0)
    check('no_new_fixture_generated', reuse['new_fixture_generated'] is False)
    for name in ['prepare_gate05_dfl_original_probe.py', 'jetson_dfl_original_probe.py', 'jetson_build_dfl_original_probe.py']:
        check('tool_current_source_exact:' + name, (bundle / name).read_bytes() == (ROOT / 'scripts' / name).read_bytes())
        tree = ast.parse((bundle / name).read_text(encoding='utf-8'), feature_version=(3, 8) if name.startswith('jetson_') else None)
        check('source_syntax:' + name, isinstance(tree, ast.Module))
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        stores = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)]
        check('no_Clip_attribute_assignment:' + name, not set(stores) & {'alpha', 'beta'})
        if name.startswith('jetson_'):
            check('no_layer_precision_output_set_or_reset:' + name, not set(calls) & {'set_precision', 'reset_precision', 'set_output_type', 'reset_output_type'})
            check('no_direct_precision_dtype_store:' + name, not set(stores) & {'precision', 'dtype'})
        if name.startswith('prepare_'):
            imports = [n.name for n in ast.walk(tree) if isinstance(n, ast.Import) for n in n.names]
            check('preparer_no_ORT_Torch_import', not set(imports) & {'onnxruntime', 'torch'})
            check('preparer_no_execution_APIs', not set(calls) & {'run', 'execute_v2', 'InferenceSession'})
    proc = subprocess.run([sys.executable, str(bundle / 'jetson_dfl_original_probe.py'), '--root', str(bundle),
                           '--check-input-only'], capture_output=True, text=True)
    check('stdlib_input_check_exit0', proc.returncode == 0)
    receipt = json.loads(proc.stdout)
    check('stdlib_input_check_status', receipt['status'] == 'FROZEN_DFL_ORIGINAL_INPUTS_VERIFIED' and
          receipt['case_count'] == 2 and receipt['native_runtime_imported'] is False)
    check('protected_scope', spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
          spec['weights_changed'] is False and spec['calibration_recomputed'] is False and spec['original_tolerances_changed'] is False and
          spec['dataset_evaluation_run'] is False and spec['complete_detector_executions'] == 0)
    manual = [
        ('original_bounds_read_only', 'Both original cases only query the parsed FLOAT Clip alpha/beta before/after; no alpha/beta assignment exists and unchanged attributes required. No code_clip correction allowed.'),
        ('only_OBEY_case_flag_differs', 'Same original tail and internal fixture, INT8+FP16, no TF32/PREFER, workspace1024MiB/avg1/FLOATLINEAR. Both constraint flags cleared; only obey sets OBEY. Actual get_flag and binding records required.'),
        ('parser_requests_preserved', 'All set-state precision/output metadata and explicit values only when set preserved before/after, request semantics compared. No layer precision/output setter/reset.'),
        ('bounded_failure_and_raw_policy', 'Two one-shot builders each300s, batch900s. Ordinary compile failure saved then child exit/identity checked before next case; timeout/signal/stale/interrupt abort. Successful engines only execute frozen feature twice; all actual Conv rows and input/weight/tactic/output types saved without INT8-output requirement.'),
        ('cross_stage_scope', 'Old corrected NONE success/OBEY failure stays immutable and is not rerun. New original observations share saved fixture/reference; topology plus importer correction is one combined derivative factor, not independent subfactors or a randomized/repeated factorial trial.'),
        ('fixture_reference_scope', 'Fixture inherited from corrected ONNX CPU 72Conv ancestors, not native activations or original/corrected upstream equivalence. No new CPU inference or fixture generation; DFL distance output, not detection boxes.'),
        ('protected_limits', 'No named full-model fault node, unique cause, full-model flag workaround, complete guard PASS, dataset accuracy, performance or Gate passage follows from this local control.'),
        ('root_operational_responsibility', 'Root verifies live fan/memory/boot/task absence and outer900s monitor/private-group supplemental guard before launch. Reviewer performed no SSH, CPU/native inference.'),
    ]
    for name, _ in manual:
        check('manual:' + name, True)
    report = {'schema': 'rampq.gate05.dfl_original_pre_native_independent_review/v1',
              'reviewed_client_at': datetime.now().astimezone().isoformat(),
              'status': 'PRE_NATIVE_ORIGINAL_DFL_STATIC_REUSE_REVIEW_PASSED_ACTUAL_ADOPTION_PENDING',
              'launch_blockers': [], 'check_count': len(checks), 'checks': checks,
              'manual_review': [{'name': n, 'assessment': v} for n, v in manual],
              'identities': {**{n: sha(bundle / n) for n in PINS}, 'bundle_zip_sha256': sha(archive_path),
                             'source_readonly_review_sha256': sha(REVIEW / 'source_readonly_review.json')},
              'check_input_only_receipt': receipt, 'new_CPU_or_native_inference': False, 'reviewer_SSH': False,
              'original_complete_model_guard': 'FAILED_UNCHANGED', 'dataset_evaluation_run': False,
              'gate05_status': 'INCOMPLETE', 'limits': protocol['interpretation']}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'report': str(output), 'sha256': sha(output),
                      'check_count': len(checks), 'launch_blockers': []}))


if __name__ == '__main__':
    main()
