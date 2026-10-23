"""Audit sealed two-Conv context and saved CPU evidence without inference."""
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
BUNDLE = ROOT / 'artifacts/gate05/20261020_twoconv_context_probe'
REVIEW = Path(__file__).resolve().parent
PINS = {
    'protocol.json': '6ba4cae53e1ca115c2b49bd7fdac91d80c15ef1cc1f854c57f01cd09ae185fa0',
    'spec.json': 'e9b48c0319f68e361977d093b25df8ceacb5f1827f8a413e5e7107f751af2a5c',
    'manifest.json': 'c809fcc573074108ff2d1f5d199ab105c2afdfe9810fed472659cd4abbb3959e',
    'jetson_twoconv_context_probe.py': 'b642c2406a524a5385b896724242224240cfa0d0161262131bba53a6f7e30820',
    'jetson_build_twoconv_context_probe.py': 'd5e9ba82c8076ff3bd88e8c981049e740c4186f5964fe415a6a9f9504fa3c978',
}
checks = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def check(name, value):
    checks.append({'name': name, 'passed': bool(value)})
    if not value:
        raise AssertionError(name)


def main():
    output = REVIEW / 'pre_native_review.json'
    assert not output.exists()
    b = BUNDLE
    spec, protocol, manifest = [read(b / n) for n in ['spec.json', 'protocol.json', 'manifest.json']]
    for name, digest in PINS.items():
        check('pinned_identity:' + name, sha(b / name) == digest)
    check('manifest_schema', manifest['schema'] == 'rampq.gate05.twoconv_context_probe_bundle/v1')
    for name, info in manifest['files'].items():
        path = b / name
        check('manifest_containment:' + name, path.resolve().parent == b.resolve())
        check('manifest_bytes:' + name, path.stat().st_size == info['bytes'])
        check('manifest_sha:' + name, sha(path) == info['sha256'])
    check('exact_payload_plus_two_seals', set(p.name for p in b.iterdir()) ==
          set(manifest['files']) | {'manifest.json', 'SHA256SUMS'})
    check('only_regular_files', all(p.is_file() and not p.is_symlink() for p in b.iterdir()))
    for line in (b / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        check('SHA256SUMS:' + name, sha(b / name) == digest)
    archive_path = b.with_suffix('.zip')
    check('zip_identity', sha(archive_path) == '599190d18de4f57bb80911ead9e436661a3a86daa4117817044079d71c25d525')
    check('zip_bytes', archive_path.stat().st_size == 10669204)
    with zipfile.ZipFile(archive_path) as archive:
        check('zip_crc', archive.testzip() is None)
        members = {'gate05_twoconv_context_probe_20261020/' + p.name for p in b.iterdir()}
        check('zip_exact_members', set(archive.namelist()) == members and len(archive.namelist()) == len(members))
        for name in members:
            check('zip_exact_payload:' + name, archive.read(name) == (b / name.split('/')[-1]).read_bytes())
    copies = {
        'source_model.onnx': protocol['source_graph'],
        'parent_spec.json': 'artifacts/gate05/20261019_corrected_qall_candidate/spec.json',
        'parent_manifest.json': 'artifacts/gate05/20261019_corrected_qall_candidate/manifest.json',
        'failure_inventory.json': 'artifacts/gate05/20261019_corrected_qall_native/failure_boundary_inventory.json',
        'retained_qall_failure_checkpoint.json': 'artifacts/gate05/20261019_corrected_qall_native/checkpoint.json',
        'protocol.json': 'artifacts/gate05/20261020_twoconv_context_protocol/protocol.json',
    }
    for local, original in copies.items():
        check('original_source_copy:' + local, (b / local).read_bytes() == (ROOT / original).read_bytes())
    check('earlier_independent_source_review', sha(REVIEW / 'source_readonly_review.json') ==
          '8c59140f287d3a6193e9dd5e0bed97067e478ac6425c19d78e1209f98e94efe3')
    for before, after in [('source_hashes_before.json', 'source_hashes_after.json'),
                          ('seal_native_source_hashes_before.json', 'seal_native_source_hashes_after.json')]:
        first, second = read(b / before), read(b / after)
        check('source_snapshots_equal:' + before, first == second)
        for name, info in first.items():
            check('current_source_fresh:' + before + ':' + name,
                  (ROOT / name).stat().st_size == info['bytes'] and sha(ROOT / name) == info['sha256'])
    for name, info in read(b / 'prepared_content_manifest.json')['files'].items():
        check('prepared_payload_unchanged:' + name,
              (b / name).stat().st_size == info['bytes'] and sha(b / name) == info['sha256'])
    check('exact_frozen_cases', spec['cases'] == protocol['cases'] and spec['case_order'] ==
          protocol['case_order'] == ['code_clip_none', 'code_clip_obey'])
    check('budgets_300_900', spec['max_build_seconds'] == protocol['per_case_max_build_seconds'] == 300 and
          spec['max_seconds'] == protocol['batch_max_seconds'] == 900)
    check('once_each_build2_execute4_repeat2', spec['maximum_attempts_per_case'] == 1 and
          spec['maximum_native_builds'] == 2 and spec['maximum_native_inferences'] == 4 and
          spec['executions_per_successful_case'] == 2)
    check('exact_common_native_policy', spec['native_build_policy'] == protocol['common_native_settings'])
    check('native_only_corrected_topology', set(spec['variants']) == {'code_clip'} and
          set(spec['cpu_control_variants']) == {'original', 'code_clip'})
    models = {}
    full_paths = {
        'original': ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles/q_all/model.onnx',
        'code_clip': ROOT / protocol['source_graph'],
    }
    for variant, entry in spec['cpu_control_variants'].items():
        path = b / entry['model_file']
        model, full = onnx.load(path), onnx.load(full_paths[variant])
        models[variant] = model
        check('slice_model_identity:' + variant, sha(path) == entry['model_sha256'])
        check('19_nodes_20_initializers:' + variant, len(model.graph.node) == 19 and len(model.graph.initializer) == 20)
        check('two_named_convs:' + variant,
              [n.name for n in model.graph.node if n.op_type == 'Conv'] == protocol['source_conv_names'])
        check('third_conv_excluded:' + variant,
              protocol['excluded_third_conv_name'] not in [n.name for n in model.graph.node])
        producer = {output: node for node in full.graph.node for output in node.output}
        selected, needed = set(), set()

        def visit(name):
            needed.add(name)
            node = producer.get(name)
            if node is None or node.name in selected:
                return
            selected.add(node.name)
            for item in node.input:
                if item:
                    visit(item)

        visit(protocol['output_name'])
        nodes = [n for n in full.graph.node if n.name in selected]
        initializers = [i for i in full.graph.initializer if i.name in needed]
        check('exact_closure_and_order:' + variant,
              [n.name for n in model.graph.node] == [n.name for n in nodes] and
              [i.name for i in model.graph.initializer] == [i.name for i in initializers])
        for node, source in zip(model.graph.node, nodes):
            check('exact_source_node:' + variant + ':' + node.name,
                  node.SerializeToString() == source.SerializeToString())
            check('selected_node_hash:' + variant + ':' + node.name,
                  hashlib.sha256(node.SerializeToString()).hexdigest() == entry['selected_node_serialized_sha256'][node.name])
        for tensor, source in zip(model.graph.initializer, initializers):
            check('exact_source_initializer:' + variant + ':' + tensor.name,
                  tensor.SerializeToString() == source.SerializeToString())
            check('selected_initializer_hash:' + variant + ':' + tensor.name,
                  hashlib.sha256(tensor.SerializeToString()).hexdigest() == entry['selected_initializer_serialized_sha256'][tensor.name])
        declarations = {v.name: v for v in list(full.graph.input) + list(full.graph.value_info) + list(full.graph.output)}
        check('one_input_and_output:' + variant, len(model.graph.input) == len(model.graph.output) == 1 and
              model.graph.input[0].name == 'images' and model.graph.output[0].name == protocol['output_name'])
        for value in list(model.graph.input) + list(model.graph.output):
            check('source_io_declaration:' + variant + ':' + value.name,
                  value.SerializeToString() == declarations[value.name].SerializeToString())
        first, second = copy.deepcopy(model), copy.deepcopy(full)
        first.ClearField('graph'); second.ClearField('graph')
        check('model_metadata_unchanged:' + variant, first.SerializeToString() == second.SerializeToString())
        onnx.checker.check_model(model, full_check=True)
        check('ONNX_static_check:' + variant, True)
    old, new = models['original'], models['code_clip']
    old_nodes, new_nodes = {n.name: n for n in old.graph.node}, {n.name: n for n in new.graph.node}
    old_init, new_init = {i.name: i for i in old.graph.initializer}, {i.name: i for i in new.graph.initializer}
    check('same_source_node_initializer_names', set(old_nodes) == set(new_nodes) and set(old_init) == set(new_init))
    changed_nodes, changed_init = set(), set()
    for clip_name in protocol['clip_names']:
        prefix = clip_name.removesuffix('_Clip')
        q_name, dq_name = prefix + '_ActivationQ', prefix + '_ActivationDQ'
        q, clip = old_nodes[q_name], old_nodes[clip_name]
        desired_q, desired_clip = copy.deepcopy(q), copy.deepcopy(clip)
        desired_q.input[0], desired_q.output[0] = clip.input[0], q.output[0] + '_before_code_clip'
        desired_clip.input[0], desired_clip.output[0] = desired_q.output[0], q.output[0]
        check('exact_Q_rewire:' + q_name, desired_q.SerializeToString() == new_nodes[q_name].SerializeToString())
        check('exact_Clip_rewire:' + clip_name, desired_clip.SerializeToString() == new_nodes[clip_name].SerializeToString())
        check('unchanged_DQ:' + dq_name, old_nodes[dq_name].SerializeToString() == new_nodes[dq_name].SerializeToString())
        scale, zero = onnx.numpy_helper.to_array(old_init[q.input[1]]), onnx.numpy_helper.to_array(old_init[q.input[2]])
        check('frozen_scalar_scale_zero:' + prefix, scale.dtype == np.float32 and scale.shape == () and
              zero.dtype == np.int8 and zero.shape == () and int(zero) == 0)
        for name, value in zip(clip.input[1:], [-127, 127]):
            previous, current = onnx.numpy_helper.to_array(old_init[name]), onnx.numpy_helper.to_array(new_init[name])
            check('preserved_quantizer_bounds:' + name, previous.dtype == np.float32 and previous.shape == () and
                  previous.tobytes() == np.asarray(np.float32(value) * scale, dtype=np.float32).tobytes() and
                  current.dtype == np.int8 and current.shape == () and int(current) == value)
            changed_init.add(name)
        changed_nodes.update([q_name, clip_name])
    for name in set(old_nodes) - changed_nodes:
        check('all_other_nodes_unchanged:' + name, old_nodes[name].SerializeToString() == new_nodes[name].SerializeToString())
    for name in set(old_init) - changed_init:
        check('all_other_initializers_unchanged:' + name, old_init[name].SerializeToString() == new_init[name].SerializeToString())
    check('only_six_nodes_and_six_bound_initializers', len(changed_nodes) == len(changed_init) == 6)
    image = np.load(b / spec['input_file'], allow_pickle=False)
    check('input_float32_finite_shape', image.dtype == np.float32 and image.shape == (1, 3, 640, 640) and np.isfinite(image).all())
    check('input_array_original_sha', hashlib.sha256(image.tobytes()).hexdigest() ==
          'b10ad22bef235bf2d2763672da55b43baca61db7c4640db6c3803e3390e722b6')
    check('input_original_bytes', (b / spec['input_file']).read_bytes() ==
          (ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate/preprocessed_input.npy').read_bytes())
    cpu = read(b / 'cpu_diagnostic.json')
    check('cpu_diagnostic_identity', sha(b / 'cpu_diagnostic.json') == spec['cpu_diagnostic_sha256'])
    check('cpu_two_slices_no_detector_native', cpu['slice_executions'] == 2 and
          cpu['complete_detector_executions'] == cpu['native_executions'] == 0)
    with np.load(b / spec['cpu_expected_outputs_file'], allow_pickle=False) as saved:
        check('cpu_saved_keys', set(saved.files) == {'original_output', 'code_clip_output'})
        arrays = {k: saved[k].copy() for k in saved.files}
    check('CPU_original_rewrite_byte_exact', arrays['original_output'].tobytes() == arrays['code_clip_output'].tobytes())
    for variant in ['original', 'code_clip']:
        info = cpu['executions'][variant]
        array = np.load(b / info['raw_output_file'], allow_pickle=False)
        check('cpu_provider_disable_optimization:' + variant, info['providers'] == ['CPUExecutionProvider'] and
              info['graph_optimization'] == 'ORT_DISABLE_ALL' and info['runs'] == 1)
        check('cpu_raw_file_identity:' + variant, sha(b / info['raw_output_file']) == info['raw_output_sha256'])
        check('cpu_raw_finite_float32_shape:' + variant, array.dtype == np.float32 and
              array.shape == (1, 32, 160, 160) and np.isfinite(array).all())
        check('cpu_saved_raw_matches_npz:' + variant, array.tobytes() == arrays[variant + '_output'].tobytes())
        check('cpu_raw_array_identity:' + variant, hashlib.sha256(array.tobytes()).hexdigest() ==
              info['summary']['raw_array_bytes_sha256'] == '40d65c0d2a887cd192f59e4c6797827012a7515239572ed1fed73d1912282782')
    for name in ['prepare_gate05_twoconv_context_probe.py', 'jetson_twoconv_context_probe.py',
                 'jetson_build_twoconv_context_probe.py']:
        check('current_tool_matches_bundle:' + name, (b / name).read_bytes() == (ROOT / 'scripts' / name).read_bytes())
        tree = ast.parse((b / name).read_text(encoding='utf-8'),
                         feature_version=(3, 8) if name.startswith('jetson_') else None)
        check('syntax_python38_if_native:' + name, isinstance(tree, ast.Module))
        if name.startswith('jetson_'):
            calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            stores = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)]
            check('no_precision_output_setter_or_reset:' + name,
                  not set(calls) & {'set_precision', 'reset_precision', 'set_output_type', 'reset_output_type'})
            check('no_precision_or_dtype_stores:' + name, not set(stores) & {'precision', 'dtype'})
    check('protected_status', spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
          spec['original_tolerances_changed'] is False and spec['calibration_recomputed'] is False and
          spec['weights_changed'] is False and spec['dataset_evaluation_run'] is False and spec['complete_detector_executions'] == 0)
    result = subprocess.run([sys.executable, str(b / 'jetson_twoconv_context_probe.py'), '--root', str(b),
                             '--check-input-only'], capture_output=True, text=True)
    check('stdlib_input_check_exit0', result.returncode == 0)
    receipt = json.loads(result.stdout)
    check('stdlib_input_check_status', receipt['status'] == 'FROZEN_TWOCONV_CONTEXT_INPUTS_VERIFIED' and
          receipt['case_count'] == 2 and receipt['native_runtime_imported'] is False)
    manual = [
        ('single_factor_flag_pair', 'Both NONE/OBEY use only corrected code_clip graph and identical three Clip corrections. Both flags cleared, only obey sets OBEY. INT8+FP16/noTF32/noPREFER/workspace1024/avg1/FLOATLINEAR common; actual flags and I/O queried.'),
        ('parser_constraints_preserved', 'All precision_is_set and output_type_is_set support/error states saved before/after; values only read when set. Precision/output request semantics compared; wrapper layer_type retained separately. No setters/resets.'),
        ('actual_mapping_pending', 'Every actual Conv inspector row retained, two source names separately mapped; missing/ambiguous explicit. Two distinct uniquely mapped rows required for mapping evidence. Raw execution may remain diagnostic if mapping ambiguous; no adopted precision PASS assumed.'),
        ('bounded_no_retry_cleanup', 'Each case once, direct Python API child budget300s and outer900s. Ordinary positive compile failure preserved; next case only after child exit and PRE/POST identity checks. Timeout/signal/stale/interrupt abort; runner_abort does not overwrite child failure.'),
        ('two_raw_repeats', 'Successful exact FLOATLINEAR bindings required, frozen input executed twice with synchronization. Both arrays and repeat/CPU descriptive deltas saved, no modified tolerance.'),
        ('hypothesis_not_cause', 'Third activation planned fusion in saved failed log motivates new context. No actual full engine or fault node claimed. No monotonic-prefix assumption or inference of full q_all success.'),
        ('root_operational_responsibility', 'Root still verifies live fan, memory, boot, no parallel task, outer900s monitor and supplemental owned-group guard. Reviewer used no SSH, native or new CPU inference.'),
    ]
    for name, _ in manual:
        check('manual:' + name, True)
    report = {'schema': 'rampq.gate05.twoconv_context_pre_native_independent_review/v1',
              'reviewed_client_at': datetime.now().astimezone().isoformat(),
              'status': 'PRE_NATIVE_STATIC_SOURCE_AND_SAVED_CPU_REVIEW_PASSED_ACTUAL_ADOPTION_PENDING',
              'launch_blockers': [], 'check_count': len(checks), 'checks': checks,
              'manual_review': [{'name': n, 'assessment': v} for n, v in manual],
              'identities': {**{n: sha(b / n) for n in PINS}, 'bundle_zip_sha256': sha(archive_path),
                             'source_readonly_review_sha256': sha(REVIEW / 'source_readonly_review.json')},
              'check_input_only_receipt': receipt, 'reviewer_new_cpu_inference': False,
              'local_review_tool_note': 'An initial inline read stopped at a Windows default GBK source decode error before signing. Re-reading the same sealed UTF-8 bytes with explicit encoding fixes only the local review tool; no native rerun, new CPU inference or source/evidence change occurred.',
              'reviewer_native_inference': False, 'reviewer_SSH': False,
              'original_complete_model_guard': 'FAILED_UNCHANGED', 'dataset_evaluation_run': False,
              'gate05_status': 'INCOMPLETE', 'limits': protocol['interpretation']}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'report': str(output), 'sha256': sha(output),
                      'check_count': len(checks), 'launch_blockers': []}))


if __name__ == '__main__':
    main()
