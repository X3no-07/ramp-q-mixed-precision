"""Audit source-exact Softmax generator, cut and saved CPU files; no inference."""
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
BUNDLE = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_probe'
REVIEW = Path(__file__).resolve().parent
PINS = {
    'protocol.json': 'bbe50e6485a72df194a13684895b4cc29a1db6ceca8ef098d05a6b5e56c56055',
    'spec.json': '45f284588d37e607478ce79e443d46a49b870ef0d65614c5ce0867130f50316b',
    'manifest.json': '6c456b5a08d34a9a3fb87d7fced5474f2c349d632076bfc0fb5eb8dd793ddde1',
    'jetson_dfl_softmax_cut_probe.py': '7f1f90a3dcc9ff40ff717af741fc2340da7c2c3ced275aa57fd1c44e6ce82d30',
    'jetson_build_dfl_softmax_cut_probe.py': 'bed3be9c7b07acf73daac884191d7199b3ebe07c88b5e664762d71a557e18146',
    'model_dfl_source_softmax.onnx': '09bd1b2903cfa5208a54a15fb85c8f1c7074b4f655d9880b49b30bc7c70a1c06',
    'model_dfl_softmax_cut_code_clip.onnx': '1f8662a55675152d57d248b16c2eebec08c4d54b5b6ebc4841f0c177d6ac2298',
    'dfl_softmax_probabilities.npy': '5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    output = REVIEW / 'pre_native_review.json'
    assert not output.exists()
    bundle = BUNDLE
    checks = []

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
        if not value:
            raise AssertionError(name)

    for name, digest in PINS.items():
        check('pinned_identity:' + name, sha(bundle / name) == digest)
    spec, protocol, manifest = [read(bundle / name) for name in ['spec.json', 'protocol.json', 'manifest.json']]
    check('manifest_schema', manifest['schema'] == 'rampq.gate05.dfl_softmax_cut_probe_bundle/v1')
    for name, info in manifest['files'].items():
        path = bundle / name
        check('manifest_path:' + name, path.resolve().parent == bundle.resolve())
        check('manifest_bytes:' + name, path.stat().st_size == info['bytes'])
        check('manifest_sha:' + name, sha(path) == info['sha256'])
    check('exact_payload_plus_seals', {p.name for p in bundle.iterdir()} == set(manifest['files']) | {'manifest.json', 'SHA256SUMS'})
    check('regular_files_no_symlinks', all(p.is_file() and not p.is_symlink() for p in bundle.iterdir()))
    for line in (bundle / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        check('SHA256SUMS:' + name, sha(bundle / name) == digest)
    archive_path = bundle.with_suffix('.zip')
    check('zip_identity', sha(archive_path) == 'aeff82c52ff3b2ec1d83b4b06c21f014b30699a1e428a0c5fb37bcd3f6cb25dd')
    check('zip_size', archive_path.stat().st_size == 13574095)
    with zipfile.ZipFile(archive_path) as archive:
        check('zip_crc', archive.testzip() is None)
        members = {'gate05_dfl_softmax_cut_probe_20261020/' + p.name for p in bundle.iterdir()}
        check('zip_exact_members', set(archive.namelist()) == members and len(archive.namelist()) == len(members))
        for name in members:
            check('zip_original_payload:' + name, archive.read(name) == (bundle / name.split('/')[-1]).read_bytes())
    parent = ROOT / 'artifacts/gate05/20261020_dfl_context_probe'
    copies = {
        'saved_dfl_logits.npy': parent / 'dfl_transpose_fixture.npy',
        'cpu_expected_outputs.npz': parent / 'cpu_expected_outputs.npz',
        'saved_cpu_reference.npy': parent / 'cpu_actual_code_clip.npy',
        'parent_fixture_lineage.json': parent / 'fixture_lineage.json',
        'parent_dfl_spec.json': parent / 'spec.json',
        'parent_dfl_manifest.json': parent / 'manifest.json',
        'source_model.onnx': ROOT / protocol['source_graph'],
        'parent_spec.json': ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate/spec.json',
        'parent_manifest.json': ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate/manifest.json',
        'parent_dfl_original_checkpoint.json': ROOT / 'artifacts/gate05/20261020_dfl_original_native/checkpoint.json',
        'parent_next_control_plan.json': ROOT / 'artifacts/gate05/20261020_dfl_original_native/next_control_plan.json',
        'protocol.json': ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_protocol/protocol.json',
    }
    for name, source in copies.items():
        check('historical_source_exact:' + name, (bundle / name).read_bytes() == source.read_bytes())
    check('previous_source_review_identity', sha(REVIEW / 'source_readonly_review.json') ==
          '62a49becab274e17ac31ca6dd1e32a1ad2ae89e447709f9be646fd9c262ae45a')
    for before, after in [('source_hashes_before.json', 'source_hashes_after.json'),
                          ('seal_native_source_hashes_before.json', 'seal_native_source_hashes_after.json')]:
        first, last = read(bundle / before), read(bundle / after)
        check('source_snapshots_equal:' + before, first == last)
        for name, info in first.items():
            check('source_current_fresh:' + before + ':' + name,
                  (ROOT / name).stat().st_size == info['bytes'] and sha(ROOT / name) == info['sha256'])
    for name, info in read(bundle / 'prepared_content_manifest.json')['files'].items():
        check('prepared_payload_unchanged:' + name,
              (bundle / name).stat().st_size == info['bytes'] and sha(bundle / name) == info['sha256'])
    check('frozen_cases', spec['cases'] == protocol['cases'] and spec['case_order'] == ['code_clip_none', 'code_clip_obey'])
    check('counts_and_budgets', spec['maximum_native_builds'] == 2 and spec['maximum_native_inferences'] == 4 and
          spec['executions_per_successful_case'] == 2 and spec['maximum_attempts_per_case'] == 1 and
          spec['max_build_seconds'] == 300 and spec['max_seconds'] == 900)
    check('common_native_settings_exact', spec['native_build_policy'] == protocol['common_native_settings'])
    check('CPU_counts_only_two_no72Conv', spec['new_CPU_execution_runs_this_stage'] == 2 and spec['new_72Conv_upstream_runs'] == 0)
    source = onnx.load(bundle / 'source_model.onnx')
    source_nodes = {n.name: n for n in source.graph.node}
    source_init = {i.name: i for i in source.graph.initializer}
    producer = {value: node for node in source.graph.node for value in node.output}
    declarations = {v.name: v for v in list(source.graph.input) + list(source.graph.value_info) + list(source.graph.output)}
    identity = read(bundle / 'graph_identity_audit.json')
    models = {}
    for label, input_name, output_name, expected_count, init_count in [
        ('Softmax_generator', protocol['fixture_generation']['generator_input'], protocol['input_name'], 1, 0),
        ('code_clip_cut', protocol['input_name'], protocol['output_name'], 8, 7),
    ]:
        entry = identity[label]
        model = onnx.load(bundle / entry['model_file']); models[label] = model
        check('model_identity:' + label, sha(bundle / entry['model_file']) == entry['model_sha256'])
        check('model_counts:' + label, len(model.graph.node) == expected_count and len(model.graph.initializer) == init_count)
        needed, selected = set(), set()

        def visit(name):
            needed.add(name)
            if name == input_name:
                return
            node = producer.get(name)
            if node is None or node.name in selected:
                return
            selected.add(node.name)
            for value in node.input:
                if value:
                    visit(value)

        visit(output_name)
        required_nodes = [n for n in source.graph.node if n.name in selected]
        required_init = [i for i in source.graph.initializer if i.name in needed]
        check('exact_dependency_closure:' + label,
              [n.name for n in model.graph.node] == [n.name for n in required_nodes] and
              [i.name for i in model.graph.initializer] == [i.name for i in required_init])
        for node in model.graph.node:
            check('exact_source_node:' + label + ':' + node.name, node.SerializeToString() == source_nodes[node.name].SerializeToString())
            check('identity_node_hash:' + label + ':' + node.name,
                  hashlib.sha256(node.SerializeToString()).hexdigest() == entry['node_serialized_sha256'][node.name])
        for tensor in model.graph.initializer:
            check('exact_source_initializer:' + label + ':' + tensor.name, tensor.SerializeToString() == source_init[tensor.name].SerializeToString())
            check('identity_initializer_hash:' + label + ':' + tensor.name,
                  hashlib.sha256(tensor.SerializeToString()).hexdigest() == entry['initializer_serialized_sha256'][tensor.name])
        check('one_input_one_output:' + label, len(model.graph.input) == len(model.graph.output) == 1 and
              model.graph.input[0].name == input_name and model.graph.output[0].name == output_name)
        for value in list(model.graph.input) + list(model.graph.output):
            check('source_io_decl:' + label + ':' + value.name, value.SerializeToString() == declarations[value.name].SerializeToString())
        first, last = copy.deepcopy(model), copy.deepcopy(source)
        first.ClearField('graph'); last.ClearField('graph')
        check('model_metadata_unchanged:' + label, first.SerializeToString() == last.SerializeToString())
        onnx.checker.check_model(model, full_check=True)
        check('ONNX_static_check:' + label, True)
    generator = models['Softmax_generator']
    check('generator_exact_source_Softmax_axis1', generator.graph.node[0].op_type == 'Softmax' and
          generator.graph.node[0].name == protocol['source_Softmax_node_name'] and
          hashlib.sha256(generator.graph.node[0].SerializeToString()).hexdigest() == protocol['source_Softmax_node_serialized_sha256'] and
          next(a.i for a in generator.graph.node[0].attribute if a.name == 'axis') == 1)
    cut = models['code_clip_cut']
    check('cut_no_Softmax_and_one_Conv', all(n.op_type != 'Softmax' for n in cut.graph.node) and
          [n.name for n in cut.graph.node if n.op_type == 'Conv'] == protocol['source_conv_names'])
    check('cut_first_Q_consumes_probabilities', cut.graph.node[0].op_type == 'QuantizeLinear' and cut.graph.node[0].input[0] == protocol['input_name'])
    lineage, cpu, plan = [read(bundle / name) for name in ['fixture_lineage.json', 'cpu_diagnostic.json', 'cpu_execution_plan.json']]
    for name, key in [('fixture_lineage.json', 'fixture_lineage_sha256'), ('cpu_diagnostic.json', 'cpu_diagnostic_sha256'),
                      ('graph_identity_audit.json', 'graph_identity_audit_sha256')]:
        check('derived_record_identity:' + name, sha(bundle / name) == spec[key])
    logits = np.load(bundle / 'saved_dfl_logits.npy', allow_pickle=False)
    probabilities = np.load(bundle / spec['input_file'], allow_pickle=False)
    check('old_logits_metadata', logits.dtype == np.float32 and logits.shape == (1, 16, 4, 8400) and np.isfinite(logits).all())
    check('probability_fixture_metadata_domain', probabilities.dtype == np.float32 and probabilities.shape == (1, 16, 4, 8400) and
          np.isfinite(probabilities).all() and float(probabilities.min()) >= 0 and float(probabilities.max()) <= 1)
    check('probability_fixture_raw_identity', hashlib.sha256(probabilities.tobytes()).hexdigest() ==
          lineage['fixture']['raw_array_bytes_sha256'] == 'd4c4b21e1d1a2e025d5b3eabaade7e99f3093c0cf7511795f9477a4ffd069ae7')
    check('probability_lineage_file_identity', lineage['fixture_file'] == spec['input_file'] == 'dfl_softmax_probabilities.npy' and
          lineage['fixture_sha256'] == spec['input_file_sha256'] == sha(bundle / spec['input_file']))
    sums = probabilities.sum(axis=1, dtype=np.float64)
    check('probability_sums_descriptive_recompute', lineage['probability_axis1_sums_descriptive'] ==
          {'min': float(sums.min()), 'max': float(sums.max()), 'max_abs_from_one': float(np.abs(sums - 1).max())})
    check('generator_single_CPU_no_oldreruns', lineage['new_generator_CPU_runs'] == 1 and
          lineage['new_72Conv_or_old_DFL_or_full_detector_runs'] == 0 and lineage['provider'] == 'CPUExecutionProvider' and
          lineage['graph_optimization'] == 'ORT_DISABLE_ALL' and lineage['threads'] == 4)
    check('two_CPU_plan_frozen', plan['status'] == 'FROZEN_BEFORE_FIRST_CPU_EXECUTION' and
          plan['ordered_runs'] == ['exact_source_Softmax_once', 'corrected_eight_node_cut_once'] and
          plan['maximum_total_new_CPU_runs'] == 2 and plan['new_72Conv_or_old_DFL_or_full_detector_runs'] == 0)
    actual = np.load(bundle / cpu['actual_output_file'], allow_pickle=False)
    reference = np.load(bundle / 'saved_cpu_reference.npy', allow_pickle=False)
    check('local_CPU_composition_counts', cpu['new_CPU_executions'] == 2 and cpu['generator_CPU_runs'] == cpu['cut_CPU_runs'] == 1 and
          cpu['new_72Conv_or_old_DFL_or_full_detector_runs'] == cpu['native_executions'] == 0)
    check('local_CPU_output_metadata', actual.dtype == reference.dtype == np.float32 and actual.shape == reference.shape == (1, 4, 8400) and np.isfinite(actual).all())
    check('local_CPU_composition_raw_byte_exact', actual.tobytes() == reference.tobytes() and cpu['both_outputs_byte_exact'] is True and
          cpu['byte_identical_saved_corrected_DFL_output'] is True and cpu['differing_elements'] == 0 and cpu['max_absolute_difference'] == 0)
    check('local_CPU_NPY_same_saved_old_bytes', sha(bundle / cpu['actual_output_file']) == cpu['actual_output_sha256'] ==
          protocol['saved_corrected_CPU_npy_sha256'] == 'fd4bcdcee24cd53651fdd7a508423c802f3b432c759647204c7b748f22abd50d')
    with np.load(bundle / spec['cpu_expected_outputs_file'], allow_pickle=False) as saved:
        check('saved_NPZ_two_keys', set(saved.files) == {'original_output', 'code_clip_output'})
        check('saved_expected_NPZ_both_equal_new_cut', all(saved[name].tobytes() == actual.tobytes() for name in saved.files))
    for name in ['prepare_gate05_dfl_softmax_cut_probe.py', 'jetson_dfl_softmax_cut_probe.py', 'jetson_build_dfl_softmax_cut_probe.py']:
        check('tool_current_source_exact:' + name, (bundle / name).read_bytes() == (ROOT / 'scripts' / name).read_bytes())
        tree = ast.parse((bundle / name).read_text(encoding='utf-8'), feature_version=(3, 8) if name.startswith('jetson_') else None)
        check('tool_source_syntax:' + name, isinstance(tree, ast.Module))
        if name.startswith('jetson_'):
            calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            stores = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)]
            check('no_layer_precision_output_setter_or_reset:' + name,
                  not set(calls) & {'set_precision', 'reset_precision', 'set_output_type', 'reset_output_type'})
            check('no_direct_precision_dtype_store:' + name, not set(stores) & {'precision', 'dtype'})
    proc = subprocess.run([sys.executable, str(bundle / 'jetson_dfl_softmax_cut_probe.py'), '--root', str(bundle),
                           '--check-input-only'], capture_output=True, text=True)
    check('sealed_stdlib_inputcheck_exit0', proc.returncode == 0)
    receipt = json.loads(proc.stdout)
    check('sealed_stdlib_inputcheck_status', receipt['status'] == 'FROZEN_DFL_SOFTMAX_CUT_INPUTS_VERIFIED' and
          receipt['case_count'] == 2 and receipt['native_runtime_imported'] is False)
    check('original_guard_and_scope', spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
          spec['new_tolerances_defined'] is False and spec['weights_changed'] is False and spec['calibration_recomputed'] is False and
          spec['dataset_evaluation_run'] is False and spec['complete_detector_executions'] == 0)
    manual = [
        ('source_exact_composition', 'One exact source Softmax axis1 once creates the probability fixture from old authenticated logits; one exact eight-node cut CPU execution is byte-equal to saved old DFL CPU outputs. Only two new CPU runs, no72Conv or oldnative reruns.'),
        ('probability_boundary_not_logits', 'Native consumes external FLOAT[1,16,4,8400] Softmax probabilities. Both parsed network and source cut contain no Softmax. [0,1] inputdomain is not a new output acceptance tolerance; probability sums are descriptive.'),
        ('only_OBEY_case_factor', 'Same cut/input/Clip correction path for NONE/OBEY; INT8+FP16/noTF32/noPREFER/workspace1024/avg1/FLOATLINEAR, actual flags queried. Only OBEY toggles.'),
        ('parser_requests_preserved', 'Only named Clip alpha/beta corrected to -127/+127. Every parser precision/output request and unsupported/error state recorded before/after; request semantics preserved. No layer setter/reset.'),
        ('bounded_safe_case_execution', 'Two once-only builders300s each, batch900s; ordinary compile failures saved and next case only after child exit/inputrevalidation. Timeout/signal/stale/interrupt abort and cleanup. Success fixedprobabilities repeated twice with raw saved.'),
        ('actual_formats_pending', 'One DFL source Conv mapping, actual input/weight/output/tactic each independently recorded. No output QDQ or requirement for actual INT8 output; adopted precision remains pending receipt review.'),
        ('no_unique_Softmax_cause', 'Cut also changes external input boundary, compiler fusion context and uses CPU-generated probabilities; a changed result cannot isolate Softmax as unique fullmodel fault or establish a fullmodel workaround. Output DFL distances only, not boxes.'),
        ('root_launch_responsibility', 'Root still verifies current boardboot/fan/memory/no parallel work and outer900s/private group+guard. Reviewer uses no SSH or new CPU/native inference.'),
    ]
    for name, _ in manual:
        check('manual:' + name, True)
    report = {'schema': 'rampq.gate05.dfl_softmax_cut_pre_native_independent_review/v1',
              'client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'status': 'PRE_NATIVE_SOFTMAX_CUT_STATIC_AND_SAVED_CPU_REVIEW_PASSED_ADOPTION_PENDING',
              'launch_blockers': [], 'check_count': len(checks), 'checks': checks,
              'manual_review': [{'name': n, 'assessment': v} for n, v in manual],
              'identities': {**{n: sha(bundle / n) for n in PINS}, 'bundle_zip_sha256': sha(archive_path),
                             'source_readonly_review_sha256': sha(REVIEW / 'source_readonly_review.json')},
              'check_input_only_receipt': receipt, 'new_CPU_or_native_inference_run_by_review': False,
              'reviewer_SSH': False, 'original_complete_model_guard': 'FAILED_UNCHANGED',
              'dataset_evaluation_run': False, 'gate05_status': 'INCOMPLETE', 'limits': protocol['cross_stage_policy']}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'report': str(output), 'sha256': sha(output),
                      'check_count': len(checks), 'launch_blockers': []}))


if __name__ == '__main__':
    main()
