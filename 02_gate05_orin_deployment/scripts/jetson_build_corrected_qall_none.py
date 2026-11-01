"""Build one new full corrected q_all NONE control, without executing its engine.

The source_bundle spec and reference contract remain unchanged legacy OBEY
records. Only the outer frozen protocol authorizes this separate NONE attempt.
--check-input-only uses the standard library and performs no inference/build.
The caller must provide the frozen external monitor and owned process guard.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback


PROTOCOL_SHA256 = '572120502a6db021945c63cdd781d81e2a4b1771f5d9a13c3f58d0da7278eac1'
HELPER_RELATIVE_PATH = 'jetson_build_precision_candidate.py'
HELPER_SHA256 = 'd8431345da90482f0d5798e2d6524bf03591ecef653edfad5ef3e28a509eb19f'
GRAPH_SHA256 = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
PARENT_GRAPH_SHA256 = '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d'
SOURCE_SPEC_SHA256 = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
SOURCE_MANIFEST_SHA256 = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
REWRITE_AUDIT_SHA256 = '842eff15adfedd75735815cbe4040d17a645653e3b62511149afacbf8f923be9'
LEGACY_CONTRACT_SHA256 = '87fcbf04e88fd311784736cf77cddfdd82fc2608ed3de19882467e58707fb430'
CPU_ANALYSIS_SHA256 = '2af9ab8e1e8b76cbafd1622d75eedd825fe6f42f2dcdcdcfe115a7110965ab87'
CPU_MANIFEST_SHA256 = '2d6a12602162c3fbc0c7e373d6f36bf58355c3cf5cadab98aebf3730d2a17791'
CPU_RAW_SHA256 = '1fe8d81720679ad47fb6f6d7d47b3019dc2cbd7fccdc72ab6683f8a5d2ce705c'
INPUT_SHA256 = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
FAILED_CHECKPOINT_SHA256 = '66e149025bb0d777afe79a013797704c7878fdcad30140ac256aac305db48471'
CPU_REUSE_REVIEW_SHA256 = '1c9e21a845ae1e376611f18bb1ba1afe6daaf6e38e073fafd0597d8dadc480c2'
COMPILER_ANALYSIS_SHA256 = '78bd8f7372d92c623ff63e26646161e19aae7350bb53072a6b4149147396c306'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def contained(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe manifest path: ' + str(name))
    path = root / relative
    require(root in path.resolve().parents and not path.is_symlink(), 'Escaping or symlink path: ' + str(name))
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), 'Symlink parent: ' + str(name))
    return path


def verify_manifest(directory, manifest):
    require(isinstance(manifest.get('files'), dict) and manifest['files'], 'Empty manifest')
    for name, info in manifest['files'].items():
        path = contained(directory, name)
        require(path.is_file() and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                'Frozen file mismatch: ' + str(directory / name))


def validate_inputs(root):
    require(PROTOCOL_SHA256 != 'PENDING' and sha(root / 'protocol.json') == PROTOCOL_SHA256,
            'New NONE protocol identity differs')
    protocol = read(root / 'protocol.json')
    require(protocol['schema'] == 'rampq.gate05.corrected_qall_none_control_protocol/v1' and
            protocol['source_graph_sha256'] == GRAPH_SHA256 and protocol['source_spec_sha256'] == SOURCE_SPEC_SHA256 and
            protocol['source_manifest_sha256'] == SOURCE_MANIFEST_SHA256 and
            protocol['source_rewrite_audit_sha256'] == REWRITE_AUDIT_SHA256 and
            protocol['parent_failed_checkpoint_sha256'] == FAILED_CHECKPOINT_SHA256,
            'New protocol source identity differs')
    require(protocol['legacy_source_spec_precision_constraints'] == 'obey' and
            protocol['current_precision_constraints'] == 'none' and protocol['legacy_source_contract_not_rewritten'] is True,
            'New protocol does not distinguish old OBEY and new NONE')
    native = protocol['native_build']
    require(native['attempts'] == 1 and native['max_seconds'] == 7200 and native['workspace_mib'] == 1024 and
            native['avg_timing_iterations'] == 1 and native['profiling_verbosity'] == 'DETAILED' and
            native['INT8'] is True and native['FP16'] is True and native['TF32'] is False and
            native['OBEY_PRECISION_CONSTRAINTS'] is False and native['PREFER_PRECISION_CONSTRAINTS'] is False and
            native['layer_precision_or_output_type_setters_or_resets'] is False and
            native['required_clip_count'] == 88 and native['clip_alpha'] == -127.0 and native['clip_beta'] == 127.0 and
            native['all_external_bindings_format'] == 'LINEAR', 'New native protocol settings differ')
    saved_cpu = protocol['source_CPU_equivalence']
    require(saved_cpu['directory'] == 'cpu_equivalence' and saved_cpu['analysis_sha256'] == CPU_ANALYSIS_SHA256 and
            saved_cpu['manifest_sha256'] == CPU_MANIFEST_SHA256 and saved_cpu['raw_outputs_sha256'] == CPU_RAW_SHA256 and
            saved_cpu['review_sha256'] == CPU_REUSE_REVIEW_SHA256 and
            protocol['source_diagnostic_analysis_sha256'] == COMPILER_ANALYSIS_SHA256 and
            saved_cpu['new_CPU_runs'] == protocol['new_CPU_runs'] == 0 and
            protocol['capture']['automatic_inference'] is False and protocol['graph_weight_scale_zero_changes'] is False,
            'New CPU/inference/source-change protocol limits differ')
    require(protocol['protected_failures']['original_complete_model_guard'] == 'FAILED_UNCHANGED' and
            protocol['protected_failures']['original_tolerances_changed'] is False and
            protocol['protected_failures']['old_OBEY_failed_full_attempt_preserved'] is True and
            protocol['protected_failures']['overall_gate_status'] == 'INCOMPLETE', 'Protected protocol failures changed')
    control = read(root / 'control_spec.json')
    require(control['schema'] == 'rampq.gate05.corrected_qall_none_control/v1', 'Wrong outer control schema')
    require(control['protocol_sha256'] == PROTOCOL_SHA256, 'Wrong outer protocol reference')
    require(control['source_bundle_directory'] == 'source_bundle' and
            control['cpu_equivalence_directory'] == 'cpu_equivalence', 'Wrong preserved source directories')
    require(control['runtime_precision_constraints'] == 'none' and
            control['legacy_source_precision_constraints'] == 'obey', 'Legacy and new runtime policies conflated')
    require(control['maximum_native_builds'] == 1 and control['max_build_seconds'] == 7200 and
            control['new_CPU_execution_runs_this_stage'] == 0 and control['native_inference_executions'] == 0,
            'New build/CPU/inference budgets differ')
    require(control['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
            control['gate05_status'] == 'INCOMPLETE', 'Protected old failure or Gate state changed')
    for key, expected in (('graph_sha256', GRAPH_SHA256), ('source_spec_sha256', SOURCE_SPEC_SHA256),
                          ('source_manifest_sha256', SOURCE_MANIFEST_SHA256),
                          ('cpu_equivalence_analysis_sha256', CPU_ANALYSIS_SHA256),
                          ('cpu_equivalence_manifest_sha256', CPU_MANIFEST_SHA256)):
        require(control[key] == expected, 'Wrong outer source identity: ' + key)
    manifest = read(root / 'manifest.json')
    require(manifest['schema'] == 'rampq.gate05.corrected_qall_none_control_bundle/v1', 'Wrong outer manifest schema')
    require('manifest.json' not in manifest['files'], 'Outer manifest must not contain itself')
    required = {'control_spec.json', 'protocol.json', Path(__file__).name, HELPER_RELATIVE_PATH,
                'source_bundle/manifest.json', 'source_bundle/spec.json', 'source_bundle/model.onnx',
                'cpu_equivalence/manifest.json', 'cpu_equivalence/analysis.json',
                'cpu_equivalence/raw_outputs.npz', 'parent_failed_checkpoint.json',
                'saved_full_qall_reference_reuse_review.json', 'compiler_path_analysis.json'}
    require(required <= set(manifest['files']), 'Incomplete outer manifest')
    verify_manifest(root, manifest)
    require(sha(contained(root, Path(__file__).name)) == sha(Path(__file__).resolve()), 'Running unsealed builder')
    require(sha(contained(root, HELPER_RELATIVE_PATH)) == HELPER_SHA256, 'Original helper changed')
    require(sha(root / 'parent_failed_checkpoint.json') == FAILED_CHECKPOINT_SHA256,
            'Preserved failed full-model checkpoint changed')
    require(sha(root / 'saved_full_qall_reference_reuse_review.json') == CPU_REUSE_REVIEW_SHA256 and
            sha(root / 'compiler_path_analysis.json') == COMPILER_ANALYSIS_SHA256,
            'Frozen full CPU source reuse review or compiler-path analysis changed')
    source, cpu = root / 'source_bundle', root / 'cpu_equivalence'
    for name, expected in (('spec.json', SOURCE_SPEC_SHA256), ('manifest.json', SOURCE_MANIFEST_SHA256),
                           ('model.onnx', GRAPH_SHA256), ('semantic_rewrite_audit.json', REWRITE_AUDIT_SHA256),
                           ('reference_contract.json', LEGACY_CONTRACT_SHA256), ('preprocessed_input.npy', INPUT_SHA256)):
        require(sha(source / name) == expected, 'Legacy source identity changed: ' + name)
    legacy_manifest = read(source / 'manifest.json')
    verify_manifest(source, legacy_manifest)
    require({path.name for path in source.iterdir()} == set(legacy_manifest['files']) | {'manifest.json', 'SHA256SUMS'} and
            len(list(source.iterdir())) == 21, 'Preserved source bundle must retain exactly its original 21 files')
    require(all('source_bundle/' + path.name in manifest['files'] for path in source.iterdir()),
            'Outer manifest omits legacy source members')
    legacy = read(source / 'spec.json')
    require(legacy['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1' and
            legacy['configuration'] == 'q_all' and legacy['precisionConstraints'] == 'obey',
            'Legacy source spec changed or was relabeled NONE')
    require(legacy['quantized_conv_count'] == 88 and legacy['fp16_conv_count'] == 0 and
            len(legacy['requests']) == len({row['onnx_node'] for row in legacy['requests']}) == 88 and
            all(row['precision'] == 'INT8' for row in legacy['requests']), 'Wrong legacy source Conv requests')
    require(len(legacy['clip_names']) == len(set(legacy['clip_names'])) == 88 and
            legacy['activation_quantizer_topology'] == 'Q_INT8_CODE_CLIP_DQ' and
            legacy['requires_native_clip_importer_correction'] is True, 'Wrong source QDQ/Clip topology')
    require(legacy['graph_sha256'] == GRAPH_SHA256 and legacy['parent_graph_sha256'] == PARENT_GRAPH_SHA256 and
            legacy['semantic_rewrite_audit_sha256'] == legacy['common_preparation_sha256'] == REWRITE_AUDIT_SHA256,
            'Wrong source rewrite identity')
    require(legacy['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED', 'Legacy guard changed')
    require(legacy['expected_binding_contract'] == native['external_binding_contract'],
            'New protocol and legacy external binding contracts differ')
    for filename, key in (('graph_audit.json', 'graph_audit_sha256'), ('module_map.json', 'module_map_sha256'),
                           ('preflight.json', 'failed_preflight_sha256'), ('diagnostic_scope.json', 'diagnostic_scope_sha256')):
        require(sha(source / filename) == legacy[key], 'Legacy audit differs: ' + filename)
    contract = read(source / 'reference_contract.json')
    require(contract['native_build']['new_precisionConstraints'] == 'obey' and
            contract['native_build']['attempts'] == 1 and contract['native_build']['max_seconds'] == 7200,
            'Legacy contract changed; it must remain a separate old OBEY attempt')
    require(contract['not_a_gate_amendment'] is True and contract['gate05_status'] == 'INCOMPLETE' and
            contract['protected_failure']['original_complete_model_guard'] == 'FAILED_UNCHANGED' and
            contract['protected_failure']['tolerances_changed'] is False, 'Legacy failure boundary changed')
    rewrite = read(source / 'semantic_rewrite_audit.json')
    require(rewrite['status'] == 'STATIC_BYTE_REWRITE_CHECKS_PASSED_NO_INFERENCE' and
            rewrite['candidate_graph_sha256'] == GRAPH_SHA256 and rewrite['parent_graph_sha256'] == PARENT_GRAPH_SHA256 and
            rewrite['source_conv_count'] == 88 and rewrite['candidate_node_count'] == 795 and
            rewrite['candidate_initializer_count'] == 703 and rewrite['clip_names'] == legacy['clip_names'],
            'Legacy semantic rewrite proof identity differs')
    require(len(rewrite['rewrites']) == 88 and
            {row['conv_name'] for row in rewrite['rewrites']} == {row['onnx_node'] for row in legacy['requests']} and
            all(row['code_clip_bounds'] == [-127, 127] and row['dq_input_and_output_unchanged'] is True
                for row in rewrite['rewrites']), 'Source rewrite inventory differs')
    preflight, scope = read(source / 'preflight.json'), read(source / 'diagnostic_scope.json')
    require(preflight['status'] == legacy['original_local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED' and
            scope['original_tolerances_changed'] is False and scope['dataset_evaluation_run'] is False,
            'Legacy failed preflight or scope changed')
    for name, expected in (('analysis.json', CPU_ANALYSIS_SHA256), ('manifest.json', CPU_MANIFEST_SHA256),
                           ('raw_outputs.npz', CPU_RAW_SHA256)):
        require(sha(cpu / name) == expected, 'Saved full CPU equivalence identity changed: ' + name)
    cpu_manifest = read(cpu / 'manifest.json')
    verify_manifest(cpu, cpu_manifest)
    require({path.name for path in cpu.iterdir()} == set(cpu_manifest['files']) | {'manifest.json'} and
            len(list(cpu.iterdir())) == 6, 'Saved CPU equivalence must retain exactly its original six files')
    require(all('cpu_equivalence/' + name in manifest['files'] for name in cpu_manifest['files']),
            'Outer manifest omits saved CPU proof members')
    analysis = read(cpu / 'analysis.json')
    require(analysis['schema'] == 'rampq.gate05.corrected_qall_cpu_equivalence/v1' and
            analysis['status'] == 'FULL_FIXED_INPUT_CPU_REWRITE_BYTE_EQUIVALENCE_VERIFIED' and
            analysis['candidate_onnx_sha256'] == GRAPH_SHA256 and analysis['parent_onnx_sha256'] == PARENT_GRAPH_SHA256 and
            analysis['candidate_spec_sha256'] == SOURCE_SPEC_SHA256 and
            analysis['candidate_manifest_sha256'] == SOURCE_MANIFEST_SHA256 and
            analysis['input_npy_sha256'] == INPUT_SHA256 and analysis['raw_outputs_sha256'] == CPU_RAW_SHA256,
            'Saved CPU proof does not cover exact full source model/input')
    expected_outputs = {row['name']: row['shape'] for row in legacy['expected_binding_contract'] if not row['input']}
    require(set(analysis['source_and_candidate_outputs']) == set(expected_outputs) and len(expected_outputs) == 2,
            'Saved full CPU proof output set differs')
    for name, row in analysis['source_and_candidate_outputs'].items():
        require(row['shape'] == expected_outputs[name] and row['dtype'] == 'float32' and
                row['byte_identical'] is True and row['different_elements'] == 0 and row['max_abs'] == 0.0,
                'Saved full CPU output equality differs: ' + name)
    require(analysis['all_original_sources_hash_unchanged'] is True and
            analysis['original_complete_model_pytorch_ort_guard'] == 'FAILED_UNCHANGED' and
            analysis['new_tolerances_defined'] is False and analysis['native_execution_run'] is False and
            analysis['numerical_migration_pass_claim'] is False and analysis['dataset_evaluation_run'] is False,
            'Saved CPU equivalence scope changed')
    require((cpu / 'source_hashes_before.json').read_bytes() == (cpu / 'source_hashes_after.json').read_bytes(),
            'Saved CPU source preservation records differ')
    return control, legacy, protocol


def load_legacy_helper(root):
    path = contained(root, HELPER_RELATIVE_PATH)
    require(sha(path) == HELPER_SHA256, 'Legacy helper changed before import')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('_frozen_qall_none_legacy_helper', str(path))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    require(sha(Path(helper.__file__)) == HELPER_SHA256, 'Wrong imported helper identity')
    return helper


def finite(value):
    return value if math.isfinite(value) else repr(value)


def tensor_info(tensor):
    if tensor is None:
        return None
    return {'name': tensor.name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'is_shape_tensor': bool(tensor.is_shape_tensor)}


def layer_constraints(network):
    """Inspect all layers including Q/DQ/SHAPE; never read unset requests."""
    rows = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        row = {'index': index, 'name': layer.name, 'layer_type': str(layer.type),
               'num_inputs': layer.num_inputs, 'num_outputs': layer.num_outputs}
        try:
            is_set = bool(layer.precision_is_set)
            row['precision_is_set'] = {'supported': True, 'value': is_set}
            if is_set:
                try:
                    row['precision'] = str(layer.precision)
                except Exception as error:
                    row['precision_value_error'] = {'error_type': type(error).__name__, 'error': str(error)}
        except Exception as error:
            row['precision_is_set'] = {'supported': False, 'error_type': type(error).__name__, 'error': str(error)}
        row['output_types'] = []
        for output_index in range(layer.num_outputs):
            output = {'index': output_index}
            try:
                is_set = bool(layer.output_type_is_set(output_index))
                output['output_type_is_set'] = {'supported': True, 'value': is_set}
                if is_set:
                    try:
                        output['requested_output_type'] = str(layer.get_output_type(output_index))
                    except Exception as error:
                        output['requested_output_type_error'] = {'error_type': type(error).__name__, 'error': str(error)}
            except Exception as error:
                output['output_type_is_set'] = {'supported': False, 'error_type': type(error).__name__, 'error': str(error)}
            try:
                tensor = layer.get_output(output_index)
                output['tensor'] = tensor_info(tensor)
            except Exception as error:
                output['tensor_query_error'] = {'error_type': type(error).__name__, 'error': str(error)}
            row['output_types'].append(output)
        rows.append(row)
    return rows


def constraint_values(rows):
    # IActivationLayer casting changes the visible type wrapper, not requests.
    values = []
    for row in rows:
        selected = {key: row[key] for key in ('index', 'name', 'precision_is_set', 'precision', 'precision_value_error')
                    if key in row}
        selected['outputs'] = [{key: output[key] for key in ('index', 'output_type_is_set',
                               'requested_output_type', 'requested_output_type_error') if key in output}
                              for output in row['output_types']]
        values.append(selected)
    return values


def actual_flags(config, trt):
    flags = {}
    for name in dir(trt.BuilderFlag):
        if name.isupper():
            try:
                flags[name] = {'supported': True, 'enabled': bool(config.get_flag(getattr(trt.BuilderFlag, name)))}
            except Exception as error:
                flags[name] = {'supported': False, 'error_type': type(error).__name__, 'error': str(error)}
    return {'raw_config_flags': int(config.flags), 'get_flag': flags}


def save_actual_conv_rows(run, inspector_raw, legacy):
    inspector = json.loads(inspector_raw)
    rows = inspector.get('Layers', []) if isinstance(inspector, dict) else inspector
    require(isinstance(rows, list), 'Unexpected inspector JSON schema')
    convs = [row for row in rows if 'Convolution' in str(row.get('LayerType', ''))]
    write(run / 'conv_rows.json', convs)
    mappings = []
    for request in legacy['requests']:
        name = request['onnx_node']
        indices = [index for index, row in enumerate(convs) if name in row.get('Name', '')]
        mappings.append({'source_conv_name': name, 'source_request': request,
                         'actual_row_indices': indices, 'status': 'UNIQUE_SOURCE_MAPPING' if len(indices) == 1 else
                         ('MISSING_SOURCE_MAPPING' if not indices else 'AMBIGUOUS_SOURCE_MAPPING')})
    unique = [row['actual_row_indices'][0] for row in mappings if row['status'] == 'UNIQUE_SOURCE_MAPPING']
    complete = len(convs) == len(unique) == len(set(unique)) == 88
    write(run / 'conv_source_mapping.json', {'source_conv_count': 88, 'actual_conv_row_count': len(convs),
          'mapping': mappings, 'complete_one_to_one_mapping': complete,
          'status': 'COMPLETE_SOURCE_MAPPING_PENDING_PRECISION_AUDIT' if complete else
                    'MISSING_OR_AMBIGUOUS_MAPPING_PENDING_AUDIT',
          'mapping_method': 'Exact source node string contained in raw inspector Name; no guessed remapping',
          'actual_precision_pass_claim': False, 'seven_prior_depthwise_tactic_gaps_waived': False})
    write(run / 'conv_descriptors.json', {'actual_conv_row_count': len(convs),
          'descriptors': [{'name': row.get('Name'), 'inputs': row.get('Inputs'), 'weights': row.get('Weights'),
                           'outputs': row.get('Outputs'), 'activation': row.get('Activation'),
                           'tactic_name': row.get('TacticName'), 'tactic_value': row.get('TacticValue'),
                           'raw_row_index': index, 'actual_precision_requires_independent_review': True}
                          for index, row in enumerate(convs)],
          'output_int8_not_assumed': True, 'actual_precision_audit': 'PENDING'})
    return len(convs), complete


def terminate_as_interrupt(_signum, _frame):
    raise KeyboardInterrupt('External monitor requested termination')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check-input-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    control, legacy, _protocol = validate_inputs(root)
    helper = load_legacy_helper(root)
    identity = {'builder_file': Path(__file__).name, 'builder_sha256': sha(Path(__file__).resolve()),
                'helper_relative_path': HELPER_RELATIVE_PATH, 'helper_sha256': sha(Path(helper.__file__)),
                'protocol_sha256': PROTOCOL_SHA256, 'control_spec_sha256': sha(root / 'control_spec.json'),
                'outer_manifest_sha256': sha(root / 'manifest.json'),
                'legacy_source_spec_sha256': SOURCE_SPEC_SHA256, 'legacy_source_manifest_sha256': SOURCE_MANIFEST_SHA256}
    if args.check_input_only:
        print(json.dumps({'status': 'FROZEN_FULL_QALL_NONE_CONTROL_INPUTS_VERIFIED_NO_EXECUTION',
              'tool_identity': identity, 'graph_sha256': GRAPH_SHA256, 'legacy_source_precision_constraints': 'obey',
              'new_runtime_precision_constraints': 'none', 'source_conv_count': 88, 'clip_corrections_required': 88,
              'binding_count': len(legacy['expected_binding_contract']), 'saved_full_CPU_proof_reused': True,
              'new_CPU_execution_runs': 0, 'native_build_run': False, 'native_inference_run': False,
              'native_runtime_imported': False,
              'maximum_native_builds': 1, 'max_build_seconds': 7200,
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE'}, indent=2))
        return
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Native build requires Jetson aarch64 Linux')
    require(helper.query(['pgrep', '-x', 'trtexec'])['returncode'] == 1, 'Another trtexec process is running')
    results = root / 'results'
    require(not results.exists() or not any(results.iterdir()), 'This new control already has an attempt/results')
    marker = root / 'native_build_attempt.json'
    with marker.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'ONE_NATIVE_BUILD_ATTEMPT_CLAIMED', 'pid': os.getpid(),
                   'protocol_sha256': PROTOCOL_SHA256, 'runtime_precision_constraints': 'none',
                   'utc': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write('\n')
    run = results / datetime.now(timezone.utc).strftime('build_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, terminate_as_interrupt)
    signal.signal(signal.SIGINT, terminate_as_interrupt)
    started = time.monotonic()
    common = {'schema': 'rampq.gate05.corrected_qall_none_build_result/v1', 'configuration': 'q_all',
              'graph_sha256': GRAPH_SHA256, 'tool_identity': identity,
              'legacy_source_precision_constraints': 'obey', 'runtime_precision_constraints': 'none',
              'separate_new_control_not_legacy_attempt': True, 'new_CPU_execution_runs': 0,
              'saved_full_CPU_equivalence_analysis_sha256': CPU_ANALYSIS_SHA256,
              'single_image_inference': 'NOT_RUN', 'dataset_evaluation': 'NOT_RUN',
              'actual_precision_audit': 'PENDING', 'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
              'gate05_status': 'INCOMPLETE', 'numerical_pass_claim': False, 'accuracy_pass_claim': False,
              'performance_pass_claim': False, 'max_build_seconds': 7200, 'maximum_native_builds': 1,
              'run_dir': str(run), 'phase': 'STARTING', 'status': 'BUILD_RUNNING'}
    write(run / 'tool_hashes.json', identity)
    write(root / 'latest_build.json', common)
    print('RESULT_DIR:', run, flush=True)
    try:
        import tensorrt as trt
        require(trt.__version__.startswith('8.5.'), 'Expected TensorRT 8.5')
        write(run / 'environment.json', {'tensorrt': trt.__version__, 'python': platform.python_version(),
              'machine': platform.machine(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pid': os.getpid(), 'pgid': os.getpgrp(),
              'power_mode': helper.query(['/usr/sbin/nvpmodel', '-q']), 'memory': helper.query(['free', '-m']),
              'fan_service': helper.query(['systemctl', 'is-active', 'nvfancontrol.service']),
              'jtop_service': helper.query(['systemctl', 'is-active', 'jtop.service']),
              'jetson_linux': Path('/etc/nv_tegra_release').read_text()})
        common['phase'] = 'PARSING'
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        importer = trt.OnnxParser(network, logger)
        model = root / 'source_bundle' / 'model.onnx'
        parsed = importer.parse(model.read_bytes())
        errors = [str(importer.get_error(index)) for index in range(importer.num_errors)]
        write(run / 'parser_errors.json', {'parse_returned': bool(parsed), 'errors': errors})
        require(parsed, 'ONNX parser failed: ' + repr(errors))
        before = layer_constraints(network)
        write(run / 'parsed_constraints_before.json', before)
        layers = [network.get_layer(index) for index in range(network.num_layers)]
        require(len({layer.name for layer in layers}) == len(layers), 'Duplicate parsed layer names')
        by_name = {layer.name: layer for layer in layers}
        require({layer.name for layer in layers if layer.type == trt.LayerType.CONVOLUTION} ==
                {row['onnx_node'] for row in legacy['requests']}, 'Parsed full source Conv inventory differs')
        write(run / 'parsed_layers_before.json', [{'index': index, 'name': layer.name, 'type': str(layer.type),
              'inputs': [tensor_info(layer.get_input(i)) for i in range(layer.num_inputs)],
              'outputs': [tensor_info(layer.get_output(i)) for i in range(layer.num_outputs)]}
              for index, layer in enumerate(layers)])
        clip_before, corrections = [], []
        for name in legacy['clip_names']:
            clip = by_name[name]
            require(clip.type == trt.LayerType.ACTIVATION, 'Requested Clip did not parse as activation: ' + name)
            clip.__class__ = trt.IActivationLayer
            require(clip.type == trt.ActivationType.CLIP and clip.num_inputs == clip.num_outputs == 1,
                    'Wrong parsed Clip: ' + name)
            clip_before.append({'name': name, 'activation_type': str(clip.type),
                                'alpha': finite(clip.alpha), 'beta': finite(clip.beta),
                                'inputs': [tensor_info(clip.get_input(0))], 'outputs': [tensor_info(clip.get_output(0))]})
        write(run / 'parsed_clip_attributes_before_correction.json', {'count': len(clip_before), 'layers': clip_before})
        for row in clip_before:
            clip = by_name[row['name']]
            clip.alpha, clip.beta = -127.0, 127.0
            require(clip.alpha == -127.0 and clip.beta == 127.0, 'Clip correction failed')
            corrections.append({'name': row['name'], 'before': {'alpha': row['alpha'], 'beta': row['beta']},
                                'after': {'alpha': clip.alpha, 'beta': clip.beta}})
        write(run / 'parsed_clip_attributes_after_correction.json', {'count': len(corrections), 'layers': corrections})
        require(len(corrections) == 88, 'Wrong correction count')
        imported = []
        for is_input, count in ((True, network.num_inputs), (False, network.num_outputs)):
            for index in range(count):
                tensor = network.get_input(index) if is_input else network.get_output(index)
                require(tensor.dtype == trt.float32, 'External binding must be FLOAT')
                tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
                imported.append(dict(tensor_info(tensor), input=is_input, allowed_formats=int(tensor.allowed_formats)))
        require(len(imported) == 3 and network.num_inputs == 1 and network.num_outputs == 2,
                'Expected one input and two outputs')
        require(helper.contract_signature(imported) == helper.contract_signature(legacy['expected_binding_contract']),
                'External source binding contract differs')
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.clear_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.clear_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
        config.avg_timing_iterations = 1
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        flags = actual_flags(config, trt)
        for name, enabled in (('INT8', True), ('FP16', True), ('TF32', False),
                               ('OBEY_PRECISION_CONSTRAINTS', False), ('PREFER_PRECISION_CONSTRAINTS', False)):
            require(flags['get_flag'][name] == {'supported': True, 'enabled': enabled}, 'Wrong actual builder flag: ' + name)
        after = layer_constraints(network)
        write(run / 'parsed_constraints_after.json', after)
        write(run / 'parsed_layers_after.json', [{'index': index, 'name': layer.name, 'type': str(layer.type),
              'inputs': [tensor_info(layer.get_input(i)) for i in range(layer.num_inputs)],
              'outputs': [tensor_info(layer.get_output(i)) for i in range(layer.num_outputs)]}
              for index, layer in enumerate(layers)])
        require(constraint_values(before) == constraint_values(after), 'Parser layer precision/output requests changed')
        request = dict(common, schema='rampq.gate05.corrected_qall_none_build_request/v1',
              status='FROZEN_FULL_QALL_NONE_NATIVE_REQUEST', flags=flags,
              command=['TensorRT Python Builder.build_serialized_network'], source_conv_count=88,
              legacy_source_requests=legacy['requests'], bindings_in_network_order=imported,
              explicit_clip_importer_corrections=corrections, explicit_layer_precision_constraint_count=0,
              explicit_layer_output_type_constraint_count=0, layer_constraint_reset_count=0,
              parser_constraints_before_after_identical=True, native_int8_QDQ_defaults_preserved=True,
              workspace_bytes_actual=int(config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)),
              avg_timing_iterations_actual=int(config.avg_timing_iterations),
              profiling_verbosity_actual=str(config.profiling_verbosity),
              protected_legacy_contract_sha256=LEGACY_CONTRACT_SHA256,
              requirement='All 88 source Convs need independent actual format, named integer tactic and fusion audit; prior seven depthwise gaps are not waived.',
              limits='One new NONE build using unchanged corrected full graph and saved CPU rewrite proof. No automatic inference, accuracy, speed or Gate pass claim.')
        write(run / 'build_request.json', request)
        validate_inputs(root)
        common['phase'] = 'BUILDING'
        write(root / 'latest_build.json', common)
        build_started = time.monotonic()
        serialized = builder.build_serialized_network(network, config)
        common['build_seconds_not_benchmark'] = time.monotonic() - build_started
        write(run / 'parsed_constraints_after_build.json', layer_constraints(network))
        validate_inputs(root)
        require(serialized is not None, 'TensorRT returned no serialized engine')
        (run / 'model.engine').write_bytes(bytes(serialized))
        common.update(engine_sha256=sha(run / 'model.engine'), engine_bytes=(run / 'model.engine').stat().st_size,
                      phase='INSPECTING_ENGINE')
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine((run / 'model.engine').read_bytes())
        require(engine is not None, 'Cannot deserialize built engine')
        inspector_raw = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
        (run / 'engine_layers.json').write_text(inspector_raw, encoding='utf-8')
        bindings = [{'name': engine.get_binding_name(index), 'input': bool(engine.binding_is_input(index)),
                     'shape': list(engine.get_binding_shape(index)), 'dtype': str(engine.get_binding_dtype(index)),
                     'format': str(engine.get_binding_format(index))} for index in range(engine.num_bindings)]
        write(run / 'bindings.json', {'bindings': bindings})
        count, mapped = save_actual_conv_rows(run, inspector_raw, legacy)
        common.update(bindings=bindings, engine_layers=engine.num_layers, actual_conv_row_count=count,
                      source_mapping_complete=mapped)
        require(len(bindings) == 3 and helper.contract_signature(bindings) ==
                helper.contract_signature(legacy['expected_binding_contract']), 'Actual binding contract differs')
        require(all(row['format'] == 'TensorFormat.LINEAR' for row in bindings), 'Actual binding format differs')
        validate_inputs(root)
        common.update(status='BUILT_AND_DESERIALIZED_BUILD_ONLY', phase='COMPLETE',
                      elapsed_seconds=time.monotonic() - started, engine_execution_context_created=False,
                      native_inference_executions=0)
    except BaseException as error:
        common.update(status='BUILD_OR_CONTRACT_VALIDATION_FAILED', elapsed_seconds=time.monotonic() - started,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      native_inference_executions=0)
        write(run / 'build_failure.json', common)
        write(run / 'build_result.json', common)
        write(root / 'latest_build.json', common)
        raise
    write(run / 'build_result.json', common)
    write(root / 'latest_build.json', common)
    print('FULL_CORRECTED_QALL_NONE_BUILT_NO_GATE_OR_NUMERICAL_PASS_CLAIM', flush=True)


if __name__ == '__main__':
    main()
