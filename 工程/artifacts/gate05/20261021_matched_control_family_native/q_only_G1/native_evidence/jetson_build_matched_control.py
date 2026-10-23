"""Build one separately frozen matched-family NONE control without inference.

The original parent bundle and its legacy OBEY metadata are immutable lineage.
The member spec and separate native protocol authorize one new 7200-second
attempt. Desired FP16 masks are not injected as layer constraints and do not
prove actual FP16 adoption. --check-input-only imports no GPU runtime.
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


PROTOCOL_SHA256 = '12ce0b79c1e765f3293cc529f01023e6a58671e8ad688f0884fcf216931a3459'
FAMILY_PROTOCOL_SHA256 = '27af69e97a8c4220653294381c69fbf8fa412044bb09e94d7d888d2355cee98c'
INVENTORY_SHA256 = '3088b68899698fe4bcd2f7f61d368651e63a5da0bc1a72311b2bdc72f008e70b'
HELPER_RELATIVE_PATH = 'jetson_build_precision_candidate.py'
HELPER_SHA256 = 'd8431345da90482f0d5798e2d6524bf03591ecef653edfad5ef3e28a509eb19f'
HELPER_PINS = {
    'jetson_build_precision_candidate.py': HELPER_SHA256,
    'jetson_monitored_build.py': '71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1',
    'jetson_owned_group_guard.py': '4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4',
    'jetson_collect_isolated_numeric_probe.py': 'af9b7f823e4f889813fb88b6e8a529a6b1a6423b0c7ead0d5704513b517775eb',
}
INPUT_SHA256 = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
CONFIGURATIONS = ('fp16_control', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1')
PINS = {'fp16_control': {'graph_sha256': 'f449b83ec71b15c54b648e45446fb523cd0710194a60f3dde6e1f14f52a7bd73',
                  'cpu_expected_outputs_sha256': '899619b1adab736e361c399b5f7609a2b5b4031e9e70d5ba304d397bc8b1a3d2',
                  'cpu_equivalence_sha256': '70efadc807aa35d43b3ae68ba59b9509d8579d1e4118f53c0f1e0deea3845479',
                  'semantic_rewrite_audit_sha256': '0516e9ed7f85c01d7ae5abc0ec793b9eb08d2686c2eb7f4e0acb18c7300e3331',
                  'parent_graph_sha256': 'f449b83ec71b15c54b648e45446fb523cd0710194a60f3dde6e1f14f52a7bd73',
                  'parent_spec_sha256': '4b279517da8d847dca9fe60a592b914f7b68f24bd5c4b85b67881172a9f3cfb9',
                  'parent_manifest_sha256': 'f7cabe1ce2530a62c20398c8e79c80d98de4c640f2e6df697013f1751809c330',
                  'quantized_conv_count': 0},
 'q_only_G1': {'graph_sha256': '3f7d481fdf5268bea3d66c2176827dffaf3bd0628f5a2c412d48e3b6c53e9356',
               'cpu_expected_outputs_sha256': '8c227bc9dc9f948f26320a022f1ea6cfcbc86257637176da35a42d866e9195c1',
               'cpu_equivalence_sha256': 'acae7989e1c9a3f478a4eeeba8012416f22a5d079b86d59af7ef75030f39556a',
               'semantic_rewrite_audit_sha256': 'b75b98bf1107ed81574eb432aa6f266d4b8401d4d63246a6f8557a9b9f631ea3',
               'parent_graph_sha256': '946fe83ea185fde114096f9d7b64b5ec9d4bc69afc87476b9c7548cf97a04b91',
               'parent_spec_sha256': 'cd6bde58334770339b9259002ce3b2a00563bbba4340c3bc8bfa8e914aa4e67f',
               'parent_manifest_sha256': '32b4187f6f694e3efc843ec41c3132f48b4bdf34b84bd2399097b2998507b06a',
               'quantized_conv_count': 6},
 'r_only_G1': {'graph_sha256': '2aade19ee4d42501f7ba7a2956ea91ebcffec7934e1a21c58d02741a2d92926a',
               'cpu_expected_outputs_sha256': 'e8bb85a8fb94e649a5f935c9b96f537c8c235de7b34fb331bc4d769ea7192d8d',
               'cpu_equivalence_sha256': '79a6e98f8a8680b4a50b2135f3e20cd8e85a8832270c7a82192c75830f5b5b79',
               'semantic_rewrite_audit_sha256': 'ae40d9eba456c08d36742fe92bdeeb4a1be70dcec08e77fbb80ef37deb8fa250',
               'parent_graph_sha256': '4f94499d817be12e1286433276ae5a697e30ba8c66916f014a4c9b5e97076c06',
               'parent_spec_sha256': '4c84b8dfd32dacb9d900b1d758d822bd00ca61265bd9e956ec9d8110db709da9',
               'parent_manifest_sha256': '93ee4e2e9be6cc372d1d87edc3b4d12adc1ccd99bea65e0cdbadc19235a71fd9',
               'quantized_conv_count': 82},
 'r_only_G1_G2': {'graph_sha256': '193c913150fb22b01174b8cf95d4b8443f14937099942890c7488b4414431d79',
                  'cpu_expected_outputs_sha256': 'b6dd390b383d237919f3313a0dc086a54462fefc36354141497689a2a222cd69',
                  'cpu_equivalence_sha256': '7ff7d895dd36eef35046b04db929539728ce63576755429338f7afef9d0e9d73',
                  'semantic_rewrite_audit_sha256': '673ef4b33d4ef146d34636e1dc0b45c5119ec1905945b06aa60355dad22cf10d',
                  'parent_graph_sha256': '25f7afbd9c860ea86bc56ac6e803fb832231a10bec84eab95520532a0faa6225',
                  'parent_spec_sha256': 'db44dba09e252b881bdac5545694f333772a9422911e548eaebc08321bdf313e',
                  'parent_manifest_sha256': 'c7853f869b3dc1faec1ac33b077a03510547969ff210253a5cd856883ec48c2e',
                  'quantized_conv_count': 77},
 'r_only_G1_G5': {'graph_sha256': '6ffd47b3e22838791dd33e6124c4871640cb52e6687a47a1ded201e128c28cec',
                  'cpu_expected_outputs_sha256': '69e0c558cd13c4b267188a2e806c68cf2ad982ce82976ace7605556ca745d8b3',
                  'cpu_equivalence_sha256': '5d2b1aa7794e6c4bc3c1ad5126a27e701d911804d372d71da8aa97783b025fbc',
                  'semantic_rewrite_audit_sha256': 'f3181ec5ecd360fb25014abc7d8563ddbdc7cd240bac7e0d1446f86998759383',
                  'parent_graph_sha256': '9e8c7221dc2100e490a3947f3058a9225fee3496cb7c78d0144e2865c7d0dbcb',
                  'parent_spec_sha256': 'e013d721396c513333d136e95f72544f3d6198b86df44988d2363e46c3008053',
                  'parent_manifest_sha256': '7195473e36ad03af3ac51f85e8ea292edc225c57ba47d17a5e6c58093aa4fbfc',
                  'quantized_conv_count': 74}}


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


def load_legacy_helper(root):
    path = contained(root, HELPER_RELATIVE_PATH)
    require(sha(path) == HELPER_SHA256, 'Legacy helper changed before import')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('_frozen_matched_control_legacy_helper', str(path))
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


def save_actual_conv_rows(run, inspector_raw, spec):
    inspector = json.loads(inspector_raw)
    rows = inspector.get('Layers', []) if isinstance(inspector, dict) else inspector
    require(isinstance(rows, list), 'Unexpected inspector JSON schema')
    convs = [row for row in rows if 'Convolution' in str(row.get('LayerType', ''))]
    write(run / 'conv_rows.json', convs)
    mappings = []
    for request in spec['requests']:
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


def validate_inputs(root):
    require(PROTOCOL_SHA256 != 'PENDING' and sha(root / 'native_protocol.json') == PROTOCOL_SHA256,
            'Separate native-family protocol identity differs')
    native = read(root / 'native_protocol.json')
    require(native['schema'] == 'rampq.gate05.matched_control_family_native_protocol/v1' and
            native['family_preparation_protocol_sha256'] == FAMILY_PROTOCOL_SHA256 and
            native['configuration_order'] == list(CONFIGURATIONS) and native['native_inference_executions'] == 0 and
            native['new_CPU_model_executions'] == 0 and native['gate05_status'] == 'INCOMPLETE',
            'Wrong separate native-family scope')
    policy = native['builder_policy']
    require(policy['precision_constraints'] == 'none' and policy['FP16'] is True and policy['TF32'] is False and
            policy['OBEY_PRECISION_CONSTRAINTS'] is False and policy['PREFER_PRECISION_CONSTRAINTS'] is False and
            policy['workspace_mib'] == 1024 and policy['avg_timing_iterations'] == 1 and
            policy['profiling_verbosity'] == 'DETAILED' and policy['all_external_bindings'] == 'FLOAT/LINEAR' and
            policy['layer_precision_or_output_type_setters_or_resets'] == 0 and
            policy['weights_scales_calibration_or_group_membership_changed'] is False,
            'Wrong frozen native builder policy')
    require(native['execution_policy']['serial_only'] is True and native['execution_policy']['no_concurrent_GPU_work'] is True and
            native['execution_policy']['no_automatic_retry'] is True, 'Native serial/no-retry supervision policy changed')
    spec = read(root / 'spec.json')
    require(spec['schema'] == 'rampq.gate05.matched_control_family_candidate/v1' and
            spec['configuration'] in CONFIGURATIONS, 'Wrong matched-family member')
    name = spec['configuration']
    pins = PINS[name]
    budget = native['native_builds'][name]
    require(budget['attempts'] == spec['maximum_native_builds'] == 1 and
            budget['max_seconds'] == spec['max_build_seconds'] == 7200 and spec['native_build_authorized'] is True,
            'Member needs exactly one separately authorized 7200-second build')
    require(budget['quantized_conv_count'] == pins['quantized_conv_count'] and
            budget['desired_fp16_conv_count'] == 88 - pins['quantized_conv_count'], 'Native budget precision partition differs')
    require(spec['native_protocol_file'] == 'native_protocol.json' and spec['native_protocol_sha256'] == PROTOCOL_SHA256 and
            spec['family_protocol_file'] == 'family_protocol.json' and spec['family_protocol_sha256'] == FAMILY_PROTOCOL_SHA256 and
            spec['source_inventory_file'] == 'source_inventory.json' and spec['source_inventory_sha256'] == INVENTORY_SHA256,
            'Wrong member protocol/inventory references')
    require(sha(root / 'family_protocol.json') == FAMILY_PROTOCOL_SHA256 and
            sha(root / 'source_inventory.json') == INVENTORY_SHA256, 'Preparation protocol/inventory changed')
    manifest = read(root / 'manifest.json')
    require(manifest['schema'] == 'rampq.gate05.matched_control_family_bundle/v1' and
            manifest['configuration'] == name and 'manifest.json' not in manifest['files'], 'Wrong member manifest')
    required = {'spec.json', 'spec_draft.json', 'native_protocol.json', 'family_protocol.json', 'source_inventory.json',
                'model.onnx', 'preprocessed_input.npy', 'cpu_equivalence.json', 'cpu_expected_outputs.npz',
                'cpu_candidate_outputs.npz', 'semantic_rewrite_audit.json', 'prepared_content_manifest.json',
                'parent_bundle/spec.json', 'parent_bundle/model.onnx', 'parent_bundle/manifest.json',
                HELPER_RELATIVE_PATH, Path(__file__).name}
    require(required <= set(manifest['files']), 'Member manifest omits required frozen inputs')
    verify_manifest(root, manifest)
    for filename, digest in HELPER_PINS.items():
        require(filename in manifest['files'] and sha(root / filename) == digest, 'Frozen supervision/helper changed: ' + filename)
    require(spec['native_builder_file'] == Path(__file__).name and
            sha(root / spec['native_builder_file']) == spec['native_builder_sha256'] == sha(Path(__file__).resolve()),
            'Running unsealed or different matched-family builder')
    require(sha(root / HELPER_RELATIVE_PATH) == HELPER_SHA256, 'Original build helper changed')
    for key in ('graph_sha256', 'cpu_expected_outputs_sha256', 'cpu_equivalence_sha256',
                'semantic_rewrite_audit_sha256', 'parent_graph_sha256', 'parent_spec_sha256',
                'parent_manifest_sha256', 'quantized_conv_count'):
        require(spec[key] == pins[key], 'Frozen member identity differs: ' + name + '/' + key)
    require(spec['graph_file'] == 'model.onnx' and sha(root / 'model.onnx') == pins['graph_sha256'] and
            spec['input_file'] == 'preprocessed_input.npy' and sha(root / spec['input_file']) == spec['input_file_sha256'] == INPUT_SHA256,
            'Frozen member model or fixed input differs')
    count = pins['quantized_conv_count']
    require(spec['desired_restored_fp16_conv_count'] == 88 - count and
            spec['requests_are_desired_precision_masks_not_manual_layer_constraints'] is True and
            spec['restored_FP16_adoption_requires_independent_actual_audit'] is True,
            'Desired FP16 precision mask scope changed')
    requests = spec['requests']
    require(len(requests) == len({row['onnx_node'] for row in requests}) == 88 and
            sum(row['precision'] == 'INT8' for row in requests) == count and
            sum(row['precision'] == 'FP16' for row in requests) == 88 - count,
            'Wrong desired source Conv precision partition')
    require(spec['legacy_source_precision_constraints'] == 'obey' and spec['runtime_precision_constraints'] == 'none' and
            spec['explicit_layer_precision_or_output_type_setters_or_resets'] is False and spec['native_inference_executions'] == 0,
            'Legacy OBEY and new NONE policies conflated')
    require(spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and spec['gate05_status'] == 'INCOMPLETE' and
            spec['diagnostic_only'] is True and spec['new_tolerances_defined'] is False and
            spec['weights_or_calibration_changed'] is False and spec['dataset_evaluation_run'] is False and
            spec['native_build_or_inference_run_by_preparer'] is False,
            'Protected member provenance/acceptance scope changed')
    policy = spec['native_build_policy']
    require(policy['precision_constraints'] == 'none' and policy['FP16'] is True and policy['INT8'] is (count > 0) and
            policy['TF32'] is False and policy['PREFER_PRECISION_CONSTRAINTS'] is False and
            policy['OBEY_PRECISION_CONSTRAINTS'] is False and policy['workspace_mib'] == 1024 and
            policy['avg_timing_iterations'] == 1 and policy['profiling_verbosity'] == 'DETAILED' and
            policy['all_external_bindings'] == 'FLOAT/LINEAR' and
            policy['layer_precision_or_output_type_setters_or_resets'] is False,
            'Wrong native member settings')
    inventory = read(root / 'source_inventory.json')
    entry = inventory['configurations'][name]
    require(entry['requests'] == requests and entry['quantized_conv_count'] == count and
            entry['desired_restored_fp16_conv_count'] == 88 - count and
            entry['expected_binding_contract'] == spec['expected_binding_contract'] and
            entry['source_convs'] == 88 and entry['source_precision_constraints'] == 'obey',
            'Frozen source inventory differs from member')
    require(spec['common_source_provenance'] == inventory['common_source_provenance'], 'Common source provenance changed')
    parent = root / spec['parent_bundle_directory']
    require(parent == root / 'parent_bundle', 'Wrong immutable parent bundle directory')
    for filename, key, inventory_key in (('model.onnx', 'parent_graph_sha256', 'model'),
                                        ('spec.json', 'parent_spec_sha256', 'spec'),
                                        ('manifest.json', 'parent_manifest_sha256', 'manifest')):
        require(sha(parent / filename) == spec[key] == entry[inventory_key]['sha256'], 'Parent identity changed: ' + filename)
    parent_manifest = read(parent / 'manifest.json')
    verify_manifest(parent, parent_manifest)
    require({path.name for path in parent.iterdir()} == set(parent_manifest['files']) | {'manifest.json', 'SHA256SUMS'} and
            len(list(parent.iterdir())) == entry['source_directory_file_count'] == 11,
            'Parent bundle must retain its original exact 11-file set')
    require(all('parent_bundle/' + path.name in manifest['files'] for path in parent.iterdir()), 'Outer manifest omits original parent')
    parent_spec = read(parent / 'spec.json')
    require(parent_spec['precisionConstraints'] == 'obey' and parent_spec['configuration'] == name and
            parent_spec['requests'] == requests and parent_spec['expected_binding_contract'] == spec['expected_binding_contract'],
            'Parent metadata was relabeled or request/output inventory changed')
    require(spec['semantic_rewrite_audit_file'] == 'semantic_rewrite_audit.json' and
            sha(root / spec['semantic_rewrite_audit_file']) == pins['semantic_rewrite_audit_sha256'], 'Own rewrite audit changed')
    rewrite = read(root / 'semantic_rewrite_audit.json')
    require(rewrite['schema'] == 'rampq.gate05.matched_control_family_rewrite/v1' and
            rewrite['status'] == 'EXACT_ALLOWED_REWRITE_VERIFIED' and rewrite['configuration'] == name and
            rewrite['family_protocol_sha256'] == FAMILY_PROTOCOL_SHA256 and
            rewrite['candidate_graph_sha256'] == pins['graph_sha256'] and
            rewrite['parent_graph_sha256'] == pins['parent_graph_sha256'] and rewrite['quantized_convs'] == count and
            rewrite['restored_FP16_desired_only'] == 88 - count and rewrite['changed_node_count'] == 2 * count and
            rewrite['changed_bound_initializer_count'] == 2 * count and
            rewrite['all_Conv_weight_bias_scale_zero_and_unapproved_node_initializer_bytes_preserved'] is True and
            rewrite['all_model_shell_and_IO_bytes_preserved'] is True and rewrite['no_calibration_recompute'] is True,
            'Own exact allowed rewrite proof differs')
    require(spec['clip_names'] == rewrite['clip_names'] and len(spec['clip_names']) == len(set(spec['clip_names'])) == count and
            spec['requires_native_clip_importer_correction'] is (count > 0) and len(rewrite['rewrites']) == count and
            {row['conv_name'] for row in rewrite['rewrites']} == {row['onnx_node'] for row in requests if row['precision'] == 'INT8'} and
            all(row['code_clip_bounds'] == [-127, 127] and row['dq_input_and_output_unchanged'] is True for row in rewrite['rewrites']),
            'Named quantized Clip rewrite inventory differs')
    require(spec['activation_quantizer_topology'] == ('Q_INT8_CODE_CLIP_DQ' if count else 'NO_QDQ_FLOAT_CONTROL'),
            'Wrong member topology')
    if count == 0:
        require(pins['graph_sha256'] == pins['parent_graph_sha256'] and not spec['clip_names'],
                'FP16 control graph must be exact source bytes without Clip correction')
    require(spec['cpu_equivalence_file'] == 'cpu_equivalence.json' and
            sha(root / spec['cpu_equivalence_file']) == pins['cpu_equivalence_sha256'] and
            spec['cpu_expected_outputs_file'] == 'cpu_expected_outputs.npz' and
            sha(root / spec['cpu_expected_outputs_file']) == pins['cpu_expected_outputs_sha256'] == sha(root / 'cpu_candidate_outputs.npz'),
            'Own saved CPU equivalence/reference identity changed')
    cpu = read(root / 'cpu_equivalence.json')
    expected_runs = 2 if count else 1
    require(cpu['schema'] == 'rampq.gate05.matched_control_family_CPU_equivalence/v1' and
            cpu['status'] == 'MEMBER_CPU_EQUIVALENCE_AND_REFERENCE_PASSED' and cpu['configuration'] == name and
            cpu['new_CPU_model_runs'] == spec['new_CPU_execution_runs_this_member'] == expected_runs and
            cpu['source_CPU_runs'] == (1 if count else 0) and cpu['candidate_CPU_runs'] == 1 and
            spec['family_total_new_CPU_runs'] == 9 and cpu['provider'] == 'CPUExecutionProvider' and
            cpu['graph_optimization'] == 'ORT_DISABLE_ALL' and cpu['threads'] == 4 and
            cpu['original_complete_model_guard'] == 'FAILED_UNCHANGED' and cpu['new_tolerances_defined'] is False and
            cpu['native_or_dataset_run'] is False, 'Own saved CPU proof scope differs')
    outputs = {row['name']: row for row in spec['expected_binding_contract'] if not row['input']}
    require(spec['expected_output_npz_keys'] == {'output0': 'main_output',
            '/model.10/m/m.0/attn/pe/conv/Conv_output_0': 'pe_output'} and
            len(cpu['outputs']) == len(outputs) == 2 and {row['name'] for row in cpu['outputs']} == set(outputs),
            'Own CPU reference output map differs')
    for row in cpu['outputs']:
        require(row['contract_valid'] is True and row['byte_exact_source'] is True and
                row['expected_npz_key'] == spec['expected_output_npz_keys'][row['name']] and
                row['candidate']['shape'] == outputs[row['name']]['shape'] and row['candidate']['dtype'] == 'float32' and
                row['candidate']['all_finite'] is True, 'Own CPU output is not proved finite/source-equivalent')
    for before, after in (('source_hashes_before.json', 'source_hashes_after.json'),
                          ('seal_native_source_hashes_before.json', 'seal_native_source_hashes_after.json')):
        require((root / before).read_bytes() == (root / after).read_bytes(), 'Source preservation records differ')
    prepared = read(root / 'prepared_content_manifest.json')
    require(prepared['status'] == 'MEMBER_CPU_EQUIVALENCE_PASSED_NOT_SEALED' and prepared['configuration'] == name,
            'Own preparation checkpoint is not successful')
    verify_manifest(root, prepared)
    draft = read(root / 'spec_draft.json')
    allowed = {'native_builder_file', 'native_builder_sha256', 'sealed_client_at', 'native_protocol_file',
               'native_protocol_sha256', 'native_build_authorized', 'maximum_native_builds', 'max_build_seconds'}
    require(all(spec[key] == value for key, value in draft.items() if key not in allowed) and
            set(spec) - set(draft) <= allowed and draft['native_build_authorized'] is False and
            draft['maximum_native_builds'] == 0 and draft['max_build_seconds'] is None,
            'Sealing changed preparation metadata beyond separately authorized native fields')
    require(len(spec['expected_binding_contract']) == 3 and
            sum(row['input'] for row in spec['expected_binding_contract']) == 1 and
            all(row['dtype'] == 'DataType.FLOAT' and all(value > 0 for value in row['shape'])
                for row in spec['expected_binding_contract']), 'Wrong FLOAT external binding contract')
    return spec, parent_spec, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--check-input-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    spec, parent_spec, _native = validate_inputs(root)
    helper = load_legacy_helper(root)
    count = spec['quantized_conv_count']
    identity = {'builder_file': Path(__file__).name, 'builder_sha256': sha(Path(__file__).resolve()),
                'helper_relative_path': HELPER_RELATIVE_PATH, 'helper_sha256': sha(Path(helper.__file__)),
                'native_protocol_sha256': PROTOCOL_SHA256, 'family_protocol_sha256': FAMILY_PROTOCOL_SHA256,
                'spec_sha256': sha(root / 'spec.json'), 'manifest_sha256': sha(root / 'manifest.json'),
                'parent_spec_sha256': sha(root / 'parent_bundle/spec.json'),
                'parent_manifest_sha256': sha(root / 'parent_bundle/manifest.json'),
                'graph_sha256': sha(root / 'model.onnx'),
                'cpu_equivalence_sha256': sha(root / 'cpu_equivalence.json'),
                'cpu_expected_outputs_sha256': sha(root / 'cpu_expected_outputs.npz')}
    if args.check_input_only:
        print(json.dumps({'status': 'FROZEN_MATCHED_CONTROL_INPUTS_VERIFIED_NO_EXECUTION', 'native_runtime_imported': False,
              'configuration': spec['configuration'], 'tool_identity': identity,
              'quantized_conv_count': count, 'desired_restored_fp16_conv_count': 88 - count,
              'legacy_parent_precision_constraints': 'obey', 'runtime_precision_constraints': 'none',
              'source_conv_count': 88, 'clip_corrections_required': count, 'binding_count': 3,
              'saved_own_CPU_reference_reused': True, 'new_CPU_execution_runs': 0,
              'native_build_run': False, 'native_inference_run': False, 'maximum_native_builds': 1, 'max_build_seconds': 7200,
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE'}, indent=2))
        return
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Native build requires Jetson aarch64 Linux')
    require(helper.query(['pgrep', '-x', 'trtexec'])['returncode'] == 1, 'Another trtexec process is running')
    results = root / 'results'
    require(not results.exists() or not any(results.iterdir()), 'This member already has an attempt/results')
    with (root / 'native_build_attempt.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'ONE_MEMBER_NATIVE_ATTEMPT_CLAIMED', 'configuration': spec['configuration'],
                   'pid': os.getpid(), 'native_protocol_sha256': PROTOCOL_SHA256,
                   'runtime_precision_constraints': 'none', 'UTC': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write('\n')
    run = results / datetime.now(timezone.utc).strftime('build_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, terminate_as_interrupt)
    signal.signal(signal.SIGINT, terminate_as_interrupt)
    started = time.monotonic()
    common = {'schema': 'rampq.gate05.matched_control_build_result/v1', 'configuration': spec['configuration'],
              'graph_sha256': spec['graph_sha256'], 'tool_identity': identity,
              'legacy_parent_precision_constraints': 'obey', 'legacy_source_precision_constraints': 'obey',
              'runtime_precision_constraints': 'none', 'separate_new_control_not_legacy_attempt': True,
              'quantized_conv_count': count, 'desired_restored_fp16_conv_count': 88 - count,
              'preparation_CPU_model_runs': spec['new_CPU_execution_runs_this_member'], 'new_CPU_execution_runs': 0,
              'single_image_inference': 'NOT_RUN', 'dataset_evaluation': 'NOT_RUN', 'actual_precision_audit': 'PENDING',
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
              'numerical_pass_claim': False, 'accuracy_pass_claim': False, 'performance_pass_claim': False,
              'desired_FP16_mask_is_not_adoption_proof': True, 'max_build_seconds': 7200, 'maximum_native_builds': 1,
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
        model = root / spec['graph_file']
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
                {row['onnx_node'] for row in spec['requests']}, 'Parsed full source Conv inventory differs')
        write(run / 'parsed_layers_before.json', [{'index': index, 'name': layer.name, 'type': str(layer.type),
              'inputs': [tensor_info(layer.get_input(i)) for i in range(layer.num_inputs)],
              'outputs': [tensor_info(layer.get_output(i)) for i in range(layer.num_outputs)]}
              for index, layer in enumerate(layers)])
        clip_before, corrections = [], []
        for name in spec['clip_names']:
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
        require(len(corrections) == count, 'Wrong correction count')
        imported = []
        for is_input, binding_count in ((True, network.num_inputs), (False, network.num_outputs)):
            for index in range(binding_count):
                tensor = network.get_input(index) if is_input else network.get_output(index)
                require(tensor.dtype == trt.float32, 'External binding must be FLOAT')
                tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
                imported.append(dict(tensor_info(tensor), input=is_input, allowed_formats=int(tensor.allowed_formats)))
        require(len(imported) == 3 and network.num_inputs == 1 and network.num_outputs == 2,
                'Expected one input and two outputs')
        require(helper.contract_signature(imported) == helper.contract_signature(spec['expected_binding_contract']),
                'External source binding contract differs')
        config = builder.create_builder_config()
        if count:
            config.set_flag(trt.BuilderFlag.INT8)
        else:
            config.clear_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.clear_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.clear_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
        config.avg_timing_iterations = 1
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        flags = actual_flags(config, trt)
        for name, enabled in (('INT8', count > 0), ('FP16', True), ('TF32', False),
                               ('OBEY_PRECISION_CONSTRAINTS', False), ('PREFER_PRECISION_CONSTRAINTS', False)):
            require(flags['get_flag'][name] == {'supported': True, 'enabled': enabled}, 'Wrong actual builder flag: ' + name)
        after = layer_constraints(network)
        write(run / 'parsed_constraints_after.json', after)
        write(run / 'parsed_layers_after.json', [{'index': index, 'name': layer.name, 'type': str(layer.type),
              'inputs': [tensor_info(layer.get_input(i)) for i in range(layer.num_inputs)],
              'outputs': [tensor_info(layer.get_output(i)) for i in range(layer.num_outputs)]}
              for index, layer in enumerate(layers)])
        require(constraint_values(before) == constraint_values(after), 'Parser layer precision/output requests changed')
        request = dict(common, schema='rampq.gate05.matched_control_build_request/v1',
              status='FROZEN_MATCHED_CONTROL_NATIVE_REQUEST', flags=flags,
              command=['TensorRT Python Builder.build_serialized_network'], source_conv_count=88,
              desired_source_requests=spec['requests'], bindings_in_network_order=imported,
              explicit_clip_importer_corrections=corrections, explicit_layer_precision_constraint_count=0,
              explicit_layer_output_type_constraint_count=0, layer_constraint_reset_count=0,
              parser_constraints_before_after_identical=True, native_int8_QDQ_defaults_preserved=True,
              workspace_bytes_actual=int(config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)),
              avg_timing_iterations_actual=int(config.avg_timing_iterations),
              profiling_verbosity_actual=str(config.profiling_verbosity),
              legacy_parent_spec_sha256=sha(root / 'parent_bundle/spec.json'),
              requirement='All 88 desired source requests need independent actual format and fusion audit. Quantized inputs/weights need integer evidence; restored requests need actual FP16 adoption. Global flags and masks are not precision proof.',
              limits='One independently frozen matched NONE build using its own graph and saved CPU reference. No manual precision/output requests, automatic inference, accuracy, speed or Gate pass claim.')
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
        actual_conv_count, mapped = save_actual_conv_rows(run, inspector_raw, spec)
        common.update(bindings=bindings, engine_layers=engine.num_layers, actual_conv_row_count=actual_conv_count,
                      source_mapping_complete=mapped)
        require(len(bindings) == 3 and helper.contract_signature(bindings) ==
                helper.contract_signature(spec['expected_binding_contract']), 'Actual binding contract differs')
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
    print('MATCHED_CONTROL_BUILT_NO_PRECISION_NUMERICAL_OR_GATE_PASS_CLAIM', flush=True)


if __name__ == '__main__':
    main()
