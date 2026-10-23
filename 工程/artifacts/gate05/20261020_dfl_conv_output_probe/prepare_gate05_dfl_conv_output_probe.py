"""Prepare the exact six-node DFL QDQ/Conv output-boundary control.

The probability fixture is copied without regeneration. Exactly one new CPU
execution runs this six-node cut. Its C-order array bytes must equal the saved
three-dimensional DFL reference; NPY headers and output contracts differ.
Preparation and sealing are exclusive stages. A failed preparation is retained
and cannot be sealed or retried in its existing artifact directory.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate'
PARENT = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_probe'
PARENT_CHECKPOINT = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_native/checkpoint.json'
NEXT_PLAN = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_native/next_control_plan.json'
PROTOCOL = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_protocol/protocol.json'
OUTPUT = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_probe'
ARCHIVE_ROOT = 'gate05_dfl_conv_output_probe_20261020'
PROTOCOL_SHA = '8f6a55f640915b84d7300d1595ebb895092ccf0374e40a27b6aaf812f406e09b'
SOURCE_SHA = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
SOURCE_SPEC_SHA = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
SOURCE_MANIFEST_SHA = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
PARENT_SPEC_SHA = '45f284588d37e607478ce79e443d46a49b870ef0d65614c5ce0867130f50316b'
PARENT_MANIFEST_SHA = '6c456b5a08d34a9a3fb87d7fced5474f2c349d632076bfc0fb5eb8dd793ddde1'
PARENT_CHECKPOINT_SHA = '142eb219ccbeea141237cd1a89ecdb2c47c70b99d35159d9912ba8cea767c53f'
PARENT_DFL_SPEC_SHA = '934e075aa58dde1829b600d3d011f5ad51c4f1da1af78f7791944f492781b14d'
PARENT_DFL_MANIFEST_SHA = '3836e82ed53afb14fe16d3b9431864c0c4a45fd0d3418c7c06a4f6fcd46ed1d0'
PARENT_MODEL_SHA = '1f8662a55675152d57d248b16c2eebec08c4d54b5b6ebc4841f0c177d6ac2298'
PROBABILITY_SHA = '5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520'
SAVED_CPU_SHA = 'fd4bcdcee24cd53651fdd7a508423c802f3b432c759647204c7b748f22abd50d'
SAVED_EXPECTED_SHA = 'fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511'
PLAN_SHA = '14d7710f373fd56a8d68ec15219e234c96a9325938e25ad703696b879160dfc8'
INPUT_NAME = '/model.23/dfl/Softmax_output_0'
OUTPUT_NAME = '/model.23/dfl/conv/Conv_output_0'
CONVS = ['/model.23/dfl/conv/Conv']
CLIPS = ['rampq_model_23_dfl_conv_Clip']
CASE_ORDER = ['code_clip_none', 'code_clip_obey']
NODE_ORDER = ['rampq_model_23_dfl_conv_ActivationQ', 'rampq_model_23_dfl_conv_Clip',
              'rampq_model_23_dfl_conv_ActivationDQ', 'rampq_model_23_dfl_conv_WeightQ',
              'rampq_model_23_dfl_conv_WeightDQ', '/model.23/dfl/conv/Conv']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def message_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def check_bundle(folder, manifest_sha):
    require(sha(folder / 'manifest.json') == manifest_sha, 'Frozen source manifest changed')
    paths = [folder / 'manifest.json']
    for name, expected in read(folder / 'manifest.json')['files'].items():
        path = folder / name
        require(path.resolve().parent == folder.resolve() and path.is_file() and not path.is_symlink() and
                info(path) == expected, 'Frozen source differs: ' + str(path))
        paths.append(path)
    return paths


def source_hashes(paths):
    return {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): info(path)
            for path in sorted(set(paths))}


def validate_sources(record):
    for name, expected in record.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT) and path.is_file() and info(path) == expected,
                'Prepared historical source became stale: ' + name)


def source_value(model, name, shape):
    import onnx
    values = {value.name: value for value in list(model.graph.input) +
              list(model.graph.value_info) + list(model.graph.output)}
    require(name in values, 'Source tensor declaration missing: ' + name)
    value = copy.deepcopy(values[name])
    require(value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
            [dim.dim_value for dim in value.type.tensor_type.shape.dim] == shape,
            'Source tensor declaration differs: ' + name)
    return value


def extract(full, parent):
    import onnx
    from onnx import helper
    producers = {name: node for node in full.graph.node for name in node.output}
    initializers = {value.name: value for value in full.graph.initializer}
    needed, pending = set(), [OUTPUT_NAME]
    while pending:
        name = pending.pop()
        if name == INPUT_NAME or name in initializers:
            continue
        require(name in producers, 'Unexpected source closure dependency: ' + name)
        node = producers[name]
        if node.name not in needed:
            needed.add(node.name)
            pending.extend(value for value in node.input if value)
    selected = [copy.deepcopy(node) for node in full.graph.node if node.name in needed]
    constant_names = {name for node in selected for name in node.input if name in initializers}
    constants = [copy.deepcopy(value) for value in full.graph.initializer if value.name in constant_names]
    require([node.name for node in selected] == NODE_ORDER and len(constants) == 7 and
            Counter(node.op_type for node in selected) == Counter({
                'QuantizeLinear': 2, 'DequantizeLinear': 2, 'Clip': 1, 'Conv': 1}),
            'Six-node DFL closure inventory differs')
    parent_nodes = {node.name: node for node in parent.graph.node}
    parent_initializers = {value.name: value for value in parent.graph.initializer}
    require(len(parent.graph.node) == 8 and len(parent.graph.initializer) == 7 and
            set(parent_nodes) - needed == {'/model.23/dfl/Constant_1', '/model.23/dfl/Reshape_1'} and
            all(node.SerializeToString() == parent_nodes[node.name].SerializeToString() for node in selected) and
            all(value.SerializeToString() == parent_initializers[value.name].SerializeToString() for value in constants),
            'Cut must only remove the old trailing Constant and Reshape')
    result = copy.deepcopy(full)
    result.graph.CopyFrom(helper.make_graph(selected, 'exact_corrected_dfl_conv_output_cut',
        [source_value(full, INPUT_NAME, [1, 16, 4, 8400])],
        [source_value(full, OUTPUT_NAME, [1, 1, 4, 8400])], constants))
    onnx.checker.check_model(result, full_check=True)
    source_nodes = {node.name: node for node in full.graph.node}
    require(all(node.SerializeToString() == source_nodes[node.name].SerializeToString()
                for node in result.graph.node) and
            all(value.SerializeToString() == initializers[value.name].SerializeToString()
                for value in result.graph.initializer), 'Selected source serialized bytes changed')
    return result


def identity(model):
    return {'nodes': len(model.graph.node), 'initializers': len(model.graph.initializer),
        'conv_count': sum(node.op_type == 'Conv' for node in model.graph.node),
        'node_serialized_sha256': {node.name: message_sha(node) for node in model.graph.node},
        'initializer_serialized_sha256': {value.name: message_sha(value) for value in model.graph.initializer}}


def summary(array):
    import numpy as np
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'elements': int(array.size),
        'all_finite': bool(np.isfinite(array).all()), 'min': float(array.min()), 'max': float(array.max()),
        'raw_array_bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def execute_cpu_once(model_path, probabilities):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    require(session.get_providers() == ['CPUExecutionProvider'] and
            [value.name for value in session.get_inputs()] == [INPUT_NAME] and
            [value.name for value in session.get_outputs()] == [OUTPUT_NAME], 'CPU runtime provider or I/O differs')
    result = session.run([OUTPUT_NAME], {INPUT_NAME: probabilities})[0]
    require(result.dtype == np.float32 and result.shape == (1, 1, 4, 8400) and np.isfinite(result).all(),
            'CPU output shape, dtype or finite check failed')
    return result


def prepare(output, protocol_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import onnx
    import onnxruntime as ort
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Fresh destination required; no CPU retry')
    require(len(PROTOCOL_SHA) == len(PARENT_CHECKPOINT_SHA) == 64,
            'Root-frozen protocol and parent checkpoint pins must be supplied before preparation')
    require(sha(protocol_path) == PROTOCOL_SHA, 'Frozen protocol changed')
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rampq.gate05.dfl_conv_output_protocol/v1' and
            protocol['case_order'] == CASE_ORDER and protocol['batch_max_seconds'] == 900 and
            protocol['source_conv_names'] == CONVS and protocol['clip_names'] == CLIPS and
            protocol['input_name'] == INPUT_NAME and protocol['output_name'] == OUTPUT_NAME and
            protocol['input_shape'] == [1, 16, 4, 8400] and protocol['output_shape'] == [1, 1, 4, 8400] and
            protocol['graph_node_count'] == 6 and protocol['initializer_count'] == 7 and
            protocol['cpu_composition_check']['total_new_CPU_runs'] == 1 and
            protocol['cpu_composition_check']['cut_CPU_runs'] == 1 and
            protocol['cpu_composition_check']['new_Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0 and
            protocol['parent_softmax_cut_checkpoint_sha256'] == PARENT_CHECKPOINT_SHA,
            'Frozen protocol scope or I/O differs')
    expected_cases = [{'id': name, 'variant': 'code_clip',
        'precision_constraint_flag': 'none' if name.endswith('_none') else 'obey',
        'max_build_seconds': 300, 'iterations': 1, 'apply_native_clip_attribute_correction': True}
        for name in CASE_ORDER]
    require(protocol['cases'] == expected_cases, 'Frozen case definitions differ')
    paths = check_bundle(SOURCE, SOURCE_MANIFEST_SHA)
    paths.extend(check_bundle(PARENT, PARENT_MANIFEST_SHA))
    pins = {SOURCE / 'model.onnx': SOURCE_SHA, SOURCE / 'spec.json': SOURCE_SPEC_SHA,
        PARENT / 'spec.json': PARENT_SPEC_SHA,
        PARENT / 'parent_dfl_spec.json': PARENT_DFL_SPEC_SHA,
        PARENT / 'parent_dfl_manifest.json': PARENT_DFL_MANIFEST_SHA,
        PARENT / 'model_dfl_softmax_cut_code_clip.onnx': PARENT_MODEL_SHA,
        PARENT / 'dfl_softmax_probabilities.npy': PROBABILITY_SHA,
        PARENT / 'cpu_actual_code_clip.npy': SAVED_CPU_SHA,
        PARENT / 'cpu_expected_outputs.npz': SAVED_EXPECTED_SHA,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA, NEXT_PLAN: PLAN_SHA, protocol_path: PROTOCOL_SHA}
    for relative, expected in read(PARENT_CHECKPOINT)['protected_identities'].items():
        pins[ROOT / relative] = expected
    for path, expected in pins.items():
        require(sha(path) == expected, 'Pinned historical source differs: ' + str(path))
    paths.extend([*pins, Path(__file__).resolve()])
    before = source_hashes(paths)
    full = onnx.load(str(SOURCE / 'model.onnx'))
    parent_graph = onnx.load(str(PARENT / 'model_dfl_softmax_cut_code_clip.onnx'))
    model = extract(full, parent_graph)
    output.mkdir(parents=True, exist_ok=False)
    copies = {SOURCE / 'model.onnx': 'source_model.onnx', SOURCE / 'spec.json': 'parent_spec.json',
        SOURCE / 'manifest.json': 'parent_manifest.json',
        PARENT / 'spec.json': 'parent_softmax_cut_spec.json',
        PARENT / 'manifest.json': 'parent_softmax_cut_manifest.json',
        PARENT / 'parent_dfl_spec.json': 'parent_dfl_spec.json',
        PARENT / 'parent_dfl_manifest.json': 'parent_dfl_manifest.json',
        PARENT / 'fixture_lineage.json': 'parent_fixture_lineage.json',
        PARENT / 'dfl_softmax_probabilities.npy': 'dfl_softmax_probabilities.npy',
        PARENT / 'cpu_actual_code_clip.npy': 'saved_cpu_reference_3d.npy',
        PARENT / 'cpu_expected_outputs.npz': 'saved_expected_outputs_3d.npz',
        PARENT / 'failure_inventory.json': 'failure_inventory.json',
        PARENT / 'original_failed_preflight.json': 'original_failed_preflight.json',
        PARENT / 'retained_qall_failure_checkpoint.json': 'retained_qall_failure_checkpoint.json',
        PARENT_CHECKPOINT: 'parent_softmax_cut_checkpoint.json', NEXT_PLAN: 'parent_next_control_plan.json',
        protocol_path: 'protocol.json', Path(__file__).resolve(): Path(__file__).name}
    for path, name in copies.items():
        shutil.copy2(path, output / name)
    write(output / 'source_hashes_before.json', before)
    model_file = 'model_dfl_conv_output_code_clip.onnx'
    onnx.save(model, str(output / model_file))
    write(output / 'graph_identity_audit.json', {
        'schema': 'rampq.gate05.dfl_conv_output_graph_identity/v1',
        'status': 'EXACT_SIX_NODE_SOURCE_CLOSURE_VERIFIED', 'source_graph_sha256': SOURCE_SHA,
        'parent_eight_node_graph_sha256': PARENT_MODEL_SHA,
        'all_selected_nodes_initializers_and_io_declarations_exact_source_bytes': True,
        'removed_parent_nodes_only': ['/model.23/dfl/Constant_1', '/model.23/dfl/Reshape_1'],
        'new_external_output_contract': {'name': OUTPUT_NAME, 'shape': [1, 1, 4, 8400], 'dtype': 'FLOAT'},
        'model_file': model_file, 'model_sha256': sha(output / model_file), **identity(model)})
    write(output / 'cpu_execution_plan.json', {
        'schema': 'rampq.gate05.dfl_conv_output_cpu_plan/v1',
        'status': 'FROZEN_BEFORE_ONLY_CPU_EXECUTION', 'protocol_sha256': PROTOCOL_SHA,
        'new_six_node_CPU_runs': 1, 'maximum_total_new_CPU_runs': 1,
        'probability_fixture_generation_runs': 0, 'Softmax_72Conv_old_DFL_or_full_detector_runs': 0,
        'provider': 'CPUExecutionProvider', 'graph_optimization': 'ORT_DISABLE_ALL', 'threads': 4,
        'reference_policy': 'Pure reshape saved3DCPU array to4D; compare C-order array bytes, not whole NPY header bytes.',
        'on_failure': 'PRESERVE_NO_RETRY_RESELECTION_OR_NATIVE_START'})
    probabilities = np.load(output / 'dfl_softmax_probabilities.npy', allow_pickle=False)
    saved_reference = np.load(output / 'saved_cpu_reference_3d.npy', allow_pickle=False)
    require(sha(output / 'dfl_softmax_probabilities.npy') == PROBABILITY_SHA and
            probabilities.dtype == np.float32 and probabilities.shape == (1, 16, 4, 8400) and
            probabilities.flags.c_contiguous and np.isfinite(probabilities).all(), 'Reused probability fixture invalid')
    require(sha(output / 'saved_cpu_reference_3d.npy') == SAVED_CPU_SHA and
            saved_reference.dtype == np.float32 and saved_reference.shape == (1, 4, 8400) and
            saved_reference.flags.c_contiguous and np.isfinite(saved_reference).all(), 'Saved3D CPU reference invalid')
    with np.load(output / 'saved_expected_outputs_3d.npz', allow_pickle=False) as old:
        require(set(old.files) == {'original_output', 'code_clip_output'} and
                old['original_output'].tobytes(order='C') == saved_reference.tobytes(order='C') and
                old['code_clip_output'].tobytes(order='C') == saved_reference.tobytes(order='C'),
                'Historical CPU NPZ differs from saved raw bytes')
    reference_4d = saved_reference.reshape(1, 1, 4, 8400)
    require(reference_4d.tobytes(order='C') == saved_reference.tobytes(order='C'), 'Pure shape conversion changed bytes')
    np.save(output / 'reshaped_saved_cpu_reference_4d.npy', reference_4d)
    np.savez_compressed(output / 'cpu_expected_outputs.npz', code_clip_output=reference_4d)
    reference_lineage = {'schema': 'rampq.gate05.dfl_conv_output_reference_shape_lineage/v1',
        'status': 'PURE_ARRAY_RESHAPE_WITHOUT_INFERENCE',
        'old_3d_file': 'saved_cpu_reference_3d.npy', 'old_3d_file_sha256': SAVED_CPU_SHA,
        'old_3d_array': summary(saved_reference),
        'reshaped_4d_file': 'reshaped_saved_cpu_reference_4d.npy',
        'reshaped_4d_file_sha256': sha(output / 'reshaped_saved_cpu_reference_4d.npy'),
        'reshaped_4d_array': summary(reference_4d),
        'old_and_reshaped_C_order_array_bytes_identical': True,
        'new_expected_NPZ_file': 'cpu_expected_outputs.npz',
        'new_expected_NPZ_sha256': sha(output / 'cpu_expected_outputs.npz'),
        'expected_output_npz_keys': {OUTPUT_NAME: 'code_clip_output'},
        'NPY_whole_file_hash_equality_required': False,
        'new_CPU_executions_for_reshape': 0}
    write(output / 'reference_shape_lineage.json', reference_lineage)
    actual = execute_cpu_once(output / model_file, probabilities)
    np.save(output / 'cpu_actual_code_clip.npy', actual)
    equal = actual.tobytes(order='C') == saved_reference.tobytes(order='C')
    delta = actual.astype(np.float64) - reference_4d.astype(np.float64)
    diagnostic = {'schema': 'rampq.gate05.dfl_conv_output_cpu/v1',
        'status': 'ONE_CPU_CONV_OUTPUT_RAW_BYTE_EQUIVALENCE_PASSED' if equal else 'CPU_RAW_BYTE_EQUIVALENCE_FAILED',
        'new_CPU_executions': 1, 'new_probability_Softmax_72Conv_old_DFL_or_full_detector_runs': 0,
        'native_executions': 0, 'provider': 'CPUExecutionProvider', 'graph_optimization': 'ORT_DISABLE_ALL', 'threads': 4,
        'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
        'probability_fixture_sha256': PROBABILITY_SHA,
        'actual_output_file': 'cpu_actual_code_clip.npy', 'actual_output_file_sha256': sha(output / 'cpu_actual_code_clip.npy'),
        'actual_output': summary(actual), 'old_saved_reference_file_sha256': SAVED_CPU_SHA,
        'old_saved_reference_shape': [1, 4, 8400], 'new_output_shape': [1, 1, 4, 8400],
        'C_order_array_bytes_identical_saved_reference': equal, 'both_outputs_byte_exact': equal,
        'differing_elements': int(np.count_nonzero(delta)), 'max_absolute_difference': float(np.abs(delta).max()),
        'shape_header_difference_is_not_numeric_difference': True,
        'output_meaning': 'DFL_DISTANCES_WITH_SINGLETON_CHANNEL_NOT_DECODED_BOXES',
        'original_complete_model_guard': 'FAILED_UNCHANGED', 'new_tolerances_defined': False}
    write(output / 'cpu_diagnostic.json', diagnostic)
    after = source_hashes(paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'Historical source changed during the single CPU execution')
    require(equal, 'Six-node CPU array differs; preserve evidence and forbid retry or native start')
    parent_spec = read(PARENT / 'spec.json')
    fixture_lineage = {'schema': 'rampq.gate05.dfl_conv_output_reused_probability/v1',
        'status': 'EXACT_PROBABILITY_FILE_REUSED_WITHOUT_GENERATION',
        'fixture_file': 'dfl_softmax_probabilities.npy', 'fixture_sha256': PROBABILITY_SHA,
        'fixture': summary(probabilities), 'source_tensor_name': INPUT_NAME,
        'parent_softmax_cut_spec_sha256': PARENT_SPEC_SHA,
        'parent_fixture_lineage_file': 'parent_fixture_lineage.json',
        'parent_fixture_lineage_sha256': sha(output / 'parent_fixture_lineage.json'),
        'new_fixture_generation_runs': 0,
        'meaning': 'Prior exact source Softmax CPU probabilities; not native upstream data or a replacement Gate reference.'}
    write(output / 'fixture_lineage.json', fixture_lineage)
    variant = {'model_file': model_file, 'model_sha256': sha(output / model_file), 'nodes': 6, 'initializers': 7,
        'expected_output_npz_keys': {OUTPUT_NAME: 'code_clip_output'},
        'selected_node_serialized_sha256': identity(model)['node_serialized_sha256'],
        'selected_initializer_serialized_sha256': identity(model)['initializer_serialized_sha256']}
    spec = {'schema': 'rampq.gate05.dfl_conv_output_probe/v1',
        'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
        'source_configuration': 'q_all', 'source_graph_file': 'source_model.onnx', 'source_graph_sha256': SOURCE_SHA,
        'parent_spec_file': 'parent_spec.json', 'parent_spec_sha256': SOURCE_SPEC_SHA,
        'parent_manifest_file': 'parent_manifest.json', 'parent_manifest_sha256': SOURCE_MANIFEST_SHA,
        'parent_softmax_cut_spec_file': 'parent_softmax_cut_spec.json', 'parent_softmax_cut_spec_sha256': PARENT_SPEC_SHA,
        'parent_softmax_cut_manifest_file': 'parent_softmax_cut_manifest.json', 'parent_softmax_cut_manifest_sha256': PARENT_MANIFEST_SHA,
        'parent_softmax_cut_checkpoint_file': 'parent_softmax_cut_checkpoint.json', 'parent_softmax_cut_checkpoint_sha256': PARENT_CHECKPOINT_SHA,
        'parent_dfl_spec_file': 'parent_dfl_spec.json', 'parent_dfl_spec_sha256': PARENT_DFL_SPEC_SHA,
        'parent_dfl_manifest_file': 'parent_dfl_manifest.json', 'parent_dfl_manifest_sha256': PARENT_DFL_MANIFEST_SHA,
        'protocol_file': 'protocol.json', 'protocol_sha256': PROTOCOL_SHA,
        'input_file': 'dfl_softmax_probabilities.npy', 'input_file_sha256': PROBABILITY_SHA,
        'input_binding': {'name': INPUT_NAME, 'input': True, 'shape': [1, 16, 4, 8400], 'dtype': 'DataType.FLOAT'},
        'output_bindings_in_order': [{'name': OUTPUT_NAME, 'input': False, 'shape': [1, 1, 4, 8400],
            'dtype': 'DataType.FLOAT', 'role': 'dfl_conv_distances_with_singleton_channel_not_decoded_boxes', 'kind': 'primary'}],
        'conv_names': CONVS, 'source_conv_names': CONVS, 'clip_names': CLIPS,
        'source_activation_boundaries': copy.deepcopy(parent_spec['source_activation_boundaries']),
        'activation_scale': parent_spec['activation_scale'], 'variants': {'code_clip': variant},
        'cpu_expected_outputs_file': 'cpu_expected_outputs.npz',
        'cpu_expected_outputs_sha256': sha(output / 'cpu_expected_outputs.npz'),
        'cpu_diagnostic_file': 'cpu_diagnostic.json', 'cpu_diagnostic_sha256': sha(output / 'cpu_diagnostic.json'),
        'reference_shape_lineage_file': 'reference_shape_lineage.json',
        'reference_shape_lineage_sha256': sha(output / 'reference_shape_lineage.json'),
        'fixture_lineage_file': 'fixture_lineage.json', 'fixture_lineage_sha256': sha(output / 'fixture_lineage.json'),
        'graph_identity_audit_file': 'graph_identity_audit.json', 'graph_identity_audit_sha256': sha(output / 'graph_identity_audit.json'),
        'cases': copy.deepcopy(protocol['cases']), 'case_order': CASE_ORDER,
        'max_seconds': 900, 'max_build_seconds': 300, 'maximum_attempts_per_case': 1,
        'maximum_native_builds': 2, 'maximum_native_inferences': 4, 'executions_per_successful_case': 2,
        'native_importer_clip_attribute_correction': {'variant': 'code_clip', 'layer_names': CLIPS,
            'activation_alpha': -127.0, 'activation_beta': 127.0,
            'when': 'After parsing; preserve every parser layer precision/output request.'},
        'native_build_policy': copy.deepcopy(protocol['common_native_settings']),
        'new_CPU_execution_runs_this_stage': 1, 'new_probability_fixture_generation_runs': 0,
        'new_Softmax_72Conv_old_DFL_or_full_detector_runs': 0,
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
        'corrected_full_qall_failure': 'FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE_UNCHANGED',
        'new_tolerances_defined': False, 'original_tolerances_changed': False,
        'calibration_recomputed': False, 'weights_changed': False, 'dataset_evaluation_run': False,
        'complete_detector_executions': 0, 'native_build_or_execution_run_by_preparer': False,
        'limits': 'New original Conv output boundary changes native format-selection context; local outcomes do not uniquely attribute full failure to removed Reshape or prove full repair.'}
    write(output / 'spec_draft.json', spec)
    (output / 'README.md').write_text('''# DFL 六节点 Conv 输出边界对照

从同一修正q_all源图提取exact6节点7initializer：Activation Q/Clip/DQ、
Weight Q/DQ和源Conv，截止原Conv_output_0 FLOAT[1,1,4,8400]。只删除父
8节点图末Constant/Reshape，所选节点、常量与源I/O声明保持原字节。
概率fixture按原NPY字节复制，不生成概率、不跑Softmax/72Conv/旧DFL。

仅执行新6节点图CPU一次；输出C顺序float32原始数组字节必须与旧
FLOAT[1,4,8400] CPU距离完全相同。旧数组纯reshape为4D生成新expectedNPZ，
保存各自shape/hash；NPY头不同不算数值差异。CPU失败保留，不重选或重试，
不启动native。输出是带单通道维的DFL distance，不是解码检测框。

native code_clip none/obey各一次300秒，batch900秒，成功后重复执行两次。
只校正原Clip导入alpha/beta为-127/+127，不设置或重置解析器精度/输出请求。
实际Conv输出允许FP32，格式与tactic必须单独核验。新输出切口改变native
边界，结果变化不能把Reshape定为完整图唯一原因。原失败和Gate未通过保留。
''', encoding='utf-8')
    write(output / 'prepared_content_manifest.json', {
        'schema': 'rampq.gate05.dfl_conv_output_prepared_content/v1',
        'status': 'ONE_CPU_CONV_OUTPUT_CHECK_PASSED_NOT_SEALED',
        'files': {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}})
    print(json.dumps({'status': 'ONE_CPU_CONV_OUTPUT_CHECK_PASSED_NOT_SEALED', 'output': str(output),
        'model_sha256': variant['model_sha256'], 'probability_fixture_sha256': PROBABILITY_SHA,
        'new_expected_NPZ_sha256': spec['cpu_expected_outputs_sha256'],
        'new_CPU_actual_NPY_sha256': sha(output / 'cpu_actual_code_clip.npy'),
        'old_CPU_NPY_sha256': SAVED_CPU_SHA, 'new_CPU_runs': 1, 'new_probability_generation_runs': 0,
        'new_CPU_shape': [1, 1, 4, 8400], 'old_CPU_shape': [1, 4, 8400],
        'C_order_array_bytes_identical_saved_reference': True, 'source_bytes_unchanged': True}, indent=2))


def seal(output, native_paths):
    require(output.is_dir() and not output.with_suffix('.zip').exists() and
            not (output / 'manifest.json').exists() and not (output / 'spec.json').exists() and
            not (output / 'preparation_failure.json').exists(), 'Successful prepared unsealed destination required')
    prepared = read(output / 'prepared_content_manifest.json')
    require(prepared['status'] == 'ONE_CPU_CONV_OUTPUT_CHECK_PASSED_NOT_SEALED', 'Wrong preparation status')
    for name, expected in prepared['files'].items():
        path = output / name
        require(path.resolve().parent == output and info(path) == expected, 'Prepared file became stale: ' + name)
    sources = read(output / 'source_hashes_before.json')
    require(sources == read(output / 'source_hashes_after.json'), 'Before/after source records differ')
    validate_sources(sources)
    require(sha(output / 'protocol.json') == PROTOCOL_SHA and
            read(output / 'cpu_diagnostic.json')['C_order_array_bytes_identical_saved_reference'], 'Protocol/CPU evidence invalid')
    require(len(native_paths) == 2 and len({path.name for path in native_paths}) == 2 and
            all(path.parent == ROOT / 'scripts' and path.is_file() and not path.is_symlink() for path in native_paths),
            'Native tools must be distinct workspace scripts')
    before = source_hashes(native_paths)
    for path in native_paths:
        shutil.copy2(path, output / path.name)
    spec = read(output / 'spec_draft.json')
    spec['native_runner_file'], spec['native_builder_file'] = [path.name for path in native_paths]
    spec['native_runner_sha256'], spec['native_builder_sha256'] = [sha(path) for path in native_paths]
    spec['sealed_client_at'] = datetime.now().astimezone().isoformat()
    write(output / 'spec.json', spec)
    write(output / 'seal_native_source_hashes_before.json', before)
    after = source_hashes(native_paths)
    write(output / 'seal_native_source_hashes_after.json', after)
    require(before == after, 'Native source changed during sealing')
    validate_sources(sources)
    files = {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}
    require(len(files) == len(list(output.iterdir())), 'Unexpected subdirectory in bundle')
    write(output / 'manifest.json', {'schema': 'rampq.gate05.dfl_conv_output_probe_bundle/v1', 'files': files})
    sums = dict(files)
    sums['manifest.json'] = info(output / 'manifest.json')
    (output / 'SHA256SUMS').write_text(''.join(value['sha256'] + '  ' + name + '\n'
        for name, value in sorted(sums.items())), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, ARCHIVE_ROOT + '/' + path.name)
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'ZIP CRC check failed')
        for path in output.iterdir():
            require(hashlib.sha256(archive.read(ARCHIVE_ROOT + '/' + path.name)).hexdigest() == sha(path),
                    'ZIP payload bytes differ: ' + path.name)
    print(json.dumps({'status': 'DFL_CONV_OUTPUT_PROBE_SEALED', 'output': str(output),
        'archive': str(archive_path), 'archive_bytes': archive_path.stat().st_size,
        'archive_sha256': sha(archive_path), 'spec_sha256': sha(output / 'spec.json'),
        'manifest_sha256': sha(output / 'manifest.json'), 'protocol_sha256': PROTOCOL_SHA,
        'native_runner_sha256': spec['native_runner_sha256'], 'native_builder_sha256': spec['native_builder_sha256'],
        'source_bytes_unchanged': True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--prepare-only', action='store_true')
    stage.add_argument('--seal-only', action='store_true')
    parser.add_argument('--native-runner', type=Path, default=ROOT / 'scripts/jetson_dfl_conv_output_probe.py')
    parser.add_argument('--native-builder', type=Path, default=ROOT / 'scripts/jetson_build_dfl_conv_output_probe.py')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / 'artifacts/gate05').resolve()), 'Unsafe artifact destination')
    if args.prepare_only:
        existed_before = output.exists()
        try:
            prepare(output, args.protocol.resolve())
        except BaseException as exc:
            if not existed_before and output.is_dir() and not (output / 'preparation_failure.json').exists():
                write(output / 'preparation_failure.json', {'schema': 'rampq.gate05.dfl_conv_output_failure/v1',
                    'status': 'FAILED_NO_RETRY_RESELECTION_OR_NATIVE', 'error_type': type(exc).__name__,
                    'error': str(exc), 'traceback': traceback.format_exc(),
                    'original_complete_model_guard': 'FAILED_UNCHANGED'})
            raise
    else:
        seal(output, [args.native_runner.resolve(), args.native_builder.resolve()])


if __name__ == '__main__':
    main()
