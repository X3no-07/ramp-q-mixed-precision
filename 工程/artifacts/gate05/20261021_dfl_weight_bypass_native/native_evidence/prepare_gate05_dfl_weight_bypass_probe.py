"""Prepare one frozen DFL weight-QDQ bypass with its own single-run CPU reference.

Only Conv.input[1] is rewired to the unchanged source FP32 weight initializer.
The activation Q/Clip/DQ path and four-dimensional I/O are preserved. Removing
weight quantization changes numerical semantics; the old reference is descriptive.
Preparation and sealing are exclusive, and a failed preparation is not retried.
This script never builds or executes a native engine.
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
PARENT = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_probe'
PARENT_CHECKPOINT = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_native/checkpoint.json'
NEXT_PLAN = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_native/next_control_plan.json'
PROTOCOL = ROOT / 'artifacts/gate05/20261021_dfl_weight_bypass_protocol/protocol.json'
OUTPUT = ROOT / 'artifacts/gate05/20261021_dfl_weight_bypass_probe'
ARCHIVE_ROOT = 'gate05_dfl_weight_bypass_probe_20261021'
PROTOCOL_SHA = '9b2ea26ff12ecd5a4b5a699b09c511a3ea6c2187125d037117bb85f57b7b1170'
SOURCE_SHA = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
SOURCE_SPEC_SHA = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
SOURCE_MANIFEST_SHA = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
PARENT_SPEC_SHA = '9e9d31b55e4358d39a08a7869aab77582f094f1d6152599c727196e2b77c3897'
PARENT_MANIFEST_SHA = '104137a621b3ea5c65a019555a1d92fc8c876fe0bf9293b5ab7cfe2e6fe71f77'
PARENT_CHECKPOINT_SHA = 'c5e4a11a37d32a6de9df22c42463de0a6a58ad4b92c97b64f4c318f71dd9947e'
PARENT_MODEL_SHA = '805abb78ed4e4d5abc8041b2916dadc83370a14ef1776fcc0a03f163c16bcf81'
PROBABILITY_SHA = '5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520'
PARENT_EXPECTED_SHA = '711fb41fa18bef712bd069c88a552e1bc918d1d1034b6f6d32d802cbd769d0c6'
PLAN_SHA = '124615758af8e5a45a074effbbc335a3fbba4de2e37352252a2d58fcba8d5a61'
INPUT_NAME = '/model.23/dfl/Softmax_output_0'
OUTPUT_NAME = '/model.23/dfl/conv/Conv_output_0'
WEIGHT_NAME = 'model.23.dfl.conv.weight'
CONVS = ['/model.23/dfl/conv/Conv']
CLIPS = ['rampq_model_23_dfl_conv_Clip']
REMOVED = ['rampq_model_23_dfl_conv_WeightQ', 'rampq_model_23_dfl_conv_WeightDQ']
NODE_ORDER = ['rampq_model_23_dfl_conv_ActivationQ', 'rampq_model_23_dfl_conv_Clip',
              'rampq_model_23_dfl_conv_ActivationDQ', '/model.23/dfl/conv/Conv']
CASE_ORDER = ['weight_bypass_none', 'weight_bypass_obey']


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


def extract(full, parent):
    import onnx
    from onnx import helper, numpy_helper
    import numpy as np
    source_nodes = {node.name: node for node in full.graph.node}
    source_initializers = {value.name: value for value in full.graph.initializer}
    parent_nodes = {node.name: node for node in parent.graph.node}
    parent_initializers = {value.name: value for value in parent.graph.initializer}
    require(len(parent.graph.node) == 6 and len(parent.graph.initializer) == 7 and
            set(parent_nodes) == set(NODE_ORDER + REMOVED), 'Wrong frozen six-node parent inventory')
    require(all(node.SerializeToString() == source_nodes[node.name].SerializeToString() for node in parent.graph.node) and
            all(value.SerializeToString() == source_initializers[value.name].SerializeToString()
                for value in parent.graph.initializer), 'Parent nodes/initializers differ from failed full source')
    selected = [copy.deepcopy(node) for node in parent.graph.node if node.name not in REMOVED]
    require([node.name for node in selected] == NODE_ORDER, 'Unexpected retained node order')
    conv = selected[-1]
    require(list(conv.input) == ['rampq_model_23_dfl_conv_a_dq', 'rampq_model_23_dfl_conv_w_dq'],
            'Source Conv input contract differs')
    conv.input[1] = WEIGHT_NAME
    used = {name for node in selected for name in node.input if name in source_initializers}
    constants = [copy.deepcopy(value) for value in parent.graph.initializer if value.name in used]
    require(len(constants) == 5 and used == {WEIGHT_NAME, 'rampq_model_23_dfl_conv_a_scale',
            'rampq_model_23_dfl_conv_a_zero', 'rampq_model_23_dfl_conv_clip_min',
            'rampq_model_23_dfl_conv_clip_max'} and Counter(node.op_type for node in selected) ==
            Counter({'QuantizeLinear': 1, 'DequantizeLinear': 1, 'Clip': 1, 'Conv': 1}),
            'Weight bypass must contain four nodes and five used initializers')
    restored = copy.deepcopy(conv)
    restored.input[1] = 'rampq_model_23_dfl_conv_w_dq'
    require(restored.SerializeToString() == source_nodes[conv.name].SerializeToString() and
            all(node.SerializeToString() == source_nodes[node.name].SerializeToString() for node in selected[:-1]) and
            all(value.SerializeToString() == source_initializers[value.name].SerializeToString() for value in constants),
            'Mutation exceeds the single declared Conv.input[1] rewire')
    weight = numpy_helper.to_array(source_initializers[WEIGHT_NAME])
    require(weight.dtype == np.float32 and weight.shape == (1, 16, 1, 1) and
            np.array_equal(weight.reshape(-1), np.arange(16, dtype=np.float32)), 'Unexpected original DFL weight values')
    require(len(parent.graph.input) == len(parent.graph.output) == 1 and parent.graph.input[0].name == INPUT_NAME and
            parent.graph.output[0].name == OUTPUT_NAME, 'Parent I/O differs')
    result = copy.deepcopy(parent)
    result.graph.CopyFrom(helper.make_graph(selected, 'corrected_dfl_weight_qdq_bypass',
        [copy.deepcopy(parent.graph.input[0])], [copy.deepcopy(parent.graph.output[0])], constants))
    onnx.checker.check_model(result, full_check=True)
    require(result.graph.input[0].SerializeToString() == parent.graph.input[0].SerializeToString() and
            result.graph.output[0].SerializeToString() == parent.graph.output[0].SerializeToString(), 'I/O bytes changed')
    audit = {'schema': 'rampq.gate05.dfl_weight_bypass_graph_identity/v1',
        'status': 'ONLY_DECLARED_WEIGHT_QDQ_BYPASS_VERIFIED', 'source_graph_sha256': SOURCE_SHA,
        'parent_six_node_graph_sha256': PARENT_MODEL_SHA, 'removed_nodes': REMOVED,
        'declared_node_mutation': {'node': conv.name, 'field': 'input[1]',
            'old': 'rampq_model_23_dfl_conv_w_dq', 'new': WEIGHT_NAME},
        'all_retained_activation_nodes_and_used_initializers_exact_source_bytes': True,
        'Conv_restored_input1_serialized_bytes_exact_source': True, 'input_output_declarations_exact_parent_bytes': True,
        'original_weight_file_values_changed': False, 'weight_shape': [1, 16, 1, 1],
        'weight_values': list(range(16)), 'unused_weight_scale_zero_source_lineage_only': {
            name: message_sha(source_initializers[name]) for name in
            ('rampq_model_23_dfl_conv_w_scale', 'rampq_model_23_dfl_conv_w_zero')},
        'semantic_change': 'Weight quantization removed; not an equivalent rewrite or full-model repair.',
        'actual_native_weight_precision_not_inferred_from_initializer_dtype': True}
    return result, audit


def identity(model):
    return {'nodes': len(model.graph.node), 'initializers': len(model.graph.initializer),
        'conv_count': sum(node.op_type == 'Conv' for node in model.graph.node),
        'node_serialized_sha256': {node.name: message_sha(node) for node in model.graph.node},
        'initializer_serialized_sha256': {value.name: message_sha(value) for value in model.graph.initializer}}


def summary(array):
    import numpy as np
    selected = array[np.isfinite(array)]
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'elements': int(array.size),
        'all_finite': bool(np.isfinite(array).all()), 'min': float(selected.min()) if selected.size else None,
        'max': float(selected.max()) if selected.size else None,
        'raw_array_bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def execute_cpu_once(model_path, probabilities):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    require(session.get_providers() == ['CPUExecutionProvider'] and
            [value.name for value in session.get_inputs()] == [INPUT_NAME] and
            [value.name for value in session.get_outputs()] == [OUTPUT_NAME], 'CPU runtime provider or I/O differs')
    return session.run([OUTPUT_NAME], {INPUT_NAME: probabilities})[0]


def prepare(output, protocol_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import onnx
    import onnxruntime as ort
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Fresh destination required; no CPU retry')
    require(sha(protocol_path) == PROTOCOL_SHA, 'Frozen protocol changed')
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rampq.gate05.dfl_weight_bypass_protocol/v1' and
            protocol['case_order'] == CASE_ORDER and protocol['batch_max_seconds'] == 900 and
            protocol['source_conv_names'] == CONVS and protocol['clip_names'] == CLIPS and
            protocol['input_name'] == INPUT_NAME and protocol['output_name'] == OUTPUT_NAME and
            protocol['input_shape'] == [1, 16, 4, 8400] and protocol['output_shape'] == [1, 1, 4, 8400] and
            protocol['graph_node_count'] == 4 and protocol['initializer_count'] == 5 and
            protocol['cpu_composition_check']['total_new_CPU_runs'] == 1 and
            protocol['cpu_composition_check']['candidate_CPU_runs'] == 1 and
            protocol['cpu_composition_check']['new_Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0 and
            protocol['parent_conv_output_checkpoint_sha256'] == PARENT_CHECKPOINT_SHA and
            protocol['parent_conv_output_spec_sha256'] == PARENT_SPEC_SHA and
            protocol['parent_conv_output_manifest_sha256'] == PARENT_MANIFEST_SHA and
            protocol['parent_conv_output_graph_sha256'] == PARENT_MODEL_SHA and
            protocol['removed_node_names'] == REMOVED, 'Frozen protocol scope or I/O differs')
    expected_cases = [{'id': name, 'variant': 'weight_bypass',
        'precision_constraint_flag': 'none' if name.endswith('_none') else 'obey',
        'max_build_seconds': 300, 'iterations': 1, 'apply_native_clip_attribute_correction': True}
        for name in CASE_ORDER]
    require(protocol['cases'] == expected_cases, 'Frozen case definitions differ')
    paths = check_bundle(SOURCE, SOURCE_MANIFEST_SHA)
    paths.extend(check_bundle(PARENT, PARENT_MANIFEST_SHA))
    pins = {SOURCE / 'model.onnx': SOURCE_SHA, SOURCE / 'spec.json': SOURCE_SPEC_SHA,
        PARENT / 'spec.json': PARENT_SPEC_SHA, PARENT / 'model_dfl_conv_output_code_clip.onnx': PARENT_MODEL_SHA,
        PARENT / 'dfl_softmax_probabilities.npy': PROBABILITY_SHA,
        PARENT / 'cpu_expected_outputs.npz': PARENT_EXPECTED_SHA,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA, NEXT_PLAN: PLAN_SHA, protocol_path: PROTOCOL_SHA}
    for relative, expected in read(PARENT_CHECKPOINT)['protected_identities'].items():
        pins[ROOT / relative] = expected
    for path, expected in pins.items():
        require(sha(path) == expected, 'Pinned historical source differs: ' + str(path))
    paths.extend([*pins, Path(__file__).resolve()])
    before = source_hashes(paths)
    model, graph_audit = extract(onnx.load(str(SOURCE / 'model.onnx')),
                                onnx.load(str(PARENT / 'model_dfl_conv_output_code_clip.onnx')))
    output.mkdir(parents=True, exist_ok=False)
    copies = {SOURCE / 'model.onnx': 'source_model.onnx', SOURCE / 'spec.json': 'parent_spec.json',
        SOURCE / 'manifest.json': 'parent_manifest.json', PARENT / 'spec.json': 'parent_conv_output_spec.json',
        PARENT / 'manifest.json': 'parent_conv_output_manifest.json',
        PARENT / 'model_dfl_conv_output_code_clip.onnx': 'parent_conv_output_model.onnx',
        PARENT / 'fixture_lineage.json': 'parent_fixture_lineage.json',
        PARENT / 'dfl_softmax_probabilities.npy': 'dfl_softmax_probabilities.npy',
        PARENT / 'cpu_actual_code_clip.npy': 'saved_parent_cpu_reference_4d.npy',
        PARENT / 'cpu_expected_outputs.npz': 'saved_parent_expected_outputs.npz',
        PARENT / 'failure_inventory.json': 'failure_inventory.json',
        PARENT / 'original_failed_preflight.json': 'original_failed_preflight.json',
        PARENT / 'retained_qall_failure_checkpoint.json': 'retained_qall_failure_checkpoint.json',
        PARENT_CHECKPOINT: 'parent_conv_output_checkpoint.json', NEXT_PLAN: 'parent_next_control_plan.json',
        protocol_path: 'protocol.json', Path(__file__).resolve(): Path(__file__).name}
    for path, name in copies.items():
        shutil.copy2(path, output / name)
    write(output / 'source_hashes_before.json', before)
    model_file = 'model_dfl_weight_bypass.onnx'
    onnx.save(model, str(output / model_file))
    graph_audit.update({'model_file': model_file, 'model_sha256': sha(output / model_file), **identity(model)})
    write(output / 'graph_identity_audit.json', graph_audit)
    write(output / 'cpu_execution_plan.json', {'schema': 'rampq.gate05.dfl_weight_bypass_cpu_plan/v1',
        'status': 'FROZEN_BEFORE_ONLY_CPU_EXECUTION', 'protocol_sha256': PROTOCOL_SHA,
        'new_candidate_CPU_runs': 1, 'maximum_total_new_CPU_runs': 1, 'probability_fixture_generation_runs': 0,
        'Softmax_72Conv_old_DFL_or_full_detector_runs': 0, 'provider': 'CPUExecutionProvider',
        'graph_optimization': 'ORT_DISABLE_ALL', 'threads': 4,
        'reference_policy': 'Own finite candidate4D output only; old6node reference descriptive, no equality requirement.',
        'expected_key': 'weight_bypass_output', 'on_failure': 'PRESERVE_NO_RETRY_RESELECTION_OR_NATIVE_START'})
    probabilities = np.load(output / 'dfl_softmax_probabilities.npy', allow_pickle=False)
    saved_reference = np.load(output / 'saved_parent_cpu_reference_4d.npy', allow_pickle=False)
    require(sha(output / 'dfl_softmax_probabilities.npy') == PROBABILITY_SHA and
            probabilities.dtype == np.float32 and probabilities.shape == (1, 16, 4, 8400) and
            probabilities.flags.c_contiguous and np.isfinite(probabilities).all() and
            float(probabilities.min()) >= 0.0 and float(probabilities.max()) <= 1.0, 'Reused probability fixture invalid')
    require(saved_reference.dtype == np.float32 and saved_reference.shape == (1, 1, 4, 8400) and
            saved_reference.flags.c_contiguous and np.isfinite(saved_reference).all(), 'Saved parent4D CPU reference invalid')
    with np.load(output / 'saved_parent_expected_outputs.npz', allow_pickle=False) as old:
        require(set(old.files) == {'code_clip_output'} and old['code_clip_output'].dtype == np.float32 and
                old['code_clip_output'].shape == saved_reference.shape and
                old['code_clip_output'].tobytes(order='C') == saved_reference.tobytes(order='C'),
                'Historical CPU NPZ differs from saved raw bytes')
    actual = execute_cpu_once(output / model_file, probabilities)
    np.save(output / 'cpu_actual_weight_bypass.npy', actual)
    require(actual.dtype == np.float32 and actual.shape == (1, 1, 4, 8400) and np.isfinite(actual).all(),
            'Own new candidate CPU output must be finite FLOAT4D')
    np.savez_compressed(output / 'cpu_expected_outputs.npz', weight_bypass_output=actual)
    delta = actual.astype(np.float64) - saved_reference.astype(np.float64)
    diagnostic = {'schema': 'rampq.gate05.dfl_weight_bypass_cpu/v1', 'status': 'OWN_ONE_RUN_CPU_REFERENCE_ESTABLISHED',
        'new_CPU_executions': 1, 'new_probability_Softmax_72Conv_old_DFL_or_full_detector_runs': 0,
        'native_executions': 0, 'provider': 'CPUExecutionProvider', 'graph_optimization': 'ORT_DISABLE_ALL', 'threads': 4,
        'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
        'probability_fixture_sha256': PROBABILITY_SHA, 'actual_output_file': 'cpu_actual_weight_bypass.npy',
        'actual_output_file_sha256': sha(output / 'cpu_actual_weight_bypass.npy'), 'actual_output': summary(actual),
        'new_expected_NPZ_sha256': sha(output / 'cpu_expected_outputs.npz'), 'expected_key': 'weight_bypass_output',
        'saved_parent_reference_file_sha256': sha(output / 'saved_parent_cpu_reference_4d.npy'),
        'saved_parent_reference': summary(saved_reference), 'old_reference_adopted_as_candidate_expected': False,
        'old_reference_comparison_descriptive_only': {'differing_elements': int(np.count_nonzero(delta)),
            'max_absolute_difference': float(np.abs(delta).max()), 'mean_absolute_difference': float(np.abs(delta).mean()),
            'C_order_array_bytes_identical': actual.tobytes(order='C') == saved_reference.tobytes(order='C'),
            'equality_required': False, 'new_tolerance_or_numerical_acceptance_rule': False},
        'output_meaning': 'DFL_DISTANCES_WITH_SINGLETON_CHANNEL_NOT_DECODED_BOXES',
        'semantic_change': protocol['semantic_change'], 'original_complete_model_guard': 'FAILED_UNCHANGED',
        'new_tolerances_defined': False}
    write(output / 'cpu_diagnostic.json', diagnostic)
    after = source_hashes(paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'Historical source changed during the single CPU execution')
    parent_spec = read(PARENT / 'spec.json')
    fixture_lineage = {'schema': 'rampq.gate05.dfl_weight_bypass_reused_probability/v1',
        'status': 'EXACT_PROBABILITY_FILE_REUSED_WITHOUT_GENERATION',
        'fixture_file': 'dfl_softmax_probabilities.npy', 'fixture_sha256': PROBABILITY_SHA,
        'fixture': summary(probabilities), 'source_tensor_name': INPUT_NAME,
        'parent_conv_output_spec_sha256': PARENT_SPEC_SHA, 'parent_fixture_lineage_file': 'parent_fixture_lineage.json',
        'parent_fixture_lineage_sha256': sha(output / 'parent_fixture_lineage.json'), 'new_fixture_generation_runs': 0,
        'meaning': 'Prior exact source Softmax CPU probabilities, not native upstream or replacement Gate reference.'}
    write(output / 'fixture_lineage.json', fixture_lineage)
    variant = {'model_file': model_file, 'model_sha256': sha(output / model_file), 'nodes': 4, 'initializers': 5,
        'expected_output_npz_keys': {OUTPUT_NAME: 'weight_bypass_output'},
        'selected_node_serialized_sha256': identity(model)['node_serialized_sha256'],
        'selected_initializer_serialized_sha256': identity(model)['initializer_serialized_sha256']}
    spec = {'schema': 'rampq.gate05.dfl_weight_bypass_probe/v1',
        'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
        'source_configuration': 'q_all', 'source_graph_file': 'source_model.onnx', 'source_graph_sha256': SOURCE_SHA,
        'parent_spec_file': 'parent_spec.json', 'parent_spec_sha256': SOURCE_SPEC_SHA,
        'parent_manifest_file': 'parent_manifest.json', 'parent_manifest_sha256': SOURCE_MANIFEST_SHA,
        'parent_conv_output_spec_file': 'parent_conv_output_spec.json', 'parent_conv_output_spec_sha256': PARENT_SPEC_SHA,
        'parent_conv_output_manifest_file': 'parent_conv_output_manifest.json', 'parent_conv_output_manifest_sha256': PARENT_MANIFEST_SHA,
        'parent_conv_output_checkpoint_file': 'parent_conv_output_checkpoint.json', 'parent_conv_output_checkpoint_sha256': PARENT_CHECKPOINT_SHA,
        'parent_conv_output_model_file': 'parent_conv_output_model.onnx', 'parent_conv_output_graph_sha256': PARENT_MODEL_SHA,
        'protocol_file': 'protocol.json', 'protocol_sha256': PROTOCOL_SHA,
        'input_file': 'dfl_softmax_probabilities.npy', 'input_file_sha256': PROBABILITY_SHA,
        'input_binding': {'name': INPUT_NAME, 'input': True, 'shape': [1, 16, 4, 8400], 'dtype': 'DataType.FLOAT'},
        'output_bindings_in_order': [{'name': OUTPUT_NAME, 'input': False, 'shape': [1, 1, 4, 8400],
            'dtype': 'DataType.FLOAT', 'role': 'dfl_conv_distances_with_singleton_channel_not_decoded_boxes', 'kind': 'primary'}],
        'conv_names': CONVS, 'source_conv_names': CONVS, 'clip_names': CLIPS,
        'source_activation_boundaries': copy.deepcopy(parent_spec['source_activation_boundaries']),
        'activation_scale': parent_spec['activation_scale'], 'variants': {'weight_bypass': variant},
        'removed_node_names': REMOVED, 'declared_node_mutation': copy.deepcopy(protocol['declared_node_mutation']),
        'semantic_change': protocol['semantic_change'], 'own_candidate_CPU_reference_required': True,
        'cpu_expected_outputs_file': 'cpu_expected_outputs.npz', 'cpu_expected_outputs_sha256': sha(output / 'cpu_expected_outputs.npz'),
        'cpu_diagnostic_file': 'cpu_diagnostic.json', 'cpu_diagnostic_sha256': sha(output / 'cpu_diagnostic.json'),
        'fixture_lineage_file': 'fixture_lineage.json', 'fixture_lineage_sha256': sha(output / 'fixture_lineage.json'),
        'graph_identity_audit_file': 'graph_identity_audit.json', 'graph_identity_audit_sha256': sha(output / 'graph_identity_audit.json'),
        'cases': copy.deepcopy(protocol['cases']), 'case_order': CASE_ORDER, 'max_seconds': 900, 'max_build_seconds': 300,
        'maximum_attempts_per_case': 1, 'maximum_native_builds': 2, 'maximum_native_inferences': 4,
        'executions_per_successful_case': 2,
        'native_importer_clip_attribute_correction': {'variant': 'weight_bypass', 'layer_names': CLIPS,
            'activation_alpha': -127.0, 'activation_beta': 127.0,
            'when': 'After parsing; preserve every parser layer precision/output request actually present.'},
        'native_build_policy': copy.deepcopy(protocol['common_native_settings']), 'new_CPU_execution_runs_this_stage': 1,
        'new_probability_fixture_generation_runs': 0, 'new_Softmax_72Conv_old_DFL_or_full_detector_runs': 0,
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
        'corrected_full_qall_failure': 'FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE_UNCHANGED',
        'new_tolerances_defined': False, 'original_tolerances_changed': False, 'calibration_recomputed': False,
        'original_weight_initializer_values_changed': False, 'weight_quantization_path_removed': True,
        'dataset_evaluation_run': False, 'complete_detector_executions': 0, 'native_build_or_execution_run_by_preparer': False,
        'limits': 'New weight-bypass semantics and parser context; own CPU reference, actual precision independently inspected; no unique full-model cause or repair claim.'}
    write(output / 'spec_draft.json', spec)
    (output / 'README.md').write_text('''# DFL 权重 QDQ 旁路局部对照

同修正 full q_all 源和父6节点图，仅移除 WeightQ/WeightDQ，并将
Conv.input[1] 改接原始 FP32 weight initializer；保留激活 Q/Clip/DQ
及权重原字节，得到4节点5使用中initializer。未添加 output QDQ 或新尺度。
这改变权重量化数学语义，并非与旧 W8A8 等价的修复。

复用原概率 NPY，不生成概率/不跑 Softmax、72Conv、旧 DFL 或完整检测器。
仅运行新4节点CPU一次，有限 FLOAT[1,1,4,8400] 结果独立保存为
weight_bypass_output 参考。旧6节点CPU仅描述比较，不要求字节相等。
失败保留，禁止重选/重试/seal/native。输出为DFL距离，不是解码框。

native NONE/OBEY 各构建一次300秒，batch900秒，成功各执行2次；
仅 Clip 导入属性校正为±127，不设置/重置任何层精度或输出请求，
不制造先前QDQ请求数量。源FP32权重不证明实际FP32计算，实际格式/tactic另审。
原full失败与Gate INCOMPLETE保留；本局部结果不构成完整图修复或唯一根因。
''', encoding='utf-8')
    write(output / 'prepared_content_manifest.json', {'schema': 'rampq.gate05.dfl_weight_bypass_prepared_content/v1',
        'status': 'OWN_ONE_RUN_CPU_REFERENCE_PREPARED_NOT_SEALED',
        'files': {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}})
    print(json.dumps({'status': 'OWN_ONE_RUN_CPU_REFERENCE_PREPARED_NOT_SEALED', 'output': str(output),
        'model_sha256': variant['model_sha256'], 'probability_fixture_sha256': PROBABILITY_SHA,
        'new_expected_NPZ_sha256': spec['cpu_expected_outputs_sha256'],
        'new_CPU_actual_NPY_sha256': sha(output / 'cpu_actual_weight_bypass.npy'), 'new_CPU_runs': 1,
        'old_reference_equality_required': False, 'source_bytes_unchanged': True}, indent=2))


def seal(output, native_paths):
    require(output.is_dir() and not output.with_suffix('.zip').exists() and not (output / 'manifest.json').exists() and
            not (output / 'spec.json').exists() and not (output / 'preparation_failure.json').exists(),
            'Successful prepared unsealed destination required')
    prepared = read(output / 'prepared_content_manifest.json')
    require(prepared['status'] == 'OWN_ONE_RUN_CPU_REFERENCE_PREPARED_NOT_SEALED', 'Wrong preparation status')
    for name, expected in prepared['files'].items():
        path = output / name
        require(path.resolve().parent == output and info(path) == expected, 'Prepared file became stale: ' + name)
    sources = read(output / 'source_hashes_before.json')
    require(sources == read(output / 'source_hashes_after.json'), 'Before/after source records differ')
    validate_sources(sources)
    diagnostic = read(output / 'cpu_diagnostic.json')
    require(sha(output / 'protocol.json') == PROTOCOL_SHA and diagnostic['status'] == 'OWN_ONE_RUN_CPU_REFERENCE_ESTABLISHED' and
            diagnostic['new_CPU_executions'] == 1 and diagnostic['actual_output']['all_finite'] and
            diagnostic['old_reference_adopted_as_candidate_expected'] is False, 'Protocol/own CPU evidence invalid')
    require([path.name for path in native_paths] == ['jetson_dfl_weight_bypass_probe.py', 'jetson_build_dfl_weight_bypass_probe.py'] and
            all(path.parent == ROOT / 'scripts' and path.is_file() and not path.is_symlink() for path in native_paths),
            'Expected distinct new workspace native tools')
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
    write(output / 'manifest.json', {'schema': 'rampq.gate05.dfl_weight_bypass_probe_bundle/v1', 'files': files})
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
    print(json.dumps({'status': 'DFL_WEIGHT_BYPASS_PROBE_SEALED', 'output': str(output), 'archive': str(archive_path),
        'archive_bytes': archive_path.stat().st_size, 'archive_sha256': sha(archive_path),
        'spec_sha256': sha(output / 'spec.json'), 'manifest_sha256': sha(output / 'manifest.json'),
        'protocol_sha256': PROTOCOL_SHA, 'native_runner_sha256': spec['native_runner_sha256'],
        'native_builder_sha256': spec['native_builder_sha256'], 'source_bytes_unchanged': True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--prepare-only', action='store_true')
    stage.add_argument('--seal-only', action='store_true')
    parser.add_argument('--native-runner', type=Path, default=ROOT / 'scripts/jetson_dfl_weight_bypass_probe.py')
    parser.add_argument('--native-builder', type=Path, default=ROOT / 'scripts/jetson_build_dfl_weight_bypass_probe.py')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / 'artifacts/gate05').resolve()), 'Unsafe artifact destination')
    if args.prepare_only:
        existed_before = output.exists()
        try:
            prepare(output, args.protocol.resolve())
        except BaseException as exc:
            if not existed_before and output.is_dir() and not (output / 'preparation_failure.json').exists():
                write(output / 'preparation_failure.json', {'schema': 'rampq.gate05.dfl_weight_bypass_failure/v1',
                    'status': 'FAILED_NO_RETRY_RESELECTION_OR_NATIVE', 'error_type': type(exc).__name__,
                    'error': str(exc), 'traceback': traceback.format_exc(), 'original_complete_model_guard': 'FAILED_UNCHANGED'})
            raise
    else:
        seal(output, [args.native_runner.resolve(), args.native_builder.resolve()])


if __name__ == '__main__':
    main()
