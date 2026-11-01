"""Freeze a source Softmax probability fixture and corrected eight-node DFL cut.

Exactly two new CPU executions are permitted: the one-node source Softmax once,
then the eight-node corrected cut once. No old upstream or DFL tail is rerun.
Preparation and sealing are separate exclusive stages; failure forbids sealing.
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
PARENT = ROOT / 'artifacts/gate05/20261020_dfl_context_probe'
ORIGINAL_CHECKPOINT = ROOT / 'artifacts/gate05/20261020_dfl_original_native/checkpoint.json'
NEXT_PLAN = ROOT / 'artifacts/gate05/20261020_dfl_original_native/next_control_plan.json'
PROTOCOL = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_protocol/protocol.json'
OUTPUT = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_probe'
ARCHIVE_ROOT = 'gate05_dfl_softmax_cut_probe_20261020'
PROTOCOL_SHA = 'bbe50e6485a72df194a13684895b4cc29a1db6ceca8ef098d05a6b5e56c56055'
SOURCE_SHA = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
SOURCE_SPEC_SHA = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
SOURCE_MANIFEST_SHA = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
PARENT_SPEC_SHA = '934e075aa58dde1829b600d3d011f5ad51c4f1da1af78f7791944f492781b14d'
PARENT_MANIFEST_SHA = '3836e82ed53afb14fe16d3b9431864c0c4a45fd0d3418c7c06a4f6fcd46ed1d0'
ORIGINAL_CHECKPOINT_SHA = '6a85e00d80464f07a9ce4d88c09191f2ae3e0e79b1c7b78ff4c744cd6077e665'
PLAN_SHA = 'de9e44fcd058cd6d855174af6934f657e5c312ee5284f8463322c0c029160be2'
LOGITS_SHA = 'cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab'
EXPECTED_SHA = 'fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511'
SAVED_CPU_SHA = 'fd4bcdcee24cd53651fdd7a508423c802f3b432c759647204c7b748f22abd50d'
SOFTMAX_NODE_SHA = 'b3b4c7e2d24556983e61ae3a731e6fc82625f83a8b8c671b0e98596c7f258adc'
LOGITS_NAME = '/model.23/dfl/Transpose_output_0'
INPUT_NAME = '/model.23/dfl/Softmax_output_0'
OUTPUT_NAME = '/model.23/dfl/Reshape_1_output_0'
SOFTMAX_NAME = '/model.23/dfl/Softmax'
CONVS = ['/model.23/dfl/conv/Conv']
CLIPS = ['rampq_model_23_dfl_conv_Clip']
CASE_ORDER = ['code_clip_none', 'code_clip_obey']


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
    require(sha(folder / 'manifest.json') == manifest_sha, 'Source manifest changed')
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
    for relative, expected in record.items():
        path = ROOT / relative
        require(path.resolve().is_relative_to(ROOT) and path.is_file() and info(path) == expected,
                'Recorded source became stale: ' + relative)


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


def extract(model, input_name, output_name, output_shape, graph_name, node_count, initializer_count):
    import onnx
    from onnx import helper
    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    needed, pending = set(), [output_name]
    while pending:
        name = pending.pop()
        if name == input_name or name in initializers:
            continue
        require(name in producers, 'Unexpected external closure dependency: ' + name)
        node = producers[name]
        if node.name not in needed:
            needed.add(node.name)
            pending.extend(value for value in node.input if value)
    selected = [copy.deepcopy(node) for node in model.graph.node if node.name in needed]
    constant_names = {name for node in selected for name in node.input if name in initializers}
    constants = [copy.deepcopy(value) for value in model.graph.initializer if value.name in constant_names]
    require(len(selected) == node_count and len(constants) == initializer_count, 'Closure inventory differs')
    if node_count == 1:
        require(selected[0].name == SOFTMAX_NAME and selected[0].op_type == 'Softmax' and
                message_sha(selected[0]) == SOFTMAX_NODE_SHA and
                {attr.name: helper.get_attribute_value(attr) for attr in selected[0].attribute} == {'axis': 1},
                'Source Softmax node or axis differs')
    else:
        require([node.name for node in selected if node.op_type == 'Conv'] == CONVS and
                [node.name for node in selected if node.op_type == 'Clip'] == CLIPS and
                Counter(node.op_type for node in selected) == Counter({
                    'QuantizeLinear': 2, 'DequantizeLinear': 2, 'Clip': 1,
                    'Conv': 1, 'Constant': 1, 'Reshape': 1}), 'Corrected DFL cut operators differ')
    result = copy.deepcopy(model)
    result.graph.CopyFrom(helper.make_graph(selected, graph_name,
        [source_value(model, input_name, [1, 16, 4, 8400])],
        [source_value(model, output_name, output_shape)], constants))
    onnx.checker.check_model(result, full_check=True)
    source_nodes = {node.name: node for node in model.graph.node}
    require(all(node.SerializeToString() == source_nodes[node.name].SerializeToString()
                for node in result.graph.node), 'Extracted node bytes changed')
    require(all(value.SerializeToString() == initializers[value.name].SerializeToString()
                for value in result.graph.initializer), 'Extracted initializer bytes changed')
    return result


def identity(model):
    return {'nodes': len(model.graph.node), 'initializers': len(model.graph.initializer),
        'conv_count': sum(node.op_type == 'Conv' for node in model.graph.node),
        'node_serialized_sha256': {node.name: message_sha(node) for node in model.graph.node},
        'initializer_serialized_sha256': {value.name: message_sha(value) for value in model.graph.initializer}}


def summary(array):
    import numpy as np
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'all_finite': bool(np.isfinite(array).all()),
            'min': float(array.min()), 'max': float(array.max()),
            'raw_array_bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def execute_cpu(model_path, input_name, output_name, values, expected_shape):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    require(session.get_providers() == ['CPUExecutionProvider'] and
            [value.name for value in session.get_inputs()] == [input_name] and
            [value.name for value in session.get_outputs()] == [output_name], 'CPU provider or tensor names differ')
    result = session.run([output_name], {input_name: values})[0]
    require(result.dtype == np.float32 and list(result.shape) == expected_shape and np.isfinite(result).all(),
            'CPU output dtype, shape or finite check failed')
    return result


def prepare(output, protocol_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import onnx
    import onnxruntime as ort
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Fresh output required; no CPU retry')
    require(sha(protocol_path) == PROTOCOL_SHA, 'Frozen protocol changed')
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rampq.gate05.dfl_softmax_cut_protocol/v1' and
            protocol['case_order'] == CASE_ORDER and protocol['batch_max_seconds'] == 900 and
            protocol['source_conv_names'] == CONVS and protocol['clip_names'] == CLIPS and
            protocol['cpu_composition_check']['total_new_CPU_runs'] == 2, 'Frozen protocol scope differs')
    source_paths = check_bundle(SOURCE, SOURCE_MANIFEST_SHA)
    source_paths.extend(check_bundle(PARENT, PARENT_MANIFEST_SHA))
    pins = {SOURCE / 'model.onnx': SOURCE_SHA, SOURCE / 'spec.json': SOURCE_SPEC_SHA,
        PARENT / 'spec.json': PARENT_SPEC_SHA, PARENT / 'dfl_transpose_fixture.npy': LOGITS_SHA,
        PARENT / 'cpu_expected_outputs.npz': EXPECTED_SHA, PARENT / 'cpu_actual_code_clip.npy': SAVED_CPU_SHA,
        ORIGINAL_CHECKPOINT: ORIGINAL_CHECKPOINT_SHA, NEXT_PLAN: PLAN_SHA, protocol_path: PROTOCOL_SHA}
    for relative, expected in read(ORIGINAL_CHECKPOINT)['protected_identities'].items():
        pins[ROOT / relative] = expected
    for path, expected in pins.items():
        require(sha(path) == expected, 'Historical pinned source differs: ' + str(path))
    source_paths.extend([*pins, Path(__file__).resolve()])
    before = source_hashes(source_paths)
    full = onnx.load(str(SOURCE / 'model.onnx'))
    generator = extract(full, LOGITS_NAME, INPUT_NAME, [1, 16, 4, 8400],
                        'exact_source_dfl_softmax_probability_fixture', 1, 0)
    cut = extract(full, INPUT_NAME, OUTPUT_NAME, [1, 4, 8400],
                  'exact_corrected_dfl_after_softmax_cut', 8, 7)
    output.mkdir(parents=True, exist_ok=False)
    copies = {SOURCE / 'model.onnx': 'source_model.onnx', SOURCE / 'spec.json': 'parent_spec.json',
        SOURCE / 'manifest.json': 'parent_manifest.json',
        PARENT / 'spec.json': 'parent_dfl_spec.json', PARENT / 'manifest.json': 'parent_dfl_manifest.json',
        PARENT / 'fixture_lineage.json': 'parent_fixture_lineage.json',
        PARENT / 'dfl_transpose_fixture.npy': 'saved_dfl_logits.npy',
        PARENT / 'cpu_expected_outputs.npz': 'cpu_expected_outputs.npz',
        PARENT / 'cpu_actual_code_clip.npy': 'saved_cpu_reference.npy',
        PARENT / 'original_failed_preflight.json': 'original_failed_preflight.json',
        PARENT / 'retained_qall_failure_checkpoint.json': 'retained_qall_failure_checkpoint.json',
        PARENT / 'failure_inventory.json': 'failure_inventory.json',
        ORIGINAL_CHECKPOINT: 'parent_dfl_original_checkpoint.json', NEXT_PLAN: 'parent_next_control_plan.json',
        protocol_path: 'protocol.json', Path(__file__).resolve(): Path(__file__).name}
    for path, name in copies.items():
        shutil.copy2(path, output / name)
    write(output / 'source_hashes_before.json', before)
    generator_file, cut_file = 'model_dfl_source_softmax.onnx', 'model_dfl_softmax_cut_code_clip.onnx'
    onnx.save(generator, str(output / generator_file))
    onnx.save(cut, str(output / cut_file))
    audit = {'schema': 'rampq.gate05.dfl_softmax_cut_graph_identity/v1',
        'status': 'EXACT_SOURCE_GENERATOR_AND_CUT_VERIFIED', 'source_graph_sha256': SOURCE_SHA,
        'all_nodes_initializers_and_IO_declarations_exact_source_bytes': True,
        'Softmax_generator': {'model_file': generator_file, 'model_sha256': sha(output / generator_file), **identity(generator)},
        'code_clip_cut': {'model_file': cut_file, 'model_sha256': sha(output / cut_file), **identity(cut)},
        'new_72Conv_upstream_executions': 0, 'new_full_detector_executions': 0}
    write(output / 'graph_identity_audit.json', audit)
    write(output / 'cpu_execution_plan.json', {'schema': 'rampq.gate05.dfl_softmax_cut_cpu_plan/v1',
        'status': 'FROZEN_BEFORE_FIRST_CPU_EXECUTION', 'protocol_sha256': PROTOCOL_SHA,
        'ordered_runs': ['exact_source_Softmax_once', 'corrected_eight_node_cut_once'],
        'maximum_total_new_CPU_runs': 2, 'provider': 'CPUExecutionProvider',
        'graph_optimization': 'ORT_DISABLE_ALL', 'threads': 4,
        'new_72Conv_or_old_DFL_or_full_detector_runs': 0,
        'on_failure': 'PRESERVE_FAILURE_NO_RETRY_RESELECTION_OR_NATIVE_START'})
    logits = np.load(output / 'saved_dfl_logits.npy', allow_pickle=False)
    reference = np.load(output / 'saved_cpu_reference.npy', allow_pickle=False)
    require(sha(output / 'saved_dfl_logits.npy') == LOGITS_SHA and logits.dtype == np.float32 and
            logits.shape == (1, 16, 4, 8400) and logits.flags.c_contiguous and np.isfinite(logits).all(), 'Saved logits invalid')
    require(sha(output / 'saved_cpu_reference.npy') == SAVED_CPU_SHA and reference.dtype == np.float32 and
            reference.shape == (1, 4, 8400) and np.isfinite(reference).all(), 'Saved CPU reference invalid')
    with np.load(output / 'cpu_expected_outputs.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'original_output', 'code_clip_output'} and
                saved['code_clip_output'].tobytes(order='C') == reference.tobytes(order='C') and
                saved['original_output'].tobytes(order='C') == reference.tobytes(order='C'), 'Reused CPU NPZ differs')
    probabilities = execute_cpu(output / generator_file, LOGITS_NAME, INPUT_NAME, logits, [1, 16, 4, 8400])
    probabilities = np.ascontiguousarray(probabilities)
    require(float(probabilities.min()) >= 0.0 and float(probabilities.max()) <= 1.0,
            'Source Softmax probabilities outside numeric domain')
    np.save(output / 'dfl_softmax_probabilities.npy', probabilities)
    sums = probabilities.sum(axis=1, dtype=np.float64)
    fixture_lineage = {'schema': 'rampq.gate05.dfl_softmax_probability_fixture/v1',
        'status': 'ONE_SOURCE_SOFTMAX_CPU_RUN_SAVED', 'fixture_file': 'dfl_softmax_probabilities.npy',
        'fixture_sha256': sha(output / 'dfl_softmax_probabilities.npy'), 'fixture': summary(probabilities),
        'fixture_tensor_name': INPUT_NAME, 'source_graph_sha256': SOURCE_SHA,
        'source_Softmax_node_name': SOFTMAX_NAME, 'source_Softmax_node_serialized_sha256': SOFTMAX_NODE_SHA,
        'source_Softmax_axis': 1, 'generator_file': generator_file, 'generator_sha256': sha(output / generator_file),
        'saved_logits_file': 'saved_dfl_logits.npy', 'saved_logits_sha256': LOGITS_SHA,
        'probability_axis1_sums_descriptive': {'min': float(sums.min()), 'max': float(sums.max()),
                                             'max_abs_from_one': float(np.abs(sums - 1).max())},
        'provider': 'CPUExecutionProvider', 'graph_optimization': 'ORT_DISABLE_ALL', 'threads': 4,
        'new_generator_CPU_runs': 1, 'new_72Conv_or_old_DFL_or_full_detector_runs': 0,
        'meaning': 'Exact source Softmax CPU probabilities from previously frozen corrected ONNX logits; not native Softmax, image pixels, or a replacement Gate reference.'}
    write(output / 'fixture_lineage.json', fixture_lineage)
    actual = execute_cpu(output / cut_file, INPUT_NAME, OUTPUT_NAME, probabilities, [1, 4, 8400])
    np.save(output / 'cpu_actual_code_clip.npy', actual)
    equal = actual.tobytes(order='C') == reference.tobytes(order='C')
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    diagnostic = {'schema': 'rampq.gate05.dfl_softmax_cut_cpu/v1',
        'status': 'TWO_CPU_RUNS_COMPOSITION_BYTE_EQUIVALENCE_PASSED' if equal else 'CPU_COMPOSITION_BYTE_EQUIVALENCE_FAILED',
        'new_CPU_executions': 2, 'generator_CPU_runs': 1, 'cut_CPU_runs': 1,
        'new_72Conv_or_old_DFL_or_full_detector_runs': 0, 'native_executions': 0,
        'provider': 'CPUExecutionProvider', 'graph_optimization': 'ORT_DISABLE_ALL', 'threads': 4,
        'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
        'actual_output_file': 'cpu_actual_code_clip.npy', 'actual_output_sha256': sha(output / 'cpu_actual_code_clip.npy'),
        'actual_output': summary(actual), 'saved_reference_sha256': SAVED_CPU_SHA,
        'byte_identical_saved_corrected_DFL_output': equal, 'both_outputs_byte_exact': equal,
        'differing_elements': int(np.count_nonzero(delta)), 'max_absolute_difference': float(np.abs(delta).max()),
        'fixture_sha256': fixture_lineage['fixture_sha256'], 'expected_NPZ_sha256_unchanged': EXPECTED_SHA,
        'output_meaning': 'DFL_DISTANCES_NOT_DECODED_BOXES',
        'original_complete_model_guard': 'FAILED_UNCHANGED', 'limits': protocol['cpu_composition_check']['proves']}
    write(output / 'cpu_diagnostic.json', diagnostic)
    after = source_hashes(source_paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'Historical sources changed during two CPU runs')
    require(equal and sha(output / 'cpu_actual_code_clip.npy') == SAVED_CPU_SHA,
            'New cut CPU composition differs; preserve and forbid retry/native start')
    parent_spec = read(PARENT / 'spec.json')
    variant = {'model_file': cut_file, 'model_sha256': sha(output / cut_file), 'nodes': 8, 'initializers': 7,
        'expected_output_npz_keys': {OUTPUT_NAME: 'code_clip_output'},
        'selected_node_serialized_sha256': identity(cut)['node_serialized_sha256'],
        'selected_initializer_serialized_sha256': identity(cut)['initializer_serialized_sha256']}
    spec = {'schema': 'rampq.gate05.dfl_softmax_cut_probe/v1',
        'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
        'source_configuration': 'q_all', 'source_graph_file': 'source_model.onnx', 'source_graph_sha256': SOURCE_SHA,
        'parent_spec_file': 'parent_spec.json', 'parent_spec_sha256': SOURCE_SPEC_SHA,
        'parent_manifest_file': 'parent_manifest.json', 'parent_manifest_sha256': SOURCE_MANIFEST_SHA,
        'parent_dfl_spec_file': 'parent_dfl_spec.json', 'parent_dfl_spec_sha256': PARENT_SPEC_SHA,
        'parent_dfl_manifest_file': 'parent_dfl_manifest.json', 'parent_dfl_manifest_sha256': PARENT_MANIFEST_SHA,
        'parent_dfl_original_checkpoint_file': 'parent_dfl_original_checkpoint.json',
        'parent_dfl_original_checkpoint_sha256': ORIGINAL_CHECKPOINT_SHA,
        'protocol_file': 'protocol.json', 'protocol_sha256': PROTOCOL_SHA,
        'input_file': 'dfl_softmax_probabilities.npy', 'input_file_sha256': fixture_lineage['fixture_sha256'],
        'input_binding': {'name': INPUT_NAME, 'input': True, 'shape': [1, 16, 4, 8400], 'dtype': 'DataType.FLOAT'},
        'output_bindings_in_order': copy.deepcopy(parent_spec['output_bindings_in_order']),
        'conv_names': CONVS, 'source_conv_names': CONVS, 'clip_names': CLIPS,
        'source_activation_boundaries': copy.deepcopy(parent_spec['source_activation_boundaries']),
        'activation_scale': parent_spec['activation_scale'], 'variants': {'code_clip': variant},
        'cpu_expected_outputs_file': 'cpu_expected_outputs.npz', 'cpu_expected_outputs_sha256': EXPECTED_SHA,
        'cpu_diagnostic_file': 'cpu_diagnostic.json', 'cpu_diagnostic_sha256': sha(output / 'cpu_diagnostic.json'),
        'fixture_lineage_file': 'fixture_lineage.json', 'fixture_lineage_sha256': sha(output / 'fixture_lineage.json'),
        'graph_identity_audit_file': 'graph_identity_audit.json',
        'graph_identity_audit_sha256': sha(output / 'graph_identity_audit.json'),
        'cases': copy.deepcopy(protocol['cases']), 'case_order': CASE_ORDER,
        'max_seconds': 900, 'max_build_seconds': 300, 'maximum_attempts_per_case': 1,
        'maximum_native_builds': 2, 'maximum_native_inferences': 4, 'executions_per_successful_case': 2,
        'native_importer_clip_attribute_correction': {'variant': 'code_clip', 'layer_names': CLIPS,
            'activation_alpha': -127.0, 'activation_beta': 127.0,
            'when': 'After parsing; preserve every parser layer precision/output request.'},
        'native_build_policy': copy.deepcopy(protocol['common_native_settings']),
        'new_CPU_execution_runs_this_stage': 2, 'new_72Conv_upstream_runs': 0,
        'saved_logits_sha256': LOGITS_SHA, 'new_probability_fixture_generated': True,
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
        'corrected_full_qall_failure': 'FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE_UNCHANGED',
        'new_tolerances_defined': False, 'original_tolerances_changed': False,
        'calibration_recomputed': False, 'weights_changed': False, 'dataset_evaluation_run': False,
        'complete_detector_executions': 0, 'native_build_or_execution_run_by_preparer': False,
        'limits': protocol['cross_stage_policy']}
    write(output / 'spec_draft.json', spec)
    (output / 'README.md').write_text('# Local comparison of native DFL with precise clipping after Softmax\n\nOnly extract the original Softmax single node, zero initializer graph, and the subsequent 8 nodes from the corrected q_all.\n7initializer DFL slicing, all selected nodes, constants, and I/O declarations preserve the original serialized bytes.\noriginal Softmax axis=1, perform one CPU probability generation fixture using authenticated previous logits\n8-node cut followed by execution on CPU must produce an output that exactly matches the saved original DFL CPU raw bytes.\nStrictly limit to two CPU passes, without re-running the 72Conv upstream, the old 9-node DFL, or the full detector; no re-selection or retry.\n\nnative input is FLOAT[1,16,4,8400] probability, not logits or images. Output is\nFLOAT[1,4,8400] DFL distance, not a detection box. Old expected NPZ raw bytes reuse.\nOne code_clip none/obey build each, limited to 300 seconds, with a batch limit of 900 seconds; repeat execution twice only after success.\nCorrect only the original Clip import of alpha/beta to -127/+127, without setting or resetting parsing precision constraints;\nConvolution output allows FP32; the actual format and tactic must be recorded separately.\n\nThe cut simultaneously modifies the input boundaries and native fusion context, and uses CPU-generated probabilities, thus the result\nChanges cannot be attributed to Softmax as the sole reason for the complete graph failure. Original complete guardrail FAILED_UNCHANGED,\nFailure of OBEY with original/corrected differs in NONE values, Gate0.5 still failed.\n', encoding='utf-8')
    write(output / 'prepared_content_manifest.json', {
        'schema': 'rampq.gate05.dfl_softmax_cut_prepared_content/v1',
        'status': 'TWO_CPU_COMPOSITION_CHECKS_PASSED_NOT_SEALED',
        'files': {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}})
    print(json.dumps({'status': 'TWO_CPU_COMPOSITION_CHECKS_PASSED_NOT_SEALED', 'output': str(output),
        'code_clip_model_sha256': variant['model_sha256'], 'probability_fixture_sha256': fixture_lineage['fixture_sha256'],
        'cpu_expected_outputs_sha256': EXPECTED_SHA, 'new_CPU_runs': 2,
        'probabilities': summary(probabilities), 'probability_axis1_sums': fixture_lineage['probability_axis1_sums_descriptive'],
        'cut_output_byte_identical_saved_DFL_CPU': True, 'new_72Conv_or_old_DFL_or_full_detector_runs': 0,
        'source_bytes_unchanged': True}, indent=2))


def seal(output, native_paths):
    require(output.is_dir() and not output.with_suffix('.zip').exists() and
            not (output / 'manifest.json').exists() and not (output / 'spec.json').exists() and
            not (output / 'preparation_failure.json').exists(), 'Successful prepared unsealed paths required')
    prepared = read(output / 'prepared_content_manifest.json')
    require(prepared['status'] == 'TWO_CPU_COMPOSITION_CHECKS_PASSED_NOT_SEALED', 'Wrong preparation status')
    for name, expected in prepared['files'].items():
        path = output / name
        require(path.resolve().parent == output and info(path) == expected, 'Prepared file became stale: ' + name)
    sources = read(output / 'source_hashes_before.json')
    require(sources == read(output / 'source_hashes_after.json'), 'Source before/after records differ')
    validate_sources(sources)
    require(sha(output / 'protocol.json') == PROTOCOL_SHA and
            read(output / 'cpu_diagnostic.json')['byte_identical_saved_corrected_DFL_output'], 'Protocol/CPU evidence invalid')
    require(len(native_paths) == 2 and len({path.name for path in native_paths}) == 2 and
            all(path.parent == ROOT / 'scripts' and path.is_file() and not path.is_symlink() for path in native_paths),
            'Native scripts must be distinct workspace scripts')
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
    require(len(files) == len(list(output.iterdir())), 'Unexpected artifact subdirectory')
    write(output / 'manifest.json', {'schema': 'rampq.gate05.dfl_softmax_cut_probe_bundle/v1', 'files': files})
    sums = dict(files)
    sums['manifest.json'] = info(output / 'manifest.json')
    (output / 'SHA256SUMS').write_text(''.join(value['sha256'] + '  ' + name + '\n'
        for name, value in sorted(sums.items())), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, ARCHIVE_ROOT + '/' + path.name)
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'ZIP CRC failed')
        for path in output.iterdir():
            require(hashlib.sha256(archive.read(ARCHIVE_ROOT + '/' + path.name)).hexdigest() == sha(path),
                    'ZIP bytes differ from sealed bundle: ' + path.name)
    print(json.dumps({'status': 'DFL_SOFTMAX_CUT_PROBE_SEALED', 'output': str(output),
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
    parser.add_argument('--native-runner', type=Path, default=ROOT / 'scripts/jetson_dfl_softmax_cut_probe.py')
    parser.add_argument('--native-builder', type=Path, default=ROOT / 'scripts/jetson_build_dfl_softmax_cut_probe.py')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / 'artifacts/gate05').resolve()), 'Unsafe artifact destination')
    if args.prepare_only:
        existed_before = output.exists()
        try:
            prepare(output, args.protocol.resolve())
        except BaseException as exc:
            if not existed_before and output.is_dir() and not (output / 'preparation_failure.json').exists():
                write(output / 'preparation_failure.json', {'schema': 'rampq.gate05.dfl_softmax_cut_failure/v1',
                    'status': 'FAILED_NO_RETRY_RESELECTION_OR_NATIVE', 'error_type': type(exc).__name__,
                    'error': str(exc), 'traceback': traceback.format_exc(),
                    'original_complete_model_guard': 'FAILED_UNCHANGED'})
            raise
    else:
        seal(output, [args.native_runner.resolve(), args.native_builder.resolve()])


if __name__ == '__main__':
    main()
