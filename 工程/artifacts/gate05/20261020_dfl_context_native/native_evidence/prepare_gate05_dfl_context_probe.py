"""Prepare one authenticated CPU internal fixture and the exact DFL tail pair.

Exactly one large upstream CPU ancestor execution generates logits. Two local
tail executions use that same fixture. Preparation and sealing are exclusive;
failed CPU preparation cannot be retried in the same artifact directory.
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
CORRECTED = ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate'
ORIGINAL = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles/q_all'
PROTOCOL = ROOT / 'artifacts/gate05/20261020_dfl_context_protocol/protocol.json'
INVENTORY = ROOT / 'artifacts/gate05/20261019_corrected_qall_native/failure_boundary_inventory.json'
CHECKPOINT = ROOT / 'artifacts/gate05/20261019_corrected_qall_native/checkpoint.json'
OPTIONS = ROOT / 'artifacts/gate05/20261020_twoconv_context_review/next_context_static_options.json'
OUTPUT = ROOT / 'artifacts/gate05/20261020_dfl_context_probe'
ARCHIVE_ROOT = 'gate05_dfl_context_probe_20261020'
PROTOCOL_SHA = 'b2723549e6acee51c86aeba3b72c8451601ba4add6088f46f7443536a1ab74ee'
CORRECTED_SHA = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
CORRECTED_SPEC_SHA = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
CORRECTED_MANIFEST_SHA = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
ORIGINAL_SHA = '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d'
ORIGINAL_MANIFEST_SHA = 'd7fb5fa24234ec38c11a88003a8cb5b7447257e4ca844e77e88247460524950b'
IMAGE_SHA = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
INVENTORY_SHA = 'a2780d83f1de24ab166553a8a3d341f04cf3c6f74ae41cabd5885124bc81d112'
CHECKPOINT_SHA = '66e149025bb0d777afe79a013797704c7878fdcad30140ac256aac305db48471'
OPTIONS_SHA = '41aaa27de8b5f275cf3632933e3d1609c03a0028b2d6906f1d8e1f4022ee6378'
PREFLIGHT_SHA = '505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21'
CONVS = ['/model.23/dfl/conv/Conv']
PREFIX = 'rampq_model_23_dfl_conv'
CLIPS = [PREFIX + '_Clip']
INPUT_NAME = '/model.23/dfl/Transpose_output_0'
OUTPUT_NAME = '/model.23/dfl/Reshape_1_output_0'
CASE_ORDER = ['code_clip_none', 'code_clip_obey']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def message_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check_bundle(folder, manifest_sha):
    require(sha(folder / 'manifest.json') == manifest_sha, 'Frozen source manifest changed')
    paths = [folder / 'manifest.json']
    for name, expected in read(folder / 'manifest.json')['files'].items():
        path = folder / name
        require(path.resolve().parent == folder.resolve() and path.is_file() and
                not path.is_symlink() and info(path) == expected, 'Frozen bundle differs: ' + str(path))
        paths.append(path)
    return paths


def source_hashes(paths):
    return {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): info(path)
            for path in sorted(set(paths))}


def validate_recorded_sources(record):
    for relative, expected in record.items():
        path = ROOT / relative
        require(path.resolve().is_relative_to(ROOT) and path.is_file() and info(path) == expected,
                'Recorded source became stale: ' + relative)


def value_info(model, name, expected_shape):
    import onnx
    values = {value.name: value for value in list(model.graph.input) +
              list(model.graph.value_info) + list(model.graph.output)}
    require(name in values, 'No source tensor declaration: ' + name)
    value = copy.deepcopy(values[name])
    require(value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
            [dim.dim_value for dim in value.type.tensor_type.shape.dim] == expected_shape,
            'Source tensor declaration differs: ' + name)
    return value


def extract(model, input_name, input_shape, output_name, output_shape, graph_name,
            expected_nodes, expected_initializers, expected_convs):
    import onnx
    from onnx import helper
    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    needed, pending = set(), [output_name]
    while pending:
        name = pending.pop()
        if name == input_name or name in initializers:
            continue
        require(name in producers, 'Unexpected external ancestor dependency: ' + name)
        node = producers[name]
        if node.name not in needed:
            needed.add(node.name)
            pending.extend(item for item in node.input if item)
    selected = [copy.deepcopy(node) for node in model.graph.node if node.name in needed]
    constant_names = {name for node in selected for name in node.input if name in initializers}
    constants = [copy.deepcopy(value) for value in model.graph.initializer if value.name in constant_names]
    conv_names = [node.name for node in selected if node.op_type == 'Conv']
    require(len(selected) == expected_nodes and len(constants) == expected_initializers and
            len(conv_names) == expected_convs, 'Dependency closure inventory differs')
    if expected_convs == 1:
        require(conv_names == CONVS and Counter(node.op_type for node in selected) == Counter({
            'Softmax': 1, 'QuantizeLinear': 2, 'Clip': 1, 'DequantizeLinear': 2,
            'Conv': 1, 'Constant': 1, 'Reshape': 1}), 'DFL tail operator inventory differs')
    extracted = copy.deepcopy(model)
    extracted.graph.CopyFrom(helper.make_graph(selected, graph_name,
        [value_info(model, input_name, input_shape)],
        [value_info(model, output_name, output_shape)], constants))
    onnx.checker.check_model(extracted, full_check=True)
    source_nodes = {node.name: node for node in model.graph.node}
    require(all(node.SerializeToString() == source_nodes[node.name].SerializeToString()
                for node in extracted.graph.node), 'Extracted node differs from source')
    require(all(value.SerializeToString() == initializers[value.name].SerializeToString()
                for value in extracted.graph.initializer), 'Extracted initializer differs from source')
    available = {input_name} | constant_names
    for node in selected:
        require(set(node.input) - {''} <= available, 'Closure has unfrozen dependency: ' + node.name)
        available.update(node.output)
    require(output_name in available, 'Closure output missing')
    return extracted


def identity(model):
    return {'nodes': len(model.graph.node), 'initializers': len(model.graph.initializer),
            'conv_count': sum(node.op_type == 'Conv' for node in model.graph.node),
            'node_serialized_sha256': {node.name: message_sha(node) for node in model.graph.node},
            'initializer_serialized_sha256': {value.name: message_sha(value) for value in model.graph.initializer}}


def check_tail_rewrite(original, corrected):
    import numpy as np
    from onnx import numpy_helper
    old_nodes, new_nodes = ({node.name: node for node in model.graph.node} for model in (original, corrected))
    old_init, new_init = ({value.name: value for value in model.graph.initializer} for model in (original, corrected))
    require(old_nodes.keys() == new_nodes.keys() and old_init.keys() == new_init.keys(), 'Tail names changed')
    q, clip, dq = (old_nodes[PREFIX + suffix] for suffix in ('_ActivationQ', '_Clip', '_ActivationDQ'))
    expected_q, expected_clip = copy.deepcopy(q), copy.deepcopy(clip)
    expected_q.input[0], expected_q.output[0] = clip.input[0], q.output[0] + '_before_code_clip'
    expected_clip.input[0], expected_clip.output[0] = expected_q.output[0], q.output[0]
    require(new_nodes[q.name].SerializeToString() == expected_q.SerializeToString() and
            new_nodes[clip.name].SerializeToString() == expected_clip.SerializeToString(), 'Unexpected DFL quantizer rewrite')
    require(all(node.SerializeToString() == new_nodes[name].SerializeToString()
                for name, node in old_nodes.items() if name not in {q.name, clip.name}), 'Non-boundary tail node changed')
    scale, zero = (numpy_helper.to_array(old_init[name]) for name in q.input[1:])
    require(scale.shape == zero.shape == () and scale.dtype == np.float32 and zero.dtype == np.int8 and
            float(scale) == 0.007873943075537682 and int(zero) == 0, 'Frozen DFL scale/zero changed')
    old_bounds = []
    for name, bound in zip(clip.input[1:], (-127, 127)):
        old_array, new_array = numpy_helper.to_array(old_init[name]), numpy_helper.to_array(new_init[name])
        require(old_array.dtype == np.float32 and old_array.shape == () and
                old_array.tobytes() == np.asarray(np.float32(bound) * scale, dtype=np.float32).tobytes() and
                new_array.dtype == np.int8 and new_array.shape == () and int(new_array) == bound, 'DFL Clip bounds changed')
        old_bounds.append(float(old_array))
    require(all(value.SerializeToString() == new_init[name].SerializeToString()
                for name, value in old_init.items() if name not in clip.input[1:]), 'Non-bound initializer changed')
    return {'prefix': PREFIX, 'q_name': q.name, 'clip_name': clip.name, 'dq_name': dq.name,
            'scale_initializer': q.input[1], 'scale': float(scale), 'zero_initializer': q.input[2], 'zero': 0,
            'original_float_clip_bounds': old_bounds, 'corrected_int8_clip_bounds': [-127, 127],
            'all_other_nodes_and_initializers_exact_source_bytes': True}


def summary(array):
    import numpy as np
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'all_finite': bool(np.isfinite(array).all()),
            'min': float(array.min()), 'max': float(array.max()),
            'raw_array_bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def execute_cpu(model_path, input_name, output_name, array, expected_shape):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    require(session.get_providers() == ['CPUExecutionProvider'] and
            [value.name for value in session.get_inputs()] == [input_name] and
            [value.name for value in session.get_outputs()] == [output_name], 'Unexpected CPU I/O or provider')
    output = session.run([output_name], {input_name: array})[0]
    require(output.dtype == np.float32 and list(output.shape) == expected_shape and np.isfinite(output).all(),
            'CPU output has wrong dtype/shape or nonfinite values')
    return output


def prepare(output, protocol_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import onnx
    import onnxruntime as ort
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Choose fresh preparation paths; no retry')
    require(sha(protocol_path) == PROTOCOL_SHA, 'Root-frozen protocol changed')
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rampq.gate05.dfl_context_protocol/v1' and
            protocol['case_order'] == CASE_ORDER and protocol['batch_max_seconds'] == 900 and
            protocol['source_conv_names'] == CONVS and protocol['clip_names'] == CLIPS,
            'Protocol scope differs')
    paths = check_bundle(CORRECTED, CORRECTED_MANIFEST_SHA)
    paths.extend(check_bundle(ORIGINAL, ORIGINAL_MANIFEST_SHA))
    pinned = {CORRECTED / 'model.onnx': CORRECTED_SHA, CORRECTED / 'spec.json': CORRECTED_SPEC_SHA,
              CORRECTED / 'preprocessed_input.npy': IMAGE_SHA, CORRECTED / 'preflight.json': PREFLIGHT_SHA,
              ORIGINAL / 'model.onnx': ORIGINAL_SHA, INVENTORY: INVENTORY_SHA, CHECKPOINT: CHECKPOINT_SHA,
              OPTIONS: OPTIONS_SHA, protocol_path: PROTOCOL_SHA}
    for path, expected in pinned.items():
        require(sha(path) == expected, 'Pinned source differs: ' + str(path))
    paths.extend([*pinned, Path(__file__).resolve()])
    before = source_hashes(paths)
    corrected_full, original_full = onnx.load(str(CORRECTED / 'model.onnx')), onnx.load(str(ORIGINAL / 'model.onnx'))
    upstream = extract(corrected_full, 'images', [1, 3, 640, 640], INPUT_NAME, [1, 16, 4, 8400],
                       'exact_corrected_qall_dfl_fixture_ancestors', 638, 576, 72)
    models = {variant: extract(full, INPUT_NAME, [1, 16, 4, 8400], OUTPUT_NAME, [1, 4, 8400],
                               'exact_dfl_tail_' + variant, 9, 7, 1)
              for variant, full in (('original', original_full), ('code_clip', corrected_full))}
    boundary = check_tail_rewrite(models['original'], models['code_clip'])
    output.mkdir(parents=True, exist_ok=False)
    copies = {CORRECTED / 'model.onnx': 'source_model.onnx',
              CORRECTED / 'preprocessed_input.npy': 'source_images.npy',
              CORRECTED / 'spec.json': 'parent_spec.json', CORRECTED / 'manifest.json': 'parent_manifest.json',
              CORRECTED / 'preflight.json': 'original_failed_preflight.json',
              ORIGINAL / 'spec.json': 'original_qall_spec.json', ORIGINAL / 'manifest.json': 'original_qall_manifest.json',
              INVENTORY: 'failure_inventory.json', CHECKPOINT: 'retained_qall_failure_checkpoint.json',
              OPTIONS: 'static_options.json', protocol_path: 'protocol.json',
              Path(__file__).resolve(): Path(__file__).name}
    for path, name in copies.items():
        shutil.copy2(path, output / name)
    write(output / 'source_hashes_before.json', before)
    upstream_name = 'model_dfl_fixture_ancestors.onnx'
    onnx.save(upstream, str(output / upstream_name))
    for variant, model in models.items():
        onnx.save(model, str(output / ('model_dfl_' + variant + '.onnx')))
    graph_audit = {'schema': 'rampq.gate05.dfl_context_graph_identity/v1',
                   'status': 'EXACT_SOURCE_DEPENDENCY_CLOSURES_VERIFIED',
                   'source_graphs': {'original': ORIGINAL_SHA, 'code_clip': CORRECTED_SHA},
                   'all_selected_nodes_and_initializers_exact_respective_source_bytes': True,
                   'upstream_ancestor': {'model_file': upstream_name, 'model_sha256': sha(output / upstream_name),
                                         **identity(upstream)},
                   'tail_variants': {variant: identity(model) for variant, model in models.items()},
                   'sole_quantizer_rewrite': boundary, 'tail_output_meaning': 'DFL_DISTANCES_NOT_DECODED_BOXES',
                   'original_complete_model_guard': 'FAILED_UNCHANGED'}
    write(output / 'graph_identity_audit.json', graph_audit)
    write(output / 'cpu_execution_plan.json', {'schema': 'rampq.gate05.dfl_cpu_execution_plan/v1',
        'status': 'FROZEN_BEFORE_FIRST_CPU_EXECUTION', 'protocol_sha256': PROTOCOL_SHA,
        'upstream_ancestor_runs': 1, 'original_tail_runs': 1, 'code_clip_tail_runs': 1,
        'order': ['corrected_upstream_once', 'original_tail_once', 'code_clip_tail_once'],
        'provider': 'CPUExecutionProvider', 'threads': 4, 'graph_optimization': 'ORT_DISABLE_ALL',
        'on_failure': 'PRESERVE_FAILURE_NO_RETRY_OR_FIXTURE_RESELECTION',
        'complete_detector_output_executions': 0, 'native_executions': 0})
    image = np.load(output / 'source_images.npy', allow_pickle=False)
    require(sha(output / 'source_images.npy') == IMAGE_SHA and image.dtype == np.float32 and
            image.shape == (1, 3, 640, 640) and image.flags.c_contiguous and np.isfinite(image).all(), 'Invalid frozen image')
    fixture = execute_cpu(output / upstream_name, 'images', INPUT_NAME, image, [1, 16, 4, 8400])
    fixture = np.ascontiguousarray(fixture)
    np.save(output / 'dfl_transpose_fixture.npy', fixture)
    fixture_lineage = {'schema': 'rampq.gate05.dfl_fixture_lineage/v1',
        'status': 'ONE_CORRECTED_ANCESTOR_CPU_RUN_SAVED', 'fixture_file': 'dfl_transpose_fixture.npy',
        'fixture_sha256': sha(output / 'dfl_transpose_fixture.npy'), 'fixture': summary(fixture),
        'fixture_tensor_name': INPUT_NAME, 'source_graph_file': 'source_model.onnx', 'source_graph_sha256': CORRECTED_SHA,
        'source_image_file': 'source_images.npy', 'source_image_sha256': IMAGE_SHA,
        'ancestor_graph_file': upstream_name, 'ancestor_graph_sha256': sha(output / upstream_name),
        'ancestor_nodes': 638, 'ancestor_initializers': 576, 'ancestor_conv_count': 72,
        'ancestor_cpu_executions': 1, 'provider': 'CPUExecutionProvider', 'threads': 4, 'graph_optimization': 'ORT_DISABLE_ALL',
        'meaning': 'Corrected ONNX CPU internal Transpose logits before Softmax. Neither image pixels nor decoded boxes; not a native fixture or replacement Gate reference.',
        'does_not_prove_original_corrected_upstream_equality': True, 'complete_detector_output_executions': 0}
    write(output / 'fixture_lineage.json', fixture_lineage)
    arrays, variants, executions = {}, {}, {}
    for variant, model in models.items():
        filename = 'model_dfl_' + variant + '.onnx'
        result = execute_cpu(output / filename, INPUT_NAME, OUTPUT_NAME, fixture, [1, 4, 8400])
        raw_name = 'cpu_actual_' + variant + '.npy'
        np.save(output / raw_name, result)
        arrays[variant + '_output'] = result.copy()
        executions[variant] = {'runs': 1, 'model_file': filename, 'same_fixture_sha256': fixture_lineage['fixture_sha256'],
            'raw_output_file': raw_name, 'raw_output_sha256': sha(output / raw_name), 'summary': summary(result)}
        variants[variant] = {'model_file': filename, 'model_sha256': sha(output / filename),
            'expected_output_npz_keys': {OUTPUT_NAME: variant + '_output'}, 'nodes': 9, 'initializers': 7,
            'selected_node_serialized_sha256': identity(model)['node_serialized_sha256'],
            'selected_initializer_serialized_sha256': identity(model)['initializer_serialized_sha256']}
    np.savez_compressed(output / 'cpu_expected_outputs.npz', **arrays)
    equal = arrays['original_output'].tobytes(order='C') == arrays['code_clip_output'].tobytes(order='C')
    delta = arrays['original_output'].astype(np.float64) - arrays['code_clip_output'].astype(np.float64)
    diagnostic = {'schema': 'rampq.gate05.dfl_context_cpu/v1',
        'status': 'DFL_TAIL_SAME_FIXTURE_CPU_BYTE_EQUIVALENCE_PASSED' if equal else 'DFL_TAIL_CPU_BYTE_EQUIVALENCE_FAILED',
        'ancestor_cpu_executions': 1, 'ancestor_conv_count': 72, 'tail_cpu_executions': 2,
        'complete_detector_output_executions': 0, 'native_executions': 0,
        'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
        'provider': 'CPUExecutionProvider', 'threads': 4, 'graph_optimization': 'ORT_DISABLE_ALL',
        'fixture_lineage_file': 'fixture_lineage.json', 'fixture_sha256': fixture_lineage['fixture_sha256'],
        'executions': executions, 'both_outputs_byte_exact': equal,
        'differing_elements': int(np.count_nonzero(delta)), 'max_absolute_difference': float(np.abs(delta).max()),
        'output_meaning': 'DFL_DISTANCES_NOT_DECODED_BOXES', 'original_complete_model_guard': 'FAILED_UNCHANGED',
        'tolerances_changed': False, 'dataset_evaluation_run': False,
        'limits': protocol['cpu_tail_validation']['proves']}
    write(output / 'cpu_diagnostic.json', diagnostic)
    after = source_hashes(paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'Frozen source changed during fixture or tail CPU execution')
    require(equal, 'DFL tail equality failed; retain failure and do not retry or seal')
    spec = {'schema': 'rampq.gate05.dfl_context_probe/v1',
        'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
        'source_configuration': 'q_all', 'source_graph_file': 'source_model.onnx', 'source_graph_sha256': CORRECTED_SHA,
        'source_original_qall_graph_sha256': ORIGINAL_SHA,
        'parent_spec_file': 'parent_spec.json', 'parent_spec_sha256': CORRECTED_SPEC_SHA,
        'parent_manifest_file': 'parent_manifest.json', 'parent_manifest_sha256': CORRECTED_MANIFEST_SHA,
        'protocol_file': 'protocol.json', 'protocol_sha256': PROTOCOL_SHA,
        'input_file': 'dfl_transpose_fixture.npy', 'input_file_sha256': fixture_lineage['fixture_sha256'],
        'input_binding': {'name': INPUT_NAME, 'input': True, 'shape': [1, 16, 4, 8400], 'dtype': 'DataType.FLOAT'},
        'output_bindings_in_order': [{'name': OUTPUT_NAME, 'input': False, 'shape': [1, 4, 8400],
            'dtype': 'DataType.FLOAT', 'role': 'dfl_distances_not_decoded_boxes', 'kind': 'primary'}],
        'conv_names': CONVS, 'source_conv_names': CONVS, 'clip_names': CLIPS,
        'source_activation_boundaries': [boundary], 'activation_scale': boundary['scale'],
        'variants': {'code_clip': variants['code_clip']}, 'cpu_control_variants': variants,
        'cpu_expected_outputs_file': 'cpu_expected_outputs.npz',
        'cpu_expected_outputs_sha256': sha(output / 'cpu_expected_outputs.npz'),
        'cpu_diagnostic_file': 'cpu_diagnostic.json', 'cpu_diagnostic_sha256': sha(output / 'cpu_diagnostic.json'),
        'graph_identity_audit_file': 'graph_identity_audit.json',
        'graph_identity_audit_sha256': sha(output / 'graph_identity_audit.json'),
        'fixture_lineage_file': 'fixture_lineage.json', 'fixture_lineage_sha256': sha(output / 'fixture_lineage.json'),
        'cases': copy.deepcopy(protocol['cases']), 'case_order': CASE_ORDER,
        'max_seconds': 900, 'max_build_seconds': 300, 'maximum_attempts_per_case': 1,
        'maximum_native_builds': 2, 'maximum_native_inferences': 4, 'executions_per_successful_case': 2,
        'native_importer_clip_attribute_correction': {'variant': 'code_clip', 'layer_names': CLIPS,
            'activation_alpha': -127.0, 'activation_beta': 127.0,
            'when': 'After ONNX parsing; preserve all parser layer precision/output constraints.'},
        'native_build_policy': copy.deepcopy(protocol['common_native_settings']),
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
        'corrected_full_qall_failure': 'FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE_UNCHANGED',
        'retained_qall_failure_checkpoint_sha256': CHECKPOINT_SHA,
        'failure_boundary_inventory_sha256': INVENTORY_SHA,
        'original_tolerances_changed': False, 'calibration_recomputed': False, 'weights_changed': False,
        'dataset_evaluation_run': False, 'complete_detector_executions': 0,
        'ancestor_cpu_executions': 1, 'ancestor_conv_count': 72, 'tail_cpu_executions': 2,
        'native_build_or_execution_run_by_preparer': False, 'limits': protocol['interpretation']}
    write(output / 'spec_draft.json', spec)
    (output / 'README.md').write_text('''# DFL Softmax/量化/卷积/形状转换上下文

从实际失败的修正 q_all 源图提取 images 到 DFL Transpose logits 的精确祖先图，
共638节点、576 initializer、72Conv。使用原固定图片NPY，仅在CPU执行一次，
四线程、ORT_DISABLE_ALL，生成并冻结 FLOAT[1,16,4,8400] 内部fixture。
这是大段backbone/回归头的CPU诊断数据，不是native数据或新的Gate参考。

原始q_all与修正q_all的DFL tail各9节点、7 initializer；所有所选节点与常量
保留各自源图序列化字节。两tail共用上述fixture，各CPU执行一次，必须输出
FLOAT[1,4,8400]字节完全一致；这是DFL distance，不是已解码目标框。
失败不得重选fixture或重跑。没有执行完整检测器输出或数据集。

native只构建code_clip tail，none/obey各一次、300秒，batch900秒，成功后
重复执行两次。只校正一个Clip的原生导入alpha/beta为-127/+127，不设置或
清除解析器精度约束。DFL末端没有输出QDQ，不要求实际卷积输出INT8。
原完整护栏FAILED_UNCHANGED和完整修正q_all构建失败保留，本局部上下文
不能确定完整失败节点、证明完整修复或Gate 0.5通过。
''', encoding='utf-8')
    write(output / 'prepared_content_manifest.json', {'schema': 'rampq.gate05.dfl_context_prepared_content/v1',
        'status': 'PREPARED_DFL_FIXTURE_AND_TAIL_CPU_EQUIVALENT_NOT_SEALED',
        'files': {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}})
    print(json.dumps({'status': 'PREPARED_DFL_FIXTURE_AND_TAIL_CPU_EQUIVALENT_NOT_SEALED',
        'output': str(output), 'source_bytes_unchanged': True,
        'code_clip_model_sha256': variants['code_clip']['model_sha256'],
        'original_model_sha256': variants['original']['model_sha256'],
        'fixture_file_sha256': fixture_lineage['fixture_sha256'], 'fixture': summary(fixture),
        'upstream_graph_sha256': sha(output / upstream_name),
        'cpu_expected_outputs_sha256': spec['cpu_expected_outputs_sha256'],
        'tail_outputs_byte_identical': equal, 'tail_output': executions['code_clip']['summary'],
        'ancestor_cpu_executions': 1, 'ancestor_conv_count': 72, 'tail_cpu_executions': 2,
        'complete_detector_output_executions': 0}, indent=2))


def seal(output, native_paths):
    require(output.is_dir() and not output.with_suffix('.zip').exists() and
            not (output / 'manifest.json').exists() and not (output / 'spec.json').exists() and
            not (output / 'preparation_failure.json').exists(), 'Choose prepared, previously unsealed and successful paths')
    content = read(output / 'prepared_content_manifest.json')
    require(content['status'] == 'PREPARED_DFL_FIXTURE_AND_TAIL_CPU_EQUIVALENT_NOT_SEALED', 'Wrong preparation status')
    for name, expected in content['files'].items():
        path = output / name
        require(path.resolve().parent == output and info(path) == expected, 'Prepared file changed: ' + name)
    sources = read(output / 'source_hashes_before.json')
    require(sources == read(output / 'source_hashes_after.json'), 'Preparation sources drifted')
    validate_recorded_sources(sources)
    require(read(output / 'cpu_diagnostic.json')['both_outputs_byte_exact'] and
            sha(output / 'protocol.json') == PROTOCOL_SHA, 'CPU equality/protocol invalid')
    require(len(native_paths) == 2 and len({path.name for path in native_paths}) == 2 and
            all(path.parent == ROOT / 'scripts' and path.is_file() and not path.is_symlink() for path in native_paths),
            'Native scripts must be separate workspace scripts')
    native_before = source_hashes(native_paths)
    for path in native_paths:
        shutil.copy2(path, output / path.name)
    spec = read(output / 'spec_draft.json')
    spec['native_runner_file'], spec['native_builder_file'] = [path.name for path in native_paths]
    spec['native_runner_sha256'], spec['native_builder_sha256'] = [sha(path) for path in native_paths]
    spec['sealed_client_at'] = datetime.now().astimezone().isoformat()
    write(output / 'spec.json', spec)
    write(output / 'seal_native_source_hashes_before.json', native_before)
    native_after = source_hashes(native_paths)
    write(output / 'seal_native_source_hashes_after.json', native_after)
    require(native_before == native_after, 'Native source changed while sealing')
    validate_recorded_sources(sources)
    files = {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}
    require(len(files) == len(list(output.iterdir())), 'Unexpected artifact subdirectory')
    write(output / 'manifest.json', {'schema': 'rampq.gate05.dfl_context_probe_bundle/v1', 'files': files})
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
                    'ZIP payload differs: ' + path.name)
    print(json.dumps({'status': 'DFL_CONTEXT_CPU_FIXTURE_AND_TAIL_BUNDLE_SEALED', 'output': str(output),
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
    parser.add_argument('--native-runner', type=Path, default=ROOT / 'scripts/jetson_dfl_context_probe.py')
    parser.add_argument('--native-builder', type=Path, default=ROOT / 'scripts/jetson_build_dfl_context_probe.py')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / 'artifacts/gate05').resolve()), 'Unsafe artifact destination')
    if args.prepare_only:
        existed_before = output.exists()
        try:
            prepare(output, args.protocol.resolve())
        except BaseException as exc:
            if not existed_before and output.is_dir() and not (output / 'preparation_failure.json').exists():
                write(output / 'preparation_failure.json', {
                    'schema': 'rampq.gate05.dfl_preparation_failure/v1',
                    'status': 'FAILED_NO_RETRY_OR_FIXTURE_RESELECTION',
                    'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                    'original_complete_model_guard': 'FAILED_UNCHANGED'})
            raise
    else:
        seal(output, [args.native_runner.resolve(), args.native_builder.resolve()])


if __name__ == '__main__':
    main()
