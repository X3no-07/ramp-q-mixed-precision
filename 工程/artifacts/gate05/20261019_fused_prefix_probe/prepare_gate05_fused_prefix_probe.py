"""Prepare a CPU-verified single-output Conv/SiLU/output-QDQ prefix pair.

The original variant retains the exact selected source nodes and initializers.
The code_clip variant moves only the two activation Clips across their Q nodes.
This preparer never connects to a Jetson, uses CUDA, or executes a full detector.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GRAPH_SHA256 = '946fe83ea185fde114096f9d7b64b5ec9d4bc69afc87476b9c7548cf97a04b91'
FIXED_INPUT_SHA256 = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
ARCHIVE_ROOT = 'gate05_fused_prefix_probe_20261019'


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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def frozen_files(folder):
    manifest = read(folder / 'manifest.json')
    paths = [folder / 'manifest.json']
    for name, expected in manifest['files'].items():
        path = folder / name
        require(path.resolve().parent == folder.resolve() and not path.is_symlink() and info(path) == expected,
                'Frozen source differs: ' + str(path))
        paths.append(path)
    return paths


def source_hashes(paths):
    return {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): info(path)
            for path in sorted(set(paths))}


def summary(value):
    import numpy as np
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'all_finite': bool(np.isfinite(value).all()),
            'min': float(value.min()), 'max': float(value.max()),
            'raw_array_bytes_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/gate05/20261019_fused_prefix_probe')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() and not output.with_suffix('.zip').exists(),
            'Choose a fresh output; preserve previous evidence')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort

    bundle = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles_prefer/q_only_G1'
    input_bundle = ROOT / 'artifacts/gate05/20261019_firstconv_numeric_probe'
    source_paths = frozen_files(bundle) + frozen_files(input_bundle) + [Path(__file__).resolve()]
    before = source_hashes(source_paths)
    source_spec = read(bundle / 'spec.json')
    mapping = read(bundle / 'module_map.json')
    require(source_spec['configuration'] == 'q_only_G1' and
            source_spec['original_local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED',
            'Unexpected original configuration or full-model guard')
    require(sha(bundle / 'model.onnx') == SOURCE_GRAPH_SHA256 == source_spec['graph_sha256'],
            'Frozen original q_only_G1 graph SHA differs')
    require(sha(input_bundle / 'preprocessed_input.npy') == FIXED_INPUT_SHA256,
            'Frozen 000020 preprocessed input SHA differs')
    require(sha(bundle / 'preflight.json') == source_spec['failed_preflight_sha256'],
            'Frozen full-model failed preflight SHA differs')
    source = onnx.load(str(bundle / 'model.onnx'))
    producers = {name: node for node in source.graph.node for name in node.output}
    source_nodes = {node.name: node for node in source.graph.node}
    initializers = {value.name: value for value in source.graph.initializer}
    convs = [node for node in source.graph.node if node.op_type == 'Conv']
    conv, excluded_conv = convs[:2]
    require(conv.name == '/model.0/conv/Conv' and excluded_conv.name == '/model.1/conv/Conv',
            'Frozen first and second Conv identities differ')
    for selected_conv in (conv, excluded_conv):
        mapped = [row for row in mapping['mapping'] if row['onnx_node'] == selected_conv.name]
        requests = [row for row in source_spec['requests'] if row['onnx_node'] == selected_conv.name]
        require(len(mapped) == len(requests) == 1 and mapped[0]['quantized'] is True and
                mapped[0]['group'] == requests[0]['group'] == 'G1' and requests[0]['precision'] == 'INT8',
                'Selected source Conv is not the mapped quantized G1 Conv')
    input_value = copy.deepcopy(source.graph.input[0])
    require(len(source.graph.input) == 1 and input_value.name == 'images' and
            input_value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
            [dim.dim_value for dim in input_value.type.tensor_type.shape.dim] == [1, 3, 640, 640],
            'Frozen input binding differs')
    boundaries = []
    boundary_metadata = []
    for selected_conv in (conv, excluded_conv):
        dq = producers[selected_conv.input[0]]
        require(dq.op_type == 'DequantizeLinear' and len(dq.input) == 3, 'Activation DQ missing')
        q = producers[dq.input[0]]
        require(q.op_type == 'QuantizeLinear' and len(q.input) == 3, 'Activation Q missing')
        clip = producers[q.input[0]]
        require(clip.op_type == 'Clip' and len(clip.input) == 3, 'Activation Float Clip missing')
        require(list(q.input[1:]) == list(dq.input[1:]) and not q.attribute and not dq.attribute and not clip.attribute,
                'Unexpected scalar Q/DQ or Clip contract')
        scale = numpy_helper.to_array(initializers[q.input[1]])
        zero = numpy_helper.to_array(initializers[q.input[2]])
        lower = numpy_helper.to_array(initializers[clip.input[1]])
        upper = numpy_helper.to_array(initializers[clip.input[2]])
        require(scale.dtype == np.float32 and scale.ndim == 0 and float(scale) > 0 and
                zero.dtype == np.int8 and zero.ndim == 0 and int(zero) == 0,
                'Expected positive FLOAT scalar scale and INT8 zero=0')
        require(lower.dtype == upper.dtype == np.float32 and lower.ndim == upper.ndim == 0 and
                float(lower) == -float(upper) and
                np.float32(scale * np.float32(127)).tobytes() == upper.tobytes() and
                np.float32(scale * np.float32(-127)).tobytes() == lower.tobytes() and
                float(np.float32(upper / scale)) == 127 and float(np.float32(lower / scale)) == -127,
                'Frozen Clip bounds do not match the symmetric signed 127 scale interval')
        boundaries.append((clip, q, dq))
        boundary_metadata.append({'consumer_conv_name': selected_conv.name, 'clip_name': clip.name,
                                  'q_name': q.name, 'dq_name': dq.name, 'clip_float_input_name': clip.input[0],
                                  'dq_output_name': dq.output[0], 'scale_initializer': q.input[1],
                                  'zero_initializer': q.input[2], 'scale_float32': float(scale), 'zero_int8': int(zero),
                                  'original_float_clip_bounds': [float(lower), float(upper)],
                                  'original_float_clip_bounds_times_scale_check': 'EXACT_FLOAT32_127_TIMES_SCALE',
                                  'signed_symmetric_quantization_range': [-127, 127],
                                  'storage_dtype_range': [-128, 127], 'code_clip_bounds': [-127, 127],
                                  'swish_output_int8_scale': float(scale) if selected_conv.name == excluded_conv.name else None})
    require(boundaries[0][0].input[0] == input_value.name, 'First Float Clip input is not original images')
    second_dq = boundaries[1][2]
    output_value = helper.make_tensor_value_info(second_dq.output[0], onnx.TensorProto.FLOAT, [1, 16, 320, 320])
    # Walk backwards from the sole output to select dependencies, then retain
    # their original relative order. The second Conv cannot be a dependency.
    needed_nodes = set()
    frontier = [output_value.name]
    while frontier:
        tensor_name = frontier.pop()
        if tensor_name in producers:
            node = producers[tensor_name]
            if node.name not in needed_nodes:
                needed_nodes.add(node.name)
                frontier.extend(name for name in node.input if name)
        else:
            require(tensor_name in initializers or tensor_name == input_value.name,
                    'Unexpected external prefix dependency: ' + tensor_name)
    selected_nodes = [copy.deepcopy(node) for node in source.graph.node if node.name in needed_nodes]
    require(len(selected_nodes) == 11 and [node.op_type for node in selected_nodes] ==
            ['Clip', 'QuantizeLinear', 'DequantizeLinear', 'QuantizeLinear', 'DequantizeLinear',
             'Conv', 'Sigmoid', 'Mul', 'Clip', 'QuantizeLinear', 'DequantizeLinear'] and
            excluded_conv.name not in needed_nodes, 'Unexpected exact prefix node inventory')
    selected_constants = {name for node in selected_nodes for name in node.input if name in initializers}
    constants = [copy.deepcopy(value) for value in source.graph.initializer if value.name in selected_constants]
    require(len(constants) == 12, 'Unexpected selected initializer inventory')
    require(selected_nodes[6].input[0] == conv.output[0] and
            list(selected_nodes[7].input) == [conv.output[0], selected_nodes[6].output[0]] and
            boundaries[1][0].input[0] == selected_nodes[7].output[0], 'Original SiLU path differs')
    original = copy.deepcopy(source)
    original.graph.CopyFrom(helper.make_graph(selected_nodes, 'original_fused_prefix_single_dq_output',
                                             [input_value], [output_value], constants))
    onnx.checker.check_model(original, full_check=True)
    require(all(node.SerializeToString() == source_nodes[node.name].SerializeToString()
                for node in original.graph.node), 'Original variant node bytes changed')
    require(all(value.SerializeToString() == initializers[value.name].SerializeToString()
                for value in original.graph.initializer), 'Original variant initializer bytes changed')
    transformed = copy.deepcopy(original)
    rewrites = []
    patched_nodes = {node.name: node for node in transformed.graph.node}
    patched_constants = {value.name: value for value in transformed.graph.initializer}
    changed_node_names = set()
    changed_constant_names = set()
    for clip, q, dq in boundaries:
        new_clip, new_q = patched_nodes[clip.name], patched_nodes[q.name]
        new_q_tensor = q.output[0] + '_before_code_clip'
        require(new_q_tensor not in producers and new_q_tensor not in initializers, 'New code tensor name collision')
        new_q.input[0] = clip.input[0]
        new_q.output[0] = new_q_tensor
        new_clip.input[0] = new_q_tensor
        new_clip.output[0] = q.output[0]
        changes = []
        for constant_name, bound in zip(clip.input[1:], (-127, 127)):
            old = initializers[constant_name]
            patched_constants[constant_name].CopyFrom(numpy_helper.from_array(np.asarray(bound, dtype=np.int8), constant_name))
            changed_constant_names.add(constant_name)
            changes.append({'name': constant_name, 'source_dtype': 'float32',
                            'source_value': float(numpy_helper.to_array(old)), 'candidate_dtype': 'int8',
                            'candidate_value': bound, 'source_serialized_sha256': message_sha(old),
                            'candidate_serialized_sha256': message_sha(patched_constants[constant_name])})
        changed_node_names.update((clip.name, q.name))
        rewrites.append({'clip_name': clip.name, 'q_name': q.name, 'dq_name_unchanged': dq.name,
                         'original_order': [clip.name, q.name, dq.name], 'candidate_order': [q.name, clip.name, dq.name],
                         'q_input_0': {'source': q.input[0], 'candidate': new_q.input[0]},
                         'q_output_0': {'source': q.output[0], 'candidate': new_q.output[0]},
                         'clip_input_0': {'source': clip.input[0], 'candidate': new_clip.input[0]},
                         'clip_output_0': {'source': clip.output[0], 'candidate': new_clip.output[0]},
                         'clip_bound_inputs_unchanged': list(clip.input[1:]), 'bound_initializer_changes': changes,
                         'source_clip_serialized_sha256': message_sha(clip), 'source_q_serialized_sha256': message_sha(q),
                         'candidate_clip_serialized_sha256': message_sha(new_clip),
                         'candidate_q_serialized_sha256': message_sha(new_q)})
    reordered = []
    clip_to_q = {clip.name: q.name for clip, q, _ in boundaries}
    q_names = set(clip_to_q.values())
    for node in original.graph.node:
        if node.name in clip_to_q:
            reordered.extend((copy.deepcopy(patched_nodes[clip_to_q[node.name]]), copy.deepcopy(patched_nodes[node.name])))
        elif node.name not in q_names:
            reordered.append(copy.deepcopy(patched_nodes[node.name]))
    del transformed.graph.node[:]
    transformed.graph.node.extend(reordered)
    transformed.graph.name = 'code_clip_fused_prefix_single_dq_output'
    for node in transformed.graph.node:
        if node.name not in changed_node_names:
            require(node.SerializeToString() == source_nodes[node.name].SerializeToString(), 'Unapproved node change')
        else:
            expected = copy.deepcopy(source_nodes[node.name])
            for clip, q, _ in boundaries:
                if node.name == q.name:
                    expected.input[0] = clip.input[0]
                    expected.output[0] = q.output[0] + '_before_code_clip'
                if node.name == clip.name:
                    expected.input[0] = q.output[0] + '_before_code_clip'
                    expected.output[0] = q.output[0]
            require(node.SerializeToString() == expected.SerializeToString(), 'Unexpected Clip/Q node mutation')
    require(all(value.SerializeToString() == initializers[value.name].SerializeToString()
                for value in transformed.graph.initializer if value.name not in changed_constant_names),
            'Unapproved initializer change')
    require(len(changed_node_names) == len(changed_constant_names) == 4 and
            len(transformed.graph.node) == len(original.graph.node) and
            len(transformed.graph.initializer) == len(original.graph.initializer), 'Candidate inventory changed unexpectedly')
    require(original.graph.output[0].SerializeToString() == transformed.graph.output[0].SerializeToString() and
            len(transformed.graph.output) == len(original.graph.output) == 1,
            'Both variants must have the same sole original second ActivationDQ output')
    onnx.checker.check_model(transformed, full_check=True)

    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    shutil.copy2(input_bundle / 'preprocessed_input.npy', output / 'preprocessed_input.npy')
    shutil.copy2(bundle / 'preflight.json', output / 'original_failed_preflight.json')
    write(output / 'source_hashes_before.json', before)
    x = np.load(output / 'preprocessed_input.npy', allow_pickle=False)
    require(sha(output / 'preprocessed_input.npy') == FIXED_INPUT_SHA256 and x.dtype == np.float32 and
            x.shape == (1, 3, 640, 640) and x.flags.c_contiguous and bool(np.isfinite(x).all()) and
            float(x.min()) >= 0 and float(x.max()) <= 1, 'Frozen copied input is invalid')
    models = {'original': original, 'code_clip': transformed}
    arrays, variants, execution = {}, {}, {}
    for variant, model in models.items():
        filename = 'model_fused_prefix_' + variant + '.onnx'
        onnx.save(model, str(output / filename))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(output / filename), sess_options=options, providers=['CPUExecutionProvider'])
        require(session.get_providers() == ['CPUExecutionProvider'] and
                [value.name for value in session.get_inputs()] == [input_value.name] and
                [value.name for value in session.get_outputs()] == [output_value.name], 'CPU ORT I/O or provider differs')
        result = session.run([output_value.name], {input_value.name: x})[0]
        raw_filename = 'cpu_actual_' + variant + '.npy'
        np.save(output / raw_filename, result)
        arrays[variant + '_output'] = result.copy()
        execution[variant] = {'providers': session.get_providers(), 'graph_optimization': 'ORT_DISABLE_ALL',
                              'runs': 1, 'raw_result_file': raw_filename, 'raw_result_sha256': sha(output / raw_filename),
                              'output': summary(result)}
        variants[variant] = {'model_file': filename, 'model_sha256': sha(output / filename),
                             'nodes': len(model.graph.node), 'initializers': len(model.graph.initializer),
                             'expected_output_npz_keys': {output_value.name: variant + '_output'},
                             'selected_node_serialized_sha256': {node.name: message_sha(node) for node in model.graph.node},
                             'selected_initializer_serialized_sha256': {value.name: message_sha(value) for value in model.graph.initializer}}
        del session
    np.savez_compressed(output / 'cpu_expected_outputs.npz', **arrays)
    difference = arrays['original_output'].astype(np.float64) - arrays['code_clip_output'].astype(np.float64)
    checks = {'original_selected_nodes_exact_source_bytes': True,
              'original_selected_initializers_exact_source_bytes': True,
              'candidate_only_two_activation_clip_q_rewrites': True,
              'candidate_dq_scale_zero_weights_bias_conv_silu_exact_source_bytes': True,
              'both_source_clip_bounds_exact_float32_symmetric_127_times_scale': True,
              'single_original_second_activation_dq_float_output_no_extra_taps': True,
              'second_conv_excluded': excluded_conv.name not in needed_nodes,
              'cpu_outputs_bit_exact': arrays['original_output'].tobytes(order='C') == arrays['code_clip_output'].tobytes(order='C'),
              'all_cpu_outputs_float32_shape_finite': all(value.dtype == np.float32 and value.shape == (1, 16, 320, 320) and
                                                        bool(np.isfinite(value).all()) for value in arrays.values())}
    diagnostic = {'schema': 'rampq.gate05.fused_prefix_cpu/v1',
                  'status': 'ISOLATED_FUSED_PREFIX_CPU_EXACT_CHECKS_PASSED' if all(checks.values()) else 'ISOLATED_FUSED_PREFIX_CPU_REVIEW_REQUIRED',
                  'checks': checks, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
                  'fixed_input': summary(x), 'source_activation_boundaries': boundary_metadata, 'cpu_execution': execution,
                  'pair_max_absolute_error': float(np.abs(difference).max()),
                  'pair_differing_element_count': int(np.count_nonzero(difference)),
                  'complete_detector_executions': 0, 'native_execution': False, 'cuda_used': False,
                  'full_model_numerical_guard': 'FAILED_UNCHANGED',
                  'limits': 'Single frozen input and isolated CPU prefix only; native SWISH/INT8 fusion is untested by this preparer.'}
    write(output / 'cpu_diagnostic.json', diagnostic)
    spec = {'schema': 'rampq.gate05.fused_prefix_probe/v1',
            'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
            'source_configuration': 'q_only_G1', 'source_graph_sha256': SOURCE_GRAPH_SHA256,
            'source_bundle_manifest_sha256': sha(bundle / 'manifest.json'), 'source_spec_sha256': sha(bundle / 'spec.json'),
            'source_module_map_sha256': sha(bundle / 'module_map.json'),
            'source_input_bundle_manifest_sha256': sha(input_bundle / 'manifest.json'), 'source_image': '000020.jpg',
            'input_file': 'preprocessed_input.npy', 'input_file_sha256': FIXED_INPUT_SHA256,
            'input_array_original_bytes_copied': True,
            'input_binding': {'name': input_value.name, 'input': True, 'shape': [1, 3, 640, 640], 'dtype': 'DataType.FLOAT'},
            'output_bindings_in_order': [{'name': output_value.name, 'input': False, 'shape': [1, 16, 320, 320],
                                          'dtype': 'DataType.FLOAT', 'role': 'second_conv_dequantized_activation', 'kind': 'primary'}],
            'conv_name': conv.name, 'excluded_second_conv_name': excluded_conv.name,
            'clip_names': [clip.name for clip, _, _ in boundaries],
            'source_activation_boundaries': boundary_metadata,
            'source_second_activation_scale': boundary_metadata[1]['scale_float32'],
            'source_second_activation_quantization_range': [-127, 127],
            'source_second_activation_float_clip_bounds': boundary_metadata[1]['original_float_clip_bounds'],
            'source_second_activation_q_name': boundaries[1][1].name,
            'source_second_activation_dq_name': second_dq.name,
            'source_second_activation_dq_tensor_name': output_value.name,
            'source_second_activation_scale_initializer': boundaries[1][1].input[1],
            'variants': variants, 'cpu_expected_outputs_file': 'cpu_expected_outputs.npz',
            'cpu_expected_outputs_sha256': sha(output / 'cpu_expected_outputs.npz'),
            'cpu_diagnostic_sha256': sha(output / 'cpu_diagnostic.json'),
            'source_nodes_serialized_sha256': {node.name: message_sha(node) for node in selected_nodes},
            'source_initializers_serialized_sha256': {value.name: message_sha(value) for value in constants},
            'code_clip_explicit_changes': rewrites,
            'original_variant_nodes_and_initializers_exact_source_bytes': True,
            'candidate_dq_scale_zero_weights_bias_conv_silu_bytes_unchanged': True,
            'single_original_float_dq_output_no_extra_pre_or_post_taps': True,
            'requires_explicit_native_importer_clip_attribute_correction': True,
            'native_importer_clip_attribute_correction': {'variant': 'code_clip',
                'layer_names': [clip.name for clip, _, _ in boundaries], 'activation_alpha': -127.0, 'activation_beta': 127.0,
                'when': 'After native ONNX parsing and before build; assert both original named Clip activation layers found.',
                'reason': 'TRT 8.5.2.2 importer previously misread INT8 Clip constant values; native runner must record parsed and corrected attributes.'},
            'native_fusion_check_intent': 'Allow original first Conv+SiLU and second input Q/DQ fusion with no pre/post intermediate taps; inspect INT8 SWISH output using unchanged second activation scale.',
            'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
            'calibration_recomputed': False, 'original_tolerances_changed': False, 'weights_changed': False,
            'dataset_evaluation_run': False, 'complete_detector_executions': 0,
            'native_build_or_execution_run_by_preparer': False,
            'limits': 'Isolated original-vs-code-Clip prefix on one frozen image only. Native fusion, full-model numerical migration, strict arithmetic, detector accuracy, and performance remain unproven.'}
    write(output / 'spec.json', spec)
    after = source_hashes(source_paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'A frozen source file changed')
    (output / 'README.md').write_text('''# 首 Conv / SiLU / 第二层 activation QDQ 融合前缀探针

从冻结 exact q_only_G1 图抽取 11 个节点，截止原第二 Conv 输入 ActivationDQ；
不含第二 Conv。原图 images FLOAT[1,3,640,640] 为唯一输入，原 tensor
`rampq_model_1_conv_a_dq` FLOAT[1,16,320,320] 为唯一输出，没有额外前后 tap。
original 逐个原样复制所选源节点与 initializer 的序列化字节。

code_clip 只把两处 Float Clip -> Q 改为 Q -> INT8 Clip[-127,127]，DQ 不动。
保留 Clip/Q 名字，Q 输出换为唯一新 tensor，Clip 输出接回原 Q tensor；
四个 Clip bound initializer 保留名字并改为 INT8 -127/+127。所有显式改线、
常量原值/新值和原/新序列化 SHA 记录在 spec.json。DQ、scale、zero、权重、
bias、Conv 和原 Sigmoid/Mul SiLU 节点逐字节不变。已核对两处原边界等于
float32(scale * -127/+127)，原第二 scale 和范围也在 spec 中单列。

固定 000020 输入直接按原 NPY 字节复制。两图各用一次 CPUExecutionProvider、
ORT_DISABLE_ALL，保留原始输出后核对原字节完全相同；输出期望 NPZ 的按名字
映射在 spec 中。CPU 结果不证明 TensorRT 融合行为。

后续 native runner 必须在 ONNX 解析后把 code_clip 两个原命名 Clip activation
的 alpha/beta 显式核对并修正为 -127/+127，同时保留解析前后数值证据。
本准备器只在本地 CPU 执行，不连接 Orin、不运行 CUDA、不跑完整检测器/数据集，
不改变原完整候选、标定、权重和阈值。原完整 PyTorch/ORT 护栏 FAILED_UNCHANGED。
''', encoding='utf-8')
    files = {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}
    write(output / 'manifest.json', {'schema': 'rampq.gate05.fused_prefix_probe_bundle/v1', 'files': files})
    checksum_files = dict(files)
    checksum_files['manifest.json'] = info(output / 'manifest.json')
    (output / 'SHA256SUMS').write_text(''.join(value['sha256'] + '  ' + name + '\n'
                                            for name, value in sorted(checksum_files.items())), encoding='utf-8')
    with zipfile.ZipFile(output.with_suffix('.zip'), 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            require(path.is_file(), 'Unexpected output directory entry')
            archive.write(path, ARCHIVE_ROOT + '/' + path.name)
    require(all(checks.values()), 'CPU exact checks failed; raw evidence retained')
    print(json.dumps({'status': diagnostic['status'], 'output': str(output),
                      'archive': str(output.with_suffix('.zip')), 'archive_bytes': output.with_suffix('.zip').stat().st_size,
                      'archive_sha256': sha(output.with_suffix('.zip')), 'spec_sha256': sha(output / 'spec.json'),
                      'input_binding': spec['input_binding'], 'output_bindings_in_order': spec['output_bindings_in_order'],
                      'conv_name': spec['conv_name'], 'clip_names': spec['clip_names'],
                      'source_second_activation_scale': spec['source_second_activation_scale'],
                      'variants': {name: {key: value[key] for key in ('model_file', 'model_sha256', 'expected_output_npz_keys')}
                                   for name, value in variants.items()},
                      'checks': checks, 'all_source_hashes_unchanged': True, 'complete_detector_executions': 0,
                      'native_execution': False}, indent=2))


if __name__ == '__main__':
    main()
