"""Extract the frozen first Conv+SiLU only and save fixed-input CPU expectations.

This does not run either complete detector, import a CUDA runtime, recalibrate,
define a new acceptance threshold, or contact the Jetson. Each selected node and
initializer is copied byte-for-byte from its stated frozen source model.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXED_IMAGE_SHA256 = '494bbd9d09de29f3a992d434da03c252b4f6cf1f28099fd5a443d0c797f5ec52'
ARCHIVE_ROOT = 'gate05_firstconv_numeric_probe_20261019'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def message_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def file_info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify_manifest(folder):
    manifest = read(folder / 'manifest.json')
    paths = [folder / 'manifest.json']
    for name, expected in manifest['files'].items():
        path = folder / name
        require(path.resolve().parent == folder.resolve() and not path.is_symlink(), 'Input escapes frozen folder')
        require(path.is_file() and file_info(path) == expected, 'Frozen source differs: ' + str(path))
        paths.append(path)
    return manifest, paths


def source_hashes(paths):
    return {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): file_info(path)
            for path in sorted(set(paths))}


def first_conv_and_silu(model, module_map, spec):
    convs = [node for node in model.graph.node if node.op_type == 'Conv']
    require(convs, 'Source has no Conv')
    conv = convs[0]  # Actual graph order, then exact mapping; no guessed layer name.
    mappings = [row for row in module_map['mapping'] if row['onnx_node'] == conv.name]
    requests = [row for row in spec['requests'] if row['onnx_node'] == conv.name]
    require(len(mappings) == len(requests) == 1, 'First Conv lacks an exact unique frozen map/request')
    mapping, request = mappings[0], requests[0]
    require(request['module'] == mapping['pytorch_module'] and request['weight_input'] == mapping['weight_input'] and
            request['precision'] == 'INT8' and mapping['quantized'] is True and request['group'] == mapping['group'] == 'G1',
            'First Conv is not the frozen requested G1 INT8 Conv')
    graph_inputs = [value.name for value in model.graph.input]
    require(graph_inputs == [request['original_activation_input']], 'First Conv original input is not the graph input')
    require(len(conv.output) == 1 and len(conv.input) == 3, 'Expected fused Conv with a bias and one output')
    pre = conv.output[0]
    sigmoids = [node for node in model.graph.node if node.op_type == 'Sigmoid' and list(node.input) == [pre]]
    require(len(sigmoids) == 1 and len(sigmoids[0].output) == 1, 'First Conv lacks one direct Sigmoid branch')
    sigmoid = sigmoids[0]
    muls = [node for node in model.graph.node if node.op_type == 'Mul' and
            len(node.input) == 2 and set(node.input) == {pre, sigmoid.output[0]}]
    require(len(muls) == 1 and len(muls[0].output) == 1, 'First Conv lacks the exact Sigmoid*Conv SiLU pattern')
    return conv, sigmoid, muls[0], mapping, request


def extract(model, input_value, output_values, graph_name):
    from onnx import helper
    producers = {}
    by_name = {}
    for node in model.graph.node:
        require(node.name and node.name not in by_name, 'Source node names must be unique')
        by_name[node.name] = node
        for name in node.output:
            require(name not in producers, 'Duplicate source tensor producer')
            producers[name] = node
    constants = {value.name: value for value in model.graph.initializer}
    needed_nodes, needed_constants = set(), set()

    def visit(name):
        if not name or name == input_value.name:
            return
        if name in constants:
            needed_constants.add(name)
            return
        require(name in producers, 'Subgraph has an unresolved external tensor: ' + name)
        node = producers[name]
        if node.name in needed_nodes:
            return
        needed_nodes.add(node.name)
        for source in node.input:
            visit(source)

    for value in output_values:
        visit(value.name)
    selected = [copy.deepcopy(node) for node in model.graph.node if node.name in needed_nodes]
    initializers = [copy.deepcopy(value) for value in model.graph.initializer if value.name in needed_constants]
    available = {input_value.name} | needed_constants
    for node in selected:
        require(set(node.input) - {''} <= available, 'Extracted graph is not topologically closed')
        available.update(node.output)
    require(all(value.name in available for value in output_values), 'An output is absent from the extracted graph')
    probe = copy.deepcopy(model)
    probe.graph.CopyFrom(helper.make_graph(selected, graph_name, [copy.deepcopy(input_value)],
                                          [copy.deepcopy(value) for value in output_values], initializers))
    require(all(node.SerializeToString() == by_name[node.name].SerializeToString() for node in probe.graph.node),
            'A selected node was rewritten')
    require(all(value.SerializeToString() == constants[value.name].SerializeToString() for value in probe.graph.initializer),
            'A selected initializer was rewritten')
    return probe


def array_summary(value):
    import numpy as np
    require(value.dtype == np.float32 and bool(np.isfinite(value).all()), 'CPU output must be finite float32')
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'all_finite': True,
            'min': float(value.min()), 'max': float(value.max()), 'mean': float(value.mean(dtype=np.float64)),
            'rms': float(np.sqrt(np.mean(np.square(value.astype(np.float64))))),
            'raw_float32_bytes_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def difference(left, right):
    import numpy as np
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
            'rms_difference': float(np.sqrt(np.mean(delta * delta))),
            'bit_exact': bool(np.array_equal(left, right)),
            'interpretation': 'Descriptive only; no new pass/fail threshold or full-model conclusion'}


def torch_cpu_check(models, value, expected, selected_conv):
    """Optional independent two small CPU Conv calculations; no detector loading."""
    import numpy as np
    from onnx import helper, numpy_helper
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(4)
    saved, descriptions = {}, {}
    for variant, model in models.items():
        constants = {item.name: numpy_helper.to_array(item).copy() for item in model.graph.initializer}
        produced = {name: node for node in model.graph.node for name in node.output}
        arrays = {next(iter(model.graph.input)).name: value}

        def evaluate(name):
            if name in arrays:
                return arrays[name]
            if name in constants:
                return constants[name]
            node = produced[name]
            inputs = [evaluate(source) for source in node.input]
            if node.op_type == 'Clip':
                output = np.clip(inputs[0], inputs[1], inputs[2])
            elif node.op_type in ('QuantizeLinear', 'DequantizeLinear'):
                attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
                scale, zero = inputs[1], inputs[2]
                if scale.ndim:
                    shape = [1] * inputs[0].ndim
                    shape[attrs.get('axis', 1)] = scale.size
                    scale, zero = scale.reshape(shape), zero.reshape(shape)
                if node.op_type == 'QuantizeLinear':
                    bounds = np.iinfo(inputs[2].dtype)
                    output = np.clip(np.rint(inputs[0] / scale) + zero, bounds.min, bounds.max).astype(inputs[2].dtype)
                else:
                    output = (inputs[0].astype(np.float32) - zero.astype(np.float32)) * scale
            else:
                raise RuntimeError('Only frozen pre-Conv Clip/QDQ may be evaluated here: ' + node.op_type)
            arrays[node.output[0]] = output
            return output

        conv = next(node for node in model.graph.node if node.name == selected_conv.name)
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in conv.attribute}
        pads = attrs.get('pads', [0, 0, 0, 0])
        require(pads[:2] == pads[2:] and attrs.get('auto_pad', b'NOTSET') == b'NOTSET', 'Unsupported Conv padding')
        x = torch.from_numpy(np.array(evaluate(conv.input[0]), dtype=np.float32, copy=True))
        weight = torch.from_numpy(np.array(evaluate(conv.input[1]), dtype=np.float32, copy=True))
        bias = torch.from_numpy(np.array(evaluate(conv.input[2]), dtype=np.float32, copy=True))
        require(all(tensor.device.type == 'cpu' for tensor in (x, weight, bias)), 'CPU tensors required')
        with torch.inference_mode():
            pre = functional.conv2d(x, weight, bias, tuple(attrs.get('strides', [1, 1])), tuple(pads[:2]),
                                    tuple(attrs.get('dilations', [1, 1])), attrs.get('group', 1))
            post = functional.silu(pre)
            post_separate = pre * torch.sigmoid(pre)
        values = {'pre_silu': pre.numpy().copy(), 'post_silu_fused': post.numpy().copy(),
                  'post_silu_sigmoid_mul': post_separate.numpy().copy()}
        descriptions[variant] = {}
        for role, output in values.items():
            key = variant + '_' + role
            saved[key] = output
            ort_key = variant + ('_pre_silu' if role == 'pre_silu' else '_post_silu')
            descriptions[variant][role] = dict(array_summary(output), comparison_to_ort=difference(output, expected[ort_key]))
    return saved, {'torch': torch.__version__, 'device': 'cpu', 'complete_model_loaded_or_run': False,
                   'convolutions_executed': 2, 'outputs': descriptions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/gate05/20261019_firstconv_numeric_probe')
    parser.add_argument('--torch-cpu', action='store_true', help='Also calculate the two isolated Conv+SiLU cases with CPU PyTorch')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Choose a fresh output; preserve prior probes')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort

    bundle = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles_prefer/q_only_G1'
    prior = ROOT / 'artifacts/gate05/20261019_qonly_fixed_image_diagnostic'
    original_image = ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1/000020.jpg'
    fp32_source = ROOT / 'artifacts/jetson_trt_fp16_20261016_v1/model.onnx'
    _bundle_manifest, bundle_paths = verify_manifest(bundle)
    prior_manifest, prior_paths = verify_manifest(prior)
    spec, mapping, diagnostic = read(bundle / 'spec.json'), read(bundle / 'module_map.json'), read(prior / 'diagnostic.json')
    require(spec['configuration'] == 'q_only_G1' and spec['quantized_conv_count'] == 6 and spec['fp16_conv_count'] == 82,
            'Frozen source configuration differs')
    require(sha(fp32_source) == spec['common_source_fp32_onnx_sha256'] == mapping['onnx_sha256'], 'Original FP32 source differs')
    require(sha(bundle / 'model.onnx') == spec['graph_sha256'] == diagnostic['onnx_sha256'], 'Frozen q_only source differs')
    require(sha(bundle / 'module_map.json') == spec['module_map_sha256'], 'Source module mapping differs')
    require(sha(original_image) == diagnostic['image_sha256'] == FIXED_IMAGE_SHA256, 'Fixed image SHA256 differs')
    image_manifest = original_image.parent / 'manifest.json'
    require(read(image_manifest)['files']['000020.jpg']['sha256'] == FIXED_IMAGE_SHA256, 'Original image manifest differs')
    input_source = prior / 'preprocessed_input.npy'
    require(sha(input_source) == prior_manifest['files']['preprocessed_input.npy']['sha256'], 'Frozen preprocessed array differs')
    require(read(prior / 'protocol.json')['existing_tolerances_changed'] is False and
            read(bundle / 'preflight.json')['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED', 'Original failure/scope differs')
    sources = bundle_paths + prior_paths + [fp32_source, original_image, image_manifest, Path(__file__).resolve()]
    before = source_hashes(sources)
    source_models = {'qdq': onnx.load(str(bundle / 'model.onnx')), 'fp32': onnx.load(str(fp32_source))}
    conv, sigmoid, mul, selected_mapping, request = first_conv_and_silu(source_models['qdq'], mapping, spec)
    float_nodes = {node.name: node for node in source_models['fp32'].graph.node}
    require(conv.name in float_nodes and float_nodes[conv.name].op_type == 'Conv', 'Original FP32 source lacks the mapped Conv')
    floating_conv = float_nodes[conv.name]
    require(floating_conv.input[0] == request['original_activation_input'] and
            floating_conv.input[1] == request['weight_input'] and floating_conv.input[2] == conv.input[2] and
            list(floating_conv.output) == list(conv.output), 'Floating first Conv identity differs')
    original_conv_restored = copy.deepcopy(conv)
    original_conv_restored.input[0] = floating_conv.input[0]
    original_conv_restored.input[1] = floating_conv.input[1]
    require(original_conv_restored.SerializeToString() == floating_conv.SerializeToString(),
            'Source Conv differs beyond frozen activation/weight QDQ input edges')
    for node in (sigmoid, mul):
        require(node.name in float_nodes and node.SerializeToString() == float_nodes[node.name].SerializeToString(),
                'Original SiLU node bytes differ')
    inferred = onnx.shape_inference.infer_shapes(source_models['qdq'], strict_mode=True, data_prop=True)
    values = {value.name: value for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)}
    input_value = source_models['qdq'].graph.input[0]
    require([dim.dim_value for dim in input_value.type.tensor_type.shape.dim] == [1, 3, 640, 640] and
            input_value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, 'Unexpected graph input declaration')
    # Same two declarations, in the same order; no new Identity/arithmetic nodes.
    roles = [('post_silu', mul.output[0], 'primary'), ('pre_silu', conv.output[0], 'auxiliary')]
    output_values = [values[name] for _role, name, _meaning in roles]
    for value in output_values:
        require(value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
                [dim.dim_value for dim in value.type.tensor_type.shape.dim] == [1, 16, 320, 320], 'Unexpected first Conv output shape/type')
    models = {variant: extract(model, input_value, output_values, 'exact_first_conv_silu_' + variant)
              for variant, model in source_models.items()}
    require([node.op_type for node in models['qdq'].graph.node] ==
            ['Clip', 'QuantizeLinear', 'DequantizeLinear', 'QuantizeLinear', 'DequantizeLinear', 'Conv', 'Sigmoid', 'Mul'],
            'QDQ probe contains operators beyond the frozen first Conv and SiLU')
    require([node.op_type for node in models['fp32'].graph.node] == ['Conv', 'Sigmoid', 'Mul'], 'FP32 probe is not one exact Conv+SiLU')
    for key in ('qdq', 'fp32'):
        onnx.checker.check_model(models[key], full_check=True)
    require([value.SerializeToString() for value in models['qdq'].graph.output] ==
            [value.SerializeToString() for value in models['fp32'].graph.output], 'Probe output declarations differ')
    qconst = {value.name: value for value in models['qdq'].graph.initializer}
    fconst = {value.name: value for value in models['fp32'].graph.initializer}
    for name, value in fconst.items():
        require(value.SerializeToString() == qconst[name].SerializeToString(), 'Original floating weight/bias bytes differ')
    require(len(qconst) == 8 and len(fconst) == 2, 'Unexpected initializer inventory')
    require(numpy_helper.to_array(qconst[selected_mapping['weight_input']]).shape == (16, 3, 3, 3), 'Unexpected first Conv weight shape')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'source_hashes_before.json', before)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    shutil.copy2(input_source, output / 'preprocessed_input.npy')
    shutil.copy2(bundle / 'preflight.json', output / 'original_failed_preflight.json')
    require((output / 'preprocessed_input.npy').read_bytes() == input_source.read_bytes(), 'Input array copy is not byte-identical')
    x = np.load(output / 'preprocessed_input.npy', allow_pickle=False)
    require(x.shape == (1, 3, 640, 640) and x.dtype == np.float32 and x.flags.c_contiguous and
            bool(np.isfinite(x).all()) and float(x.min()) >= 0 and float(x.max()) <= 1, 'Input is not the frozen finite FP32 tensor')
    expected, cpu_variants, variant_specs = {}, {}, {}
    for variant, model in models.items():
        filename = 'model_firstconv_' + variant + '.onnx'
        onnx.save(model, str(output / filename))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(output / filename), sess_options=options, providers=['CPUExecutionProvider'])
        require(session.get_providers() == ['CPUExecutionProvider'], 'Only CPUExecutionProvider is permitted')
        require([value.name for value in session.get_inputs()] == [input_value.name] and
                [value.name for value in session.get_outputs()] == [name for _role, name, _kind in roles], 'ORT named I/O differs')
        started = time.monotonic()
        results = session.run([name for _role, name, _kind in roles], {input_value.name: x})
        descriptions = []
        for (role, name, kind), value in zip(roles, results):
            key = variant + '_' + role
            require(value.shape == (1, 16, 320, 320), 'Wrong CPU output shape')
            expected[key] = value.copy()
            descriptions.append(dict(array_summary(value), name=name, role=role, kind=kind, npz_key=key))
        cpu_variants[variant] = {'providers': session.get_providers(), 'graph_optimization': 'ORT_DISABLE_ALL',
                                 'cpu_runs': 1, 'elapsed_seconds_not_benchmark': time.monotonic() - started,
                                 'outputs': descriptions}
        source_nodes = {node.name: node for node in source_models[variant].graph.node}
        source_initializers = {value.name: value for value in source_models[variant].graph.initializer}
        variant_specs[variant] = {'model_file': filename, 'model_sha256': sha(output / filename),
                                 'source_graph_sha256': sha(bundle / 'model.onnx') if variant == 'qdq' else sha(fp32_source),
                                 'nodes': len(model.graph.node), 'initializers': len(model.graph.initializer),
                                 'source_nodes_serialized_sha256': {node.name: message_sha(source_nodes[node.name]) for node in model.graph.node},
                                 'source_initializers_serialized_sha256': {value.name: message_sha(source_initializers[value.name])
                                                                          for value in model.graph.initializer},
                                 'expected_output_npz_keys': {name: variant + '_' + role for role, name, _kind in roles}}
        del session
    np.savez_compressed(output / 'cpu_expected_outputs.npz', **expected)
    torch_report = {'status': 'NOT_REQUESTED', 'cuda_used': False}
    if args.torch_cpu:
        torch_arrays, torch_report = torch_cpu_check(models, x, expected, conv)
        np.savez_compressed(output / 'cpu_torch_outputs.npz', **torch_arrays)
        torch_report.update(status='TWO_ISOLATED_CPU_CONVS_EXECUTED', cuda_used=False,
                            saved_outputs_file='cpu_torch_outputs.npz', saved_outputs_sha256=sha(output / 'cpu_torch_outputs.npz'))
    report = {'schema': 'rampq.gate05.firstconv_numeric_cpu_expectations/v1',
              'status': 'ISOLATED_FIRST_CONV_CPU_EXPECTATIONS_SAVED', 'onnx': onnx.__version__,
              'onnxruntime': ort.__version__, 'numpy': np.__version__, 'complete_detector_executions': 0,
              'cuda_used': False, 'variants': cpu_variants, 'optional_torch_cpu': torch_report,
              'qdq_vs_fp32_descriptive_difference': {role: difference(expected['qdq_' + role], expected['fp32_' + role])
                                                     for role, _name, _kind in roles},
              'numerical_pass_criterion_defined': False, 'full_model_numerical_guard': 'FAILED_UNCHANGED'}
    write(output / 'cpu_expectations.json', report)
    probe_spec = {'schema': 'rampq.gate05.firstconv_numeric_probe/v1',
                  'prepared_client_at': datetime.now().astimezone().isoformat(),
                  'diagnostic_only': True, 'source_configuration': 'q_only_G1',
                  'source_bundle_manifest_sha256': sha(bundle / 'manifest.json'),
                  'source_spec_sha256': sha(bundle / 'spec.json'), 'source_module_map_sha256': sha(bundle / 'module_map.json'),
                  'source_fp32_graph_sha256': sha(fp32_source), 'source_qonly_graph_sha256': sha(bundle / 'model.onnx'),
                  'source_input_diagnostic_manifest_sha256': sha(prior / 'manifest.json'),
                  'source_image': '000020.jpg', 'source_image_sha256': FIXED_IMAGE_SHA256,
                  'input_file': 'preprocessed_input.npy', 'input_file_sha256': sha(output / 'preprocessed_input.npy'),
                  'input_array_original_bytes_copied': True,
                  'input_binding': {'name': input_value.name, 'input': True, 'shape': [1, 3, 640, 640], 'dtype': 'DataType.FLOAT'},
                  'output_bindings_in_order': [{'name': name, 'input': False, 'shape': [1, 16, 320, 320],
                                                'dtype': 'DataType.FLOAT', 'role': role, 'kind': kind} for role, name, kind in roles],
                  'selected_conv_name': conv.name, 'selected_silu_node_names': [sigmoid.name, mul.name],
                  'first_conv_source_mapping': selected_mapping,
                  'variants': variant_specs, 'cpu_expected_outputs_file': 'cpu_expected_outputs.npz',
                  'cpu_expected_outputs_sha256': sha(output / 'cpu_expected_outputs.npz'),
                  'cpu_expectations_sha256': sha(output / 'cpu_expectations.json'),
                  'original_failed_preflight_sha256': sha(output / 'original_failed_preflight.json'),
                  'source_nodes_and_initializers_byte_identical': True, 'output_declarations_identical_between_variants': True,
                  'pre_silu_output_tap_may_change_native_fusion': True,
                  'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
                  'calibration_recomputed': False, 'original_tolerances_changed': False,
                  'new_numerical_pass_criterion_defined': False, 'dataset_evaluation_run': False,
                  'native_build_or_execution_run_by_preparer': False,
                  'limits': 'An experimental isolated first Conv+SiLU micrograph, with an extra pre-SiLU output tap. '
                            'It cannot substitute for complete-model numerical migration, actual native layer/tactic '
                            'evidence, detector accuracy, target rescue, causal attribution or performance.'}
    write(output / 'spec.json', probe_spec)
    after = source_hashes(sources)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'A frozen source file changed during preparation')
    (output / 'README.md').write_text('''# 冻结首层 Conv+SiLU 最小数值探针

从 q_only_G1 冻结图的实际首个 Conv 与 module_map 精确定位，经输入/输出依赖识别其 SiLU。
QDQ 图仅含原 Clip、activation Q/DQ、weight Q/DQ、Conv、Sigmoid、Mul 共8节点；
FP32 对照从原始 FP32 图截取对应 Conv、Sigmoid、Mul 共3节点。所有被选节点和
weights/bias/量化常量均保持各自来源中的序列化原字节。未重新校准、未改阈值。

唯一输入为 `preprocessed_input.npy`，逐字节复制先前000020固定图输入，
FLOAT[1,3,640,640]，名字 `images`。两个模型同样按顺序输出：
主输出 post_silu `/model.0/act/Mul_output_0`，辅助 pre_silu
`/model.0/conv/Conv_output_0`，均 FLOAT[1,16,320,320]。
请按tensor名字读取，不依赖原生绑定下标。精确接口、原节点/initializer SHA、
CPU期望结果 NPZ 键位在 `spec.json`。

本地只运行两个小图：CPUExecutionProvider + ORT_DISABLE_ALL。
`cpu_expected_outputs.npz` 保存四个原始 float32 数组，差值只作描述。
可选 PyTorch CPU 也只计算两个小 Conv，并分别保存 fused SiLU 和 Sigmoid*Mul；
不加载或执行完整检测器，不使用 CUDA。

暴露 pre-SiLU 输出可能改变原生融合。这只是首层定位实验，不替代完整模型数值
迁移、strict INT8/FP16格式证明、目标恢复或性能结果。原完整数值guard失败仍保留。
未运行150图评估，未建立新的PASS/FAIL门槛。原生构建及执行由独立runner完成。
''', encoding='utf-8')
    files = {path.name: file_info(path) for path in sorted(output.iterdir()) if path.is_file()}
    write(output / 'manifest.json', {'schema': 'rampq.gate05.firstconv_numeric_probe_bundle/v1', 'files': files})
    (output / 'SHA256SUMS').write_text(''.join(info['sha256'] + '  ' + name + '\n' for name, info in files.items()), encoding='utf-8')
    with zipfile.ZipFile(output.with_suffix('.zip'), 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, ARCHIVE_ROOT + '/' + path.name)
    print(json.dumps({'status': report['status'], 'output': str(output), 'archive': str(output.with_suffix('.zip')),
                      'archive_sha256': sha(output.with_suffix('.zip')), 'spec_sha256': sha(output / 'spec.json'),
                      'selected_conv_name': conv.name, 'selected_silu_node_names': [sigmoid.name, mul.name],
                      'variants': {name: {key: value for key, value in row.items() if key in
                                         ('model_file', 'model_sha256', 'nodes', 'initializers', 'expected_output_npz_keys')}
                                   for name, row in variant_specs.items()},
                      'input_array_sha256': sha(output / 'preprocessed_input.npy'), 'source_hashes_unchanged': True,
                      'complete_detector_executions': 0, 'native_execution': False,
                      'cpu_descriptive_difference': report['qdq_vs_fp32_descriptive_difference']}, indent=2))


if __name__ == '__main__':
    main()
