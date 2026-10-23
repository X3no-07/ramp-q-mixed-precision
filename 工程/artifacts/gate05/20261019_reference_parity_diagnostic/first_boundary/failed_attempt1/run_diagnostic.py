"""Read-only fixed-image CPU Conv/SiLU/Q boundary counterfactuals.

Run once in this new evidence directory. No source graph/model/calibration or
threshold is changed. The float64 convolution is a diagnostic widened-input
reference, not a replacement PyTorch execution or a new acceptance criterion.
"""
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_CONFIG_DIR'] = str(OUT / '.ultralytics')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = OUT / name
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    import ultralytics
    sys.path.insert(0, str(ROOT / 'src'))
    import gate0
    for filename in ('protocol.json', 'raw_outputs.npz', 'analysis.json', 'source_hashes_after.json'):
        assert not (OUT / filename).exists(), 'Evidence must be exclusive: ' + filename
    torch.set_num_threads(4)
    frozen = ROOT / 'artifacts/gate05/20261018_int8/input_v1'
    sources = {
        'model': frozen / 'model_q_all.onnx',
        'map': frozen / 'module_map.json',
        'calibration': frozen / 'calibration.json',
        'preflight': frozen / 'preflight.json',
        'weights': ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1/best.pt',
        'image': ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1/000020.jpg',
        'gate0_source': ROOT / 'src/gate0.py',
        'original_diagnostic_source': ROOT / 'scripts/diagnose_gate05_qdq.py',
        'original_divergence': frozen.parent / 'local_divergence_diagnostic.json',
        'isolated_quantizer_audit': frozen.parent / 'isolated_quantizer_audit.json',
    }
    before = {name: {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
              for name, path in sources.items()}
    protocol = {
        'schema': 'rampq.gate05.first_cpu_boundary_counterfactual/v1',
        'started_client_time': datetime.now().astimezone().isoformat(),
        'scope': 'Original fixed 000020 image, only first Conv and following SiLU/second Conv activation QDQ boundary; CPU only.',
        'source_files': before,
        'script_sha256': sha(Path(__file__)),
        'versions': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                     'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'ultralytics': ultralytics.__version__},
        'execution': {'torch_threads': 4, 'ort_threads': 4, 'provider': 'CPUExecutionProvider',
                      'ort_graph_optimization': 'ORT_DISABLE_ALL'},
        'counterfactuals': [
            'Original exact ancestor prefix of q_all graph ending at second activation DQ.',
            'Same original first QDQ input/weight/bias: PyTorch F.conv2d vs runtime-input ORT Conv.',
            'PyTorch and ORT first Conv outputs each passed through Torch SiLU and ORT Sigmoid/Mul.',
            'Each of four resulting post-SiLU arrays passed through Torch round/clamp/DQ and original ONNX Clip/Q/DQ.',
            'First Conv widened-input float64 F.conv2d as descriptive rounding reference only.'
        ],
        'changes_to_source_or_guard': False, 'new_tolerance_defined': False,
        'calibration_recomputed': False, 'weights_changed': False, 'dataset_evaluation_run': False,
        'native_execution_run': False, 'original_full_model_guard': 'FAILED_UNCHANGED',
        'limits': ['Counterfactuals are diagnostic, not a full-model acceptance check.',
                   'A prefix may change backend execution context; equality with original saved statistics is checked.',
                   'float64 widened-input convolution is not exact real arithmetic and does not decide a pass.'],
    }
    write('protocol.json', protocol)
    graph = onnx.load(str(sources['model']))
    mapping = json.loads(sources['map'].read_text(encoding='utf-8'))['mapping']
    calibration = json.loads(sources['calibration'].read_text(encoding='utf-8'))
    preflight = json.loads(sources['preflight'].read_text(encoding='utf-8'))
    assert preflight['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert sha(sources['weights']) == calibration['model_sha256']
    assert sha(sources['image']) == preflight['image_sha256']
    nodes = {node.name: node for node in graph.graph.node}
    initializers = {value.name: value for value in graph.graph.initializer}
    first, second = mapping[:2]
    assert first['pytorch_module'] == 'model.0.conv' and second['pytorch_module'] == 'model.1.conv'
    p0, p1 = first['qdq_prefix'], second['qdq_prefix']
    conv0 = nodes[first['onnx_node']]
    pre_name = conv0.output[0]
    act_name = nodes['/model.0/act/Mul'].output[0]
    q1_name, dq1_name = p1 + '_a_q', p1 + '_a_dq'
    selected_outputs = [(p0 + '_a_dq', onnx.TensorProto.FLOAT), (p0 + '_w_dq', onnx.TensorProto.FLOAT),
                        (pre_name, onnx.TensorProto.FLOAT), (act_name, onnx.TensorProto.FLOAT),
                        (q1_name, onnx.TensorProto.INT8), (dq1_name, onnx.TensorProto.FLOAT)]
    producers = {output: node for node in graph.graph.node for output in node.output}
    need = {name for name, _ in selected_outputs}
    chosen = set()
    stack = list(need)
    while stack:
        name = stack.pop()
        node = producers.get(name)
        if node is not None and node.name not in chosen:
            chosen.add(node.name)
            stack.extend(node.input)
    prefix = copy.deepcopy(graph)
    del prefix.graph.node[:]
    prefix.graph.node.extend(copy.deepcopy(node) for node in graph.graph.node if node.name in chosen)
    used = {name for node in prefix.graph.node for name in node.input}
    del prefix.graph.initializer[:]
    prefix.graph.initializer.extend(copy.deepcopy(value) for value in graph.graph.initializer if value.name in used)
    del prefix.graph.output[:]
    prefix.graph.output.extend(helper.make_tensor_value_info(name, dtype, None) for name, dtype in selected_outputs)
    del prefix.graph.value_info[:]
    onnx.save(prefix, str(OUT / 'original_prefix.onnx'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    def session(model):
        return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])

    def micro(nodes_, inputs, outputs, name):
        model = helper.make_model(helper.make_graph(nodes_, name, inputs, outputs),
                                  opset_imports=list(graph.opset_import))
        model.ir_version = graph.ir_version
        return model

    x, *_ = gate0.image_tensor(sources['image'], 640, torch.device('cpu'))
    assert x.dtype == torch.float32
    prefix_raw = dict(zip([n for n, _ in selected_outputs], session(prefix).run(None, {'images': x.numpy()})))
    model = gate0.load_fused_model(sources['weights'], torch.device('cpu'))
    first_module = dict(model.named_modules())['model.0.conv']
    act_module = dict(model.named_modules())['model.0.act']
    assert isinstance(act_module, torch.nn.SiLU)
    w = first_module.weight.detach()
    b = first_module.bias.detach()
    np.testing.assert_array_equal(w.numpy(), numpy_helper.to_array(initializers[first['weight_input']]))
    np.testing.assert_array_equal(b.numpy(), numpy_helper.to_array(initializers[conv0.input[2]]))
    s0 = numpy_helper.to_array(initializers[p0 + '_a_scale'])
    s1 = numpy_helper.to_array(initializers[p1 + '_a_scale'])
    wscale = numpy_helper.to_array(initializers[p0 + '_w_scale'])
    torch_xdq = (torch.round(x / float(s0)).clamp(-127, 127) * float(s0)).numpy()
    torch_wdq = (torch.round(w / torch.from_numpy(wscale[:, None, None, None].copy())).clamp(-127, 127)
                 * torch.from_numpy(wscale[:, None, None, None].copy())).numpy()
    np.testing.assert_array_equal(torch_xdq, prefix_raw[p0 + '_a_dq'])
    np.testing.assert_array_equal(torch_wdq, prefix_raw[p0 + '_w_dq'])
    with torch.inference_mode():
        pre_pt = F.conv2d(torch.from_numpy(torch_xdq), torch.from_numpy(torch_wdq), b,
                          first_module.stride, first_module.padding, first_module.dilation, first_module.groups).numpy()
        pre_f64 = F.conv2d(torch.from_numpy(torch_xdq).double(), torch.from_numpy(torch_wdq).double(), b.double(),
                           first_module.stride, first_module.padding, first_module.dilation, first_module.groups).numpy()
    conv_node = copy.deepcopy(conv0)
    del conv_node.input[:]
    conv_node.input.extend(['x', 'w', 'b'])
    del conv_node.output[:]
    conv_node.output.append('y')
    conv_model = micro([conv_node],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(array.shape)) for name, array in
         [('x', torch_xdq), ('w', torch_wdq), ('b', b.numpy())]],
        [helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, list(pre_pt.shape))], 'same_input_runtime_conv')
    onnx.checker.check_model(conv_model)
    onnx.save(conv_model, str(OUT / 'same_input_conv.onnx'))
    pre_ort_replay = session(conv_model).run(None, {'x': torch_xdq, 'w': torch_wdq, 'b': b.numpy()})[0]
    pre_ort = prefix_raw[pre_name]
    shape = list(pre_pt.shape)
    silu_model = micro([helper.make_node('Sigmoid', ['x'], ['sigmoid']), helper.make_node('Mul', ['x', 'sigmoid'], ['y'])],
                       [helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, shape)],
                       [helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, shape)], 'same_input_silu')
    onnx.save(silu_model, str(OUT / 'same_input_silu.onnx'))
    silu_session = session(silu_model)
    quant_model = micro([helper.make_node('Clip', ['x', 'lo', 'hi'], ['clipped']),
                         helper.make_node('QuantizeLinear', ['clipped', 'scale', 'zero'], ['q']),
                         helper.make_node('DequantizeLinear', ['q', 'scale', 'zero'], ['dq'])],
                        [helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, shape),
                         helper.make_tensor_value_info('lo', onnx.TensorProto.FLOAT, []),
                         helper.make_tensor_value_info('hi', onnx.TensorProto.FLOAT, []),
                         helper.make_tensor_value_info('scale', onnx.TensorProto.FLOAT, []),
                         helper.make_tensor_value_info('zero', onnx.TensorProto.INT8, [])],
                        [helper.make_tensor_value_info('q', onnx.TensorProto.INT8, shape),
                         helper.make_tensor_value_info('dq', onnx.TensorProto.FLOAT, shape)], 'same_input_qdq')
    onnx.save(quant_model, str(OUT / 'same_input_qdq.onnx'))
    quant_session = session(quant_model)
    raw = {'input': x.numpy(), 'input0_qdq': torch_xdq, 'weight0_qdq': torch_wdq, 'bias0': b.numpy(),
           'scale0': s0, 'scale1': s1, 'conv_pt': pre_pt, 'conv_ort_prefix': pre_ort,
           'conv_ort_replay': pre_ort_replay, 'conv_widened_f64_reference': pre_f64,
           'prefix_post_silu': prefix_raw[act_name], 'prefix_q1': prefix_raw[q1_name],
           'prefix_dq1': prefix_raw[dq1_name]}

    def diff(a, b):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        return {'shape': list(a.shape), 'different': int(np.count_nonzero(a != b)),
                'max_abs': float(d.max()), 'mean_abs': float(d.mean()), 'rms': float(np.sqrt(np.mean(d*d)))}

    same_quantizer = {}
    for conv_backend, pre in [('pt', pre_pt), ('ort', pre_ort)]:
        with torch.inference_mode():
            post_pt = F.silu(torch.from_numpy(pre.copy())).numpy()
        post_ort = silu_session.run(None, {'x': pre})[0]
        for silu_backend, post in [('pt', post_pt), ('ort', post_ort)]:
            tag = 'conv_' + conv_backend + '_silu_' + silu_backend
            with torch.inference_mode():
                q = torch.round(torch.from_numpy(post.copy()) / float(s1)).clamp(-127, 127).to(torch.int8).numpy()
                dq = q.astype(np.float32) * s1
            bound = np.asarray(s1 * np.float32(127), dtype=np.float32)
            oq, odq = quant_session.run(None, {'x': post, 'lo': -bound, 'hi': bound,
                                             'scale': s1, 'zero': np.asarray(0, dtype=np.int8)})
            raw[tag + '_post'] = post
            raw[tag + '_q_pt'] = q
            raw[tag + '_dq_pt'] = dq
            raw[tag + '_q_ort'] = oq
            raw[tag + '_dq_ort'] = odq
            same_quantizer[tag] = {'integer_codes': diff(q, oq), 'dequantized_values': diff(dq, odq)}
    q_pt = raw['conv_pt_silu_pt_q_pt']
    q_ort = raw['prefix_q1']
    mask = q_pt != q_ort
    coords = np.argwhere(mask)
    raw['actual_q1_disagreement_indices'] = coords
    coordinates = []
    for coord in coords:
        index = tuple(int(i) for i in coord)
        row = {'index_nchw': list(index), 'scale': float(s1), 'pytorch_actual_code': int(q_pt[index]),
               'ort_original_prefix_code': int(q_ort[index]), 'conv_values': {}, 'counterfactuals': {}}
        for tag, a in [('pt', pre_pt), ('ort', pre_ort), ('ort_replay', pre_ort_replay), ('widened_f64', pre_f64)]:
            row['conv_values'][tag] = float(a[index])
        for conv_backend in ('pt', 'ort'):
            for silu_backend in ('pt', 'ort'):
                tag = 'conv_' + conv_backend + '_silu_' + silu_backend
                value = raw[tag + '_post'][index]
                normalized32 = np.float32(value / s1)
                normalized64 = float(value) / float(s1)
                half = float(np.floor(normalized64) + 0.5)
                row['counterfactuals'][tag] = {
                    'post_silu_float32': float(value), 'normalized_float32': float(normalized32),
                    'normalized_float64_division': normalized64, 'nearest_halfstep': half,
                    'signed_distance_to_halfstep_in_code_units': normalized64 - half,
                    'float32_ulp_at_post_silu': float(np.spacing(abs(value))),
                    'torch_code': int(raw[tag + '_q_pt'][index]), 'ort_code': int(raw[tag + '_q_ort'][index]),
                }
        coordinates.append(row)
    counterfactual_code_differences = {}
    for a in ('conv_pt_silu_pt', 'conv_pt_silu_ort', 'conv_ort_silu_pt', 'conv_ort_silu_ort'):
        counterfactual_code_differences[a] = {
            'versus_actual_pytorch': diff(raw[a + '_q_pt'], q_pt),
            'versus_original_ort_prefix': diff(raw[a + '_q_ort'], q_ort),
        }
    original_divergence = json.loads(sources['original_divergence'].read_text(encoding='utf-8'))
    analysis = {
        'schema': 'rampq.gate05.first_cpu_boundary_counterfactual_analysis/v1',
        'status': 'FIRST_BOUNDARY_COUNTERFACTUALS_COMPLETED', 'original_full_model_guard': 'FAILED_UNCHANGED',
        'source_qdq_input_and_weight_bitexact': True,
        'same_input_first_conv_pt_vs_ort': diff(pre_pt, pre_ort_replay),
        'ort_runtime_input_conv_vs_original_prefix': diff(pre_ort_replay, pre_ort),
        'ort_same_input_silu_vs_original_prefix': diff(raw['conv_ort_silu_ort_post'], raw['prefix_post_silu']),
        'ort_same_input_quantizer_vs_original_prefix': diff(raw['conv_ort_silu_ort_q_ort'], q_ort),
        'pt_conv_vs_widened_f64_reference': diff(pre_pt, pre_f64),
        'ort_conv_vs_widened_f64_reference': diff(pre_ort, pre_f64),
        'same_input_silu_pt_vs_ort': {
            'pytorch_conv_source': diff(raw['conv_pt_silu_pt_post'], raw['conv_pt_silu_ort_post']),
            'ort_conv_source': diff(raw['conv_ort_silu_pt_post'], raw['conv_ort_silu_ort_post'])},
        'same_input_quantizer_comparisons': same_quantizer,
        'actual_q1_code_differences': diff(q_pt, q_ort),
        'counterfactual_code_differences': counterfactual_code_differences,
        'all_actual_q1_disagreement_coordinates': coordinates,
        'matches_existing_first_conv_max_error': diff(pre_pt, pre_ort)['max_abs'] == original_divergence[0]['output']['max_abs'],
        'existing_second_conv_input_qdq_disagreements': original_divergence[1]['input_qdq']['unequal'],
        'reproduces_existing_second_conv_input_qdq_count': len(coords) == original_divergence[1]['input_qdq']['unequal'],
        'acceptance_or_tolerance_changed': False,
        'limits': protocol['limits'] + ['This isolates the earliest boundary only; later layers may have additional backend differences.'],
    }
    np.savez_compressed(OUT / 'raw_outputs.npz', **raw)
    after = {name: {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
             for name, path in sources.items()}
    assert before == after, 'A source file changed during read-only experiment'
    analysis['source_files_unchanged'] = True
    analysis['raw_outputs_sha256'] = sha(OUT / 'raw_outputs.npz')
    analysis['actual_q1_disagreement_count'] = len(coords)
    write('source_hashes_after.json', after)
    write('analysis.json', analysis)
    manifest = {'schema': 'rampq.gate05.cpu_boundary_evidence/v1', 'files': {
        path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
        for path in sorted(OUT.iterdir()) if path.is_file() and path.name != 'manifest.json'}}
    write('manifest.json', manifest)
    print(json.dumps({'status': analysis['status'], 'actual_q1_disagreement_count': len(coords),
                      'same_quantizer_different_counts': {k: v['integer_codes']['different'] for k, v in same_quantizer.items()},
                      'ort_conv_replay_vs_prefix': analysis['ort_runtime_input_conv_vs_original_prefix'],
                      'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
