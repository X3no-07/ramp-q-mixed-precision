"""Prepare one explicit-Q/DQ Conv baseline from frozen Gate 0 calibration."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/gate05/20261018_int8/input_v1'
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_CONFIG_DIR'] = str(OUT / '.ultralytics')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort
    import torch
    sys.path.insert(0, str(ROOT / 'src'))
    import gate0

    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    source = ROOT / 'artifacts/jetson_trt_fp16_20261016_v1/model.onnx'
    weights = ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1/best.pt'
    cal_path = ROOT / 'artifacts/calibration/screen.json'
    calibration = json.loads(cal_path.read_text(encoding='utf-8-sig'))
    map_path = ROOT / 'artifacts/gate05/onnx_module_map.json'
    mapping = json.loads(map_path.read_text(encoding='utf-8-sig'))
    assert sha(source) == mapping['onnx_sha256']
    assert sha(weights) == calibration['model_sha256']
    assert set(calibration['activation_max_abs']) == {r['pytorch_module'] for r in mapping['mapping']}
    protocol = {
        'schema': 'rampq.gate05.int8_build_protocol/v1', 'client_date': '2026-10-18',
        'configuration': 'q_all', 'source_weights_sha256': sha(weights), 'source_onnx_sha256': sha(source),
        'calibration_sha256': sha(cal_path), 'module_map_sha256': sha(map_path),
        'calibration_image_list_sha256': calibration['image_list_sha256'],
        'intervention': 'All 88 grouped Conv inputs and weights; not every network operator',
        'activation_quantization': 'Per-tensor symmetric scale=max(frozen_absmax/127,1e-12), zero-point=0',
        'activation_clip': 'Clip to [-127*scale,127*scale] before ONNX INT8 Q to preserve Gate0 narrow range',
        'weight_quantization': 'Per-output-channel symmetric axis=0, scale=max(absmax,1e-12)/127, zero-point=0',
        'conv_output': 'No new output Q; subsequent Conv input Q follows the frozen Gate0 intervention',
        'floating_ops': 'FP16 enabled with FP32 fallback, TF32 disabled; engine inspector must verify every Conv',
        'local_preflight': {'image': '000020.jpg, first existing Selection image',
                            'box_rtol': 0.005, 'box_atol': 0.5, 'score_rtol': 0.02, 'score_atol': 0.002,
                            'ort_graph_optimizations': 'disabled for FP32 Q/DQ comparison'},
        'build_pass': 'Build/deserialization succeeds and all 88 Conv identities map to verified INT8 weighted kernels',
        'execution_pass': 'Real Jetson execution succeeds with finite [1,14,8400] output on the fixed image',
        'not_a_pass_condition': 'INT8 matching FP16 detection count or accuracy; quantization may change detections',
        'limits': 'No dataset evaluation, no new calibration, no recovery or bidirectional claim, no latency benchmark',
        'dataset_gate_decision_rules': 'Still to be frozen before fixed-dataset evaluation',
        'references': ['https://github.com/NVIDIA/TensorRT/issues/2383',
                       'https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html'],
    }
    write(OUT / 'protocol.json', protocol)
    model = gate0.load_fused_model(weights, torch.device('cpu'))
    convs = {name: module for name, module, group in gate0.list_conv_modules(model)}
    graph = onnx.load(str(source))
    initializers = {x.name: x for x in graph.graph.initializer}
    mapping_by_node = {row['onnx_node']: row for row in mapping['mapping']}
    assert len(mapping_by_node) == len(mapping['mapping']) == 88
    new_nodes = []
    new_initializers = []
    rows = []
    for node in graph.graph.node:
        if node.op_type != 'Conv':
            new_nodes.append(copy.deepcopy(node))
            continue
        row = mapping_by_node[node.name]
        module = convs[row['pytorch_module']]
        original_weight = numpy_helper.to_array(initializers[node.input[1]])
        np.testing.assert_array_equal(original_weight, module.weight.detach().numpy())
        if module.bias is not None:
            np.testing.assert_array_equal(numpy_helper.to_array(initializers[node.input[2]]), module.bias.detach().numpy())
        prefix = 'rampq_' + row['pytorch_module'].replace('.', '_')
        activation_scale = np.asarray(max(calibration['activation_max_abs'][row['pytorch_module']] / 127.0, 1e-12), dtype=np.float32)
        weight_scale = (module.weight.detach().abs().amax(dim=tuple(range(1, module.weight.ndim))).clamp_min(1e-12) / 127.0).numpy()
        bound = np.asarray(activation_scale * np.float32(127.0), dtype=np.float32)
        constants = {'a_scale': activation_scale, 'a_zero': np.asarray(0, dtype=np.int8),
                     'clip_min': -bound, 'clip_max': bound, 'w_scale': weight_scale,
                     'w_zero': np.zeros(weight_scale.shape, dtype=np.int8)}
        for suffix, value in constants.items():
            new_initializers.append(numpy_helper.from_array(value, name=prefix + '_' + suffix))
        def name(suffix):
            return prefix + '_' + suffix
        new_nodes.extend([
            helper.make_node('Clip', [node.input[0], name('clip_min'), name('clip_max')], [name('clipped')], name=name('Clip')),
            helper.make_node('QuantizeLinear', [name('clipped'), name('a_scale'), name('a_zero')], [name('a_q')], name=name('ActivationQ')),
            helper.make_node('DequantizeLinear', [name('a_q'), name('a_scale'), name('a_zero')], [name('a_dq')], name=name('ActivationDQ')),
            helper.make_node('QuantizeLinear', [node.input[1], name('w_scale'), name('w_zero')], [name('w_q')], name=name('WeightQ'), axis=0),
            helper.make_node('DequantizeLinear', [name('w_q'), name('w_scale'), name('w_zero')], [name('w_dq')], name=name('WeightDQ'), axis=0),
        ])
        replacement = copy.deepcopy(node)
        replacement.input[0] = name('a_dq')
        replacement.input[1] = name('w_dq')
        new_nodes.append(replacement)
        rows.append(dict(row, activation_scale=float(activation_scale), weight_scale=weight_scale.tolist(),
                         qdq_prefix=prefix, quantized=True))
    del graph.graph.node[:]
    graph.graph.node.extend(new_nodes)
    graph.graph.initializer.extend(new_initializers)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(OUT / 'model_q_all.onnx'))
    write(OUT / 'module_map.json', dict(mapping, mapping=rows, configuration='q_all'))
    (OUT / 'calibration.json').write_bytes(cal_path.read_bytes())

    # Test rounding and narrow-range saturation independently of network accuracy.
    scalar = np.float32(0.125)
    probe = np.asarray([-300, -128, -127.51, -127, -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 127, 128, 300], dtype=np.float32) * scalar
    probe_model = helper.make_model(helper.make_graph([
        helper.make_node('Clip', ['x','lo','hi'], ['c']),
        helper.make_node('QuantizeLinear', ['c','s','z'], ['q']),
        helper.make_node('DequantizeLinear', ['q','s','z'], ['y']),
    ], 'narrow_range_probe', [helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [len(probe)])],
        [helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [len(probe)])],
        [numpy_helper.from_array(np.asarray(v,dtype=d),name=n) for n,v,d in
         [('lo',-127*scalar,np.float32),('hi',127*scalar,np.float32),('s',scalar,np.float32),('z',0,np.int8)]]),
        opset_imports=[helper.make_opsetid('',13)])
    probe_model.ir_version = graph.ir_version
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    observed_probe = ort.InferenceSession(probe_model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider']).run(None, {'x':probe})[0]
    expected_probe = (torch.round(torch.from_numpy(probe)/scalar).clamp(-127,127)*scalar).numpy()
    np.testing.assert_array_equal(observed_probe, expected_probe)

    x, _, _, _, _ = gate0.image_tensor(ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1/000020.jpg', 640, torch.device('cpu'))
    wrappers = gate0.install_qdq_wrappers(model, calibration['activation_max_abs'])
    gate0.enable_configuration(wrappers, 'q_all')
    with torch.inference_mode():
        expected = model(x)
        expected = expected[0] if isinstance(expected, (tuple, list)) else expected
        expected = expected.detach().numpy()
    session = ort.InferenceSession(str(OUT / 'model_q_all.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    actual = session.run(None, {'images':x.numpy()})[0]
    assert expected.shape == actual.shape == (1,14,8400)
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    report = {'scope':'One-image Windows CPU Q/DQ preparation check, not TensorRT hardware evidence',
              'protocol_sha256':sha(OUT/'protocol.json'), 'onnx_sha256':sha(OUT/'model_q_all.onnx'),
              'conv_weights_and_biases_exactly_match_source':True, 'qdq_convs':len(rows),
              'narrow_range_probe':'PASS', 'max_abs_box_difference':float(np.abs(expected[:,:4]-actual[:,:4]).max()),
              'max_abs_score_difference':float(np.abs(expected[:,4:]-actual[:,4:]).max()),
              'image_sha256':sha(ROOT/'artifacts/jetson_fp16_smoke_20261016_v1/000020.jpg'),
              'status':'PENDING'}
    np.savez_compressed(OUT/'local_raw_outputs.npz', pytorch_qdq=expected, onnx_qdq=actual)
    try:
        np.testing.assert_allclose(actual[:,:4],expected[:,:4],rtol=0.005,atol=0.5)
        np.testing.assert_allclose(actual[:,4:],expected[:,4:],rtol=0.02,atol=0.002)
        report['status']='LOCAL_QDQ_PREFLIGHT_PASS'
    except AssertionError as exc:
        report['status']='LOCAL_QDQ_PREFLIGHT_FAILED'
        report['error']=str(exc)
        write(OUT/'preflight.json',report)
        raise
    write(OUT/'preflight.json',report)
    files = sorted(p for p in OUT.iterdir() if p.is_file())
    write(OUT/'manifest.json',{'schema':'rampq.gate05.qdq_inputs/v1','configuration':'q_all',
         'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}})
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
