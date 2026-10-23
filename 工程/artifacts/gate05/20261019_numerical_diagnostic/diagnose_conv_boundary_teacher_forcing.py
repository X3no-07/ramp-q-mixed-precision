"""Fixed-image causal numerical diagnostic; never replaces the failed preflight.

Inject ONNX Runtime Conv outputs into the existing PyTorch Q/DQ model to isolate
floating Conv boundary drift from the intervening non-Conv graph operations.
This is a diagnostic intervention, not an alternate quantization configuration.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_CONFIG_DIR'] = str(OUT / '.ultralytics')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import helper
    sys.path.insert(0, str(ROOT / 'src'))
    import gate0
    torch.set_num_threads(4)
    frozen = ROOT / 'artifacts/gate05/20261018_int8/input_v1'
    paths = {name: frozen / name for name in
             ('model_q_all.onnx', 'module_map.json', 'calibration.json', 'preflight.json',
              'local_raw_outputs.npz', 'protocol.json')}
    paths['weights'] = ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1/best.pt'
    paths['image'] = ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1/000020.jpg'
    hashes_before = {name: sha(path) for name, path in paths.items()}
    preflight = json.loads(paths['preflight.json'].read_text(encoding='utf-8'))
    protocol = json.loads(paths['protocol.json'].read_text(encoding='utf-8'))
    assert preflight['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert hashes_before['model_q_all.onnx'] == preflight['onnx_sha256']
    assert hashes_before['image'] == preflight['image_sha256']
    mapping = json.loads(paths['module_map.json'].read_text(encoding='utf-8'))['mapping']
    calibration = json.loads(paths['calibration.json'].read_text(encoding='utf-8'))
    assert len(mapping) == 88
    graph = onnx.load(str(paths['model_q_all.onnx']))
    nodes = {node.name: node for node in graph.graph.node}
    producers = {name: node for node in graph.graph.node for name in node.output}
    primary_name = graph.graph.output[0].name
    references = {}
    for row in mapping:
        conv = nodes[row['onnx_node']]
        dq = producers[conv.input[0]]
        quantize = producers[dq.input[0]]
        clip = producers[quantize.input[0]]
        assert (dq.op_type, quantize.op_type, clip.op_type) == ('DequantizeLinear', 'QuantizeLinear', 'Clip')
        references[row['pytorch_module']] = {'conv_output': conv.output[0], 'preclip_input': clip.input[0],
                                             'quantized_input': conv.input[0]}
    names = list(dict.fromkeys([primary_name] + [name for row in references.values() for name in row.values()]))
    existing = {value.name for value in graph.graph.output}
    for name in names:
        if name not in existing:
            graph.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    x, *_ = gate0.image_tensor(paths['image'], 640, torch.device('cpu'))
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    observed = dict(zip(names, session.run(names, {'images': x.numpy()})))
    saved = np.load(paths['local_raw_outputs.npz'], allow_pickle=False)

    def statistics(actual, expected):
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        return {'elements': int(actual.size), 'unequal': int(np.count_nonzero(difference)),
                'max_abs': float(difference.max()), 'mean_abs': float(difference.mean())}

    def frozen_guard(actual, expected):
        box = protocol['local_preflight']
        return {'boxes_within_original_guard': bool(np.allclose(actual[:, :4], expected[:, :4],
                    rtol=box['box_rtol'], atol=box['box_atol'])),
                'scores_within_original_guard': bool(np.allclose(actual[:, 4:], expected[:, 4:],
                    rtol=box['score_rtol'], atol=box['score_atol'])),
                'boxes': statistics(actual[:, :4], expected[:, :4]),
                'scores': statistics(actual[:, 4:], expected[:, 4:])}

    model = gate0.load_fused_model(paths['weights'], torch.device('cpu'))
    wrappers = gate0.install_qdq_wrappers(model, calibration['activation_max_abs'])
    gate0.enable_configuration(wrappers, 'q_all')
    trials = []
    first_name = mapping[0]['pytorch_module']
    raw_outputs = {'onnx_instrumented': observed[primary_name],
                   'onnx_original_saved': saved['onnx_qdq'], 'pytorch_original_saved': saved['pytorch_qdq']}
    for mode in ('unforced_repeat_control', 'replace_first_conv_output', 'replace_all_conv_outputs'):
        rows, handles = [], []
        for wrapper in wrappers:
            def hook(mod, inputs, output):
                ref = references[mod.name]
                actual_input = inputs[0].detach().numpy()
                actual_quantized = (torch.round(inputs[0] / mod.activation_scale).clamp(-127, 127) *
                                    mod.activation_scale).detach().numpy()
                expected_quantized = observed[ref['quantized_input']]
                row = {'module': mod.name, 'group': mod.group,
                       'preclip_input': statistics(actual_input, observed[ref['preclip_input']]),
                       'quantized_input': statistics(actual_quantized, expected_quantized),
                       'computed_conv_output': statistics(output.detach().numpy(), observed[ref['conv_output']])}
                changed = actual_quantized != expected_quantized
                if np.any(changed):
                    normalized = actual_input[changed] / np.float32(mod.activation_scale.item())
                    samples = []
                    for index in np.argwhere(changed)[:8]:
                        key = tuple(int(value) for value in index)
                        before = float(actual_input[key])
                        onnx_before = float(observed[ref['preclip_input']][key])
                        samples.append({'index': list(key), 'pytorch_preclip': before, 'onnx_preclip': onnx_before,
                                        'scale': float(mod.activation_scale.item()),
                                        'pytorch_normalized_fp32': float(np.float32(before) / np.float32(mod.activation_scale.item())),
                                        'onnx_normalized_fp32': float(np.float32(onnx_before) / np.float32(mod.activation_scale.item())),
                                        'pytorch_qdq': float(actual_quantized[key]), 'onnx_qdq': float(expected_quantized[key])})
                    row['changed_quantizer_bins'] = {
                        'count': int(changed.sum()),
                        'min_abs_distance_to_half_integer': float(np.abs(normalized - np.floor(normalized) - .5).min()),
                        'samples': samples}
                replace = mode == 'replace_all_conv_outputs' or (
                    mode == 'replace_first_conv_output' and mod.name == first_name)
                row['output_replaced_with_onnx_reference'] = replace
                rows.append(row)
                if replace:
                    return torch.from_numpy(observed[ref['conv_output']].copy())
                return output
            handles.append(wrapper.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                actual = model(x)
                actual = actual[0] if isinstance(actual, (tuple, list)) else actual
                actual = actual.detach().numpy().copy()
        finally:
            for handle in handles:
                handle.remove()
        assert len(rows) == 88 and np.isfinite(actual).all()
        raw_outputs[mode] = actual
        trials.append({'mode': mode, 'replaced_conv_outputs': sum(row['output_replaced_with_onnx_reference'] for row in rows),
                       'final_raw_output_to_onnx': frozen_guard(actual, observed[primary_name]),
                       'all_88_intervening_preclip_input_max_abs': max(row['preclip_input']['max_abs'] for row in rows),
                       'all_88_quantized_input_changed_elements': sum(row['quantized_input']['unequal'] for row in rows),
                       'per_conv_boundaries': rows})
        print(mode, json.dumps(trials[-1]['final_raw_output_to_onnx']), flush=True)
    hashes_after = {name: sha(path) for name, path in paths.items()}
    assert hashes_after == hashes_before
    result = {'schema': 'rampq.gate05.fixed_image_conv_boundary_diagnostic/v1',
              'client_date': '2026-10-19', 'status': 'DIAGNOSTIC_COMPLETE_ORIGINAL_FAILURE_RETAINED',
              'original_full_preflight_status': preflight['status'],
              'original_frozen_inputs_unchanged': True, 'input_sha256': hashes_before,
              'onnx_instrumented_to_original_saved': frozen_guard(observed[primary_name], saved['onnx_qdq']),
              'pytorch_repeat_to_original_saved': frozen_guard(raw_outputs['unforced_repeat_control'], saved['pytorch_qdq']),
              'trials': trials, 'elapsed_seconds': time.monotonic() - started,
              'environment': {'python': sys.version, 'torch': torch.__version__, 'onnx': onnx.__version__,
                              'onnxruntime': ort.__version__, 'numpy': np.__version__,
                              'torch_threads': torch.get_num_threads(), 'torch_mkldnn': torch.backends.mkldnn.enabled,
                              'ort_optimization': 'ORT_DISABLE_ALL', 'ort_provider': 'CPUExecutionProvider'},
              'scope': 'One original Selection image, causal floating-boundary intervention only; no dataset evaluation, '
                       'no model/range/tolerance changes, no TensorRT or hardware evidence. Teacher forcing is not an executable '
                       'model candidate and cannot pass or replace the failed original numerical guard.'}
    (OUT / 'teacher_forcing_result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(OUT / 'raw_outputs.npz', **raw_outputs)
    print('CAUSAL_CONV_BOUNDARY_DIAGNOSTIC_COMPLETE', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
