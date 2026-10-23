"""CPU-only synthetic operator equivalence at all 88 frozen activation scales."""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'artifacts/gate05/20261019_all_code_clip_scale_equivalence'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    source = ROOT / 'artifacts/gate05/20261018_int8/input_v1/model_q_all.onnx'
    original_sha = sha(source)
    model = onnx.load(source)
    producers = {output: node for node in model.graph.node for output in node.output}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    results = []
    for quantizer in model.graph.node:
        if quantizer.op_type != 'QuantizeLinear' or quantizer.input[0] not in producers:
            continue
        clip = producers[quantizer.input[0]]
        if clip.op_type != 'Clip':
            continue
        dequantizer = next(n for n in model.graph.node if n.op_type == 'DequantizeLinear' and n.input[0] == quantizer.output[0])
        scale = np.float32(numpy_helper.to_array(initializers[quantizer.input[1]]))
        edges = np.arange(-300, 301, dtype=np.float32) * np.float32(0.5) * scale
        x = np.concatenate([edges, np.nextafter(edges, np.float32(-np.inf)), np.nextafter(edges, np.float32(np.inf)),
                            np.asarray([-1000, -10, -2, -1, -0., 0., 1, 2, 10, 1000], np.float32)])
        x = x.reshape(1, 1, 1, -1)
        input_info = helper.make_tensor_value_info(clip.input[0], TensorProto.FLOAT, list(x.shape))
        output_info = helper.make_tensor_value_info(dequantizer.output[0], TensorProto.FLOAT, list(x.shape))
        nodes = [copy.deepcopy(n) for n in (clip, quantizer, dequantizer)]
        tensors = [copy.deepcopy(initializers[name]) for name in (clip.input[1], clip.input[2], quantizer.input[1], quantizer.input[2])]
        old = helper.make_model(helper.make_graph(nodes, 'frozen_operator', [input_info], [output_info], tensors),
                                opset_imports=copy.deepcopy(model.opset_import))
        old.ir_version = model.ir_version
        new = copy.deepcopy(old)
        c, q, dq = [copy.deepcopy(n) for n in new.graph.node]
        q.input[0] = clip.input[0]
        q.output[0] = 'code_clip_unclipped'
        c.input[:] = ['code_clip_unclipped', 'code_min', 'code_max']
        c.output[0] = quantizer.output[0]
        del new.graph.node[:]
        new.graph.node.extend([q, c, dq])
        del new.graph.initializer[:]
        new.graph.initializer.extend([copy.deepcopy(initializers[quantizer.input[1]]), copy.deepcopy(initializers[quantizer.input[2]]),
                                      numpy_helper.from_array(np.asarray(-127, np.int8), 'code_min'),
                                      numpy_helper.from_array(np.asarray(127, np.int8), 'code_max')])
        arrays = []
        for graph in (old, new):
            onnx.checker.check_model(graph)
            session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
            arrays.append(session.run(None, {clip.input[0]: x})[0])
        delta = np.abs(arrays[0].astype(np.float64) - arrays[1].astype(np.float64))
        results.append({'clip': clip.name, 'quantizer': quantizer.name, 'scale': float(scale),
                        'samples': x.size, 'max_abs': float(delta.max()), 'bit_exact': bool(np.array_equal(*arrays)),
                        'input_raw_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                        'old_output_raw_sha256': hashlib.sha256(arrays[0].tobytes()).hexdigest(),
                        'rewritten_output_raw_sha256': hashlib.sha256(arrays[1].tobytes()).hexdigest()})
    assert len(results) == 88 and sha(source) == original_sha
    report = {'status': 'ALL_FROZEN_SCALE_SYNTHETIC_OPERATOR_CHECKS_COMPLETE',
              'client_at': datetime.now().astimezone().isoformat(), 'source_sha256': original_sha,
              'operators': len(results), 'samples_per_operator': results[0]['samples'],
              'all_bit_exact': all(row['bit_exact'] for row in results), 'results': results,
              'onnxruntime': ort.__version__, 'graph_optimization': 'ORT_DISABLE_ALL',
              'cpu_only': True, 'dataset_evaluation_run': False,
              'limits': 'Finite synthetic thresholds at all frozen scales, not exhaustive float32 proof or complete-model parity.'}
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'operators': len(results), 'all_bit_exact': report['all_bit_exact'],
                      'maximum_difference': max(row['max_abs'] for row in results), 'report_sha256': sha(OUT / 'analysis.json')}))


if __name__ == '__main__':
    main()
