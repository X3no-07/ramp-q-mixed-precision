"""Prepare signed synthetic Clip/QDQ boundary equivalence and native inputs."""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'artifacts/gate05/20261019_code_clip_signed_probe'
SOURCE = ROOT / 'artifacts/gate05/20261019_first_activation_boundary_probe'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    source = onnx.load(SOURCE / 'model_activation_clip.onnx')
    original = copy.deepcopy(source)
    init = {t.name: numpy_helper.to_array(t) for t in source.graph.initializer}
    scale = np.float32(init['rampq_model_0_conv_a_scale'])
    # Include every integer and half-integer neighborhood, saturation, signed
    # zero, infinities excluded. This is synthetic operator diagnosis, no data.
    centers = np.arange(-300, 301, dtype=np.float32) * np.float32(0.5) * scale
    samples = np.concatenate([centers, np.nextafter(centers, np.float32(-np.inf)),
                              np.nextafter(centers, np.float32(np.inf)),
                              np.asarray([-10, -2, -1, -0., 0., 1, 2, 10], np.float32)])
    samples = samples.reshape(1, 1, 1, -1)
    shape = list(samples.shape)
    for value in [original.graph.input[0], original.graph.output[0]]:
        for i, dim in enumerate(value.type.tensor_type.shape.dim):
            dim.dim_value = shape[i]
    rewritten = copy.deepcopy(original)
    clip, quantize, dequantize = rewritten.graph.node
    clip_copy = copy.deepcopy(clip)
    del rewritten.graph.node[:]
    quantize.input[0] = 'images'
    quantize.output[0] = 'rampq_code_clip_unclipped_q'
    clip_copy.input[:] = ['rampq_code_clip_unclipped_q', 'rampq_code_clip_min', 'rampq_code_clip_max']
    clip_copy.output[0] = 'rampq_model_0_conv_a_q'
    rewritten.graph.node.extend([quantize, clip_copy, dequantize])
    kept = [t for t in rewritten.graph.initializer if 'clip_min' not in t.name and 'clip_max' not in t.name]
    del rewritten.graph.initializer[:]
    rewritten.graph.initializer.extend(kept + [numpy_helper.from_array(np.asarray(-127, np.int8), name='rampq_code_clip_min'),
                                               numpy_helper.from_array(np.asarray(127, np.int8), name='rampq_code_clip_max')])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    outputs, variants = {}, {}
    for name, model in [('clip', original), ('code_clip', rewritten)]:
        onnx.checker.check_model(model)
        path = OUT / ('model_' + name + '.onnx')
        onnx.save(model, path)
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        values = session.run(None, {'images': samples})
        outputs[name + '_dequantized'] = values[0]
        variants[name] = {'model_file': path.name, 'model_sha256': sha(path),
                          'expected_output_npz_keys': {'rampq_model_0_conv_a_dq': name + '_dequantized'}}
    np.save(OUT / 'synthetic_signed_input.npy', samples)
    np.savez_compressed(OUT / 'cpu_expected_outputs.npz', **outputs)
    assert np.array_equal(outputs['clip_dequantized'], outputs['code_clip_dequantized'])
    scales = json.loads((ROOT / 'artifacts/calibration/screen.json').read_text(encoding='utf-8-sig'))
    save('cpu_equivalence.json', {'status': 'SIGNED_OPERATOR_PAIRED_CPU_OUTPUTS_BIT_EXACT',
                                 'samples': samples.size, 'finite': bool(np.isfinite(samples).all()),
                                 'input_min': float(samples.min()), 'input_max': float(samples.max()),
                                 'activation_scale': float(scale),
                                 'sample_design': 'Integer/half-integer thresholds +/- 1 float32 ULP; saturation; signed zero',
                                 'all_samples_bit_exact': True, 'input_is_synthetic_not_dataset': True,
                                 'onnxruntime': ort.__version__, 'original_full_model_guard': 'FAILED_UNCHANGED',
                                 'limits': 'First quantizer scale only. No claim yet for all frozen scales or complete model.'})
    binding = lambda name, is_input: {'name': name, 'input': is_input, 'shape': shape, 'dtype': 'DataType.FLOAT'}
    save('spec.json', {'schema': 'rampq.gate05.code_clip_signed_probe/v1',
                       'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
                       'input_file': 'synthetic_signed_input.npy', 'input_file_sha256': sha(OUT / 'synthetic_signed_input.npy'),
                       'input_binding': binding('images', True),
                       'output_bindings_in_order': [binding('rampq_model_0_conv_a_dq', False)],
                       'variants': variants, 'cpu_expected_outputs_file': 'cpu_expected_outputs.npz',
                       'cpu_expected_outputs_sha256': sha(OUT / 'cpu_expected_outputs.npz'),
                       'selected_clip_node_name': 'rampq_model_0_conv_Clip',
                       'source_graph_sha256': sha(SOURCE / 'model_activation_clip.onnx'),
                       'changes': 'Synthetic shape. Paired rewrite moves Clip after ActivationQ, uses INT8 [-127,127] bounds; original scale/zero/DQ unchanged.',
                       'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
                       'original_tolerances_changed': False, 'dataset_evaluation_run': False,
                       'limits': 'Signed synthetic operator diagnostic only; not a complete model repair or accuracy test.'})
    shutil.copyfile(__file__, OUT / Path(__file__).name)
    save('manifest.json', {'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()}})
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, 'gate05_code_clip_signed_probe_20261019/' + p.name)
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive), 'signed_samples': samples.size}))


if __name__ == '__main__':
    main()
