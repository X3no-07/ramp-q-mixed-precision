"""Prepare an equivalent first-Conv code Clip graph, leaving original intact."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'artifacts/gate05/20261019_firstconv_code_clip_probe'
SOURCE = ROOT / 'artifacts/gate05/20261019_firstconv_numeric_probe'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    source_path = SOURCE / 'model_firstconv_qdq.onnx'
    model = onnx.load(source_path)
    before = model.SerializeToString()
    clip, quantize = [copy.deepcopy(n) for n in model.graph.node[:2]]
    assert clip.op_type == 'Clip' and quantize.op_type == 'QuantizeLinear'
    quantize.input[0] = 'images'
    quantize.output[0] = 'rampq_code_clip_unclipped_q'
    clip.input[:] = ['rampq_code_clip_unclipped_q', 'rampq_code_clip_min', 'rampq_code_clip_max']
    clip.output[0] = 'rampq_model_0_conv_a_q'
    remainder = [copy.deepcopy(n) for n in model.graph.node[2:]]
    del model.graph.node[:]
    model.graph.node.extend([quantize, clip] + remainder)
    tensors = [copy.deepcopy(t) for t in model.graph.initializer if t.name not in ('rampq_model_0_conv_clip_min', 'rampq_model_0_conv_clip_max')]
    del model.graph.initializer[:]
    model.graph.initializer.extend(tensors + [numpy_helper.from_array(np.asarray(-127, np.int8), 'rampq_code_clip_min'),
                                             numpy_helper.from_array(np.asarray(127, np.int8), 'rampq_code_clip_max')])
    onnx.checker.check_model(model)
    model_path = OUT / 'model_firstconv_code_clip.onnx'
    onnx.save(model, model_path)
    shutil.copyfile(SOURCE / 'preprocessed_input.npy', OUT / 'preprocessed_input.npy')
    shutil.copyfile(SOURCE / 'original_failed_preflight.json', OUT / 'original_failed_preflight.json')
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    image = np.load(OUT / 'preprocessed_input.npy', allow_pickle=False)
    outputs = session.run(None, {'images': image})
    arrays = {'code_clip_post_silu': outputs[0], 'code_clip_pre_silu': outputs[1]}
    np.savez_compressed(OUT / 'cpu_expected_outputs.npz', **arrays)
    with np.load(SOURCE / 'cpu_expected_outputs.npz', allow_pickle=False) as old:
        checks = {'post_bit_exact': bool(np.array_equal(old['qdq_post_silu'], outputs[0])),
                  'pre_bit_exact': bool(np.array_equal(old['qdq_pre_silu'], outputs[1]))}
    assert all(checks.values())
    (OUT / 'cpu_equivalence.json').write_text(json.dumps(checks, indent=2) + '\n')
    spec = json.loads((SOURCE / 'spec.json').read_text())
    spec['schema'] = 'rampq.gate05.firstconv_code_clip_probe/v1'
    spec['variants'] = {'code_clip': {'model_file': model_path.name, 'model_sha256': sha(model_path),
                                    'expected_output_npz_keys': {'/model.0/act/Mul_output_0': 'code_clip_post_silu',
                                                                 '/model.0/conv/Conv_output_0': 'code_clip_pre_silu'}}}
    spec['cpu_expected_outputs_sha256'] = sha(OUT / 'cpu_expected_outputs.npz')
    spec['source_isolated_graph_sha256'] = sha(source_path)
    spec['code_clip_equivalence_on_fixed_input'] = checks
    spec['native_build_or_execution_run_by_preparer'] = False
    spec['changes'] = 'Move original float Clip before Q to INT8 code Clip[-127,127] after Q; preserve original scale/zero/weights/bias/Conv/SiLU/output declarations.'
    spec['requires_explicit_native_importer_clip_attribute_correction'] = True
    (OUT / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    shutil.copyfile(__file__, OUT / Path(__file__).name)
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()}
    (OUT / 'manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, 'gate05_firstconv_code_clip_probe_20261019/' + p.name)
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive), 'cpu_checks': checks}))


if __name__ == '__main__':
    main()
