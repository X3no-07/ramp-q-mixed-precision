"""Execute two isolated first activation Q/DQ graphs; native diagnostic only."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from jetson_firstconv_numeric_probe import compare, require, sha, stats, stop, write

ROOT = Path(__file__).resolve().parent
SPEC_SCHEMA = 'rampq.gate05.first_activation_boundary_probe/v1'
VARIANTS_TO_RUN = ('clip', 'no_clip')
FORCE_CLIP_PRECISION = None
COMPLETION_STATUS = 'TWO_ISOLATED_ACTIVATION_BOUNDARY_CASES_SAVED'


def main():
    spec = json.loads((ROOT / 'spec.json').read_text())
    require(spec['schema'] == SPEC_SCHEMA, 'Wrong spec')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, info in manifest['files'].items():
        path = (ROOT / name).resolve()
        require(ROOT in path.parents and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Bad manifest ' + name)
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Native Jetson required')
    import numpy as np
    import tensorrt as trt
    import torch
    require(trt.__version__.startswith('8.5.') and torch.cuda.is_available(), 'Wrong native runtime')
    run = ROOT / 'results' / ('probe_' + time.strftime('%Y%m%dT%H%M%S'))
    run.mkdir(parents=True, exist_ok=False)
    write(ROOT / 'latest_probe.json', {'run': str(run), 'spec_sha256': sha(ROOT / 'spec.json')})
    image = np.load(str(ROOT / spec['input_file']), allow_pickle=False)
    require(image.dtype == np.float32 and list(image.shape) == spec['input_binding']['shape'], 'Input contract')
    require(sha(ROOT / spec['input_file']) == spec['input_file_sha256'], 'Input hash')
    require(sha(ROOT / spec['cpu_expected_outputs_file']) == spec['cpu_expected_outputs_sha256'], 'CPU hash')
    with np.load(str(ROOT / spec['cpu_expected_outputs_file']), allow_pickle=False) as data:
        expected = {key: data[key].copy() for key in data.files}
    logger = trt.Logger(trt.Logger.WARNING)
    cases, native = {}, {}
    for name in VARIANTS_TO_RUN:
        variant = spec['variants'][name]
        require(sha(ROOT / variant['model_file']) == variant['model_sha256'], 'Graph hash')
        out = run / name
        out.mkdir()
        command = ['/usr/src/tensorrt/bin/trtexec', '--onnx=' + str(ROOT / variant['model_file']),
                   '--saveEngine=' + str(out / 'model.engine'), '--int8', '--noTF32',
                   '--inputIOFormats=fp32:chw', '--outputIOFormats=fp32:chw',
                   '--avgTiming=1', '--memPoolSize=workspace:1024', '--profilingVerbosity=detailed', '--buildOnly']
        if FORCE_CLIP_PRECISION is not None:
            require(name == 'clip' and FORCE_CLIP_PRECISION == 'fp32', 'Unsupported Clip intervention')
            command += ['--precisionConstraints=obey',
                        '--layerPrecisions=' + spec['selected_clip_node_name'] + ':fp32',
                        '--layerOutputTypes=' + spec['selected_clip_node_name'] + ':fp32']
        write(out / 'request.json', {'command': command, 'case': name, 'max_build_seconds': 120,
                                    'spec_sha256': sha(ROOT / 'spec.json'), 'onnx_sha256': variant['model_sha256']})
        print('BOUNDARY_BUILD_START ' + name, flush=True)
        with (out / 'trtexec.log').open('wb') as handle:
            p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            try:
                code = p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                stop(p)
                raise RuntimeError('Boundary build timeout: ' + name)
        require(code == 0, 'Boundary build failed: ' + name)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine((out / 'model.engine').read_bytes())
        require(engine is not None, 'Engine unavailable')
        layers = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        write(out / 'engine_layers.json', layers)
        context = engine.create_execution_context()
        require(context is not None and engine.num_bindings == 2, 'Wrong context/binding count')
        desired = {row['name']: row for row in [spec['input_binding']] + spec['output_bindings_in_order']}
        buffers, bindings = [], []
        for i in range(engine.num_bindings):
            tensor = engine.get_binding_name(i)
            require(tensor in desired, 'Unknown binding')
            target = desired[tensor]
            require(list(engine.get_binding_shape(i)) == target['shape'] and
                    bool(engine.binding_is_input(i)) == target['input'] and
                    engine.get_binding_dtype(i) == trt.float32 and
                    engine.get_binding_format(i) == trt.TensorFormat.LINEAR, 'Wrong binding')
            buffers.append(torch.from_numpy(image.copy()).cuda() if target['input'] else
                           torch.empty(target['shape'], dtype=torch.float32, device='cuda'))
            bindings.append({'name': tensor, 'input': target['input'], 'shape': target['shape'], 'dtype': str(engine.get_binding_dtype(i))})
        input_index = next(i for i, row in enumerate(bindings) if row['input'])
        output_index = next(i for i, row in enumerate(bindings) if not row['input'])
        require(np.array_equal(buffers[input_index].cpu().numpy(), image), 'CUDA input transfer differs')
        saved = {}
        for repeat in range(2):
            torch.cuda.synchronize()
            require(context.execute_v2([int(value.data_ptr()) for value in buffers]), 'Execution failed')
            torch.cuda.synchronize()
            saved['first' if repeat == 0 else 'repeat'] = buffers[output_index].cpu().numpy().copy()
        np.savez_compressed(str(out / 'raw_outputs.npz'), **saved)
        expected_key = variant['expected_output_npz_keys'][bindings[output_index]['name']]
        result = {'status': 'ISOLATED_ACTIVATION_BOUNDARY_OUTPUTS_SAVED', 'diagnostic_only': True,
                  'case': name, 'bindings': bindings, 'stats': stats(saved['first'], np),
                  'comparison_to_cpu_ort': compare(saved['first'], expected[expected_key], np),
                  'repeat_comparison': compare(saved['repeat'], saved['first'], np),
                  'engine_sha256': sha(out / 'model.engine'), 'raw_outputs_sha256': sha(out / 'raw_outputs.npz'),
                  'spec_sha256': sha(ROOT / 'spec.json'), 'onnx_sha256': variant['model_sha256'],
                  'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'tensorrt': trt.__version__, 'torch': torch.__version__,
                  'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'dataset_evaluation_run': False,
                  'limits': 'Only fixed input and isolated slice. Removing Clip is a diagnostic intervention; original model is unchanged.'}
        write(out / 'result.json', result)
        cases[name], native[name] = result, saved['first']
        print('BOUNDARY_CASE_DONE ' + name, flush=True)
        del context, buffers, engine, runtime
        torch.cuda.empty_cache()
    write(run / 'summary.json', {'status': COMPLETION_STATUS, 'cases': cases,
                                'native_clip_vs_no_clip': compare(native['clip'], native['no_clip'], np) if set(native) == {'clip', 'no_clip'} else None,
                                'requested_clip_precision': FORCE_CLIP_PRECISION,
                                'diagnostic_only': True, 'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
                                'dataset_evaluation_run': False})


if __name__ == '__main__':
    main()
