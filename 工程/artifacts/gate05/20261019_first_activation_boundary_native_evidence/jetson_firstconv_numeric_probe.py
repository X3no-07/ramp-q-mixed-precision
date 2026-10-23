"""Native isolated first-Conv numeric diagnosis; Python 3.8 / TensorRT 8.5.

Four small builds use the same frozen input and two named output taps. This is
not a detector migration, accuracy, or performance pass.
"""
import hashlib
import json
from pathlib import Path
import platform
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CASES = (('fp32_no_tf32', 'fp32', False), ('fp16_no_tf32', 'fp32', True),
         ('qdq_fp16_no_tf32', 'qdq', True), ('qdq_fp32_no_tf32', 'qdq', False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stats(array, np):
    value = array.astype(np.float64)
    finite = np.isfinite(value)
    selected = value[finite]
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'all_finite': bool(finite.all()), 'nonfinite_count': int((~finite).sum()),
            'min': float(selected.min()) if selected.size else None,
            'max': float(selected.max()) if selected.size else None,
            'mean': float(selected.mean()) if selected.size else None,
            'rms': float(np.sqrt(np.mean(selected * selected))) if selected.size else None,
            'raw_bytes_sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def compare(actual, expected, np):
    require(actual.shape == expected.shape, 'Comparison shape mismatch')
    valid = np.isfinite(actual) & np.isfinite(expected)
    delta = np.abs(actual[valid].astype(np.float64) - expected[valid].astype(np.float64))
    return {'all_pairs_finite': bool(valid.all()), 'nonfinite_pair_count': int((~valid).sum()),
            'max_abs': float(delta.max()) if delta.size else None,
            'mean_abs': float(delta.mean()) if delta.size else None,
            'rms_difference': float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
            'abs_percentiles_50_90_99_999': np.percentile(delta, [50, 90, 99, 99.9]).tolist() if delta.size else None,
            'bit_exact': bool(np.array_equal(actual, expected)),
            'descriptive_only_no_new_pass_criterion': True}


def stop(process):
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        process.send_signal(sig)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def main():
    spec_path = ROOT / 'spec.json'
    spec = json.loads(spec_path.read_text())
    require(spec['schema'] == 'rampq.gate05.firstconv_numeric_probe/v1', 'Wrong spec')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, info in manifest['files'].items():
        path = (ROOT / name).resolve()
        require(ROOT in path.parents, 'Manifest escape')
        require(path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Bad file ' + name)
    require(sha(ROOT / spec['input_file']) == spec['input_file_sha256'], 'Input hash')
    require(sha(ROOT / spec['cpu_expected_outputs_file']) == spec['cpu_expected_outputs_sha256'], 'Expected hash')
    require(spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED', 'Prior guard changed')
    for variant in spec['variants'].values():
        require(sha(ROOT / variant['model_file']) == variant['model_sha256'], 'Graph hash')
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Native Jetson required')
    import numpy as np
    import tensorrt as trt
    import torch
    require(trt.__version__.startswith('8.5.'), 'Expected TensorRT 8.5')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    run = ROOT / 'results' / ('probe_' + time.strftime('%Y%m%dT%H%M%S'))
    run.mkdir(parents=True, exist_ok=False)
    write(ROOT / 'latest_probe.json', {'run': str(run), 'spec_sha256': sha(spec_path)})
    host_input = np.load(str(ROOT / spec['input_file']), allow_pickle=False)
    require(host_input.dtype == np.float32 and host_input.shape == (1, 3, 640, 640), 'Input contract')
    with np.load(str(ROOT / spec['cpu_expected_outputs_file']), allow_pickle=False) as archive:
        expected = {key: archive[key].copy() for key in archive.files}
    logger = trt.Logger(trt.Logger.WARNING)
    results, native_arrays = {}, {}
    for case, variant_name, fp16 in CASES:
        variant = spec['variants'][variant_name]
        out = run / case
        out.mkdir()
        command = ['/usr/src/tensorrt/bin/trtexec', '--onnx=' + str(ROOT / variant['model_file']),
                   '--saveEngine=' + str(out / 'model.engine'), '--noTF32',
                   '--inputIOFormats=fp32:chw', '--outputIOFormats=fp32:chw,fp32:chw',
                   '--avgTiming=1', '--memPoolSize=workspace:1024',
                   '--profilingVerbosity=detailed', '--buildOnly']
        if fp16:
            command.append('--fp16')
        if variant_name == 'qdq':
            command += ['--int8', '--precisionConstraints=prefer',
                        '--layerPrecisions=' + spec['selected_conv_name'] + ':int8']
        else:
            precision = 'fp16' if fp16 else 'fp32'
            command += ['--precisionConstraints=obey',
                        '--layerPrecisions=' + spec['selected_conv_name'] + ':' + precision,
                        '--layerOutputTypes=' + spec['selected_conv_name'] + ':' + precision]
        request = {'case': case, 'variant': variant_name, 'command': command,
                   'max_build_seconds': 300, 'spec_sha256': sha(spec_path),
                   'onnx_sha256': variant['model_sha256'], 'diagnostic_only': True}
        write(out / 'request.json', request)
        print('FIRSTCONV_BUILD_START ' + case, flush=True)
        started = time.monotonic()
        with (out / 'trtexec.log').open('wb') as handle:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                stop(process)
                write(out / 'result.json', dict(request, status='BUILD_TIMEOUT_INCONCLUSIVE'))
                raise RuntimeError('Small build timeout: ' + case)
        require(code == 0, 'Build failed: ' + case)
        build_seconds = time.monotonic() - started
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine((out / 'model.engine').read_bytes())
        require(engine is not None, 'Deserialize failed')
        layers = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        write(out / 'engine_layers.json', layers)
        context = engine.create_execution_context()
        require(context is not None, 'Context failed')
        buffers, bindings = [], []
        desired = [spec['input_binding']] + spec['output_bindings_in_order']
        desired_by_name = {row['name']: row for row in desired}
        require(engine.num_bindings == len(desired), 'Wrong binding count')
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = list(engine.get_binding_shape(i))
            dtype = engine.get_binding_dtype(i)
            is_input = bool(engine.binding_is_input(i))
            require(name in desired_by_name, 'Unexpected binding name')
            target = desired_by_name[name]
            require(shape == target['shape'] and is_input == target['input'] and dtype == trt.float32,
                    'Wrong binding contract: ' + name)
            require(engine.get_binding_format(i) == trt.TensorFormat.LINEAR, 'Nonlinear binding')
            buffer = torch.from_numpy(host_input.copy()).cuda() if is_input else torch.empty(shape, dtype=torch.float32, device='cuda')
            buffers.append(buffer)
            bindings.append({'name': name, 'shape': shape, 'input': is_input, 'dtype': str(dtype),
                             'format': str(engine.get_binding_format(i))})
        require(len({row['name'] for row in bindings}) == len(desired), 'Duplicate binding')
        input_index = next(i for i, row in enumerate(bindings) if row['input'])
        require(np.array_equal(buffers[input_index].cpu().numpy(), host_input), 'CUDA input transfer differs')
        torch.cuda.synchronize()
        actual = {}
        for repeat in range(2):
            require(context.execute_v2([int(buffer.data_ptr()) for buffer in buffers]), 'execute_v2 failed')
            torch.cuda.synchronize()
            for i, row in enumerate(bindings):
                if not row['input']:
                    key = variant['expected_output_npz_keys'][row['name']]
                    array = buffers[i].cpu().numpy().copy()
                    actual[('first_' if repeat == 0 else 'repeat_') + key] = array
        np.savez_compressed(str(out / 'raw_outputs.npz'), **actual)
        outputs = {}
        for name, key in variant['expected_output_npz_keys'].items():
            outputs[name] = {'npz_key': 'first_' + key, 'stats': stats(actual['first_' + key], np),
                             'comparison_to_cpu_ort': compare(actual['first_' + key], expected[key], np),
                             'repeat_comparison': compare(actual['repeat_' + key], actual['first_' + key], np)}
        convs = [row for row in layers.get('Layers', []) if 'Convolution' in row.get('LayerType', '')]
        require(len(convs) == 1 and spec['selected_conv_name'] in convs[0]['Name'], 'Source Conv mapping ambiguous')
        report = dict(request, status='ISOLATED_FIRSTCONV_NATIVE_OUTPUTS_SAVED',
                      engine_sha256=sha(out / 'model.engine'), bindings=bindings, outputs=outputs,
                      convolution_layers=convs, raw_outputs_sha256=sha(out / 'raw_outputs.npz'),
                      build_seconds_not_benchmark=build_seconds,
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      board_time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), tensorrt=trt.__version__, torch=torch.__version__,
                      original_complete_model_numerical_guard='FAILED_UNCHANGED',
                      pre_silu_output_tap_may_change_native_fusion=True,
                      accuracy_pass_claim=False, performance_pass_claim=False, dataset_evaluation_run=False)
        write(out / 'result.json', report)
        results[case] = report
        native_arrays[case] = {name: actual['first_' + key] for name, key in variant['expected_output_npz_keys'].items()}
        print('FIRSTCONV_CASE_DONE ' + case, flush=True)
        del context, buffers, engine, runtime
        torch.cuda.empty_cache()
    cross = {}
    for first, second in (('qdq_fp16_no_tf32', 'qdq_fp32_no_tf32'), ('fp16_no_tf32', 'fp32_no_tf32')):
        cross[first + '_vs_' + second] = {name: compare(native_arrays[first][name], native_arrays[second][name], np)
                                         for name in native_arrays[first]}
    write(run / 'summary.json', {'schema': 'rampq.gate05.firstconv_numeric_native/v1',
                                'status': 'FOUR_ISOLATED_FIRSTCONV_CASES_SAVED', 'cases': results,
                                'cross_native_comparisons': cross, 'spec_sha256': sha(spec_path),
                                'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
                                'dataset_evaluation_run': False,
                                'limits': 'Experimental isolated graph with pre-SiLU output tap; no full-model parity, accuracy, causality, or performance pass.'})
    print('FOUR_ISOLATED_FIRSTCONV_CASES_SAVED', flush=True)


if __name__ == '__main__':
    main()
