"""Build one frozen explicit-INT8 Q/DQ graph using TensorRT 8.5 on Jetson."""
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def query(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=20)
    return {'command': args, 'returncode': p.returncode, 'output': p.stdout}


def main():
    assert platform.system() == 'Linux' and platform.machine() == 'aarch64'
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, info in manifest['files'].items():
        assert (ROOT / name).stat().st_size == info['bytes'] and sha(ROOT / name) == info['sha256'], name
    preflight_status = json.loads((ROOT / 'preflight.json').read_text())['status']
    diagnostic_only = preflight_status != 'LOCAL_QDQ_PREFLIGHT_PASS'
    if diagnostic_only:
        scope = json.loads((ROOT / 'diagnostic_scope.json').read_text())
        isolated = json.loads((ROOT / 'isolated_quantizer_audit.json').read_text())
        assert preflight_status == 'LOCAL_QDQ_PREFLIGHT_FAILED'
        assert scope['status'] == 'DIAGNOSTIC_BUILD_ONLY_FULL_PARITY_FAILED'
        assert scope['failed_preflight_sha256'] == sha(ROOT / 'preflight.json')
        assert scope['isolated_audit_sha256'] == sha(ROOT / 'isolated_quantizer_audit.json')
        assert isolated['weight_qdq_all_88_exact'] and len(isolated['same_input_quantizer']) == 88
        assert all(row['different'] == 0 for row in isolated['same_input_quantizer'])
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert protocol['configuration'] == manifest['configuration'] == 'q_all'
    average_timing = manifest.get('kernel_selection_avg_timing')
    assert average_timing in (None, 1), 'Unexpected kernel selection timing override'
    binding_contract = manifest.get('expected_binding_contract')
    if binding_contract is not None:
        candidate = json.loads((ROOT / 'candidate.json').read_text())
        assert sha(ROOT / 'model_q_all.onnx') == candidate['candidate_sha256']
        assert candidate['all_arithmetic_nodes_and_initializers_byte_identical']
        assert candidate['quantization_groups_weights_ranges_and_main_output_unchanged']
        assert binding_contract == [
            {'name': 'images', 'input': True, 'shape': [1,3,640,640], 'dtype': 'DataType.FLOAT'},
            {'name': 'output0', 'input': False, 'shape': [1,14,8400], 'dtype': 'DataType.FLOAT'},
            {'name': candidate['extra_output'], 'input': False, 'shape': [1,128,20,20], 'dtype': 'DataType.FLOAT'}]
    import tensorrt as trt
    assert trt.__version__.startswith('8.5.'), trt.__version__
    executable = '/usr/src/tensorrt/bin/trtexec'
    help_info = query([executable, '--help'])
    for option in ('--buildOnly', '--memPoolSize', '--profilingVerbosity', '--noTF32', '--fp16', '--int8'):
        assert option in help_info['output'], option
    if average_timing is not None:
        assert '--avgTiming' in help_info['output']
    run = ROOT / 'results' / datetime.now(timezone.utc).strftime('build_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    output = run / 'model_q_all.engine'
    # Explicit Q/DQ supplies scales. Do not add an implicit calibrator or random ranges.
    command = [executable, '--onnx=' + str(ROOT / 'model_q_all.onnx'), '--saveEngine=' + str(output),
               '--int8', '--fp16', '--noTF32', '--memPoolSize=workspace:1024', '--profilingVerbosity=detailed', '--buildOnly']
    if average_timing is not None:
        command.append('--avgTiming=' + str(average_timing))
    environment = {'tensorrt': trt.__version__, 'python': platform.python_version(),
                   'machine': platform.machine(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   'power_mode': query(['/usr/sbin/nvpmodel', '-q']),
                   'memory': query(['free', '-m']), 'fan_service': query(['systemctl', 'is-active', 'nvfancontrol.service']),
                   'jetson_linux': Path('/etc/nv_tegra_release').read_text()}
    write(run / 'environment.json', environment)
    write(run / 'build_request.json', {'command': command, 'configuration': 'q_all',
          'onnx_sha256': sha(ROOT / 'model_q_all.onnx'), 'protocol_sha256': sha(ROOT / 'protocol.json'),
          'precision': 'Explicit signed INT8 Q/DQ on all 88 Conv inputs/weights; FP16/FP32 floating operators',
          'calibrator': None, 'workspace_mib': 1024, 'diagnostic_only': diagnostic_only,
          'local_preflight_status': preflight_status,
          'expected_binding_contract': binding_contract,
          'kernel_selection_avg_timing_override': average_timing,
          'limits': 'Workspace is not total RAM; actual layer precision inspection required. Failed numerical parity remains failed.'})
    (run / 'trtexec_help.txt').write_text(help_info['output'])
    write(ROOT / 'latest_build.json', {'run_dir': str(run), 'status': 'BUILD_RUNNING'})
    print('RESULT_DIR:', run, flush=True)
    print('FILES_SHA256_OK', flush=True)
    started = time.monotonic()
    with (run / 'trtexec_build.log').open('w') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    print('BUILD_RUNNING_SECONDS:', round(time.monotonic()-started), flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    if code != 0 or not output.is_file():
        report = {'status': 'BUILD_FAILED', 'returncode': code, 'elapsed_seconds': time.monotonic()-started,
                  'diagnostic_only': diagnostic_only, 'local_preflight_status': preflight_status}
        write(run / 'build_result.json', report)
        write(ROOT / 'latest_build.json', dict(report, run_dir=str(run)))
        print('\n'.join((run/'trtexec_build.log').read_text(errors='replace').splitlines()[-50:]), flush=True)
        raise RuntimeError('TensorRT Q/DQ build failed; retain logs and do not call this INT8 evidence')
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(output.read_bytes())
    assert engine is not None
    bindings = [{'name': engine.get_binding_name(i), 'input': bool(engine.binding_is_input(i)),
                 'shape': list(engine.get_binding_shape(i)), 'dtype': str(engine.get_binding_dtype(i))}
                for i in range(engine.num_bindings)]
    assert [b['shape'] for b in bindings if b['input']] == [[1,3,640,640]]
    if binding_contract is None:
        assert [b['shape'] for b in bindings if not b['input']] == [[1,14,8400]]
    else:
        assert len({b['name'] for b in bindings}) == len(bindings) == len(binding_contract)
        assert sorted(bindings, key=lambda b:b['name']) == sorted(binding_contract, key=lambda b:b['name'])
    inspector = engine.create_engine_inspector()
    (run / 'engine_layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    report = {'status': 'BUILT_AND_DESERIALIZED', 'configuration': 'q_all', 'engine_sha256': sha(output),
              'engine_bytes': output.stat().st_size, 'onnx_sha256': sha(ROOT/'model_q_all.onnx'),
              'engine_layers': engine.num_layers, 'bindings': bindings,
              'expected_binding_contract': binding_contract,
              'primary_output_name': 'output0' if binding_contract is not None else None,
              'kernel_selection_avg_timing_override': average_timing,
              'elapsed_seconds': round(time.monotonic()-started,3), 'layer_precision_review': 'pending',
              'diagnostic_only': diagnostic_only, 'local_preflight_status': preflight_status,
              'detection_inference': 'not run', 'benchmark': 'not run'}
    write(run / 'build_result.json', report)
    write(ROOT / 'latest_build.json', dict(report, run_dir=str(run)))
    print(json.dumps(report, indent=2), flush=True)
    print('TENSORRT_QDQ_BUILD_OK', flush=True)


if __name__ == '__main__':
    main()
