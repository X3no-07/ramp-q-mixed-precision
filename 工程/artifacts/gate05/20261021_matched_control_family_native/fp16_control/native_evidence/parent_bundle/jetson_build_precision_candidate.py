"""Build one manifest-frozen precision candidate on Jetson; hardware diagnostics only."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def query(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, timeout=20)
    return {'command': command, 'returncode': result.returncode, 'output': result.stdout}


def contract_signature(rows):
    assert len({row['name'] for row in rows}) == len(rows), 'Duplicate binding name'
    assert sum(row['input'] is True for row in rows) == 1
    return sorted((row['name'], row['input'], tuple(row['shape']), row['dtype']) for row in rows)


def parse_and_validate(trt, path, spec):
    """Validate exact imported names before trtexec can silently ignore a CLI constraint."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(path.read_bytes()), '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    names = [layer.name for layer in layers]
    assert len(names) == len(set(names)), 'Parsed layer names must be unique'
    by_name = {layer.name: layer for layer in layers}
    requested = []
    for row in spec['requests']:
        name = row['onnx_node']
        assert name in by_name, 'Requested source Conv absent from parsed network: ' + name
        layer = by_name[name]
        assert layer.type == trt.LayerType.CONVOLUTION and layer.num_outputs == 1, name
        assert not any(character in name for character in (':', ',', '*')), 'Unsupported CLI layer name: ' + name
        requested.append(dict(row, parsed_layer_type=str(layer.type), parsed_outputs=layer.num_outputs))
    parsed_convs = {layer.name for layer in layers if layer.type == trt.LayerType.CONVOLUTION}
    assert parsed_convs == {row['onnx_node'] for row in spec['requests']}, 'Parsed Conv inventory differs'
    io = []
    for is_input, count in ((True, network.num_inputs), (False, network.num_outputs)):
        for i in range(count):
            value = network.get_input(i) if is_input else network.get_output(i)
            io.append({'name': value.name, 'input': is_input, 'shape': list(value.shape), 'dtype': str(value.dtype)})
    assert contract_signature(io) == contract_signature(spec['expected_binding_contract']), 'Parsed graph I/O differs'
    result = {'status': 'ALL_REQUESTED_CONV_NAMES_EXIST_IN_PARSED_NETWORK',
              'parsed_layer_count': network.num_layers, 'source_conv_count': len(parsed_convs),
              'requests': requested, 'bindings_in_network_order': io,
              'limits': 'Parsed names validate request addressing; native adoption still requires final engine inspection.'}
    del by_name, layers, parser, network, builder
    gc.collect()
    return result


def build_command(executable, root, run, spec):
    command = [executable, '--onnx=' + str(root / 'model.onnx'), '--saveEngine=' + str(run / 'model.engine'),
               '--fp16', '--noTF32', '--memPoolSize=workspace:1024', '--profilingVerbosity=detailed',
               '--buildOnly', '--avgTiming=1', '--precisionConstraints=obey',
               '--inputIOFormats=fp32:chw', '--outputIOFormats=fp32:chw,fp32:chw']
    if spec['quantized_conv_count']:
        command.append('--int8')
    restored = [row['onnx_node'] for row in spec['requests'] if row['precision'] == 'FP16']
    if restored:
        constraints = ','.join(name + ':fp16' for name in restored)
        command.extend(['--layerPrecisions=' + constraints, '--layerOutputTypes=' + constraints])
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    assert platform.system() == 'Linux' and platform.machine() == 'aarch64'
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, info in manifest['files'].items():
        path = root / name
        assert path.resolve().parent == root, 'Bundle inputs must be direct children'
        assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], name
    spec = json.loads((root / 'spec.json').read_text())
    assert spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1'
    for filename, field in [('model.onnx', 'graph_sha256'), ('graph_audit.json', 'graph_audit_sha256'),
                            ('module_map.json', 'module_map_sha256'), ('preflight.json', 'failed_preflight_sha256'),
                            ('diagnostic_scope.json', 'diagnostic_scope_sha256')]:
        assert sha(root / filename) == spec[field], filename
    scope = json.loads((root / 'diagnostic_scope.json').read_text())
    preflight = json.loads((root / 'preflight.json').read_text())
    assert spec['diagnostic_only'] is True and preflight['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert scope['status'] == 'DIAGNOSTIC_MATRIX_BUILD_AND_FIXED_IMAGE_ONLY'
    assert scope['failed_preflight_sha256'] == spec['failed_preflight_sha256']
    assert scope['original_tolerances_changed'] is False and scope['dataset_evaluation_run'] is False
    assert len(spec['requests']) == len({row['onnx_node'] for row in spec['requests']}) == 88
    assert set(row['precision'] for row in spec['requests']) <= {'INT8', 'FP16'}
    assert sum(row['precision'] == 'INT8' for row in spec['requests']) == spec['quantized_conv_count']
    assert sum(row['precision'] == 'FP16' for row in spec['requests']) == spec['fp16_conv_count']
    assert spec['workspace_mib'] == 1024 and spec['avg_timing'] == 1 and spec['max_build_seconds'] == 1800
    assert query(['pgrep', '-x', 'trtexec'])['returncode'] == 1, 'Another trtexec is running'
    import tensorrt as trt
    assert trt.__version__.startswith('8.5.'), trt.__version__
    executable = '/usr/src/tensorrt/bin/trtexec'
    help_info = query([executable, '--help'])
    for option in ('--int8', '--fp16', '--precisionConstraints', '--layerPrecisions', '--layerOutputTypes',
                   '--inputIOFormats', '--outputIOFormats', '--avgTiming', '--buildOnly'):
        assert option in help_info['output'], option
    run = root / 'results' / datetime.now(timezone.utc).strftime('build_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    command = build_command(executable, root, run, spec)
    common = {'configuration': spec['configuration'], 'onnx_sha256': sha(root / 'model.onnx'),
              'spec_sha256': sha(root / 'spec.json'), 'manifest_sha256': sha(root / 'manifest.json'),
              'diagnostic_only': True, 'local_preflight_status': preflight['status'],
              'expected_binding_contract': spec['expected_binding_contract'],
              'primary_output_name': spec['primary_output_name'],
              'graph_audit_sha256': spec['graph_audit_sha256'], 'module_map_sha256': spec['module_map_sha256']}
    write(run / 'environment.json', {'tensorrt': trt.__version__, 'python': platform.python_version(),
          'machine': platform.machine(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          'power_mode': query(['/usr/sbin/nvpmodel', '-q']), 'memory': query(['free', '-m']),
          'fan_service': query(['systemctl', 'is-active', 'nvfancontrol.service']),
          'jetson_linux': Path('/etc/nv_tegra_release').read_text()})
    write(run / 'build_request.json', dict(common, command=command,
          restored_fp16_compute_and_output_constraints=[row['onnx_node'] for row in spec['requests'] if row['precision'] == 'FP16'],
          external_io='Explicit FLOAT linear input and two outputs; reformatting allowed',
          limits='obey requests may fail; actual engine formats/precision must be audited. Failed numerical parity remains failed.'))
    (run / 'trtexec_help.txt').write_text(help_info['output'])
    write(root / 'latest_build.json', {'run_dir': str(run), 'status': 'BUILD_RUNNING'})
    print('RESULT_DIR:', run, flush=True)
    started = time.monotonic()
    try:
        parsed = parse_and_validate(trt, root / 'model.onnx', spec)
        write(run / 'parsed_network_validation.json', parsed)
        with (run / 'trtexec_build.log').open('w') as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        code = process.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        print('BUILD_RUNNING_SECONDS:', round(time.monotonic() - started), flush=True)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        assert code == 0 and (run / 'model.engine').is_file(), 'trtexec build failed; return code ' + str(code)
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine((run / 'model.engine').read_bytes())
        assert engine is not None, 'Deserialization failed'
        bindings = [{'name': engine.get_binding_name(i), 'input': bool(engine.binding_is_input(i)),
                     'shape': list(engine.get_binding_shape(i)), 'dtype': str(engine.get_binding_dtype(i)),
                     'format': str(engine.get_binding_format(i))} for i in range(engine.num_bindings)]
        common.update(engine_sha256=sha(run / 'model.engine'), engine_bytes=(run / 'model.engine').stat().st_size,
                      bindings=bindings, engine_layers=engine.num_layers)
        inspector = engine.create_engine_inspector()
        (run / 'engine_layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        assert contract_signature(bindings) == contract_signature(spec['expected_binding_contract']), 'Actual engine I/O differs'
        assert all(row['format'] == str(trt.TensorFormat.LINEAR) for row in bindings), 'Nonlinear external binding'
        report = dict(common, status='BUILT_AND_DESERIALIZED', elapsed_seconds=time.monotonic() - started,
                      actual_precision_audit='PENDING', single_image_inference='NOT_RUN', dataset_evaluation='NOT_RUN')
    except BaseException as error:
        report = dict(common, status='BUILD_OR_CONTRACT_VALIDATION_FAILED', elapsed_seconds=time.monotonic() - started,
                      error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc(),
                      single_image_inference='NOT_RUN', dataset_evaluation='NOT_RUN')
        write(run / 'build_result.json', report)
        write(root / 'latest_build.json', dict(report, run_dir=str(run)))
        print(json.dumps(report, indent=2), flush=True)
        raise
    write(run / 'build_result.json', report)
    write(root / 'latest_build.json', dict(report, run_dir=str(run)))
    print(json.dumps(report, indent=2), flush=True)
    print('TENSORRT_PRECISION_CANDIDATE_BUILT', flush=True)


if __name__ == '__main__':
    main()
