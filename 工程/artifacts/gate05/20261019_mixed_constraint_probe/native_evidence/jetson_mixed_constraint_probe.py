"""Four bounded build-only constraint probes of the same exact two-Conv graph."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time


VARIANTS = ('compute_output_obey', 'compute_only_obey', 'compute_output_prefer', 'global_fp16_only')
FIRST = '/model.2/cv2/conv/Conv'
SECOND = '/model.3/conv/Conv'
KERNEL_MARKERS = ('controller error detected', 'HC died;', 'unregister_netdevice: waiting',
                  'Out of memory:', 'oom-kill:', 'Kernel panic')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_bundle(root):
    manifest = read(root / 'manifest.json')
    require(manifest['schema'] == 'rampq.gate05.mixed_constraint_probe_bundle/v1', 'Unexpected manifest schema')
    for name, info in manifest['files'].items():
        path = root / name
        require(Path(name).name == name and path.resolve().parent == root, 'Unsafe manifest path')
        require(path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Input hash differs: ' + name)
    spec = read(root / 'spec.json')
    require(spec['schema'] == 'rampq.gate05.mixed_constraint_probe/v1' and spec['diagnostic_only'] is True,
            'Unexpected probe spec/scope')
    require(spec['source_configuration'] == 'r_only_G1' and spec['variants'] == list(VARIANTS),
            'Frozen source/variant inventory differs')
    require(spec['max_seconds_per_variant'] == 90 and spec['max_total_seconds'] == 360 and
            spec['workspace_mib'] == 1024 and spec['avg_timing'] == 1, 'Build bounds differ')
    require(sha(root / 'model.onnx') == spec['graph_sha256'] and
            sha(root / 'preflight.json') == spec['failed_preflight_sha256'], 'Spec input hash differs')
    require(read(root / 'preflight.json')['status'] == spec['original_numerical_parity_status'] ==
            'LOCAL_QDQ_PREFLIGHT_FAILED', 'Original parity failure must remain recorded')
    require(spec['serialized_nodes_and_initializers_unchanged'] is True and
            spec['auxiliary_graph_outputs_added'] is False and spec['nodes'] == 9 and spec['initializers'] == 10,
            'Exact extraction contract differs')
    require(spec['external_input']['shape'] == [1, 48, 160, 160] and
            spec['external_output']['shape'] == [1, 64, 80, 80], 'Frozen static shapes differ')
    require([(row['onnx_node'], row['precision']) for row in spec['source_requests']] ==
            [(FIRST, 'FP16'), (SECOND, 'INT8')], 'Frozen requested precisions differ')
    return spec


def command_for(root, engine, name):
    require(name in VARIANTS, 'Unexpected constraint variant')
    command = ['/usr/src/tensorrt/bin/trtexec', '--onnx=' + str(root / 'model.onnx'),
               '--saveEngine=' + str(engine), '--int8', '--fp16', '--noTF32',
               '--memPoolSize=workspace:1024', '--avgTiming=1', '--profilingVerbosity=detailed', '--buildOnly',
               '--inputIOFormats=fp32:chw', '--outputIOFormats=fp32:chw']
    if name != 'global_fp16_only':
        command += ['--precisionConstraints=' + ('prefer' if name == 'compute_output_prefer' else 'obey'),
                    '--layerPrecisions=' + FIRST + ':fp16']
        if name != 'compute_only_obey':
            command += ['--layerOutputTypes=' + FIRST + ':fp16']
    return command


def snapshot():
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1])
    temperatures = {}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            temperatures[(zone / 'type').read_text().strip()] = int((zone / 'temp').read_text()) / 1000.0
        except (OSError, ValueError):
            pass
    require(available is not None and temperatures, 'Memory/thermal readings unavailable')
    return {'available_kib': available, 'temperatures_c': temperatures}


def safe_to_continue(boot, kernel, collector):
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot, 'Board rebooted')
    require(collector.poll() is None, 'Kernel diagnostic collector exited')
    require(subprocess.run(['systemctl', 'is-active', '--quiet', 'nvfancontrol.service'], timeout=10).returncode == 0,
            'Fan service is inactive')
    require(not any(marker in kernel.read_text(errors='replace') for marker in KERNEL_MARKERS),
            'New kernel safety event')
    sample = snapshot()
    require(max(sample['temperatures_c'].values()) < 85, '85 C temperature stop limit')
    return sample


def require_no_trtexec():
    result = subprocess.run(['pgrep', '-x', 'trtexec'], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True, timeout=10)
    require(result.returncode == 1, 'Another trtexec is live or process check failed: ' + result.stdout)


def stop(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=2)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def format_observations(layers):
    observations = []
    convs = [row for row in layers if 'Convolution' in row.get('LayerType', '')]
    for name, requested in [(FIRST, 'FP16'), (SECOND, 'INT8')]:
        matching = [row for row in convs if re.search(re.escape(name) + r'(?=$|[\s,+|)])', row.get('Name', ''))]
        item = {'source_conv': name, 'requested_precision': requested, 'native_matches': len(matching),
                'requested_formats_verified': False, 'scope': 'Visible native formats only; no inference or whole-model precision claim'}
        if len(matching) == 1:
            row = matching[0]
            inputs = [value.get('Format/Datatype', '') for value in row.get('Inputs', [])]
            outputs = [value.get('Format/Datatype', '') for value in row.get('Outputs', [])]
            weights = row.get('Weights', {})
            expected_count = 64 * 48 if requested == 'FP16' else 64 * 64 * 3 * 3
            if requested == 'FP16':
                verified = (weights.get('Count') == expected_count and
                            weights.get('Type', '').lower() in ('half', 'fp16', 'float16') and inputs and outputs and
                            all('FP16' in value.upper() for value in inputs + outputs))
            else:
                index = row.get('ConvXAsActInputIdx', 0)
                index = index if isinstance(index, int) and index >= 0 else 0
                verified = (weights.get('Count') == expected_count and
                            weights.get('Type', '').lower() in ('int8', 'int8_t', 'signed int8', 'signed byte') and
                            len(inputs) > index and 'INT8' in inputs[index].upper())
            item.update(native_layer=row['Name'], native_layer_type=row['LayerType'],
                        weights=weights, expected_source_weight_count=expected_count,
                        input_formats=inputs, output_formats=outputs,
                        tactic_name=row.get('TacticName'), tactic_value=row.get('TacticValue'),
                        requested_formats_verified=bool(verified))
        observations.append(item)
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--result-name', default='results')
    parser.add_argument('--check-only', action='store_true', help='Verify inputs/commands only, without importing GPU libraries')
    args = parser.parse_args()
    root = args.root.resolve()
    spec = verify_bundle(root)
    result = (root / args.result_name).resolve()
    require(result.parent == root and result != root, 'Results must be a fresh direct child directory')
    if args.check_only:
        print(json.dumps({'status': 'MIXED_PROBE_INPUTS_VERIFIED_ONLY', 'graph_sha256': spec['graph_sha256'],
                          'commands': {name: command_for(root, result / (name + '.engine'), name) for name in VARIANTS},
                          'inference_run': False}, indent=2), flush=True)
        return
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Real Jetson Linux aarch64 required')

    def terminate(_signum, _frame):
        raise KeyboardInterrupt('Probe termination requested')

    signal.signal(signal.SIGTERM, terminate)
    import fcntl
    lock = (root.parent / '.gate05_mixed_constraint_probe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require_no_trtexec()
    initial = snapshot()
    require(initial['available_kib'] >= 3 * 1024 * 1024, 'Less than 3 GiB available before diagnostic probe')
    result.mkdir(exist_ok=False)
    started = time.monotonic()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    rows = []
    summary = {'schema': 'rampq.gate05.mixed_constraint_probe_results/v1', 'status': 'RUNNING',
               'boot_id': boot, 'spec_sha256': sha(root / 'spec.json'), 'manifest_sha256': sha(root / 'manifest.json'),
               'graph_sha256': sha(root / 'model.onnx'), 'runner_sha256': sha(Path(__file__)),
               'results': rows, 'inference_run': False, 'dataset_evaluation_run': False,
               'full_model_precision_pass_claim': False, 'original_numerical_parity_status': 'LOCAL_QDQ_PREFLIGHT_FAILED'}
    save(result / 'summary.json', summary)
    collector, process = None, None
    kernel = result / 'kernel.log'
    try:
        import tensorrt as trt
        require(trt.__version__.startswith('8.5.'), 'Expected TensorRT 8.5')
        summary['tensorrt'] = trt.__version__
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        with kernel.open('wb', buffering=0) as kernel_handle:
            collector = subprocess.Popen(['journalctl', '-k', '-b', '-f', '-n', '0', '-o', 'short-monotonic', '--no-pager'],
                                         stdout=kernel_handle, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
            time.sleep(0.5)
            for name in VARIANTS:
                remaining = 360 - (time.monotonic() - started) - 5
                if remaining <= 0:
                    summary['status'] = 'TOTAL_BUILD_TIME_LIMIT_REACHED'
                    break
                require_no_trtexec()
                initial = safe_to_continue(boot, kernel, collector)
                require(initial['available_kib'] >= 3 * 1024 * 1024, 'Less than 3 GiB available before next variant')
                engine_path, log_path = result / (name + '.engine'), result / (name + '.log')
                command = command_for(root, engine_path, name)
                cap = min(90.0, remaining)
                variant_started = time.monotonic()
                row = {'variant': name, 'command': command, 'max_seconds': cap, 'initial_resources': initial,
                       'build_status': 'RUNNING', 'inference_run': False}
                rows.append(row)
                save(result / 'summary.json', summary)
                with log_path.open('wb', buffering=0) as handle:
                    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, start_new_session=True)
                    try:
                        low_memory = 0
                        while process.poll() is None:
                            sample = safe_to_continue(boot, kernel, collector)
                            low_memory = low_memory + 1 if sample['available_kib'] < 384 * 1024 else 0
                            require(low_memory < 3, 'Available memory stayed below 384 MiB')
                            elapsed = time.monotonic() - variant_started
                            if elapsed >= cap:
                                stop(process)
                                row['build_status'] = 'TIMEOUT_INCONCLUSIVE'
                                break
                            time.sleep(min(2, max(0.01, cap - elapsed)))
                    finally:
                        if process.poll() is None:
                            stop(process)
                code = process.poll()
                row.update(returncode=code, seconds=round(time.monotonic() - variant_started, 3),
                           log_sha256=sha(log_path))
                process = None
                require_no_trtexec()
                safe_to_continue(boot, kernel, collector)
                if row['build_status'] != 'TIMEOUT_INCONCLUSIVE':
                    row['build_status'] = 'BUILD_FAILED' if code != 0 else 'BUILD_RETURNED_ZERO'
                if code == 0 and engine_path.is_file() and row['build_status'] != 'TIMEOUT_INCONCLUSIVE':
                    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
                    require(engine is not None, 'Probe engine deserialization failed')
                    inspector = engine.create_engine_inspector()
                    layers = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
                    (result / (name + '_layers.json')).write_text(layers, encoding='utf-8')
                    bindings = [{'name': engine.get_binding_name(i), 'input': bool(engine.binding_is_input(i)),
                                 'shape': list(engine.get_binding_shape(i)), 'dtype': str(engine.get_binding_dtype(i)),
                                 'format': str(engine.get_binding_format(i))} for i in range(engine.num_bindings)]
                    row.update(build_status='BUILT_AND_DESERIALIZED', engine_sha256=sha(engine_path),
                               engine_bytes=engine_path.stat().st_size, native_bindings=bindings,
                               native_conv_format_observations=format_observations(json.loads(layers)['Layers']),
                               layers_sha256=sha(result / (name + '_layers.json')))
                    expected = [(spec['external_input']['name'], True, spec['external_input']['shape']),
                                (spec['external_output']['name'], False, spec['external_output']['shape'])]
                    row['external_contract_verified'] = (len(bindings) == 2 and
                        sorted((value['name'], value['input'], value['shape']) for value in bindings) == sorted(expected) and
                        all(value['dtype'] == 'DataType.FLOAT' and value['format'] == 'TensorFormat.LINEAR' for value in bindings))
                    del inspector, engine
                else:
                    if code == 0 and row['build_status'] != 'TIMEOUT_INCONCLUSIVE':
                        row['build_status'] = 'BUILD_FAILED_MISSING_ENGINE'
                    row['errors'] = [line for line in log_path.read_text(errors='replace').splitlines() if '[E]' in line][-10:]
                save(result / 'summary.json', summary)
                print(json.dumps({key: value for key, value in row.items() if key != 'native_conv_format_observations'}), flush=True)
            if summary['status'] == 'RUNNING':
                summary['status'] = 'FOUR_BUILD_DIAGNOSTICS_FINISHED_NATIVE_FORMAT_REVIEW_REQUIRED'
    except BaseException as error:
        summary['status'] = 'PROBE_STOPPED_REQUIRES_INSPECTION'
        summary['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        if process is not None:
            stop(process)
        if collector is not None:
            stop(collector)
        summary['elapsed_seconds'] = round(time.monotonic() - started, 3)
        save(result / 'summary.json', summary)
        files = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(result.iterdir()) if path.is_file() and path.name != 'manifest.json'}
        save(result / 'manifest.json', {'schema': 'rampq.gate05.mixed_constraint_probe_evidence/v1', 'files': files})
        print(json.dumps({'status': summary['status'], 'result': str(result),
                          'elapsed_seconds': summary['elapsed_seconds'], 'inference_run': False}), flush=True)


if __name__ == '__main__':
    main()
