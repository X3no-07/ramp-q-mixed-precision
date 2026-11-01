"""Build one frozen q_only_G1 code-Clip diagnostic candidate on Jetson."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

from jetson_build_precision_candidate import contract_signature, query, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    assert platform.system() == 'Linux' and platform.machine() == 'aarch64'
    for name, info in json.loads((root / 'manifest.json').read_text())['files'].items():
        path = (root / name).resolve()
        assert path.parent == root and sha(path) == info['sha256'] and path.stat().st_size == info['bytes']
    spec = json.loads((root / 'spec.json').read_text())
    assert spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1'
    assert spec['configuration'] == 'q_only_G1' and spec['precisionConstraints'] == 'prefer'
    assert spec['quantized_conv_count'] == 6 and spec['fp16_conv_count'] == 82
    assert spec['requires_native_clip_importer_correction'] is True and len(spec['clip_names']) == 6
    assert spec['workspace_mib'] == 1024 and spec['avg_timing'] == 1 and spec['max_build_seconds'] == 1800
    assert spec['diagnostic_only'] is True
    for filename, key in [('model.onnx', 'graph_sha256'), ('graph_audit.json', 'graph_audit_sha256'),
                          ('module_map.json', 'module_map_sha256'), ('preflight.json', 'failed_preflight_sha256'),
                          ('diagnostic_scope.json', 'diagnostic_scope_sha256')]:
        assert sha(root / filename) == spec[key], filename
    preflight = json.loads((root / 'preflight.json').read_text())
    assert preflight['status'] == spec['original_local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    scope = json.loads((root / 'diagnostic_scope.json').read_text())
    assert scope['status'] == 'DIAGNOSTIC_MATRIX_BUILD_AND_FIXED_IMAGE_ONLY'
    assert scope['original_tolerances_changed'] is False and scope['dataset_evaluation_run'] is False
    assert scope['failed_preflight_sha256'] == spec['failed_preflight_sha256']
    assert query(['pgrep', '-x', 'trtexec'])['returncode'] == 1
    import tensorrt as trt
    assert trt.__version__.startswith('8.5.')
    run = root / 'results' / datetime.now(timezone.utc).strftime('build_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    common = {'configuration': spec['configuration'], 'onnx_sha256': sha(root / 'model.onnx'),
              'requested_precision_constraints_mode': 'prefer', 'spec_sha256': sha(root / 'spec.json'),
              'manifest_sha256': sha(root / 'manifest.json'), 'diagnostic_only': True,
              'local_preflight_status': preflight['status'], 'expected_binding_contract': spec['expected_binding_contract'],
              'primary_output_name': spec['primary_output_name'], 'graph_audit_sha256': spec['graph_audit_sha256'],
              'module_map_sha256': spec['module_map_sha256'], 'native_importer_clip_correction_required': True,
              'parent_graph_sha256': spec['parent_graph_sha256'], 'dataset_evaluation_run': False}
    write(run / 'environment.json', {'tensorrt': trt.__version__, 'python': platform.python_version(),
          'machine': platform.machine(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'power_mode': query(['/usr/sbin/nvpmodel', '-q']),
          'memory': query(['free', '-m']), 'fan_service': query(['systemctl', 'is-active', 'nvfancontrol.service']),
          'jetson_linux': Path('/etc/nv_tegra_release').read_text()})
    write(root / 'latest_build.json', dict(common, run_dir=str(run), status='BUILD_RUNNING'))
    print('RESULT_DIR:', run, flush=True)
    started = time.monotonic()
    try:
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        importer = trt.OnnxParser(network, logger)
        assert importer.parse((root / 'model.onnx').read_bytes()), [str(importer.get_error(i)) for i in range(importer.num_errors)]
        layers = [network.get_layer(i) for i in range(network.num_layers)]
        by_name = {layer.name: layer for layer in layers}
        assert len(by_name) == len(layers)
        requests = spec['requests']
        assert len(requests) == len({row['onnx_node'] for row in requests}) == 88
        assert {layer.name for layer in layers if layer.type == trt.LayerType.CONVOLUTION} == {row['onnx_node'] for row in requests}
        corrections, restored = [], []
        for name in spec['clip_names']:
            clip = by_name[name]
            assert clip.type == trt.LayerType.ACTIVATION
            clip.__class__ = trt.IActivationLayer
            assert clip.type == trt.ActivationType.CLIP
            before = {key: value if math.isfinite(value) else repr(value)
                      for key, value in (('alpha', clip.alpha), ('beta', clip.beta))}
            clip.alpha, clip.beta = -127., 127.
            corrections.append({'name': name, 'before': before, 'after': {'alpha': clip.alpha, 'beta': clip.beta}})
        for row in requests:
            layer = by_name[row['onnx_node']]
            assert layer.type == trt.LayerType.CONVOLUTION and layer.num_outputs == 1
            assert row['precision'] == ('INT8' if row['group'] == 'G1' else 'FP16')
            if row['precision'] == 'FP16':
                layer.precision = trt.float16
                layer.set_output_type(0, trt.float16)
                restored.append(row['onnx_node'])
        assert len(restored) == 82
        imported = []
        for is_input, count in ((True, network.num_inputs), (False, network.num_outputs)):
            for index in range(count):
                value = network.get_input(index) if is_input else network.get_output(index)
                value.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
                imported.append({'name': value.name, 'input': is_input, 'shape': list(value.shape), 'dtype': str(value.dtype)})
        assert contract_signature(imported) == contract_signature(spec['expected_binding_contract'])
        config = builder.create_builder_config()
        for flag in (trt.BuilderFlag.INT8, trt.BuilderFlag.FP16, trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS):
            config.set_flag(flag)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
        config.avg_timing_iterations = 1
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        write(run / 'parsed_network_validation.json', {'status': 'ALL_REQUESTED_CONV_NAMES_EXIST_IN_PARSED_NETWORK',
              'source_conv_count': 88, 'requests': requests, 'bindings_in_network_order': imported,
              'clip_importer_corrections': corrections, 'restored_fp16_compute_and_output_constraints': restored})
        write(run / 'build_request.json', dict(common, command=['TensorRT Python Builder.build_serialized_network'],
              builder_flags=['INT8', 'FP16', 'PREFER_PRECISION_CONSTRAINTS', 'TF32_DISABLED'],
              workspace_mib=1024, avg_timing_iterations=1, restored_fp16_compute_and_output_constraints=restored,
              explicit_clip_importer_corrections=corrections,
              actual_precision_requirement='Original complete per-source format and named-integer-tactic criteria; imported requests are not adoption proof.',
              limits='One new diagnostic derivative. Original model, calibration, numerical guard and frozen engines unchanged.'))
        serialized = builder.build_serialized_network(network, config)
        assert serialized is not None
        (run / 'model.engine').write_bytes(bytes(serialized))
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine((run / 'model.engine').read_bytes())
        assert engine is not None
        bindings = [{'name': engine.get_binding_name(i), 'input': bool(engine.binding_is_input(i)),
                     'shape': list(engine.get_binding_shape(i)), 'dtype': str(engine.get_binding_dtype(i)),
                     'format': str(engine.get_binding_format(i))} for i in range(engine.num_bindings)]
        common.update(engine_sha256=sha(run / 'model.engine'), engine_bytes=(run / 'model.engine').stat().st_size,
                      bindings=bindings, engine_layers=engine.num_layers)
        (run / 'engine_layers.json').write_text(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        assert contract_signature(bindings) == contract_signature(spec['expected_binding_contract'])
        assert all(row['format'] == 'TensorFormat.LINEAR' for row in bindings)
        report = dict(common, status='BUILT_AND_DESERIALIZED', elapsed_seconds=time.monotonic() - started,
                      actual_precision_audit='PENDING', single_image_inference='NOT_RUN', dataset_evaluation='NOT_RUN')
    except BaseException as error:
        report = dict(common, status='BUILD_OR_CONTRACT_VALIDATION_FAILED', elapsed_seconds=time.monotonic() - started,
                      error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc(),
                      single_image_inference='NOT_RUN', dataset_evaluation='NOT_RUN')
        write(run / 'build_result.json', report)
        write(root / 'latest_build.json', dict(report, run_dir=str(run)))
        raise
    write(run / 'build_result.json', report)
    write(root / 'latest_build.json', dict(report, run_dir=str(run)))
    print('CORRECTED_QONLY_CANDIDATE_BUILT_NO_NUMERICAL_PASS_CLAIM', flush=True)


if __name__ == '__main__':
    main()
