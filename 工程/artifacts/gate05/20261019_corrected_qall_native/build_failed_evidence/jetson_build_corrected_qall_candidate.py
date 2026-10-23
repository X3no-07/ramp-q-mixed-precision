"""Build one frozen 88-INT8 code-Clip q_all diagnostic candidate on Jetson.

--check-input-only validates frozen inputs without importing TensorRT or running
the graph. Native execution requires the external monitored private process group.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time
import traceback

import jetson_build_precision_candidate as helper


EXPECTED_HELPER_SHA256 = 'd8431345da90482f0d5798e2d6524bf03591ecef653edfad5ef3e28a509eb19f'
EXPECTED_GRAPH_SHA256 = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
EXPECTED_PARENT_GRAPH_SHA256 = '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d'
EXPECTED_REWRITE_AUDIT_SHA256 = '842eff15adfedd75735815cbe4040d17a645653e3b62511149afacbf8f923be9'
EXPECTED_REFERENCE_CONTRACT_SHA256 = '87fcbf04e88fd311784736cf77cddfdd82fc2608ed3de19882467e58707fb430'
EXPECTED_SPEC_SHA256 = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
EXPECTED_MANIFEST_SHA256 = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
BUDGET_RATIONALE = ('One bounded diagnostic build only. The original 88-INT8 q_all native build took '
                    'about 55 minutes, exceeding the old 1800-second mixed-precision candidate limit. '
                    '7200 seconds is an explicit separate build budget, not a numerical tolerance change.')


def validate_inputs(root):
    assert helper.sha(Path(helper.__file__)) == EXPECTED_HELPER_SHA256, 'Original build helper changed'
    assert helper.sha(root / 'spec.json') == EXPECTED_SPEC_SHA256, 'Frozen spec changed'
    assert helper.sha(root / 'manifest.json') == EXPECTED_MANIFEST_SHA256, 'Frozen manifest changed'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, info in manifest['files'].items():
        path = (root / name).resolve()
        assert path.parent == root and path.stat().st_size == info['bytes'] and helper.sha(path) == info['sha256'], name
    spec = json.loads((root / 'spec.json').read_text(encoding='utf-8'))
    assert spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1'
    assert spec['configuration'] == 'q_all' and spec['precisionConstraints'] == 'obey'
    assert spec['quantized_conv_count'] == 88 and spec['fp16_conv_count'] == 0
    assert len(spec['requests']) == len({row['onnx_node'] for row in spec['requests']}) == 88
    assert all(row['precision'] == 'INT8' for row in spec['requests'])
    assert spec['activation_quantizer_topology'] == 'Q_INT8_CODE_CLIP_DQ'
    assert spec['requires_native_clip_importer_correction'] is True
    assert len(spec['clip_names']) == len(set(spec['clip_names'])) == 88
    assert spec['workspace_mib'] == 1024 and spec['avg_timing'] == 1 and spec['max_build_seconds'] == 7200
    assert spec['diagnostic_only'] is True
    assert spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED'
    assert spec['graph_sha256'] == EXPECTED_GRAPH_SHA256
    assert spec['parent_graph_sha256'] == EXPECTED_PARENT_GRAPH_SHA256
    assert spec['semantic_rewrite_audit_sha256'] == EXPECTED_REWRITE_AUDIT_SHA256
    assert spec['common_preparation_sha256'] == spec['semantic_rewrite_audit_sha256']
    assert spec['common_preparation_sha256'] != spec['parent_common_preparation_sha256']
    assert spec['native_diagnostic_reference_contract_sha256'] == EXPECTED_REFERENCE_CONTRACT_SHA256
    assert helper.sha(root / 'reference_contract.json') == EXPECTED_REFERENCE_CONTRACT_SHA256
    contract = json.loads((root / 'reference_contract.json').read_text(encoding='utf-8'))
    assert contract['not_a_gate_amendment'] is True and contract['gate05_status'] == 'INCOMPLETE'
    assert contract['protected_failure']['original_complete_model_guard'] == 'FAILED_UNCHANGED'
    assert contract['protected_failure']['tolerances_changed'] is False
    assert contract['candidate_rewrite']['quantized_convs'] == contract['candidate_rewrite']['native_clip_count'] == 88
    assert contract['native_build']['attempts'] == 1 and contract['native_build']['max_seconds'] == 7200
    assert contract['native_build']['new_precisionConstraints'] == 'obey'
    assert contract['native_build']['explicit_restored_fp16_layer_constraints'] == 0
    for filename, key in [('model.onnx', 'graph_sha256'), ('graph_audit.json', 'graph_audit_sha256'),
                          ('module_map.json', 'module_map_sha256'), ('preflight.json', 'failed_preflight_sha256'),
                          ('diagnostic_scope.json', 'diagnostic_scope_sha256'),
                          ('semantic_rewrite_audit.json', 'semantic_rewrite_audit_sha256')]:
        assert helper.sha(root / filename) == spec[key], filename
    preflight = json.loads((root / 'preflight.json').read_text(encoding='utf-8'))
    assert preflight['status'] == spec['original_local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    scope = json.loads((root / 'diagnostic_scope.json').read_text(encoding='utf-8'))
    assert scope['status'] == 'DIAGNOSTIC_MATRIX_BUILD_AND_FIXED_IMAGE_ONLY'
    assert scope['original_tolerances_changed'] is False and scope['dataset_evaluation_run'] is False
    assert scope['failed_preflight_sha256'] == spec['failed_preflight_sha256']
    return spec, preflight


def tool_identity(root):
    return {'builder_file': Path(__file__).name, 'builder_sha256': helper.sha(Path(__file__)),
            'helper_file': Path(helper.__file__).name, 'helper_sha256': helper.sha(Path(helper.__file__)),
            'direct_runtime_python_files': {path.name: {'bytes': path.stat().st_size, 'sha256': helper.sha(path)}
                                            for path in sorted(root.glob('*.py'))}}


def tensor_info(value):
    return {'name': value.name, 'shape': list(value.shape), 'dtype': str(value.dtype)}


def finite_or_repr(value):
    return value if math.isfinite(value) else repr(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--check-input-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    spec, preflight = validate_inputs(root)
    identity = tool_identity(root)
    if args.check_input_only:
        print(json.dumps({'status': 'FROZEN_QALL_INPUTS_VERIFIED_NO_NATIVE_EXECUTION',
                          'configuration': 'q_all', 'onnx_sha256': spec['graph_sha256'],
                          'spec_sha256': helper.sha(root / 'spec.json'), 'manifest_sha256': helper.sha(root / 'manifest.json'),
                          'quantized_conv_count': 88, 'fp16_conv_count': 0, 'clip_correction_count': 88,
                          'requested_precision_constraints_mode': 'obey', 'max_build_seconds': 7200,
                          'budget_rationale': BUDGET_RATIONALE, 'tool_identity': identity,
                          'native_execution_run': False, 'original_complete_model_numerical_guard': 'FAILED_UNCHANGED'}, indent=2))
        return
    assert platform.system() == 'Linux' and platform.machine() == 'aarch64'
    assert helper.query(['pgrep', '-x', 'trtexec'])['returncode'] == 1
    import tensorrt as trt
    assert trt.__version__.startswith('8.5.')
    run = root / 'results' / datetime.now(timezone.utc).strftime('build_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    common = {'configuration': 'q_all', 'onnx_sha256': helper.sha(root / 'model.onnx'),
              'requested_precision_constraints_mode': 'obey', 'spec_sha256': helper.sha(root / 'spec.json'),
              'manifest_sha256': helper.sha(root / 'manifest.json'), 'diagnostic_only': True,
              'local_preflight_status': preflight['status'], 'expected_binding_contract': spec['expected_binding_contract'],
              'primary_output_name': spec['primary_output_name'], 'graph_audit_sha256': spec['graph_audit_sha256'],
              'module_map_sha256': spec['module_map_sha256'], 'native_importer_clip_correction_required': True,
              'parent_graph_sha256': spec['parent_graph_sha256'], 'dataset_evaluation_run': False,
              'common_preparation_sha256': spec['common_preparation_sha256'],
              'native_diagnostic_reference_contract_sha256': EXPECTED_REFERENCE_CONTRACT_SHA256,
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'tool_identity': identity}
    helper.write(run / 'tool_hashes.json', identity)
    helper.write(run / 'environment.json', {'tensorrt': trt.__version__, 'python': platform.python_version(),
          'machine': platform.machine(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'power_mode': helper.query(['/usr/sbin/nvpmodel', '-q']),
          'memory': helper.query(['free', '-m']), 'fan_service': helper.query(['systemctl', 'is-active', 'nvfancontrol.service']),
          'jetson_linux': Path('/etc/nv_tegra_release').read_text()})
    helper.write(root / 'latest_build.json', dict(common, run_dir=str(run), status='BUILD_RUNNING'))
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
        assert {layer.name for layer in layers if layer.type == trt.LayerType.CONVOLUTION} == {row['onnx_node'] for row in requests}
        # These are imported parser values before any explicit correction.
        before = []
        for name in spec['clip_names']:
            clip = by_name[name]
            assert clip.type == trt.LayerType.ACTIVATION
            clip.__class__ = trt.IActivationLayer
            assert clip.type == trt.ActivationType.CLIP and clip.num_inputs == clip.num_outputs == 1
            before.append({'name': name, 'activation_type': str(clip.type),
                           'alpha': finite_or_repr(clip.alpha), 'beta': finite_or_repr(clip.beta),
                           'inputs': [tensor_info(clip.get_input(0))], 'outputs': [tensor_info(clip.get_output(0))]})
        helper.write(run / 'parsed_clip_attributes_before_correction.json', {'count': len(before), 'layers': before})
        corrections = []
        for row in before:
            clip = by_name[row['name']]
            clip.__class__ = trt.IActivationLayer
            clip.alpha, clip.beta = -127., 127.
            assert clip.alpha == -127. and clip.beta == 127.
            corrections.append({'name': row['name'], 'before': {'alpha': row['alpha'], 'beta': row['beta']},
                                'after': {'alpha': clip.alpha, 'beta': clip.beta}})
        helper.write(run / 'parsed_clip_attributes_after_correction.json', {'count': len(corrections), 'layers': corrections})
        assert len(corrections) == 88
        parsed_layers = []
        for layer in layers:
            parsed_layers.append({'name': layer.name, 'type': str(layer.type),
                                  'inputs': [tensor_info(layer.get_input(i)) for i in range(layer.num_inputs)
                                             if layer.get_input(i) is not None],
                                  'outputs': [tensor_info(layer.get_output(i)) for i in range(layer.num_outputs)]})
        for row in requests:
            layer = by_name[row['onnx_node']]
            assert layer.type == trt.LayerType.CONVOLUTION and layer.num_outputs == 1 and row['precision'] == 'INT8'
        # Q/DQ carries the INT8 request. The parent q_all has no Half restore
        # requests, so no layer.precision or output-type constraint is injected.
        imported = []
        for is_input, count in ((True, network.num_inputs), (False, network.num_outputs)):
            for index in range(count):
                value = network.get_input(index) if is_input else network.get_output(index)
                value.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
                imported.append(dict(tensor_info(value), input=is_input))
        assert helper.contract_signature(imported) == helper.contract_signature(spec['expected_binding_contract'])
        config = builder.create_builder_config()
        for flag in (trt.BuilderFlag.INT8, trt.BuilderFlag.FP16, trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS):
            config.set_flag(flag)
        config.clear_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
        config.avg_timing_iterations = 1
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        helper.write(run / 'parsed_network_validation.json', {'status': 'ALL_REQUESTED_CONV_NAMES_EXIST_IN_PARSED_NETWORK',
              'source_conv_count': 88, 'requests': requests, 'bindings_in_network_order': imported,
              'parsed_layers': parsed_layers, 'clip_importer_corrections': corrections,
              'restored_fp16_compute_and_output_constraints': [], 'explicit_layer_precision_constraint_count': 0})
        helper.write(run / 'build_request.json', dict(common, command=['TensorRT Python Builder.build_serialized_network'],
              builder_flags=['INT8', 'FP16', 'OBEY_PRECISION_CONSTRAINTS', 'PREFER_PRECISION_CONSTRAINTS_DISABLED', 'TF32_DISABLED'],
              workspace_mib=1024, avg_timing_iterations=1, max_build_seconds=7200, build_budget_rationale=BUDGET_RATIONALE,
              restored_fp16_compute_and_output_constraints=[], explicit_layer_precision_constraint_count=0,
              obey_flag_scope='Parent obey mode retained. There are zero explicit per-layer Half/INT8 constraints; Q/DQ specifies quantization. Actual native adoption is audited separately.',
              explicit_clip_importer_corrections=corrections,
              actual_precision_requirement='Unchanged complete per-source native format, named-integer-tactic and fusion criteria. Parsed requests are not adoption proof; seven prior depthwise tactic gaps are not waived.',
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
        common.update(engine_sha256=helper.sha(run / 'model.engine'), engine_bytes=(run / 'model.engine').stat().st_size,
                      bindings=bindings, engine_layers=engine.num_layers)
        (run / 'engine_layers.json').write_text(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        helper.write(run / 'bindings.json', {'bindings': bindings})
        assert helper.contract_signature(bindings) == helper.contract_signature(spec['expected_binding_contract'])
        assert all(row['format'] == 'TensorFormat.LINEAR' for row in bindings)
        report = dict(common, status='BUILT_AND_DESERIALIZED', elapsed_seconds=time.monotonic() - started,
                      actual_precision_audit='PENDING', single_image_inference='NOT_RUN', dataset_evaluation='NOT_RUN')
    except BaseException as error:
        report = dict(common, status='BUILD_OR_CONTRACT_VALIDATION_FAILED', elapsed_seconds=time.monotonic() - started,
                      error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc(),
                      single_image_inference='NOT_RUN', dataset_evaluation='NOT_RUN')
        helper.write(run / 'build_result.json', report)
        helper.write(root / 'latest_build.json', dict(report, run_dir=str(run)))
        raise
    helper.write(run / 'build_result.json', report)
    helper.write(root / 'latest_build.json', dict(report, run_dir=str(run)))
    print('CORRECTED_QALL_CANDIDATE_BUILT_NO_NUMERICAL_PASS_CLAIM', flush=True)


if __name__ == '__main__':
    main()
