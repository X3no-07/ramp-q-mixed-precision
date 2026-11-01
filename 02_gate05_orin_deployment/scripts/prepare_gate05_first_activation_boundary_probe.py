"""Prepare exact first activation Clip/Q/DQ and a no-Clip boundary diagnostic.

Only the two isolated CPU quantizers are executed. The original full candidate,
calibration, criteria and complete-model numerical failure remain unchanged.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXED_IMAGE_SHA256 = '494bbd9d09de29f3a992d434da03c252b4f6cf1f28099fd5a443d0c797f5ec52'
ARCHIVE_ROOT = 'gate05_first_activation_boundary_probe_20261019'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def message_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def manifest_files(folder):
    manifest = read(folder / 'manifest.json')
    paths = [folder / 'manifest.json']
    for name, expected in manifest['files'].items():
        path = folder / name
        require(path.resolve().parent == folder.resolve() and not path.is_symlink() and info(path) == expected,
                'Frozen source file differs: ' + name)
        paths.append(path)
    return paths


def source_hashes(paths):
    return {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): info(path) for path in sorted(set(paths))}


def summary(value):
    import numpy as np
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'all_finite': bool(np.isfinite(value).all()),
            'min': float(value.min()), 'max': float(value.max()),
            'raw_array_bytes_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/gate05/20261019_first_activation_boundary_probe')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Choose a fresh output; preserve previous probes')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort

    bundle = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles_prefer/q_only_G1'
    prior = ROOT / 'artifacts/gate05/20261019_qonly_fixed_image_diagnostic'
    image = ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1/000020.jpg'
    source_paths = manifest_files(bundle) + manifest_files(prior) + [image, image.parent / 'manifest.json', Path(__file__).resolve()]
    spec, mapping = read(bundle / 'spec.json'), read(bundle / 'module_map.json')
    diagnostic, prior_manifest = read(prior / 'diagnostic.json'), read(prior / 'manifest.json')
    require(spec['configuration'] == 'q_only_G1' and spec['original_local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED',
            'Original source identity/failure differs')
    require(sha(image) == FIXED_IMAGE_SHA256 == diagnostic['image_sha256'] ==
            read(image.parent / 'manifest.json')['files']['000020.jpg']['sha256'], 'Frozen image SHA differs')
    require(sha(bundle / 'model.onnx') == spec['graph_sha256'] == diagnostic['onnx_sha256'] and
            sha(bundle / 'module_map.json') == spec['module_map_sha256'], 'Source graph/module map differs')
    input_source = prior / 'preprocessed_input.npy'
    require(sha(input_source) == prior_manifest['files']['preprocessed_input.npy']['sha256'], 'Frozen input bytes differ')
    before = source_hashes(source_paths)
    source = onnx.load(str(bundle / 'model.onnx'))
    producers = {name: node for node in source.graph.node for name in node.output}
    node_by_name = {node.name: node for node in source.graph.node}
    initializers = {value.name: value for value in source.graph.initializer}
    conv = next(node for node in source.graph.node if node.op_type == 'Conv')
    mapped = [row for row in mapping['mapping'] if row['onnx_node'] == conv.name]
    requests = [row for row in spec['requests'] if row['onnx_node'] == conv.name]
    require(len(mapped) == len(requests) == 1 and mapped[0]['quantized'] is True and
            mapped[0]['group'] == requests[0]['group'] == 'G1' and requests[0]['precision'] == 'INT8',
            'Actual first Conv is not the frozen mapped G1 quantized Conv')
    dq = producers[conv.input[0]]
    require(dq.op_type == 'DequantizeLinear' and len(dq.input) == 3, 'First Conv input is not frozen DQ')
    q = producers[dq.input[0]]
    require(q.op_type == 'QuantizeLinear' and len(q.input) == 3, 'Frozen activation Q missing')
    clip = producers[q.input[0]]
    require(clip.op_type == 'Clip' and len(clip.input) == 3, 'Frozen activation Clip missing')
    input_value = source.graph.input[0]
    require(len(source.graph.input) == 1 and input_value.name == requests[0]['original_activation_input'] == clip.input[0] and
            input_value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
            [dim.dim_value for dim in input_value.type.tensor_type.shape.dim] == [1, 3, 640, 640], 'Input declaration differs')
    require(list(q.input[1:]) == list(dq.input[1:]) and q.input[1] in initializers and q.input[2] in initializers,
            'Q/DQ do not share the original scale and zero point')
    require(not q.attribute and not dq.attribute, 'Expected original scalar activation Q/DQ without an axis attribute')
    scale = numpy_helper.to_array(initializers[q.input[1]]).copy()
    zero = numpy_helper.to_array(initializers[q.input[2]]).copy()
    lower = numpy_helper.to_array(initializers[clip.input[1]]).copy()
    upper = numpy_helper.to_array(initializers[clip.input[2]]).copy()
    require(scale.dtype == np.float32 and scale.ndim == 0 and float(scale) > 0 and
            zero.dtype == np.int8 and zero.ndim == 0 and int(zero) == 0, 'Frozen signed scalar quantizer contract differs')
    require(lower.dtype == upper.dtype == np.float32 and lower.ndim == upper.ndim == 0 and
            float(lower) == -1.0 and float(upper) == 1.0, 'Expected exact frozen [-1,+1] Clip bounds')
    output_value = copy.deepcopy(input_value)
    output_value.name = dq.output[0]
    selected_names = {clip.name, q.name, dq.name}
    clip_nodes = [copy.deepcopy(node) for node in source.graph.node if node.name in selected_names]
    require([node.op_type for node in clip_nodes] == ['Clip', 'QuantizeLinear', 'DequantizeLinear'], 'Unexpected three-node order')
    clip_constants = {name for node in clip_nodes for name in node.input if name in initializers}
    require(clip_constants == {q.input[1], q.input[2], clip.input[1], clip.input[2]}, 'Unexpected quantizer constants')
    models = {}
    for variant in ('clip', 'no_clip'):
        nodes = copy.deepcopy(clip_nodes if variant == 'clip' else [q, dq])
        if variant == 'no_clip':
            nodes[0].input[0] = input_value.name
        needed = {name for node in nodes for name in node.input if name in initializers}
        constants = [copy.deepcopy(value) for value in source.graph.initializer if value.name in needed]
        require(len(nodes) == (3 if variant == 'clip' else 2) and len(constants) == (4 if variant == 'clip' else 2),
                'Unexpected isolated graph inventory')
        model = copy.deepcopy(source)
        model.graph.CopyFrom(helper.make_graph(nodes, 'first_activation_boundary_' + variant,
                                               [copy.deepcopy(input_value)], [copy.deepcopy(output_value)], constants))
        for node in model.graph.node:
            expected = copy.deepcopy(node_by_name[node.name])
            if variant == 'no_clip' and node.name == q.name:
                expected.input[0] = input_value.name
            require(node.SerializeToString() == expected.SerializeToString(), 'Unexpected node modification')
        require(all(value.SerializeToString() == initializers[value.name].SerializeToString() for value in model.graph.initializer),
                'Quantizer initializer bytes changed')
        onnx.checker.check_model(model, full_check=True)
        models[variant] = model
    require(models['clip'].graph.output[0].SerializeToString() == models['no_clip'].graph.output[0].SerializeToString(),
            'Output declarations differ')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    shutil.copy2(input_source, output / 'preprocessed_input.npy')
    shutil.copy2(bundle / 'preflight.json', output / 'original_failed_preflight.json')
    write(output / 'source_hashes_before.json', before)
    require(input_source.read_bytes() == (output / 'preprocessed_input.npy').read_bytes(), 'Frozen input copy differs')
    x = np.load(output / 'preprocessed_input.npy', allow_pickle=False)
    require(x.shape == (1, 3, 640, 640) and x.dtype == np.float32 and x.flags.c_contiguous and
            bool(np.isfinite(x).all()) and float(x.min()) >= 0 and float(x.max()) <= 1, 'Fixed input is not the expected finite [0,1] array')
    clipped = np.clip(x, lower, upper)
    clip_is_exact_identity = clipped.tobytes(order='C') == x.tobytes(order='C')
    # Explicit float32 normalized values followed by round-to-nearest-even, using
    # the unchanged scalar scale/zero from the source model. No calibration here.
    limits = np.iinfo(np.int8)
    normalized = (x / scale).astype(np.float32)
    codes = np.clip(np.rint(normalized) + zero, limits.min, limits.max).astype(np.int8)
    expected_dequantized = ((codes.astype(np.float32) - zero.astype(np.float32)) * scale).astype(np.float32)
    np.save(output / 'cpu_expected_int8_codes.npy', codes)
    np.save(output / 'cpu_expected_dequantized.npy', expected_dequantized)
    actual, variants, execution = {}, {}, {}
    for variant, model in models.items():
        filename = 'model_activation_' + variant + '.onnx'
        onnx.save(model, str(output / filename))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(output / filename), sess_options=options, providers=['CPUExecutionProvider'])
        require(session.get_providers() == ['CPUExecutionProvider'] and
                [value.name for value in session.get_inputs()] == [input_value.name] and
                [value.name for value in session.get_outputs()] == [output_value.name], 'CPU ORT named I/O/provider mismatch')
        result = session.run([output_value.name], {input_value.name: x})[0]
        # Preserve raw results before any numerical/finite/bit-exact assertion.
        np.save(output / ('cpu_actual_' + variant + '.npy'), result)
        actual[variant + '_dequantized'] = result.copy()
        execution[variant] = {'providers': session.get_providers(), 'graph_optimization': 'ORT_DISABLE_ALL',
                              'runs': 1, 'raw_result_file': 'cpu_actual_' + variant + '.npy',
                              'raw_result_sha256': sha(output / ('cpu_actual_' + variant + '.npy')),
                              'output': summary(result)}
        variants[variant] = {'model_file': filename, 'model_sha256': sha(output / filename),
                             'nodes': len(model.graph.node), 'initializers': len(model.graph.initializer),
                             'expected_output_npz_keys': {output_value.name: variant + '_dequantized'},
                             'selected_node_serialized_sha256': {node.name: message_sha(node) for node in model.graph.node},
                             'selected_initializer_serialized_sha256': {value.name: message_sha(value) for value in model.graph.initializer}}
        del session
    np.savez_compressed(output / 'cpu_expected_outputs.npz', **actual)
    checks = {'clip_strict_identity_on_frozen_input_bytes': clip_is_exact_identity,
              'clip_and_no_clip_outputs_bit_exact': actual['clip_dequantized'].tobytes() == actual['no_clip_dequantized'].tobytes(),
              'clip_output_bit_exact_to_round_even_codes_times_scale': actual['clip_dequantized'].tobytes() == expected_dequantized.tobytes(),
              'no_clip_output_bit_exact_to_round_even_codes_times_scale': actual['no_clip_dequantized'].tobytes() == expected_dequantized.tobytes(),
              'all_cpu_outputs_float32_shape_finite': all(value.dtype == np.float32 and value.shape == (1,3,640,640) and
                                                        bool(np.isfinite(value).all()) for value in actual.values())}
    diagnostic = {'schema': 'rampq.gate05.first_activation_boundary_cpu/v1',
                  'status': 'ISOLATED_ACTIVATION_CPU_EXACT_CHECKS_PASSED' if all(checks.values()) else 'ISOLATED_ACTIVATION_CPU_REVIEW_REQUIRED',
                  'checks': checks, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
                  'fixed_input': summary(x), 'scale_float32': float(scale), 'zero_int8': int(zero),
                  'clip_bounds': [float(lower), float(upper)], 'expected_codes': summary(codes),
                  'expected_dequantized': summary(expected_dequantized), 'cpu_execution': execution,
                  'complete_detector_executions': 0, 'native_execution': False, 'cuda_used': False,
                  'full_model_numerical_guard': 'FAILED_UNCHANGED',
                  'limits': 'Exact operator-level check on one frozen input only. No full-model repair, migration, strict native arithmetic, accuracy or performance conclusion.'}
    write(output / 'cpu_diagnostic.json', diagnostic)
    probe_spec = {'schema': 'rampq.gate05.first_activation_boundary_probe/v1',
                  'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
                  'source_configuration': 'q_only_G1', 'source_graph_sha256': sha(bundle / 'model.onnx'),
                  'source_bundle_manifest_sha256': sha(bundle / 'manifest.json'), 'source_spec_sha256': sha(bundle / 'spec.json'),
                  'source_module_map_sha256': sha(bundle / 'module_map.json'),
                  'source_input_diagnostic_manifest_sha256': sha(prior / 'manifest.json'),
                  'source_image': '000020.jpg', 'source_image_sha256': FIXED_IMAGE_SHA256,
                  'input_file': 'preprocessed_input.npy', 'input_file_sha256': sha(output / 'preprocessed_input.npy'),
                  'input_array_original_bytes_copied': True,
                  'input_binding': {'name': input_value.name, 'input': True, 'shape': [1,3,640,640], 'dtype': 'DataType.FLOAT'},
                  'output_bindings_in_order': [{'name': output_value.name, 'input': False, 'shape': [1,3,640,640],
                                                'dtype': 'DataType.FLOAT', 'role': 'dequantized_activation', 'kind': 'primary'}],
                  'variants': variants, 'cpu_expected_outputs_file': 'cpu_expected_outputs.npz',
                  'cpu_expected_outputs_sha256': sha(output / 'cpu_expected_outputs.npz'),
                  'cpu_expected_int8_codes_file': 'cpu_expected_int8_codes.npy',
                  'cpu_expected_int8_codes_sha256': sha(output / 'cpu_expected_int8_codes.npy'),
                  'cpu_expected_dequantized_file': 'cpu_expected_dequantized.npy',
                  'cpu_expected_dequantized_sha256': sha(output / 'cpu_expected_dequantized.npy'),
                  'cpu_diagnostic_sha256': sha(output / 'cpu_diagnostic.json'),
                  'selected_source_conv_name': conv.name, 'selected_clip_node_name': clip.name,
                  'selected_activation_q_node_name': q.name, 'selected_activation_dq_node_name': dq.name,
                  'source_nodes_serialized_sha256': {node.name: message_sha(node) for node in clip_nodes},
                  'source_initializers_serialized_sha256': {name: message_sha(initializers[name]) for name in sorted(clip_constants)},
                  'no_clip_explicit_changes': {'removed_node': clip.name, 'rewired_node': q.name,
                                               'rewired_input_index': 0, 'original_input': q.input[0], 'new_input': input_value.name,
                                               'original_q_serialized_sha256': message_sha(q),
                                               'rewired_q_serialized_sha256': variants['no_clip']['selected_node_serialized_sha256'][q.name],
                                               'removed_initializers': [clip.input[1],clip.input[2]],
                                               'dq_scale_zero_and_all_remaining_initializer_bytes_unchanged': True,
                                               'reason': 'Only this fixed finite input is in [0,1], so the original exact [-1,+1] Clip is a strict byte-identical identity here.'},
                  'fixed_input_clip_identity_proved': clip_is_exact_identity,
                  'external_graph_output_is_float_dq_only': True,
                  'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
                  'calibration_recomputed': False, 'original_tolerances_changed': False,
                  'dataset_evaluation_run': False, 'native_build_or_execution_run_by_preparer': False,
                  'limits': 'Independent tiny Clip/Q/DQ boundary diagnostic. Removing Clip is an experimental paired variant '
                            'valid for the frozen input; it is not a modification or repair of the full candidate and '
                            'does not establish complete-model numerical migration or native INT8 arithmetic.'}
    write(output / 'spec.json', probe_spec)
    after = source_hashes(source_paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'A frozen source file changed')
    (output / 'README.md').write_text('# First activation Clip/Q/DQ boundary probe\n\nExtract only the first 3 original nodes before the Conv input in the frozen q_only_G1 graph: Clip, ActivationQ, ActivationDQ.\nPair no_clip graph only removes Clip and its unused bound constants, and sets the input[0] of original ActivationQ\nReconnect to original `images`; the original/updated serialization SHA values in the spec are recorded item by item. DQ, scale, zero, and the rest\ninitializer preserves raw bytes. The original complete candidates are not modified.\n\nSame input `preprocessed_input.npy` is a byte-for-byte copy of the original 000020 fixed graph preprocessing array\nFLOAT[1,3,640,640]. Each graph has only one FLOAT output with the original DQ tensor name;\nDo not expose INT8 Q as an external output. Detailed interface by name and CPU NPZ key as specified.\n\nThis input range is [0,1], with the original Clip bounds strictly at [-1,+1], and the raw bytes of the entire array before and after Clip have been verified to be identical.\nBoth small CPU graphs use CPUExecutionProvider and ORT_DISABLE_ALL. First save the raw float32 outputs,\nPerform element-wise raw bytes comparison between the two graphs, and ensure consistency with the explicit CPU expectation of rounding to nearest even followed by multiplication by the original scale.\noriginal INT8 codes are also saved in npy, solely for CPU expectation and not as external output of the graph.\n\nThis experiment does not re-run the full detector, does not connect the board, does not use CUDA, does not calibrate, does not modify thresholds, and does not perform evaluation on 150 images.\nClip removal is a paired numerical diagnostic on this one fixed input; it cannot establish complete-model repair, successful transfer,\nstrict native INT8 arithmetic, accuracy, or performance conclusions. The original full numerical failure is retained.\n',encoding='utf-8')
    files = {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}
    write(output / 'manifest.json', {'schema':'rampq.gate05.first_activation_boundary_probe_bundle/v1','files':files})
    (output / 'SHA256SUMS').write_text(''.join(value['sha256']+'  '+name+'\n' for name,value in files.items()),encoding='utf-8')
    with zipfile.ZipFile(output.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file(): archive.write(path,ARCHIVE_ROOT+'/'+path.name)
    require(all(checks.values()), 'CPU exact operator checks failed; raw outputs and diagnostic were preserved')
    print(json.dumps({'status':diagnostic['status'],'output':str(output),'archive':str(output.with_suffix('.zip')),
                      'archive_bytes':output.with_suffix('.zip').stat().st_size,'archive_sha256':sha(output.with_suffix('.zip')),
                      'spec_sha256':sha(output/'spec.json'),'input_array_sha256':sha(output/'preprocessed_input.npy'),
                      'output_name':output_value.name,'variants':variants,'checks':checks,'all_source_hashes_unchanged':True,
                      'complete_detector_executions':0,'native_execution':False},indent=2))


if __name__ == '__main__':
    main()
