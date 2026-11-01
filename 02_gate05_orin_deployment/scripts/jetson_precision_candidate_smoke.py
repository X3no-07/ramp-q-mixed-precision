"""Run one frozen precision candidate on the fixed Selection image only.

Python 3.8 / TensorRT 8.5 board script. This never evaluates a dataset and never
turns the retained failed independent-model numerical guard into a pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'

CONFIGURATIONS = ('fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5',
                  'r_only_G1_G2', 'q_only_G1')
ALL_GROUPS = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'}
QUANTIZED_GROUPS = {'fp16_control': set(), 'q_all': ALL_GROUPS,
                    'r_only_G1': ALL_GROUPS - {'G1'},
                    'r_only_G1_G5': ALL_GROUPS - {'G1', 'G5'},
                    'r_only_G1_G2': ALL_GROUPS - {'G1', 'G2'}, 'q_only_G1': {'G1'}}
FIXED_IMAGE_SHA256 = '494bbd9d09de29f3a992d434da03c252b4f6cf1f28099fd5a443d0c797f5ec52'
OLD_FP16_ENGINE_SHA256 = '64d91db3063ac28c1552a1ba6443ae28ca45d82d988f5cd375b39fc8cc76b85c'
OLD_FP16_RAW_SHA256 = 'f38d9b113c148198735dfd0a2577c17b93c8cd46ac1f97002a6c0e0a756cf45b'
NMS = {'conf_thres': 0.25, 'iou_thres': 0.7, 'max_det': 1000, 'nc': 10}
SCHEMA = 'rampq.gate05.precision_candidate_smoke/v1'
NVTX_RANGE = 'RAMPQ_QDQ_FIXED_IMAGE'
GRAPH_CONTROL_KEYS = ('common_preparation_sha256', 'common_source_fp32_onnx_sha256',
                      'common_source_weights_sha256', 'common_q_all_pe_tap_reference_sha256',
                      'common_output_declaration_sha256')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding_contract(rows):
    """Normalize the exact static named contract, independently of binding order."""
    names = [row['name'] for row in rows]
    require(len(names) == len(set(names)), 'Duplicate binding name')
    require(sum(row['input'] is True for row in rows) == 1, 'Expected one image input')
    for row in rows:
        require(isinstance(row['input'], bool), 'Binding input flag must be boolean')
        require(row['shape'] and all(type(n) is int and n > 0 for n in row['shape']),
                'Expected positive static binding dimensions')
        require(row['dtype'] in ('DataType.FLOAT', 'DataType.HALF'), 'Unsupported binding dtype')
    return sorted((row['name'], row['input'], tuple(row['shape']), row['dtype']) for row in rows)


def verify_manifest(directory, name='manifest.json'):
    """Validate complete manifest contents without allowing paths outside the bundle."""
    manifest = read(directory / name)
    verified = {}
    for relative, info in manifest['files'].items():
        path = (directory / relative).resolve()
        require(directory.resolve() in path.parents, 'Manifest path escapes bundle: ' + relative)
        require(path.is_file(), 'Missing manifest file: ' + relative)
        require(path.stat().st_size == info['bytes'], 'Manifest byte mismatch: ' + relative)
        require(sha(path) == info['sha256'], 'Manifest SHA256 mismatch: ' + relative)
        verified[relative] = info['sha256']
    return manifest, verified


def verify_requested_precision(precision, spec, diagnostic_formats_only=False):
    """Require complete per-source format inventory; keep tactic evidence separate."""
    quantized, floating = spec['quantized_conv_count'], spec['fp16_conv_count']
    requests = spec['requests']
    require(precision['status'] == 'REQUESTED_CONV_FORMATS_VERIFIED' and
            precision['requested_conv_formats_complete'] is True, 'Requested actual Conv formats incomplete')
    require(not precision['errors'] and not precision['unmapped_engine_convs'],
            'Precision audit has errors or unmapped engine Conv layers')
    require(precision['source_convs'] == 88 and precision['quantized_conv_count'] == quantized and
            precision['fp16_conv_count'] == floating and precision['verified_int8_format_convs'] == quantized and
            precision['verified_fp16_format_convs'] == floating, 'Actual requested-precision inventory incomplete')
    mappings = precision['mapping']
    require(len(mappings) == len({row['onnx_node'] for row in mappings}) == 88,
            'Actual precision mapping does not inventory all 88 unique source Conv nodes')
    mapped = {row['onnx_node']: row for row in mappings}
    require(set(mapped) == {row['onnx_node'] for row in requests}, 'Requested/mapped source Conv identities differ')
    for request in requests:
        observed = mapped[request['onnx_node']]
        require(all(observed[key] == value for key, value in request.items()),
                'Requested/mapped Conv specification differs: ' + request['onnx_node'])
        require(observed['requested_formats_verified'] is True,
                'Requested Conv format not verified: ' + request['onnx_node'])
    strict_status = precision['strict_int8_status']
    strict_ok = strict_status == ('ALL_REQUESTED_INT8_TACTICS_VERIFIED' if quantized else 'NO_INT8_REQUESTED')
    if quantized:
        actual_strict = all(row['strict_int8_tactic_verified'] is True for row in mappings if row['precision'] == 'INT8')
        require(actual_strict == strict_ok, 'Strict tactic status differs from per-Conv evidence')
    if not strict_ok:
        require(quantized > 0 and strict_status == 'INTEGER_TACTIC_REVIEW_REQUIRED' and diagnostic_formats_only,
                'INT8 tactic evidence incomplete; explicit --diagnostic-formats-only is required')
    return strict_ok


def verify_inputs(build_dir, bundle, source, diagnostic_formats_only=False):
    """Read-only checks before importing CUDA, TensorRT, Torch, or Ultralytics."""
    _manifest, files = verify_manifest(bundle)
    for name in ('model.onnx', 'spec.json', 'module_map.json', 'preflight.json',
                 'diagnostic_scope.json', 'graph_audit.json'):
        require(name in files, 'Bundle manifest omits required file: ' + name)
    spec = read(bundle / 'spec.json')
    require(spec['schema'] == 'rampq.gate05.precision_candidate_build_spec/v1', 'Unsupported build spec')
    require(spec['configuration'] in CONFIGURATIONS, 'Unsupported precision candidate')
    require(spec['diagnostic_only'] is True, 'Candidate scope must remain diagnostic')
    require(spec['original_local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED',
            'Original numerical migration failure must remain recorded')
    expected_hashes = {'model.onnx': 'graph_sha256', 'graph_audit.json': 'graph_audit_sha256',
                       'module_map.json': 'module_map_sha256', 'preflight.json': 'failed_preflight_sha256',
                       'diagnostic_scope.json': 'diagnostic_scope_sha256'}
    for name, key in expected_hashes.items():
        require(files[name] == spec[key], 'Spec SHA256 mismatch: ' + name)
    require(read(bundle / 'preflight.json')['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED',
            'Original full-model PyTorch/ORT parity failure must remain failed')
    scope = read(bundle / 'diagnostic_scope.json')
    require(scope['status'] == 'DIAGNOSTIC_MATRIX_BUILD_AND_FIXED_IMAGE_ONLY', 'Unexpected scope')
    require(scope['failed_preflight_sha256'] == files['preflight.json'], 'Scope preflight hash mismatch')
    require(scope['original_tolerances_changed'] is False and scope['dataset_evaluation_run'] is False,
            'Numerical tolerances or diagnostic dataset scope changed')
    requests = spec['requests']
    require(len(requests) == len({row['onnx_node'] for row in requests}) == 88,
            'Spec must inventory all 88 source Conv nodes exactly once')
    require(all(row['precision'] in ('INT8', 'FP16') for row in requests), 'Unsupported requested precision')
    require({row['group'] for row in requests} == ALL_GROUPS, 'Source group inventory changed')
    require(all(row['precision'] == ('INT8' if row['group'] in QUANTIZED_GROUPS[spec['configuration']] else 'FP16')
                for row in requests), 'Requested precision mask differs from the frozen configuration')
    quantized = sum(row['precision'] == 'INT8' for row in requests)
    floating = sum(row['precision'] == 'FP16' for row in requests)
    require((quantized, floating) == (spec['quantized_conv_count'], spec['fp16_conv_count']),
            'Spec requested-precision inventory mismatch')
    graph_audit = read(bundle / 'graph_audit.json')
    require(graph_audit['configuration'] == spec['configuration'] and graph_audit['graph_sha256'] == spec['graph_sha256'],
            'Frozen graph audit/spec identity mismatch')
    require(graph_audit['source_conv_count'] == 88 and graph_audit['quantized_conv_count'] == quantized and
            graph_audit['desired_restored_fp16_conv_count'] == floating, 'Frozen graph audit inventory mismatch')
    require(binding_contract(graph_audit['common_expected_binding_contract']) ==
            binding_contract(spec['expected_binding_contract']), 'Frozen graph audit/spec contract mismatch')
    build_path = build_dir / 'build_result.json'
    build = read(build_path)
    require(build['status'] == 'BUILT_AND_DESERIALIZED', 'Candidate engine not successfully built')
    require('configuration' in build and 'spec_sha256' in build,
            'Legacy build lacks candidate configuration/spec lineage; a new candidate build is required')
    require(build['configuration'] == spec['configuration'], 'Build/spec configuration mismatch')
    require(build['diagnostic_only'] is True and build['local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED',
            'Build does not retain diagnostic scope and original failure')
    engine_path = build_dir / 'model.engine'
    require(sha(engine_path) == build['engine_sha256'], 'Engine SHA256 mismatch')
    for key, value in (('onnx_sha256', files['model.onnx']), ('spec_sha256', files['spec.json']),
                       ('manifest_sha256', sha(bundle / 'manifest.json'))):
        require(build[key] == value, 'Build lineage mismatch: ' + key)
    expected = spec['expected_binding_contract']
    require(binding_contract(build['expected_binding_contract']) == binding_contract(expected),
            'Build/spec named binding contract mismatch')
    require(binding_contract(build['bindings']) == binding_contract(expected), 'Built bindings differ from spec')
    require(all(row.get('format') == 'TensorFormat.LINEAR' for row in build['bindings']),
            'Built external binding format must be LINEAR')
    primary = spec['primary_output_name']
    require(build['primary_output_name'] == primary == 'output0', 'Primary output name changed')
    primary_rows = [row for row in expected if row['name'] == primary and not row['input']]
    input_rows = [row for row in expected if row['input']]
    require(len(primary_rows) == 1 and primary_rows[0]['shape'] == [1, 14, 8400], 'Wrong primary output shape')
    require(input_rows[0]['shape'] == [1, 3, 640, 640], 'Wrong frozen input shape')
    require(len(expected) >= 3, 'Common candidate contract must include the frozen PE auxiliary output')
    precision_path = build_dir / 'precision_audit.json'
    precision = read(precision_path)
    for key, value in (('engine_sha256', build['engine_sha256']), ('onnx_sha256', files['model.onnx']),
                       ('spec_sha256', files['spec.json']), ('module_map_sha256', files['module_map.json']),
                       ('layer_file_sha256', sha(build_dir / 'engine_layers.json')),
                       ('build_result_sha256', sha(build_path))):
        require(precision[key] == value, 'Precision audit lineage mismatch: ' + key)
    strict_ok = verify_requested_precision(precision, spec, diagnostic_formats_only)
    source_manifest = read(source / 'manifest.json')
    image = source / '000020.jpg'
    require(source_manifest['selection_rule'] == 'First line of existing frozen selection list; chosen before inference',
            'Source is not the original fixed Selection image bundle')
    require(source_manifest['files']['000020.jpg']['sha256'] == sha(image) == FIXED_IMAGE_SHA256,
            'Fixed Selection image SHA256 mismatch')
    return {'build': build, 'spec': spec, 'precision': precision, 'engine_path': engine_path,
            'image_path': image, 'strict_int8_tactics_verified': strict_ok,
            'lineage': {'build_result_sha256': sha(build_path), 'spec_sha256': files['spec.json'],
                        'bundle_manifest_sha256': sha(bundle / 'manifest.json'),
                        'precision_audit_sha256': sha(precision_path), 'onnx_sha256': files['model.onnx'],
                        'graph_audit_sha256': files['graph_audit.json'],
                        'module_map_sha256': files['module_map.json'],
                        'failed_preflight_sha256': files['preflight.json'],
                        'diagnostic_scope_sha256': files['diagnostic_scope.json'],
                        'source_manifest_sha256': sha(source / 'manifest.json')}}


def reference_info(directory, kind, np):
    """Load only already saved FP16 outputs; no reference engine is executed."""
    report_path = directory / 'smoke_result.json'
    report = read(report_path)
    raw_path = directory / 'raw_outputs.npz'
    require(report['image_sha256'] == FIXED_IMAGE_SHA256 and report['nms'] == NMS,
            'Reference image/postprocessing differs')
    if kind == 'old_fp16':
        require(report['status'] == 'PASS' and report['execution_success'] is True, 'Old FP16 smoke unsuccessful')
        require(report['engine_sha256'] == OLD_FP16_ENGINE_SHA256 and sha(raw_path) == OLD_FP16_RAW_SHA256,
                'Old FP16 reference is not the original frozen output')
        detections = report['tensorrt_detections']
        graph_control = None
    else:
        require(report['schema'] == SCHEMA and report['configuration'] == 'fp16_control',
                'Matched reference must be the generic fp16_control smoke')
        require(report['execution_success'] is True and report['all_outputs_finite'] and
                all(report['all_outputs_finite'].values()),
                'Matched FP16 reference outputs not verified finite')
        require(report['raw_outputs_sha256'] == sha(raw_path), 'Matched reference raw output SHA256 mismatch')
        _manifest, files = verify_manifest(directory)
        require(files['smoke_result.json'] == sha(report_path) and files['raw_outputs.npz'] == sha(raw_path),
                'Matched reference output manifest mismatch')
        detections = report['detections']
        graph_control = report.get('shared_graph_control')
    with np.load(str(raw_path), allow_pickle=False) as data:
        raw = data['tensorrt'].astype(np.float32)
    require(raw.shape == (1, 14, 8400) and bool(np.isfinite(raw).all()), 'Invalid reference raw output')
    return {'kind': kind, 'report_sha256': sha(report_path), 'raw_outputs_sha256': sha(raw_path),
            'engine_sha256': report['engine_sha256'], 'onnx_sha256': report.get('lineage', {}).get('onnx_sha256'),
            'raw': raw, 'detections': detections, 'expected_binding_contract': report.get('expected_binding_contract'),
            'shared_graph_control': graph_control}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True, help='Hash-frozen generic candidate build bundle')
    parser.add_argument('--source', type=Path, required=True, help='Original fixed 000020 Selection smoke bundle')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fp16-smoke', type=Path, help='Optional original FP16 outputs; descriptive comparison only')
    parser.add_argument('--matched-fp16-smoke', type=Path, help='Optional generic fp16_control saved smoke outputs')
    parser.add_argument('--diagnostic-formats-only', action='store_true',
                        help='Opt into complete requested format evidence with incomplete integer tactics; no precision pass')
    parser.add_argument('--nvtx-capture', action='store_true', help='Mark engine execution only for an optional Nsight trace')
    parser.add_argument('--check-only', action='store_true', help='Verify input hashes/contracts/audit only; never import CUDA')
    args = parser.parse_args()
    verified = verify_inputs(args.build, args.bundle, args.source, args.diagnostic_formats_only)
    if args.check_only:
        print(json.dumps({'status': 'INPUTS_VERIFIED_ONLY_NO_EXECUTION', 'configuration': verified['spec']['configuration'],
                          'lineage': verified['lineage'], 'dataset_evaluation_run': False}, indent=2), flush=True)
        return
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Real Jetson aarch64 execution required')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(args.output / '.ultralytics')
    import cv2
    import numpy as np
    import torch
    import tensorrt as trt
    import ultralytics
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
    require(trt.__version__.startswith('8.5.'), 'Expected TensorRT 8.5')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0)
    references = []
    for path, kind in ((args.fp16_smoke, 'old_fp16'), (args.matched_fp16_smoke, 'matched_fp16_control')):
        if path is not None:
            references.append(reference_info(path, kind, np))
    image = cv2.imread(str(verified['image_path']))
    require(image is not None, 'Fixed image cannot be decoded')
    padded = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=image)
    array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).to('cuda')
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(verified['engine_path'].read_bytes())
    require(engine is not None, 'Engine deserialization failed')
    context = engine.create_execution_context()
    require(context is not None, 'Engine execution context unavailable')
    if args.nvtx_capture:
        context.nvtx_verbosity = trt.ProfilingVerbosity.DETAILED
    spec = verified['spec']
    expected = verified['build']['expected_binding_contract']
    primary = verified['build']['primary_output_name']
    buffers, descriptions = [], []
    output_index = None
    for index in range(engine.num_bindings):
        shape = tuple(engine.get_binding_shape(index))
        dtype = engine.get_binding_dtype(index)
        require(engine.get_binding_format(index) == trt.TensorFormat.LINEAR, 'Nonlinear I/O format unsupported')
        require(dtype in (trt.float32, trt.float16), 'Unexpected runtime binding dtype')
        torch_dtype = torch.float32 if dtype == trt.float32 else torch.float16
        is_input = bool(engine.binding_is_input(index))
        name = engine.get_binding_name(index)
        require(shape and all(size > 0 for size in shape), 'Dynamic or empty runtime binding')
        if is_input:
            require(shape == (1, 3, 640, 640), 'Input shape changed')
            buffer = tensor.to(dtype=torch_dtype).contiguous()
        else:
            buffer = torch.empty(shape, dtype=torch_dtype, device='cuda')
            if name == primary:
                output_index = index
        descriptions.append({'name': name, 'input': is_input, 'shape': list(shape), 'dtype': str(dtype)})
        buffers.append(buffer)
    require(binding_contract(descriptions) == binding_contract(expected), 'Runtime named I/O contract changed')
    require(output_index is not None, 'Primary output binding missing')
    torch.cuda.synchronize()
    print('CANDIDATE_SMOKE_INPUTS_AND_BINDINGS_VERIFIED', flush=True)
    if args.nvtx_capture:
        torch.cuda.nvtx.range_push(NVTX_RANGE)
    started = time.perf_counter()
    try:
        success = context.execute_v2([int(buffer.data_ptr()) for buffer in buffers])
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        if args.nvtx_capture:
            torch.cuda.nvtx.range_pop()
    require(success, 'TensorRT execute_v2 returned false')
    finite = {row['name']: bool(torch.isfinite(buffers[index]).all())
              for index, row in enumerate(descriptions) if not row['input']}
    saved, auxiliary = {}, {}
    for index, row in enumerate(descriptions):
        if row['input']:
            continue
        key = 'tensorrt' if index == output_index else 'auxiliary_output_{}'.format(len(auxiliary))
        if index != output_index:
            auxiliary[row['name']] = key
        saved[key] = buffers[index].detach().cpu().numpy()
    np.savez_compressed(str(args.output / 'raw_outputs.npz'), **saved)
    all_finite = bool(finite) and all(finite.values())
    detections = []
    if all_finite:
        actual = buffers[output_index].float()
        boxes = non_max_suppression(actual.clone(), **NMS)[0].detach().cpu()
        ops.scale_boxes((640, 640), boxes[:, :4], image.shape[:2])
        detections = boxes.tolist()
    shared_graph_control = {key: spec[key] for key in GRAPH_CONTROL_KEYS if key in spec}
    if len(shared_graph_control) != len(GRAPH_CONTROL_KEYS):
        shared_graph_control = None
    comparisons = []
    for reference in references:
        raw = reference.pop('raw')
        contract = reference.pop('expected_binding_contract')
        common_contract = contract is not None and binding_contract(contract) == binding_contract(expected)
        graph_verified = (reference['kind'] == 'matched_fp16_control' and common_contract and
                          shared_graph_control is not None and
                          shared_graph_control == reference['shared_graph_control'])
        reference.update({'descriptive_only': True, 'common_output_contract_verified': common_contract,
                          'shared_source_graph_hash_control_verified': graph_verified,
                          'graph_structure_confounded': not graph_verified,
                          'fusion_and_nonquantized_floating_precision_control_unverified': True,
                          'causal_attribution_allowed': False})
        if all_finite:
            delta = np.abs(saved['tensorrt'].astype(np.float32) - raw)
            reference.update(max_abs_box_output_difference=float(delta[:, :4].max()),
                             max_abs_score_output_difference=float(delta[:, 4:].max()))
        comparisons.append(reference)
    report = {'schema': SCHEMA,
              'status': 'DIAGNOSTIC_FIXED_IMAGE_EXECUTION_OK' if all_finite else 'DIAGNOSTIC_FIXED_IMAGE_NONFINITE_OUTPUT',
              'configuration': spec['configuration'], 'execution_success': True, 'diagnostic_only': True,
              'diagnostic_formats_only_opt_in': args.diagnostic_formats_only,
              'precision_audit_status': verified['precision']['status'],
              'strict_int8_status': verified['precision']['strict_int8_status'],
              'strict_int8_tactics_verified': verified['strict_int8_tactics_verified'],
              'all_requested_conv_formats_verified': True, 'all_outputs_finite': finite,
              'auxiliary_output_npz_keys': auxiliary, 'primary_output_npz_key': 'tensorrt',
              'engine_sha256': sha(verified['engine_path']), 'lineage': verified['lineage'],
              'image': '000020.jpg', 'image_sha256': FIXED_IMAGE_SHA256,
              'preprocess': {'imgsz': 640, 'batch': 1, 'rect': False, 'bgr_to_rgb': True,
                             'scale': 'float32 / 255.0', 'letterbox_auto': False, 'stride': 32},
              'bindings': descriptions, 'expected_binding_contract': expected, 'primary_output_name': primary,
              'nms': NMS, 'detections': detections, 'class_count': 10,
              'raw_outputs_sha256': sha(args.output / 'raw_outputs.npz'),
              'shared_graph_control': shared_graph_control, 'fp16_comparisons': comparisons,
              'comparison_limits': 'All comparisons are descriptive. Common graph and output declarations do not '
                                   'establish common fusion or floating precision. A protocol decision is still required '
                                   'before target rescue, causal attribution, or dataset conclusions.',
              'first_call_ms_not_benchmark': elapsed_ms, 'nvtx_capture_range': NVTX_RANGE if args.nvtx_capture else None,
              'tensorrt': trt.__version__, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'numerical_migration_status': 'ORIGINAL_LOCAL_FULL_MODEL_PYTORCH_ORT_QDQ_PARITY_FAILED_RETAINED',
              'dataset_evaluation_run': False, 'accuracy_pass_claim': False, 'performance_pass_claim': False,
              'causal_attribution_allowed': False,
              'scope': 'One existing fixed Selection image only; build/format/execution evidence only'}
    write(args.output / 'smoke_result.json', report)
    if all_finite:
        names = ['holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals',
                 'diver', 'cuttlefish', 'turtle', 'jellyfish']
        annotated = image.copy()
        for row in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in row[:4]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(annotated, '{} {:.3f}'.format(names[int(row[5])], row[4]), (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
        require(cv2.imwrite(str(args.output / 'prediction.jpg'), annotated), 'Cannot save diagnostic prediction image')
    names = ['smoke_result.json', 'raw_outputs.npz'] + (['prediction.jpg'] if all_finite else [])
    write(args.output / 'manifest.json', {'schema': 'rampq.gate05.precision_candidate_smoke_files/v1',
                                        'files': {name: {'sha256': sha(args.output / name),
                                                         'bytes': (args.output / name).stat().st_size} for name in names}})
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    require(all_finite, 'Primary or auxiliary output is nonfinite; raw outputs retained')
    print('DIAGNOSTIC_FIXED_IMAGE_EXECUTION_OK', flush=True)


if __name__ == '__main__':
    main()
