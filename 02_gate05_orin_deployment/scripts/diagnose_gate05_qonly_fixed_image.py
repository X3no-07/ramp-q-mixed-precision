"""One existing image: CPU Gate0 q_only:G1 and exact frozen ORT graph diagnostic.

Saved TensorRT outputs are read only. There is no new tolerance, native-parity
gate, calibration, dataset evaluation, engine build, or board communication.
"""
import argparse
import contextlib
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def source_record(paths):
    return {str(path.relative_to(ROOT)).replace('\\', '/'): {'sha256': sha(path), 'bytes': path.stat().st_size}
            for path in sorted(set(paths), key=str)}


def verify_receipt(folder):
    receipt = read(folder / 'remote_receipt.json')
    paths = [folder / 'remote_receipt.json']
    for name, info in receipt['files'].items():
        path = folder / name
        require(folder.resolve() in path.resolve().parents and path.stat().st_size == info['bytes'] and
                sha(path) == info['sha256'], 'Native receipt file differs: ' + name)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/gate05/20261019_qonly_fixed_image_diagnostic')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(out / '.ultralytics')
    original = ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1'
    bundle = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles_prefer/q_only_G1'
    native = ROOT / 'artifacts/gate05/20261019_precision_native/prefer/q_only_G1/post_smoke_evidence'
    reference = ROOT / 'artifacts/gate05/20261019_precision_native/fp16_control/post_smoke_evidence'
    calibration_path = ROOT / 'artifacts/gate05/20261018_int8/input_v1/calibration.json'
    calibration_original = ROOT / 'artifacts/calibration/screen.json'
    config_path = ROOT / 'config/gate0.json'
    image_path, weights_path, label_path = original / '000020.jpg', original / 'best.pt', original / '000020.txt'
    source_paths = [Path(__file__).resolve(), ROOT / 'src/gate0.py', config_path, calibration_path,
                    calibration_original, image_path, weights_path, label_path, original / 'manifest.json']
    source_paths += verify_receipt(native) + verify_receipt(reference)
    manifest = read(bundle / 'manifest.json')
    for name, info in manifest['files'].items():
        path = bundle / name
        require(path.resolve().parent == bundle and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                'Frozen prefer bundle differs: ' + name)
        source_paths.append(path)
    source_paths.append(bundle / 'manifest.json')
    before = source_record(source_paths)
    write(out / 'source_hashes_before.json', before)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    spec, calibration, config = read(bundle / 'spec.json'), read(calibration_path), read(config_path)
    native_report, fp16_report = read(native / 'smoke/smoke_result.json'), read(reference / 'smoke/smoke_result.json')
    require(spec['configuration'] == 'q_only_G1' and spec['quantized_conv_count'] == 6 and spec['fp16_conv_count'] == 82,
            'Frozen precision configuration differs')
    require(sha(calibration_path) == sha(calibration_original), 'Original frozen calibration copy differs')
    require(sha(weights_path) == calibration['model_sha256'] == spec['common_source_weights_sha256'], 'Weights differ')
    require(sha(image_path) == native_report['image_sha256'] == fp16_report['image_sha256'] ==
            read(original / 'manifest.json')['files']['000020.jpg']['sha256'], 'Fixed image differs')
    require(sha(label_path) == read(original / 'manifest.json')['files']['000020.txt']['sha256'], 'Fixed label differs')
    require(native_report['execution_success'] is True and native_report['configuration'] == 'q_only_G1' and
            native_report['strict_int8_tactics_verified'] is True and native_report['strict_int8_status'] ==
            'ALL_REQUESTED_INT8_TACTICS_VERIFIED', 'Saved q_only native execution/precision evidence differs')
    require(native_report['lineage']['onnx_sha256'] == sha(bundle / 'model.onnx') == spec['graph_sha256'] and
            native_report['lineage']['spec_sha256'] == sha(bundle / 'spec.json'), 'Saved q_only graph lineage differs')
    require(sha(native / 'smoke/raw_outputs.npz') == native_report['raw_outputs_sha256'] and
            sha(reference / 'smoke/raw_outputs.npz') == fp16_report['raw_outputs_sha256'], 'Saved raw output differs')
    nms = {'conf_thres': 0.25, 'iou_thres': 0.7, 'max_det': 1000, 'nc': 10}
    require(native_report['nms'] == fp16_report['nms'] == nms and config['imgsz'] == 640 and
            config['primary_conf'] == 0.25 and config['match_iou'] == 0.5, 'Frozen thresholds/settings differ')
    protocol = {'schema': 'rampq.gate05.qonly_fixed_image_cpu_diagnostic_protocol/v1',
                'client_started_at': datetime.now().astimezone().isoformat(), 'image': '000020.jpg',
                'configuration': 'q_only:G1', 'new_image_selection': False, 'cpu_forward_runs': 2,
                'calibration_recomputed': False, 'nms': nms, 'input_shape': [1, 3, 640, 640],
                'pytorch': 'Existing gate0.load_fused_model/install_qdq_wrappers/enable_configuration; CPU FP32 arithmetic and same frozen G1 QDQ',
                'onnxruntime': 'Exact hash-frozen prefer q_only_G1 ONNX; CPUExecutionProvider, ORT_DISABLE_ALL, intra_op_num_threads=4',
                'native_outputs': 'Read the already executed q_only_G1 and matched fp16_control saved raw outputs only',
                'precision_boundary': 'PyTorch/ORT nonquantized operators use FP32. Native q_only_G1 has six audited INT8 Conv and 82 audited FP16 Conv; fusion/backend and floating precision are confounded.',
                'new_native_parity_criterion_defined': False, 'existing_tolerances_changed': False,
                'original_q_all_preflight': 'LOCAL_QDQ_PREFLIGHT_FAILED_UNCHANGED',
                'dataset_evaluation_run': False, 'causal_or_target_recovery_claim': False}
    write(out / 'protocol.json', protocol)
    with (out / 'cpu_console.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import cv2
        import numpy as np
        import onnx
        from onnx import numpy_helper
        import onnxruntime as ort
        import torch
        import ultralytics
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import ops
        from ultralytics.utils.nms import non_max_suppression
        sys.path.insert(0, str(ROOT / 'src'))
        import gate0
        torch.set_num_threads(4)
        x, image, ratio, left, top = gate0.image_tensor(image_path, 640, torch.device('cpu'))
        require(x.device.type == 'cpu' and x.dtype == torch.float32, 'CPU FP32 input required')
        board_style = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=image)
        board_style = np.ascontiguousarray(board_style[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        require(np.array_equal(x.numpy(), board_style), 'Host Gate0 and recorded native-style preprocessing differ')
        np.save(out / 'preprocessed_input.npy', x.numpy())
        model = gate0.load_fused_model(weights_path, torch.device('cpu'))
        require(all(parameter.dtype == torch.float32 and parameter.device.type == 'cpu' for parameter in model.parameters()),
                'Reference parameters must stay CPU FP32')
        wrappers = gate0.install_qdq_wrappers(model, calibration['activation_max_abs'])
        gate0.enable_configuration(wrappers, 'q_only:G1')
        require(len(wrappers) == 88 and sum(wrapper.enabled for wrapper in wrappers) == 6, 'Gate0 mask differs')
        graph = onnx.load(str(bundle / 'model.onnx'))
        constants = {value.name: value for value in graph.graph.initializer}
        producers = {name: node for node in graph.graph.node for name in node.output}
        nodes = {node.name: node for node in graph.graph.node}
        modules = {wrapper.name: wrapper for wrapper in wrappers}
        quantizer_checks, activation_observations, recorded = [], {}, {}
        handles = []
        for request in spec['requests']:
            wrapper = modules[request['module']]
            conv = nodes[request['onnx_node']]
            require(np.array_equal(wrapper.original.weight.detach().numpy(), numpy_helper.to_array(constants[request['weight_input']])),
                    'PyTorch/source original Conv weights differ')
            if wrapper.original.bias is not None:
                require(np.array_equal(wrapper.original.bias.detach().numpy(), numpy_helper.to_array(constants[conv.input[2]])),
                        'PyTorch/source Conv bias differs')
            if not wrapper.enabled:
                continue
            activation_dq = producers[conv.input[0]]
            weight_dq = producers[conv.input[1]]
            weight_q = producers[weight_dq.input[0]]
            a_scale = numpy_helper.to_array(constants[activation_dq.input[1]])
            w_scale = numpy_helper.to_array(constants[weight_q.input[1]])
            require(np.array_equal(wrapper.activation_scale.numpy(), a_scale), 'G1 activation scale differs')
            original_weight = numpy_helper.to_array(constants[weight_q.input[0]])
            expected_weight = np.rint(original_weight / w_scale[:, None, None, None]).clip(-127, 127) * w_scale[:, None, None, None]
            require(np.array_equal(wrapper.quantized_weight.numpy(), expected_weight), 'G1 frozen weight QDQ differs')
            quantizer_checks.append({'module': wrapper.name, 'activation_scale': float(a_scale),
                                     'weight_qdq_exact': True, 'frozen_activation_scale_exact': True})
            def record_activation(mod, inputs, _output):
                value = inputs[0].detach()
                bound = float(np.float32(mod.activation_scale.item()) * np.float32(127))
                activation_observations[mod.name] = {'elements': value.numel(), 'scale': mod.activation_scale.item(),
                    'max_abs': value.abs().max().item(), 'saturated_elements_above_frozen_bound': int((value.abs() > bound).sum().item())}
            handles.append(wrapper.register_forward_hook(record_activation))
        def record_pe(_mod, _inputs, output):
            recorded['pytorch_pe_auxiliary'] = output.detach().numpy().copy()
        handles.append(modules['model.10.m.0.attn.pe.conv'].register_forward_hook(record_pe))
        try:
            with torch.inference_mode():
                prediction = model(x)
                prediction = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
                pytorch_raw = prediction.detach().numpy().copy()
        finally:
            for handle in handles:
                handle.remove()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(bundle / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        ort_outputs = dict(zip([value.name for value in session.get_outputs()], session.run(None, {'images': x.numpy()})))
        ort_raw = ort_outputs['output0']
        auxiliary_name = next(row['name'] for row in spec['expected_binding_contract'] if not row['input'] and row['name'] != 'output0')
        with np.load(native / 'smoke/raw_outputs.npz', allow_pickle=False) as saved:
            trt_raw = saved['tensorrt'].copy()
            trt_aux = saved[native_report['auxiliary_output_npz_keys'][auxiliary_name]].copy()
        with np.load(reference / 'smoke/raw_outputs.npz', allow_pickle=False) as saved:
            fp16_raw = saved['tensorrt'].copy()
        arrays = {'pytorch_qonly_g1': pytorch_raw, 'ort_qonly_g1': ort_raw,
                  'saved_trt_qonly_g1': trt_raw, 'saved_trt_fp16_control': fp16_raw,
                  'pytorch_pe_auxiliary': recorded['pytorch_pe_auxiliary'],
                  'ort_pe_auxiliary': ort_outputs[auxiliary_name], 'saved_trt_pe_auxiliary': trt_aux}
        np.savez_compressed(out / 'raw_outputs.npz', **arrays)
        detections, score_ranges = {}, {}
        for name in ('pytorch_qonly_g1', 'ort_qonly_g1', 'saved_trt_qonly_g1', 'saved_trt_fp16_control'):
            raw = arrays[name]
            require(raw.shape == (1, 14, 8400) and bool(np.isfinite(raw).all()), 'Invalid main output: ' + name)
            scores = raw[:, 4:, :]
            score_ranges[name] = {'min_score': float(scores.min()), 'max_score': float(scores.max()),
                                 'anchors_above_frozen_conf': int((scores.max(axis=1) > 0.25).sum()),
                                 'all_outputs_finite': True}
            rows = non_max_suppression(torch.from_numpy(raw.copy()), **nms)[0].cpu()
            ops.scale_boxes((640, 640), rows[:, :4], image.shape[:2])
            detections[name] = rows.tolist()
        def difference(first, second):
            delta = np.abs(first.astype(np.float32) - second.astype(np.float32))
            return {'max_abs_box_difference': float(delta[:, :4].max()),
                    'max_abs_score_difference': float(delta[:, 4:].max()),
                    'mean_abs_box_difference': float(delta[:, :4].mean()),
                    'mean_abs_score_difference': float(delta[:, 4:].mean()), 'descriptive_only': True}
        differences = {'pytorch_vs_ort': difference(pytorch_raw, ort_raw),
                       'pytorch_vs_saved_trt': difference(pytorch_raw, trt_raw),
                       'ort_vs_saved_trt': difference(ort_raw, trt_raw),
                       'saved_fp16_vs_saved_qonly_trt': difference(fp16_raw, trt_raw)}
        targets = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        for index, line in enumerate(label_path.read_text(encoding='utf-8-sig').splitlines()):
            if not line.strip():
                continue
            class_id, cx, cy, bw, bh = [float(value) for value in line.split()]
            box = ((cx - bw / 2) * width, (cy - bh / 2) * height,
                   (cx + bw / 2) * width, (cy + bh / 2) * height)
            targets.append(gate0.GroundTruth('000020.jpg:{}'.format(index), int(class_id), box,
                           bw * bh, gate0.local_contrast(gray, box)))
        matches = {name: gate0.match_targets(targets, [gate0.Detection(int(row[5]), row[4], tuple(row[:4])) for row in rows],
                                            primary_conf=0.25, match_iou=0.5) for name, rows in detections.items()}
        report = {'schema': 'rampq.gate05.qonly_fixed_image_cpu_diagnostic/v1',
                  'status': 'ONE_FIXED_IMAGE_CPU_DIAGNOSTIC_COMPLETED',
                  'client_completed_at': datetime.now().astimezone().isoformat(),
                  'python': platform.python_version(), 'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                  'onnxruntime': ort.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__,
                  'cpu_forward_runs': {'pytorch_qonly_g1': 1, 'ort_exact_qonly_g1': 1},
                  'native_inference_run_by_this_script': False, 'ort_providers': session.get_providers(),
                  'ort_graph_optimization': 'ORT_DISABLE_ALL', 'nms': nms, 'host_preprocess_bit_exact_between_gate0_and_native_style': True,
                  'image_sha256': sha(image_path), 'onnx_sha256': sha(bundle / 'model.onnx'),
                  'weights_sha256': sha(weights_path), 'calibration_sha256': sha(calibration_path),
                  'saved_native_engine_sha256': native_report['engine_sha256'],
                  'saved_native_smoke_boot_id': native_report['boot_id'], 'saved_fp16_smoke_boot_id': fp16_report['boot_id'],
                  'quantizer_checks': quantizer_checks, 'pytorch_activation_observations': activation_observations,
                  'score_ranges': score_ranges, 'detection_counts': {name: len(rows) for name, rows in detections.items()},
                  'detections': detections, 'saved_native_detections': native_report['detections'],
                  'saved_fp16_detections': fp16_report['detections'], 'raw_differences': differences,
                  'target_count': len(targets), 'target_matches': matches,
                  'annotation_boundary': 'Only the existing label is used. Extra class predictions absent from that label are not automatically counted as false positives.',
                  'precision_boundary': protocol['precision_boundary'],
                  'native_parity_criterion_defined_or_passed': False,
                  'original_q_all_preflight': 'LOCAL_QDQ_PREFLIGHT_FAILED_UNCHANGED',
                  'new_thresholds_or_tolerances_defined': False, 'dataset_evaluation_run': False,
                  'causal_or_target_recovery_claim': False,
                  'scope': 'One already fixed Selection image; CPU references and saved native outputs are descriptive only.'}
    after = source_record(source_paths)
    write(out / 'source_hashes_after.json', after)
    require(after == before, 'A source file changed during the diagnostic')
    report['all_source_files_sha256_unchanged'] = True
    report['raw_outputs_sha256'] = sha(out / 'raw_outputs.npz')
    write(out / 'diagnostic.json', report)
    counts = report['detection_counts']
    note = '# q_only: G1 fixed single-image CPU diagnosis\n\n'
    note += 'Only the existing Selection image `000020.jpg` is used, without re-calibration, image modification, threshold adjustment, or 150-image evaluation run.\n\n'
    note += 'Detection count at frozen conf=0.25: PyTorch CPU {}, ORT CPU {}, saved TRT q_only:G1 {}, saved TRT FP16 {}.\n\n'.format(
        counts['pytorch_qonly_g1'], counts['ort_qonly_g1'], counts['saved_trt_qonly_g1'], counts['saved_trt_fp16_control'])
    note += 'The unquantized portions of PyTorch/ORT use FP32; the remaining 82 convolutions in TRT have been verified as FP16, with backend and fusion differences present. Therefore, these discrepancies cannot serve as a criterion for native numerical consistency pass/fail, nor can they be attributed solely to quantization. The original q_all full numerical consistency failure remains unchanged.\n\n'
    note += 'The original main output, auxiliary output, input tensors, source file SHA values before and after, console logs, and machine-readable detection/target matching have been preserved. All source file hashes remain unchanged.\n'
    (out / 'README.md').write_text(note, encoding='utf-8')
    files = {path.relative_to(out).as_posix(): {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in sorted(out.rglob('*')) if path.is_file()}
    write(out / 'manifest.json', {'schema': 'rampq.gate05.qonly_fixed_image_diagnostic_files/v1', 'files': files})
    (out / 'SHA256SUMS').write_text(''.join(info['sha256'] + '  ' + name + '\n' for name, info in files.items()), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'output': str(out), 'detection_counts': counts,
                      'score_ranges': report['score_ranges'], 'raw_differences': report['raw_differences'],
                      'all_source_files_sha256_unchanged': True,
                      'native_parity_criterion_defined_or_passed': False}, indent=2))


if __name__ == '__main__':
    main()
