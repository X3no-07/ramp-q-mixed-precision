"""Recompute saved-output NMS/GT on CPU; no model or native execution."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['YOLO_CONFIG_DIR'] = str(OUT / '.ultralytics')
os.environ['MPLCONFIGDIR'] = str(OUT / '.matplotlib')
NMS = {'conf_thres': 0.25, 'iou_thres': 0.7, 'max_det': 1000, 'nc': 10}
CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals',
               'diver', 'cuttlefish', 'turtle', 'jellyfish']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_inputs(paths):
    return {path.relative_to(ROOT).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)}
            for path in paths}


def write_new(path, value):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(encoded)


def main():
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.utils import ops
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
    sys.path.insert(0, str(ROOT / 'src'))
    import gate0
    torch.set_num_threads(4)
    native = ROOT / 'artifacts/gate05/20261019_corrected_qonly_native/post_smoke_evidence/fixed_image_smoke'
    old = ROOT / 'artifacts/gate05/20261019_precision_native/prefer/q_only_G1/post_smoke_evidence/smoke'
    cpu = ROOT / 'artifacts/gate05/20261019_corrected_qonly_cpu_equivalence'
    image_source = ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1'
    image_path, label_path = image_source / '000020.jpg', image_source / '000020.txt'
    paths = [Path(__file__).resolve(), native / 'raw_outputs.npz', native / 'smoke_result.json',
             native / 'manifest.json', old / 'raw_outputs.npz', old / 'smoke_result.json',
             cpu / 'raw_outputs.npz', cpu / 'analysis.json', image_path, label_path,
             image_source / 'manifest.json', ROOT / 'config/gate0.json', ROOT / 'src/gate0.py',
             ROOT / 'scripts/jetson_precision_candidate_smoke.py']
    before = hash_inputs(paths)
    smoke, old_smoke, cpu_report = read(native / 'smoke_result.json'), read(old / 'smoke_result.json'), read(cpu / 'analysis.json')
    source_manifest = read(image_source / 'manifest.json')
    require(smoke['nms'] == old_smoke['nms'] == NMS, 'Existing NMS parameters differ')
    require(sha(native / 'raw_outputs.npz') == smoke['raw_outputs_sha256'] and
            sha(old / 'raw_outputs.npz') == old_smoke['raw_outputs_sha256'] and
            sha(cpu / 'raw_outputs.npz') == cpu_report['raw_outputs_sha256'], 'Saved raw hash differs')
    require(sha(image_path) == smoke['image_sha256'] == old_smoke['image_sha256'] ==
            source_manifest['files']['000020.jpg']['sha256'], 'Fixed image identity differs')
    require(sha(label_path) == source_manifest['files']['000020.txt']['sha256'], 'Original label SHA differs')
    config = read(ROOT / 'config/gate0.json')
    require(config['primary_conf'] == NMS['conf_thres'] and config['nms_iou'] == NMS['iou_thres'] and
            config['max_det'] == NMS['max_det'] and config['match_iou'] == 0.5, 'Original GT/NMS protocol differs')
    image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    require(image is not None, 'Cannot decode frozen source image')
    height, width = image.shape[:2]
    with np.load(native / 'raw_outputs.npz', allow_pickle=False) as data:
        new_raw = data[smoke['primary_output_npz_key']].copy()
    with np.load(old / 'raw_outputs.npz', allow_pickle=False) as data:
        old_raw = data[old_smoke['primary_output_npz_key']].copy()
    with np.load(cpu / 'raw_outputs.npz', allow_pickle=False) as data:
        cpu_raw = data['original_0'].copy()
    require(all(value.shape == (1, 14, 8400) and value.dtype == np.float32 and np.isfinite(value).all()
                for value in (new_raw, old_raw, cpu_raw)), 'Expected finite original FLOAT raw shape')
    results, score_stats = {}, {}
    for name, raw in [('new_native_saved_raw', new_raw), ('old_native_saved_raw', old_raw), ('cpu_original_saved_raw', cpu_raw)]:
        tensor = torch.from_numpy(raw.copy()).float()
        require(tensor.device.type == 'cpu', 'NMS tensor is not on CPU')
        detections = non_max_suppression(tensor.clone(), **NMS)[0].detach().cpu()
        ops.scale_boxes((640, 640), detections[:, :4], image.shape[:2])
        array = detections.numpy().copy()
        np.save(OUT / ('cpu_nms_' + name + '.npy'), array)
        results[name] = array
        scores = raw[:, 4:, :]
        score_stats[name] = {'global_max_score': float(scores.max()), 'global_min_score': float(scores.min()),
                             'anchors_with_any_class_score_above_025': int(np.count_nonzero(scores.max(axis=1) > 0.25)),
                             'per_class_max_score': {CLASS_NAMES[i]: float(scores[:, i, :].max()) for i in range(10)},
                             'detections_after_original_nms': len(array)}
    expected_new = np.asarray(smoke['detections'], dtype=np.float32).reshape(-1, 6)
    expected_old = np.asarray(old_smoke['detections'], dtype=np.float32).reshape(-1, 6)
    new_equal = results['new_native_saved_raw'].shape == expected_new.shape and \
        results['new_native_saved_raw'].tobytes() == expected_new.tobytes()
    old_equal = results['old_native_saved_raw'].shape == expected_old.shape and \
        results['old_native_saved_raw'].tobytes() == expected_old.tobytes()
    require(new_equal and old_equal, 'CPU NMS does not exactly reproduce saved smoke detections')
    targets = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for index, line in enumerate(label_path.read_text(encoding='utf-8-sig').splitlines()):
        if not line.strip():
            continue
        parts = [float(value) for value in line.split()]
        require(len(parts) == 5 and all(math.isfinite(value) for value in parts), 'Invalid original label line')
        class_id, cx, cy, bw, bh = parts
        require(class_id == int(class_id) and 0 <= class_id < 10 and all(0 <= value <= 1 for value in parts[1:]),
                'Invalid YOLO class/normalized coordinates')
        box = ((cx - bw / 2) * width, (cy - bh / 2) * height,
               (cx + bw / 2) * width, (cy + bh / 2) * height)
        targets.append(gate0.GroundTruth('000020.jpg:{}'.format(index), int(class_id), box,
                                        bw * bh, gate0.local_contrast(gray, box)))
    require(len(targets) == 1 and targets[0].class_id == 0, 'Expected exactly one existing holothurian GT')
    gt_match, exact_ious = {}, {}
    for name, array in results.items():
        detections = [gate0.Detection(int(row[5]), float(row[4]), tuple(float(value) for value in row[:4])) for row in array]
        gt_match[name] = gate0.match_targets(targets, detections, primary_conf=config['primary_conf'], match_iou=config['match_iou'])
        exact_ious[name] = [{'detection_index': index, 'class_id': int(row[5]), 'class_name': CLASS_NAMES[int(row[5])],
                             'confidence': float(row[4]), 'box_xyxy_original_image': [float(value) for value in row[:4]],
                             'same_class_as_unique_gt': int(row[5]) == targets[0].class_id,
                             'iou_to_unique_gt_unrounded': float(gate0.box_iou(targets[0].box, tuple(float(value) for value in row[:4]))),
                             'gt_supported_correct_detection': bool(int(row[5]) == targets[0].class_id and
                                float(row[4]) >= config['primary_conf'] and gate0.box_iou(targets[0].box, tuple(float(value) for value in row[:4])) >= config['match_iou'])}
                            for index, row in enumerate(array)]
    after = hash_inputs(paths)
    require(before == after, 'An original saved evidence file changed')
    report = {'schema': 'rampq.gate05.corrected_qonly_saved_raw_nms_gt_recheck/v1',
              'status': 'SAVED_RAW_CPU_NMS_AND_GT_RECHECK_COMPLETE',
              'nms': NMS, 'gt_match_iou_threshold_from_original_config': config['match_iou'],
              'execution_device': 'cpu', 'torch_version': torch.__version__, 'ultralytics_version': ultralytics.__version__,
              'native_smoke_ultralytics_version': smoke['ultralytics'],
              'source_image_shape_hw': [height, width], 'original_gt_label_lines': label_path.read_text().strip().splitlines(),
              'ground_truth_count': len(targets), 'ground_truth_class_names': ['holothurian'],
              'unique_gt_xyxy_original_image_unrounded': list(targets[0].box),
              'new_native_cpu_nms_detections_bit_exact_to_saved_smoke_float32': new_equal,
              'old_native_cpu_nms_detections_bit_exact_to_saved_smoke_float32': old_equal,
              'cpu_nms_detections': {name: array.tolist() for name, array in results.items()},
              'raw_score_statistics': score_stats, 'original_gate0_one_to_one_target_matches': gt_match,
              'detection_ious_to_unique_gt': exact_ious,
              'new_native_reported_detection_count': len(expected_new),
              'new_native_gt_supported_detection_count': sum(row['gt_supported_correct_detection'] for row in exact_ious['new_native_saved_raw']),
              'scallop_ground_truth_count': 0,
              'scallop_detection_interpretation': 'Saved output contains a class-2 scallop detection, but the supplied frozen label has no scallop GT; it cannot establish a second correct target.',
              'source_hashes_before': before, 'source_hashes_after': after, 'all_sources_unchanged': True,
              'original_complete_model_pytorch_ort_guard': 'FAILED_UNCHANGED',
              'new_tolerances_defined': False, 'numerical_migration_pass_claim': False,
              'accuracy_pass_claim': False, 'causal_attribution_allowed': False,
              'performance_pass_claim': False, 'dataset_evaluation_run': False,
              'new_full_model_inference_run': False, 'new_native_execution_run': False, 'cuda_used': False,
              'limits': 'Read-only saved-output postprocessing on one original Selection image. One GT match is an observation on this image; two detections are not two GT-verified targets. Original PyTorch/ORT guard remains failed, and native numerical migration/accuracy/performance are not established.'}
    write_new(OUT / 'nms_gt_analysis.json', report)
    print(json.dumps({'new_native_detections': report['cpu_nms_detections']['new_native_saved_raw'],
                      'cpu_nms_exactly_matches_native_smoke': new_equal,
                      'old_native_detection_count': len(results['old_native_saved_raw']),
                      'old_native_max_score': score_stats['old_native_saved_raw']['global_max_score'],
                      'unique_gt_box': report['unique_gt_xyxy_original_image_unrounded'],
                      'new_native_gt_ious': exact_ious['new_native_saved_raw'],
                      'new_native_gt_supported_detection_count': report['new_native_gt_supported_detection_count'],
                      'all_sources_unchanged': True, 'analysis_sha256': sha(OUT / 'nms_gt_analysis.json'),
                      'script_sha256': sha(Path(__file__))}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
