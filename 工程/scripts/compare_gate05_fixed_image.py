"""Describe target matches in saved fixed-image detections; never run inference.

This reuses Gate 0's class-aware greedy one-to-one assignment. Its counts are
descriptive observations on Selection 000020, with no accuracy or rescue gate.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from jetson_precision_candidate_smoke import FIXED_IMAGE_SHA256, NMS, SCHEMA, require

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_gate0():
    specification = importlib.util.spec_from_file_location('gate05_existing_gate0', ROOT / 'src/gate0.py')
    require(specification is not None and specification.loader is not None, 'Cannot locate existing Gate 0 matching code')
    gate0 = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = gate0
    specification.loader.exec_module(gate0)
    return gate0


def saved_detections(path, key=None):
    report = read(path)
    require(report['image_sha256'] == FIXED_IMAGE_SHA256 and report['nms'] == NMS,
            'Saved result must use the original fixed image and frozen NMS settings')
    require(report['execution_success'] is True, 'Saved result does not record successful execution')
    if report.get('schema') == SCHEMA:
        require(report['all_outputs_finite'] and all(report['all_outputs_finite'].values()), 'Nonfinite saved outputs')
        raw_path = path.parent / 'raw_outputs.npz'
        require(report['raw_outputs_sha256'] == sha(raw_path), 'Saved raw outputs SHA256 mismatch')
        selected_key = key or 'detections'
    else:
        selected_key = key or 'tensorrt_detections'
    rows = report[selected_key]
    for row in rows:
        require(len(row) == 6 and all(math.isfinite(value) for value in row), 'Invalid saved detection row')
        require(row[5] == int(row[5]) and 0 <= int(row[5]) < 10, 'Invalid detection class')
        require(0.25 <= row[4] <= 1 and row[2] >= row[0] and row[3] >= row[1], 'Invalid confidence or XYXY box')
    return report, rows, selected_key


def fixed_ground_truths(source, gate0):
    """Load the existing smoke label, with the same target IDs and contrast as Gate 0."""
    import cv2
    manifest = read(source / 'manifest.json')
    image_path, label_path = source / '000020.jpg', source / '000020.txt'
    require(sha(image_path) == manifest['files']['000020.jpg']['sha256'] == FIXED_IMAGE_SHA256,
            'Fixed image SHA256 mismatch')
    require(sha(label_path) == manifest['files']['000020.txt']['sha256'], 'Existing fixed-image label SHA256 mismatch')
    image = cv2.imread(str(image_path))
    require(image is not None, 'Cannot decode fixed image')
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    targets = []
    for index, line in enumerate(label_path.read_text(encoding='utf-8-sig').splitlines()):
        parts = line.split()
        if not parts:
            continue
        require(len(parts) == 5, 'Existing label is not YOLO class/cx/cy/w/h')
        values = [float(value) for value in parts]
        class_id, cx, cy, bw, bh = values
        require(all(math.isfinite(value) for value in values) and class_id == int(class_id) and
                0 <= class_id < 10 and all(0 <= value <= 1 for value in values[1:]), 'Invalid existing label')
        box = ((cx - bw / 2) * width, (cy - bh / 2) * height,
               (cx + bw / 2) * width, (cy + bh / 2) * height)
        targets.append(gate0.GroundTruth('000020.jpg:{}'.format(index), int(class_id), box,
                                        max(0.0, bw) * max(0.0, bh), gate0.local_contrast(gray, box)))
    return targets, {'image_sha256': sha(image_path), 'label_sha256': sha(label_path),
                     'source_manifest_sha256': sha(source / 'manifest.json')}


def compare_target_matches(ground_truths, reference, candidate, config, gate0):
    """Call the existing matching implementation independently for both saved outputs."""
    converted = []
    for rows in (reference, candidate):
        detections = [gate0.Detection(int(row[5]), float(row[4]), tuple(row[:4])) for row in rows]
        converted.append(gate0.match_targets(ground_truths, detections,
                                             primary_conf=config['primary_conf'], match_iou=config['match_iou']))
    target_rows = []
    for ref, cand in zip(converted[0], converted[1]):
        require(ref['target_id'] == cand['target_id'], 'Target identity changed between assignments')
        state = '{}_to_{}'.format('matched' if ref['matched'] else 'unmatched',
                                  'matched' if cand['matched'] else 'unmatched')
        target_rows.append({'target_id': ref['target_id'], 'class_id': ref['class_id'],
                            'reference_match': ref, 'candidate_match': cand, 'observed_match_state': state})
    return target_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Existing reference smoke_result.json')
    parser.add_argument('--candidate', type=Path, required=True, help='Existing candidate smoke_result.json')
    parser.add_argument('--source', type=Path, required=True, help='Original 000020 image/label smoke bundle')
    parser.add_argument('--reference-key', help='Existing detection key; defaults to generic/legacy TensorRT detections')
    parser.add_argument('--candidate-key', help='Existing detection key; useful for the saved PyTorch smoke output')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/gate0.json')
    parser.add_argument('--output', type=Path, required=True, help='Fresh descriptive JSON; never overwrites evidence')
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a fresh output file; preserve existing evidence')
    config = read(args.config)
    require(config['imgsz'] == 640 and config['primary_conf'] == 0.25 and config['match_iou'] == 0.5 and
            config['nms_iou'] == 0.7 and config['max_det'] == 1000, 'Original Gate 0 matching/settings changed')
    reference, ref_rows, ref_key = saved_detections(args.reference, args.reference_key)
    candidate, cand_rows, cand_key = saved_detections(args.candidate, args.candidate_key)
    gate0 = load_gate0()
    targets, source_lineage = fixed_ground_truths(args.source, gate0)
    matches = compare_target_matches(targets, ref_rows, cand_rows, config, gate0)
    labelled_classes = {target.class_id for target in targets}
    output = {'schema': 'rampq.gate05.fixed_image_target_comparison/v1',
              'status': 'DESCRIPTIVE_SAVED_FIXED_IMAGE_MATCHES_ONLY', 'image': '000020.jpg',
              'source_lineage': source_lineage, 'matching_code_sha256': sha(ROOT / 'src/gate0.py'),
              'matching_config_sha256': sha(args.config), 'matching': {'class_aware': True, 'one_to_one': True,
              'assignment': 'Existing gate0.match_targets: IoU/confidence descending greedy',
              'primary_conf': 0.25, 'match_iou': 0.5}, 'nms': NMS,
              'reference': {'report_sha256': sha(args.reference), 'detection_key': ref_key,
                            'engine_sha256': reference['engine_sha256'] if ref_key != 'pytorch_fp32_detections' else None,
                            'weights_sha256': reference.get('weights_sha256'), 'detection_count': len(ref_rows)},
              'candidate': {'report_sha256': sha(args.candidate), 'detection_key': cand_key,
                            'engine_sha256': candidate['engine_sha256'] if cand_key != 'pytorch_fp32_detections' else None,
                            'weights_sha256': candidate.get('weights_sha256'), 'detection_count': len(cand_rows)},
              'target_count': len(targets), 'target_matches': matches,
              'fixed_label_class_counts': {str(class_id): sum(target.class_id == class_id for target in targets)
                                           for class_id in sorted(labelled_classes)},
              'predictions_in_classes_absent_from_fixed_label': {
                  'reference': {str(class_id): sum(int(row[5]) == class_id for row in ref_rows)
                                for class_id in sorted({int(row[5]) for row in ref_rows} - labelled_classes)},
                  'candidate': {str(class_id): sum(int(row[5]) == class_id for row in cand_rows)
                                for class_id in sorted({int(row[5]) for row in cand_rows} - labelled_classes)}},
              'annotation_boundary': 'The existing 000020 label contains one class-0 holothurian target. '
                                     'A saved class-2 scallop prediction is unannotated in this fixed label. '
                                     'These diagnostics do not classify that extra prediction as a false positive '
                                     'or infer dataset annotation completeness.',
              'observed_match_state_counts': {state: sum(row['observed_match_state'] == state for row in matches)
                                              for state in ('matched_to_matched', 'matched_to_unmatched',
                                                            'unmatched_to_matched', 'unmatched_to_unmatched')},
              'inference_run': False, 'dataset_evaluation_run': False, 'accuracy_pass_claim': False,
              'target_rescue_claim': False, 'causal_attribution_allowed': False,
              'numerical_migration_status': 'ORIGINAL_LOCAL_FULL_MODEL_PYTORCH_ORT_QDQ_PARITY_FAILED_RETAINED',
              'limits': 'Only saved detections on existing Selection 000020. Observed match changes are not '
                        'a recovery or causality conclusion. No dataset criteria are defined or applied; '
                        'original independent-model parity failure remains unresolved.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': output['status'], 'target_count': len(targets),
                      'observed_match_state_counts': output['observed_match_state_counts'],
                      'output': str(args.output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
