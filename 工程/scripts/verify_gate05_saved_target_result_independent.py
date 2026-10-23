"""Independently recompute frozen matching and result sets from saved detections.

No evaluator implementation is imported or called. No model, NumPy, torch,
TensorRT, ONNX runtime, NMS or coordinate re-decoding is used. Matching uses
already-saved original-image pixel XYXY boxes and original physical label IDs.
Run --self-test before a separately authorized actual saved-result review.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ORDER = ['fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1']
RESTORED = ('r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2')
MATCHING_SHA = 'f3c4d80658b79ae685e569b37c5ea89af9e6e9dd4a63ce86aff2806a7fcc5adb'
EVALUATOR_SHA = 'b5ac60351e17761144a8897300ef8c8005f14e87e119b74dc1be2907be4cb895'
POST_SHA = 'd887b9f43fbecf1fec6a9dd3d1663160049cf8b36df57fe167f5d66bb372798d'
DATA_SHA = 'c4dac80ad7c59524194a9501f5ff8126cc0517e84f29f4106068c0c4b27a93aa'
POST = {'candidate_confidence': .001, 'nms_iou': .7, 'max_det': 1000, 'nc': 10,
        'max_time_img': 2., 'agnostic': False, 'multi_label': False}
IDENTITY_KEYS = ('source_graph_sha256', 'engine_sha256', 'build_request_sha256',
                 'native_precision_audit_sha256', 'float_fusion_comparability_audit_sha256',
                 'independent_ORT_reference_graph_sha256')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def contained(root, name):
    relative = PurePosixPath(name)
    require(str(relative) == name and not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,
            'Unsafe saved relative path: ' + str(name))
    path = root.joinpath(*relative.parts)
    require(path.is_file() and root in path.resolve().parents and not path.is_symlink(), 'Missing/escaping saved file: ' + name)
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), 'Saved file has symlink parent: ' + name)
    return path


class SourceReader:
    def __init__(self):
        self.identities = {}

    def raw(self, path, expected=None):
        path = path.resolve(); data = path.read_bytes(); value = hashlib.sha256(data).hexdigest()
        require(expected is None or value == expected, 'Saved source SHA differs: ' + str(path))
        require(str(path) not in self.identities or self.identities[str(path)] == value, 'Saved input drift: ' + str(path))
        self.identities[str(path)] = value
        return data

    def hash(self, path, expected=None):
        path = path.resolve(); value = digest(path)
        require(expected is None or value == expected, 'Saved source SHA differs: ' + str(path))
        require(str(path) not in self.identities or self.identities[str(path)] == value, 'Saved input drift: ' + str(path))
        self.identities[str(path)] = value
        return value

    def json(self, path, expected=None):
        return json.loads(self.raw(path, expected).decode('utf-8-sig'))

    def lines(self, path, expected=None):
        lines = self.raw(path, expected).decode('utf-8-sig').splitlines()
        require(all(line.strip() for line in lines), 'Blank saved JSONL line')
        return [json.loads(line) for line in lines]

    def recheck(self):
        for name, expected in self.identities.items():
            require(digest(Path(name)) == expected, 'Saved source changed during independent verification: ' + name)


def jpeg_geometry(data):
    require(data[:2] == b'\xff\xd8', 'Expected frozen JPEG')
    offset = 2; starts = {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}
    while offset < len(data):
        require(data[offset] == 0xff, 'Malformed JPEG marker')
        while offset < len(data) and data[offset] == 0xff:
            offset += 1
        require(offset < len(data), 'Truncated JPEG marker')
        marker = data[offset]; offset += 1
        if marker in (0xd8, 0xd9, 0x01) or 0xd0 <= marker <= 0xd7:
            continue
        require(offset + 2 <= len(data), 'Truncated JPEG segment')
        length = struct.unpack('>H', data[offset:offset+2])[0]
        require(length >= 2 and offset + length <= len(data), 'Malformed JPEG segment length')
        if marker in starts:
            require(length >= 8, 'Truncated JPEG dimensions')
            height, width = struct.unpack('>HH', data[offset+3:offset+7])
            require(width > 0 and height > 0, 'Empty JPEG')
            return width, height
        require(marker != 0xda, 'JPEG scan precedes dimensions')
        offset += length
    raise ValueError('JPEG dimensions missing')


def independently_match(truths, detections):
    """Contract algorithm, independently implemented with unrounded float math."""
    edges = []
    for truth_index, truth in enumerate(truths):
        for detection_index, detected in enumerate(detections):
            if detected['class_id'] != truth['class_id'] or detected['confidence'] < .25:
                continue
            a, b = truth['box'], detected['box']
            ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
            intersect = ix * iy
            first_area = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
            second_area = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
            denominator = first_area + second_area - intersect
            overlap = intersect / denominator if denominator > 0 else 0.
            if overlap >= .5:
                edges.append((overlap, detected['confidence'], truth_index, detection_index))
    assignments, occupied = {}, set()
    for overlap, confidence, truth_index, detection_index in sorted(edges, reverse=True):
        if truth_index not in assignments and detection_index not in occupied:
            assignments[truth_index] = (detection_index, overlap, confidence)
            occupied.add(detection_index)
    results = {}
    for truth_index, truth in enumerate(truths):
        chosen = assignments.get(truth_index)
        results[truth['target_id']] = {'matched': chosen is not None,
                'detection_index': chosen[0] if chosen else None,
                'match_iou': chosen[1] if chosen else None, 'match_confidence': chosen[2] if chosen else None,
                'class_id': truth['class_id'], 'area_ratio': truth['area_ratio'],
                'detection_id': truth['image_basename'] + ':det:' + str(chosen[0]) if chosen else None}
    return results


def proportion(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator/denominator if denominator else None,
            'undefined_reason': None if denominator else 'ZERO_DENOMINATOR'}


def independent_summary(native, eligible):
    hits = {name: {key for key, row in rows.items() if row['matched']} for name, rows in native.items()}
    baseline, quantized = hits['fp16_control'], hits['q_all']
    lost = baseline.difference(quantized); steady = baseline.intersection(quantized)
    restoration = {}
    for name in RESTORED:
        successes = lost.intersection(hits[name]); damage = steady.difference(hits[name])
        restoration[name] = {'rescued_target_ids': sorted(successes), 'still_missing_target_ids': sorted(lost.difference(hits[name])),
                'collateral_target_ids': sorted(damage), 'outside_FP16_target_ids': sorted(hits[name].difference(baseline)),
                'net_count': len(successes)-len(damage), 'rescue_rate': proportion(len(successes), len(lost)),
                'collateral_rate': proportion(len(damage), len(steady)),
                'stable_retention': proportion(len(steady.intersection(hits[name])), len(steady)),
                'net_benefit_label': 'POSITIVE_NET_COUNT' if len(successes) > len(damage) else 'NO_NET_RECOVERY_BENEFIT'}
    recovered = lost.intersection(hits['r_only_G1'])
    if not eligible:
        conclusion = 'INCONCLUSIVE_ENGINEERING_OR_REFERENCE_GAP'
    elif not baseline:
        conclusion = 'INCONCLUSIVE_NO_NATIVE_FP16_ELIGIBLE_TARGETS'
    elif not lost:
        conclusion = 'FAIL_PHENOMENON_NOT_REPRODUCED_NO_ADDED_MISS'
    elif not recovered:
        conclusion = 'FAIL_PRIMARY_G1_SAME_TARGET_RECOVERY_NOT_REPRODUCED'
    else:
        conclusion = 'PASS_LOCAL_G1_SAME_TARGET_RECOVERY_EXISTENCE'
    killed = baseline.difference(hits['q_only_G1']); dual = recovered.intersection(killed)
    auxiliary = {'destruction_target_ids': sorted(killed), 'dual_overlap_target_ids': sorted(dual),
                 'overlap_rate': proportion(len(dual), len(lost)),
                 'status': 'INCONCLUSIVE_ENGINEERING_OR_REFERENCE_GAP' if not eligible else
                           'WEAK_SAME_TARGET_BIDIRECTIONAL_PATTERN' if dual else 'NO_POSITIVE_BIDIRECTIONAL_PATTERN_ON_THIS_SET',
                 'eligible_for_bidirectional_interpretation': eligible, 'pure_G1_causality_proven': False}
    return {'engineering_eligibility': eligible, 'primary_verdict': conclusion, 'FP16_hits': len(baseline),
            'q_all_hits': len(quantized), 'added_miss_target_ids': sorted(lost), 'stable_target_ids': sorted(steady),
            'recovery': restoration, 'auxiliary_G1': auxiliary}


def load_ground_truth(args, contract, reader):
    data = args.data_root.resolve(); manifest = reader.json(data / 'manifest.json', DATA_SHA)
    fixed = contract['fixed_data_and_model']
    require(fixed['dataset_manifest_sha256'] == DATA_SHA and fixed['evaluation_images'] == manifest['evaluation_count'] == 150 and
            fixed['evaluation_targets'] == manifest['evaluation_target_count'] == 865, 'Frozen data universe differs')
    for key, expected_key in (('canonical_holdout_list_sha256', 'canonical_holdout_list_sha256'),
                             ('canonical_calibration_list_sha256', 'canonical_calibration_list_sha256'),
                             ('source_holdout_file_bytes_sha256', 'source_holdout_file_sha256'),
                             ('source_calibration_manifest_sha256', 'calibration_manifest_sha256'), ('weights_sha256', 'weights_sha256')):
        require(manifest[key] == fixed[expected_key], 'Frozen source identity differs: ' + key)
    metadata, targets = [], {}
    require(len(manifest['data']['evaluation']) == 150, 'Need all150 evaluation images')
    for row in manifest['data']['evaluation']:
        image = contained(data, row['image']); label = contained(data, row['label'])
        width, height = jpeg_geometry(reader.raw(image, row['image_sha256']))
        require(image.stat().st_size == row['image_bytes'], 'Image byte count differs')
        text = reader.raw(label, row['label_sha256']).decode('utf-8'); truths = []
        for physical_line, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            parts = line.split(); require(len(parts) == 5, 'Malformed physical label line')
            cls = int(parts[0]); cx, cy, bw, bh = [float(value) for value in parts[1:]]
            require(0 <= cls < 10 and all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, bw, bh)), 'Invalid frozen label')
            target_id = PurePosixPath(row['image']).name + ':' + str(physical_line)
            require(target_id not in targets, 'Duplicate physical target ID')
            truth = {'target_id': target_id, 'class_id': cls,
                     'box': ((cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height),
                     'area_ratio': max(0., bw)*max(0., bh), 'image_basename': PurePosixPath(row['image']).name}
            targets[target_id] = truth; truths.append(truth)
        require(len(truths) == row['target_count'], 'Frozen target count differs')
        metadata.append((row, truths, width, height))
    require(len(targets) == 865 and len({r[0]['image'] for r in metadata}) == 150, 'Target/image universe incomplete')
    return metadata, targets


def run(args):
    reader = SourceReader(); reader.hash(Path(__file__).resolve())
    contract_path, matrix_path, summary_path, target_path = (p.resolve() for p in (args.contract, args.matrix, args.summary, args.target_matrix))
    contract = reader.json(contract_path, args.contract_sha256)
    require(contract['schema'] == 'rampq.gate05.dataset_acceptance_contract/v1' and
            contract['status'] == 'FROZEN_BEFORE_DATASET_INFERENCE' and [r['name'] for r in contract['configurations']] == ORDER and
            contract['evaluator_identity']['file_sha256'] == EVALUATOR_SHA and
            contract['evaluator_identity']['matching_source_sha256'] == MATCHING_SHA and
            contract['postprocess_identity']['implementation_sha256'] == POST_SHA and
            contract['protected_history']['original_complete_PyTorch_ORT_guard'] == 'FAILED_UNCHANGED', 'Wrong frozen result contract')
    reader.hash(ROOT / 'src/gate0.py', MATCHING_SHA)
    reader.hash(ROOT / 'scripts/evaluate_gate05_saved_target_matrix_v2.py', EVALUATOR_SHA)
    reader.hash(ROOT / 'scripts/postprocess_gate05_frozen_raw_matrix_v2.py', POST_SHA)
    settings = contract['common_input_postprocess_and_matching']
    require(settings['match']['primary_confidence_min'] == .25 and settings['match']['IoU_min'] == .5 and
            settings['NMS']['candidate_confidence'] == .001 and settings['NMS']['iou'] == .7 and
            settings['NMS']['max_det'] == 1000 and settings['NMS']['classes'] == 10, 'Matching/NMS contract changed')
    matrix = reader.json(matrix_path, args.matrix_sha256); summary = reader.json(summary_path, args.summary_sha256)
    original_targets = reader.lines(target_path, args.target_matrix_sha256)
    frozen_root = Path(contract['postprocess_execution']['matrix_root'])
    require(frozen_root.is_absolute() and frozen_root.resolve() == frozen_root == matrix_path.parent and
            frozen_root.is_dir() and not frozen_root.is_symlink(), 'Saved prediction matrix must live in frozen root')
    require(matrix['schema'] == 'rampq.gate05.saved_prediction_matrix/v1' and matrix['contract_sha256'] == args.contract_sha256 and
            matrix['dataset_manifest_sha256'] == DATA_SHA and matrix['NMS_calls'] == 1800 and
            matrix['model_inference_calls'] == 0 and matrix['target_matching_or_scoring_run'] is False, 'Saved matrix scope differs')
    expected_order = [{'configuration': name, 'backend': 'native' if backend == 'trt' else 'ort'}
                      for backend in ('trt', 'ort') for name in ORDER]
    streams = matrix['streams']
    require(len(streams) == 12 and [{'configuration': s['configuration'], 'backend': s['backend']} for s in streams] == expected_order,
            'Need all twelve saved streams in frozen order')
    proof_path = contained(frozen_root, matrix['engineering_evidence']['file'])
    proof = reader.json(proof_path, matrix['engineering_evidence']['sha256'])
    expected_ids = {r['name']: {key: r[key] for key in IDENTITY_KEYS} for r in contract['configurations']}
    require(proof['contract_sha256'] == args.contract_sha256 and proof['configuration_identities'] == expected_ids and
            proof['status'] in ('ELIGIBLE', 'INELIGIBLE', 'UNKNOWN') and type(proof['all_six_configuration_audits_eligible']) is bool,
            'Engineering proof identity/state differs')
    eligible = proof['status'] == 'ELIGIBLE' and proof['all_six_configuration_audits_eligible']
    metadata, truths = load_ground_truth(args, contract, reader)
    raw_index_path = args.raw_index.resolve() if args.raw_index else frozen_root / 'raw_family_index.json'
    require(raw_index_path.parent == frozen_root, 'Source raw index must share frozen matrix root')
    raw_index = reader.json(raw_index_path, matrix['source_raw_index_sha256'])
    require(raw_index['schema'] == 'rampq.gate05.frozen_raw_family_index/v1' and
            raw_index['contract_sha256'] == args.contract_sha256 and raw_index['raw_contract_sha256'] == contract['raw_contract_sha256'] and
            raw_index['input_manifest_sha256'] == contract['input_manifest_sha256'] and raw_index['dataset_manifest_sha256'] == DATA_SHA,
            'Source raw index contract/data differs')
    require(raw_index['engineering_evidence'] == matrix['engineering_evidence'], 'Raw and detection engineering evidence differ')
    input_root = frozen_root.joinpath(*PurePosixPath(raw_index['inputs_directory']).parts)
    require(frozen_root in input_root.resolve().parents and input_root.is_dir() and not input_root.is_symlink(), 'Unsafe shared input root')
    prepared = reader.json(contained(input_root, raw_index.get('input_manifest_file', 'manifest.json')), contract['input_manifest_sha256'])
    require(prepared['schema'] == 'rampq.gate05.shared_preprocessed_inputs/v1' and prepared['data_manifest_sha256'] == DATA_SHA and
            prepared['model_executions'] == 0 and len(prepared['images']) == 150 and
            [row['image_index'] for row in prepared['images']] == list(range(150)), 'Shared input metadata differs')
    raw_sources = {}
    require(len(raw_index['streams']) == 12, 'Incomplete source raw stream universe')
    for item in raw_index['streams']:
        key = ('native' if item['backend'] == 'trt' else item['backend'], item['configuration'])
        require(key not in raw_sources, 'Duplicate source raw stream')
        path = contained(frozen_root, item['file']); saved = reader.json(path, item['sha256'])
        require(saved['schema'] == 'rampq.gate05.frozen_raw_dataset_outputs/v1' and
                saved['status'] == 'ALL150_RAW_OUTPUTS_SERIALIZED_NO_SCORING' and len(saved['images']) == 150,
                'Incomplete source raw output manifest')
        raw_sources[key] = (path, saved)
    matched = {}; input_identities = {}; stream_reviews = []
    for stream in streams:
        cfg, backend = stream['configuration'], stream['backend']; source_path, source_raw = raw_sources[(backend, cfg)]
        require(stream['postprocess_settings'] == POST and stream['postprocess_identity'] == contract['postprocess_identity'], 'Postprocess identity/settings differ')
        require(stream['graph_sha256'] == expected_ids[cfg]['source_graph_sha256' if backend == 'native' else 'independent_ORT_reference_graph_sha256'] and
                stream['engine_sha256'] == (expected_ids[cfg]['engine_sha256'] if backend == 'native' else None), 'Detection stream graph/engine differs')
        path = contained(frozen_root, stream['file']); records = reader.lines(path, stream['sha256'])
        require(len(records) == 150, 'Need all150 detection records')
        result = {}; detections_total = 0
        for index, (record, datum, frozen, raw_row) in enumerate(zip(records, metadata, prepared['images'], source_raw['images'])):
            row, targets, width, height = datum
            require(record['image'] == row['image'] == frozen['image'] == raw_row['image'] and
                    record['image_sha256'] == row['image_sha256'] == frozen['image_sha256'] and
                    record['label_sha256'] == row['label_sha256'] == frozen['label_sha256'] and
                    raw_row['image_index'] == index and frozen['original_shape'] == [height, width] and
                    record['original_width'] == width and record['original_height'] == height,
                    'Prediction data/order/coordinate-frame identity differs')
            require(record['nms_complete'] is True and record['nms_warnings'] == [] and
                    record['NMS_capture']['complete'] is True and record['NMS_capture']['warnings'] == [] and
                    all(record[key] == frozen[key] == raw_row[key] for key in ('ratio', 'left', 'top')),
                    'NMS incomplete/warned or frozen transform differs')
            input_path = contained(frozen_root, record['input_file']); expected_input = contained(input_root, frozen['input_file'])
            require(input_path == expected_input and record['input_sha256'] == frozen['input_sha256'] == raw_row['input_sha256'] and
                    raw_row['input_array_sha256'] == frozen['array_sha256'], 'Per-stream frozen input identity differs')
            reader.hash(input_path, record['input_sha256'])
            previous = input_identities.setdefault(row['image'], record['input_sha256'])
            require(previous == record['input_sha256'], 'Twelve streams did not use same image input')
            raw_path = contained(frozen_root, record['raw_output_file'])
            expected_raw_path = contained(source_path.parent, raw_row['raw_file'])
            require(raw_path == expected_raw_path and record['raw_output_npz_key'] == 'main_output' and
                    record['raw_output_sha256'] == raw_row['raw_sha256'], 'Prediction is not bound to its exact saved main raw tensor')
            reader.hash(raw_path, record['raw_output_sha256'])
            detections = record['detections']; require(len(detections) <= 1000, 'NMS max_det exceeded')
            for detection in detections:
                cls, confidence, box = detection['class_id'], detection['confidence'], detection['box']
                require(type(cls) is int and 0 <= cls < 10 and isinstance(confidence, (int, float)) and
                        math.isfinite(confidence) and .001 <= confidence <= 1 and len(box) == 4 and
                        all(isinstance(v, (int, float)) and math.isfinite(v) for v in box) and
                        0 <= box[0] <= box[2] <= width and 0 <= box[1] <= box[3] <= height, 'Malformed original-pixel detection')
            recomputed = independently_match(targets, detections)
            require(not set(result).intersection(recomputed), 'Duplicate target in stream')
            result.update(recomputed); detections_total += len(detections)
        require(set(result) == set(truths) and len(result) == 865, 'Incomplete865target stream')
        matched[(backend, cfg)] = result
        stream_reviews.append({'configuration': cfg, 'backend': backend, 'prediction_stream_sha256': stream['sha256'],
                               'images': 150, 'targets': 865, 'detections': detections_total,
                               'hits': sum(row['matched'] for row in result.values()), 'independent_recompute': True})
    require(len(input_identities) == 150, 'Shared input inventory incomplete')
    require(len(original_targets) == 865 and [r['target_id'] for r in original_targets] == list(truths), 'Original scored target order/universe differs')
    for original in original_targets:
        target_id = original['target_id']; truth = truths[target_id]
        bits = {cfg: matched[('native', cfg)][target_id]['matched'] for cfg in ORDER}
        added, stable = bits['fp16_control'] and not bits['q_all'], bits['fp16_control'] and bits['q_all']
        expected = {'target_id': target_id, 'class_id': truth['class_id'], 'area_ratio': truth['area_ratio'],
                    'native': {cfg: matched[('native', cfg)][target_id] for cfg in ORDER},
                    'ort': {cfg: matched[('ort', cfg)][target_id] for cfg in ORDER}, 'added_miss': added, 'stable': stable,
                    'recovery_bits': {cfg: {'rescued': added and bits[cfg], 'still_missing': added and not bits[cfg],
                    'collateral': stable and not bits[cfg], 'outside_FP16': not bits['fp16_control'] and bits[cfg]} for cfg in RESTORED},
                    'q_only_G1_destruction': bits['fp16_control'] and not bits['q_only_G1'],
                    'G1_dual_overlap': added and not bits['q_only_G1'] and bits['r_only_G1']}
        require(original == expected, 'Original target row differs from independent matching/sets: ' + target_id)
    native = {cfg: matched[('native', cfg)] for cfg in ORDER}
    recomputed_summary = independent_summary(native, eligible)
    for key, value in recomputed_summary.items():
        require(summary[key] == value, 'Original summary differs: ' + key)
    classes = {}
    for cls in range(10):
        subset = {cfg: {key: row for key, row in values.items() if truths[key]['class_id'] == cls} for cfg, values in native.items()}
        totals = independent_summary(subset, False)
        classes[str(cls)] = {'targets': sum(row['class_id'] == cls for row in truths.values()), 'FP16_hits': totals['FP16_hits'],
                'q_all_hits': totals['q_all_hits'], 'added_misses': len(totals['added_miss_target_ids']), 'stable': len(totals['stable_target_ids']),
                'recovery': totals['recovery'], 'auxiliary_G1': totals['auxiliary_G1'], 'descriptive_only_no_subgroup_verdict': True}
    disagreements = {cfg: sum(matched[('native', cfg)][key]['matched'] != matched[('ort', cfg)][key]['matched'] for key in truths) for cfg in ORDER}
    require(summary['class_descriptive_summaries'] == classes and summary['native_ORT_target_disagreements'] == disagreements,
            'Class counts/sets or native-ORT target disagreement differs')
    expected_metadata = {'schema': 'rampq.gate05.saved_target_matrix_result/v1', 'contract_sha256': args.contract_sha256,
            'matrix_sha256': args.matrix_sha256, 'evaluator_sha256': EVALUATOR_SHA, 'matching_source_sha256': MATCHING_SHA,
            'images': 150, 'targets': 865, 'native_streams': 6, 'ORT_streams': 6, 'shared_image_input_hashes': input_identities,
            'original_complete_model_guard': 'FAILED_UNCHANGED', 'engineering_evidence_sha256': matrix['engineering_evidence']['sha256'],
            'new_reference_role': 'independent_export_semantics', 'new_numerical_tolerance': False,
            'model_inference_executions_by_evaluator': 0, 'NMS_executions_by_evaluator': 0, 'overall_gate_completion_ownership': 'ROOT_FINAL_REVIEW'}
    for key, expected in expected_metadata.items():
        require(summary[key] == expected, 'Original summary metadata differs: ' + key)
    output = args.output.resolve()
    require(not output.exists() and output.parent.is_dir() and input_root not in output.parents and
            str(output) not in reader.identities, 'Fresh report outside protected input package required')
    reader.recheck()
    report = {'schema': 'rampq.gate05.independent_saved_target_result_review/v1',
              'status': 'ALL_SAVED_MATCHING_BITS_TARGET_SETS_AND_VERDICT_INDEPENDENTLY_RECOMPUTED',
              'verification_tool_sha256': digest(Path(__file__).resolve()), 'reviewed_UTC': datetime.now(timezone.utc).isoformat(),
              'matching_implementation': 'Independent standard-library implementation; original/evaluator functions are not imported, extracted or called.',
              'contract_sha256': args.contract_sha256, 'matrix_sha256': args.matrix_sha256,
              'summary_sha256': args.summary_sha256, 'target_matrix_sha256': args.target_matrix_sha256,
              'images': 150, 'physical_targets': 865, 'streams': stream_reviews, 'target_hit_bits_verified': 865*12,
              'full_target_rows_exact': True, 'matching_assignments_IoU_confidence_detection_IDs_exact': True,
              'summary_set_rate_net_and_auxiliary_verdicts_exact': True, 'class_summaries_exact': True,
              'native_ORT_disagreements_exact': True, 'engineering_eligibility': eligible,
              'recomputed_primary_verdict': recomputed_summary['primary_verdict'], 'recomputed_summary': recomputed_summary,
              'source_hashes_before_and_after': reader.identities, 'protected_sources_unchanged': True,
              'new_model_calls': 0, 'new_NMS_calls': 0, 'original_results_modified': False,
              'configuration_selection_or_threshold_changes': False, 'original_complete_model_guard': 'FAILED_UNCHANGED',
              'original_numerical_guard_repaired': False, 'overall_Gate_completion_decision': 'ROOT_FINAL_REVIEW',
              'limits': 'Independent recomputation of saved detections and frozen matching only; NMS is authenticated by saved pinned common postprocessing, not rerun. Conditional deployed recovery does not establish pureG1arithmeticcause, overall superiority, new independent holdout preregistration or repair of the historical PyTorch/ORT guard.'}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': report['status'], 'report': str(output), 'report_sha256': digest(output),
          'target_hit_bits_verified': 10380, 'primary_verdict': report['recomputed_primary_verdict'],
          'new_model_calls': 0, 'new_NMS_calls': 0}), flush=True)


def self_test():
    def gt(name='a', box=(0., 0., 1., 1.), cls=0):
        return {'target_id': name, 'class_id': cls, 'box': box, 'area_ratio': 1., 'image_basename': 'a.jpg'}
    def det(box=(0., 0., 1., 1.), conf=.25, cls=0):
        return {'box': box, 'confidence': conf, 'class_id': cls}
    checks = []
    def check(name, condition):
        require(condition, 'Synthetic independent result test failed: ' + name); checks.append(name)
    check('inclusive_confidence', independently_match([gt()], [det()])['a']['matched'])
    check('below_confidence', not independently_match([gt()], [det(conf=math.nextafter(.25, 0.))])['a']['matched'])
    check('same_class', not independently_match([gt()], [det(cls=1)])['a']['matched'])
    check('inclusive_IoU', independently_match([gt()], [det(box=(0., 0., 2., 1.))])['a']['matched'])
    check('below_IoU', not independently_match([gt()], [det(box=(0., 0., math.nextafter(2., 3.), 1.))])['a']['matched'])
    check('reverse_detection_tie', independently_match([gt()], [det(), det()])['a']['detection_index'] == 1)
    tied = independently_match([gt('a'), gt('b')], [det()])
    check('reverse_GT_tie', not tied['a']['matched'] and tied['b']['matched'])
    check('IoU_before_confidence', independently_match([gt()], [det(box=(0., 0., 2., 1.), conf=.99), det()])['a']['detection_index'] == 1)
    check('empty_detection', not independently_match([gt()], [])['a']['matched'])
    def family(F, Q, R):
        hit_sets = {'fp16_control': F, 'q_all': Q, 'r_only_G1': R, 'r_only_G1_G5': R, 'r_only_G1_G2': R, 'q_only_G1': set()}
        return {cfg: {key: {'matched': key in hits} for key in ('a', 'b', 'c')} for cfg, hits in hit_sets.items()}
    data = family({'a', 'b', 'c'}, {'b', 'c'}, {'a'})
    positive = independent_summary(data, True)
    check('existence_with_negative_net', positive['primary_verdict'] == 'PASS_LOCAL_G1_SAME_TARGET_RECOVERY_EXISTENCE' and positive['recovery']['r_only_G1']['net_count'] == -1)
    check('engineering_precedes_existence', independent_summary(data, False)['primary_verdict'] == 'INCONCLUSIVE_ENGINEERING_OR_REFERENCE_GAP')
    check('auxiliary_eligibility_precedes_overlap', independent_summary(data, False)['auxiliary_G1']['status'] == 'INCONCLUSIVE_ENGINEERING_OR_REFERENCE_GAP')
    check('no_baseline', independent_summary(family(set(), set(), set()), True)['primary_verdict'] == 'INCONCLUSIVE_NO_NATIVE_FP16_ELIGIBLE_TARGETS')
    check('no_added_miss', independent_summary(family({'a'}, {'a'}, {'a'}), True)['primary_verdict'] == 'FAIL_PHENOMENON_NOT_REPRODUCED_NO_ADDED_MISS')
    check('no_primary_rescue', independent_summary(family({'a', 'b'}, {'b'}, {'b'}), True)['primary_verdict'] == 'FAIL_PRIMARY_G1_SAME_TARGET_RECOVERY_NOT_REPRODUCED')
    check('undefined_zero_denominator', proportion(0, 0) == {'numerator': 0, 'denominator': 0, 'value': None, 'undefined_reason': 'ZERO_DENOMINATOR'})
    check('no_model_or_NMS_import', not any(name == p or name.startswith(p+'.') for name in sys.modules for p in ('torch', 'numpy', 'tensorrt', 'onnxruntime', 'onnx', 'ultralytics')))
    print(json.dumps({'status': 'INDEPENDENT_SYNTHETIC_MATCHING_AND_SET_TESTS_PASS', 'checks': checks,
          'new_model_calls': 0, 'new_NMS_calls': 0, 'real_saved_predictions_read': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'matrix', 'summary', 'target-matrix'):
        parser.add_argument('--' + name, type=Path); parser.add_argument('--' + name + '-sha256')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'artifacts/gate05/data_v1')
    parser.add_argument('--raw-index', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        require(all(getattr(args, name) and getattr(args, name+'_sha256') for name in ('contract', 'matrix', 'summary', 'target_matrix')) and args.output,
                'Exact frozen contract/matrix/summary/target-matrix identities and fresh report required')
        run(args)


if __name__ == '__main__':
    main()
