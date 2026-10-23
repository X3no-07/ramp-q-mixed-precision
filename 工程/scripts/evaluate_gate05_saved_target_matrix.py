"""Evaluate authenticated saved detections only; never load or run a model.

--check-data-only accepts the draft for immutable data preflight only. Scoring
rejects drafts and requires an explicitly frozen contract, authenticated matrix,
and engineering-evidence receipt. NMS is performed by a separately pinned common
postprocessor; this evaluator preserves its detection order and never repeats it.
"""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/gate05/data_v1'
LEGACY = ROOT / 'src/gate0.py'
LEGACY_SHA = 'f3c4d80658b79ae685e569b37c5ea89af9e6e9dd4a63ce86aff2806a7fcc5adb'
MANIFEST_SHA = 'c4dac80ad7c59524194a9501f5ff8126cc0517e84f29f4106068c0c4b27a93aa'
CONFIGURATIONS = ('fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1')
POSTPROCESS = {'candidate_confidence': 0.001, 'nms_iou': 0.7, 'max_det': 1000,
               'nc': 10, 'max_time_img': 2.0, 'agnostic': False, 'multi_label': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def contained(root, relative):
    path = (root / relative).resolve()
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts and root in path.parents and
            path.is_file() and not path.is_symlink(), 'Unsafe or missing saved file: ' + str(relative))
    return path


def jpeg_size(path):
    """Read original JPEG dimensions without decoding pixels or importing a model stack."""
    data = path.read_bytes()
    require(data[:2] == b'\xff\xd8', 'Frozen evaluation image is not JPEG')
    position = 2
    sof = {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}
    while position < len(data):
        require(data[position] == 0xff, 'Malformed JPEG marker')
        while position < len(data) and data[position] == 0xff:
            position += 1
        require(position < len(data), 'Incomplete JPEG marker')
        marker = data[position]
        position += 1
        if marker in {0xd8, 0xd9, 0x01} or 0xd0 <= marker <= 0xd7:
            continue
        require(position + 2 <= len(data), 'Incomplete JPEG segment')
        length = struct.unpack('>H', data[position:position+2])[0]
        require(length >= 2 and position + length <= len(data), 'Malformed JPEG segment length')
        if marker in sof:
            require(length >= 8, 'Incomplete JPEG dimensions')
            height, width = struct.unpack('>HH', data[position+3:position+7])
            require(width > 0 and height > 0, 'Empty JPEG dimensions')
            return width, height
        require(marker != 0xda, 'JPEG image data reached before dimensions')
        position += length
    raise ValueError('No JPEG dimensions found')


def ground_truths(row, data_root):
    width, height = jpeg_size(contained(data_root, row['image']))
    targets = []
    for physical_index, line in enumerate(contained(data_root, row['label']).read_text(encoding='utf-8').splitlines()):
        if not line.strip():
            continue
        parts = line.split()
        require(len(parts) == 5, 'Invalid physical label line; target omission forbidden')
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        require(0 <= class_id < 10 and all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'Invalid frozen label')
        cx, cy, bw, bh = values
        targets.append(SimpleNamespace(target_id=Path(row['image']).name + ':' + str(physical_index), class_id=class_id,
            box=((cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height),
            area_ratio=max(0.0, bw)*max(0.0, bh), contrast=0.0))
    require(len(targets) == row['target_count'], 'Frozen label target count differs')
    return targets, width, height


def verify_data(data_root, contract):
    path = data_root / 'manifest.json'
    require(sha(path) == MANIFEST_SHA, 'Frozen dataset manifest changed')
    manifest = read(path)
    fixed = contract['fixed_data_and_model']
    require(fixed['dataset_manifest_sha256'] == MANIFEST_SHA and fixed['evaluation_images'] == 150 and
            fixed['evaluation_targets'] == 865 and manifest['evaluation_count'] == 150 and
            manifest['evaluation_target_count'] == 865 and manifest['calibration_count'] == 64,
            'Frozen image/target/calibration counts differ')
    for a, b in (('canonical_holdout_list_sha256', 'canonical_holdout_list_sha256'),
                 ('canonical_calibration_list_sha256', 'canonical_calibration_list_sha256'),
                 ('source_holdout_file_bytes_sha256', 'source_holdout_file_sha256'),
                 ('source_calibration_manifest_sha256', 'calibration_manifest_sha256'),
                 ('weights_sha256', 'weights_sha256')):
        require(manifest[a] == fixed[b], 'Frozen source/list identity differs: ' + a)
    hashes_by_split = {}
    for split, expected_count in (('calibration', 64), ('evaluation', 150)):
        rows = manifest['data'][split]
        require(len(rows) == expected_count and len({r['image'] for r in rows}) == expected_count, 'Incomplete image list')
        hashes_by_split[split] = set()
        for row in rows:
            image = contained(data_root, row['image'])
            require(image.stat().st_size == row['image_bytes'] and sha(image) == row['image_sha256'], 'Image identity differs')
            hashes_by_split[split].add(row['image_sha256'])
            if split == 'evaluation':
                require(sha(contained(data_root, row['label'])) == row['label_sha256'], 'Label identity differs')
    require(len(hashes_by_split['calibration']) == 64 and len(hashes_by_split['evaluation']) == 150 and
            not hashes_by_split['calibration'] & hashes_by_split['evaluation'], 'Calibration/evaluation content overlap')
    target_map, metadata = {}, []
    for row in manifest['data']['evaluation']:
        targets, width, height = ground_truths(row, data_root)
        for target in targets:
            require(target.target_id not in target_map, 'Duplicate physical target identity')
            target_map[target.target_id] = target
        metadata.append((row, targets, width, height))
    require(len(target_map) == 865, 'Incomplete fixed target universe')
    return manifest, metadata, target_map


def legacy_matching(path):
    """Reuse the frozen original function and its exact pre-rounding assignment AST."""
    require(sha(path) == LEGACY_SHA, 'Original Gate0 matching source changed')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    original = copy.deepcopy(functions['match_targets'])
    stop = next(index for index, node in enumerate(original.body) if isinstance(node, ast.AnnAssign) and
                isinstance(node.target, ast.Name) and node.target.id == 'rows')
    raw = copy.deepcopy(original)
    raw.name = '_unrounded_assignment'
    raw.body = raw.body[:stop] + [ast.Return(value=ast.Name(id='assigned_gt', ctx=ast.Load()))]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
        copy.deepcopy(functions['box_iou']), original, raw], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path) + ':matching_only', 'exec'), namespace)
    return namespace


def match_saved(targets, detections, matcher):
    assignments = matcher['_unrounded_assignment'](targets, detections, 0.25, 0.5)
    legacy_rows = matcher['match_targets'](targets, detections, 0.25, 0.5)
    result = {}
    for index, target in enumerate(targets):
        assignment = assignments.get(index)
        require(legacy_rows[index]['matched'] == int(assignment is not None), 'Raw and original matching disagree')
        result[target.target_id] = {'matched': assignment is not None,
            'detection_index': assignment[0] if assignment else None,
            'match_iou': assignment[1] if assignment else None,
            'match_confidence': assignment[2] if assignment else None,
            'class_id': target.class_id, 'area_ratio': target.area_ratio}
    return result


def validate_frozen(contract, expected_sha, contract_path, matching_source):
    require(sha(contract_path) == expected_sha, 'Frozen dataset contract identity differs')
    require(contract.get('schema') == 'rampq.gate05.dataset_acceptance_contract/v1' and
            contract.get('status') == 'FROZEN_BEFORE_DATASET_INFERENCE', 'Drafts do not authorize target evaluation')
    evaluator = contract['evaluator_identity']
    require(evaluator['file_sha256'] == sha(Path(__file__).resolve()) and
            evaluator['matching_source_sha256'] == LEGACY_SHA == sha(matching_source), 'Evaluator identity differs')
    require([row['name'] for row in contract['configurations']] == list(CONFIGURATIONS), 'Six frozen configurations required')
    for row in contract['configurations']:
        for key in ('source_graph_sha256', 'engine_sha256', 'build_request_sha256', 'native_precision_audit_sha256',
                    'float_fusion_comparability_audit_sha256', 'independent_ORT_reference_graph_sha256'):
            require(digest_string(row[key]), 'Unbound model/audit identity: ' + row['name'] + ':' + key)
    common = contract['common_input_postprocess_and_matching']
    require(common['NMS']['candidate_confidence'] == 0.001 and common['NMS']['iou'] == 0.7 and
            common['NMS']['max_det'] == 1000 and common['NMS']['classes'] == 10 and
            common['NMS']['agnostic'] is False and common['NMS']['multi_label'] is False and
            common['match']['primary_confidence_min'] == 0.25 and common['match']['IoU_min'] == 0.5,
            'Original fixed postprocessing/matching criteria changed')
    require(contract['protected_history']['original_complete_PyTorch_ORT_guard'] == 'FAILED_UNCHANGED', 'Old guard changed')
    post = contract['postprocess_identity']
    require(isinstance(post, dict) and digest_string(post['implementation_sha256']) and
            digest_string(post['environment_sha256']) and post['device'] == 'cpu' and
            isinstance(post['library_versions'], dict) and all(isinstance(post['library_versions'].get(name), str) and
            post['library_versions'][name] for name in ('torch', 'torchvision', 'ultralytics', 'numpy')),
            'Common pinned CPU postprocessor code/environment/version identities required')


def score_stream(matrix_root, stream, metadata, matcher, input_identities, fixed_identity):
    import numpy as np
    path = contained(matrix_root, stream['file'])
    require(sha(path) == stream['sha256'] and stream['postprocess_settings'] == POSTPROCESS and
            stream['postprocess_identity'] == fixed_identity, 'Saved postprocessor identity/settings differ')
    lines = path.read_text(encoding='utf-8').splitlines()
    require(len(lines) == 150 and all(line.strip() for line in lines), 'Incomplete 150-image prediction stream')
    targets_all = {}
    for (row, targets, width, height), line in zip(metadata, lines):
        record = json.loads(line)
        require(record['image'] == row['image'] and record['image_sha256'] == row['image_sha256'] and
                record['label_sha256'] == row['label_sha256'], 'Saved prediction image/label order or identity differs')
        require(record['nms_complete'] is True and record.get('nms_warnings', []) == [], 'Incomplete or timed-out NMS')
        require(record['original_width'] == width and record['original_height'] == height, 'Coordinate frame differs')
        input_path = contained(matrix_root, record['input_file'])
        input_bytes = input_path.read_bytes()
        require(digest_string(record['input_sha256']) and hashlib.sha256(input_bytes).hexdigest() == record['input_sha256'],
                'Preprocessed input identity differs')
        previous = input_identities.setdefault(row['image'], record['input_sha256'])
        require(previous == record['input_sha256'], 'Native/ORT configurations did not use the same image input')
        array = np.load(io.BytesIO(input_bytes), allow_pickle=False)
        require(array.dtype == np.float32 and array.shape == (1, 3, 640, 640) and array.flags.c_contiguous and
                bool(np.isfinite(array).all()), 'Frozen saved input contract differs')
        raw_path = contained(matrix_root, record['raw_output_file'])
        raw_bytes = raw_path.read_bytes()
        require(hashlib.sha256(raw_bytes).hexdigest() == record['raw_output_sha256'], 'Saved raw output identity differs')
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as raw:
            primary = raw[record.get('raw_output_npz_key', 'output0')]
            require(primary.dtype == np.float32 and primary.shape == (1, 14, 8400) and
                    bool(np.isfinite(primary).all()), 'Wrong or nonfinite saved primary raw output')
        detections = []
        require(len(record['detections']) <= 1000, 'NMS detection inventory exceeds frozen max_det')
        for detection in record['detections']:
            cls, confidence, box = detection['class_id'], detection['confidence'], detection['box']
            require(type(cls) is int and 0 <= cls < 10 and isinstance(confidence, (int, float)) and
                    math.isfinite(confidence) and 0.001 <= confidence <= 1 and len(box) == 4 and
                    all(isinstance(v, (int, float)) and math.isfinite(v) for v in box) and
                    0 <= box[0] <= box[2] <= width and 0 <= box[1] <= box[3] <= height, 'Invalid saved original-pixel detection')
            detections.append(SimpleNamespace(class_id=cls, confidence=float(confidence), box=tuple(box)))
        matched = match_saved(targets, detections, matcher)
        for value in matched.values():
            value['detection_id'] = (Path(row['image']).name + ':det:' + str(value['detection_index'])
                                     if value['matched'] else None)
        require(not set(targets_all) & set(matched), 'Duplicate target across images')
        targets_all.update(matched)
    require(len(targets_all) == 865, 'Incomplete target matrix stream')
    require(sha(path) == stream['sha256'], 'Saved prediction stream changed during evaluation')
    return targets_all


def rate(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator/denominator if denominator else None,
            'undefined_reason': None if denominator else 'ZERO_DENOMINATOR'}


def summarize(native, eligible):
    hits = {cfg: {target for target, value in rows.items() if value['matched']} for cfg, rows in native.items()}
    F, Q = hits['fp16_control'], hits['q_all']
    added, stable = F-Q, F & Q
    recovery = {}
    for cfg in ('r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2'):
        R = hits[cfg]
        rescued, collateral = added & R, stable-R
        recovery[cfg] = {'rescued_target_ids': sorted(rescued), 'still_missing_target_ids': sorted(added-R),
            'collateral_target_ids': sorted(collateral), 'outside_FP16_target_ids': sorted(R-F),
            'net_count': len(rescued)-len(collateral), 'rescue_rate': rate(len(rescued), len(added)),
            'collateral_rate': rate(len(collateral), len(stable)), 'stable_retention': rate(len(stable & R), len(stable)),
            'net_benefit_label': 'POSITIVE_NET_COUNT' if len(rescued)>len(collateral) else 'NO_NET_RECOVERY_BENEFIT'}
    primary = added & hits['r_only_G1']
    if not eligible:
        verdict = 'INCONCLUSIVE_ENGINEERING_OR_REFERENCE_GAP'
    elif not F:
        verdict = 'INCONCLUSIVE_NO_NATIVE_FP16_ELIGIBLE_TARGETS'
    elif not added:
        verdict = 'FAIL_PHENOMENON_NOT_REPRODUCED_NO_ADDED_MISS'
    elif not primary:
        verdict = 'FAIL_PRIMARY_G1_SAME_TARGET_RECOVERY_NOT_REPRODUCED'
    else:
        verdict = 'PASS_LOCAL_G1_SAME_TARGET_RECOVERY_EXISTENCE'
    destruction = F-hits['q_only_G1']
    dual = added & destruction & hits['r_only_G1']
    return {'engineering_eligibility': eligible, 'primary_verdict': verdict, 'FP16_hits': len(F), 'q_all_hits': len(Q),
        'added_miss_target_ids': sorted(added), 'stable_target_ids': sorted(stable), 'recovery': recovery,
        'auxiliary_G1': {'destruction_target_ids': sorted(destruction), 'dual_overlap_target_ids': sorted(dual),
            'overlap_rate': rate(len(dual), len(added)), 'status': 'WEAK_SAME_TARGET_BIDIRECTIONAL_PATTERN' if dual else
            'NO_POSITIVE_BIDIRECTIONAL_PATTERN_ON_THIS_SET', 'pure_G1_causality_proven': False}}


def evaluate(args):
    require(args.contract_sha256 and args.matrix and args.matrix_sha256 and args.output, 'Frozen contract/matrix/output arguments required')
    contract_path, matrix_path = args.contract.resolve(), args.matrix.resolve()
    contract = read(contract_path)
    validate_frozen(contract, args.contract_sha256, contract_path, args.matching_source.resolve())
    data_root = args.data_root.resolve()
    _, metadata, targets = verify_data(data_root, contract)
    require(sha(matrix_path) == args.matrix_sha256, 'Saved matrix receipt identity differs')
    matrix, matrix_root = read(matrix_path), matrix_path.parent
    require(matrix['schema'] == 'rampq.gate05.saved_prediction_matrix/v1' and
            matrix['dataset_manifest_sha256'] == MANIFEST_SHA and matrix['contract_sha256'] == args.contract_sha256,
            'Saved matrix contract/data identity differs')
    streams = matrix['streams']
    require(len(streams) == 12 and {(s['backend'], s['configuration']) for s in streams} ==
            {(backend, cfg) for backend in ('native', 'ort') for cfg in CONFIGURATIONS}, 'All six native and six ORT streams required')
    proof = matrix['engineering_evidence']
    proof_path = contained(matrix_root, proof['file'])
    require(sha(proof_path) == proof['sha256'], 'Engineering eligibility receipt changed')
    evidence = read(proof_path)
    require(evidence['contract_sha256'] == args.contract_sha256, 'Eligibility evidence belongs to another contract')
    require(evidence['status'] in ('ELIGIBLE', 'INELIGIBLE', 'UNKNOWN') and
            type(evidence['all_six_configuration_audits_eligible']) is bool,
            'Explicit engineering eligibility state required')
    expected_evidence = {row['name']: {key: row[key] for key in ('source_graph_sha256', 'engine_sha256',
        'build_request_sha256', 'native_precision_audit_sha256', 'float_fusion_comparability_audit_sha256',
        'independent_ORT_reference_graph_sha256')} for row in contract['configurations']}
    require(evidence['configuration_identities'] == expected_evidence,
            'Engineering evidence is not bound to all six exact frozen models and audits')
    eligible = evidence['status'] == 'ELIGIBLE' and evidence['all_six_configuration_audits_eligible'] is True
    output = args.output.resolve()
    require(not output.exists(), 'Exclusive new evaluation output required')
    output.mkdir(parents=True, exist_ok=False)
    try:
        matcher, input_identities, matched = legacy_matching(args.matching_source.resolve()), {}, {}
        bindings = {r['name']: r for r in contract['configurations']}
        for stream in streams:
            cfg, backend = stream['configuration'], stream['backend']
            require(stream['graph_sha256'] == bindings[cfg]['source_graph_sha256' if backend == 'native' else
                'independent_ORT_reference_graph_sha256'], 'Stream graph differs from frozen family')
            if backend == 'native':
                require(stream['engine_sha256'] == bindings[cfg]['engine_sha256'], 'Stream engine differs from frozen family')
            matched[(backend, cfg)] = score_stream(matrix_root, stream, metadata, matcher, input_identities,
                                                  contract['postprocess_identity'])
            require(set(matched[(backend, cfg)]) == set(targets), 'Target universe differs across family')
        native = {cfg: matched[('native', cfg)] for cfg in CONFIGURATIONS}
        summary = summarize(native, eligible)
        with (output/'target_matrix.jsonl').open('x', encoding='utf-8', newline='\n') as handle:
            for target_id, target in targets.items():
                bits = {cfg: matched[('native', cfg)][target_id]['matched'] for cfg in CONFIGURATIONS}
                added = bits['fp16_control'] and not bits['q_all']
                stable = bits['fp16_control'] and bits['q_all']
                row = {'target_id': target_id, 'class_id': target.class_id, 'area_ratio': target.area_ratio,
                    'native': {cfg: matched[('native', cfg)][target_id] for cfg in CONFIGURATIONS},
                    'ort': {cfg: matched[('ort', cfg)][target_id] for cfg in CONFIGURATIONS},
                    'added_miss': added, 'stable': stable,
                    'recovery_bits': {cfg: {'rescued': added and bits[cfg], 'still_missing': added and not bits[cfg],
                        'collateral': stable and not bits[cfg], 'outside_FP16': not bits['fp16_control'] and bits[cfg]}
                        for cfg in ('r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2')},
                    'q_only_G1_destruction': bits['fp16_control'] and not bits['q_only_G1'],
                    'G1_dual_overlap': added and not bits['q_only_G1'] and bits['r_only_G1']}
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        class_counts = {}
        for cls in range(10):
            class_native = {cfg: {t: value for t, value in rows.items() if targets[t].class_id == cls}
                            for cfg, rows in native.items()}
            class_summary = summarize(class_native, eligible=False)
            class_counts[str(cls)] = {'targets': sum(t.class_id == cls for t in targets.values()),
                'FP16_hits': class_summary['FP16_hits'], 'q_all_hits': class_summary['q_all_hits'],
                'added_misses': len(class_summary['added_miss_target_ids']),
                'stable': len(class_summary['stable_target_ids']), 'recovery': class_summary['recovery'],
                'auxiliary_G1': class_summary['auxiliary_G1'], 'descriptive_only_no_subgroup_verdict': True}
        require(sha(contract_path) == args.contract_sha256 and sha(matrix_path) == args.matrix_sha256 and
                sha(proof_path) == proof['sha256'] and sha(data_root/'manifest.json') == MANIFEST_SHA and
                sha(args.matching_source.resolve()) == LEGACY_SHA and
                sha(Path(__file__).resolve()) == contract['evaluator_identity']['file_sha256'],
                'Frozen evaluator, contract, eligibility receipt or dataset became stale')
        summary.update(schema='rampq.gate05.saved_target_matrix_result/v1', client_at=datetime.now().astimezone().isoformat(),
            contract_sha256=args.contract_sha256, matrix_sha256=args.matrix_sha256, evaluator_sha256=sha(Path(__file__).resolve()),
            matching_source_sha256=LEGACY_SHA, images=150, targets=865, native_streams=6, ORT_streams=6,
            shared_image_input_hashes=input_identities, original_complete_model_guard='FAILED_UNCHANGED',
            engineering_evidence_sha256=proof['sha256'], class_descriptive_summaries=class_counts,
            new_reference_role='independent_export_semantics', new_numerical_tolerance=False,
            model_inference_executions_by_evaluator=0, NMS_executions_by_evaluator=0,
            overall_gate_completion_ownership='ROOT_FINAL_REVIEW',
            native_ORT_target_disagreements={cfg:sum(matched[('native',cfg)][t]['matched'] !=
                matched[('ort',cfg)][t]['matched'] for t in targets) for cfg in CONFIGURATIONS})
        write(output/'summary.json', summary)
    except BaseException as error:
        write(output/'evaluation_failure.json', {'status':'INCONCLUSIVE_ENGINEERING_OR_REFERENCE_GAP',
            'error_type':type(error).__name__, 'error':str(error), 'raw_results_preserved':True,
            'original_complete_model_guard':'FAILED_UNCHANGED', 'no_model_or_NMS_execution':True})
        raise


def self_test(matching_source):
    matcher = legacy_matching(matching_source)
    gt = lambda name, cls=0, box=(0.,0.,1.,1.): SimpleNamespace(target_id=name,class_id=cls,box=box,area_ratio=1.,contrast=0.)
    det = lambda confidence=.25, cls=0, box=(0.,0.,1.,1.): SimpleNamespace(class_id=cls,confidence=confidence,box=box)
    checks = []
    def check(name, value):
        checks.append({'name':name,'passed':bool(value)})
        require(value, 'Matching corner case failed: '+name)
    check('same_class_required',not match_saved([gt('a')],[det(cls=1)],matcher)['a']['matched'])
    check('confidence_equal_quarter_inclusive',match_saved([gt('a')],[det()],matcher)['a']['matched'])
    check('just_below_confidence_excluded',not match_saved([gt('a')],[det(math.nextafter(.25,0.))],matcher)['a']['matched'])
    check('IoU_equal_half_inclusive',match_saved([gt('a')],[det(box=(0.,0.,2.,1.))],matcher)['a']['match_iou']==.5)
    check('just_below_IoU_excluded',not match_saved([gt('a')],[det(box=(0.,0.,math.nextafter(2.,3.),1.))],matcher)['a']['matched'])
    check('detection_tie_reverse_index',match_saved([gt('a')],[det(),det()],matcher)['a']['detection_index']==1)
    rows=match_saved([gt('a'),gt('b')],[det()],matcher)
    check('one_detection_one_GT_reverse_GT_tie',not rows['a']['matched'] and rows['b']['matched'])
    rows=match_saved([gt('a')],[det(.99,box=(0.,0.,2.,1.)),det(.25)],matcher)
    check('IoU_priority_before_confidence',rows['a']['detection_index']==1)
    check('empty_detection_inventory',not match_saved([gt('a')],[],matcher)['a']['matched'])
    def family(F, Q, R):
        sets={'fp16_control':F,'q_all':Q,'r_only_G1':R,'r_only_G1_G5':R,'r_only_G1_G2':R,'q_only_G1':set()}
        return {cfg:{t:{'matched':t in hits} for t in ('a','b','c')} for cfg,hits in sets.items()}
    sample=family({'a','b','c'},{'b','c'},{'a'})
    result=summarize(sample,True)
    check('narrow_existence_preserves_negative_net',result['primary_verdict']=='PASS_LOCAL_G1_SAME_TARGET_RECOVERY_EXISTENCE'
        and result['recovery']['r_only_G1']['net_count']==-1 and
        result['recovery']['r_only_G1']['net_benefit_label']=='NO_NET_RECOVERY_BENEFIT')
    check('engineering_gap_has_priority',summarize(sample,False)['primary_verdict']=='INCONCLUSIVE_ENGINEERING_OR_REFERENCE_GAP')
    check('zero_FP16_baseline_inconclusive',summarize(family(set(),set(),set()),True)['primary_verdict']==
        'INCONCLUSIVE_NO_NATIVE_FP16_ELIGIBLE_TARGETS')
    check('zero_added_miss_scientific_nonreproduction',summarize(family({'a'},{'a'},{'a'}),True)['primary_verdict']==
        'FAIL_PHENOMENON_NOT_REPRODUCED_NO_ADDED_MISS')
    check('no_primary_recovery_not_substituted',summarize(family({'a','b'},{'b'},{'b'}),True)['primary_verdict']==
        'FAIL_PRIMARY_G1_SAME_TARGET_RECOVERY_NOT_REPRODUCED')
    check('zero_denominator_is_null',summarize(family({'a'},set(),{'a'}),True)['recovery']['r_only_G1']['stable_retention']==
        {'numerator':0,'denominator':0,'value':None,'undefined_reason':'ZERO_DENOMINATOR'})
    print(json.dumps({'status':'SYNTHETIC_MATCHING_CORNER_CASES_PASSED','checks':checks,
        'matching_source_sha256':LEGACY_SHA,'model_or_dataset_evaluation_run':False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--contract-sha256')
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--matching-source',type=Path,default=LEGACY)
    parser.add_argument('--matrix',type=Path)
    parser.add_argument('--matrix-sha256')
    parser.add_argument('--output',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-data-only',action='store_true')
    mode.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test(args.matching_source.resolve())
    elif args.check_data_only:
        require(args.contract is not None,'Draft or frozen contract path required for data-only check')
        contract=read(args.contract.resolve())
        _,metadata,targets=verify_data(args.data_root.resolve(),contract)
        print(json.dumps({'status':'FROZEN_DATA_IDENTITIES_AND_TARGET_UNIVERSE_VERIFIED_NO_PREDICTIONS_READ',
            'data_root':str(args.data_root.resolve()),'dataset_manifest_sha256':MANIFEST_SHA,
            'contract_file_sha256':sha(args.contract.resolve()),'images':len(metadata),'targets':len(targets),
            'prediction_or_model_or_NMS_evaluation_run':False}))
    else:
        require(args.contract is not None,'Frozen contract required')
        evaluate(args)


if __name__=='__main__':
    main()
