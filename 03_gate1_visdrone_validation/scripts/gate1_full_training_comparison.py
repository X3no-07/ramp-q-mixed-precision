"""One fixed 4530-image, 10-epoch, two-arm DEVELOPMENT comparison.

New artifacts only. No automatic retry, best-epoch selection, or sealed payload.
The original quantizers/loss implementations are imported without editing them.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/gate1/20261027_full_training_comparison_v1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['YOLO_OFFLINE'] = 'true'
import numpy as np
import torch
import gate1_frozen_box_scorer as S
import gate1_scorer_exact_features as FEATURES
P, N = S.P, S.N
# Imported historical modules assign this variable; restore the NEW output path.
os.environ['YOLO_CONFIG_DIR'] = str(OUT / 'settings')
read, write, identity, require, sha, now = P.read, P.write, P.identity, P.require, P.sha, P.now
SCHEMA = 'rampq.gate1.full_training_comparison/v1'
ARMS = ('ordinary_qat', 'native_intervention_correction')
BRANCHES = (*ARMS, 'ordinary_qat_frozen_scorer')
EPOCHS, BATCH, ACCUMULATE, TRAIN_COUNT, VAL_COUNT = 10, 4, 4, 4530, 414


def accumulation_window(batch_index, number_of_batches, accumulate=ACCUMULATE):
    """Zero-based batch: normalize a final partial window, then flush it."""
    require(0 <= batch_index < number_of_batches, 'Batch index outside epoch')
    start = batch_index // accumulate * accumulate
    width = min(accumulate, number_of_batches - start)
    return width, batch_index + 1 == start + width


def value_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def source_bindings():
    result = deepcopy(N.source_bindings())
    for name in ('gate1_full_training_comparison.py', 'launch_gate1_full_training_comparison.py',
                 'audit_gate1_full_training_comparison.py', 'gate1_frozen_box_scorer.py',
                 'gate1_scorer_exact_features.py', 'gate1_cached_ap_job.py'):
        result[name] = identity(ROOT / 'scripts' / name)
    path = ROOT / 'tests/test_gate1_full_training_comparison.py'
    result[path.name] = identity(path)
    return result


def mine_spec(q, r, gt, meta, params):
    """Same native intervention policy as the pilot, extended to new train IDs."""
    qp = P.predictions(q, params, meta['image_id'], 'q')
    rp = P.predictions(r, params, meta['image_id'], 'r')
    qm = P.MATCH.match_project_targets(gt, qp, meta['width'], meta['height'], confidence=.25, iou_threshold=.5)
    rm = P.MATCH.match_project_targets(gt, rp, meta['width'], meta['height'], confidence=.25, iou_threshold=.5)
    qhits = {x['target_id'] for x in qm['matches']}
    rhits = {x['target_id'] for x in rm['matches']}
    positives = [g for g in gt if g['score'] == 1 and 1 <= g['category'] <= 10]
    eligible = {g['target_id'] for g in qm['targets']} & {g['target_id'] for g in rm['targets']}
    boxes = r[0, :4].T.copy()
    boxes[:, :2] -= boxes[:, 2:] / 2; boxes[:, 2:] += boxes[:, :2]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - params['left']) / params['ratio'][0]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - params['top']) / params['ratio'][1]
    overlaps = P.box_iou_matrix(boxes, P.xyxy([g['bbox_xywh'] for g in positives]))
    pool = []
    for i, g in enumerate(positives):
        if g['target_id'] in eligible and g['target_id'] in qhits.symmetric_difference(rhits):
            pool.append({'gt_index': i, 'quality': float((overlaps[:, i] * r[0, 3 + g['category']]).max())})
    selected = sorted(pool, key=lambda x: (x['quality'], x['gt_index']))[:8]
    qfp = {x['detection_id'] for x in qm['detection_states'] if x['reason'] == 'false_positive'}
    rfp = {x['detection_id'] for x in rm['detection_states'] if x['reason'] == 'false_positive'}
    positive_eval = [g for g in gt if g['target_id'] in {v['target_id'] for v in rm['targets']}]
    clipped_gt = np.array([P.MATCH._xyxy(g['bbox_xywh'], meta['width'], meta['height']) for g in positive_eval]).reshape(-1, 4)
    classes = np.array([g['category'] for g in positive_eval])
    errors = []
    for d in rp:
        if d['detection_id'] not in rfp:
            continue
        ious = P.box_iou_matrix(np.array([P.MATCH._xyxy(d['bbox_xywh'], meta['width'], meta['height'])]), clipped_gt)[0]
        previous = [x for x in qp if x['detection_id'] in qfp and x['category'] == d['category']]
        already = bool(previous) and bool((P.box_iou_matrix(P.xyxy([d['bbox_xywh']]), P.xyxy([x['bbox_xywh'] for x in previous])) >= .5).any())
        if not already:
            errors.append({'anchor': d['raw_anchor_index'], 'category': d['category'] - 1, 'score': d['score'],
                           'score_bin': min(2, int((d['score'] - .25) / .25)), 'kind': N.error_kind(ious, classes, d['category']),
                           'new_after_restore': True, 'maximum_GT_IoU': float(ious.max()) if len(ious) else 0.})
    errors = sorted(errors, key=lambda x: (-x['score'], x['anchor']))[:8]
    return {'image_id': meta['image_id'], 'q_counts': qm['counts'], 'r_counts': rm['counts'],
            'rescued_targets': sorted(rhits - qhits), 'collateral_targets': sorted(qhits - rhits),
            ARMS[1]: {'positive_gt': [x['gt_index'] for x in selected], 'errors': errors},
            'training_to_original_params': params, 'native_TAL_targets_preserved': True}


def prepare():
    require(not OUT.exists(), 'Comparison output already exists; no implicit retry')
    old_path = P.OUT / 'contract.json'; old = read(old_path)
    train = read(old['train_manifest']['path']); val = read(old['training_val_manifest']['path'])
    P.FT.validate_membership(train, val)
    members = sorted(train['samples'], key=lambda r: r['image_id'])
    require(len(members) == TRAIN_COUNT and len(val['samples']) == VAL_COUNT, 'Fixed membership count differs')
    protocol = read(old['evaluation_protocol']['path'])
    evaluation = protocol['cohorts']['training_val']
    indexed_val = {r['image_id']: r for r in val['samples']}
    require(len(evaluation) == VAL_COUNT and all(indexed_val[r['image_id']]['image']['sha256'] == r['image_sha256'] and r['assigned_split'] == 'training_val' for r in evaluation), 'Evaluation member identity differs')
    selected = {r['image_id']: r for r in members}; metadata = {}
    for line in Path(old['training_annotation_assignment']['path']).read_text(encoding='utf-8-sig').splitlines():
        row = json.loads(line)
        if row['image_id'] in selected:
            require(row['assigned_split'] == 'train' and row['image_sha256'] == selected[row['image_id']]['image']['sha256'], 'Train metadata source/role differs')
            metadata[row['image_id']] = row
    require(set(metadata) == set(selected), 'Original annotation metadata incomplete')
    reused = []
    for i, row in enumerate(old['training_members']):
        require(row == selected[row['image_id']], 'Reused teacher member changed')
        reused.append({'image_id': row['image_id'], 'source_index': i,
                       'record': identity(N.OUT / 'attempt_01/training_supervision' / f'{i:04d}.json'),
                       'raw': identity(P.OUT / 'attempt_01/training_supervision' / f'{i:04d}_raw.npz')})
    require(len(reused) == 512, 'Expected exactly 512 completed teachers to reuse')
    reference = []
    for row in old['evaluation_members']:
        reference.append({'image_id': row['image_id'], 'raw': identity(P.PRIOR / 'attempt_01/fp32_reference' / (row['image_id'].split('/')[-1] + '.npz'))})
    batches = math.ceil(TRAIN_COUNT / BATCH)
    c = {'schema': SCHEMA, 'created_at_utc': now(), 'authorization': 'User 2026-10-27: Proceed as planned, use your own computer; conduct a fixed 10-round full-training-set development comparison for each of two groups',
         'previous_state': identity(ROOT / read(ROOT / 'artifacts/gate1/latest_execution_state.json')['snapshot']),
         'source_bindings': source_bindings(), 'framework': P.verify_framework_sources(),
         **{k: old[k] for k in ('candidate', 'architecture', 'formal_contract', 'train_manifest', 'training_val_manifest', 'evaluation_protocol', 'training_annotation_assignment')},
         'prior_contract': identity(old_path), 'native_pilot_contract': identity(N.OUT / 'contract.json'),
         'ranges': identity(P.PRIOR / 'attempt_01/mse_ranges.json'),
         'scorer_model': identity(S.OUT / 'attempt_01/score_base_model.json'),
         'training_members': members, 'training_annotation_metadata': [metadata[r['image_id']] for r in members],
         'evaluation_members': evaluation, 'reused_teachers': reused, 'FP32_saved_reference_subset': reference,
         'train_count': TRAIN_COUNT, 'evaluation_count': VAL_COUNT, 'epochs': EPOCHS, 'batch': BATCH, 'accumulate': ACCUMULATE,
         'batches_per_epoch': batches, 'optimizer_steps_per_epoch': math.ceil(batches / ACCUMULATE),
         'optimizer_steps_per_arm': EPOCHS * math.ceil(batches / ACCUMULATE), 'training_seed': P.SEED,
         'arms': list(ARMS), 'evaluation_order': list(BRANCHES), 'final_candidate': 'epoch_10_only_no_best_epoch_selection',
         'precision': old['precision'], 'quantizer': old['quantizer'],
         'training': 'SGD lr1e-4 momentum0.9 weight_decay1e-4; clip10; AMPoff; EMAoff; no augmentation; frozen BN/DFL/scales; identical seed/order/init; flush partial accumulation each epoch and divide loss by actual window microbatch count',
         'auxiliary': 'Same existing native correction:0.25*(selected-positive IoU loss + one-sided native TAL target overconfidence divergence); <=8 hit-change GT and8 newly appearing R errors/image. Train-only frozen teacher.',
         'comparability_limit': 'Ordinary QAT control uses the SAME G1 FP32 mask. This is not a fully trained all-INT8, HQOD, DetPTQ or real-backend comparison. Ten epochs is a fixed larger development budget, not convergence certification.',
         'scorer_transfer': 'Previously frozen generic score_base applied to ordinary_qat outputs, without fitting; post-NMS scores only; fixed confidence0.25 for every branch; transfer is exploratory and trained on overlapping training members',
         'preprocessing_training': 'Unchanged RestrictedIgnoreDataset resize/letterbox640 RGB uint8 then CUDA float/255; original integer annotation geometry mapped with atol0.002px/rtol1e-5',
         'preprocessing_evaluation': 'Unchanged historical LetterBox640; CPU float/255 then CUDA; publisher original raster ignores',
         'postprocessing': 'CPU NMS conf>0.001 IoU0.7 single label max_det1000 max_nms30000; fixed source tie/order policy; AP global cap500 before category',
         'numeric_tolerances': {'boxes': {'atol': .5, 'rtol': .005}, 'scores': {'atol': .002, 'rtol': .02},
                                'all_anchor_scope': 'Only existing128 FP32 raw references; all8400 anchors/all14 channels, no confidence/NMS subset. Missing286 references not certified. Learned weights do not imply numerical equivalence.'},
         'primary_development_rule': 'native correction AP_all > ordinary_qat AP_all AND hits_conf025 >= ordinary AND FP_conf025 <= ordinary. No threshold/epoch/seed optimization.',
         'candidate_budget': {'new_detector_candidates': 2, 'training_seeds': 1, 'new_scorer_fits': 0, 'frozen_score_transfer_branches': 1},
         'new_teacher_forwards': 2 * (TRAIN_COUNT - 512), 'training_forward_images': 2 * EPOCHS * TRAIN_COUNT,
         'evaluation_forward_images': 2 * VAL_COUNT, 'real_image_forward_budget': 2 * (TRAIN_COUNT - 512) + 2 * EPOCHS * TRAIN_COUNT + 2 * VAL_COUNT,
         'reconstruction_real_image_forwards_separate': 6, 'AP_jobs': 3,
         'timeout_seconds': 21600, 'minimum_free_disk_bytes': 8 * 1024**3, 'automatic_retry': False,
         'failure_policy': 'Preserve failed attempt, no budget/batch/tolerance changes or automatic restart',
         'Selection_payload_access': False, 'Holdout_payload_access': False, 'scientific_acceptance': False, 'actual_INT8': False}
    OUT.mkdir(parents=True)
    write(OUT / 'contract.json', c)
    print(json.dumps({'status': 'FROZEN_FULL_TRAINING_COMPARISON', 'contract': identity(OUT / 'contract.json'),
                      'training_images_per_arm': TRAIN_COUNT * EPOCHS, 'optimizer_steps_per_arm': c['optimizer_steps_per_arm'],
                      'new_real_image_forwards': c['real_image_forward_budget']}, ensure_ascii=True), flush=True)


class Worker(P.Worker):
    def __init__(self):
        self.contract = read(OUT / 'contract.json')
        self.attempt = OUT / 'attempt_01'; self.attempt.mkdir(exist_ok=False)
        self.started = time.monotonic()
        self.live = {'schema': SCHEMA, 'pid': os.getpid(), 'parent_pid': os.getppid(), 'started_at_utc': now(),
                     'contract': identity(OUT / 'contract.json'), 'forward_images': 0, 'reconstruction_forward_images': 0}
        self.progress('STARTING', 0, 1)

    def progress(self, stage, done, total, **extra):
        require(shutil.disk_usage(OUT).free >= self.contract['minimum_free_disk_bytes'], 'Disk below frozen8GiB minimum')
        super().progress(stage, done, total, elapsed_seconds=round(time.monotonic() - self.started, 3), **extra)

    def verify(self):
        c = self.contract
        require(c['schema'] == SCHEMA and c['source_bindings'] == source_bindings(), 'Frozen comparison source changed')
        require(c['framework'] == P.verify_framework_sources(), 'Framework source changed')
        for key in ('candidate', 'architecture', 'formal_contract', 'train_manifest', 'training_val_manifest', 'evaluation_protocol',
                    'training_annotation_assignment', 'prior_contract', 'native_pilot_contract', 'ranges', 'scorer_model', 'previous_state'):
            require(identity(c[key]['path']) == c[key], 'Input identity changed:' + key)
        for row in c['reused_teachers']:
            for key in ('record', 'raw'):
                require(identity(row[key]['path']) == row[key], 'Completed teacher changed')
        for row in c['FP32_saved_reference_subset']:
            require(identity(row['raw']['path']) == row['raw'], 'Saved FP32 reference changed')

    def supervision(self, dataset, maxima):
        c = self.contract; old = {r['image_id']: r for r in c['reused_teachers']}
        original_root = Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
        folder = self.attempt / 'training_supervision'; folder.mkdir()
        model = P.new_model(maxima); before = value_sha(model.state_dict())
        records = []
        self.progress('EXTEND_TRAIN_ONLY_TEACHERS', 0, TRAIN_COUNT)
        for i, meta in enumerate(c['training_annotation_metadata']):
            if meta['image_id'] in old:
                binding = old[meta['image_id']]
                original = read(binding['record']['path'])
                require(original['image_id'] == meta['image_id'], 'Reused teacher order/ID mismatch')
                record = {k: original[k] for k in ('image_id', 'q_counts', 'r_counts', 'rescued_targets', 'collateral_targets', ARMS[1])}
                record['reused_record'] = binding['record']; record['raw_source'] = binding['raw']
            else:
                sample = dataset[i]
                annotation = (original_root / meta['annotation_path']).resolve(strict=True)
                require(annotation.is_relative_to(original_root) and sha(annotation) == meta['annotation_sha256'], 'Training annotation identity differs')
                gt = P.MATCH.parse_ground_truth(annotation.read_text(encoding='utf-8-sig'), meta['image_id'])
                params = P.original_training_geometry(sample, gt, meta['width'], meta['height'])
                tensor = P.gpu_batch([sample])['img']
                P.precision(model, 'q_all'); q = self.forward(model, tensor)
                P.precision(model, 'r_only:G1'); r = self.forward(model, tensor)
                record = mine_spec(q, r, gt, meta, params)
                path = folder / f'{i:04d}_raw.npz'; np.savez_compressed(path, q=q, r=r)
                record['raw_source'] = identity(path)
                del sample, tensor, q, r
            write(folder / f'{i:04d}.json', record); records.append(record)
            if (i + 1) % 16 == 0 or i + 1 == TRAIN_COUNT:
                self.progress('EXTEND_TRAIN_ONLY_TEACHERS', i + 1, TRAIN_COUNT)
        require(before == value_sha(model.state_dict()), 'Frozen teacher state mutated')
        require(self.live['forward_images'] == c['new_teacher_forwards'], 'Teacher forward count differs')
        write(folder / 'summary.json', {'images': TRAIN_COUNT, 'reused': 512, 'new': TRAIN_COUNT - 512,
              'positive_counts': sum(len(r[ARMS[1]]['positive_gt']) for r in records),
              'error_counts': sum(len(r[ARMS[1]]['errors']) for r in records),
              'error_types': dict(Counter(e['kind'] for r in records for e in r[ARMS[1]]['errors'])),
              'teacher_state_unchanged': True, 'train_members_only': True, 'new_calibration': False})
        del model; torch.cuda.empty_cache()
        return records

    def train(self, model, dataset, records, arm, maxima):
        c = self.contract; P.seed_all(c['training_seed'])
        model.args.epochs = EPOCHS
        folder = self.attempt / 'training' / arm; folder.mkdir(parents=True)
        initial = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        initial_hash = value_sha(initial)
        model.train(); P.freeze_bn(model)
        criterion = N.NativeCorrectiveLoss(model)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=1e-4, momentum=.9, weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        generator = np.random.Generator(np.random.PCG64(c['training_seed']))
        orders = []; batches = 0; steps = 0; epoch_summaries = []
        started = time.monotonic()
        self.progress('TRAINING_' + arm, 0, EPOCHS * c['batches_per_epoch'], epoch=1, optimizer_steps=0)
        with (folder / 'batch_trace.jsonl').open('x', encoding='utf-8') as trace:
            for epoch in range(EPOCHS):
                epoch_started = time.monotonic(); order = generator.permutation(len(dataset)).tolist(); orders.append(order)
                epoch_losses = []
                for b, position in enumerate(range(0, len(order), BATCH)):
                    ids = order[position:position + BATCH]
                    batch = P.gpu_batch([dataset[j] for j in ids])
                    self.live['forward_images'] += len(ids)
                    require(self.live['forward_images'] <= c['real_image_forward_budget'], 'Training forward budget exceeded')
                    pred = model(batch['img'])
                    loss = criterion.compute(pred, batch, [records[j] for j in ids], arm)
                    require(bool(torch.isfinite(loss)), 'Nonfinite loss; no recovery')
                    denominator, update = accumulation_window(b, c['batches_per_epoch'])
                    (loss / denominator).backward(); batches += 1
                    scalar = float(loss.detach()); epoch_losses.append(scalar)
                    norm = None
                    if update:
                        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True))
                        optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
                    trace.write(json.dumps({'epoch': epoch + 1, 'epoch_batch': b + 1, 'batch': batches, 'indices': ids,
                        'loss': scalar, 'optimizer_step': steps, 'update': update, 'accumulation_denominator': denominator,
                        'grad_norm_before_clip': norm, **criterion.last_aux}, allow_nan=False) + '\n')
                    if batches % 32 == 0 or b + 1 == c['batches_per_epoch']:
                        trace.flush()
                        self.progress('TRAINING_' + arm, batches, EPOCHS * c['batches_per_epoch'],
                                      epoch=epoch + 1, optimizer_steps=steps, latest_loss=scalar,
                                      training_elapsed_seconds=round(time.monotonic() - started, 3))
                    del batch, pred, loss
                state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
                require(all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point()), 'Nonfinite checkpoint')
                torch.save({'state_dict': state, 'epoch': epoch + 1, 'arm': arm,
                            'contract_sha256': self.live['contract']['sha256']}, folder / f'epoch_{epoch+1:02d}_fp32.pt')
                epoch_summaries.append({'epoch': epoch + 1, 'mean_batch_loss': float(np.mean(epoch_losses)),
                                        'elapsed_seconds': time.monotonic() - epoch_started, 'optimizer_steps': steps})
                del state
        require(steps == c['optimizer_steps_per_arm'], 'Optimizer budget incomplete')
        final = model.state_dict()
        frozen = [n for n in initial if 'running_' in n or 'num_batches_tracked' in n or n.endswith('_scale') or '.dfl.' in n]
        require(all(torch.equal(initial[n], final[n].cpu()) for n in frozen), 'BN/DFL/quantizer frozen state changed')
        changed = sum(int((final[n].cpu() != v).sum()) for n, v in initial.items() if n.endswith('weight'))
        require(changed > 0, 'No learned weight updates')
        model.eval(); x = P.gpu_batch([dataset[0]])['img']
        with torch.inference_mode(): reference = model(x)[0].detach().cpu()
        saved = torch.load(folder / 'epoch_10_fp32.pt', map_location='cpu', weights_only=True)
        rebuilt = P.new_model(maxima); rebuilt.args.epochs = EPOCHS
        rebuilt.load_state_dict(saved['state_dict'], strict=True)
        with torch.inference_mode(): replay = rebuilt(x)[0].detach().cpu()
        require(torch.equal(reference, replay), 'Final saved CUDA rebuild differs')
        rebuilt.cpu()
        with torch.inference_mode(): cpu = rebuilt(x.cpu())[0].detach().cpu()
        require(bool(torch.isfinite(cpu).all()), 'Final CPU rebuild nonfinite')
        self.live['reconstruction_forward_images'] += 3
        write(folder / 'epoch_orders.json', orders)
        write(folder / 'training_receipt.json', {'arm': arm, 'epochs': EPOCHS, 'batches': batches, 'optimizer_steps': steps,
              'initial_state_value_sha256': initial_hash, 'final_state_value_sha256': value_sha(final),
              'changed_weight_elements': changed, 'frozen_BN_DFL_scales_unchanged': True, 'epoch_summaries': epoch_summaries,
              'epoch_orders': identity(folder / 'epoch_orders.json'), 'final_candidate': identity(folder / 'epoch_10_fp32.pt'),
              'CUDA_reconstruction_exact': True, 'CPU_CUDA_output_max_abs': float((reference - cpu).abs().max()),
              'CPU_CUDA_equivalence_certified': False, 'reconstruction_real_image_forwards_separate': 3,
              'training_elapsed_seconds': time.monotonic() - started})
        del rebuilt, optimizer, criterion, initial, saved, x, parameters; torch.cuda.empty_cache()
        return model

    def evaluate(self, model, arm):
        c = self.contract; root = Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
        references = {r['image_id']: r['raw'] for r in c['FP32_saved_reference_subset']}
        branches = [arm] + (['ordinary_qat_frozen_scorer'] if arm == ARMS[0] else [])
        folders = {}; rows = {b: [] for b in branches}; matrices = {b: {} for b in branches}
        totals = {b: Counter() for b in branches}; prefixes = {b: {} for b in branches}
        for b in branches:
            folders[b] = self.attempt / 'evaluation' / b; folders[b].mkdir(parents=True)
        rawfolder = folders[arm] / 'raw'; rawfolder.mkdir()
        scorer = read(c['scorer_model']['path']); guards = []; state_before = value_sha(model.state_dict())
        timings = Counter(); model.eval(); jobs = []
        self.progress('EVALUATION_' + arm, 0, VAL_COUNT)
        for i, meta in enumerate(c['evaluation_members']):
            tensor, params = self.image_input(root, meta)
            start = time.monotonic(); raw = self.forward(model, tensor); timings['detector_seconds'] += time.monotonic() - start
            np.savez_compressed(rawfolder / f'{i:04d}.npz', raw=raw)
            pred = P.predictions(raw, params, meta['image_id'], arm)
            predictions = {arm: pred}
            if arm == ARMS[0]:
                start = time.monotonic(); features = FEATURES.visible_features(raw, pred)
                new_scores = S.score_rows(features, np.array([d['score'] for d in pred]), scorer)
                changed = deepcopy(pred)
                for d, score in zip(changed, new_scores): d['score'] = float(score)
                changed.sort(key=lambda d: (-d['score'], d['nms_input_order']))
                predictions[branches[1]] = changed
                timings['scorer_seconds'] += time.monotonic() - start
                np.savez_compressed(folders[branches[1]] / f'features_{i:04d}.npz', features=features, new_scores=new_scores)
            annotation = (root / meta['annotation_path']).resolve(strict=True)
            require(annotation.is_relative_to(root) and sha(annotation) == meta['annotation_sha256'], 'Development annotation changed')
            gt = P.MATCH.parse_ground_truth(annotation.read_text(encoding='utf-8-sig'), meta['image_id'])
            gt8 = np.array([g['bbox_xywh'] + [g['score'], g['category'], g['truncation'], g['occlusion']] for g in gt], np.float64).reshape(-1, 8)
            for b in branches:
                detections = predictions[b]
                dt8 = np.array([d['bbox_xywh'] + [d['score'], d['category'], -1, -1] for d in detections], np.float64).reshape(-1, 8)
                prepared = P.AP.prepare_raw_image(gt8, dt8, width=meta['width'], height=meta['height'])
                matrices[b][f'gt_{i}'] = prepared['prepared_gt']; matrices[b][f'dt_{i}'] = prepared['filtered_predictions']
                match = P.MATCH.match_project_targets(gt, detections, meta['width'], meta['height'], confidence=.25, iou_threshold=.5)
                totals[b].update(match['counts'])
                prefix = meta['prefix_proxy']; prefixes[b].setdefault(prefix, Counter()).update(match['counts'])
                rows[b].append({'image_id': meta['image_id'], 'predictions': detections, 'target_matching': match,
                                'letterbox': params, 'GT_source_sha256': meta['annotation_sha256'], 'frozen_confidence': .25,
                                'max_det_saturated': len(detections) == 1000})
            if meta['image_id'] in references:
                with np.load(references[meta['image_id']]['path'], allow_pickle=False) as d: ref = d['raw'].copy()
                guards.append({'image_id': meta['image_id'], 'guard': P.EVAL.raw_guard(ref, raw)})
            if (i + 1) % 16 == 0 or i + 1 == VAL_COUNT:
                self.progress('EVALUATION_' + arm, i + 1, VAL_COUNT)
        require(state_before == value_sha(model.state_dict()), 'Trained detector changed during evaluation')
        write(folders[arm] / 'all_anchor_saved128_guard.json', {'images': 128, 'anchors_per_image': 8400, 'checks': guards,
              'full414_FP32_numeric_equivalence_certified': False, 'missing_reference_images': 286,
              'scope': 'Fixed historical full-channel/all-anchor tolerance; not an AP or learned-method acceptance rule'})
        write(folders[arm] / 'timing.json', {**dict(timings), 'images': VAL_COUNT, 'end_to_end_device_speed_certified': False})
        for b in branches:
            output = folders[b]
            np.savez_compressed(output / 'prepared_matrices.npz', **matrices[b])
            write(output / 'saved_predictions_and_targets.json', rows[b])
            write(output / 'project_metrics.json', {'branch': b, 'images': VAL_COUNT, 'conf025_counts': dict(totals[b]),
                  'prefix_counts': {k: dict(v) for k, v in prefixes[b].items()}, 'confidence': .25, 'development_only': True})
            job = output / 'AP_job.json'
            write(job, {'cohort': 'full414_reused_development', 'branch': b, 'images': VAL_COUNT,
                        'matrices': identity(output / 'prepared_matrices.npz'), 'output': str(output.resolve()),
                        'contract_sha256': self.live['contract']['sha256']})
            jobs.append(job)
        return jobs

    def score_jobs(self, jobs):
        # Sequential CPU aggregation; never run a second model-training worker.
        for i, job in enumerate(jobs):
            self.progress('OFFICIAL_SOURCE_PYTHON_AP', i, len(jobs), AP_job=str(job))
            output = job.parent
            argv = [sys.executable, '-B', '-X', 'utf8', '-u', str(ROOT / 'scripts/gate1_cached_ap_job.py'), str(job)]
            with (output / 'AP_console.txt').open('x', encoding='utf-8') as console:
                child = subprocess.Popen(argv, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                write(output / 'AP_launcher.json', {'pid': child.pid, 'parent_pid': os.getpid(), 'argv': argv, 'started_at_utc': now()})
                while child.poll() is None:
                    if time.monotonic() - self.started >= self.contract['timeout_seconds']:
                        child.kill(); child.wait(); raise TimeoutError('Frozen complete comparison time budget exceeded')
                    time.sleep(1)
                code = child.returncode
            write(output / 'AP_terminal.json', {'pid': child.pid, 'exit_code': code, 'observed_at_utc': now()})
            require(code == 0 and (output / 'metrics.json').is_file(), 'AP aggregation failed; preserve attempt')

    def run(self):
        P.configure(); self.verify(); require(torch.cuda.is_available(), 'CUDA unavailable')
        write(self.attempt / 'runtime.json', {'python': sys.version, 'executable': sys.executable, 'torch': str(torch.__version__),
              'cuda': torch.version.cuda, 'GPU': torch.cuda.get_device_name(0), 'backend': P.FT.backend_settings()})
        c = self.contract; maxima = read(c['ranges']['path'])['max_abs']
        dataset = P.SelectedTrainingDataset(c['train_manifest'], c['training_members'])
        records = self.supervision(dataset, maxima); jobs = []
        for arm in ARMS:
            model = P.new_model(maxima); P.precision(model, 'r_only:G1')
            model = self.train(model, dataset, records, arm, maxima)
            jobs.extend(self.evaluate(model, arm)); del model; torch.cuda.empty_cache()
        receipts = [read(self.attempt / 'training' / arm / 'training_receipt.json') for arm in ARMS]
        orders = [read(self.attempt / 'training' / arm / 'epoch_orders.json') for arm in ARMS]
        require(len({r['initial_state_value_sha256'] for r in receipts}) == 1 and orders[0] == orders[1], 'Matched initialization/order check failed')
        require(self.live['forward_images'] == c['real_image_forward_budget'] and self.live['reconstruction_forward_images'] == 6, 'Forward budget not exactly complete')
        self.score_jobs(jobs)
        metrics = {b: {'official_source_python': read(self.attempt / 'evaluation' / b / 'metrics.json'),
                       'project': read(self.attempt / 'evaluation' / b / 'project_metrics.json')} for b in BRANCHES}
        base = metrics[ARMS[0]]; ours = metrics[ARMS[1]]
        ap_delta = ours['official_source_python']['metrics_percent']['AP_all'] - base['official_source_python']['metrics_percent']['AP_all']
        counts = [m['project']['conf025_counts'] for m in (base, ours)]
        tp_delta = counts[1]['hits'] - counts[0]['hits']; fp_delta = counts[1]['false_positive_detections'] - counts[0]['false_positive_detections']
        result = {'status': 'COMPLETE_FIXED_LARGER_DEVELOPMENT_COMPARISON', 'metrics': metrics,
                  'correction_vs_ordinary': {'AP_delta_percentage_points': ap_delta, 'TP_delta_conf025': tp_delta,
                                             'FP_delta_conf025': fp_delta, 'joint_rule_met': ap_delta > 0 and tp_delta >= 0 and fp_delta <= 0},
                  'same_initialization_order_precision_training_budget': True, 'fully_converged_baselines_certified': False,
                  'new_scorer_fit_count': 0, 'single_seed': True, 'Selection_payload_access': False,
                  'Holdout_payload_access': False, 'scientific_acceptance': False, 'actual_INT8': False,
                  'native_current_submission': False, 'historical_failures_preserved': True}
        write(self.attempt / 'comparison_result.json', result)
        self.verify()
        manifest = [identity(p) for p in sorted(self.attempt.rglob('*')) if p.is_file() and p.name not in
                    {'live_status.json', 'events.jsonl', 'manifest.json', 'completion_receipt.json'}]
        write(self.attempt / 'manifest.json', {'files': manifest})
        self.progress('COMPLETE_FIXED_LARGER_DEVELOPMENT_COMPARISON', 1, 1)
        write(self.attempt / 'completion_receipt.json', {**self.live, 'finished_at_utc': now(),
              'elapsed_seconds': time.monotonic() - self.started, 'comparison': identity(self.attempt / 'comparison_result.json'),
              'manifest': identity(self.attempt / 'manifest.json')})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('prepare', 'run')); args = parser.parse_args()
    if args.mode == 'prepare': prepare(); return
    worker = Worker(); awake = False
    try:
        if os.name == 'nt':
            import ctypes
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        worker.run()
    except BaseException as error:
        worker.live.update(status='FAILED_ATTEMPT_PRESERVED', error_type=type(error).__name__, error=str(error), updated_at_utc=now())
        write(worker.attempt / 'live_status.json', worker.live, once=False)
        write(worker.attempt / 'failure_receipt.json', {**worker.live, 'traceback': traceback.format_exc()})
        raise
    finally:
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__': main()
