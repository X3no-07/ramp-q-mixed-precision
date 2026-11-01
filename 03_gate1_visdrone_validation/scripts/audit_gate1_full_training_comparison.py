"""Post-exit saved-array audit and delivery. No image/model forward, no refit."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import gate1_full_training_comparison as W
P, S = W.P, W.S
ROOT, OUT = W.ROOT, W.OUT
read, write, identity, require, now = W.read, W.write, W.identity, W.require, W.now


def metric_aggregation(output):
    with np.load(output / 'AP_AR_arrays.npz', allow_pickle=False) as d:
        ap, ar, selected = d['AP'].copy(), d['AR'].copy(), d['evalClass'].copy() - 1
    require(ap.shape == (10, 10) and ar.shape == (10, 10, 4) and len(selected), 'Saved AP/AR domain differs')
    values = [float(ap[selected].mean()), float(ap[selected, 0].mean()), float(ap[selected, 5].mean())]
    values += [float(ar[selected, :, k].mean()) for k in range(4)]
    actual = read(output / 'metrics.json')['metrics_percent']
    require(all(values[i] == actual[name] for i, name in enumerate(P.AP.METRIC_NAMES)), 'Seven aggregate metrics differ from saved AP/AR')


def audit():
    P.configure(); started = time.monotonic(); c = read(OUT / 'contract.json'); attempt = OUT / 'attempt_01'
    require(c['source_bindings'] == W.source_bindings(), 'Frozen source identity differs')
    require(c['framework'] == P.verify_framework_sources(), 'Framework differs')
    terminal = read(OUT / 'worker_terminal_receipt.json')
    require(terminal['exit_code'] == 0 and terminal['exit_code_directly_observed'], 'Worker exit0 not observed')
    completion = read(attempt / 'completion_receipt.json')
    require(completion['status'] == 'COMPLETE_FIXED_LARGER_DEVELOPMENT_COMPARISON' and completion['forward_images'] == c['real_image_forward_budget'] and completion['reconstruction_forward_images'] == 6, 'Incomplete budgets/receipt')
    manifest = read(attempt / 'manifest.json')['files']
    for binding in manifest:
        require(identity(binding['path']) == binding, 'Output hash differs before audit')
    for key in ('candidate', 'architecture', 'formal_contract', 'train_manifest', 'training_val_manifest', 'evaluation_protocol',
                'training_annotation_assignment', 'prior_contract', 'native_pilot_contract', 'ranges', 'scorer_model', 'previous_state'):
        require(identity(c[key]['path']) == c[key], 'Source/input binding differs:' + key)
    for r in c['reused_teachers']:
        require(identity(r['record']['path']) == r['record'] and identity(r['raw']['path']) == r['raw'], 'Reused teacher source differs')
    for binding in c['FP32_saved_reference_subset']:
        require(identity(binding['raw']['path']) == binding['raw'], 'Old FP32 raw source differs')
    training_manifest = read(c['train_manifest']['path']); val_manifest = read(c['training_val_manifest']['path'])
    P.FT.validate_membership(training_manifest, val_manifest)
    source_member_checks = 0
    for manifest_binding, header in ((c['train_manifest'], training_manifest), (c['training_val_manifest'], val_manifest)):
        root = (Path(manifest_binding['path']).parent / header['data_root']).resolve(strict=True)
        for row in header['samples']:
            for key in ('image', 'labels', 'sidecar'):
                member = row[key]; path = (root / member['path']).resolve(strict=True)
                require(path.is_relative_to(root), 'Member escaped role root')
                require(path.stat().st_size == member['bytes'] and W.sha(path) == member['sha256'], 'Source data member changed')
                source_member_checks += 1
    state_checks = []; training_orders = []; initial_hashes = []
    for arm in W.ARMS:
        folder = attempt / 'training' / arm; receipt = read(folder / 'training_receipt.json')
        require(receipt['epochs'] == 10 and receipt['batches'] == 11330 and receipt['optimizer_steps'] == 2840 and receipt['CUDA_reconstruction_exact'], 'Training budget/rebuild differs')
        orders = read(folder / 'epoch_orders.json'); training_orders.append(orders); initial_hashes.append(receipt['initial_state_value_sha256'])
        require(len(orders) == 10 and all(sorted(o) == list(range(4530)) for o in orders), 'Epoch membership differs')
        trace = [json.loads(line) for line in (folder / 'batch_trace.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(trace) == 11330 and sum(r['update'] for r in trace) == 2840, 'Trace update/batch counts differ')
        for epoch in range(10):
            epoch_trace = trace[epoch * 1133:(epoch + 1) * 1133]
            require([j for row in epoch_trace for j in row['indices']] == orders[epoch], 'Actual sample order differs')
            for b, row in enumerate(epoch_trace):
                width, update = W.accumulation_window(b, 1133)
                require(row['accumulation_denominator'] == width and row['update'] == update, 'Partial accumulation differs')
        first_frozen = None
        for epoch in range(1, 11):
            path = folder / f'epoch_{epoch:02d}_fp32.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True); state = saved['state_dict']
            require(saved['arm'] == arm and saved['epoch'] == epoch and saved['contract_sha256'] == completion['contract']['sha256'], 'Checkpoint identity differs')
            require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point()), 'Checkpoint is not finite FP32')
            frozen = {n: v for n, v in state.items() if 'running_' in n or 'num_batches_tracked' in n or n.endswith('_scale') or '.dfl.' in n}
            if first_frozen is None: first_frozen = {n: v.clone() for n, v in frozen.items()}
            require(all(torch.equal(first_frozen[n], v) for n, v in frozen.items()), 'Saved frozen buffers changed across epochs')
            state_checks.append({'arm': arm, 'epoch': epoch, 'identity': identity(path), 'finite_FP32': True})
            if epoch == 10: require(W.value_sha(state) == receipt['final_state_value_sha256'], 'Final saved state value differs')
        del saved, state, trace
    require(len(set(initial_hashes)) == 1 and training_orders[0] == training_orders[1], 'Arm initialization/orders differ')
    root = Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
    references = {r['image_id']: r['raw'] for r in c['FP32_saved_reference_subset']}
    scorer = read(c['scorer_model']['path']); raw_count = 0; matrix_count = 0; guard_count = 0; prediction_count = 0
    for arm in W.ARMS:
        branches = [arm] + (['ordinary_qat_frozen_scorer'] if arm == W.ARMS[0] else [])
        rows = {b: read(attempt / 'evaluation' / b / 'saved_predictions_and_targets.json') for b in branches}
        packed = {}; totals = {b: Counter() for b in branches}; guard = []
        for b in branches:
            with np.load(attempt / 'evaluation' / b / 'prepared_matrices.npz', allow_pickle=False) as d:
                packed[b] = {k: d[k].copy() for k in d.files}
        for i, meta in enumerate(c['evaluation_members']):
            with np.load(attempt / 'evaluation' / arm / 'raw' / f'{i:04d}.npz', allow_pickle=False) as d: raw = d['raw'].copy()
            require(raw.shape == (1, 14, 8400) and raw.dtype == np.float32 and np.isfinite(raw).all(), 'Saved full raw invalid')
            ordinary = rows[arm][i]; require(ordinary['image_id'] == meta['image_id'], 'Submission order differs')
            pred = P.predictions(raw, ordinary['letterbox'], meta['image_id'], arm)
            require(pred == ordinary['predictions'], 'Independent saved raw/NMS replay differs')
            predictions = {arm: pred}; raw_count += 1
            if arm == W.ARMS[0]:
                features = W.FEATURES.visible_features(raw, pred)
                scored = S.score_rows(features, np.array([d['score'] for d in pred]), scorer)
                with np.load(attempt / 'evaluation' / branches[1] / f'features_{i:04d}.npz', allow_pickle=False) as d:
                    require(np.array_equal(features, d['features']) and np.array_equal(scored, d['new_scores']), 'Frozen score transfer replay differs')
                updated = deepcopy(pred)
                for d, score in zip(updated, scored): d['score'] = float(score)
                updated.sort(key=lambda d: (-d['score'], d['nms_input_order']))
                predictions[branches[1]] = updated
            annotation = (root / meta['annotation_path']).resolve(strict=True)
            require(annotation.is_relative_to(root) and W.sha(annotation) == meta['annotation_sha256'], 'Original annotation differs')
            gt = P.MATCH.parse_ground_truth(annotation.read_text(encoding='utf-8-sig'), meta['image_id'])
            gt8 = np.array([g['bbox_xywh'] + [g['score'], g['category'], g['truncation'], g['occlusion']] for g in gt], np.float64).reshape(-1, 8)
            for b in branches:
                require(rows[b][i]['image_id'] == meta['image_id'] and predictions[b] == rows[b][i]['predictions'], 'Saved submission differs')
                match = P.MATCH.match_project_targets(gt, predictions[b], meta['width'], meta['height'], confidence=.25, iou_threshold=.5)
                require(match == rows[b][i]['target_matching'], 'Independent saved target matching differs')
                totals[b].update(match['counts']); prediction_count += 1
                dt8 = np.array([d['bbox_xywh'] + [d['score'], d['category'], -1, -1] for d in predictions[b]], np.float64).reshape(-1, 8)
                prepared = P.AP.prepare_raw_image(gt8, dt8, width=meta['width'], height=meta['height'])
                require(np.array_equal(prepared['prepared_gt'], packed[b][f'gt_{i}']) and
                        np.array_equal(prepared['filtered_predictions'], packed[b][f'dt_{i}']), 'Zero-tolerance saved AP matrix replay differs')
                matrix_count += 2
            if meta['image_id'] in references:
                with np.load(references[meta['image_id']]['path'], allow_pickle=False) as d: reference = d['raw'].copy()
                guard.append({'image_id': meta['image_id'], 'guard': P.EVAL.raw_guard(reference, raw)}); guard_count += 1
        require(guard == read(attempt / 'evaluation' / arm / 'all_anchor_saved128_guard.json')['checks'], 'Historical all-anchor guard replay differs')
        for b in branches:
            output = attempt / 'evaluation' / b
            require(dict(totals[b]) == read(output / 'project_metrics.json')['conf025_counts'], 'Project count aggregation differs')
            metric_aggregation(output)
            require(read(output / 'AP_terminal.json')['exit_code'] == 0, 'AP job exit0 absent')
    require((raw_count, prediction_count, matrix_count, guard_count) == (828, 1242, 2484, 256), 'Independent audit counts incomplete')
    for binding in manifest: require(identity(binding['path']) == binding, 'Output hash differs after audit')
    result = {'status': 'PASS_SAVED_RESULTS_INTEGRITY_AND_REPLAY_ONLY', 'at_utc': now(), 'pid': os.getpid(),
              'parent_pid': os.getppid(), 'output_members_hashed_before_and_after': len(manifest),
              'source_member_hashes_checked': source_member_checks, 'FP32_checkpoints': state_checks,
              'full_raw_NMS_replays': raw_count, 'prediction_target_replays': prediction_count,
              'zero_tolerance_AP_matrix_replays': matrix_count, 'all_anchor_saved128_guard_replays': guard_count,
              'new_model_forwards': 0, 'new_fits': 0, 'AP_recomputed': False,
              'seven_metric_aggregation_from_saved_arrays_exact': True, 'elapsed_seconds': time.monotonic() - started,
              'audit_PASS_does_not_accept_method': True, 'scientific_acceptance': False, 'actual_INT8': False}
    write(OUT / 'independent_saved_audit.json', result)
    comparison = read(attempt / 'comparison_result.json'); delta = comparison['correction_vs_ordinary']
    decision = {'status': 'COMPLETE_LARGER_DEVELOPMENT_COMPARISON_AUDITED', 'at_utc': now(),
                'joint_development_rule_met': delta['joint_rule_met'], 'correction_vs_ordinary': delta,
                'training_complete': True, 'saved_results_audit_PASS': True, 'independent_scientific_acceptance': False,
                'fully_converged_strong_baseline_certified': False, 'actual_INT8': False, 'Holdout_payload_access': False,
                'next_action': 'Review the fixed comparison; no automatic extra candidates, training epochs, seeds, new data, or backend claims.'}
    write(OUT / 'machine_decision.json', decision)
    lines = ['# Gate1 Results from Two Groups of 10 Training Rounds on the Full Dataset', '', '2026-10-27。', '',
             'This round involves the existing 4530 training images, two arms each with 10 epochs, and a single seed experiment with identical initialization, sequence, precision, and optimization budget. The candidate fixed epoch is set at 10, without selecting epochs or thresholds based on development AP.', '',
             '| Branch | AP50–95(%) | Fixed detection threshold of 0.25 | Fixed false detection threshold of 0.25 |', '|---|---:|---:|---:|']
    for b in W.BRANCHES:
        row = comparison['metrics'][b]; counts = row['project']['conf025_counts']
        lines.append(f"| {b} | {row['official_source_python']['metrics_percent']['AP_all']:.9f} | {counts['hits']} | {counts['false_positive_detections']} |")
    lines += ['', f"Target alignment correction relative to standard QAT under the same precision: AP difference{delta['AP_delta_percentage_points']:+.9f}in percentage points, detection difference{delta['TP_delta_conf025']:+d}, false positive difference{delta['FP_delta_conf025']:+d}. The pre-fixed condition of improved AP, unchanged detection, and no increase in false positives:{'satisfied' if delta['joint_rule_met'] else 'not satisfied'}。", '',
              'The third branch directly applies the old conventional scorer to the new conventional QAT output without retraining. Training members overlap and cross-detection migration exists, serving as a descriptive control only—this cannot be used to claim the contribution of the new scorer.', '',
              'Independent saved-artifact checks passed:20 per-epoch FP32 records,828 all-candidate raw/NMS records,1242 prediction/target-match records, and 2484 AP input matrices were checked with zero tolerance. All output-member hashes are unchanged before and after audit. The three AP jobs terminated normally, and the seven aggregate metrics exactly match the saved AP/AR arrays. The audit repeated neither model inference nor complete AP computation.', '',
              'The original numerical gate is rechecked over all 8400 candidates only for the 128 development images with existing FP32 arrays, without substituting confidence-filtered or NMS subsets. Old FP32 arrays for the other 286 images were not retained, and no new FP32 run fills that gap here; complete numerical equivalence over 414 images is therefore unqualified. The trained models differ from original FP32 outputs; saved-artifact consistency PASS cannot substitute for numerical qualification.', '',
              'The current ordinary QAT and correction group both use G1-bit floating-point, with other Conv layers using W8A8-bit floating-point QDQ. Expanding the budget to ten epochs strengthened the pilot evidence, but did not demonstrate training convergence, nor did it reproduce HQOD/DetPTQ, validate true INT8, Orin, or independent scientific validation. 414 images are repeatedly used from the training_val development set; Selection/Holdout has not been read, and all historical failures are retained.', '',
              'This phase stops automatic candidate augmentation. The next step will decide whether to expand comparisons with conventional and published methods based on this fixed result; independent data and actual backend systems require separate validation evidence.']
    report = ROOT / 'docs/85_gate1_full_training_set_two_arm_ten_epoch_results.md'
    with report.open('x', encoding='utf-8') as f: f.write('\n'.join(lines) + '\n')
    members = [identity(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in
               {'delivery_manifest.json', 'delivery_validation.json', 'launcher_terminal_receipt.json', 'audit_console.txt', 'supervisor_console.txt'}]
    members += [identity(report)]
    write(OUT / 'delivery_manifest.json', {'files': members})
    require(all(identity(b['path']) == b for b in members), 'Delivery manifest validation failed')
    write(OUT / 'delivery_validation.json', {'status': 'PASS_COMPLETE_FIXED_COMPARISON_DELIVERY', 'at_utc': now(),
          'members': len(members), 'manifest': identity(OUT / 'delivery_manifest.json'), 'report': identity(report),
          'audit': identity(OUT / 'independent_saved_audit.json'), 'decision': identity(OUT / 'machine_decision.json')})
    previous = read(ROOT / 'artifacts/gate1/latest_execution_state.json')
    number = 49
    while (ROOT / f'artifacts/gate1/current_execution_state_v{number}.json').exists(): number += 1
    snapshot = ROOT / f'artifacts/gate1/current_execution_state_v{number}.json'
    write(snapshot, {'schema': f'rampq.gate1.execution_state/v{number}', 'at_utc': now(), **decision, 'previous_pointer': previous,
                    'comparison_contract': identity(OUT / 'contract.json'), 'report': identity(report),
                    'delivery_validation': identity(OUT / 'delivery_validation.json'), 'no_active_training_worker': True})
    write(ROOT / 'artifacts/gate1/latest_execution_state.json', {'schema': 'rampq.gate1.mutable_latest_state_pointer/v1',
          'snapshot': str(snapshot.relative_to(ROOT)).replace('\\', '/'), 'snapshot_sha256': W.sha(snapshot),
          'delivery_validation': identity(OUT / 'delivery_validation.json'),
          'note': 'Fixed full4530x10epoch two-arm development comparison complete and independently audited. Read report85; no scientific/INT8 certification; no automatic candidate extension.'}, once=False)
    print(json.dumps({'status': decision['status'], 'joint_rule_met': delta['joint_rule_met'], 'delivery': str(OUT / 'delivery_validation.json')}), flush=True)


if __name__ == '__main__':
    try: audit()
    except BaseException as error:
        write(OUT / 'audit_failure_receipt.json', {'at_utc': now(), 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
        raise
