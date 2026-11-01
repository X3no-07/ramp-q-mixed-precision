"""Publish the completed conditional deployment evaluation from saved evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/gate05'
MATRIX = BASE / '20261021_dataset_matrix'
FINAL = BASE / '20261021_final_acceptance'
ORDER = ['fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    contract_path = BASE / '20261021_dataset_contracts/acceptance_contract.json'
    contract = read(contract_path)
    summary_path = MATRIX / 'target_evaluation/summary.json'
    target_path = MATRIX / 'target_evaluation/target_matrix.jsonl'
    review_path = MATRIX / 'target_evaluation/independent_result_verification.json'
    summary, review = read(summary_path), read(review_path)
    require(sha(contract_path) == '759506e78383fa262c3cb74735c24c57844d706fdb95bf95bce168d9b143f500' and
            sha(summary_path) == review['summary_sha256'] and sha(target_path) == review['target_matrix_sha256'] and
            sha(MATRIX / 'saved_prediction_matrix.json') == review['matrix_sha256'], 'Final evidence identity mismatch')
    require(review['status'] == 'ALL_SAVED_MATCHING_BITS_TARGET_SETS_AND_VERDICT_INDEPENDENTLY_RECOMPUTED' and
            review['full_target_rows_exact'] and review['target_hit_bits_verified'] == 10380 and
            review['new_model_calls'] == review['new_NMS_calls'] == 0 and review['protected_sources_unchanged'] and
            review['recomputed_primary_verdict'] == summary['primary_verdict'], 'Independent result verification incomplete')
    require(sha(ROOT / 'scripts/verify_gate05_saved_target_result_independent.py') == review['verification_tool_sha256'] ==
            '12b8d6c098ff2a88e3f0a2df234500e37c0cde719da978ad36e7896931ad0941', 'Independent verifier source changed')
    original = contract['protected_history']
    require(sha(ROOT / original['protected_preflight_path']) == original['protected_preflight_sha256'] and
            sha(ROOT / 'src/gate0.py') == 'f3c4d80658b79ae685e569b37c5ea89af9e6e9dd4a63ce86aff2806a7fcc5adb', 'Historical guard or matching implementation changed')
    post = read(MATRIX / 'postprocessed_detections/completion.json')
    require(post['NMS_calls'] == 1800 and post['original_raw_files_unchanged'] and
            post['status'] == 'ALL1800_COMMON_CPU_NMS_OUTPUTS_SAVED_NO_TARGET_SCORING', 'Common postprocessing incomplete')
    numeric = read(MATRIX / 'raw_native_ORT_descriptive_comparison.json')
    require(numeric['status'] == 'SAVED_RAW_DESCRIPTIVE_COMPARISON_COMPLETE' and numeric['all_outputs_finite'] and
            numeric['new_numeric_tolerances'] is False and numeric['numeric_equivalence_PASS_claim'] is False,
            'Raw descriptive comparison incomplete or incorrectly relabelled')
    targets = [json.loads(line) for line in target_path.read_text(encoding='utf-8').splitlines()]
    require(len(targets) == 865 and len({r['target_id'] for r in targets}) == 865, 'Target universe incomplete')
    counts = {backend: {cfg: sum(row[backend][cfg]['matched'] for row in targets) for cfg in ORDER} for backend in ('native', 'ort')}
    now = datetime.now().astimezone().isoformat()
    require(not FINAL.exists() and not (ROOT / 'docs/39_gate05_complete_150_image_acceptance_results.md').exists(), 'Final deliverable already exists; no overwrite')
    FINAL.mkdir()
    preserved = FINAL / 'protected_document_snapshots'
    preserved.mkdir()
    source_snapshots = {}
    for filename in ('README.md', 'docs/historical_status_and_plan.md'):
        source = ROOT / filename
        saved = preserved / filename
        saved.write_bytes(source.read_bytes())
        source_snapshots[filename] = {'snapshot': saved.relative_to(ROOT).as_posix(), 'sha256': sha(saved)}
    require(source_snapshots['docs/historical_status_and_plan.md']['sha256'] == contract['sources']['research_route']['sha256'],
            'Preserved original research scope citation differs from the frozen contract')
    r = summary['recovery']['r_only_G1']
    a, stable = len(summary['added_miss_target_ids']), len(summary['stable_target_ids'])
    selected_table = read(BASE / '20261021_dataset_contracts/qualification/selected_identity_table.json')
    precision = {r['name']: (r['quantized_source_convs'], r['actual_fp16_source_convs']) for r in selected_table['members']}
    rows = ['| Configuration | Reviewed source convolution INT8 / FP16 | Board-side hits / 865 | ORT hits / 865 | Inconsistency between board-side and ORT hit statuses |',
            '|---|---:|---:|---:|---:|']
    for cfg in ORDER:
        rows.append(f"| `{cfg}` | {precision[cfg][0]} / {precision[cfg][1]} | {counts['native'][cfg]} | {counts['ort'][cfg]} | {summary['native_ORT_target_disagreements'][cfg]} |")
    recovery_rows = ['| Recovery configuration | Recovered additional misses / 44 | Still missed detections | Stable target collateral misses / 678 | Net recovery count | New hits outside FP16 |',
                     '|---|---:|---:|---:|---:|---:|']
    for cfg, value in summary['recovery'].items():
        recovery_rows.append(f"| `{cfg}` | {len(value['rescued_target_ids'])} | {len(value['still_missing_target_ids'])} | {len(value['collateral_target_ids'])} | +{value['net_count']} | {len(value['outside_FP16_target_ids'])} |")
    numeric_rows = ['| Configuration | Maximum absolute difference in network coordinates | Maximum absolute difference in category scores | Maximum absolute difference in PE |', '|---|---:|---:|---:|']
    for value in numeric['configurations']:
        agg = value['aggregates']
        numeric_rows.append('| `' + value['configuration'] + '` | ' + ' | '.join(f"{agg[key]['max_abs_difference_finite_pairs']:.9g}" for key in
                            ('main_coordinates_4', 'class_scores_10', 'PE_output')) + ' |')
    report = f'''# Gate0.5: 150 images fully deployed acceptance results

Record date:{now}The six-configuration deployment acceptance has been fully completed; the pre-frozen **G1 same-target recovery existence criterion has been passed**. The original complete PyTorch/ORT numerical guardrail remains **FAILED_UNCHANGED**, and thus numerical equivalence cannot be claimed in this deployment.

## Main Results

Fixed 150 images, 865 annotated targets. Board-side FP16 baseline hits **{summary['FP16_hits']}**, full INT8 hits **{summary['q_all_hits']}** targets. Compared at the object level, FP16 detects targets that are missed by full INT8.{a}** instances, and the stable object-level detections shared by both are **{stable}number of items.

The predefined main configuration `r_only_G1` recovers **40 out of 44** targets (90.91%), while causing **3 missed detections** among the 678 stable targets (0.4425%). The net recovery count for this constrained set is **+37**, with a stable target retention rate of **99.5575%**. Still, **4 additional missed detections** remain unrecovered.

The total detection count for G1 configuration is **721**, an increase of **34** compared to full INT8\'s 687. This differs from the net recovery of +37: full INT8 detects an additional **9** targets among those missed by FP16, while G1 detects an additional **6**, resulting in a total detection difference of `40-3+(6-9)=34`. The value +37 cannot be interpreted as an increase of 37 in total detections.

The total count includes all size categories of targets. The 44 additional missed detections or the 40 recovered targets cannot be collectively labeled as \'small objects\'. The detection count is not equivalent to mAP.

## Six-configuration Results Compared to the Prespecified Secondary Benchmark

In this paper, \'full INT8\' refers to the W8A8 configuration of 88 source convolutions, and does not imply that every network operator is quantized to INT8. Floating-point paths and FLOAT output boundaries are separately preserved and reviewed.

{chr(10).join(rows)}

{chr(10).join(recovery_rows)}

G1 serves as the prespecified primary benchmark. G1+G5 and G1+G2 represent pre-frozen secondary results and are not used to modify the primary configuration based on their outcomes. These counts do not prove that G1+G5 outperforms G1 under equivalent computational cost; no evaluation of latency, energy consumption, or resource budget equivalence has been conducted in this study.

The auxiliary `q_only_G1` causes 35 FP16-detected targets to disappear. Of these, 26 also satisfy both conditions: missed by full INT8 and recovered after G1 restoration, representing 26/44 (59.09%) of the additional misses. This is a limited same-target bidirectional pattern. The auxiliary engine uses the disclosed **PREFER + 82 explicit FP16 compute/output requests**; the other main members use NONE. It cannot serve as a pure G1 causal control under identical build constraints, and it does not replace the primary comparison.

## Execution and Matching Evidence

- Each of the six board ends executes one complete 150 graph, totaling **900** times; the corresponding six complete ONNX graphs are each executed 150 times on CPU ORT, totaling **900** times. All are full graphs, with 0 warm-up, 0 additional inferences, and 0 automatic retries.
- All 1,800 primary outputs and PE outputs are finite. The board-end 6 ZIPs\' 1,218 receipt files have been recovered and verified; all related PID entries under the same boot for 30 tasks and serial queue PIDs are absent, and 6 wrapper process groups have been physically cleared, with no supervision intervention.
- The actual layer inspection JSON from the deserialization engine matches exactly with the layer inspection content of the qualified build. The environment, preceding and following library SHAs, maps device numbers/inodes, source images, and engine identities are all authenticated. The actual precision and valid fusion review of each source convolution is separated from the per-image runtime kernel and static signed INT8 arithmetic augmentation, without fabricating historical missing tactic names.
- Shared preprocessing NPY files and identical CPU post-processing environment were used across all runs; all **1,800 NMS operations** were successfully saved with no warnings or timeouts. The NMS candidate confidence=0.001, IoU=0.7, max_det=1000; target matching requires confidence≥0.25, same category, IoU≥0.5, one-to-one, and follows the original reverse priority rule `(IoU, confidence, GT_index, detection_index)`.
- Independent re-matching was performed from saved detections to validate **10,380 target detection positions**, including complete IoU, confidence, and detection IDs; all 865 target rows, recovery, collateral misses, net counts, category descriptions, and final criteria are fully consistent. No original evaluator matching function was invoked, and no model or NMS re-execution was performed.

## Numerical and Research Boundaries

The following describes the descriptive maximum absolute difference between the corresponding complete ORT outputs and the board-end stored arrays, without introducing new tolerances or considering them acceptable. The network coordinates consist of 4 coordinate channels from the original output, with category scores in an additional 10 channels; PE per column. The full report also includes per-image, mean, RMS, byte differences, and all identities.

{chr(10).join(numeric_rows)}

Although FP16 shows original array differences, the target detection positions in this set are consistent with ORT. The quantization configuration exhibits differences in **26/17/19/8/1** target detection positions, and thus cannot be considered equivalent across backends. The original PyTorch/ORT pre-check files and original tolerances remain unmodified. The new full ONNX reference is an independently deployed semantic reference and does not constitute reclassifying original failures as passes.

During actual compilation, the FP16 baseline uses FP16 for DFL Softmax/decoding, FP32 for quantization configuration, and partial SiLU/residual floating-point partitioning and implementations differ. These differences were disclosed and frozen prior to reading the outputs from the 150 images. Thus, the support here is for the **existence of object-level disappearance or recovery under the selected real-world deployment configuration**, not for proving the causal arithmetic mechanism of pure G1, the uniqueness of all differences, overall accuracy superiority, or Tensor Core performance.

New NONE `q_only_G1` PE Conv actually operates in FP32; only 81 recovery sources reach FP16, yet still fail qualification and original failure is retained. Prior to execution, a legacy same-image 6 INT8+82 FP16 engine was selected as a limited auxiliary member based on actual precision qualification, without selecting the engine using results from 150 images.

The previously fixed set was used for diagnosis and cannot be packaged into a fresh, independent holdout set; the file identities of 64 calibration images and the 150-image set remain non-overlapping. No new dataset, multiple sub-significance, size stratification superiority, or performance acceptance was conducted. Board-side clock and computer clock differ; herein, evidence is linked via SHA, boot ID, process identity, and unique attempt markers, without adjusting the board-side clock.

## Evidence and Review Entry

- [Frozen Acceptance Contract] (artifacts/gate05/20261021_dataset_contracts/acceptance_contract.json), SHA `759506e78383fa262c3cb74735c24c57844d706fdb95bf95bce168d9b143f500`.
- [Original Output Index] (artifacts/gate05/20261021_dataset_matrix/raw_family_index.json), [Engineering Qualification] (artifacts/gate05/20261021_dataset_matrix/engineering_eligibility.json).
- [Unified Detection Matrix] (artifacts/gate05/20261021_dataset_matrix/saved_prediction_matrix.json), SHA `{review['matrix_sha256']}`.
- [Target Result Summary] (artifacts/gate05/20261021_dataset_matrix/target_evaluation/summary.json), SHA `{sha(summary_path)}`.
- [Complete 865-Target Matrix] (artifacts/gate05/20261021_dataset_matrix/target_evaluation/target_matrix.jsonl), SHA `{sha(target_path)}`.
- [Independent Recalculation Record] (artifacts/gate05/20261021_dataset_matrix/target_evaluation/independent_result_verification.json), SHA `{sha(review_path)}`.
- [Original Numerical Comparison] (artifacts/gate05/20261021_dataset_matrix/raw_native_ORT_descriptive_comparison.json).
- [Final Machine State and Range Judgment] (artifacts/gate05/20261021_final_acceptance/final_decision.json), [Final File List] (artifacts/gate05/20261021_final_acceptance/final_manifest.json). The original research pathway has been saved under frozen SHA `protected_document_snapshots/`; records from stages 23–38 and failure materials are retained.

Subsequent work should be separately frozen: revalidate using data not involved in diagnosis and multiple calibration seeds, and independently verify against original numerical guardrails and backend discrepancies; after selection locking, real-world latency and power consumption should be measured. This round does not automatically initiate the next stage or re-run the 900 board-side tasks.
'''
    report_path = ROOT / 'docs/39_gate05_complete_150_image_acceptance_results.md'
    report_path.write_text(report, encoding='utf-8')
    decision = {'schema': 'rampq.gate05.final_completed_evaluation_decision/v1', 'completed_at': now,
        'status': 'COMPLETE_CONDITIONAL_DEPLOYMENT_EVALUATION', 'gate05_evaluation_complete': True,
        'primary_verdict': summary['primary_verdict'], 'unqualified_numerical_migration_PASS': False,
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'original_guard_path': original['protected_preflight_path'],
        'original_guard_sha256': original['protected_preflight_sha256'], 'thresholds_changed': False,
        'images': 150, 'physical_targets': 865, 'native_model_calls': 900, 'full_CPU_ORT_model_calls': 900,
        'common_CPU_NMS_calls': 1800, 'new_model_or_NMS_calls_by_finalizer': 0,
        'independently_verified_target_bits': 10380, 'hits': counts,
        'added_misses': a, 'stable_targets': stable, 'primary_G1_rescued': len(r['rescued_target_ids']),
        'primary_G1_stable_collateral': len(r['collateral_target_ids']), 'primary_G1_net_within_FP16_universe': r['net_count'],
        'primary_G1_total_hits_change_over_qall': counts['native']['r_only_G1']-counts['native']['q_all'],
        'native_ORT_target_disagreements': summary['native_ORT_target_disagreements'],
        'report': report_path.name, 'protected_document_snapshots': source_snapshots,
        'remaining_work_inside_frozen_evaluation': [], 'next_stage_started': False,
        'limits': ['Conditional actual-deployment existence; no pure G1 arithmetic causality',
                   'Original numerical guard not repaired; corresponding ORT/native differences retained',
                   'Auxiliary q_only uses predeclared different PREFER policy',
                   'Previously inspected diagnostic set, not fresh independent holdout',
                   'All-size target counts, not mAP or exclusive small-object claims',
                   'No latency, energy or general superiority validation']}
    write(FINAL / 'final_decision.json', decision)
    state_path = BASE / 'current_execution_state.json'
    (FINAL / 'execution_state_before_final_delivery.json').write_bytes(state_path.read_bytes())
    state = read(state_path)
    state.update(client_at=now, gate05_status='COMPLETE_CONDITIONAL_DEPLOYMENT_EVALUATION', active_native_job=None,
                 report=report_path.name, final_decision='20261021_final_acceptance/final_decision.json',
                 next=['A new independent validation stage requires a separate frozen plan; not started in this task'])
    state['dataset_evaluation'].update(status='COMPLETE_INDEPENDENTLY_RECOMPUTED_AND_DELIVERED', pending=[],
                 independent_target_bits_verified=10380, unqualified_numerical_migration_PASS=False,
                 final_decision='20261021_final_acceptance/final_decision.json')
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for filename in ('README.md', 'docs/historical_status_and_plan.md'):
        path = ROOT / filename
        old = path.read_text(encoding='utf-8')
        if filename == 'README.md':
            old = old.replace('> Current Stage: RUOD-R Gate 0 has been closed out; Gate 0.5 DFL has been reduced to six nodes (QDQ + Conv), with NONE successfully passing while OBEY still fails under the same assertion; Softmax and final Reshape are not necessary conditions for this local failure, and the original full guardrail still fails (2026-10-21)',
                              '> Current stage: Gate0.5 pilot deployment acceptance completed, G1 same-object recovery existence passed; original complete numerical guardrail still failed (2026-10-21)')
        first, rest = old.split('\n', 1)
        update = ('\n\n## Latest Acceptance Result (2026-10-21)\n\n'
                  "Gate0.5's six-configuration real-deployment acceptance has been fully completed, with independent recalculation validation of 10, 380 targets detected. In 150 images / 865 targets,"
                  'FP16 hits 722, full INT8 hits 687; 44 additional misses are observed compared to FP16 in full INT8. The designated main configuration recovers only G1,'
                  'recovering 40 of them, with 3 collateral misses on stable targets, resulting in a net recovery of +37; total hits reach 721, an increase of 34 over full INT8.\n\n'
                  '**Pass is achieved for the existence criterion of same-target recovery under the selected real-world deployment configuration. The original full PyTorch/ORT numerical guardrail remains FAILED_UNCHANGED,'
                  'and numerical migration equivalence, pure G1 arithmetic causality, mAP/performance, or overall superiority cannot be claimed.**'
                  'This statistics includes all-sized object-level targets. The different PREFER constraints of the previous auxiliary q_only have been disclosed, and the diagnostic reuse set does not falsely claim to be a fresh, independent holdout set.\n\n'
                  'Complete results, definitions of recovery and collateral misses, corresponding ORT differences, and supporting evidence are provided in '
                  '[Report 39: Gate05 complete 150-image acceptance results](39_gate05_complete_150_image_acceptance_results.md).'
                  'The previous-stage explanations are retained; the frozen original text is preserved in its original SHA at '
                  '`artifacts/gate05/20261021_final_acceptance/protected_document_snapshots/`。\n')
        path.write_text(first + update + rest, encoding='utf-8')
    paths = [report_path, ROOT/'README.md', ROOT/'docs/historical_status_and_plan.md', state_path,
             contract_path, BASE/'20261021_dataset_contracts/raw_contract.json', MATRIX/'raw_family_index.json',
             MATRIX/'engineering_eligibility.json', MATRIX/'raw_native_ORT_descriptive_comparison.json',
             MATRIX/'saved_prediction_matrix.json', MATRIX/'postprocessed_detections/completion.json',
             MATRIX/'postprocessed_detections/environment.json', summary_path, target_path, review_path,
             BASE/'20261021_dataset_native_downloads/final_native_observation.json',
             BASE/'20261021_gate05_completion_plan/assembler_final_independent_review.json',
             BASE/'20261021_dataset_contracts/qualification/selected_identity_table.json',
             BASE/'20261021_dataset_contracts/qualification/technical_decision.json',
             BASE/'20261021_dataset_contracts/qualification/float_fusion_comparability.json',
             ROOT/original['protected_preflight_path'], ROOT/'src/gate0.py']
    for filename in ('run_gate05_frozen_raw_dataset_v2.py', 'postprocess_gate05_frozen_raw_matrix_v2.py',
                     'evaluate_gate05_saved_target_matrix_v2.py', 'assemble_gate05_saved_raw_index.py',
                     'compare_gate05_saved_raw_reference.py', 'verify_gate05_saved_target_result_independent.py',
                     'finalize_gate05_saved_acceptance.py'):
        paths.append(ROOT/'scripts'/filename)
    paths.extend(path for path in FINAL.rglob('*') if path.is_file())
    for stream in read(MATRIX/'raw_family_index.json')['streams']:
        paths.append(MATRIX/stream['file'])
    files = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(paths))}
    write(FINAL/'final_manifest.json', {'schema':'rampq.gate05.final_evidence_delivery_manifest/v1', 'completed_at':now,
        'status':'COMPLETE_CONDITIONAL_DEPLOYMENT_EVALUATION', 'files':files,
        'scope':'Root evidence and nested raw/detection manifests authenticate full artifacts; per-array raw hashes remain in all12 raw manifests.',
        'new_model_or_NMS_calls':0, 'original_complete_model_numerical_guard':'FAILED_UNCHANGED'})
    print(json.dumps({'status':decision['status'], 'primary_verdict':summary['primary_verdict'],
                      'report':str(report_path), 'final_manifest_sha256':sha(FINAL/'final_manifest.json'),
                      'files':len(files), 'new_model_or_NMS_calls':0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
