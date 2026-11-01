"""Report every frozen extension cohort, with no refitting or favorable-subset selection."""
from datetime import datetime,timezone
from audit_gate1_scorer_extension_v2 import E,P,S,OUT,A
read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/'contract.json');comparison=read(A/'comparison_result.json');audit=read(OUT/'saved_delivery_audit.json')
    require(audit['status']=='PASS_FROZEN_EXTENSION_SOURCE_STATE_INPUT_MATRIX_AND_CURVE_AUDIT_ONLY','Audit not complete')
    additional={}
    for cohort,r in comparison['cohorts'].items():
        score=r['branches']['frozen_score_base'];qat=r['branches']['ordinary_qat'];ref=r['branches']['mse_r']
        additional[cohort]={
            'AP_score_minus_QAT_percentage_points':score['official']['metrics_percent']['AP_all']-qat['official']['metrics_percent']['AP_all'],
            'TP_score_minus_QAT_at_referenceR_FP_budget':score['at_FP_budget']['TP']-qat['at_FP_budget']['TP'],
            'FP_budget':r['reference_FP_budget'],'score_AP_minus_R':r['score_AP_delta_vs_R_percentage_points'],
            'TP_score_minus_R_same_FP':r['score_TP_delta_at_same_FP_budget'],
            'operating_TP_delta_vs_R':r['frozen_threshold_TP_delta'],'operating_FP_delta_vs_R':r['frozen_threshold_FP_delta'],
            'curve_joint_observation':r['joint_development_rule_met'],'frozen_operating_joint_observation':r['operating_joint_observation']}
    decision={'status':'COMPLETE_FROZEN_SCORER_286_EXTENSION_DEVELOPMENT_ONLY','at_utc':P.now(),
        'contract':identity(OUT/'contract.json'),'comparison':identity(A/'comparison_result.json'),'audit':identity(OUT/'saved_delivery_audit.json'),
        'summary_source':identity(__file__),'cohorts':comparison['cohorts'],'additional_comparisons':additional,
        'primary_new286_joint_rule_met':comparison['primary_new286_pass'],
        'all_three_overlapping_cohorts_joint_rule_met':comparison['all_three_cohorts_joint_rule_met'],
        'primary_frozen_operating_joint_observation':comparison['cohorts']['new286']['operating_joint_observation'],
        'frozen_operating_threshold':c['scorer_operating_threshold'],'scorer_was_not_refit':True,'scorer_state':c['scorer_model'],
        'new_detector_image_forwards':572,'model_fits':0,'AP_jobs':9,'observed_worker_exit_code':0,
        'old128_reused_not_reinferred':True,'prefix_proxy_not_verified_scene':True,'old128_used_to_choose_scorer':True,
        'scientific_acceptance':False,'independent_test_set':False,'actual_INT8':False,'Gate1_overall_passed':False,
        'old_numerical_FAILs_reclassified':False,'new286_raw_numeric_qualification':'NOT_RUN',
        'Selection_payload_access':False,'Holdout_payload_access':False,
        'next_action':'Frozen extension budget exhausted and all cohorts retained. Report primary286,new-prefix47,combined414 and training-fixed threshold together. Do not continue tuning these developer images or treat them as independent acceptance. Any next hypothesis,replication budget,data access or device stage must be fixed separately before execution.'}
    write(OUT/'machine_decision.json',decision)
    primary=decision['primary_new286_joint_rule_met'];robust=decision['all_three_overlapping_cohorts_joint_rule_met'];oper=decision['primary_frozen_operating_joint_observation']
    if robust:
        lead='The frozen scorer achieved the predefined AP and identical false positive budget conditions across the new 286 images, the new prefix 47 images, and the merged 414 images, demonstrating reproducibility of extended development results.'
    elif primary:
        lead='The frozen scorer achieves the prescribed joint observation criteria on the primary additional 286 images, but the results across the three predefined ranges are not fully consistent, and thus cross-range stability cannot be claimed.'
    else:
        lead='The frozen scorer fails to meet the prescribed AP and same false positive budget joint observation criteria on the primary new 286 images, and thus the positive signal from the original 128 images cannot be considered stably reproducible.'
    op_text='The fixed threshold determined on the training set also satisfies the observation condition of no degradation in detection performance and no increase in false positives on the new 286 images.' if oper else "The fixed training threshold does not simultaneously satisfy 'detection count not reduced' and 'false positive count not increased' on the new 286 images. Therefore, curve improvement cannot be directly converted into a commitment to using this fixed threshold."
    names={'mse_r':'Original mixed-precision R','frozen_score_base':'existing ordinary QAT','ordinary_qat':'scorer threshold **'}
    cohorts={'new286':'additional 286 images (primary)','new_prefix47':'new prefix 47 images (auxiliary)','full414':'merged 414 images (auxiliary)'}
    lines=['# Gate1: Frozen Ordinary Scorer - Extended Validation Results for 286 Images','','2026-10-26。','',f'**{lead}**','',op_text,'',
        'This round remains training_val extended development, not independent scientific validation, Gate1 full acceptance, or real INT8/board-end validation. All ranges and three fixed branches are preserved.','',
        '## Complete Results of AP and Same False Positive Budget','',
        '| Scope | Branch | AP50–95(%) | Maximum TP under the same reference FP budget | Actual FP | Minimum FP to reach reference TP |',
        '|---|---|---:|---:|---:|---:|']
    for cohort,r in comparison['cohorts'].items():
        for branch,m in r['branches'].items():
            t=m['at_FP_budget'];f=m['at_TP_target'];minimum=str(f['FP']) if f['achievable'] else 'Unable to achieve'
            lines.append(f"| {cohorts[cohort]} | {names[branch]} | {m['official']['metrics_percent']['AP_all']:.6f} | {t['TP']} | {t['FP']} | {minimum} |")
    lines += ['', 'For each range, the false positive budget is independently calculated based on the official IoU0.5/cap500 FP count at score ≥ 0.25 of the original R, fixed prior to new inference. The original R itself also takes the maximum TP under the same budget across the full curve, avoiding comparison with only a single unfavorable reference point; all candidates with the same score are included. The following differences are listed in full, without selecting favorable ranges:','',
        '| Scope | Reference FP budget | AP scorer − R (percentage points) | TP scorer under same budget − R | AP scorer − baseline QAT (percentage points) | TP scorer under same budget − baseline QAT |','|---|---:|---:|---:|---:|---:|']
    for cohort,d in additional.items():
        lines.append(f"| {cohorts[cohort]} | {d['FP_budget']} | {d['score_AP_minus_R']:+.6f} | {d['TP_score_minus_R_same_FP']:+d} | {d['AP_score_minus_QAT_percentage_points']:+.6f} | {d['TP_score_minus_QAT_at_referenceR_FP_budget']:+d} |")
    lines += ['', 'In the previous 128-image set, the AP of the baseline scorer increased from 11.986118% to 12.083086%, with 37 additional detections under the same official false positive budget of 2605. This 128-image result was used to select the current frozen scorer, and thus cannot be combined with the new 286-image set to claim an independent 414-image test set.','',
        '## Actual checks with fixed training-set thresholds','',
        f"scorer threshold **{c['scorer_operating_threshold']:.17g}** Only computed using the original 512 training image cache and a frozen scorer. Both original R and standard QAT use the existing 0.25 threshold; no adjustment of the score threshold is made based on the newly developed images. The training threshold is set such that the nominal false positives (FP) under the original training label definition do not exceed the FP count of R at 0.25, but this does not guarantee that the false positives in the new images are exactly equal.",'',
        '| Scope | Branch | Threshold used | Actual detections by project | Actual false positives by project |','|---|---|---:|---:|---:|']
    for cohort,r in comparison['cohorts'].items():
        for branch,m in r['branches'].items():
            t=m['project'];counts=t['operating_counts']
            lines.append(f"| {cohorts[cohort]} | {names[branch]} | {t['operating_threshold']:.9f} | {counts['hits']} | {counts['false_positive_detections']} |")
    lines += ['', op_text,'','Target matching is prioritized by IoU, while official sorting diagnosis uses score priority, with cross-category cap500 applied in the latter; these two counts are not interchangeable. The optimal threshold derived from development annotations is used solely for performance evaluation and is not used to replace the training-set threshold.','',
        'Simultaneously preserves the project counts from all branches under the original 0.25 threshold:','',
        '| Scope | Branch | Original 0.25 detection | Original 0.25 false positives |','|---|---|---:|---:|']
    for cohort,r in comparison['cohorts'].items():
        for branch,m in r['branches'].items():
            t=m['project']['conf025_counts'];lines.append(f"| {cohorts[cohort]} | {names[branch]} | {t['hits']} | {t['false_positive_detections']} |")
    lines += ['', '## Data Boundaries and Correlations','',
        'The newly added 286 images are all members of the original training_val minus the old 128 images, and their image SHA values do not overlap with the complete 4530 training list. Among 286, 239 images share the same file name prefixes as those in the old 128, while 47 images come from 21 new prefixes; the original 128 covers 37 prefixes, and the new 286 covers 53 prefixes, of which 32 are overlapping. The members of the new prefixes 47 images are determined solely based on existing metadata prior to this detection round, and no selection is made based on the current detection results.', '',
        'File name prefixes are not validated independent scene identities. The 286, 47, and 414 sets overlap with each other and do not represent three independent scientific experiments; these training_val images have also been used in earlier floating-point development checks. Even if all three ranges show improvement, this can only be described as an extended development reproducibility and cannot be claimed as independent testing or statistically significant.', '',
        'This round only validates the reproducibility of the standard scorer. The previous intervention sampling did not outperform the standard scorer, and thus the benefits of the standard scorer cannot be rebranded as G1 intervention contributions. The quality scoring itself is part of an existing research direction, and originality still requires additional mechanisms and strong control evidence.','',
        '## Execution and Independent Audit Verification','',
        'The frozen 23 detector parameters, features, normalizations, and both detectors were not retrained. Each new 286 image runs two existing detectors, completing 572 forward passes on images; the old 128 outputs are directly reused. The complete state fingerprints of both detectors remain identical before and after inference, and the initial values of ordinary QAT are consistent with the saved files from the existing round 3. The new model is trained for 0 epochs, and the activation range is recalibrated 0 times.', '',
        f"8 items passed pre-check; saved-artifact audit verification{audit['manifest_members_verified']}A set of output members,{audit['reused_members_verified']}one reusable binding, 572 original arrays, 572 new image/annotation inputs. The model's visible features and scorer outputs for 286 images are independently reconstructed, confirming that the scorer has not altered bounding boxes, classes, or candidate members; all new GT/DT filtering matrices are validated against original annotations and outputs with zero tolerance, and the original 128 matrices are cross-checked against predicted reuse.",'',
        'All 9 aggregated tasks actually reported exit code 0, with only lightweight NumPy processes handling cached matrices. The audit recalculated 63 aggregated metrics from saved AP/AR arrays and re-computed IoU0.5 matching flags and full group tie curves; no detector re-execution or full AP pipeline was performed. The training threshold was independently recalculated from cached data and model parameters, and was consistent. The actual threshold matching for the fixed 32 new images was also recalculated and verified.', '',
        "The first version of the saved audit terminated at the end of the 9th group due to an undefined 'now' name at the final report timestamp, and the PASS file was not generated. The original frozen script, contract, and error receipts were preserved as-is; an independent audit_v2 was implemented using P.now, completing a read-only verification and saving the results. This error does not involve model inference, scorer training, or the 9 formal AP tasks, and does not alter their completion status.", '',
        'The new 286 images did not reconstruct the original FP32 full candidate reference, and thus do not claim numerical qualification; the historical complete 8400 candidate values remain as FAIL. Frozen parameters, saved consistency, AP or ranking improvements cannot substitute for original numerical qualification or board-level acceptance.','',
        '## Stage Conclusion and Future Boundaries','',lead,'',op_text,'',
        "The current budget has been fully executed and sealed; all ranges have reported results. No additional threshold search, candidate expansion, training rounds, or feature filtering will be performed to meet a 'must succeed' requirement. If further evidence is needed for publication, a specific problem must first be frozen, and a data access scheme not previously used in repeated development must be established, along with supplementary stability, strong baselines, and actual computational overhead measurements.", '',
        'Selection load not read; Holdout remains sealed; no cloud resources or scheduled tasks were created. Evidence directory is `artifacts/gate1/20261026_scorer_extension_286_v1`; primary files include `contract.json`, training threshold calibration record, 9 group saved matrix/evaluation curves, `attempt_01/comparison_result.json`, completion and actual exit acknowledgments, `saved_delivery_audit.json`, `machine_decision.json`, and final delivery verification.']
    report=P.ROOT/'docs/73_gate1_frozen_scorer_286_image_extension_review.md'
    with report.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    print(P.json.dumps({'status':decision['status'],'primary_new286_joint_rule_met':primary,'all_three_joint_rule_met':robust,
        'frozen_operating_primary_joint_observation':oper,'comparisons':additional,'report':identity(report)},ensure_ascii=True))


if __name__=='__main__':main()
