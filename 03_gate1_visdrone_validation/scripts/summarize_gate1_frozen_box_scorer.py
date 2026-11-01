"""Assess all fixed scorers against the pre-execution rules; never select a winner."""
from datetime import datetime,timezone
import numpy as np
from audit_gate1_frozen_box_scorer import S,P,OUT,A

read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/'contract.json');comparison=read(A/'comparison_result.json');audit=read(OUT/'saved_delivery_audit.json')
    require(audit['status']=='PASS_SAVED_SCORER_INPUT_STATE_POOL_AND_CURVE_AUDIT_ONLY','Audit not complete')
    by_label={m['branch']:m for m in comparison['metrics']};ranks=comparison['ranking_comparisons']
    reference=by_label['mse_r'];basetp=ranks['mse_r']['at_FP_budget']['TP']
    old_QAT=read(P.OUT/'attempt_01/evaluation/ordinary_qat/metrics.json')
    decisions={}
    for arm in S.ARMS:
        m=by_label[arm];ap=m['metrics_percent']['AP_all'];tp=ranks[arm]['at_FP_budget']['TP']
        delta_ap=ap-reference['metrics_percent']['AP_all'];delta_tp=tp-basetp
        versus={}
        for other in S.ARMS:
            if other==arm:continue
            versus[other]={'AP_delta_percentage_points':ap-by_label[other]['metrics_percent']['AP_all'],
                           'TP_delta_at_fixed_FP_budget':tp-ranks[other]['at_FP_budget']['TP']}
        decisions[arm]={'AP_delta_vs_R_percentage_points':delta_ap,'TP_delta_vs_R_at_fixed_FP_budget':delta_tp,
            'AP_delta_vs_old_ordinary_QAT_percentage_points':ap-old_QAT['metrics_percent']['AP_all'],
            'project_conf025_hits_delta':m['project_counts_conf025']['hits']-reference['project_counts_conf025']['hits'],
            'project_conf025_FP_delta':m['project_counts_conf025']['false_positive_detections']-reference['project_counts_conf025']['false_positive_detections'],
            'development_joint_rule_met':delta_ap>0 and delta_tp>=0,'versus_other_scorers':versus}
    treatment=decisions['score_intervention']
    added_value=all(x['AP_delta_percentage_points']>0 and x['TP_delta_at_fixed_FP_budget']>=0 for x in treatment['versus_other_scorers'].values())
    decision={'status':'COMPLETE_FROZEN_BOX_SCORER_DEVELOPMENT_PILOT','at_utc':datetime.now(timezone.utc).isoformat(),
        'contract':identity(OUT/'contract.json'),'comparison':identity(A/'comparison_result.json'),'audit':identity(OUT/'saved_delivery_audit.json'),
        'summary_source':identity(__file__),'metrics':comparison['metrics'],'ranking_comparisons':ranks,
        'FP_budget':comparison['FP_budget'],'TP_target':comparison['TP_target'],'decisions':decisions,
        'intervention_descriptive_advantage_over_both_scorers':added_value,
        'intervention_meets_development_joint_rule':treatment['development_joint_rule_met'],
        'registered_candidates_meeting_development_joint_rule':[arm for arm in S.ARMS if decisions[arm]['development_joint_rule_met']],
        'Gate1_overall_passed':False,'independent_scientific_acceptance_passed':False,'deployment_threshold_frozen':False,
        'old_ordinary_QAT_AP_percent':old_QAT['metrics_percent']['AP_all'],
        'all_candidate_scores_and_results_preserved':True,'new_detector_forwards':0,'detector_weights_changed':False,
        'new_scorer_states':3,'optimizer_updates_each':100,'selected_positive_rows':audit['positive_selected_each'],
        'selected_negative_rows':audit['negative_selected_each'],'candidate_pool_boxes_categories_unchanged':True,
        'GT_or_Q_features_at_score_time':False,'full_scoring_current_native_oracle':False,'scientific_acceptance':False,
        'independent_test_set':False,'actual_INT8':False,'old_numerical_FAILs_reclassified':False,
        'same_FP_thresholds_are_descriptive_only':True,'Selection_payload_access':False,'Holdout_payload_access':False,
        'next_action':'Close fixed3candidate pilot after source/delivery audit; report generic scoring usefulness separately from intervention-specific extra value. No additional sweep,seed,feature,threshold or data access is automatically authorized by a negative or positive development observation.'}
    write(OUT/'machine_decision.json',decision)
    names={'mse_r':'Original mixed-precision R','score_base':'Standard scorer','score_hard':'Standard hard-example scorer','score_intervention':'Intervention-based sampling scorer'}
    if decisions['score_base']['development_joint_rule_met']:
        lead='The standard scorer, pre-specified in advance, meets the joint observation condition for this development cycle: relative to the original R, it improves AP while maintaining the same false positive budget and achieving or better detection performance. This constitutes a development signal, not a full scientific acceptance or gate-level approval of Gate1 or any independent component.'
    elif treatment['development_joint_rule_met']:
        lead='The intervention-based sampling scorer meets the predefined development joint observation criteria: relative to the original R, AP improves and detection performance does not degrade under the same false positive budget.'
    else:lead='The scorer fails to achieve the required joint improvement; reductions in fixed-threshold false positives or partial metric improvements cannot be considered a pass.'
    intervention_lead='The intervention-based sampling scorer meets the predefined development joint observation criteria relative to the original R.' if treatment['development_joint_rule_met'] else 'The intervention-based sampling scorer fails to meet the predefined development joint observation criteria relative to the original R.'
    extra='The intervention-based scorer also demonstrates a descriptive phenomenon of joint improvement relative to the other two scorers.' if added_value else 'The intervention sampling has not demonstrated a relative advantage over the standard scorer when matched against hard examples, and thus its additional contribution remains unproven.'
    lines=['# Gate1: Frozen bounding box scorer: results and fixed false positive budget check','','2026-10-26。','',f'**{lead}**','',intervention_lead+' '+extra,
        '','These judgments apply only to the 128 training_val images repeatedly used in development; they do not constitute independent scientific acceptance, real INT8, or board-level approval.','',
        '## Scoring Results','','| Branch | AP50–95(%) | Fixed 0.25 threshold project detection | Fixed 0.25 threshold project false positives |','|---|---:|---:|---:|']
    for arm,m in by_label.items():
        t=m['project_counts_conf025'];lines.append(f"| {names[arm]} | {m['metrics_percent']['AP_all']:.6f} | {t['hits']} | {t['false_positive_detections']} |")
    lines += ['',f"The AP of ordinary QAT on the same 128 image is{old_QAT['metrics_percent']['AP_all']:.6f}%. It serves as a development counterpart for another detection model and does not include its predicted bounding boxes in the fixed R candidate pool for this study.",'',
        'Project-level matching prioritizes IoU over the official scoring method, which differs from the standard score-based matching definition. After score adjustment, the 0.25 threshold count is also affected by the score scaling, thus it must be combined with the following identical false positive budget check and complete AP evaluation.','',
        '## Same False Positive Budget Ranking Check','',
        f"The reference rule is fixed prior to new candidate execution: B takes the official IoU0.5, cap500 false positive count at score ≥ 0.25 of the original R, i.e., **{comparison['FP_budget']}**; T is the true positive count at the same score point **{comparison['TP_target']}**. For fair comparison, the original R itself is evaluated using the maximum true positives (TP) under false positive (FP) ≤ B across the full curve, rather than only at the 0.25 score point. All tied candidates are included as a whole, without splitting tied groups.",'',
        '| Branch | Maximum TP under FP ≤ B | Corresponding FP | Minimum FP required to reach T |','|---|---:|---:|---:|']
    for arm,rank in ranks.items():
        point=rank['at_FP_budget'];target_point=rank['at_TP_target'];minfp=str(target_point['FP']) if target_point['achievable'] else 'Unable to achieve'
        lines.append(f"| {names[arm]} | {point['TP']} | {point['FP']} | {minfp} |")
    lines += ['', 'These curve points are derived from development annotations for performance evaluation, and the resulting score thresholds are not directly deployable. Final deployment thresholds must be established independently on development or calibration data separate from the validation set. The optimal point on the curve is not used as justification for modifying the original 0.25 threshold.','',
        '## Intervention Group Relative to Controls','', '| Comparison Object | AP Difference (in percentage points) | TP Difference under Same FP Budget |','|---|---:|---:|']
    lines.append(f"| Original R | {treatment['AP_delta_vs_R_percentage_points']:+.6f} | {treatment['TP_delta_vs_R_at_fixed_FP_budget']:+d} |")
    for arm,d in treatment['versus_other_scorers'].items():lines.append(f"| {names[arm]} | {d['AP_delta_percentage_points']:+.6f} | {d['TP_delta_at_fixed_FP_budget']:+d} |")
    lines += ['', '## Input and Supervision Boundaries','',
        'The scorer has only 22 models visible inputs and 23 trainable parameters. Features include original and second-highest class scores, class gaps and entropy, bounding box width and height, center coordinates, overlap and scores with neighboring instances of the same class, and overlap with other classes, as well as one-hot class encoding. Ground truth box distances, matching IoU, true/false labels, and Q-path outputs are not included as inputs to the scorer. Ground truth is used solely for training soft labels and offline sampling.', '',
        f"512 images from existing training data generate{audit['training_rows']}unignored candidate training entries. The intervention group selects these.{audit['positive_selected_each']}a set of positive routine cases,{audit['negative_selected_each']}a negative routine; the standard hard example group is matched by category/scale and category/confidence interval/error type to have the same number of instances. The original 601 intervention GTs and 882 erroneous channels retain only the members still available within a fixed candidate pool, without synthesizing missing data. The two groups' sampling overlap: positive examples{audit['hard_intervention_positive_overlap']}current retained members and{audit['hard_intervention_negative_overlap']}。",'',
        'All scorers are initialized from the same zero-residual state and undergo 100 fixed full-batch Adam updates. The common loss assigns equal weight to each training image, with an additional balanced binary cross-entropy (BCE) loss of 0.25 weight for both positive and negative examples in the two sampling groups; an L2 coefficient of 0.0005. No optimal iteration, random seed, feature, threshold, or parameter is selected. Mean and variance normalization is fitted solely from training features, and development images are not provided with labels or loss during training.', '',
        '## Verified Completeness','',
        f"10 pre-checks passed. Audit verification{audit['manifest_members_verified']}A list of current saved members and{audit['reused_members_verified']}reused bindings. Independently reconstruct development features for 128 images and deterministic training-probe features for 32 images; check the three saved scorers, unchanged boxes/classes/identities in 384 development candidate pools, and all seven AP/AR aggregates. An IoU match flag for the official-style curve is recomputed from saved matrices. The complete tied-score curve matches an independent recomputation. Neither model training nor detector forwards are repeated.",'',
        'This round performs 0 new detector image forwards, 0 updates to existing weights, and 0 calibrations. Training features use the fixed NMS pool from saved R arrays; the development pool is reused directly. NMS is not rerun after scoring and no boxes are added; only scores and submission order change. Old raw outputs and every historical all-candidate numerical FAIL remain unchanged. These scoring outputs do not certify original-tensor numerical qualification or device qualification.', '',
        'All workflow completion receipts are complete, and the launcher observed a propagation exit code of 0. An additional final complete delivery hash check is performed, and the saved consistency PASS is not equated with method effectiveness PASS.', '',
        '## Explanation and Termination Criteria','',lead,'',extra,'',
        'If the standard scorer is effective, it indicates that model-visible features possess a meaningful ranking value. Only if the intervention sampling consistently and stably outperforms both the standard training and the standard hard example setup under the same architecture, does it justify further investigation of the selection information. Currently, there is only one fixed development cohort and one deterministic linear structure, and these development graphs have already been involved in prior diagnostic analyses. Any observed improvements require independent and repeated validation with newly registered data.', '',
        'Linear scoring, quality-aware scoring, and hard sample training have already been extensively studied in prior work. This round does not treat the scorer itself as a novel contribution. The standard scorer serves as a pre-registered control and is not rebranded post-hoc as an intervention method. Additional neighbor computation and ranking incur real costs; this round does not measure end-to-end GPU or board-level latency, nor does it claim such costs are negligible.', '',
        "All three candidates are fully retained. After the budget ends, the criteria will not be altered or the screening process continued simply due to user requests for 'pass'. The selection load has not been read, and the holdout set remains sealed. No cloud resources are available, and no new timing checks are scheduled.", '',
        'Evidence Directory: `artifacts/gate1/20261026_frozen_box_scorer_pilot_v1`. Primary documents include `contract.json`, `attempt_01/comparison_result.json`, `attempt_01/FP_budget_reference.json`, saved curves, three scorers, `saved_delivery_audit.json`, `machine_decision.json`, the launcher final state, and final delivery verification.']
    report=P.ROOT/'docs/71_gate1_frozen_box_scorer_results_and_false_positive_budget.md'
    with report.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    print(P.json.dumps({'status':decision['status'],'decisions':decisions,'intervention_added_value':added_value,'report':identity(report)},ensure_ascii=True))


if __name__=='__main__':main()
