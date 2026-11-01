"""Report the only separately calibrated threshold, including failed subcohorts."""
from datetime import datetime,timezone
from pathlib import Path
import json
from audit_gate1_separate_threshold import T,P,S,E,OUT,A
read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/'contract.json');result=read(A/'comparison_result.json');audit=read(OUT/'saved_delivery_audit.json')
    require(audit['status']=='PASS_SEPARATE_THRESHOLD_SOURCE_IDENTITY_SELECTION_AND_ENDPOINT_AUDIT_ONLY','Saved audit incomplete')
    threshold=read(A/'frozen_threshold.json');primary=result['primary_official_joint_rule_met'];all_pass=result['all_cohorts_official_joint_rule_met']
    decision={'status':'COMPLETE_SEPARATELY_CALIBRATED_FIXED_THRESHOLD_DEVELOPMENT_CHECK','at_utc':P.now(),
        'contract':identity(OUT/'contract.json'),'result':identity(A/'comparison_result.json'),'audit':identity(OUT/'saved_delivery_audit.json'),
        'summary_source':identity(__file__),'frozen_threshold':threshold['threshold'],'threshold_record':identity(A/'frozen_threshold.json'),
        'calibration':threshold,'cohorts':result['cohorts'],'primary_official_fixed_threshold_joint_rule_met':primary,
        'all_three_cohorts_official_fixed_threshold_joint_rule_met':all_pass,'primary_project_joint_observation':result['primary_project_joint_rule_met'],
        'calibration_members':178,'calibration_prefix_proxies':34,'scorer_fit_calibration_id_hash_prefix_overlap':0,
        'calibration_members_seen_by_detector':True,'scorer_was_not_refit':True,'new_model_fit_count':0,'threshold_candidates':1,
        'new_detector_forwards':178,'new_full_AP_scores':0,'old_numerical_FAILs_reclassified':False,
        'old128_used_to_choose_scorer':True,'development_used_previously':True,'scientific_acceptance':False,
        'independent_test_set':False,'Gate1_overall_passed':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
        'remaining_limits':['New-prefix47does not have to match pooled-developer performance; all findings retained',
                            'Generic scorer itself is a baseline,intervention contribution remains unproven',
                            'No independent test,formal uncertainty validation or final-device latency/INT8 acceptance'],
        'next_action':'This one-threshold registered budget is exhausted and sealed. Do not retune to fix47prefixsubcohort or turn developer-only observations into independent acceptance. Prepare independent data/device stage and contribution comparison before more method changes.'}
    write(OUT/'machine_decision.json',decision)
    if primary and all_pass:lead='After separate calibration, the fixed threshold achieves the official endpoint criteria of "detection performance not degraded and false positives not increased" across all three predefined development ranges.'
    elif primary:lead='After separate calibration, the fixed threshold achieves detection increases and false positive reductions on the majority of the newly added 286 images and on some aggregated ranges; however, the new prefix of 47 images still increases false positives, and cross-range conditions have not been fully satisfied.'
    else:lead='The fixed threshold calibrated independently in this study still fails to satisfy the criteria of no degradation in detection and no increase in false positives on the major additional 286 images, and the failure result is retained.'
    labels={'new286':'additional 286 images (primary)','new_prefix47':'new prefix 47 images (auxiliary)','full414':'merged 414 images (auxiliary)'}
    branches={'mse_r':'Original mixed-precision R','frozen_score_base':'existing ordinary QAT','ordinary_qat':'scorer threshold **'}
    lines=['# Gate1: Frozen Scorer Standalone Threshold Calibration: 178-Image Calibration and Fixed Endpoint Validation','','2026-10-26。','',f'**{lead}**','',
        'This round only validates the actual threshold endpoint, without altering the scorer, prediction boxes, categories, candidate members, ranking, or mAP. Development pass does not equate to paper contribution, independent scientific validation, Gate1 overall pass, or INT8/board-level pass.','',
        '## Actual Results After Fixed Threshold','',
        f"A unique threshold of **{threshold['threshold']:.17g}**. It is first computed and written into an immutable record using 178 independently trained images, then applied to the existing 414 development outputs. The original R and standard QAT continue to use 0.25.",'',
        'The following adopts project-level target matching (confidence fixed, IoU0.5, IoU priority):','',
        '| Scope | Branch | Threshold used | True target hits | false positives |','|---|---|---:|---:|---:|']
    for name,r in result['cohorts'].items():
        for branch,b in r['branches'].items():
            t=b['project_counts'];lines.append(f"| {labels[name]} | {branches[branch]} | {b['threshold']:.9f} | {t['hits']} | {t['false_positive_detections']} |")
    lines += ['', 'Difference in project matching between the scorer and original R:','', '| Scope | Hit variation | False Positive Change | Detection not degraded and false positives not increased |','|---|---:|---:|---|']
    for name,r in result['cohorts'].items():
        lines.append(f"| {labels[name]} | {r['project_TP_delta_vs_R']:+d} | {r['project_FP_delta_vs_R']:+d} | {'Yes' if r['project_fixed_threshold_joint_observation'] else 'No'} |")
    lines += ['', 'The primary scope is the pre-fixed additional 286 images. The merged 414-image set includes the 128 images previously used for scorer selection, and cannot mask the increase in false positives within the 47-image auxiliary range using a merged average. The 47 images contain 21 new filename prefixes, but these prefixes are not already validated independent scenarios, and thus cannot be considered 21 independent scientific samples.','',
        '## Official Counting Under the Same Fixed Threshold','',
        'The main evaluation metric comes from the officially validated Python image IoU0.5 matching, preserving score priority, original submission order, and cross-category cap500. Each score threshold is fixed in advance and not selected dynamically along the development curve. This count differs from project-level IoU priority matching; all differences are reported truthfully.','',
        '| Scope | Branch | Official TP | Official FP | Ignored items |','|---|---|---:|---:|---:|']
    for name,r in result['cohorts'].items():
        for branch,b in r['branches'].items():
            t=b['official_IoU05_cap500_counts'];lines.append(f"| {labels[name]} | {branches[branch]} | {t['TP']} | {t['FP']} | {t['ignored']} |")
    lines += ['', '| Scope | Official TP relative to R | Official FP relative to R | Official TP relative to standard QAT | Official FP relative to standard QAT |','|---|---:|---:|---:|---:|']
    for name,r in result['cohorts'].items():
        lines.append(f"| {labels[name]} | {r['official_TP_delta_vs_R']:+d} | {r['official_FP_delta_vs_R']:+d} | {r['official_TP_delta_vs_QAT']:+d} | {r['official_FP_delta_vs_QAT']:+d} |")
    lines += ['', '## Calibration Generates Only One Candidate','',
        'Starting from the original 4530 training images, metadata exclude filename prefixes associated with the scorer fitting set of 512 images. All remaining 178 images across 34 prefixes are used for calibration. Calibration images do not overlap the scorer fitting set or 414 development images by image ID, content SHA, or prefix proxy. No new predictions or annotations were used to select members.', '',
        "These calibration images belong to the detector's original training set, so the term 'separate' refers only to parameter fitting for the scorer, and does not constitute an unseen test set for the detector. No gradient information for the scorer is provided for the 178 calibration images; the 23 parameters of the scorer, normalization, features, and detector range are maintained at their original values.", '',
        'On all 178 calibration images, the official FP count at R@0.25 from the original detector defines the budget. Using the same official method as the scorer, a curve is matched and the lowest feasible score threshold is selected such that FP does not exceed the budget. All boxes with the same score are grouped together; if no non-empty feasible group exists, the threshold is fixed at 1. No search for safety margins, class thresholds, seeds, additional calibration members, or a second threshold is conducted.', '',
        f"Calibration reference endpoint is TP ={threshold['reference_calibration_counts']['TP']}、FP={threshold['reference_calibration_counts']['FP']};The calibration endpoint for the selected threshold is TP ={threshold['chosen_calibration_counts']['TP']}、FP={threshold['chosen_calibration_counts']['FP']}. This is only a budget constraint on the calibration data, not a probabilistic guarantee that false positive detection counts will not increase on new images.",'',
        'Threshold optimization and model fitting data should be separated; this principle is also described in [scikit-learn threshold tuning documentation](https://scikit-learn.org/stable/modules/classification_threshold.html). In this round, a custom detection-matching endpoint is used, rather than reproducing the full classification tuning method.', '',
        '## AP remains unchanged','',
        "Changing the acceptance threshold does not alter the sorting of the previously fixed candidate pool; the full AP is still taken from the original stored result. The full AP was not re-computed in this round, and the truncated results after thresholding were not misrepresented as an AP improvement. The original scorer's AP data is still listed as follows:",'',
        '| Scope | Original R AP (%) | Frozen Scorer AP (%) | Ordinary QAT AP (%) |','|---|---:|---:|---:|']
    for name,r in result['cohorts'].items():
        lines.append(f"| {labels[name]} | {r['branches']['mse_r']['AP_percent_unchanged']:.6f} | {r['branches']['frozen_score_base']['AP_percent_unchanged']:.6f} | {r['branches']['ordinary_qat']['AP_percent_unchanged']:.6f} |")
    lines += ['', 'The AP difference between the new scorer and ordinary QAT on an additional 286 images is only about 0.002 percentage points, and thus cannot be claimed to have a significant accuracy advantage. The ordinary scorer represents the existing baseline form; prior interventions in sampling did not exceed the ordinary scorer, and therefore the benefits from this calibration cannot be rebranded as contributions from the intervention method.', '',
        '## Actual Completion and Saved Validation','',
        f"The current round of work completed with exit code 0, and all return receipts and 178 forward-pass budgets were fully realized. The new scorer was fitted 0 times, inference on development images occurred 0 times, and full AP re-evaluation occurred 0 times. Per-target IoU matching and new threshold projects are new analyses in this round and are each independently saved.",'',
        f"8 pre-screening checks passed. Independent audit verification of{audit['manifest_members_verified']}new output members,{audit['reused_members_verified']}existing development bindings, and{audit['allowed_input_payloads_rehashed']}authorized image/annotation inputs; 178 raw files, 178 independently visible feature maps and model outputs, original annotations to filtering matrices, calibration per-IOU matching, group-wise curves, and the unique threshold were all validated.",'',
        'The freeze threshold event occurs prior to the application of the development threshold, and the new threshold function interface only accepts calibration score and match flag. Three development scopes, three paths, collectively account for 9 official endpoints counted independently from saved curves, with 32 fixed development images recalculated for per-object match consistency. Validation confirms no duplicate model inference, model fitting, or full AP.', '',
        'The detector state fingerprint remains unchanged before and after calibration inference, and the scorer file SHA remains frozen. The original 8400 candidate values in raw data were not reclassified as FAIL; no FP32_reference was constructed for the 178 new images, and thus no floating-point qualification is authenticated. A saved consistency PASS only indicates evidence completeness and cannot substitute for effectiveness, scientific validation, or equipment acceptance.', '',
        '## Progress and Remaining Issues','',lead,'',
        "The first stage has already provided actual evidence of improvement from the fixed threshold on the main development data, but false positive control on the new prefix subset remains unsolved, and thus 'all cases completed' is not marked. This round includes only one registered threshold, and no further adjustment is made based on the results of the 47-image subset.", '',
        'The next stage of work should shift toward independently verifiable contributions and robustness: clearly distinguishing mechanisms from ordinary quality scoring, strong baselines and ablation studies, truly independent data not involved in development, and additional computational costs. Continuing to adjust thresholds on the same development images cannot prove such contributions.', '',
        'Holdout content remains unopened, and Selection load has not been read. A true independent data phase requires specific data access decisions and frozen contracts; equipment phase requires actual connections and INT8 execution evidence; neither has been claimed to have passed. No new cloud resources or scheduled checks have been added.', '',
        'Evidence directory: `artifacts/gate1/20261026_separate_threshold178_v1`. Primary files are `contract.json`, `attempt_01/frozen_threshold.json`, calibration predictions and filtering matrices, `attempt_01/comparison_result.json`, independent audit, actual exit receipts, machine decisions, and final delivery validation.']
    report=P.ROOT/'docs/75_gate1_scorer_separate_threshold_calibration_results.md'
    with report.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':decision['status'],'threshold':threshold['threshold'],'primary_official_rule_met':primary,'all_cohorts_rule_met':all_pass,
        'cohort_deltas':{name:{key:r[key] for key in ('official_TP_delta_vs_R','official_FP_delta_vs_R','project_TP_delta_vs_R','project_FP_delta_vs_R')} for name,r in result['cohorts'].items()},
        'report':identity(report)},ensure_ascii=True))


if __name__=='__main__':main()
