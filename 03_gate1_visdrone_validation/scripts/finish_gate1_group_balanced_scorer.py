"""Report all fixed group/feature ablation outcomes and seal their evidence."""
from pathlib import Path
import json
import numpy as np
from audit_gate1_group_balanced_scorer import G,P,S,T,E,J,OUT,A
read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/'contract.json');r=read(A/'comparison_result.json');audit=read(OUT/'saved_delivery_audit.json')
    terminal=read(OUT/'launcher_terminal_receipt.json');process=read(OUT/'terminal_validation.json')
    require(audit['status']=='PASS3SCORER_GROUP_WEIGHT_FEATURE_CALIBRATION_AND_METRIC_AUDIT_ONLY','Saved audit incomplete')
    require(terminal['observed_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and terminal['stderr_bytes']==0 and process['matching_active_pilot_count']==0,'Abnormal worker terminal state')
    spec=read(OUT/'saved_audit_contract.json')
    require(identity(spec['script']['path'])==spec['script'] and identity(spec['score_replay_helper']['path'])==spec['score_replay_helper'],'Frozen audit source changed')
    primary=[arm for arm,s in r['summary'].items() if s['primary_new286_joint_observation']]
    allcohort=[arm for arm,s in r['summary'].items() if s['all3cohorts_joint_observation']]
    robust=[arm for arm,s in r['summary'].items() if s['all3cohorts_and_allprefixremovals_robust_observation']]
    extra=[arm for arm,s in r['summary'].items() if s['primary_extra_value_vs_old_generic_scorer']]
    oldthreshold=read(c['previous_threshold_result']['path']);qat={};qat_contrasts={}
    for cohort,entry in r['cohorts'].items():
        prior=oldthreshold['cohorts'][cohort]['branches']['ordinary_qat']
        folder=E.OUT/'attempt_01/evaluation'/cohort/'ordinary_qat'
        require(identity(folder/'official_IoU05_rank_curve.npz')==prior['cached_curve_source'],'Prior QAT curve changed')
        with np.load(folder/'official_IoU05_rank_curve.npz',allow_pickle=False) as d:counts=T.fixed_counts(d['score'],d['flag'],.25)
        require(counts==prior['official_IoU05_cap500_counts'],'Prior QAT endpoint differs')
        ap=read(folder/'metrics.json')['metrics_percent']['AP_all']
        require(ap==prior['AP_percent_unchanged'],'Prior QAT AP differs')
        qat[cohort]={'AP_percent':ap,'threshold':.25,'official_fixed_counts':counts,'reused_only':True}
        qat_contrasts[cohort]={arm:{'AP_delta':m['AP_percent']-ap,'TP_delta':m['official_fixed_counts']['TP']-counts['TP'],'FP_delta':m['official_fixed_counts']['FP']-counts['FP']} for arm,m in entry['arms'].items()}
    decision={'status':'COMPLETE3ARM_PREFIX_FEATURE_SCORER_DEVELOPMENT_ABLATION','at_utc':P.now(),'contract':identity(OUT/'contract.json'),
        'result':identity(A/'comparison_result.json'),'audit':identity(OUT/'saved_delivery_audit.json'),'finish_source':identity(__file__),
        'summary':r['summary'],'cohorts':r['cohorts'],'ordinary_QAT_reference':qat,'contrasts_vs_ordinary_QAT':qat_contrasts,'primary_rule_met_arms':primary,'all3cohort_rule_met_arms':allcohort,
        'all3cohorts_and_allprefixremovals_rule_met_arms':robust,'primary_extra_value_vs_old_generic_arms':extra,
        'all3newarms_retained':True,'oldgeneric_reference_retained':True,'new_scorer_fits':3,'updates_each':100,'new_detector_forwards':0,
        'calibration_members':178,'fit_members':512,'new_AP_jobs':9,'cumulative_scoring_candidates':6,
        'no_causal_claim_from_ablations':True,'no_novelty_claim_for_groupbalancing_or_feature_removal':True,
        'independent_test_set':False,'scientific_acceptance':False,'Gate1_overall_passed':False,'actual_INT8':False,
        'Selection_payload_access':False,'Holdout_payload_access':False,
        'next_action':'Fixed3arm ablation exhausted and sealed. Report all9cohortoutcomes and oldgeneric/ordinaryQAT contrasts. Do not select a favorablethreshold/group/feature combination after results or rename known groupweighting as novel. Further paper-method evidence needs a frozen distinct hypothesis and independent data/device scope; keep Holdout sealed unless explicitly opened.'}
    write(OUT/'machine_decision.json',decision)
    if robust:lead='Some registered candidates meet the joint observation criteria across all predefined ranges and leave-one-prefix groups, yet they still originate from repeatedly used development data and thus cannot be equated to independent scientific validation.'
    elif allcohort:lead='Some candidates meet the joint criteria in aggregated metrics across three ranges, but the leave-one-prefix group robustness condition is not fully satisfied—stability has not been proven.'
    elif primary:lead='Some candidates achieve the joint observation criteria on the primary 286 images, but no candidate satisfies the criteria across all three ranges and all leave-one-prefix groups; stability remains unresolved.'
    else:lead='All three new candidates fail to meet the predefined joint observation criteria on the primary 286 images; group balance and removal of source-related features have not resolved the issue in this round.'
    if not extra:lead+='Relative to the previous standard scorer, no group achieves simultaneously improved AP, unchanged detection, and no increase in false positives on the primary 286 images; this round fails to demonstrate additional value of the new method.'
    names={'prefix_balanced_full22':'group equal-weight plus full 22-dim','image_balanced_quality8':'image equal-weight plus relative quality 8-dim','prefix_balanced_quality8':'group equally weighted plus relative quality in 8 dimensions'}
    cohorts={'new286':'additional 286 images (primary)','new_prefix47':'new prefix 47 images (auxiliary)','full414':'merged 414 images (auxiliary)'}
    thresholds=read(A/'all3frozen_thresholds.json')
    lines=['# Gate1 Scorer Group Balance and Feature Ablation: Complete Results','','2026-10-26。','',f'**{lead}**','',
        'This round only includes the three pre-defined new scorers, all of which are retained. All data are from training_val repeatedly used in development and are not independent test sets; group balancing and feature selection are not considered original contributions.','',
        '## All Ranges and Fixed Threshold Endpoints','',
        '| Scope | Branch | AP50–95(%) | Fixed Threshold | Official TP | Official FP | TP Relative to R | FP Relative to R |','|---|---|---:|---:|---:|---:|---:|---:|']
    for cohort,entry in r['cohorts'].items():
        ref=entry['reference_R_official_counts'];base=entry['old_generic_scorer_official_counts']
        lines.append(f"| {cohorts[cohort]} | Original R | {entry['reference_R_AP']:.6f} | 0.250000000 | {ref['TP']} | {ref['FP']} | 0 | 0 |")
        lines.append(f"| {cohorts[cohort]} | Original Baseline Scorer | {entry['old_generic_scorer_AP']:.6f} | 0.174738524 | {base['TP']} | {base['FP']} | {base['TP']-ref['TP']:+d} | {base['FP']-ref['FP']:+d} |")
        q=qat[cohort];qc=q['official_fixed_counts']
        lines.append(f"| {cohorts[cohort]} | Baseline QAT (with existing cache) | {q['AP_percent']:.6f} | 0.250000000 | {qc['TP']} | {qc['FP']} | {qc['TP']-ref['TP']:+d} | {qc['FP']-ref['FP']:+d} |")
        for arm,m in entry['arms'].items():
            counts=m['official_fixed_counts'];lines.append(f"| {cohorts[cohort]} | {names[arm]} | {m['AP_percent']:.6f} | {m['threshold']:.9f} | {counts['TP']} | {counts['FP']} | {m['TP_delta_vs_R']:+d} | {m['FP_delta_vs_R']:+d} |")
    lines += ['', 'Official endpoint policy follows score-priority IoU0.5 matching, cross-category cap500, original submission order, and ignore rules. AP is computed using a fixed candidate pool; actual endpoints use unique scalar thresholds pre-frozen across 178 calibration images for each model. No threshold re-tuning is performed based on the development set.', '',
        '## Additional Value Over the Original Baseline Scorer','',
        '| Scope | New branch | AP Difference (in percentage points) | Fixed endpoint TP difference | Fixed endpoint FP difference |','|---|---|---:|---:|---:|']
    for cohort,entry in r['cohorts'].items():
        for arm,m in entry['arms'].items():lines.append(f"| {cohorts[cohort]} | {names[arm]} | {m['AP_delta_vs_old_generic_scorer']:+.6f} | {m['TP_delta_vs_old_generic_scorer']:+d} | {m['FP_delta_vs_old_generic_scorer']:+d} |")
    lines += ['', 'among the main 286 images, candidates that simultaneously satisfy AP improvement, unchanged endpoint detection, and no increase in false positives compared to the original baseline scorer:'+('、'.join(names[a] for a in extra) if extra else 'none')+'This is only a list of results from the registration criteria, not a post-hoc selection of the best model, nor does it establish causality or originality.', '',
        '## Comparison with Standard QAT Baseline','',
        '| Scope | New branch | AP Difference (in percentage points) | Fixed endpoint TP difference | Fixed endpoint FP difference |','|---|---|---:|---:|---:|']
    for cohort,arms in qat_contrasts.items():
        for arm,m in arms.items():lines.append(f"| {cohorts[cohort]} | {names[arm]} | {m['AP_delta']:+.6f} | {m['TP_delta']:+d} | {m['FP_delta']:+d} |")
    lines += ['', 'Standard QAT reuses previously saved results from the same development scope and the original 0.25 endpoint. This round verifies the curves, counts, and AP sources without retraining or re-scoring. The calibration methods for different branches are explicitly stated in the table; endpoint differences cannot be directly interpreted as equivalent false positive rates or equivalent computational advantages. Real INT8 runtime performance was not measured in this round.', '',
        '## All Joint Conditions and Group Sensitivity','',
        '| New branch | Main 286-image joint condition | Three-range joint condition | Three-range joint condition including all leave-one prefix groups |','|---|---|---|---|']
    for arm,s in r['summary'].items():lines.append(f"| {names[arm]} | {'Yes' if s['primary_new286_joint_observation'] else 'No'} | {'Yes' if s['all3cohorts_joint_observation'] else 'No'} | {'Yes' if s['all3cohorts_and_allprefixremovals_robust_observation'] else 'No'} |")
    lines += ['', 'The main joint condition requires AP to exceed R, at least R@0.25 in fixed threshold TP, and no more than R@0.25 in FP. Additional robustness checks are performed on the three predefined ranges and the count of each prefix group removal. All results are preserved and no favorable subset is selected. Prefix groups are not validated independent scenarios; leaving out one group is not an independent repetition or statistical confidence interval.','',
        '| Scope | New branch | Project hit change | Project false positive change | Number of joint improvements after leave-one prefix group removal |','|---|---|---:|---:|---:|']
    for cohort,entry in r['cohorts'].items():
        for arm,m in entry['arms'].items():
            s=m['prefix_sensitivity'];lines.append(f"| {cohorts[cohort]} | {names[arm]} | {s['pooled_TP_delta']:+d} | {s['pooled_FP_delta']:+d} | {s['leave_one_prefix_joint_count']}/{s['prefix_proxies']} |")
    lines += ['', '## Only two factors are changed','',
        'The existing image-equal-weight 22-dimensional model is reused directly; the three new models fill the other cells of the corresponding 2×2 ablation. Group-equal weighting gives all 58 training-prefix groups the same loss weight, followed by equal image weighting within each group and equal candidate weighting within each image. This prevents a prefix group with 50 images from naturally receiving 50 times the weight of a group with 1 image. Prefix identity is not provided to the inference model.', '',
        'The 8-dimensional quality model retains 4 score-quality quantities and 4 relative-neighbor quantities, removing absolute width/height, center location, and class one-hot features. Removed coefficients are exactly 0 in the common 22-dimensional slot, leaving 9 effective parameters; the complete 22-dimensional model has 23. This design tests source associations without assuming all source features are wrong or interpreting the three-arm differences as a complete causal explanation.', '',
        'Three models use native 512 images, 213716 cached candidate soft labels, without reading new images or training the detector. Normalization fits only the original training cache, with zero-residual initialization, fixed 100 CPU float64 Adam updates, lr0.03, betas0.9/0.999, eps1e-8, L2 coefficients 0.0005. No hard examples or intervention auxiliary losses are added; no best epoch, group weights, seeds, or regularization are selected.', '',
        '## The threshold is fixed once on the calibration dataset','',
        '| Branch | Unique threshold | Calibration TP relative to R | Calibration FP relative to R |','|---|---:|---:|---:|']
    for arm,t in thresholds.items():lines.append(f"| {names[arm]} | {t['threshold']:.17g} | {t['calibration_TP_delta']:+d} | {t['calibration_FP_delta']:+d} |")
    lines += ['', 'These 178 training images were not involved in the scorer fitting and have no image ID, SHA, or prefix proxy overlap with the 512 fitting images or the 414 development images; they remain part of the original detector training data and do not constitute an independent test set for the detector. Each group maintains the same official FP budget and is processed uniformly across the group. All three thresholds are frozen only after being applied to the development outputs; the calibration budget does not guarantee that the number of false positives on new images will not increase.','',
        '## Ranking diagnosis under the same false positive budget','',
        '| Scope | New branch | R reference FP budget | TP within new group budget | Actual FP |','|---|---|---:|---:|---:|']
    for cohort,entry in r['cohorts'].items():
        for arm,m in entry['arms'].items():
            p=m['at_R_FP_budget'];lines.append(f"| {cohorts[cohort]} | {names[arm]} | {entry['reference_R_FP_budget']} | {p['TP']} | {p['FP']} |")
    lines += ['', 'The table evaluates the ranking along the entire curve derived from development labels; the thresholds obtained from this curve are not deployable thresholds. It does not replace the previously fixed threshold endpoints and does not allow using it to mask failures of the fixed endpoints.','',
        '## Integrity and interpretability boundaries','',
        f"The actual exit code observed in the main workflow is 0, all three new evaluators completed 100 updates; all nine newly submitted AP tasks achieved exit0. 7 pre-checks were passed. Independent audit verification{audit['manifest_members_verified']}a number of new saved members and{audit['input_cache_bindings_verified']}existing cache bindings. Independently recompute per-image/group loss weights and normalization, verify removed coefficients are exactly 0, check the three saved states, the 178-image calibration matrix and unique threshold, all scoring/ranking/candidate-identity and filtering matrices for 414 images, nine sets of seven-metric aggregates, and the complete tied-score curve. Project matching is recomputed consistently for each arm on 32 fixed development probes.",'',
        'No model re-fitting, detector image inference, or full AP scoring is performed. The previous engineering optimizations for bit-level equivalence and CPU timing are retained. No additional board or CPU latency tests are conducted in this round. It is not valid to claim that the entire feature computation is free simply because the effective feature dimension is reduced.', '',
        'This design is influenced by prior diagnostic developments, and in conjunction with the previous 3 evaluators, it accumulates a new set of 6 candidates, all results are retained. The data is reused throughout development and is not considered a standalone scientific validation. Prior work on prefix balancing and group robust learning has already been established, such as in [Group DRO paper] (https://arxiv.org/abs/1911.08731); this round only implements group equal-weight ERM and does not claim to reproduce the complete algorithm or present it as a new contribution.', '',
        lead, '',
        "After budget exhaustion, no further attempts are made to pursue 'success' by selecting features, adjusting group weights, changing thresholds, or removing groups. If the current ablation is insufficient, the paper requires a new, distinguishable mechanism and independent validation, rather than merely renaming an ordinary scorer. Selection load is not read, holdout data is sealed, true INT8/Jetson and Gate1 are not passed; no new cloud resources or scheduled tasks are available.", '',
        'Evidence directory: `artifacts/gate1/20261026_group_balanced_scorer_v1`. All models, loss weights, thresholds, 9 groups of AP/curve/project matching, one group retained for record, independent audit, machine decisions, and final delivery verification are preserved.']
    report=P.ROOT/'docs/80_gate1_group_balancing_and_feature_ablation_results.md'
    with report.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    latest=P.ROOT/'artifacts/gate1/latest_execution_state.json';previous=read(latest)
    require(previous['snapshot']=='artifacts/gate1/current_execution_state_v43.json','Latest pointer advanced elsewhere')
    require(identity(P.ROOT/previous['snapshot'])['sha256']==previous['snapshot_sha256'],'Previous snapshot changed')
    expected={}
    def add(b):
        if b['path'] in expected:require(expected[b['path']]==b,'Conflicting file identity')
        expected[b['path']]=b
    for b in read(A/'manifest.json')['files']+c['fit_files']+c['reused_files']+list(c['source_bindings'].values()):add(b)
    for key in ('previous_robustness_contract','previous_robustness_delivery','previous_threshold_contract','previous_threshold_delivery','previous_threshold_result','old_score_model','evaluation_protocol'):add(c[key])
    for row in c['calibration_caches']:
        for key in ('features','predictions','matrices'):add(row[key])
    for row in c['development_features']:add(row['features'])
    root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
    for meta in c['calibration_members']+c['cohorts']['full414']:
        path=(root/meta['annotation_path']).resolve(strict=True);require(path.is_relative_to(root),'Annotation path escaped')
        b=identity(path);require(b['sha256']==meta['annotation_sha256'],'Allowed annotation changed');add(b)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():add(identity(path))
    for path in (P.ROOT/previous['snapshot'],P.ROOT/'docs/79_gate1_group_balancing_and_feature_ablation_launch.md',report,
        P.ROOT/'scripts/audit_gate1_group_balanced_scorer.py',P.ROOT/'scripts/audit_gate1_frozen_box_scorer.py',Path(__file__)):add(identity(path))
    for b in expected.values():require(identity(b['path'])==b,'Delivery binding differs')
    write(OUT/'delivery_manifest.json',{'schema':'rampq.group_feature_scorer.delivery/v1','at_utc':P.now(),'files':[expected[k] for k in sorted(expected)],'mutable_latest_pointer_excluded':True,'all3arms_and_all3cohorts_retained':True})
    validation={'status':'PASS3SCORER_ABLATION_DELIVERY_INTEGRITY_ONLY','at_utc':P.now(),'manifest':identity(OUT/'delivery_manifest.json'),'members_verified':len(expected),
        'missing':0,'mismatched':0,'audit':identity(OUT/'saved_delivery_audit.json'),'machine_decision':identity(OUT/'machine_decision.json'),'report':identity(report),
        'observed_worker_exit_code':0,'primary_rule_met_arms':primary,'all_robust_rule_met_arms':robust,'scientific_acceptance':False,'Gate1_overall_passed':False,'actual_INT8':False,'Holdout_accessed':False}
    write(OUT/'delivery_validation.json',validation)
    snap=P.ROOT/'artifacts/gate1/current_execution_state_v44.json'
    write(snap,{'schema':'rampq.gate1.execution_state/v44','status':'COMPLETE3ARM_PREFIX_FEATURE_SCORER_DEVELOPMENT_ABLATION','at_utc':P.now(),
        'previous_snapshot':previous,'current_experiment':'artifacts/gate1/20261026_group_balanced_scorer_v1','report':report.name,
        'delivery_validation':identity(OUT/'delivery_validation.json'),'machine_decision':identity(OUT/'machine_decision.json'),
        'all3arms_retained':True,'primary_rule_met_arms':primary,'all3cohort_rule_met_arms':allcohort,'all3cohorts_and_allprefixremovals_rule_met_arms':robust,
        'primary_extra_value_vs_old_generic_arms':extra,'new_scorer_fits':3,'updates_each':100,'new_detector_forwards':0,'AP_jobs':9,
        'cumulative_scoring_candidates':6,'independent_contribution_demonstrated':False,'scientific_acceptance':False,'Gate1_overall_passed':False,'actual_INT8':False,
        'Selection_payload_access':False,'Holdout_payload_access':False,'old_failures_preserved':True,'no_active_worker':True,'next_action':decision['next_action']})
    pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v44.json','snapshot_sha256':identity(snap)['sha256'],
        'delivery_validation':identity(OUT/'delivery_validation.json'),'note':'Registered3arm prefixweight/quality8scorer ablation complete/audited. Allmodels/thresholds/9cohortjobs retained. Read report80 for actual primary/robustness/control results;not independent/Gate1/INT8 acceptance,Holdout sealed.'}
    tmp=latest.with_name('latest_execution_state.v44.pending.json');write(tmp,pointer);tmp.replace(latest)
    print(json.dumps({'status':validation['status'],'members_verified':len(expected),'primary_rule_met_arms':primary,'all3cohort_rule_met_arms':allcohort,
        'robust_rule_met_arms':robust,'primary_extra_value_arms':extra,'latest':pointer['snapshot'],'report':identity(report)},ensure_ascii=True))


if __name__=='__main__':main()
