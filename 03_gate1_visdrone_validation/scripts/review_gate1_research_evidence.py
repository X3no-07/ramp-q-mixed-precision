"""Read-only research-route review; no data payload, fitting or new scoring."""
from pathlib import Path
from collections import defaultdict,Counter
import json
import statistics
import gate1_visual_evidence_probe as V
P=V.P;ROOT=V.ROOT
OUT=ROOT/'artifacts/gate1/20261026_research_evidence_review_v1'
read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    latest=ROOT/'artifacts/gate1/latest_execution_state.json';prior=read(latest)
    require(prior['snapshot']=='artifacts/gate1/current_execution_state_v46.json' and identity(ROOT/prior['snapshot'])['sha256']==prior['snapshot_sha256'],'Unexpected prior state')
    OUT.mkdir(exist_ok=False)
    inputs={'formal_contract':ROOT/'artifacts/gate1/20261023_formal_ignore_training_v1/formal_contract.json',
        'formal_events':ROOT/'artifacts/gate1/20261023_formal_ignore_training_v1/formal_attempt_01/events.jsonl',
        'formal_completion':ROOT/'artifacts/gate1/20261023_formal_ignore_training_v1/formal_attempt_01/completion_receipt.json',
        'QAT_pilot_contract':P.OUT/'contract.json','extension_comparison':V.E.OUT/'attempt_01/comparison_result.json',
        'formal_FP32_metrics':ROOT/'artifacts/gate1/20261024_formal_candidate_evaluation_v1/scores/fp32/metrics.json',
        'group_scorer_decision':ROOT/'artifacts/gate1/20261026_group_balanced_scorer_v1/machine_decision.json',
        'group_scorer_delivery':ROOT/'artifacts/gate1/20261026_group_balanced_scorer_v1/delivery_validation.json',
        'visual_probe_comparison':V.OUT/'attempt_01/comparison_result.json','visual_probe_delivery':V.OUT/'delivery_validation.json',
        'robustness_report':ROOT/'docs/77_gate1_scorer_robustness_and_cpu_cost_results.md'}
    bindings={k:identity(p) for k,p in inputs.items()}
    contract={'schema':'rampq.gate1.research_evidence_review/v1','at_utc':P.now(),'source':identity(__file__),'inputs':bindings,
        'previous_snapshot':prior,'new_model_fits':0,'new_image_forwards':0,'new_AP_jobs':0,'new_thresholds':0,
        'scope':'Aggregate immutable training scalar logs and existing final metrics; distinguish baseline adequacy,method effect and deployment evidence. No new image/annotation/prediction payload or method search.',
        'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False}
    write(OUT/'contract.json',contract)
    by=defaultdict(list);events=Counter()
    for line in inputs['formal_events'].open(encoding='utf-8-sig'):
        r=json.loads(line);events[r['event']]+=1
        if r['event']=='training_batch':by[r['epoch']].append(r['loss'])
    rows=[{'epoch':k,'batches':len(v),'mean_logged_batch_loss':statistics.fmean(v),'median_logged_batch_loss':statistics.median(v)} for k,v in sorted(by.items())]
    require([r['epoch'] for r in rows]==list(range(1,31)) and all(r['batches']==1133 for r in rows) and events['training_batch']==33990,'Training log coverage differs')
    prev5=statistics.fmean(r['mean_logged_batch_loss'] for r in rows[20:25]);last5=statistics.fmean(r['mean_logged_batch_loss'] for r in rows[25:30]);drop=100*(last5/prev5-1)
    pilot=read(inputs['QAT_pilot_contract']);require(pilot['train_count']==512 and pilot['epochs']==3 and pilot['optimizer_steps_per_arm']==96,'Pilot baseline budget differs')
    comparison=read(inputs['extension_comparison']);full=comparison['cohorts']['full414']['branches']
    aps={branch:full[branch]['official']['metrics_percent']['AP_all'] for branch in ('mse_r','frozen_score_base','ordinary_qat')}
    fp32=read(inputs['formal_FP32_metrics'])['metrics_percent']['AP_all'];aps['formal_fp32']=fp32
    visual=read(inputs['visual_probe_comparison']);groups=read(inputs['group_scorer_decision'])
    result={'status':'COMPLETE_RESEARCH_EVIDENCE_REVIEW_NO_NEW_EXPERIMENTS','formal_training_epoch_log':rows,'event_counts':dict(events),
        'epoch21to25_mean_logged_loss':prev5,'epoch26to30_mean_logged_loss':last5,'relative_change_percent':drop,
        'training_loss_not_validation_AP':True,'FP32_convergence_proven':False,'undertraining_as_cause_proven':False,
        'ordinary_QAT_pilot_budget':{'images':512,'epochs':3,'optimizer_updates':96,'single_seed_development_pilot':True},
        'fully_trained_strong_QAT_comparison_proven':False,'existing_full414_AP50_95_percent':aps,
        'oldgeneric_AP_delta_vs_R':aps['frozen_score_base']-aps['mse_r'],
        'oldgeneric_AP_delta_vs_ordinaryQAT':aps['frozen_score_base']-aps['ordinary_qat'],
        'oldgeneric_AP_gap_to_FP32':aps['frozen_score_base']-fp32,
        'three_new_scalar_arms_robust_passes':groups['all3cohorts_and_allprefixremovals_rule_met_arms'],
        'visual_probe_diagnostic_promising':visual['diagnostic_promising'],'visual_probe_TP_delta_vs_old':visual['TP_delta_vs_old_generic_at_budget'],
        'recommendation':'Stop automatic scorer/module accumulation. Next priority is a separately frozen adequate ordinaryQAT and established task-aware quantization baseline on the actual intended backend,with identical data/compute accounting. Preserve all failed variants. Retain the original mechanism as a paper contribution only if it adds stable independently validated value over those controls.',
        'new_model_fits':0,'new_image_forwards':0,'new_AP_jobs':0,'new_candidate_selection':False,
        'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False}
    write(OUT/'review_result.json',result)
    report=ROOT/'docs/83_gate1_research_review_and_baseline_first_plan.md'
    text=f'''# Gate1 Research Pathway Review and Baseline Priority Plan

2026-10-26.

**We recommend pausing automatic scorer addition and recovery modules, and instead allocating the next computational budget to strengthen trustworthy baseline models. Only after establishing a solid baseline should we decide whether the original mechanism warrants retention as the paper\'s main contribution. The current results are insufficient to support a high-quality paper contribution and do not justify the claim that the entire quantization direction is fundamentally infeasible.**

## Newly Identified Evidence Gaps

Formal FP32 training did complete 30 epochs and 33990 batches; the training-completion and saved-state-consistency conclusions are unchanged. The original per-batch scalar loss averages over epochs 21–25 are{prev5:.6f}, and over epochs 26–30 are{last5:.6f}, relative change{drop:+.3f}%; log-recorded AP validation is delayed and not used for convergence judgment in this round.

The training loss continues to decrease, so it cannot be claimed that failure is necessarily due to non-convergence. Similarly, it cannot be claimed that the baseline has been sufficiently trained. The batch-wise averages include the original final batch size and existing loss scaling, and are only described within the same training log. They cannot be converted into AP or ranked against external models.

The current ordinary QAT comparison only covers 512 images, 3 epochs, 96 optimizer updates, and a single training seed. It is sufficient as a control for a pilot experiment under the same budget, but it does not justify claiming that ordinary QAT or current task-aware quantization methods have been sufficiently trained. The past three correction arms and this pilot were all under the same budget and are not re-evaluated as successful or invalid by this review.

## Existing Full 414-Image AP Gap

| Existing branches | AP50–95(%) |
|---|---:|
| Official FP32 | {fp32:.9f} |
| MSE Range R | {aps['mse_r']:.9f} |
| Standard QAT Pilot | {aps['ordinary_qat']:.9f} |
| Original Baseline Scorer | {aps['frozen_score_base']:.9f} |

The old scorer improves relative to R by{result['oldgeneric_AP_delta_vs_R']:+.9f}percentage points, improving relative to the standard QAT pilot by{result['oldgeneric_AP_delta_vs_ordinaryQAT']:+.9f}percentage points, still trailing FP32 by{result['oldgeneric_AP_gap_to_FP32']:+.9f}percentage points. Together with report 77, the evidence shows that the aggregate false-positive reduction also depends on a few prefix groups; a very small aggregate AP difference cannot establish a stable method advantage.

The three new groups/feature candidates in Report 80 do not satisfy all range and leave-a-group conditions. One local visual candidate in Report 82 on calibration training tiling 178 retains 56 fewer true positives (TPs) compared to the previous scorer under the FP budget, although BCE slightly decreases, and the joint criterion is not met. A total of 7 scorer candidates are retained; these results do not support continuing with intuitive module addition and acceptance.

## Proceed in this order next

1. **First, fix a trustworthy baseline protocol.** Agree in a separate new directory on sufficient training budget, control implementation, optimization settings, data roles, and actual target backend; retain the original 30-round candidates, without covering continuation training or rewriting contracts. Complete the qualification and training sufficiency evidence for conventional QAT and published task-aware quantization baselines. Do not treat the current unmerged Conv floating-point QDQ as a genuine INT8 backend, nor assume that adding more training rounds will necessarily improve AP.
2. **Then, evaluate whether the original mechanism provides additional value.** Only compare the original precision allocation or mitigation strategy with conventional and task-aware baselines when data, training, computational budget, and post-processing criteria are comparable. Require prior evidence for AP, fixed endpoint detection and false positive rates, and additional overheads; do not select the best threshold, prefix, seed, or module combination from development results. Independent validation requires clearly open, independent data ranges; holdout sets remain sealed.
3. **Decide retention or replacement based on results.** If a strong baseline already covers current gains, the original method becomes a control and negative result, and its main contribution is withdrawn; if stable, reproducible, and practically valuable quantization degradation remains, then formulate a new, distinguishable research hypothesis around the identified mechanism. The new hypothesis must be explicitly differentiated from prior work and cannot treat local validation, standard distillation, or task consistency as novel.

## Constraints from prior work

[IoU-Net](https://arxiv.org/abs/1807.11590) has learned localization quality, [GFLV2](https://arxiv.org/abs/2011.12885) studies lightweight quality ranking; standard quality scoring is not a unique contribution of this project. [HQOD](https://arxiv.org/abs/2408.02561) investigates inconsistency between classification and localization quality under quantization and proposes task-related losses and IoU losses; the claim that \'quantization causes classification-localization inconsistency\' is not a new discovery. [DetPTQ](https://arxiv.org/abs/2304.09785) has already used detection task losses to guide layer-wise quantization parameter selection. Their published results cannot be directly considered as performance outcomes of the local YOLO11n implementation; applicability and implementation validation are required in the next phase.

## Actual Completion and Unexecuted Actions in This Round

This round only reads 33,990 existing training batch records and aggregated results, with no new training, image inference, AP, candidates, or thresholds. Loss statistics, source identities, and machine conclusions from the 30 rounds are preserved. Full training of the new baseline, local reproduction of HQOD/DetPTQ, genuine INT8 implementation, and new independent scientific validation have not been executed. This report does not constitute a commitment to publication in any journal and does not imply that the paper has been saved or recovered.

Combined with Report No. 82, the most clear action at present is to strengthen the experimental basis before deciding on the method, rather than continuing to search for success stories around the 7 scorers.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    for k,b in bindings.items():require(identity(inputs[k])==b,'Read-only source changed')
    files=[*bindings.values(),identity(__file__),identity(ROOT/prior['snapshot']),identity(OUT/'contract.json'),identity(OUT/'review_result.json'),identity(report)]
    write(OUT/'delivery_manifest.json',{'files':files,'mutable_latest_excluded':True})
    for b in files:require(identity(b['path'])==b,'Review delivery identity changed')
    validation={'status':'PASS_RESEARCH_REVIEW_SOURCE_AND_AGGREGATE_CHECK_ONLY','at_utc':P.now(),'manifest':identity(OUT/'delivery_manifest.json'),
        'members_verified':len(files),'missing':0,'mismatched':0,'report':identity(report),'training_batches_recomputed':33990,'epochs':30,
        'new_model_fits':0,'new_image_forwards':0,'new_AP_jobs':0,'scientific_acceptance':False,'actual_INT8':False}
    write(OUT/'delivery_validation.json',validation)
    snap=ROOT/'artifacts/gate1/current_execution_state_v47.json'
    write(snap,{'schema':'rampq.gate1.execution_state/v47','status':'COMPLETE_EVIDENCE_REVIEW_BASELINE_ADEQUACY_NEXT','at_utc':P.now(),'previous_snapshot':prior,
        'report':identity(report),'delivery_validation':identity(OUT/'delivery_validation.json'),'review_result':identity(OUT/'review_result.json'),
        'last_experiment_delivery':identity(V.OUT/'delivery_validation.json'),'cumulative_scorer_candidates':7,'automatic_scorer_search_stopped':True,
        'formal_training_complete':True,'FP32_convergence_proven':False,'fully_trained_strong_QAT_baseline':False,
        'new_baseline_training_started':False,'scientific_acceptance':False,'actual_INT8':False,'Gate1_overall_passed':False,
        'Holdout_payload_access':False,'no_active_worker':True,'next_action':result['recommendation']})
    P.FT.atomic_json(latest,{'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v47.json','snapshot_sha256':identity(snap)['sha256'],
        'delivery_validation':identity(OUT/'delivery_validation.json'),'note':'One visual probe negative and sealed(report82);read-only33000+batch review complete(report83). Stop automatic scorer accumulation;next priority is adequate ordinary/task-aware quantization baselines with real backend evidence. No new baseline training started;Holdout sealed.'})
    print(json.dumps({'status':validation['status'],'members_verified':len(files),'last5_loss_change_percent':drop,'ordinaryQAT_updates':96,'AP_delta_oldscore_vs_QAT':result['oldgeneric_AP_delta_vs_ordinaryQAT'],'snapshot':'v47','report':identity(report)},ensure_ascii=True))


if __name__=='__main__':main()
