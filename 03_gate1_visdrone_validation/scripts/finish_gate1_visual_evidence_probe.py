"""Report one finite information probe and seal all controls, including negatives."""
from pathlib import Path
import json
import gate1_visual_evidence_probe as V
P=V.P;OUT=V.OUT;ROOT=V.ROOT;A=OUT/'attempt_01'
read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/'contract.json');r=read(A/'comparison_result.json');audit=read(OUT/'saved_delivery_audit.json')
    terminal=read(OUT/'launcher_terminal_receipt.json');process=read(OUT/'terminal_validation.json');ae=read(OUT/'audit_exit_receipt.json')
    require(audit['status']=='PASS_LOCAL_VISUAL_SAVED_INFORMATION_DIAGNOSTIC_AUDIT_ONLY' and ae['exit_code']==0,'Independent saved audit incomplete')
    require(terminal['observed_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and terminal['stderr_bytes']==0 and process['matching_active_probe_count']==0,'Abnormal process terminal state')
    latest=ROOT/'artifacts/gate1/latest_execution_state.json';prior=read(latest)
    require(prior['snapshot']=='artifacts/gate1/current_execution_state_v45.json' and identity(ROOT/prior['snapshot'])['sha256']==prior['snapshot_sha256'],'Latest state moved/changed')
    positive=r['diagnostic_promising'];ref=r['reference_R_at025'];old=r['old_generic_at_R_FP_budget'];new=r['local_visual_at_R_FP_budget'];loss=r['image_balanced_quality_BCE']
    lead='Local visual information provided additional quality ranking signals on training calibration images with non-overlapping prefix segments prior to scorer fitting, satisfying the two pre-defined diagnostic conditions in this round.' if positive else "This fixed local visual linear probing does not simultaneously satisfy the diagnostic condition of 'outperforming the previous scorer under the same false positive budget and achieving lower quality BCE'. Therefore, this candidate cannot be claimed to have improved based on this result."
    decision={'status':'COMPLETE_ONE_LOCAL_VISUAL_INFORMATION_PROBE','at_utc':P.now(),'diagnostic_promising':positive,
        'contract':identity(OUT/'contract.json'),'comparison':identity(A/'comparison_result.json'),'audit':identity(OUT/'saved_delivery_audit.json'),
        'finish_source':identity(__file__),'calibration_reference':ref,'old_generic_at_R_FP_budget':old,'local_visual_at_R_FP_budget':new,
        'TP_delta_vs_old_generic_at_budget':r['TP_delta_vs_old_generic_at_budget'],'image_balanced_quality_BCE':loss,
        'BCE_delta_vs_old_generic':r['BCE_delta_vs_old_generic'],'new_candidates':1,'cumulative_scorer_candidates':7,
        'new_image_forwards':690,'new_detector_training_updates':0,'new_scorer_updates':100,'new_full_AP_jobs':0,
        'development_evaluated':False,'new_deployment_thresholds':0,'no_novelty_claim':True,'scientific_acceptance':False,
        'actual_INT8':False,'Gate1_overall_passed':False,'training_val_payload_access':False,'Selection_payload_access':False,'Holdout_payload_access':False,
        'next_action':('A positive information diagnostic is not a new method or generalization result. Freeze a separate same-candidate development/calibration protocol and include an image-independent control before any broader inference/scoring; no new feature/candidate search. A paper still needs a distinct mechanism and genuinely independent validation.' if positive else 'Registered information probe exhausted;preserve negative. Do not automatically add nonlinear heads,feature variants,seeds or new thresholds to search for a pass. Reconsider the distinct research mechanism and independent validation scope;all previousresults remain valid historical evidence.')}
    write(OUT/'machine_decision.json',decision)
    report=ROOT/'docs/82_gate1_local_visual_information_feasibility_results.md'
    text=f'''# Gate1 Local Visual Information Feasibility Diagnosis Results

2026-10-26.

**{lead}**

This validation addresses whether adding local visual features from the model can supplement information beyond the original score and bounding box statistics. Even if the results are favorable, it does not directly answer whether the 414 development images, an independent test set, originality, or real INT8 performance have been improved.

## Results on Fixed 178 Images

| Branch | Same R False Positive Budget TP | Actual FP | Per-Image Weighted Soft Quality BCE |
|---|---:|---:|---:|
| Official R@0.25 | {ref['TP']} | {ref['FP']} | {loss['original_R']:.9f} |
| Original 22-Dimensional Standard Scorer | {old['TP']} | {old['FP']} | {loss['old_generic']:.9f} |
| New 150-Dimensional Local Visual Probe | {new['TP']} | {new['FP']} | {loss['local_visual']:.9f} |

New probe relative to old scorer, budgeted TP change **{r['TP_delta_vs_old_generic_at_budget']:+d}**, BCE change **{r['BCE_delta_vs_old_generic']:+.9f}**. Precondition requires the former to be strictly greater than 0 and the latter strictly less than 0; this run{'satisfied' if positive else 'does not simultaneously satisfy'}.

The false positive budget is fixed at the official IoU0.5/cap500 FP count from the original R@0.25, with all groups treated equally and not split by score. The curve uses the existing official matching and ignores submission rules. This is a ranking curve diagnosis using calibrated labels, and the selected points do not equate to validated deployable thresholds. No new deployment threshold is frozen in this round, no full AP score is added, and the TP numbers in the table cannot be interpreted as development set AP or independent scientific validation.

## What Actually Changed

The detector was completely frozen, remaining in its original 30-epoch FP32 state with the MSE-range R path (G1 retained floating-point precision, all other fused Conv layers were kept in floating-point QDQ). A single forward pass at the 16th layer P3 output produced a 64×80×80 tensor. The center of each candidate bounding box was sampled using bilinear interpolation to extract a 64-dimensional vector. Additionally, an 8-point average was computed around the bounding box with a 0.75 scaling of width and height to form a 64-dimensional center-minus-neighborhood mean. The resulting vector was concatenated with the original 22-dimensional features, forming a linear scorer with 151 parameters. The surrounding features are not verified pure background; thus, they cannot be used as prior knowledge to eliminate targets.

Retained the original 512 images, 213716 candidate soft quality labels, with only one 100-step CPU float64 Adam fitting. Equal-weight per image, normalization only applied to the image itself, zero-residual initialization, lr0.03, betas0.9/0.999, eps1e-8, L2 coefficients 0.0005; no hard example or precision intervention auxiliary loss added, no candidate selection, feature layout, or optimal epoch tuning. Combined with the previous 6 scoring candidate models, a total of 7 candidates are now accumulated, and all previous candidates are retained.

690 new forward passes were used to extract previously un-saved P3 features, without claiming any zero-forward cache experiment. The two original preprocessing pipelines—on the 512 fitting images and 178 calibration images—were preserved. The complete 8400-candidate raw arrays for all 690 images were bitwise identical to historical R. The detector\'s saved state hash remained consistent before and after. Therefore, the observed score differences stem solely from the new scorer, not from modifications to bounding boxes, classes, the detector, NMS, or image membership.

## Data and Interpretation Boundaries

The 178 calibration training images and 512 fitting images have no overlapping IDs, content SHA, or filename prefixes; however, both sets are drawn from the original training data. The 178 images were already used in the prior scorer calibration, and this round is assumed to be influenced by existing development diagnostics. This can only be described as \'prefix-non-overlapping information diagnostics\'—it cannot be called a fully independent test. The prefix is also not a validated independent scenario identity.

This round does not load the 414 training/val development images, their labels, or detection results, nor does it compute a new AP. The Selection and Holdout loads were not read. The original all-anchor qualification failures and historical failure records were preserved as-is. The consistency of the original arrays in this round does not constitute a reclassification of these failures as passes.

Local visual quality scoring belongs to an established research direction: [IoU-Net](https://arxiv.org/abs/1807.11590) learns localization quality, [GFLV2](https://arxiv.org/abs/2011.12885) studies distribution statistics and quality ranking. This work is not a complete reproduction of these methods, nor does it package ordinary local feature linear probing as an original contribution.

## Independent Review and Delivery

6 pre-checks passed. Main task observes exit code 0, no stderr, no failure acknowledgment, actual process terminates. Independent audit verified{audit['manifest_members_verified']}a new saved member and{audit['input_cache_bindings_verified']}one existing cached binding; recompute all 690 local features using independent NumPy bilinear interpolation, with maximum absolute difference{audit['visual_replay_max_abs_error']:.9g}checked element-wise against pre-frozen atol=1e-4, rtol=2e-5. This is only the CPU validation tolerance for newly added features, and does not alter the original full 8400 candidate numerical thresholds.

An independent audit was conducted on the fitting normalization, 150-dimensional scores for the 178 images, original 22-dimensional reference, original candidate identities, order, and soft quality labels, BCE loss, ignore filtering matrix, IoU0.5 matching, the full same-score curve, and the original R false positive budget. No model re-fitting, detector execution, or full AP computation was performed during the audit. The audit passing only indicates that the saved computations are reproducible.

{lead}

{"If proceeding further, a separate freeze of the same candidate's development protocol, threshold usage, and image-free comparison should be established to evaluate whether this local information can yield cross-group benefits; it is also essential to distinguish between routine visual validation and the unique contribution of precision intervention." if positive else "This fixed candidate's budget has ended. There is no evidence supporting the direct addition of a larger classification head, more visual features, or a seed change to achieve passage; we should first clearly define the distinguishable research mechanisms and independent validation scopes."} Independent scientific acceptance, true INT8, Jetson, and Gate1 as a whole have not passed.

Evidence directory: `artifacts/gate1/20261026_local_visual_information_probe_v1`
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    expected={}
    def add(b):
        if b['path'] in expected:require(expected[b['path']]==b,'Conflicting delivery identity')
        expected[b['path']]=b
    for b in c['sources'].values():add(b)
    for k in ('candidate','architecture','training_manifest','evaluation_protocol','old_scorer','ranges','previous_contract','previous_delivery','old_reference_entries','old_scored_entries','preflight_tests'):add(c[k])
    for role in ('fit','calibration'):
        root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
        for row in c[role]:
            for k in (('features','raw','geometry') if role=='fit' else ('features','predictions','matrices')):add(row[k])
            for key,sha_key in (('image_path','image_sha256'),('annotation_path','annotation_sha256')):
                path=(root/row['metadata'][key]).resolve(strict=True);require(path.is_relative_to(root),'Input path escaped')
                b=identity(path);require(b['sha256']==row['metadata'][sha_key],'Allowed image/annotation identity differs');add(b)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():add(identity(path))
    spec=read(OUT/'saved_audit_contract.json');add(spec['script']);add(spec['score_helper'])
    for path in (Path(__file__),ROOT/'scripts/register_gate1_visual_probe.py',ROOT/'docs/81_gate1_local_visual_information_feasibility_launch.md',report,ROOT/prior['snapshot']):add(identity(path))
    for b in expected.values():require(identity(b['path'])==b,'Final delivery member changed')
    write(OUT/'delivery_manifest.json',{'schema':'rampq.local_visual_information_probe.delivery/v1','at_utc':P.now(),'files':[expected[k] for k in sorted(expected)],'mutable_latest_excluded':True})
    validation={'status':'PASS_LOCAL_VISUAL_PROBE_DELIVERY_INTEGRITY_ONLY','at_utc':P.now(),'manifest':identity(OUT/'delivery_manifest.json'),
        'members_verified':len(expected),'missing':0,'mismatched':0,'audit':identity(OUT/'saved_delivery_audit.json'),
        'machine_decision':identity(OUT/'machine_decision.json'),'report':identity(report),'diagnostic_promising':positive,
        'observed_worker_exit_code':0,'scientific_acceptance':False,'actual_INT8':False,'Gate1_overall_passed':False,'Holdout_accessed':False}
    write(OUT/'delivery_validation.json',validation)
    snap=ROOT/'artifacts/gate1/current_execution_state_v46.json'
    write(snap,{'schema':'rampq.gate1.execution_state/v46','status':'COMPLETE_ONE_LOCAL_VISUAL_INFORMATION_PROBE','at_utc':P.now(),
        'previous_snapshot':prior,'report':identity(report),'delivery_validation':identity(OUT/'delivery_validation.json'),
        'machine_decision':identity(OUT/'machine_decision.json'),'diagnostic_promising':positive,'new_candidates':1,'cumulative_scorer_candidates':7,
        'new_image_forwards':690,'all8400raw_same_as_original_R':True,'new_scorer_updates':100,'new_detector_training_updates':0,'new_full_AP_jobs':0,
        'new_deployment_thresholds':0,'development_evaluated':False,'scientific_acceptance':False,'actual_INT8':False,'Gate1_overall_passed':False,
        'training_val_payload_access':False,'Selection_payload_access':False,'Holdout_payload_access':False,'old_failures_preserved':True,
        'no_active_worker':True,'next_action':decision['next_action']})
    P.FT.atomic_json(latest,{'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v46.json',
        'snapshot_sha256':identity(snap)['sha256'],'delivery_validation':identity(OUT/'delivery_validation.json'),
        'note':'One local visual information probe complete/audited:512fit/178calibration only. Read report82 for actual result. No developer AP/newdeploythreshold/independent science/INT8 claim;Holdout sealed.'})
    print(json.dumps({'status':validation['status'],'members_verified':len(expected),'diagnostic_promising':positive,'TP_delta_vs_old_generic_at_budget':r['TP_delta_vs_old_generic_at_budget'],
        'BCE_delta_vs_old_generic':r['BCE_delta_vs_old_generic'],'snapshot':'v46','report':identity(report)},ensure_ascii=True))


if __name__=='__main__':main()
