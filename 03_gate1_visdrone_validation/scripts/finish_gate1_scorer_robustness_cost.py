"""Write and seal all fixed-output robustness and CPU cost observations."""
from pathlib import Path
import json
from audit_gate1_scorer_robustness_cost import D,P,T,E,S,OUT,A
read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/'contract.json');result=read(A/'diagnostic_result.json');audit=read(OUT/'saved_delivery_audit.json')
    terminal=read(OUT/'launcher_terminal_receipt.json');process=read(OUT/'terminal_validation.json')
    require(audit['status']=='PASS_FROZEN_DIAGNOSTIC_SOURCE_FP_GROUP_AND_TIMING_SUMMARY_AUDIT_ONLY','Saved audit not complete')
    require(terminal['propagated_worker_exit_code']==0 and process['matching_active_pilot_count']==0,'Worker not normally terminated')
    decision={'status':'COMPLETE_FROZEN_SCORER_ROBUSTNESS_COST_DIAGNOSTIC','at_utc':P.now(),
        'contract':identity(OUT/'contract.json'),'result':identity(A/'diagnostic_result.json'),'audit':identity(OUT/'saved_delivery_audit.json'),
        'finish_source':identity(__file__),'cohort_summaries':{name:{k:r[k] for k in ('images','prefix_proxies','pooled_TP_delta','pooled_FP_delta','leave_one_prefix_joint_count','all_leave_one_prefix_joint_observation','FP_geometry_net_change','added_FP_anchors','removed_FP_anchors','gained_GT_ids','lost_GT_ids')} for name,r in result['cohorts'].items()},
        'CPU_cost':result['CPU_overhead'],'historical_FP32_AP_gap_percentage_points':result['AP_gap_to_historical_FP32_percentage_points'],
        'pooled_FP_benefit_is_sensitive_to_one_prefix_group':not result['cohorts']['full414']['all_leave_one_prefix_joint_observation'],
        'scorer_robust_independent_contribution_demonstrated':False,'new_model_fits':0,'new_threshold_candidates':0,'new_detector_forwards':0,
        'scientific_acceptance':False,'Gate1_overall_passed':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
        'next_action':'Do not tune/deletethe9999975prefix to claim success. PooledFPbenefit is fragile and generic qualityranking is prior art. A numerically exact implementation optimization may reduce currentCPUcost but does not create a detection-method contribution; independent device/data evidence remains separate.'}
    write(OUT/'machine_decision.json',decision)
    timing=result['CPU_overhead'];names={'new286':'286 images added','new_prefix47':'47 images with new prefix','full414':'414 images merged'}
    lines=['# Gate1 Frozen Scorer: Stability, False Positive Composition, and Additional CPU Overhead','','2026-10-26。','',
        '**The aggregated improvement has been re-verified, but the net false positive reduction is concentrated within a prefix-group subset, and is still insufficient to claim robustness. The additional CPU cost incurred by feature computation in the current implementation remains significant.**','',
        'The model, scorer, thresholds, and data members are all kept frozen. This round is a post-hoc diagnosis and does not involve any new model inference, parameter tuning, full AP evaluation, or threshold candidates. The diagnosis does not modify the previously established pass criteria, nor does it treat the removal of any group as a new method.','',
        '## Concentration of Aggregate Gains','',
        '| Scope | Net Hit Change | Net False Positive Change | Number of Prefix Groups | Number of Cases Still Showing Joint Improvement After Removal of Any Single Group |','|---|---:|---:|---:|---:|']
    for name,r in result['cohorts'].items():lines.append(f"| {names[name]} | {r['pooled_TP_delta']:+d} | {r['pooled_FP_delta']:+d} | {r['prefix_proxies']} | {r['leave_one_prefix_joint_count']}/{r['prefix_proxies']} |")
    lines += ['', 'Among all 414 images, the 34 images under prefix 9999975 contributed a net reduction of 78 false positives and a net increase of 1 hit. The remaining 380 images showed a net increase of 86 hits and a net increase of 49 false positives. Within the added 286 images, 25 images share the same prefix and contributed a net reduction of 61 false positives; after removal, the remaining 261 images showed a net increase of 50 hits and a net increase of 47 false positives.', '',
        "Therefore, the claim of 'overall false positive reduction of 29' does not imply consistent improvement across all scenarios. All original groups are retained, especially no group that would undermine the conclusion is removed, and no group that would favor the conclusion is deleted. The filename prefix is merely an existing proxy, not a validated scenario identity; this exhaustive removal check is not an independent replication, confidence interval, or statistical significance test.", '',
        'In the new prefix 47 images, removing any one of the 21 prefix groups still fails to meet the joint improvement condition. The false positive issue cannot be resolved by removing the worst-performing group.', '',
        '## Origin of False Positive Changes','',
        'The following represents the net change in false positive geometry within a fixed candidate pool: first, same-class IoU ≥ 0.5; then, other-class IoU ≥ 0.5; followed by maximum IoU in [0.1, 0.5); finally, low-overlap residuals. This is a post-hoc description of preserved false positive annotations, not a causal proof or an official error classification; low overlap cannot be directly labeled as pure background.','',
        '| Scope | Same-class Overlap | Cross-class Overlap | Low IoU with Nearby Objects | Low Overlap Residual | Total |','|---|---:|---:|---:|---:|---:|']
    for name,r in result['cohorts'].items():
        d=r['FP_geometry_net_change'];lines.append(f"| {names[name]} | "+' | '.join(f'{d[k]:+d}' for k in D.KINDS)+f" | {sum(d.values()):+d} |")
    lines += ['', 'Across all 414 images, false positives from same-class overlap decreased by 206, but false positives from other-class overlap increased by 32, localization neighborhood increased by 99, and low-overlap residuals increased by 46, resulting in a net reduction of 29 false positives. The scorer does not simultaneously improve all errors. For the new prefix 47 images, the changes are −16, +11, +30, and +10, respectively, resulting in a net increase of 35 false positives.', '',
        '| Scope | Newly Appeared FP Anchors | Anchors No Longer Classified as FP | Newly Detected Target IDs | Loss hits target ID |','|---|---:|---:|---:|---:|']
    for name,r in result['cohorts'].items():lines.append(f"| {names[name]} | {r['added_FP_anchors']} | {r['removed_FP_anchors']} | {r['gained_GT_ids']} | {r['lost_GT_ids']} |")
    lines += ['', 'FP identity is matched by the same image and raw_anchor_index; target identity follows the original annotation physical row. These two identities must not be conflated. Across all 414 images, there are a net gain of 87 hits, consisting of 179 newly added hits and 92 lost hits. The net gain cannot be interpreted as preserving all original hits.','',
        '## Actual CPU time spent by current post-processing','',
        'On the same i9-14900HX computer, for the saved output of existing 286 images, each image is preheated 1 times and fixed evaluated 3 times, resulting in a total of 858 samples. These samples cover the candidate count with the minimum of 2, median of 436, and maximum of 1000. In this round, only the current code is tested, without modifying the implementation or comparing multiple optimized versions.','',
        '| Operation | Median (ms) | P95(ms) | Mean (ms) | Maximum (ms) |','|---|---:|---:|---:|---:|']
    for key,label in (('score_only','Linear scoring only computed'),('features_score_stable_sort','22-dimensional features + scoring + stable ranking')):
        r=timing[key];lines.append(f"| {label} | {r['median_ms']:.6f} | {r['p95_ms']:.6f} | {r['mean_ms']:.6f} | {r['maximum_ms']:.6f} |")
    lines += ['', 'Timing does not include file/annotation reading, image decoding, detector execution, original NMS, CUDA output transfer, and final serialization. This is not end-to-end speed, power consumption, or Jetson latency. The 23 linear parameters are minimal, but neighbor feature computation requires pairwise overlap calculations among bounding boxes. The main cost in the current implementation lies in the feature stage, and thus the additional computation cannot be considered free.', '',
        'Future engineering optimizations could verify output equivalence; even if successful, they can only demonstrate reduced cost of the current implementation and cannot replace evidence of originality or independent generalization of the detection method.','',
        '## Precision and paper contribution boundaries','',
        f"Scorer frozen, AP over 414 images is{result['frozen_full414_AP_percent']:.6f}%, historical FP32 same-set AP is{result['historical_FP32_full414_AP_percent']:.6f}%, difference is{result['AP_gap_to_historical_FP32_percentage_points']:+.6f}percentage points. This historical baseline reuses previously saved results without rerunning old scoring; a significant FP32 precision gap remains.",'',
        '[IoU-Net](https://arxiv.org/abs/1807.11590) has already leveraged localization confidence to improve candidate selection; [GFLV2](https://arxiv.org/abs/2011.12885) has utilized statistical edge distribution to estimate localization quality and improve ranking; [DetPTQ](https://arxiv.org/abs/2304.09785) has studied quantization setting selection relevant to detection tasks. These works indicate that quality scoring and task-related quantization are not novel contributions here. This round does not claim to reproduce their full methods, nor does it substitute measured values from the paper with its own empirical results.', '',
        'The marginal improvement from a conventional scorer cannot be rebranded as an intervention contribution in a failed control. Current evidence lacks a stable mechanism that outperforms strong baselines, as well as actual data not involved in development and real INT8/board-level costs. Therefore, the paper method is not marked as successful, and Gate1 is not passed.','',
        '## Saved and Reviewed','',
        f"A single process completes acknowledgments with exit code 0 fully. 9 pre-checks passed; independent audit verified{audit['manifest_members_verified']}outputs generated,{audit['reused_members_verified']}bindings reused and all{audit['FP_details_independently_checked']}FP geometric records, recalculated all leave-one-prefix group results across three ranges, 858 timing samples, and nanosecond-to-millisecond summaries; no re-execution of timing, training, or model inference was performed.",'',
        "After timing completion, each baseline path was cross-checked for consistency of features, scores, and rankings against existing cache. All 414 images' per-target gains and losses, each group's removal results, FP additions/removals by identity and type, and all 858 original nanosecond samples were preserved. The timing reflects the conditions of this process and hardware, but does not guarantee equivalence on other machines.", '',
        'Original raw values FAIL, original contracts, thresholds, results, and reports are preserved as-is; Selection load not read, Holdout sealed, no cloud resources, no new scheduled tasks. Evidence directory: `artifacts/gate1/20261026_scorer_robustness_cost_v1`.']
    report=P.ROOT/'docs/77_gate1_scorer_robustness_and_cpu_cost_results.md'
    with report.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    latest=P.ROOT/'artifacts/gate1/latest_execution_state.json';previous=read(latest)
    require(previous['snapshot']=='artifacts/gate1/current_execution_state_v39.json','Latest pointer advanced elsewhere')
    require(identity(P.ROOT/previous['snapshot'])['sha256']==previous['snapshot_sha256'],'Previous snapshot changed')
    expected={}
    def add(b):
        if b['path'] in expected:require(expected[b['path']]==b,'Conflicting file identity')
        expected[b['path']]=b
    for b in read(A/'manifest.json')['files']+c['reused_files']+list(c['source_bindings'].values()):add(b)
    for key in ('previous_threshold_contract','previous_threshold_delivery','previous_threshold_decision','extension_contract','extension_delivery','frozen_threshold_record','scorer_model','evaluation_protocol','historical_FP32_AP'):add(c[key])
    root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
    for meta in c['cohorts']['full414']:
        path=(root/meta['annotation_path']).resolve(strict=True);require(path.is_relative_to(root),'Allowed annotation escaped')
        b=identity(path);require(b['sha256']==meta['annotation_sha256'],'Annotation changed');add(b)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():add(identity(path))
    for path in (P.ROOT/previous['snapshot'],P.ROOT/'docs/76_gate1_frozen_scorer_stability_and_cost_diagnostics_launch.md',report,
                 P.ROOT/'scripts/audit_gate1_scorer_robustness_cost.py',Path(__file__)):add(identity(path))
    for b in expected.values():require(identity(b['path'])==b,'Delivery file changed')
    write(OUT/'delivery_manifest.json',{'schema':'rampq.scorer_robustness_cost.delivery/v1','at_utc':P.now(),'files':[expected[k] for k in sorted(expected)],'mutable_latest_pointer_excluded':True})
    validation={'status':'PASS_FROZEN_DIAGNOSTIC_DELIVERY_INTEGRITY_ONLY','at_utc':P.now(),'manifest':identity(OUT/'delivery_manifest.json'),
        'members_verified':len(expected),'missing':0,'mismatched':0,'audit':identity(OUT/'saved_delivery_audit.json'),'machine_decision':identity(OUT/'machine_decision.json'),
        'report':identity(report),'observed_worker_exit_code':0,'scientific_acceptance':False,'Gate1_overall_passed':False,'actual_INT8':False,'Holdout_accessed':False}
    write(OUT/'delivery_validation.json',validation)
    snapshot=P.ROOT/'artifacts/gate1/current_execution_state_v40.json'
    write(snapshot,{'schema':'rampq.gate1.execution_state/v40','status':'COMPLETE_FROZEN_SCORER_ROBUSTNESS_COST_DIAGNOSTIC','created_at_utc':P.now(),
        'previous_snapshot':previous,'current_experiment':'artifacts/gate1/20261026_scorer_robustness_cost_v1','report':report.name,
        'delivery_validation':identity(OUT/'delivery_validation.json'),'machine_decision':identity(OUT/'machine_decision.json'),
        'all_models_and_threshold_frozen':True,'new_detector_forwards':0,'FP_benefit_sensitive_to_one_prefix_group':True,
        'scorer_feature_score_sort_CPU_median_ms':timing['features_score_stable_sort']['median_ms'],
        'scorer_feature_score_sort_CPU_p95_ms':timing['features_score_stable_sort']['p95_ms'],'independent_contribution_demonstrated':False,
        'scientific_acceptance':False,'Gate1_overall_passed':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
        'no_active_worker':True,'next_action':decision['next_action']})
    pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v40.json',
        'snapshot_sha256':identity(snapshot)['sha256'],'delivery_validation':identity(OUT/'delivery_validation.json'),
        'note':'Fixedscorer robustness/cost diagnostic complete:pooledFPbenefit depends on9999975prefix;CPUfeatures+score+sortmedian13.9539ms,p9568.678275ms.No newmethod or independent success;Holdout sealed.'}
    tmp=latest.with_name('latest_execution_state.v40.pending.json');write(tmp,pointer);tmp.replace(latest)
    print(json.dumps({'status':validation['status'],'members':len(expected),'snapshot':pointer['snapshot'],'report':identity(report)},ensure_ascii=True))


if __name__=='__main__':main()
