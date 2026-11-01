"""Independent saved replay/timing audit, Chinese report and exact-optimization seal."""
from collections import Counter
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import run_gate1_scorer_exact_CPU as B
P=B.P;S=B.S;O=B.O;D=B.D;OUT=B.OUT;A=OUT/'attempt_01'
read,write,identity,require=P.read,P.write,P.identity,P.require


def summarize_independent(a):
    a=np.asarray(a,np.float64)
    return {'samples':len(a),'median_ms':float(np.quantile(a,.5)),'p95_ms':float(np.quantile(a,.95)),
        'maximum_ms':float(max(a)),'mean_ms':float(sum(a)/len(a)),'minimum_ms':float(min(a))}


def main():
    P.torch.set_num_threads(1)
    c=read(OUT/'contract.json');result=read(A/'optimization_result.json');complete=read(A/'completion_receipt.json')
    terminal=read(OUT/'launcher_terminal_receipt.json');process=read(OUT/'terminal_validation.json')
    require(terminal['observed_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and process['matching_active_pilot_count']==0,'Abnormal terminal state')
    require(complete['status']=='COMPLETE_SINGLE_EXACT_CPU_FEATURE_OPTIMIZATION' and complete['contract']==identity(OUT/'contract.json'),'Incomplete worker/contract identity')
    spec=read(OUT/'saved_audit_contract.json');require(identity(spec['script']['path'])==spec['script'],'Frozen audit source changed')
    for b in c['source_bindings'].values():require(identity(b['path'])==b,'Frozen source changed')
    for key in ('previous_cost_contract','previous_cost_delivery','previous_cost_decision','scorer_model','frozen_threshold_record','preflight_initial_failure'):
        require(identity(c[key]['path'])==c[key],'Bound source changed: '+key)
    xml=ET.parse(OUT/'preflight_tests.xml').getroot()
    require(sum(int(x.get('tests',0)) for x in xml.iter('testsuite'))==11,'Qualification count differs')
    require(sum(int(x.get(k,0)) for x in xml.iter('testsuite') for k in ('errors','failures','skipped'))==0,'Final preflight not PASS')
    manifest=read(A/'manifest.json')['files']
    for b in manifest:require(identity(b['path'])==b,'Saved output changed')
    paths={str(p.resolve()) for p in A.rglob('*') if p.is_file() and p.name not in {'events.jsonl','live_status.json','manifest.json','completion_receipt.json'}}
    require(paths=={b['path'] for b in manifest} and len(paths)==len(manifest),'Output manifest coverage differs')
    records=read(A/'equivalence_checks.json');require(len(records)==592,'Exact output membership incomplete')
    model=read(c['scorer_model']['path']);member_bindings={}
    for i,(m,r) in enumerate(zip(c['members'],records)):
        require(m['scope']==r['scope'] and m['index']==r['index'] and m['image_id']==r['image_id'],'Exact replay order/member differs')
        for key in ('raw','prediction_source','features'):
            binding=m[key];member_bindings[binding['path']]=binding
        source=read(m['prediction_source']['path']);pred=source['predictions'] if m['prediction_branch'] is None else source[m['prediction_branch']]
        if isinstance(pred,dict):pred=pred['predictions']
        with np.load(m['raw']['path'],allow_pickle=False) as d:raw=d[m['raw_key']].copy()
        with np.load(m['features']['path'],allow_pickle=False) as d:cached=d['features'].copy();p=d['original_score'].copy()
        before=S.visible_features(raw,pred);after=O.visible_features(raw,pred)
        require(np.array_equal(before,cached) and np.array_equal(after,cached),'Independent bitwise feature replay failed')
        first=S.score_rows(before,p,model);second=S.score_rows(after,p,model)
        require(np.array_equal(first,second),'Independent frozen-model score replay failed')
        firstsort=sorted(range(len(pred)),key=lambda j:(-float(first[j]),pred[j]['nms_input_order']))
        secondsort=sorted(range(len(pred)),key=lambda j:(-float(second[j]),pred[j]['nms_input_order']))
        require(firstsort==secondsort,'Independent stable sort replay failed')
        require(r['features_bitwise_equal'] and r['scores_bitwise_equal'] and r['stable_sort_equal'] and r['candidates']==len(pred),'Saved exact replay receipt differs')
        if (i+1)%64==0:P.FT.atomic_json(OUT/'audit_live_status.json',{'stage':'INDEPENDENT592EXACT_REPLAY','done':i+1,'total':592,'at_utc':P.now()})
    for b in member_bindings.values():require(identity(b['path'])==b,'Frozen cache/input changed')
    timings=read(A/'paired_CPU_timings.json');require(len(timings)==858,'Paired timing budget incomplete');seen=set()
    for row in timings:
        key=(row['index'],row['repeat']);require(key not in seen,'Duplicate paired timing');seen.add(key)
        member=c['members'][row['index']];require(member['scope']=='new286' and member['image_id']==row['image_id'],'Benchmark member differs')
        order=['original','optimized'] if (row['index']+row['repeat'])%2==0 else ['optimized','original']
        require(order==row['measurement_order'],'Paired ordering did not alternate')
        require(row['candidate_count']==records[row['index']]['candidates'] and all(type(row[k]) is int and row[k]>0 for k in ('original_ns','optimized_ns')),'Timing units/candidate count invalid')
    require(seen=={(i,j) for i in range(286) for j in range(3)},'Paired timing coverage differs')
    original=np.array([x['original_ns']/1e6 for x in timings]);optimized=np.array([x['optimized_ns']/1e6 for x in timings])
    for label,values in (('original_CPU',original),('optimized_CPU',optimized)):
        check=summarize_independent(values)
        require(all(np.isclose(v,result[label][k],atol=1e-12,rtol=1e-12) for k,v in check.items()),'Independent paired timing summary differs')
    ratio=result['original_CPU']['median_ms']/result['optimized_CPU']['median_ms']
    require(result['ratio_of_median_times']==ratio and result['median_paired_time_ratio']==float(np.median(original/optimized)),'Speed ratio definition differs')
    require(result['optimized_faster_sample_count']==int((optimized<original).sum()),'Paired faster count differs')
    require(result['scope_member_counts']=={'new286':286,'old128':128,'calibration178':178} and result['all_features_scores_and_orders_exact'],'Exact scope differs')
    audit={'status':'PASS592BITWISE_REPLAY_AND_PAIRED_TIMING_SUMMARY_AUDIT_ONLY','at_utc':P.now(),'contract':identity(OUT/'contract.json'),
        'manifest_members_verified':len(manifest),'input_cache_files_verified':len(member_bindings),'members_exactly_replayed':592,
        'paired_timing_samples_verified':858,'model_fits':0,'detector_forwards':0,'threshold_changes':0,'benchmark_repeated':False,
        'new_scientific_method_validated':False,'scientific_acceptance':False,'actual_INT8':False,'Holdout_accessed':False}
    write(OUT/'saved_delivery_audit.json',audit)
    decision={'status':'COMPLETE_SINGLE_EXACT_CPU_IMPLEMENTATION_ENGINEERING_CHECK','at_utc':P.now(),'contract':identity(OUT/'contract.json'),
        'result':identity(A/'optimization_result.json'),'audit':identity(OUT/'saved_delivery_audit.json'),'audit_finish_source':identity(__file__),
        'bitwise_members':592,'paired_samples_each':858,'original_CPU':result['original_CPU'],'optimized_CPU':result['optimized_CPU'],
        'ratio_of_median_times':ratio,'median_paired_time_ratio':result['median_paired_time_ratio'],'optimized_faster_sample_count':result['optimized_faster_sample_count'],
        'exact_outputs_no_accuracy_changes':True,'frozen_model_unchanged':True,'frozen_threshold_unchanged':True,
        'scientific_method_contribution_proved':False,'pooled_FP_robustness_issue_unchanged':True,'new_prefix47_FP_growth_unchanged':True,
        'new_detector_forwards':0,'new_AP_scores':0,'new_models_fit':0,'threshold_searches':0,'optimization_candidates':1,
        'Gate1_overall_passed':False,'scientific_acceptance':False,'actual_INT8':False,'Jetson_acceptance':False,'Selection_payload_access':False,'Holdout_payload_access':False,
        'next_action':'The exact CPUengineering improvement can be reused only in a new unfrozen runner,never by editing sealed historical sources. Detection-method novelty/robustness remain unsupported. Further independent data or final-device work needs concrete access/evidence and fixedscope; no more favorable-threshold/seed/model search on these dev images.'}
    write(OUT/'machine_decision.json',decision)
    lines=['# Gate1 Scoring Module Feature Computation: Strict Equivalence with CPU Optimization','','2026-10-26。','',
        f"**The fixed 128-row tiling implementation has passed bit-level consistency checks across 592 saved outputs. In paired measurements, the CPU median time for features, scoring, and stable ranking decreased from{result['original_CPU']['median_ms']:.3f}ms to{result['optimized_CPU']['median_ms']:.3f}ms, with a median ratio of{ratio:.2f}。**",'',
        'This result represents current engineering optimization within the implementation. The detection box, class, score, ranking, threshold, and existing detection accuracy remain unchanged; it does not address the growth of false positives in new prefix 47 images or insufficient paper contributions.','',
        '## Unique Implementation and Consistency','',
        'The original implementation constructed a full pairwise intersection array per image. The new implementation uses a fixed 128-row tile, separately computing the intersection of width and height, preserving the same float64 computation order and identical index rules, while reducing temporary arrays. Only the tile size and one code implementation are registered—no faster configuration is searched.', '',
        "All 592 members include 286 newly developed images, 128 original development images, and 178 calibration images. Each image's 22-dimensional feature vector matches bit-for-bit with the original implementation and original cache, with bit-for-bit equal scoring from the same frozen 23-parameter model and strictly identical stable ranking indices. No error tolerance was relaxed, no new images or annotations were read, and no new detector inference was introduced.",'',
        'The threshold remains at 0.17473852372359241, and the model SHA value is preserved. The new function is provided only as a new engineering entry point; historical frozen scripts, contracts, and weights are unchanged. Future experiments using this implementation require binding its identity in a new contract and cannot override the original training source.','',
        '## On-machine Paired CPU Timing','',
        '| Implementation | Median (ms) | P95ms | Mean (ms) | Max (ms) |','|---|---:|---:|---:|---:|']
    for key,label in (('original_CPU','original full overlapping array'),('optimized_CPU','fixed 128-row tiling')):
        r=result[key];lines.append(f"| {label} | {r['median_ms']:.6f} | {r['p95_ms']:.6f} | {r['mean_ms']:.6f} | {r['maximum_ms']:.6f} |")
    lines += ['', f"Use the same 286 saved image outputs. Each implementation warms up 1 time per image, followed by 3 paired measurements, totaling 858 samples per implementation. Image index and repetition number determine alternating execution order between the new and old implementations. Among858 pairs, the optimized implementation is faster in{result['optimized_faster_sample_count']}pairs. The median ratio of per-pair execution time was{result['median_paired_time_ratio']:.3f}, which differs from the overall median ratio of the two groups, and both are preserved.",'',
        'Timing only includes 22-dimensional features, frozen CPU scoring, and stable Python index-based ranking. File I/O, image decoding, detector inference, NMS, CUDA output transfer, annotation reading, and final serialization are excluded. The test machine is an i9-14900HX; this is not a Jetson, power, or end-to-end detection pipeline performance metric. The CPU segment ratio cannot be interpreted as an end-to-end acceleration factor.', '',
        '## Original False Positive and Robustness Issues Remain Unchanged','',
        "The original 414-image set showed a net increase of 87 false positives and a net decrease of 29 false negatives under a fixed threshold. However, one filename prefix group contributed a net reduction of 78 false positives; removing the rest of that group still resulted in a net increase of 49 false positives. The new prefix 47 images showed an increase of 35 false positives. This engineering optimization is strictly equivalent to the original output, and all these facts remain unchanged—no changes to the implementation have been used to 'correct' the statistical results.", '',
        'The complete 414-image AP of the standard scorer remains at 11.940007%, with the historical FP32 value at 12.440710%, resulting in a gap of approximately 0.501 percentage points. Quality scoring and localization quality estimation have been previously studied; this round does not demonstrate a new detection mechanism, independent generalization, or original paper contribution.', '',
        '## Testing and Independent Verification','',
        'Final 11 pre-checks passed, including 0/1/4/10/127/128/129/257/1000 candidate boundary, tie-handling, frozen score reproducibility, and input validation. During the initial pre-check, a test incorrectly treated the first tied box containing an extra random box as having no higher-scoring neighbor; this test has since been modified to isolate 4 cases of tied boxes, with the original failing JUnit retained, and the bit-wise equality test for features already passed prior to the fix.', '',
        'The main process observed an exit code of 0 and received a completion acknowledgment. A saved-artifact audit independently replayed all 592 new and old feature, score, and ranking outputs, re-verifying each input identity, 858 original nanosecond timestamps, alternating measurement order, coverage, and millisecond summaries; no duplicated baseline timing, detector inference, or model training was detected.', '',
        'This is an engineering improvement that is reusable and cannot be considered a complete research success. All previous FAILs and 47 false positive detections are preserved. The Selection load is not read, the Holdout is sealed, and no cloud resources or new scheduling tasks are involved.', '',
        'New implementation: [gate1_scorer_exact_features.py]([LOCAL_PROJECT]/legacy_engineering/scripts/gate1_scorer_exact_features.py). Evidence directory: `artifacts/gate1/20261026_scorer_exact_CPU_optimization_v1`. Robustness diagnostics are referenced in report 77.']
    report=P.ROOT/'docs/78_gate1_exact_scorer_feature_computation_optimization.md'
    with report.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    latest=P.ROOT/'artifacts/gate1/latest_execution_state.json';previous=read(latest)
    require(previous['snapshot']=='artifacts/gate1/current_execution_state_v41.json','Latest pointer changed elsewhere')
    require(identity(P.ROOT/previous['snapshot'])['sha256']==previous['snapshot_sha256'],'Previous snapshot changed')
    expected={}
    def add(b):
        if b['path'] in expected:require(expected[b['path']]==b,'Conflicting file identity')
        expected[b['path']]=b
    for b in manifest+list(c['source_bindings'].values())+list(member_bindings.values()):add(b)
    for key in ('previous_cost_contract','previous_cost_delivery','previous_cost_decision','scorer_model','frozen_threshold_record','preflight_initial_failure'):add(c[key])
    for path in sorted(OUT.rglob('*')):
        if path.is_file():add(identity(path))
    for path in (P.ROOT/previous['snapshot'],P.ROOT/'docs/77_gate1_scorer_robustness_and_cpu_cost_results.md',report,Path(__file__)):add(identity(path))
    for b in expected.values():require(identity(b['path'])==b,'Delivery member changed')
    write(OUT/'delivery_manifest.json',{'schema':'rampq.exact_scorer_CPU_optimization.delivery/v1','at_utc':P.now(),'files':[expected[k] for k in sorted(expected)],'mutable_latest_pointer_excluded':True})
    validation={'status':'PASS_EXACT_FEATURE_OPTIMIZATION_DELIVERY_INTEGRITY_ONLY','at_utc':P.now(),'manifest':identity(OUT/'delivery_manifest.json'),
        'members_verified':len(expected),'missing':0,'mismatched':0,'audit':identity(OUT/'saved_delivery_audit.json'),
        'machine_decision':identity(OUT/'machine_decision.json'),'report':identity(report),'observed_worker_exit_code':0,
        'bitwise_members':592,'paired_timing_samples':858,'scientific_acceptance':False,'Gate1_overall_passed':False,'actual_INT8':False,'Holdout_accessed':False}
    write(OUT/'delivery_validation.json',validation)
    snap=P.ROOT/'artifacts/gate1/current_execution_state_v42.json'
    write(snap,{'schema':'rampq.gate1.execution_state/v42','status':'COMPLETE_EXACT_CPU_SCORER_ENGINEERING_OPTIMIZATION','created_at_utc':P.now(),
        'previous_snapshot':previous,'current_experiment':'artifacts/gate1/20261026_scorer_exact_CPU_optimization_v1','report':report.name,
        'delivery_validation':identity(OUT/'delivery_validation.json'),'machine_decision':identity(OUT/'machine_decision.json'),
        'bitwise_members_verified':592,'paired_timing_samples_each':858,'old_CPU_median_ms':result['original_CPU']['median_ms'],
        'optimized_CPU_median_ms':result['optimized_CPU']['median_ms'],'ratio_of_median_times':ratio,
        'all_existing_accuracy_and_threshold_results_unchanged':True,'47prefix_new_FP_growth_unchanged':True,'pooled_FP_sensitivity_to9999975prefix_unchanged':True,
        'new_detector_forwards':0,'new_models':0,'new_thresholds':0,'scientific_contribution_demonstrated':False,
        'Gate1_overall_passed':False,'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
        'no_active_worker':True,'next_action':decision['next_action']})
    pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v42.json','snapshot_sha256':identity(snap)['sha256'],
        'delivery_validation':identity(OUT/'delivery_validation.json'),'note':'SingleexactCPUfeature optimization complete/audited:592bitwiseoutputs unchanged,858pairedtimings. See report78 for actual CPUcost. No detection/novelty/independent/INT8 acceptance; known FP fragility preserved,Holdout sealed.'}
    tmp=latest.with_name('latest_execution_state.v42.pending.json');write(tmp,pointer);tmp.replace(latest)
    print(json.dumps({'status':validation['status'],'members_verified':len(expected),'old_median_ms':result['original_CPU']['median_ms'],
        'new_median_ms':result['optimized_CPU']['median_ms'],'ratio_of_median_times':ratio,'report':identity(report),'latest':pointer['snapshot']},ensure_ascii=True))


if __name__=='__main__':main()
