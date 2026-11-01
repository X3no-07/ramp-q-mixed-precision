"""Saved-only independent conservation, grouping, geometry and timing audit."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import gate1_scorer_robustness_cost as D
P=D.P;T=D.T;E=D.E;S=D.S;OUT=D.OUT;A=OUT/'attempt_01'
read,write,identity,require=P.read,P.write,P.identity,P.require


def independent_type(iou,categories,category):
    same=max([float(v) for v,c in zip(iou,categories) if c==category]+[0.])
    maximum=max([float(v) for v in iou]+[0.])
    if same>=.5:return 'same_class_overlap'
    if maximum>=.5:return 'other_class_overlap'
    if maximum>=.1:return 'near_object_low_IoU'
    return 'low_overlap_remainder'


def main():
    P.torch.set_num_threads(1)
    c=read(OUT/'contract.json');completion=read(A/'completion_receipt.json');terminal=read(OUT/'launcher_terminal_receipt.json')
    require(completion['status']=='COMPLETE_FROZEN_SCORER_ROBUSTNESS_COST_DIAGNOSTIC' and completion['new_detector_forwards']==0 and completion['new_model_fits']==0,'Completion scope mismatch')
    require(terminal['propagated_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and terminal['stderr_bytes']==0,'Abnormal worker terminal state')
    require(completion['contract']==identity(OUT/'contract.json'),'Contract changed')
    for b in c['source_bindings'].values():require(identity(b['path'])==b,'Frozen source changed')
    for key in ('previous_threshold_contract','previous_threshold_delivery','previous_threshold_decision','extension_contract','extension_delivery','frozen_threshold_record','scorer_model','evaluation_protocol','historical_FP32_AP'):
        require(identity(c[key]['path'])==c[key],'Input binding changed: '+key)
    for b in c['reused_files']:require(identity(b['path'])==b,'Reused artifact changed')
    spec=read(OUT/'saved_audit_contract.json');require(identity(spec['script']['path'])==spec['script'],'Frozen audit changed')
    xml=ET.parse(OUT/'preflight_tests.xml').getroot()
    require(sum(int(x.get('tests',0)) for x in xml.iter('testsuite'))==9,'Qualification test count differs')
    require(sum(int(x.get(k,0)) for x in xml.iter('testsuite') for k in ('errors','failures','skipped'))==0,'Qualification incomplete')
    manifest=read(A/'manifest.json')['files']
    for b in manifest:require(identity(b['path'])==b,'Saved output changed')
    expected={str(p.resolve()) for p in A.rglob('*') if p.is_file() and p.name not in {'events.jsonl','live_status.json','manifest.json','completion_receipt.json'}}
    require(expected=={b['path'] for b in manifest} and len(expected)==len(manifest),'Manifest coverage mismatch')
    per=read(A/'per_image_diagnostics.json');details=read(A/'FP_geometry_details.json');result=read(A/'diagnostic_result.json');robust=read(A/'robustness_diagnostics.json')
    require(result['cohorts']==robust['cohorts'],'Diagnostic result differs from detailed robustness')
    require([r['image_id'] for r in per]==[r['image_id'] for r in c['cohorts']['full414']],'Developer member/order differs')
    extension=Path(c['extension_root']);threshold=Path(c['threshold_root'])
    base=read(extension/'attempt_01/evaluation/full414/mse_r/saved_predictions_and_targets.json')
    score=read(extension/'attempt_01/evaluation/full414/frozen_score_base/saved_predictions_and_targets.json')
    matches=read(threshold/'attempt_01/developer_project_threshold_matches.json')
    detail_index=defaultdict(list)
    for row in details:detail_index[(row['image_id'],row['branch'])].append(row)
    root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
    fp_total=0;permap={}
    for meta,r,b,s,m in zip(c['cohorts']['full414'],per,base,score,matches):
        image_id=meta['image_id'];require(r['image_id']==b['image_id']==s['image_id']==m['image_id']==image_id,'Image identity differs')
        require(m['threshold']==c['fixed_threshold'] and r['prefix_proxy']==meta['prefix_proxy'],'Frozen threshold/prefix changed')
        old_match=b['project_conf025'];new_match=m['matching']
        old_hits={x['target_id'] for x in old_match['matches']};new_hits={x['target_id'] for x in new_match['matches']}
        require(r['newly_hit_target_ids']==sorted(new_hits-old_hits) and r['lost_hit_target_ids']==sorted(old_hits-new_hits),'Target identity change differs')
        require(r['TP_delta']==len(new_hits)-len(old_hits)==r['new_TP']-r['base_TP'],'TP conservation mismatch')
        ann=(root/meta['annotation_path']).resolve(strict=True);require(ann.is_relative_to(root) and P.sha(ann)==meta['annotation_sha256'],'Developer annotation changed')
        gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),image_id);eligible={x['target_id'] for x in old_match['targets']}
        positives=[g for g in gt if g['target_id'] in eligible]
        boxes=np.array([P.MATCH._xyxy(g['bbox_xywh'],meta['width'],meta['height']) for g in positives]).reshape(-1,4);categories=[g['category'] for g in positives]
        area=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
        fp_sets={}
        for branch,rows,matching,key in (('mse_r',b['predictions'],old_match,'base_FP_types'),('scored_fixed_threshold',s['predictions'],new_match,'new_FP_types')):
            pred={x['detection_id']:x for x in rows};fps=[pred[x['detection_id']] for x in matching['detection_states'] if x['reason']=='false_positive']
            detail_rows=detail_index[(image_id,branch)]
            indexed={x['raw_anchor_index']:x for x in detail_rows};fp_sets[branch]={x['raw_anchor_index'] for x in fps}
            require(len(indexed)==len(detail_rows)==len(fps) and set(indexed)==fp_sets[branch],'FP detail coverage differs')
            counted=Counter({k:0 for k in D.KINDS})
            for predrow in fps:
                saved=indexed[predrow['raw_anchor_index']];box=np.array(P.MATCH._xyxy(predrow['bbox_xywh'],meta['width'],meta['height']))
                inter=np.maximum(0,np.minimum(boxes[:,2:],box[2:])-np.maximum(boxes[:,:2],box[:2])).prod(1)
                denom=area+np.maximum(0,box[2:]-box[:2]).prod()-inter;iou=inter/np.maximum(1e-12,denom)
                kind=independent_type(iou,categories,predrow['category'])
                require(kind==saved['kind'] and saved['category']==predrow['category'] and saved['score']==predrow['score'],'Independent FP geometry/category/score differs')
                require(np.isclose(saved['max_GT_IoU'],float(iou.max()) if len(iou) else 0.,atol=1e-12,rtol=1e-12),'FP maximum IoU differs')
                counted[kind]+=1
            require(dict(counted)==r[key],'Per-image FP type aggregate differs');fp_total+=len(fps)
        oldfp=fp_sets['mse_r'];newfp=fp_sets['scored_fixed_threshold']
        require(r['added_FP_anchor_ids']==sorted(newfp-oldfp) and r['removed_FP_anchor_ids']==sorted(oldfp-newfp),'FP identity set difference differs')
        require(r['FP_delta']==len(newfp)-len(oldfp)==r['new_FP']-r['base_FP'],'FP conservation mismatch')
        for branch,other in (('mse_r',newfp),('scored_fixed_threshold',oldfp)):
            require(all(x['FP_anchor_in_other_branch']==(x['raw_anchor_index'] in other) for x in detail_index[(image_id,branch)]),'FP cross-path identity differs')
        permap[image_id]=r
    require(fp_total==len(details),'FP detail count differs')
    for name in ('new286','new_prefix47','full414'):
        saved=result['cohorts'][name];rows=[permap[m['image_id']] for m in c['cohorts'][name]]
        groups=defaultdict(lambda:[0,0,0])
        for row in rows:
            values=groups[row['prefix_proxy']];values[0]+=row['TP_delta'];values[1]+=row['FP_delta'];values[2]+=1
        tp=sum(x[0] for x in groups.values());fp=sum(x[1] for x in groups.values())
        require(saved['pooled_TP_delta']==tp and saved['pooled_FP_delta']==fp and saved['prefix_proxies']==len(groups),'Group totals differ')
        leaves=[]
        for prefix,(dtp,dfp,n) in sorted(groups.items()):
            leaves.append({'omitted_prefix_proxy':prefix,'omitted_images':n,'remaining_TP_delta':tp-dtp,'remaining_FP_delta':fp-dfp,'remaining_joint_observation':tp-dtp>=0 and fp-dfp<=0})
        require(leaves==saved['leave_one_prefix_out'],'Exhaustive leave-one-group diagnostic differs')
        require(saved['leave_one_prefix_joint_count']==sum(x['remaining_joint_observation'] for x in leaves),'Leave-one-group summary differs')
        require(saved['all_leave_one_prefix_joint_observation']==all(x['remaining_joint_observation'] for x in leaves),'All-group robustness flag differs')
        for basekey,newkey in (('base_FP_types','base'),('new_FP_types','new')):
            count=Counter({k:0 for k in D.KINDS})
            for row in rows:count.update(row[basekey])
            require(dict(count)==saved['FP_geometry_counts'][newkey],'Cohort FP geometry total differs')
        require(saved['added_FP_anchors']==sum(len(r['added_FP_anchor_ids']) for r in rows) and saved['removed_FP_anchors']==sum(len(r['removed_FP_anchor_ids']) for r in rows),'FP addition/removal total differs')
    samples=read(A/'CPU_timing_samples.json');timing=read(A/'CPU_overhead_summary.json')
    require(len(samples)==858 and timing['sample_count']==858 and timing['images']==286,'Timing sample budget differs')
    seen=set()
    for row in samples:
        key=(row['index'],row['repeat']);require(key not in seen,'Duplicated timing sample');seen.add(key)
        member=c['benchmark_members'][row['index']]
        require(row['image_id']==member['image_id'] and row['repeat'] in (0,1,2),'Timing member/repetition differs')
        require(type(row['score_only_ns']) is int and row['score_only_ns']>0 and type(row['features_score_stable_sort_ns']) is int and row['features_score_stable_sort_ns']>0,'Invalid nanosecond measurements')
        if row['repeat']==0:
            folder=extension/'attempt_01/new_inference'/f"{row['index']:04d}"
            with np.load(folder/'scorer_features.npz',allow_pickle=False) as d:require(len(d['original_score'])==row['candidates'],'Timing candidate count differs')
    require(seen=={(i,j) for i in range(286) for j in range(3)},'Timing coverage differs')
    for key,field in (('score_only','score_only_ns'),('features_score_stable_sort','features_score_stable_sort_ns')):
        values=np.array([r[field] for r in samples],np.float64)/1000000.
        expected={'samples':len(values),'median_ms':float(np.quantile(values,.5)),'p95_ms':float(np.quantile(values,.95)),
            'maximum_ms':float(max(values)),'mean_ms':float(sum(values)/len(values)),'minimum_ms':float(min(values))}
        for k,value in expected.items():require(np.isclose(timing[key][k],value,atol=1e-12,rtol=1e-12),'Independent timing millisecond summary differs')
    require(result['CPU_overhead']==timing and not timing['end_to_end_speedup_measured'] and not timing['Jetson_latency_measured'],'Timing scope differs')
    fp32=read(c['historical_FP32_AP']['path'])['metrics_percent']['AP_all']
    score_AP=read(extension/'attempt_01/evaluation/full414/frozen_score_base/metrics.json')['metrics_percent']['AP_all']
    require(result['historical_FP32_full414_AP_percent']==fp32 and result['frozen_full414_AP_percent']==score_AP and result['AP_gap_to_historical_FP32_percentage_points']==score_AP-fp32,'Historical AP comparison differs')
    audit={'status':'PASS_FROZEN_DIAGNOSTIC_SOURCE_FP_GROUP_AND_TIMING_SUMMARY_AUDIT_ONLY','at_utc':P.now(),'contract':identity(OUT/'contract.json'),
        'completion':identity(A/'completion_receipt.json'),'manifest_members_verified':len(manifest),'reused_members_verified':len(c['reused_files']),
        'developer_members':414,'FP_details_independently_checked':len(details),'cohort_prefix_sensitivity_recomputed':3,
        'timing_samples_verified':858,'timing_repetitions_each':3,'timing_unit_ns_to_ms_verified':True,'benchmark_rerun':False,
        'new_detector_forwards':0,'new_models_fitted':0,'new_thresholds':0,'full_AP_scores':0,'posthoc_diagnostic_only':True,
        'preflight_tests':9,'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False}
    write(OUT/'saved_delivery_audit.json',audit)
    print(json.dumps({k:audit[k] for k in ('status','manifest_members_verified','FP_details_independently_checked','timing_samples_verified')},ensure_ascii=True))


if __name__=='__main__':main()
