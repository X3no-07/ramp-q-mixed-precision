"""Saved-only calibration, threshold freeze, and developer endpoint audit."""
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import gate1_separate_threshold as T
import audit_gate1_frozen_box_scorer as I
P=T.P;S=T.S;E=T.E;J=T.J;OUT=T.OUT;A=OUT/'attempt_01'
read,write,identity,require=P.read,P.write,P.identity,P.require


def progress(stage,done,total):
    P.FT.atomic_json(OUT/'audit_live_status.json',{'stage':stage,'done':done,'total':total,'at_utc':P.now()})


def counts_independent(score,flag,threshold):
    values=[int(v) for s,v in zip(score,flag) if float(s)>=threshold]
    return {'TP':values.count(1),'FP':values.count(0),'ignored':values.count(-1)}


def main():
    torch.set_num_threads(1)
    c=read(OUT/'contract.json');complete=read(A/'completion_receipt.json');terminal=read(OUT/'launcher_terminal_receipt.json')
    require(complete['status']=='COMPLETE_SEPARATE_THRESHOLD_CALIBRATION_DEVELOPMENT_CHECK' and complete['forward_images']==178,'Completion/budget incomplete')
    require(terminal['propagated_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and terminal['stderr_bytes']==0,'Abnormal terminal state')
    require(complete['contract']==identity(OUT/'contract.json'),'Contract changed')
    for b in c['source_bindings'].values():require(identity(b['path'])==b,'Frozen source changed')
    for key in ('previous_contract','previous_delivery','previous_decision','scorer_model','candidate','architecture','ranges','evaluation_protocol','training_manifest','training_val_manifest','assignment_metadata'):
        require(identity(c[key]['path'])==c[key],'Bound source changed: '+key)
    for b in c['reused_files']:require(identity(b['path'])==b,'Reused developer artifact changed')
    spec=read(OUT/'saved_audit_contract.json')
    for key in ('script','independent_helpers'):
        require(identity(spec[key]['path'])==spec[key],'Audit source changed')
    xml=ET.parse(OUT/'preflight_tests.xml').getroot()
    require(sum(int(x.get('tests',0)) for x in xml.iter('testsuite'))==8,'Missing tests')
    require(sum(int(x.get(k,0)) for x in xml.iter('testsuite') for k in ('errors','failures','skipped'))==0,'Preflight not PASS')
    manifest=read(A/'manifest.json')['files']
    for b in manifest:require(identity(b['path'])==b,'Saved output changed')
    expected={str(p.resolve()) for p in A.rglob('*') if p.is_file() and p.name not in {'live_status.json','events.jsonl','manifest.json','completion_receipt.json'}}
    require(expected=={b['path'] for b in manifest} and len(expected)==len(manifest),'Output coverage differs')
    train=read(c['training_manifest']['path'])['samples']
    chosen=T.choose_calibration_members(train,c['scorer_fit_members'])
    require(chosen==c['calibration_members'] and len(chosen)==178 and len({T.prefix(r) for r in chosen})==34,'Calibration cohort differs')
    ids={r['image_id'] for r in chosen};fitids={r['image_id'] for r in c['scorer_fit_members']};dev=c['evaluation_cohorts']['full414']
    require(ids.isdisjoint(fitids) and ids.isdisjoint(r['image_id'] for r in dev),'Image ID overlap')
    chash={r['image']['sha256'] for r in chosen};cprefix={T.prefix(r) for r in chosen}
    require(chash.isdisjoint(r['image']['sha256'] for r in c['scorer_fit_members']) and chash.isdisjoint(r['image_sha256'] for r in dev),'Image content overlap')
    require(cprefix.isdisjoint(T.prefix(r) for r in c['scorer_fit_members']) and cprefix.isdisjoint(r['prefix_proxy'] for r in dev),'Prefix proxy overlap')
    frozen=read(A/'frozen_state_check.json');require(frozen['before']==frozen['after'] and frozen['equal'] and frozen['scorer']==c['scorer_model'],'Frozen detector/scorer state differs')
    root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True);model=read(c['scorer_model']['path'])
    gt_images=[];r_images=[];s_images=[];allowed_inputs=0
    for i,meta in enumerate(c['calibration_metadata']):
        require(meta['image_id']==chosen[i]['image_id'] and meta['assigned_split']=='train','Calibration source role/order differs')
        for kind in ('image','annotation'):
            path=(root/meta[kind+'_path']).resolve(strict=True)
            require(path.is_relative_to(root) and P.sha(path)==meta[kind+'_sha256'],'Allowed calibration member changed');allowed_inputs+=1
        folder=A/'calibration'/f'{i:04d}';record=read(folder/'predictions.json')
        require(record['image_id']==meta['image_id'],'Calibration saved identity differs')
        with np.load(folder/'raw_and_features.npz',allow_pickle=False) as raw,np.load(folder/'prepared_matrices.npz',allow_pickle=False) as matrix:
            a=raw['raw'];require(a.shape==(1,14,8400) and a.dtype==np.float32 and np.isfinite(a).all(),'Raw domain differs')
            pred=P.predictions(a,record['letterbox_params'],meta['image_id'],'calibration_R')
            require(pred==record['mse_r'],'NMS/restored prediction replay differs')
            feature=I.independently_features(a,pred)
            require(np.allclose(feature,raw['features'],atol=1e-12,rtol=1e-12),'Independent model-visible features differ')
            require(np.array_equal(raw['original_score'],np.array([p['score'] for p in pred])),'Original score differs')
            score=I.independently_scores(raw['features'],raw['original_score'],model)
            require(np.array_equal(score,raw['new_score']),'Frozen scorer replay differs')
            old={r['raw_anchor_index']:r for r in pred};scores={r['raw_anchor_index']:float(s) for r,s in zip(pred,score)}
            require(len(record['scored'])==len(old) and len({r['raw_anchor_index'] for r in record['scored']})==len(old),'Candidate pool changed')
            require(record['scored']==sorted(record['scored'],key=lambda r:(-r['score'],r['nms_input_order'])),'Submission sort differs')
            for row in record['scored']:
                r=old[row['raw_anchor_index']]
                require({k:v for k,v in row.items() if k!='score'}=={k:v for k,v in r.items() if k!='score'} and row['score']==scores[row['raw_anchor_index']],'Frozen boxes/categories/score differ')
            gt=P.MATCH.parse_ground_truth((root/meta['annotation_path']).read_text(encoding='utf-8-sig'),meta['image_id'])
            g=np.array([x['bbox_xywh']+[x['score'],x['category'],x['truncation'],x['occlusion']] for x in gt],float).reshape(-1,8)
            combined=[]
            for key in ('mse_r','scored'):
                combined.append(np.array([x['bbox_xywh']+[x['score'],x['category'],-1,-1] for x in record[key]],float).reshape(-1,8))
            prep=P.AP.prepare_raw_image(g,np.concatenate(combined),width=meta['width'],height=meta['height']);count=len(matrix['mse_r'])
            require(len(matrix['scored'])==count and len(prep['filtered_predictions'])==2*count,'Filtered pool count differs')
            require(np.array_equal(matrix['gt'],prep['prepared_gt']) and np.array_equal(matrix['mse_r'],prep['filtered_predictions'][:count]) and np.array_equal(matrix['scored'],prep['filtered_predictions'][count:]),'Original source/filtered matrix differs')
            gt_images.append(matrix['gt'].copy());r_images.append(matrix['mse_r'].copy());s_images.append(matrix['scored'].copy())
        if (i+1)%32==0:progress('VERIFY178_RAW_FEATURES_AND_MATRICES',i+1,178)
    rs,rf=J.rank_entries(gt_images,r_images);ss,sf=J.rank_entries(gt_images,s_images)
    for name,score,flag in (('reference_rank_entries',rs,rf),('scored_rank_entries',ss,sf)):
        with np.load(A/'calibration'/f'{name}.npz',allow_pickle=False) as d:
            require(np.array_equal(score,d['score']) and np.array_equal(flag,d['flag']),'Single-IoU matching entries differ')
            curve=I.independent_curve(score,flag)
            require(all(np.array_equal(d[k],curve[k]) for k in curve),'Independent whole-tie curve differs')
    threshold=read(A/'frozen_threshold.json');reference=counts_independent(rs,rf,.25)
    curve=I.independent_curve(ss,sf);eligible=np.flatnonzero(curve['FP']<=reference['FP'])
    expected_tau=float(curve['threshold'][eligible[-1]]) if len(eligible) else 1.
    require(threshold['threshold']==expected_tau and not threshold['uses_training_val_labels'],'Frozen threshold was not selected only from calibration')
    require(threshold['reference_calibration_counts']==reference and threshold['chosen_calibration_counts']==counts_independent(ss,sf,expected_tau),'Calibration endpoints differ')
    events=[json.loads(line) for line in (A/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    stages=[r['status'] for r in events]
    require(stages.index('SCORER_DISJOINT_CALIBRATION_THRESHOLD_FROZEN')<stages.index('APPLY_FROZEN_THRESHOLD_TO_REUSED_DEVELOPMENT_OUTPUTS'),'Developer comparison preceded threshold freeze')
    result=read(A/'comparison_result.json');require(result['threshold']==expected_tau and result['new_full_AP_scores']==0 and result['model_fits']==0,'Budget/threshold differs')
    matches=read(A/'developer_project_threshold_matches.json')
    require([r['image_id'] for r in matches]==[r['image_id'] for r in dev],'Developer member/order differs')
    indexed={r['image_id']:r for r in matches};previous=Path(c['prior_extension_root'])
    scored=read(previous/'attempt_01/evaluation/full414/frozen_score_base/saved_predictions_and_targets.json')
    probes=set(np.linspace(0,413,32,dtype=int).tolist())
    for i,meta in enumerate(dev):
        ann=(root/meta['annotation_path']).resolve(strict=True)
        require(ann.is_relative_to(root) and P.sha(ann)==meta['annotation_sha256'],'Developer annotation identity changed');allowed_inputs+=1
        require(matches[i]['threshold']==expected_tau,'Per-image threshold differs')
        if i in probes:
            gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id'])
            replay=P.MATCH.match_project_targets(gt,scored[i]['predictions'],meta['width'],meta['height'],confidence=expected_tau,iou_threshold=.5)
            require(replay==matches[i]['matching'],'Sampled actual threshold target-matching differs')
    endpoint_checks=[]
    for name in ('new286','new_prefix47','full414'):
        cohort=result['cohorts'][name]
        for branch in E.BRANCHES:
            folder=previous/'attempt_01/evaluation'/name/branch;point=cohort['branches'][branch]
            tau=expected_tau if branch=='frozen_score_base' else .25
            require(point['threshold']==tau,'Developer threshold changed')
            with np.load(folder/'official_IoU05_rank_curve.npz',allow_pickle=False) as d:
                counted=counts_independent(d['score'],d['flag'],tau)
            require(point['official_IoU05_cap500_counts']==counted,'Fixed official threshold count differs')
            require(point['AP_percent_unchanged']==read(folder/'metrics.json')['metrics_percent']['AP_all'],'Fixed threshold changed reused AP')
            if branch=='frozen_score_base':
                count=Counter()
                for meta in c['evaluation_cohorts'][name]:count.update(indexed[meta['image_id']]['matching']['counts'])
                require(point['project_counts']==dict(count),'Fixed project-threshold aggregation differs')
            else:require(point['project_counts']==read(folder/'project_metrics.json')['conf025_counts'],'Reference project counts differ')
            endpoint_checks.append({'cohort':name,'branch':branch,'fixed_official_counts':counted})
        r=cohort['branches']['mse_r']['official_IoU05_cap500_counts'];s=cohort['branches']['frozen_score_base']['official_IoU05_cap500_counts']
        require(cohort['official_TP_delta_vs_R']==s['TP']-r['TP'] and cohort['official_FP_delta_vs_R']==s['FP']-r['FP'],'Primary deltas differ')
        require(cohort['official_fixed_threshold_joint_rule_met']==(s['TP']>=r['TP'] and s['FP']<=r['FP']),'Joint observation differs')
    audit={'status':'PASS_SEPARATE_THRESHOLD_SOURCE_IDENTITY_SELECTION_AND_ENDPOINT_AUDIT_ONLY','at_utc':P.now(),
        'contract':identity(OUT/'contract.json'),'completion':identity(A/'completion_receipt.json'),'manifest_members_verified':len(manifest),
        'reused_members_verified':len(c['reused_files']),'calibration_members':178,'prefix_proxies':34,'scorer_fit_calibration_id_hash_prefix_overlap':0,
        'development_calibration_id_hash_prefix_overlap':0,'raw_arrays_verified':178,'features_independently_reconstructed':178,
        'allowed_input_payloads_rehashed':allowed_inputs,'frozen_scorer_and_detector_state_unchanged':True,'threshold_independently_recomputed':True,
        'threshold_frozen_before_developer_comparison':True,'endpoint_checks':endpoint_checks,'project_target_matching_probes':sorted(probes),
        'preflight_tests':8,'new_detector_forwards':178,'new_model_fits':0,'new_full_AP_scores':0,
        'model_inference_repeated_in_audit':False,'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False}
    write(OUT/'saved_delivery_audit.json',audit);progress('COMPLETE_SEPARATE_THRESHOLD_AUDIT',1,1)
    print(json.dumps({k:audit[k] for k in ('status','manifest_members_verified','raw_arrays_verified','threshold_independently_recomputed')},ensure_ascii=True))


if __name__=='__main__':main()
