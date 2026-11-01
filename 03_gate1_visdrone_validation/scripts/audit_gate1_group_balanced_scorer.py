"""Read-only audit of3scorer states,weights,thresholds,9metric jobs and sensitivity."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import gate1_group_balanced_scorer as G
import audit_gate1_frozen_box_scorer as I
P=G.P;S=G.S;T=G.T;E=G.E;J=G.J;OUT=G.OUT;A=OUT/'attempt_01'
read,write,identity,require=P.read,P.write,P.identity,P.require


def audit_progress(stage,done,total):
    P.FT.atomic_json(OUT/'audit_live_status.json',{'stage':stage,'done':done,'total':total,'at_utc':P.now()})


def main():
    P.torch.set_num_threads(1)
    c=read(OUT/'contract.json');complete=read(A/'completion_receipt.json');terminal=read(OUT/'launcher_terminal_receipt.json')
    require(complete['status']=='COMPLETE3ARM_PREFIX_FEATURE_SCORER_DEVELOPMENT_ABLATION' and complete['new_model_fit_count']==3 and complete['new_detector_forwards']==0,'Completion budget differs')
    require(terminal['observed_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and terminal['stderr_bytes']==0,'Abnormal worker terminal state')
    require(complete['contract']==identity(OUT/'contract.json'),'Contract changed')
    for b in c['source_bindings'].values():require(identity(b['path'])==b,'Frozen source changed')
    for key in ('previous_robustness_contract','previous_robustness_delivery','previous_threshold_contract','previous_threshold_delivery','previous_threshold_result','old_score_model','evaluation_protocol'):
        require(identity(c[key]['path'])==c[key],'Bound input changed: '+key)
    spec=read(OUT/'saved_audit_contract.json');require(identity(spec['script']['path'])==spec['script'] and identity(spec['score_replay_helper']['path'])==spec['score_replay_helper'],'Audit source changed')
    bound={}
    for b in c['fit_files']+c['reused_files']:bound[b['path']]=b
    for r in c['calibration_caches']:
        for key in ('features','predictions','matrices'):bound[r[key]['path']]=r[key]
    for r in c['development_features']:bound[r['features']['path']]=r['features']
    for b in bound.values():require(identity(b['path'])==b,'Saved input differs')
    xml=ET.parse(OUT/'preflight_tests.xml').getroot()
    require(sum(int(x.get('tests',0)) for x in xml.iter('testsuite'))==7,'Qualification count differs')
    require(sum(int(x.get(k,0)) for x in xml.iter('testsuite') for k in ('errors','failures','skipped'))==0,'Qualification not PASS')
    manifest=read(A/'manifest.json')['files']
    for b in manifest:require(identity(b['path'])==b,'Saved output differs')
    actual={str(p.resolve()) for p in A.rglob('*') if p.is_file() and p.name not in {'live_status.json','events.jsonl','manifest.json','completion_receipt.json'}}
    require(actual=={b['path'] for b in manifest} and len(actual)==len(manifest),'Output manifest coverage differs')
    xs=[];ps=[];ys=[];image=[]
    for i,b in enumerate(c['fit_files']):
        with np.load(b['path'],allow_pickle=False) as d:
            xs.append(d['features'].copy());ps.append(d['original_score'].copy());ys.append(d['target'].copy());image.append(np.full(len(d['target']),i,np.int64))
    x,p,y,image=np.concatenate(xs),np.concatenate(ps),np.concatenate(ys),np.concatenate(image);del xs,ps,ys
    require(x.shape==(213716,22) and len(y)==213716,'Training cache shape differs')
    mean,scale=S.standardize_training(x);prefixes=[T.prefix(r) for r in c['fit_members']];group_sizes=Counter(prefixes)
    require(dict(group_sizes)==c['group_distribution_metadata'] and len(group_sizes)==58,'Training group metadata differs')
    image_sizes=np.bincount(image);models={};weight_summaries={}
    for arm in G.ARMS:
        m=read(A/f'{arm}_model.json');models[arm]=m
        active=list(range(22)) if arm.endswith('full22') else list(G.QUALITY_INDICES);removed=sorted(set(range(22))-set(active))
        require(m['active_feature_indices']==active and m['active_parameter_slots']==len(active)+1 and m['steps']==100,'Model features/budget differ')
        require(np.array_equal(np.array(m['weight'])[removed],np.zeros(len(removed))),'Removed absolute/class feature learned a coefficient')
        require(np.array_equal(np.array(m['mean']),mean) and np.array_equal(np.array(m['scale']),scale),'Normalizer not trained only on fit cache')
        require([r['step'] for r in m['trace']]==[1,11,21,31,41,51,61,71,81,91,100] and all(np.isfinite(r['loss']) for r in m['trace']),'Optimization trace incomplete')
        expected_per_image=np.array([1/(58*group_sizes[prefixes[i]]*image_sizes[i]) if arm.startswith('prefix_balanced') else 1/(512*image_sizes[i]) for i in range(512)])
        with np.load(A/f'{arm}_row_weights.npz',allow_pickle=False) as d:w=d['weights'].copy()
        require(np.allclose(w,expected_per_image[image],atol=1e-18,rtol=1e-14) and np.isclose(w.sum(),1),'Independent row loss weights differ')
        image_weights=np.bincount(image,weights=w,minlength=512);group_weights={}
        for name in sorted(group_sizes):group_weights[name]=float(sum(image_weights[i] for i,prefix in enumerate(prefixes) if prefix==name))
        if arm.startswith('prefix_balanced'):require(all(np.isclose(v,1/58,atol=1e-12,rtol=1e-12) for v in group_weights.values()),'Prefix total weight not uniform')
        else:require(np.allclose(image_weights,np.full(512,1/512),atol=1e-12,rtol=1e-12),'Image total weight not uniform')
        predicted=I.independently_scores(x,p,m)
        require(np.array_equal(predicted,S.score_rows(x,p,m)),'Independent saved-score replay differs')
        weight_summaries[arm]={'image_weights':image_weights.tolist(),'prefix_weights':group_weights,'sum':float(w.sum()),'removed_feature_indices':removed,'removed_weights_all_zero':True}
    del x,p,y,image
    thresholds=read(A/'all3frozen_thresholds.json');events=[json.loads(line) for line in (A/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    stages=[r['status'] for r in events]
    require(stages.index('ALL3CALIBRATION_THRESHOLDS_FROZEN_BEFORE_DEVELOPER_SCORING')<stages.index('SCORE3FIXEDMODELS_ON414DEVELOPER_FEATURES'),'Developer scoring preceded threshold freeze')
    calgt=[];calt={arm:[] for arm in G.ARMS};root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
    for i,(meta,cached) in enumerate(zip(c['calibration_members'],c['calibration_caches'])):
        with np.load(cached['features']['path'],allow_pickle=False) as d:x=d['features'].copy();p=d['original_score'].copy()
        old=read(cached['predictions']['path'])['mse_r'];ann=(root/meta['annotation_path']).resolve(strict=True)
        require(ann.is_relative_to(root) and P.sha(ann)==meta['annotation_sha256'],'Calibration GT changed')
        gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id']);gt8=np.array([g['bbox_xywh']+[g['score'],g['category'],g['truncation'],g['occlusion']] for g in gt],float).reshape(-1,8)
        arrays=[]
        for arm,m in models.items():
            values=I.independently_scores(x,p,m);pred=[{**d,'score':float(v)} for d,v in zip(old,values)];pred=sorted(pred,key=lambda d:(-d['score'],d['nms_input_order']))
            arrays.append(np.array([d['bbox_xywh']+[d['score'],d['category'],-1,-1] for d in pred],float).reshape(-1,8))
        prepared=P.AP.prepare_raw_image(gt8,np.concatenate(arrays),width=meta['width'],height=meta['height'])
        with np.load(A/'calibration'/f'{i:04d}_matrices.npz',allow_pickle=False) as d:
            require(np.array_equal(prepared['prepared_gt'],d['gt']),'Calibration original GT transport differs')
            k=len(d[G.ARMS[0]]);require(len(prepared['filtered_predictions'])==3*k,'Calibration pool count differs');calgt.append(d['gt'].copy())
            for j,arm in enumerate(G.ARMS):
                require(np.array_equal(prepared['filtered_predictions'][j*k:(j+1)*k],d[arm]),'Independent calibration rescore/matrix differs');calt[arm].append(d[arm].copy())
        if (i+1)%32==0:audit_progress('VERIFY3CALIBRATION_MODELS_AND_MATRICES',i+1,178)
    with np.load(T.OUT/'attempt_01/calibration/reference_rank_entries.npz',allow_pickle=False) as d:reference_budget=int(((d['score']>=.25)&(d['flag']==0)).sum())
    for arm in G.ARMS:
        score,flag=J.rank_entries(calgt,calt[arm])
        with np.load(A/'calibration'/f'{arm}_entries.npz',allow_pickle=False) as d:
            require(np.array_equal(score,d['score']) and np.array_equal(flag,d['flag']),'Calibration single-IoU flags differ')
        curve=I.independent_curve(score,flag);eligible=np.flatnonzero(curve['FP']<=reference_budget)
        tau=float(curve['threshold'][eligible[-1]]) if len(eligible) else 1.
        require(tau==thresholds[arm]['threshold'] and not thresholds[arm]['uses_training_val_labels'],'Calibration-only threshold differs')
    base_rows=read(E.OUT/'attempt_01/evaluation/full414/mse_r/saved_predictions_and_targets.json');base={r['image_id']:r for r in base_rows}
    full={arm:read(A/'evaluation/full414'/arm/'saved_predictions_and_targets.json') for arm in G.ARMS}
    prepared_by_arm={arm:{} for arm in G.ARMS};probe=set(np.linspace(0,413,32,dtype=int).tolist())
    for i,(meta,feature) in enumerate(zip(c['cohorts']['full414'],c['development_features'])):
        image_id=meta['image_id'];old=base[image_id]['predictions']
        with np.load(feature['features']['path'],allow_pickle=False) as d:x=d['features'].copy();p=d['original_score'].copy()
        require(np.array_equal(p,np.array([r['score'] for r in old])),'Development feature row identity differs')
        ann=(root/meta['annotation_path']).resolve(strict=True);require(ann.is_relative_to(root) and P.sha(ann)==meta['annotation_sha256'],'Development annotation changed')
        gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),image_id);gt8=np.array([g['bbox_xywh']+[g['score'],g['category'],g['truncation'],g['occlusion']] for g in gt],float).reshape(-1,8)
        arrays=[]
        for arm,m in models.items():
            row=full[arm][i];require(row['image_id']==image_id and row['frozen_threshold']==thresholds[arm]['threshold'],'Developer membership/threshold differs')
            values=I.independently_scores(x,p,m);expected=[{**d,'score':float(v)} for d,v in zip(old,values)];expected=sorted(expected,key=lambda r:(-r['score'],r['nms_input_order']))
            require(expected==row['predictions'],'Independent score/order or unchanged box/category identity differs')
            arrays.append(np.array([r['bbox_xywh']+[r['score'],r['category'],-1,-1] for r in expected],float).reshape(-1,8))
            if i in probe:
                matching=P.MATCH.match_project_targets(gt,expected,meta['width'],meta['height'],confidence=thresholds[arm]['threshold'],iou_threshold=.5)
                require(matching==row['project_operational'],'Frozen-threshold project matching probe differs')
        prep=P.AP.prepare_raw_image(gt8,np.concatenate(arrays),width=meta['width'],height=meta['height'])
        require(len(prep['filtered_predictions'])%3==0,'Developer pool count differs');k=len(prep['filtered_predictions'])//3
        for j,arm in enumerate(G.ARMS):prepared_by_arm[arm][image_id]=(prep['prepared_gt'],prep['filtered_predictions'][j*k:(j+1)*k])
        if (i+1)%64==0:audit_progress('VERIFY3DEVELOPER_SCORE_OUTPUTS_AND_MATRICES',i+1,414)
    result=read(A/'comparison_result.json');job_checks=[]
    for cohort in G.COHORTS:
        for arm in G.ARMS:
            folder=A/'evaluation'/cohort/arm;rows=read(folder/'saved_predictions_and_targets.json');pm=read(folder/'project_metrics.json');m=read(folder/'metrics.json')
            require([r['image_id'] for r in rows]==[r['image_id'] for r in c['cohorts'][cohort]],'AP member/order differs')
            require(read(folder/'AP_launcher_receipt.json')['observed_exit_code']==0 and m['torch_imported'] is False,'AP job exit/runtime differs')
            require(m['matrices']==identity(folder/'prepared_matrices.npz'),'AP input binding differs')
            project=Counter();project_rows=[];gt_images=[];dt_images=[]
            with np.load(folder/'prepared_matrices.npz',allow_pickle=False) as d:
                for i,row in enumerate(rows):
                    image_id=row['image_id'];original=base[image_id]['project_conf025']['counts'];counts=row['project_operational']['counts']
                    project.update(counts);project_rows.append({'prefix':row['prefix_proxy'],'TP':counts['hits']-original['hits'],'FP':counts['false_positive_detections']-original['false_positive_detections']})
                    require(np.array_equal(d[f'gt_{i}'],prepared_by_arm[arm][image_id][0]) and np.array_equal(d[f'dt_{i}'],prepared_by_arm[arm][image_id][1]),'Canonical GT/DT transport differs')
                    gt_images.append(d[f'gt_{i}'].copy());dt_images.append(d[f'dt_{i}'].copy())
            require(dict(project)==pm['counts'],'Developer project aggregate differs')
            with np.load(folder/'AP_AR_arrays.npz',allow_pickle=False) as d:
                ap,ar,idx=d['AP'],d['AR'],d['evalClass']-1
                require(ap.shape==(10,10) and ar.shape==(10,10,4) and len(idx)>0,'AP array shape differs')
                values=[np.mean(ap[idx]),np.mean(ap[idx,0]),np.mean(ap[idx,5]),*[np.mean(ar[idx,:,j]) for j in range(4)]]
                require(all(np.isclose(v,m['metrics_percent'][k],atol=1e-10,rtol=1e-12) for k,v in zip(P.AP.METRIC_NAMES,values)),'Seven AP/AR metric aggregates differ')
            expected_score,expected_flag=J.rank_entries(gt_images,dt_images)
            with np.load(folder/'official_IoU05_rank_curve.npz',allow_pickle=False) as d:
                require(np.array_equal(expected_score,d['score']) and np.array_equal(expected_flag,d['flag']),'Official IoU0.5 flag replay differs')
                curve=I.independent_curve(d['score'],d['flag']);require(all(np.array_equal(d[k],curve[k]) for k in curve),'Whole-tie curve differs')
                counted=T.fixed_counts(d['score'],d['flag'],thresholds[arm]['threshold'])
            r=result['cohorts'][cohort]['arms'][arm]
            require(r['official_fixed_counts']==counted and r['project_counts']==dict(project) and r['AP_percent']==m['metrics_percent']['AP_all'],'Reported AP/endpoint differs')
            group=defaultdict(lambda:[0,0,0])
            for row in project_rows:
                v=group[row['prefix']];v[0]+=row['TP'];v[1]+=row['FP'];v[2]+=1
            tp=sum(v[0] for v in group.values());fp=sum(v[1] for v in group.values());saved=r['prefix_sensitivity']
            leaves=[{'omitted_prefix_proxy':p,'omitted_images':n,'remaining_TP_delta':tp-a,'remaining_FP_delta':fp-b,'remaining_joint_observation':tp-a>=0 and fp-b<=0} for p,(a,b,n) in sorted(group.items())]
            require(leaves==saved['leave_one_prefix_out'] and saved['pooled_TP_delta']==tp and saved['pooled_FP_delta']==fp,'Independent prefix sensitivity differs')
            ref=result['cohorts'][cohort]['reference_R_official_counts'];apref=result['cohorts'][cohort]['reference_R_AP']
            require(r['primary_style_joint_AP_and_endpoint_observation']==(r['AP_percent']>apref and counted['TP']>=ref['TP'] and counted['FP']<=ref['FP']),'Primary observation flag differs')
            job_checks.append({'cohort':cohort,'arm':arm,'AP_aggregates_verified':7,'endpoint_and_whole_tie_curve_verified':True,'prefix_removals_verified':len(leaves)})
            audit_progress('VERIFY9AP_JOBS_AND_FIXED_RULES',len(job_checks),9)
    audit={'status':'PASS3SCORER_GROUP_WEIGHT_FEATURE_CALIBRATION_AND_METRIC_AUDIT_ONLY','at_utc':P.now(),'contract':identity(OUT/'contract.json'),
        'completion':identity(A/'completion_receipt.json'),'manifest_members_verified':len(manifest),'input_cache_bindings_verified':len(bound),
        'fit_rows':213716,'fit_images':512,'fit_prefix_proxies':58,'calibration_images':178,'developer_images':414,
        'models_verified':3,'updates_each':100,'normalization_training_only_exact':True,'weight_summaries':weight_summaries,
        'calibration_thresholds_independently_recomputed':3,'thresholds_frozen_before_developer_scoring':True,
        'all3developer_candidate_pool_and_scores_verified':True,'project_matching_probe_indices':sorted(probe),'AP_jobs_verified':job_checks,
        'preflight_tests':7,'model_fit_repeated_in_audit':False,'new_detector_forwards':0,'full_AP_scoring_repeated_in_audit':False,
        'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False}
    write(OUT/'saved_delivery_audit.json',audit);audit_progress('COMPLETE3SCORER_SAVED_AUDIT',1,1)
    print(json.dumps({k:audit[k] for k in ('status','manifest_members_verified','input_cache_bindings_verified','calibration_thresholds_independently_recomputed')},ensure_ascii=True))


if __name__=='__main__':main()
