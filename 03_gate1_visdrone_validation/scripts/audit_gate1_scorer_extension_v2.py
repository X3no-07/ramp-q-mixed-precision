"""Read-only saved-artifact audit of the frozen scorer extension."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import gate1_scorer_extension as E
import audit_gate1_frozen_box_scorer as I
P=E.P;S=E.S;OUT=E.OUT;A=OUT/'attempt_01'
read,write,identity,require=P.read,P.write,P.identity,P.require


def state_hash(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        a=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def audit_progress(stage,done,total):
    P.FT.atomic_json(OUT/'audit_live_status.json',{'stage':stage,'done':done,'total':total,'at_utc':P.now()})


def main():
    torch.set_num_threads(1)
    c=read(OUT/'contract.json');complete=read(A/'completion_receipt.json');terminal=read(OUT/'launcher_terminal_receipt.json')
    require(complete['status']=='COMPLETE_FROZEN_SCORER_EXTENSION_DEVELOPMENT_ONLY' and complete['forward_images']==572,'Completion/budget incomplete')
    require(terminal['propagated_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and terminal['stderr_bytes']==0,'Abnormal worker terminal state')
    require(complete['contract']==identity(OUT/'contract.json'),'Contract changed')
    for b in c['source_bindings'].values():require(identity(b['path'])==b,'Source changed')
    for key in ('previous_contract','previous_delivery','previous_decision','scorer_model','candidate','architecture','evaluation_protocol','training_manifest','training_val_manifest','ranges','QAT_candidate','QAT_training_receipt','training_threshold_calibration','training_threshold_arrays'):
        require(identity(c[key]['path'])==c[key],'Input changed: '+key)
    for b in c['reused_files']+c['training_threshold_inputs']:require(identity(b['path'])==b,'Reused member changed')
    spec=read(OUT/'saved_audit_contract_v2.json')
    require(identity(spec['previous_audit_contract']['path'])==spec['previous_audit_contract'] and identity(spec['previous_failure']['path'])==spec['previous_failure'],'Preserved audit writer failure changed')
    require(identity(spec['script']['path'])==spec['script'] and identity(spec['independent_helpers']['path'])==spec['independent_helpers'],'Audit source changed')
    xml=ET.parse(OUT/'preflight_tests.xml').getroot()
    require(sum(int(x.get('tests',0)) for x in xml.iter('testsuite'))==8,'Missing qualification tests')
    require(sum(int(x.get(k,0)) for x in xml.iter('testsuite') for k in ('errors','failures','skipped'))==0,'Preflight not PASS')
    manifest=read(A/'manifest.json')['files']
    for b in manifest:require(identity(b['path'])==b,'Saved output changed')
    expected={str(p.resolve()) for p in A.rglob('*') if p.is_file() and p.name not in {'live_status.json','events.jsonl','manifest.json','completion_receipt.json'}}
    require(expected=={b['path'] for b in manifest} and len(expected)==len(manifest),'Manifest coverage differs')
    old=read(c['previous_contract']['path']);protocol=read(c['evaluation_protocol']['path'])
    require(E.partition_members(protocol['cohorts']['training_val'],old['evaluation_members'])==c['cohorts'],'Metadata-defined cohort changed')
    model=read(c['scorer_model']['path'])
    old_scores=[];labels=[];new_scores=[]
    for b in c['training_threshold_inputs']:
        with np.load(b['path'],allow_pickle=False) as d:
            old_scores.append(d['original_score'].copy());labels.append(d['target'].copy())
            new_scores.append(I.independently_scores(d['features'],d['original_score'],model))
    old_scores=np.concatenate(old_scores);labels=np.concatenate(labels);new_scores=np.concatenate(new_scores)
    with np.load(c['training_threshold_arrays']['path'],allow_pickle=False) as d:
        require(np.array_equal(old_scores,d['original_score']) and np.array_equal(labels,d['target']) and np.array_equal(new_scores,d['new_score']),'Training-only calibration arrays differ')
    calibration=read(c['training_threshold_calibration']['path'])
    budget=int(((labels==0)&(old_scores>=.25)).sum())
    traincurve=I.independent_curve(new_scores,(labels>0).astype(np.int8));good=np.flatnonzero(traincurve['FP']<=budget)
    require(len(good)>0 and float(traincurve['threshold'][good[-1]])==c['scorer_operating_threshold']==calibration['threshold'],'Frozen training-only threshold differs')
    require(not calibration['uses_development_labels'],'Calibration used development labels')
    frozen=read(A/'frozen_detector_state_check.json')
    require(frozen['equal'] and frozen['before']==frozen['after'],'Detector state mutated')
    payload=torch.load(c['QAT_candidate']['path'],map_location='cpu',weights_only=True)
    require(state_hash(payload['state_dict'])==frozen['before']['ordinary_qat'],'QAT state differs from frozen checkpoint')
    del payload
    canonical={};all_records={};gt_by_id={};root=Path(protocol['prepared_root']).resolve(strict=True)
    full_index={r['image_id']:i for i,r in enumerate(c['cohorts']['full414'])}
    for branch in E.BRANCHES:
        folder=A/'evaluation/full414'/branch
        with np.load(folder/'prepared_matrices.npz',allow_pickle=False) as d:
            canonical[branch]={row['image_id']:(d[f'gt_{i}'].copy(),d[f'dt_{i}'].copy()) for i,row in enumerate(c['cohorts']['full414'])}
        rows=read(folder/'saved_predictions_and_targets.json');all_records[branch]={r['image_id']:r for r in rows}
        require([r['image_id'] for r in rows]==[r['image_id'] for r in c['cohorts']['full414']],'Full414 submission order differs')
    sample_indices=set(np.linspace(0,285,32,dtype=int).tolist());raw_members=0;input_files=0
    for i,meta in enumerate(c['cohorts']['new286']):
        for kind in ('image','annotation'):
            p=(root/meta[kind+'_path']).resolve(strict=True)
            require(p.is_relative_to(root) and P.sha(p)==meta[kind+'_sha256'],'Authorized new input changed');input_files+=1
        gt=P.MATCH.parse_ground_truth((root/meta['annotation_path']).read_text(encoding='utf-8-sig'),meta['image_id']);gt_by_id[meta['image_id']]=gt
        folder=A/'new_inference'/f'{i:04d}';records=read(folder/'predictions.json')
        params=records['mse_r']['letterbox_params']
        with np.load(folder/'raw.npz',allow_pickle=False) as raw,np.load(folder/'scorer_features.npz',allow_pickle=False) as cached:
            for b in ('mse_r','ordinary_qat'):
                a=raw[b];require(a.shape==(1,14,8400) and a.dtype==np.float32 and np.isfinite(a).all(),'Raw output domain differs')
                require((a[:,2:4]>0).all() and (a[:,4:]>=0).all() and (a[:,4:]<=1).all(),'Raw geometry/score domain differs')
                reproduced=P.predictions(a,params,meta['image_id'],b)
                require(reproduced==records[b]['predictions'],'Saved NMS/restored predictions differ from raw')
                raw_members+=1
            rp=records['mse_r']['predictions'];sp=records['frozen_score_base']['predictions']
            feature=I.independently_features(raw['mse_r'],rp)
            require(np.allclose(feature,cached['features'],atol=1e-12,rtol=1e-12),'Independent model-visible feature reconstruction differs')
            expected_score=I.independently_scores(cached['features'],cached['original_score'],model)
            require(np.array_equal(expected_score,cached['new_score']),'Independent scorer replay differs')
            oldrows={r['raw_anchor_index']:r for r in rp};scores={r['raw_anchor_index']:float(s) for r,s in zip(rp,expected_score)}
            require(len(sp)==len(oldrows) and len({r['raw_anchor_index'] for r in sp})==len(oldrows),'Scored candidate pool changed')
            require(sp==sorted(sp,key=lambda p:(-p['score'],p['nms_input_order'])),'Score submission order differs')
            for row in sp:
                original=oldrows[row['raw_anchor_index']]
                require({k:v for k,v in row.items() if k!='score'}=={k:v for k,v in original.items() if k!='score'},'Scorer changed boxes/category/identity')
                require(row['score']==scores[row['raw_anchor_index']],'Scorer score differs')
        raw_gt=np.array([g['bbox_xywh']+[g['score'],g['category'],g['truncation'],g['occlusion']] for g in gt],float).reshape(-1,8)
        combined=[]
        for branch in E.BRANCHES:
            pred=records[branch]['predictions']
            combined.append(np.array([p['bbox_xywh']+[p['score'],p['category'],-1,-1] for p in pred],float).reshape(-1,8))
        prepared=P.AP.prepare_raw_image(raw_gt,np.concatenate(combined),width=meta['width'],height=meta['height'])
        k=len(canonical['mse_r'][meta['image_id']][1]);lengths=[k,k,len(prepared['filtered_predictions'])-2*k];offset=0
        for branch,length in zip(E.BRANCHES,lengths):
            original=canonical[branch][meta['image_id']]
            require(np.array_equal(original[0],prepared['prepared_gt']) and np.array_equal(original[1],prepared['filtered_predictions'][offset:offset+length]),'Canonical original GT/DT matrix differs')
            require(all_records[branch][meta['image_id']]==records[branch],'Full414 new record differs from original per-image output')
            offset+=length
            if i in sample_indices:
                threshold=c['scorer_operating_threshold'] if branch=='frozen_score_base' else .25
                operational=P.MATCH.match_project_targets(gt,records[branch]['predictions'],meta['width'],meta['height'],confidence=threshold,iou_threshold=.5)
                require(operational==records[branch]['project_operational'],'Sampled frozen-operating-threshold matching differs')
        if (i+1)%32==0:audit_progress('VERIFY_NEW_RAW_FEATURES_FIXED_BOXES_AND_MATRICES',i+1,286)
    for branch,oldroot in c['old_cohort_roots'].items():
        oldrows=read(Path(oldroot)/'saved_predictions_and_targets.json')
        with np.load(Path(oldroot)/'prepared_matrices.npz',allow_pickle=False) as d:
            for i,(meta,r) in enumerate(zip(c['cohorts']['old128'],oldrows)):
                require(np.array_equal(d[f'gt_{i}'],canonical[branch][meta['image_id']][0]) and np.array_equal(d[f'dt_{i}'],canonical[branch][meta['image_id']][1]),'Old128 matrices changed during combined aggregation')
                require(r['predictions']==all_records[branch][meta['image_id']]['predictions'],'Old128 prediction reuse differs')
    checks=[];cohort_decisions=read(A/'comparison_result.json')['cohorts']
    for cohort in ('new286','new_prefix47','full414'):
        curves={};official={};project={}
        for branch in E.BRANCHES:
            folder=A/'evaluation'/cohort/branch;job=read(folder/'AP_job.json');launch=read(folder/'AP_launcher_receipt.json');m=read(folder/'metrics.json')
            require(launch['observed_exit_code']==0 and m['torch_imported'] is False,'AP job terminal/runtime differs')
            require(job['matrices']==identity(folder/'prepared_matrices.npz')==m['matrices'],'AP input binding differs')
            require(job['images']==len(c['cohorts'][cohort]) and job['cohort']==cohort and job['branch']==branch,'AP job cohort differs')
            rows=read(folder/'saved_predictions_and_targets.json');nominal=Counter();operating=Counter();expected_score=[];expected_flag=[]
            require([r['image_id'] for r in rows]==[r['image_id'] for r in c['cohorts'][cohort]],'Cohort order/members differ')
            with np.load(folder/'prepared_matrices.npz',allow_pickle=False) as matrix:
                for i,r in enumerate(rows):
                    require(r==all_records[branch][r['image_id']],'Cohort image record differs from full414')
                    require(np.array_equal(matrix[f'gt_{i}'],canonical[branch][r['image_id']][0]) and np.array_equal(matrix[f'dt_{i}'],canonical[branch][r['image_id']][1]),'Cohort packed matrices differ')
                    nominal.update(r['project_conf025']['counts']);operating.update(r['project_operational']['counts'])
                for category in range(1,11):
                    for i in range(len(rows)):
                        gt=matrix[f'gt_{i}'];dt=matrix[f'dt_{i}'][:500]
                        _,match=P.AP.eval_res(gt[gt[:,5]==category,:5],dt[dt[:,5]==category,:5],.5)
                        expected_score.extend(match[:,4]);expected_flag.extend(match[:,5])
            pm=read(folder/'project_metrics.json')
            require(dict(nominal)==pm['conf025_counts'] and dict(operating)==pm['operating_counts'],'Project count aggregation differs')
            with np.load(folder/'AP_AR_arrays.npz',allow_pickle=False) as d:
                ap,ar,idx=d['AP'],d['AR'],d['evalClass']-1
                require(ap.shape==(10,10) and ar.shape==(10,10,4) and len(idx)>0,'Metric array domain differs')
                values=[np.mean(ap[idx]),np.mean(ap[idx,0]),np.mean(ap[idx,5]),*[np.mean(ar[idx,:,j]) for j in range(4)]]
                require(all(np.isclose(v,m['metrics_percent'][k],atol=1e-10,rtol=1e-12) for k,v in zip(P.AP.METRIC_NAMES,values)),'AP metric aggregate differs')
            with np.load(folder/'official_IoU05_rank_curve.npz',allow_pickle=False) as d:
                require(np.array_equal(d['score'],np.array(expected_score)) and np.array_equal(d['flag'],np.array(expected_flag)),'Official-style curve entries differ')
                curve=I.independent_curve(d['score'],d['flag'])
                require(all(np.array_equal(curve[k],d[k]) for k in curve),'Independent whole-tie curve differs')
                curves[branch]={**curve,'score':d['score'].copy(),'flag':d['flag'].copy()}
            official[branch]=m;project[branch]=pm
            checks.append({'cohort':cohort,'branch':branch,'AP_aggregates':7,'one_IoU_match_flags_rechecked':True,'whole_tie_curve_rechecked':True})
            audit_progress('VERIFY_NINE_JOBS_AND_CURVES',len(checks),9)
        base=curves['mse_r'];budget=int(((base['score']>=.25)&(base['flag']==0)).sum());target=int(((base['score']>=.25)&(base['flag']==1)).sum())
        decision=cohort_decisions[cohort]
        require(budget==decision['reference_FP_budget'] and target==decision['reference_TP_target'],'FP/TP reference formula differs')
        for branch in E.BRANCHES:
            require(S.at_fp_budget(curves[branch],budget)==decision['branches'][branch]['at_FP_budget'],'Fixed FP point differs')
            require(S.at_tp_target(curves[branch],target)==decision['branches'][branch]['at_TP_target'],'Fixed TP point differs')
        dap=official['frozen_score_base']['metrics_percent']['AP_all']-official['mse_r']['metrics_percent']['AP_all']
        dtp=decision['branches']['frozen_score_base']['at_FP_budget']['TP']-decision['branches']['mse_r']['at_FP_budget']['TP']
        require(dap==decision['score_AP_delta_vs_R_percentage_points'] and dtp==decision['score_TP_delta_at_same_FP_budget'],'Cohort primary deltas differ')
        require((dap>0 and dtp>=0)==decision['joint_development_rule_met'],'Primary rule result differs')
    result={'status':'PASS_FROZEN_EXTENSION_SOURCE_STATE_INPUT_MATRIX_AND_CURVE_AUDIT_ONLY','at_utc':P.now(),
        'audit_contract_v2':identity(OUT/'saved_audit_contract_v2.json'),'original_audit_writer_failure_preserved':True,
        'contract':identity(OUT/'contract.json'),'completion':identity(A/'completion_receipt.json'),'manifest_members_verified':len(manifest),
        'reused_members_verified':len(c['reused_files'])+len(c['training_threshold_inputs']),'new_images':286,'new_raw_arrays_verified':raw_members,
        'new_image_annotation_payloads_rehashed':input_files,'old128_reused_for_each_branch':True,'new_prefix_metadata_members_verified':47,
        'frozen_detector_state_fingerprints_unchanged':True,'QAT_state_equals_frozen_file':True,'scorer_bytes_unchanged':True,
        'features_independently_reconstructed':286,'scored_box_pools_unchanged_verified':286,'training_only_threshold_independently_recomputed':True,
        'operating_matching_probes':sorted(sample_indices),'AP_jobs_verified':checks,'seven_AP_metrics_reaggregated_each':True,
        'new_detector_forwards':572,'new_model_fit_count':0,'model_inference_repeated_in_audit':False,'full_AP_scoring_repeated_in_audit':False,
        'preflight_tests':8,'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False}
    write(OUT/'saved_delivery_audit.json',result);audit_progress('COMPLETE_SAVED_EXTENSION_AUDIT',1,1)
    print(json.dumps({k:result[k] for k in ('status','manifest_members_verified','new_raw_arrays_verified','AP_jobs_verified')},ensure_ascii=True))


if __name__=='__main__':main()
