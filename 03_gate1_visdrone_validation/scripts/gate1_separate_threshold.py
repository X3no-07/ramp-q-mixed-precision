"""Single threshold from scorer-disjoint training members; frozen developer checks.

No scorer fitting, no new AP scoring, no Selection/Holdout payload access.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/gate1/20261026_separate_threshold178_v1'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'settings')
import numpy as np
import torch
import gate1_scorer_extension as E
import gate1_cached_ap_job as J
P=E.P;S=E.S
read,write,identity,require,now,sha=P.read,P.write,P.identity,P.require,P.now,P.sha


def prefix(row):return row['image_id'].split('/')[-1].split('_')[0]


def choose_calibration_members(train,fit):
    fitids={r['image_id'] for r in fit};fitprefix={prefix(r) for r in fit}
    require(len({r['image_id'] for r in train})==len(train),'Duplicate training member')
    members=sorted((r for r in train if r['image_id'] not in fitids and prefix(r) not in fitprefix),key=lambda r:r['image_id'])
    require(not ({r['image']['sha256'] for r in members}&{r['image']['sha256'] for r in fit}),'Calibration/fit image hash overlap')
    return members


def fixed_counts(score,flag,threshold):
    require(np.isfinite(score).all() and np.isin(flag,[-1,0,1]).all() and len(score)==len(flag),'Invalid curve entries')
    kept=score>=threshold
    return {'TP':int((kept&(flag==1)).sum()),'FP':int((kept&(flag==0)).sum()),'ignored':int((kept&(flag==-1)).sum())}


def select_threshold(original_score,original_flag,new_score,new_flag):
    ref=fixed_counts(original_score,original_flag,.25)
    curve=J.curve_from_entries(new_score,new_flag)
    point=S.at_fp_budget(curve,ref['FP'])
    if point['threshold'] is None:point={'TP':0,'FP':0,'threshold':1.0}
    actual=fixed_counts(new_score,new_flag,point['threshold'])
    require(actual['FP']<=ref['FP'],'Calibrated FP budget exceeded')
    return {'threshold':point['threshold'],'reference_threshold':.25,'reference_calibration_counts':ref,
            'chosen_calibration_counts':actual,'calibration_TP_delta':actual['TP']-ref['TP'],'calibration_FP_delta':actual['FP']-ref['FP'],
            'rule':'Lowest whole-score-group threshold with officialIoU0.5 cap500 calibrationFP<=R_at0.25FP; all178fixed calibration members; one threshold; no safety factor or new feature/scorer fitting',
            'score_ties_split':False,'threshold_is_frozen_before_development_counts':True,'uses_training_val_labels':False}


def sources():
    bindings=deepcopy(E.source_bindings())
    for path in (Path(__file__),ROOT/'scripts/launch_gate1_separate_threshold.py',ROOT/'tests/test_gate1_separate_threshold.py'):
        bindings[path.name]=identity(path)
    return bindings


def prepare():
    require(not (OUT/'contract.json').exists(),'Threshold contract already frozen')
    extension=read(E.OUT/'contract.json');scoring=read(S.OUT/'contract.json')
    train=read(extension['training_manifest']['path']);val=read(extension['training_val_manifest']['path']);P.FT.validate_membership(train,val)
    chosen=choose_calibration_members(train['samples'],scoring['training_members'])
    require(len(chosen)==178 and len({prefix(r) for r in chosen})==34,'Unexpected metadata-defined calibration members')
    selected={r['image_id']:r for r in chosen};metadata={}
    assignment=Path(scoring['training_annotation_assignment']['path'])
    for line in assignment.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['image_id'] in selected:
            require(row['assigned_split']=='train' and row['image_sha256']==selected[row['image_id']]['image']['sha256'],'Calibration source role/hash differs')
            metadata[row['image_id']]=row
    require(set(metadata)==set(selected),'Missing calibration metadata')
    full=extension['cohorts']['full414']
    require({r['image_sha256'] for r in full}.isdisjoint(r['image']['sha256'] for r in chosen),'Calibration/development image overlap')
    require({r['prefix_proxy'] for r in full}.isdisjoint(prefix(r) for r in chosen),'Calibration/development filename-prefix overlap')
    reused=[]
    for cohort in ('new286','new_prefix47','full414'):
        for branch in E.BRANCHES:
            folder=E.OUT/'attempt_01/evaluation'/cohort/branch
            for name in ('metrics.json','project_metrics.json','official_IoU05_rank_curve.npz'):
                reused.append(identity(folder/name))
    for branch in E.BRANCHES:
        reused.append(identity(E.OUT/'attempt_01/evaluation/full414'/branch/'saved_predictions_and_targets.json'))
    c={'schema':'rampq.gate1.separate_scorer_threshold/v1','created_at_utc':now(),
       'authorization':'User: Get this done quickly; execute separate calibration, freeze thresholds, and check existing development results, preserving all outputs',
       'source_bindings':sources(),'framework':P.verify_framework_sources(),
       'previous_contract':identity(E.OUT/'contract.json'),'previous_delivery':identity(E.OUT/'delivery_validation.json'),
       'previous_decision':identity(E.OUT/'machine_decision.json'),'scorer_model':extension['scorer_model'],
       'candidate':extension['candidate'],'architecture':extension['architecture'],'ranges':extension['ranges'],
       'evaluation_protocol':extension['evaluation_protocol'],'training_manifest':extension['training_manifest'],'training_val_manifest':extension['training_val_manifest'],
       'assignment_metadata':identity(assignment),'calibration_members':chosen,'calibration_metadata':[metadata[r['image_id']] for r in chosen],
       'scorer_fit_members':scoring['training_members'],'evaluation_cohorts':extension['cohorts'],
       'calibration_count':178,'calibration_prefix_proxies':34,'scorer_fit_calibration_id_hash_prefix_overlap':0,
       'development_calibration_id_hash_prefix_overlap':0,'prefix_is_verified_scene_identity':False,
       'calibration_images_seen_in_original_detector_training':True,'calibration_not_used_for_scorer_parameter_fit':True,
       'calibration_selection':'ALL remaining formal-training members whose filename-prefix proxy is absent from512scorer-fit images; alphabetical image_id order, chosen by metadata before reading predictions/labels',
       'threshold_rule':'One threshold: maximum TP prefix / lowest whole-score-group threshold with calibrationFP<=originalR@0.25FP, officialIoU0.5,cap500. No safety-margin sweep, no class-specific thresholds, no retries based on test outcome.',
       'primary_evaluation':'new286 fixed-threshold officialTP>=R@0.25TP and officialFP<=R@0.25FP; project same-target matching also reported. No training_val labels used to choose threshold.',
       'supporting_evaluation':['new_prefix47','full414'],'all_cohorts_reported':True,
       'strong_comparator':'existingordinaryQAT@0.25, allsavedpredictions/curves reused; no extra QAT forward',
       'model_fit_count':0,'new_detector_image_forward_budget':178,'real_image_forward_budget':178,'full_AP_score_count':0,
       'scorer_changed':False,'raw_outputs_numeric_qualification':'NOT_RUN; original all-anchorFAIL retained',
       'reused_files':reused,'prior_extension_root':str(E.OUT.resolve()),'baseline_threshold':.25,
       'scientific_acceptance':False,'independent_test_set':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
       'timeout_seconds':2400,'automatic_retry':False,
       'method_reference':{'url':'https://scikit-learn.org/stable/modules/classification_threshold.html','principle':'Threshold validation members separate from score-model fitting; no claim of exact sklearn reproduction'}}
    write(OUT/'contract.json',c)
    print(json.dumps({'status':'FROZEN_SCORER_DISJOINT_THRESHOLD_CHECK','contract':identity(OUT/'contract.json'),'calibration_images':178,'prefix_proxies':34,'detector_forwards':178,'new_scorer_fits':0},ensure_ascii=True))


class Worker(P.Worker):
    def __init__(self):
        self.contract=read(OUT/'contract.json');self.attempt=OUT/'attempt_01';self.attempt.mkdir(exist_ok=False)
        self.started=time.monotonic();self.live={'pid':os.getpid(),'parent_pid':os.getppid(),'started_at_utc':now(),'status':'STARTING',
            'contract':identity(OUT/'contract.json'),'forward_images':0,'new_model_fit_count':0}
        self.progress('STARTING',0,1)

    def verify(self):
        c=self.contract;require(c['source_bindings']==sources(),'Frozen sources changed')
        require(c['framework']==P.verify_framework_sources(),'Framework changed')
        for key in ('previous_contract','previous_delivery','previous_decision','scorer_model','candidate','architecture','ranges','evaluation_protocol','training_manifest','training_val_manifest','assignment_metadata'):
            require(identity(c[key]['path'])==c[key],'Bound input changed: '+key)
        for b in c['reused_files']:require(identity(b['path'])==b,'Reused developer output changed')

    def calibration(self,root):
        c=self.contract;maxima=read(c['ranges']['path'])['max_abs'];scorer=read(c['scorer_model']['path'])
        model=P.new_model(maxima);P.precision(model,'r_only:G1');before=E.state_value_fingerprint(model)
        output=self.attempt/'calibration';output.mkdir()
        gt_images=[];r_images=[];s_images=[]
        for i,meta in enumerate(c['calibration_metadata']):
            folder=output/f'{i:04d}';folder.mkdir()
            tensor,params=self.image_input(root,meta);raw=self.forward(model,tensor)
            rp=P.predictions(raw,params,meta['image_id'],'calibration_R');feature=S.visible_features(raw,rp)
            prob=np.array([p['score'] for p in rp]);score=S.score_rows(feature,prob,scorer)
            sp=deepcopy(rp)
            for p,s in zip(sp,score):p['score']=float(s)
            sp=sorted(sp,key=lambda p:(-p['score'],p['nms_input_order']))
            ann=(root/meta['annotation_path']).resolve(strict=True)
            require(ann.is_relative_to(root) and sha(ann)==meta['annotation_sha256'],'Calibration annotation changed')
            gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id'])
            g=np.array([x['bbox_xywh']+[x['score'],x['category'],x['truncation'],x['occlusion']] for x in gt],float).reshape(-1,8)
            r=np.array([x['bbox_xywh']+[x['score'],x['category'],-1,-1] for x in rp],float).reshape(-1,8)
            s=np.array([x['bbox_xywh']+[x['score'],x['category'],-1,-1] for x in sp],float).reshape(-1,8)
            prepared=P.AP.prepare_raw_image(g,np.concatenate((r,s)),width=meta['width'],height=meta['height'])
            dt=prepared['filtered_predictions'];require(len(dt)%2==0,'R/scorer ignore-filter count differs')
            half=len(dt)//2;gt_images.append(prepared['prepared_gt']);r_images.append(dt[:half]);s_images.append(dt[half:])
            np.savez_compressed(folder/'raw_and_features.npz',raw=raw,features=feature,original_score=prob,new_score=score)
            np.savez_compressed(folder/'prepared_matrices.npz',gt=prepared['prepared_gt'],mse_r=dt[:half],scored=dt[half:])
            write(folder/'predictions.json',{'image_id':meta['image_id'],'image_source_sha256':meta['image_sha256'],'annotation_source_sha256':meta['annotation_sha256'],
                'letterbox_params':params,'mse_r':rp,'scored':sp,'features_extracted_without_GT':True})
            if (i+1)%16==0 or i==177:self.progress('INFER_FROZEN_R_ON_SEPARATE_CALIBRATION_MEMBERS',i+1,178)
        after=E.state_value_fingerprint(model);require(before==after,'Frozen detector state changed')
        write(self.attempt/'frozen_state_check.json',{'before':before,'after':after,'equal':True,'scorer':identity(c['scorer_model']['path']),'fit_count':0})
        del model;torch.cuda.empty_cache()
        self.progress('CALIBRATION_SINGLE_IOU_MATCHING_ONLY',0,1)
        rs,rf=J.rank_entries(gt_images,r_images);ss,sf=J.rank_entries(gt_images,s_images)
        np.savez_compressed(output/'reference_rank_entries.npz',score=rs,flag=rf,**J.curve_from_entries(rs,rf))
        np.savez_compressed(output/'scored_rank_entries.npz',score=ss,flag=sf,**J.curve_from_entries(ss,sf))
        threshold=select_threshold(rs,rf,ss,sf)
        threshold.update(created_at_utc=now(),contract=identity(OUT/'contract.json'),scorer_model=c['scorer_model'],
            calibration_count=178,reference_entries=identity(output/'reference_rank_entries.npz'),scored_entries=identity(output/'scored_rank_entries.npz'))
        write(self.attempt/'frozen_threshold.json',threshold)
        self.progress('SCORER_DISJOINT_CALIBRATION_THRESHOLD_FROZEN',1,1,frozen_threshold=threshold['threshold'])
        return threshold

    def evaluate(self,root,threshold):
        c=self.contract;tau=threshold['threshold'];previous=Path(c['prior_extension_root'])
        require((self.attempt/'frozen_threshold.json').is_file(),'Freeze calibration result before reading evaluation outcomes')
        self.progress('APPLY_FROZEN_THRESHOLD_TO_REUSED_DEVELOPMENT_OUTPUTS',0,414)
        full_scored=read(previous/'attempt_01/evaluation/full414/frozen_score_base/saved_predictions_and_targets.json')
        require([r['image_id'] for r in full_scored]==[r['image_id'] for r in c['evaluation_cohorts']['full414']],'Developer row order differs')
        per_image=[]
        for i,(meta,row) in enumerate(zip(c['evaluation_cohorts']['full414'],full_scored)):
            ann=(root/meta['annotation_path']).resolve(strict=True)
            require(ann.is_relative_to(root) and sha(ann)==meta['annotation_sha256'],'Developer annotation changed')
            gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id'])
            match=P.MATCH.match_project_targets(gt,row['predictions'],meta['width'],meta['height'],confidence=tau,iou_threshold=.5)
            per_image.append({'image_id':meta['image_id'],'threshold':tau,'matching':match})
            if (i+1)%32==0 or i==413:self.progress('APPLY_FROZEN_THRESHOLD_TO_REUSED_DEVELOPMENT_OUTPUTS',i+1,414)
        write(self.attempt/'developer_project_threshold_matches.json',per_image)
        indexed={r['image_id']:r for r in per_image};cohorts={}
        for name in ('new286','new_prefix47','full414'):
            branches={}
            for branch in E.BRANCHES:
                folder=previous/'attempt_01/evaluation'/name/branch
                m=read(folder/'metrics.json');pm=read(folder/'project_metrics.json')
                chosen=tau if branch=='frozen_score_base' else .25
                with np.load(folder/'official_IoU05_rank_curve.npz',allow_pickle=False) as data:
                    official=fixed_counts(data['score'],data['flag'],chosen)
                    old_op=fixed_counts(data['score'],data['flag'],read(previous/'contract.json')['scorer_operating_threshold']) if branch=='frozen_score_base' else None
                if branch=='frozen_score_base':
                    count=Counter()
                    for row in c['evaluation_cohorts'][name]:count.update(indexed[row['image_id']]['matching']['counts'])
                    project=dict(count)
                else:project=pm['conf025_counts']
                branches[branch]={'threshold':chosen,'official_IoU05_cap500_counts':official,'project_counts':project,
                    'AP_percent_unchanged':m['metrics_percent']['AP_all'],'previous_training_fit_threshold_counts':old_op,
                    'cached_curve_source':identity(folder/'official_IoU05_rank_curve.npz')}
            base=branches['mse_r'];score=branches['frozen_score_base'];qat=branches['ordinary_qat']
            a,b=score['official_IoU05_cap500_counts'],base['official_IoU05_cap500_counts']
            sp,bp=score['project_counts'],base['project_counts']
            cohorts[name]={'images':len(c['evaluation_cohorts'][name]),'branches':branches,
                'official_TP_delta_vs_R':a['TP']-b['TP'],'official_FP_delta_vs_R':a['FP']-b['FP'],
                'official_fixed_threshold_joint_rule_met':a['TP']>=b['TP'] and a['FP']<=b['FP'],
                'project_TP_delta_vs_R':sp['hits']-bp['hits'],'project_FP_delta_vs_R':sp['false_positive_detections']-bp['false_positive_detections'],
                'project_fixed_threshold_joint_observation':sp['hits']>=bp['hits'] and sp['false_positive_detections']<=bp['false_positive_detections'],
                'official_TP_delta_vs_QAT':a['TP']-qat['official_IoU05_cap500_counts']['TP'],
                'official_FP_delta_vs_QAT':a['FP']-qat['official_IoU05_cap500_counts']['FP']}
        result={'status':'COMPLETE_SEPARATE_THRESHOLD_CALIBRATION_DEVELOPMENT_CHECK','threshold':tau,'calibration':threshold,'cohorts':cohorts,
            'primary_cohort':'new286','primary_official_joint_rule_met':cohorts['new286']['official_fixed_threshold_joint_rule_met'],
            'all_cohorts_official_joint_rule_met':all(r['official_fixed_threshold_joint_rule_met'] for r in cohorts.values()),
            'primary_project_joint_rule_met':cohorts['new286']['project_fixed_threshold_joint_observation'],
            'model_fits':0,'new_detector_forwards':178,'new_full_AP_scores':0,'scorer_selection_used_previous_development':True,
            'independent_test_set':False,'scientific_acceptance':False,'actual_INT8':False,'Selection_accessed':False,'Holdout_accessed':False,
            'all_old_failures_preserved':True,'one_threshold_candidate_only':True}
        write(self.attempt/'comparison_result.json',result)

    def run(self):
        P.configure();self.verify();require(torch.cuda.is_available(),'CUDA required')
        root=Path(read(self.contract['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
        write(self.attempt/'runtime.json',{'python':P.sys.version,'executable':P.sys.executable,'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'GPU':torch.cuda.get_device_name(0),'backend':P.FT.backend_settings(),'model_fits':0})
        threshold=self.calibration(root);self.evaluate(root,threshold);self.verify()
        require(self.live['forward_images']==178,'Forward budget incomplete')
        files=[identity(p) for p in sorted(self.attempt.rglob('*')) if p.is_file() and p.name not in {'live_status.json','events.jsonl','manifest.json','completion_receipt.json'}]
        write(self.attempt/'manifest.json',{'files':files});self.progress('COMPLETE_SEPARATE_THRESHOLD_CALIBRATION_DEVELOPMENT_CHECK',1,1)
        write(self.attempt/'completion_receipt.json',{**self.live,'finished_at_utc':now(),'elapsed_seconds':time.monotonic()-self.started,
            'comparison':identity(self.attempt/'comparison_result.json'),'manifest':identity(self.attempt/'manifest.json')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));args=parser.parse_args()
    if args.mode=='prepare':prepare();return
    worker=Worker();awake=False
    try:
        if os.name=='nt':
            import ctypes
            awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        worker.run()
    except BaseException as error:
        worker.live.update(status='FAILED_ATTEMPT_PRESERVED',error_type=type(error).__name__,error=str(error),updated_at_utc=now())
        write(worker.attempt/'live_status.json',worker.live,once=False)
        write(worker.attempt/'failure_receipt.json',{**worker.live,'traceback':traceback.format_exc()});raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
