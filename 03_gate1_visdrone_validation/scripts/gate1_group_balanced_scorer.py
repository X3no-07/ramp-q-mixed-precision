"""Registered3arm scorer ablation: prefix weighting and source-shortcut removal.

Reuse saved512fit/178calibration/414developer features. No detector forward.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/gate1/20261026_group_balanced_scorer_v1'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'settings')
import numpy as np
import torch
from torch.nn import functional as F
import gate1_separate_threshold as T
import gate1_scorer_robustness_cost as D
P=T.P;S=T.S;E=T.E;J=T.J
read,write,identity,require,now=P.read,P.write,P.identity,P.require,P.now
QUALITY_INDICES=(0,1,2,3,8,9,10,11)
ARMS=('prefix_balanced_full22','image_balanced_quality8','prefix_balanced_quality8')
COHORTS=('new286','new_prefix47','full414')


def row_weights(image_index,image_prefixes,prefix_balanced):
    image_index=np.asarray(image_index,dtype=np.int64)
    require(len(image_index)>0 and image_index.min()==0 and image_index.max()==len(image_prefixes)-1,'Invalid fitted image indices')
    n=np.bincount(image_index,minlength=len(image_prefixes));require((n>0).all(),'Every fit image must have candidate rows')
    if prefix_balanced:
        groups=Counter(image_prefixes);g=len(groups)
        per_image=np.array([1/(g*groups[p]) for p in image_prefixes],np.float64)
    else:per_image=np.full(len(image_prefixes),1/len(image_prefixes),np.float64)
    weights=per_image[image_index]/n[image_index]
    require(np.isclose(weights.sum(),1,atol=1e-12),'Unnormalized row objective')
    return weights


def fit(x,y,p,image_index,image_prefixes,arm):
    require(arm in ARMS,'Unregistered scorer arm')
    active=tuple(range(22)) if arm.endswith('full22') else QUALITY_INDICES
    mean,scale=S.standardize_training(x)
    z=(x-mean)/scale;excluded=sorted(set(range(22))-set(active));z[:,excluded]=0
    weights=row_weights(image_index,image_prefixes,arm.startswith('prefix_balanced'))
    z=torch.from_numpy(np.ascontiguousarray(z));target=torch.from_numpy(np.asarray(y,np.float64))
    offset=torch.from_numpy(S.logit(p));weights_tensor=torch.from_numpy(weights)
    w=torch.zeros(22,dtype=torch.float64,requires_grad=True);b=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([w,b],lr=.03,betas=(.9,.999),eps=1e-8);trace=[]
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits=offset+z@w+b
        common=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*weights_tensor).sum()
        regularizer=.0005*(w.square().sum()+b.square());loss=common+regularizer
        require(bool(torch.isfinite(loss)),'Nonfinite scorer loss')
        loss.backward();require(torch.isfinite(w.grad).all() and torch.isfinite(b.grad),'Nonfinite scorer gradient')
        optimizer.step()
        if step%10==0 or step==99:trace.append({'step':step+1,'loss':float(loss.detach()),'base':float(common.detach()),'regularizer':float(regularizer.detach())})
    weight=w.detach().numpy();require(np.array_equal(weight[excluded],np.zeros(len(excluded))),'Removed feature received nonzero coefficient')
    model={'arm':arm,'features':list(S.FEATURES),'active_feature_indices':list(active),'active_feature_names':[S.FEATURES[i] for i in active],
        'coefficient_slots':23,'active_parameter_slots':len(active)+1,'mean':mean.tolist(),'scale':scale.tolist(),'weight':weight.tolist(),
        'bias':float(b.detach()),'steps':100,'train_rows':len(y),'fit_images':len(image_prefixes),'fit_prefixes':len(set(image_prefixes)),
        'weighting':'equalprefix_equalimagewithinprefix_equalcandidatewithinimage' if arm.startswith('prefix_balanced') else 'equalimage_equalcandidatewithinimage',
        'trace':trace,'class_or_absolute_geometry_feature_used':len(active)==22,'GT_used_only_as_training_target':True,
        'prefix_at_inference':False,'Q_path_at_inference':False,'detector_updated':False}
    return model,weights


def sources():
    bindings=deepcopy(T.sources())
    for p in (Path(__file__),ROOT/'scripts/gate1_scorer_robustness_cost.py',ROOT/'scripts/launch_gate1_group_balanced_scorer.py',ROOT/'tests/test_gate1_group_balanced_scorer.py'):
        bindings[p.name]=identity(p)
    return bindings


def prepare():
    require(not (OUT/'contract.json').exists(),'Scorer ablation contract already frozen')
    threshold=read(T.OUT/'contract.json');oldscore=read(S.OUT/'contract.json');ext=read(E.OUT/'contract.json')
    fit_files=[identity(S.OUT/'attempt_01/training_cache'/f'{i:04d}.npz') for i in range(512)]
    calibration=[]
    for i,meta in enumerate(threshold['calibration_metadata']):
        folder=T.OUT/'attempt_01/calibration'/f'{i:04d}'
        calibration.append({'index':i,'image_id':meta['image_id'],'features':identity(folder/'raw_and_features.npz'),
            'predictions':identity(folder/'predictions.json'),'matrices':identity(folder/'prepared_matrices.npz')})
    newmap={r['image_id']:i for i,r in enumerate(ext['cohorts']['new286'])};oldmap={r['image_id']:i for i,r in enumerate(ext['cohorts']['old128'])}
    development=[]
    for meta in ext['cohorts']['full414']:
        if meta['image_id'] in newmap:
            i=newmap[meta['image_id']];path=E.OUT/'attempt_01/new_inference'/f'{i:04d}/scorer_features.npz'
        else:i=oldmap[meta['image_id']];path=S.OUT/'attempt_01/development_cache'/f'{i:04d}.npz'
        development.append({'image_id':meta['image_id'],'features':identity(path)})
    reused=[]
    for name in COHORTS:
        for branch in E.BRANCHES:
            folder=E.OUT/'attempt_01/evaluation'/name/branch
            for fn in ('metrics.json','project_metrics.json','official_IoU05_rank_curve.npz'):reused.append(identity(folder/fn))
    for branch in ('mse_r','ordinary_qat'):reused.append(identity(E.OUT/'attempt_01/evaluation/full414'/branch/'saved_predictions_and_targets.json'))
    c={'schema':'rampq.gate1.prefix_balanced_scorer_ablation/v1','at_utc':now(),'source_bindings':sources(),
       'authorization':'User: Continue; conduct fixed 3 evaluator comparisons based on revenue concentration and training group imbalance diagnosis, maintaining data isolation and reporting truthfully',
       'previous_robustness_contract':identity(D.OUT/'contract.json'),'previous_robustness_delivery':identity(D.OUT/'delivery_validation.json'),
       'previous_threshold_contract':identity(T.OUT/'contract.json'),'previous_threshold_delivery':identity(T.OUT/'delivery_validation.json'),
       'previous_threshold_result':identity(T.OUT/'attempt_01/comparison_result.json'),'old_score_model':threshold['scorer_model'],
       'evaluation_protocol':threshold['evaluation_protocol'],'fit_members':oldscore['training_members'],
       'fit_files':fit_files,'calibration_members':threshold['calibration_metadata'],'calibration_caches':calibration,
       'development_features':development,'cohorts':threshold['evaluation_cohorts'],'reused_files':reused,
       'arms':list(ARMS),'candidate_budget':3,'cumulative_new_scoring_candidates_including_previous3':6,
       'features_full':list(S.FEATURES),'quality8indices':list(QUALITY_INDICES),
       'design':'2x2factorial:oldimagebalancedfull22(reused),newprefixbalancedfull22,newimagebalancedquality8,newprefixbalancedquality8. Quality8removesabsoluteWH/XYandclassonehot; keep original4scorequality+4neighborrelativefeatures.',
       'fit_rule':'Same512fitimages/213716candidateGTIoUtargets,train-onlynormalization,zeroresidualinit,100fullbatch CPUfloat64 Adam steps,lr0.03,betas0.9/0.999,eps1e-8,L2coefficient0.0005. No selectedhard/intervention auxiliary.',
       'group_weight_rule':'For58filenameprefixproxies,equalprefixweight;withinprefixequalimageweight;withinimageequalcandidateweight. No group label needed for scoring.',
       'group_distribution_metadata':dict(Counter(T.prefix(r) for r in oldscore['training_members'])),
       'threshold_rule':'Eachnewmodel exactlyone scalar threshold using all178scorer-fit-disjoint calibrationmembers,officialIoU0.5/cap500,FP<=sameR@0.25budget; samewholegroup rule andabstention as frozenT.select_threshold; freezeALL3thresholdsbeforedeveloper scoring.',
       'evaluation':'All3models x new286/new_prefix47/full414 AP9jobs. Report fixedthreshold officialTP/FP,project same-target counts,FP-budgetcurve and exhaustiveleave-one-prefix sensitivity. OldR,oldfull22scoreandordinaryQAT metrics reused.',
       'primary_rule':'new286 AP>R_AP and fixedofficialTP>=R@0.25TP and fixedofficialFP<=R@0.25FP. Additional robustness requires samejointobservation onnew_prefix47/full414and all leave-one-prefix PROJECTtotals.',
       'extra_value_rule':'Reporteachnewarm vsoldimagefull22score at its existingseparatecalibrationthreshold0.1747385237; superiority requires APincrease and fixedendpointTPnotlower/FPnothigher. No renaming knownprefixweighting as novel intervention.',
       'all_outcomes_retained':True,'primary_cohort':'new286','AP_job_budget':9,'AP_job_max_parallelism':2,'timeout_seconds':3600,
       'new_detector_forwards':0,'new_full_fit_images':0,'scorer_fit_count':3,'updates_each':100,'new_image_decodes':0,
       'labels_scope':'Reusefitlabels,calibrationmatrices,existing414training_valannotationonly. No Selection/Holdout payload.',
       'prefix_proxy_is_verified_scene':False,'adaptive_method_design_used_previous_developer_results':True,'independent_test_set':False,
       'related_work':{'url':'https://arxiv.org/abs/1911.08731','scope':'group-awaregeneralizationispriorart;thisusesfixedgroupbalancedERM,notfullGroupDROoritsreportedaccuracy'},
       'scientific_acceptance':False,'Gate1_overall_passed':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,'automatic_retry':False}
    write(OUT/'contract.json',c)
    print(json.dumps({'status':'FROZEN3SCORER_ABLATION','contract':identity(OUT/'contract.json'),'new_scorers':3,'new_detector_forwards':0,'AP_jobs':9},ensure_ascii=True))


class Worker(E.Worker):
    def __init__(self):
        self.contract=read(OUT/'contract.json');self.attempt=OUT/'attempt_01';self.attempt.mkdir(exist_ok=False)
        self.started=time.monotonic();self.children=[]
        self.live={'pid':os.getpid(),'parent_pid':os.getppid(),'started_at_utc':now(),'status':'STARTING','contract':identity(OUT/'contract.json'),
            'forward_images':0,'new_detector_forwards':0,'new_model_fit_count':0}
        self.progress('STARTING',0,1)

    def verify(self):
        c=self.contract;require(c['source_bindings']==sources(),'Frozen sources changed')
        for key in ('previous_robustness_contract','previous_robustness_delivery','previous_threshold_contract','previous_threshold_delivery','previous_threshold_result','old_score_model','evaluation_protocol'):
            require(identity(c[key]['path'])==c[key],'Bound source changed: '+key)
        bound={}
        for b in c['fit_files']+c['reused_files']:bound[b['path']]=b
        for row in c['calibration_caches']:
            for key in ('features','predictions','matrices'):bound[row[key]['path']]=row[key]
        for row in c['development_features']:bound[row['features']['path']]=row['features']
        for b in bound.values():require(identity(b['path'])==b,'Saved input changed')

    def fit_models(self):
        c=self.contract;xs=[];ys=[];ps=[];images=[]
        for i,b in enumerate(c['fit_files']):
            with np.load(b['path'],allow_pickle=False) as d:
                xs.append(d['features'].copy());ys.append(d['target'].copy());ps.append(d['original_score'].copy());images.append(np.full(len(d['target']),i,np.int64))
        x,y,p,image=np.concatenate(xs),np.concatenate(ys),np.concatenate(ps),np.concatenate(images)
        del xs,ys,ps,images
        require(x.shape==(213716,22),'Existing fit dataset shape changed')
        prefixes=[T.prefix(r) for r in c['fit_members']];models={}
        for arm in ARMS:
            self.progress('FIT_'+arm,0,100);m,weights=fit(x,y,p,image,prefixes,arm)
            write(self.attempt/f'{arm}_model.json',m);np.savez_compressed(self.attempt/f'{arm}_row_weights.npz',weights=weights)
            require(np.array_equal(S.score_rows(x[:1000],p[:1000],m),S.score_rows(x[:1000],p[:1000],read(self.attempt/f'{arm}_model.json'))),'Saved scorer state replay differs')
            models[arm]=m;self.live['new_model_fit_count']+=1;self.progress('FIT_'+arm,100,100)
        return models

    def calibrate(self,models,root):
        c=self.contract;output=self.attempt/'calibration';output.mkdir()
        gts=[];by_arm={arm:[] for arm in ARMS}
        for i,(meta,cached) in enumerate(zip(c['calibration_members'],c['calibration_caches'])):
            original=read(cached['predictions']['path'])['mse_r']
            with np.load(cached['features']['path'],allow_pickle=False) as d:x=d['features'].copy();p=d['original_score'].copy()
            ann=(root/meta['annotation_path']).resolve(strict=True);require(ann.is_relative_to(root) and P.sha(ann)==meta['annotation_sha256'],'Calibration annotation changed')
            gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id'])
            gt8=np.array([g['bbox_xywh']+[g['score'],g['category'],g['truncation'],g['occlusion']] for g in gt],float).reshape(-1,8)
            arrays=[]
            for arm,m in models.items():
                score=S.score_rows(x,p,m);pred=deepcopy(original)
                for row,v in zip(pred,score):row['score']=float(v)
                pred=sorted(pred,key=lambda d:(-d['score'],d['nms_input_order']))
                arrays.append(np.array([d['bbox_xywh']+[d['score'],d['category'],-1,-1] for d in pred],float).reshape(-1,8))
            prepared=P.AP.prepare_raw_image(gt8,np.concatenate(arrays),width=meta['width'],height=meta['height'])
            with np.load(cached['matrices']['path'],allow_pickle=False) as d:
                k=len(d['mse_r']);require(np.array_equal(d['gt'],prepared['prepared_gt']),'Calibration GT differs from previous frozen matrix')
            require(len(prepared['filtered_predictions'])==3*k,'Calibration candidate pool membership differs')
            packed={'gt':prepared['prepared_gt']};gts.append(prepared['prepared_gt'])
            for j,arm in enumerate(ARMS):by_arm[arm].append(prepared['filtered_predictions'][j*k:(j+1)*k]);packed[arm]=by_arm[arm][-1]
            np.savez_compressed(output/f'{i:04d}_matrices.npz',**packed)
            if (i+1)%32==0 or i==177:self.progress('APPLY3SCORERS_TO178CALIBRATIONCACHES',i+1,178)
        with np.load(T.OUT/'attempt_01/calibration/reference_rank_entries.npz',allow_pickle=False) as d:rs,rf=d['score'].copy(),d['flag'].copy()
        thresholds={}
        for arm in ARMS:
            score,flag=J.rank_entries(gts,by_arm[arm]);np.savez_compressed(output/f'{arm}_entries.npz',score=score,flag=flag,**J.curve_from_entries(score,flag))
            thresholds[arm]=T.select_threshold(rs,rf,score,flag)
            thresholds[arm].update(model=identity(self.attempt/f'{arm}_model.json'),created_at_utc=now())
        write(self.attempt/'all3frozen_thresholds.json',thresholds)
        self.progress('ALL3CALIBRATION_THRESHOLDS_FROZEN_BEFORE_DEVELOPER_SCORING',3,3)
        return thresholds

    def build_evaluations(self,models,thresholds,root):
        c=self.contract;base=read(E.OUT/'attempt_01/evaluation/full414/mse_r/saved_predictions_and_targets.json')
        require([r['image_id'] for r in base]==[r['image_id'] for r in c['cohorts']['full414']],'Developer base order differs')
        all_records={arm:{} for arm in ARMS};matrices={arm:{} for arm in ARMS}
        for i,(meta,record,feature_binding) in enumerate(zip(c['cohorts']['full414'],base,c['development_features'])):
            require(meta['image_id']==record['image_id']==feature_binding['image_id'],'Developer feature membership differs')
            with np.load(feature_binding['features']['path'],allow_pickle=False) as d:x=d['features'].copy();p=d['original_score'].copy()
            require(np.array_equal(p,np.array([r['score'] for r in record['predictions']])),'Feature/original prediction order differs')
            ann=(root/meta['annotation_path']).resolve(strict=True);require(ann.is_relative_to(root) and P.sha(ann)==meta['annotation_sha256'],'Developer annotation changed')
            gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id']);gt8=np.array([g['bbox_xywh']+[g['score'],g['category'],g['truncation'],g['occlusion']] for g in gt],float).reshape(-1,8)
            arrays=[]
            for arm,m in models.items():
                scores=S.score_rows(x,p,m);pred=deepcopy(record['predictions'])
                for row,v in zip(pred,scores):row['score']=float(v)
                pred=sorted(pred,key=lambda d:(-d['score'],d['nms_input_order']))
                match=P.MATCH.match_project_targets(gt,pred,meta['width'],meta['height'],confidence=thresholds[arm]['threshold'],iou_threshold=.5)
                all_records[arm][meta['image_id']]={'image_id':meta['image_id'],'predictions':pred,'project_operational':match,
                    'frozen_threshold':thresholds[arm]['threshold'],'prefix_proxy':meta['prefix_proxy']}
                arrays.append(np.array([d['bbox_xywh']+[d['score'],d['category'],-1,-1] for d in pred],float).reshape(-1,8))
            prepared=P.AP.prepare_raw_image(gt8,np.concatenate(arrays),width=meta['width'],height=meta['height'])
            require(len(prepared['filtered_predictions'])%3==0,'Developer candidate pool count differs');k=len(prepared['filtered_predictions'])//3
            for j,arm in enumerate(ARMS):matrices[arm][meta['image_id']]=(prepared['prepared_gt'],prepared['filtered_predictions'][j*k:(j+1)*k])
            if (i+1)%32==0 or i==413:self.progress('SCORE3FIXEDMODELS_ON414DEVELOPER_FEATURES',i+1,414)
        jobs=[]
        for cohort in COHORTS:
            for arm in ARMS:
                folder=self.attempt/'evaluation'/cohort/arm;folder.mkdir(parents=True)
                packed={};rows=[];count=Counter()
                for i,meta in enumerate(c['cohorts'][cohort]):
                    g,d=matrices[arm][meta['image_id']];packed[f'gt_{i}']=g;packed[f'dt_{i}']=d
                    row=all_records[arm][meta['image_id']];rows.append(row);count.update(row['project_operational']['counts'])
                np.savez_compressed(folder/'prepared_matrices.npz',**packed);write(folder/'saved_predictions_and_targets.json',rows)
                write(folder/'project_metrics.json',{'cohort':cohort,'branch':arm,'images':len(rows),'threshold':thresholds[arm]['threshold'],'counts':dict(count)})
                job={'cohort':cohort,'branch':arm,'images':len(rows),'matrices':identity(folder/'prepared_matrices.npz'),'output':str(folder.resolve()),'contract_sha256':self.live['contract']['sha256']}
                path=folder/'AP_job.json';write(path,job);jobs.append(path)
        return jobs,all_records,base

    def collect(self,thresholds,records,base):
        c=self.contract;oldthreshold=read(c['previous_threshold_result']['path']);oldbase={r['image_id']:r['project_conf025'] for r in base};allresults={}
        for cohort in COHORTS:
            rf=E.OUT/'attempt_01/evaluation'/cohort/'mse_r'
            reference_AP=read(rf/'metrics.json')['metrics_percent']['AP_all']
            with np.load(rf/'official_IoU05_rank_curve.npz',allow_pickle=False) as d:
                rscore,rflag=d['score'].copy(),d['flag'].copy();budget=T.fixed_counts(rscore,rflag,.25)['FP']
            ref=T.fixed_counts(rscore,rflag,.25);previous=oldthreshold['cohorts'][cohort]['branches']['frozen_score_base']
            baseline_AP=previous['AP_percent_unchanged'];baseline_count=previous['official_IoU05_cap500_counts']
            comparisons={}
            for arm in ARMS:
                folder=self.attempt/'evaluation'/cohort/arm;m=read(folder/'metrics.json');project=read(folder/'project_metrics.json')
                with np.load(folder/'official_IoU05_rank_curve.npz',allow_pickle=False) as d:
                    official=T.fixed_counts(d['score'],d['flag'],thresholds[arm]['threshold']);at_budget=S.at_fp_budget({k:d[k] for k in ('threshold','TP','FP')},budget)
                ap=m['metrics_percent']['AP_all'];rows=[]
                for meta in c['cohorts'][cohort]:
                    original=oldbase[meta['image_id']]['counts'];new=records[arm][meta['image_id']]['project_operational']['counts']
                    rows.append({'image_id':meta['image_id'],'prefix_proxy':meta['prefix_proxy'],'TP_delta':new['hits']-original['hits'],'FP_delta':new['false_positive_detections']-original['false_positive_detections']})
                sensitivity=D.group_sensitivity(rows)
                fixed_joint=official['TP']>=ref['TP'] and official['FP']<=ref['FP']
                comparisons[arm]={'AP_percent':ap,'metrics_source':identity(folder/'metrics.json'),'threshold':thresholds[arm]['threshold'],
                    'official_fixed_counts':official,'project_counts':project['counts'],'at_R_FP_budget':at_budget,
                    'AP_delta_vs_R':ap-reference_AP,'TP_delta_vs_R':official['TP']-ref['TP'],'FP_delta_vs_R':official['FP']-ref['FP'],
                    'AP_delta_vs_old_generic_scorer':ap-baseline_AP,'TP_delta_vs_old_generic_scorer':official['TP']-baseline_count['TP'],
                    'FP_delta_vs_old_generic_scorer':official['FP']-baseline_count['FP'],
                    'primary_style_joint_AP_and_endpoint_observation':ap>reference_AP and fixed_joint,
                    'extra_value_over_old_generic_scorer':ap>baseline_AP and official['TP']>=baseline_count['TP'] and official['FP']<=baseline_count['FP'],
                    'prefix_sensitivity':sensitivity}
            allresults[cohort]={'images':len(c['cohorts'][cohort]),'reference_R_AP':reference_AP,'reference_R_official_counts':ref,'reference_R_FP_budget':budget,
                'old_generic_scorer_AP':baseline_AP,'old_generic_scorer_official_counts':baseline_count,'arms':comparisons}
        summary={arm:{'primary_new286_joint_observation':allresults['new286']['arms'][arm]['primary_style_joint_AP_and_endpoint_observation'],
            'all3cohorts_joint_observation':all(allresults[k]['arms'][arm]['primary_style_joint_AP_and_endpoint_observation'] for k in COHORTS),
            'all3cohorts_and_allprefixremovals_robust_observation':all(allresults[k]['arms'][arm]['primary_style_joint_AP_and_endpoint_observation'] and allresults[k]['arms'][arm]['prefix_sensitivity']['all_leave_one_prefix_joint_observation'] for k in COHORTS),
            'primary_extra_value_vs_old_generic_scorer':allresults['new286']['arms'][arm]['extra_value_over_old_generic_scorer']} for arm in ARMS}
        write(self.attempt/'comparison_result.json',{'status':'COMPLETE3ARM_PREFIX_FEATURE_SCORER_DEVELOPMENT_ABLATION','cohorts':allresults,'summary':summary,
            'scorer_fits':3,'updates_each':100,'new_detector_forwards':0,'AP_jobs':9,'all3candidates_preserved':True,
            'independent_test_set':False,'scientific_acceptance':False,'actual_INT8':False,'Selection_accessed':False,'Holdout_accessed':False,'prior_art_not_claimed_novel':True})

    def run(self):
        P.configure();self.verify();models=self.fit_models()
        root=Path(read(self.contract['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
        thresholds=self.calibrate(models,root);jobs,records,base=self.build_evaluations(models,thresholds,root)
        self.execute_jobs(jobs);self.collect(thresholds,records,base);self.verify()
        require(self.live['new_model_fit_count']==3 and self.live['forward_images']==0,'Fit/forward budget incomplete')
        files=[identity(p) for p in sorted(self.attempt.rglob('*')) if p.is_file() and p.name not in {'events.jsonl','live_status.json','manifest.json','completion_receipt.json'}]
        write(self.attempt/'manifest.json',{'files':files});self.progress('COMPLETE3ARM_PREFIX_FEATURE_SCORER_DEVELOPMENT_ABLATION',1,1)
        write(self.attempt/'completion_receipt.json',{**self.live,'finished_at_utc':now(),'elapsed_seconds':time.monotonic()-self.started,
            'manifest':identity(self.attempt/'manifest.json'),'comparison':identity(self.attempt/'comparison_result.json')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));args=parser.parse_args()
    if args.mode=='prepare':prepare();return
    worker=Worker()
    try:worker.run()
    except BaseException as error:
        stopped=[]
        for child in worker.children:
            if child.poll() is None:
                stop=subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                stopped.append({'owned_live_AP_pid':child.pid,'tree_stop_returncode':stop.returncode});child.wait()
        worker.live.update(status='FAILED_ATTEMPT_PRESERVED',error_type=type(error).__name__,error=str(error),updated_at_utc=now())
        write(worker.attempt/'live_status.json',worker.live,once=False);write(worker.attempt/'failure_receipt.json',{**worker.live,'traceback':traceback.format_exc(),'owned_children_stopped':stopped});raise


if __name__=='__main__':main()
