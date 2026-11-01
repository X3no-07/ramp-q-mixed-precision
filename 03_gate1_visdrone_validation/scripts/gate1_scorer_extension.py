"""Frozen scorer extension to remaining training_val members. No refitting."""
from __future__ import annotations
import argparse
import hashlib
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/gate1/20261026_scorer_extension_286_v1'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'settings')
import numpy as np
import torch
import gate1_frozen_box_scorer as S
P=S.P
BRANCHES=('mse_r','frozen_score_base','ordinary_qat')
read,write,identity,require,sha,now=P.read,P.write,P.identity,P.require,P.sha,P.now


def partition_members(full,old):
    oldids={r['image_id'] for r in old};fullids={r['image_id'] for r in full}
    require(len(oldids)==len(old) and len(fullids)==len(full) and oldids<=fullids,'Cohort uniqueness/subset mismatch')
    new=[r for r in full if r['image_id'] not in oldids]
    prefixes={r['prefix_proxy'] for r in old}
    fresh=[r for r in new if r['prefix_proxy'] not in prefixes]
    require(not ({r['image_sha256'] for r in new}&{r['image_sha256'] for r in old}),'Old/new image hash overlap')
    return {'new286':new,'new_prefix47':fresh,'full414':full,'old128':old}


def select_training_threshold(original_score,target,new_score):
    budget=int(((target==0)&(original_score>=.25)).sum())
    flags=(target>0).astype(np.int8)
    curve=S.grouped_curve(new_score,flags)
    point=S.at_fp_budget(curve,budget)
    require(point['threshold'] is not None,'No valid training-only operating threshold')
    return {'threshold':point['threshold'],'nominal_train_FP_budget':budget,'chosen_train_point':point,
            'old_nominal_train_TP':int(((target>0)&(original_score>=.25)).sum()),
            'rule':'Lowest whole-score-group threshold with nominal training FP<=old R confidence0.25 FP count',
            'uses_development_labels':False,'source_labels':'PROJECT one-to-one TP/FP labels at confidence0.001 from original scorer training cache',
            'is_nominal_training_budget_not_guarantee_on_new_images':True}


def source_bindings():
    bindings=deepcopy(S.source_bindings())
    for path in (Path(__file__),ROOT/'scripts/gate1_cached_ap_job.py',ROOT/'scripts/launch_gate1_scorer_extension.py',ROOT/'tests/test_gate1_scorer_extension.py'):
        bindings[path.name]=identity(path)
    return bindings


def state_value_fingerprint(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        data=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(data.dtype).encode());digest.update(str(data.shape).encode());digest.update(data.tobytes())
    return digest.hexdigest()


def prepare():
    require(not (OUT/'contract.json').exists(),'Extension contract exists')
    old=read(S.OUT/'contract.json');protocol=read(old['evaluation_protocol']['path'])
    train_manifest=read(old['train_manifest']['path']);val_manifest=read(old['training_val_manifest']['path'])
    P.FT.validate_membership(train_manifest,val_manifest)
    val_index={r['image_id']:r for r in val_manifest['samples']}
    require(all(r['assigned_split']=='training_val' and val_index[r['image_id']]['image']['sha256']==r['image_sha256'] for r in protocol['cohorts']['training_val']),'Training_val role/source differs')
    cohorts=partition_members(protocol['cohorts']['training_val'],old['evaluation_members'])
    require([len(cohorts[k]) for k in ('new286','new_prefix47','full414','old128')]==[286,47,414,128],'Unexpected cohort membership')
    train_sha={r['image']['sha256'] for r in train_manifest['samples']}
    require(train_sha.isdisjoint(r['image_sha256'] for r in cohorts['new286']),'Train/new image overlap')
    model_path=S.OUT/'attempt_01/score_base_model.json';model=read(model_path)
    original=[];target=[];scored=[];train_bindings=[]
    for i in range(512):
        path=S.OUT/'attempt_01/training_cache'/f'{i:04d}.npz';train_bindings.append(identity(path))
        with np.load(path,allow_pickle=False) as data:
            original.append(data['original_score'].copy());target.append(data['target'].copy())
            scored.append(S.score_rows(data['features'],data['original_score'],model))
    calibration=select_training_threshold(np.concatenate(original),np.concatenate(target),np.concatenate(scored))
    calibration.update(model=identity(model_path),training_cache_members=train_bindings)
    write(OUT/'training_threshold_calibration.json',calibration)
    np.savez_compressed(OUT/'training_threshold_calibration_arrays.npz',original_score=np.concatenate(original),target=np.concatenate(target),new_score=np.concatenate(scored))
    old_roots={'mse_r':old['baseline_root'],'frozen_score_base':str(S.OUT/'attempt_01/evaluation/score_base'),
               'ordinary_qat':str(P.OUT/'attempt_01/evaluation/ordinary_qat')}
    reused=[]
    for root in old_roots.values():
        for name in ('metrics.json','prepared_matrices.npz','saved_predictions_and_targets.json'):
            reused.append(identity(Path(root)/name))
    qat_receipt=read(P.OUT/'attempt_01/training/ordinary_qat/training_receipt.json')
    c={'schema':'rampq.gate1.frozen_scorer_extension/v1','created_at_utc':now(),
       'authorization':'User: Must proceed; freeze the ordinary scorers that have shown observed development improvements, and revalidate them on the remaining development members; success must be determined by actual measurement',
       'source_bindings':source_bindings(),'framework':P.verify_framework_sources(),
       'previous_contract':identity(S.OUT/'contract.json'),'previous_delivery':identity(S.OUT/'delivery_validation.json'),
       'previous_decision':identity(S.OUT/'machine_decision.json'),'scorer_model':identity(model_path),
       'candidate':old['candidate'],'architecture':old['architecture'],'evaluation_protocol':old['evaluation_protocol'],
       'training_manifest':old['train_manifest'],'training_val_manifest':old['training_val_manifest'],'training_members':old['training_members'],
       'ranges':identity(Path(old['reused_stage_root'])/'mse_ranges.json'),
       'QAT_candidate':qat_receipt['final_candidate'],'QAT_training_receipt':identity(P.OUT/'attempt_01/training/ordinary_qat/training_receipt.json'),
       'cohorts':cohorts,'primary_cohort':'new286','supporting_cohorts':['new_prefix47','full414'],
       'prefix_proxy_is_verified_scene_identity':False,'prefix_new_cohort_count':47,'prefix_new_proxy_count':21,
       'new286_prefix_overlap_with_old128':32,'new286_hash_overlap_training':0,
       'scorer_selected_after_old128_results':True,'remaining_training_val_was_used_in_historical_other_development':True,
       'independent_test_set':False,'old_cohort_order':[r['image_id'] for r in old['evaluation_members']],
       'old_cohort_roots':old_roots,'reused_files':reused,'training_threshold_inputs':train_bindings,
       'training_threshold_calibration':identity(OUT/'training_threshold_calibration.json'),
       'training_threshold_arrays':identity(OUT/'training_threshold_calibration_arrays.npz'),
       'scorer_operating_threshold':calibration['threshold'],'baseline_operating_threshold':.25,
       'branches':list(BRANCHES),'new_detector_image_forward_budget':572,'real_image_forward_budget':572,
       'detector_forwards_per_new_image':2,'new_model_fit_count':0,'new_calibration_image_forwards':0,
       'inference':'286new training_val images, fixedR and previous ordinaryQAT,640,batch1,FP32 unfused QDQ,noAMP. Existing128prediction/matrix outputs reused, no old image forward.',
       'scoring':'Frozen score_base JSON only; original fixedR NMS pool retained, same boxes/categories; change score/order only; no NMS after scoring.',
       'AP_job_budget':9,'AP_job_max_parallelism':2,'AP_job_timeout_seconds':1200,
       'evaluation':'3fixedbranches x new286,prefix47,full414. New official Python AP and official IoU0.5/cap500 score curves. All cohorts reported, no favorable-subset selection.',
       'primary_rule':'On new286: frozen score AP>R AP and maxTP at referenceR FP budget is not below R curve. FP budget=#R officialFP at score0.25 in the SAME cohort. Whole score ties retained.',
       'robust_development_rule':'Joint AP/FP-budget rule holds on ALL3predefined cohorts, not only full414. OrdinaryQAT deltas also reported. No scientific acceptance from these overlapping developer cohorts.',
       'operating_rule':'Separately report PROJECT counts at frozen training-only scorer threshold vsR at0.25; operational joint observation requires TP>=R and FP<=R. This is not imposed by retuning on new images.',
       'ranking_thresholds_are_descriptive_only':True,'new286_raw_numeric_qualification':'NOT_RUN; older full-anchor FAILs stay FAIL; no subset gate replacement',
       'old_failures_preserved':True,'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
       'timeout_seconds':3600,'automatic_retry':False}
    write(OUT/'contract.json',c)
    print(json.dumps({'status':'FROZEN_EXTENSION','contract':identity(OUT/'contract.json'),'new_images':286,'new_prefix_images':47,
                      'training_only_threshold':calibration['threshold'],'new_forwards':572,'AP_jobs':9},ensure_ascii=True))


class Worker(P.Worker):
    def __init__(self):
        self.contract=read(OUT/'contract.json');self.attempt=OUT/'attempt_01';self.attempt.mkdir(exist_ok=False)
        self.started=time.monotonic();self.children=[]
        self.live={'pid':os.getpid(),'parent_pid':os.getppid(),'started_at_utc':now(),'status':'STARTING','forward_images':0,
                   'contract':identity(OUT/'contract.json'),'new_model_fit_count':0}
        self.progress('STARTING',0,1)

    def verify(self):
        c=self.contract
        require(c['source_bindings']==source_bindings(),'Frozen sources changed')
        require(c['framework']==P.verify_framework_sources(),'Framework changed')
        for key in ('previous_contract','previous_delivery','previous_decision','scorer_model','candidate','architecture','evaluation_protocol','training_manifest','training_val_manifest','ranges','QAT_candidate','QAT_training_receipt','training_threshold_calibration','training_threshold_arrays'):
            require(identity(c[key]['path'])==c[key],'Bound source changed: '+key)
        for b in c['reused_files']+c['training_threshold_inputs']:require(identity(b['path'])==b,'Reused member changed')

    def image_gt(self,root,meta):
        ann=(root/meta['annotation_path']).resolve(strict=True)
        require(ann.is_relative_to(root) and sha(ann)==meta['annotation_sha256'],'Annotation identity differs')
        return P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id'])

    def infer_new(self,root):
        c=self.contract;maxima=read(c['ranges']['path'])['max_abs'];scorer=read(c['scorer_model']['path'])
        r=P.new_model(maxima);P.precision(r,'r_only:G1')
        q=P.new_model(maxima);q.load_state_dict(torch.load(c['QAT_candidate']['path'],map_location='cpu',weights_only=True)['state_dict'],strict=True)
        q.eval();P.precision(q,'r_only:G1')
        before={'mse_r':state_value_fingerprint(r),'ordinary_qat':state_value_fingerprint(q)}
        folder=self.attempt/'new_inference';folder.mkdir()
        all_records={b:{} for b in BRANCHES};all_matrices={b:{} for b in BRANCHES}
        for i,meta in enumerate(c['cohorts']['new286']):
            x,params=self.image_input(root,meta)
            raw_r=self.forward(r,x);raw_q=self.forward(q,x)
            output=folder/f'{i:04d}';output.mkdir()
            np.savez_compressed(output/'raw.npz',mse_r=raw_r,ordinary_qat=raw_q)
            rp=P.predictions(raw_r,params,meta['image_id'],'mse_r')
            qp=P.predictions(raw_q,params,meta['image_id'],'ordinary_qat')
            features=S.visible_features(raw_r,rp)
            rescore=S.score_rows(features,np.array([p['score'] for p in rp],np.float64),scorer)
            sp=deepcopy(rp)
            for p,score in zip(sp,rescore):p['score']=float(score)
            sp=sorted(sp,key=lambda p:(-p['score'],p['nms_input_order']))
            np.savez_compressed(output/'scorer_features.npz',features=features,original_score=np.array([p['score'] for p in rp]),new_score=rescore)
            gt=self.image_gt(root,meta)
            gt8=np.array([g['bbox_xywh']+[g['score'],g['category'],g['truncation'],g['occlusion']] for g in gt],np.float64).reshape(-1,8)
            for branch,pred in zip(BRANCHES,(rp,sp,qp)):
                dt8=np.array([p['bbox_xywh']+[p['score'],p['category'],-1,-1] for p in pred],np.float64).reshape(-1,8)
                prepared=P.AP.prepare_raw_image(gt8,dt8,width=meta['width'],height=meta['height'])
                all_matrices[branch][meta['image_id']]=(prepared['prepared_gt'],prepared['filtered_predictions'])
                op=c['scorer_operating_threshold'] if branch=='frozen_score_base' else .25
                match=P.MATCH.match_project_targets(gt,pred,meta['width'],meta['height'],confidence=.25,iou_threshold=.5)
                operational=match if op==.25 else P.MATCH.match_project_targets(gt,pred,meta['width'],meta['height'],confidence=op,iou_threshold=.5)
                all_records[branch][meta['image_id']]={'image_id':meta['image_id'],'predictions':pred,'project_conf025':match,
                    'frozen_operating_threshold':op,'project_operational':operational,'letterbox_params':params,'image_sha256':meta['image_sha256']}
            write(output/'predictions.json',{b:all_records[b][meta['image_id']] for b in BRANCHES})
            if (i+1)%16==0 or i==285:self.progress('INFER_TWO_FROZEN_DETECTORS_AND_SCORE_NEW_MEMBERS',i+1,286)
        after={'mse_r':state_value_fingerprint(r),'ordinary_qat':state_value_fingerprint(q)}
        require(before==after,'Frozen detector state changed during inference')
        write(self.attempt/'frozen_detector_state_check.json',{'before':before,'after':after,'equal':True,'new_model_fit_count':0,
              'scorer_model':identity(c['scorer_model']['path'])})
        del r,q;torch.cuda.empty_cache()
        require(self.live['forward_images']==572,'Inference budget incomplete')
        return all_records,all_matrices

    def assemble(self,root,records,matrices):
        c=self.contract
        for branch,oldroot in c['old_cohort_roots'].items():
            rows=read(Path(oldroot)/'saved_predictions_and_targets.json')
            require([r['image_id'] for r in rows]==c['old_cohort_order'],'Old saved order differs')
            with np.load(Path(oldroot)/'prepared_matrices.npz',allow_pickle=False) as data:
                for i,(meta,row) in enumerate(zip(c['cohorts']['old128'],rows)):
                    require(meta['image_id']==row['image_id'],'Old metadata differs')
                    matrices[branch][row['image_id']]=(data[f'gt_{i}'].copy(),data[f'dt_{i}'].copy())
                    op=c['scorer_operating_threshold'] if branch=='frozen_score_base' else .25
                    base_match=row['target_matching']
                    operational=base_match
                    if op!=.25:
                        gt=self.image_gt(root,meta)
                        operational=P.MATCH.match_project_targets(gt,row['predictions'],meta['width'],meta['height'],confidence=op,iou_threshold=.5)
                    records[branch][row['image_id']]={'image_id':row['image_id'],'predictions':row['predictions'],
                        'project_conf025':base_match,'project_operational':operational,'frozen_operating_threshold':op,'reused_old_predictions':True}
        jobs=[]
        for cohort in ('new286','new_prefix47','full414'):
            members=c['cohorts'][cohort]
            for branch in BRANCHES:
                output=self.attempt/'evaluation'/cohort/branch;output.mkdir(parents=True)
                packed={};rows=[];nominal=Counter();operating=Counter()
                for i,meta in enumerate(members):
                    gt,dt=matrices[branch][meta['image_id']];packed[f'gt_{i}']=gt;packed[f'dt_{i}']=dt
                    row=records[branch][meta['image_id']];rows.append(row)
                    nominal.update(row['project_conf025']['counts']);operating.update(row['project_operational']['counts'])
                np.savez_compressed(output/'prepared_matrices.npz',**packed)
                write(output/'saved_predictions_and_targets.json',rows)
                write(output/'project_metrics.json',{'cohort':cohort,'branch':branch,'images':len(members),'conf025_counts':dict(nominal),
                    'operating_counts':dict(operating),'operating_threshold':c['scorer_operating_threshold'] if branch=='frozen_score_base' else .25})
                job={'cohort':cohort,'branch':branch,'images':len(members),'matrices':identity(output/'prepared_matrices.npz'),'output':str(output.resolve()),
                     'contract_sha256':self.live['contract']['sha256']}
                path=output/'AP_job.json';write(path,job);jobs.append(path)
        return jobs

    def execute_jobs(self,jobs):
        pending=list(jobs);running=[];completed=[];events=[]
        while pending or running:
            while pending and len(running)<2:
                path=pending.pop(0);job=read(path);folder=Path(job['output'])
                stdout=(folder/'AP_stdout.txt').open('x',encoding='utf-8');stderr=(folder/'AP_stderr.txt').open('x',encoding='utf-8')
                argv=[sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/gate1_cached_ap_job.py'),str(path)]
                proc=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                self.children.append(proc)
                record={'job':identity(path),'pid':proc.pid,'argv':argv,'started_at_utc':now()};events.append(record)
                running.append((proc,path,stdout,stderr,time.monotonic(),record))
            for item in list(running):
                proc,path,stdout,stderr,started,record=item
                code=proc.poll()
                require(time.monotonic()-started<1200,'AP job time budget exhausted')
                if code is None:continue
                stdout.close();stderr.close();running.remove(item)
                result={**record,'finished_at_utc':now(),'observed_exit_code':code}
                write(path.parent/'AP_launcher_receipt.json',result)
                require(code==0 and (path.parent/'metrics.json').exists(),'Cached AP job failed; attempt preserved')
                completed.append(str(path));self.progress('CACHED_AP_JOBS',len(completed),9,
                    completed_jobs=completed,active_jobs=[{'pid':p.pid,'job':str(j)} for p,j,*_ in running])
            if running:time.sleep(1)
        write(self.attempt/'AP_jobs_receipt.json',{'jobs':events,'all_observed_exit_codes_zero':True,'completed':completed,'max_parallelism':2})

    def collect(self):
        c=self.contract;result={}
        for cohort in ('new286','new_prefix47','full414'):
            entries={}
            curves={}
            for branch in BRANCHES:
                folder=self.attempt/'evaluation'/cohort/branch
                entries[branch]={'official':read(folder/'metrics.json'),'project':read(folder/'project_metrics.json')}
                with np.load(folder/'official_IoU05_rank_curve.npz',allow_pickle=False) as d:
                    curves[branch]={k:d[k].copy() for k in ('score','flag','threshold','TP','FP')}
            ref=curves['mse_r'];budget=int(((ref['score']>=.25)&(ref['flag']==0)).sum());target=int(((ref['score']>=.25)&(ref['flag']==1)).sum())
            for branch in BRANCHES:
                entries[branch]['at_FP_budget']=S.at_fp_budget(curves[branch],budget)
                entries[branch]['at_TP_target']=S.at_tp_target(curves[branch],target)
            apdelta=entries['frozen_score_base']['official']['metrics_percent']['AP_all']-entries['mse_r']['official']['metrics_percent']['AP_all']
            tpdelta=entries['frozen_score_base']['at_FP_budget']['TP']-entries['mse_r']['at_FP_budget']['TP']
            op=entries['frozen_score_base']['project']['operating_counts'];base=entries['mse_r']['project']['operating_counts']
            result[cohort]={'members':len(c['cohorts'][cohort]),'reference_FP_budget':budget,'reference_TP_target':target,'branches':entries,
                'score_AP_delta_vs_R_percentage_points':apdelta,'score_TP_delta_at_same_FP_budget':tpdelta,
                'joint_development_rule_met':apdelta>0 and tpdelta>=0,
                'frozen_threshold_TP_delta':op['hits']-base['hits'],'frozen_threshold_FP_delta':op['false_positive_detections']-base['false_positive_detections'],
                'operating_joint_observation':op['hits']>=base['hits'] and op['false_positive_detections']<=base['false_positive_detections']}
        write(self.attempt/'comparison_result.json',{'status':'COMPLETE_FROZEN_SCORER_EXTENSION_DEVELOPMENT_ONLY','cohorts':result,
            'new_detector_image_forwards':572,'new_model_fit_count':0,'AP_jobs':9,'frozen_threshold':c['scorer_operating_threshold'],
            'primary_new286_pass':result['new286']['joint_development_rule_met'],
            'all_three_cohorts_joint_rule_met':all(r['joint_development_rule_met'] for r in result.values()),
            'old128_results_were_used_to_choose_scorer':True,'independent_test_set':False,'scientific_acceptance':False,
            'actual_INT8':False,'Holdout_accessed':False,'Selection_accessed':False,'old_failures_preserved':True})

    def run(self):
        P.configure();self.verify();require(torch.cuda.is_available(),'CUDA required for frozen detector extension')
        root=Path(read(self.contract['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
        write(self.attempt/'runtime.json',{'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'GPU':torch.cuda.get_device_name(0),'backend':P.FT.backend_settings(),'new_model_fit_count':0})
        records,matrices=self.infer_new(root)
        jobs=self.assemble(root,records,matrices);del records,matrices
        self.execute_jobs(jobs);self.collect();self.verify()
        files=[identity(p) for p in sorted(self.attempt.rglob('*')) if p.is_file() and p.name not in {'events.jsonl','live_status.json','manifest.json','completion_receipt.json'}]
        write(self.attempt/'manifest.json',{'files':files})
        self.progress('COMPLETE_FROZEN_SCORER_EXTENSION_DEVELOPMENT_ONLY',1,1)
        write(self.attempt/'completion_receipt.json',{**self.live,'finished_at_utc':now(),'elapsed_seconds':time.monotonic()-self.started,
            'manifest':identity(self.attempt/'manifest.json'),'comparison':identity(self.attempt/'comparison_result.json')})


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
        terminated=[]
        for child in worker.children:
            if child.poll() is None:
                if os.name=='nt':
                    stop=subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    terminated.append({'owned_live_AP_job_pid':child.pid,'tree_stop_returncode':stop.returncode})
                else:child.terminate()
                child.wait()
        worker.live.update(status='FAILED_ATTEMPT_PRESERVED',error_type=type(error).__name__,error=str(error),updated_at_utc=now())
        write(worker.attempt/'live_status.json',worker.live,once=False)
        write(worker.attempt/'failure_receipt.json',{**worker.live,'traceback':traceback.format_exc(),'owned_children_terminated':terminated});raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
