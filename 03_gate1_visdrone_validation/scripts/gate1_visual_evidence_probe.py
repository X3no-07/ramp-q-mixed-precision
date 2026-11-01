"""One frozen local-visual-information probe on512fit/178calibration images only.

This is an information diagnostic, not a new method claim or development AP run.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import shutil
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
import gate1_frozen_box_scorer as S
import gate1_separate_threshold as T
import gate1_scorer_extension as E
import gate1_cached_ap_job as J
P=S.P
ROOT=P.ROOT
OUT=ROOT/'artifacts/gate1/20261026_local_visual_information_probe_v1'
read,write,identity,require=P.read,P.write,P.identity,P.require
OFFSETS=((0.,0.),(-.75,-.75),(0.,-.75),(.75,-.75),(-.75,0.),(.75,0.),(-.75,.75),(0.,.75),(.75,.75))


def local_visual(feature,boxes):
    """128visual features, no labels/classes/scores/group identity accepted."""
    require(tuple(feature.shape)==(1,64,80,80),'Frozen P3 feature domain differs')
    boxes=np.asarray(boxes,np.float32).reshape(-1,4)
    require(np.isfinite(boxes).all() and (boxes[:,2:]>0).all(),'Invalid canvas XYWH')
    if not len(boxes):return np.empty((0,128),np.float32)
    b=torch.from_numpy(boxes.copy()).to(feature.device)
    offsets=torch.tensor(OFFSETS,device=feature.device,dtype=feature.dtype)
    coords=b[:,None,:2]+b[:,None,2:]*offsets[None,:,:]
    grid=(coords*(2./640)-1).unsqueeze(0)
    sampled=F.grid_sample(feature,grid,mode='bilinear',padding_mode='border',align_corners=False)[0].permute(1,2,0)
    center=sampled[:,0];contrast=center-sampled[:,1:].mean(1)
    return torch.cat((center,contrast),1).detach().cpu().numpy().copy()


def source_bindings():
    return {p.name:identity(p) for p in (Path(__file__),ROOT/'scripts/launch_gate1_visual_evidence_probe.py',
        ROOT/'tests/test_gate1_visual_evidence_probe.py',Path(P.__file__),Path(S.__file__),Path(T.__file__),Path(E.__file__),Path(J.__file__),
        Path(P.DS.__file__),Path(F.__file__))}


def prepare():
    OUT.mkdir(exist_ok=False)
    s=read(S.OUT/'contract.json');t=read(T.OUT/'contract.json')
    fit=s['training_annotation_metadata'];cal=t['calibration_metadata']
    require(len(fit)==512 and len(cal)==178,'Membership count differs')
    for key in ('image_id','image_sha256','prefix_proxy'):
        require(not ({m[key] for m in fit}&{m[key] for m in cal}),'Fit/calibration isolation differs')
    records=[];teacher=Path(s['teacher_array_root'])
    for i,m in enumerate(fit):records.append({'index':i,'metadata':m,'features':identity(S.OUT/'attempt_01/training_cache'/f'{i:04d}.npz'),
        'raw':identity(teacher/f'{i:04d}_raw.npz'),'geometry':identity(teacher/f'{i:04d}.json')})
    calibration=[]
    for i,m in enumerate(cal):
        folder=T.OUT/'attempt_01/calibration'/f'{i:04d}'
        calibration.append({'index':i,'metadata':m,'features':identity(folder/'raw_and_features.npz'),
            'predictions':identity(folder/'predictions.json'),'matrices':identity(folder/'prepared_matrices.npz')})
    c={'schema':'rampq.gate1.local_visual_information_probe/v1','created_at_utc':P.now(),
       'authorization':'User: Proceed! What now; validate new local visual information instead of continuing to tune old scorer features, with limited diagnosis confined to training data only',
       'hypothesis':'P3center activation and center-minus-surround features add useful quality ranking information beyond the existing22scalar features on prefix-disjoint scorer calibration data.',
       'sources':source_bindings(),'framework':P.verify_framework_sources(),
       'preflight_tests':identity(ROOT/'artifacts/gate1/20261026_local_visual_probe_preflight.xml'),
       'previous_contract':identity(ROOT/'artifacts/gate1/20261026_group_balanced_scorer_v1/contract.json'),
       'previous_delivery':identity(ROOT/'artifacts/gate1/20261026_group_balanced_scorer_v1/delivery_validation.json'),
       'candidate':s['candidate'],'architecture':s['architecture'],'training_manifest':s['train_manifest'],
       'training_members':s['training_members'],'evaluation_protocol':s['evaluation_protocol'],
       'old_scorer':t['scorer_model'],'ranges':t['ranges'],'fit':records,'calibration':calibration,
       'old_reference_entries':identity(T.OUT/'attempt_01/calibration/reference_rank_entries.npz'),
       'old_scored_entries':identity(T.OUT/'attempt_01/calibration/scored_rank_entries.npz'),
       'feature_spec':{'layer':16,'shape':[1,64,80,80],'offsets':OFFSETS,'canvas':640,'interpolation':'bilinear,align_corners=False,padding_mode=border',
           'visual_features':'64center activations followed by64center-minus-mean-of8surround activations; same frozen R forward',
           'total_features':150,'original_features':list(S.FEATURES),'GT_or_group_as_input':False,
           'independent_CPU_replay_atol':1e-4,'independent_CPU_replay_rtol':2e-5},
       'training':'One150feature residual linear scorer+1bias; same512images/213716savedsofttargets; imagebalanced common BCE;100CPUfloat64 Adam updates;lr.03,betas.9/.999,eps1e-8,L2.0005;normalization fit only;zero residual initialization;no auxiliary/hyperparameter/feature/seed search.',
       'primary_diagnostic':'On178scorer-fit-prefix-disjoint calibration images, compare NEW and old22feature scorer: lower image-balanced soft-quality BCE AND strictly higher official IoU.5/cap500 TP at the SAME originalR@.25 FP budget. This is a ranking curve diagnostic,not a frozen deployment threshold or independent scientific test.',
       'stop_rule':'Exactly1newcandidate/100updates,690newimageforwards to obtain previously unsaved P3 and require full8400raw identical to bound originalR. No development AP or additional candidates based on outcome.',
       'candidate_budget':1,'cumulative_new_scorer_candidates':7,'real_image_forward_budget':690,'new_detector_training_updates':0,
       'new_full_AP_jobs':0,'new_deployment_thresholds':0,'timeout_seconds':2400,'automatic_retry':False,
       'fit_preprocess':'Existing SelectedTrainingDataset with original exact GPUfloat normalization;calibration uses existing CPU-normalized LetterBox. Both raw outputs must equal original saved R bitwise.',
       'prior_art':[{'name':'IoU-Net','url':'https://arxiv.org/abs/1807.11590'},{'name':'GFLV2','url':'https://arxiv.org/abs/2011.12885'}],
       'not_full_prior_art_reproduction':True,'no_novelty_claim':True,'scorer_fit_disjoint_not_detector_train_disjoint':True,
       'training_val_payload_access':False,'Selection_payload_access':False,'Holdout_payload_access':False,'scientific_acceptance':False,'actual_INT8':False}
    write(OUT/'contract.json',c)
    print(json.dumps({'status':'FROZEN_ONE_LOCAL_VISUAL_INFORMATION_PROBE','contract':identity(OUT/'contract.json'),'fit':512,'calibration':178,'image_forwards':690},ensure_ascii=True))


class Worker(P.Worker):
    def __init__(self):
        self.contract=read(OUT/'contract.json');self.attempt=OUT/'attempt_01';self.attempt.mkdir(exist_ok=False)
        self.started=time.monotonic();self.live={'pid':P.os.getpid(),'parent_pid':P.os.getppid(),'started_at_utc':P.now(),
            'contract':identity(OUT/'contract.json'),'forward_images':0,'new_model_fit_count':0,'status':'STARTING'}
        self.progress('STARTING',0,1)

    def verify(self):
        c=self.contract;require(c['sources']==source_bindings(),'Frozen probe source changed')
        require(c['framework']==P.verify_framework_sources(),'Framework source changed')
        for k in ('candidate','architecture','training_manifest','evaluation_protocol','old_scorer','ranges','previous_contract','previous_delivery','old_reference_entries','old_scored_entries'):
            require(identity(c[k]['path'])==c[k],'Bound input changed: '+k)
        for row in c['fit']:
            for k in ('features','raw','geometry'):require(identity(row[k]['path'])==row[k],'Fit cache changed')
        for row in c['calibration']:
            for k in ('features','predictions','matrices'):require(identity(row[k]['path'])==row[k],'Calibration cache changed')
        require(shutil.disk_usage(ROOT).free>8*1024**3,'Insufficient8GiB free working space')

    def extract(self):
        c=self.contract;root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
        dataset=P.SelectedTrainingDataset(c['training_manifest'],c['training_members'])
        model=P.new_model(read(c['ranges']['path'])['max_abs']);P.precision(model,'r_only:G1')
        before=E.state_value_fingerprint(model);capture={}
        def hook(_module,_args,result):capture['p3']=result.detach().clone()
        handle=model.model[16].register_forward_hook(hook);checks=[]
        for role in ('fit','calibration'):
            folder=self.attempt/role;folder.mkdir()
            for i,row in enumerate(c[role]):
                meta=row['metadata']
                if role=='fit':
                    tensor=P.gpu_batch([dataset[i]])['img']
                    with np.load(row['raw']['path'],allow_pickle=False) as d:expected=d['r'].copy()
                    with np.load(row['features']['path'],allow_pickle=False) as d:anchors=d['raw_anchor'].copy();oldx=d['features'].copy();prob=d['original_score'].copy()
                else:
                    tensor,_=self.image_input(root,meta)
                    with np.load(row['features']['path'],allow_pickle=False) as d:expected=d['raw'].copy();oldx=d['features'].copy();prob=d['original_score'].copy()
                    anchors=np.array([d['raw_anchor_index'] for d in read(row['predictions']['path'])['mse_r']],np.int64)
                capture.clear();raw=self.forward(model,tensor)
                require(np.array_equal(raw,expected),'Frozen8400anchor raw output changed during new feature capture: '+meta['image_id'])
                require(np.array_equal(raw[0,4:][:,anchors].T.max(1),prob),'Saved anchor score alignment differs')
                boxes=raw[0,:4][:,anchors].T.copy();p3=capture['p3'];visual=local_visual(p3,boxes)
                require(visual.shape==(len(oldx),128) and np.isfinite(visual).all(),'Local visual feature domain differs')
                np.savez_compressed(folder/f'{i:04d}.npz',p3=p3.cpu().numpy(),boxes=boxes,raw_anchor=anchors,visual=visual,
                    input_sha256=np.array(hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()),
                    raw_sha256=np.array(hashlib.sha256(raw.tobytes()).hexdigest()))
                checks.append({'role':role,'index':i,'image_id':meta['image_id'],'raw_all8400_bitwise_identical':True,'visual_rows':len(visual)})
                if (i+1)%16==0 or i+1==len(c[role]):self.progress('EXTRACT_FROZEN_R_P3_'+role.upper(),i+1,len(c[role]))
        handle.remove();after=E.state_value_fingerprint(model);require(before==after,'Frozen R detector state changed')
        write(self.attempt/'extraction_checks.json',{'checks':checks,'before':before,'after':after,'equal':True,'forwards':self.live['forward_images']})
        del model;torch.cuda.empty_cache()

    def fit(self):
        xs=[];ys=[];ps=[];images=[]
        for i,row in enumerate(self.contract['fit']):
            with np.load(row['features']['path'],allow_pickle=False) as d:x=d['features'].copy();y=d['target'].copy();p=d['original_score'].copy()
            with np.load(self.attempt/'fit'/f'{i:04d}.npz',allow_pickle=False) as d:v=d['visual'].astype(np.float64)
            xs.append(np.column_stack((x,v)));ys.append(y);ps.append(p);images.append(np.full(len(p),i,np.int64))
        x,y,p,image=np.concatenate(xs),np.concatenate(ys),np.concatenate(ps),np.concatenate(images);del xs,ys,ps,images
        require(x.shape==(213716,150),'Training dimensions differ')
        mean,scale=S.standardize_training(x);z=torch.from_numpy(np.ascontiguousarray((x-mean)/scale));target=torch.from_numpy(y);offset=torch.from_numpy(S.logit(p))
        counts=np.bincount(image);weights=1/(len(counts)*counts[image]);weights=torch.from_numpy(weights)
        w=torch.zeros(150,dtype=torch.float64,requires_grad=True);b=torch.zeros((),dtype=torch.float64,requires_grad=True)
        optimizer=torch.optim.Adam([w,b],lr=.03,betas=(.9,.999),eps=1e-8);trace=[]
        for step in range(100):
            optimizer.zero_grad(set_to_none=True);logits=offset+z@w+b
            base=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*weights).sum()
            reg=.0005*(w.square().sum()+b.square());loss=base+reg
            require(bool(torch.isfinite(loss)),'Nonfinite probe loss');loss.backward()
            require(torch.isfinite(w.grad).all() and torch.isfinite(b.grad),'Nonfinite probe gradients');optimizer.step()
            if step%10==0 or step==99:
                trace.append({'step':step+1,'loss':float(loss.detach()),'base':float(base.detach()),'regularizer':float(reg.detach())})
                self.progress('FIT_ONE150FEATURE_SCORER',step+1,100)
        model={'mean':mean.tolist(),'scale':scale.tolist(),'weight':w.detach().numpy().tolist(),'bias':float(b.detach()),
            'steps':100,'features':[*S.FEATURES,*[f'center_c{k}' for k in range(64)],*[f'center_minus_surround_c{k}' for k in range(64)]],
            'trace':trace,'fit_images':512,'fit_rows':len(y),'parameters':151,'detector_weights_updated':False,'GT_as_feature':False}
        require(np.isfinite(np.r_[model['weight'],model['bias']]).all(),'Invalid probe state')
        write(self.attempt/'visual_scorer_model.json',model);self.live['new_model_fit_count']=1
        return model

    def diagnose(self,model):
        c=self.contract;oldmodel=read(c['old_scorer']['path']);root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
        gt_all=[];dt_all=[];losses=[];saved=[];target_rows=[]
        folder=self.attempt/'calibration_predictions';folder.mkdir()
        for i,row in enumerate(c['calibration']):
            meta=row['metadata'];old=read(row['predictions']['path']);pred=old['mse_r']
            with np.load(row['features']['path'],allow_pickle=False) as d:x=d['features'].copy();p=d['original_score'].copy();oldscore=d['new_score'].copy()
            require(np.array_equal(S.score_rows(x,p,oldmodel),oldscore),'Old22feature scorer changed')
            with np.load(self.attempt/'calibration'/f'{i:04d}.npz',allow_pickle=False) as d:v=d['visual'].astype(np.float64)
            scores=S.score_rows(np.column_stack((x,v)),p,model)
            new=sorted([{**d,'score':float(s)} for d,s in zip(pred,scores)],key=lambda d:(-d['score'],d['nms_input_order']))
            ann=(root/meta['annotation_path']).resolve(strict=True);require(ann.is_relative_to(root) and P.sha(ann)==meta['annotation_sha256'],'Allowed calibration GT changed')
            gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id'])
            matched=P.MATCH.match_project_targets(gt,pred,meta['width'],meta['height'],confidence=.001,iou_threshold=.5)
            bymatch={d['detection_id']:d['iou'] for d in matched['matches']};bystate={d['detection_id']:d['reason'] for d in matched['detection_states']}
            valid=np.array([j for j,d in enumerate(pred) if bystate[d['detection_id']] in ('true_positive','false_positive')],np.int64)
            y=np.array([bymatch.get(d['detection_id'],0.) for d in pred]);per={}
            require(len(valid)>0,'No valid calibration quality rows')
            for name,values in (('original_R',p),('old_generic',oldscore),('local_visual',scores)):
                clipped=np.clip(values[valid],1e-12,1-1e-12)
                per[name]=float(-(y[valid]*np.log(clipped)+(1-y[valid])*np.log1p(-clipped)).mean())
            losses.append(per);target_rows.append({'index':i,'valid':len(valid),'TP_rows':int((y[valid]>0).sum())})
            with np.load(row['matrices']['path'],allow_pickle=False) as d:original_gt=d['gt'].copy()
            gt8=np.array([g['bbox_xywh']+[g['score'],g['category'],g['truncation'],g['occlusion']] for g in gt],np.float64).reshape(-1,8)
            dt8=np.array([d['bbox_xywh']+[d['score'],d['category'],-1,-1] for d in new],np.float64).reshape(-1,8)
            prep=P.AP.prepare_raw_image(gt8,dt8,width=meta['width'],height=meta['height']);require(np.array_equal(prep['prepared_gt'],original_gt),'Canonical calibration GT differs')
            gt_all.append(prep['prepared_gt']);dt_all.append(prep['filtered_predictions'])
            np.savez_compressed(folder/f'{i:04d}.npz',score=scores,valid=valid,target=y,gt=prep['prepared_gt'],dt=prep['filtered_predictions'])
            saved.append({'index':i,'image_id':meta['image_id'],'predictions':new,'quality_losses':per})
            if (i+1)%16==0 or i==177:self.progress('CALIBRATION_ONLY_INFORMATION_DIAGNOSTIC',i+1,178)
        write(folder/'prediction_records.json',saved)
        score,flag=J.rank_entries(gt_all,dt_all);curve=J.curve_from_entries(score,flag)
        np.savez_compressed(self.attempt/'local_visual_IoU05_rank_curve.npz',score=score,flag=flag,**curve)
        with np.load(c['old_reference_entries']['path'],allow_pickle=False) as d:ref=T.fixed_counts(d['score'],d['flag'],.25)
        with np.load(c['old_scored_entries']['path'],allow_pickle=False) as d:oldpoint=S.at_fp_budget({k:d[k] for k in ('threshold','TP','FP')},ref['FP'])
        point=S.at_fp_budget(curve,ref['FP']);loss={k:float(np.mean([r[k] for r in losses])) for k in losses[0]}
        promising=point['TP']>oldpoint['TP'] and loss['local_visual']<loss['old_generic']
        result={'status':'COMPLETE_ONE_LOCAL_VISUAL_CALIBRATION_INFORMATION_PROBE','diagnostic_promising':promising,
            'calibration_images':178,'calibration_prefix_proxies':len({r['metadata']['prefix_proxy'] for r in c['calibration']}),
            'reference_R_at025':ref,'old_generic_at_R_FP_budget':oldpoint,'local_visual_at_R_FP_budget':point,
            'TP_delta_vs_old_generic_at_budget':point['TP']-oldpoint['TP'],'image_balanced_quality_BCE':loss,
            'BCE_delta_vs_old_generic':loss['local_visual']-loss['old_generic'],'valid_target_rows_by_image':target_rows,
            'ranking_threshold_is_diagnostic_only':True,'new_deployment_threshold':None,'new_full_AP_jobs':0,'development_evaluated':False,
            'new_detector_image_forwards':690,'new_scorer_fits':1,'updates':100,'cumulative_new_scorer_candidates':7,
            'not_an_independent_test':True,'not_a_novel_method_claim':True,'not_a_complete_IoUNet_reproduction':True,
            'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
            'interpretation':'Information feasibility only. Even a positive diagnostic needs a distinct mechanism, frozen development check and genuinely independent scientific validation. Negative result ends this registered candidate; no automatic feature/model search.'}
        write(self.attempt/'comparison_result.json',result)

    def run(self):
        P.configure();self.verify();self.extract();model=self.fit();self.diagnose(model);self.verify()
        require(self.live['forward_images']==690 and self.live['new_model_fit_count']==1,'Probe budget incomplete')
        files=[identity(p) for p in sorted(self.attempt.rglob('*')) if p.is_file() and p.name not in ('events.jsonl','live_status.json','manifest.json','completion_receipt.json')]
        write(self.attempt/'manifest.json',{'files':files});self.progress('COMPLETE_ONE_LOCAL_VISUAL_CALIBRATION_INFORMATION_PROBE',1,1)
        write(self.attempt/'completion_receipt.json',{**self.live,'finished_at_utc':P.now(),'elapsed_seconds':time.monotonic()-self.started,
            'manifest':identity(self.attempt/'manifest.json'),'comparison':identity(self.attempt/'comparison_result.json')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));args=parser.parse_args()
    if args.mode=='prepare':prepare();return
    worker=Worker()
    try:worker.run()
    except BaseException as error:
        write(worker.attempt/'failure_receipt.json',{'status':'FAILED_LOCAL_VISUAL_PROBE_NO_AUTOMATIC_RETRY','at_utc':P.now(),
            'live':worker.live,'error':repr(error),'traceback':traceback.format_exc(),'historical_results_changed':False})
        raise


if __name__=='__main__':main()
