"""Independent NumPy sampling/score/curve audit, no forward or refit."""
from collections import Counter
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import gate1_visual_evidence_probe as V
import audit_gate1_frozen_box_scorer as I
P=V.P;S=V.S;T=V.T;J=V.J;OUT=V.OUT;A=OUT/'attempt_01'
read,write,identity,require=P.read,P.write,P.identity,P.require


def cpu_local_visual(feature,boxes):
    f=feature[0];b=np.asarray(boxes,np.float32);off=np.asarray(V.OFFSETS,np.float32)
    coords=b[:,None,:2]+b[:,None,2:]*off[None,:,:]
    normalized=coords*np.float32(2/640)-np.float32(1)
    xy=((normalized+np.float32(1))*np.float32(.5))*np.float32(80)-np.float32(.5)
    xy=np.clip(xy,0,79);lo=np.floor(xy).astype(np.int64);hi=np.minimum(lo+1,79);frac=xy-lo
    x,y=lo[...,0],lo[...,1];xx,yy=hi[...,0],hi[...,1];dx,dy=frac[...,0,None],frac[...,1,None]
    sample=f[:,y,x].transpose(1,2,0)*(1-dx)*(1-dy)+f[:,y,xx].transpose(1,2,0)*dx*(1-dy)+f[:,yy,x].transpose(1,2,0)*(1-dx)*dy+f[:,yy,xx].transpose(1,2,0)*dx*dy
    return np.column_stack((sample[:,0],sample[:,0]-sample[:,1:].mean(1)))


def progress(stage,done,total):
    P.FT.atomic_json(OUT/'audit_live_status.json',{'stage':stage,'done':done,'total':total,'at_utc':P.now()})


def main():
    c=read(OUT/'contract.json');spec=read(OUT/'saved_audit_contract.json');terminal=read(OUT/'launcher_terminal_receipt.json')
    require(spec['script']==identity(__file__) and spec['score_helper']==identity(I.__file__),'Frozen audit changed')
    require(terminal['observed_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and terminal['stderr_bytes']==0,'Probe did not terminate cleanly')
    complete=read(A/'completion_receipt.json');require(complete['forward_images']==690 and complete['new_model_fit_count']==1 and complete['contract']==identity(OUT/'contract.json'),'Probe budget/contract differs')
    require(c['sources']==V.source_bindings() and c['framework']==P.verify_framework_sources(),'Frozen sources changed')
    for k in ('candidate','architecture','training_manifest','evaluation_protocol','old_scorer','ranges','previous_contract','previous_delivery','old_reference_entries','old_scored_entries','preflight_tests'):
        require(identity(c[k]['path'])==c[k],'Bound source changed')
    tests=ET.parse(c['preflight_tests']['path']).getroot();require(sum(int(x.get('tests',0)) for x in tests.iter('testsuite'))==6 and sum(int(x.get(k,0)) for x in tests.iter('testsuite') for k in ('failures','errors','skipped'))==0,'Preflight tests differ')
    manifest=read(A/'manifest.json')['files']
    for b in manifest:require(identity(b['path'])==b,'Saved probe member changed')
    actual={str(p.resolve()) for p in A.rglob('*') if p.is_file() and p.name not in ('live_status.json','events.jsonl','manifest.json','completion_receipt.json')}
    require(actual=={b['path'] for b in manifest} and len(actual)==len(manifest),'Manifest coverage differs')
    capture=read(A/'extraction_checks.json');require(capture['equal'] and capture['before']==capture['after'] and len(capture['checks'])==690,'Frozen detector capture checks differ')
    xs=[];maxerr=0.;bindings=0
    for role in ('fit','calibration'):
        for i,row in enumerate(c[role]):
            for k in (('features','raw','geometry') if role=='fit' else ('features','predictions','matrices')):
                require(identity(row[k]['path'])==row[k],'Saved input cache changed');bindings+=1
            with np.load(row['features']['path'],allow_pickle=False) as d:oldx=d['features'].copy();oldanchors=d['raw_anchor'].copy() if role=='fit' else None
            with np.load(row['raw']['path'] if role=='fit' else row['features']['path'],allow_pickle=False) as d:raw=d['r' if role=='fit' else 'raw'].copy()
            with np.load(A/role/f'{i:04d}.npz',allow_pickle=False) as d:
                anchors=d['raw_anchor'].copy();boxes=d['boxes'].copy();visual=d['visual'].copy();p3=d['p3'].copy()
                require(str(d['raw_sha256'])==P.hashlib.sha256(raw.tobytes()).hexdigest(),'All8400 raw identity mismatch')
            if role=='fit':require(np.array_equal(anchors,oldanchors),'Fit anchor order changed')
            else:require(np.array_equal(anchors,[r['raw_anchor_index'] for r in read(row['predictions']['path'])['mse_r']]),'Calibration anchor order changed')
            require(np.array_equal(boxes,raw[0,:4][:,anchors].T) and p3.shape==(1,64,80,80) and np.isfinite(p3).all(),'Canvas box/P3 domain differs')
            reconstructed=cpu_local_visual(p3,boxes);error=float(np.max(np.abs(reconstructed-visual))) if len(visual) else 0.;maxerr=max(maxerr,error)
            require(np.allclose(reconstructed,visual,atol=c['feature_spec']['independent_CPU_replay_atol'],rtol=c['feature_spec']['independent_CPU_replay_rtol']),'Independent NumPy bilinear feature replay differs')
            if role=='fit':xs.append(np.column_stack((oldx,visual.astype(np.float64))))
            if (i+1)%32==0 or i+1==len(c[role]):progress('INDEPENDENT_NUMPY_SAMPLING_'+role.upper(),i+1,len(c[role]))
    model=read(A/'visual_scorer_model.json');x=np.concatenate(xs);del xs
    mean,scale=S.standardize_training(x);require(x.shape==(213716,150) and np.array_equal(mean,model['mean']) and np.array_equal(scale,model['scale']),'Training-only normalization differs')
    require(model['steps']==100 and model['parameters']==151 and [r['step'] for r in model['trace']]==[1,11,21,31,41,51,61,71,81,91,100],'Model/optimizer budget differs');del x
    result=read(A/'comparison_result.json');records=read(A/'calibration_predictions/prediction_records.json');oldmodel=read(c['old_scorer']['path'])
    root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True);gts=[];dts=[];losses=[]
    for i,(row,record) in enumerate(zip(c['calibration'],records)):
        meta=row['metadata'];pred=read(row['predictions']['path'])['mse_r']
        require(record['image_id']==meta['image_id'],'Calibration saved order changed')
        with np.load(row['features']['path'],allow_pickle=False) as d:x=d['features'].copy();p=d['original_score'].copy();oldscore=d['new_score'].copy()
        with np.load(A/'calibration'/f'{i:04d}.npz',allow_pickle=False) as d:v=d['visual'].astype(np.float64)
        values=I.independently_scores(np.column_stack((x,v)),p,model)
        require(np.array_equal(I.independently_scores(x,p,oldmodel),oldscore),'Original scorer replay differs')
        expected=sorted([{**d,'score':float(s)} for d,s in zip(pred,values)],key=lambda d:(-d['score'],d['nms_input_order']))
        require(expected==record['predictions'],'Saved new score/order/pool differs')
        ann=(root/meta['annotation_path']).resolve(strict=True);require(ann.is_relative_to(root) and P.sha(ann)==meta['annotation_sha256'],'Allowed GT changed')
        gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id']);match=P.MATCH.match_project_targets(gt,pred,meta['width'],meta['height'],confidence=.001,iou_threshold=.5)
        bymatch={r['detection_id']:r['iou'] for r in match['matches']};states={r['detection_id']:r['reason'] for r in match['detection_states']}
        valid=np.array([j for j,r in enumerate(pred) if states[r['detection_id']] in ('true_positive','false_positive')],np.int64);y=np.array([bymatch.get(r['detection_id'],0.) for r in pred]);per={}
        for label,scores in (('original_R',p),('old_generic',oldscore),('local_visual',values)):
            z=np.clip(scores[valid],1e-12,1-1e-12);per[label]=float(-(y[valid]*np.log(z)+(1-y[valid])*np.log1p(-z)).mean())
        require(per==record['quality_losses'],'Independent calibration loss differs');losses.append(per)
        gt8=np.array([g['bbox_xywh']+[g['score'],g['category'],g['truncation'],g['occlusion']] for g in gt],float).reshape(-1,8)
        dt8=np.array([r['bbox_xywh']+[r['score'],r['category'],-1,-1] for r in expected],float).reshape(-1,8)
        prepared=P.AP.prepare_raw_image(gt8,dt8,width=meta['width'],height=meta['height'])
        with np.load(A/'calibration_predictions'/f'{i:04d}.npz',allow_pickle=False) as d:
            require(np.array_equal(d['score'],values) and np.array_equal(d['valid'],valid) and np.array_equal(d['target'],y) and np.array_equal(d['gt'],prepared['prepared_gt']) and np.array_equal(d['dt'],prepared['filtered_predictions']),'Prediction/GT matrix transport differs');gts.append(d['gt'].copy());dts.append(d['dt'].copy())
        if (i+1)%32==0 or i==177:progress('INDEPENDENT178SCORES_TARGETS_AND_MATRICES',i+1,178)
    score,flag=J.rank_entries(gts,dts)
    with np.load(A/'local_visual_IoU05_rank_curve.npz',allow_pickle=False) as d:
        require(np.array_equal(d['score'],score) and np.array_equal(d['flag'],flag),'OneIoU official matching replay differs');curve=I.independent_curve(score,flag)
        require(all(np.array_equal(d[k],curve[k]) for k in curve),'Independent whole-score-tie curve differs')
    with np.load(c['old_reference_entries']['path'],allow_pickle=False) as d:ref=T.fixed_counts(d['score'],d['flag'],.25)
    with np.load(c['old_scored_entries']['path'],allow_pickle=False) as d:oldpoint=S.at_fp_budget(I.independent_curve(d['score'],d['flag']),ref['FP'])
    point=S.at_fp_budget(curve,ref['FP']);loss={k:float(np.mean([r[k] for r in losses])) for k in losses[0]}
    require(ref==result['reference_R_at025'] and oldpoint==result['old_generic_at_R_FP_budget'] and point==result['local_visual_at_R_FP_budget'] and loss==result['image_balanced_quality_BCE'],'Final diagnostic aggregates differ')
    require(result['diagnostic_promising']==(point['TP']>oldpoint['TP'] and loss['local_visual']<loss['old_generic']),'Diagnostic rule differs')
    audit={'status':'PASS_LOCAL_VISUAL_SAVED_INFORMATION_DIAGNOSTIC_AUDIT_ONLY','at_utc':P.now(),'contract':identity(OUT/'contract.json'),
        'manifest_members_verified':len(manifest),'input_cache_bindings_verified':bindings,'new_detector_forwards':0,'model_fits_repeated':0,
        'NumPy_visual_images_verified':690,'visual_replay_max_abs_error':maxerr,'predeclared_visual_tolerances':c['feature_spec'],
        'all8400raw_identity_checks':690,'calibration_score_and_target_checks':178,'whole_tie_curve_and_original_budget_recomputed':True,
        'normalization_fit_only':True,'diagnostic_promising':result['diagnostic_promising'],'scientific_acceptance':False,'actual_INT8':False,
        'training_val_payload_access':False,'Selection_payload_access':False,'Holdout_payload_access':False}
    write(OUT/'saved_delivery_audit.json',audit);progress('COMPLETE_INDEPENDENT_VISUAL_PROBE_AUDIT',1,1)
    print(json.dumps({k:audit[k] for k in ('status','manifest_members_verified','NumPy_visual_images_verified','visual_replay_max_abs_error','diagnostic_promising')}))


if __name__=='__main__':main()
