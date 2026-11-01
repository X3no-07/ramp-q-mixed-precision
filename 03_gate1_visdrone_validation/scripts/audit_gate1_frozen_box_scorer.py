"""Saved scoring artifact audit; no detector inference or model fitting."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import gate1_frozen_box_scorer as S

P=S.P;OUT=S.OUT;A=OUT/"attempt_01"
read,write,identity,require=P.read,P.write,P.identity,P.require


def independently_features(raw,rows):
    """Scalar/vector feature reconstruction independent of training extractor."""
    if not rows:return np.empty((0,22),np.float64)
    a=np.array([r["raw_anchor_index"] for r in rows]);v=raw[0].T[a].astype(np.float64)
    p=v[:,4:];top=p.max(1);runner=np.sort(p,axis=1)[:,-2];cat=p.argmax(1)
    prob=p/np.maximum(p.sum(1,keepdims=True),1e-12)
    box=np.c_[v[:,:2]-v[:,2:4]/2,v[:,:2]+v[:,2:4]/2]
    area=np.maximum(0,box[:,2:]-box[:,:2]).prod(1)
    result=[]
    def lg(x):x=np.clip(x,1e-6,1-1e-6);return np.log(x)-np.log1p(-x)
    for i in range(len(rows)):
        inter=np.maximum(0,np.minimum(box[:,2:],box[i,2:])-np.maximum(box[:,:2],box[i,:2])).prod(1)
        overlap=inter/np.maximum(1e-12,area+area[i]-inter);overlap[i]=0
        same=cat==cat[i];near=same&(overlap>=.5)
        higher=(top>top[i])|((top==top[i])&(np.arange(len(rows))<i))
        f=[lg(top[i]),lg(runner[i]),top[i]-runner[i],-(prob[i]*np.log(np.maximum(prob[i],1e-12))).sum()/np.log(10),
           np.log(max(v[i,2]/640,1e-6)),np.log(max(v[i,3]/640,1e-6)),v[i,0]/640,v[i,1]/640,
           np.log1p(near.sum()),float(np.max(np.where(same&higher,overlap,0))),float(np.max(np.where(~same,overlap,0))),
           float(np.max(np.where(near,top,0))),*np.eye(10)[cat[i]].tolist()]
        result.append(f)
    return np.clip(np.array(result,np.float64).reshape(-1,22),-16,16)


def independently_scores(x,p,m):
    x=(x-np.asarray(m["mean"]))/np.asarray(m["scale"])
    offset=np.log(np.clip(p,1e-6,1-1e-6))-np.log1p(-np.clip(p,1e-6,1-1e-6))
    z=offset+x@np.array(m["weight"])+float(m["bias"])
    out=np.zeros(len(z));positive=z>=0
    out[positive]=1/(1+np.exp(-np.minimum(z[positive],700)))
    ex=np.exp(np.maximum(z[~positive],-700));out[~positive]=ex/(1+ex)
    return np.clip(out,1e-12,1-1e-12)


def independent_curve(score,flag):
    order=sorted(range(len(score)),key=lambda i:(-score[i],i))
    threshold=[];tp=[];fp=[];a=b=0;k=0
    while k<len(order):
        value=score[order[k]]
        while k<len(order) and score[order[k]]==value:
            a+=int(flag[order[k]]==1);b+=int(flag[order[k]]==0);k+=1
        threshold.append(value);tp.append(a);fp.append(b)
    return {"threshold":np.array(threshold),"TP":np.array(tp,dtype=np.int64),"FP":np.array(fp,dtype=np.int64)}


def main():
    P.torch.set_num_threads(1)
    c=read(OUT/"contract.json");complete=read(A/"completion_receipt.json")
    terminal=read(OUT/"launcher_terminal_receipt.json")
    require(complete["status"]=="COMPLETE_FIXED_BOX_SCORER_DEVELOPMENT_PILOT","Incomplete worker")
    require(complete["new_detector_image_forwards"]==0 and terminal["propagated_worker_exit_code"]==0,"Forward/exit condition failed")
    require(terminal["completion_receipt_exists"] and not terminal["failure_receipt_exists"] and terminal["stderr_bytes"]==0,"Abnormal terminal receipt")
    require(complete["contract"]==identity(OUT/"contract.json"),"Contract changed")
    for b in c["source_bindings"].values():require(identity(b["path"])==b,"Source changed")
    for b in c["previous_source_bindings"].values():require(identity(b["path"])==b,"Previous source changed")
    for b in c["reused_files"]:require(identity(b["path"])==b,"Old artifact changed")
    for key in ("candidate","architecture","train_manifest","training_val_manifest","evaluation_protocol","previous_contract","previous_delivery","previous_comparison","training_annotation_assignment"):
        require(identity(c[key]["path"])==c[key],"Input changed: "+key)
    spec=read(OUT/"saved_audit_contract.json")
    require(identity(spec["script"]["path"])==spec["script"],"Frozen audit changed")
    tree=ET.parse(OUT/"preflight_tests.xml").getroot()
    require(sum(int(x.get('tests',0)) for x in tree.iter('testsuite'))==10,"Missing tests")
    require(sum(int(x.get(k,0)) for x in tree.iter('testsuite') for k in ('errors','failures','skipped'))==0,"Preflight not PASS")
    manifest=read(A/"manifest.json")["files"]
    for b in manifest:require(identity(b["path"])==b,"Saved output changed")
    actual={str(p.resolve()) for p in A.rglob('*') if p.is_file() and p.name not in {'live_status.json','events.jsonl','manifest.json','completion_receipt.json'}}
    require(actual=={b['path'] for b in manifest} and len(actual)==len(manifest),"Manifest membership differs")
    root=Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
    # Rehash only explicit annotation members actually used; no image directory read.
    for meta in c["training_annotation_metadata"]+c["evaluation_members"]:
        path=(root/meta['annotation_path']).resolve(strict=True)
        require(path.is_relative_to(root) and P.sha(path)==meta['annotation_sha256'],"Allowed annotation differs")
    xs=[];ys=[];ps=[];sizes=[];positive=negative=commonp=commonn=0;offset=0
    chosen=read(A/"training_cache/selection.json")
    for i,meta in enumerate(c["training_annotation_metadata"]):
        r=read(A/"training_cache"/f"{i:04d}.json")
        require(r['image_id']==meta['image_id'] and r['offset']==offset,"Training cache order/offset differs")
        def pkey(x):return (x['category'],x['size_bin'])
        def nkey(x):return (x['category'],x['score_bin'],x['kind'])
        require(Counter(map(pkey,r['hard_positive']))==Counter(map(pkey,r['intervention_positive'])),"Positive strata differ")
        require(Counter(map(nkey,r['hard_negative']))==Counter(map(nkey,r['intervention_negative'])),"Negative strata differ")
        positive+=len(r['intervention_positive']);negative+=len(r['intervention_negative'])
        commonp+=len({x['row'] for x in r['hard_positive']}&{x['row'] for x in r['intervention_positive']})
        commonn+=len({x['row'] for x in r['hard_negative']}&{x['row'] for x in r['intervention_negative']})
        with np.load(A/'training_cache'/f'{i:04d}.npz',allow_pickle=False) as d:
            x,y,p=d['features'].copy(),d['target'].copy(),d['original_score'].copy()
            require(x.shape==(r['rows'],22) and np.isfinite(x).all() and ((y==0)|(y>=.5)).all() and (y<=1).all(),"Training cache domain differs")
            for arm,pk,nk in (('score_hard','hard_positive','hard_negative'),('score_intervention','intervention_positive','intervention_negative')):
                for q in r[pk]:require(y[q['row']]>=.5,"Positive label invalid")
                for q in r[nk]:require(y[q['row']]==0 and p[q['row']]>=.25,"Negative label invalid")
                expectedp=[offset+q['row'] for q in r[pk]];expectedn=[offset+q['row'] for q in r[nk]]
                require(set(expectedp).issubset(chosen[arm]['positive']) and set(expectedn).issubset(chosen[arm]['negative']),"Global selection differs")
            xs.append(x);ys.append(y);ps.append(p);sizes.append(len(x));offset+=len(x)
    train_probe_indices=np.linspace(0,511,32,dtype=int).tolist()
    for i in train_probe_indices:
        old=read(Path(c['teacher_array_root'])/f'{i:04d}.json')
        with np.load(Path(c['teacher_array_root'])/f'{i:04d}_raw.npz',allow_pickle=False) as raw,np.load(A/'training_cache'/f'{i:04d}.npz',allow_pickle=False) as cached:
            prediction=P.predictions(raw['r'],old['training_to_original_params'],old['image_id'],'audit_R')
            reconstructed=independently_features(raw['r'],prediction)
            by_anchor={p['raw_anchor_index']:j for j,p in enumerate(prediction)}
            selected=[by_anchor[int(k)] for k in cached['raw_anchor']]
            require(np.allclose(reconstructed[selected],cached['features'],atol=1e-12,rtol=1e-12),'Sampled training feature replay differs')
    x=np.concatenate(xs);y=np.concatenate(ys);p=np.concatenate(ps)
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-8,scale,1.)
    require(positive==len(chosen['score_intervention']['positive']) and negative==len(chosen['score_intervention']['negative']),"Selected count differs")
    require(len(chosen['score_hard']['positive'])==positive and len(chosen['score_hard']['negative'])==negative,"Global matched budgets differ")
    models={arm:read(A/f'{arm}_model.json') for arm in S.ARMS}
    for arm,m in models.items():
        require(m['steps']==100 and m['features']==list(S.FEATURES) and len(m['weight'])==22,"Model shape/budget differs")
        require(np.array_equal(mean,np.array(m['mean'])) and np.array_equal(scale,np.array(m['scale'])),"Normalization did not fit training features exactly")
        require(not m['GT_as_feature'] and not m['Q_path_at_score_time'] and not m['detector_weights_updated'],"Input/model scope violation")
        require([r['step'] for r in m['trace']]==[1,11,21,31,41,51,61,71,81,91,100],"Optimization trace incomplete")
        require(all(np.isfinite(t['loss']) for t in m['trace']),"Nonfinite trace")
        require(np.array_equal(independently_scores(x[:2048],p[:2048],m),S.score_rows(x[:2048],p[:2048],m)),"Saved score replay differs")
    del x,y,p,xs,ys,ps
    metrics=read(A/'comparison_result.json')['metrics'];guards=[]
    for i,meta in enumerate(c['evaluation_members']):
        cached=read(A/'development_cache'/f'{i:04d}.json')
        require(cached['image_id']==meta['image_id'],"Development member differs")
        with np.load(cached['original_raw']['path'],allow_pickle=False) as raw,np.load(A/'development_cache'/f'{i:04d}.npz',allow_pickle=False) as d:
            independent=independently_features(raw['raw'],cached['predictions'])
            require(np.allclose(independent,d['features'],atol=1e-12,rtol=1e-12),"Independent visible features differ")
            require(np.array_equal(d['original_score'],np.array([r['score'] for r in cached['predictions']])),'Cached original scores differ')
            for arm,m in models.items():
                # Checked against actual submitted rows below by anchor identity.
                expected_score=independently_scores(d['features'],d['original_score'],m)
                guards.append((arm,i,expected_score))
    per_arm={arm:{} for arm in S.ARMS}
    for arm,i,scores in guards:per_arm[arm][i]=scores
    metric_checks=[];budget=read(A/'FP_budget_reference.json')
    for m in metrics:
        arm=m['branch']
        folder=Path(c['baseline_root']) if arm=='mse_r' else A/'evaluation'/arm
        require(read(folder/'metrics.json')==m,"Metric record differs")
        with np.load(folder/'AP_AR_arrays.npz',allow_pickle=False) as d:
            ap,ar,idx=d['AP'],d['AR'],d['evalClass']-1
            values=[np.mean(ap[idx]),np.mean(ap[idx,0]),np.mean(ap[idx,5]),*[np.mean(ar[idx,:,j]) for j in range(4)]]
            for key,v in zip(P.AP.METRIC_NAMES,values):require(np.isclose(v,m['metrics_percent'][key],atol=1e-10,rtol=1e-12),"AP aggregate differs")
        if arm!='mse_r':
            rows=read(folder/'saved_predictions_and_targets.json');total=Counter()
            for i,r in enumerate(rows):
                cached=read(A/'development_cache'/f'{i:04d}.json');old={p['raw_anchor_index']:p for p in cached['predictions']}
                require(r['image_id']==c['evaluation_members'][i]['image_id'] and len(r['predictions'])==len(old),"Candidate pool changed")
                expected={p['raw_anchor_index']:float(s) for p,s in zip(cached['predictions'],per_arm[arm][i])}
                require(len(set(p['raw_anchor_index'] for p in r['predictions']))==len(old),"Duplicate submitted anchor")
                require(r['predictions']==sorted(r['predictions'],key=lambda p:(-p['score'],p['nms_input_order'])),"Submission sort differs")
                for p in r['predictions']:
                    initial=old[p['raw_anchor_index']]
                    require({k:v for k,v in p.items() if k!='score'}=={k:v for k,v in initial.items() if k!='score'},"Boxes/categories/identities changed")
                    require(p['score']==expected[p['raw_anchor_index']],"Submitted score differs from saved head")
                total.update(r['target_matching']['counts'])
            require(dict(total)==m['project_counts_conf025'],"Project count aggregate differs")
        curvepath=A/'reference_official_IoU05_rank_curve.npz' if arm=='mse_r' else folder/'official_IoU05_rank_curve.npz'
        with np.load(curvepath,allow_pickle=False) as d:
            expected_score=[];expected_flag=[]
            with np.load(folder/'prepared_matrices.npz',allow_pickle=False) as mat:
                for category in range(1,11):
                    for i in range(128):
                        gt=mat[f'gt_{i}'];dt=mat[f'dt_{i}'][:500]
                        _,match=P.AP.eval_res(gt[gt[:,5]==category,:5],dt[dt[:,5]==category,:5],.5)
                        expected_score.extend(match[:,4]);expected_flag.extend(match[:,5])
            require(np.array_equal(d['score'],np.array(expected_score)) and np.array_equal(d['flag'],np.array(expected_flag)),"One-IoU curve match flags differ from submitted matrices")
            new=independent_curve(d['score'],d['flag'])
            require(all(np.array_equal(new[k],d[k]) for k in new),"Independent whole-tie-group curve differs")
            if arm=='mse_r':
                require(int(((d['flag']==0)&(d['score']>=.25)).sum())==budget['FP_budget'],"Reference FP formula differs")
                require(int(((d['flag']==1)&(d['score']>=.25)).sum())==budget['TP_target'],"Reference TP formula differs")
        metric_checks.append({'branch':arm,'seven_aggregates_rechecked':True,'whole_tie_group_curve_rechecked':True})
    result={'status':'PASS_SAVED_SCORER_INPUT_STATE_POOL_AND_CURVE_AUDIT_ONLY','at_utc':datetime.now(timezone.utc).isoformat(),
        'contract':identity(OUT/'contract.json'),'manifest_members_verified':len(manifest),'reused_members_verified':len(c['reused_files']),
        'training_images':512,'development_images':128,'training_rows':offset,'positive_selected_each':positive,'negative_selected_each':negative,
        'hard_intervention_positive_overlap':commonp,'hard_intervention_negative_overlap':commonn,'training_only_normalization_exact':True,
        'development_features_independently_reconstructed':128,'unchanged_candidate_boxes_categories_verified':384,
        'training_feature_independent_probes':train_probe_indices,'curve_match_flags_rechecked':True,
        'new_detector_forwards':0,'model_fitting_repeated':False,'full_AP_scoring_repeated':False,'new_scorer_states_verified':3,
        'metrics_and_curves':metric_checks,'optimizer_updates_each':100,'preflight_tests':10,
        'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False}
    write(OUT/'saved_delivery_audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','manifest_members_verified','training_rows','positive_selected_each','negative_selected_each')},ensure_ascii=True))


if __name__=='__main__':main()
