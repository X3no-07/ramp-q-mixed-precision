"""Finite frozen-box scoring pilot, using saved model outputs only.

The detector, boxes, labels, pre-existing NMS candidate pool and all historic
artifacts are read-only. No new model image forward or sealed-data payload.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/gate1/20261026_frozen_box_scorer_pilot_v1"
os.environ["YOLO_CONFIG_DIR"]=str(OUT/"settings")
import numpy as np
import torch
from torch.nn import functional as F
import gate1_intervention_training_v2 as P
import gate1_native_target_correction as N

ARMS=("score_base","score_hard","score_intervention")
FEATURES=("logit_score","logit_runnerup","class_margin","class_entropy",
          "log_width_over640","log_height_over640","center_x_over640","center_y_over640",
          "log_same_class_neighbors_IoU05","max_higher_same_class_IoU","max_other_class_IoU","max_same_neighbor_score",
          *[f"class_{k}" for k in range(10)])
read,write,sha,identity,require,now=P.read,P.write,P.sha,P.identity,P.require,P.now


def logit(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
    return np.log(p)-np.log1p(-p)


def visible_features(raw,predictions):
    """No annotations, target IDs, Q path or label-derived quantities admitted."""
    n=len(predictions)
    if not n:return np.empty((0,len(FEATURES)),np.float64)
    ids=np.array([d["raw_anchor_index"] for d in predictions],dtype=np.int64)
    require(raw.ndim==3 and raw.shape[0:2]==(1,14) and np.isfinite(raw).all(),"Raw domain differs")
    require(len(set(ids.tolist()))==n,"Duplicate raw anchor in fixed NMS pool")
    scores=raw[0,4:][:,ids].T
    require(scores.shape==(n,10),"Class score indexing differs")
    scores=scores.astype(np.float64)
    cls=np.array([d["category"]-1 for d in predictions])
    require(np.array_equal(scores.argmax(1),cls),"Fixed candidate category differs from raw argmax")
    maximum=scores.max(1);second=np.partition(scores,-2,axis=1)[:,-2]
    require(np.array_equal(maximum,np.array([d["score"] for d in predictions])),"Candidate score differs from raw")
    distribution=scores/np.maximum(scores.sum(1,keepdims=True),1e-12)
    entropy=-(distribution*np.log(np.maximum(distribution,1e-12))).sum(1)/np.log(10)
    boxes=raw[0,:4][:,ids].T
    require(boxes.shape==(n,4) and (boxes[:,2:]>0).all(),"Fixed candidate geometry differs")
    boxes=boxes.astype(np.float64)
    corners=np.column_stack((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2))
    overlap=P.box_iou_matrix(corners,corners)
    np.fill_diagonal(overlap,0)
    same=cls[:,None]==cls[None,:]
    close=same&(overlap>=.5)
    higher=(maximum[None,:]>maximum[:,None])|((maximum[None,:]==maximum[:,None])&(np.arange(n)[None,:]<np.arange(n)[:,None]))
    out=np.column_stack((logit(maximum),logit(second),maximum-second,entropy,
        np.log(np.maximum(boxes[:,2]/640,1e-6)),np.log(np.maximum(boxes[:,3]/640,1e-6)),
        boxes[:,0]/640,boxes[:,1]/640,np.log1p(close.sum(1)),
        np.where(same&higher,overlap,0).max(1),np.where(~same,overlap,0).max(1),
        np.where(close,maximum[None,:],0).max(1),np.eye(10)[cls]))
    out=np.clip(out,-16,16)
    require(out.shape==(n,len(FEATURES)) and np.isfinite(out).all(),"Feature shape/finite mismatch")
    return out


def standardize_training(x):
    mean=x.mean(0);scale=x.std(0)
    scale=np.where(scale>1e-8,scale,1.)
    return mean,scale


def score_rows(x,p,model):
    z=(np.asarray(x,dtype=np.float64)-np.array(model["mean"]))/np.array(model["scale"])
    logits=logit(p)+z@np.array(model["weight"])+float(model["bias"])
    scores=np.empty_like(logits)
    positive=logits>=0
    scores[positive]=1/(1+np.exp(-np.minimum(logits[positive],700)))
    ex=np.exp(np.maximum(logits[~positive],-700));scores[~positive]=ex/(1+ex)
    return np.clip(scores,1e-12,1-1e-12)


def fit_scorer(x,y,p,image_index,selected_pos,selected_neg,arm):
    require(arm in ARMS and len(x)==len(y)==len(p)==len(image_index),"Invalid training input")
    require(len(selected_pos)>0 and len(selected_neg)>0,"Empty selected supervision")
    mean,scale=standardize_training(x)
    z=torch.from_numpy(np.ascontiguousarray((x-mean)/scale))
    target=torch.from_numpy(y.astype(np.float64));offset=torch.from_numpy(logit(p))
    counts=np.bincount(image_index)
    weights=1/(counts[image_index]*len(counts))
    require(np.isclose(weights.sum(),1),"Image-balanced objective normalization differs")
    weights=torch.from_numpy(weights)
    w=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    b=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([w,b],lr=.03,betas=(.9,.999),eps=1e-8)
    pos=torch.tensor(selected_pos,dtype=torch.long);neg=torch.tensor(selected_neg,dtype=torch.long)
    trace=[]
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        predicted=offset+z@w+b
        element=F.binary_cross_entropy_with_logits(predicted,target,reduction="none")
        base=(element*weights).sum()
        aux=predicted.sum()*0 if arm=="score_base" else .5*(element[pos].mean()+element[neg].mean())
        regularizer=.0005*(w.square().sum()+b.square())
        loss=base+.25*aux+regularizer
        require(bool(torch.isfinite(loss)),"Nonfinite score loss")
        loss.backward()
        require(torch.isfinite(w.grad).all() and torch.isfinite(b.grad),"Nonfinite score gradient")
        optimizer.step()
        if step%10==0 or step==99:trace.append({"step":step+1,"loss":float(loss.detach()),"base":float(base.detach()),"auxiliary":float(aux.detach()),"regularizer":float(regularizer.detach())})
    model={"arm":arm,"features":list(FEATURES),"mean":mean.tolist(),"scale":scale.tolist(),
           "weight":w.detach().numpy().tolist(),"bias":float(b.detach()),"steps":100,
           "train_rows":len(x),"selected_positive_rows":len(pos),"selected_negative_rows":len(neg),
           "trace":trace,"compute":"CPU float64; residual logistic scorer; original score offset",
           "detector_weights_updated":False,"GT_as_feature":False,"Q_path_at_score_time":False}
    require(np.isfinite(np.r_[model["weight"],model["bias"]]).all(),"Invalid learned state")
    return model


def grouped_curve(scores,flags):
    require(len(scores)==len(flags) and np.isfinite(scores).all() and np.isin(flags,[-1,0,1]).all(),"Invalid curve input")
    if not len(scores):return {"threshold":np.empty(0),"TP":np.empty(0,np.int64),"FP":np.empty(0,np.int64)}
    order=np.argsort(-scores,kind="stable");s=scores[order];f=flags[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    return {"threshold":s[ends],"TP":np.cumsum(f==1)[ends],"FP":np.cumsum(f==0)[ends]}


def at_fp_budget(curve,budget):
    good=np.flatnonzero(curve["FP"]<=budget)
    if not len(good):return {"TP":0,"FP":0,"threshold":None}
    k=good[-1]
    return {"TP":int(curve["TP"][k]),"FP":int(curve["FP"][k]),"threshold":float(curve["threshold"][k])}


def at_tp_target(curve,target):
    good=np.flatnonzero(curve["TP"]>=target)
    if not len(good):return {"achievable":False,"TP":None,"FP":None,"threshold":None}
    k=good[0]
    return {"achievable":True,"TP":int(curve["TP"][k]),"FP":int(curve["FP"][k]),"threshold":float(curve["threshold"][k])}


def official_rank_entries(gt_images,dt_images):
    scores=[];flags=[]
    for category in range(1,11):
        for gt,dt in zip(gt_images,dt_images):
            g=gt[gt[:,5]==category,:5]
            d=dt[:500];d=d[d[:,5]==category,:5]
            _,matched=P.AP.eval_res(g,d,.5)
            scores.extend(matched[:,4]);flags.extend(matched[:,5])
    return np.array(scores,np.float64),np.array(flags,np.int8)


def source_bindings():
    result=deepcopy(N.source_bindings())
    for path in (Path(__file__),ROOT/"tests/test_gate1_frozen_box_scorer.py"):
        result[path.name]=identity(path)
    return result


def prepare():
    require(not (OUT/"contract.json").exists(),"Scoring contract already frozen")
    previous=read(N.OUT/"contract.json")
    c=deepcopy(previous)
    teacher=Path(previous["teacher_array_root"])
    old_supervision=N.OUT/"attempt_01/training_supervision"
    baseline=Path(previous["reused_stage_root"])/"evaluation/mse_r"
    reused=[identity(p) for p in sorted(baseline.rglob("*")) if p.is_file()]
    reused += [identity(teacher/f"{i:04d}{suffix}") for i in range(512) for suffix in (".json","_raw.npz")]
    reused += [identity(old_supervision/f"{i:04d}.json") for i in range(512)]
    c.update(schema="rampq.gate1.frozen_box_scorer_development/v1",created_at_utc=now(),
        authorization='User: OK, try it; execute a limited comparator for the frozen detector, and record results truthfully',
        source_bindings=source_bindings(),previous_source_bindings=previous["source_bindings"],
        previous_contract=identity(N.OUT/"contract.json"),previous_delivery=identity(N.OUT/"delivery_validation.json"),
        previous_comparison=identity(N.OUT/"attempt_01/comparison_result.json"),reused_files=reused,
        arms=list(ARMS),features=list(FEATURES),teacher_array_root=str(teacher),
        intervention_supervision_root=str(old_supervision),baseline_root=str(baseline),
        scorer="residual logit(original_score)+linear22standardized_features+bias; 23parameters; zero initialization; original boxes/categories fixed",
        fit="100fullbatch CPUfloat64 Adam updates,lr0.03,betas0.9/0.999,eps1e-8; image-balanced common BCE +0.25 balanced selected BCE +0.0005norm2; no model/seed/feature/hyperparameter sweep",
        training_target="GT-aware PROJECT matching at confidence0.001,IoU0.5. TP soft target=matchedIoU,FP target0; ignored/nonfinite/empty excluded. Targets never input features.",
        selected_supervision="positive original601interventionGT when a TP is present in fixed R pool; negative original882interventionerrors only if still FP at0.001; hard controls exact positive class/size and negative class/confidence/error-type counts",
        candidate_pool="Existing MSE R: conf0.001,NMS0.7,max1000. Do not rerun NMS after scoring. Same boxes/categories/anchor IDs; only scores and submission order change. Keep entire original pool even when new scores<0.001.",
        evaluation="128previously-used training_val developer images only; official Python AP; fixed conf0.25 project counts; official IoU0.5 cap500 global score curves; evaluate all3scorers and original R",
        FP_budget_rule="B=#FP of original R at score>=0.25 in new official IoU0.5/cap500 curve. Freeze this formula now; evaluate whole tied score groups, maxTP with FP<=B. Also minFP attaining baselineTP. Curve thresholds are descriptive, not deployable thresholds learned on test labels.",
        primary_success="Candidate AP>original_R and original_R officialTP not decreased at fixed original_R officialFP budget. Extra intervention value requires improvement over matched hard and base scorers; no scientific acceptance from reused development set.",
        ranking_diag_scope="New one-IoU cached-matrix matching for FP-budget diagnosis; do not repeat original R full AP or old model inference.",
        new_detector_image_forwards=0,calibration_repeated=False,optimizer_steps_per_scorer=100,
        real_image_forward_budget=0,reconstruction_forwards_separate=0,new_teacher_model_forwards=0,
        optimizer_steps_per_arm=100,epochs=None,batch=None,accumulate=None,
        training="Frozen detector; only the22feature linear scorer and bias are optimized,100fullbatch CPUfloat64 updates",
        auxiliary="Common image-balanced soft-quality BCE; matched hard/intervention selected positive and negative BCE,weight0.25; score_base has no selected auxiliary",
        evaluation_order=list(ARMS),new_evaluation_order=list(ARMS),
        actual_INT8_execution=False,scientific_acceptance=False,Selection_payload_access=False,Holdout_payload_access=False,
        candidate_budget=3,timeout_seconds=2400,automatic_retry=False)
    write(OUT/"contract.json",c)
    print(json.dumps({"status":"FROZEN","contract":identity(OUT/"contract.json"),"new_candidates":3,"detector_forwards":0},ensure_ascii=True))


class Worker:
    def __init__(self):
        self.contract=read(OUT/"contract.json");self.attempt=OUT/"attempt_01";self.attempt.mkdir(exist_ok=False)
        self.started=time.monotonic()
        self.live={"pid":os.getpid(),"parent_pid":os.getppid(),"started_at_utc":now(),"status":"STARTING",
                   "contract":identity(OUT/"contract.json"),"new_detector_image_forwards":0}
        self.progress("STARTING",0,1)

    def progress(self,status,done,total,**extra):
        require(time.monotonic()-self.started<2400,"Finite scoring time budget exhausted")
        self.live.update(status=status,done=done,total=total,updated_at_utc=now(),**extra)
        write(self.attempt/"live_status.json",self.live,once=False)
        with (self.attempt/"events.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(self.live)+"\n")
        print(json.dumps({k:self.live[k] for k in ("status","done","total")}),flush=True)

    def verify(self):
        c=self.contract
        require(c["source_bindings"]==source_bindings(),"Frozen sources changed")
        require(c["framework"]==P.verify_framework_sources(),"Framework changed")
        for key in ("candidate","architecture","train_manifest","training_val_manifest","evaluation_protocol","previous_contract","previous_delivery","previous_comparison","training_annotation_assignment"):
            require(identity(c[key]["path"])==c[key],"Input binding changed: "+key)
        for b in c["reused_files"]:require(identity(b["path"])==b,"Reused saved member changed")

    def annotation(self,root,meta):
        path=(root/meta["annotation_path"]).resolve(strict=True)
        require(path.is_relative_to(root) and sha(path)==meta["annotation_sha256"],"Allowed annotation identity differs")
        return P.MATCH.parse_ground_truth(path.read_text(encoding="utf-8-sig"),meta["image_id"])

    def build_training(self,root):
        c=self.contract;teacher=Path(c["teacher_array_root"]);supervision=Path(c["intervention_supervision_root"])
        output=self.attempt/"training_cache";output.mkdir()
        matrices=[];targets=[];probabilities=[];image_indices=[];selection={arm:{"positive":[],"negative":[]} for arm in ARMS}
        rows_total=0;receipts=[]
        for i,meta in enumerate(c["training_annotation_metadata"]):
            old=read(teacher/f"{i:04d}.json");old2=read(supervision/f"{i:04d}.json")
            with np.load(teacher/f"{i:04d}_raw.npz",allow_pickle=False) as data:raw=data["r"].copy()
            pred=P.predictions(raw,old["training_to_original_params"],meta["image_id"],"fixed_R")
            # Feature extraction happens without calling or receiving any GT reader.
            feature=visible_features(raw,pred)
            gt=self.annotation(root,meta)
            matched=P.MATCH.match_project_targets(gt,pred,meta["width"],meta["height"],confidence=.001,iou_threshold=.5)
            by_match={m["detection_id"]:m for m in matched["matches"]}
            by_state={m["detection_id"]:m["reason"] for m in matched["detection_states"]}
            valid=[j for j,d in enumerate(pred) if by_state[d["detection_id"]] in {"true_positive","false_positive"}]
            require(valid,"No usable training candidates")
            local={j:k for k,j in enumerate(valid)}
            soft=np.array([by_match[d["detection_id"]]["iou"] if d["detection_id"] in by_match else 0. for d in pred],np.float64)
            changed={p["target_id"] for p in old["positives"]}
            positive_gt={g["target_id"]:g for g in gt if g["score"]==1 and 1<=g["category"]<=10}
            pos_pool=[];neg_pool=[]
            eligible={t["target_id"] for t in matched["targets"]}
            boxes=np.array([P.MATCH._xyxy(g["bbox_xywh"],meta["width"],meta["height"]) for g in gt if g["target_id"] in eligible]).reshape(-1,4)
            categories=np.array([g["category"] for g in gt if g["target_id"] in eligible])
            ratio=old["training_to_original_params"]["ratio"]
            negative_keys={(e["anchor"],e["category"]) for e in old2["native_intervention_correction"]["errors"]}
            for j in valid:
                d=pred[j];base={"row":local[j],"category":d["category"]-1,"score":d["score"],"anchor":d["raw_anchor_index"]}
                if soft[j]>0:
                    target_id=by_match[d["detection_id"]]["target_id"];g=positive_gt[target_id]
                    area=g["bbox_xywh"][2]*g["bbox_xywh"][3]*ratio[0]*ratio[1]
                    pos_pool.append({**base,"size_bin":0 if area<1024 else (1 if area<9216 else 2),"target_id":target_id})
                elif d["score"]>=.25:
                    box=np.array([P.MATCH._xyxy(d["bbox_xywh"],meta["width"],meta["height"])])
                    kind=N.error_kind(P.box_iou_matrix(box,boxes)[0],categories,d["category"])
                    neg_pool.append({**base,"score_bin":min(2,int((d["score"]-.25)/.25)),"kind":kind})
            selected_p=sorted((x for x in pos_pool if x["target_id"] in changed),key=lambda x:(x["score"],x["anchor"]))
            selected_n=sorted((x for x in neg_pool if (x["anchor"],x["category"]) in negative_keys),key=lambda x:(-x["score"],x["anchor"]))
            require(len(selected_p)<=8 and len(selected_n)<=8,"Selection cap exceeded")
            control_p=P.match_counts_to_controls(selected_p,pos_pool,lambda x:(x["category"],x["size_bin"]),lambda x:(x["score"],x["anchor"]))
            control_n=P.match_counts_to_controls(selected_n,neg_pool,lambda x:(x["category"],x["score_bin"],x["kind"]),lambda x:(-x["score"],x["anchor"]))
            for arm,ps,ns in (("score_intervention",selected_p,selected_n),("score_hard",control_p,control_n),("score_base",selected_p,selected_n)):
                selection[arm]["positive"].extend(rows_total+x["row"] for x in ps)
                selection[arm]["negative"].extend(rows_total+x["row"] for x in ns)
            x=feature[valid];y=soft[valid];p=np.array([pred[j]["score"] for j in valid],np.float64)
            np.savez_compressed(output/f"{i:04d}.npz",features=x,target=y,original_score=p,raw_anchor=np.array([pred[j]["raw_anchor_index"] for j in valid]),
                                category=np.array([pred[j]["category"] for j in valid]))
            write(output/f"{i:04d}.json",{"image_id":meta["image_id"],"rows":len(valid),"offset":rows_total,
                "hard_positive":control_p,"hard_negative":control_n,"intervention_positive":selected_p,"intervention_negative":selected_n,
                "input_raw":identity(teacher/f"{i:04d}_raw.npz"),"GT_used_only_for_labels_and_selection":True})
            matrices.append(x);targets.append(y);probabilities.append(p);image_indices.append(np.full(len(valid),i,np.int64));rows_total+=len(valid)
            receipts.append({"image_id":meta["image_id"],"rows":len(valid),"TP_rows":int((y>0).sum()),"FP_rows":int((y==0).sum())})
            if (i+1)%32==0:self.progress("TRAINING_FEATURES_FROM_SAVED_R_ONLY",i+1,512)
        write(output/"selection.json",selection)
        write(output/"summary.json",{"images":512,"rows":rows_total,"selected_positive":len(selection["score_intervention"]["positive"]),
            "selected_negative":len(selection["score_intervention"]["negative"]),"rows_by_image":receipts,"detector_forwards":0})
        return np.concatenate(matrices),np.concatenate(targets),np.concatenate(probabilities),np.concatenate(image_indices),selection

    def build_development(self,root):
        c=self.contract;folder=self.attempt/"development_cache";folder.mkdir()
        old=read(Path(c["baseline_root"])/"saved_predictions_and_targets.json")
        require([r["image_id"] for r in old]==[r["image_id"] for r in c["evaluation_members"]],"Development order differs")
        for i,(meta,row) in enumerate(zip(c["evaluation_members"],old)):
            name=meta["image_id"].split("/")[-1]+".npz"
            with np.load(Path(c["baseline_root"])/"raw"/name,allow_pickle=False) as data:raw=data["raw"].copy()
            x=visible_features(raw,row["predictions"])
            np.savez_compressed(folder/f"{i:04d}.npz",features=x,original_score=np.array([p["score"] for p in row["predictions"]],np.float64))
            write(folder/f"{i:04d}.json",{"image_id":meta["image_id"],"predictions":row["predictions"],"original_raw":identity(Path(c["baseline_root"])/"raw"/name),
                "features_have_no_GT_or_Q_inputs":True})
        write(folder/"summary.json",{"images":128,"detector_forwards":0,"uses_fixed_existing_NMS_pool":True})

    def evaluate(self,root,arm,model):
        c=self.contract;folder=self.attempt/"evaluation"/arm;folder.mkdir(parents=True)
        allgt=[];alldt=[];records=[];counts=Counter()
        for i,meta in enumerate(c["evaluation_members"]):
            cached=read(self.attempt/"development_cache"/f"{i:04d}.json")
            with np.load(self.attempt/"development_cache"/f"{i:04d}.npz",allow_pickle=False) as d:
                scores=score_rows(d["features"],d["original_score"],model)
            pred=deepcopy(cached["predictions"])
            for row,score in zip(pred,scores):row["score"]=float(score)
            pred=sorted(pred,key=lambda x:(-x["score"],x["nms_input_order"]))
            gt=self.annotation(root,meta)
            gt8=np.array([g["bbox_xywh"]+[g["score"],g["category"],g["truncation"],g["occlusion"]] for g in gt],np.float64).reshape(-1,8)
            dt8=np.array([d["bbox_xywh"]+[d["score"],d["category"],-1,-1] for d in pred],np.float64).reshape(-1,8)
            prepared=P.AP.prepare_raw_image(gt8,dt8,width=meta["width"],height=meta["height"])
            allgt.append(prepared["prepared_gt"]);alldt.append(prepared["filtered_predictions"])
            matching=P.MATCH.match_project_targets(gt,pred,meta["width"],meta["height"],confidence=.25,iou_threshold=.5)
            counts.update(matching["counts"])
            records.append({"image_id":meta["image_id"],"predictions":pred,"target_matching":matching})
            if (i+1)%32==0:self.progress("SCORE_FIXED_DEVELOPMENT_BOXES_"+arm,i+1,128)
        write(folder/"saved_predictions_and_targets.json",records)
        np.savez_compressed(folder/"prepared_matrices.npz",**{f"gt_{i}":r for i,r in enumerate(allgt)},**{f"dt_{i}":r for i,r in enumerate(alldt)})
        self.progress("NEW_SCORER_OFFICIAL_PYTHON_AP_"+arm,0,1)
        ap=P.AP.calc_accuracy(allgt,alldt,capture_intermediates=False)
        np.savez_compressed(folder/"AP_AR_arrays.npz",AP=ap["AP"],AR=ap["AR"],evalClass=ap["evalClass"])
        score,flag=official_rank_entries(allgt,alldt);curve=grouped_curve(score,flag)
        np.savez_compressed(folder/"official_IoU05_rank_curve.npz",score=score,flag=flag,**curve)
        metrics={"branch":arm,"images":128,"metrics_percent":ap["metrics"],"project_counts_conf025":dict(counts),
                 "fixed_detector_boxes_categories_and_NMS_pool":True,"scoring_GT_inputs":False,"new_detector_forwards":0,"scientific_acceptance":False}
        write(folder/"metrics.json",metrics)
        return metrics,curve

    def run(self):
        P.configure();self.verify();c=self.contract
        root=Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
        x,y,p,images,selection=self.build_training(root)
        self.build_development(root)
        models={}
        for arm in ARMS:
            self.progress("FIT_LINEAR_SCORER_"+arm,0,100)
            chosen=selection[arm];model=fit_scorer(x,y,p,images,chosen["positive"],chosen["negative"],arm)
            path=self.attempt/f"{arm}_model.json";write(path,model)
            require(np.array_equal(score_rows(x[:1000],p[:1000],model),score_rows(x[:1000],p[:1000],read(path))),"Saved scorer reconstruction differs")
            models[arm]=model;self.progress("FIT_LINEAR_SCORER_"+arm,100,100)
        del x,y,p,images
        with np.load(Path(c["baseline_root"])/"prepared_matrices.npz",allow_pickle=False) as data:
            basegt=[data[f"gt_{i}"].copy() for i in range(128)];basedt=[data[f"dt_{i}"].copy() for i in range(128)]
        score,flag=official_rank_entries(basegt,basedt)
        reference=grouped_curve(score,flag)
        np.savez_compressed(self.attempt/"reference_official_IoU05_rank_curve.npz",score=score,flag=flag,**reference)
        budget=int(((flag==0)&(score>=.25)).sum());target=int(((flag==1)&(score>=.25)).sum())
        write(self.attempt/"FP_budget_reference.json",{"FP_budget":budget,"TP_target":target,"confidence":.25,
            "metric":"new official-style IoU0.5 cap500 diagnostic; differs from PROJECT counts","source":identity(Path(c["baseline_root"])/"prepared_matrices.npz"),
            "formula_was_frozen_before_candidate_results":True,"curve_thresholds_are_descriptive_not_deployable":True})
        metrics=[read(Path(c["baseline_root"])/"metrics.json")];comparisons={"mse_r":{"at_FP_budget":at_fp_budget(reference,budget),"at_TP_target":at_tp_target(reference,target)}}
        for arm in ARMS:
            metric,curve=self.evaluate(root,arm,models[arm]);metrics.append(metric)
            comparisons[arm]={"at_FP_budget":at_fp_budget(curve,budget),"at_TP_target":at_tp_target(curve,target)}
        self.verify()
        summary={"status":"COMPLETE_FIXED_BOX_SCORER_DEVELOPMENT_PILOT","metrics":metrics,"FP_budget":budget,"TP_target":target,
            "ranking_comparisons":comparisons,"new_detector_image_forwards":0,"candidate_count":3,"optimizer_steps_each":100,
            "all_candidates_retained":True,"old_failures_preserved":True,"Holdout_accessed":False,"Selection_accessed":False,"scientific_acceptance":False}
        write(self.attempt/"comparison_result.json",summary)
        files=[identity(p) for p in sorted(self.attempt.rglob("*")) if p.is_file() and p.name not in {"live_status.json","events.jsonl","manifest.json","completion_receipt.json"}]
        write(self.attempt/"manifest.json",{"files":files})
        self.progress("COMPLETE_FIXED_BOX_SCORER_DEVELOPMENT_PILOT",1,1)
        write(self.attempt/"completion_receipt.json",{**self.live,"finished_at_utc":now(),"elapsed_seconds":time.monotonic()-self.started,
             "manifest":identity(self.attempt/"manifest.json"),"comparison":identity(self.attempt/"comparison_result.json")})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","run"));args=parser.parse_args()
    if args.mode=="prepare":prepare();return
    worker=Worker();awake=False
    try:
        if os.name=="nt":
            import ctypes
            awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        worker.run()
    except BaseException as e:
        worker.live.update(status="FAILED_ATTEMPT_PRESERVED",error_type=type(e).__name__,error=str(e),updated_at_utc=now())
        write(worker.attempt/"live_status.json",worker.live,once=False)
        write(worker.attempt/"failure_receipt.json",{**worker.live,"traceback":traceback.format_exc()});raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=="__main__":main()
