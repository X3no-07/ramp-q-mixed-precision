"""Finite developer-only correction pilot; preserve native TAL score targets.

Reuse the completed teacher arrays and baselines. No new calibration, no
Selection/Holdout payload, no new inference architecture, no parameter search.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/gate1/20261026_native_target_correction_pilot_v1"
os.environ["YOLO_CONFIG_DIR"]=str(OUT/"settings")
import numpy as np
import torch
from torch.nn import functional as F
import gate1_intervention_training_v2 as P
from gate1_ignore_negative_loss import build_negative_keep_mask

PREVIOUS=P.OUT
ARMS=("localization_only","native_hard_correction","native_intervention_correction")
KINDS=("same_class_overlap","other_class_overlap","near_object_low_IoU","low_overlap_remainder")
read,write,sha,identity,require,now=P.read,P.write,P.sha,P.identity,P.require,P.now


def overconfidence_divergence(logit,target):
    """BCE minus target entropy, only on the overconfident side.

    dL/dlogit = sigmoid(logit)-target >= 0 in active entries; zero otherwise.
    Targets are the SAME detached native TAL targets as the base classifier.
    """
    target=target.detach()
    require(bool(((target>=0)&(target<=1)).all()),"Target outside probability domain")
    entropy=-(torch.xlogy(target,target)+torch.xlogy(1-target,1-target))
    excess=(F.binary_cross_entropy_with_logits(logit,target,reduction="none")-entropy).clamp_min(0)
    return torch.where(logit.detach().sigmoid()>target,excess,torch.zeros_like(excess))


def aligned_iou(box,truth):
    inter=(torch.minimum(box[:,2:],truth[2:])-torch.maximum(box[:,:2],truth[:2])).clamp_min(0).prod(1)
    union=(box[:,2:]-box[:,:2]).clamp_min(0).prod(1)+(truth[2:]-truth[:2]).clamp_min(0).prod()-inter
    return inter/union.clamp_min(1e-7)


def error_kind(iou,classes,category):
    same=iou[classes==category]
    if len(same) and same.max()>=.5:return KINDS[0]
    if len(iou) and iou.max()>=.5:return KINDS[1]
    if len(iou) and iou.max()>=.1:return KINDS[2]
    return KINDS[3]


def mine_errors(old,q,r,gt,meta):
    params=old["training_to_original_params"]
    qp=P.predictions(q,params,meta["image_id"],"q")
    rp=P.predictions(r,params,meta["image_id"],"r")
    qm=P.MATCH.match_project_targets(gt,qp,meta["width"],meta["height"],confidence=.25,iou_threshold=.5)
    rm=P.MATCH.match_project_targets(gt,rp,meta["width"],meta["height"],confidence=.25,iou_threshold=.5)
    require(qm["counts"]==old["q_counts"] and rm["counts"]==old["r_counts"],"Saved teacher matching changed")
    qfp={s["detection_id"] for s in qm["detection_states"] if s["reason"]=="false_positive"}
    rfp={s["detection_id"] for s in rm["detection_states"] if s["reason"]=="false_positive"}
    eligible={t["target_id"] for t in rm["targets"]}
    positives=[g for g in gt if g["target_id"] in eligible]
    boxes=np.array([P.MATCH._xyxy(g["bbox_xywh"],meta["width"],meta["height"]) for g in positives]).reshape(-1,4)
    classes=np.array([g["category"] for g in positives])
    pool=[]
    for d in rp:
        if d["detection_id"] not in rfp:continue
        clipped=np.array([P.MATCH._xyxy(d["bbox_xywh"],meta["width"],meta["height"])])
        iou=P.box_iou_matrix(clipped,boxes)[0]
        prior=[x for x in qp if x["detection_id"] in qfp and x["category"]==d["category"]]
        already=bool(prior) and bool((P.box_iou_matrix(P.xyxy([d["bbox_xywh"]]),P.xyxy([x["bbox_xywh"] for x in prior]))>=.5).any())
        pool.append({"anchor":d["raw_anchor_index"],"category":d["category"]-1,"score":d["score"],
                     "score_bin":min(2,int((d["score"]-.25)/.25)),"kind":error_kind(iou,classes,d["category"]),
                     "new_after_restore":not already,"maximum_GT_IoU":float(iou.max()) if len(iou) else 0.})
    chosen=sorted((x for x in pool if x["new_after_restore"]),key=lambda x:(-x["score"],x["anchor"]))[:8]
    controls=P.match_counts_to_controls(chosen,pool,lambda x:(x["category"],x["score_bin"],x["kind"]),lambda x:(-x["score"],x["anchor"]))
    return {"image_id":meta["image_id"],"q_counts":qm["counts"],"r_counts":rm["counts"],
            "original_supervision":identity(PREVIOUS/"attempt_01/training_supervision"/f"{meta['_index']:04d}.json"),
            "localization_only":{"positive_gt":old["intervention"]["positive_gt"],"errors":[]},
            "native_hard_correction":{"positive_gt":old["hard_example"]["positive_gt"],"errors":controls},
            "native_intervention_correction":{"positive_gt":old["intervention"]["positive_gt"],"errors":chosen},
            "eligible_errors":len(pool),"error_targets_are_dynamic_native_TAL_not_background_labels":True,
            "ignore_policy":"official teacher prefilter; during training original native ignore mask and positive priority",
            "rescued_targets":old["rescued_targets"],"collateral_targets":old["collateral_targets"]}


class NativeCorrectiveLoss(P.IgnoreAwareDetectionLoss):
    def __init__(self,model):
        super().__init__(model)
        self.native_targets=None
        self._capture_handle=self.assigner.register_forward_hook(self.capture_targets)
        self.last_aux={}

    def capture_targets(self,module,args,result):
        self.native_targets=result[2].detach()

    def compute(self,preds,batch,records,arm):
        require(arm in (*ARMS,"ordinary_qat"),"Unregistered loss arm")
        self.native_targets=None
        assignment,base,_=super().get_assigned_targets_and_loss(preds,batch)
        fg,gt_index,_target_boxes,anchor_points,stride=assignment
        fg=fg.bool()
        scores=preds["scores"].permute(0,2,1).contiguous()
        require(self.native_targets is not None and self.native_targets.shape==scores.shape,"TAL targets were not captured")
        keep=build_negative_keep_mask((anchor_points*stride).detach(),fg,batch["ignore_boxes_xyxy"],tuple(batch["img"].shape[-2:]))
        box=self.bbox_decode(anchor_points,preds["boxes"].permute(0,2,1).contiguous())*stride
        zero=scores.sum()*0
        loc=[];errors=[];active_errors=0;ignored_errors=0;assigned_errors=0
        if arm!="ordinary_qat":
            for b,record in enumerate(records):
                spec=record[arm]
                indices=torch.where(batch["batch_idx"]==b)[0]
                for local_gt in spec["positive_gt"]:
                    selected=fg[b]&(gt_index[b]==local_gt)
                    if not selected.any():continue
                    t=batch["bboxes"][indices[local_gt]]*640
                    truth=torch.cat((t[:2]-t[2:]/2,t[:2]+t[2:]/2))
                    loc.append(self.hyp.box*(1-aligned_iou(box[b,selected],truth)).mean())
                for e in spec["errors"]:
                    anchor,cls=e["anchor"],e["category"]
                    if not bool(keep[b,anchor]):ignored_errors+=1;continue
                    logit=scores[b,anchor,cls];target=self.native_targets[b,anchor,cls]
                    errors.append(self.hyp.cls*overconfidence_divergence(logit,target))
                    active_errors+=int(logit.detach().sigmoid()>target)
                    assigned_errors+=int(fg[b,anchor])
        loc_loss=torch.stack(loc).mean() if loc else zero
        error_loss=torch.stack(errors).mean() if errors else zero
        aux=.25*(loc_loss+error_loss)
        self.last_aux={"positive_objects":len(loc),"selected_error_channels_kept":len(errors),"active_error_channels":active_errors,
                       "ignored_error_channels":ignored_errors,"error_anchors_currently_foreground":assigned_errors,
                       "localization_loss":float(loc_loss.detach()),"overconfidence_loss":float(error_loss.detach()),
                       "auxiliary_loss":float(aux.detach()),"extra_positive_classification_BCE":False}
        return base.sum()+aux


def source_bindings():
    result=deepcopy(P.source_bindings())
    for p in (Path(__file__),ROOT/"tests/test_gate1_native_target_correction.py"):
        result[p.name]=identity(p)
    return result


def prepare():
    require(not (OUT/"contract.json").exists(),"Contract already frozen")
    prior=read(PREVIOUS/"contract.json")
    require(read(PREVIOUS/"delivery_validation.json")["status"]=="PASS_COMPLETE_FINITE_PILOT_DELIVERY_INTEGRITY_ONLY","Previous delivery incomplete")
    c=deepcopy(prior)
    teacher=PREVIOUS/"attempt_01/training_supervision"
    reused=list(prior["reused_files"])
    reused += [identity(teacher/f"{i:04d}{suffix}") for i in range(512) for suffix in (".json","_raw.npz")]
    for label in prior["arms"]:
        reused += [identity(p) for p in sorted((PREVIOUS/"attempt_01/evaluation"/label).rglob("*")) if p.is_file()]
        reused.append(identity(PREVIOUS/"attempt_01/training"/label/"training_receipt.json"))
    c.update(schema="rampq.gate1.native_target_correction_pilot/v1",created_at_utc=now(),
             authorization='User 2026-10-26: Proceed; conduct a limited development comparison for native target alignment error correction based on the failure evidence from the previous round',
             source_bindings=source_bindings(),previous_source_bindings=prior["source_bindings"],
             previous_contract=identity(PREVIOUS/"contract.json"),previous_delivery=identity(PREVIOUS/"delivery_validation.json"),
             previous_comparison=identity(PREVIOUS/"attempt_01/comparison_result.json"),
             teacher_array_root=str(teacher.resolve()),reused_files=reused,
             reused_training_raw=[],arms=list(ARMS),evaluation_order=list(ARMS),
             new_evaluation_order=list(ARMS),reused_metric_order=read(PREVIOUS/"attempt_01/comparison_result.json")["metrics"],
             real_image_forward_budget=3*3*512+3*128,
             new_teacher_model_forwards=0,reconstruction_forwards_separate=9,
             calibration_and_baseline_inference_repeated=False,
             auxiliary="0.25*(mean selected positive GT IoU loss + mean one-sided native-TAL-target classification divergence on selected error channels). No extra positive classification BCE. Original ignore mask/positive priority preserved.",
             error_mining="all four FP geometry types from saved Q/R outputs, confidence0.25,IoU0.5, max8 new R errors/image; hard controls exact category/confidence-bin/error-type counts; same previous601GT positive budgets",
             error_types=list(KINDS),
             localization_only_ablation="same intervention-positive601GT, only IoU auxiliary; no extra error-classification term",
             hypothesis="Test whether native-target alignment and richer error supervision help; separately test if intervention selection adds value over matched hard selection. No guaranteed improvement or novelty.",
             related_work=[{"name":"GFL","url":"https://arxiv.org/abs/2006.04388","scope":"joint localization quality and classification is prior art; no exact GFL reproduction claim"},
                           {"name":"HQOD","url":"https://arxiv.org/abs/2408.02561","scope":"classification/localization harmony in quantization is prior art; no exact HQOD reproduction claim"}])
    write(OUT/"contract.json",c)
    print(json.dumps({"status":"FROZEN","contract":identity(OUT/"contract.json"),"new_forwards":c["real_image_forward_budget"]},ensure_ascii=True))


class Worker(P.Worker):
    def __init__(self):
        self.contract=read(OUT/"contract.json");self.attempt=OUT/"attempt_01"
        self.attempt.mkdir(exist_ok=False);self.started=time.monotonic()
        self.live={"schema":self.contract["schema"],"pid":os.getpid(),"parent_pid":os.getppid(),"started_at_utc":now(),
                   "contract":identity(OUT/"contract.json"),"forward_images":0,"status":"STARTING"}
        self.progress("STARTING",0,1)

    def verify(self):
        c=self.contract
        require(c["source_bindings"]==source_bindings(),"Frozen source changed")
        require(c["framework"]==P.verify_framework_sources(),"Framework changed")
        for key in ("candidate","architecture","formal_contract","train_manifest","training_val_manifest","evaluation_protocol","calibration_contract","old_ranges","previous_contract","previous_delivery","previous_comparison","training_annotation_assignment"):
            require(identity(c[key]["path"])==c[key],"Bound input changed: "+key)
        for b in c["reused_files"]+c["reference_bindings"]:
            require(identity(b["path"])==b,"Reused source/output changed")

    def train(self,model,dataset,records,arm,maxima):
        P.seed_all()
        folder=self.attempt/"training"/arm;folder.mkdir(parents=True,exist_ok=False)
        initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        initial_sha=hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        model.train();P.freeze_bn(model)
        criterion=NativeCorrectiveLoss(model)
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1e-4,momentum=.9,weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        generator=np.random.Generator(np.random.PCG64(P.SEED))
        steps=0;batches=0;trace=[];orders=[]
        for epoch in range(3):
            order=generator.permutation(len(dataset)).tolist();orders.append(order)
            for position in range(0,len(order),4):
                indices=order[position:position+4]
                batch=P.gpu_batch([dataset[j] for j in indices])
                self.live["forward_images"]+=len(indices)
                require(self.live["forward_images"]<=self.contract["real_image_forward_budget"],"Forward budget exceeded")
                pred=model(batch["img"])
                loss=criterion.compute(pred,batch,[records[j] for j in indices],arm)
                require(bool(torch.isfinite(loss)),"Nonfinite loss; preserve attempt")
                (loss/4).backward();batches+=1
                with (folder/"auxiliary_trace.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps({"epoch":epoch+1,"batch":batches,"indices":indices,**criterion.last_aux})+"\n")
                if batches%4==0:
                    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
                    trace.append({"epoch":epoch+1,"batch":batches,"step":steps,"loss":float(loss.detach()),"grad_norm_before_clip":float(norm),**criterion.last_aux})
                if batches%8==0:self.progress("TRAINING_"+arm,batches,384,epoch=epoch+1,optimizer_steps=steps,latest_loss=float(loss.detach()))
                del batch,pred,loss
            state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            require(all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()),"Nonfinite state")
            torch.save({"state_dict":state,"epoch":epoch+1,"arm":arm,"contract_sha256":self.live["contract"]["sha256"]},folder/f"epoch_{epoch+1:02d}_fp32.pt")
        require(steps==96,"Optimizer count incomplete")
        changed=sum(int((model.state_dict()[n].cpu()!=v).sum()) for n,v in initial.items() if n.endswith("weight"))
        frozen=[n for n in initial if "running_" in n or "num_batches_tracked" in n or n.endswith("_scale")]
        require(changed>0 and all(torch.equal(initial[n],model.state_dict()[n].cpu()) for n in frozen),"Update/frozen-state mismatch")
        model.eval();x=P.gpu_batch([dataset[0]])["img"]
        with torch.inference_mode():reference=model(x)[0].detach().cpu()
        restored=torch.load(folder/"epoch_03_fp32.pt",map_location="cpu",weights_only=True)
        independent=P.new_model(maxima);independent.load_state_dict(restored["state_dict"],strict=True)
        with torch.inference_mode():replay=independent(x)[0].detach().cpu()
        require(torch.equal(reference,replay),"Independent CUDA rebuild differs")
        independent.cpu()
        with torch.inference_mode():cpu=independent(x.cpu())[0].detach().cpu()
        require(torch.isfinite(cpu).all(),"CPU rebuild nonfinite")
        write(folder/"epoch_orders.json",orders)
        write(folder/"training_receipt.json",{"arm":arm,"epochs":3,"batches":384,"optimizer_steps":steps,
            "initial_state_value_sha256":initial_sha,"trace":trace,"changed_weight_elements":changed,"frozen_BN_and_scales_unchanged":True,
            "final_candidate":identity(folder/"epoch_03_fp32.pt"),"CUDA_reconstruction_exact":True,
            "CPU_CUDA_output_max_abs":float((reference-cpu).abs().max()),"CPU_CUDA_equivalence_certified":False,
            "reconstruction_real_image_forwards_outside_training_budget":3,"epoch_orders":identity(folder/"epoch_orders.json")})
        del independent,optimizer,criterion,initial;torch.cuda.empty_cache()
        return model

    def run(self):
        P.configure();self.verify();require(torch.cuda.is_available(),"CUDA required")
        write(self.attempt/"runtime.json",{"python":P.sys.version,"executable":P.sys.executable,"torch":str(torch.__version__),
              "cuda":torch.version.cuda,"GPU":torch.cuda.get_device_name(0),"backend":P.FT.backend_settings()})
        c=self.contract
        root=Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
        teacher=Path(c["teacher_array_root"])
        records=[];mining=self.attempt/"training_supervision";mining.mkdir()
        for i,meta0 in enumerate(c["training_annotation_metadata"]):
            meta={**meta0,"_index":i};ann=(root/meta["annotation_path"]).resolve(strict=True)
            require(ann.is_relative_to(root) and sha(ann)==meta["annotation_sha256"],"Original annotation changed")
            gt=P.MATCH.parse_ground_truth(ann.read_text(encoding="utf-8-sig"),meta["image_id"])
            with np.load(teacher/f"{i:04d}_raw.npz",allow_pickle=False) as data:q,r=data["q"].copy(),data["r"].copy()
            record=mine_errors(read(teacher/f"{i:04d}.json"),q,r,gt,meta);records.append(record)
            write(mining/f"{i:04d}.json",record)
            if (i+1)%32==0:self.progress("DERIVE_NEW_ERROR_SUPERVISION_FROM_SAVED_ARRAYS",i+1,512)
        summary={"images":512,"new_model_forwards":0,"positive_counts":sum(len(r["native_intervention_correction"]["positive_gt"]) for r in records),
                 "error_counts":sum(len(r["native_intervention_correction"]["errors"]) for r in records),
                 "error_types":dict(Counter(e["kind"] for r in records for e in r["native_intervention_correction"]["errors"])),
                 "matched_category_confidence_type_counts":True}
        require(summary["positive_counts"]==601 and summary["error_counts"]>0,"Supervision budget unexpected")
        write(mining/"summary.json",summary)
        dataset=P.SelectedTrainingDataset(c["train_manifest"],c["training_members"])
        maxima=read(Path(c["reused_stage_root"])/"mse_ranges.json")["max_abs"]
        metrics=deepcopy(c["reused_metric_order"])
        for arm in ARMS:
            model=P.new_model(maxima);model=self.train(model,dataset,records,arm,maxima)
            metrics.append(self.evaluate(model,arm));del model;torch.cuda.empty_cache()
        receipts=[read(self.attempt/"training"/arm/"training_receipt.json") for arm in ARMS]
        old_initial=read(PREVIOUS/"attempt_01/training/ordinary_qat/training_receipt.json")["initial_state_value_sha256"]
        require({r["initial_state_value_sha256"] for r in receipts}=={old_initial},"Initial model differs from ordinary QAT comparator")
        require(all(read(self.attempt/"training"/arm/"epoch_orders.json")==read(self.attempt/"training"/ARMS[0]/"epoch_orders.json") for arm in ARMS),"Training orders differ")
        self.verify();require(self.live["forward_images"]==4992,"Forward budget incomplete")
        write(self.attempt/"comparison_result.json",{"status":"COMPLETE_SINGLE_SEED_DEVELOPMENT_PILOT_NOT_SCIENTIFIC_ACCEPTANCE",
            "metrics":metrics,"arms":list(ARMS),"training_arms_same_initialization":True,"old_failures_preserved":True,
            "Holdout_accessed":False,"Selection_accessed":False,"real_INT8":False})
        files=[identity(p) for p in sorted(self.attempt.rglob("*")) if p.is_file() and p.name not in {"events.jsonl","live_status.json","manifest.json","completion_receipt.json"}]
        write(self.attempt/"manifest.json",{"files":files})
        self.progress("COMPLETE_SINGLE_SEED_DEVELOPMENT_PILOT",1,1)
        write(self.attempt/"completion_receipt.json",{**self.live,"finished_at_utc":now(),"elapsed_seconds":time.monotonic()-self.started,
            "comparison":identity(self.attempt/"comparison_result.json"),"manifest":identity(self.attempt/"manifest.json")})


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
        write(worker.attempt/"failure_receipt.json",{**worker.live,"traceback":traceback.format_exc()})
        raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=="__main__":main()
