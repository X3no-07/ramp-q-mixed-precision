"""Independent saved-state, budget, input and metric audit. No model inference."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import gate1_native_target_correction as N

P=N.P
OUT=N.OUT
A=OUT/"attempt_01"
read,write,identity,require=P.read,P.write,P.identity,P.require


def independent_kind(ious,categories,category):
    same=[float(v) for v,c in zip(ious,categories) if c==category]
    maximum=float(max(ious)) if len(ious) else 0.
    if same and max(same)>=.5:return "same_class_overlap"
    if maximum>=.5:return "other_class_overlap"
    if maximum>=.1:return "near_object_low_IoU"
    return "low_overlap_remainder"


def metric_folder(c,label):
    if label=="fp32_reference":return Path(c["reused_stage_root"])/"fp32_reference"
    if label in N.ARMS:return A/"evaluation"/label
    if label in ("ordinary_qat","hard_example","intervention"):
        return N.PREVIOUS/"attempt_01/evaluation"/label
    return Path(c["reused_stage_root"])/"evaluation"/label


def main():
    torch.set_num_threads(1)
    c=read(OUT/"contract.json")
    completion=read(A/"completion_receipt.json")
    require(completion["status"]=="COMPLETE_SINGLE_SEED_DEVELOPMENT_PILOT" and completion["forward_images"]==4992,"Completion/budget incomplete")
    require(completion["contract"]==identity(OUT/"contract.json"),"Contract identity differs")
    for b in c["source_bindings"].values():require(identity(b["path"])==b,"Source changed")
    for b in c["previous_source_bindings"].values():require(identity(b["path"])==b,"Previous source changed")
    for key in ("candidate","architecture","formal_contract","train_manifest","training_val_manifest","evaluation_protocol","calibration_contract","old_ranges","previous_contract","previous_delivery","previous_comparison","training_annotation_assignment"):
        require(identity(c[key]["path"])==c[key],"Input identity differs: "+key)
    for b in c["reused_files"]+c["reference_bindings"]:
        require(identity(b["path"])==b,"Reused artifact differs")
    audit_contract=read(OUT/"saved_audit_contract.json")
    require(identity(audit_contract["script"]["path"])==audit_contract["script"],"Audit source changed")
    tree=ET.parse(OUT/"preflight_tests.xml").getroot()
    tests=sum(int(s.get("tests",0)) for s in tree.iter("testsuite"))
    issues=sum(int(s.get(k,0)) for s in tree.iter("testsuite") for k in ("errors","failures","skipped"))
    require(tests==10 and issues==0,"Qualification incomplete")
    header=read(c["train_manifest"]["path"])
    train_root=(Path(c["train_manifest"]["path"]).parent/header["data_root"]).resolve(strict=True)
    checked_inputs=0
    for row in c["training_members"]:
        for kind in ("image","labels","sidecar"):
            b=row[kind];path=(train_root/b["path"]).resolve(strict=True)
            require(path.is_relative_to(train_root),"Path escaped")
            actual=identity(path)
            require(actual["bytes"]==b["bytes"] and actual["sha256"]==b["sha256"],"Train input changed")
            checked_inputs+=1
    eval_root=Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
    for rows,kinds in ((c["training_annotation_metadata"],("annotation",)),(c["evaluation_members"],("image","annotation")),(c["calibration_members"],("image",))):
        for row in rows:
            for kind in kinds:
                path=(eval_root/row[kind+"_path"]).resolve(strict=True)
                require(path.is_relative_to(eval_root) and identity(path)["sha256"]==row[kind+"_sha256"],"Original allowed input differs")
                checked_inputs+=1
    manifest=read(A/"manifest.json")["files"]
    for b in manifest:require(identity(b["path"])==b,"Saved output changed")
    expected={str(p.resolve()) for p in A.rglob("*") if p.is_file() and p.name not in {"events.jsonl","live_status.json","completion_receipt.json","manifest.json"}}
    require(expected=={b["path"] for b in manifest} and len(expected)==len(manifest),"Manifest coverage differs")
    teacher=Path(c["teacher_array_root"])
    positive=errors=common_pos=common_errors=0;types=Counter();records=[]
    for i,meta in enumerate(c["training_annotation_metadata"]):
        r=read(A/"training_supervision"/f"{i:04d}.json");records.append(r)
        old=read(teacher/f"{i:04d}.json")
        require(r["image_id"]==meta["image_id"]==c["training_members"][i]["image_id"],"Supervision order differs")
        require(r["original_supervision"]==identity(teacher/f"{i:04d}.json"),"Original supervision binding differs")
        t,h,loc=[r[k] for k in ("native_intervention_correction","native_hard_correction","localization_only")]
        require(t["positive_gt"]==loc["positive_gt"]==old["intervention"]["positive_gt"],"Intervention GT selection changed")
        require(h["positive_gt"]==old["hard_example"]["positive_gt"] and not loc["errors"],"Hard GT/ablation changed")
        require(len(t["positive_gt"])==len(h["positive_gt"])<=8,"Positive budget differs")
        require(len(t["errors"])==len(h["errors"])<=8,"Error budget differs")
        def stratum(e):return (e["category"],e["score_bin"],e["kind"])
        require(Counter(map(stratum,t["errors"]))==Counter(map(stratum,h["errors"])),"Error strata budget differs")
        require(all(e["new_after_restore"] for e in t["errors"]),"Treatment includes old error")
        positive+=len(t["positive_gt"]);errors+=len(t["errors"])
        common_pos+=len(set(t["positive_gt"])&set(h["positive_gt"]))
        common_errors+=len({(e["anchor"],e["category"]) for e in t["errors"]}&{(e["anchor"],e["category"]) for e in h["errors"]})
        types.update(e["kind"] for e in t["errors"])
        gt=P.MATCH.parse_ground_truth((eval_root/meta["annotation_path"]).read_text(encoding="utf-8-sig"),meta["image_id"])
        positive_gt=[g for g in gt if g["score"]==1 and 1<=g["category"]<=10]
        ratio=old["training_to_original_params"]["ratio"]
        def pos_stratum(k):
            g=positive_gt[k];area=g["bbox_xywh"][2]*g["bbox_xywh"][3]*ratio[0]*ratio[1]
            return g["category"],0 if area<1024 else (1 if area<9216 else 2)
        require(Counter(map(pos_stratum,t["positive_gt"]))==Counter(map(pos_stratum,h["positive_gt"])),"Positive class/size strata differ")
        # Original positive GTs can be filtered by ignores, so use stored teacher targets only
        # to get their membership, never rematch scores or reconstruct model outputs.
        filtered=P.MATCH.filter_ignored_regions(gt,[],meta["width"],meta["height"])
        eligible=[g for g in filtered["ground_truth"] if g["score"]==1 and 1<=g["category"]<=10 and
                  P.MATCH._xyxy(g["bbox_xywh"],meta["width"],meta["height"])[2]>P.MATCH._xyxy(g["bbox_xywh"],meta["width"],meta["height"])[0] and
                  P.MATCH._xyxy(g["bbox_xywh"],meta["width"],meta["height"])[3]>P.MATCH._xyxy(g["bbox_xywh"],meta["width"],meta["height"])[1]]
        boxes=np.asarray([P.MATCH._xyxy(g["bbox_xywh"],meta["width"],meta["height"]) for g in eligible]).reshape(-1,4)
        categories=np.array([g["category"] for g in eligible])
        with np.load(teacher/f"{i:04d}_raw.npz",allow_pickle=False) as data:
            raw=data["r"]
            for arm in (t,h):
                require(len({(e["anchor"],e["category"]) for e in arm["errors"]})==len(arm["errors"]),"Duplicate error channel")
                for e in arm["errors"]:
                    anchor,cls=e["anchor"],e["category"]
                    require(float(raw[0,4+cls,anchor])==e["score"] and min(2,int((e["score"]-.25)/.25))==e["score_bin"],"Error score/bin differs from raw")
                    b=raw[0,:4,anchor].astype(np.float64)
                    # Teacher NMS used FP32 XYWH->XYXY; preserve that arithmetic before inverse transform.
                    xy=raw[0,:4,anchor].copy();xy[:2]-=xy[2:]/2;xy[2:]+=xy[:2]
                    xy=xy.astype(np.float64)
                    params=old["training_to_original_params"]
                    xy[[0,2]]=(xy[[0,2]]-params["left"])/ratio[0]
                    xy[[1,3]]=(xy[[1,3]]-params["top"])/ratio[1]
                    xy[[0,2]]=np.clip(xy[[0,2]],0,meta["width"])
                    xy[[1,3]]=np.clip(xy[[1,3]],0,meta["height"])
                    inter=np.maximum(0,np.minimum(boxes[:,2:],xy[2:])-np.maximum(boxes[:,:2],xy[:2])).prod(1)
                    denom=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)+np.maximum(0,xy[2:]-xy[:2]).prod()-inter
                    ious=inter/np.maximum(1e-12,denom)
                    require(independent_kind(ious,categories,cls+1)==e["kind"],"Independent error geometry partition differs")
    require(positive==601 and errors>0,"Supervision coverage incomplete")
    train_receipts={arm:read(A/"training"/arm/"training_receipt.json") for arm in N.ARMS}
    ordinary=read(N.PREVIOUS/"attempt_01/training/ordinary_qat/training_receipt.json")
    checkpoints=[];aux_totals={};orders=[]
    for arm,r in train_receipts.items():
        require(r["epochs"]==3 and r["batches"]==384 and r["optimizer_steps"]==96,"Training budget differs")
        require(r["initial_state_value_sha256"]==ordinary["initial_state_value_sha256"],"Initialization differs from ordinary QAT")
        require(r["CUDA_reconstruction_exact"] and r["frozen_BN_and_scales_unchanged"] and r["changed_weight_elements"]>0,"Training/save receipt incomplete")
        require([t["step"] for t in r["trace"]]==list(range(1,97)),"Optimizer steps missing")
        order=read(A/"training"/arm/"epoch_orders.json");orders.append(order)
        rng=np.random.Generator(np.random.PCG64(c["training_seed"]))
        require(order==[rng.permutation(512).tolist() for _ in range(3)],"Registered epoch order differs")
        log=[json.loads(line) for line in (A/"training"/arm/"auxiliary_trace.jsonl").read_text(encoding="utf-8").splitlines()]
        require(len(log)==384 and [x["batch"] for x in log]==list(range(1,385)),"Auxiliary trace incomplete")
        counts=Counter()
        for j,row in enumerate(log):
            epoch,offset=divmod(j,128);indices=order[epoch][offset*4:offset*4+4]
            require(row["indices"]==indices,"Batch order differs")
            selected_pos=sum(len(records[k][arm]["positive_gt"]) for k in indices)
            selected_err=sum(len(records[k][arm]["errors"]) for k in indices)
            require(row["positive_objects"]<=selected_pos,"Positive assignment exceeds selected budget")
            require(row["selected_error_channels_kept"]+row["ignored_error_channels"]==selected_err,"Error use budget differs")
            require(row["active_error_channels"]<=row["selected_error_channels_kept"],"Active channel count differs")
            require(not row["extra_positive_classification_BCE"],"Old positive BCE was used")
            require(all(np.isfinite(row[k]) and row[k]>=0 for k in ("localization_loss","overconfidence_loss","auxiliary_loss")),"Invalid auxiliary loss")
            counts.update({k:row[k] for k in ("positive_objects","selected_error_channels_kept","active_error_channels","ignored_error_channels","error_anchors_currently_foreground")})
        aux_totals[arm]=dict(counts)
        for epoch in range(1,4):
            path=A/"training"/arm/f"epoch_{epoch:02d}_fp32.pt"
            data=torch.load(path,map_location="cpu",weights_only=True)
            require(data["epoch"]==epoch and data["arm"]==arm and data["contract_sha256"]==identity(OUT/"contract.json")["sha256"],"Checkpoint identity differs")
            state=data["state_dict"]
            require(all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values() if v.is_floating_point()),"Checkpoint nonFP32/nonfinite")
            require(sum(k.endswith("activation_scale") for k in state)==88 and sum(k.endswith("weight_scale") for k in state)==88,"Quantizer state missing")
            checkpoints.append(identity(path))
    require(orders[0]==orders[1]==orders[2],"Arm orders differ")
    results=read(A/"comparison_result.json")
    metrics=[];guards={}
    for metric in results["metrics"]:
        label=metric["branch"];folder=metric_folder(c,label)
        require(read(folder/"metrics.json")==metric,"Reported metrics differ")
        with np.load(folder/"AP_AR_arrays.npz",allow_pickle=False) as d:
            ap,ar,classes=d["AP"],d["AR"],d["evalClass"]
            require(ap.shape==(10,10) and ar.shape==(10,10,4) and len(classes)>0,"Metric array domain differs")
            idx=classes-1
            computed=[np.mean(ap[idx]),np.mean(ap[idx,0]),np.mean(ap[idx,5]),*[np.mean(ar[idx,:,k]) for k in range(4)]]
        keys=("AP_all","AP_50","AP_75","AR_1","AR_10","AR_100","AR_500")
        require(all(np.isclose(v,metric["metrics_percent"][k],atol=1e-10,rtol=1e-12) for k,v in zip(keys,computed)),"Seven metric aggregates differ")
        if label in N.ARMS:
            predictions=read(folder/"saved_predictions_and_targets.json")
            require([r["image_id"] for r in predictions]==[m["image_id"] for m in c["evaluation_members"]],"Development membership differs")
            count=Counter()
            for r in predictions:count.update(r["target_matching"]["counts"])
            require(dict(count)==metric["project_counts_conf025"],"Saved target counts differ")
            saved={g["image_id"]:g["guard"] for g in read(folder/"all_anchor_guard.json")}
            require(len(saved)==128,"Guard coverage differs")
            passed=0
            for row in c["evaluation_members"]:
                name=row["image_id"].split("/")[-1]+".npz"
                with np.load(folder/"raw"/name,allow_pickle=False) as d,np.load(Path(c["reused_stage_root"])/"fp32_reference"/name,allow_pickle=False) as ref:
                    require(d["raw"].shape==(1,14,8400) and d["raw"].dtype==np.float32 and np.isfinite(d["raw"]).all(),"Raw domain differs")
                    guard=P.EVAL.raw_guard(ref["raw"],d["raw"])
                    require(guard==saved[row["image_id"]],"All-anchor guard differs")
                    passed+=int(guard["allclose_pass"])
            guards[label]={"images":128,"passed":passed,"failed":128-passed,"full8400anchors_recomputed":True}
        metrics.append(metric)
    result={"status":"PASS_SAVED_SOURCE_STATE_BUDGET_AND_METRIC_AGGREGATE_AUDIT_ONLY","at_utc":datetime.now(timezone.utc).isoformat(),
        "contract":identity(OUT/"contract.json"),"completion":identity(A/"completion_receipt.json"),
        "manifest_members_verified":len(manifest),"reused_bindings_verified":len(c["reused_files"])+len(c["reference_bindings"]),
        "authorized_input_members_rehashed":checked_inputs,"preflight_tests":10,"training_images":512,"development_images":128,
        "real_image_forwards":4992,"reconstruction_forwards_separate":9,"new_teacher_model_forwards":0,
        "positive_GT_selected":positive,"error_channels_selected":errors,"error_types":dict(types),
        "hard_intervention_positive_overlap":common_pos,"hard_intervention_error_overlap":common_errors,
        "auxiliary_usage_totals":aux_totals,"checkpoints":checkpoints,"all_arms_ordinary_QAT_initialization_equal":True,
        "epochs_each":3,"updates_each":96,"epoch_orders_equal":True,"new_arms_guard":guards,"metrics":metrics,
        "model_inference_rerun":False,"full_AP_scoring_rerun":False,"calibration_rerun":False,
        "independent_error_geometry_recomputed":True,"teacher_target_matching_repeated_in_audit":False,
        "scientific_acceptance":False,"actual_INT8":False,"Selection_payload_access":False,"Holdout_payload_access":False}
    write(OUT/"saved_delivery_audit.json",result)
    print(json.dumps({k:result[k] for k in ("status","manifest_members_verified","positive_GT_selected","error_channels_selected","error_types")},ensure_ascii=True))


if __name__=="__main__":main()
