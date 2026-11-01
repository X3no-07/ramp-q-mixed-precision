"""Read-only saved-artifact audit for the new finite corrective-QAT pilot.

No image inference, training, calibration, AP matching or sealed-data access.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import qualify_gate1_float_baseline as EVAL

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/gate1/20261026_intervention_training_pilot_v2"
ATTEMPT=OUT/"attempt_01"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(path):
    p=Path(path).resolve(strict=True)
    with p.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":digest}


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    torch.set_num_threads(1)
    output=OUT/"saved_delivery_audit.json"
    require(not output.exists(),"Audit output already exists")
    c=read(OUT/"contract.json")
    complete=read(ATTEMPT/"completion_receipt.json")
    require(complete["status"]=="COMPLETE_SINGLE_SEED_DEVELOPMENT_PILOT","Worker not complete")
    require(complete["contract"]==identity(OUT/"contract.json"),"Contract identity differs")
    require(complete["forward_images"]==c["real_image_forward_budget"]==6014,"Forward count differs")
    prior=Path(c["reused_stage_root"])
    for key in ("previous_contract","previous_failure","training_annotation_assignment"):
        require(identity(c[key]["path"])==c[key],"Prior input identity differs")
    reused=list(c["previous_source_bindings"].values())+c["reused_files"]+[r["raw"] for r in c["reused_training_raw"]]
    for item in reused:require(identity(item["path"])==item,"Reused completed-stage file changed")
    old_failure=read(c["previous_failure"]["path"])
    require(old_failure["forward_images"]==708 and old_failure["status"]=="FAILED_ATTEMPT_PRESERVED","Prior failure not preserved")
    for binding in c["source_bindings"].values():require(identity(binding["path"])==binding,"Frozen source changed")
    for binding in c["reference_bindings"]:require(identity(binding["path"])==binding,"Historical reference changed")
    for key in ("candidate","architecture","formal_contract","train_manifest","training_val_manifest","evaluation_protocol","calibration_contract","old_ranges"):
        require(identity(c[key]["path"])==c[key],f"Frozen input changed: {key}")
    # Hash only explicitly authorized train/development/calibration members, never scan a data directory.
    header=read(c["train_manifest"]["path"])
    train_root=(Path(c["train_manifest"]["path"]).parent/header["data_root"]).resolve(strict=True)
    input_files_checked=0
    for row in c["training_members"]:
        for kind in ("image","labels","sidecar"):
            member=row[kind];p=(train_root/member["path"]).resolve(strict=True)
            require(p.is_relative_to(train_root),"Train member path escaped")
            actual=identity(p)
            require(actual["bytes"]==member["bytes"] and actual["sha256"]==member["sha256"],"Training member changed after execution")
            input_files_checked+=1
    eval_root=Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
    for row in c["training_annotation_metadata"]:
        p=(eval_root/row["annotation_path"]).resolve(strict=True)
        require(row["assigned_split"]=="train" and p.is_relative_to(eval_root),"Training annotation role/path mismatch")
        require(identity(p)["sha256"]==row["annotation_sha256"],"Original training annotation changed")
        input_files_checked+=1
    for row in c["evaluation_members"]+c["calibration_members"]:
        kinds=("image","annotation") if row["assigned_split"]=="training_val" else ("image",)
        for kind in kinds:
            p=(eval_root/row[kind+"_path"]).resolve(strict=True)
            require(p.is_relative_to(eval_root) and identity(p)["sha256"]==row[kind+"_sha256"],"Development/calibration input changed")
            input_files_checked+=1
    files=read(ATTEMPT/"manifest.json")["files"]
    for item in files:require(identity(item["path"])==item,"Saved artifact digest mismatch")
    require(len({i["path"] for i in files})==len(files),"Duplicate manifest member")
    expected={str(p.resolve()) for p in ATTEMPT.rglob("*") if p.is_file() and p.name not in {"live_status.json","events.jsonl","completion_receipt.json","manifest.json"}}
    require(expected=={i["path"] for i in files},"Manifest coverage mismatch")
    suites=ET.parse(OUT/"preflight_tests.xml").getroot()
    tests=sum(int(s.attrib.get("tests",0)) for s in suites.iter("testsuite"))
    failures=sum(int(s.attrib.get("failures",0))+int(s.attrib.get("errors",0))+int(s.attrib.get("skipped",0)) for s in suites.iter("testsuite"))
    require(tests==10 and failures==0,"Preflight qualification incomplete")
    geo=read(OUT/"geometry_preflight.json")
    require(geo["images"]==512 and geo["status"]=="PASS_ORIGINAL_TRAINING_GEOMETRY_ALL_512","Geometry bridge qualification incomplete")
    require(geo["pilot_source"]==c["source_bindings"]["gate1_intervention_training_v2.py"],"Geometry preflight qualified another source")
    historical_checks=read(prior/"fp32_reference/historical_hash_checks.json")
    require(len(historical_checks)==128 and all(v["historical_input_and_full_raw_SHA_exact"] for v in historical_checks),"Reference reconstruction incomplete")
    mining=ATTEMPT/"training_supervision"
    selected={r["image_id"] for r in c["training_members"]}
    records=[read(mining/f"{i:04d}.json") for i in range(512)]
    require({r["image_id"] for r in records}==selected,"Training supervision membership differs")
    np_count=nn_count=0;common_pos=common_neg=0
    for index,row in enumerate(records):
        treatment,control=row["intervention"],row["hard_example"]
        for spec in (treatment,control):
            require(len(set(spec["positive_gt"]))==len(spec["positive_gt"]),"Duplicate positive supervision")
            require(len(set(map(tuple,spec["negative"])))==len(spec["negative"]),"Duplicate negative supervision")
        require(len(treatment["positive_gt"])==len(control["positive_gt"])<=8,"Positive budget differs")
        require(len(treatment["negative"])==len(control["negative"])<=8,"Negative budget differs")
        np_count+=len(treatment["positive_gt"]);nn_count+=len(treatment["negative"])
        common_pos+=len(set(treatment["positive_gt"])&set(control["positive_gt"]))
        common_neg+=len(set(map(tuple,treatment["negative"]))&set(map(tuple,control["negative"])))
        require(row["matching_coordinate_system"]=="original_publisher_integer_ignore_regions","Original matching geometry missing")
        require(set(row["rescued_targets"]).isdisjoint(row["collateral_targets"]),"Changed target identity overlap")
        require(set(x["target_id"] for x in row["positives"]).issubset(set(row["rescued_targets"]+row["collateral_targets"])),"Treatment includes non-intervention positive")
        # Check the matched control strata independently from saved raw and original labels.
        meta=c["training_annotation_metadata"][index]
        numbers=np.loadtxt(eval_root/meta["annotation_path"],delimiter=",",dtype=np.float64,ndmin=2)
        truth=numbers[(numbers[:,4]==1)&(numbers[:,5]>=1)&(numbers[:,5]<=10)]
        params=row["training_to_original_params"]
        def pos_stratum(k):
            t=truth[k];area=t[2]*t[3]*params["ratio"][0]*params["ratio"][1]
            return (int(t[5])-1,0 if area<1024 else (1 if area<9216 else 2))
        require(Counter(map(pos_stratum,treatment["positive_gt"]))==Counter(map(pos_stratum,control["positive_gt"])),"Positive class/size budget mismatch")
        with np.load(mining/f"{index:04d}_raw.npz",allow_pickle=False) as data:
            require(set(data.files)=={"q","r"},"Mining raw keys differ")
            for label in ("q","r"):
                require(data[label].shape==(1,14,8400) and data[label].dtype==np.float32 and np.isfinite(data[label]).all(),"Mining raw array domain differs")
            raw=data["r"]
            def neg_stratum(pair):
                anchor,cls=pair;score=float(raw[0,4+cls,anchor])
                require(score>=.25,"Negative below frozen threshold")
                return (cls,min(2,int((score-.25)/.25)))
            require(Counter(map(neg_stratum,treatment["negative"]))==Counter(map(neg_stratum,control["negative"])),"Negative class/confidence budget mismatch")
            centers=np.concatenate([np.stack(np.meshgrid(np.arange(640//s)+.5,np.arange(640//s)+.5),-1).reshape(-1,2)*s for s in (8,16,32)])
            boxes=numbers[:,:4].copy();boxes[:,2:]+=boxes[:,:2]
            for anchor,cls in set(map(tuple,treatment["negative"]))|set(map(tuple,control["negative"])):
                center=(centers[anchor]-[params["left"],params["top"]])/params["ratio"]
                require(not np.any((boxes[:,0]<=center[0])&(center[0]<boxes[:,2])&(boxes[:,1]<=center[1])&(center[1]<boxes[:,3])),"Negative anchor center falls in GT/ignore")
                b=raw[0,:4,anchor].astype(np.float64)
                low=(b[:2]-b[2:]/2-[params["left"],params["top"]])/params["ratio"]
                high=(b[:2]+b[2:]/2-[params["left"],params["top"]])/params["ratio"]
                area=np.maximum(0,high-low).prod()
                intersections=np.maximum(0,np.minimum(boxes[:,2:],high)-np.maximum(boxes[:,:2],low)).prod(1)
                ratios=intersections/np.maximum(1e-12,np.minimum(area,np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)))
                require(not np.any(ratios>=.1),"Negative overlaps original target or ignore region")
    receipts={arm:read(ATTEMPT/"training"/arm/"training_receipt.json") for arm in c["arms"]}
    require(len({r["initial_state_value_sha256"] for r in receipts.values()})==1,"Training initialization differs")
    checkpoint_files=[]
    for arm,receipt in receipts.items():
        require(receipt["optimizer_steps"]==96 and receipt["batches"]==384 and receipt["epochs"]==3,"Training budget incomplete")
        require(receipt["CUDA_reconstruction_exact"] and receipt["frozen_BN_and_scales_unchanged"] and receipt["changed_weight_elements"]>0,"Training/save acceptance incomplete")
        trace=receipt["trace"]
        require([x["step"] for x in trace]==list(range(1,97)),"Missing or duplicated optimizer step")
        require(all(np.isfinite(x["loss"]) and np.isfinite(x["grad_norm_before_clip"]) for x in trace),"Nonfinite trace")
        for epoch in range(1,4):
            path=ATTEMPT/"training"/arm/f"epoch_{epoch:02d}_fp32.pt"
            payload=torch.load(path,map_location="cpu",weights_only=True)
            require(payload["epoch"]==epoch and payload["arm"]==arm,"Checkpoint identity differs")
            require(payload["contract_sha256"]==identity(OUT/"contract.json")["sha256"],"Checkpoint contract differs")
            state=payload["state_dict"]
            floating=[v for v in state.values() if v.is_floating_point()]
            require(floating and all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in floating),"Checkpoint dtype/nonfinite")
            require(sum(n.endswith("activation_scale") for n in state)==88 and sum(n.endswith("weight_scale") for n in state)==88,"Saved quantizer scale coverage differs")
            checkpoint_files.append(identity(path))
    keys=("AP_all","AP_50","AP_75","AR_1","AR_10","AR_100","AR_500")
    metrics=[];guards={}
    for label in ["fp32_reference",*c["evaluation_order"]]:
        source=ATTEMPT if label in c["arms"] else prior
        folder=source/"fp32_reference" if label=="fp32_reference" else source/"evaluation"/label
        m=read(folder/"metrics.json")
        with np.load(folder/"AP_AR_arrays.npz",allow_pickle=False) as data:
            ap,ar,classes=data["AP"],data["AR"],data["evalClass"]
            require(ap.shape==(10,10) and ar.shape==(10,10,4) and len(classes)>0,"Saved AP array domain differs")
            idx=classes-1
            recomputed=[np.mean(ap[idx]),np.mean(ap[idx,0]),np.mean(ap[idx,5]),*[np.mean(ar[idx,:,i]) for i in range(4)]]
        require(all(np.isclose(m["metrics_percent"][k],v,atol=1e-10,rtol=1e-12) for k,v in zip(keys,recomputed)),"Saved AP aggregate differs")
        if label!="fp32_reference":
            records_eval=read(folder/"saved_predictions_and_targets.json")
            require([r["image_id"] for r in records_eval]==[r["image_id"] for r in c["evaluation_members"]],"Evaluation order differs")
            count=Counter()
            for row in records_eval:count.update(row["target_matching"]["counts"])
            require(dict(count)==m["project_counts_conf025"],"Saved target-count sum differs")
            with np.load(folder/"prepared_matrices.npz",allow_pickle=False) as data:
                require(set(data.files)=={f"{prefix}_{i}" for i in range(128) for prefix in ("gt","dt")},"Prepared matrix membership differs")
                require(all(data[k].ndim==2 and data[k].shape[1]==8 and np.isfinite(data[k]).all() for k in data.files),"Prepared matrix domain differs")
            raw_paths=list((folder/"raw").glob("*.npz"))
            require(len(raw_paths)==128,"Raw array coverage incomplete")
            g=read(folder/"all_anchor_guard.json")
            require(len(g)==128,"Guard coverage differs")
            by_id={item["image_id"].split("/")[-1]:item for item in g}
            passed=0
            for path in raw_paths:
                with np.load(path,allow_pickle=False) as data:
                    require(data["raw"].shape==(1,14,8400) and data["raw"].dtype==np.float32 and np.isfinite(data["raw"]).all(),"Saved raw array domain differs")
                    with np.load(prior/"fp32_reference"/path.name,allow_pickle=False) as ref:
                        guard=EVAL.raw_guard(ref["raw"],data["raw"])
                    require(guard==by_id[path.stem]["guard"],"Full all-anchor guard recomputation differs")
                    passed+=int(guard["allclose_pass"])
            guards[label]={"images":128,"passed":passed,"failed":128-passed,"all_8400_anchor_guards_recomputed":True}
        metrics.append(m)
    result={"status":"PASS_SAVED_ARTIFACT_INTEGRITY_AND_AGGREGATE_AUDIT_ONLY","at_utc":datetime.now(timezone.utc).isoformat(),
        "contract":identity(OUT/"contract.json"),"completion":identity(ATTEMPT/"completion_receipt.json"),
        "manifest_members_verified":len(files),"reused_files_verified":len(reused),"preflight_tests":10,"real_image_forwards":6014,
        "prior_attempt_forwards":708,"cumulative_forwards_excluding_9_reconstruction_checks":6722,
        "authorized_input_files_rehashed":input_files_checked,
        "checkpoints":checkpoint_files,"training_arms_initialization_equal":True,"optimizer_updates_each":96,
        "selected_positive_targets":np_count,"selected_negative_anchors":nn_count,
        "hard_intervention_positive_overlap":common_pos,"hard_intervention_negative_overlap":common_neg,
        "seven_metrics_reaggregated_from_saved_AP_arrays":True,"full_AP_matching_rerun":False,
        "model_inference_rerun":False,"calibration_rerun":False,"scientific_acceptance":False,"real_INT8":False,
        "mining_saved_arrays_and_control_strata_checked":512,"negative_background_geometry_independently_checked":True,"mining_matching_recomputed":False,
        "Holdout_accessed":False,"Selection_payload_accessed":False,"metrics":metrics,"guard_coverage":guards}
    with output.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"status":result["status"],"members":len(files),"positive":np_count,"negative":nn_count},ensure_ascii=True))


if __name__=="__main__":main()
