"""Describe all frozen pilot arms using only already saved scores and GT identities."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/gate1/20261026_intervention_training_pilot_v2"
A=OUT/"attempt_01"


def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def ident(p):
    p=Path(p).resolve(strict=True)
    with p.open("rb") as f:h=hashlib.file_digest(f,"sha256").hexdigest()
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":h}


def need(ok,msg):
    if not ok:raise ValueError(msg)


def write(p,obj):
    with Path(p).open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    c=read(OUT/"contract.json")
    audit=read(OUT/"saved_delivery_audit.json")
    need(audit["status"]=="PASS_SAVED_ARTIFACT_INTEGRITY_AND_AGGREGATE_AUDIT_ONLY","Audit incomplete")
    prior=Path(c["reused_stage_root"])
    labels=["mse_q","mse_r",*c["arms"]]
    all_sets={};sources=[]
    for label in labels:
        folder=A if label in c["arms"] else prior
        p=folder/"evaluation"/label/"saved_predictions_and_targets.json"
        sources.append(ident(p));all_sets[label]={}
        for r in read(p):
            t=r["target_matching"]
            all_sets[label][r["image_id"]]={"eligible":{x["target_id"] for x in t["targets"]},
                "hits":{x["target_id"] for x in t["matches"]},"counts":t["counts"]}
    ids=[r["image_id"] for r in c["evaluation_members"]]
    need(all(list(all_sets[k])==ids for k in labels),"Saved evaluation membership/order differs")
    paired=[];summary={label:Counter() for label in c["arms"]};totals=Counter()
    for image_id in ids:
        q,r=all_sets["mse_q"][image_id],all_sets["mse_r"][image_id]
        need(q["eligible"]==r["eligible"],"Q/R eligible identity differs")
        rescued=r["hits"]-q["hits"];collateral=q["hits"]-r["hits"]
        totals.update(rescued_by_initial_G1=len(rescued),lost_by_initial_G1=len(collateral))
        row={"image_id":image_id,"initial_G1_rescued_ids":sorted(rescued),"initial_G1_lost_ids":sorted(collateral),"arms":{}}
        for arm in c["arms"]:
            s=all_sets[arm][image_id]
            need(s["eligible"]==r["eligible"],"Training arm eligible identity differs")
            gain=s["hits"]-r["hits"];loss=r["hits"]-s["hits"]
            counts={"kept_initial_G1_rescues":len(s["hits"]&rescued),"lost_initial_G1_rescues":len(rescued-s["hits"]),
                    "new_hits_vs_initial_R":len(gain),"lost_hits_vs_initial_R":len(loss),
                    "FP_delta_vs_initial_R":s["counts"]["false_positive_detections"]-r["counts"]["false_positive_detections"]}
            summary[arm].update(counts)
            row["arms"][arm]={**counts,"gained_target_ids":sorted(gain),"lost_target_ids":sorted(loss)}
        paired.append(row)
    metrics={m["branch"]:m for m in audit["metrics"]}
    treatment=metrics["intervention"]
    differences={}
    for label in ["fp32_reference","legacy_q","legacy_r","mse_q","mse_r","ordinary_qat","hard_example"]:
        b=metrics[label];t=treatment
        delta={"AP50_95_percentage_points":t["metrics_percent"]["AP_all"]-b["metrics_percent"]["AP_all"],
               "hits":t["project_counts_conf025"]["hits"]-b["project_counts_conf025"]["hits"],
               "false_positives":t["project_counts_conf025"]["false_positive_detections"]-b["project_counts_conf025"]["false_positive_detections"]}
        delta["joint_AP_up_hits_not_down_FP_not_up_observation"]=delta["AP50_95_percentage_points"]>0 and delta["hits"]>=0 and delta["false_positives"]<=0
        differences[label]=delta
    diagnostic={"scope":"post-training descriptive accounting of the fixed 128 development images at confidence0.25; no new model or scoring run",
                "sources":sources,"initial_G1":dict(totals),"arms":{k:dict(v) for k,v in summary.items()},"paired_images":paired,
                "scientific_acceptance":False,"independent_test_set":False}
    write(OUT/"saved_target_diagnostics.json",diagnostic)
    decisions={"status":"COMPLETE_SINGLE_SEED_DEVELOPMENT_PILOT", "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "contract":ident(OUT/"contract.json"),"saved_artifact_audit":ident(OUT/"saved_delivery_audit.json"),
        "target_diagnostics":ident(OUT/"saved_target_diagnostics.json"),"summary_script":ident(__file__),
        "training_completed":True,"checkpoint_CUDA_reconstruction_exact":True,
        "CPU_CUDA_numerical_equivalence_certified":False,"CPU_CUDA_max_abs_by_arm":{
            arm:read(A/"training"/arm/"training_receipt.json")["CPU_CUDA_output_max_abs"] for arm in c["arms"]},
        "all_anchor_guard":audit["guard_coverage"],"training_val_development_metrics":list(metrics.values()),
        "intervention_minus_each_comparator":differences,
        "current_128_submission_native_reevaluation":False,"real_INT8":False,"scientific_acceptance":False,
        "Selection_payload_access":False,"Holdout_payload_access":False,
        "inference_on_saved_analysis":False,"calibration_repeated":False,"old_failures_preserved":True,
        "limits":["one training seed, three epochs,512training and128development images",
                  "all data remain development evidence; filename prefix is not verified scene identity",
                  "same initialization/optimizer/order/batch and auxiliary-count budget; wall-time is reported separately",
                  "fixed final epoch3, all arms retained; no best epoch/seed/threshold selection",
                  "training used floating W8A8 QDQ and labels, not pure PTQ or real integer deployment"],
        "next_action":"Finite pilot closed. Interpret all comparator deltas before freezing any additional training budget or arranging cloud/device access."}
    write(OUT/"machine_decision.json",decisions)
    print(json.dumps({"status":decisions["status"],"intervention_minus_comparators":differences,"target_accounting":{k:dict(v) for k,v in summary.items()}},ensure_ascii=True))


if __name__=="__main__":main()
