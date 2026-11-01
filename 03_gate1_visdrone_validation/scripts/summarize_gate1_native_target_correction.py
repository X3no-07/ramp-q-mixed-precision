"""Account for all fixed arms from saved development outputs, without inference."""
from collections import Counter
from datetime import datetime,timezone
import numpy as np
from audit_gate1_native_target_correction import N,P,OUT,A,metric_folder

read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/"contract.json")
    audit=read(OUT/"saved_delivery_audit.json")
    require(audit["status"]=="PASS_SAVED_SOURCE_STATE_BUDGET_AND_METRIC_AGGREGATE_AUDIT_ONLY","Audit incomplete")
    by_label={m["branch"]:m for m in audit["metrics"]}
    all_sets={};sources=[];fp_types={};fp_details=[]
    root=P.Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
    truth={}
    for row in c["evaluation_members"]:
        ann=(root/row["annotation_path"]).resolve(strict=True)
        require(ann.is_relative_to(root) and P.sha(ann)==row["annotation_sha256"],"Annotation changed")
        truth[row["image_id"]]={g["target_id"]:g for g in P.MATCH.parse_ground_truth(ann.read_text(encoding="utf-8-sig"),row["image_id"])}
    labels=("mse_q","mse_r","ordinary_qat","intervention",*N.ARMS)
    for label in labels:
        path=metric_folder(c,label)/"saved_predictions_and_targets.json"
        sources.append(identity(path));count=Counter({k:0 for k in N.KINDS});sets={}
        for meta,row in zip(c["evaluation_members"],read(path)):
            require(meta["image_id"]==row["image_id"],"Saved prediction order differs")
            match=row["target_matching"]
            sets[row["image_id"]]={"eligible":{g["target_id"] for g in match["targets"]},"hits":{g["target_id"] for g in match["matches"]}}
            eligible=[truth[row["image_id"]][g["target_id"]] for g in match["targets"]]
            boxes=np.array([P.MATCH._xyxy(g["bbox_xywh"],meta["width"],meta["height"]) for g in eligible],dtype=float).reshape(-1,4)
            classes=np.array([g["category"] for g in eligible])
            fps={s["detection_id"] for s in match["detection_states"] if s["reason"]=="false_positive"}
            classified=0
            for d in row["predictions"]:
                if d["detection_id"] not in fps:continue
                box=np.array([P.MATCH._xyxy(d["bbox_xywh"],meta["width"],meta["height"])])
                ious=P.box_iou_matrix(box,boxes)[0]
                kind=N.error_kind(ious,classes,d["category"]);count[kind]+=1;classified+=1
                fp_details.append({"branch":label,"image_id":row["image_id"],"detection_id":d["detection_id"],"kind":kind})
            require(classified==match["counts"]["false_positive_detections"],"FP diagnostic coverage differs")
        all_sets[label]=sets
        require(sum(count.values())==by_label[label]["project_counts_conf025"]["false_positive_detections"],"FP total differs")
        fp_types[label]=dict(count)
    totals={label:Counter() for label in labels[2:]};pairs=[];initial=Counter()
    for meta in c["evaluation_members"]:
        image_id=meta["image_id"];q,r=all_sets["mse_q"][image_id],all_sets["mse_r"][image_id]
        require(q["eligible"]==r["eligible"],"Eligible GTs differ")
        rescued=r["hits"]-q["hits"];initial.update(G1_rescued=len(rescued),G1_collateral=len(q["hits"]-r["hits"]))
        row={"image_id":image_id,"initial_G1_rescued_ids":sorted(rescued),"arms":{}}
        for label in labels[2:]:
            arm=all_sets[label][image_id];require(arm["eligible"]==r["eligible"],"Arm eligible GTs differ")
            gains=arm["hits"]-r["hits"];losses=r["hits"]-arm["hits"]
            count={"new_hits_vs_initial_R":len(gains),"lost_hits_vs_initial_R":len(losses),
                   "kept_initial_G1_rescues":len(rescued&arm["hits"]),"lost_initial_G1_rescues":len(rescued-arm["hits"])}
            totals[label].update(count);row["arms"][label]={**count,"gained_ids":sorted(gains),"lost_ids":sorted(losses)}
        pairs.append(row)
    differences={}
    for arm in N.ARMS:
        t=by_label[arm];comparison={}
        for label,b in by_label.items():
            if label==arm:continue
            delta={"AP50_95_percentage_points":t["metrics_percent"]["AP_all"]-b["metrics_percent"]["AP_all"],
                   "hits":t["project_counts_conf025"]["hits"]-b["project_counts_conf025"]["hits"],
                   "false_positives":t["project_counts_conf025"]["false_positive_detections"]-b["project_counts_conf025"]["false_positive_detections"]}
            delta["descriptive_joint_AP_up_hits_not_down_FP_not_up"]=delta["AP50_95_percentage_points"]>0 and delta["hits"]>=0 and delta["false_positives"]<=0
            comparison[label]=delta
        differences[arm]=comparison
    diagnostic={"scope":"fixed128development images,confidence0.25; all FP flags and target matchings reused",
        "sources":sources,"initial_G1":dict(initial),"target_summary":{k:dict(v) for k,v in totals.items()},
        "paired_images":pairs,"FP_geometry_counts":fp_types,"FP_geometry_details":fp_details,
        "geometry_partition_posthoc":True,"geometry_partition_not_causal":True,"low_overlap_not_proof_of_background":True,
        "model_inference":False,"new_AP_scoring":False,"scientific_acceptance":False}
    write(OUT/"saved_result_diagnostics.json",diagnostic)
    result={"status":"COMPLETE_FINITE_NATIVE_TARGET_CORRECTION_DEVELOPMENT_PILOT","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "contract":identity(OUT/"contract.json"),"saved_audit":identity(OUT/"saved_delivery_audit.json"),"summary_source":identity(__file__),
        "diagnostics":identity(OUT/"saved_result_diagnostics.json"),"metrics":list(by_label.values()),"differences":differences,
        "all_3_new_arms_completed":True,"epochs_each":3,"updates_each":96,"same_initialization_as_ordinary_QAT":True,
        "FP_error_training_strata":audit["error_types"],"positive_GT_selected":601,"error_channels_selected":audit["error_channels_selected"],
        "all_anchor_guard":audit["new_arms_guard"],"checkpoint_CUDA_reconstruction_exact":True,
        "CPU_CUDA_equivalence_certified":False,"CPU_CUDA_max_abs_by_arm":{
            arm:read(A/"training"/arm/"training_receipt.json")["CPU_CUDA_output_max_abs"] for arm in N.ARMS},
        "scientific_acceptance":False,"actual_INT8":False,"independent_test_set":False,"old_failures_preserved":True,
        "Selection_payload_access":False,"Holdout_payload_access":False,"native_current_submission_reevaluation":False,
        "candidate_selection":"all registered arms retained; fixed final epoch3; no selection of best threshold or seed",
        "limitation":"same development data reused after prior diagnostics; one training seed; confidence counts do not replace AP or all-anchor guard",
        "next_action":"Close this finite budget after delivery. Interpret intervention versus matched hard,localization-only,ordinary QAT and initial R separately before any further training stage."}
    write(OUT/"machine_decision.json",result)
    compact={arm:{k:differences[arm][k] for k in ("mse_r","ordinary_qat") if k!=arm} for arm in N.ARMS}
    print(P.json.dumps({"status":result["status"],"comparisons":compact,"FP_types":fp_types},ensure_ascii=True))


if __name__=="__main__":main()
