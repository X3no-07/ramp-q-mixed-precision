"""Post hoc geometric partition of SAVED false positives; no new scoring or tuning."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import gate1_ignore_aware_evaluation as MATCH
from summarize_gate1_intervention_pilot_v2 import ROOT, OUT, A, read, write, ident, need

KINDS=("same_class_overlap_candidate","other_class_overlap","near_object_low_IoU","low_overlap_remainder")


def kind(iou,classes,category):
    same=iou[classes==category]
    if len(same) and same.max()>=.5:return KINDS[0]
    if len(iou) and iou.max()>=.5:return KINDS[1]
    if len(iou) and iou.max()>=.1:return KINDS[2]
    return KINDS[3]


def main():
    for expected,values,classes,category in [
        (KINDS[0],[.5],[1],1),(KINDS[1],[.6],[2],1),
        (KINDS[2],[.1],[1],1),(KINDS[3],[.09],[1],1)]:
        need(kind(np.array(values),np.array(classes),category)==expected,"Partition boundary self-check failed")
    c=read(OUT/"contract.json")
    prior=Path(c["reused_stage_root"])
    root=Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
    contract={"diagnostic_only":True,"defined_after_pilot_results":True,"source":ident(__file__),
              "pilot_contract":ident(OUT/"contract.json"),"priority":list(KINDS),"high_iou":.5,"near_iou":.1,
              "uses_already_saved_FP_flags_confidence":.25,"labels":"only existing128training_val members",
              "not_causal_attribution":True,"not_a_new_acceptance_gate":True,
              "low_overlap_is_not_proof_of_pure_background":True,"no_threshold_or_candidate_selection":True}
    write(OUT/"fp_diagnostic_contract.json",contract)
    gt={}
    sources=[]
    for meta in c["evaluation_members"]:
        p=(root/meta["annotation_path"]).resolve(strict=True)
        need(p.is_relative_to(root) and ident(p)["sha256"]==meta["annotation_sha256"],"Annotation binding differs")
        gt[meta["image_id"]]={g["target_id"]:g for g in MATCH.parse_ground_truth(p.read_text(encoding="utf-8-sig"),meta["image_id"])}
    results={};details=[]
    for label in ["mse_r",*c["arms"]]:
        folder=A if label in c["arms"] else prior
        p=folder/"evaluation"/label/"saved_predictions_and_targets.json"
        sources.append(ident(p));count=Counter({k:0 for k in KINDS})
        for meta,row in zip(c["evaluation_members"],read(p)):
            need(meta["image_id"]==row["image_id"],"Wrong evaluation order")
            match=row["target_matching"]
            eligible=[gt[row["image_id"]][x["target_id"]] for x in match["targets"]]
            boxes=np.array([MATCH._xyxy(g["bbox_xywh"],meta["width"],meta["height"]) for g in eligible],dtype=float).reshape(-1,4)
            classes=np.array([g["category"] for g in eligible])
            areas=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
            fps={x["detection_id"] for x in match["detection_states"] if x["reason"]=="false_positive"}
            evaluated=0
            for d in row["predictions"]:
                if d["detection_id"] not in fps:continue
                b=np.array(MATCH._xyxy(d["bbox_xywh"],meta["width"],meta["height"]))
                area=np.maximum(0,b[2:]-b[:2]).prod()
                intersection=np.maximum(0,np.minimum(boxes[:,2:],b[2:])-np.maximum(boxes[:,:2],b[:2])).prod(1)
                iou=intersection/np.maximum(1e-12,areas+area-intersection)
                partition=kind(iou,classes,d["category"]);count[partition]+=1;evaluated+=1
                details.append({"branch":label,"image_id":row["image_id"],"detection_id":d["detection_id"],
                                "partition":partition,"maximum_eligible_GT_IoU":float(iou.max()) if len(iou) else 0.0})
            need(evaluated==match["counts"]["false_positive_detections"],"FP coverage differs")
        m=read(folder/"evaluation"/label/"metrics.json")
        need(sum(count.values())==m["project_counts_conf025"]["false_positive_detections"],"FP total differs")
        results[label]=dict(count)
    result={"contract":ident(OUT/"fp_diagnostic_contract.json"),"sources":sources,"counts":results,
            "intervention_minus_initial_R":{k:results["intervention"][k]-results["mse_r"][k] for k in KINDS},
            "no_inference_or_scoring_rerun":True,"all_FP_flags_preserved":True,"not_causal_attribution":True,
            "low_overlap_is_not_proof_of_pure_background":True,"details":details}
    write(OUT/"saved_false_positive_diagnostics.json",result)
    print(json.dumps({"counts":results,"deltas":result["intervention_minus_initial_R"]}))


if __name__=="__main__":main()
