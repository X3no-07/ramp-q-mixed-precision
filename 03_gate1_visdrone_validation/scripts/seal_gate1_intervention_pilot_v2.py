"""Seal the finite pilot and update only the mutable execution pointer."""
from datetime import datetime, timezone
from pathlib import Path
import json
from summarize_gate1_intervention_pilot_v2 import ROOT, OUT, A, read, write, ident, need


def main():
    c=read(OUT/"contract.json")
    audit=read(OUT/"saved_delivery_audit.json")
    outcome=read(OUT/"machine_decision.json")
    exit_record=read(OUT/"process_exit_observation.json")
    terminal=read(OUT/"terminal_validation.json")
    need(audit["status"]=="PASS_SAVED_ARTIFACT_INTEGRITY_AND_AGGREGATE_AUDIT_ONLY","Audit incomplete")
    need(exit_record["exited"] and exit_record["exit_code"]==0 and terminal["matching_active_pilot_count"]==0,"No verified normal termination")
    need(not (A/"failure_receipt.json").exists(),"Current attempt failed")
    need(read(A/"completion_receipt.json")["forward_images"]==6014,"Budget incomplete")
    guard=read(OUT/"saved_audit_contract.json")
    need(ident(guard["script"]["path"])==guard["script"],"Frozen saved audit changed")
    latest=ROOT/"artifacts/gate1/latest_execution_state.json"
    previous=read(latest)
    need(previous["snapshot"]=="artifacts/gate1/current_execution_state_v29.json","Execution pointer advanced elsewhere")
    need(ident(ROOT/previous["snapshot"])["sha256"]==previous["snapshot_sha256"],"Prior snapshot changed")
    expected={}
    def add(binding):
        path=binding["path"]
        if path in expected:need(expected[path]==binding,"Conflicting duplicate file binding")
        expected[path]=binding
    for item in read(A/"manifest.json")["files"]:add(item)
    for item in list(c["source_bindings"].values())+list(c["previous_source_bindings"].values())+c["reused_files"]+c["reference_bindings"]:
        add(item)
    for key in ("candidate","architecture","formal_contract","train_manifest","training_val_manifest","evaluation_protocol","calibration_contract","old_ranges","previous_contract","previous_failure","training_annotation_assignment"):
        add(c[key])
    # Only the already authorized member lists are expanded. No data-directory scan.
    header=read(c["train_manifest"]["path"])
    train_root=(Path(c["train_manifest"]["path"]).parent/header["data_root"]).resolve(strict=True)
    for row in c["training_members"]:
        for kind in ("image","labels","sidecar"):
            member=row[kind];p=(train_root/member["path"]).resolve(strict=True)
            need(p.is_relative_to(train_root),"Path escaped training root")
            add({"path":str(p),"bytes":member["bytes"],"sha256":member["sha256"]})
    eval_root=Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
    for rows,kinds in ((c["training_annotation_metadata"],("annotation",)),(c["evaluation_members"],("image","annotation")),(c["calibration_members"],("image",))):
        for row in rows:
            for kind in kinds:
                p=(eval_root/row[kind+"_path"]).resolve(strict=True)
                need(p.is_relative_to(eval_root),"Path escaped allowed root")
                b=ident(p);need(b["sha256"]==row[kind+"_sha256"],"Authorized input changed")
                add(b)
    for root in (OUT,Path(c["previous_contract"]["path"]).parent):
        for p in sorted(root.rglob("*")):
            if p.is_file():add(ident(p))
    for p in [ROOT/previous["snapshot"],ROOT/"docs/65_gate1_precision_intervention_correction_pilot_launch.md",
              ROOT/"docs/66_gate1_correction_coordinate_fix_and_training_resumption.md",ROOT/"docs/67_gate1_intervention_correction_pilot_results_and_false_positive_diagnostics.md",
              ROOT/"scripts/preflight_gate1_training_geometry_v2.py",ROOT/"scripts/audit_gate1_intervention_pilot_v2.py",
              ROOT/"scripts/summarize_gate1_intervention_pilot_v2.py",ROOT/"scripts/diagnose_gate1_pilot_fp_v2.py",Path(__file__)]:
        add(ident(p))
    for b in expected.values():need(ident(b["path"])==b,"Delivery binding differs: "+b["path"])
    manifest={"schema":"rampq.corrective_qat.finite_pilot_delivery/v2","created_at_utc":datetime.now(timezone.utc).isoformat(),
              "files":[expected[k] for k in sorted(expected)],"contains_authorized_train_calibration_training_val_only":True,
              "mutable_latest_pointer_excluded":True,"post_delivery_snapshot_binds_this_manifest":True}
    write(OUT/"delivery_manifest.json",manifest)
    validation={"status":"PASS_COMPLETE_FINITE_PILOT_DELIVERY_INTEGRITY_ONLY","at_utc":datetime.now(timezone.utc).isoformat(),
                "manifest":ident(OUT/"delivery_manifest.json"),"members_verified":len(expected),"missing":0,"mismatched":0,
                "saved_audit":ident(OUT/"saved_delivery_audit.json"),"machine_decision":ident(OUT/"machine_decision.json"),
                "report":ident(ROOT/"docs/67_gate1_intervention_correction_pilot_results_and_false_positive_diagnostics.md"),
                "worker_exit_code":0,"scientific_acceptance":False,"real_INT8":False,"Holdout_accessed":False}
    write(OUT/"delivery_validation.json",validation)
    state={"schema":"rampq.gate1.execution_state/v30","status":"COMPLETE_CORRECTIVE_QAT_PILOT_NO_OVERALL_IMPROVEMENT",
           "created_at_utc":datetime.now(timezone.utc).isoformat(),"previous_snapshot":previous,
           "current_experiment":"artifacts/gate1/20261026_intervention_training_pilot_v2",
           "report":"docs/67_gate1_intervention_correction_pilot_results_and_false_positive_diagnostics.md","delivery_validation":ident(OUT/"delivery_validation.json"),
           "machine_decision":ident(OUT/"machine_decision.json"),"contract_sha256":ident(OUT/"contract.json")["sha256"],
           "three_arms_completed":True,"candidate_rule":"fixed_epoch3_all_three_arms_retained",
           "intervention_vs_start":outcome["intervention_minus_each_comparator"]["mse_r"],
           "numerical_qualification_passed":False,"scientific_acceptance":False,"real_INT8":False,
           "Selection_payload_access":False,"Holdout_payload_access":False,"old_failures_preserved":True,
           "active_heavy_worker":False,"heartbeat_recreated":False,"cloud_resources_created":False,
           "next_action":"Pilot budget exhausted normally and all outputs sealed. Do not rerun completed stages or tune frozen parameters. Saved diagnostic suggests examining localization, duplicate and class errors before registering a further training budget; do not call this experiment a method success."}
    snap=ROOT/"artifacts/gate1/current_execution_state_v30.json"
    write(snap,state)
    pointer={"schema":"rampq.gate1.mutable_latest_state_pointer/v1","snapshot":"artifacts/gate1/current_execution_state_v30.json",
             "snapshot_sha256":ident(snap)["sha256"],"delivery_validation":ident(OUT/"delivery_validation.json"),
             "note":"Finite3arm pilot completed/audited. Intervention adds70hits but517FP vs start; AP-0.162391pp. No overall improvement; all old guards fail. No active training, no cloud upload, Holdout sealed."}
    temp=latest.with_name("latest_execution_state.v30.pending.json")
    write(temp,pointer);temp.replace(latest)
    print(json.dumps({"status":validation["status"],"members":len(expected),"latest":pointer},ensure_ascii=True))


if __name__=="__main__":main()
