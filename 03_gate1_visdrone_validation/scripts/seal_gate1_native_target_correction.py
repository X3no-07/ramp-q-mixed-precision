"""Seal this registered finite pilot; preserve historical files and snapshots."""
from datetime import datetime,timezone
from pathlib import Path
import json
from audit_gate1_native_target_correction import N,P,OUT,A

read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/"contract.json");decision=read(OUT/"machine_decision.json")
    audit=read(OUT/"saved_delivery_audit.json");terminal=read(OUT/"terminal_validation.json")
    exit_record=read(OUT/"process_exit_observation.json")
    require(exit_record["exited"] and terminal["matching_active_pilot_count"]==0 and terminal["stderr_final_bytes"]==0,"Termination not verified")
    require(exit_record["completion_receipt_exists"] and not exit_record["failure_receipt_exists"],"Terminal receipt check incomplete")
    require(not (A/"failure_receipt.json").exists(),"Current attempt failed")
    require(read(A/"completion_receipt.json")["forward_images"]==4992,"Budget incomplete")
    audit_contract=read(OUT/"saved_audit_contract.json")
    require(identity(audit_contract["script"]["path"])==audit_contract["script"],"Independent audit source changed")
    latest=P.ROOT/"artifacts/gate1/latest_execution_state.json";previous=read(latest)
    require(previous["snapshot"]=="artifacts/gate1/current_execution_state_v31.json","Latest pointer changed elsewhere")
    require(identity(P.ROOT/previous["snapshot"])["sha256"]==previous["snapshot_sha256"],"Previous state changed")
    expected={}
    def add(b):
        if b["path"] in expected:require(expected[b["path"]]==b,"Conflicting identity")
        expected[b["path"]]=b
    for b in read(A/"manifest.json")["files"]+c["reused_files"]+c["reference_bindings"]+list(c["source_bindings"].values())+list(c["previous_source_bindings"].values()):add(b)
    for key in ("candidate","architecture","formal_contract","train_manifest","training_val_manifest","evaluation_protocol","calibration_contract","old_ranges","previous_contract","previous_delivery","previous_comparison","training_annotation_assignment"):
        add(c[key])
    header=read(c["train_manifest"]["path"])
    train_root=(Path(c["train_manifest"]["path"]).parent/header["data_root"]).resolve(strict=True)
    for row in c["training_members"]:
        for kind in ("image","labels","sidecar"):
            b=row[kind];p=(train_root/b["path"]).resolve(strict=True)
            require(p.is_relative_to(train_root),"Training path escaped")
            add({"path":str(p),"bytes":b["bytes"],"sha256":b["sha256"]})
    root=Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
    for rows,kinds in ((c["training_annotation_metadata"],("annotation",)),(c["evaluation_members"],("image","annotation")),(c["calibration_members"],("image",))):
        for row in rows:
            for kind in kinds:
                p=(root/row[kind+"_path"]).resolve(strict=True)
                require(p.is_relative_to(root),"Input path escaped")
                actual=identity(p);require(actual["sha256"]==row[kind+"_sha256"],"Allowed input changed")
                add(actual)
    for root in (OUT,N.PREVIOUS):
        for p in sorted(root.rglob("*")):
            if p.is_file():add(identity(p))
    for p in (P.ROOT/previous["snapshot"],P.ROOT/"docs/68_gate1_native_objective_aligned_correction_pilot_launch.md",
              P.ROOT/"docs/69_gate1_native_objective_aligned_correction_results.md",P.ROOT/"scripts/audit_gate1_native_target_correction.py",
              P.ROOT/"scripts/summarize_gate1_native_target_correction.py",P.ROOT/"scripts/report_gate1_native_target_correction.py",Path(__file__)):
        add(identity(p))
    for b in expected.values():require(identity(b["path"])==b,"Delivery file differs: "+b["path"])
    manifest={"schema":"rampq.native_target_correction.delivery/v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
              "files":[expected[k] for k in sorted(expected)],"only_explicitly_authorized_data_members":True,
              "mutable_pointer_excluded":True}
    write(OUT/"delivery_manifest.json",manifest)
    validation={"status":"PASS_FINITE_PILOT_DELIVERY_INTEGRITY_ONLY","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "manifest":identity(OUT/"delivery_manifest.json"),"members_verified":len(expected),"missing":0,"mismatched":0,
        "audit":identity(OUT/"saved_delivery_audit.json"),"machine_decision":identity(OUT/"machine_decision.json"),
        "report":identity(P.ROOT/"docs/69_gate1_native_objective_aligned_correction_results.md"),"worker_exit_code":exit_record["exit_code"],
        "exit_code_directly_observed":False,"completion_and_process_exit_verified":True,
        "scientific_acceptance":False,"actual_INT8":False,"Holdout_accessed":False}
    write(OUT/"delivery_validation.json",validation)
    candidate="native_intervention_correction"
    state={"schema":"rampq.gate1.execution_state/v32","status":"COMPLETE_FINITE_NATIVE_TARGET_CORRECTION_PILOT",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"previous_snapshot":previous,
        "current_experiment":"artifacts/gate1/20261026_native_target_correction_pilot_v1",
        "report":"docs/69_gate1_native_objective_aligned_correction_results.md","delivery_validation":identity(OUT/"delivery_validation.json"),
        "machine_decision":identity(OUT/"machine_decision.json"),"contract_sha256":identity(OUT/"contract.json")["sha256"],
        "new_training_arms":list(N.ARMS),"training_complete":True,"all_final_epoch3_candidates_retained":True,
        "intervention_vs_initial_R":decision["differences"][candidate]["mse_r"],
        "intervention_vs_native_hard":decision["differences"][candidate]["native_hard_correction"],
        "all_anchor_guard":audit["new_arms_guard"],"CPU_CUDA_equivalence_certified":False,
        "scientific_acceptance":False,"actual_INT8":False,"Selection_payload_access":False,"Holdout_payload_access":False,
        "no_active_worker":True,"old_failures_preserved":True,"cloud_resources_created":False,"heartbeat_recreated":False,
        "next_action":"Registered finite budget complete and sealed. Read report69 and all11fixed comparator outcomes; do not repeat prior inference/calibration/scoring, overwrite contracts or pick best threshold/seed. Any later budget must be explicitly fixed before execution; current evidence remains development only."}
    snap=P.ROOT/"artifacts/gate1/current_execution_state_v32.json";write(snap,state)
    pointer={"schema":"rampq.gate1.mutable_latest_state_pointer/v1","snapshot":"artifacts/gate1/current_execution_state_v32.json",
        "snapshot_sha256":identity(snap)["sha256"],"delivery_validation":identity(OUT/"delivery_validation.json"),
        "note":"Finite native-TAL-target correction3arm pilot complete and audited. See report69 for AP/hit/FP tradeoffs; all registered outcomes retained, no active worker, Holdout sealed."}
    temporary=latest.with_name("latest_execution_state.v32.pending.json")
    write(temporary,pointer);temporary.replace(latest)
    print(json.dumps({"status":validation["status"],"members_verified":len(expected),"latest_snapshot":pointer["snapshot"],"snapshot_sha256":pointer["snapshot_sha256"]},ensure_ascii=True))


if __name__=="__main__":main()
