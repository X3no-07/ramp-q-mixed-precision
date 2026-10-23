"""Independent static and saved-metadata review; no model or dataset execution."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
PACKAGE=ROOT/"artifacts/gate1/20261023_fp32_live_capture_v1"
FRAMEWORK=ROOT.parent/"05_水下小目标检测_工程/检测工程/.venv/Lib/site-packages/ultralytics"


def ident(path):
    raw=path.read_bytes()
    return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)


def spans(path,names):
    tree=ast.parse(path.read_text(encoding="utf-8"))
    result={}
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            result.setdefault(node.name,[]).append([node.lineno,node.end_lineno])
    return result


def main():
    destination=OUT/"fp32_source_review.json"
    if destination.exists():raise RuntimeError("do not overwrite review")
    manifest=read(PACKAGE/"manifest.json")
    verified={}
    for rel,expected in manifest["files"].items():
        assert ident(PACKAGE/rel)==expected,rel
        verified[rel]=expected
    for rel,expected in manifest["external_sources"].items():
        assert ident(ROOT/rel)==expected,rel
    # Nested reference/outer manifests are independently rechecked as manifests,
    # in addition to being direct members of the component package.
    nested={}
    for rel in ("synthetic_capture/epoch_0001/manifest.json","synthetic_capture/epoch_0001/state/manifest.json"):
        path=PACKAGE/rel
        obj=read(path)
        for member,expected in obj["files"].items():assert ident(path.parent/member)==expected,member
        nested[rel]={"files":len(obj["files"]),**ident(path)}
    pins=read(PACKAGE/"runtime_source_fingerprints.json")
    inventory={p.relative_to(FRAMEWORK).as_posix():ident(p)["sha256"] for p in sorted(FRAMEWORK.rglob("*.py"))}
    assert pins["ultralytics_python_tree_sha256"]==hashlib.sha256(canonical(inventory).encode()).hexdigest()
    assert pins["ultralytics_python_files"]==len(inventory)
    for field,rel in (("capture_sha256","scripts/gate1_fp32_live_capture.py"),("state_format_sha256","scripts/gate1_fp32_reference_state.py")):
        assert pins[field]==ident(ROOT/rel)["sha256"]
    capture=read(PACKAGE/"synthetic_capture/epoch_0001/capture_receipt.json")
    state=read(PACKAGE/"synthetic_capture/epoch_0001/state/snapshot_receipt.json")
    architecture=read(PACKAGE/"synthetic_capture/epoch_0001/architecture.json")
    arch_sha=hashlib.sha256(canonical(architecture).encode()).hexdigest()
    assert arch_sha==capture["architecture_sha256"]==state["metadata"]["architecture_sha256"]
    state_sha=hashlib.sha256(canonical(state["tensors"]).encode()).hexdigest()
    assert state_sha==capture["live_state_before_sha256"]==capture["live_state_after_sha256"]
    assert capture["sources"]==pins
    assert capture["ema_updates_before"]==capture["ema_updates_after"]==2
    assert capture["state_and_update_counter_unchanged"] is True
    assert capture["scientific_reference_accepted"] is False
    assert capture["reconstruction"]["status"]=="PASS_STRICT_STATE_AND_SYNTHETIC_CPU_OUTPUT_RECONSTRUCTION_ONLY"
    assert all(v["dtype"] in {"torch.float32","torch.int64"} for v in state["tensors"].values())
    tests=read(PACKAGE/"test_execution_receipt.json")
    assert tests["tests_run"]==23 and tests["failures"]==tests["errors"]==tests["exit_code"]==0
    assert tests["full_terminal_transcript_retained_as_artifact"] is False
    sources={rel:ident(ROOT/rel) for rel in ("scripts/gate1_fp32_live_capture.py","tests/test_gate1_fp32_live_capture.py",
             "scripts/gate1_fp32_reference_state.py","scripts/gate1_ignore_trainer.py")}
    framework={rel:ident(FRAMEWORK/rel) for rel in ("engine/trainer.py","models/yolo/detect/train.py","nn/tasks.py","utils/torch_utils.py")}
    obj={"schema":"rampq.gate1.fp32_live_capture_independent_source_review/v1",
         "status":"PASS_SOURCE_AND_COMPONENT_METADATA_FOR_SYNTHETIC_CPU_TRAINER_SMOKE_ONLY",
         "issues":[],"sources":sources,"framework_sources":framework,
         "component_manifest":{"path":(PACKAGE/"manifest.json").relative_to(ROOT).as_posix(),**ident(PACKAGE/"manifest.json")},
         "verified_direct_component_members":len(verified),"verified_nested_manifests":nested,
         "current_ultralytics_python_inventory_matches_component_pins":True,
         "reviewed_source_spans":{
             "scripts/gate1_fp32_live_capture.py":spans(ROOT/"scripts/gate1_fp32_live_capture.py",{"architecture","reconstruct_and_verify","LiveFP32Capture","FP32LiveCaptureMixin"}),
             "scripts/gate1_ignore_trainer.py":spans(ROOT/"scripts/gate1_ignore_trainer.py",{"RestrictedIgnoreTrainer","save_model"}),
             "framework/engine/trainer.py":spans(FRAMEWORK/"engine/trainer.py",{"save_model","optimizer_step"})},
         "findings":[
             {"topic":"real_trainer_order","conclusion":"Source proves the callback call occurs after RestrictedIgnoreTrainer finite checks and before super().save_model(); DetectionTrainer does not override save_model, so that super resolves to BaseTrainer whose FP16 conversion acts on deepcopy(EMA). Runtime execution of this actual MRO remains the root smoke's responsibility."},
             {"topic":"mixin_order","conclusion":"FP32LiveCaptureMixin must be first in MRO and must not be combined with fp32_capture_callback. Its test uses a substitute Base.save_model, so the component test alone is not a real DetectionTrainer lifecycle proof."},
             {"topic":"thread_and_noninterference","conclusion":"Constructor/callback require original main thread; nonblocking hook lock prevents hook reentry. Before/after tensor hashes and EMA update count detect supported mutations. Deepcopy forwards protect live state and head caches. These checks do not exclude arbitrary external concurrent writers, which caller protocol must forbid."},
             {"topic":"already_fp32_origin","conclusion":"All floating source tensors must already be finite FP32, CPU, AMP args/effective false, enabled eval EMA with completed updates. The hook performs no half-to-float recovery. Dtype checks cannot independently detect a caller's earlier FP16-to-FP32 lift; fresh no-checkpoint training provenance belongs to the bound root protocol."},
             {"topic":"restricted_state_and_architecture","conclusion":"Only a tensor state payload is loaded via frozen FORMAT weights_only=True. Architecture comes from a canonical allowed-key/module cfg dictionary, not an external YAML/model pickle. Fresh DetectionModel strict load requires no missing/unexpected keys; state shape/dtype/value hashes and analytic-input output structure/dtype/value hashes must match the frozen live copy. Rebuilding inference-only does not rebuild the custom training criterion or optimizer/resume state."},
             {"topic":"state_commit","conclusion":"A selected epoch writes into a new .pending directory and commits via same-parent rename only after source identity, exact state, reconstruction, and unchanged live state/update checks; existing epoch outputs cannot be overwritten."},
             {"topic":"package_evidence","conclusion":"All 15 direct component members, external source pins and both nested manifests match. Saved architecture and state metadata bind consistently; 23/23 tests are reported by an honest terminal-summary receipt. Full raw unittest transcript was not retained and this review does not claim independent test rerun or independent tensor deserialization/forward."}],
         "required_root_checks":["bind current Dataset/Trainer/capture/format source bytes in the training contract before run",
                                 "fresh random architecture and generated synthetic-only inputs; prohibit checkpoint/model weight reads",
                                 "execute actual one-epoch CPU DetectionTrainer optimizer/EMA lifecycle with workers=0 and AMP=false",
                                 "observe capture completion before framework half serialization via actual training events",
                                 "independently audit captured new state/architecture/output and preserved live FP32 EMA"],
         "allowed_claim":"source-reviewed CPU synthetic live-capture component, pending actual Trainer smoke",
         "not_accepted":["real_dataset_training","GPU_capture","old_checkpoint_recovery","complete_training_or_optimizer_resume",
                         "official_AP","scientific_reference","scientific_Gate1","quantization","Orin_deployment","performance"],
         "old_checkpoint_loaded":False,"real_data_or_holdout_read":False,"source_mutated":False,
         "review_type":"independent_static_source_and_hash_bound_saved_metadata_audit_not_runtime_execution"}
    destination.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":obj["status"],"review":ident(destination),"verified_component_members":len(verified)}))


if __name__=="__main__":main()
