"""Preserve failed one-thread reconstruction and its resolved thread diagnosis."""
from pathlib import Path
import hashlib
import json

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
def identity(path):
    raw=path.read_bytes()
    return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}

if (OUT/"manifest.json").exists():raise RuntimeError("never overwrite")
diagnosis=json.loads((OUT/"thread_diagnosis/diagnosis.json").read_text())
assert diagnosis["before_select_device"]==1 and diagnosis["after_select_device"]==8
assert diagnosis["results"][0]["threads"]==1 and diagnosis["results"][0]["exact_output_matches_capture"] is False
assert diagnosis["results"][1]["threads"]==8 and diagnosis["results"][1]["exact_output_matches_capture"] is True
failure={"schema":"rampq.gate1.synthetic_training_output_audit_failure/v1",
         "status":"FAILED_DECLARED_ONE_THREAD_CPU_RECONSTRUCTION_CONTEXT",
         "attempt_01_audit_failed_at":"independent CPU output differs after verified new state-only load and strict reconstruction",
         "attempt_02_new_state_diagnosis":"fresh CPU model exact at 8 threads and not exact at 1 thread",
         "root_cause":"Ultralytics select_device(cpu) calls torch.set_num_threads(NUM_THREADS=8) after runner's initial setting of 1",
         "fixed_input_and_state_values_equal":True,"precision_threshold_changed":False,
         "receipt_threads_declaration_accepted":False,"strict_offline_01_accepted":False,
         "normal_model_checkpoint_deserialized":False,"old_weights_or_real_data_read":False,
         "required_new_attempt":"reassert effective intraop=1 after Trainer construction, verify actual context at start/save and record values; independently reconstruct new state in one-thread process",
         "scientific_gate_pass":False}
(OUT/"failure_receipt.json").write_text(json.dumps(failure,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
files={p.relative_to(OUT).as_posix():identity(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
manifest={"schema":"rampq.gate1.synthetic_training_failed_audit_manifest/v1","status":failure["status"],"files":files}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":failure["status"],"files":len(files),"manifest":identity(OUT/"manifest.json")}))
