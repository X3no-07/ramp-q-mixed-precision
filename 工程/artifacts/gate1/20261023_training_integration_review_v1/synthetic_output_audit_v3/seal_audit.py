"""Seal independent audit source, stdout, result and private runtime settings."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
def ident(path):
    raw=path.read_bytes()
    return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}

require_new=OUT/"manifest.json"
if require_new.exists():raise RuntimeError("never overwrite audit manifest")
result=json.loads((OUT/"audit_result.json").read_text(encoding="utf-8"))
assert result["status"]=="PASS_ACTUAL_SYNTHETIC_CPU_RECORDS_AND_INDEPENDENT_NEW_FP32_RECONSTRUCTION_ONLY"
assert result["issues"]==[] and result["independent_analytic_cpu_output_exact"] is True
assert result["changed_parameter_tensors"]==238 and result["changed_parameter_elements"]==2209385
for key in ("preparation","outputs"):
    info=result[key]
    assert ident(ROOT/info["path"])=={k:info[k] for k in ("bytes","sha256")}
for info in result["nested_manifests"]:
    assert ident(ROOT/info["path"])=={k:info[k] for k in ("bytes","sha256")}
files={p.relative_to(OUT).as_posix():ident(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
manifest={"schema":"rampq.gate1.independent_synthetic_training_output_audit_manifest/v1",
          "status":result["status"],"files":files,"input_manifests":{
              "preparation":result["preparation"],"outputs":result["outputs"],"nested":result["nested_manifests"]},
          "prior_failed_context_diagnosis":{"path":"artifacts/gate1/20261023_training_integration_review_v1/synthetic_output_audit_v1/manifest.json",
              **ident(ROOT/"artifacts/gate1/20261023_training_integration_review_v1/synthetic_output_audit_v1/manifest.json")},
          "ordinary_model_checkpoints_deserialized":False,"old_weights_or_real_data_used":False,"scientific_gate_pass":False}
require_new.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":result["status"],"files":len(files),"manifest":ident(require_new)}))
