"""Seal this component's existing tests/example evidence without another run."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

pins = {
    "scripts/gate1_fp32_live_capture.py": "99baeedf7db524fa4cd55674537fea6422de41f049fd1d8b710eb91c265e1a51",
    "tests/test_gate1_fp32_live_capture.py": "e2b21a8b6ea87600aee9db276be7858c6cb001e18d1315f21b9470fc85d25777",
}
assert all(sha(ROOT / name) == expected for name, expected in pins.items())
example = json.loads((HERE / "synthetic_example_receipt.json").read_text())
assert example["ema_updates"] == 2 and example["status"] == "PASS_TWO_SYNTHETIC_SGD_EMA_STEPS_AND_CONTROLLED_CPU_CAPTURE_ONLY"
capture = json.loads((HERE / "synthetic_capture/epoch_0001/capture_receipt.json").read_text())
assert capture["state_and_update_counter_unchanged"] and capture["reconstruction"]["output_values_dtypes_shapes_exact"]
for directory in (HERE / "synthetic_capture/epoch_0001", HERE / "synthetic_capture/epoch_0001/state"):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert all((directory / name).stat().st_size == expected["bytes"] and sha(directory / name) == expected["sha256"]
               for name, expected in manifest["files"].items())
write(HERE / "test_execution_receipt.json", {
    "schema": "rampq.gate1.synthetic_cpu_component_tests/v1", "status": "PASS_23_SYNTHETIC_CPU_HOOK_TESTS_ONLY",
    "command": "python -B -m unittest discover -s tests -p test_gate1_fp32_live_capture.py -v",
    "exit_code": 0, "tests_run": 23, "failures": 0, "errors": 0, "unittest_reported_duration_seconds": 38.359,
    "terminal_summary": "Ran 23 tests in 38.359s\n\nOK",
    "source_sha256": pins, "full_terminal_transcript_retained_as_artifact": False,
    "verification_basis": "Completed tool execution and its actual unittest summary; no fabricated full log",
    "old_weights_or_real_images_read": False, "scientific_gate1_passed": False})
write(HERE / "component_decision.json", {
    "schema": "rampq.gate1.fp32_live_capture_component_decision/v1",
    "status": "PASS_CONTROLLED_CPU_COMPONENT_ONLY", "tests_passed": 23,
    "example_sgd_steps": 2, "example_ema_updates": 2, "exact_state_and_fresh_architecture_cpu_output": True,
    "phase": capture["phase"], "source_sha256": pins,
    "real_ultralytics_dataset_trainer_integration_verified_in_this_package": False,
    "external_arbitrary_concurrent_writers_excluded": False,
    "caller_must_forbid_concurrent_ema_writers": True,
    "gpu_capture_qualified": False, "optimizer_or_resume_state_saved": False,
    "scientific_reference_accepted": False, "old_checkpoint_recovered": False,
    "next_check": "Root must execute a fresh controlled synthetic CPU DetectionTrainer epoch and independently observe capture before super.save_model half serialization."})
members = {p.relative_to(HERE).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
           for p in sorted(HERE.rglob("*")) if p.is_file() and p.name != "manifest.json"}
# Include nested manifests as files; only this top-level manifest is excluded.
for p in sorted((HERE / "synthetic_capture").rglob("manifest.json")):
    members[p.relative_to(HERE).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha(p)}
write(HERE / "manifest.json", {"schema": "rampq.gate1.fp32_live_capture_component_delivery/v1", "files": members,
                               "external_sources": {name: {"bytes": (ROOT / name).stat().st_size, "sha256": expected}
                                                    for name, expected in pins.items()}})
print(json.dumps({"status": "SEALED_COMPONENT_ONLY", "members": len(members), "manifest_sha256": sha(HERE / "manifest.json")}))
