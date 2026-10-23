"""Freeze the direct prototype source review and synthetic CPU test evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNNER_SHA = "7974e7ef5e671d644ad2c2ee7ec675debeed686d72935a5e8bb04adc3e968073"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


assert not (HERE / "manifest.json").exists(), "Review is already frozen"
sources = [ROOT / path for path in ("scripts/probe_gate1_body32_head16.py", "scripts/audit_gate1_body32_head16.py",
                                    "tests/test_gate1_body32_head16.py", "tests/test_gate1_body32_head16_audit.py", "scripts/audit_gate1_decode_probe.py")]
assert sha(sources[0]) == RUNNER_SHA, "Runner changed after source review"
before = {str(path): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sources}
command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_gate1_body32_head16*.py", "-v"]
result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, encoding="utf-8")
log = result.stdout + result.stderr
(HERE / "synthetic_test_results.txt").write_text(log, encoding="utf-8")
assert result.returncode == 0 and "\nOK\n" in log.replace("\r\n", "\n"), log
count = int(re.search(r"Ran (\d+) tests", log).group(1)); assert count == 30, count
assert before == {str(path): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sources}, "Source changed during tests"
for path in sources: shutil.copyfile(path, HERE / path.name)
write(HERE / "review.json", {
    "schema": "rampq.gate1.body32_head16_prototype_source_review/v1", "status": "PASS_BOUNDED_DIRECT_PROTOTYPE_REVIEW_ONLY",
    "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_identities": before,
    "synthetic_cpu_tests_passed": count, "test_command": command,
    "guard_name": "GATE1_BODY_FP32_HEAD_TOWERS_FP16_DECODE_FP32_DEBUG_NUMERIC_CHECK",
    "reviewed_checks": [
        "Exactly one checkpoint deserialization creates the candidate. The fresh image is independently decoded, letterboxed, normalized and sent into a direct mixed forward before loading reference NPZ array values.",
        "The original body DAG modules 0..22 execute once each with saved dependencies. Parameters, floating buffers and all observed body output tensors must be FP32.",
        "Only original Detect cv2/cv3 towers and their floating buffers are FP16; fresh FP32 Detect input features are explicitly cast to FP16. Their box/class output tensors must be FP16.",
        "Original head.forward and its implicit FP16 decoder are never called. Geometry is made from fresh FP32 body features and FP32 stride. Frozen functional DFL, geometry and sigmoid use explicit FP32, with every returned stage dtype checked.",
        "The mixed final outputs/logits must be byte-identical to the earlier source-FP32/head-FP16 endpoint; input and feature/cast hashes must match independently saved references after the fresh computation.",
        "The independently named all-anchor prototype guard retains box rtol .005/atol .5 and score rtol .02/atol .002, relabels geometry branches correctly, and cannot replace historical failed guards.",
        "The NumPy auditor recomputes full-array fixed guards, exact endpoint/reference bytes, cast statistics, geometry, scope/load/forward reports and approximate FP64 decoding sanity. Original full-model FP16 failure is separately recomputed.",
        "No NMS, target matching, label parsing, Selection, Holdout, AP, quantization, TensorRT, Orin, latency or energy computation is included.",
    ],
    "issues_repaired_before_freeze": ["Added explicit floating buffer dtype checks for body, head towers and DFL, plus actual parameter/buffer dtype count provenance.", "Gave the prototype numerical check a distinct name and geometry branch labels.", "Aligned independent receipt/summary/runtime precision and fresh-forward scope metadata checks."],
    "conditional_execution": "Only fixed pilot3 first; expand to fixed full109 only after all fresh-forward identity controls and independent saved-output audit pass. Never alter tolerances or choose a checkpoint from the new outputs.",
    "limits": ["Synthetic CPU tests and reviewed assertions do not prove real GPU internal arithmetic precision.", "Auditor does not load the model or regenerate convolutions; runtime body/head parameter and output precision assertions retain reviewed-runner limits.", "Fresh features are deduplicated into hashes/statistics; their source/reference and FP16 casts are independently compared against earlier saved tensors.", "Reproducing an earlier engineering endpoint is implementation consistency evidence, not an independent effect or generalization experiment.", "No scientific Gate1, all-FP16 deployment, original pre-save FP32 state recovery, accuracy or performance acceptance follows."],
    "scientific_gate_pass": False, "accepted_float_baseline_ready": False, "old_guard_replaced": False,
    "selection_detector_inference": False, "holdout_accessed": False})
write(HERE / "manifest.json", {"files": {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(HERE.iterdir()) if path.is_file() and path.name != "manifest.json"}})
print(json.dumps({"tests": count, "sources": before, "review_manifest_sha256": sha(HERE / "manifest.json")}, ensure_ascii=False, indent=2))
