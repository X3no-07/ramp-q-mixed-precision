"""Freeze independently reviewed frontier sources and synthetic CPU test evidence."""
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
RUNNER_SHA = "f5ffa1741f1c4bd97e8c3b36372489f28e3704965ade951a1dc38ce55d347ba8"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


assert not (HERE / "manifest.json").exists(), "Review already frozen"
sources = [ROOT / p for p in ("scripts/probe_gate1_upstream_frontier.py", "scripts/audit_gate1_upstream_frontier.py",
                              "tests/test_gate1_upstream_frontier.py", "tests/test_gate1_upstream_frontier_audit.py",
                              "scripts/audit_gate1_decode_probe.py")]
assert sha(sources[0]) == RUNNER_SHA, "Runner changed after review"
before = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in sources}
command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_gate1_upstream_frontier*.py", "-v"]
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
log = result.stdout + result.stderr
(HERE / "synthetic_test_results.txt").write_text(log, encoding="utf-8")
assert result.returncode == 0 and "\nOK\n" in log.replace("\r\n", "\n"), log
count = int(re.search(r"Ran (\d+) tests", log).group(1))
assert count == 39, count
assert before == {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in sources}, "Source changed during tests"
for source in sources:
    shutil.copyfile(source, HERE / source.name)
write(HERE / "review.json", {
    "schema": "rampq.gate1.upstream_frontier_source_review/v1", "status": "PASS_BOUNDED_DIAGNOSTIC_IMPLEMENTATION_REVIEW_ONLY",
    "created_at_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "independent upstream frontier reviewer",
    "source_identities": before, "synthetic_cpu_tests_passed": count, "test_command": command,
    "cuts": [-1, 4, 6, 10, 22], "derived_frontiers": {"-1": [-1], "4": [4], "6": [4, 6], "10": [4, 6, 10], "22": [16, 19, 22]},
    "stored_crossed_variants": 10, "fixed_guard_comparisons": 12,
    "reviewed_checks": [
        "DAG f references resolve independently; every skip dependency crossing a cut is retained.",
        "Frozen CPU topology is required to equal actual loaded module graph; all 23 native prefix modules and final outputs must replay exactly before interpretation.",
        "Native full manual output arrays are stored. Per-cut same-dtype suffix output controls are deduplicated into exact assertions and byte hashes bound to stored native values.",
        "Only union native frontier tensors and ten crossed logits/decoded outputs are stored; cast reports are independently reconstructed with NumPy source.astype(receiver_dtype).",
        "Input-cut FP32-to-FP16 and cut22 head endpoints must exactly alias prior native/head outputs and cannot count as independent new evidence.",
        "Every historical/reference raw identity, fixed all-anchor guard and aggregate is independently rechecked from saved arrays.",
        "Independent NumPy FP64 decode sanity has predeclared box allowance .002+1e-5*abs(expected) and score allowance 2e-7+3e-6*abs(expected); these are distinct from unchanged fixed FP32/FP16 guards.",
        "Runner preserves debug-only cohort, fixed pilot order, no automatic retries, source/checkpoint/runtime SHA bindings and no Selection/Holdout/NMS/AP access.",
    ],
    "issues_repaired_before_freeze": ["Missing nl metadata was added.", "All-native-module exactness was expanded from final-output equality to every captured prefix module.", "Storage was reduced to two crossed directions and cast hashes without repeated decode intermediate arrays.", "Repeated input/head endpoints now require explicit byte-identical alias checks."],
    "conditional_execution": "Only fixed pilot3 first; full109 only after exact reconstruction controls and saved-array audit succeed, regardless of diagnostic guard outcome.",
    "limits": ["Synthetic tests and source review do not establish real GPU execution equivalence.", "Auditor independently verifies stored bytes, casts, guards and approximate decoding; it does not regenerate convolution outputs or establish a unique layer cause.", "Unstored prefix tensor checks, model-state immutability and deduplicated same-dtype replay histories retain reviewed-runner assertion limits.", "No old numerical guard is replaced or relaxed; Gate1 scientific acceptance remains false."],
    "scientific_gate_pass": False, "old_guard_replaced": False, "tolerances_changed": False,
    "selection_detector_inference": False, "holdout_accessed": False})
write(HERE / "manifest.json", {"files": {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(HERE.iterdir()) if path.is_file() and path.name != "manifest.json"}})
print(json.dumps({"tests": count, "sources": before, "review_manifest_sha256": sha(HERE / "manifest.json")}, ensure_ascii=False, indent=2))
