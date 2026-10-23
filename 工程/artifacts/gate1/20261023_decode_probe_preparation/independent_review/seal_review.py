"""Freeze the independently authored decode-probe review and synthetic tests."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


runner = ROOT / "scripts/probe_gate1_decode_precision.py"
auditor = ROOT / "scripts/audit_gate1_decode_probe.py"
tests = ROOT / "tests/test_gate1_decode_probe_audit.py"
assert sha(runner) == "f76f3bfd677487fa7d87947f866daaf7407ddd455ad3e4d9e399da719e481ec0"
assert sha(auditor) == "04a7d2256ab0a3d90025b2453ef2bc44b043e8b1493b7866c9acc603838dc1ef"
assert sha(tests) == "ddcb6e93e34d05c728021ce33e9d0e4ab9eab140444dea6acc7642eb95a08e56"
for source in (runner, auditor, tests):
    destination = HERE / source.name
    assert not destination.exists()
    shutil.copyfile(source, destination)
write("review.json", {
    "schema": "rampq.gate1.decode_precision_source_review/v1",
    "status": "NO_BLOCKING_SOURCE_CORRECTNESS_BUG_IDENTIFIED",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "timing": "Source review and tests were performed while parent prepared/executed the three-image pilot; this sealed record is not represented as a pre-pilot registration.",
    "runner_sha256": sha(runner), "auditor_sha256": sha(auditor), "synthetic_tests_sha256": sha(tests),
    "synthetic_tests": {"count": 22, "passed": 22, "elapsed_seconds": 1.226},
    "framework_source_observations": [
        {"path": "nn/modules/head.py", "lines": "168-216,232-239", "finding": "Pinned forward emits decoded y and dict containing box logits, class logits, feats; decoder uses saved runtime anchor/stride cache, DFL, xywh and stride multiplication."},
        {"path": "nn/modules/block.py", "lines": "60-81", "finding": "DFL views Bx4xreg_maxxN, transposes to Bxreg_maxx4xN, softmax along bin dimension, applies fixed-bin 1x1 convolution and views Bx4xN."},
        {"path": "utils/tal.py", "lines": "424-450", "finding": "Anchors/strides inherit feature dtype; corner subtraction/addition, midpoint and width-height subtraction must preserve order to retain native FP16 rounding."}
    ],
    "review_findings": [
        "Independent checkpoint loads and float-lifted model-state equality prevent differing weights from masquerading as dtype intervention.",
        "Captured logits and runtime geometry are reused; each requested decode dtype casts logits, anchors, strides and bin weights together.",
        "Functional DFL convolution and dist2bbox operation order match pinned framework code; native exact reconstruction catches overlooked operation or cache differences.",
        "Mandatory historical raw-byte equality and normalized-input/uint8 hashes bind execution to the already frozen engineering debug cohort.",
        "Pilot-to-full expansion depends on identity/native-reconstruction validity, not favorable allclose results. No posthoc tolerance adjustment.",
        "The independent saved-array auditor recomputes all five comparisons at unchanged tolerances and checks actual bin values, anchors, strides, exact elementary decode stages, approximate nonlinear equations, native/historical bytes, manifest identities and summary totals."
    ],
    "remaining_limits": [
        "Native torch.equal reconstruction alone is value equality; saved-array auditor strengthens it to dtype/shape/C-order byte equality.",
        "NumPy nonlinear arithmetic is approximate and cannot certify bitwise CUDA softmax or convolution; the audit clearly separates exact and approximate checks.",
        "No model, pixel decoding, GPU, Selection, Holdout, NMS, official AP or target scoring is performed by the independent review/auditor.",
        "No operation-level cause is established from a source review, and any diagnostic comparison PASS cannot accept the original failed guard or Gate1."
    ],
    "scientific_gate_pass": False, "old_guard_replaced": False, "tolerances_changed": False,
    "selection_detector_inference": False, "holdout_accessed": False
})
write("manifest.json", {"schema": "rampq.gate1.decode_precision_independent_review_manifest/v1", "files": {
    item.relative_to(HERE).as_posix(): {"bytes": item.stat().st_size, "sha256": sha(item)}
    for item in sorted(HERE.rglob("*")) if item.is_file() and item.name != "manifest.json"
}})
print(json.dumps({"review": str(HERE / "review.json"), "manifest_sha256": sha(HERE / "manifest.json")}))
