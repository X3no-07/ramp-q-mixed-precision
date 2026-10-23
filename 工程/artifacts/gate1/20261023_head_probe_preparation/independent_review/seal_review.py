"""Freeze source review and synthetic tests; code copies are inspection evidence."""
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


bindings = {
    "scripts/probe_gate1_head_precision.py": "842e16b7b743c6044975ab2faad5725ead19cd8c793575340c9ee81ddaaef23f",
    "scripts/audit_gate1_head_probe.py": "cadac8b9c00ce4cb6c8063ceeb496ce224530a2c5c0fbdd988007d03995991a9",
    "tests/test_gate1_head_probe_audit.py": "e184515802e147b0b58b664b422aab7de9e568aecdC18d40bd626e364c2872c6".lower(),
    "scripts/audit_gate1_decode_probe.py": "04a7d2256ab0a3d90025b2453ef2bc44b043e8b1493b7866c9acc603838dc1ef",
    "tests/test_gate1_decode_probe_audit.py": "ddcb6e93e34d05c728021ce33e9d0e4ab9eab140444dea6acc7642eb95a08e56",
}
for relative, expected in bindings.items():
    source = ROOT / relative
    assert sha(source) == expected
    target = HERE / source.name
    assert not target.exists()
    shutil.copyfile(source, target)
write("review.json", {
    "schema": "rampq.gate1.head_precision_source_review/v1",
    "status": "NO_BLOCKING_SOURCE_CORRECTNESS_BUG_IDENTIFIED",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "timing": "Source reviewed while parent prepared/executed the head pilot; final independent auditor/tests sealed after pilot completed and while full109 ran. This is not described as a pre-pilot registration.",
    "source_bindings": bindings,
    "synthetic_tests": {"count": 15, "passed": 15, "elapsed_seconds": 1.606},
    "review_findings": [
        "Head intervention crosses captured Detect input feature origin with original unfused receiving head dtype; it calls pinned forward_head rather than reimplementing convolutions.",
        "Receiving head parameters and all child modules are checked for dtype/eval status. BatchNorm statistics are not trained or recalibrated.",
        "Explicit feature casts are part of the head arithmetic intervention. Four crossed head outputs share FP32 DFL, box geometry and sigmoid decoding.",
        "Mandatory native head logits, original forward bytes and saved decoder-probe captures are exactness conditions, independent of whether diagnostic guards pass.",
        "Features and head state are compared against pre-intervention copies; per-scale feature SHA before/after is saved for independent array verification.",
        "The source adapter changes only schema and runner binding for reviewed decoder runtime helpers; original head protocol and runner remain independently SHA-bound and are preserved.",
        "The auditor independently recomputes all eight named guards, logit difference counts/means/maxima, summary totals, feature hashes, native/reference identities and four FP32 decode stage checks.",
        "reference_fp32_vs_fp32_features_fp16_head aliases same_fp32_features_head_fp32_vs_fp16; reference_fp32_vs_fp16_features_fp32_head aliases same_fp32_head_features_fp32_vs_fp16. These are not additional independent contrasts."
    ],
    "remaining_limits": [
        "Saved feature tensors and head logits do not independently prove convolution/BatchNorm execution. Auditor does not deserialize model weights or rerun head arithmetic.",
        "Feature before/after SHA values are independently checked against saved bytes; head-state immutability remains a reviewed-runner assertion.",
        "NumPy nonlinear decoder checks are approximate and cannot certify bitwise CUDA softmax or convolution.",
        "Feature-origin contrast covers upstream network arithmetic; head contrast covers complete box/class towers, including convolution, BatchNorm and activation, rather than an isolated operation.",
        "No NMS, target matching, official AP, Selection, Holdout, quantization, TensorRT, Orin, performance or scientific baseline acceptance."
    ],
    "original_guard_replaced": False, "tolerances_changed": False, "scientific_gate_pass": False,
    "selection_detector_inference": False, "holdout_accessed": False,
    "copy_note": "Copied scripts are byte-for-byte inspection evidence. Run original workspace entrypoints because imports depend on their original script layout."
})
write("manifest.json", {"schema": "rampq.gate1.head_precision_independent_review_manifest/v1", "files": {
    item.relative_to(HERE).as_posix(): {"bytes": item.stat().st_size, "sha256": sha(item)}
    for item in sorted(HERE.rglob("*")) if item.is_file() and item.name != "manifest.json"
}})
print(json.dumps({"review": str(HERE / "review.json"), "manifest_sha256": sha(HERE / "manifest.json")}))
