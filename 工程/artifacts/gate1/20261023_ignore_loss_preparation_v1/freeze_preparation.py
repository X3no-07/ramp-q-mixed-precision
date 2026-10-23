"""Seal local synthetic evidence and source identities without real data access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def ident(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if (OUT/"manifest.json").exists() or (OUT/"final_tests_v2_receipt.json").exists():
        raise RuntimeError("Already sealed; do not overwrite")
    xml = ET.parse(OUT/"final_tests_v2.xml").getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.iter("testsuite"))
    total = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    assert total == 55 and failures == 0
    gap = json.loads((OUT/"source_gap_review.json").read_text(encoding="utf-8"))
    for relative, expected in gap["workspace_sources"].items():
        assert ident(ROOT/relative) == expected, relative
    for relative, source in gap["sources"].items():
        actual = ident(OUT/source["snapshot"])
        assert actual == {k: source[k] for k in ("bytes", "sha256")}, relative
        pin = source["historical_training_contract_sha256"]
        if pin is not None:
            assert actual["sha256"] == pin, relative
    assert gap["sources"]["utils/loss.py"]["historical_training_contract_sha256"] is None
    assert gap["sources"]["utils/tal.py"]["historical_training_contract_sha256"] is None
    final_sources = {name: ident(ROOT/name) for name in (
        "scripts/gate1_ignore_negative_loss.py", "tests/test_gate1_ignore_negative_loss.py")}
    save("final_tests_v2_receipt.json", {
        "schema": "rampq.gate1.ignore_loss_synthetic_test_receipt/v1",
        "status": "PASS_SYNTHETIC_PRIMITIVE_ONLY", "tests_passed": total,
        "failures_and_errors": failures, "observed_pytest_output": "55 passed in 8.62s",
        "junit_xml": ident(OUT/"final_tests_v2.xml"), "console": ident(OUT/"final_tests_v2_console.txt"),
        "final_sources": final_sources, "supported_dtypes": ["torch.float32", "torch.float64"],
        "low_precision_and_finite_input_overflow_rejected": True,
        "initial_44_tests_preserved": True, "real_data_or_weights_used": False,
        "gpu_used": False, "training_pipeline_integrated": False,
        "scientific_gate_or_official_AP_passed": False})
    files = {p.relative_to(OUT).as_posix(): ident(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    save("manifest.json", {
        "schema": "rampq.gate1.ignore_loss_preparation_manifest/v1",
        "status": "PASS_SOURCE_GAP_REVIEW_AND_SYNTHETIC_CLASSIFICATION_PRIMITIVE_ONLY",
        "files": files, "external_workspace_sources": final_sources,
        "complete_ignore_aware_training_solved": False,
        "training_pipeline_integrated": False, "original_scientific_gates_changed": False})
    sealed = json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    for name, expected in sealed["files"].items():
        assert ident(OUT/name) == expected, name
    print(json.dumps({"status": sealed["status"], "files_verified": len(files), "manifest": ident(OUT/"manifest.json"),
                      "final_sources": final_sources, "source_gap_review": ident(OUT/"source_gap_review.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
