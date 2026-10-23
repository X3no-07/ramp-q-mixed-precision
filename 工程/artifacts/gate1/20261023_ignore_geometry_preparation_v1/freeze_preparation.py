"""Seal synthetic-only preparation after independent review; no real data I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def ident(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Pass the source-bound independent review JSON path")
    review_path = Path(sys.argv[1]).resolve()
    if not review_path.is_file() or not review_path.is_relative_to(ROOT):
        raise RuntimeError("Review must be a workspace JSON file")
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Preparation manifest already exists; never overwrite")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    # Root independently verifies review semantics; this seal additionally records
    # its exact identity and refuses known negative dispositions.
    if review.get("issues") != [] or not str(review.get("status", "")).upper().startswith("PASS"):
        raise RuntimeError("Independent review still has issues")
    source = json.loads((OUT / "source_review.json").read_text(encoding="utf-8"))
    for rel, expected in source["workspace_sources"].items():
        assert ident(ROOT / rel) == expected, rel
    for rel, expected in source["framework_sources"].items():
        expected_id = {k: expected[k] for k in ("bytes", "sha256")}
        assert ident(OUT / expected["snapshot"]) == expected_id, rel
        assert ident(Path(expected["original_path"])) == expected_id, rel
    receipt = json.loads((OUT / "final_tests_receipt.json").read_text(encoding="utf-8"))
    assert receipt["counts"] == {"tests": 85, "failures": 0, "errors": 0, "skipped": 0}
    assert ident(OUT / "final_tests.xml") == receipt["junit_xml"]
    assert ident(OUT / "final_tests.log") == receipt["console"]
    for rel, expected in receipt["final_sources"].items():
        assert ident(ROOT / rel) == expected, rel
    members = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and "runtime_config" not in path.relative_to(OUT).parts and path.name != "manifest.json":
            members[path.relative_to(OUT).as_posix()] = ident(path)
    obj = {"schema": "rampq.gate1.ignore_geometry_preparation_manifest/v1",
           "status": "SEALED_SYNTHETIC_PRIMITIVES_ONLY_REAL_TRAINING_NOT_INTEGRATED",
           "files": members,
           "workspace_sources": receipt["final_sources"],
           "independent_review": {"path": review_path.relative_to(ROOT).as_posix(), **ident(review_path)},
           "excluded_mutable_runtime_config": "runtime_config/",
           "real_data_or_weights_used": False, "training_pipeline_integrated": False,
           "official_AP_or_scientific_gate_or_deployment_pass": False}
    (OUT / "manifest.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": obj["status"], "files": len(members), "manifest": ident(OUT / "manifest.json")}))


if __name__ == "__main__":
    main()
