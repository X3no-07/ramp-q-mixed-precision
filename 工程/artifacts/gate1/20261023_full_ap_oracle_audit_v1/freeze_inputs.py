"""Freeze approved synthetic membership, tolerance policy and native wrapper inputs."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_new(name, obj):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    suites = []
    for identifier, path, native_schema in (
        ("frozen20", ROOT / "artifacts/gate1/20261023_official_evaluator_readiness/synthetic_fixtures.json", "rampq.gate1.synthetic_official_oracle_execution/v1"),
        ("stress12", HERE / "stress_fixtures.json", "rampq.gate1.synthetic_official_stress_oracle_execution/v1"),
    ):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        cases = fixture["cases"]
        image_count = sum(len(case["images"]) for case in cases)
        match_images = sum(len(case["images"]) for case in cases if case["kind"] == "match")
        aggregate_count = sum(case["kind"] == "aggregate" for case in cases)
        suites.append({"id": identifier, "fixture": identity(path), "case_count": len(cases), "case_ids": [case["id"] for case in cases],
                       "kinds": dict(Counter(case["kind"] for case in cases)), "image_count": image_count,
                       "matrix_count": image_count * 3 + match_images * 10 * 2, "metric_count": aggregate_count * 7,
                       "native_schema": native_schema, "native_runtime": "11.3.0"})
    assert [(s["case_count"], s["matrix_count"], s["metric_count"]) for s in suites] == [(20, 166, 77), (12, 54, 84)]
    policy = {"schema": "rampq.gate1.full_ap_synthetic_comparison_policy/v1", "data_role": "synthetic",
              "client_date": "2026-10-23", "suites": suites,
              "finite_metric_tolerance": {"atol": 1e-10, "rtol": 1e-12, "reference": "native", "unit": "percentage"},
              "matrix_tolerance": {"atol": 0, "rtol": 0, "shape_exact": True, "dtype": "float64"},
              "nan_pattern_exact": True, "infinity_allowed": False,
              "roundoff_rationale": "Binary64 reductions may differ by summation path; allowance applies only to final synthetic AP/AR percentage scalars",
              "native_internal_tables_compared": False, "arbitrary_input_equivalence_certified": False,
              "original_dtype_guard_modified": False, "gate1_scientific_pass": False,
              "implementation_frozen_before_comparison": True, "handwritten_metric_answers_used": False,
              "model_executed": False, "real_data_read": False, "selection_or_holdout_accessed": False}
    write_new("comparison_policy.json", policy)
    members = [HERE / name for name in ("AUDIT_DESIGN.md", "generate_stress_fixtures.py", "stress_fixtures.json", "run_full_ap_stress_oracle.m", "freeze_inputs.py", "comparison_policy.json")]
    write_new("inputs_manifest.json", {"schema": "rampq.gate1.full_ap_synthetic_inputs_manifest/v1", "path_base": "workspace",
                                      "status": "FROZEN_BEFORE_STRESS_NATIVE_OR_PYTHON_COMPARISON", "files": [identity(path) for path in members],
                                      "file_count": len(members), "old_source_or_fixture_modified": False})
    print("INPUTS_MANIFEST_SHA256", sha(HERE / "inputs_manifest.json"))
    print("POLICY_SHA256", sha(HERE / "comparison_policy.json"))
    print("STRESS_FIXTURE_SHA256", sha(HERE / "stress_fixtures.json"))
    print("WRAPPER_SHA256", sha(HERE / "run_full_ap_stress_oracle.m"))


if __name__ == "__main__":
    main()
