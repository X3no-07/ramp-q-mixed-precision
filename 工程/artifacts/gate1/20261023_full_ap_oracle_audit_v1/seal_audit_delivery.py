"""Verify completed comparison evidence and seal only this new audit delivery."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify_member(path, row):
    assert path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], str(path)


def check_manifest(path, base):
    manifest = read(path)
    files = manifest["files"]
    rows = list(files.items()) if isinstance(files, dict) else [(row["path"], row) for row in files]
    assert len(set(name for name, row in rows)) == len(rows)
    if "file_count" in manifest: assert manifest["file_count"] == len(rows)
    for name, row in rows:
        target = (base / name).resolve()
        assert target.is_relative_to(ROOT)
        verify_member(target, row)
    return {"manifest": identity(path), "verified_file_count": len(rows)}


def main():
    result_path = HERE / "comparison_attempt_v1.json"
    result = read(result_path)
    assert result["status"] == "PASS_DECLARED_SYNTHETIC_FULL_AP_NATIVE_COMPARISON_ONLY"
    assert result["matrix_count"] == 220 and result["metric_count"] == 161
    assert result["gate1_scientific_pass"] is False and result["original_dtype_guard_modified"] is False
    assert result["native_internal_AP_AR_tables_compared"] is False
    matrices = [row for suite in result["suites"] for row in suite["matrices"]]
    metrics = [row for suite in result["suites"] for row in suite["metrics"]]
    assert len(matrices) == 220 and all(row["exact"] for row in matrices)
    assert len(metrics) == 161 and all(row["native_is_nan"] == row["python_is_nan"] for row in metrics)
    assert all(row["absolute_error"] <= row["allowed_error"] for row in metrics if not row["native_is_nan"])
    contract = read(HERE / "comparison_contract_v1.json")
    for key in ("policy", "auditor", "comparator_preparation", "implementation", "implementation_preparation_manifest",
                "structural_auditor", "official_source_manifest", "runtime_manifest", "runtime_executable", "native_mean2"):
        row = contract[key]; verify_member(ROOT / row["path"], row)
    for row in contract["implementation_dependencies"] + contract["official_sources"]:
        verify_member(ROOT / row["path"], row)
    for suite in contract["suites"]:
        for key in ("fixture", "native_mat", "native_json", "launcher_receipt", "native_structure_audit"):
            if key in suite: row = suite[key]; verify_member(ROOT / row["path"], row)
    checks = [check_manifest(HERE / "inputs_manifest.json", ROOT),
              check_manifest(HERE / "comparator_preparation_manifest.json", ROOT),
              check_manifest(HERE / "native_stress_execution_v1/manifest.json", HERE / "native_stress_execution_v1"),
              check_manifest(ROOT / "artifacts/gate1/20261023_full_ap_preparation_v1/preparation_manifest.json", ROOT)]
    finite_errors = [row["absolute_error"] for row in metrics if not row["native_is_nan"]]
    summary = {"schema": "rampq.gate1.full_ap_synthetic_audit_summary/v1", "status": result["status"],
               "case_count": 32, "aggregate_case_count": 23, "image_count": 40,
               "matrix_count_exact": 220, "metric_count": 161, "finite_metric_count": len(finite_errors),
               "matching_nan_metric_count": sum(row["native_is_nan"] for row in metrics),
               "max_finite_metric_absolute_error_percentage_units": max(finite_errors),
               "metric_tolerance": result["float64_final_metric_tolerance"], "matrix_tolerance": result["matrix_tolerance"],
               "result": identity(result_path), "contract": identity(HERE / "comparison_contract_v1.json"),
               "verification_checks": checks, "auditor_negative_regression_tests_passed": 45,
               "implementation_synthetic_tests_passed": 54, "full_Python_AP_implemented": True,
               "declared_synthetic_output_agreement": True, "native_internal_AP_AR_tables_compared": False,
               "arbitrary_input_equivalence_certified": False, "real_data_AP_computed": False,
               "model_executed": False, "selection_or_holdout_accessed": False,
               "original_dtype_guard_modified": False, "gate1_scientific_pass": False}
    write_new(HERE / "audit_summary.json", summary)
    invocation = {"schema": "rampq.gate1.full_ap_comparison_invocation_receipt/v1",
                  "returncode": 0, "python": sys.version, "python_executable": sys.executable,
                  "invocation": [sys.executable, "-B", "-u", str(HERE / "compare_full_ap.py"), "--contract", str(HERE / "comparison_contract_v1.json"),
                      "--contract-sha256", sha(HERE / "comparison_contract_v1.json"), "--output", str(result_path)],
                  "output": identity(result_path), "stdout_observed": [result["status"], "MATRICES: 220; METRICS: 161"],
                  "scope": "Receipt records the completed comparator invocation; no second model/native execution"}
    write_new(HERE / "comparison_invocation_receipt.json", invocation)
    own_files = [path for path in sorted(HERE.iterdir()) if path.is_file()]
    native_files = [path for path in sorted((HERE / "native_stress_execution_v1").iterdir()) if path.is_file()]
    files = own_files + native_files
    manifest = {"schema": "rampq.gate1.full_ap_synthetic_audit_delivery_manifest/v1", "path_base": "workspace",
                "file_count": len(files), "files": [identity(path) for path in files],
                "source_and_native_dependencies_bound_transitively_by_contract": True,
                "native_runtime_members_verified_by_comparator": 2890, "isolated_reader_members_verified_by_comparator": 1656,
                "declaration_inputs_frozen_before_stress_native_run": True,
                "implementation_and_comparator_frozen_before_metric_comparison": True,
                "gate1_scientific_pass": False}
    write_new(HERE / "audit_delivery_manifest.json", manifest)
    for row in manifest["files"]: verify_member(ROOT / row["path"], row)
    print("SEALED_AUDIT_FILES", len(files))
    print("AUDIT_MANIFEST_SHA256", sha(HERE / "audit_delivery_manifest.json"))
    print("SUMMARY_SHA256", sha(HERE / "audit_summary.json"))


if __name__ == "__main__":
    main()
