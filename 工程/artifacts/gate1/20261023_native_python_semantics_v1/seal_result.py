"""Seal partial synthetic comparison evidence without broad equivalence claims."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    result_path = HERE / "comparison_result_v1.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["matrix_checks"] == result["equal_matrix_checks"] == 166
    assert result["different_matrix_checks"] == 0
    summary = {"schema": "rampq.gate1.native_python_partial_comparison_decision/v1",
               "client_date": "2026-10-23", "status": result["status"], "data_role": "synthetic",
               "native_runtime": "GNU Octave 11.3.0", "official_source_commit": "005445782213e20cb91bc50a597db3dd949e749a",
               "prefilter_cases": 20, "prefilter_images": 22, "prefilter_prepared_matrix_checks": 66,
               "matching_cases": 5, "categories_each": 10, "matching_GT_DT_matrix_checks": 100,
               "total_matrix_checks": 166, "strictly_equal": 166, "differences": 0,
               "absolute_tolerance": 0, "relative_tolerance": 0,
               "result": identity(result_path),
               "interpretation": "The existing Python region prefilter and separately named official-style single-image matcher matched the pinned native source outputs for all compared matrices on these 20 fixed synthetic fixtures only.",
               "not_established": ["A complete Python AP/AR implementation or equivalence", "Arbitrary-input or real-data evaluator equivalence", "Cross-runtime or cross-version tie equivalence", "Project IoU-first matcher equivalence", "Real-data model accuracy or Gate1 scientific acceptance"],
               "full_Python_AP_implemented": False, "official_AP_equivalence_certified": False,
               "project_IoU_first_matcher_called_or_modified": False, "model_executed": False,
               "real_data_Selection_Holdout_accessed": False, "gate1_scientific_pass": False}
    with (HERE / "decision.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    names = ["compare_partial_semantics.py", "test_partial_semantics.py", "freeze_conditions.py",
             "bind_execution_contract.py", "seal_result.py", "adapter_synthetic_regression_v1.xml",
             "predeclared_conditions.json", "execution_contract_v1.json", "comparison_result_v1.json", "decision.json"]
    manifest = {"schema": "rampq.gate1.native_python_partial_semantics_manifest/v1", "path_base": "workspace",
                "file_count": len(names), "files": [identity(HERE / name) for name in names],
                "source_inputs_bound_transitively_via_execution_contract": True,
                "no_native_AP_values_fabricated_or_compared_against_Python": True}
    with (HERE / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"manifest": identity(HERE / "manifest.json"), "decision": identity(HERE / "decision.json"), "result": identity(result_path)}))


if __name__ == "__main__":
    main()
