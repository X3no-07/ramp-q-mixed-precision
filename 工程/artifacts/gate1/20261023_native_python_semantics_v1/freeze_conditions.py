"""Freeze exact partial-comparison conditions before native intermediates are read."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
READINESS = ROOT / "artifacts/gate1/20261023_official_evaluator_readiness"


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    fixture_path = READINESS / "synthetic_fixtures.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    root = ET.parse(HERE / "adapter_synthetic_regression_v1.xml").getroot()
    suites = list(root.iter("testsuite"))
    regression = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    assert regression["failures"] == regression["errors"] == regression["skipped"] == 0
    conditions = {
        "schema": "rampq.gate1.native_python_partial_comparison_predeclared_conditions/v1",
        "client_date": "2026-10-23", "status": "FIXED_BEFORE_NATIVE_OUTPUT_COMPARISON",
        "fixture": identity(fixture_path), "case_ids": [row["id"] for row in fixture["cases"]],
        "all_cases_required": True, "prefilter_case_count": 20, "prefilter_image_count": sum(len(c["images"]) for c in fixture["cases"]),
        "matching_case_ids": [row["id"] for row in fixture["cases"] if row["kind"] == "match"],
        "matching_categories": list(range(1, 11)),
        "matrix_fields": ["filtered_gt", "filtered_predictions", "prepared_gt", "class_matches.gt_output", "class_matches.dt_output"],
        "absolute_tolerance": 0, "relative_tolerance": 0, "matrix_shape_and_row_order_compared": True,
        "no_native_arrays_used_as_adapter_inputs": True,
        "comparison_source": identity(HERE / "compare_partial_semantics.py"),
        "test_source": identity(HERE / "test_partial_semantics.py"),
        "test_result": identity(HERE / "adapter_synthetic_regression_v1.xml"), "regression": regression,
        "python_source": identity(ROOT / "scripts/gate1_ignore_aware_evaluation.py"),
        "native_structure_audit_pass_required_before_execution": True,
        "success_rule": "All 20 case prefilter/prepared matrices and 5 matching cases times 10 classes GT/DT output matrices compare exactly, including empty shapes, original duplicates, score-zero GT and declared row order",
        "failure_rule": "Save every differing matrix and retain mismatch status; do not edit existing matcher, choose a subset, relax tolerances, or repair native results",
        "AP_metrics_compared": False, "full_Python_AP_implemented": False,
        "project_IoU_first_matcher_called_or_modified": False,
        "official_AP_equivalence_certified": False, "gate1_scientific_pass": False,
        "real_data_models_Selection_Holdout_accessed": False,
    }
    with (HERE / "predeclared_conditions.json").open("x", encoding="utf-8") as stream:
        json.dump(conditions, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(identity(HERE / "predeclared_conditions.json")))


if __name__ == "__main__":
    main()
