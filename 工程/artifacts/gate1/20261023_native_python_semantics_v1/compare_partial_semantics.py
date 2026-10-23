"""Compare fixed synthetic native intermediates to existing partial Python semantics.

This is not a Python AP implementation. Mismatches are recorded, never repaired.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREPARATION = ROOT / "artifacts/gate1/20261023_native_oracle_comparison_preparation_v1"
READINESS = ROOT / "artifacts/gate1/20261023_official_evaluator_readiness"
PYTHON_SOURCE = ROOT / "scripts/gate1_ignore_aware_evaluation.py"
PYTHON_SOURCE_SHA = "4e45ff26a86a83ff66e997f15e4c4794246c6440c304446972809ff32db09961"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    imported = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(imported)
    return imported


def input_records(image, image_id):
    gt, dt = [], []
    for index, row in enumerate(image["raw_ground_truth"]):
        gt.append({"target_id": f"{image_id}:gt:{index}", "physical_line_zero_based": index,
                   "bbox_xywh": list(row[:4]), "score": row[4], "category": int(row[5]),
                   "truncation": int(row[6]), "occlusion": int(row[7])})
    for index, row in enumerate(image["raw_predictions"]):
        dt.append({"detection_id": f"{image_id}:dt:{index}", "physical_line_zero_based": index,
                   "bbox_xywh": list(row[:4]), "score": row[4], "category": int(row[5])})
    return gt, dt


def reconstruct_rows(filtered, raw_gt, raw_dt):
    gt_map = {row["target_id"]: source for row, source in zip(raw_gt[0], raw_gt[1])}
    dt_map = {row["detection_id"]: source for row, source in zip(raw_dt[0], raw_dt[1])}
    gt = np.asarray([gt_map[row["target_id"]] for row in filtered["ground_truth"]], dtype=np.float64).reshape((-1, 8))
    dt = np.asarray([dt_map[row["detection_id"]] for row in filtered["predictions"]], dtype=np.float64).reshape((-1, 8))
    prepared = gt.copy()
    prepared[gt[:, 4] == 0, 4] = 1
    prepared[gt[:, 4] == 1, 4] = 0
    return gt, dt, prepared


def python_intermediates(evaluation, case, image, image_id):
    """Adapt declared Python results; native outputs are never adapter inputs."""
    gt_records, dt_records = input_records(image, image_id)
    width, height = image["width"], image["height"]
    filtered = evaluation.filter_ignored_regions(gt_records, dt_records, width, height)
    gt, dt, prepared = reconstruct_rows(filtered, (gt_records, image["raw_ground_truth"]),
                                         (dt_records, image["raw_predictions"]))
    result = {"filtered_gt": gt, "filtered_predictions": dt, "prepared_gt": prepared}
    if case["kind"] == "match":
        official_style = evaluation.match_official_style(gt_records, dt_records, width, height,
                                                         iou_threshold=case["threshold"], max_detections=500)
        require(len(filtered["predictions"]) <= 500, "Fixed matching fixture unexpectedly exceeds maxDets=500")
        matched_targets = {row["target_id"] for row in official_style["matches"]}
        dt_by_id = {row["detection_id"]: row for row in filtered["predictions"]}
        gt_by_category = {}
        for category in range(1, 11):
            original = [row for row in filtered["ground_truth"] if row["category"] == category]
            ordered = sorted(original, key=lambda row: row["score"] == 0)
            rows = [list(row["bbox_xywh"]) + [-1 if row["score"] == 0 else 1 if row["target_id"] in matched_targets else 0]
                    for row in ordered]
            gt_by_category[category] = np.asarray(rows, dtype=np.float64).reshape((-1, 5))
        matches = []
        for category in range(1, 11):
            states = [state for state in official_style["detection_states"]
                      if dt_by_id[state["detection_id"]]["category"] == category]
            rows = []
            for state in states:
                row = dt_by_id[state["detection_id"]]
                flag = {"true_positive": 1, "ignored": -1, "false_positive": 0}[state["state"]]
                rows.append(list(row["bbox_xywh"]) + [row["score"], flag])
            matches.append({"category": category, "gt_output": gt_by_category[category],
                            "dt_output": np.asarray(rows, dtype=np.float64).reshape((-1, 6))})
        result["class_matches"] = matches
    return result


def compare_array(native, python, label):
    equal = native.shape == python.shape and np.array_equal(native, python)
    record = {"field": label, "native_shape": list(native.shape), "python_shape": list(python.shape),
              "equal": bool(equal), "atol": 0, "rtol": 0}
    if not equal:
        record.update(native_values=native.tolist(), python_values=python.tolist())
        if native.shape == python.shape:
            indices = np.argwhere(native != python)
            record["differing_elements"] = int(len(indices))
            record["first_difference_index"] = indices[0].tolist() if len(indices) else None
    return record


def compare_loaded(native_results, fixture, audit_helpers, evaluation):
    native_cases = audit_helpers.cell(native_results, len(fixture["cases"]), "results")
    checks = []
    for case, native_raw in zip(fixture["cases"], native_cases):
        native_case = audit_helpers.struct(native_raw, case["id"])
        require(audit_helpers.text(native_case["id"], "case id") == case["id"], "Case ordering mismatch")
        native_images = audit_helpers.cell(native_case["intermediates"], len(case["images"]), "intermediates")
        for image_index, (image, raw_native) in enumerate(zip(case["images"], native_images)):
            label = f"{case['id']}.image{image_index}"
            python = python_intermediates(evaluation, case, image, label)
            native = audit_helpers.struct(raw_native, label)
            for field in ("filtered_gt", "filtered_predictions", "prepared_gt"):
                value = audit_helpers.matrix(native[field], 8, label + "." + field)
                checks.append(compare_array(value, python[field], label + "." + field))
            if case["kind"] == "match":
                native_categories = audit_helpers.cell(native["class_matches"], 10, label + ".class_matches")
                for category, raw_category in enumerate(native_categories, 1):
                    native_category = audit_helpers.struct(raw_category, label)
                    require(audit_helpers.scalar(native_category["category"], "category") == category, "Class ordering mismatch")
                    for field, columns in (("gt_output", 5), ("dt_output", 6)):
                        value = audit_helpers.matrix(native_category[field], columns, label + "." + field)
                        checks.append(compare_array(value, python["class_matches"][category - 1][field],
                                                    label + f".category{category}." + field))
    differences = [row for row in checks if not row["equal"]]
    return {"matrix_checks": len(checks), "equal_matrix_checks": len(checks) - len(differences),
            "different_matrix_checks": len(differences), "checks": checks,
            "status": "MATCH_FIXED_SYNTHETIC_PARTIAL_INTERMEDIATES_ONLY" if not differences else "MISMATCH_FIXED_SYNTHETIC_PARTIAL_INTERMEDIATES"}


def compare(contract_path, contract_sha, output):
    contract_path = Path(contract_path).resolve()
    require(sha(contract_path) == contract_sha, "Comparison contract identity mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract["schema"] == "rampq.gate1.native_python_partial_comparison_contract/v1", "Wrong comparison contract")
    require(contract["absolute_tolerance"] == contract["relative_tolerance"] == 0, "Comparison tolerances changed")
    require(contract["comparison_source"]["sha256"] == sha(Path(__file__)), "Comparison source changed after contract")
    for name in ("python_source", "fixture", "native_structure_audit", "native_mat", "auditor_source"):
        row = contract[name]
        path = ROOT / row["path"]
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Comparison input changed: " + name)
    require(contract["python_source"]["sha256"] == PYTHON_SOURCE_SHA, "Existing Python source identity changed")
    structure = json.loads((ROOT / contract["native_structure_audit"]["path"]).read_text(encoding="utf-8"))
    require(structure["status"] == "PASS_NATIVE_SYNTHETIC_OUTPUT_STRUCTURE_IDENTITIES_ONLY", "Native structure audit did not pass")
    fixture = json.loads((ROOT / contract["fixture"]["path"]).read_text(encoding="utf-8"))
    require(contract["case_ids"] == [case["id"] for case in fixture["cases"]], "Fixed comparison case order changed")
    require(contract["matrix_fields"] == ["filtered_gt", "filtered_predictions", "prepared_gt", "class_matches.gt_output", "class_matches.dt_output"], "Fixed compared fields changed")
    helpers = module(ROOT / contract["auditor_source"]["path"], "frozen_native_structure_audit")
    reader_path = PREPARATION / "mat_reader_dependencies"
    helpers.verify_reader_dependency(reader_path)
    sys.path.insert(0, str(reader_path))
    from scipy.io import loadmat
    evaluation = module(ROOT / contract["python_source"]["path"], "existing_partial_evaluation")
    native = loadmat(ROOT / contract["native_mat"]["path"], squeeze_me=False, struct_as_record=False,
                     chars_as_strings=True, mat_dtype=True, verify_compressed_data_integrity=True)
    result = compare_loaded(native["results"], fixture, helpers, evaluation)
    result.update(schema="rampq.gate1.native_python_partial_comparison/v1", data_role="synthetic",
                  comparison_contract=identity(contract_path), python_source=contract["python_source"],
                  native_structure_audit=contract["native_structure_audit"], native_mat=contract["native_mat"],
                  fixture_count=20, native_matching_fixture_count=5, categories_per_matching_fixture=10,
                  no_AP_metric_values_compared=True, full_Python_AP_implemented=False,
                  official_AP_equivalence_certified=False, project_IoU_first_matcher_called=False,
                  project_matcher_modified=False, existing_Python_source_modified=False,
                  gate1_scientific_pass=False, model_or_real_data_accessed=False,
                  tie_order_policy="Compare the declared Python order literally to this runtime's native row order; record mismatches without repair",
                  scope="Only these fixed synthetic prefilter and single-image matcher intermediates; no general or real-data equivalence claim")
    output = Path(output).resolve()
    require(output.is_relative_to(HERE) and not output.exists(), "New result inside this comparison tree required")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"], result["equal_matrix_checks"], "/", result["matrix_checks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.contract, args.contract_sha256, args.output)


if __name__ == "__main__":
    main()
