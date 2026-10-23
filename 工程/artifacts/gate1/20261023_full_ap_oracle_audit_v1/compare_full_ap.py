"""Compare frozen synthetic Python full AP against saved native output only.

Native output must be generated before this command, and the immutable policy,
fixture, implementation and outputs must be bound in the invocation contract.
No native subprocess, image/model access, package installation or AP oracle
answer substitution occurs in this auditor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
METRICS = ("AP_all", "AP_50", "AP_75", "AR_1", "AR_10", "AR_100", "AR_500")
ATOL = 1e-10
RTOL = 1e-12
POLICY_SCHEMA = "rampq.gate1.full_ap_synthetic_comparison_policy/v1"
CONTRACT_SCHEMA = "rampq.gate1.full_ap_synthetic_comparison_contract/v1"
STRUCTURE_AUDIT = ROOT / "artifacts/gate1/20261023_native_oracle_comparison_preparation_v1/audit_native_oracle.py"
READER_PATH = STRUCTURE_AUDIT.parent / "mat_reader_dependencies"
SOURCE_ROOT = ROOT / "artifacts/gate1/20261022_split_refinement/evaluation_sources/official_toolkit"
FUNCTIONS = ("dropObjectsInIgr", "createIntImg", "evalRes", "compOas", "calcAccuracy", "VOCap")


class ComparisonError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ComparisonError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size,
            "sha256": sha(path)}


def verify(row, label):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}, f"Wrong identity fields: {label}")
    path = (ROOT / row["path"]).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), f"Missing or escaped bound file: {label}")
    require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], f"Identity mismatch: {label}")
    return path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_fields(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), f"Fields differ: {label}")


def scalar_metric(value, label):
    array = np.asarray(value)
    require(array.shape in {(), (1,), (1, 1)} and array.size == 1 and array.dtype.kind == "f"
            and array.dtype.itemsize == 8, f"Expected float64 scalar metric: {label}")
    result = float(array.item())
    require(not math.isinf(result), f"Infinite metric: {label}")
    return result


def compare_metric(candidate, native, label):
    candidate, native = scalar_metric(candidate, label + ".python"), scalar_metric(native, label + ".native")
    require(math.isnan(candidate) == math.isnan(native), f"NaN pattern differs: {label}")
    if math.isnan(native):
        return {"name": label, "native_is_nan": True, "python_is_nan": True,
                "native": None, "python": None, "absolute_error": None, "allowed_error": None}
    error = abs(candidate - native)
    tolerance = ATOL + RTOL * abs(native)
    require(error <= tolerance, f"Metric differs: {label}; absolute_error={error}; allowed={tolerance}")
    return {"name": label, "native_is_nan": False, "python_is_nan": False,
            "native": native, "python": candidate, "absolute_error": error, "allowed_error": tolerance}


def matrix(value, columns, label):
    require(isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == columns
            and value.dtype == np.float64 and bool(np.isfinite(value).all()),
            f"Expected finite float64 m-by-{columns} matrix: {label}")
    return value


def compare_matrix(candidate, native, columns, label):
    candidate, native = matrix(candidate, columns, label + ".python"), matrix(native, columns, label + ".native")
    require(candidate.shape == native.shape, f"Matrix shape differs: {label}")
    require(np.array_equal(candidate, native), f"Matrix values differ: {label}")
    return {"name": label, "shape": list(native.shape), "dtype": "float64", "exact": True}


def validate_fixture(fixture, expected):
    require(fixture.get("data_role") == "synthetic" and fixture.get("expected_oracle_outputs") is None
            and fixture.get("no_real_data_or_model_access") is True, "Fixture violates synthetic-only policy")
    cases = fixture["cases"]
    require(len(cases) == expected["case_count"], "Fixture case count differs")
    require([case["id"] for case in cases] == expected["case_ids"]
            and len(set(expected["case_ids"])) == len(cases), "Fixture case identity/order differs")
    require(dict(Counter(case["kind"] for case in cases)) == expected["kinds"], "Fixture kinds differ")
    for case in cases:
        require(case["kind"] in {"aggregate", "match", "region"} and case["images"], "Unsupported/empty case")
        if case["kind"] == "match":
            require(0 < case["threshold"] <= 1, "Wrong matching threshold")
        for image in case["images"]:
            require(isinstance(image["width"], int) and isinstance(image["height"], int)
                    and image["width"] > 0 and image["height"] > 0, "Wrong image shape")
            for key in ("raw_ground_truth", "raw_predictions"):
                values = np.asarray(image[key], dtype=np.float64).reshape((-1, 8))
                require(bool(np.isfinite(values).all()), "Nonfinite synthetic input")
    return cases


def verify_native_launcher(bound, declared, contract):
    launcher_path = verify(bound["launcher_receipt"], "native launcher receipt")
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    require(launcher["returncode"] == 0 and launcher["fixture_count"] == declared["case_count"], "Native launcher failed/count differs")
    require(launcher["runtime_manifest"] == contract["runtime_manifest"], "Native runtime manifest differs")
    require(launcher["real_data_read"] is False and launcher["model_executed"] is False, "Native launcher scope differs")
    for key in ("oracle_results.mat", "oracle_results.json"):
        expected = bound["native_mat"] if key.endswith(".mat") else bound["native_json"]
        require(launcher["outputs"][key] == expected, "Launcher output identity differs")
    if declared["id"] == "frozen20":
        require(launcher["schema"] == "rampq.gate1.native_synthetic_oracle_launcher/v2"
                and launcher["official_oracle_executed"] is True, "Wrong old native launcher")
        require(launcher["fixture_sha256"] == declared["fixture"]["sha256"], "Old native fixture differs")
        require(launcher["source_manifest_sha256"] == contract["official_source_manifest"]["sha256"], "Old native source manifest differs")
        require(launcher["runtime"]["version"] == declared["native_runtime"]
                and launcher["runtime"]["exe_sha256"] == contract["runtime_executable"]["sha256"], "Old native runtime identity differs")
        require(launcher["runtime_file_count_verified"] == 2890 and launcher["verified_sources"] == contract["official_sources"], "Old native dependencies differ")
        for key in ("selection_accessed", "holdout_accessed", "official_sources_edited", "custom_mean2_used"):
            require(launcher[key] is False, "Old native scope differs: " + key)
        require(Path(launcher["mean2_resolved_path"]).resolve() == verify(contract["native_mean2"], "real native mean2"), "Old native mean2 differs")
        verify(launcher["wrapper"], "old native launcher source")
    else:
        require(launcher["schema"] == "rampq.gate1.full_ap_stress_launcher/v1"
                and launcher["status"] == "EXECUTED_12_NATIVE_STRESS_CASES_ONLY", "Wrong stress native launcher")
        require(launcher["inputs"]["comparison_policy.json"] == contract["policy"], "Native launch policy differs")
        require(launcher["inputs"]["stress_fixtures.json"] == declared["fixture"], "Native stress fixture differs")
        for row in launcher["inputs"].values():
            verify(row, "native stress input")
        require(launcher["official_source_manifest"] == contract["official_source_manifest"]
                and launcher["official_sources_verified"] == 11, "Native stress official source differs")
        require(launcher["runtime_version"] == declared["native_runtime"] and launcher["runtime_files_verified"] == 2890,
                "Native stress runtime differs")
        require(launcher["executable"] == contract["runtime_executable"] and launcher["mean2"] == contract["native_mean2"],
                "Native stress executable/mean2 identity differs")
        for key in ("selection_or_holdout_accessed", "official_sources_modified", "system_configuration_modified"):
            require(launcher[key] is False, "Native stress scope differs: " + key)
        verify(launcher["launcher"], "stress launcher source")
    return identity(launcher_path)


def compare_suite(fixture, native_results, native_receipt, module, structural, expected):
    """Compute fresh Python outputs; never feed native arrays into implementation."""
    cases = validate_fixture(fixture, expected)
    records = structural.cell(native_results, len(cases), "results")
    receipt = structural.struct(native_receipt, "receipt")
    require(structural.text(receipt["schema"], "receipt.schema") == expected["native_schema"], "Native schema differs")
    require(structural.text(receipt["runtime"], "receipt.runtime") == expected["native_runtime"], "Native runtime differs")
    require(structural.scalar(receipt["fixture_count"], "fixture_count") == len(cases), "Native case count differs")
    require(structural.scalar(receipt["case_errors"], "case_errors") == 0, "Native errors present")
    for key in ("real_data_read", "model_executed", "official_AP_equivalence_certified"):
        require(structural.native_bool(receipt[key], key) is False, f"Native scope changed: {key}")
    resolved = structural.struct(receipt["resolved_functions"], "resolved_functions")
    require(set(resolved) == set(FUNCTIONS), "Native source function set differs")
    for name in FUNCTIONS:
        actual = Path(structural.text(resolved[name], name)).resolve()
        require(actual == (SOURCE_ROOT / "utils" / f"{name}.m").resolve(), f"Native source path differs: {name}")
    matrices, metrics, summaries = [], [], []
    for case, raw_record in zip(cases, records):
        label = case["id"]
        record = structural.struct(raw_record, label)
        for field in ("id", "kind", "purpose"):
            require(structural.text(record[field], label) == case[field], f"Native case identity differs: {label}")
        require(structural.text(record["status"], label) == "EXECUTED_NATIVE_SOURCE", f"Native case failed: {label}")
        expected_fields = {"id", "kind", "purpose", "status", "intermediates"}
        if case["kind"] == "aggregate":
            expected_fields.add("metrics")
        require_fields(record, expected_fields, label)
        intermediates = structural.cell(record["intermediates"], len(case["images"]), label)
        allgt, alldt = [], []
        before_matrices, before_metrics = len(matrices), len(metrics)
        for index, (image, raw_intermediate) in enumerate(zip(case["images"], intermediates)):
            image_label = f"{label}.image{index}"
            # Only fixture-created arrays enter the new Python implementation.
            raw_gt = np.asarray(image["raw_ground_truth"], dtype=np.float64).reshape((-1, 8))
            raw_dt = np.asarray(image["raw_predictions"], dtype=np.float64).reshape((-1, 8))
            fresh = module.prepare_raw_image(raw_gt.copy(), raw_dt.copy(), width=image["width"], height=image["height"])
            require_fields(fresh, ("filtered_gt", "filtered_predictions", "prepared_gt"), image_label + ".python")
            intermediate = structural.struct(raw_intermediate, image_label)
            expected_image_fields = {"filtered_gt", "filtered_predictions", "prepared_gt"}
            if case["kind"] == "match":
                expected_image_fields.add("class_matches")
            require_fields(intermediate, expected_image_fields, image_label + ".native")
            for key in ("filtered_gt", "filtered_predictions", "prepared_gt"):
                matrices.append(compare_matrix(fresh[key], intermediate[key], 8, image_label + "." + key))
            gt, dt = fresh["prepared_gt"], fresh["filtered_predictions"]
            allgt.append(gt)
            alldt.append(dt)
            if case["kind"] == "match":
                classes = structural.cell(intermediate["class_matches"], 10, image_label + ".class_matches")
                for category, raw_class in enumerate(classes, 1):
                    class_label = f"{image_label}.class{category}"
                    observed = structural.struct(raw_class, class_label)
                    require_fields(observed, ("category", "gt_output", "dt_output"), class_label)
                    require(structural.scalar(observed["category"], class_label) == category, "Native category/order differs")
                    candidate_gt, candidate_dt = module.eval_res(gt[gt[:, 5] == category, :5].copy(),
                        dt[dt[:, 5] == category, :5].copy(), case["threshold"])
                    matrices.append(compare_matrix(candidate_gt, observed["gt_output"], 5, class_label + ".gt_output"))
                    matrices.append(compare_matrix(candidate_dt, observed["dt_output"], 6, class_label + ".dt_output"))
        if case["kind"] == "aggregate":
            fresh = module.calc_accuracy(allgt, alldt, capture_intermediates=False)
            require(isinstance(fresh, dict) and "metrics" in fresh, f"No Python metric outputs: {label}")
            observed = structural.struct(record["metrics"], label + ".metrics")
            require_fields(observed, METRICS, label + ".native.metrics")
            require_fields(fresh["metrics"], METRICS, label + ".python.metrics")
            for name in METRICS:
                metrics.append(compare_metric(fresh["metrics"][name], observed[name], f"{label}.{name}"))
        summaries.append({"id": label, "kind": case["kind"], "images": len(case["images"]),
                          "matrix_count": len(matrices) - before_matrices, "metric_count": len(metrics) - before_metrics,
                          "status": "PASS_DECLARED_SYNTHETIC_OUTPUTS"})
    require(len(matrices) == expected["matrix_count"] and len(metrics) == expected["metric_count"], "Comparison coverage count differs")
    finite_errors = [row["absolute_error"] for row in metrics if row["absolute_error"] is not None]
    return {"cases": summaries, "matrix_count": len(matrices), "metric_count": len(metrics), "matrices": matrices,
            "metrics": metrics, "max_finite_metric_absolute_error": max(finite_errors, default=None),
            "metric_nan_count": sum(row["native_is_nan"] for row in metrics),
            "native_internal_AP_AR_tables_compared": False}


def audit(contract_path, contract_sha):
    require(sha(contract_path) == contract_sha, "Invocation contract hash differs")
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    require(contract["schema"] == CONTRACT_SCHEMA, "Wrong invocation contract schema")
    policy_path = verify(contract["policy"], "policy")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    require(policy["schema"] == POLICY_SCHEMA and policy["data_role"] == "synthetic", "Wrong policy schema/scope")
    require(policy["finite_metric_tolerance"] == {"atol": ATOL, "rtol": RTOL, "reference": "native", "unit": "percentage"}, "Metric tolerance changed")
    require(policy["matrix_tolerance"] == {"atol": 0, "rtol": 0, "shape_exact": True, "dtype": "float64"}, "Matrix comparison changed")
    require(policy["nan_pattern_exact"] is True and policy["infinity_allowed"] is False, "Nonfinite policy changed")
    require(policy["native_internal_tables_compared"] is False and policy["gate1_scientific_pass"] is False,
            "Claim boundary changed")
    implementation_path = verify(contract["implementation"], "implementation")
    for row in contract["implementation_dependencies"]:
        verify(row, "implementation dependency")
    require(verify(contract["structural_auditor"], "structural auditor") == STRUCTURE_AUDIT, "Unapproved MAT structural reader")
    require(verify(contract["auditor"], "this auditor") == Path(__file__).resolve(), "Auditor identity differs")
    source_manifest_path = verify(contract["official_source_manifest"], "official source manifest")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_sources = [{"path": str((source_manifest_path.parent / row["path"]).relative_to(ROOT)).replace("\\", "/"),
                         "bytes": row["bytes"], "sha256": row["sha256"]} for row in source_manifest["files"]]
    require(contract["official_sources"] == expected_sources, "Official source membership/order differs")
    for row in contract["official_sources"]:
        verify(row, "official source")
    require(len(contract["official_sources"]) == 11, "Official source member count differs")
    runtime_manifest_path = verify(contract["runtime_manifest"], "runtime manifest")
    runtime_members = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))["files"]
    require(len(runtime_members) == 2890 and len({row["path"] for row in runtime_members}) == 2890,
            "Runtime dependency membership differs")
    for row in runtime_members:
        verify(row, "native runtime dependency")
    verify(contract["runtime_executable"], "native runtime executable")
    verify(contract["native_mean2"], "native package mean2")
    structural = load_module(STRUCTURE_AUDIT, "rampq_structural_native_reader_frozen")
    reader_receipt = structural.verify_reader_dependency(READER_PATH)
    sys.path.insert(0, str(READER_PATH))
    from scipy.io import loadmat
    import scipy
    implementation = load_module(implementation_path, "rampq_full_ap_frozen_implementation")
    suites = []
    require([suite["id"] for suite in contract["suites"]] == [suite["id"] for suite in policy["suites"]], "Suite membership/order differs")
    for bound, declared in zip(contract["suites"], policy["suites"]):
        fixture_path = verify(bound["fixture"], "fixture")
        require(bound["fixture"] == declared["fixture"], "Fixture identity differs from predeclared policy")
        native_mat = verify(bound["native_mat"], "native MAT")
        native_json = verify(bound["native_json"], "native JSON")
        launcher_identity = verify_native_launcher(bound, declared, contract)
        if bound.get("native_structure_audit"):
            prior = json.loads(verify(bound["native_structure_audit"], "prior native structural audit").read_text(encoding="utf-8"))
            require(prior["status"] == "PASS_NATIVE_SYNTHETIC_OUTPUT_STRUCTURE_IDENTITIES_ONLY", "Native structural audit not passed")
        native = loadmat(native_mat, squeeze_me=False, struct_as_record=False, chars_as_strings=True,
                         mat_dtype=True, verify_compressed_data_integrity=True)
        require(set(native) - {"__header__", "__version__", "__globals__"} == {"receipt", "results"}, "Unexpected MAT variables")
        serial = json.loads(native_json.read_text(encoding="utf-8"))
        require_fields(serial, ("receipt", "results"), "native JSON")
        structural.compare_serialization(native["receipt"], serial["receipt"], "receipt")
        structural.compare_serialization(native["results"], serial["results"], "results")
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        result = compare_suite(fixture, native["results"], native["receipt"], implementation, structural, declared)
        suites.append({"id": bound["id"], "launcher_receipt": launcher_identity, **result})
    return {"schema": "rampq.gate1.full_ap_synthetic_native_comparison/v1",
            "status": "PASS_DECLARED_SYNTHETIC_FULL_AP_NATIVE_COMPARISON_ONLY", "policy": identity(policy_path),
            "contract": identity(contract_path), "implementation": identity(implementation_path),
            "suites": suites, "matrix_count": sum(row["matrix_count"] for row in suites),
            "metric_count": sum(row["metric_count"] for row in suites),
            "float64_final_metric_tolerance": policy["finite_metric_tolerance"], "matrix_tolerance": policy["matrix_tolerance"],
            "reader": {"scipy": scipy.__version__, "numpy": np.__version__, "dependency_verification": reader_receipt},
            "comparison_scope": "Only the fixed synthetic inputs and saved native final metrics/matrices",
            "native_internal_AP_AR_tables_compared": False, "arbitrary_input_equivalence_certified": False,
            "real_data_AP_computed": False, "model_executed": False, "selection_or_holdout_accessed": False,
            "original_dtype_guard_modified": False, "gate1_scientific_pass": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = args.output.resolve()
    require(target.is_relative_to(HERE) and not target.exists(), "Comparison output must be new and inside this audit directory")
    try:
        result = audit(args.contract.resolve(), args.contract_sha256)
    except Exception as error:
        failure = {"schema": "rampq.gate1.full_ap_synthetic_native_comparison_attempt/v1",
                   "status": "FAIL_CLOSED_COMPARISON_ATTEMPT", "contract_path": str(args.contract.resolve()),
                   "contract_sha256_argument": args.contract_sha256,
                   "error_type": type(error).__name__, "error_message": str(error), "gate1_scientific_pass": False}
        with target.open("x", encoding="utf-8") as stream:
            json.dump(failure, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        raise
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"])
    print(f"MATRICES: {result['matrix_count']}; METRICS: {result['metric_count']}")


if __name__ == "__main__":
    main()
