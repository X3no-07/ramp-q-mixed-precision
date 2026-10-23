"""Audit saved native synthetic oracle structure, identities, and MAT/JSON parity.

No AP implementation or handwritten expected AP values are present here.
The existing Python matcher is neither run nor certified by this auditor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
READINESS = ROOT / "artifacts/gate1/20261023_official_evaluator_readiness"
SOURCES = ROOT / "artifacts/gate1/20261022_split_refinement/evaluation_sources"
NATIVE_SCHEMA = "rampq.gate1.synthetic_official_oracle_execution/v1"
LAUNCHER_SCHEMA = "rampq.gate1.native_synthetic_oracle_launcher/v2"
METRICS = ("AP_all", "AP_50", "AP_75", "AR_1", "AR_10", "AR_100", "AR_500")
FUNCTIONS = ("dropObjectsInIgr", "createIntImg", "evalRes", "compOas", "calcAccuracy", "VOCap")
COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"
READINESS_MANIFEST_SHA = "10e6d14bfd0b0c557bf6e995790a7d4290d869e45a7c94bbfd3bd09e89a7952a"
SOURCE_MANIFEST_SHA = "1d45d3ecaabf5f3f8a8885436a0a22718a890a17f8ad9d5ec4f9c274c3c079ae"
SCIPY_WHEEL = "scipy-1.18.1-cp312-cp312-win_amd64.whl"
SCIPY_WHEEL_SHA = "5e4d44984abc0020154ea81b247adeddcc3ac5527b975ff798bd1ba0adc513c2"


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")
    return digest.hexdigest()


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_identity(path, expected, label):
    path = Path(path)
    require(path.is_file(), f"Missing {label}: {path}")
    require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
            f"Identity mismatch: {label}")


def struct(value, label):
    if isinstance(value, dict):
        return value
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    if hasattr(value, "_fieldnames"):
        return {name: getattr(value, name) for name in value._fieldnames}
    raise AuditError(f"Expected scalar MAT struct: {label}")


def cell(value, count, label):
    require(isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 2,
            f"Expected two-dimensional MAT cell: {label}")
    require(value.size == count and (value.shape[0] == 1 or value.shape[1] == 1),
            f"Unexpected cell count/shape: {label}")
    return list(value.ravel(order="F"))


def text(value, label):
    if isinstance(value, str):
        return value
    require(isinstance(value, np.ndarray) and value.dtype.kind in "US" and value.size == 1,
            f"Expected scalar native string: {label}")
    return str(value.item())


def scalar(value, label, *, allow_nan=False):
    array = np.asarray(value)
    require(array.size == 1 and array.dtype.kind in "fiu", f"Expected numeric scalar: {label}")
    result = float(array.item())
    require(not math.isinf(result) and (allow_nan or not math.isnan(result)),
            f"Disallowed nonfinite scalar: {label}")
    return result


def native_bool(value, label):
    array = np.asarray(value)
    require(array.size == 1 and array.dtype.kind in "bfiu" and array.item() in (0, 1),
            f"Expected native logical scalar: {label}")
    return bool(array.item())


def matrix(value, columns, label):
    require(isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == columns,
            f"Wrong numeric matrix dimensions: {label}; expected m x {columns}")
    require(value.dtype.kind in "fiu" and bool(np.isfinite(value).all()),
            f"Expected finite numeric matrix: {label}")
    return value.astype(np.float64, copy=False)


def rows8(rows):
    return np.asarray(rows, dtype=np.float64).reshape((-1, 8))


def row_counter(array):
    return Counter(tuple(row) for row in array.tolist())


def subsequence(kept, original, label):
    cursor = 0
    for row in kept:
        while cursor < len(original) and not np.array_equal(row, original[cursor]):
            cursor += 1
        require(cursor < len(original), f"Filtered rows not an ordered input subsequence: {label}")
        cursor += 1


def compare_serialization(native, serial, label):
    """Only serialization losses in dimension-one/empty arrays and NaN are allowed."""
    if hasattr(native, "_fieldnames") or isinstance(native, dict):
        native = struct(native, label)
        require(isinstance(serial, dict) and set(serial) == set(native), f"JSON struct fields differ: {label}")
        for key, value in native.items():
            compare_serialization(value, serial[key], f"{label}.{key}")
        return
    if isinstance(native, str):
        require(native == serial, f"JSON string differs: {label}")
        return
    require(isinstance(native, np.ndarray), f"Unsupported MAT type: {label}")
    if native.dtype == object:
        # Scalar MAT structs are wrapped as object arrays by scipy without squeezing.
        if native.size == 1 and hasattr(native.item(), "_fieldnames") and isinstance(serial, dict):
            compare_serialization(native.item(), serial, label)
            return
        values = list(native.ravel(order="F"))
        serial_values = serial if isinstance(serial, list) else [serial] if len(values) == 1 else None
        require(isinstance(serial_values, list) and len(serial_values) == len(values),
                f"JSON cell count differs: {label}")
        for index, (value, other) in enumerate(zip(values, serial_values)):
            compare_serialization(value, other, f"{label}[{index}]")
        return
    if native.dtype.kind in "US":
        require(text(native, label) == serial, f"JSON string differs: {label}")
        return
    require(native.dtype.kind in "bfiu", f"Unsupported MAT array class: {label}")
    if native.size == 0:
        require(serial == [], f"JSON empty array differs: {label}")
        return
    other = np.asarray(serial, dtype=object)
    permitted_shapes = {native.shape, np.squeeze(native).shape}
    if native.ndim == 2 and 1 in native.shape:
        permitted_shapes.add((native.size,))
    require(other.shape in permitted_shapes and other.size == native.size,
            f"JSON numeric array dimensions differ: {label}")
    for index, (value, candidate) in enumerate(zip(native.ravel(order="C"), other.ravel(order="C"))):
        if math.isnan(float(value)):
            require(candidate is None, f"JSON must preserve MAT NaN as null: {label}[{index}]")
        elif native.dtype.kind == "b":
            require(isinstance(candidate, bool) and bool(value) == candidate,
                    f"JSON logical differs: {label}[{index}]")
        else:
            require(isinstance(candidate, (int, float)) and not isinstance(candidate, bool)
                    and math.isfinite(candidate) and float(value) == float(candidate),
                    f"JSON finite numeric value differs: {label}[{index}]")


def same_source_path(reported, expected, aliases=()):
    """Prefer live samefile; recorded aliases are an explicit receipt trust boundary."""
    expected = Path(expected).resolve()
    reported_path = Path(reported)
    if reported_path.exists():
        return os.path.samefile(reported_path, expected), "live_samefile"
    candidate = str(reported).replace("\\", "/")
    for alias in aliases:
        prefix = alias["native_prefix"].replace("\\", "/").rstrip("/")
        if candidate.casefold().startswith((prefix + "/").casefold()):
            suffix = candidate[len(prefix):].lstrip("/")
            rewritten = Path(alias["real_prefix"]) / suffix
            return rewritten.resolve() == expected, "recorded_alias_only"
    return os.path.normcase(str(reported_path)) == os.path.normcase(str(expected)), "direct_path_only"


def validate_match(intermediate, gt, dt, label):
    records = cell(intermediate["class_matches"], 10, label + ".class_matches")
    for category, raw_record in enumerate(records, 1):
        record = struct(raw_record, label)
        require(set(record) == {"category", "gt_output", "dt_output"}, f"Wrong matcher fields: {label}")
        require(scalar(record["category"], label) == category, f"Wrong matcher category/order: {label}")
        gt0, dt0 = gt[gt[:, 5] == category, :5], dt[dt[:, 5] == category, :5]
        gt1 = matrix(record["gt_output"], 5, label + ".gt_output")
        dt1 = matrix(record["dt_output"], 6, label + ".dt_output")
        require(row_counter(gt1[:, :4]) == row_counter(gt0[:, :4]), f"Matcher changed GT rows: {label}")
        require(row_counter(dt1[:, :5]) == row_counter(dt0), f"Matcher changed DT rows: {label}")
        require(set(gt1[:, 4]) <= {-1, 0, 1} and set(dt1[:, 5]) <= {-1, 0, 1},
                f"Invalid native match flag: {label}")
        require(np.count_nonzero(gt1[:, 4] == -1) == np.count_nonzero(gt0[:, 4] == 1),
                f"Ignored GT row count changed: {label}")
        ignored_geometry = row_counter(gt1[gt1[:, 4] == -1, :4])
        require(ignored_geometry == row_counter(gt0[gt0[:, 4] == 1, :4]),
                f"Ignored GT geometry changed: {label}")
        require(bool(np.all(np.diff((gt1[:, 4] == -1).astype(int)) >= 0)),
                f"Native GT ignored-last ordering changed: {label}")
        require(bool(np.all(np.diff(dt1[:, 4]) <= 0)), f"Native DT score ordering changed: {label}")


def validate_native(results, receipt, json_bundle, fixture, source_root, aliases=()):
    receipt_fields = struct(receipt, "receipt")
    require(text(receipt_fields["schema"], "receipt.schema") == NATIVE_SCHEMA, "Wrong native receipt schema")
    require(scalar(receipt_fields["fixture_count"], "fixture_count") == 20, "Wrong native fixture count")
    require(scalar(receipt_fields["case_errors"], "case_errors") == 0, "Native case errors present")
    for field in ("real_data_read", "model_executed", "official_AP_equivalence_certified"):
        require(native_bool(receipt_fields[field], field) is False, f"Scope flag changed: {field}")
    require(bool(text(receipt_fields["runtime"], "runtime")), "Missing native runtime version")
    require(bool(text(receipt_fields["computer"], "computer")), "Missing native computer identity")
    resolved = struct(receipt_fields["resolved_functions"], "resolved_functions")
    require(set(resolved) == set(FUNCTIONS), "Native resolved function set differs")
    path_checks = []
    for name in FUNCTIONS:
        reported = text(resolved[name], name)
        expected = Path(source_root) / "utils" / (name + ".m")
        equal, method = same_source_path(reported, expected, aliases)
        require(equal, f"Native official function shadowed: {name}")
        path_checks.append({"function": name, "reported": reported, "expected": str(expected), "method": method})
    cases = fixture["cases"]
    require(fixture["data_role"] == "synthetic" and fixture["expected_oracle_outputs"] is None,
            "Fixture is not unlabelled synthetic oracle input")
    require(len(cases) == 20 and Counter(c["kind"] for c in cases) == {"aggregate": 11, "region": 4, "match": 5},
            "Wrong fixture coverage")
    records = cell(results, len(cases), "results")
    observations = []
    for expected, raw_record in zip(cases, records):
        label = expected["id"]
        record = struct(raw_record, label)
        allowed = {"id", "kind", "purpose", "intermediates", "status"}
        if expected["kind"] == "aggregate":
            allowed.add("metrics")
        require(set(record) == allowed, f"Wrong result fields: {label}")
        for field in ("id", "kind", "purpose"):
            require(text(record[field], label + "." + field) == expected[field], f"Fixture identity/order differs: {label}")
        require(text(record["status"], label) == "EXECUTED_NATIVE_SOURCE", f"Native case not executed: {label}")
        intermediates = cell(record["intermediates"], len(expected["images"]), label + ".intermediates")
        image_observations = []
        for image_index, (raw_intermediate, image) in enumerate(zip(intermediates, expected["images"])):
            image_label = f"{label}.image{image_index}"
            intermediate = struct(raw_intermediate, image_label)
            allowed_intermediate = {"filtered_gt", "filtered_predictions", "prepared_gt"}
            if expected["kind"] == "match":
                allowed_intermediate.add("class_matches")
            require(set(intermediate) == allowed_intermediate, f"Wrong intermediate fields: {image_label}")
            gt = matrix(intermediate["filtered_gt"], 8, image_label + ".filtered_gt")
            dt = matrix(intermediate["filtered_predictions"], 8, image_label + ".filtered_predictions")
            prepared = matrix(intermediate["prepared_gt"], 8, image_label + ".prepared_gt")
            raw_gt, raw_dt = rows8(image["raw_ground_truth"]), rows8(image["raw_predictions"])
            subsequence(gt, raw_gt[raw_gt[:, 5] != 0], image_label + ".filtered_gt")
            subsequence(dt, raw_dt, image_label + ".filtered_predictions")
            inverted = gt.copy()
            inverted[gt[:, 4] == 0, 4] = 1
            inverted[gt[:, 4] == 1, 4] = 0
            require(np.array_equal(prepared, inverted), f"Prepared score masks differ: {image_label}")
            if expected["kind"] == "match":
                validate_match(intermediate, prepared, dt, image_label)
            image_observations.append({"filtered_gt_shape": list(gt.shape), "filtered_predictions_shape": list(dt.shape),
                                       "prepared_gt_shape": list(prepared.shape), "surviving_score_zero_gt": int(np.count_nonzero(gt[:, 4] == 0))})
        observed = {"id": label, "kind": expected["kind"], "status": "STRUCTURE_CONSISTENT", "images": image_observations}
        if expected["kind"] == "aggregate":
            metrics = struct(record["metrics"], label + ".metrics")
            require(set(metrics) == set(METRICS), f"Wrong metric set: {label}")
            observed["native_metrics_observed_not_handwritten_expected"] = {}
            for name in METRICS:
                value = scalar(metrics[name], label + "." + name, allow_nan=True)
                observed["native_metrics_observed_not_handwritten_expected"][name] = {"value": None if math.isnan(value) else value, "is_nan": math.isnan(value)}
        observations.append(observed)
    require(isinstance(json_bundle, dict) and set(json_bundle) == {"receipt", "results"}, "Wrong JSON top-level structure")
    compare_serialization(receipt, json_bundle["receipt"], "receipt")
    compare_serialization(results, json_bundle["results"], "results")
    return {"runtime": text(receipt_fields["runtime"], "runtime"), "computer": text(receipt_fields["computer"], "computer"),
            "resolved_function_path_checks": path_checks, "fixture_count": 20,
            "kinds": {"aggregate": 11, "region": 4, "match": 5}, "observations": observations,
            "MAT_authoritative_for_empty_dimensions_and_NaN": True, "JSON_serialization_consistent": True}


def verify_frozen_inputs():
    require(sha(READINESS / "manifest.json") == READINESS_MANIFEST_SHA, "Frozen readiness manifest changed")
    require(sha(SOURCES / "source_manifest.json") == SOURCE_MANIFEST_SHA, "Frozen official source manifest changed")
    readiness_manifest = json.loads((READINESS / "manifest.json").read_text(encoding="utf-8"))
    require(readiness_manifest["file_count"] == len(readiness_manifest["files"]) == 12, "Wrong readiness member count")
    for row in readiness_manifest["files"]:
        verify_identity(READINESS / row["path"], row, "readiness " + row["path"])
    source_manifest = json.loads((SOURCES / "source_manifest.json").read_text(encoding="utf-8"))
    require(source_manifest["commit"] == COMMIT and len(source_manifest["files"]) == 11, "Wrong source identity scope")
    for row in source_manifest["files"]:
        verify_identity(SOURCES / row["path"], row, "source " + row["path"])
    contract = json.loads((READINESS / "fixture_contract.json").read_text(encoding="utf-8"))
    for row in contract["bound_local_files"]:
        verify_identity(READINESS / row["path"], row, "fixture-bound " + row["path"])
    return json.loads((READINESS / "synthetic_fixtures.json").read_text(encoding="utf-8")), source_manifest


def verify_reader_dependency(reader_path):
    reader_path = Path(reader_path).resolve()
    require(reader_path == (HERE / "mat_reader_dependencies").resolve() and reader_path.is_dir(),
            "MAT reader must use the approved isolated dependency tree")
    wheel = HERE / SCIPY_WHEEL
    require(sha(wheel) == SCIPY_WHEEL_SHA, "MAT reader wheel identity mismatch")
    with zipfile.ZipFile(wheel) as archive:
        members = [row for row in archive.infolist() if not row.is_dir()]
        expected_files = set()
        for member in members:
            path = (reader_path / member.filename).resolve()
            require(path.is_relative_to(reader_path), "Wheel member escaped isolated dependency root")
            require(path.is_file() and path.stat().st_size == member.file_size, "MAT reader member missing/size changed")
            with archive.open(member) as stream:
                expected_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            require(sha(path) == expected_sha, "MAT reader extracted member identity changed: " + member.filename)
            expected_files.add(path)
    actual_files = {path.resolve() for path in reader_path.rglob("*") if path.is_file()}
    require(actual_files == expected_files, "Unexpected files in the isolated MAT reader tree")
    return {"wheel": identity(wheel), "verified_extracted_member_count": len(expected_files),
            "existing_Python_environment_modified": False}


def audit_execution(execution, reader_path=None):
    execution = Path(execution).resolve()
    fixture, source_manifest = verify_frozen_inputs()
    launcher = json.loads((execution / "launcher_receipt.json").read_text(encoding="utf-8"))
    require(launcher["schema"] == LAUNCHER_SCHEMA, "Wrong launcher schema")
    require(launcher["returncode"] == 0 and launcher["official_oracle_executed"] is True,
            "No successful native oracle execution")
    require(launcher["source_commit"] == COMMIT and launcher["fixture_count"] == 20, "Wrong launched source/fixture scope")
    for field in ("real_data_read", "model_executed", "official_AP_equivalence_certified"):
        require(launcher[field] is False, "Launcher scope changed: " + field)
    for field in ("selection_accessed", "holdout_accessed", "official_sources_edited", "custom_mean2_used"):
        require(launcher[field] is False, "Launcher scope changed: " + field)
    require(launcher["source_manifest_sha256"] == sha(SOURCES / "source_manifest.json"), "Wrong launch source manifest")
    for name, key in (("synthetic_fixtures.json", "fixture_sha256"), ("run_synthetic_oracle.m", "runner_sha256"),
                      ("fixture_contract.json", "fixture_contract_sha256")):
        require(launcher[key] == sha(READINESS / name), "Wrong launched preparation identity: " + name)
    actual_sources = {row["path"]: (row["bytes"], row["sha256"]) for row in source_manifest["files"]}
    reported_sources = {}
    for row in launcher["verified_sources"]:
        relative = row["path"].replace("\\", "/")
        if relative.startswith("artifacts/gate1/20261022_split_refinement/evaluation_sources/"):
            relative = relative.split("evaluation_sources/", 1)[1]
        reported_sources[relative] = (row["bytes"], row["sha256"])
    require(len(launcher["verified_sources"]) == len(reported_sources) == 11 and reported_sources == actual_sources,
            "Launcher source member list differs")
    runtime = launcher["runtime"]
    require(Path(runtime["executable"]).is_file() and sha(runtime["executable"]) == runtime["exe_sha256"], "Runtime executable identity mismatch")
    require(isinstance(runtime["command"], list) and len(runtime["command"]) >= 3 and runtime["command"][0] == runtime["executable"],
            "Missing exact native launch command")
    require(bool(launcher["mean2_resolved_path"]), "Missing real native mean2 path")
    for key in ("wrapper", "runtime_manifest", "subst_query"):
        verify_identity(ROOT / launcher[key]["path"], launcher[key], "launcher-bound " + key)
    for name in ("oracle_results.mat", "oracle_results.json", "native_console.txt", "runtime_preflight.json"):
        require((ROOT / launcher["outputs"][name]["path"]).resolve() == (execution / name).resolve(), "Wrong native output path identity")
        verify_identity(execution / name, launcher["outputs"][name], "native output " + name)
    preflight = json.loads((execution / "runtime_preflight.json").read_text(encoding="utf-8"))
    require(preflight["version"] == runtime["version"] and preflight["mean2_resolved_path"] == launcher["mean2_resolved_path"],
            "Runtime preflight differs from launcher")
    require(Path(launcher["mean2_resolved_path"]).is_file(), "Native mean2 file not accessible for verification")
    runtime_files = json.loads((ROOT / launcher["runtime_manifest"]["path"]).read_text(encoding="utf-8"))
    require(runtime_files["schema"] == "rampq.gate1.octave_portable_runtime_files/v1"
            and runtime_files["path_base"] == "workspace", "Wrong runtime manifest schema/path base")
    members = runtime_files["files"]
    require(len(members) == launcher["runtime_file_count_verified"] and len({row["path"] for row in members}) == len(members),
            "Runtime manifest member count differs")
    runtime_root = (execution.parent / "runtime").resolve()
    verified_paths = []
    for row in members:
        member = (ROOT / row["path"]).resolve()
        require(member.is_relative_to(runtime_root), "Runtime dependency escaped the new runtime tree")
        verify_identity(member, row, "runtime dependency " + row["path"])
        verified_paths.append(member)
    require(any(os.path.samefile(runtime["executable"], path) for path in verified_paths), "Runtime executable missing from dependency manifest")
    require(any(os.path.samefile(launcher["mean2_resolved_path"], path) for path in verified_paths), "Native mean2 missing from dependency manifest")
    require(reader_path is not None, "Explicit isolated MAT reader path required")
    reader_identity = verify_reader_dependency(reader_path)
    sys.path.insert(0, str(Path(reader_path).resolve()))
    from scipy.io import loadmat
    import scipy
    native = loadmat(execution / "oracle_results.mat", squeeze_me=False, struct_as_record=False,
                     chars_as_strings=True, mat_dtype=True, verify_compressed_data_integrity=True)
    require(set(native) - {"__header__", "__version__", "__globals__"} == {"receipt", "results"}, "Unexpected MAT top-level variables")
    serial = json.loads((execution / "oracle_results.json").read_text(encoding="utf-8"))
    result = validate_native(native["results"], native["receipt"], serial, fixture,
                             SOURCES / "official_toolkit", [launcher["alias"]] if launcher.get("alias") else ())
    require(result["runtime"] == runtime["version"], "MAT runtime version differs from launcher")
    result.update({"schema": "rampq.gate1.native_synthetic_oracle_saved_output_audit/v1",
                   "status": "PASS_NATIVE_SYNTHETIC_OUTPUT_STRUCTURE_IDENTITIES_ONLY",
                   "execution": str(execution), "launcher": identity(execution / "launcher_receipt.json"),
                   "source_manifest": identity(SOURCES / "source_manifest.json"),
                   "frozen_readiness_manifest": identity(READINESS / "manifest.json"),
                   "mat_reader": {"scipy": scipy.__version__, "numpy": np.__version__, "python": sys.version},
                   "mat_reader_dependency_verification": reader_identity,
                   "official_AP_equivalence_certified": False, "existing_Python_matcher_certified": False,
                   "full_Python_AP_implemented": False, "gate1_scientific_pass": False,
                   "model_executed_by_auditor": False, "real_data_read_by_auditor": False,
                   "handwritten_expected_AP_values_used": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader-path", type=Path)
    args = parser.parse_args()
    target = args.output.resolve()
    require(target.is_relative_to(HERE) and not target.exists(), "Audit output must be new and inside this preparation tree")
    result = audit_execution(args.execution, args.reader_path)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"])


if __name__ == "__main__":
    main()
