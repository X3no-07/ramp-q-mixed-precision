"""Seal completed acceptance prerequisites after root authorizes final execution.

This script is prepared before final reports/manifests exist. A separately
reviewed seal_inputs.json pins every dynamic identity. --check-readiness only
reports missing paths, performs no large hashing, and writes nothing. Normal
execution requires all sources, reviews and final native manifests to be fixed.

Only file-byte integrity is checked. No runtime/model/test is executed, no MAT,
image, annotation, tensor or archive is decoded, and no Holdout path is opened.
The prior 122-member small delivery is rechecked without descending into its
older multi-GB numerical evidence. Runtime rows are verified once by SHA; the
auditor's already-recorded 1535 extracted SciPy checks are a stated dependency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = "artifacts/gate1/"
MAX_TOP_BYTES = 16 * 1024 * 1024
SMALL_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".ps1", ".txt", ".xml", ".log", ".m", ".svg", ".png", ".pt", ".mat", ".sig"}
EXPECTED_DYNAMIC_PATHS = {
    "report": "48_Gate1_训练前置组件与官方原生评分验证.md",
    "state": BASE + "current_execution_state_v9.json",
    "decision": BASE + "20261023_acceptance_prerequisites_delivery_v1/decision.json",
    "build_result": BASE + "20261023_acceptance_prerequisites_delivery_v1/build_result.py",
    "final_review": BASE + "20261023_acceptance_prerequisites_review_v1/final_review.json",
    "runtime_preparation_manifest": BASE + "20261023_native_oracle_runtime_v1/preparation_manifest.json",
    "native_semantics_manifest": BASE + "20261023_native_python_semantics_v1/manifest.json",
    "native_audit": BASE + "20261023_native_oracle_comparison_preparation_v1/native_output_audit_v1.json",
    "native_audit_manifest": BASE + "20261023_native_oracle_comparison_preparation_v1/native_output_audit_manifest_v1.json",
}
# (ROOT-relative manifest, member base, expected count, immutable SHA256)
FIXED_NESTED = [
    (BASE + "20261023_upstream_diagnosis_delivery/delivery_manifest.json", "root", 122,
     "aa272247b7e449e6bd923979cb27bbdc23c3a88e951fc7ce9bba0a90bed5d29d"),
    (BASE + "20261023_ignore_loss_preparation_v1/manifest.json", "local", 17,
     "136e68f3dec791b7f5e6449f00565d061a1f8d3b8453dad33bfbedc752c39c58"),
    (BASE + "20261023_fp32_reference_preparation_v1/manifest.json", "local", 8,
     "449710399204a83837619e6f0083c5893b0998fc1829839eece018b3aa9703a0"),
    (BASE + "20261023_acceptance_prerequisites_review_v1/source_review_manifest.json", "local", 6,
     "6249440e2af410419a10b7326b6053615f99d96d731d4cbde90134c9390ccfe2"),
    (BASE + "20261023_official_evaluator_readiness/manifest.json", "local", 12,
     "10e6d14bfd0b0c557bf6e995790a7d4290d869e45a7c94bbfd3bd09e89a7952a"),
    (BASE + "20261023_native_oracle_comparison_preparation_v1/preparation_manifest.json", "root", 14,
     "7bd3a6006e8962ce87321acfb3f63b6ddb13bf2ad01b49d35ea795de6290b52f"),
]
LIVE_COMPONENTS = {
    "scripts/gate1_ignore_negative_loss.py": "67fa0f7a171d26b1f8d2ebd922333f30bb97ffb94b523a58f6f784ed572b3c1f",
    "tests/test_gate1_ignore_negative_loss.py": "e6b182bf70abeabcb5b2a843b50bf288d7282f1e6d33ffe4aab2346e32516fe4",
    "scripts/gate1_fp32_reference_state.py": "02ccca03a1bfd9ab330a17a740c13abc7cf68fc83b1d4cc1a073c8983d452505",
    "tests/test_gate1_fp32_reference_state.py": "69942391ce10a2b0e978cb8b23d4ae83012bb434518973e12584a7eed9f88b5a",
}
FALSE_SCOPE_KEYS = {
    "gate1_scientific_pass", "scientific_gate1_passed", "scientific_gate_pass", "accepted_float_baseline_ready",
    "official_AP_equivalence_certified", "full_Python_AP_implemented", "new_precision_model_accepted_or_deployed",
    "selection_detector_inference", "holdout_accessed", "holdout_opened_or_listed", "real_data_read",
    "training_pipeline_integrated", "complete_ignore_aware_training_solved", "live_training_hook_integrated",
}
CACHE: dict[str, tuple[tuple, dict]] = {}
BYTES_HASHED = 0
CACHE_HITS = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def opaque_name(name):
    return any(any(marker in part.casefold() for marker in ("holdout", "test-dev", "test_dev"))
               for part in PurePosixPath(str(name).replace("\\", "/")).parts)


def forbidden_volatile(name):
    parts = [part.casefold() for part in PurePosixPath(str(name).replace("\\", "/")).parts]
    return any(part in {"__pycache__", "test_cases", "test_temp", "tmp", "temp", ".pytest_cache"}
               or ("mpl" in part and ("cache" in part or "config" in part)) for part in parts) or "latest_execution_state.json" in parts


def normalized_relative(name):
    rel = PurePosixPath(str(name).replace("\\", "/"))
    require(bool(rel.parts) and not rel.is_absolute() and ".." not in rel.parts
            and not any(":" in part for part in rel.parts), "Unsafe manifest member path")
    require(not opaque_name(rel.as_posix()), "Holdout/test-dev paths remain opaque")
    return rel


def safe(base, name):
    rel = normalized_relative(name)
    path = base
    for part in rel.parts:
        path = path / part
        require(not path.is_symlink() and not path.is_junction(), "Reparse links forbidden: " + str(path))
    path = path.resolve(strict=True)
    require(path.is_relative_to(ROOT) and path.is_file(), "File leaves workspace or is not regular")
    return path


def signature(path):
    value = path.stat()
    require(stat.S_ISREG(value.st_mode), "Not a regular file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def identity(path):
    global BYTES_HASHED, CACHE_HITS
    path = safe(ROOT, path.relative_to(ROOT).as_posix())
    key, before = str(path).casefold(), signature(path)
    if key in CACHE:
        CACHE_HITS += 1
        require(CACHE[key][0] == before, "Previously hashed evidence changed: " + str(path))
        return dict(CACHE[key][1])
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(before == signature(path), "Evidence changed during hash: " + str(path))
    result = {"bytes": before[2], "sha256": digest}
    CACHE[key] = before, result
    BYTES_HASHED += before[2]
    return dict(result)


def read_json(path):
    identity(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def expected_identity(value):
    require(isinstance(value, dict) and type(value.get("bytes")) is int and value["bytes"] >= 0
            and isinstance(value.get("sha256"), str) and re.fullmatch("[0-9a-f]{64}", value["sha256"]),
            "Invalid expected file identity")
    return {"bytes": value["bytes"], "sha256": value["sha256"]}


def member_rows(document):
    files = document.get("files")
    if isinstance(files, dict):
        rows = list(files.items())
    elif isinstance(files, list):
        rows = [(row["path"], row) for row in files]
    else:
        raise ValueError("Unsupported manifest membership representation")
    names = [normalized_relative(name).as_posix() for name, _ in rows]
    require(bool(rows) and len(set(name.casefold() for name in names)) == len(names), "Empty/duplicate manifest membership")
    return [(name, expected_identity(row)) for name, (_, row) in zip(names, rows)]


def scope_not_elevated(document):
    # Native execution and partial synthetic agreement may be true; no blanket
    # rejection of the now-completed native-oracle fact is applied.
    for key in FALSE_SCOPE_KEYS:
        if key in document:
            require(document[key] is False, "Scientific/data/integration scope elevated: " + key)


def verify_source_bindings(document):
    verified = 0
    for field in ("sources", "source_identities", "external_workspace_sources", "final_sources"):
        values = document.get(field)
        if not isinstance(values, dict):
            continue
        for name, expected in values.items():
            if not isinstance(expected, dict) or "bytes" not in expected or "sha256" not in expected:
                continue
            candidate = Path(name)
            relative = candidate.relative_to(ROOT).as_posix() if candidate.is_absolute() else str(name)
            require(identity(safe(ROOT, relative)) == expected_identity(expected), "Reviewed source changed: " + relative)
            verified += 1
    return verified


def verify_manifest(relative, mode, count, expected_sha):
    path = safe(ROOT, relative)
    actual = identity(path)
    require(actual["sha256"] == expected_sha, "Frozen manifest identity changed: " + relative)
    document = read_json(path)
    rows = member_rows(document)
    require(count is None or len(rows) == count, "Manifest member count changed: " + relative)
    if "file_count" in document:
        require(document["file_count"] == len(rows), "Declared file_count inconsistent")
    base = ROOT if mode == "root" else path.parent
    members = []
    for name, expected in rows:
        require(not forbidden_volatile(name), "Volatile scope unexpectedly listed in sealed manifest: " + name)
        member = safe(base, name)
        require(identity(member) == expected, "Manifest member changed: " + name)
        members.append(member)
    # Intentionally do not require directory == member list. New appended audits
    # coexist with already-frozen source-review/preparation manifests.
    receipt = {"manifest": relative, "base": "engineering_project_root" if mode == "root" else path.parent.relative_to(ROOT).as_posix(),
               "membership_format": type(document["files"]).__name__, "declared_count": len(rows),
               "verified_count": len(rows), "extra_directory_files_not_part_of_this_manifest": "not traversed",
               **actual}
    return receipt, document, members


def add_small(top, path):
    relative = path.relative_to(ROOT).as_posix()
    require(not forbidden_volatile(relative), "Mutable/cache/test temporary evidence forbidden in top list")
    if path.suffix.casefold() in SMALL_SUFFIXES and signature(path)[2] <= MAX_TOP_BYTES:
        top.add(path)


def check_review(path):
    review = read_json(path)
    require(str(review.get("status", "")).startswith("PASS") and review.get("issues", []) == [], "Incomplete final/source review")
    scope_not_elevated(review)
    return {"path": path.relative_to(ROOT).as_posix(), "status": review["status"],
            "reviewed_source_bindings_verified": verify_source_bindings(review), **identity(path)}


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    require(config.get("schema") == "rampq.gate1.acceptance_prerequisites_seal_inputs/v1", "Missing reviewed seal input schema")
    require(set(config.get("bindings", {})) == set(EXPECTED_DYNAMIC_PATHS), "Dynamic seal binding keys differ")
    for key, value in config["bindings"].items():
        expected_identity(value)
        require(value.get("path") == EXPECTED_DYNAMIC_PATHS[key], "Unexpected dynamic path: " + key)
    require(isinstance(config.get("additional_reviews", []), list), "additional_reviews must be a list")
    for value in config.get("additional_reviews", []):
        expected_identity(value)
        normalized_relative(value.get("path", ""))
    return config


def readiness(config_path):
    names = [str(config_path.relative_to(ROOT).as_posix())] + [item[0] for item in FIXED_NESTED]
    names += list(EXPECTED_DYNAMIC_PATHS.values()) + list(LIVE_COMPONENTS)
    names += [BASE + "20261023_native_oracle_runtime_v1/runtime_manifest.json"]
    missing = [name for name in names if not (ROOT / normalized_relative(name).as_posix()).is_file()]
    if config_path.is_file():
        config = load_config(config_path)
        missing += [value["path"] for value in config.get("additional_reviews", []) if not (ROOT/value["path"]).is_file()]
    return sorted(set(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"seal_inputs.json")
    parser.add_argument("--check-readiness", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    require(config_path.is_relative_to(HERE), "Seal input config must stay in this new delivery directory")
    missing = readiness(config_path)
    if args.check_readiness:
        print(json.dumps({"status": "PATHS_PRESENT_NOT_YET_HASH_VERIFIED" if not missing else "WAITING_FOR_FIXED_INPUTS",
                          "missing_paths": missing, "large_files_hashed": False, "files_written": False}, ensure_ascii=False))
        return
    require(not missing, "Missing final inputs: " + json.dumps(missing, ensure_ascii=False))
    validation_path, manifest_path = HERE/"delivery_validation.json", HERE/"delivery_manifest.json"
    require(not validation_path.exists() and not manifest_path.exists(), "Delivery already sealed; preserve it")
    config_path = safe(ROOT, config_path.relative_to(ROOT).as_posix())
    config = load_config(config_path)
    identity(config_path)
    dynamic = {}
    top = {Path(__file__).resolve(), config_path}
    for key, binding in config["bindings"].items():
        path = safe(ROOT, binding["path"])
        require(identity(path) == expected_identity(binding), "Dynamic final identity differs: " + key)
        dynamic[key] = path
        add_small(top, path)
    for name, expected in LIVE_COMPONENTS.items():
        path = safe(ROOT, name)
        require(identity(path)["sha256"] == expected, "Live component differs from reviewed final source: " + name)
        top.add(path)
    nested = []
    for relative, mode, count, pin in FIXED_NESTED:
        receipt, document, members = verify_manifest(relative, mode, count, pin)
        nested.append(receipt)
        top.add(safe(ROOT, relative))
        # Prior small package is transitively retained; avoid copying its 122
        # members into this top list or traversing its old nested array manifests.
        if count != 122:
            for member in members:
                add_small(top, member)
        if relative.endswith("20261023_ignore_loss_preparation_v1/manifest.json"):
            verify_source_bindings(document)
    reviews = [check_review(dynamic["final_review"]), check_review(safe(ROOT, BASE + "20261023_acceptance_prerequisites_review_v1/source_review.json"))]
    for binding in config.get("additional_reviews", []):
        path = safe(ROOT, binding["path"])
        require(identity(path) == expected_identity(binding), "Additional final review differs")
        reviews.append(check_review(path))
        add_small(top, path)
    for key in ("state", "decision"):
        scope_not_elevated(read_json(dynamic[key]))

    runtime_path = dynamic["runtime_preparation_manifest"]
    runtime_receipt, runtime, members = verify_manifest(runtime_path.relative_to(ROOT).as_posix(), "local", 37, identity(runtime_path)["sha256"])
    require(runtime.get("status") == "SEALED_NATIVE_SYNTHETIC_ORACLE_EXECUTION_ONLY", "Runtime package not final")
    scope_not_elevated(runtime)
    nested.append(runtime_receipt)
    for member in members:
        add_small(top, member)
    runtime_rows_path = safe(ROOT, BASE + "20261023_native_oracle_runtime_v1/runtime_manifest.json")
    declared = runtime.get("transitive_runtime_manifest", {})
    require(identity(runtime_rows_path) == expected_identity(declared)
            and declared.get("bound_runtime_file_count") == 2890, "Runtime transitive identity/count differs")
    runtime_rows_receipt, _, _ = verify_manifest(runtime_rows_path.relative_to(ROOT).as_posix(), "root", 2890, declared["sha256"])
    nested.append(runtime_rows_receipt)
    top.add(runtime_rows_path)

    semantics_path = dynamic["native_semantics_manifest"]
    semantics_receipt, semantics, members = verify_manifest(semantics_path.relative_to(ROOT).as_posix(), "root", 10, identity(semantics_path)["sha256"])
    scope_not_elevated(semantics)
    nested.append(semantics_receipt)
    for member in members:
        add_small(top, member)

    partial = read_json(safe(ROOT, BASE + "20261023_native_python_semantics_v1/comparison_result_v1.json"))
    require(partial.get("status") == "MATCH_FIXED_SYNTHETIC_PARTIAL_INTERMEDIATES_ONLY"
            and partial.get("matrix_checks") == partial.get("equal_matrix_checks") == 166
            and partial.get("different_matrix_checks") == 0 and partial.get("fixture_count") == 20
            and partial.get("native_matching_fixture_count") == 5
            and partial.get("categories_per_matching_fixture") == 10
            and partial.get("no_AP_metric_values_compared") is True, "Partial native/Python metadata differs")
    scope_not_elevated(partial)
    for field in ("comparison_contract", "python_source", "native_structure_audit", "native_mat"):
        expected = partial[field]
        require(identity(safe(ROOT, expected["path"])) == expected_identity(expected), "Partial comparison input changed: " + field)

    native_audit_manifest_path = dynamic["native_audit_manifest"]
    audit_manifest_receipt, _, members = verify_manifest(native_audit_manifest_path.relative_to(ROOT).as_posix(), "root", 4,
                                                       identity(native_audit_manifest_path)["sha256"])
    nested.append(audit_manifest_receipt)
    for member in members:
        add_small(top, member)

    audit = read_json(dynamic["native_audit"])
    require(audit.get("status") == "PASS_NATIVE_SYNTHETIC_OUTPUT_STRUCTURE_IDENTITIES_ONLY"
            and audit.get("fixture_count") == 20 and audit.get("kinds") == {"aggregate": 11, "region": 4, "match": 5}
            and audit.get("MAT_authoritative_for_empty_dimensions_and_NaN") is True
            and audit.get("JSON_serialization_consistent") is True, "Native saved-output audit incomplete")
    scope_not_elevated(audit)
    scipy = audit["mat_reader_dependency_verification"]
    require(scipy.get("verified_extracted_member_count") == 1535
            and scipy.get("existing_Python_environment_modified") is False, "SciPy audit receipt differs")
    wheel = scipy["wheel"]
    wheel_path = Path(wheel["path"])
    require(wheel_path.is_absolute() and wheel_path.is_relative_to(ROOT), "Unexpected wheel scope")
    require(identity(safe(ROOT, wheel_path.relative_to(ROOT).as_posix())) == expected_identity(wheel), "Verified SciPy wheel changed")

    files = {path.relative_to(ROOT).as_posix(): identity(path) for path in sorted(top)}
    require(all(entry["bytes"] <= MAX_TOP_BYTES for entry in files.values()), "Top list must stay small; large packages are transitive")
    for key, (expected_stat, _) in CACHE.items():
        require(signature(Path(key)) == expected_stat, "Evidence changed immediately before sealing")
    validation = {
        "schema": "rampq.gate1.acceptance_prerequisites_delivery_validation/v1",
        "status": "PASS_COMPLETED_SYNTHETIC_COMPONENT_AND_NATIVE_EVIDENCE_BINDINGS_ONLY",
        "client_date": "2026-10-23", "at_utc": datetime.now(timezone.utc).isoformat(),
        "nested_manifests": nested, "reviews": reviews, "small_top_level_files": len(files),
        "unique_paths_hashed": len(CACHE), "total_file_bytes_hashed_once": BYTES_HASHED,
        "duplicate_path_hash_requests_served_from_cache": CACHE_HITS,
        "old_delivery_122_small_members_verified_without_old_array_traversal": True,
        "runtime_files_verified_as_opaque_bytes": 2890,
        "saved_partial_semantics_metadata": {"matrix_checks": 166, "equal_matrix_checks": 166,
                                             "different_matrix_checks": 0, "AP_metric_values_compared": False},
        "scipy_extracted_1535_files_rehashed_here": False,
        "scipy_dependency": {"policy": "Rely on the sealed native saved-output auditor's existing extracted-member SHA verification; rehash official wheel once here.",
                              "native_audit": identity(dynamic["native_audit"]), "wheel": expected_identity(wheel),
                              "audited_extracted_member_count": 1535},
        "hashing_policy": "Opaque byte SHA256 only; stat-guarded per-resolved-path cache. No archive/MAT/model/tensor/image/annotation decoding and no old multi-GB NPZ traversal.",
        "large_runtime_and_wheel_files_bound_transitively": True,
        "mutable_latest_caches_and_test_temporaries_excluded": True,
        "model_or_real_data_executed_or_parsed": False, "selection_detector_inference": False,
        "holdout_opened_or_listed": False, "gate1_scientific_pass": False,
        "accepted_float_baseline_ready": False, "official_AP_equivalence_certified": False,
        "complete_ignore_aware_training_solved": False, "training_pipeline_integrated": False,
        "live_training_hook_integrated": False,
        "old_gate05_guard": "FAILED_UNCHANGED", "original_gate1_dtype_guard": "FAIL_FIXED_TOLERANCES",
        "limits": ["Integrity and completed bounded synthetic engineering evidence only; no new efficacy/accuracy/AP or deployment acceptance.",
                   "Native saved-output structure and partial Python matrix agreement do not certify full Python AP.",
                   "The loss primitive is not a complete augmented training pipeline; the FP32 state format does not prove live provenance or recover old FP16 precision."]}
    with validation_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    files[validation_path.relative_to(ROOT).as_posix()] = identity(validation_path)
    manifest = {
        "schema": "rampq.gate1.acceptance_prerequisites_delivery_manifest/v1",
        "status": "COMPLETED_BOUNDED_TRAINING_PRIMITIVES_AND_NATIVE_SYNTHETIC_ORACLE_ONLY",
        "client_date": "2026-10-23", "at_utc": datetime.now(timezone.utc).isoformat(),
        "file_base": "engineering_project_root", "files": dict(sorted(files.items())),
        "nested_manifests_verified": nested, "gate1_scientific_pass": False,
        "accepted_float_baseline_ready": False, "official_AP_equivalence_certified": False,
        "selection_detector_inference": False, "holdout_opened_or_listed": False,
        "complete_ignore_aware_training_solved": False, "training_pipeline_integrated": False,
        "live_training_hook_integrated": False,
        "original_gate1_dtype_guard": "FAIL_FIXED_TOLERANCES", "old_gate05_guard": "FAILED_UNCHANGED"}
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": manifest["status"], "small_files": len(files),
                      "nested_manifests_verified": len(nested), "runtime_files_verified": 2890,
                      "manifest_sha256": identity(manifest_path)["sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
