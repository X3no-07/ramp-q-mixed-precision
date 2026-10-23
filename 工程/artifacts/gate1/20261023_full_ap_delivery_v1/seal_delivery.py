"""Seal bounded full-AP synthetic agreement and ignore-geometry preparation.

Prepared for root review; execution is separate. --check-readiness writes
nothing and hashes no files. Normal execution requires a separately reviewed
seal_inputs.json binding final paths, sizes and SHA256 identities. Validation
is written first and the final delivery manifest is the last write.

Only opaque file bytes and small JSON metadata are read. No model, runtime,
test, MAT reader, real image/annotation, tensor, or archive is executed/decoded.
The old 125-member small delivery is verified without following its historical
large-array/dependency manifests. Runtime/SciPy extraction checks rely on the
sealed comparator/launcher receipts; their large trees are not rescanned.
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
DELIVERY = BASE + "20261023_full_ap_delivery_v1/"
AUDIT = BASE + "20261023_full_ap_oracle_audit_v1/"
REVIEW = BASE + "20261023_full_ap_final_review_v1/"
MAX_FILE_BYTES = 16 * 1024 * 1024
EXPECTED_PATHS = {
    "report": "49_Gate1_AP聚合原生对照与忽略几何验证.md",
    "state": BASE + "current_execution_state_v10.json",
    "decision": DELIVERY + "decision.json",
    "build_result": DELIVERY + "build_result.py",
    "final_review": REVIEW + "final_review.json",
    "source_review": REVIEW + "source_review.json",
    "ap_preparation_manifest": BASE + "20261023_full_ap_preparation_v1/preparation_manifest.json",
    "geometry_manifest": BASE + "20261023_ignore_geometry_preparation_v1/manifest.json",
    "audit_manifest": AUDIT + "audit_delivery_manifest.json",
    "stress_manifest": AUDIT + "native_stress_execution_v1/manifest.json",
    "inputs_manifest": AUDIT + "inputs_manifest.json",
    "comparator_manifest": AUDIT + "comparator_preparation_manifest.json",
    "comparison_result": AUDIT + "comparison_attempt_v1.json",
    "comparison_summary": AUDIT + "audit_summary.json",
    "comparison_contract": AUDIT + "comparison_contract_v1.json",
    "native_launcher": AUDIT + "native_stress_execution_v1/launcher_receipt.json",
    "native_constants": DELIVERY + "native_constants_probe.json",
    "prior_identity_check": DELIVERY + "prior_delivery_identity_check.json",
    "launcher_source": DELIVERY + "launch_stress.py",
    "constants_source": DELIVERY + "inspect_native_constants.py",
}
# Bases are declared by source schemas, never inferred from path spelling.
MANIFEST_KINDS = {
    "rampq.gate1.acceptance_prerequisites_delivery_manifest/v1": ("root", 125),
    "rampq.gate1.full_ap_preparation_manifest/v1": ("root", 7),
    "rampq.gate1.ignore_geometry_preparation_manifest/v1": ("local", 17),
    "rampq.gate1.full_ap_synthetic_audit_delivery_manifest/v1": ("root", 25),
    "rampq.gate1.full_ap_native_stress_output_manifest/v1": ("local", 4),
    "rampq.gate1.full_ap_synthetic_inputs_manifest/v1": ("root", 6),
    "rampq.gate1.full_ap_comparator_preparation/v1": ("root", 7),
}
NESTED_BINDINGS = ("ap_preparation_manifest", "geometry_manifest", "audit_manifest", "stress_manifest", "inputs_manifest", "comparator_manifest")
PRIOR_PATH = BASE + "20261023_acceptance_prerequisites_delivery_v1/delivery_manifest.json"
PRIOR_SHA = "12cd0fde964a34d0890a269741255d715e2e299d0f4a55b1dab2cf7cac18d07f"
LIVE_SOURCES = {
    "scripts/gate1_official_ap.py": "a76a3ae45f2859fecc89909f903011b483514ee7113b93b881947fc9dbcbb54c",
    "tests/test_gate1_official_ap.py": "055fb6955afcf3869e5b7157e3b4ad1ce5494646df04b7c5823e073f253d63c3",
    "scripts/gate1_ignore_geometry.py": "4ecb3eedc4c5597f64e2290d73de4d6a4d40f4cd265e06c0dbb0ca03371dfd56",
    "tests/test_gate1_ignore_geometry.py": "675f1627b11af41295b3b8dd475baf697c8dbb4ae00d4822c75a24104bcb1df8",
}
FALSE_SCOPE_KEYS = {
    "gate1_scientific_pass", "scientific_gate1_passed", "scientific_gate_pass", "accepted_float_baseline_ready",
    "official_AP_equivalence_certified", "arbitrary_input_equivalence_certified", "official_AP_arbitrary_input_equivalence_certified",
    "real_data_AP_computed", "new_precision_model_accepted_or_deployed", "new_precision_model_deployed",
    "selection_detector_inference", "holdout_accessed", "holdout_opened_or_listed", "selection_or_holdout_accessed",
    "training_pipeline_integrated", "complete_ignore_aware_training_solved", "live_training_hook_integrated",
    "original_dtype_guard_modified", "native_internal_AP_AR_tables_compared",
}
CACHE: dict[str, tuple[tuple, dict]] = {}
BYTES_HASHED = 0
CACHE_HITS = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_name(name):
    value = PurePosixPath(str(name).replace("\\", "/"))
    require(bool(value.parts) and not value.is_absolute() and ".." not in value.parts
            and not any(":" in part for part in value.parts), "Unsafe relative path")
    parts = [part.casefold() for part in value.parts]
    require(not any(any(token in part for token in ("holdout", "test-dev", "test_dev")) for part in parts),
            "Holdout/test-dev must remain opaque")
    require(not any(part in {"__pycache__", ".pytest_cache", "test_temp", "test_cases", "temp", "tmp", "runtime_config", "mpl_config"}
                    or part.startswith("test_temp_") for part in parts)
            and "latest_execution_state.json" not in parts, "Mutable cache/test temporary path forbidden")
    return value.as_posix()


def safe(base, name):
    path = base
    for part in PurePosixPath(relative_name(name)).parts:
        path = path / part
        require(not path.is_symlink() and not path.is_junction(), "Reparse path forbidden")
    path = path.resolve(strict=True)
    require(path.is_relative_to(ROOT) and path.is_file(), "Path leaves workspace or is not regular")
    return path


def signature(path):
    value = path.stat()
    require(stat.S_ISREG(value.st_mode), "Not a regular file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def identity(path):
    global BYTES_HASHED, CACHE_HITS
    path = safe(ROOT, path.relative_to(ROOT).as_posix())
    key, before = str(path).casefold(), signature(path)
    require(before[2] <= MAX_FILE_BYTES, "This sealer admits small evidence files only")
    if key in CACHE:
        CACHE_HITS += 1
        require(CACHE[key][0] == before, "Previously checked evidence changed")
        return dict(CACHE[key][1])
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(before == signature(path), "Evidence changed during hashing")
    result = {"bytes": before[2], "sha256": digest}
    CACHE[key] = before, result
    BYTES_HASHED += before[2]
    return dict(result)


def expected_identity(value):
    require(isinstance(value, dict) and type(value.get("bytes")) is int and 0 <= value["bytes"] <= MAX_FILE_BYTES
            and isinstance(value.get("sha256"), str) and re.fullmatch("[0-9a-f]{64}", value["sha256"]),
            "Missing valid small-file identity")
    return {"bytes": value["bytes"], "sha256": value["sha256"]}


def read_json(path):
    identity(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    identity(path)
    return value


def checked_binding(binding):
    path = safe(ROOT, binding["path"])
    require(identity(path) == expected_identity(binding), "Bound identity changed: " + binding["path"])
    return path


def verify_source_maps(document):
    checked = []
    for key in ("sources", "source_pins", "workspace_sources", "source_identities", "external_workspace_sources", "final_sources"):
        values = document.get(key)
        if not isinstance(values, dict):
            continue
        for name, expected in values.items():
            if isinstance(expected, dict) and "bytes" in expected and "sha256" in expected:
                path = safe(ROOT, name)
                require(identity(path) == expected_identity(expected), "Reviewed source changed: " + name)
                checked.append(path)
    return checked


def manifest_rows(document):
    files = document.get("files")
    rows = list(files.items()) if isinstance(files, dict) else [(row["path"], row) for row in files] if isinstance(files, list) else None
    require(bool(rows), "Missing manifest members")
    names = [relative_name(name) for name, _ in rows]
    require(len(set(name.casefold() for name in names)) == len(names), "Duplicate manifest member")
    if "file_count" in document:
        require(document["file_count"] == len(rows), "Declared manifest count differs")
    return [(name, expected_identity(row)) for name, (_, row) in zip(names, rows)]


def verify_manifest(path, *, optional_base=None, optional_count=None):
    document = read_json(path)
    if optional_base is None:
        require(document.get("schema") in MANIFEST_KINDS, "Unrecognized manifest schema/base")
        mode, count = MANIFEST_KINDS[document["schema"]]
    else:
        mode, count = optional_base, optional_count
        require(mode in {"root", "local"}, "Invalid additional manifest base")
    declared_base = document.get("root", document.get("path_base", document.get("file_base")))
    if declared_base is not None:
        allowed = {"root": {"workspace", "engineering_project_root", "root"}, "local": {"local", "manifest_directory"}}
        require(declared_base in allowed[mode], "Manifest root/schema metadata conflict")
    rows = manifest_rows(document)
    require(count is None or len(rows) == count, "Manifest count differs")
    base = ROOT if mode == "root" else path.parent
    members = []
    for name, expected in rows:
        member = safe(base, name)
        require(identity(member) == expected, "Manifest member changed: " + name)
        members.append(member)
    members += verify_source_maps(document)
    return {"path": path.relative_to(ROOT).as_posix(), **identity(path), "schema": document["schema"],
            "member_base": mode, "membership_format": type(document["files"]).__name__, "verified_members": len(rows)}, document, members


def scope_not_elevated(document):
    if isinstance(document, dict):
        for key, value in document.items():
            if key in FALSE_SCOPE_KEYS:
                require(value is False, "Scientific/arbitrary-input/integration scope elevated: " + key)
            scope_not_elevated(value)
    elif isinstance(document, list):
        for value in document:
            scope_not_elevated(value)


def check_review(path):
    document = read_json(path)
    require(str(document.get("status", "")).startswith("PASS") and document.get("issues", []) == [], "Review not final PASS")
    scope_not_elevated(document)
    sources = verify_source_maps(document)
    require(bool(sources), "Final/source review must bind actual sources")
    return {"path": path.relative_to(ROOT).as_posix(), **identity(path), "status": document["status"],
            "source_bindings_verified": len(sources)}, sources


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    require(config.get("schema") == "rampq.gate1.full_ap_delivery_seal_inputs/v1", "Wrong seal input schema")
    require(set(config.get("bindings", {})) == set(EXPECTED_PATHS), "Seal binding keys differ")
    for key, binding in config["bindings"].items():
        require(binding.get("path") == EXPECTED_PATHS[key], "Unexpected dynamic path: " + key)
        expected_identity(binding)
    for key in ("additional_reviews", "additional_manifests", "additional_small_files"):
        require(isinstance(config.get(key, []), list), key + " must be a list")
        for binding in config.get(key, []):
            relative_name(binding["path"])
            expected_identity(binding)
    return config


def readiness(config_path):
    names = [config_path.relative_to(ROOT).as_posix(), PRIOR_PATH, *EXPECTED_PATHS.values(), *LIVE_SOURCES]
    if config_path.is_file():
        config = load_config(config_path)
        names += [binding["path"] for key in ("additional_reviews", "additional_manifests", "additional_small_files") for binding in config.get(key, [])]
    return sorted({name for name in names if not (ROOT / relative_name(name)).is_file()})


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "seal_inputs.json")
    parser.add_argument("--check-readiness", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    require(config_path.is_relative_to(HERE), "Config must stay in this new delivery directory")
    missing = readiness(config_path)
    if args.check_readiness:
        print(json.dumps({"status": "PATHS_PRESENT_NOT_HASH_VERIFIED" if not missing else "WAITING_FOR_FIXED_INPUTS",
                          "missing_paths": missing, "files_hashed": False, "files_written": False}, ensure_ascii=False))
        return
    require(not missing, "Final inputs missing: " + json.dumps(missing, ensure_ascii=False))
    validation_path, manifest_path = HERE / "delivery_validation.json", HERE / "delivery_manifest.json"
    require(not validation_path.exists() and not manifest_path.exists(), "Preserve existing sealed outputs")
    config = load_config(config_path)
    top = {safe(ROOT, Path(__file__).resolve().relative_to(ROOT).as_posix()), safe(ROOT, config_path.relative_to(ROOT).as_posix())}
    dynamic = {key: checked_binding(binding) for key, binding in config["bindings"].items()}
    top.update(dynamic.values())
    prior_path = safe(ROOT, PRIOR_PATH)
    require(identity(prior_path)["sha256"] == PRIOR_SHA, "Prior delivery manifest changed")
    prior_receipt, _, _ = verify_manifest(prior_path)
    top.add(prior_path)
    nested = [prior_receipt]
    nested_documents = {}
    for key in NESTED_BINDINGS:
        receipt, document, members = verify_manifest(dynamic[key])
        nested.append(receipt)
        nested_documents[key] = document
        top.update(members)
    for name, pin in LIVE_SOURCES.items():
        path = safe(ROOT, name)
        require(identity(path)["sha256"] == pin, "Reviewed stable source changed: " + name)
        top.add(path)
    reviews = []
    for path in [dynamic["source_review"], dynamic["final_review"], *(checked_binding(row) for row in config.get("additional_reviews", []))]:
        receipt, sources = check_review(path)
        reviews.append(receipt)
        top.add(path)
        top.update(sources)
    for binding in config.get("additional_manifests", []):
        path = checked_binding(binding)
        receipt, _, members = verify_manifest(path, optional_base=binding.get("member_base"), optional_count=binding.get("file_count"))
        nested.append(receipt)
        top.add(path)
        top.update(members)
    top.update(checked_binding(row) for row in config.get("additional_small_files", []))
    # Scope checks do not reject implemented full AP or bounded synthetic passes.
    # They reject elevating them to arbitrary-input/real-data/scientific claims.
    state, decision = read_json(dynamic["state"]), read_json(dynamic["decision"])
    for document in (state, decision, *nested_documents.values()):
        scope_not_elevated(document)
    require(state.get("same_backend_dtype_guard") == "FAIL_FIXED_TOLERANCES"
            and state.get("old_gate05_numeric_guard") == "FAILED_UNCHANGED", "Original failed guards must remain unchanged")
    summary = read_json(dynamic["comparison_summary"])
    result = read_json(dynamic["comparison_result"])
    contract = read_json(dynamic["comparison_contract"])
    scope_not_elevated(summary)
    scope_not_elevated(result)
    require(summary.get("status") == result.get("status") == "PASS_DECLARED_SYNTHETIC_FULL_AP_NATIVE_COMPARISON_ONLY", "Comparison not completed")
    require(summary.get("case_count") == 32 and summary.get("aggregate_case_count") == 23 and summary.get("image_count") == 40
            and summary.get("matrix_count_exact") == result.get("matrix_count") == 220
            and summary.get("metric_count") == result.get("metric_count") == 161
            and summary.get("finite_metric_count") == 133 and summary.get("matching_nan_metric_count") == 28, "Saved comparison counts differ")
    expected_tol = {"atol": 1e-10, "rtol": 1e-12, "reference": "native", "unit": "percentage"}
    require(summary.get("metric_tolerance") == result.get("float64_final_metric_tolerance") == expected_tol, "Predeclared metric tolerance differs")
    require(summary.get("matrix_tolerance") == {"atol": 0, "rtol": 0, "shape_exact": True, "dtype": "float64"}, "Matrix tolerance changed")
    require(summary.get("max_finite_metric_absolute_error_percentage_units") == 7.105427357601002e-15, "Saved maximum error differs")
    for row in (summary["result"], summary["contract"], result["policy"], result["contract"], result["implementation"]):
        top.add(checked_binding(row))
    require(contract["implementation"]["sha256"] == LIVE_SOURCES["scripts/gate1_official_ap.py"], "Comparator used another implementation")
    launcher = read_json(dynamic["native_launcher"])
    require(launcher.get("status") == "EXECUTED_12_NATIVE_STRESS_CASES_ONLY" and launcher.get("returncode") == 0
            and launcher.get("runtime_files_verified") == 2890 and launcher.get("fixture_count") == 12
            and launcher.get("native_case_errors") == 0, "Native launch verification not complete")
    reader = result["reader"]["dependency_verification"]
    require(reader.get("verified_extracted_member_count") == 1535 and reader.get("existing_Python_environment_modified") is False,
            "Actual comparator reader-file verification receipt differs")
    # The audit manifest's 1656 count is the ZIP's entries including directories;
    # the comparator's actual verification receipt counts 1535 extracted files.
    require(nested_documents["audit_manifest"].get("native_runtime_members_verified_by_comparator") == 2890
            and nested_documents["audit_manifest"].get("isolated_reader_members_verified_by_comparator") == 1656,
            "Audit dependency entry metadata differs")
    files = {path.relative_to(ROOT).as_posix(): identity(path) for path in sorted(top)}
    for key, (expected_stat, _) in CACHE.items():
        require(signature(Path(key)) == expected_stat, "Evidence changed immediately before seal")
    common = {"client_date": "2026-10-23", "at_utc": datetime.now(timezone.utc).isoformat(),
              "gate1_scientific_pass": False, "accepted_float_baseline_ready": False,
              "arbitrary_input_equivalence_certified": False, "real_data_AP_computed": False,
              "training_pipeline_integrated": False, "selection_detector_inference": False, "holdout_opened_or_listed": False,
              "original_gate1_dtype_guard": "FAIL_FIXED_TOLERANCES", "old_gate05_guard": "FAILED_UNCHANGED"}
    validation = {"schema": "rampq.gate1.full_ap_delivery_validation/v1", **common,
        "status": "PASS_BOUNDED_SYNTHETIC_AP_AND_GEOMETRY_EVIDENCE_BINDINGS_ONLY",
        "nested_manifests": nested, "reviews": reviews, "small_files_before_validation": len(files),
        "unique_files_hashed": len(CACHE), "total_bytes_hashed_once": BYTES_HASHED, "duplicate_hash_requests_cached": CACHE_HITS,
        "prior_125_small_members_verified_without_deep_array_traversal": True,
        "synthetic_comparison_metadata": {"cases": 32, "aggregate_cases": 23, "exact_matrices": 220, "metrics": 161,
            "matching_nan_metrics": 28, "max_finite_metric_error_percentage_units": 7.105427357601002e-15,
            "full_AP_implemented": True, "declared_synthetic_agreement": True, "native_internal_AP_AR_tables_compared": False},
        "large_dependencies_not_rescanned_here": {"runtime_files_preverified": 2890, "reader_files_preverified": 1535,
            "reader_wheel_entries_including_directories": 1656, "receipt": dynamic["comparison_result"].relative_to(ROOT).as_posix(),
            "policy": "Trust sealed actual launcher/comparator verification receipts; no repeated archive/tree scan"},
        "mutable_latest_caches_and_test_temporaries_excluded": True,
        "model_or_real_data_executed_or_parsed": False,
        "hashing_policy": "Small opaque-byte SHA256, per-path stat guard/cache; no MAT/tensor/image/archive parsing or numerical recomputation",
        "limits": ["Finite synthetic cases and chosen native runtime only; no arbitrary-input or real-data AP certification.",
                   "Geometry preparation is not a completed Dataset/Trainer/criterion integration.",
                   "Scientific baseline and board deployment acceptance remain separate and unfinished."]}
    write_new(validation_path, validation)
    files[validation_path.relative_to(ROOT).as_posix()] = identity(validation_path)
    manifest = {"schema": "rampq.gate1.full_ap_delivery_manifest/v1", **common,
                "status": "COMPLETED_BOUNDED_FULL_AP_NATIVE_SYNTHETIC_AND_GEOMETRY_PREPARATION_ONLY",
                "file_base": "engineering_project_root", "files": dict(sorted(files.items())), "file_count": len(files),
                "nested_manifests_verified": nested, "full_Python_AP_implemented": True, "declared_synthetic_output_agreement": True}
    write_new(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "small_files": len(files), "nested_manifests_verified": len(nested),
                      "manifest_sha256": identity(manifest_path)["sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
