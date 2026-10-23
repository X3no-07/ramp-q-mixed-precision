"""Seal synthetic-output auditor preparation, without executing any oracle."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def report(path):
    suites = ET.parse(path).getroot()
    result = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    assert result["failures"] == result["errors"] == result["skipped"] == 0
    return result


def main():
    final_regression = HERE / "auditor_synthetic_regression_v4.xml"
    old_regression = HERE / "existing_python_regression.xml"
    current, old = report(final_regression), report(old_regression)
    assert current["tests"] == 39 and old["tests"] == 37
    dependency = json.loads((HERE / "mat_reader_dependency_receipt.json").read_text(encoding="utf-8"))
    preparation = {
        "schema": "rampq.gate1.native_synthetic_oracle_auditor_preparation/v1",
        "client_date": "2026-10-23", "status": "AUDITOR_PREPARED_NATIVE_OUTPUT_NOT_YET_AUDITED",
        "data_role": "synthetic", "fixture_count": 20,
        "auditor": identity(HERE / "audit_native_oracle.py"),
        "tests_source": identity(HERE / "test_audit_native_oracle.py"),
        "auditor_synthetic_regression": {**current, "result": identity(final_regression)},
        "existing_Python_semantic_regression": {**old, "result": identity(old_regression),
                                                "oracle_equivalence_proven": False},
        "native_launcher_schema_expected": "rampq.gate1.native_synthetic_oracle_launcher/v2",
        "runtime_manifest_schema_expected": "rampq.gate1.octave_portable_runtime_files/v1",
        "auditor_success_scope": "Saved native synthetic output structure, source/runtime identities and MAT/JSON serialization consistency only",
        "synthetic_test_metric_values_are_fabricated_auditor_inputs_not_oracle_answers": True,
        "earlier_attempts": [
            {"xml": "auditor_synthetic_regression.xml", "result": "36 pass, 1 fixture setup error: default pytest temporary directory was not writable"},
            {"xml": "auditor_synthetic_regression_v2.xml", "result": "37 pass before final nested-MAT and dependency-integrity tests were added"},
            {"xml": "auditor_synthetic_regression_v3.xml", "result": "37 pass, 2 test configuration failures: long MAT struct fields and directory-versus-file ZIP count; fixed in final tests"}],
        "mat_reader_dependency": dependency,
        "frozen_readiness_manifest": identity(ROOT / "artifacts/gate1/20261023_official_evaluator_readiness/manifest.json"),
        "official_source_manifest": identity(ROOT / "artifacts/gate1/20261022_split_refinement/evaluation_sources/source_manifest.json"),
        "official_AP_equivalence_certified": False, "existing_Python_matcher_certified": False,
        "full_Python_AP_implemented": False, "gate1_scientific_pass": False,
        "model_executed": False, "real_image_annotation_or_archive_read": False,
        "selection_or_holdout_accessed": False, "old_sources_modified": False,
        "existing_interpreter_packages_modified": False,
    }
    target = HERE / "preparation.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(preparation, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    names = ["AUDIT_DESIGN.md", "audit_native_oracle.py", "test_audit_native_oracle.py", "fetch_mat_reader.py",
             "freeze_preparation.py", "preparation.json", "existing_python_regression.xml",
             "auditor_synthetic_regression.xml", "auditor_synthetic_regression_v2.xml",
             "auditor_synthetic_regression_v3.xml", "auditor_synthetic_regression_v4.xml",
             "mat_reader_dependency_receipt.json", "mat_reader_wheel_listing.json",
             "scipy-1.18.1-cp312-cp312-win_amd64.whl"]
    manifest = {"schema": "rampq.gate1.native_synthetic_oracle_auditor_preparation_manifest/v1",
                "file_count": len(names), "path_base": "workspace", "files": [identity(HERE / name) for name in names],
                "isolated_extracted_SciPy_files_bound_transitively_to_official_wheel": True,
                "fabricated_test_temp_MAT_files_are_not_native_outputs_and_are_not_delivery_evidence": True}
    with (HERE / "preparation_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"preparation_manifest": identity(HERE / "preparation_manifest.json"), "auditor": preparation["auditor"]}))


if __name__ == "__main__":
    main()
