"""Freeze source/receipt/file identities, without reading native metric arrays."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_MANIFEST = ROOT / "artifacts/gate1/20261022_split_refinement/evaluation_sources/source_manifest.json"
RUNTIME = ROOT / "artifacts/gate1/20261023_native_oracle_runtime_v1"
STRUCTURAL = ROOT / "artifacts/gate1/20261023_native_oracle_comparison_preparation_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    assert sha(HERE / "comparison_policy.json") == "ff75048e986f5ec4bd2e956fca41a2370eb62b525b62b386cf3bd07f192f0b70"
    assert sha(HERE / "inputs_manifest.json") == "1d77ec04575579a1233e230f30ab65eb0d82537938ae5ce347459d90bae7fb91"
    assert sha(ROOT / "scripts/gate1_official_ap.py") == "a76a3ae45f2859fecc89909f903011b483514ee7113b93b881947fc9dbcbb54c"
    assert sha(ROOT / "artifacts/gate1/20261023_full_ap_preparation_v1/preparation_manifest.json") == "27ac8449421b67b6d8a86c5903f3f1e045a9350fa5402ab06b137e5358f2ad1e"
    assert sha(SOURCE_MANIFEST) == "1d45d3ecaabf5f3f8a8885436a0a22718a890a17f8ad9d5ec4f9c274c3c079ae"
    assert sha(RUNTIME / "runtime_manifest.json") == "2490915f6f61adfde6fe85a5ba7cd249d77877a4461b7c42e418fe05d656d9e2"
    tests = ET.parse(HERE / "comparator_regression_v2.xml").getroot()
    count = sum(int(s.attrib["tests"]) for s in tests.iter("testsuite"))
    assert count == 45
    assert all(int(s.attrib[key]) == 0 for s in tests.iter("testsuite") for key in ("errors", "failures", "skipped"))
    source_review = {"schema": "rampq.gate1.full_ap_independent_source_review/v1",
                     "status": "NO_BLOCKER_BEFORE_NATIVE_COMPARISON", "implementation": identity(ROOT / "scripts/gate1_official_ap.py"),
                     "official_source_manifest": identity(SOURCE_MANIFEST),
                     "reviewed_semantics": ["cap submitted rows globally before category selection and sorting",
                         "prepared GT score inversion", "evalRes full matrix flags, ignored last, score-first stable ties",
                         "raw continuous XYWH with IoF for ignored GT", "recall denominator includes ignored GT rows",
                         "AP from maxDets500 and VOC precision envelope integration", "evalClass repeats once per class occurrence per image",
                         "NaN for empty evalClass", "ten thresholds and four maxDets", "raw adapter reuses only pinned raster prefilter"],
                     "domain_limits": ["finite binary64 inputs, finite geometry arithmetic and declared category/ignore fields",
                         "raw adapter inherits positive prediction widths and heights and integer category-zero region coordinates",
                         "stable tie policy tested only against pinned Octave runtime, not all MATLAB releases"],
                     "issues": [], "native_metric_answers_used_to_review_or_change_implementation": False,
                     "native_internal_tables_verified": False, "gate1_scientific_pass": False}
    write_new(HERE / "implementation_source_review.json", source_review)
    frozen_files = [HERE / name for name in ("compare_full_ap.py", "test_compare_full_ap.py", "comparator_regression_v1.xml",
                    "comparator_regression_v2.xml", "prepare_comparison_contract.py", "implementation_source_review.json", "inputs_manifest.json")]
    preparation = {"schema": "rampq.gate1.full_ap_comparator_preparation/v1", "status": "FROZEN_BEFORE_METRIC_COMPARISON",
                   "auditor_synthetic_regression": {"tests": 45, "failures": 0, "errors": 0, "skipped": 0},
                   "tests_use_fabricated_auditor_inputs_not_oracle_answers": True,
                   "files": [identity(path) for path in frozen_files], "file_count": len(frozen_files),
                   "policy": identity(HERE / "comparison_policy.json"), "native_metric_arrays_loaded_by_this_preparer": False}
    write_new(HERE / "comparator_preparation_manifest.json", preparation)
    policy = read(HERE / "comparison_policy.json")
    suites = []
    for declared, execution in zip(policy["suites"], (RUNTIME / "execution_01", HERE / "native_stress_execution_v1")):
        suite = {"id": declared["id"], "fixture": declared["fixture"],
                 "native_mat": identity(execution / "oracle_results.mat"), "native_json": identity(execution / "oracle_results.json"),
                 "launcher_receipt": identity(execution / "launcher_receipt.json")}
        if declared["id"] == "frozen20":
            suite["native_structure_audit"] = identity(STRUCTURAL / "native_output_audit_v1.json")
        suites.append(suite)
    sources = [{"path": str((SOURCE_MANIFEST.parent / row["path"]).relative_to(ROOT)).replace("\\", "/"),
                "bytes": row["bytes"], "sha256": row["sha256"]} for row in read(SOURCE_MANIFEST)["files"]]
    contract = {"schema": "rampq.gate1.full_ap_synthetic_comparison_contract/v1", "policy": identity(HERE / "comparison_policy.json"),
                "auditor": identity(HERE / "compare_full_ap.py"), "comparator_preparation": identity(HERE / "comparator_preparation_manifest.json"),
                "implementation": identity(ROOT / "scripts/gate1_official_ap.py"),
                "implementation_preparation_manifest": identity(ROOT / "artifacts/gate1/20261023_full_ap_preparation_v1/preparation_manifest.json"),
                "implementation_dependencies": [identity(ROOT / "scripts/gate1_ignore_aware_evaluation.py")],
                "structural_auditor": identity(STRUCTURAL / "audit_native_oracle.py"),
                "official_source_manifest": identity(SOURCE_MANIFEST), "official_sources": sources,
                "runtime_manifest": identity(RUNTIME / "runtime_manifest.json"),
                "runtime_executable": identity(RUNTIME / "runtime/octave-11.3.0-w64/mingw64/bin/octave-cli.exe"),
                "native_mean2": identity(RUNTIME / "runtime/octave-11.3.0-w64/mingw64/share/octave/packages/image-2.20.0/mean2.m"),
                "suites": suites, "output_name": "comparison_attempt_v1.json", "gate1_scientific_pass": False}
    write_new(HERE / "comparison_contract_v1.json", contract)
    print("CONTRACT_SHA256", sha(HERE / "comparison_contract_v1.json"))
    print("AUDITOR_SHA256", sha(HERE / "compare_full_ap.py"))
    print("PREPARATION_SHA256", sha(HERE / "comparator_preparation_manifest.json"))


if __name__ == "__main__":
    main()
