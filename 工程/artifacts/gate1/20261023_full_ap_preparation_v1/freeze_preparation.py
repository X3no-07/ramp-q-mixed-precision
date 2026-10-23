"""Freeze source and pure-synthetic test evidence before any native comparison."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    implementation = ROOT / "scripts/gate1_official_ap.py"
    tests = ROOT / "tests/test_gate1_official_ap.py"
    official = ROOT / "artifacts/gate1/20261022_split_refinement/evaluation_sources/official_toolkit"
    dependencies = [ROOT / "scripts/gate1_ignore_aware_evaluation.py",
                    official / "utils/calcAccuracy.m", official / "utils/VOCap.m",
                    official / "utils/evalRes.m", official / "utils/compOas.m",
                    official / "utils/dropObjectsInIgr.m", official / "utils/createIntImg.m",
                    ROOT / "artifacts/gate1/20261023_official_evaluator_readiness/synthetic_fixtures.json",
                    ROOT / "artifacts/gate1/20261023_official_evaluator_readiness/prepare_synthetic_fixtures.py",
                    ROOT / "artifacts/gate1/20261023_official_evaluator_readiness/run_synthetic_oracle.m",
                    ROOT / "artifacts/gate1/20261023_full_ap_delivery_v1/native_constants_probe.json"]
    expected_prefilter = "4e45ff26a86a83ff66e997f15e4c4794246c6440c304446972809ff32db09961"
    assert identity(dependencies[0])["sha256"] == expected_prefilter
    suites = list(ET.parse(HERE / "synthetic_tests.xml").getroot().iter("testsuite"))
    test_count = sum(int(suite.attrib["tests"]) for suite in suites)
    assert test_count == 54
    assert all(int(suite.attrib["errors"]) == int(suite.attrib["failures"]) == int(suite.attrib["skipped"]) == 0 for suite in suites)
    source_pins = {path.relative_to(ROOT).as_posix(): identity(path) for path in [implementation, tests, *dependencies]}
    receipt = {"schema": "rampq.gate1.full_ap_preparation/v1", "status": "FROZEN_BEFORE_NATIVE_COMPARISON",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "official_commit": "005445782213e20cb91bc50a597db3dd949e749a",
        "test_count": test_count, "test_failures": 0, "source_pins": source_pins,
        "test_data_role": "synthetic", "native_metric_answers_read_before_freeze": False,
        "native_matcher_answers_read_before_freeze": False,
        "native_values_inspected": "threshold float64 constants only, pure arithmetic Octave 11.3.0 probe",
        "real_data_read": False, "model_executed": False, "GPU_used": False,
        "old_sealed_evidence_modified": False, "official_AP_equivalence_certified": False,
        "comparison_status": "NOT_YET_RUN_AFTER_THIS_FREEZE",
        "scope": "Complete source translation and causal synthetic tests only; independent oracle audit is separate"}
    write_new(HERE / "preparation_receipt.json", receipt)
    files = [implementation, tests, *sorted(HERE.glob("*"))]
    members = {path.relative_to(ROOT).as_posix(): identity(path) for path in files if path.is_file()}
    write_new(HERE / "preparation_manifest.json", {"schema": "rampq.gate1.full_ap_preparation_manifest/v1",
        "root": "workspace", "files": members, "file_count": len(members),
        "frozen_before_comparison": True, "native_equivalence_certified": False})
    print(json.dumps({"status": receipt["status"], "tests": test_count,
        "implementation": identity(implementation), "tests_source": identity(tests),
        "manifest": identity(HERE / "preparation_manifest.json")}))


if __name__ == "__main__":
    main()
