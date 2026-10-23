"""Freeze qualification code/protocol from metadata and completed synthetic tests.

No model, checkpoint, real image, or annotation contents are opened here.
"""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNNER = ROOT / "scripts" / "qualify_gate1_float_baseline.py"
TESTS = ROOT / "tests" / "test_gate1_float_qualification.py"
SPEC = importlib.util.spec_from_file_location("qualified_preparation_protocol", RUNNER)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def create(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def bind(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


ast.parse(RUNNER.read_text(encoding="utf-8"))
ast.parse(TESTS.read_text(encoding="utf-8"))
contract_dir = ROOT / "artifacts" / "gate1" / "20261022_split_refinement" / "training_contract_v3"
Q.require(digest(contract_dir / "contract.json") == Q.CONTRACT_SHA256, "Training contract changed")
Q.require(digest(contract_dir / "manifest.json") == Q.CONTRACT_MANIFEST_SHA256, "Training manifest changed")
contract = Q.read_json(contract_dir / "contract.json")
plan = Path(contract["split_plan"])
Q.require(digest(plan / "manifest.json") == Q.PLAN_MANIFEST_SHA256, "Split manifest changed")
Q.require(digest(plan / "split_assignment.jsonl") == Q.ASSIGNMENT_SHA256, "Split assignment changed")
debug_rows = Q.select_debug_rows(Q.read_jsonl(plan / "split_assignment.jsonl"))
Q.require(digest(ROOT / "scripts" / "gate1_ignore_aware_evaluation.py") == Q.EVALUATOR_SHA256, "Evaluator changed")
review = HERE / "source_review"
Q.require(digest(review / "source_review.json") == Q.SOURCE_REVIEW_SHA256, "Review inventory changed")
Q.require(digest(review / "qualification_contract_prepared.json") == Q.PREPARED_SOURCE_CONTRACT_SHA256, "Review protocol changed")
Q.require(digest(review / "manifest.json") == Q.SOURCE_REVIEW_MANIFEST_SHA256, "Review manifest changed")
Q.verify_manifest(review)
source_root = Path(sys.executable).resolve().parents[1] / "Lib" / "site-packages" / "ultralytics"
live_sources = {}
for relative, entry in Q.read_json(review / "source_review.json")["framework_files"].items():
    source = source_root / relative
    Q.require(digest(source) == entry["sha256"], "Reviewed framework source changed")
    live_sources[relative] = bind(source)
for relative, expected in Q.SUPPLEMENTAL_SOURCE_SHA256.items():
    source = source_root / relative
    Q.require(digest(source) == expected, "Supplemental framework source changed")
    live_sources[relative] = bind(source)
    destination = HERE / "supplemental_framework_sources" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    Q.require(not destination.exists(), "Supplemental snapshot exists")
    shutil.copyfile(source, destination)
for source, name in ((RUNNER, "runner_source.py"), (TESTS, "synthetic_tests_source.py")):
    destination = HERE / name
    Q.require(not destination.exists(), "Frozen source snapshot exists")
    shutil.copyfile(source, destination)
create(HERE / "qualification_protocol.json", Q.protocol())
console = HERE / "synthetic_tests_v5_console.txt"
Q.require(console.is_file() and "Ran 20 tests" in console.read_text(encoding="utf-8") and "\nOK\n" in console.read_text(encoding="utf-8").replace("\r\n", "\n"), "Final synthetic tests did not pass")
state = {
    "schema": "rampq.gate1.float_qualification_preparation/v1",
    "status": "PREPARED_AND_SYNTHETIC_TESTED_NOT_EXECUTED",
    "at_utc": datetime.now(timezone.utc).isoformat(),
    "runner": bind(RUNNER), "tests": bind(TESTS), "protocol": bind(HERE / "qualification_protocol.json"),
    "training_contract": bind(contract_dir / "contract.json"), "source_review_manifest": bind(review / "manifest.json"),
    "debug_rows_from_pinned_metadata": len(debug_rows), "debug_source_split": "train",
    "live_framework_source_files_verified": live_sources,
    "synthetic_validation": {"tests": 20, "failures": 0, "errors": 0, "console": bind(console)},
    "earlier_synthetic_attempts": {"v1": "17 passed; one test cleanup failed due Windows sandbox temp-directory removal permissions; no inference",
                                   "v2": "18 passed", "v3": "19 passed", "v4": "20 passed before final protocol/structure hardening"},
    "checkpoints_loaded": False, "real_image_pixels_or_annotation_contents_opened": False,
    "gpu_inference_started": False, "selection_detector_inference": False, "holdout_accessed": False,
    "accepted_float_baseline_ready": False, "official_visdrone_ap": False,
    "limits": Q.LIMITS,
}
create(HERE / "preparation_receipt.json", state)
files = {str(path.relative_to(HERE)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": digest(path)}
         for path in sorted(HERE.rglob("*")) if path.is_file() and path != HERE / "preparation_manifest.json"}
create(HERE / "preparation_manifest.json", {"schema": "rampq.gate1.float_qualification_preparation_delivery/v1", "files": files,
                                           "external_source_files": {str(RUNNER): bind(RUNNER), str(TESTS): bind(TESTS)}})
print(json.dumps({"status": state["status"], "debug_rows": len(debug_rows), "synthetic_tests": 20, "files": len(files)}, ensure_ascii=False))
