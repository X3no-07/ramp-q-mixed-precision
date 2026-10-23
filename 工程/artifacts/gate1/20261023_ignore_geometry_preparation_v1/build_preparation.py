"""Snapshot code/contract metadata and summarize CPU synthetic tests only."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PKG = Path(sys.executable).parent.parent / "Lib/site-packages/ultralytics"


def ident(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write(name, obj):
    path = OUT / name
    if path.exists():
        raise RuntimeError(f"Output exists: {name}")
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    if any((OUT / p).exists() for p in ("source_review.json", "final_tests_receipt.json", "sources")):
        raise RuntimeError("New source-review outputs already exist")
    helper = ROOT / "scripts/gate1_ignore_geometry.py"
    spec = importlib.util.spec_from_file_location("gate1_ignore_geometry", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = ROOT / "artifacts/gate1/20261022_split_refinement/training_contract_v3/contract.json"
    pins = json.loads(contract.read_text(encoding="utf-8"))["environment_fingerprint"]["framework_source_sha256"]
    source_dir = OUT / "sources"
    source_dir.mkdir()
    ranges = {"data/base.py": [[228, 289], [415, 434]], "data/augment.py": [[1532, 1593], [1729, 1778], [1836, 1887], [2809, 2855]],
              "data/dataset.py": [[302, 332], [376, 438]], "utils/instance.py": []}
    sources = {}
    for rel, spans in ranges.items():
        original = PKG / rel
        dest = source_dir / rel.replace("/", "__")
        shutil.copyfile(original, dest)
        actual = ident(dest)
        pin = pins.get(rel)
        if pin is not None and actual["sha256"] != pin:
            raise RuntimeError(f"Previously pinned source changed: {rel}")
        sources[rel] = {"original_path": str(original), "snapshot": dest.relative_to(OUT).as_posix(), **actual,
                        "historical_training_contract_sha256": pin,
                        "historical_pin_matches_current": actual["sha256"] == pin if pin else None,
                        "reviewed_line_spans": spans}
    source_paths = [helper, ROOT / "tests/test_gate1_ignore_geometry.py", ROOT / "scripts/gate1_ignore_negative_loss.py",
                    ROOT / "scripts/prepare_gate1_visdrone.py", contract, OUT / "README.md", Path(__file__)]
    old_loss = ident(ROOT / "scripts/gate1_ignore_negative_loss.py")
    assert old_loss["sha256"] == "67fa0f7a171d26b1f8d2ebd922333f30bb97ffb94b523a58f6f784ed572b3c1f"
    write("source_review.json", {
        "schema": "rampq.gate1.ignore_geometry_source_review/v1",
        "status": "SYNTHETIC_PRIMITIVES_ONLY_REAL_TRAINING_NOT_INTEGRATED",
        "framework_sources": sources,
        "workspace_sources": {p.relative_to(ROOT).as_posix(): ident(p) for p in source_paths},
        "policy": module.POLICY, "restricted_options": module.RESTRICTED_OPTIONS,
        "source_findings": [
            {"source": "prepare_gate1_visdrone.py:161-204", "finding": "Publisher score=0/category0/11 and preparation-quality exclusions are distinguishable; both may be present in ignored sidecars."},
            {"source": "data/base.py:267-280", "finding": "Actual loader resize applies separate ceil-rounded pixel dimensions. Sidecar uses both actual output ratios."},
            {"source": "data/augment.py:1729-1778,1836-1887", "finding": "LetterBox has public get_params/apply_instances; ratio and integer pad are reused exactly by synthetic geometry."},
            {"source": "data/augment.py:1532-1593", "finding": "RandomFlip has sampled decision plus absolute/normalized dimensions; this primitive requires the absolute canvas dimensions."},
            {"source": "data/augment.py:2809-2855", "finding": "Ordinary v8_transforms composes affine and other operations even when selected mix probabilities are zero; a new explicit chain must be built."},
            {"source": "data/dataset.py:409-438", "finding": "Existing collate behavior does not itself transform ignore geometry or validate its identity; custom sidecar collation here is not installed into Dataset."}],
        "limits": {"base_and_instance_current_only_not_historically_pinned": True,
                   "actual_file_hash_or_split_membership_verified": False,
                   "identity_digests_structure_and_order_only": True,
                   "prepared_train_4534_compatibility_checked": False,
                   "zeroarea_quarantine_requires_new_data_contract": True,
                   "dataset_format_trainer_criterion_integrated": False,
                   "new_augmentation_policy_cannot_replace_old_training": True,
                   "real_images_labels_sidecars_or_weights_read": False,
                   "selection_or_holdout_accessed": False, "gpu_used": False,
                   "official_AP_or_scientific_gate_or_deployment_pass": False}})
    root = ET.parse(OUT / "final_tests.xml").getroot()
    suites = list(root.iter("testsuite"))
    counts = {k: sum(int(s.attrib.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    assert counts == {"tests": 85, "failures": 0, "errors": 0, "skipped": 0}, counts
    write("final_tests_receipt.json", {"schema": "rampq.gate1.ignore_geometry_synthetic_test_receipt/v1",
          "status": "PASS_CPU_SYNTHETIC_GEOMETRY_ONLY", "counts": counts,
          "tests_include": {"real_letterbox_crosschecks": 10, "real_flip_crosschecks": 4,
                            "installed_loader_with_in_memory_imread_full_chain_crosschecks": 3,
                            "ragged_identity_and_loss_mask_linkage_dtypes": 2},
          "final_sources": {p.relative_to(ROOT).as_posix(): ident(p) for p in (helper, ROOT / "tests/test_gate1_ignore_geometry.py")},
          "junit_xml": ident(OUT / "final_tests.xml"), "console": ident(OUT / "final_tests.log"),
          "runtime_packages": {k: importlib.metadata.version(k) for k in ("numpy", "torch", "ultralytics", "pytest")},
          "python": sys.version, "interpreter": sys.executable,
          "failed_attempts_preserved": ["initial_tests.xml", "initial_tests.log", "second_tests.xml", "second_tests.log"],
          "intermediate_82_test_pass_preserved": ["third_tests.xml", "third_tests.log"],
          "real_data_or_weights_used": False, "gpu_used": False, "training_pipeline_integrated": False})
    print("IGNORE_GEOMETRY_PREPARATION_BUILT_85_SYNTHETIC_TESTS")


if __name__ == "__main__":
    main()
