"""Seal synthetic Dataset component sources/tests/fixtures; no real data/model."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
FRAMEWORK=ROOT.parent/"05_水下小目标检测_工程/检测工程/.venv/Lib/site-packages/ultralytics"


def ident(path):
    raw=path.read_bytes()
    return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def write(name,obj):
    path=OUT/name
    if path.exists(): raise RuntimeError(f"do not overwrite {path}")
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    if (OUT/"manifest.json").exists(): raise RuntimeError("already sealed")
    spec=importlib.util.spec_from_file_location("dataset_component",ROOT/"scripts/gate1_ignore_dataset.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture=module.build_synthetic_fixture(OUT/"frozen_fixture")
    ds=module.RestrictedIgnoreDataset(fixture["train_manifest"],fixture["train_manifest_sha256"],image_size=64,
                                     augment=True,fliplr=1.,flipud=1.)
    batch=ds.collate_fn([ds[0],ds[1]])
    write("fixture_receipt.json",{"schema":"rampq.gate1.ignore_dataset_fixture_receipt/v1","data_role":"synthetic_only",
          "fixture":{k:(str(Path(v).relative_to(OUT)) if k.endswith("manifest") else v) for k,v in fixture.items()},
          "batch":{"img":{"shape":list(batch["img"].shape),"dtype":str(batch["img"].dtype)},
                   "cls":{"shape":list(batch["cls"].shape),"dtype":str(batch["cls"].dtype)},
                   "bboxes":{"shape":list(batch["bboxes"].shape),"dtype":str(batch["bboxes"].dtype)},
                   "batch_idx":batch["batch_idx"].tolist(),"image_hw":batch["image_hw"],
                   "ignore_boxes_xyxy":[v.tolist() for v in batch["ignore_boxes_xyxy"]],
                   "image_identities":batch["image_identities"],"sample_identities":batch["sample_identities"],
                   "transform_trace":batch["transform_trace"]},
          "not_executed":["real_data_training","criterion_or_model_forward","GPU","official_AP","Orin_deployment"]})
    sources={}
    snap=OUT/"sources";snap.mkdir()
    for rel in ("scripts/gate1_ignore_dataset.py","tests/test_gate1_ignore_dataset.py","scripts/gate1_ignore_geometry.py"):
        source=ROOT/rel
        target=snap/source.name
        shutil.copyfile(source,target)
        sources[rel]=ident(source)
        assert ident(target)==sources[rel]
    framework={}
    for rel in ("data/base.py","data/dataset.py","data/augment.py","models/yolo/detect/train.py","data/build.py"):
        source=FRAMEWORK/rel
        target=snap/("framework__"+rel.replace("/","__"))
        shutil.copyfile(source,target)
        framework[rel]={"original_path":str(source),"snapshot":target.relative_to(OUT).as_posix(),**ident(source)}
    xml=ET.parse(OUT/"tests_attempt2.xml").getroot()
    suite=xml.find("testsuite") if xml.tag=="testsuites" else xml
    counts={key:int(suite.attrib.get(key,"0")) for key in ("tests","failures","errors","skipped")}
    assert counts=={"tests":45,"failures":0,"errors":0,"skipped":0},counts
    write("source_review.json",{"schema":"rampq.gate1.ignore_dataset_source_review/v1",
          "review_type":"component_author_implementation_review_not_independent_acceptance",
          "workspace_sources":sources,"framework_sources":framework,"geometry_source_sha256":module.GEOMETRY_SHA256,
          "claims":["exact role manifest membership validated before any member file read",
                    "image/positive YOLO label/ignore JSONL bytes and decoded dimensions verified",
                    "manifest plus three bytes identities rechecked every getitem",
                    "same actual resize and centered padding and flip parameters used for all three",
                    "publisher ignores remain independent from positive classes",
                    "invalid source geometry rejects whole sample; empty file explicitly means zero regions",
                    "synthetic-only manifest and prohibited role/path rejection"],
          "limitations":["Dataset component and CPU DataLoader workers=0 only; actual Trainer integration belongs to root run",
                         "no official task metrics or complete training or AP/deployment claim",
                         "synthetic-only is a caller manifest assertion plus checked fixture provenance; not a pixel classifier for synthetic content"],
          "old_evidence_edited":False})
    write("tests_receipt.json",{"schema":"rampq.gate1.ignore_dataset_tests_receipt/v1","counts":counts,
          "python":platform.python_version(),"workspace_sources":sources,
          "console":ident(OUT/"tests_attempt2.log"),"junit_xml":ident(OUT/"tests_attempt2.xml"),
          "attempt1":{"status":"SETUP_ENVIRONMENT_ERROR_NO_DATASET_CODE_EXECUTED","reason":"pytest sandbox Temp numbered directory creation denied; explicit new owned basetemp used on attempt2",
                      "console":ident(OUT/"tests_attempt1.log"),"junit_xml":ident(OUT/"tests_attempt1.xml")}})
    readme="""# Restricted ignore Dataset component (synthetic only)

Implemented `RestrictedIgnoreDataset(manifest_path, manifest_sha256, split_role=..., image_size=..., class_count=10, augment=False, fliplr=0, flipud=0, seed=0, epoch=0)` and `build_synthetic_fixture(new_directory)`.

Only the caller hash-bound manifest's explicit members may load. Each entry's role, image ID, dimensions and image/positive-label/ignore-sidecar bytes bind independently. A train/ prefix is never a membership rule. Selection/Holdout/test paths and roles are rejected before member reads. This version requires `data_role=synthetic_only`; real training needs a separate reviewed contract.

Fresh verified PNG bytes decode with cv2; positives read separate normalized YOLO xywh; sidecar JSONL passes the old frozen fail-closed geometry helper. The same actual longest-side ceil resize, centered fixed LetterBox and explicit vertical/horizontal flip decisions transform images, positives and ignores. Nothing invokes the ordinary probability-zero complex augmentation chain. Explicit zero-row sidecars are allowed; zero/negative/fully-clipped source boxes reject the whole sample, never drop a row. No image or labels are silently skipped.

Batch outputs are RGB uint8 BCHW, float32 cls [N,1], normalized xywh [N,4], batch_idx [N], plus one float32 pixel XYXY tensor per image under ignore_boxes_xyxy, image_identities, sample_identities, source_records, image_hw and transform_trace. Only workers=0 CPU DataLoader is tested here. Ragged ignores require the Trainer/criterion to move all tensors to the model's device; ordinary DetectionTrainer preprocess only moves top-level tensors.

45/45 synthetic file-I/O and data-contract tests pass. Initial pytest attempt failed before Dataset code because the sandbox's default Temp path could not create a numbered directory; the unchanged source passes with a new owned basetemp. Both logs remain bound. Frozen fixture is six newly generated synthetic PNGs (four train, two training_val), all image hashes distinct and image IDs disjoint; three distinct publisher-ignore reasons are retained. The first score-0 region spans 90% of each original dimension and overlaps positives so a separate root Trainer smoke can verify positive priority and ignored negatives are used.

This package proves the Dataset component on synthetic files. It does not prove actual Detector/Trainer integration, real training, official AP, clean evaluation, scientific Gate1 acceptance, quantization, or deployment. No real images, old checkpoints, GPU, Selection or Holdout were read.
"""
    (OUT/"README.md").write_text(readme,encoding="utf-8")
    files={}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and not any(p.startswith("temp_attempt") or p=="__pycache__" for p in path.relative_to(OUT).parts):
            files[path.relative_to(OUT).as_posix()]=ident(path)
    manifest={"schema":"rampq.gate1.ignore_dataset_component_manifest/v1","status":"SEALED_SYNTHETIC_DATASET_COMPONENT_ONLY",
              "files":files,"workspace_sources":sources,"counts":counts,"real_data_or_weights_used":False,
              "actual_DetectionTrainer_verified_by_this_package":False,"scientific_gate_pass":False,
              "excluded_mutable_test_temporary_directories":["temp_attempt2/"]}
    write("manifest.json",manifest)
    print(json.dumps({"status":manifest["status"],"files":len(files),"manifest":ident(OUT/"manifest.json"),"source":sources["scripts/gate1_ignore_dataset.py"]}))


if __name__=="__main__":main()
