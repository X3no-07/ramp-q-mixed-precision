"""Capture local source evidence and metadata only; never import a detector."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
ULTRA = ROOT.parent / "05_水下小目标检测_工程/检测工程/.venv/Lib/site-packages/ultralytics"
REFINE = ROOT / "artifacts/gate1/20261022_split_refinement"
PREPARED = ROOT / "artifacts/gate1/20261022_preparation/prepared_v2"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(name: str, value: object) -> None:
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    assert (ROOT / "scripts/gate1_ignore_aware_evaluation.py").is_file(), ROOT
    names = [
        "__init__.py", "cfg/__init__.py", "cfg/default.yaml", "data/augment.py", "data/loaders.py", "data/utils.py",
        "engine/model.py", "engine/predictor.py", "engine/trainer.py", "models/yolo/detect/predict.py",
        "models/yolo/detect/val.py", "nn/autobackend.py", "nn/backends/pytorch.py", "nn/modules/head.py",
        "nn/tasks.py", "utils/nms.py", "utils/ops.py", "utils/torch_utils.py",
    ]
    files = {}
    for name in names:
        source = ULTRA / name
        snapshot = OUT / "framework_sources" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        assert not snapshot.exists()
        shutil.copyfile(source, snapshot)
        assert sha(source) == sha(snapshot)
        files[name] = {"path": str(source), "snapshot": str(snapshot.relative_to(OUT)),
                       "sha256": sha(source), "bytes": source.stat().st_size,
                       "line_count": len(source.read_text(encoding="utf-8").splitlines())}
    evaluator = ROOT / "scripts/gate1_ignore_aware_evaluation.py"
    shutil.copyfile(evaluator, OUT / evaluator.name)
    assignment = REFINE / "split_plan_v1/split_assignment.jsonl"
    rows = [json.loads(line) for line in assignment.read_text(encoding="utf-8").splitlines()]
    debug = sorted((row for row in rows if row["assigned_split"] == "debug"), key=lambda row: row["image_id"])
    assert len(debug) == 109
    assert all(row["source_split"] == "train" and not row["requires_annotation_quality_quarantine"] for row in debug)
    inventory = [{key: row[key] for key in ("image_id", "image_path", "image_sha256", "annotation_path",
                 "annotation_sha256", "width", "height", "valid_target_count", "refined_component")} for row in debug]
    dump("debug_metadata_inventory.json", {"status": "METADATA_ONLY_NO_IMAGE_OR_ANNOTATION_BODY_OPENED",
         "source_assignment": str(assignment), "source_assignment_sha256": sha(assignment),
         "count": len(debug), "component_count": len({row["refined_component"] for row in debug}),
         "valid_target_count_before_ignore_raster_prefilter": sum(row["valid_target_count"] for row in debug),
         "prepared_root": str(PREPARED), "images": inventory,
         "selection_or_holdout_image_label_bytes_read": False})
    project_sources = {
        "project_evaluator": evaluator,
        "evaluation_policy": REFINE / "evaluation_sources/evaluation_policy.json",
        "training_contract": REFINE / "training_contract_v3/contract.json",
        "split_assignment": assignment,
        "effective_training_args_snapshot": REFINE / "float_training_v2/effective_training_args.json",
    }
    evidence = {key: {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
                for key, path in project_sources.items()}
    claims = [
        {"id": "predict_defaults", "file": "engine/model.py", "lines": [519, 526],
         "finding": "Model.predict sets rect=True by default, overriding default.yaml rect=False. Single-image same_shapes=True plus rect=True makes .pt LetterBox auto use stride-aligned rectangle rather than necessarily square640."},
        {"id": "predict_input_routes", "file": "engine/predictor.py", "lines": [167, 220],
         "finding": "NumPy route LetterBoxes, converts BGR to RGB and normalizes after dtype conversion; Tensor route does not LetterBox, channel-flip or normalize again. Do not treat manually LetterBoxed Tensor Results orig_shape as the original source image."},
        {"id": "letterbox_rounding", "file": "data/augment.py", "lines": [1745, 1765],
         "finding": "Gain is min(newH/H,newW/W); resized dimensions round each axis. Centered remainder uses round(d-0.1),round(d+0.1), so odd padding is asymmetric. auto=True reduces remainder modulo stride. Record actual integer left/top padding and nominal gain."},
        {"id": "letterbox_pixels", "file": "data/augment.py", "lines": [1792, 1807],
         "finding": "Apply image with cv2.resize then cv2.copyMakeBorder constant114; default interpolation is cv2.INTER_LINEAR, center=True and scaleup=True."},
        {"id": "predict_precision_fusion", "file": "engine/predictor.py", "lines": [428, 438],
         "finding": "Predictor deepcopy model then AutoBackend(fp16=args.quantize==16, fuse=True). Legacy half/int8 aliases are mapped to quantize by cfg.__init__.py:594-603. Fixed manual raw dtype bypasses these defaults deliberately."},
        {"id": "checkpoint_float", "file": "nn/tasks.py", "lines": [1938, 1982],
         "finding": "load_checkpoint chooses EMA before model, casts candidate to FP32, defaults fuse=False, then eval().to(device); compatibility updates include inplace=True by default."},
        {"id": "saved_precision", "file": "engine/trainer.py", "lines": [720, 758],
         "finding": "save_model serializes a deepcopy of EMA cast to half, with framework finite-resync/clamp behavior. The training runner guard is a separate protection. Casting saved EMA to float does not restore original training FP32 EMA values."},
        {"id": "stripped_checkpoint", "file": "utils/torch_utils.py", "lines": [806, 829],
         "finding": "strip_optimizer replaces model with EMA, casts model half, clears EMA and epoch=-1. Use final receipt/results to identify selected epoch; final checkpoint epoch alone is insufficient."},
        {"id": "decoded_layout", "file": "nn/modules/head.py", "lines": [179, 238],
         "finding": "Non-end2end eval returns (decoded_y, raw_heads) unless export. Decoded y concatenates pixel-space xywh and sigmoid class scores along channel axis, hence nc10 gives Bx14xN without an objectness channel. Expected square640 strides8/16/32 gives N8400; runtime must verify, not assume."},
        {"id": "fusion", "file": "nn/backends/pytorch.py", "lines": [54, 78],
         "finding": "Backend loads checkpoint then optionally fuses Conv+BN and finally converts dtype; predictor requests fuse=True. Raw unfused and default predictor fused are different arithmetic paths."},
        {"id": "nms_mutation", "file": "utils/nms.py", "lines": [77, 95],
         "finding": "NMS transposes prediction as a view and overwrites first four channels converting xywh to xyxy. Clone input after saving/comparing raw output."},
        {"id": "nms_policy", "file": "utils/nms.py", "lines": [117, 170],
         "finding": "Best-class-only by default, strict confidence greater-than cutoff, class offsets, max_nms cap, then torchvision or TorchNMS chosen from already imported sys.modules. max_det caps output; time limit can interrupt batches. Pin CPU FP32 and explicit torchvision import for qualification."},
        {"id": "restoration_clipping", "file": "utils/ops.py", "lines": [122, 165],
         "finding": "scale_boxes reverses gain/pad but default xyxy path clips to image. Project ignore raster must see raw original-coordinate boxes first, so use explicit unclipped inverse transform before matcher; keep optional clipped copy separately."},
        {"id": "effective_training_cap", "file": "models/yolo/detect/val.py", "lines": [65, 97],
         "finding": "Framework raises default max_det300 to observed dataset maximum902; this is not manual tuning. Parent verified train max351 and training_val max902. Qualification uses its separately fixed max_det1000."},
        {"id": "loader_label_deduplication", "file": "data/utils.py", "lines": [365, 372],
         "finding": "YOLO loader uses np.unique on label rows and drops duplicates in cache. Parent verified three dropped positives, one in each specified train image; original prepared label bytes remain untouched."},
    ]
    dump("source_review.json", {"schema": "rampq.gate1.float_qualification_source_review/v1",
         "status": "LOCAL_SOURCE_AND_METADATA_REVIEW_ONLY", "installed_version": "8.4.142",
         "framework_files": files, "project_evidence": evidence, "claims": claims,
         "evaluator_adapter": {"path": str(evaluator), "sha256": sha(evaluator),
             "parse_ground_truth_lines": [35, 63], "validation_lines": [76, 104],
             "matching_lines": [183, 258],
             "ground_truth": "Read original eight-column annotation for each permitted debug image, retaining category0/11, score0, truncation and occlusion. Do not reconstruct GT from YOLO labels.",
             "target_id": "image_id + ':' + original physical zero-based line index, blank lines counted; parser strips optional BOM but preserves physical indexing.",
             "prediction": "Restored unclipped xyxy -> [x1,y1,x2-x1,y2-y1]; best YOLO class0..9 mapped explicitly to publisher category1..10; score is decoded class probability. Unique detection_id includes image_id, dtype and post-NMS row index.",
             "ignore_order": "category0 pixel-raster prefilter on raw original boxes, then project clips surviving boxes, performs positive IoU-first matching, then same-class score0 absorption. Category11 is not a universal ignore region.",
             "identity_dedup": "Do not deduplicate original GT; each physical annotation row remains a distinct identity even if YOLO training cache deduplicated exact label rows.",
             "official_AP": False, "official_oracle_equivalence_verified": False},
         "training_disclosures": {"effective_max_det": 902, "qualification_max_det": 1000,
             "loader_dropped_positive_duplicate_rows": 3,
             "affected_train_images": ["0000140_00118_d_0000002", "9999945_00000_d_0000114", "9999987_00000_d_0000049"],
             "duplicate_count_source": "parent current training logs/cache metadata audit; no model or image body read by this source-review task",
             "training_checkpoint_precision": "serialized FP16 EMA, cast back to FP32 for FP32 compute"},
         "models_executed": False, "image_or_annotation_bodies_opened": False,
         "selection_detector_inference": False, "holdout_accessed": False,
         "scientific_Gate1_pass": False, "old_numerical_guard": "FAILED_UNCHANGED"})

    contract = {
        "schema": "rampq.gate1.float_qualification_contract_prepared/v1",
        "status": "PREPARED_PENDING_RUNNER_FREEZE_AND_TRAINING_COMPLETION_NOT_EXECUTED",
        "scope": "Float inference pipeline and FP32/FP16 numerical qualification on existing debug only; not a research-selection experiment, official AP or Gate1 scientific acceptance.",
        "prerequisites": ["Original fixed 30-epoch run completed without rejected recovery; final receipt verified.",
            "Copy selected best.pt from verified final receipt to new immutable qualification output; bind SHA before loading.",
            "Verify source assignment, all109 debug image and original annotation SHAs; reject Selection/testdev paths.",
            "Freeze runner SHA, environment, framework source hashes and this contract before any qualification model execution.",
            "Run synthetic geometry, category and target-identity adapter checks first; failed checks prohibit inference."],
        "checkpoint": {"path": str(REFINE / "float_training_v2/run/weights/best.pt"),
            "final_weight_sha256": None, "weight_sha_binding": "required from final training receipt, pending completion",
            "selection": "Already frozen train-derived training_val builtin ignore-unaware mAP50-95, later epoch on tie; qualification debug must not select/reselect checkpoint.",
            "serialized_precision": "FP16 EMA from framework; final optimizer stripping may move it from ema key to model key.",
            "load": "Two independent same-SHA loads; EMA preferred else model; explicit float().eval() followed by CUDA float or half, fuse=False in both. Set inplace compatibility and verify head flags. No autocast.",
            "not_recovered": "FP32 computation from saved FP16 EMA is not recovery of original FP32 training EMA."},
        "data": {"assignment": str(assignment), "assignment_sha256": sha(assignment), "role": "debug", "count": 109,
            "order": "ascending image_id", "inventory": "debug_metadata_inventory.json", "inventory_sha256": sha(OUT / "debug_metadata_inventory.json"),
            "selection_read_for_inference": False, "holdout_accessed": False,
            "scope_limit": "Debug images were previously used for engineering checks and are not a fresh independent holdout."},
        "preprocessing": {"batch": 1, "image_size": [640, 640], "stride": 32,
            "loader": "OpenCV BGR uint8 decoded file; verify decoded dimensions against metadata.",
            "letterbox": {"new_shape": [640, 640], "auto": False, "scale_fill": False, "scaleup": True,
                "center": True, "interpolation": "cv2.INTER_LINEAR", "padding_value": 114},
            "tensor": "BGR to RGB, HWC to contiguous CHW, batch axis; canonical CPU FP32 divide by255 exactly once, then transfer/cast to respective CUDA dtype.",
            "same_input": "FP16 input must be canonical FP32 input converted to half; record FP32 input SHA and exact letterbox integer padding/gain per image.",
            "predictor_equivalence": "Not claimed: default predict rect=True can generate rectangle; default predictor fuses and clips. No default-predict/raw numerical comparison."},
        "model": {"expected_classes": 10, "expected_decoded_shape": [1, 14, 8400],
            "layout": "B,C,N; first4 pixel xywh in letterbox640 domain, then10 independent sigmoid class scores, no objectness channel",
            "head_flags": {"end2end": False, "export": False, "xyxy": False}, "augment": False, "fusion": False,
            "raw_extraction": "tuple[0] for eval output; fail unexpected layout rather than silently transpose or truncate.",
            "torch_controls": {"inference_mode": True, "autocast": False, "cudnn_benchmark": False,
                "allow_tf32_matmul": False, "allow_tf32_cudnn": False, "float32_matmul_precision": "highest"}},
        "numerical_policy": {"reference": "FP32 compute output", "comparison": "FP16 output promoted to FP32",
            "boxes": {"channels": [0, 1, 2, 3], "rtol": 0.005, "atol": 0.5, "units": "letterbox-input pixels; xywh"},
            "scores": {"channels": "4:14", "rtol": 0.02, "atol": 0.002},
            "formula": "abs(fp16_as_fp32-fp32) <= atol + rtol*abs(fp32)",
            "domain": "All decoded coordinates and all10 class scores at all8400 anchors for each of109 images; no confidence mask for raw tolerance.",
            "required_finite": "All saved model floating state, both inputs and both decoded outputs finite; raw widths/heights positive and scores within[0,1].",
            "pass": "All required prerequisites and all images pass shape/finite/all-element bounds; zero tolerated out-of-bound values.",
            "failure": "Preserve failed evidence, mark numerical qualification failed/incomplete; do not relax bounds, fuse, change checkpoint, retrain or tune using debug outcomes automatically."},
        "postprocessing": {"nms": {"implementation": "ultralytics.utils.nms.non_max_suppression", "device": "cpu", "dtype": "float32",
                "input": "independent detached CPU FP32 clone of decoded output; raw comparison/summary completed before NMS mutates input",
                "conf_thres": 0.001, "conf_comparison": ">", "iou_thres": 0.7, "classes": None, "agnostic": False,
                "multi_label": False, "nc": 10, "max_det": 1000, "max_nms": 30000, "max_wh": 7680,
                "end2end": False, "rotated": False, "backend": "explicit torchvision import, torchvision.ops.nms CPU",
                "truncation": "record output count reaching1000 and any NMS time-limit warning; no silent censoring"},
            "box_restoration": "xyxy subtract recorded integer left/top pad then divide by recorded gain; keep raw original-coordinate boxes unclipped for category0 raster filter. Do not use default scale_boxes xyxy clipping before evaluator.",
            "adapter": "class_id0..9 +1; bbox_xywh=[x1,y1,x2-x1,y2-y1] from original coordinates, preserve floating coordinates and detection order.",
            "project_matching": {"module": str(evaluator), "sha256": sha(evaluator), "function": "match_project_targets",
                "confidence": 0.25, "confidence_comparison": ">=", "iou_threshold": 0.5,
                "ground_truth": "Original annotation bytes via parse_ground_truth; physical line identity retained."},
            "reporting": "Per-dtype target hits/misses, same-target hit differences and false-positive/ignore counts are diagnostics under this project policy. NMS count/order equality and target-hit equality are not substituted for raw allclose. Disclose threshold/NMS boundary differences.",
            "official_AP": False},
        "evidence": {"source_review": "source_review.json", "source_review_sha256": sha(OUT / "source_review.json"),
            "per_image": "source SHA, tensor shape/input hash, gain/pad, output shape/dtype/finite, separate box/score maximum error and allclose fail counts, postNMS count, project matching summaries",
            "artifacts": "Frozen contract/runner, source and environment records, selected checkpoint SHA, per-image metrics JSONL, aggregate qualification verdict, raw and postprocessing diagnostic failures."},
        "boundaries": {"training_modified": False, "training_started_by_this_task": False, "model_executed_by_this_task": False,
            "official_visdrone_AP": False, "Gate1_scientific_pass": False, "old_numerical_guard": "FAILED_UNCHANGED",
            "hardware_scope": "Local RTX5060 Laptop GPU path only; no Orin/TensorRT qualification or performance claim."},
    }
    dump("qualification_contract_prepared.json", contract)
    print(json.dumps({"status": "SOURCE_REVIEW_AND_PENDING_CONTRACT_WRITTEN", "framework_sources": len(files),
                      "debug_metadata_rows": len(debug), "model_executed": False, "holdout_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
