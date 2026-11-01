"""Qualify the completed Gate1 candidate on its 109 engineering debug images.

This does not train, choose a checkpoint, access Selection/Holdout, calculate AP,
or accept a scientific baseline. The entry point refuses an incomplete training
receipt before loading weights or opening image pixels. All run outputs are new.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np


CONTRACT_SHA256 = "6dfc1b1c8b2d89948c5804cfda8f22c92cbd721b3dfaf0cffdc314b2b2bd4fc0"
CONTRACT_MANIFEST_SHA256 = "f80772ea5122ff64c3c19d7ac7deeefedd4b11cfb6566807a70db778221461ca"
TRAINING_RUNNER_SHA256 = "99730c52afcd363250054b41bf47bc0aceb8db1bbb8f0179c0b1742f2d951ef6"
PLAN_MANIFEST_SHA256 = "c757d7665ec5b3336433ec22ce50fc9104e7f8c2352fd919e2481e470508232d"
ASSIGNMENT_SHA256 = "50113cde0843076f9524a55b75d54392fced1c5e1092bb9d28d84649272ea025"
SOURCE_MANIFEST_SHA256 = "e657bc24d4cdf16b868bf91111647a933bec95d90385b7fc8ceab483020f8048"
EVALUATOR_SHA256 = "4e45ff26a86a83ff66e997f15e4c4794246c6440c304446972809ff32db09961"
SOURCE_REVIEW_SHA256 = "483fe5963296bb69c96594c79cc913efda6a9215979823e1b12bb1e23a9b4a68"
PREPARED_SOURCE_CONTRACT_SHA256 = "5a3c400c0a5606756ea56e90a86a54b7d5c237014152590a17c776e25cd8ee2a"
SOURCE_REVIEW_MANIFEST_SHA256 = "4bd6b90cb2f282b26aa6502fa454ddce00e1c4decaf52e6e568065a2e1d0a488"
SUPPLEMENTAL_SOURCE_SHA256 = {
    "nn/modules/conv.py": "548ee1f3fa28542ad46b920e4d4271594efa09bc76c4ab3067a08a966f30648d",
    "nn/modules/block.py": "f815c656fc5c0d6f689322f1ef13655349c5d013792fdf08a7b9156751e21be3",
    "utils/tal.py": "cdf158d6cc61a91810a23194c7d0848ae5456aaf5d60341db79b38d16e9c1d8b",
}
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
FINAL_STATUS = "COMPLETE_FLOAT_TRAINING_REQUIRES_SCIENTIFIC_REVIEW"
EXPECTED_DEBUG_COUNT = 109
THRESHOLDS = (0.2, 0.25, 0.3)
LIMITS = [
    "The 109 debug images were previously used for engineering work and are not independent Selection or Holdout evidence.",
    "FP32 reference weights are the saved fp16 EMA checkpoint converted to float; original full-precision training weights are not recovered.",
    "Both PyTorch models are unfused and use one fixed preprocessing/postprocessing contract; this is not TensorRT or Orin validation.",
    "Same-backend dtype tolerances are fixed before execution; failure is retained without relaxing thresholds.",
    "Project ignore-aware target matching is not official VisDrone AP and has not been certified against a MATLAB/Octave oracle.",
    "Finite output, target hits, or a dtype guard PASS do not automatically accept a scientific float baseline.",
    "The original Gate0.5 FAILED_UNCHANGED numerical guard is a separate artifact and is never replaced by this check.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    require(not rel.is_absolute() and rel.parts and ".." not in rel.parts, f"Unsafe relative path: {relative}")
    cursor = root
    for part in rel.parts:
        cursor /= part
        require(not cursor.is_symlink(), f"Symbolic link forbidden: {cursor}")
    resolved = cursor.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), "File escapes its root")
    return resolved


def verify_manifest(root: Path) -> None:
    files = read_json(safe_file(root, "manifest.json"))["files"]
    require(isinstance(files, dict) and bool(files), "Empty manifest")
    for relative, expected in files.items():
        path = safe_file(root, relative)
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], f"Manifest mismatch: {relative}")


def protocol() -> dict:
    return {
        "schema": "rampq.gate1.float_qualification_protocol/v1",
        "training_contract_sha256": CONTRACT_SHA256,
        "training_contract_manifest_sha256": CONTRACT_MANIFEST_SHA256,
        "split_assignment_sha256": ASSIGNMENT_SHA256,
        "reviewed_source_contract_sha256": PREPARED_SOURCE_CONTRACT_SHA256,
        "reviewed_source_inventory_sha256": SOURCE_REVIEW_SHA256,
        "reviewed_source_manifest_sha256": SOURCE_REVIEW_MANIFEST_SHA256,
        "supplemental_framework_source_sha256": SUPPLEMENTAL_SOURCE_SHA256,
        "data_role": "debug", "expected_images": EXPECTED_DEBUG_COUNT,
        "checkpoint": "best.pt selected solely by the frozen training_val proxy; no debug-based selection",
        "model": {"source": "two independent CPU loads of the identical verified saved checkpoint", "fusion": False,
                  "reference": "saved fp16 EMA converted to CUDA FP32", "comparison": "same saved fp16 EMA converted to CUDA FP16",
                  "batch": 1, "imgsz": 640, "augment": False, "class_count": 10},
        "preprocessing": {"decode": "cv2.imdecode color BGR, EXIF orientation ignored; dimensions checked against source manifest",
                          "letterbox": "Ultralytics LetterBox 640x640, auto=False, scale_fill=False, scaleup=True, center=True, stride=32, value=114, INTER_LINEAR",
                          "input": "BGR to RGB, CHW contiguous; shared FP32 uint8/255 normalization; FP16 input is that FP32 tensor cast to half",
                          "same_preprocessed_uint8_for_both_models": True},
        "raw": {"layout": "[1,14,8400]: first four channels decoded pixel xywh at 640; ten sigmoid class scores; no objectness",
                "comparison": "all anchors, channel order retained, FP32 is asymmetric allclose reference",
                "box_rtol": 0.005, "box_atol": 0.5, "score_rtol": 0.02, "score_atol": 0.002,
                "failure_policy": "retain FAIL; no tolerance change; finite/shape failure aborts matching"},
        "postprocessing": {"backend": "Ultralytics non_max_suppression on CPU FP32 clone, torchvision imported explicitly",
                           "candidate_confidence_strictly_greater_than": 0.001, "nms_iou": 0.7, "max_det": 1000,
                           "nc": 10, "agnostic": False, "multi_label": False, "max_nms": 30000, "max_wh": 7680,
                           "max_time_img": 60.0, "return_anchor_indices": True,
                           "time_limit_warning_policy": "abort and preserve the attempt; never accept a truncated NMS result",
                           "max_det_saturation_policy": "report each branch/image reaching 1000, without changing the fixed cap",
                           "restore": "subtract actual integer left/top padding and divide recorded gain; do not clip before ignore-region filtering",
                           "mapping": "YOLO class 0..9 maps explicitly to publisher category 1..10"},
        "matching": {"implementation_sha256": EVALUATOR_SHA256, "function": "match_project_targets",
                     "target_id": "image_id:zero_based_original_physical_annotation_line",
                     "confidence_primary_inclusive": 0.25, "confidence_sensitivity_inclusive": [0.2, 0.3],
                     "target_iou": 0.5, "size_scope": "all eligible sizes; no small-only metric", "official_AP": False},
        "selection_detector_inference": False, "holdout_accessed": False,
        "accepted_float_baseline_ready": False, "limits": LIMITS,
    }


def validate_completion(receipt: dict, contract: dict, contract_sha: str, contract_manifest_sha: str) -> dict:
    """Metadata-only completion guard. Does not load or open any checkpoint."""
    require(receipt.get("schema") == "rampq.gate1.float_training_attempt/v1", "Wrong training receipt schema")
    require(receipt.get("mode") == "train" and receipt.get("status") == FINAL_STATUS
            and receipt.get("phase") == "COMPLETE" and receipt.get("training_started") is True,
            "Only a completed final training attempt is accepted")
    require(receipt.get("contract_sha256") == contract_sha == CONTRACT_SHA256, "Wrong frozen contract identity")
    require(receipt.get("contract_manifest_sha256") == contract_manifest_sha == CONTRACT_MANIFEST_SHA256, "Wrong contract manifest identity")
    require(receipt.get("runner_sha256") == contract.get("runner_sha256") == TRAINING_RUNNER_SHA256, "Wrong frozen training runner")
    require(contract.get("fixed_args", {}).get("epochs") == 30 and contract["fixed_args"].get("amp") is False, "Training contract was changed")
    require(receipt.get("selection_detector_inference") is False and receipt.get("holdout_accessed") is False, "Forbidden data-role access in training receipt")
    require(receipt.get("recovery_events") == [] and receipt.get("external_restarts") == 0, "Training recovery/restart requires review")
    require(receipt.get("effective_batch") == 4 and receipt.get("effective_amp") is False, "Effective training parameters differ")
    history = receipt.get("epoch_history", [])
    require(len(history) == 30 and [event.get("epoch_zero_based") for event in history] == list(range(30))
            and [event.get("epoch_one_based") for event in history] == list(range(1, 31)), "Training did not finish exactly 30 ordered epochs")
    require(all(isinstance(event.get("fitness"), (int, float)) and not isinstance(event["fitness"], bool)
                and math.isfinite(event["fitness"]) for event in history), "Nonfinite or missing epoch fitness")
    expected_best = max(range(30), key=lambda index: (history[index]["fitness"], index))
    saved = receipt.get("checkpoint_history", [])
    require(len(saved) == 30 and [event.get("epoch_zero_based") for event in saved] == list(range(30)), "Incomplete checkpoint provenance")
    require(saved[expected_best].get("best_saved_this_epoch") is True, "Best epoch not saved")
    checkpoints = receipt.get("checkpoints", {})
    require(set(checkpoints) == {"best.pt", "last.pt"}, "Final checkpoint identities missing")
    for name, epoch in (("best.pt", expected_best), ("last.pt", 29)):
        entry = checkpoints[name]
        require(entry.get("training_epoch_zero_based") == epoch and entry.get("training_epoch_one_based") == epoch + 1, f"Wrong {name} epoch provenance")
        require(isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64 and all(c in "0123456789abcdef" for c in entry["sha256"]), "Malformed checkpoint SHA")
        require(isinstance(entry.get("bytes"), int) and entry["bytes"] > 0, "Invalid checkpoint size")
    names = receipt.get("checkpoint_classes", {})
    require({str(key): value for key, value in names.items()} == {str(index): value for index, value in enumerate(NAMES)}, "Checkpoint classes differ")
    return {"epochs": 30, "best_training_epoch_one_based": expected_best + 1,
            "last_training_epoch_one_based": 30, "checkpoint_selection_uses_debug": False}


def select_debug_rows(rows: list[dict], expected_count: int = EXPECTED_DEBUG_COUNT) -> list[dict]:
    selected = [dict(row) for row in rows if row.get("assigned_split") == "debug"]
    require(len(selected) == expected_count, "Debug image count differs from the fixed plan")
    require(len({row["image_id"] for row in rows}) == len(rows), "Duplicate image identities")
    for row in selected:
        require(row.get("source_split") == "train" and row["image_id"].startswith("train/"), "Debug must come from source train")
        require(row["image_path"].startswith("images/train/") and row["annotation_path"].startswith("annotations/train/"), "Forbidden debug source path")
        require(not row.get("requires_annotation_quality_quarantine") and not row.get("annotation_quality_target_ids"), "Quarantined positive annotation cannot be evaluated")
    return sorted(selected, key=lambda row: row["image_id"])


def raw_guard(reference: np.ndarray, comparison: np.ndarray) -> dict:
    ref, cmp = np.asarray(reference), np.asarray(comparison)
    shapes_ok = ref.shape == cmp.shape == (1, 14, 8400)
    finite = bool(np.isfinite(ref).all() and np.isfinite(cmp).all())
    result = {"shape_reference": list(ref.shape), "shape_comparison": list(cmp.shape),
              "shape_pass": shapes_ok, "finite_pass": finite,
              "guard_name": "GATE1_SAME_PYTORCH_UNFUSED_FP32_FP16_DTYPE_GUARD",
              "old_gate05_guard_replaced": False, "tolerances_changed": False}
    if not shapes_ok or not finite:
        return result | {"status": "FAIL_SHAPE_OR_FINITE", "allclose_pass": False}
    ref, cmp = ref.astype(np.float64), cmp.astype(np.float64)
    for name, section, rtol, atol in (("boxes", slice(0, 4), 0.005, 0.5), ("scores", slice(4, 14), 0.02, 0.002)):
        absolute = np.abs(cmp[:, section] - ref[:, section])
        tolerance = atol + rtol * np.abs(ref[:, section])
        result[name] = {"rtol": rtol, "atol": atol, "elements": int(absolute.size),
                        "violations": int(np.count_nonzero(absolute > tolerance)),
                        "max_absolute_error": float(absolute.max(initial=0)),
                        "max_error_over_tolerance": float((absolute / tolerance).max(initial=0))}
    result["raw_geometry"] = {key: {"positive_width_height": bool((value[:, 2:4] > 0).all()),
                                    "class_scores_within_0_1": bool(((value[:, 4:] >= 0) & (value[:, 4:] <= 1)).all())}
                              for key, value in (("fp32", ref), ("fp16", cmp))}
    same = ref[:, 4:].argmax(axis=1) == cmp[:, 4:].argmax(axis=1)
    active = np.maximum(ref[:, 4:].max(axis=1), cmp[:, 4:].max(axis=1)) > 0.001
    result["class_argmax"] = {"all_anchors_equal": int(same.sum()), "all_anchors_total": int(same.size),
                              "candidate_union_equal": int(same[active].sum()), "candidate_union_total": int(active.sum()),
                              "criterion": "reported separately; class argmax equality is not substituted for the fixed elementwise score guard"}
    result["allclose_pass"] = result["boxes"]["violations"] == result["scores"]["violations"] == 0
    result["status"] = "PASS" if result["allclose_pass"] else "FAIL_FIXED_TOLERANCES"
    return result


def restored_predictions(detections: np.ndarray, anchor_ids: np.ndarray, params: dict, image_id: str, dtype: str) -> tuple[list[dict], dict]:
    """Restore without clipping; the frozen matcher performs ignore filtering first."""
    det = np.asarray(detections, dtype=np.float64)
    ids = np.asarray(anchor_ids).reshape(-1)
    require(det.ndim == 2 and det.shape[1] == 6 and len(det) == len(ids), "NMS output/index shape mismatch")
    require(bool(np.isfinite(det).all()), "Nonfinite NMS detections")
    ratio_x, ratio_y = params["ratio"]
    require(ratio_x > 0 and ratio_y > 0, "Invalid letterbox ratio")
    result = []
    for index, (row, anchor) in enumerate(zip(det, ids)):
        x1, y1, x2, y2, score, category = map(float, row)
        require(category.is_integer() and 0 <= category <= 9 and 0 <= score <= 1, "Invalid NMS category or score")
        x1, x2 = (x1 - params["left"]) / ratio_x, (x2 - params["left"]) / ratio_x
        y1, y2 = (y1 - params["top"]) / ratio_y, (y2 - params["top"]) / ratio_y
        require(x2 > x1 and y2 > y1, "Nonpositive restored NMS geometry")
        require(float(anchor).is_integer() and 0 <= int(anchor) < 8400, "Invalid anchor identity")
        result.append({"detection_id": f"{image_id}:{dtype}:anchor:{int(anchor)}:nms:{index}",
                       "bbox_xywh": [x1, y1, x2 - x1, y2 - y1], "score": score,
                       "category": int(category) + 1, "yolo_class_zero_based": int(category),
                       "raw_anchor_index": int(anchor), "nms_input_order": index})
    return result, {"finite": True, "positive_unclipped_geometry": True, "category_mapping_valid": True,
                    "count": len(result), "coordinates_clipped_before_ignore_filter": False}


def aggregate_target_comparison(fp32: dict, fp16: dict) -> dict:
    left = {row["target_id"]: row["hit"] for row in fp32["targets"]}
    right = {row["target_id"]: row["hit"] for row in fp16["targets"]}
    require(left.keys() == right.keys(), "FP32/FP16 eligible target identities differ")
    return {"eligible_targets": len(left), "fp32_hits": sum(left.values()), "fp16_hits": sum(right.values()),
            "both_hit": sum(left[key] and right[key] for key in left),
            "fp32_only_hit": sum(left[key] and not right[key] for key in left),
            "fp16_only_hit": sum(right[key] and not left[key] for key in left),
            "both_miss": sum(not left[key] and not right[key] for key in left)}


def validate_unfused_structure(counts: dict) -> None:
    require(counts["Conv2d"] == 88 and counts["BatchNorm2d"] > 0,
            "Expected 88 Conv2d and retained BatchNorm in the unfused model")
    require(counts["UltralyticsConv"] > 0 and counts["UltralyticsConv"] == counts["UltralyticsConv_with_BatchNorm"],
            "A Conv wrapper was already fused before qualification")


def validate_inputs(receipt_path: Path, contract_dir: Path) -> tuple[dict, list[dict], Path, dict]:
    require(sha(contract_dir / "contract.json") == CONTRACT_SHA256, "Only the frozen training_contract_v3 is supported")
    require(sha(contract_dir / "manifest.json") == CONTRACT_MANIFEST_SHA256, "Frozen contract manifest changed")
    verify_manifest(contract_dir)
    contract, receipt = read_json(contract_dir / "contract.json"), read_json(receipt_path)
    completion = validate_completion(receipt, contract, sha(contract_dir / "contract.json"), sha(contract_dir / "manifest.json"))
    require(sha(receipt_path.parent / "frozen_contract.json") == CONTRACT_SHA256, "Attempt frozen contract differs")
    for name, entry in receipt["checkpoints"].items():
        expected_path = (receipt_path.parent / "run" / "weights" / name).resolve(strict=True)
        path = Path(entry["path"]).resolve(strict=True)
        require(path == expected_path and not path.is_symlink(), "Checkpoint path not in the completed attempt")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], f"Final {name} bytes changed")
    plan_dir = Path(contract["split_plan"]).resolve(strict=True)
    require(sha(plan_dir / "manifest.json") == contract["split_plan_manifest_sha256"] == PLAN_MANIFEST_SHA256, "Wrong split plan identity")
    verify_manifest(plan_dir)
    require(sha(plan_dir / "split_assignment.jsonl") == contract["split_assignment_sha256"] == ASSIGNMENT_SHA256, "Assignment changed")
    plan = read_json(plan_dir / "split_plan.json")
    prepared = Path(plan["source_prepared"]).resolve(strict=True)
    require(sha(prepared / "image_manifest.jsonl") == plan["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256, "Source metadata changed")
    rows = select_debug_rows(read_jsonl(plan_dir / "split_assignment.jsonl"))
    source_rows = {row["image_id"]: row for row in read_jsonl(prepared / "image_manifest.jsonl")}
    for row in rows:
        source = source_rows[row["image_id"]]
        for key in ("source_split", "image_path", "image_sha256", "annotation_path", "annotation_sha256", "width", "height"):
            require(row[key] == source[key], f"Refined/source metadata disagree: {row['image_id']}:{key}")
    completion.update({"receipt": identity(receipt_path), "contract": identity(contract_dir / "contract.json"),
                       "split_plan_manifest": identity(plan_dir / "manifest.json"),
                       "checkpoint_identities": receipt["checkpoints"], "debug_image_count": len(rows)})
    return receipt, rows, prepared, completion


def configure_runtime(output: Path) -> tuple[Any, Any, Any, Any, Any, dict]:
    config = output / "ultralytics_config"
    config.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.dont_write_bytecode = True
    import cv2
    import torch
    import torchvision  # explicitly selects the known torchvision CPU NMS path
    import ultralytics
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils import SETTINGS
    SETTINGS.update({key: False for key in ("sync", "wandb", "comet", "clearml", "mlflow", "dvc", "raytune", "tensorboard")})
    require(torch.cuda.is_available(), "CUDA is required; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    root = Path(ultralytics.__file__).resolve().parent
    files = ("nn/tasks.py", "nn/modules/head.py", "nn/modules/conv.py", "nn/modules/block.py", "data/augment.py", "utils/nms.py", "utils/ops.py", "utils/metrics.py", "utils/tal.py")
    sources = {name: identity(root / name) for name in files}
    environment = {"python": sys.version, "executable": identity(Path(sys.executable)),
                   "packages": {"torch": torch.__version__, "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
                   "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu": torch.cuda.get_device_name(0), "framework_sources": sources,
                   "torchvision_nms_source": identity(Path(torchvision.ops.boxes.__file__).resolve()),
                   "tf32": False, "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "autocast_enabled": False, "deterministic_algorithms": True, "cudnn_benchmark": False,
                   "network_download_requested": False, "existing_environment_modified": False}
    for relative, expected in read_json(output / "frozen_training_contract.json")["environment_fingerprint"]["framework_source_sha256"].items():
        require(sha(root / relative) == expected, f"Training-frozen framework source changed: {relative}")
    reviewed = read_json(output / "source_review.json")
    for relative, expected in reviewed["framework_files"].items():
        require(sha(root / relative) == expected["sha256"], f"Qualification-reviewed source changed: {relative}")
    for relative, expected in SUPPLEMENTAL_SOURCE_SHA256.items():
        require(sha(root / relative) == expected, f"Supplemental inference source changed: {relative}")
    environment["reviewed_source_inventory_sha256"] = sha(output / "source_review.json")
    environment["all_reviewed_framework_sources_verified"] = len(reviewed["framework_files"])
    return torch, cv2, LetterBox, non_max_suppression, torchvision, environment


def load_models(torch: Any, checkpoint: Path) -> tuple[dict, list[dict]]:
    models, provenance = {}, []
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        # This verified, local same-version checkpoint is loaded directly. Do not
        # use a high-level predictor/backend that may fuse or inherit train args.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        key = "ema" if isinstance(payload.get("ema"), torch.nn.Module) else "model"
        model = payload.get(key)
        require(isinstance(model, torch.nn.Module), "Saved checkpoint has no model")
        saved = Counter(str(value.dtype) for value in model.state_dict().values())
        require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite saved model state")
        require(set(str(value.dtype) for value in model.parameters()) == {"torch.float16"}, "Expected fp16 EMA checkpoint serialization")
        require({int(k): v for k, v in model.names.items()} == dict(enumerate(NAMES)), "Loaded class names differ")
        head = model.model[-1]
        require(head.nc == 10 and not getattr(head, "end2end", False) and not getattr(head, "xyxy", False)
                and not getattr(head, "export", False), "Unsupported head/output semantics")
        require(max(int(x) for x in model.stride) == 32, "Unexpected model stride")
        modules = list(model.modules())
        wrappers = [module for module in modules if type(module).__name__ == "Conv" and type(module).__module__.startswith("ultralytics.nn.modules")]
        structure = {"Conv2d": sum(isinstance(module, torch.nn.Conv2d) for module in modules),
                     "BatchNorm2d": sum(isinstance(module, torch.nn.BatchNorm2d) for module in modules),
                     "UltralyticsConv": len(wrappers),
                     "UltralyticsConv_with_BatchNorm": sum(isinstance(getattr(module, "bn", None), torch.nn.BatchNorm2d) for module in wrappers)}
        validate_unfused_structure(structure)
        for module in model.modules():
            if hasattr(module, "inplace"):
                module.inplace = True
        model.float().eval()
        models[name] = model.to(device="cuda:0", dtype=dtype)
        provenance.append({"branch": name, "checkpoint_key": key, "saved_state_dtypes": dict(saved),
                           "saved_floating_parameters_dtype": "torch.float16", "cpu_load": True,
                           "fusion_called": False, "effective_dtype": str(dtype),
                           "unfused_structure": structure,
                           "reference_is_original_fp32_training_state": False})
    return models, provenance


def guarded_nms(nms: Any, raw: Any) -> tuple[Any, Any]:
    """Reject the framework's time-limit warning instead of accepting truncation."""
    from ultralytics.utils import LOGGER

    class TimeLimitHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            message = record.getMessage()
            if "NMS time limit" in message:
                self.messages.append(message)

    handler = TimeLimitHandler()
    LOGGER.addHandler(handler)
    try:
        result = nms(raw.clone(), conf_thres=0.001, iou_thres=0.7, agnostic=False,
                     multi_label=False, max_det=1000, nc=10, max_time_img=60.0,
                     max_nms=30000, max_wh=7680, return_idxs=True)
    finally:
        LOGGER.removeHandler(handler)
    require(not handler.messages, "NMS time limit warning: " + "; ".join(handler.messages))
    return result


def run(receipt_path: Path, contract_dir: Path, output: Path, result: dict) -> None:
    receipt, rows, prepared, completion = validate_inputs(receipt_path, contract_dir)
    result["input_validation"] = completion
    result["phase"] = "COPYING_DEBUG_AND_FINAL_CHECKPOINT"
    write_json(output / "qualification_receipt.json", result)
    shutil.copyfile(contract_dir / "contract.json", output / "frozen_training_contract.json")
    shutil.copyfile(receipt_path, output / "final_training_receipt.json")
    shutil.copyfile(Path(__file__), output / "runner_source.py")
    review_dir = Path(__file__).resolve().parents[1] / "artifacts" / "gate1" / "20261022_float_qualification_preparation" / "source_review"
    require(sha(review_dir / "source_review.json") == SOURCE_REVIEW_SHA256, "Qualification source review changed")
    require(sha(review_dir / "qualification_contract_prepared.json") == PREPARED_SOURCE_CONTRACT_SHA256, "Reviewed preparation contract changed")
    require(sha(review_dir / "manifest.json") == SOURCE_REVIEW_MANIFEST_SHA256, "Source review manifest changed")
    verify_manifest(review_dir)
    shutil.copyfile(review_dir / "source_review.json", output / "source_review.json")
    shutil.copyfile(review_dir / "qualification_contract_prepared.json", output / "reviewed_source_contract.json")
    evaluator = Path(__file__).with_name("gate1_ignore_aware_evaluation.py")
    require(sha(evaluator) == EVALUATOR_SHA256, "Frozen ignore-aware evaluator changed")
    shutil.copyfile(evaluator, output / "ignore_aware_evaluation_source.py")
    checkpoint = output / "best.pt"
    shutil.copyfile(Path(receipt["checkpoints"]["best.pt"]["path"]), checkpoint)
    require(sha(checkpoint) == receipt["checkpoints"]["best.pt"]["sha256"], "Copied checkpoint differs")
    copied = []
    for row in rows:
        for path_key, hash_key, subdir in (("image_path", "image_sha256", "images"), ("annotation_path", "annotation_sha256", "annotations")):
            source = safe_file(prepared, row[path_key])
            require(sha(source) == row[hash_key], "Debug source bytes changed")
            destination = output / "debug" / subdir / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            require(not destination.exists(), "Copied debug filename collision")
            shutil.copyfile(source, destination)
            require(sha(destination) == row[hash_key], "Debug copy bytes differ")
        copied.append(row)
    write_json(output / "debug_source_manifest.json", {"rows": copied, "selection_or_holdout_copied": False})
    write_json(output / "qualification_protocol.json", protocol())
    torch, cv2, LetterBox, nms, _, environment = configure_runtime(output)
    write_json(output / "environment.json", environment)
    models, provenance = load_models(torch, checkpoint)
    write_json(output / "model_provenance.json", {"models": provenance, "checkpoint": identity(checkpoint)})
    spec = importlib.util.spec_from_file_location("frozen_gate1_ignore_evaluation", output / "ignore_aware_evaluation_source.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    letterbox = LetterBox((640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32, padding_value=114, interpolation=cv2.INTER_LINEAR)
    raw_dir, details_dir = output / "raw", output / "per_image"
    raw_dir.mkdir()
    details_dir.mkdir()
    summary = {str(conf): {"fp32": Counter(), "fp16": Counter(), "paired_targets": Counter()} for conf in THRESHOLDS}
    guards = []
    result.update(phase="INFERENCE_DEBUG_ONLY", inference_started=True)
    write_json(output / "qualification_receipt.json", result)
    with torch.inference_mode(), torch.autocast("cuda", enabled=False), (output / "image_progress.jsonl").open("x", encoding="utf-8") as progress:
        for number, row in enumerate(rows, 1):
            name = Path(row["image_path"]).stem
            image_file = output / "debug" / "images" / Path(row["image_path"]).name
            image = cv2.imdecode(np.frombuffer(image_file.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(image is not None and image.shape == (row["height"], row["width"], 3), "Decoded dimensions/channels differ")
            params = letterbox.get_params({"img": image})
            padded = letterbox.apply_image({"img": image}, params)["img"]
            require(padded.shape == (640, 640, 3), "Letterbox shape differs")
            array = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))
            normalized = torch.from_numpy(array).unsqueeze(0).float().div(255)
            normalized_sha = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
            tensor = normalized.to("cuda:0")
            raw, arrays = {}, {}
            for dtype in ("fp32", "fp16"):
                out = models[dtype](tensor if dtype == "fp32" else tensor.half(), augment=False)
                decoded = out[0] if isinstance(out, (tuple, list)) else out
                require(isinstance(decoded, torch.Tensor), "Unsupported model output")
                require(decoded.dtype == (torch.float32 if dtype == "fp32" else torch.float16), "Actual raw model dtype differs from its fixed branch")
                arrays[dtype] = decoded.detach().cpu().numpy().copy()
                raw[dtype] = decoded.detach().float().cpu().clone()
            torch.cuda.synchronize()
            np.savez_compressed(raw_dir / f"{name}.npz", fp32=arrays["fp32"], fp16=arrays["fp16"])
            guard = raw_guard(arrays["fp32"], arrays["fp16"])
            guard["image_id"] = row["image_id"]
            guards.append(guard)
            write_json(output / "raw_dtype_guard_progress.json", {"images_completed": number, "guards": guards})
            require(guard["shape_pass"] and guard["finite_pass"], "Raw shape/finite check failed; preserved raw tensors")
            require(all(check["positive_width_height"] and check["class_scores_within_0_1"] for check in guard["raw_geometry"].values()), "Invalid decoded box/score domains")
            gt = evaluation.parse_ground_truth((output / "debug" / "annotations" / Path(row["annotation_path"]).name).read_text(encoding="utf-8"), row["image_id"])
            item = {"image_id": row["image_id"], "image_sha256": row["image_sha256"], "annotation_sha256": row["annotation_sha256"],
                    "letterbox_params": {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()},
                    "preprocessed_uint8_sha256": hashlib.sha256(array.tobytes()).hexdigest(), "raw_guard": guard,
                    "normalized_fp32_input_sha256": normalized_sha,
                    "raw_output_dtypes": {dtype: str(value.dtype) for dtype, value in arrays.items()},
                    "original_ground_truth": gt, "predictions": {}, "matching": {}, "coordinate_checks": {}}
            predictions = {}
            for dtype in ("fp32", "fp16"):
                # NMS mutates xywh channels through a transpose view. Only a clone
                # is passed, after the immutable raw tensors were saved/compared.
                detections, anchors = guarded_nms(nms, raw[dtype])
                predictions[dtype], coordinates = restored_predictions(detections[0].numpy(), anchors[0].numpy(), params, row["image_id"], dtype)
                item["predictions"][dtype] = predictions[dtype]
                coordinates["max_det_cap"] = 1000
                coordinates["max_det_saturated"] = len(predictions[dtype]) == 1000
                coordinates["NMS_time_limit_warning"] = False
                item["coordinate_checks"][dtype] = coordinates
            for conf in THRESHOLDS:
                left = evaluation.match_project_targets(gt, predictions["fp32"], row["width"], row["height"], confidence=conf, iou_threshold=0.5)
                right = evaluation.match_project_targets(gt, predictions["fp16"], row["width"], row["height"], confidence=conf, iou_threshold=0.5)
                pair = aggregate_target_comparison(left, right)
                item["matching"][str(conf)] = {"fp32": left, "fp16": right, "paired_targets": pair}
                summary[str(conf)]["fp32"].update(left["counts"])
                summary[str(conf)]["fp16"].update(right["counts"])
                summary[str(conf)]["paired_targets"].update(pair)
            write_json(details_dir / f"{name}.json", item)
            progress.write(json.dumps({"image_id": row["image_id"], "images_completed": number, "raw_guard_status": guard["status"]}) + "\n")
            progress.flush()
            result["images_completed"] = number
            write_json(output / "qualification_receipt.json", result)
            print(f"DEBUG_QUALIFICATION {number}/{len(rows)} {guard['status']}", flush=True)
    require(sha(receipt_path) == completion["receipt"]["sha256"], "Final training receipt changed during qualification")
    source_checks = []
    for row in rows:
        for key, digest in (("image_path", "image_sha256"), ("annotation_path", "annotation_sha256")):
            require(sha(safe_file(prepared, row[key])) == row[digest], "Debug source changed during qualification")
        source_checks.append(row["image_id"])
    write_json(output / "source_after_check.json", {"verified_images_and_original_annotations": len(source_checks), "unchanged": True})
    for threshold in summary.values():
        for dtype in ("fp32", "fp16"):
            counts = threshold[dtype]
            counts["project_target_recall"] = counts["hits"] / counts["eligible_targets"] if counts["eligible_targets"] else None
            denominator = counts["hits"] + counts["false_positive_detections"]
            counts["project_detection_precision"] = counts["hits"] / denominator if denominator else None
    guard_pass = all(guard["status"] == "PASS" for guard in guards)
    write_json(output / "matching_summary.json", {"metric": "PROJECT_IGNORE_AWARE_TARGET_MATCHING", "official_AP": False,
               "oracle_equivalence_verified": False, "data_role": "debug", "size_scope": "all eligible sizes; no small-only metric", "thresholds": summary, "limits": LIMITS})
    write_json(output / "raw_dtype_guard.json", {"status": "PASS" if guard_pass else "FAIL_FIXED_TOLERANCES", "images": len(guards),
               "box_rtol": 0.005, "box_atol": 0.5, "score_rtol": 0.02, "score_atol": 0.002,
               "guard_failed_images": [row["image_id"] for row in guards if row["status"] != "PASS"],
               "guards": guards, "old_gate05_guard_replaced": False})
    result.update(status="COMPLETE_DEBUG_FLOAT_QUALIFICATION_REQUIRES_REVIEW", phase="COMPLETE", images_completed=len(rows),
                  same_backend_dtype_guard="PASS" if guard_pass else "FAIL_FIXED_TOLERANCES", raw_finite_and_shape=True,
                  debug_project_matching_complete=True, accepted_float_baseline_ready=False,
                  scientific_gate_pass=False, official_visdrone_ap=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-receipt", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.final_receipt, args.contract_dir, args.output):
        require(not any(word in str(path).casefold() for word in ("selection", "holdout", "testdev", "test-dev", "test-challenge")), "Forbidden data-role path")
    require(not args.output.exists() and not args.output.is_symlink(), "Output must be a new directory")
    receipt_path, contract_dir = args.final_receipt.resolve(strict=True), args.contract_dir.resolve(strict=True)
    output = args.output.resolve()
    require(not any(output.is_relative_to(root) for root in (receipt_path.parent, contract_dir)), "Output cannot modify frozen training inputs")
    # Bind the source metadata before creating an output under an unintended tree.
    contract = read_json(contract_dir / "contract.json")
    plan_root = Path(contract["split_plan"]).resolve(strict=True)
    prepared = Path(read_json(plan_root / "split_plan.json")["source_prepared"]).resolve(strict=True)
    require(not any(output.is_relative_to(root) for root in (prepared, plan_root)), "Output cannot modify prepared inputs")
    output.mkdir(parents=True)
    started = time.monotonic()
    result = {"schema": "rampq.gate1.float_qualification_attempt/v1", "status": "VALIDATING_COMPLETED_TRAINING", "phase": "INPUT_CHECKS",
              "started_at_utc": now(), "runner_sha256": sha(Path(__file__)), "inference_started": False,
              "selection_detector_inference": False, "holdout_accessed": False, "accepted_float_baseline_ready": False,
              "scientific_gate_pass": False, "official_visdrone_ap": False, "limits": LIMITS}
    code = 0
    try:
        run(receipt_path, contract_dir, output, result)
    except Exception as error:
        result.update(status="FAILED_PRESERVED_NO_RETRY", error_type=type(error).__name__, error=str(error))
        code = 1
    finally:
        result.update(finished_at_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_json(output / "qualification_receipt.json", result)
        files = {str(path.relative_to(output)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha(path)}
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        write_json(output / "manifest.json", {"files": files})
    print(json.dumps({"status": result["status"], "output": str(output), "inference_started": result["inference_started"]}, ensure_ascii=False), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
