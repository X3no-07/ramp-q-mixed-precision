"""Audit a completed debug qualification from saved files, without loading a model.

Raw dtype checks and target/count partitions are recomputed independently. The
frozen project matcher is reused only for ignore-aware matching consistency; it
is not an independent official VisDrone oracle. All reads stay in the supplied
qualification directory. No image decoding, training or checkpoint deserialization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import types
from typing import Any

import numpy as np


QUALIFIER_SHA = "b981869ad658e0619454cdbe2361c8a34bd029adf2a8bbb7f85590853c736f34"
EVALUATOR_SHA = "4e45ff26a86a83ff66e997f15e4c4794246c6440c304446972809ff32db09961"
CONTRACT_SHA = "6dfc1b1c8b2d89948c5804cfda8f22c92cbd721b3dfaf0cffdc314b2b2bd4fc0"
ASSIGNMENT_SHA = "50113cde0843076f9524a55b75d54392fced1c5e1092bb9d28d84649272ea025"
DEBUG_CANONICAL_SHA = "059b06216d35535d91c21b34955ba545331664b748edd4f822802cd78e927e75"
DEBUG_IDENTITY_KEYS = ("image_id", "image_path", "image_sha256", "annotation_path", "annotation_sha256", "width", "height", "valid_target_count", "refined_component")
THRESHOLDS = ("0.2", "0.25", "0.3")
LIMITS = [
    "Debug images are reused engineering data, not independent Selection/Holdout evidence.",
    "Exact debug identities are bound to metadata derived before execution from inventory SHA d056e3782aabce761d268cf3875ca8981e15be591bfa8f12d3cc8a42e017e7a1. Full split isolation is not reconstructed again here.",
    "The frozen project matcher is reused for ignore-aware consistency and is not an independent official VisDrone oracle or AP implementation.",
    "Images are hashed, not decoded. Normalized/preprocessed input hashes cannot be reconstructed without decoding; stored hash syntax alone is checked.",
    "Predictions are checked against their saved raw anchors, but torchvision NMS selection/order and model computation are not rerun.",
    "Checkpoint bytes are hashed only; finite model parameters and unfused runtime structure are not independently deserialized.",
    "A consistent qualification with a failed dtype guard remains a failed dtype qualification. This audit never relaxes tolerances or replaces the original Gate0.5 guard.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    require(not rel.is_absolute() and bool(rel.parts) and ".." not in rel.parts, f"Unsafe path {relative}")
    cursor = root
    for part in rel.parts:
        cursor /= part
        require(not cursor.is_symlink(), f"Symlink forbidden: {relative}")
    resolved = cursor.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), f"File escapes root: {relative}")
    return resolved


def verify_manifest(root: Path) -> dict:
    manifest_path = safe_file(root, "manifest.json")
    files = read_json(manifest_path)["files"]
    require(isinstance(files, dict) and bool(files), "Empty manifest")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Symlinks forbidden in qualification")
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(files), "Manifest omits files or lists nonexistent files")
    for relative, entry in files.items():
        path = safe_file(root, relative)
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], f"Manifest mismatch: {relative}")
    return {"sha256": sha(manifest_path), "files_verified": len(files)}


def calculate_raw_guard(reference: np.ndarray, comparison: np.ndarray) -> tuple[dict, dict]:
    """Recalculate saved-raw metrics with NumPy, asymmetric FP32 reference."""
    shapes = reference.shape == comparison.shape == (1, 14, 8400)
    finite = bool(np.isfinite(reference).all() and np.isfinite(comparison).all())
    result = {"shape_reference": list(reference.shape), "shape_comparison": list(comparison.shape),
              "shape_pass": shapes, "finite_pass": finite,
              "guard_name": "GATE1_SAME_PYTORCH_UNFUSED_FP32_FP16_DTYPE_GUARD",
              "old_gate05_guard_replaced": False, "tolerances_changed": False}
    domains = {"dtype_pass": reference.dtype == np.dtype("float32") and comparison.dtype == np.dtype("float16")}
    if not shapes or not finite:
        return result | {"status": "FAIL_SHAPE_OR_FINITE", "allclose_pass": False}, domains | {"domain_pass": False}
    a, b = reference.astype(np.float64), comparison.astype(np.float64)
    passes = []
    for name, lo, hi, rtol, atol in (("boxes", 0, 4, .005, .5), ("scores", 4, 14, .02, .002)):
        x, y = a[:, lo:hi], b[:, lo:hi]
        errors = np.abs(y - x)
        bounds = atol + rtol * np.abs(x)
        passes.append(bool(np.allclose(y, x, rtol=rtol, atol=atol, equal_nan=False)))
        result[name] = {"rtol": rtol, "atol": atol, "elements": int(errors.size),
                        "violations": int((errors > bounds).sum()), "max_absolute_error": float(np.max(errors)),
                        "max_error_over_tolerance": float(np.max(errors / bounds))}
    result["raw_geometry"] = {}
    for name, values in (("fp32", a), ("fp16", b)):
        result["raw_geometry"][name] = {"positive_width_height": bool(np.min(values[:, 2:4]) > 0),
                                       "class_scores_within_0_1": bool(np.min(values[:, 4:]) >= 0 and np.max(values[:, 4:]) <= 1)}
    equal = np.argmax(a[:, 4:], axis=1) == np.argmax(b[:, 4:], axis=1)
    active = np.maximum(np.max(a[:, 4:], axis=1), np.max(b[:, 4:], axis=1)) > .001
    result["class_argmax"] = {"all_anchors_equal": int(equal.sum()), "all_anchors_total": int(equal.size),
                              "candidate_union_equal": int(equal[active].sum()), "candidate_union_total": int(active.sum()),
                              "criterion": "reported separately; class argmax equality is not substituted for the fixed elementwise score guard"}
    result["allclose_pass"] = all(passes)
    result["status"] = "PASS" if all(passes) else "FAIL_FIXED_TOLERANCES"
    domains["domain_pass"] = all(all(values.values()) for values in result["raw_geometry"].values())
    return result, domains


def parse_original_annotation(text: str, image_id: str) -> list[dict]:
    """Preserve zero-based physical line IDs independently of evaluator parsing."""
    result = []
    for line, raw in enumerate(text.removeprefix("\ufeff").splitlines()):
        if not raw.strip():
            continue
        fields = raw.split(",")
        if len(fields) == 9 and not fields[-1].strip():
            fields.pop()
        require(len(fields) == 8, f"Invalid annotation physical line {line}")
        values = [float(value.strip()) for value in fields]
        require(all(math.isfinite(value) for value in values) and all(value.is_integer() for value in values[4:]), "Invalid annotation values")
        x, y, width, height, score, category, truncation, occlusion = values
        require(score in (0, 1) and 0 <= category <= 11 and truncation in (0, 1, 2, 3) and occlusion in (0, 1, 2), "Invalid annotation attributes")
        result.append({"target_id": f"{image_id}:{line}", "physical_line_zero_based": line,
                       "bbox_xywh": [x, y, width, height], "score": int(score), "category": int(category),
                       "truncation": int(truncation), "occlusion": int(occlusion)})
    return result


def canonical_debug_identity(rows: list[dict]) -> str:
    selected = sorted(({key: row[key] for key in DEBUG_IDENTITY_KEYS} for row in rows), key=lambda row: row["image_id"])
    payload = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_letterbox(params: dict, width: int, height: int) -> None:
    ratio = min(640 / width, 640 / height)
    resized = [round(width * ratio), round(height * ratio)]
    dx, dy = (640 - resized[0]) / 2, (640 - resized[1]) / 2
    expected = {"orig_shape": [height, width], "new_shape": [640, 640], "ratio": [ratio, ratio],
                "new_unpad": resized, "top": round(dy - .1), "bottom": round(dy + .1),
                "left": round(dx - .1), "right": round(dx + .1)}
    require(params == expected, "Letterbox metadata differs from fixed geometry")


def verify_predictions(predictions: list[dict], raw: np.ndarray, params: dict, image_id: str, branch: str) -> None:
    require(len(predictions) <= 1000, "NMS cap exceeded")
    seen = set()
    for order, prediction in enumerate(predictions):
        anchor = prediction["raw_anchor_index"]
        require(isinstance(anchor, int) and not isinstance(anchor, bool) and 0 <= anchor < 8400 and anchor not in seen, "Invalid/duplicate raw anchor")
        seen.add(anchor)
        require(prediction["nms_input_order"] == order and prediction["detection_id"] == f"{image_id}:{branch}:anchor:{anchor}:nms:{order}", "Prediction identity/order mismatch")
        values = raw[0, :, anchor].astype(np.float32)
        category = int(np.argmax(values[4:]))
        score = float(values[4 + category])
        require(prediction["category"] == category + 1 and prediction["yolo_class_zero_based"] == category, "Category +1/argmax mapping differs")
        require(prediction["score"] == score and .001 < score <= 1, "Prediction score differs from raw anchor")
        half_size = values[2:4] / np.float32(2)
        corners = np.concatenate((values[:2] - half_size, values[:2] + half_size)).astype(np.float64)
        x1, y1, x2, y2 = corners
        x1, x2 = (x1 - params["left"]) / params["ratio"][0], (x2 - params["left"]) / params["ratio"][0]
        y1, y2 = (y1 - params["top"]) / params["ratio"][1], (y2 - params["top"]) / params["ratio"][1]
        box = np.asarray(prediction["bbox_xywh"], dtype=np.float64)
        require(box.shape == (4,) and bool(np.isfinite(box).all()) and bool((box[2:] > 0).all()), "Invalid restored prediction geometry")
        require(bool(np.allclose(box, [x1, y1, x2 - x1, y2 - y1], rtol=1e-12, atol=1e-9)), "Prediction restoration differs from raw anchor")


def recount_matching(matching: dict, predictions: list[dict]) -> dict:
    targets, matches, states = matching["targets"], matching["matches"], matching["detection_states"]
    target_map = {row["target_id"]: row["hit"] for row in targets}
    require(len(target_map) == len(targets) and all(type(value) is bool for value in target_map.values()), "Invalid target hit map")
    matched_targets = {row["target_id"] for row in matches}
    matched_detections = {row["detection_id"] for row in matches}
    require(len(matched_targets) == len(matched_detections) == len(matches), "Matching is not one-to-one")
    require(matched_targets == {key for key, hit in target_map.items() if hit}, "Hit flags differ from matched target IDs")
    require(not matched_targets.intersection(row["target_id"] for row in matching["excluded_ground_truth"]), "Excluded GT was matched")
    state_map = {row["detection_id"]: row for row in states}
    require(len(state_map) == len(states) == len(predictions) and set(state_map) == {row["detection_id"] for row in predictions}, "Detection states do not partition predictions")
    require(matched_detections == {key for key, value in state_map.items() if value["reason"] == "true_positive"}, "Detection true positives differ")
    for match in matches:
        require(state_map[match["detection_id"]].get("target_id") == match["target_id"], "State target ID differs")
        require(.5 <= match["iou"] <= 1 and 0 <= match["confidence"] <= 1, "Invalid match IoU/confidence")
    reasons = Counter(row["reason"] for row in states)
    allowed = {"true_positive", "false_positive", "absorbed_by_same_class_score_0", "category_0_raster_coverage_ge_0.5", "outside_evaluated_categories", "below_confidence", "empty_after_clipping"}
    require(set(reasons) <= allowed, "Unknown detection state")
    counts = {"eligible_targets": len(targets), "hits": len(matches), "misses": len(targets) - len(matches),
              "false_positive_detections": reasons["false_positive"], "score_0_absorbed_detections": reasons["absorbed_by_same_class_score_0"],
              "category_0_dropped_detections": reasons["category_0_raster_coverage_ge_0.5"]}
    require(matching["counts"] == counts, "Matching counts differ from target/state partitions")
    return counts


def paired_partition(left: dict, right: dict) -> dict:
    a = {row["target_id"]: row["hit"] for row in left["targets"]}
    b = {row["target_id"]: row["hit"] for row in right["targets"]}
    require(a.keys() == b.keys(), "Branch eligible target IDs differ")
    parts = Counter((a[key], b[key]) for key in a)
    return {"eligible_targets": len(a), "fp32_hits": sum(a.values()), "fp16_hits": sum(b.values()),
            "both_hit": parts[True, True], "fp32_only_hit": parts[True, False],
            "fp16_only_hit": parts[False, True], "both_miss": parts[False, False]}


def audit(root: Path, output: Path) -> dict:
    manifest_check = verify_manifest(root)
    receipt = read_json(safe_file(root, "qualification_receipt.json"))
    require(receipt.get("schema") == "rampq.gate1.float_qualification_attempt/v1" and receipt.get("status") == "COMPLETE_DEBUG_FLOAT_QUALIFICATION_REQUIRES_REVIEW" and receipt.get("phase") == "COMPLETE", "Only completed qualifications may be audited")
    for field in ("selection_detector_inference", "holdout_accessed", "accepted_float_baseline_ready", "scientific_gate_pass", "official_visdrone_ap"):
        require(receipt.get(field) is False, f"Forbidden/overstated receipt flag: {field}")
    require(receipt.get("images_completed") == 109 and receipt.get("inference_started") is True and receipt.get("raw_finite_and_shape") is True and receipt.get("debug_project_matching_complete") is True, "Incomplete debug qualification")
    require(receipt.get("runner_sha256") == sha(safe_file(root, "runner_source.py")) == QUALIFIER_SHA, "Qualifier source identity differs")
    require(sha(safe_file(root, "frozen_training_contract.json")) == CONTRACT_SHA, "Frozen contract differs")
    evaluator_file = safe_file(root, "ignore_aware_evaluation_source.py")
    require(sha(evaluator_file) == EVALUATOR_SHA, "Frozen evaluator differs")
    protocol = read_json(safe_file(root, "qualification_protocol.json"))
    require(protocol.get("data_role") == "debug" and protocol.get("expected_images") == 109 and protocol.get("split_assignment_sha256") == ASSIGNMENT_SHA, "Protocol source role differs")
    require(protocol["raw"]["box_rtol"] == .005 and protocol["raw"]["box_atol"] == .5 and protocol["raw"]["score_rtol"] == .02 and protocol["raw"]["score_atol"] == .002, "Tolerances differ from frozen policy")
    require(protocol["matching"]["target_iou"] == .5 and protocol["matching"]["implementation_sha256"] == EVALUATOR_SHA, "Matching policy differs")
    # Execute only the hash-verified pure evaluator without creating __pycache__
    # in the immutable input directory or invoking its command-line entry point.
    evaluator = types.ModuleType("audited_frozen_project_evaluator")
    evaluator.__file__ = str(evaluator_file)
    exec(compile(evaluator_file.read_bytes(), str(evaluator_file), "exec"), evaluator.__dict__)
    sources = read_json(safe_file(root, "debug_source_manifest.json"))
    require(sources.get("selection_or_holdout_copied") is False, "Forbidden source access")
    rows = sources["rows"]
    require(canonical_debug_identity(rows) == DEBUG_CANONICAL_SHA, "Debug member metadata differs from frozen 109-image identity")
    ids = [row["image_id"] for row in rows]
    require(len(rows) == 109 and len(set(ids)) == 109 and ids == sorted(ids), "Wrong debug count/order")
    expected_stems = set()
    summary = {threshold: {"fp32": Counter(), "fp16": Counter(), "paired_targets": Counter()} for threshold in THRESHOLDS}
    guards, checks = [], []
    checkpoint = safe_file(root, "best.pt")
    training = read_json(safe_file(root, "final_training_receipt.json"))
    require(training.get("status") == "COMPLETE_FLOAT_TRAINING_REQUIRES_SCIENTIFIC_REVIEW" and training.get("phase") == "COMPLETE", "Training receipt is not completed")
    require(sha(checkpoint) == training["checkpoints"]["best.pt"]["sha256"] and checkpoint.stat().st_size == training["checkpoints"]["best.pt"]["bytes"], "Qualification checkpoint bytes differ")
    for row in rows:
        image_id = row["image_id"]
        require(row.get("assigned_split") == "debug" and row.get("source_split") == "train" and image_id.startswith("train/") and not row.get("requires_annotation_quality_quarantine") and not row.get("annotation_quality_target_ids"), "Non-debug/quality-quarantined source")
        require(row["image_path"].startswith("images/train/") and row["annotation_path"].startswith("annotations/train/"), "Forbidden source paths")
        stem = Path(row["image_path"]).stem
        require(image_id == f"train/{stem}" and stem not in expected_stems, "Source identity/stem collision")
        expected_stems.add(stem)
        for path_key, hash_key, folder in (("image_path", "image_sha256", "images"), ("annotation_path", "annotation_sha256", "annotations")):
            copied = safe_file(root, f"debug/{folder}/{Path(row[path_key]).name}")
            require(sha(copied) == row[hash_key], "Copied source bytes differ")
        item = read_json(safe_file(root, f"per_image/{stem}.json"))
        require(item["image_id"] == image_id and item["image_sha256"] == row["image_sha256"] and item["annotation_sha256"] == row["annotation_sha256"], "Per-image source identity differs")
        annotation = safe_file(root, f"debug/annotations/{Path(row['annotation_path']).name}")
        ground_truth = parse_original_annotation(annotation.read_text(encoding="utf-8"), image_id)
        require(item["original_ground_truth"] == ground_truth, "Original physical-line GT identity/content differs")
        check_letterbox(item["letterbox_params"], row["width"], row["height"])
        for key in ("preprocessed_uint8_sha256", "normalized_fp32_input_sha256"):
            require(isinstance(item[key], str) and len(item[key]) == 64 and set(item[key]) <= set("0123456789abcdef"), "Malformed stored input SHA")
        with np.load(safe_file(root, f"raw/{stem}.npz"), allow_pickle=False) as archive:
            require(set(archive.files) == {"fp32", "fp16"}, "Unexpected raw NPZ keys")
            arrays = {branch: archive[branch] for branch in ("fp32", "fp16")}
        guard, domains = calculate_raw_guard(arrays["fp32"], arrays["fp16"])
        guard["image_id"] = image_id
        require(guard == item["raw_guard"], "Per-image dtype guard disagrees with raw tensors")
        require(guard["shape_pass"] and guard["finite_pass"] and domains["dtype_pass"] and domains["domain_pass"], "Completed qualification has invalid raw dtype/shape/finite/domain")
        require(item["raw_output_dtypes"] == {branch: str(value.dtype) for branch, value in arrays.items()}, "Reported raw dtype differs")
        guards.append(guard)
        for branch in ("fp32", "fp16"):
            predictions = item["predictions"][branch]
            verify_predictions(predictions, arrays[branch], item["letterbox_params"], image_id, branch)
            coordinate = item["coordinate_checks"][branch]
            require(coordinate == {"finite": True, "positive_unclipped_geometry": True, "category_mapping_valid": True,
                                  "count": len(predictions), "coordinates_clipped_before_ignore_filter": False, "max_det_cap": 1000,
                                  "max_det_saturated": len(predictions) == 1000, "NMS_time_limit_warning": False}, "Coordinate check report differs")
        require(set(item["matching"]) == set(THRESHOLDS), "Unexpected confidence thresholds")
        for threshold in THRESHOLDS:
            matching = item["matching"][threshold]
            for branch in ("fp32", "fp16"):
                predictions = item["predictions"][branch]
                fresh = evaluator.match_project_targets(ground_truth, predictions, row["width"], row["height"], confidence=float(threshold), iou_threshold=.5)
                require(fresh == matching[branch], "Frozen project matcher recomputation differs")
                summary[threshold][branch].update(recount_matching(matching[branch], predictions))
            partition = paired_partition(matching["fp32"], matching["fp16"])
            require(partition == matching["paired_targets"], "Paired target partition differs")
            summary[threshold]["paired_targets"].update(partition)
        checks.append({"image_id": image_id, "raw_guard_status": guard["status"], **domains, "matching_thresholds_recomputed": 3})
    require({path.stem for path in (root / "raw").glob("*.npz")} == expected_stems and {path.stem for path in (root / "per_image").glob("*.json")} == expected_stems, "Missing/extra raw or per-image records")
    for threshold in summary.values():
        for branch in ("fp32", "fp16"):
            counts = threshold[branch]
            counts["project_target_recall"] = counts["hits"] / counts["eligible_targets"] if counts["eligible_targets"] else None
            denominator = counts["hits"] + counts["false_positive_detections"]
            counts["project_detection_precision"] = counts["hits"] / denominator if denominator else None
    saved_summary = read_json(safe_file(root, "matching_summary.json"))
    require(saved_summary["thresholds"] == summary and saved_summary.get("official_AP") is False and saved_summary.get("oracle_equivalence_verified") is False and saved_summary.get("data_role") == "debug", "Aggregate matching summary differs")
    failed = [guard["image_id"] for guard in guards if guard["status"] != "PASS"]
    dtype_status = "FAIL_FIXED_TOLERANCES" if failed else "PASS"
    saved_guard = read_json(safe_file(root, "raw_dtype_guard.json"))
    require(saved_guard == {"status": dtype_status, "images": 109, "box_rtol": .005, "box_atol": .5, "score_rtol": .02, "score_atol": .002,
                            "guard_failed_images": failed, "guards": guards, "old_gate05_guard_replaced": False}, "Aggregate dtype summary differs")
    require(receipt.get("same_backend_dtype_guard") == dtype_status, "Receipt overrides actual dtype verdict")
    progress = [json.loads(line) for line in safe_file(root, "image_progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(progress == [{"image_id": image_id, "images_completed": number, "raw_guard_status": guard["status"]} for number, (image_id, guard) in enumerate(zip(ids, guards), 1)], "Image progress differs")
    require(read_json(safe_file(root, "raw_dtype_guard_progress.json")) == {"images_completed": 109, "guards": guards}, "Guard progress differs")
    require(read_json(safe_file(root, "source_after_check.json")) == {"verified_images_and_original_annotations": 109, "unchanged": True}, "Source-after report differs")
    require(verify_manifest(root) == manifest_check, "Qualification changed during audit")
    result = {"schema": "rampq.gate1.float_qualification_audit/v1", "status": "PASS_SAVED_OUTPUT_CONSISTENCY_ONLY", "qualification_manifest": manifest_check,
              "images": 109, "dtype_guard_preserved": dtype_status, "dtype_failed_images": failed, "thresholds": summary, "per_image_checks": checks,
              "model_executed": False, "checkpoint_deserialized": False, "image_pixels_decoded": False, "external_source_inputs_opened": False,
              "selection_detector_inference": False, "holdout_accessed": False, "scientific_gate_pass": False, "official_visdrone_ap": False,
              "independent_official_oracle": False, "old_gate05_guard_replaced": False, "limits": LIMITS}
    write_json(output / "recomputed_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.qualification, args.output):
        require(not any(token in str(path).casefold() for token in ("selection", "holdout", "test-dev", "testdev", "test-challenge")), "Forbidden data-role path")
    require(not args.qualification.is_symlink(), "Qualification symlink forbidden")
    root = args.qualification.resolve(strict=True)
    output = args.output.resolve()
    require(not output.exists() and not args.output.is_symlink() and not output.is_relative_to(root) and not root.is_relative_to(output), "Output must be a new sibling directory")
    output.mkdir(parents=True)
    result = {"schema": "rampq.gate1.float_qualification_audit_attempt/v1", "status": "AUDITING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "runner_sha256": sha(Path(__file__)), "qualification": str(root), "model_executed": False, "scientific_gate_pass": False, "limits": LIMITS}
    code = 0
    try:
        summary = audit(root, output)
        result.update(status=summary["status"], dtype_guard_preserved=summary["dtype_guard_preserved"], images=summary["images"])
    except Exception as error:
        result.update(status="FAIL_SAVED_OUTPUT_AUDIT_PRESERVED", error_type=type(error).__name__, error=str(error))
        code = 1
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "audit_receipt.json", result)
    files = {path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    write_json(output / "manifest.json", {"files": files})
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
