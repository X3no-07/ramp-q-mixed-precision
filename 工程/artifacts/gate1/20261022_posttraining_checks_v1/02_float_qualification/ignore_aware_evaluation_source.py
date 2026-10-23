"""Ignore-aware, single-image Gate1 matching; no model and no AP implementation.

The raster prefilter follows the inspected VisDrone MATLAB source semantics.
``match_project_targets`` uses the project's IoU-first deterministic matching;
``match_official_style`` is a separately named score-first matching translation.
Neither function has been certified against a MATLAB/Octave oracle. This module
does not download data, perform NMS, aggregate AP, or choose model checkpoints.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"
MAX_IMAGE_PIXELS = 32_000_000
EVALUATED_CATEGORIES = frozenset(range(1, 11))


class EvaluationInputError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationInputError(message)


def _csv_rows(text: str) -> list[tuple[int, list[float]]]:
    rows = []
    for line, raw in enumerate(text.lstrip("\ufeff").splitlines()):
        if not raw.strip():
            continue
        fields = raw.split(",")
        if len(fields) == 9 and not fields[-1].strip():
            fields.pop()
        require(len(fields) == 8, f"Expected eight CSV fields at physical line {line}")
        try:
            values = [float(value.strip()) for value in fields]
        except ValueError as error:
            raise EvaluationInputError(f"Nonnumeric CSV field at physical line {line}") from error
        require(all(math.isfinite(value) for value in values), f"Nonfinite CSV value at physical line {line}")
        rows.append((line, values))
    return rows


def parse_ground_truth(text: str, image_id: str) -> list[dict[str, Any]]:
    result = []
    for line, values in _csv_rows(text):
        require(all(value.is_integer() for value in values[4:]), f"Noninteger GT flags at line {line}")
        x, y, width, height, score, category, truncation, occlusion = values
        require(score in (0, 1) and 0 <= category <= 11, f"Invalid GT score/category at line {line}")
        require(truncation in (0, 1, 2, 3) and occlusion in (0, 1, 2), f"Invalid GT attributes at line {line}")
        result.append({"target_id": f"{image_id}:{line}", "physical_line_zero_based": line,
                       "bbox_xywh": [x, y, width, height], "score": int(score), "category": int(category),
                       "truncation": int(truncation), "occlusion": int(occlusion)})
    return result


def parse_predictions(text: str, image_id: str) -> list[dict[str, Any]]:
    result = []
    for line, values in _csv_rows(text):
        require(values[5].is_integer() and 0 <= values[5] <= 11, f"Invalid prediction category at line {line}")
        result.append({"detection_id": f"{image_id}:det:{line}", "bbox_xywh": values[:4],
                       "score": values[4], "category": int(values[5]),
                       "physical_line_zero_based": line})
    return result


def _validated_inputs(ground_truth: list[dict], predictions: list[dict], width: int, height: int) -> tuple[list[dict], list[dict]]:
    require(isinstance(width, int) and not isinstance(width, bool) and isinstance(height, int)
            and not isinstance(height, bool) and width > 0 and height > 0 and width * height <= MAX_IMAGE_PIXELS,
            "Image dimensions must be positive integers within the pixel limit")
    gt, dt = deepcopy(ground_truth), deepcopy(predictions)
    for records, id_key in ((gt, "target_id"), (dt, "detection_id")):
        require(len({row[id_key] for row in records}) == len(records), f"Duplicate {id_key}")
        for index, row in enumerate(records):
            require(isinstance(row[id_key], str) and bool(row[id_key]), f"Invalid {id_key}")
            box = row["bbox_xywh"]
            require(len(box) == 4 and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in box), f"Invalid box: {row[id_key]}")
            require(math.isfinite(box[0] + box[2]) and math.isfinite(box[1] + box[3]), "Box arithmetic overflow")
            require(math.isfinite(box[2] * box[3]), "Box area arithmetic overflow")
            require(isinstance(row["category"], int) and not isinstance(row["category"], bool)
                    and 0 <= row["category"] <= 11, f"Invalid category: {row[id_key]}")
            require(isinstance(row["score"], (int, float)) and not isinstance(row["score"], bool)
                    and math.isfinite(row["score"]), f"Invalid score: {row[id_key]}")
            if id_key == "target_id":
                require(row["score"] in (0, 1), f"GT score must be 0 or 1: {row[id_key]}")
                if row["score"] == 1 and row["category"] in EVALUATED_CATEGORIES:
                    require(box[2] > 0 and box[3] > 0, f"Invalid positive GT geometry; quarantine before evaluation: {row[id_key]}")
                physical = row.get("physical_line_zero_based", index)
                require(isinstance(physical, int) and physical >= 0, "Invalid physical GT line")
                row["physical_line_zero_based"] = physical
            else:
                require(box[2] > 0 and box[3] > 0, f"Prediction width/height must be positive: {row[id_key]}")
            row["input_index"] = index
    require(len({row["physical_line_zero_based"] for row in gt}) == len(gt), "Duplicate physical GT lines")
    return gt, dt


def matlab_round(value: float) -> int:
    """MATLAB ties away from zero, rather than Python's ties-to-even."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _ignore_integral(ground_truth: list[dict], width: int, height: int) -> np.ndarray | None:
    regions = [row for row in ground_truth if row["category"] == 0]
    if not regions:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    for row in regions:
        require(all(float(value).is_integer() for value in row["bbox_xywh"]),
                "Official category-0 raster assignment requires integer region coordinates")
        x, y, box_width, box_height = [max(1, int(value)) for value in row["bbox_xywh"]]
        # MATLAB writes x:x+w and y:y+h inclusively. It does not use a continuous rectangle union.
        mask[y - 1:min(height, y + box_height), x - 1:min(width, x + box_width)] = 1
    integral = np.cumsum(mask, axis=0, dtype=np.int64)
    np.cumsum(integral, axis=1, out=integral)
    return integral


def raster_ignore_fraction(box: list[float], integral: np.ndarray | None, width: int, height: int) -> float:
    if integral is None:
        return 0.0
    x, y, box_width, box_height = [max(1, matlab_round(value)) for value in box]
    x, y = min(width, x), min(height, y)
    right, bottom = min(width, x + box_width), min(height, y + box_height)
    occupied = int(integral[y - 1, x - 1]) + int(integral[bottom - 1, right - 1])
    occupied -= int(integral[y - 1, right - 1]) + int(integral[bottom - 1, x - 1])
    return occupied / (box_width * box_height)


def filter_ignored_regions(ground_truth: list[dict], predictions: list[dict], width: int, height: int) -> dict[str, Any]:
    """Translate category-0 prefilter semantics; retain raw surviving XYWH boxes.

    The 50% boundary is inclusive. Overlapping ignore regions count once through
    their pixel union. score=0 and category=11 do not create category-0 masks.
    """
    gt, dt = _validated_inputs(ground_truth, predictions, width, height)
    integral = _ignore_integral(gt, width, height)
    result = {"ground_truth": [], "predictions": [], "removed_ground_truth": [], "removed_predictions": [],
              "official_commit": OFFICIAL_COMMIT, "oracle_equivalence_verified": False}
    for records, kept_key, removed_key, id_key in (
        (gt, "ground_truth", "removed_ground_truth", "target_id"),
        (dt, "predictions", "removed_predictions", "detection_id"),
    ):
        for row in records:
            if kept_key == "ground_truth" and row["category"] == 0:
                result[removed_key].append({id_key: row[id_key], "reason": "category_0_region_definition"})
                continue
            fraction = raster_ignore_fraction(row["bbox_xywh"], integral, width, height)
            if fraction >= 0.5:
                result[removed_key].append({id_key: row[id_key], "reason": "category_0_raster_coverage_ge_0.5", "fraction": fraction})
            else:
                result[kept_key].append(row)
    return result


def _xyxy(box: list[float], width: int | None = None, height: int | None = None) -> tuple[float, float, float, float]:
    x, y, box_width, box_height = box
    result = (x, y, x + box_width, y + box_height)
    if width is None:
        return result
    return tuple(min(max(value, 0.0), maximum) for value, maximum in zip(result, (width, height, width, height)))


def overlap(detection_box: tuple[float, ...], gt_box: tuple[float, ...], *, ignored_gt: bool = False) -> float:
    left, top = max(detection_box[0], gt_box[0]), max(detection_box[1], gt_box[1])
    right, bottom = min(detection_box[2], gt_box[2]), min(detection_box[3], gt_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    detection_area = max(0.0, detection_box[2] - detection_box[0]) * max(0.0, detection_box[3] - detection_box[1])
    gt_area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
    denominator = detection_area if ignored_gt else detection_area + gt_area - intersection
    return intersection / denominator if denominator > 0 else 0.0


def match_project_targets(ground_truth: list[dict], predictions: list[dict], width: int, height: int,
                          *, confidence: float = 0.25, iou_threshold: float = 0.5) -> dict[str, Any]:
    """Project target identity matcher, with official-style ignore handling.

    First apply the category-0 raster prefilter using raw boxes. Clip surviving
    target/prediction boxes to [0,W]x[0,H]. Match eligible positives by descending
    (IoU, confidence, original GT physical line, original detection input index).
    Only afterwards can unmatched predictions be absorbed by same-class score-0
    GT using intersection / prediction area >= the matching threshold.
    """
    require(math.isfinite(confidence) and 0 <= confidence <= 1 and 0 < iou_threshold <= 1, "Invalid project thresholds")
    filtered = filter_ignored_regions(ground_truth, predictions, width, height)
    positive, ignored_gt, detections = [], [], []
    excluded_gt = list(filtered["removed_ground_truth"])
    detection_states = {row["detection_id"]: row for row in filtered["removed_predictions"]}
    for row in filtered["ground_truth"]:
        box = _xyxy(row["bbox_xywh"], width, height)
        if row["category"] not in EVALUATED_CATEGORIES:
            reason = "outside_evaluated_categories"
        elif box[2] <= box[0] or box[3] <= box[1]:
            reason = "empty_after_clipping"
        elif row["score"] == 0:
            ignored_gt.append((row, box))
            reason = "publisher_score_0"
        else:
            positive.append((row, box))
            continue
        excluded_gt.append({"target_id": row["target_id"], "reason": reason})
    for row in filtered["predictions"]:
        box = _xyxy(row["bbox_xywh"], width, height)
        if row["category"] not in EVALUATED_CATEGORIES:
            reason = "outside_evaluated_categories"
        elif row["score"] < confidence:
            reason = "below_confidence"
        elif box[2] <= box[0] or box[3] <= box[1]:
            reason = "empty_after_clipping"
        else:
            detections.append((row, box))
            continue
        detection_states[row["detection_id"]] = {"detection_id": row["detection_id"], "reason": reason}
    pairs = []
    for gt_index, (gt, gt_box) in enumerate(positive):
        for detection_index, (dt, dt_box) in enumerate(detections):
            if gt["category"] == dt["category"]:
                value = overlap(dt_box, gt_box)
                if value >= iou_threshold:
                    pairs.append((value, dt["score"], gt["physical_line_zero_based"], dt["input_index"], gt_index, detection_index))
    used_gt, used_dt, matches = set(), set(), []
    for value, score, _, _, gt_index, detection_index in sorted(pairs, reverse=True):
        if gt_index in used_gt or detection_index in used_dt:
            continue
        gt, dt = positive[gt_index][0], detections[detection_index][0]
        used_gt.add(gt_index)
        used_dt.add(detection_index)
        matches.append({"target_id": gt["target_id"], "detection_id": dt["detection_id"], "iou": value, "confidence": score})
        detection_states[dt["detection_id"]] = {"detection_id": dt["detection_id"], "reason": "true_positive", "target_id": gt["target_id"]}
    for index, (dt, dt_box) in enumerate(detections):
        if index in used_dt:
            continue
        absorbing = [gt["target_id"] for gt, gt_box in ignored_gt if gt["category"] == dt["category"]
                     and overlap(dt_box, gt_box, ignored_gt=True) >= iou_threshold]
        detection_states[dt["detection_id"]] = {"detection_id": dt["detection_id"],
            "reason": "absorbed_by_same_class_score_0" if absorbing else "false_positive", "ignore_target_ids": absorbing}
    hits = {match["target_id"] for match in matches}
    targets = [{"target_id": gt["target_id"], "hit": gt["target_id"] in hits} for gt, _ in positive]
    states = [detection_states[row["detection_id"]] for row in predictions]
    return {"metric": "PROJECT_IGNORE_AWARE_TARGET_MATCHING", "official_AP": False,
            "oracle_equivalence_verified": False, "official_commit": OFFICIAL_COMMIT,
            "confidence": confidence, "iou_threshold": iou_threshold, "targets": targets, "matches": matches,
            "excluded_ground_truth": excluded_gt, "detection_states": states,
            "counts": {"eligible_targets": len(positive), "hits": len(matches), "misses": len(positive) - len(matches),
                       "false_positive_detections": sum(row["reason"] == "false_positive" for row in states),
                       "score_0_absorbed_detections": sum(row["reason"] == "absorbed_by_same_class_score_0" for row in states),
                       "category_0_dropped_detections": len(filtered["removed_predictions"])}}


def match_official_style(ground_truth: list[dict], predictions: list[dict], width: int, height: int,
                         *, iou_threshold: float = 0.5, max_detections: int = 500) -> dict[str, Any]:
    """Uncertified translation of the source's score-first per-class matcher.

    Keep raw coordinates, prefilter ignores, take the first max_detections in
    submitted input order across all classes, then match each class by descending
    confidence. Exact ties preserve detection input order and prefer the later
    original GT row among equal-overlap positives. This does not aggregate AP.
    """
    require(0 < iou_threshold <= 1 and isinstance(max_detections, int) and max_detections >= 1, "Invalid official-style thresholds/cap")
    filtered = filter_ignored_regions(ground_truth, predictions, width, height)
    capped = filtered["predictions"][:max_detections]
    matches, detection_states = [], []
    for category in range(1, 11):
        gt = sorted((row for row in filtered["ground_truth"] if row["category"] == category), key=lambda row: row["score"] == 0)
        dt = sorted((row for row in capped if row["category"] == category), key=lambda row: -row["score"])
        used = set()
        for detection in dt:
            best_overlap, best_index, best_state = iou_threshold, None, None
            for index, target in enumerate(gt):
                ignored = target["score"] == 0
                if index in used:
                    continue
                if best_state is not None and ignored:
                    break
                value = overlap(_xyxy(detection["bbox_xywh"]), _xyxy(target["bbox_xywh"]), ignored_gt=ignored)
                if value >= best_overlap:
                    best_overlap, best_index = value, index
                    best_state = "ignored" if ignored else "true_positive"
            state = {"detection_id": detection["detection_id"], "state": best_state or "false_positive"}
            if best_index is not None:
                state.update(target_id=gt[best_index]["target_id"], overlap=best_overlap)
                if best_state == "true_positive":
                    used.add(best_index)
                    matches.append({"target_id": gt[best_index]["target_id"], "detection_id": detection["detection_id"], "iou": best_overlap})
            detection_states.append(state)
    return {"metric": "UNCERTIFIED_OFFICIAL_STYLE_SINGLE_IMAGE_MATCHING", "official_AP": False,
            "oracle_equivalence_verified": False, "official_commit": OFFICIAL_COMMIT,
            "max_detections": max_detections, "iou_threshold": iou_threshold, "matches": matches,
            "detection_states": detection_states, "region_prefilter": filtered,
            "uncertainty": "MATLAB/Octave oracle and cross-version sort-tie equivalence not verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True, help="Single-image payload with raw GT CSV and prediction CSV; no image/model access")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file, never overwritten")
    args = parser.parse_args()
    for path in (args.input_json, args.output):
        require(not any(token in str(path).casefold() for token in ("test-dev", "testdev", "holdout", "test-challenge")), "Holdout paths are prohibited by this development CLI")
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing existing output")
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    require(payload["data_role"] in {"synthetic", "debug", "training_val", "selection"}, "Unsupported development data role")
    gt = parse_ground_truth(payload["ground_truth_csv"], payload["image_id"])
    dt = parse_predictions(payload["prediction_csv"], payload["image_id"])
    result = match_project_targets(gt, dt, payload["width"], payload["height"],
                                   confidence=payload.get("confidence", 0.25), iou_threshold=payload.get("iou_threshold", 0.5))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
