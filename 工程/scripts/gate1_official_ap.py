"""Source-faithful VisDrone AP/AR aggregation for finite in-memory matrices.

Translation target: official commit 005445782213e20cb91bc50a597db3dd949e749a,
utils/{calcAccuracy,VOCap,evalRes,compOas}.m.  The optional raw adapter reuses
only the separately pinned category-0 raster prefilter, never its matcher.

Prepared GT rows are [x,y,w,h,ignore,category,truncation,occlusion], where the
publisher score has ALREADY been inverted (0 -> ignore=1, 1 -> ignore=0).
Detection rows are [x,y,w,h,score,category,unused,unused].  All inputs must be
finite real numeric matrices; categories are integers 0..11 and GT ignore is
0/1. Geometry is raw, unclipped XYWH, with finite endpoints and areas. Empty
inputs are [] or correctly shaped zero-row matrices. Scores are finite
but need not be within [0,1]. No input is mutated, no NMS is performed, and
there is no image/model/dataset file access in this module. The raw adapter
hash-reads and imports its pinned Python prefilter source.

Stable ties preserve submitted row order within each image, then image order
across images. This policy is explicit rather than a cross-version claim
about every MATLAB/Octave sort implementation. Native agreement must be
established externally after freezing this source; the module certifies none.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


OFFICIAL_COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"
IOU_THRESHOLDS = tuple(0.5 + 0.05 * index for index in range(10))
# A pure arithmetic probe of the chosen Octave 11.3.0 runtime observed exactly
# these ten float64 values, including 0.8500000000000001. No AP answer was used.
THRESHOLD_RUNTIME_SCOPE = "Octave 11.3.0 colon 0.5:0.05:0.95; not all MATLAB versions"
MAX_DETECTIONS = (1, 10, 100, 500)
METRIC_NAMES = ("AP_all", "AP_50", "AP_75", "AR_1", "AR_10", "AR_100", "AR_500")
TIE_POLICY = "stable: submitted detection row order within image, then input image order"
PREFILTER_SHA256 = "4e45ff26a86a83ff66e997f15e4c4794246c6440c304446972809ff32db09961"


class OfficialAPInputError(ValueError):
    """Input is outside the declared finite-matrix translation domain."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialAPInputError(message)


def _matrix(value: Any, columns: int, name: str) -> np.ndarray:
    try:
        original = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise OfficialAPInputError(f"{name}: invalid matrix") from error
    _require(original.dtype.kind in "iuf", f"{name}: require real numeric values, not bool/object/string/complex")
    if original.ndim == 1 and original.size == 0:
        original = original.reshape(0, columns)
    _require(original.ndim == 2 and original.shape[1] == columns, f"{name}: require Nx{columns} matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(original, dtype=np.float64, copy=True)
    _require(bool(np.isfinite(result).all()), f"{name}: nonfinite value")
    return result


def _boxes(value: Any, columns: int, name: str) -> np.ndarray:
    result = _matrix(value, columns, name)
    with np.errstate(over="ignore", invalid="ignore"):
        endpoints = result[:, :2] + result[:, 2:4]
        areas = result[:, 2] * result[:, 3]
    _require(bool(np.isfinite(endpoints).all() and np.isfinite(areas).all()), f"{name}: box arithmetic overflow")
    return result


def _rows8(value: Any, name: str, *, ground_truth: bool) -> np.ndarray:
    result = _boxes(value, 8, name)
    categories = result[:, 5]
    _require(bool(((categories == np.floor(categories)) & (categories >= 0) & (categories <= 11)).all()),
             f"{name}: category must be integer 0..11")
    if ground_truth:
        _require(bool(np.isin(result[:, 4], (0, 1)).all()), f"{name}: GT ignore/score must be 0 or 1")
    return result


def comp_oas(dt: Any, gt: Any, ignore: Any = None) -> np.ndarray:
    """Mirror compOas raw continuous overlap (ignore uses detection area).

    Zero/negative widths and heights are admitted, as in the source: no
    positive intersection then yields zero overlap. Arithmetic overflow is
    rejected rather than silently converted to NaN/Inf.
    """
    detections = _boxes(dt, 4, "dt")
    targets = _boxes(gt, 4, "gt")
    if ignore is None:
        ignored = np.zeros(len(targets), dtype=bool)
    else:
        values = np.asarray(ignore)
        _require(values.ndim == 1 and values.shape == (len(targets),), "ignore: require length-N vector")
        _require(values.dtype.kind in "biuf" and bool(np.isin(values, (0, 1)).all()), "ignore: require 0/1 values")
        ignored = values.astype(bool)
    result = np.zeros((len(detections), len(targets)), dtype=np.float64)
    for d, detection in enumerate(detections):
        dx, dy, dw, dh = map(float, detection)
        for g, target in enumerate(targets):
            gx, gy, gw, gh = map(float, target)
            width = min(dx + dw, gx + gw) - max(dx, gx)
            if width <= 0:
                continue
            height = min(dy + dh, gy + gh) - max(dy, gy)
            if height <= 0:
                continue
            intersection = width * height
            denominator = dw * dh if ignored[g] else dw * dh + gw * gh - intersection
            _require(math.isfinite(intersection) and math.isfinite(denominator) and denominator > 0,
                     "overlap arithmetic outside finite positive-denominator domain")
            result[d, g] = intersection / denominator
    return result


def eval_res(gt0: Any, dt0: Any, threshold: float = 0.7, *, allow_multiple: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Independent evalRes mirror, including every -1/0/1 output flag.

    Equal-overlap nonignored candidates prefer the later stable GT row.
    After any match, the first encountered ignore row ends the scan. Ignored
    GT can absorb multiple detections. The source's optional mul behavior is
    retained literally (a further match to GT already flagged 1 gets dt=-1).
    """
    _require(isinstance(threshold, (int, float, np.integer, np.floating)) and not isinstance(threshold, (bool, np.bool_))
             and math.isfinite(float(threshold)) and 0 <= threshold <= 1, "threshold: require finite value in [0,1]")
    _require(isinstance(allow_multiple, bool), "allow_multiple: require bool")
    targets = _boxes(gt0, 5, "gt0")
    detections = _boxes(dt0, 5, "dt0")
    _require(bool(np.isin(targets[:, 4], (0, 1)).all()), "gt0: ignore must be 0 or 1")
    detections = detections[np.argsort(-detections[:, 4], kind="stable")]
    targets = targets[np.argsort(targets[:, 4], kind="stable")]
    targets[:, 4] *= -1
    detections = np.column_stack((detections, np.zeros(len(detections), dtype=np.float64)))
    overlaps = comp_oas(detections[:, :4], targets[:, :4], targets[:, 4] == -1)
    for d in range(len(detections)):
        best_overlap, best_gt, best_match = float(threshold), -1, 0
        for g in range(len(targets)):
            match = targets[g, 4]
            if match == 1 and not allow_multiple:
                continue
            if best_match != 0 and match == -1:
                break
            if overlaps[d, g] < best_overlap:
                continue
            best_overlap, best_gt = float(overlaps[d, g]), g
            best_match = 1 if match == 0 else -1
        if best_match == -1:
            detections[d, 5] = -1
        elif best_match == 1:
            targets[best_gt, 4] = 1
            detections[d, 5] = 1
    return targets, detections


def voc_ap(recall: Any, precision: Any) -> float:
    """VOCap precision-envelope integration, not 11- or 101-point sampling."""
    rec = np.asarray(recall)
    prec = np.asarray(precision)
    _require(rec.ndim == prec.ndim == 1 and rec.shape == prec.shape, "recall/precision: require equal length vectors")
    _require(rec.dtype.kind in "iuf" and prec.dtype.kind in "iuf", "recall/precision: require real numeric values")
    rec, prec = rec.astype(np.float64, copy=True), prec.astype(np.float64, copy=True)
    _require(bool(np.isfinite(rec).all() and np.isfinite(prec).all()), "recall/precision: nonfinite value")
    _require(bool(((rec >= 0) & (rec <= 1)).all() and ((prec >= 0) & (prec <= 1)).all()
                  and (np.diff(rec) >= 0).all()), "recall/precision: outside monotone [0,1] domain")
    padded_rec = np.concatenate(([0.0], rec, [1.0]))
    padded_prec = np.concatenate(([0.0], prec, [0.0]))
    for index in range(len(padded_prec) - 2, -1, -1):
        padded_prec[index] = max(padded_prec[index], padded_prec[index + 1])
    changes = np.flatnonzero(padded_rec[1:] != padded_rec[:-1]) + 1
    return float(np.sum((padded_rec[changes] - padded_rec[changes - 1]) * padded_prec[changes]))


def calc_accuracy(allgt: Sequence[Any], alldet: Sequence[Any], *, capture_intermediates: bool = True) -> dict[str, Any]:
    """Mirror calcAccuracy over prepared eight-column per-image matrices.

    Quirks intentionally preserved: cap input rows ACROSS ALL CLASSES before
    filtering by class; count all GT rows (including ignore) in the recall
    denominator; exclude ignored dt from both TP and FP; repeat evalClass once
    per image containing that class; compute AP only from the cap=500 loop.
    NaN is retained for all seven aggregate metrics if evalClass is empty.
    """
    _require(isinstance(allgt, (list, tuple)) and isinstance(alldet, (list, tuple)), "allgt/alldet: require image sequences")
    _require(len(allgt) == len(alldet), "allgt/alldet: unequal image counts")
    _require(isinstance(capture_intermediates, bool), "capture_intermediates: require bool")
    gt_images = [_rows8(rows, f"allgt[{i}]", ground_truth=True) for i, rows in enumerate(allgt)]
    dt_images = [_rows8(rows, f"alldet[{i}]", ground_truth=False) for i, rows in enumerate(alldet)]
    ap = np.zeros((10, 10), dtype=np.float64)
    ar = np.zeros((10, 10, 4), dtype=np.float64)
    eval_class, intermediates = [], []
    for category in range(1, 11):
        eval_class.extend(category for gt in gt_images if np.any(gt[:, 5] == category))
        for t, threshold in enumerate(IOU_THRESHOLDS):
            for c, cap in enumerate(MAX_DETECTIONS):
                gt_flags, det_matches, image_traces = [], [], []
                for image_index, (gt, dt) in enumerate(zip(gt_images, dt_images)):
                    selected_gt_indices = np.flatnonzero(gt[:, 5] == category)
                    # The source indexes det with a logical vector of cap length.
                    # It selects only the first cap rows, then that class, with
                    # no sorting before the cap and no per-class cap instead.
                    selected_dt_indices = np.flatnonzero(dt[:cap, 5] == category)
                    gt1, dt1 = eval_res(gt[selected_gt_indices, :5], dt[selected_dt_indices, :5], threshold)
                    gt_flags.append(gt1[:, 4])
                    det_matches.append(dt1[:, 4:6])
                    if capture_intermediates:
                        image_traces.append({"image_index_zero_based": image_index,
                            "selected_gt_input_indices_zero_based": selected_gt_indices.copy(),
                            "selected_dt_input_indices_zero_based": selected_dt_indices.copy(),
                            "gt_output": gt1.copy(), "dt_output": dt1.copy()})
                gt_match = np.concatenate(gt_flags) if gt_flags else np.zeros(0, dtype=np.float64)
                det_match = np.concatenate(det_matches, axis=0) if det_matches else np.zeros((0, 2), dtype=np.float64)
                rank = np.argsort(-det_match[:, 0], kind="stable")
                tp = np.cumsum(det_match[rank, 1] == 1, dtype=np.int64)
                fp = np.cumsum(det_match[rank, 1] == 0, dtype=np.int64)
                recall = tp / max(1, len(gt_match))
                precision = tp / np.maximum(1, fp + tp)
                if recall.size:
                    ar[category - 1, t, c] = float(np.max(recall)) * 100
                if cap == 500:
                    ap[category - 1, t] = voc_ap(recall, precision) * 100
                if capture_intermediates:
                    intermediates.append({"category": category, "iou_threshold": threshold, "max_detections": cap,
                        "by_image": image_traces, "gt_match_flags": gt_match.copy(),
                        "det_match_score_and_flag": det_match.copy(), "rank_zero_based": rank.copy(),
                        "tp_cumulative": tp.copy(), "fp_cumulative": fp.copy(),
                        "recall": recall.copy(), "precision": precision.copy(),
                        "recall_denominator_including_ignored_gt": max(1, len(gt_match))})
    eval_class_array = np.asarray(eval_class, dtype=np.int64)
    if eval_class:
        selected = eval_class_array - 1
        values = (np.mean(ap[selected, :]), np.mean(ap[selected, 0]), np.mean(ap[selected, 5]),
                  *(np.mean(ar[selected, :, cap_index]) for cap_index in range(4)))
    else:
        values = (float("nan"),) * 7
    return {"schema": "rampq.gate1.python_official_ap_mirror/v1", "official_commit": OFFICIAL_COMMIT,
            "AP": ap, "AR": ar, "evalClass": eval_class_array,
            "metrics": {name: float(value) for name, value in zip(METRIC_NAMES, values)},
            "iou_thresholds": np.asarray(IOU_THRESHOLDS, dtype=np.float64),
            "max_detections": np.asarray(MAX_DETECTIONS, dtype=np.int64),
            "intermediates": intermediates, "tie_policy": TIE_POLICY,
            "threshold_runtime_scope": THRESHOLD_RUNTIME_SCOPE,
            "input_image_count": len(gt_images), "capture_intermediates": capture_intermediates,
            "scope": {"official_AP_equivalence_certified": False, "NMS_executed": False,
                      "model_executed": False, "real_data_files_read_by_module": False,
                      "prepared_input_contract": "publisher GT score inverted BEFORE calc_accuracy",
                      "empty_evalClass_metrics": "NaN", "units": "percent"}}


def _load_prefilter() -> Any:
    import hashlib
    source = Path(__file__).with_name("gate1_ignore_aware_evaluation.py")
    _require(hashlib.sha256(source.read_bytes()).hexdigest() == PREFILTER_SHA256, "Pinned raster prefilter source changed")
    spec = importlib.util.spec_from_file_location("gate1_official_ap_pinned_prefilter", source)
    _require(spec is not None and spec.loader is not None, "Cannot load pinned raster prefilter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_raw_image(raw_gt8: Any, raw_dt8: Any, *, width: int, height: int) -> dict[str, np.ndarray]:
    """Synthetic/in-memory raw adapter: category-0 filter, then score inversion.

    This path inherits the pinned prefilter's explicit pixel limit and geometry
    domain (positive prediction w/h and positive score-1 evaluated GT w/h;
    integer category-0 region coordinates). Unused attributes remain unchanged.
    It imports no oracle result, opens no image/data files, and never calls the
    old score-first matcher. Only the pinned Python source file is hash-read.
    """
    raw_gt = _rows8(raw_gt8, "raw_gt8", ground_truth=True)
    raw_dt = _rows8(raw_dt8, "raw_dt8", ground_truth=False)
    prefilter = _load_prefilter()
    gt_records = [{"target_id": f"synthetic:gt:{i}", "physical_line_zero_based": i,
                   "bbox_xywh": row[:4].tolist(), "score": int(row[4]), "category": int(row[5])}
                  for i, row in enumerate(raw_gt)]
    dt_records = [{"detection_id": f"synthetic:dt:{i}", "bbox_xywh": row[:4].tolist(),
                   "score": float(row[4]), "category": int(row[5])}
                  for i, row in enumerate(raw_dt)]
    filtered = prefilter.filter_ignored_regions(gt_records, dt_records, width, height)
    kept_gt = raw_gt[[row["input_index"] for row in filtered["ground_truth"]], :].copy()
    kept_dt = raw_dt[[row["input_index"] for row in filtered["predictions"]], :].copy()
    prepared = kept_gt.copy()
    prepared[:, 4] = 1 - prepared[:, 4]
    return {"filtered_gt": kept_gt, "filtered_predictions": kept_dt, "prepared_gt": prepared}


def evaluate_raw_images(images: Sequence[dict[str, Any]], *, capture_intermediates: bool = True) -> dict[str, Any]:
    """Convenience adapter for pure synthetic fixture input dictionaries."""
    _require(isinstance(images, (list, tuple)), "images: require list or tuple")
    prepared = [prepare_raw_image(image["raw_ground_truth"], image["raw_predictions"],
                                 width=image["width"], height=image["height"]) for image in images]
    result = calc_accuracy([row["prepared_gt"] for row in prepared],
                           [row["filtered_predictions"] for row in prepared],
                           capture_intermediates=capture_intermediates)
    result["raw_adapter_intermediates"] = prepared
    return result
