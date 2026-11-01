"""Fail-closed ignore sidecar geometry primitives, NOT Dataset/Trainer integration.

Only an explicit original -> loader resize -> centered LetterBox -> flips chain
is supported. The caller must reuse the image transform's already sampled
parameters. No random parameters, image/label files, or weights are read here.
The restricted chain is a proposed NEW training contract; it does not replace
the old 30-epoch experiment. Source policies are engineering training choices,
not official evaluation semantics. Nonpositive/empty source rows are errors.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re

import numpy as np

POLICY = {
    "schema": "rampq.gate1.ignore_geometry_policy/v1",
    "scope": "synthetic_geometry_and_ragged_collation_primitives_only",
    "chain": ["explicit_actual_loader_resize", "same_sampled_centered_letterbox", "same_sampled_vertical_horizontal_flip"],
    "letterbox": {"auto": False, "scale_fill": False, "center": True},
    "forbidden": ["mosaic", "mixup", "cutmix", "copy_paste", "random_perspective", "rotation", "shear", "translation", "random_scale_crop", "spatial_albumentations", "multi_scale"],
    "publisher_sources_separate": ["score_0", "category_0", "category_11"],
    "empty_or_nonpositive_publisher_geometry": "reject_entire_sample_for_explicit_quarantine_never_silent_drop",
    "quality_excluded_positive": "reject_separate_policy_required_never_convert_to_publisher_ignore",
    "geometry": "clipped_final_canvas_pixel_xyxy",
    "inside": "x1 <= anchor_x < x2 and y1 <= anchor_y < y2",
    "positive_priority": "downstream_loss_keeps_all_channels_of_TAL_positive_anchors",
    "official_training_or_AP_rule": False,
    "training_pipeline_integrated": False,
}

RESTRICTED_OPTIONS = {
    "pipeline": "explicit_loader_letterbox_flip_v1",
    "mosaic": 0., "mixup": 0., "cutmix": 0., "copy_paste": 0.,
    "degrees": 0., "translate": 0., "scale": 0., "shear": 0., "perspective": 0.,
    "spatial_albumentations": False, "multi_scale": False,
    "letterbox_auto": False, "letterbox_scale_fill": False, "letterbox_center": True,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hw(value, name):
    _require(isinstance(value, (list, tuple)) and len(value) == 2, f"{name} must contain height,width")
    _require(all(type(v) is int and v > 0 for v in value), f"{name} requires positive integer dimensions")
    return tuple(value)


def _identity(value):
    _require(isinstance(value, Mapping), "identity must be a mapping")
    _require(set(value) == {"image_id", "image_sha256", "sidecar_sha256"}, "identity requires exact image and sidecar digest fields")
    image_id = value["image_id"]
    _require(isinstance(image_id, str) and re.fullmatch(r"train/[A-Za-z0-9_.-]+", image_id) is not None,
             "only explicit train image identities are supported")
    _require(all(isinstance(value[k], str) and re.fullmatch(r"[0-9a-f]{64}", value[k]) is not None
                 for k in ("image_sha256", "sidecar_sha256")), "identities require lowercase SHA256 digests")
    return dict(value)


def validate_restricted_options(options):
    """Validate proposed chain metadata; this never builds framework transforms.

    A probability-zero ordinary v8_transforms chain is not this explicit chain.
    Random scale/crop is disabled; only actual loader and LetterBox resizing is
    supported. Additional transforms require a separately reviewed contract.
    """
    _require(isinstance(options, Mapping) and set(options) == set(RESTRICTED_OPTIONS), "restricted options require the exact explicit-chain fields")
    for key, expected in RESTRICTED_OPTIONS.items():
        value = options[key]
        if isinstance(expected, bool):
            _require(type(value) is bool and value == expected, f"unsupported restricted option: {key}")
        elif isinstance(expected, float):
            _require(type(value) in {int, float} and math.isfinite(value) and value == expected, f"unsupported restricted option: {key}")
        else:
            _require(value == expected, f"unsupported restricted option: {key}")
    return dict(options)


def _coords(value, name):
    _require(isinstance(value, (list, tuple)) and len(value) == 4, f"{name} requires four coordinates")
    _require(all(type(v) in {int, float} and math.isfinite(v) for v in value), f"{name} requires finite numeric coordinates")
    return [float(v) for v in value]


def _boxes(value, hw):
    _require(isinstance(value, np.ndarray) and value.dtype in {np.dtype("float32"), np.dtype("float64")},
             "boxes require a float32/float64 ndarray")
    _require(value.ndim == 2 and value.shape[1] == 4, "boxes require [N,4]")
    _require(bool(np.isfinite(value).all()), "boxes must be finite")
    _require(bool((value[:, 2:] > value[:, :2]).all()), "boxes must have positive width and height")
    h, w = hw
    _require(bool((value >= 0).all()) and bool((value <= np.array([w, h, w, h])).all()),
             "boxes must already be clipped to their canvas")


def _validate(sample):
    _require(isinstance(sample, Mapping) and sample.get("schema") == "rampq.gate1.ignore_geometry_sample/v1", "invalid sample schema")
    _identity(sample["identity"])
    _hw(sample["original_hw"], "original_hw")
    _boxes(sample["boxes"], _hw(sample["canvas_hw"], "canvas_hw"))
    _require(isinstance(sample["records"], tuple) and len(sample["records"]) == len(sample["boxes"]), "one provenance record is required per box")
    _require(len({r["target_id"] for r in sample["records"]}) == len(sample["records"]), "duplicate target identities")
    _require(isinstance(sample["trace"], tuple), "trace must be a tuple")


def from_prepared_rows(identity, original_hw, rows, *, dtype=np.float64):
    """Validate prepared sidecar rows supplied in memory; never read sidecar files.

    This fails the whole sample for publisher zero-area geometry or a quality
    exclusion. Such a failure requires an explicit new quarantine decision.
    The two supplied hashes bind identity metadata only; file hash verification
    remains the future loader's responsibility.
    """
    identity = _identity(identity)
    original_hw = _hw(original_hw, "original_hw")
    _require(np.dtype(dtype) in {np.dtype("float32"), np.dtype("float64")}, "dtype must be float32/float64")
    _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)), "rows must be a sequence")
    h, w = original_hw
    boxes, records, seen = [], [], set()
    for row in rows:
        _require(isinstance(row, Mapping), "sidecar rows must be mappings")
        _require(row.get("image_id") == identity["image_id"], "sidecar image identity mismatch")
        line = row.get("physical_line_zero_based")
        _require(type(line) is int and line >= 0, "source line identity must be a nonnegative integer")
        target_id = f"{identity['image_id']}:{line}"
        _require(row.get("target_id") == target_id and target_id not in seen, "target line identity mismatch or duplicate")
        seen.add(target_id)
        score, category = row.get("score"), row.get("category_original")
        _require(type(score) is int and score in {0, 1} and type(category) is int and 0 <= category <= 11,
                 "invalid source score/category")
        publisher = score == 0 or category in {0, 11}
        _require(type(row.get("publisher_ignored")) is bool and row["publisher_ignored"] == publisher,
                 "publisher flag inconsistent with source score/category")
        _require(row.get("excluded_for_preparation_quality") is False, "quality exclusion requires a separate quarantine policy")
        _require(publisher and row.get("ignored") is True, "only publisher-ignore sidecar rows are supported")
        x, y, bw, bh = _coords(row.get("original_xywh"), "original_xywh")
        _require(bw > 0 and bh > 0, "nonpositive publisher geometry requires explicit quarantine")
        original = [x, y, x + bw, y + bh]
        _require(all(math.isfinite(v) for v in original), "original coordinate arithmetic overflow")
        _require(_coords(row.get("original_xyxy"), "original_xyxy") == original, "original coordinate representations disagree")
        clipped = [max(0., min(float(w), original[0])), max(0., min(float(h), original[1])),
                   max(0., min(float(w), original[2])), max(0., min(float(h), original[3]))]
        _require(_coords(row.get("clipped_xyxy"), "clipped_xyxy") == clipped, "prepared clipped geometry disagrees with source canvas")
        _require(clipped[2] > clipped[0] and clipped[3] > clipped[1], "empty publisher geometry requires explicit quarantine")
        reasons = tuple(name for flag, name in ((score == 0, "score_0"), (category == 0, "category_0"), (category == 11, "category_11")) if flag)
        _require(isinstance(row.get("ignore_reasons"), list) and sorted(row["ignore_reasons"]) == sorted(reasons), "ignore reasons disagree with publisher source")
        boxes.append(clipped)
        records.append({"target_id": target_id, "physical_line_zero_based": line, "score": score,
                        "category_original": category, "source_reasons": reasons,
                        "publisher_ignored": True, "excluded_for_preparation_quality": False})
    sample = {"schema": "rampq.gate1.ignore_geometry_sample/v1", "identity": identity,
              "original_hw": original_hw, "canvas_hw": original_hw,
              "boxes": np.array(boxes, dtype=dtype).reshape(-1, 4), "records": tuple(records), "trace": ()}
    _validate(sample)  # includes underflow-to-empty after a requested dtype cast
    return sample


def _new(sample, boxes, hw, step):
    _validate(sample)
    _boxes(boxes, hw)
    return {**sample, "identity": dict(sample["identity"]), "canvas_hw": hw, "boxes": boxes,
            "records": tuple(dict(r) for r in sample["records"]), "trace": (*sample["trace"], step)}


def apply_loader_resize(sample, actual_resized_hw):
    """Use both actual output dimensions, including ceil-rounded loader resize."""
    _validate(sample)
    _require(not sample["trace"], "loader resize must be the first and only resize step")
    hw = _hw(actual_resized_hw, "actual_resized_hw")
    h, w = sample["canvas_hw"]
    rh, rw = hw[0] / h, hw[1] / w
    boxes = sample["boxes"].copy()
    boxes[:, [0, 2]] *= rw
    boxes[:, [1, 3]] *= rh
    # Clip only roundoff at the known canvas, never delete a region.
    boxes = np.clip(boxes, 0, np.array([hw[1], hw[0], hw[1], hw[0]], dtype=boxes.dtype))
    return _new(sample, boxes, hw, {"kind": "loader_resize", "from_hw": (h, w), "to_hw": hw, "ratio_wh": (rw, rh)})


def apply_letterbox_params(sample, params):
    """Reuse one image LetterBox.get_params result; do not sample/recompute it."""
    _validate(sample)
    _require(len(sample["trace"]) == 1 and sample["trace"][0]["kind"] == "loader_resize", "LetterBox must follow actual loader resize")
    _require(isinstance(params, Mapping), "LetterBox params must be a mapping")
    _require(_hw(params.get("orig_shape"), "orig_shape") == sample["canvas_hw"], "LetterBox input canvas mismatch")
    hw = _hw(params.get("new_shape"), "new_shape")
    ratio = params.get("ratio")
    _require(isinstance(ratio, (list, tuple)) and len(ratio) == 2 and all(type(v) in {int, float} and math.isfinite(v) and v > 0 for v in ratio), "invalid LetterBox ratios")
    _require(ratio[0] == ratio[1], "scale-fill/stretch is outside the restricted contract")
    h, w = sample["canvas_hw"]
    r = float(ratio[0])
    _require(r in {min(hw[0] / h, hw[1] / w), min(min(hw[0] / h, hw[1] / w), 1.)}, "ratio is not standard LetterBox scaleup true/false")
    unpad = (round(w * r), round(h * r))
    _require(tuple(params.get("new_unpad", ())) == unpad, "LetterBox resize rounding mismatch")
    dw, dh = (hw[1] - unpad[0]) / 2, (hw[0] - unpad[1]) / 2
    pads = {"left": round(dw - .1), "right": round(dw + .1), "top": round(dh - .1), "bottom": round(dh + .1)}
    _require(all(type(params.get(k)) is int and params[k] == v and v >= 0 for k, v in pads.items()),
             "only centered non-auto LetterBox padding is supported")
    boxes = sample["boxes"].copy()
    boxes *= r
    boxes[:, [0, 2]] += pads["left"]
    boxes[:, [1, 3]] += pads["top"]
    boxes = np.clip(boxes, 0, np.array([hw[1], hw[0], hw[1], hw[0]], dtype=boxes.dtype))
    return _new(sample, boxes, hw, {"kind": "letterbox", "params": {"orig_shape": (h, w), "new_shape": hw, "ratio": tuple(ratio), "new_unpad": unpad, **pads}})


def apply_flip_params(sample, params):
    """Reuse image RandomFlip.get_params; coordinates remain pixel XYXY."""
    _validate(sample)
    _require(len(sample["trace"]) >= 2 and sample["trace"][1]["kind"] == "letterbox", "flips must follow LetterBox")
    _require(isinstance(params, Mapping) and type(params.get("flip")) is bool, "flip decision must be explicit bool")
    direction = params.get("direction")
    _require(direction in {"horizontal", "vertical"}, "invalid flip direction")
    _require(direction not in [s.get("direction") for s in sample["trace"][2:]], "duplicate flip direction")
    h, w = sample["canvas_hw"]
    _require(params.get("h") == h and params.get("w") == w, "flip canvas mismatch or normalized geometry")
    _require(params.get("flip_idx") is None, "keypoint remapping is outside this detector geometry API")
    boxes = sample["boxes"].copy()
    if params["flip"]:
        a, b, length = (0, 2, w) if direction == "horizontal" else (1, 3, h)
        boxes[:, [a, b]] = length - boxes[:, [b, a]]
    return _new(sample, boxes, (h, w), {"kind": "flip", "direction": direction, "flip": params["flip"]})


def collate_ignores(samples, expected_identities, final_image_hw, *, torch_dtype=None, device="cpu"):
    """Preserve input order with explicit identities and one tensor per image.

    This only collates the ignore sidecar. It does not collate images, positives,
    run Format, transfer a training batch, or connect the actual criterion.
    """
    import torch
    _require(isinstance(samples, Sequence) and len(samples) > 0, "provide a nonempty sample sequence")
    _require(isinstance(expected_identities, Sequence) and len(expected_identities) == len(samples), "one expected identity per image is required")
    hw = _hw(final_image_hw, "final_image_hw")
    dtype = torch.float32 if torch_dtype is None else torch_dtype
    _require(dtype in {torch.float32, torch.float64}, "loss geometry supports only float32/float64")
    boxes, identities, provenance = [], [], []
    for sample, expected in zip(samples, expected_identities):
        _validate(sample)
        _require(sample["identity"] == _identity(expected), "batch image/sidecar identity or order mismatch")
        _require(sample["canvas_hw"] == hw, "batch final canvas mismatch; multi-scale is not integrated")
        _require(len(sample["trace"]) >= 2 and sample["trace"][1]["kind"] == "letterbox", "untransformed source geometry cannot be collated")
        tensor = torch.tensor(sample["boxes"].copy(), dtype=dtype, device=device)
        _require(bool(torch.isfinite(tensor).all()) and bool((tensor[:, 2:] > tensor[:, :2]).all()), "cast produced invalid loss geometry")
        boxes.append(tensor)
        identities.append(dict(expected))
        provenance.append(tuple(dict(r) for r in sample["records"]))
    return {"ignore_boxes_xyxy": boxes, "image_identities": tuple(identities), "source_records": tuple(provenance),
            "image_hw": hw, "batch_indices": tuple(range(len(samples))), "training_pipeline_integrated": False}
