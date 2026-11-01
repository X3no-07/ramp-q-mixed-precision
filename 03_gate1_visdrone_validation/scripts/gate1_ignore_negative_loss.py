"""Synthetic-tested classification-loss primitive; NOT a training integration.

The caller must supply publisher-ignore regions after every image augmentation,
clipped to the final image canvas in pixel XYXY coordinates. An anchor is inside
a region iff x1 <= x < x2 and y1 <= y < y2. This class-agnostic, anchor-center
policy is a new engineering choice, not the official VisDrone evaluation rule.
All classification channels of an assigned positive anchor remain unchanged.
Assignment, box/DFL losses, data loading and augmentation are outside this API.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

POLICY = {
    "schema": "rampq.gate1.ignore_negative_loss_policy/v1",
    "scope": "classification_BCE_primitive_only_after_augmentation",
    "geometry": "clipped_final_canvas_pixel_xyxy",
    "inside": "x1 <= anchor_x < x2 and y1 <= anchor_y < y2",
    "ignore_class_policy": "class_agnostic_for_unassigned_anchors_only",
    "positive_priority": "preserve_all_class_channels_at_every_fg_mask_anchor",
    "normalization": "original_target_scores_sum_clamp_min_1_no_area_rescaling",
    "supported_float_dtypes": ["torch.float32", "torch.float64"],
    "official_visdrone_training_or_AP_rule": False,
    "training_pipeline_integrated": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_float(tensor: torch.Tensor, name: str) -> None:
    _require(isinstance(tensor, torch.Tensor), f"{name} must be a tensor")
    _require(tensor.is_floating_point(), f"{name} must have floating dtype")
    _require(tensor.dtype in {torch.float32, torch.float64}, f"{name} only supports float32 or float64")
    _require(bool(torch.isfinite(tensor).all()), f"{name} must be finite")


def build_negative_keep_mask(
    anchor_centers_xy: torch.Tensor,
    fg_mask: torch.Tensor,
    ignore_boxes_xyxy: Sequence[torch.Tensor],
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Return [B,A] keep flags; never infer or transform ignore geometry.

    Ragged ignore lists must contain an explicit [0,4] tensor for an image with
    no regions. Empty, reversed, out-of-canvas and nonfinite boxes are rejected,
    so an upstream geometry error cannot silently become an unmasked negative.
    """
    _require(isinstance(image_hw, tuple) and len(image_hw) == 2, "image_hw must be (height, width)")
    _require(all(type(v) is int and v > 0 for v in image_hw), "image_hw dimensions must be positive integers")
    height, width = image_hw
    _finite_float(anchor_centers_xy, "anchor_centers_xy")
    _require(anchor_centers_xy.ndim == 2 and anchor_centers_xy.shape[1] == 2, "anchor centers require [A,2]")
    _require(anchor_centers_xy.shape[0] > 0, "at least one anchor is required")
    _require(not anchor_centers_xy.requires_grad, "anchor geometry must be detached")
    _require(isinstance(fg_mask, torch.Tensor) and fg_mask.dtype == torch.bool, "fg_mask must be boolean")
    _require(fg_mask.ndim == 2 and fg_mask.shape[1] == anchor_centers_xy.shape[0] and fg_mask.shape[0] > 0,
             "fg_mask requires nonempty [B,A] matching centers")
    _require(fg_mask.device == anchor_centers_xy.device, "geometry and fg_mask devices must match")
    _require(isinstance(ignore_boxes_xyxy, Sequence) and len(ignore_boxes_xyxy) == fg_mask.shape[0],
             "provide one ragged ignore tensor per image")
    _require(bool(((anchor_centers_xy >= 0) & (anchor_centers_xy < anchor_centers_xy.new_tensor([width, height]))).all()),
             "anchor centers must be within the final image canvas")

    inside = torch.zeros_like(fg_mask)
    for batch_index, boxes in enumerate(ignore_boxes_xyxy):
        _finite_float(boxes, "ignore_boxes_xyxy")
        _require(boxes.ndim == 2 and boxes.shape[1] == 4, "each ignore tensor requires [N,4]")
        _require(boxes.device == anchor_centers_xy.device, "ignore and anchor devices must match")
        _require(not boxes.requires_grad, "ignore geometry must be detached")
        _require(bool((boxes[:, 2:] > boxes[:, :2]).all()), "ignore boxes must have strictly positive width and height")
        _require(bool((boxes >= 0).all()) and bool((boxes <= boxes.new_tensor([width, height, width, height])).all()),
                 "ignore boxes must already be clipped to the final image canvas")
        # Union each region directly; no count-dependent loss rescaling.
        for box in boxes:
            inside[batch_index] |= ((anchor_centers_xy[:, 0] >= box[0])
                                    & (anchor_centers_xy[:, 0] < box[2])
                                    & (anchor_centers_xy[:, 1] >= box[1])
                                    & (anchor_centers_xy[:, 1] < box[3]))
    return fg_mask | ~inside


def masked_classification_bce(
    logits: torch.Tensor,
    target_scores: torch.Tensor,
    fg_mask: torch.Tensor,
    anchor_centers_xy: torch.Tensor,
    ignore_boxes_xyxy: Sequence[torch.Tensor],
    image_hw: tuple[int, int],
    *,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply a new engineering mask to otherwise ordinary YOLO class BCE.

    Inputs are not modified. The loss is before the framework's cls gain and
    batch-size multiplier. Keep the original target-score denominator even if
    every background anchor is ignored. No assignment or regression is run.
    """
    _finite_float(logits, "logits")
    _finite_float(target_scores, "target_scores")
    _require(logits.ndim == 3 and all(size > 0 for size in logits.shape), "logits require nonempty [B,A,C]")
    _require(logits.shape == target_scores.shape, "target_scores shape must match logits")
    _require(logits.dtype == target_scores.dtype and logits.device == target_scores.device,
             "logits and target_scores dtype/device must match")
    _require(not target_scores.requires_grad, "assigned target_scores must be detached")
    _require(bool(((target_scores >= 0) & (target_scores <= 1)).all()), "target_scores must lie in [0,1]")
    keep = build_negative_keep_mask(anchor_centers_xy, fg_mask, ignore_boxes_xyxy, image_hw)
    _require(keep.shape == logits.shape[:2] and keep.device == logits.device, "mask shape/device must match logits")
    _require(bool((target_scores[~fg_mask] == 0).all()), "unassigned anchors must have zero target scores")
    elementwise = F.binary_cross_entropy_with_logits(logits, target_scores, reduction="none")
    if class_weights is not None:
        _finite_float(class_weights, "class_weights")
        _require(class_weights.shape in {(logits.shape[-1],), (1, 1, logits.shape[-1])}, "class_weights require [C] or [1,1,C]")
        _require(class_weights.dtype == logits.dtype and class_weights.device == logits.device,
                 "class_weights dtype/device must match logits")
        _require(not class_weights.requires_grad and bool((class_weights >= 0).all()), "class_weights must be detached and nonnegative")
        elementwise = elementwise * class_weights
    _require(bool(torch.isfinite(elementwise).all()), "elementwise BCE must remain finite after class weighting")
    denominator = target_scores.sum().clamp_min(1)
    _require(bool(torch.isfinite(denominator)), "original target-score denominator must remain finite")
    masked = torch.where(keep.unsqueeze(-1), elementwise, torch.zeros_like(elementwise))
    numerator = masked.sum()
    _require(bool(torch.isfinite(numerator)), "masked BCE reduction must remain finite")
    loss = numerator / denominator
    _require(bool(torch.isfinite(loss)), "classification loss must remain finite")
    return {"loss": loss, "denominator": denominator,
            "keep_mask": keep, "ignored_negative_mask": ~keep}
