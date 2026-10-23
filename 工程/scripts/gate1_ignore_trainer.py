"""Pinned Ultralytics integration for a NEW, synthetic-only training contract.

The adapted framework loss block is AGPL-3.0, as is its Ultralytics source.

This does not launch real-data training or replace any earlier experiment.
The TAL/box/DFL path mirrors the pinned framework implementation; only the
classification BCE excludes unassigned anchors whose centers are in publisher
ignore boxes. Validation is explicitly a lifecycle placeholder, with no AP.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import ultralytics
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

try:
    from .gate1_ignore_geometry import RESTRICTED_OPTIONS, validate_restricted_options
    from .gate1_ignore_negative_loss import masked_classification_bce
except ImportError:
    from gate1_ignore_geometry import RESTRICTED_OPTIONS, validate_restricted_options
    from gate1_ignore_negative_loss import masked_classification_bce


FRAMEWORK_PINS = {
    "ultralytics.utils.loss": "357053aa9a24ad2c4d541c3ada3a27232a10d679f89c7b4957b9a84a7da4ec1d",
    "ultralytics.nn.tasks": "2371b1bc9b1a17782c55259cf760fcd603f0bd012feb55737e7809dafe89c807",
    "ultralytics.models.yolo.detect.train": "76d6b181259edc09b03f7cea1f9d59bd4eda2f7964146bbb2e76d4f7201dcf5f",
    "ultralytics.engine.trainer": "4b929f223830328ffed7331508917c6a65eab3408b5628e7c561274cd750e43e",
}

INTEGRATION_POLICY = {
    "schema": "rampq.gate1.ignore_trainer_integration_policy/v1",
    "scope": "synthetic_only_actual_DetectionTrainer_lifecycle",
    "classification": "ignore_center_half_open_only_at_unassigned_anchors_all_channels",
    "positive_priority": "preserve_entire_classification_row_at_every_TAL_positive",
    "unchanged": ["TAL", "box", "DFL", "target_score_denominator", "gains", "batch_multiplier"],
    "validation": "SMOKE_NO_TASK_AP",
    "official_training_or_AP_rule": False,
    "real_data_training_authorized_by_this_module": False,
    "old_guard_replaced": False,
    "scientific_gate_pass": False,
}

SAFE_SYNTHETIC_OVERRIDES = {
    "model": "synthetic-bound-config.yaml", "data": "synthetic-bound-manifest.json",
    "task": "detect", "mode": "train", "device": "cpu", "workers": 0,
    "epochs": 1, "batch": 2, "nbs": 2, "imgsz": 64,
    "amp": False, "pretrained": False, "resume": False, "compile": False,
    "channels_last": False, "optimizer": "SGD", "warmup_epochs": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0,
    "perspective": 0.0, "multi_scale": 0.0, "close_mosaic": 0,
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "bgr": 0.0,
    "fliplr": 0.0, "flipud": 0.0, "rect": False, "cache": False,
    "val": False, "plots": False, "profile": False, "save": True,
    "save_period": -1, "fraction": 1.0, "single_cls": False,
    "classes": None, "freeze": None, "distill_model": None,
    "augmentations": None, "time": None, "cls_pw": 0.0,
    "deterministic": True, "seed": 0, "verbose": False,
}
_ADJUSTABLE = {"project", "name", "exist_ok", "imgsz", "batch", "nbs", "seed",
               "lr0", "lrf", "momentum", "weight_decay", "fliplr", "flipud",
               "box", "cls", "dfl", "patience", "cls_pw", "verbose"}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_framework_sources():
    """Fail closed on a different installed framework; never changes its files."""
    _require(ultralytics.__version__ == "8.4.142", "requires reviewed Ultralytics 8.4.142")
    observed = {}
    for name, expected in FRAMEWORK_PINS.items():
        source = Path(importlib.import_module(name).__file__).resolve()
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        _require(actual == expected, f"framework source changed: {name}")
        observed[name] = {"path": str(source), "sha256": actual}
    return observed


def validate_class_weights(weights, class_count, device):
    if weights is None:
        return
    _require(isinstance(weights, torch.Tensor) and weights.shape in {(class_count,), (1, 1, class_count)},
             "class weights require [C] or [1,1,C]")
    _require(weights.dtype == torch.float32 and weights.device == device,
             "class weights must already be FP32 on the model device")
    _require(not weights.requires_grad and bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()),
             "class weights must be finite, detached and nonnegative")


def validate_synthetic_overrides(overrides=None):
    """Reject configuration drift rather than relying on probability-zero transforms."""
    supplied = dict(overrides or {})
    _require(set(supplied) <= set(SAFE_SYNTHETIC_OVERRIDES) | _ADJUSTABLE,
             "unsupported synthetic training override")
    merged = {**SAFE_SYNTHETIC_OVERRIDES, **supplied}
    for key, expected in SAFE_SYNTHETIC_OVERRIDES.items():
        if key not in _ADJUSTABLE:
            _require(type(merged[key]) is type(expected) and merged[key] == expected,
                     f"restricted synthetic option changed: {key}")
    for key in ("batch", "nbs", "imgsz"):
        _require(type(merged[key]) is int and merged[key] > 0, f"{key} must be an explicit positive integer")
    _require(merged["imgsz"] >= 64 and merged["imgsz"] % 32 == 0, "imgsz must be >=64 and a multiple of32")
    _require(merged["batch"] == merged["nbs"], "synthetic smoke requires nbs=batch for an optimizer step each batch")
    _require(type(merged["seed"]) is int and merged["seed"] >= 0, "seed must be nonnegative integer")
    for key in ("fliplr", "flipud", "cls_pw"):
        value = merged[key]
        _require(type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1, f"invalid {key}")
    for key in ("lr0", "lrf", "momentum", "weight_decay", "box", "cls", "dfl"):
        if key in merged:
            _require(type(merged[key]) in {int, float} and math.isfinite(merged[key]) and merged[key] >= 0,
                     f"invalid nonnegative finite {key}")
    return merged


def _validate_contract(contract):
    _require(isinstance(contract, Mapping), "training contract must be a mapping")
    _require(contract.get("schema") == "rampq.gate1.synthetic_ignore_trainer_contract/v1"
             and contract.get("synthetic_only") is True, "only an explicitly synthetic contract is supported")
    nc = contract.get("class_count")
    names = contract.get("names")
    _require(type(nc) is int and nc > 0 and isinstance(names, list) and len(names) == nc
             and all(isinstance(n, str) and n for n in names), "class count and names disagree")
    validate_restricted_options(contract.get("restricted_options"))
    datasets = contract.get("datasets")
    _require(isinstance(datasets, Mapping) and set(datasets) == {"train", "training_val"},
             "contract requires explicit train and training_val manifests")
    for role, entry in datasets.items():
        _require(isinstance(entry, Mapping) and set(entry) == {"manifest_path", "manifest_sha256"},
                 "dataset binding requires an exact path and SHA256")
        path = Path(entry["manifest_path"])
        _require(path.is_absolute() and path.is_file(), f"missing bound {role} manifest")
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["manifest_sha256"],
                 f"bound {role} manifest digest mismatch")
        header = json.loads(path.read_text(encoding="utf-8-sig"))
        _require(header.get("schema") == "rampq.gate1.ignore_dataset_manifest/v1"
                 and header.get("data_role") == "synthetic_only",
                 f"bound {role} manifest must declare the reviewed synthetic-only Dataset schema")
    return deepcopy(dict(contract))


class IgnoreAwareDetectionLoss(v8DetectionLoss):
    """One pinned TAL invocation; the only altered objective is background BCE."""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        self.source_pins = verify_framework_sources()
        head = model.model[-1]
        _require(getattr(head, "one2one_cv2", None) is None, "end-to-end/one2one heads need separate review")
        _require(all(v.dtype == torch.float32 for v in model.state_dict().values() if v.is_floating_point()),
                 "ignore integration requires FP32 model tensors")
        validate_class_weights(getattr(model, "class_weights", None), head.nc, next(model.parameters()).device)
        super().__init__(model, tal_topk, tal_topk2)
        self.last_mask_receipt = None
        self.observed_batches = []

    def get_assigned_targets_and_loss(self, preds, batch):
        # These are the pinned framework's ordering, normalization and reductions.
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (preds["boxes"].permute(0, 2, 1).contiguous(),
                                   preds["scores"].permute(0, 2, 1).contiguous())
        _require(pred_scores.dtype == torch.float32 and pred_distri.dtype == torch.float32,
                 "FP16/AMP loss execution is outside this synthetic integration")
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        _require("ignore_boxes_xyxy" in batch and "image_identities" in batch,
                 "explicit ragged ignore geometry and image identities are required")
        _require(len(batch["image_identities"]) == batch_size, "one identity is required per image")
        _require(all(isinstance(b, torch.Tensor) and b.dtype == torch.float32
                     for b in batch["ignore_boxes_xyxy"]), "integration ignore boxes must be explicit FP32")
        image_hw = tuple(int(v) for v in batch["img"].shape[-2:])
        _require(image_hw == tuple(int(v) for v in imgsz), "loss anchors and final image canvas disagree")
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        validate_class_weights(self.class_weights, self.nc, self.device)
        # The pinned TAL's empty-GT branch returns an all-zero float fg_mask.
        # Normalize only the mask consumed by our strict helper; return the
        # untouched framework assignment tuple and preserve its box/DFL path.
        helper_fg_mask = fg_mask.bool()
        masked = masked_classification_bce(
            pred_scores, target_scores.to(dtype), helper_fg_mask, (anchor_points * stride_tensor).detach(),
            batch["ignore_boxes_xyxy"], image_hw, class_weights=self.class_weights,
        )
        loss[1] = masked["loss"]
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor,
            )
        else:
            loss[0] += pred_distri[..., :0].sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        with torch.no_grad():
            ordinary_bce = self.bce(pred_scores, target_scores.to(dtype))
            if self.class_weights is not None:
                ordinary_bce *= self.class_weights
            ordinary_cls = ordinary_bce.sum() / target_scores_sum
        self.last_mask_receipt = {
            "schema": "rampq.gate1.ignore_training_batch_receipt/v1",
            "batch_size": batch_size,
            "num_anchors_per_image": int(fg_mask.shape[1]),
            "image_hw": image_hw,
            "assigned_positive_anchors": int(fg_mask.sum().item()),
            "ignored_negative_anchors": int(masked["ignored_negative_mask"].sum().item()),
            "kept_anchors": int(masked["keep_mask"].sum().item()),
            "positive_rows_preserved": bool(masked["keep_mask"][helper_fg_mask].all()),
            "target_score_denominator": float(masked["denominator"].item()),
            "classification_before_gain": float(masked["loss"].detach().item()),
            "ordinary_classification_before_gain": float(ordinary_cls.item()),
            "empty_ignore": all(len(b) == 0 for b in batch["ignore_boxes_xyxy"]),
            "classification_only_intervention": True,
        }
        self.observed_batches.append(dict(self.last_mask_receipt))
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss,
                dict(zip(self.loss_names, loss.detach())))


class IgnoreAwareDetectionModel(DetectionModel):
    def init_criterion(self):
        return IgnoreAwareDetectionLoss(self)


class RestrictedIgnoreTrainer(DetectionTrainer):
    """Actual Trainer lifecycle constrained to a one-epoch CPU synthetic smoke.

    Model architecture is supplied as a dictionary, not a checkpoint. The
    dataset factory receives keyword arguments mode, img_path, trainer, batch.
    The FP32 capture callback runs before the framework's FP16 serialization.
    """

    def __init__(self, *, training_contract, model_cfg, dataset_factory=None,
                 fp32_capture_callback=None, overrides=None, _callbacks=None):
        self.integration_source_pins = verify_framework_sources()
        self.training_contract = _validate_contract(training_contract)
        _require(isinstance(model_cfg, dict), "synthetic model architecture must be an in-memory dictionary")
        self.synthetic_model_cfg = deepcopy(model_cfg)
        self.dataset_factory = dataset_factory
        self.fp32_capture_callback = fp32_capture_callback
        self.integration_events = []
        super().__init__(cfg=DEFAULT_CFG, overrides=validate_synthetic_overrides(overrides), _callbacks=_callbacks)

    def get_dataset(self):
        c = self.training_contract
        return {"nc": c["class_count"], "names": dict(enumerate(c["names"])), "channels": 3,
                "train": c["datasets"]["train"]["manifest_path"],
                "val": c["datasets"]["training_val"]["manifest_path"]}

    def setup_model(self):
        self.model = self.get_model(cfg=self.synthetic_model_cfg, weights=None, verbose=self.args.verbose)
        self.resume = False
        return None

    def get_model(self, cfg=None, weights=None, verbose=True):
        _require(weights is None, "weight loading is forbidden in the synthetic integration")
        _require(isinstance(cfg, dict), "only an in-memory architecture dictionary is allowed")
        return IgnoreAwareDetectionModel(deepcopy(cfg), ch=3, nc=self.data["nc"], verbose=verbose)

    def build_dataset(self, img_path, mode="train", batch=None):
        _require(mode in {"train", "val"}, "unsupported dataset mode")
        role = "train" if mode == "train" else "training_val"
        binding = self.training_contract["datasets"][role]
        _require(str(img_path) == binding["manifest_path"], "dataset path is outside the bound contract")
        if self.dataset_factory is not None:
            dataset = self.dataset_factory(mode=mode, img_path=img_path, trainer=self, batch=batch)
        else:
            try:
                from .gate1_ignore_dataset import RestrictedIgnoreDataset
            except ImportError:
                from gate1_ignore_dataset import RestrictedIgnoreDataset
            dataset = RestrictedIgnoreDataset(
                binding["manifest_path"], binding["manifest_sha256"], split_role=role,
                image_size=self.args.imgsz, class_count=self.data["nc"], augment=mode == "train",
                fliplr=self.args.fliplr if mode == "train" else 0.0,
                flipud=self.args.flipud if mode == "train" else 0.0, seed=self.args.seed,
            )
        self.integration_events.append({"event": "bound_dataset_built", "role": role, "samples": len(dataset)})
        return dataset

    def _build_train_pipeline(self):
        _require(not getattr(self, "_oom_retries", 0), "OOM recovery/automatic batch reduction is forbidden")
        return super()._build_train_pipeline()

    def preprocess_batch(self, batch):
        _require(self.args.multi_scale == 0.0 and self.args.amp is False, "unsupported dynamic training options")
        _require("ignore_boxes_xyxy" in batch and "image_identities" in batch, "missing ignore sidecar batch")
        _require(isinstance(batch["ignore_boxes_xyxy"], (tuple, list)), "ragged ignore geometry is required")
        _require(batch["img"].dtype == torch.uint8, "Dataset must supply uint8 images before normalization")
        _require(len(batch["ignore_boxes_xyxy"]) == len(batch["img"])
                 and len(batch["image_identities"]) == len(batch["img"]), "one geometry and identity per image is required")
        _require(all(isinstance(b, torch.Tensor) and b.dtype == torch.float32 and not b.requires_grad
                     and b.ndim == 2 and b.shape[1] == 4 for b in batch["ignore_boxes_xyxy"]),
                 "Dataset ignore sidecars must be detached FP32 [N,4] before device transfer")
        batch = super().preprocess_batch(batch)
        batch["ignore_boxes_xyxy"] = [b.to(device=self.device) for b in batch["ignore_boxes_xyxy"]]
        return batch

    def get_validator(self):
        self.integration_events.append({"event": "validator_disabled", "status": "SMOKE_NO_TASK_AP"})
        return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

    def validate(self):
        self.integration_events.append({"event": "validation_skipped", "status": "SMOKE_NO_TASK_AP"})
        self.best_fitness = 0.0
        return {}, 0.0  # lifecycle sentinel; never an accuracy or AP statistic

    def final_eval(self):
        self.integration_events.append({"event": "final_eval_skipped", "status": "SMOKE_NO_TASK_AP",
                                        "reason": "no half/fused checkpoint validation in synthetic capture proof"})

    def _handle_nan_recovery(self, epoch):
        _require(self.loss is not None and bool(torch.isfinite(self.loss).all()),
                 "nonfinite loss: fail closed; checkpoint recovery forbidden")
        for module in (self.model, self.ema.ema):
            _require(all(bool(torch.isfinite(v).all()) for v in module.state_dict().values() if v.is_floating_point()),
                     "nonfinite model/EMA: fail closed; checkpoint recovery forbidden")
        return False

    def save_model(self):
        self._handle_nan_recovery(self.epoch)
        self.integration_events.append({"event": "save_model_before_fp32_callback", "epoch": self.epoch})
        if self.fp32_capture_callback is not None:
            self.fp32_capture_callback(self)
            self.integration_events.append({"event": "fp32_callback_completed", "epoch": self.epoch})
        result = super().save_model()
        self.integration_events.append({"event": "framework_half_checkpoint_serialized", "epoch": self.epoch})
        return result
