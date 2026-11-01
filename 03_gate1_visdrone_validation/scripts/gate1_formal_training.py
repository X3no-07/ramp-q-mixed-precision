"""New bound CUDA training path; previous synthetic-only modules stay frozen.

The loss and image/ignore transforms reuse the reviewed components. Real-data
membership, CUDA capture and fixed-final-epoch selection have new contracts.
Training completion is candidate generation, never scientific acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import DEFAULT_CFG

import gate1_ignore_dataset as D
import gate1_fp32_live_capture as C
from gate1_ignore_trainer import (IgnoreAwareDetectionModel, RestrictedIgnoreTrainer,
    SAFE_SYNTHETIC_OVERRIDES, verify_framework_sources)

require = D._require
SCHEMA = "rampq.gate1.bound_ignore_dataset_manifest/v2"
DATA_ROLES = {"real_training_v1", "synthetic_cuda_qualification_v1"}
POLICY = {**D.POLICY, "schema": "rampq.gate1.bound_ignore_dataset_policy/v2",
          "current_execution_data_role": "explicit_bound_real_or_CUDA_qualification_manifest",
          "real_data_training_verified": False,
          "invalid_source_sample": "exclude_whole_flagged_image_in_NEW_contract_preserve_originals_no_role_transfer"}
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def now():
    return datetime.now(timezone.utc).isoformat()


def info(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class BoundIgnoreDataset(D.RestrictedIgnoreDataset):
    """New header validation with the frozen loader and geometry implementation."""
    def __init__(self, manifest_path, manifest_sha256, *, split_role, data_role,
                 image_size=640, class_count=10, augment=False, fliplr=0., flipud=0., seed=0, epoch=0):
        require(data_role in DATA_ROLES and split_role in {"train", "training_val"}, "Unsupported declared dataset role")
        require(type(image_size) is int and image_size >= 64 and image_size % 32 == 0, "Invalid image size")
        require(class_count == 10 and type(class_count) is int, "Requires ten classes")
        require(type(augment) is bool and all(type(p) in {int, float} and 0 <= p <= 1 for p in (fliplr, flipud)), "Invalid flips")
        require((augment and split_role == "train") or (fliplr == 0 and flipud == 0), "Validation flips forbidden")
        require(type(seed) is int and seed >= 0 and type(epoch) is int and epoch >= 0, "Invalid seed or epoch")
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        D._reject_forbidden_absolute(self.manifest_path)
        require(isinstance(manifest_sha256, str) and D._DIGEST.fullmatch(manifest_sha256), "Invalid manifest SHA")
        self.manifest_sha256 = manifest_sha256
        raw = self.manifest_path.read_bytes()
        require(D._sha(raw) == manifest_sha256, "Manifest byte identity mismatch")
        manifest = D._json(raw)
        require(set(manifest) == {"schema", "data_role", "split_role", "class_count", "data_root", "policy", "samples"}, "Manifest fields differ")
        require(manifest["schema"] == SCHEMA and manifest["policy"] == POLICY and manifest["data_role"] == data_role,
                "New dataset schema, policy or execution role mismatch")
        require(manifest["split_role"] == split_role and manifest["class_count"] == class_count, "Manifest split/class mismatch")
        D._safe_path_text(manifest["data_root"])
        self.root = (self.manifest_path.parent / manifest["data_root"]).resolve(strict=True)
        require(self.root.is_relative_to(self.manifest_path.parent), "Data root escaped manifest parent")
        D._reject_forbidden_absolute(self.root)
        require(isinstance(manifest["samples"], list) and bool(manifest["samples"]), "Explicit nonempty membership required")
        self.entries = deepcopy(manifest["samples"])
        ids, paths = set(), set()
        for row in self.entries:
            require(set(row) == {"image_id", "split_role", "original_hw", "image", "labels", "sidecar"}, "Sample fields differ")
            require(isinstance(row["image_id"], str) and D._IMAGE_ID.fullmatch(row["image_id"]) and row["image_id"] not in ids,
                    "Invalid or duplicate sample ID")
            ids.add(row["image_id"])
            require(row["split_role"] == split_role and isinstance(row["original_hw"], list) and len(row["original_hw"]) == 2 and
                    all(type(v) is int and v > 0 for v in row["original_hw"]), "Sample role/dimensions mismatch")
            for key in ("image", "labels", "sidecar"):
                member = row[key]
                require(isinstance(member, dict) and set(member) == {"path", "bytes", "sha256"}, "Member fields differ")
                D._safe_path_text(member["path"])
                require(type(member["bytes"]) is int and member["bytes"] >= 0 and isinstance(member["sha256"], str)
                        and D._DIGEST.fullmatch(member["sha256"]), "Invalid member identity")
                require(member["path"] not in paths, "Duplicate member path")
                paths.add(member["path"])
        self.split_role, self.imgsz, self.class_count = split_role, image_size, class_count
        self.data_role = data_role
        self.augment, self.fliplr, self.flipud, self.seed, self.epoch = augment, float(fliplr), float(flipud), seed, epoch
        self.geometry = D._load_geometry()
        self.rect = self.mosaic = self.use_segments = self.use_keypoints = self.use_obb = False
        self.data = {"nc": class_count, "channels": 3, "names": dict(enumerate(NAMES))}
        self.labels, self.im_files, self.label_files = [], [], []
        self._entry_json = tuple(json.dumps(row, sort_keys=True, allow_nan=False) for row in self.entries)
        for entry in self.entries:
            _, labels, _, members = self._load_sample(entry)
            self.im_files.append(str(members["image"]))
            self.label_files.append(str(members["labels"]))
            self.labels.append({"im_file": str(members["image"]), "shape": tuple(entry["original_hw"]),
                "cls": labels[:, :1].astype(np.float32), "bboxes": labels[:, 1:].astype(np.float32),
                "segments": [], "keypoints": None, "normalized": True, "bbox_format": "xywh"})

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["ignore_metadata"]["policy"] = deepcopy(POLICY)
        sample["ignore_metadata"]["data_role"] = self.data_role
        return sample


def validate_membership(train, val):
    """Check both manifest member sets before Dataset opens their member bytes."""
    require(train["schema"] == val["schema"] == SCHEMA and train["data_role"] == val["data_role"] in DATA_ROLES,
            "Dataset pair roles differ")
    require(train["split_role"] == "train" and val["split_role"] == "training_val", "Dataset pair split mismatch")
    for field in ("image_id",):
        left = [r[field] for r in train["samples"]]
        right = [r[field] for r in val["samples"]]
        require(len(set(left)) == len(left) and len(set(right)) == len(right) and set(left).isdisjoint(right), "Dataset pair ID leakage")
    left = {r["image"]["sha256"] for r in train["samples"]}
    right = {r["image"]["sha256"] for r in val["samples"]}
    require(left.isdisjoint(right), "Dataset pair image-content leakage")


def backend_settings():
    return {"threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def enforce_backend():
    require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1, "CPU thread drift")
    require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(), "Deterministic policy drift")
    require(not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic and
            not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32, "CUDA backend policy drift")
    return backend_settings()


class CUDALiveFP32Capture(C.LiveFP32Capture):
    """Synchronize source CUDA EMA; exact state, CPU and CUDA reconstruction."""
    @torch.no_grad()
    def _capture(self, trainer):
        require(trainer.args.amp is False and trainer.amp is False, "AMP must be disabled")
        epoch = trainer.epoch + 1
        if epoch not in self.selected_epochs:
            return None
        settings = enforce_backend()
        require(C.sources() == self.expected_sources, "Capture runtime source drift")
        ema = trainer.ema
        require(ema.enabled is True and type(ema.updates) is int and ema.updates > 0, "Updated enabled EMA required")
        live = ema.ema
        require(not live.training and all(v.device == trainer.device for v in live.state_dict().values()) and
                trainer.device.type == "cuda", "Live EMA must already be on the bound CUDA device")
        require(all(v.dtype == torch.float32 for v in live.state_dict().values() if v.is_floating_point()), "Source must already be FP32")
        torch.cuda.synchronize(trainer.device)
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(trainer.device).clone()
        before_updates = ema.updates
        frozen = C.FORMAT.freeze_state(live.state_dict())
        before = C.FORMAT.describe_state(frozen)
        arch = C.architecture(live)
        arch_sha = hashlib.sha256(C.canonical(arch).encode()).hexdigest()
        final = self.output_root / f"epoch_{epoch:04d}"
        pending = final.with_name(final.name + ".pending")
        require(not final.exists() and not pending.exists(), "Capture destination must be new")
        pending.mkdir(parents=True)
        C.write_json(pending / "architecture.json", arch)
        C.FORMAT.write_snapshot(frozen, pending / "state", {"training_contract_sha256": self.training_contract_sha256,
            "architecture_sha256": arch_sha, "epoch_one_based": epoch, "source_stage": C.FORMAT.STAGE})
        cpu_copy = deepcopy(live).cpu().eval()
        cpu_check = C.reconstruct_and_verify(pending / "state", arch, self.expected_sources, cpu_copy, self.synthetic_input)
        del cpu_copy
        with torch.random.fork_rng(devices=[]):
            rebuilt = DetectionModel(deepcopy(arch["cfg"]), ch=3, nc=arch["nc"], verbose=False)
        rebuilt.load_state_dict(frozen, strict=True)
        rebuilt.names = dict(enumerate(arch["names"]))
        rebuilt = rebuilt.to(trainer.device).float().eval()
        gpu_copy = deepcopy(live).eval()
        probe = self.synthetic_input.to(trainer.device)
        left = C.describe_output(gpu_copy(probe.clone()))
        right = C.describe_output(rebuilt(probe.clone()))
        torch.cuda.synchronize(trainer.device)
        require(left == right, "CUDA reconstructed synthetic outputs differ")
        del rebuilt, gpu_copy, probe
        after = C.FORMAT.describe_state(C.FORMAT.freeze_state(live.state_dict()))
        require(before == after and ema.updates == before_updates, "Live EMA changed during capture")
        require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state(trainer.device)),
                "Capture changed training RNG")
        require(C.sources() == self.expected_sources and enforce_backend() == settings, "Capture source/backend drift")
        receipt = {"schema": "rampq.gate1.cuda_fp32_live_capture/v1", "status": "PASS_BOUND_CUDA_FP32_CAPTURE_AND_EXACT_RECONSTRUCTION_ONLY",
            "phase": C.PHASE, "epoch_one_based": epoch, "ema_updates_before": before_updates, "ema_updates_after": ema.updates,
            "training_contract_sha256": self.training_contract_sha256, "sources": self.expected_sources,
            "architecture_sha256": arch_sha, "device": str(trainer.device), "amp": False,
            "cuda_synchronize_before_state_copy": True, "source_already_FP32": True, "half_lift_performed": False,
            "state_and_update_counter_unchanged": True, "training_torch_RNG_unchanged": True,
            "backend_settings": settings, "cpu_reconstruction": cpu_check,
            "cuda_reconstruction": {"input": C._tensor_description(self.synthetic_input), "output": right,
                                    "output_values_shapes_dtypes_exact": True},
            "live_state_sha256": hashlib.sha256(C.canonical(before).encode()).hexdigest(),
            "main_thread_nonreentrant": True, "caller_requires_no_concurrent_EMA_writers": True,
            "optimizer_resume_certified": False, "scientific_reference_accepted": False}
        C.write_json(pending / "capture_receipt.json", receipt)
        C.write_json(pending / "manifest.json", {"files": {p.relative_to(pending).as_posix():
            {k: v for k, v in info(p).items() if k != "path"} for p in sorted(pending.rglob("*")) if p.is_file()}})
        pending.rename(final)
        self.receipts.append(receipt)
        return receipt


def validate_contract(contract):
    require(contract.get("schema") == "rampq.gate1.bound_cuda_training_contract/v1", "Wrong training contract")
    require(contract["data_role"] in DATA_ROLES and contract["candidate_selection"] == "fixed_final_epoch_EMA_no_AP_selection",
            "Unsupported data role or candidate selection")
    require(contract["class_count"] == 10 and contract["names"] == NAMES, "Class semantics changed")
    args = contract["overrides"]
    require(args["device"] == "0" and args["batch"] == 4 and args["nbs"] == 64 and args["imgsz"] == 640 and
            args["amp"] is False and args["workers"] == 0 and args["resume"] is False and args["val"] is False,
            "Bound CUDA training configuration mismatch")
    expected_epochs = 30 if contract["data_role"] == "real_training_v1" else 1
    require(args["epochs"] == expected_epochs and contract["epochs"] == expected_epochs, "Epoch budget changed")
    immutable = {k: v for k, v in SAFE_SYNTHETIC_OVERRIDES.items() if k not in
        {"device", "epochs", "batch", "nbs", "imgsz", "model", "data", "fliplr", "seed", "warmup_epochs", "pretrained"}}
    require(all(args[k] == v for k, v in immutable.items()), "Restricted augmentation/training policy drift")
    manifests = {}
    for role in ("train", "training_val"):
        binding = contract["datasets"][role]
        require(info(binding["manifest_path"])["sha256"] == binding["manifest_sha256"], "Dataset binding changed")
        manifests[role] = D._json(Path(binding["manifest_path"]).read_bytes())
        require(manifests[role]["data_role"] == contract["data_role"] and manifests[role]["policy"] == POLICY, "Dataset execution role/policy changed")
        require(len(manifests[role]["samples"]) == contract["counts"][role], "Bound dataset count changed")
    validate_membership(manifests["train"], manifests["training_val"])
    return deepcopy(contract)


class BoundCUDATrainer(DetectionTrainer):
    """One GPU, explicit manifests, fixed final epoch; AP evaluated separately."""

    def preprocess_batch(self, batch):
        # Call the shared ancestor explicitly: the frozen synthetic adapter's
        # zero-argument super() is bound to its own class, not to this adapter.
        require(self.args.multi_scale == 0.0 and self.args.amp is False, "Unsupported dynamic training options")
        require(batch["img"].dtype == torch.uint8, "Dataset must provide uint8 images")
        require("ignore_boxes_xyxy" in batch and "image_identities" in batch, "Missing sidecar geometry/identity")
        require(isinstance(batch["ignore_boxes_xyxy"], (tuple, list)) and
                len(batch["ignore_boxes_xyxy"]) == len(batch["image_identities"]) == len(batch["img"]), "One ragged ignore set per image required")
        require(all(isinstance(b, torch.Tensor) and b.dtype == torch.float32 and not b.requires_grad and
                    b.ndim == 2 and b.shape[1] == 4 for b in batch["ignore_boxes_xyxy"]), "Ignore boxes must be detached FP32 [N,4]")
        result = DetectionTrainer.preprocess_batch(self, batch)
        result["ignore_boxes_xyxy"] = [b.to(device=self.device) for b in result["ignore_boxes_xyxy"]]
        return result

    def __init__(self, contract, model_cfg, *, event_callback=None, capture_callback=None):
        self.training_contract = validate_contract(contract)
        self.integration_source_pins = verify_framework_sources()
        self.model_cfg = deepcopy(model_cfg)
        require(hashlib.sha256(C.canonical(model_cfg).encode()).hexdigest() == contract["model_cfg_sha256"], "Model graph changed")
        self.event_callback = event_callback or (lambda name, **fields: None)
        self.capture_callback = capture_callback
        self.integration_events = []
        self.optimizer_steps_observed = 0
        self.batch_count_observed = 0
        super().__init__(cfg=DEFAULT_CFG, overrides=deepcopy(contract["overrides"]))

    def get_dataset(self):
        c = self.training_contract
        return {"nc": 10, "channels": 3, "names": dict(enumerate(NAMES)),
                "train": c["datasets"]["train"]["manifest_path"], "val": c["datasets"]["training_val"]["manifest_path"]}

    def setup_model(self):
        self.model = IgnoreAwareDetectionModel(deepcopy(self.model_cfg), ch=3, nc=10, verbose=False)
        weights = self.training_contract.get("initialization_weights")
        if weights is not None:
            require(info(weights["path"]) == weights, "Initialization weights changed")
            initial, _ = load_checkpoint(weights["path"], device="cpu", fuse=False)
            self.model.load(initial, verbose=True)
            del initial
            self.event_callback("pretrained_initialization_loaded", weights=weights,
                note="published_checkpoint_initialization_not_recovery_of_old_FP32_EMA")
        self.resume = False
        return None

    def build_dataset(self, img_path, mode="train", batch=None):
        role = "train" if mode == "train" else "training_val"
        require(mode in {"train", "val"}, "Unsupported dataset mode")
        binding = self.training_contract["datasets"][role]
        require(str(img_path) == binding["manifest_path"], "Dataset path outside contract")
        dataset = BoundIgnoreDataset(binding["manifest_path"], binding["manifest_sha256"], split_role=role,
            data_role=self.training_contract["data_role"], image_size=self.args.imgsz, class_count=10,
            augment=mode == "train", fliplr=self.args.fliplr if mode == "train" else 0.,
            flipud=self.args.flipud if mode == "train" else 0., seed=self.args.seed)
        self.event_callback("bound_dataset_built", role=role, samples=len(dataset))
        return dataset

    def _build_train_pipeline(self):
        require(int(self.batch_size) == 4 and not getattr(self, "_oom_retries", 0), "Batch reduction/recovery forbidden")
        return super()._build_train_pipeline()

    def optimizer_step(self):
        enforce_backend()
        require(self.batch_size == 4 and not self.amp, "Optimizer configuration drift")
        before = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad} \
            if self.optimizer_steps_observed < 2 else None
        updates = self.ema.updates
        super().optimizer_step()
        self.optimizer_steps_observed += 1
        require(self.ema.updates == updates + 1, "EMA update count mismatch")
        if before is not None:
            changed = sum(int((before[n] != p.detach()).sum().item()) for n, p in self.model.named_parameters() if n in before)
            require(changed > 0, "Optimizer did not change observed parameter")
            self.event_callback("initial_optimizer_step_verified", step=self.optimizer_steps_observed,
                trainable_parameter_changed_elements=changed, ema_updates=self.ema.updates)

    def get_validator(self):
        self.event_callback("AP_validation_deferred", policy="fixed_final_epoch_no_task_AP_selection")
        return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

    def validate(self):
        self.best_fitness = 0.0
        return {}, 0.0  # Framework lifecycle sentinel only, never a task score.

    def final_eval(self):
        self.event_callback("framework_half_checkpoint_final_eval_skipped", candidate="fixed_final_FP32_EMA")

    def _handle_nan_recovery(self, epoch):
        require(self.loss is not None and bool(torch.isfinite(self.loss).all()), "Nonfinite training loss; recovery forbidden")
        for module in (self.model, self.ema.ema):
            require(all(bool(torch.isfinite(v).all()) for v in module.state_dict().values() if v.is_floating_point()),
                    "Nonfinite live model/EMA; recovery forbidden")
        return False

    def save_model(self):
        self._handle_nan_recovery(self.epoch)
        require(self.capture_callback is not None, "CUDA FP32 capture callback required")
        self.event_callback("save_before_FP32_capture", epoch=self.epoch + 1, ema_updates=self.ema.updates)
        captured = self.capture_callback(self)
        require(captured is not None, "Every epoch requires an FP32 reference capture")
        self.event_callback("FP32_reference_committed", epoch=self.epoch + 1, ema_updates=self.ema.updates,
            status=captured["status"], cuda_exact=captured["cuda_reconstruction"]["output_values_shapes_dtypes_exact"])
        # Preserve ordinary framework recovery files separately. Their half EMA
        # and best-fitness sentinel are not the selected scientific reference.
        saved = super().save_model()
        self.event_callback("framework_compatibility_checkpoint_saved", epoch=self.epoch + 1,
                            native_best_is_task_selected=False, scientific_reference="separate_FP32_capture")
        return saved
