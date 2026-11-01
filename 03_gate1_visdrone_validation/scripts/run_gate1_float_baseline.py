"""Freeze, validate, or execute one isolated Gate1 float-baseline training attempt.

Only official-train-derived train/training_val pixels and YOLO labels are opened.
Selection and test-dev are never passed to Ultralytics. This runner does not
certify official VisDrone AP, scientific acceptance, or quantization recovery.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback


WEIGHT_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
ACTIVE_ROLES = {"train", "training_val", "calibration_pool", "debug", "selection"}
ALLOWED_ROLES = ACTIVE_ROLES | {"quarantine", "duplicate_excluded"}
FIXED_ARGS = {
    "task": "detect", "mode": "train", "epochs": 30, "imgsz": 640,
    "batch": 4, "workers": 0, "amp": False, "cache": False, "plots": False,
    "val": True, "device": "0", "seed": 20261022, "deterministic": True,
    "optimizer": "SGD", "lr0": 0.01, "momentum": 0.937,
    "weight_decay": 0.0005, "patience": 0, "close_mosaic": 10,
    "save": True, "save_period": -1, "resume": False, "pretrained": True,
    "fraction": 1.0, "time": None, "exist_ok": False, "verbose": True,
    "single_cls": False,
}
FRAMEWORK_SOURCES = [
    "engine/trainer.py", "engine/model.py", "engine/validator.py", "utils/metrics.py", "utils/torch_utils.py",
    "models/yolo/detect/train.py", "models/yolo/detect/val.py", "cfg/default.yaml",
    "cfg/__init__.py", "data/dataset.py", "data/augment.py", "nn/tasks.py",
]
LIMITS = [
    "Builtin training_val validation is ignore-unaware and is only a checkpoint-selection proxy.",
    "Fitness uses builtin mAP50-95; tied best fitness saves the later epoch.",
    "No Selection detector inference and no Holdout access are authorized by this runner.",
    "A completed run is not automatically an accepted scientific baseline or official VisDrone AP.",
    "Filename/pixel grouping does not establish true camera or scene identity.",
    "Framework checkpoint serialization is fp16 EMA; training uses amp=False, and no quantization is applied.",
    "Ordinary YOLO training loss does not mask ignored regions as negative samples; ignored annotations are only omitted from YOLO positive labels.",
    "Thirty epochs is the fixed first candidate-training budget, not evidence of sufficient convergence or a final research method.",
    "Float qualification of preprocessing, box restoration, ignore-aware target hits, and FP32/FP16 numerics remains a separate pending step.",
]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_file(root: Path, relative: str) -> Path:
    """Reject traversal, absolute paths, and symbolic links before resolving."""
    rel = Path(relative)
    require(not rel.is_absolute() and rel.parts and ".." not in rel.parts, f"Unsafe relative file: {relative}")
    cursor = root
    for part in rel.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), f"Symbolic link is forbidden: {cursor}")
    path = cursor.resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), f"File outside root: {path}")
    return path


def file_identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_manifest(root: Path) -> dict:
    manifest = json.loads(safe_file(root, "manifest.json").read_text(encoding="utf-8"))
    require(isinstance(manifest.get("files"), dict) and bool(manifest["files"]), "Empty metadata manifest")
    for relative, expected in manifest["files"].items():
        path = safe_file(root, relative)
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], f"Manifest mismatch: {relative}")
    return manifest


def validate_roles(rows: list[dict]) -> None:
    require(len({r["image_id"] for r in rows}) == len(rows), "Duplicate image ID")
    for row in rows:
        role = row["assigned_split"]
        require(role in ALLOWED_ROLES, f"Unexpected role: {role}")
        require(row["source_split"] in {"train", "val"}, "Unknown source split; Holdout is forbidden")
        if row["source_split"] == "val":
            require(role == "selection", "Official val must remain Selection")
        elif role in {"train", "training_val"}:
            require(row["is_content_canonical"], "Noncanonical content in training")
            require(not row.get("requires_annotation_quality_quarantine", False), "Quality quarantine reached training")
        require(not (role == "selection" and row["source_split"] != "val"), "Selection must be official val only")
    for field in ("image_sha256", "prefix_proxy", "refined_component"):
        roles = defaultdict(set)
        for row in rows:
            if row["assigned_split"] in ACTIVE_ROLES:
                roles[row[field]].add(row["assigned_split"])
        require(all(len(value) == 1 for value in roles.values()), f"Active-role leakage: {field}")


def validate_plan(plan_dir: Path, verify_source_files: bool = False) -> tuple[dict, list[dict], dict]:
    plan_dir = plan_dir.resolve(strict=True)
    require(not plan_dir.is_symlink(), "Plan directory cannot be a symbolic link")
    manifest = verify_manifest(plan_dir)
    required = {"split_plan.json", "split_assignment.jsonl", "splits/train.txt", "splits/training_val.txt"}
    require(required <= set(manifest["files"]), "Split-plan manifest lacks required identities")
    plan = json.loads(safe_file(plan_dir, "split_plan.json").read_text(encoding="utf-8"))
    require(plan["holdout_accessed"] is False, "Holdout access is outside contract")
    require(plan["status"] == "REFINED_SPLIT_PLAN_NOT_SCIENTIFIC_GATE_PASS", "Unknown split-plan status")
    prepared = Path(plan["source_prepared"]).resolve(strict=True)
    source_manifest = safe_file(prepared, "image_manifest.jsonl")
    require(sha(source_manifest) == plan["source_manifest_sha256"], "Prepared source manifest changed")
    require(sha(safe_file(prepared, "manifest.json")) == plan["source_file_manifest_sha256"], "Prepared file manifest changed")
    source_rows = read_jsonl(source_manifest)
    source_by_id = {r["image_id"]: r for r in source_rows}
    rows = read_jsonl(safe_file(plan_dir, "split_assignment.jsonl"))
    require(len(source_by_id) == len(source_rows) == len(rows), "Manifest coverage mismatch")
    validate_roles(rows)
    for row in rows:
        source = source_by_id.get(row["image_id"])
        require(source is not None, "Unknown image in refined split")
        require(row["previous_assigned_split"] == source["assigned_split"], "Previous role changed")
        for key, value in source.items():
            if key != "assigned_split":
                require(row.get(key) == value, f"Source identity changed: {row['image_id']} / {key}")
    counts = dict(Counter(row["assigned_split"] for row in rows))
    require(counts == plan["counts"], "Split counts disagree")
    require(counts.get("train", 0) >= 3000 and counts.get("training_val", 0) >= 400, "Insufficient formal training data")
    val_groups = Counter(r["refined_component"] for r in rows if r["assigned_split"] == "training_val")
    require(len(val_groups) >= 12 and max(val_groups.values()) <= 64, "Training validation diversity policy failed")
    selected = [r for r in rows if r["assigned_split"] in {"train", "training_val"}]
    identities = []
    for role in ("train", "training_val"):
        listed = [str(Path(line).resolve(strict=True)) for line in safe_file(plan_dir, f"splits/{role}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = [str((prepared / row["image_path"]).resolve(strict=True)) for row in rows if row["assigned_split"] == role]
        require(listed == expected, f"Split-list order/membership mismatch: {role}")
    for row in selected:
        image = safe_file(prepared, row["image_path"])
        label = safe_file(prepared, row["label_path"])
        require(image.is_relative_to(prepared / "images" / "train"), "Image is not official-train-derived")
        require(label.is_relative_to(prepared / "labels" / "train"), "Label is not official-train-derived")
        if verify_source_files:
            require(sha(image) == row["image_sha256"] and sha(label) == row["label_sha256"], f"Training source bytes changed: {row['image_id']}")
        identities.append({"image_id": row["image_id"], "role": row["assigned_split"], "image": str(image), "label": str(label), "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"]})
    audit = {"status": "PASS_PLAN_METADATA_AND_ROLE_VALIDATION", "plan_dir": str(plan_dir), "source_prepared": str(prepared), "rows": len(rows), "counts": counts,
             "training_val_components": len(val_groups), "training_val_largest_component": max(val_groups.values()),
             "plan_manifest_sha256": sha(plan_dir / "manifest.json"), "split_assignment_sha256": sha(plan_dir / "split_assignment.jsonl"),
             "source_manifest_sha256": sha(source_manifest), "selected_source_files_verified": verify_source_files,
             "selected_source_identities": identities, "selection_pixels_or_labels_opened": False, "holdout_accessed": False}
    return plan, rows, audit


def isolate_settings(root: Path) -> None:
    config = root / "ultralytics_config"
    config.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
    os.environ.pop("PYTHONPATH", None)
    sys.dont_write_bytecode = True


def runtime_snapshot() -> tuple[dict, dict, Path]:
    import torch
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT

    root = Path(ultralytics.__file__).parent
    packages = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics", "numpy", "Pillow")}
    sources = {relative: file_identity(root / relative) for relative in FRAMEWORK_SOURCES}
    env = {"python": sys.version, "executable": str(Path(sys.executable).resolve()), "python_executable_sha256": sha(Path(sys.executable)),
           "platform": platform.platform(), "packages": packages, "cuda_available": torch.cuda.is_available(),
           "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "framework_sources": sources,
           "gpu": [{"index": i, "name": torch.cuda.get_device_name(i), "total_memory": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]}
    return env, jsonable(DEFAULT_CFG_DICT), root


def disable_integrations(output: Path) -> None:
    import ultralytics
    ultralytics.settings.update({"sync": False, "wandb": False, "comet": False, "clearml": False,
        "mlflow": False, "dvc": False, "raytune": False, "tensorboard": False,
        "runs_dir": str(output / "runs"), "weights_dir": str(output / "weights")})


def environment_fingerprint(env: dict) -> dict:
    return {key: env[key] for key in ("python", "python_executable_sha256", "packages", "torch_cuda", "cudnn", "gpu") } | {
        "framework_source_sha256": {key: value["sha256"] for key, value in env["framework_sources"].items()}}


def implementation_evidence(root: Path) -> dict:
    patterns = {
        "engine/trainer.py": ["self._oom_retries >= 3", "self.args.batch = self.batch_size = max", "if self.best_fitness == self.fitness:",
            "self.best_fitness < fitness", "def _handle_nan_recovery", "if epoch == self.start_epoch:", "self.nan_recovery_attempts += 1", "self.epoch += 1", 'self.run_callbacks("on_fit_epoch_end")', "ema = deepcopy", "torch.nan_to_num_(v)"],
        "utils/metrics.py": ["w = [0.0, 0.0, 0.0, 1.0]"],
    }
    evidence = {}
    for relative, needles in patterns.items():
        path = root / relative
        lines = path.read_text(encoding="utf-8").splitlines()
        matches = [{"line": i + 1, "text": line.strip()} for i, line in enumerate(lines) if any(needle in line for needle in needles)]
        evidence[relative] = {"sha256": sha(path), "matches": matches}
    return {"source": "Installed Ultralytics source, locally pinned by SHA256; no claim of official VisDrone evaluator equivalence", "evidence": evidence}


def freeze_contract(plan_dir: Path, weights: Path, output: Path) -> dict:
    require(sha(weights) == WEIGHT_SHA256, "Unexpected official starting-weight bytes")
    plan, rows, audit = validate_plan(plan_dir, verify_source_files=False)
    isolate_settings(output)
    env, defaults, framework_root = runtime_snapshot()
    require(env["cuda_available"] and bool(env["gpu"]), "This contract requires the existing CUDA GPU")
    disable_integrations(output)
    shutil.copyfile(Path(__file__).resolve(), output / "runner_source.py")
    for relative in FRAMEWORK_SOURCES:
        destination = output / "framework_sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(framework_root / relative, destination)
    write_json(output / "environment.json", env)
    write_json(output / "framework_defaults.json", defaults)
    write_json(output / "implementation_evidence.json", implementation_evidence(framework_root))
    write_json(output / "plan_validation.json", audit)
    contract = {"schema": "rampq.gate1.float_training_contract/v1", "status": "FROZEN_NOT_STARTED", "frozen_at_utc": now(),
        "runner_sha256": sha(Path(__file__).resolve()), "weights": file_identity(weights), "split_plan": str(plan_dir),
        "split_plan_manifest_sha256": audit["plan_manifest_sha256"], "split_assignment_sha256": audit["split_assignment_sha256"],
        "source_manifest_sha256": audit["source_manifest_sha256"], "counts": audit["counts"], "fixed_args": FIXED_ARGS,
        "framework_defaults_sha256": sha(output / "framework_defaults.json"), "environment_sha256": sha(output / "environment.json"),
        "environment_fingerprint": environment_fingerprint(env), "checkpoint_policy": {
            "selection_data_role": "training_val", "selection_metric": "Ultralytics builtin ignore-unaware mAP50-95", "ties": "later epoch",
            "scientific_acceptance": False, "official_visdrone_ap": False},
        "recovery_policy": {"external_restarts": 0, "framework_source_modified": False,
            "custom_trainer_guard": "Abort before a rebuilt pipeline if batch differs from 4; inspect loss/loss_items/tloss after every training batch and loss/fitness before default epoch NaN recovery. Before every checkpoint save, reject nonfinite live/EMA state or fp16 EMA overflow before framework resync/clamping can run.",
            "default_framework_behavior": "First-epoch memory error may halve batch up to 3 times; NaN recovery may reload last.pt up to 3 times. save_model may resync nonfinite EMA or clamp its fp16 serialization. Guards prevent accepting those altered attempts.",
            "if_triggered": "Fail this attempt, preserve receipts/checkpoints, require a new reviewed contract; no automatic restart."},
        "training_started": False, "selection_detector_inference": False, "holdout_accessed": False, "limits": LIMITS}
    write_json(output / "contract.json", contract)
    (output / "TRAINING_CONTRACT.md").write_text(
        '# Gate1 First-round float training candidate contract\n\n'
        'Status: frozen, training has not yet started. 30 epochs is the fixed budget for the first round and cannot be used to conclude that sufficient convergence has been achieved.\n\n'
        'YOLO11n official COCO initial weights; 10 classes VisDrone; 640; batch size 4; workers 0; AMP disabled;'
        "SGD，lr0 0.01，momentum 0.937，weight decay 0.0005；seed 20261022；deterministic；"
        'patience 0; disable mosaic in the last 10 epochs. All other default values are fixed by framework_defaults.json and the source SHA.'
        'Fully pass and save the final effective parameters during execution.\n\n'
        'Only copy the train/training_val splits from the new split to an independent output directory, and verify image and label SHA values file by file; do not load selection images or labels.'
        'Do not access test-dev. For training checkpoints, use the built-in, unknown-ignored mAP50-95 metric; preserve the later epoch with the same score.'
        'Positive labels during training exclude ignored items, but the standard YOLO loss does not suppress negative sample loss in ignored regions.\n\n'
        'The original environment is not modified. A custom trainer terminates when the batch deviates by 4 or when loss/fitness becomes non-finite;'
        'Before each save, also verify the full state of live and EMA, and ensure that EMA converted to fp16 remains finite, to prevent silent re-synchronization or weight interruption by the framework;'
        'Preserve failure evidence and do not automatically restart. The original OOM halving and NaN recovery code from the framework is saved in framework_sources;'
        'Specific line numbers are listed in implementation_evidence.json. The additional on_fit_epoch_end callback in final_eval is separately recorded as the final best weights validation;'
        'Do not count as the 31st training epoch.\n\n'
        'Completion is defined only as the generation of training candidates. accepted_float_baseline_ready remains false; preprocessing, bounding box restoration, ignore-aware hits,'
        'floating-point qualification checks for FP32/FP16 are conducted separately. The built-in mAP is not the official VisDrone AP; the formal quantization research contract has not yet been frozen.\n',
        encoding="utf-8")
    manifest = {str(path.relative_to(output)).replace("\\", "/"): {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "manifest.json", {"files": manifest})
    return contract


def verify_contract(contract_dir: Path) -> dict:
    verify_manifest(contract_dir)
    contract = json.loads((contract_dir / "contract.json").read_text(encoding="utf-8"))
    require(contract["status"] == "FROZEN_NOT_STARTED" and contract["training_started"] is False, "Contract is not frozen before training")
    require(contract["runner_sha256"] == sha(Path(__file__).resolve()), "Runner changed after freeze")
    require(contract["fixed_args"] == FIXED_ARGS, "Fixed training parameters changed")
    return contract


def materialize_dataset(rows: list[dict], prepared: Path, output: Path) -> dict:
    dataset = output / "dataset"
    copied = []
    for role, target in (("train", "train"), ("training_val", "val")):
        paths = []
        for row in rows:
            if row["assigned_split"] != role:
                continue
            image = safe_file(prepared, row["image_path"])
            label = safe_file(prepared, row["label_path"])
            require(sha(image) == row["image_sha256"] and sha(label) == row["label_sha256"], f"Source bytes changed before copying: {row['image_id']}")
            image_dest = dataset / "images" / target / image.name
            label_dest = dataset / "labels" / target / f"{image.stem}.txt"
            require(not image_dest.exists() and not label_dest.exists(), "Destination name collision")
            image_dest.parent.mkdir(parents=True, exist_ok=True)
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, image_dest)
            shutil.copyfile(label, label_dest)
            require(sha(image_dest) == row["image_sha256"] and sha(label_dest) == row["label_sha256"], "Copied dataset integrity failure")
            paths.append(str(image_dest))
            copied.append({"image_id": row["image_id"], "role": role, "image_sha256": sha(image_dest), "label_sha256": sha(label_dest)})
        (output / f"{role}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    data = {"path": str(dataset), "train": str(output / "train.txt"), "val": str(output / "training_val.txt"), "names": NAMES, "nc": 10}
    write_json(output / "data.yaml", data)
    write_json(output / "copied_dataset_manifest.json", {"files": copied, "selection_or_holdout_copied": False})
    return data


def enforce_batch(batch: int, events: list[dict], epoch: int | None) -> None:
    if batch != FIXED_ARGS["batch"]:
        events.append({"kind": "BATCH_CHANGE_ABORT", "epoch_zero_based": epoch, "effective_batch": batch, "expected_batch": FIXED_ARGS["batch"], "at_utc": now()})
        raise RuntimeError("Framework attempted to change frozen batch; this attempt is invalid and will not restart")


def enforce_finite(loss_finite: bool, fitness_finite: bool, events: list[dict], epoch: int | None) -> None:
    if not loss_finite or not fitness_finite:
        events.append({"kind": "NONFINITE_ABORT", "epoch_zero_based": epoch, "loss_finite": loss_finite, "fitness_finite": fitness_finite, "at_utc": now()})
        raise RuntimeError("Nonfinite loss/fitness; default framework recovery is forbidden for this frozen attempt")


def append_epoch_event(receipt: dict, event: dict, epochs: int = 30) -> None:
    """The installed final_eval deliberately emits an extra fit-epoch callback."""
    epoch = event["epoch_zero_based"]
    require(epoch >= 0, "Negative training epoch")
    if epoch >= epochs:
        receipt.setdefault("final_best_checkpoint_validation", []).append(event)
    else:
        require(all(item["epoch_zero_based"] != epoch for item in receipt["epoch_history"]), "Duplicate training-epoch event")
        receipt["epoch_history"].append(event)


def enforce_checkpoint_state(live_state: dict, ema_state: dict, events: list[dict], epoch: int) -> dict:
    """Reject save_model repairs before the framework can mutate/serialize EMA."""
    import torch

    counts = {"live_tensors": 0, "ema_tensors": 0, "ema_fp16_checks": 0}
    for scope, state in (("live", live_state), ("ema", ema_state)):
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            counts[scope + "_tensors"] += 1
            reason = None
            if not bool(torch.isfinite(tensor).all()):
                reason = "NONFINITE_" + scope.upper() + "_STATE"
            elif scope == "ema" and tensor.is_floating_point():
                counts["ema_fp16_checks"] += 1
                if not bool(torch.isfinite(tensor.detach().half()).all()):
                    reason = "EMA_FP16_SERIALIZATION_OVERFLOW"
            if reason is not None:
                events.append({"kind": "CHECKPOINT_STATE_ABORT", "reason": reason,
                               "state_key": key, "epoch_zero_based": epoch, "at_utc": now()})
                raise RuntimeError(f"Checkpoint state violates frozen contract: {reason}: {key}")
    return counts


def finite_loss_tree(value) -> bool:
    """The installed criterion returns named loss dictionaries, not just tensors."""
    import torch

    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_loss_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_loss_tree(item) for item in value)
    if isinstance(value, numbers.Real):
        return math.isfinite(value)
    raise TypeError(f"Unsupported loss value type: {type(value).__name__}")


def train_once(contract_dir: Path, output: Path, receipt: dict) -> None:
    contract = verify_contract(contract_dir)
    weights = Path(contract["weights"]["path"]).resolve(strict=True)
    require(sha(weights) == WEIGHT_SHA256, "Starting weights changed after freeze")
    plan_dir = Path(contract["split_plan"]).resolve(strict=True)
    plan, rows, audit = validate_plan(plan_dir, verify_source_files=True)
    require(audit["plan_manifest_sha256"] == contract["split_plan_manifest_sha256"], "Plan changed after freeze")
    require(audit["split_assignment_sha256"] == contract["split_assignment_sha256"], "Assignments changed after freeze")
    isolate_settings(output)
    env, defaults, _ = runtime_snapshot()
    require(environment_fingerprint(env) == contract["environment_fingerprint"], "Runtime changed after contract freeze")
    require(defaults == json.loads((contract_dir / "framework_defaults.json").read_text(encoding="utf-8")), "Framework defaults changed")
    require(env["cuda_available"], "CUDA unavailable")
    disable_integrations(output)
    write_json(output / "environment.json", env)
    write_json(output / "source_validation.json", audit)
    shutil.copyfile(Path(__file__).resolve(), output / "runner_source.py")
    shutil.copyfile(contract_dir / "contract.json", output / "frozen_contract.json")
    receipt.update({"phase": "COPYING_ISOLATED_DATASET", "contract_sha256": sha(contract_dir / "contract.json"), "contract_manifest_sha256": sha(contract_dir / "manifest.json"), "counts": audit["counts"], "selection_detector_inference": False, "holdout_accessed": False, "recovery_events": [], "epoch_history": [], "checkpoint_history": []})
    write_json(output / "attempt_receipt.json", receipt)
    materialize_dataset(rows, Path(plan["source_prepared"]), output)
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    def flush():
        write_json(output / "attempt_receipt.json", receipt)

    class ContractTrainer(DetectionTrainer):
        def _build_train_pipeline(self):
            try:
                enforce_batch(int(self.batch_size), receipt["recovery_events"], getattr(self, "epoch", None))
            finally:
                flush()
            return super()._build_train_pipeline()

        def _handle_nan_recovery(self, epoch):
            loss_ok = self.loss is None or bool(self.loss.isfinite())
            fitness_ok = self.fitness is None or math.isfinite(float(self.fitness))
            try:
                enforce_finite(loss_ok, fitness_ok, receipt["recovery_events"], epoch)
            finally:
                flush()
            return False

        def save_model(self):
            try:
                checked = enforce_checkpoint_state(unwrap_model(self.model).state_dict(), self.ema.ema.state_dict(),
                                                   receipt["recovery_events"], int(self.epoch))
                receipt.setdefault("checkpoint_state_checks", []).append({"epoch_zero_based": int(self.epoch), **checked})
            finally:
                flush()
            return super().save_model()

    def on_train_start(trainer):
        enforce_batch(int(trainer.batch_size), receipt["recovery_events"], getattr(trainer, "epoch", None))
        actual = jsonable(vars(trainer.args))
        for key, value in FIXED_ARGS.items():
            # Device selection normalizes the string 0 to cuda:0 in some versions.
            if key != "device":
                require(actual.get(key) == value, f"Effective parameter drift: {key}: {actual.get(key)!r} != {value!r}")
        write_json(output / "effective_training_args.json", actual)
        receipt.update({"status": "RUNNING_FROZEN_FLOAT_TRAINING", "phase": "TRAINING", "training_started": True,
                        "effective_batch": int(trainer.batch_size), "effective_device": str(trainer.device), "effective_amp": bool(trainer.amp),
                        "train_dataset_images": len(trainer.train_loader.dataset), "validation_dataset_images": len(trainer.test_loader.dataset),
                        "effective_training_args_sha256": sha(output / "effective_training_args.json"), "save_dir": str(trainer.save_dir)})
        require(len(trainer.train_loader.dataset) == audit["counts"]["train"] and len(trainer.test_loader.dataset) == audit["counts"]["training_val"], "Framework silently removed dataset images")
        require(str(trainer.device) == "cuda:0" and not trainer.amp, "Frozen GPU/AMP policy changed")
        flush()

    def on_fit_epoch_end(trainer):
        enforce_batch(int(trainer.batch_size), receipt["recovery_events"], trainer.epoch)
        enforce_finite(trainer.loss is None or bool(trainer.loss.isfinite()), trainer.fitness is None or math.isfinite(float(trainer.fitness)), receipt["recovery_events"], trainer.epoch)
        event = {"epoch_zero_based": int(trainer.epoch), "epoch_one_based": int(trainer.epoch) + 1,
                 "fitness": float(trainer.fitness), "best_fitness": float(trainer.best_fitness), "batch": int(trainer.batch_size),
                 "metrics": jsonable(trainer.metrics), "at_utc": now()}
        if int(trainer.epoch) >= FIXED_ARGS["epochs"]:
            # final_eval replaces metrics but leaves fitness from the last training epoch.
            event["last_training_fitness"] = event.pop("fitness")
            event["best_training_fitness"] = event.pop("best_fitness")
            event["validation_checkpoint"] = "best.pt"
            event["source"] = "framework_final_eval_extra_callback"
        append_epoch_event(receipt, event, FIXED_ARGS["epochs"])
        flush()

    def on_train_batch_end(trainer):
        observed = [getattr(trainer, name, None) for name in ("loss", "loss_items", "tloss")]
        try:
            enforce_finite(all(finite_loss_tree(value) for value in observed),
                           True, receipt["recovery_events"], int(trainer.epoch))
        finally:
            if receipt["recovery_events"]:
                flush()

    def on_model_save(trainer):
        receipt["checkpoint_history"].append({"epoch_zero_based": int(trainer.epoch), "best_saved_this_epoch": bool(trainer.best_fitness == trainer.fitness),
            "fitness": float(trainer.fitness), "best_fitness": float(trainer.best_fitness), "at_utc": now()})
        flush()

    model = YOLO(str(weights))
    for event, callback in (("on_train_start", on_train_start), ("on_train_batch_end", on_train_batch_end), ("on_fit_epoch_end", on_fit_epoch_end), ("on_model_save", on_model_save)):
        model.add_callback(event, callback)
    receipt.update({"phase": "STARTING_FRAMEWORK", "external_restarts": 0})
    flush()
    kwargs = dict(defaults)
    kwargs.update(FIXED_ARGS)
    kwargs.update({"model": str(weights), "data": str(output / "data.yaml"), "project": str(output), "name": "run"})
    # Every default is passed explicitly, preventing hidden checkpoint overrides.
    write_json(output / "requested_full_training_args.json", kwargs)
    result = model.train(trainer=ContractTrainer, **kwargs)
    trainer = model.trainer
    require(not receipt["recovery_events"], "Recovery guard triggered")
    require(len(receipt["epoch_history"]) == FIXED_ARGS["epochs"], "Training did not complete exactly 30 epochs")
    require(len(model.names) == 10, "Checkpoint class count changed")
    require(all(bool(torch.isfinite(p).all()) for p in model.model.parameters()), "Nonfinite final selected checkpoint parameters")
    checkpoints = {}
    saved = receipt["checkpoint_history"]
    for name in ("best.pt", "last.pt"):
        path = Path(trainer.save_dir) / "weights" / name
        require(path.is_file(), f"Missing checkpoint: {path}")
        source_events = saved if name == "last.pt" else [event for event in saved if event["best_saved_this_epoch"]]
        require(bool(source_events), "No checkpoint provenance event")
        checkpoints[name] = file_identity(path) | {"training_epoch_zero_based": source_events[-1]["epoch_zero_based"], "training_epoch_one_based": source_events[-1]["epoch_zero_based"] + 1,
            "epoch_provenance": "on_model_save callback before framework final stripping; final serialized epoch may be -1"}
    receipt.update({"status": "COMPLETE_FLOAT_TRAINING_REQUIRES_SCIENTIFIC_REVIEW", "phase": "COMPLETE", "checkpoints": checkpoints,
        "trained_model_parameters_finite": True, "effective_batch": int(trainer.batch_size), "checkpoint_classes": model.names,
        "diagnostic_builtin_metrics": jsonable(result.results_dict), "accepted_scientific_baseline": False,
        "accepted_float_baseline_ready": False, "official_visdrone_ap": False, "selection_detector_inference": False, "holdout_accessed": False})
    flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "dry-validate", "train"))
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not args.output.is_symlink() and not output.exists(), f"Refusing existing output: {output}")
    protected = [path.resolve() for path in (args.split_plan, args.contract) if path is not None]
    if args.split_plan is not None:
        plan_input = json.loads((args.split_plan / "split_plan.json").read_text(encoding="utf-8"))
        protected.append(Path(plan_input["source_prepared"]).resolve())
    elif args.contract is not None:
        contract_input = json.loads((args.contract / "contract.json").read_text(encoding="utf-8"))
        plan_input = json.loads((Path(contract_input["split_plan"]) / "split_plan.json").read_text(encoding="utf-8"))
        protected.append(Path(plan_input["source_prepared"]).resolve())
    require(not any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected), "Output overlaps frozen inputs")
    # The only output tree is created once. All later errors receive a durable receipt.
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "rampq.gate1.float_training_attempt/v1", "mode": args.mode, "status": "VALIDATING", "phase": "INPUT_CHECKS",
               "started_at_utc": now(), "training_started": False, "runner_sha256": sha(Path(__file__).resolve()), "limits": LIMITS}
    started = time.monotonic()
    try:
        if args.mode == "freeze":
            require(args.split_plan is not None and args.weights is not None and args.contract is None, "Freeze requires --split-plan and --weights only")
            contract = freeze_contract(args.split_plan.resolve(strict=True), args.weights.resolve(strict=True), output)
            receipt.update({"status": contract["status"], "phase": "COMPLETE", "contract_sha256": sha(output / "contract.json")})
        elif args.mode == "dry-validate":
            require(args.contract is not None and args.split_plan is None and args.weights is None, "Dry validation requires --contract only")
            contract = verify_contract(args.contract.resolve(strict=True))
            plan, rows, audit = validate_plan(Path(contract["split_plan"]), verify_source_files=True)
            require(audit["plan_manifest_sha256"] == contract["split_plan_manifest_sha256"], "Plan changed after freeze")
            require(sha(Path(contract["weights"]["path"])) == WEIGHT_SHA256, "Weights changed after freeze")
            write_json(output / "plan_validation.json", audit)
            receipt.update({"status": "PASS_DRY_VALIDATION_NO_COPY_NO_TRAINING", "phase": "COMPLETE", "validated_counts": audit["counts"], "training_pixels_opened_for_hashing_only": True,
                            "selection_pixels_or_labels_opened": False, "holdout_accessed": False, "dataset_copied": False, "contract_sha256": sha(args.contract / "contract.json")})
        else:
            require(args.contract is not None and args.split_plan is None and args.weights is None, "Training requires --contract only")
            train_once(args.contract.resolve(strict=True), output, receipt)
    except BaseException as exc:
        receipt.update({"status": "FAILED_PRESERVED_NO_RESTART", "error_type": type(exc).__name__, "error": str(exc)})
        (output / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["finished_at_utc"] = now()
        write_json(output / "attempt_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "output": str(output), "training_started": receipt["training_started"], "elapsed_seconds": receipt["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
