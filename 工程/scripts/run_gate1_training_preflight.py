"""One-epoch engineering check on train-derived debug/validation data only.

This does not produce an accepted VisDrone baseline or evaluate Selection/Holdout.
Use the existing environment; all artifacts and settings stay in the output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone


NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    prepared = args.prepared.resolve(strict=True)
    weights = args.weights.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if args.batch < 1 or args.batch > 8:
        raise ValueError("Preflight batch must be between 1 and 8")

    manifest_path = prepared / "image_manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records_by_path = {(prepared / row["image_path"]).resolve(): row for row in manifest}
    if len(records_by_path) != len(manifest):
        raise ValueError("Duplicate image paths in prepared manifest")
    verified_sources = []

    def image_list(split: str, limit: int) -> list[Path]:
        paths = [Path(line).resolve(strict=True) for line in (prepared / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = sorted(paths, key=lambda p: p.name)[:limit]
        if not paths:
            raise ValueError(f"Empty {split}")
        for path in paths:
            if not path.is_relative_to(prepared / "images" / "train"):
                raise ValueError(f"Only official-train-derived images permitted: {path}")
            row = records_by_path.get(path)
            if row is None or row["source_split"] != "train" or row["assigned_split"] != split:
                raise ValueError(f"Manifest role does not match {split}: {path}")
            label = (prepared / row["label_path"]).resolve(strict=True)
            if not label.is_relative_to(prepared / "labels" / "train"):
                raise ValueError(f"Training label outside allowed directory: {label}")
            if sha256(path) != row["image_sha256"] or sha256(label) != row["label_sha256"]:
                raise ValueError(f"Prepared image or label changed: {path}")
            verified_sources.append({"image": str(path), "label": str(label), "role": split,
                                     "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"]})
        return paths

    train_paths = image_list("debug", 64)
    val_paths = image_list("training_val", 32)
    if set(train_paths) & set(val_paths):
        raise ValueError("Preflight train/validation overlap")
    train_hashes = {sha256(p) for p in train_paths}
    val_hashes = {sha256(p) for p in val_paths}
    if train_hashes & val_hashes:
        raise ValueError("Preflight train/validation duplicate content")
    train_groups = {p.name.split("_")[0] for p in train_paths}
    val_groups = {p.name.split("_")[0] for p in val_paths}
    if train_groups & val_groups:
        raise ValueError("Preflight prefix-group overlap")

    # Require membership in the declared lists, and exclude every other role by path.
    blocked = set()
    for split in ("train", "calibration_pool", "selection", "quarantine", "duplicate_excluded"):
        blocked.update(Path(line).resolve() for line in (prepared / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip())
    if blocked & (set(train_paths) | set(val_paths)):
        raise ValueError("Preflight data overlaps a prohibited data role")

    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(), output / "runner_source.py")
    # Isolate Ultralytics' automatic labels/*.cache and every other generated file.
    preflight_dataset = output / "dataset"
    copied_lists = {}
    for role, paths, destination_role in (("debug", train_paths, "train"), ("training_val", val_paths, "val")):
        copied = []
        for path in paths:
            row = records_by_path[path]
            image_dest = preflight_dataset / "images" / destination_role / path.name
            label_dest = preflight_dataset / "labels" / destination_role / f"{path.stem}.txt"
            image_dest.parent.mkdir(parents=True, exist_ok=True)
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, image_dest)
            shutil.copyfile(prepared / row["label_path"], label_dest)
            if sha256(image_dest) != row["image_sha256"] or sha256(label_dest) != row["label_sha256"]:
                raise ValueError(f"Preflight copy hash mismatch: {path}")
            copied.append(image_dest)
        copied_lists[role] = copied
    (output / "train_debug_subset.txt").write_text("\n".join(map(str, copied_lists["debug"])) + "\n", encoding="utf-8")
    (output / "training_val_subset.txt").write_text("\n".join(map(str, copied_lists["training_val"])) + "\n", encoding="utf-8")
    # JSON is valid YAML and avoids ambiguity for Windows paths and Unicode.
    data = {"path": str(preflight_dataset), "train": str(output / "train_debug_subset.txt"), "val": str(output / "training_val_subset.txt"), "names": NAMES, "nc": 10}
    write_json(output / "preflight_data.yaml", data)
    config_dir = output / "ultralytics_config"
    config_dir.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
    started = time.monotonic()
    contract = {
        "schema": "rampq.gate1.training_preflight/v1",
        "runner_sha256": sha256(Path(__file__).resolve()),
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One-epoch engineering check; not a final baseline, Selection result, official VisDrone AP, or Holdout result",
        "prepared_manifest_sha256": sha256(manifest_path),
        "prepared_manifest_rows": len(manifest),
        "verified_source_roles_and_hashes": verified_sources,
        "preflight_dataset_copy": str(preflight_dataset),
        "weights_sha256": sha256(weights),
        "training_list_sha256": sha256(output / "train_debug_subset.txt"),
        "validation_list_sha256": sha256(output / "training_val_subset.txt"),
        "train_images": len(train_paths),
        "validation_images": len(val_paths),
        "epochs": 1,
        "amp": False,
        "batch": args.batch,
        "imgsz": 640,
        "seed": 20261022,
        "workers": 0,
        "official_val_or_testdev_access": False,
        "ignore_policy": "Training labels exclude ignored annotations and retain sidecars. Built-in YOLO validation is ignore-unaware and is an engineering diagnostic only.",
        "external_automatic_retries": 0,
        "framework_recovery_policy": "Ultralytics default: first-epoch OOM may reduce batch and retry up to 3 times; framework NaN recovery is also possible. Preserve console and report effective batch; this is not a zero-retry experiment.",
    }
    write_json(output / "preflight_receipt.json", contract)
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO

        ultralytics.settings.update({"sync": False, "wandb": False, "comet": False, "clearml": False, "mlflow": False, "dvc": False, "raytune": False, "tensorboard": False, "runs_dir": str(output / "run"), "weights_dir": str(output / "weights")})
        contract.update({"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__, "cuda_available": torch.cuda.is_available(), "framework_trainer_source_sha256": sha256(Path(ultralytics.__file__).parent / "engine" / "trainer.py")})
        write_json(output / "preflight_receipt.json", contract)
        model = YOLO(str(weights))
        result = model.train(
            data=str(output / "preflight_data.yaml"),
            epochs=1, imgsz=640, batch=args.batch, device=args.device,
            workers=0, amp=False, cache=False, plots=False, val=True,
            project=str(output), name="run", exist_ok=False,
            seed=20261022, deterministic=True, optimizer="SGD",
            lr0=0.01, momentum=0.937, weight_decay=0.0005,
            patience=0, close_mosaic=0, save=True, verbose=True,
        )
        save_dir = Path(model.trainer.save_dir).resolve()
        checkpoints = {}
        for name in ("best.pt", "last.pt"):
            path = save_dir / "weights" / name
            if not path.is_file():
                raise RuntimeError(f"Missing checkpoint: {path}")
            checkpoints[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        if len(model.names) != 10:
            raise RuntimeError(f"Expected ten VisDrone classes, found {model.names}")
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.model.parameters()):
            raise RuntimeError("Non-finite trained model parameters")
        contract.update({"status": "PASS_ENGINEERING_TRAINING_PREFLIGHT_ONLY", "checkpoint_classes": model.names, "trained_source_conv_count": sum(isinstance(module, torch.nn.Conv2d) for module in model.model.modules()), "trained_model_parameters_finite": True, "effective_batch": int(model.trainer.batch_size), "save_dir": str(save_dir), "checkpoints": checkpoints, "diagnostic_metrics": {k: float(v) for k, v in result.results_dict.items()}})
    except BaseException as exc:
        contract.update({"status": "FAILED_NO_EXTERNAL_RETRY", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        contract["elapsed_seconds"] = time.monotonic() - started
        contract["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "preflight_receipt.json", contract)
    print(json.dumps({"status": contract["status"], "output": str(output), "elapsed_seconds": contract["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
