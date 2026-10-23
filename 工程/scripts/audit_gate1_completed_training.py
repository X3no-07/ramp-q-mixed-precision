"""Independently audit a finished Gate1 candidate run; no detector inference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys


NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
COMPLETE = "COMPLETE_FLOAT_TRAINING_REQUIRES_SCIENTIFIC_REVIEW"


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(item) for item in value)
    return not isinstance(value, (float, int)) or math.isfinite(value)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def member(root: Path, name: str) -> Path:
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest member")
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "Manifest member is outside its root")
    return path


def verify_manifest(root: Path) -> int:
    manifest = read(root / "manifest.json")
    for name, expected in manifest["files"].items():
        path = member(root, name)
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], f"Manifest mismatch: {name}")
    return len(manifest["files"])


def validate_completed_receipt(receipt: dict) -> dict:
    require(receipt["status"] == COMPLETE and receipt["phase"] == "COMPLETE", "Training is not complete")
    require(receipt["training_started"] is True, "Training did not start")
    require(receipt["effective_batch"] == 4 and receipt["effective_amp"] is False, "Frozen batch/AMP changed")
    require(receipt["train_dataset_images"] == 4534 and receipt["validation_dataset_images"] == 414, "Dataset coverage changed")
    require(receipt["recovery_events"] == [] and receipt["external_restarts"] == 0, "Recovery/restart invalidates this attempt")
    require(receipt["selection_detector_inference"] is False and receipt["holdout_accessed"] is False, "Prohibited data access declared")
    history = receipt["epoch_history"]
    require([item["epoch_zero_based"] for item in history] == list(range(30)), "Exactly 30 ordered training epochs required")
    best = -math.inf
    best_epoch = None
    best_flags = []
    for epoch, event in enumerate(history):
        require(event["epoch_one_based"] == epoch + 1 and event["batch"] == 4 and finite(event), "Invalid epoch history")
        fitness = event["fitness"]
        require(fitness == event["metrics"]["metrics/mAP50-95(B)"], "Fitness is not the declared builtin mAP proxy")
        if fitness >= best:
            best_epoch = epoch
        best = max(best, fitness)
        require(event["best_fitness"] == best, "Best-fitness history mismatch")
        best_flags.append(fitness == best)
    saves = receipt["checkpoint_history"]
    checks = receipt["checkpoint_state_checks"]
    require([event["epoch_zero_based"] for event in saves] == list(range(30)), "Checkpoint save history incomplete")
    require([event["epoch_zero_based"] for event in checks] == list(range(30)), "Pre-save state checks incomplete")
    for epoch, (save, check) in enumerate(zip(saves, checks)):
        require(save["best_saved_this_epoch"] is best_flags[epoch], "Best checkpoint tie policy mismatch")
        require(save["fitness"] == history[epoch]["fitness"] and save["best_fitness"] == history[epoch]["best_fitness"], "Save provenance mismatch")
        require(check["live_tensors"] > 0 and check["ema_tensors"] > 0 and check["ema_fp16_checks"] > 0, "Missing tensor checks")
    final = receipt["final_best_checkpoint_validation"]
    require(len(final) == 1 and final[0]["epoch_zero_based"] == 30, "Final best validation must be separate from training epochs")
    require(final[0]["source"] == "framework_final_eval_extra_callback" and final[0]["validation_checkpoint"] == "best.pt", "Final validation source mismatch")
    require("fitness" not in final[0] and final[0]["last_training_fitness"] == history[-1]["fitness"], "Stale final fitness mislabeled")
    require(final[0]["best_training_fitness"] == best and finite(final), "Invalid final validation")
    for name, epoch in (("best.pt", best_epoch), ("last.pt", 29)):
        identity = receipt["checkpoints"][name]
        require(identity["training_epoch_zero_based"] == epoch and identity["training_epoch_one_based"] == epoch + 1, "Checkpoint epoch provenance mismatch")
    require(receipt["accepted_scientific_baseline"] is False and receipt["accepted_float_baseline_ready"] is False and receipt["official_visdrone_ap"] is False, "Training-only result overclaims acceptance")
    return {"epochs": 30, "selected_best_epoch_zero_based": best_epoch, "selected_best_builtin_fitness": best}


def validate_args(requested: dict, effective: dict, observed_objects: int, run_dir: Path) -> dict:
    expected = dict(requested)
    require(expected["max_det"] == 300, "Unknown starting max_det policy")
    expected["max_det"] = max(300, observed_objects)
    expected["save_dir"] = str(run_dir)
    require(effective == expected, "Effective arguments differ beyond source-defined max_det/save_dir")
    return {"requested_max_det": 300, "effective_max_det": expected["max_det"], "observed_max_objects": observed_objects,
            "reason": "Pinned DetectionValidator._check_max_det raises an unchanged default max_det to the observed train/training_val maximum"}


def audit(attempt: Path, contract_dir: Path, output: Path) -> dict:
    contract_members = verify_manifest(contract_dir)
    contract = read(contract_dir / "contract.json")
    receipt = read(attempt / "attempt_receipt.json")
    result = validate_completed_receipt(receipt)
    require(receipt["contract_sha256"] == sha(contract_dir / "contract.json"), "Wrong final training contract")
    require(receipt["contract_manifest_sha256"] == sha(contract_dir / "manifest.json"), "Wrong final contract manifest")
    require(sha(attempt / "runner_source.py") == contract["runner_sha256"] == receipt["runner_sha256"], "Runner identity drift")
    require(sha(attempt / "frozen_contract.json") == sha(contract_dir / "contract.json"), "Run contract copy drift")
    require(sha(Path(contract["weights"]["path"])) == contract["weights"]["sha256"], "Initial weight drift")
    for item in read(contract_dir / "environment.json")["framework_sources"].values():
        require(sha(Path(item["path"])) == item["sha256"], "Installed framework source changed since freeze")
    plan_dir = Path(contract["split_plan"]).resolve(strict=True)
    require(sha(plan_dir / "manifest.json") == contract["split_plan_manifest_sha256"], "Split identity drift")
    plan_members = verify_manifest(plan_dir)
    rows = [json.loads(line) for line in (plan_dir / "split_assignment.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_rows = {r["image_id"]: r for r in rows if r["assigned_split"] in {"train", "training_val"}}
    copied = read(attempt / "copied_dataset_manifest.json")
    require(copied["selection_or_holdout_copied"] is False, "Prohibited copied data")
    require(len(copied["files"]) == len(expected_rows) == 4948, "Copied dataset coverage mismatch")
    require(len({r["image_id"] for r in copied["files"]}) == 4948, "Duplicate copied image IDs")
    import numpy as np
    label_counts = Counter()
    duplicate_rows = []
    maximum = 0
    for record in copied["files"]:
        original = expected_rows[record["image_id"]]
        require(original["source_split"] == "train" and record["role"] == original["assigned_split"], "Copied role mismatch")
        require(record["image_sha256"] == original["image_sha256"] and record["label_sha256"] == original["label_sha256"], "Copied identity mismatch")
        role = "train" if record["role"] == "train" else "val"
        stem = record["image_id"].split("/")[1]
        image_name = Path(original["image_path"]).name
        image = member(attempt, f"dataset/images/{role}/{image_name}")
        label = member(attempt, f"dataset/labels/{role}/{stem}.txt")
        require(sha(image) == record["image_sha256"] and sha(label) == record["label_sha256"], "Copied dataset bytes changed")
        values = [[float(v) for v in line.split()] for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(all(len(line) == 5 and finite(line) for line in values), "Invalid label row")
        unique = np.unique(np.asarray(values, dtype=np.float32).reshape(-1, 5), axis=0)
        duplicates = len(values) - len(unique)
        label_counts[record["role"] + "_source_rows"] += len(values)
        label_counts[record["role"] + "_unique_loader_rows"] += len(unique)
        maximum = max(maximum, len(unique))
        if duplicates:
            duplicate_rows.append({"image_id": record["image_id"], "role": record["role"], "duplicates": duplicates})
    require(label_counts["train_source_rows"] == 227425 and label_counts["training_val_source_rows"] == 28007, "Original label counts changed")
    require(label_counts["train_unique_loader_rows"] == 227422 and label_counts["training_val_unique_loader_rows"] == 28007, "Loader deduplication differs from recorded startup")
    require(maximum == 902, "Observed maximum-object policy changed")
    run_dir = attempt / "run"
    requested = read(attempt / "requested_full_training_args.json")
    effective = read(attempt / "effective_training_args.json")
    for key, value in contract["fixed_args"].items():
        require(requested[key] == value and effective[key] == value, f"Frozen parameter changed: {key}")
    argument_policy = validate_args(requested, effective, maximum, run_dir)
    require(sha(attempt / "effective_training_args.json") == receipt["effective_training_args_sha256"], "Effective parameter receipt drift")
    csv_rows = list(csv.DictReader(io.StringIO((run_dir / "results.csv").read_text(encoding="utf-8"))))
    require(len(csv_rows) == 30, "Result CSV epoch count mismatch")
    for epoch, row in enumerate(csv_rows):
        require(int(float(row["epoch"])) == epoch + 1, "Result CSV epoch order mismatch")
        require(all(math.isfinite(float(value)) for value in row.values()), "Nonfinite CSV metric")
        require(float(row["metrics/mAP50-95(B)"]) == receipt["epoch_history"][epoch]["fitness"], "CSV/receipt fitness mismatch")
    config = output / "ultralytics_config"
    config.mkdir(exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.pop("PYTHONPATH", None)
    sys.dont_write_bytecode = True
    import torch
    checkpoints = {}
    for name in ("best.pt", "last.pt"):
        path = member(attempt, "run/weights/" + name)
        expected = receipt["checkpoints"][name]
        require(path == Path(expected["path"]).resolve(strict=True), "Checkpoint path outside expected run")
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], "Checkpoint identity mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = checkpoint.get("ema") if checkpoint.get("ema") is not None else checkpoint.get("model")
        require(model is not None, "No checkpoint model")
        names = model.names
        require([names[i] for i in range(10)] == NAMES and len(names) == 10, "Wrong checkpoint class mapping")
        tensors = [v for v in model.state_dict().values() if isinstance(v, torch.Tensor)]
        require(all(bool(torch.isfinite(v).all()) for v in tensors), "Nonfinite checkpoint state")
        require(checkpoint["epoch"] == -1, "Expected framework final optimizer stripping")
        checkpoints[name] = {**expected, "serialized_epoch": checkpoint["epoch"], "state_tensor_count": len(tensors),
                             "all_state_tensors_finite": True, "inference_executed": False}
    result.update({"status": "PASS_COMPLETED_TRAINING_CANDIDATE_AUDIT_ONLY", "contract_members": contract_members,
                   "plan_members": plan_members, "copied_images_and_labels_verified": 4948, "label_counts": dict(label_counts),
                   "loader_duplicate_rows": duplicate_rows, "effective_argument_policy": argument_policy, "checkpoints": checkpoints,
                   "accepted_float_baseline_ready": False, "gate1_scientific_pass": False, "official_AP": False,
                   "selection_pixels_or_labels_read": False, "holdout_accessed": False,
                   "limits": ["No detector inference or numeric equivalence check", "Builtin metrics are ignore-unaware training proxies", "Checkpoint FP16 EMA cannot reconstruct the pre-save FP32 training EMA"]})
    for name in ("attempt_receipt.json", "effective_training_args.json", "requested_full_training_args.json"):
        shutil.copyfile(attempt / name, output / name)
    shutil.copyfile(run_dir / "results.csv", output / "results.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    attempt = args.attempt.resolve(strict=True)
    contract = args.contract.resolve(strict=True)
    output = args.output.resolve()
    require(not output.exists() and not args.output.is_symlink(), "Refusing existing audit output")
    plan_dir = Path(read(contract / "contract.json")["split_plan"]).resolve(strict=True)
    prepared = Path(read(plan_dir / "split_plan.json")["source_prepared"]).resolve(strict=True)
    require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (attempt, contract, plan_dir, prepared)), "Audit output overlaps frozen inputs")
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "FAILED_OR_NOT_READY", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "auditor_sha256": sha(Path(__file__).resolve()), "attempt": str(attempt), "contract": str(contract)}
    try:
        result.update(audit(attempt, contract, output))
    except BaseException as error:
        result.update({"error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        write(output / "audit.json", result)
    print(json.dumps({k: result[k] for k in ("status", "epochs", "selected_best_epoch_zero_based")}), flush=True)


if __name__ == "__main__":
    main()
