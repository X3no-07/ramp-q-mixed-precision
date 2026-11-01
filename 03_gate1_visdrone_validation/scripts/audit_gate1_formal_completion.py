"""Audit the fixed 30-epoch training; no task-data inference or task AP."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def validate_events(events, epochs=30, batches=1133, updates=2557):
    expected_epochs = list(range(1, epochs + 1))
    for event in ("epoch_started", "FP32_reference_committed", "framework_compatibility_checkpoint_saved", "epoch_completed_with_FP32_reference"):
        require([r["epoch"] for r in events if r["event"] == event] == expected_epochs, "Epoch event coverage mismatch: " + event)
    samples = [r for r in events if r["event"] == "training_batch"]
    require(len(samples) == epochs * batches, "Wrong actual batch count")
    previous = 0
    for epoch in expected_epochs:
        group = [r for r in samples if r["epoch"] == epoch]
        require([r["batch"] for r in group] == list(range(1, batches + 1)), "Batch order/coverage mismatch")
        for i, row in enumerate(group, 1):
            require(math.isfinite(row["loss"]) and row["mask"]["positive_rows_preserved"] is True, "Nonfinite loss or positive row loss")
            require(row["mask"]["batch_size"] == (2 if i == batches else 4), "Unexpected actual batch size")
            require(row["optimizer_steps"] == row["ema_updates"] and row["optimizer_steps"] >= previous, "Update counters diverged")
            previous = row["optimizer_steps"]
        positions = {event: next(i for i, r in enumerate(events) if r["event"] == event and r.get("epoch") == epoch)
                     for event in ("save_before_FP32_capture", "FP32_reference_committed", "framework_compatibility_checkpoint_saved", "epoch_completed_with_FP32_reference")}
        require(list(positions.values()) == sorted(positions.values()), "FP32 capture/half serialization ordering differs")
    require(previous == updates, "Final optimizer update count mismatch")
    return {"batches": len(samples), "optimizer_updates": previous,
            "batches_with_ignored_negatives": sum(r["mask"]["ignored_negative_anchors"] > 0 for r in samples),
            "all_positive_rows_preserved": True}


def normalize_csv(text):
    rows = list(csv.reader(text.splitlines()))
    require(len(rows) == 31 and rows[0] == ["epoch", "time", "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss", "lr/pg0", "lr/pg1", "lr/pg2"], "Unexpected training CSV header/count")
    result = []
    for epoch, fields in enumerate(rows[1:], 1):
        require(int(fields[0]) == epoch, "CSV epoch order differs")
        require(len(fields) == (8 if epoch == 30 else 11), "Unexpected CSV row length")
        values = [float(v) for v in fields]
        require(all(math.isfinite(v) for v in values), "Nonfinite training CSV")
        # The final validate() returned {}, so native CSV omitted three empty
        # validation placeholders. Never read its LR columns as validation loss.
        result.append({"epoch": epoch, "framework_time_seconds": values[1],
            "train_box_loss": values[2], "train_cls_loss": values[3], "train_dfl_loss": values[4],
            "lr": values[-3:], "task_validation_metrics": None, "raw_field_count": len(fields)})
    return result


def run(base, output):
    import numpy as np
    import torch
    from ultralytics.nn.tasks import DetectionModel
    import gate1_fp32_live_capture as C

    base, output = Path(base).resolve(strict=True), Path(output).resolve()
    require(not output.exists(), "New audit result directory required")
    output.mkdir(parents=True)
    attempt = base / "formal_attempt_01"
    contract = read(base / "formal_contract.json")
    completion = read(attempt / "completion_receipt.json")
    require(completion["status"] == "COMPLETE_FIXED_BUDGET_FP32_TRAINING_CANDIDATE_ONLY" and
            completion["completed_epochs"] == completion["captured_epochs"] == 30, "Training completion not established")
    contract_sha = sha(base / "formal_contract.json")
    require(contract_sha == completion["contract_sha256"] and completion["counts"] == {"train": 4530, "training_val": 414}, "Completion contract/counts differ")
    require(completion["optimizer_steps"] == completion["ema_updates"] == 2557 and completion["recovery_or_external_restart"] is False,
            "Update budget/recovery mismatch")
    require(not (attempt / "failure_receipt.json").exists() and (base / "formal_stderr.txt").stat().st_size == 0, "Training error output present")
    for binding in contract["source_bindings"].values():
        require(identity(binding["path"]) == binding, "Training source drift")
    require(C.sources() == contract["runtime_sources"], "Training framework source drift")
    pair, member_files = {}, 0
    for role in ("train", "training_val"):
        binding = contract["datasets"][role]
        path = Path(binding["manifest_path"])
        require(sha(path) == binding["manifest_sha256"], "Training dataset manifest changed")
        manifest = read(path)
        pair[role] = manifest
        root = (path.parent / manifest["data_root"]).resolve(strict=True)
        require(root.is_relative_to(path.parent) and manifest["split_role"] == role and manifest["data_role"] == "real_training_v1", "Dataset role/root mismatch")
        require(len(manifest["samples"]) == contract["counts"][role], "Dataset count changed")
        for sample in manifest["samples"]:
            require(sample["split_role"] == role and sample["image_id"].startswith("train/"), "Unexpected sample role")
            for key in ("image", "labels", "sidecar"):
                member = sample[key]
                source = (root / member["path"]).resolve(strict=True)
                require(source.is_relative_to(root) and not source.is_symlink(), "Dataset member escaped root")
                observed = identity(source)
                require(all(observed[k] == member[k] for k in ("bytes", "sha256")), "Training member changed")
                member_files += 1
    require({r["image_id"] for r in pair["train"]["samples"]}.isdisjoint(r["image_id"] for r in pair["training_val"]["samples"]), "Training role ID overlap")
    require({r["image"]["sha256"] for r in pair["train"]["samples"]}.isdisjoint(r["image"]["sha256"] for r in pair["training_val"]["samples"]), "Training role content overlap")
    # Validate source membership and component/prefix separation from preserved
    # metadata only. No Selection image, label, annotation or sidecar is opened.
    membership = read(base / "membership_decision.json")
    for key in ("source_assignment", "shared_source_sidecar"):
        require(identity(membership[key]["path"]) == membership[key], "Original source metadata changed")
    assignments = [json.loads(line) for line in Path(membership["source_assignment"]["path"]).read_text(encoding="utf-8").splitlines()]
    excluded = {r["image_id"] for r in membership["excluded_images"]}
    by_id = {r["image_id"]: r for r in assignments}
    for role, manifest in pair.items():
        require([r["image_id"] for r in manifest["samples"]] == [r["image_id"] for r in assignments if r["assigned_split"] == role and r["image_id"] not in excluded], "New role membership differs from policy")
        for r in manifest["samples"]:
            original = by_id[r["image_id"]]
            require(r["image"]["sha256"] == original["image_sha256"] and r["labels"]["sha256"] == original["label_sha256"], "New data differs from original member")
    for field in ("image_sha256", "prefix_proxy", "refined_component"):
        owners = {}
        for row in assignments:
            if row["assigned_split"] in {"train", "training_val", "calibration_pool", "debug", "selection"} and row["image_id"] not in excluded:
                owners.setdefault(row[field], set()).add(row["assigned_split"])
        require(all(len(v) == 1 for v in owners.values()), "Active split leakage")
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    event_check = validate_events(events)
    normalized = normalize_csv((attempt / "run/results.csv").read_text(encoding="utf-8"))
    require({p.name for p in (attempt / "fp32_ema").iterdir()} == {f"epoch_{e:04d}" for e in range(1, 31)}, "Missing or pending epoch capture")
    captures, updates, files = [], [], 0
    for epoch in range(1, 31):
        folder = attempt / "fp32_ema" / f"epoch_{epoch:04d}"
        for rel, expected in read(folder / "manifest.json")["files"].items():
            path = (folder / rel).resolve(strict=True)
            require(path.is_relative_to(folder), "Capture path escaped")
            observed = identity(path)
            require(all(observed[k] == expected[k] for k in ("bytes", "sha256")), "Capture file mismatch")
            files += 1
        receipt = read(folder / "capture_receipt.json")
        require(receipt["epoch_one_based"] == epoch and receipt["training_contract_sha256"] == contract_sha, "Capture epoch/contract mismatch")
        require(receipt["status"] == "PASS_BOUND_CUDA_FP32_CAPTURE_AND_EXACT_RECONSTRUCTION_ONLY" and receipt["source_already_FP32"] and
                receipt["state_and_update_counter_unchanged"] and receipt["training_torch_RNG_unchanged"], "Capture did not pass")
        require(receipt["cpu_reconstruction"]["output_values_dtypes_shapes_exact"] and
                receipt["cuda_reconstruction"]["output_values_shapes_dtypes_exact"], "Capture output check failed")
        state, metadata = C.FORMAT.load_snapshot(folder / "state")
        require(metadata["epoch_one_based"] == epoch and metadata["training_contract_sha256"] == contract_sha, "State metadata mismatch")
        require(C.FORMAT.describe_state(state) and all(v.dtype == torch.float32 for v in state.values() if v.is_floating_point()), "State not FP32")
        captures.append(identity(folder / "capture_receipt.json"))
        updates.append(receipt["ema_updates_before"])
        print(json.dumps({"capture_checked": epoch, "total": 30}), flush=True)
    require(all(a < b for a, b in zip(updates, updates[1:])) and updates[-1] == 2557, "Saved EMA update history mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    arch = read(folder / "architecture.json")
    require(hashlib.sha256(C.canonical(arch).encode()).hexdigest() == receipt["architecture_sha256"], "Final architecture digest mismatch")
    with torch.random.fork_rng(devices=[]):
        model = DetectionModel(deepcopy(arch["cfg"]), ch=3, nc=10, verbose=False)
    model.load_state_dict(state, strict=True)
    model.names = dict(enumerate(arch["names"]))
    model = model.float().eval()
    probe = C.deterministic_synthetic_input()
    with torch.inference_mode():
        require(C.describe_output(model(probe.clone())) == receipt["cpu_reconstruction"]["output"], "Independent final CPU replay differs")
        model = model.to("cuda:0")
        require(C.describe_output(model(probe.to("cuda:0"))) == receipt["cuda_reconstruction"]["output"], "Independent final CUDA replay differs")
    result = {"schema": "rampq.gate1.formal_training_completion_audit/v1", "status": "PASS_COMPLETED_FORMAL_TRAINING_AND_FINAL_FP32_RECONSTRUCTION_CANDIDATE_ONLY",
        "as_of_utc": datetime.now(timezone.utc).isoformat(), "auditor": identity(__file__), "training_contract": identity(base / "formal_contract.json"),
        "completion_receipt": identity(attempt / "completion_receipt.json"), "epochs": 30, "event_checks": event_check,
        "dataset_member_files_verified": member_files, "FP32_captures_verified": 30, "capture_manifest_files_verified": files,
        "final_state": identity(folder / "state/reference_state.pt"), "final_architecture": identity(folder / "architecture.json"),
        "final_capture_receipt": identity(folder / "capture_receipt.json"), "fresh_CPU_and_CUDA_synthetic_replay_exact": True,
        "member_order_matches_frozen_assignment_minus_four_flagged_images": True, "source_bindings_unchanged": True,
        "CSV_final_row_missing_three_unused_validation_placeholders": True, "original_CSV_modified": False,
        "validation_metrics_computed": False, "candidate_selection": contract["candidate_selection"],
        "inference_state_only_not_optimizer_resume_certification": True, "scientific_acceptance": False,
        "selection_pixels_or_scores_used": False, "holdout_opened_or_listed": False,
        "original_gate1_dtype_guard": "FAIL_FIXED_TOLERANCES", "old_gate05_numeric_guard": "FAILED_UNCHANGED"}
    for name, value in (("audit_result.json", result), ("normalized_training_history.json", normalized), ("capture_identities.json", captures)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.training, args.output)
