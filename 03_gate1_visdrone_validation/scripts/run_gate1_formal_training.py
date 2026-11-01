"""Run one immutable bound CUDA contract and persist progress/checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(contract_path):
    contract_path = Path(contract_path).resolve(strict=True)
    contract = read(contract_path)
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    attempt = Path(contract["attempt_directory"])
    require(not attempt.exists(), "Attempt output must be new; automatic resume/restart is forbidden")
    attempt.mkdir(parents=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", YOLO_CONFIG_DIR=str(attempt / "settings"), YOLO_OFFLINE="true",
        YOLO_VERBOSE="False", WANDB_MODE="disabled", COMET_MODE="DISABLED", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import torch
    import ultralytics
    from ultralytics.utils import callbacks
    import gate1_formal_training as F
    import gate1_fp32_live_capture as C

    started = time.monotonic()
    live = {"schema": "rampq.gate1.bound_cuda_training_live/v1", "status": "VERIFYING_FROZEN_INPUTS",
            "pid": os.getpid(), "contract": str(contract_path), "contract_sha256": contract_sha,
            "data_role": contract["data_role"], "planned_epochs": contract["epochs"], "completed_epochs": 0,
            "counts": contract["counts"], "batch": 4, "amp": False, "optimizer_steps": 0, "ema_updates": 0,
            "task_AP_computed": False, "gate1_scientific_pass": False, "holdout_accessed": False,
            "selection_inference": False, "recovery_or_external_restart": False, "started_at_utc": F.now()}

    def flush():
        live["updated_at_utc"] = F.now()
        live["elapsed_seconds"] = time.monotonic() - started
        F.atomic_json(attempt / "live_status.json", live)
        human = (f"Status:{live['status']}\n\n rounds:{live['completed_epochs']}/{live['planned_epochs']}\n\n"
                 f"Current batch:{live.get('batch_in_epoch', 0)}/{live.get('batches_per_epoch', 0)}\n\n"
                 f"Optimizer update:{live['optimizer_steps']}; EMA update:{live['ema_updates']}\n\n"
                 f"Training image:{live['counts']['train']}; Training validation image:{live['counts']['training_val']}\n\n"
                 f"Update time (UTC):{live['updated_at_utc']}\n\n task AP and scientific acceptance are handled separately. \n")
        temp = attempt / "progress.md.writing"
        temp.write_text(human, encoding="utf-8")
        temp.replace(attempt / "progress.md")

    def event(name, **fields):
        with (attempt / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at_utc": F.now(), "event": name, **fields}, ensure_ascii=False, allow_nan=False) + "\n")

    sleep_prevented = False
    try:
        flush()
        for expected in contract["source_bindings"].values():
            require(F.info(expected["path"]) == expected, "Bound training implementation or prerequisite changed")
        require(C.sources() == contract["runtime_sources"], "Pinned framework/runtime source identity changed")
        require(torch.cuda.is_available(), "CUDA unavailable")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        ultralytics.settings.update({**{k: False for k in ("sync", "wandb", "comet", "clearml", "mlflow", "dvc", "raytune", "tensorboard")},
                                     "runs_dir": str(attempt / "unused_runs"), "weights_dir": str(attempt / "unused_weights")})
        callbacks.add_integration_callbacks = lambda trainer: event("external_integration_registration_disabled")
        runtime = {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__),
                   "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu": torch.cuda.get_device_name(0), "ultralytics": ultralytics.__version__}
        F.atomic_json(attempt / "runtime.json", runtime)
        if contract["data_role"] == "real_training_v1":
            qbinding = contract["required_CUDA_qualification"]
            require(F.info(qbinding["path"]) == qbinding, "CUDA qualification receipt changed")
            qualification = read(qbinding["path"])
            require(qualification["status"] == "PASS_ACTUAL_BOUND_CUDA_TRAINER_AND_FP32_CAPTURE_ONLY" and
                    qualification["runtime"] == runtime and qualification["source_bindings"] == contract["source_bindings"] and
                    qualification["runtime_sources"] == contract["runtime_sources"], "CUDA qualification does not bind this implementation/runtime")
            ignored = {"epochs", "project", "name", "model", "data"}
            require({k: v for k, v in qualification["qualified_overrides"].items() if k not in ignored} ==
                    {k: v for k, v in contract["overrides"].items() if k not in ignored}, "Formal configuration differs from CUDA qualification")
            require(qualification["initialization_weights"] == contract["initialization_weights"], "Initialization differs from qualification")
        config = read(contract["model_cfg_path"])
        capture = F.CUDALiveFP32Capture(attempt / "fp32_ema", contract_sha, list(range(1, contract["epochs"] + 1)),
            expected_source_sha256=contract["runtime_sources"])
        trainer = F.BoundCUDATrainer(contract, config, event_callback=event, capture_callback=capture)
        torch.set_num_threads(1)  # Observe settings after framework device selection.
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        F.enforce_backend()
        require(all(f.__module__ == "ultralytics.utils.callbacks.base" for funcs in trainer.callbacks.values() for f in funcs),
                "Unexpected external integration callback")
        F.atomic_json(attempt / "base_callback_inventory.json", {k: [f.__module__ + "." + f.__qualname__ for f in funcs]
                      for k, funcs in trainer.callbacks.items()})
        batch_receipts = []

        def on_start(t):
            F.enforce_backend()
            require(str(t.device) == "cuda:0" and not t.amp and t.batch_size == 4, "Effective CUDA/AMP/batch mismatch")
            require(len(t.train_loader.dataset) == contract["counts"]["train"] and
                    len(t.test_loader.dataset) == contract["counts"]["training_val"], "Dataset silently lost members")
            for key, expected in contract["overrides"].items():
                if key != "device":
                    require(getattr(t.args, key) == expected, "Effective training option changed: " + key)
            live.update(status="RUNNING_FORMAL_IGNORE_AWARE_CUDA_TRAINING" if contract["data_role"] == "real_training_v1" else "RUNNING_CUDA_QUALIFICATION",
                        batches_per_epoch=len(t.train_loader), backend_settings=F.backend_settings(),
                        trainable_parameters=sum(p.numel() for p in t.model.parameters() if p.requires_grad))
            event("actual_training_started", effective_device=str(t.device), counts=contract["counts"], backend=F.backend_settings())
            flush()

        def on_epoch(t):
            t.train_loader.dataset.set_epoch(t.epoch)
            live.update(epoch_one_based=t.epoch + 1, batch_in_epoch=0)
            event("epoch_started", epoch=t.epoch + 1)
            flush()

        def on_batch(t):
            F.enforce_backend()
            require(bool(torch.isfinite(t.loss).all()), "Nonfinite batch loss; abort without recovery")
            receipt = dict(t.model.criterion.last_mask_receipt)
            require(receipt["positive_rows_preserved"], "Positive anchor classification row lost")
            live["batch_in_epoch"] += 1
            live.update(optimizer_steps=t.optimizer_steps_observed, ema_updates=t.ema.updates,
                        latest_loss=float(t.loss.detach().item()), latest_mask_receipt=receipt)
            if contract["data_role"] == "synthetic_cuda_qualification_v1":
                batch_receipts.append(receipt)
            event("training_batch", epoch=t.epoch + 1, batch=live["batch_in_epoch"], optimizer_steps=t.optimizer_steps_observed,
                  ema_updates=t.ema.updates, loss=live["latest_loss"], mask=receipt)
            if live["batch_in_epoch"] % 25 == 0 or live["batch_in_epoch"] == 1:
                flush()
            if (attempt / "stop_request.json").exists():
                raise RuntimeError("Explicit stop request observed; partial attempt preserved")

        def on_fit(t):
            live.update(completed_epochs=t.epoch + 1, optimizer_steps=t.optimizer_steps_observed, ema_updates=t.ema.updates,
                        latest_FP32_reference=str(attempt / "fp32_ema" / f"epoch_{t.epoch+1:04d}" / "state/reference_state.pt"))
            event("epoch_completed_with_FP32_reference", epoch=t.epoch + 1, optimizer_steps=t.optimizer_steps_observed,
                  ema_updates=t.ema.updates, backend=F.backend_settings())
            flush()
            print(json.dumps({"status": live["status"], "completed_epochs": live["completed_epochs"], "planned_epochs": live["planned_epochs"],
                              "ema_updates": t.ema.updates}, ensure_ascii=True), flush=True)

        for key, callback in (("on_train_start", on_start), ("on_train_epoch_start", on_epoch),
                              ("on_train_batch_end", on_batch), ("on_fit_epoch_end", on_fit)):
            trainer.add_callback(key, callback)
        if os.name == "nt":
            import ctypes
            sleep_prevented = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        event("actual_DetectionTrainer_train_call_enter", contract_sha256=contract_sha)
        trainer.train()
        require(live["completed_epochs"] == contract["epochs"] and len(capture.receipts) == contract["epochs"], "Fixed epoch/capture budget not completed")
        require(trainer.optimizer_steps_observed == trainer.ema.updates and trainer.ema.updates > 0, "Final optimizer/EMA count mismatch")
        for expected in contract["source_bindings"].values():
            require(F.info(expected["path"]) == expected, "Frozen implementation changed while training")
        live.update(status="COMPLETE_FIXED_BUDGET_FP32_TRAINING_CANDIDATE_ONLY", completed_at_utc=F.now(),
                    final_FP32_reference=live["latest_FP32_reference"], captured_epochs=len(capture.receipts))
        flush()
        if contract["data_role"] == "synthetic_cuda_qualification_v1":
            require(trainer.optimizer_steps_observed == 2 and len(batch_receipts) == 32,
                    "CUDA fixture must make two real optimizer steps across 32 accumulated batches")
            require(any(r["assigned_positive_anchors"] > 0 and r["ignored_negative_anchors"] > 0 for r in batch_receipts), "CUDA fixture failed to exercise both loss branches")
            F.atomic_json(attempt / "qualification_receipt.json", {
                "status": "PASS_ACTUAL_BOUND_CUDA_TRAINER_AND_FP32_CAPTURE_ONLY", "runtime": runtime,
                "source_bindings": contract["source_bindings"], "runtime_sources": contract["runtime_sources"],
                "qualified_overrides": contract["overrides"], "initialization_weights": contract["initialization_weights"],
                "contract_sha256": contract_sha, "epochs": 1, "optimizer_steps": 2, "ema_updates": trainer.ema.updates,
                "actual_batches": 32, "actual_warmup_iterations": trainer._get_warmup_iterations(len(trainer.train_loader)),
                "qualification_scope": "one_epoch_CUDA_accumulation16_training_and_FP32_capture_no_warmup_iterations",
                "mask_receipts": batch_receipts, "capture_receipt": F.info(attempt / "fp32_ema/epoch_0001/capture_receipt.json"),
                "scientific_acceptance": False, "real_training_data_used": False})
        F.atomic_json(attempt / "completion_receipt.json", {**live, "runtime": runtime,
            "ordinary_best_checkpoint_task_selected": False, "optimizer_resume_certified": False})
        print(json.dumps({"status": live["status"], "epochs": live["completed_epochs"], "final_reference": live["final_FP32_reference"]}, ensure_ascii=True))
    except BaseException as error:
        live.update(status="FAILED_BOUND_CUDA_TRAINING_ATTEMPT_PRESERVED", error_type=type(error).__name__, error=str(error))
        flush()
        F.atomic_json(attempt / "failure_receipt.json", {**live, "traceback": traceback.format_exc()})
        raise
    finally:
        if sleep_prevented:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    main(parser.parse_args().contract)
