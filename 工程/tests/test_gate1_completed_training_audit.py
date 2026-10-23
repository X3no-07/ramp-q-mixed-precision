"""Synthetic completion/provenance contracts; no images or trained models."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/audit_gate1_completed_training.py"
SPEC = importlib.util.spec_from_file_location("completed_audit", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def receipt():
    history = [{"epoch_zero_based": i, "epoch_one_based": i + 1, "batch": 4, "fitness": 0.5,
                "best_fitness": 0.5, "metrics": {"metrics/mAP50-95(B)": 0.5}} for i in range(30)]
    return {"status": M.COMPLETE, "phase": "COMPLETE", "training_started": True, "effective_batch": 4,
            "effective_amp": False, "train_dataset_images": 4534, "validation_dataset_images": 414,
            "recovery_events": [], "external_restarts": 0, "selection_detector_inference": False,
            "holdout_accessed": False, "accepted_scientific_baseline": False, "accepted_float_baseline_ready": False,
            "official_visdrone_ap": False, "epoch_history": history,
            "checkpoint_history": [{"epoch_zero_based": i, "best_saved_this_epoch": True, "fitness": 0.5, "best_fitness": 0.5} for i in range(30)],
            "checkpoint_state_checks": [{"epoch_zero_based": i, "live_tensors": 499, "ema_tensors": 499, "ema_fp16_checks": 418} for i in range(30)],
            "final_best_checkpoint_validation": [{"epoch_zero_based": 30, "source": "framework_final_eval_extra_callback",
                "validation_checkpoint": "best.pt", "last_training_fitness": 0.5, "best_training_fitness": 0.5}],
            "checkpoints": {name: {"training_epoch_zero_based": 29, "training_epoch_one_based": 30} for name in ["best.pt", "last.pt"]}}


class CompletionAudit(unittest.TestCase):
    def test_ties_choose_latest_epoch(self):
        self.assertEqual(M.validate_completed_receipt(receipt())["selected_best_epoch_zero_based"], 29)

    def test_running_or_partial_run_is_rejected(self):
        r = receipt(); r["status"] = "RUNNING_FROZEN_FLOAT_TRAINING"
        with self.assertRaisesRegex(ValueError, "not complete"):
            M.validate_completed_receipt(r)
        r = receipt(); r["epoch_history"].pop()
        with self.assertRaisesRegex(ValueError, "30 ordered"):
            M.validate_completed_receipt(r)

    def test_batch_recovery_and_holdout_are_not_accepted(self):
        for key, value in [("effective_batch", 2), ("recovery_events", [{"kind": "repair"}]), ("holdout_accessed", True)]:
            r = receipt(); r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.validate_completed_receipt(r)

    def test_checkpoint_epoch_and_final_callback_cannot_be_relabelled(self):
        r = receipt(); r["checkpoints"]["best.pt"]["training_epoch_zero_based"] = 0
        with self.assertRaisesRegex(ValueError, "provenance"):
            M.validate_completed_receipt(r)
        r = receipt(); r["final_best_checkpoint_validation"][0]["fitness"] = 0.5
        with self.assertRaisesRegex(ValueError, "Stale"):
            M.validate_completed_receipt(r)

    def test_nonfinite_history_is_rejected(self):
        r = receipt(); r["epoch_history"][0]["metrics"]["val/loss"] = float("nan")
        with self.assertRaises(ValueError):
            M.validate_completed_receipt(r)

    def test_only_source_defined_effective_default_change_is_allowed(self):
        requested = {"max_det": 300, "batch": 4}
        path = Path("example/run")
        effective = {"max_det": 902, "batch": 4, "save_dir": str(path)}
        self.assertEqual(M.validate_args(requested, effective, 902, path)["effective_max_det"], 902)
        bad = deepcopy(effective); bad["max_det"] = 1000
        with self.assertRaises(ValueError):
            M.validate_args(requested, bad, 902, path)
        bad = deepcopy(effective); bad["batch"] = 8
        with self.assertRaises(ValueError):
            M.validate_args(requested, bad, 902, path)


if __name__ == "__main__":
    unittest.main()
