"""Synthetic guards only: no real dataset/model imports or detector inference."""
import importlib.util
from pathlib import Path
import uuid
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "run_gate1_float_baseline.py"
SPEC = importlib.util.spec_from_file_location("gate1_float_runner", SOURCE)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def row(key, role, source="train", **changes):
    value = {"image_id": key, "assigned_split": role, "source_split": source,
             "is_content_canonical": True, "requires_annotation_quality_quarantine": False,
             "image_sha256": key, "prefix_proxy": key, "refined_component": key}
    value.update(changes)
    return value


class FloatContractSafety(unittest.TestCase):
    def test_only_training_roles_and_official_val_selection_are_allowed(self):
        RUNNER.validate_roles([row("a", "train"), row("b", "training_val"), row("c", "selection", source="val")])
        for bad in [row("x", "training_val", source="val"), row("x", "train", source="test-dev"), row("x", "selection")]:
            with self.assertRaises(ValueError):
                RUNNER.validate_roles([bad])

    def test_content_prefix_and_component_leakage_are_independent_guards(self):
        for field in ("image_sha256", "prefix_proxy", "refined_component"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                RUNNER.validate_roles([row("a", "train", **{field: "shared"}), row("b", "training_val", **{field: "shared"})])

    def test_quality_and_noncanonical_training_are_rejected(self):
        for changes in ({"requires_annotation_quality_quarantine": True}, {"is_content_canonical": False}):
            with self.assertRaises(ValueError):
                RUNNER.validate_roles([row("x", "train", **changes)])

    def test_batch_reduction_aborts_and_records_reason(self):
        events = []
        RUNNER.enforce_batch(4, events, 0)
        self.assertEqual(events, [])
        with self.assertRaises(RuntimeError):
            RUNNER.enforce_batch(2, events, 0)
        self.assertEqual(events[0]["kind"], "BATCH_CHANGE_ABORT")
        self.assertEqual(events[0]["effective_batch"], 2)

    def test_nonfinite_first_epoch_also_aborts_without_recovery(self):
        for loss_ok, fitness_ok in ((False, True), (True, False), (False, False)):
            events = []
            with self.assertRaises(RuntimeError):
                RUNNER.enforce_finite(loss_ok, fitness_ok, events, 0)
            self.assertEqual(events[0]["kind"], "NONFINITE_ABORT")

    def test_safefile_rejects_traversal_and_outside_file(self):
        root = SOURCE.parents[1] / "artifacts" / "gate1" / "20261022_split_refinement" / "training_contract_tests" / ("fixture_" + uuid.uuid4().hex)
        root.mkdir(parents=True, exist_ok=False)
        (root / "ok.txt").write_text("fixture", encoding="utf-8")
        self.assertEqual(RUNNER.safe_file(root, "ok.txt"), root / "ok.txt")
        for unsafe in ("../outside", str((root / "ok.txt").resolve())):
            with self.assertRaises(ValueError):
                RUNNER.safe_file(root, unsafe)

    def test_duplicate_ids_and_unknown_roles_fail(self):
        with self.assertRaises(ValueError):
            RUNNER.validate_roles([row("a", "train"), row("a", "train")])
        with self.assertRaises(ValueError):
            RUNNER.validate_roles([row("a", "holdout")])

    def test_frozen_budget_is_not_exposed_as_cli_tuning_options(self):
        self.assertEqual(RUNNER.FIXED_ARGS["epochs"], 30)
        self.assertEqual(RUNNER.FIXED_ARGS["batch"], 4)
        self.assertFalse(RUNNER.FIXED_ARGS["amp"])
        self.assertEqual(RUNNER.FIXED_ARGS["close_mosaic"], 10)

    def test_final_eval_callback_does_not_become_epoch31(self):
        receipt = {"epoch_history": []}
        for epoch in range(31):
            RUNNER.append_epoch_event(receipt, {"epoch_zero_based": epoch})
        self.assertEqual(len(receipt["epoch_history"]), 30)
        self.assertEqual(receipt["final_best_checkpoint_validation"], [{"epoch_zero_based": 30}])
        with self.assertRaises(ValueError):
            RUNNER.append_epoch_event(receipt, {"epoch_zero_based": 29})

    def test_checkpoint_guard_checks_buffers_as_well_as_parameters(self):
        import torch
        for scope in ("live", "ema"):
            live = {"weight": torch.ones(2), "running_mean": torch.zeros(2)}
            ema = {key: value.clone() for key, value in live.items()}
            (live if scope == "live" else ema)["running_mean"][0] = float("nan")
            events = []
            with self.assertRaises(RuntimeError):
                RUNNER.enforce_checkpoint_state(live, ema, events, 2)
            self.assertEqual(events[0]["reason"], "NONFINITE_" + scope.upper() + "_STATE")
            self.assertEqual(events[0]["state_key"], "running_mean")

    def test_checkpoint_guard_rejects_fp16_overflow_without_clamping(self):
        import torch
        live = {"weight": torch.ones(2)}
        ema = {"weight": torch.tensor([1.0, 70000.0])}
        events = []
        with self.assertRaises(RuntimeError):
            RUNNER.enforce_checkpoint_state(live, ema, events, 0)
        self.assertEqual(events[0]["reason"], "EMA_FP16_SERIALIZATION_OVERFLOW")
        self.assertEqual(ema["weight"][1].item(), 70000.0)

    def test_checkpoint_guard_accepts_finite_state_and_integer_buffers(self):
        import torch
        state = {"weight": torch.tensor([1.0, -65504.0]), "num_batches_tracked": torch.tensor(4)}
        events = []
        checked = RUNNER.enforce_checkpoint_state(state, state, events, 0)
        self.assertEqual(checked, {"live_tensors": 2, "ema_tensors": 2, "ema_fp16_checks": 1})
        self.assertEqual(events, [])

    def test_named_loss_dictionary_and_nested_values_are_checked(self):
        import torch
        loss = {"box_loss": torch.tensor(1.5), "cls_loss": torch.tensor(3.0), "dfl_loss": torch.tensor(1.0)}
        self.assertTrue(RUNNER.finite_loss_tree(loss))
        self.assertTrue(RUNNER.finite_loss_tree([None, 2.0, (torch.tensor([1.0, 2.0]), loss)]))
        self.assertFalse(RUNNER.finite_loss_tree({"box_loss": torch.tensor(1.0), "cls_loss": torch.tensor(float("nan"))}))
        self.assertFalse(RUNNER.finite_loss_tree([loss, {"hidden": float("inf")}]))
        with self.assertRaises(TypeError):
            RUNNER.finite_loss_tree({"unknown": "not a loss"})


if __name__ == "__main__":
    unittest.main()
