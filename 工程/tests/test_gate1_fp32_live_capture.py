"""Synthetic CPU live EMA hook tests. No old weights, images, or network."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/gate1/20261023_fp32_live_capture_v1"
(BASE / "ultralytics_config").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(BASE / "ultralytics_config")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import torch
torch.set_num_threads(1)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA

SPEC = importlib.util.spec_from_file_location("fp32_live_capture", ROOT / "scripts/gate1_fp32_live_capture.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

CFG = {"nc": 2, "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [16, 3, 2]],
                               [-1, 1, "Conv", [32, 3, 2]]], "head": [[[0, 1, 2], 1, "Detect", ["nc"]]]}


class LiveCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_pins = M.sources()

    def setUp(self):
        (BASE / "test_cases").mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=BASE / "test_cases")
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "captures"
        torch.manual_seed(713)
        model = DetectionModel(cfg=deepcopy(CFG), ch=3, nc=2, verbose=False).cpu().float()
        model.names = {0: "synthetic_zero", 1: "synthetic_one"}
        ema = ModelEMA(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            loss = next(model.parameters()).square().mean()
            loss.backward()
            optimizer.step()
            ema.update(model)
        self.trainer = SimpleNamespace(args=SimpleNamespace(amp=False), amp=False, epoch=0, ema=ema)
        self.capture = M.LiveFP32Capture(self.output, "a" * 64, [1], expected_source_sha256=self.source_pins)

    def test_controlled_capture_roundtrip_and_strict_architecture_forward(self):
        before = M.FORMAT.describe_state(M.FORMAT.freeze_state(self.trainer.ema.ema.state_dict()))
        receipt = self.capture(self.trainer)
        self.assertEqual(receipt["ema_updates_before"], 2)
        self.assertEqual(receipt["ema_updates_after"], 2)
        self.assertTrue(receipt["reconstruction"]["output_values_dtypes_shapes_exact"])
        self.assertFalse(receipt["scientific_reference_accepted"])
        self.assertFalse(receipt["arbitrary_external_concurrent_writers_excluded_by_lock"])
        self.assertEqual(before, M.FORMAT.describe_state(M.FORMAT.freeze_state(self.trainer.ema.ema.state_dict())))
        outer = self.output / "epoch_0001"
        self.assertTrue(outer.is_dir())
        self.assertFalse((self.output / "epoch_0001.pending").exists())
        manifest = json.loads((outer / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            self.assertEqual(M.sha(outer / name), expected["sha256"])
        state, metadata = M.FORMAT.load_snapshot(outer / "state")
        self.assertEqual(before, M.FORMAT.describe_state(state))
        self.assertEqual(metadata["training_contract_sha256"], "a" * 64)

    def test_copy_forward_does_not_change_live_head_cache(self):
        head = self.trainer.ema.ema.model[-1]
        before = {name: deepcopy(getattr(head, name, None)) for name in ("shape", "anchors", "strides")}
        self.capture(self.trainer)
        for name, old in before.items():
            new = getattr(head, name, None)
            self.assertTrue(torch.equal(old, new) if isinstance(old, torch.Tensor) else old == new)

    def test_half_and_lift_are_not_recovered(self):
        self.trainer.ema.ema.half()
        with self.assertRaisesRegex(ValueError, "already be FP32"):
            self.capture(self.trainer)
        self.assertFalse(self.output.exists())

    def test_nonfinite_live_state_rejected_before_directory(self):
        next(self.trainer.ema.ema.parameters()).view(-1)[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.capture(self.trainer)
        self.assertFalse(self.output.exists())

    def test_amp_config_and_effective_amp_must_both_be_false(self):
        for args_amp, effective_amp in ((True, False), (False, True), ("fp32", False), (0, False)):
            self.trainer.args.amp, self.trainer.amp = args_amp, effective_amp
            with self.assertRaisesRegex(ValueError, "AMP"):
                self.capture(self.trainer)

    def test_eval_and_updates_are_required(self):
        self.trainer.ema.ema.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            self.capture(self.trainer)
        self.trainer.ema.ema.eval()
        self.trainer.ema.updates = 0
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.capture(self.trainer)

    def test_disabled_ema_rejected(self):
        self.trainer.ema.enabled = False
        with self.assertRaisesRegex(ValueError, "Enabled"):
            self.capture(self.trainer)

    def test_selected_epoch_only_no_best_selection(self):
        self.trainer.epoch = 1
        self.assertIsNone(self.capture(self.trainer))
        self.assertFalse(self.output.exists())
        self.assertEqual(self.capture.receipts, [])

    def test_no_overwrite_existing_epoch(self):
        self.capture(self.trainer)
        original = M.sha(self.output / "epoch_0001" / "capture_receipt.json")
        with self.assertRaisesRegex(ValueError, "must be new"):
            self.capture(self.trainer)
        self.assertEqual(original, M.sha(self.output / "epoch_0001" / "capture_receipt.json"))

    def test_other_thread_is_rejected(self):
        failures = []
        def run():
            try:
                self.capture(self.trainer)
            except Exception as exc:
                failures.append(str(exc))
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        self.assertEqual(len(failures), 1)
        self.assertIn("main training thread", failures[0])
        self.assertFalse(self.output.exists())

    def test_reentrant_hook_lock_fails_closed(self):
        self.capture._lock.acquire()
        try:
            with self.assertRaisesRegex(ValueError, "reentrant"):
                self.capture(self.trainer)
        finally:
            self.capture._lock.release()

    def test_sources_are_pinned_before_capture(self):
        self.capture.expected_sources["capture_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.capture(self.trainer)
        self.assertFalse(self.output.exists())

    def test_constructor_rejects_unpinned_source_or_bad_contract(self):
        with self.assertRaisesRegex(ValueError, "pre-pinned"):
            M.LiveFP32Capture(self.output, "a" * 64, [1], expected_source_sha256={})
        for contract in ("", "A" * 64, "a" * 63):
            with self.assertRaisesRegex(ValueError, "contract"):
                M.LiveFP32Capture(self.output, contract, [1], expected_source_sha256=self.source_pins)

    def test_invalid_selected_epochs(self):
        for epochs in ([], [0], [True], [1, 1]):
            with self.assertRaisesRegex(ValueError, "epochs"):
                M.LiveFP32Capture(self.output, "a" * 64, epochs, expected_source_sha256=self.source_pins)

    def test_arbitrary_input_is_not_accepted_only_analytic_recipe(self):
        left = M.deterministic_synthetic_input()
        right = M.deterministic_synthetic_input()
        self.assertTrue(torch.equal(left, right))
        self.assertEqual(float(left.flatten()[1]), float(torch.tensor(1 / 250, dtype=torch.float32)))
        for shape in ((1, 3, 63, 64), (2, 3, 64, 64), (1, 3, 512, 64), (1, 1, 64, 64)):
            with self.assertRaises(ValueError):
                M.deterministic_synthetic_input(shape)

    def test_counter_mutation_during_capture_prevents_commit(self):
        original = M.FORMAT.write_snapshot
        def changing(*args, **kwargs):
            result = original(*args, **kwargs)
            self.trainer.ema.updates += 1
            return result
        with patch.object(M.FORMAT, "write_snapshot", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "counter changed"):
                self.capture(self.trainer)
        self.assertFalse((self.output / "epoch_0001").exists())
        self.assertTrue((self.output / "epoch_0001.pending").exists())
        self.assertFalse((self.output / "epoch_0001.pending" / "capture_receipt.json").exists())

    def test_state_mutation_during_capture_prevents_commit(self):
        original = M.FORMAT.write_snapshot
        def changing(*args, **kwargs):
            result = original(*args, **kwargs)
            next(self.trainer.ema.ema.parameters()).add_(0.0001)
            return result
        with patch.object(M.FORMAT, "write_snapshot", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "counter changed"):
                self.capture(self.trainer)
        self.assertFalse((self.output / "epoch_0001").exists())

    def test_capture_executes_without_grad(self):
        original = M.FORMAT.write_snapshot
        flags = []
        def observing(*args, **kwargs):
            flags.append(torch.is_grad_enabled())
            return original(*args, **kwargs)
        with torch.enable_grad(), patch.object(M.FORMAT, "write_snapshot", side_effect=observing):
            self.capture(self.trainer)
        self.assertEqual(flags, [False])

    def test_rng_unchanged_by_reconstruction(self):
        before = torch.get_rng_state().clone()
        self.capture(self.trainer)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_mixin_capture_precedes_super_half_serialization(self):
        events = []
        source_trainer = self.trainer
        class Base:
            def save_model(self):
                events.append("framework_half_serialization")
                self.half_copy = deepcopy(self.ema.ema).half()
                return "saved"
        class Trainer(M.FP32LiveCaptureMixin, Base):
            pass
        trainer = Trainer()
        trainer.__dict__.update(source_trainer.__dict__)
        capture = trainer.configure_fp32_capture(self.output, "a" * 64, [1], expected_source_sha256=self.source_pins)
        original = capture._capture
        def observing(value):
            result = original(value)
            events.append("exact_fp32_capture_complete")
            return result
        with patch.object(capture, "_capture", side_effect=observing):
            self.assertEqual(trainer.save_model(), "saved")
        self.assertEqual(events, ["exact_fp32_capture_complete", "framework_half_serialization"])
        self.assertEqual(next(trainer.half_copy.parameters()).dtype, torch.float16)
        self.assertEqual(next(trainer.ema.ema.parameters()).dtype, torch.float32)

    def test_mixin_double_adapter_and_unconfigured_save_rejected(self):
        class Base:
            def save_model(self):
                raise AssertionError("must fail before super")
        class Trainer(M.FP32LiveCaptureMixin, Base):
            pass
        trainer = Trainer()
        with self.assertRaisesRegex(ValueError, "Configure"):
            trainer.save_model()
        trainer.fp32_capture_callback = self.capture
        with self.assertRaisesRegex(ValueError, "either"):
            trainer.configure_fp32_capture(self.output, "a" * 64, [1], expected_source_sha256=self.source_pins)

    def test_unreviewed_module_and_names_mismatch_rejected(self):
        model = self.trainer.ema.ema
        model.names = {0: "zero"}
        with self.assertRaisesRegex(ValueError, "Names"):
            M.architecture(model)
        model.names = {0: "zero", 1: "one"}
        model.yaml["backbone"][0][2] = "UnreviewedModule"
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            M.architecture(model)

    def test_custom_activation_and_non_rgb_cfg_rejected(self):
        model = self.trainer.ema.ema
        model.yaml["activation"] = "nn.SiLU()"
        with self.assertRaisesRegex(ValueError, "activation"):
            M.architecture(model)
        model.yaml.pop("activation")
        model.yaml["channels"] = True
        with self.assertRaisesRegex(ValueError, "three-channel"):
            M.architecture(model)


if __name__ == "__main__":
    unittest.main()
