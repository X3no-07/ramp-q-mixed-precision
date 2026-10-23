"""Synthetic CPU format/precision tests; no old checkpoint or real data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fp32_reference_format", ROOT/"scripts/gate1_fp32_reference_state.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class ReferenceStateTests(unittest.TestCase):
    def setUp(self):
        base = (ROOT/"artifacts/gate1/20261023_fp32_reference_preparation_v1/test_cases").resolve()
        assert base.is_relative_to(ROOT)
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        assert Path(self.tmp.name).resolve().is_relative_to(base)
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)/"snapshot"
        self.metadata = {"training_contract_sha256": "a"*64, "architecture_sha256": "b"*64,
                         "epoch_one_based": 1, "source_stage": M.STAGE}

    def state(self):
        return {"weight": torch.tensor([1.00012362, -0.34567893, 0.00001356789], dtype=torch.float32),
                "bn.running_mean": torch.tensor([0.12345679]), "bn.num_batches_tracked": torch.tensor(7),
                "enabled": torch.tensor([True, False])}

    def test_exact_mantissa_roundtrip_vs_half_counterexample(self):
        state = self.state()
        self.assertFalse(torch.equal(state["weight"], state["weight"].half().float()))
        receipt = M.write_snapshot(state, self.output, self.metadata)
        loaded, metadata = M.load_snapshot(self.output)
        self.assertEqual(metadata, self.metadata)
        for name in state:
            self.assertEqual(state[name].dtype, loaded[name].dtype)
            self.assertTrue(torch.equal(state[name], loaded[name]))
        self.assertTrue(receipt["all_values_and_dtypes_exact"])
        self.assertFalse(receipt["accepted_scientific_reference"])
        self.assertFalse(receipt["old_checkpoint_recovered"])

    def test_does_not_mutate_alias_or_require_gradient(self):
        source = torch.tensor([1.00012362, 3.456789], requires_grad=True)
        view = source[::2]
        before = source.detach().clone()
        M.write_snapshot({"view": view}, self.output, self.metadata)
        self.assertTrue(torch.equal(source, before)); self.assertTrue(source.requires_grad)
        self.assertIsNone(source.grad)
        loaded, _ = M.load_snapshot(self.output)
        self.assertFalse(loaded["view"].requires_grad)
        self.assertNotEqual(loaded["view"].data_ptr(), source.data_ptr())

    def test_float16_and_float64_sources_rejected_before_directory_creation(self):
        for dtype in (torch.float16, torch.float64, torch.bfloat16):
            with self.assertRaises(ValueError): M.write_snapshot({"w": torch.ones(1, dtype=dtype)}, self.output, self.metadata)
            self.assertFalse(self.output.exists())

    def test_nonfinite_and_complex_sources_rejected(self):
        for value in (torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.ones(1, dtype=torch.complex64)):
            with self.assertRaises(ValueError): M.write_snapshot({"w": value}, self.output, self.metadata)
        self.assertFalse(self.output.exists())

    def test_integer_only_and_non_tensor_sources_rejected(self):
        for state in ({"counter": torch.tensor(1)}, {"not_tensor": 1}, {}, {"": torch.ones(1)}):
            with self.assertRaises(ValueError): M.write_snapshot(state, self.output, self.metadata)

    def test_post_save_provenance_and_bad_metadata_rejected(self):
        for updates in ({"source_stage": "LIFTED_FP16_CHECKPOINT"}, {"epoch_one_based": True},
                        {"training_contract_sha256": "bad"}, {"architecture_sha256": "A"*64}, {"extra": 1}):
            with self.assertRaises(ValueError): M.write_snapshot(self.state(), self.output, dict(self.metadata, **updates))
        self.assertFalse(self.output.exists())

    def test_declared_stage_does_not_prove_origin(self):
        lifted = {"w": torch.tensor([1.234567]).half().float()}
        receipt = M.write_snapshot(lifted, self.output, self.metadata)
        self.assertFalse(receipt["training_origin_independently_proven"])
        self.assertFalse(receipt["live_training_hook_integrated"])
        self.assertFalse(receipt["accepted_scientific_reference"])

    def test_existing_destination_never_overwritten(self):
        self.output.mkdir(); sentinel = self.output/"keep.txt"; sentinel.write_text("keep")
        with self.assertRaises(ValueError): M.write_snapshot(self.state(), self.output, self.metadata)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_tampered_payload_rejected_before_deserialization(self):
        M.write_snapshot(self.state(), self.output, self.metadata)
        with (self.output/"reference_state.pt").open("ab") as stream: stream.write(b"tamper")
        with self.assertRaises(ValueError): M.load_snapshot(self.output)

    def test_tampered_receipt_rejected(self):
        M.write_snapshot(self.state(), self.output, self.metadata)
        path = self.output/"snapshot_receipt.json"
        value = json.loads(path.read_text()); value["status"] = "ACCEPTED"
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): M.load_snapshot(self.output)


if __name__ == "__main__": unittest.main()
