"""Synthetic CPU-only decoder/protocol tests; no real models/images or GPU work."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_gate1_decode_precision.py"
SPEC = importlib.util.spec_from_file_location("decode_probe_under_test", PATH)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)
TEST_TEMP_ROOT = PATH.parents[1] / "artifacts" / "gate1" / "20261023_decode_probe_preparation" / "runner_tests"


def workspace_temp():
    # Windows sandbox's redirected system TEMP may be read-only to this process.
    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT)


def protocol_fixture(stage="pilot3"):
    rows = [{"image_id": f"train/image{i:03}", "assigned_split": "debug", "source_split": "train",
             "image_path": f"images/train/image{i:03}.jpg"} for i in range(109)]
    p = {"schema": P.SCHEMA, "runner_sha256": "a" * 64, "data_role": "debug",
         "selection_detector_inference": False, "holdout_accessed": False,
         "historical_equality_required": True, "native_reconstruction_required": True,
         "controls": deepcopy(P.CONTROLS), "raw_tolerances": deepcopy(P.TOLERANCES),
         "qualifier": {"sha256": P.QUALIFIER_SHA}, "checkpoint": {"sha256": P.CHECKPOINT_SHA},
         "image_ids": [r["image_id"] for r in rows[:3 if stage == "pilot3" else 109]], "stage": stage,
         "package_versions": {"torch": "synthetic"}, "framework_source_sha256": {"head.py": "b" * 64}}
    return p, rows


def inputs(reg_max=4, n=6, dtype=torch.float32):
    boxes = (torch.arange(4 * reg_max * n, dtype=torch.float32).reshape(1, 4 * reg_max, n) % 23 - 11) / 7
    classes = torch.linspace(-4, 4, 10 * n).reshape(1, 10, n)
    anchors = torch.stack((torch.arange(n) + 0.5, torch.arange(n) + 3.5))
    strides = torch.full((1, n), 8)
    return tuple(x.to(dtype=dtype) for x in (boxes, classes, anchors, strides))


class DecoderTests(unittest.TestCase):
    def test_uniform_dfl_closed_form(self):
        x = list(inputs())
        x[0].zero_()
        result = P.decode_functional(torch, *x, 4, torch.float32)
        self.assertTrue(torch.equal(result["probabilities"], torch.full((1, 4, 4, 6), 0.25)))
        self.assertTrue(torch.equal(result["distances"], torch.full((1, 4, 6), 1.5)))
        self.assertTrue(torch.equal(result["boxes"][:, :2], x[2].unsqueeze(0) * 8))
        self.assertTrue(torch.equal(result["boxes"][:, 2:], torch.full((1, 2, 6), 24.0)))

    def test_matches_independent_native_style_module_both_dtypes(self):
        for dtype in (torch.float32, torch.float16):
            boxes, classes, anchors, strides = inputs(dtype=dtype)
            layer = torch.nn.Conv2d(4, 1, 1, bias=False).to(dtype=dtype)
            layer.weight.data.copy_(torch.arange(4, dtype=dtype).view(1, 4, 1, 1))
            distances = layer(boxes.view(1, 4, 4, 6).transpose(2, 1).softmax(1)).view(1, 4, 6)
            lt, rb = distances.chunk(2, 1)
            xy1, xy2 = anchors.unsqueeze(0) - lt, anchors.unsqueeze(0) + rb
            expected = torch.cat((torch.cat(((xy1 + xy2) / 2, xy2 - xy1), 1) * strides,
                                  classes.sigmoid()), 1)
            actual = P.decode_functional(torch, boxes, classes, anchors, strides, 4, dtype)
            self.assertTrue(torch.equal(actual["decoded"], expected))
            self.assertEqual(actual["decoded"].dtype, dtype)

    def test_casts_all_arithmetic_from_same_captured_logits(self):
        captured = inputs(dtype=torch.float16)
        snapshots = [x.clone() for x in captured]
        result = P.decode_functional(torch, *captured, 4, torch.float32)
        expected = P.decode_functional(torch, *(x.float() for x in captured), 4, torch.float32)
        for key in result:
            self.assertEqual(result[key].dtype, torch.float32)
            self.assertTrue(torch.equal(result[key], expected[key]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(captured, snapshots)))

    def test_geometry_operation_order_exposes_half_rounding(self):
        # A real counterexample to the tempting but different lt+rb simplification.
        anchors = torch.tensor([[[79.5], [79.5]]], dtype=torch.float16)
        lt = torch.tensor([[[2.333984375], [7.99609375]]], dtype=torch.float16)
        rb = torch.tensor([[[3.001953125], [6.00390625]]], dtype=torch.float16)
        xy1, xy2 = anchors - lt, anchors + rb
        ordered = xy2 - xy1
        simplified = lt + rb
        self.assertFalse(torch.equal(ordered, simplified))

    def test_rejects_bad_shape_nonfinite_or_precision(self):
        original = inputs()
        for bad in (torch.float64, torch.int64):
            with self.assertRaises(ValueError):
                P.decode_functional(torch, *original, 4, bad)
        with self.assertRaises(ValueError):
            P.decode_functional(torch, *original, 1, torch.float32)
        bad = list(original); bad[0] = bad[0][:, :-1]
        with self.assertRaises(ValueError):
            P.decode_functional(torch, *bad, 4, torch.float32)
        bad = list(inputs()); bad[1][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            P.decode_functional(torch, *bad, 4, torch.float32)


class ProtocolTests(unittest.TestCase):
    def check(self, p, rows):
        return P.validate_protocol(p, "a" * 64, rows)

    def test_pilot_and_full_exact_order(self):
        for stage in ("pilot3", "full109"):
            p, rows = protocol_fixture(stage)
            selected = self.check(p, rows)
            self.assertEqual([x["image_id"] for x in selected], p["image_ids"])

    def test_rejects_changed_source_checkpoint_or_runner(self):
        for field in ("runner_sha256", "qualifier", "checkpoint"):
            p, rows = protocol_fixture()
            if field == "runner_sha256":
                p[field] = "b" * 64
            else:
                p[field]["sha256"] = "b" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(p, rows)

    def test_rejects_tolerance_and_control_relaxation(self):
        for container, key, value in (("raw_tolerances", "box_atol", 1.0), ("controls", "tf32", True)):
            p, rows = protocol_fixture(); p[container][key] = value
            with self.assertRaises(ValueError):
                self.check(p, rows)

    def test_rejects_data_leakage_and_disabled_exactness(self):
        changes = [("data_role", "selection"), ("selection_detector_inference", True), ("holdout_accessed", True),
                   ("historical_equality_required", False), ("native_reconstruction_required", False)]
        for key, value in changes:
            p, rows = protocol_fixture(); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(p, rows)

    def test_rejects_non_debug_or_duplicate_cohort(self):
        for change in ("selection", "duplicate", "path"):
            p, rows = protocol_fixture()
            if change == "selection": rows[0]["assigned_split"] = "selection"
            if change == "duplicate": rows[1] = deepcopy(rows[0])
            if change == "path": rows[0]["image_path"] = "images/val/unseen.jpg"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check(p, rows)

    def test_rejects_unknown_duplicate_ids_or_reordered_full(self):
        for change in ("unknown", "duplicate", "order"):
            p, rows = protocol_fixture("full109" if change == "order" else "pilot3")
            if change == "unknown": p["image_ids"][0] = "train/unknown"
            if change == "duplicate": p["image_ids"][1] = p["image_ids"][0]
            if change == "order": p["image_ids"].reverse()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check(p, rows)


class IntegrityTests(unittest.TestCase):
    def test_bytes_equal_rejects_close_dtype_shape_or_signed_zero_difference(self):
        a = np.array([0.0, 1.0], dtype=np.float32)
        self.assertTrue(P.bytes_equal(a, a.copy()))
        self.assertFalse(P.bytes_equal(a, np.array([-0.0, 1.0], dtype=np.float32)))
        self.assertFalse(P.bytes_equal(a, a.astype(np.float64)))
        self.assertFalse(P.bytes_equal(a, a.reshape(1, 2)))
        self.assertFalse(P.bytes_equal(a, np.array([0.0, 1.0000001], dtype=np.float32)))

    def test_bound_identity_rejects_tampering(self):
        with workspace_temp() as temp:
            path = Path(temp).resolve() / "input.bin"; path.write_bytes(b"frozen")
            entry = P.identity(path)
            self.assertEqual(P.bound_identity(entry, "fixture"), path)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError): P.bound_identity(entry, "fixture")

    def test_safe_file_rejects_escape(self):
        with workspace_temp() as temp:
            root = Path(temp).resolve(); path = root / "ok.txt"; path.write_text("ok")
            self.assertEqual(P.safe_file(root, "ok.txt"), path)
            with self.assertRaises(ValueError): P.safe_file(root, "../escape")
            with self.assertRaises(ValueError): P.safe_file(root, str(path))

    def test_lifted_state_includes_buffers(self):
        left = torch.nn.BatchNorm1d(2).float()
        right = deepcopy(left).half()
        result = P.compare_lifted_states(torch, left, right)
        self.assertTrue(result["equal_after_float32_lift"])
        right.running_mean[0] = 1
        with self.assertRaises(ValueError): P.compare_lifted_states(torch, left, right)


if __name__ == "__main__":
    unittest.main()
