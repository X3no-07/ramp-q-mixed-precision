"""Synthetic CPU-only head probe tests; no real images/models/GPU execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load("head_probe_under_test", ROOT / "scripts" / "probe_gate1_head_precision.py")
D = load("frozen_decoder_for_head_tests", ROOT / "scripts" / "probe_gate1_decode_precision.py")


def fixture(stage="pilot3"):
    rows = [{"image_id": f"train/fixture{i:03}", "assigned_split": "debug", "source_split": "train",
             "image_path": f"images/train/fixture{i:03}.jpg"} for i in range(109)]
    for i, image_id in enumerate(H.PILOT_IDS):
        rows[i].update(image_id=image_id, image_path=f"images/{image_id}.jpg")
    p = {"schema": H.SCHEMA, "runner_sha256": "a" * 64, "decoder_runner": {"sha256": H.DECODER_SHA},
         "data_role": "debug", "selection_detector_inference": False, "holdout_accessed": False,
         "historical_equality_required": True, "native_reconstruction_required": True,
         "intervention": deepcopy(H.INTERVENTION), "controls": deepcopy(D.CONTROLS),
         "raw_tolerances": deepcopy(D.TOLERANCES), "qualifier": {"sha256": D.QUALIFIER_SHA},
         "checkpoint": {"sha256": D.CHECKPOINT_SHA}, "stage": stage,
         "image_ids": H.PILOT_IDS.copy() if stage == "pilot3" else [row["image_id"] for row in rows],
         "package_versions": {"torch": "synthetic"}, "framework_source_sha256": {"head.py": "b" * 64}}
    return p, rows


class SyntheticHead(torch.nn.Module):
    """Small deterministic eval head with the pinned head's public structure."""
    def __init__(self):
        super().__init__()
        self.nl = 3
        self.nc = 10
        self.reg_max = 4
        self.cv2 = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(c, 16, 1), torch.nn.BatchNorm2d(16),
                                                          torch.nn.SiLU()) for c in (2, 4, 8)])
        self.cv3 = torch.nn.ModuleList([torch.nn.Conv2d(c, 10, 1) for c in (2, 4, 8)])
        # Make both branch weights exactly representable by a serialized half model.
        self.half().float().eval()

    @property
    def one2many(self):
        return {"box_head": self.cv2, "cls_head": self.cv3}

    def forward_head(self, x, box_head, cls_head):
        b = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(b, 16, -1) for i in range(3)], -1)
        scores = torch.cat([cls_head[i](x[i]).view(b, 10, -1) for i in range(3)], -1)
        return {"boxes": boxes, "scores": scores, "feats": x}


def feature_fixture():
    return [(torch.arange(c * h * h).reshape(1, c, h, h).float() - 10) / 13
            for c, h in ((2, 4), (4, 2), (8, 1))]


class HeadArithmeticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_native_forward_head_reconstruction_exact_both_dtypes(self):
        for dtype in (torch.float32, torch.float16):
            head = SyntheticHead().to(dtype=dtype)
            features = [x.to(dtype=dtype) for x in feature_fixture()]
            original = head.forward_head(features, **head.one2many)
            reconstruction = H.cross_head(torch, head, features, dtype)
            self.assertTrue(H.tensor_exact(torch, original["boxes"], reconstruction["boxes"]))
            self.assertTrue(H.tensor_exact(torch, original["scores"], reconstruction["scores"]))

    def test_full_cross_uses_receiving_head_dtype_and_preserves_features_state(self):
        heads = {"fp32": SyntheticHead()}
        heads["fp16"] = deepcopy(heads["fp32"]).half()
        sources = {"fp32": feature_fixture(), "fp16": [x.half() for x in feature_fixture()]}
        feature_before = {key: [x.clone() for x in values] for key, values in sources.items()}
        state_before = {key: {k: v.clone() for k, v in head.state_dict().items()} for key, head in heads.items()}
        outputs = {}
        for source in sources:
            for branch, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
                result = H.cross_head(torch, heads[branch], sources[source], dtype)
                outputs[source, branch] = result
                self.assertEqual(result["boxes"].dtype, dtype)
                self.assertTrue(all(v.dtype == dtype for v in result["feats"]))
                self.assertEqual(tuple(result["boxes"].shape), (1, 16, 21))
                for scale, feature in enumerate(result["feats"]):
                    self.assertTrue(H.tensor_exact(torch, feature, sources[source][scale].to(dtype=dtype)))
        self.assertTrue(all(H.tensor_exact(torch, value, saved) for key in sources for value, saved in zip(sources[key], feature_before[key])))
        self.assertTrue(all(H.tensor_exact(torch, value, state_before[key][name]) for key, head in heads.items() for name, value in head.state_dict().items()))
        self.assertFalse(torch.equal(outputs["fp32", "fp32"]["boxes"], outputs["fp16", "fp32"]["boxes"]))

    def test_unified_fp32_decoding_from_each_combination(self):
        head32 = SyntheticHead()
        head16 = deepcopy(head32).half()
        original = feature_fixture()
        for features in (original, [x.half() for x in original]):
            for head, dtype in ((head32, torch.float32), (head16, torch.float16)):
                logits = H.cross_head(torch, head, features, dtype)
                anchors = torch.full((2, 21), 0.5, dtype=dtype)
                strides = torch.full((1, 21), 8, dtype=dtype)
                decoded = D.decode_functional(torch, logits["boxes"], logits["scores"], anchors, strides, 4, torch.float32)
                self.assertEqual(decoded["decoded"].dtype, torch.float32)
                self.assertEqual(tuple(decoded["decoded"].shape), (1, 14, 21))
                self.assertTrue(bool(torch.isfinite(decoded["decoded"]).all()))

    def test_rejects_train_submodule_dtype_and_scale_mismatch(self):
        features = feature_fixture()
        head = SyntheticHead().train()
        with self.assertRaises(ValueError): H.cross_head(torch, head, features, torch.float32)
        head.eval(); head.cv2[0][1].train()
        with self.assertRaises(ValueError): H.cross_head(torch, head, features, torch.float32)
        head.eval()
        with self.assertRaises(ValueError): H.cross_head(torch, head, features, torch.float16)
        with self.assertRaises(ValueError): H.cross_head(torch, head, features[:2], torch.float32)
        with self.assertRaises(ValueError): H.cross_head(torch, head, features, torch.float64)

    def test_exactness_rejects_dtype_shape_value_changes(self):
        x = torch.tensor([1.0], dtype=torch.float32)
        self.assertTrue(H.tensor_exact(torch, x, x.clone()))
        self.assertFalse(H.tensor_exact(torch, x, x.half()))
        self.assertFalse(H.tensor_exact(torch, x, x.reshape(1, 1)))
        self.assertFalse(H.tensor_exact(torch, x, x + 0.001))


class HeadProtocolTests(unittest.TestCase):
    def validate(self, p, rows):
        return H.validate_head_protocol(p, "a" * 64, rows, D)

    def test_pilot_and_full_valid(self):
        for stage in ("pilot3", "full109"):
            p, rows = fixture(stage)
            self.assertEqual([r["image_id"] for r in self.validate(p, rows)], p["image_ids"])

    def test_adapter_changes_only_two_schema_binding_fields(self):
        p, _ = fixture(); original = deepcopy(p)
        adapted = H.decoder_adapter(p, D)
        self.assertEqual(p, original)
        changed = {key for key in p if p[key] != adapted[key]}
        self.assertEqual(changed, {"schema", "runner_sha256"})
        self.assertEqual(adapted["schema"], D.SCHEMA)
        self.assertEqual(adapted["runner_sha256"], D.sha(ROOT / "scripts" / "probe_gate1_decode_precision.py"))

    def test_rejects_changed_runner_decoder_and_intervention(self):
        for key in ("runner_sha256", "decoder_runner", "intervention"):
            p, rows = fixture()
            if key == "runner_sha256": p[key] = "b" * 64
            if key == "decoder_runner": p[key]["sha256"] = "b" * 64
            if key == "intervention": p[key]["all_comparisons_decode_dtype"] = "fp16"
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(p, rows)

    def test_rejects_pilot_replacement_or_reordering(self):
        for mode in ("replace", "reverse"):
            p, rows = fixture()
            if mode == "replace": p["image_ids"][0] = rows[3]["image_id"]
            if mode == "reverse": p["image_ids"].reverse()
            with self.assertRaises(ValueError): self.validate(p, rows)

    def test_preserves_original_role_control_and_tolerance_guards(self):
        for key in ("holdout_accessed", "selection_detector_inference", "controls", "raw_tolerances"):
            p, rows = fixture()
            if key in ("holdout_accessed", "selection_detector_inference"): p[key] = True
            if key == "controls": p[key]["tf32"] = True
            if key == "raw_tolerances": p[key]["box_atol"] = 2
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(p, rows)

    def test_frozen_decoder_loaded_from_exact_identity(self):
        path = ROOT / "scripts" / "probe_gate1_decode_precision.py"
        entry = {"path": str(path), "bytes": path.stat().st_size, "sha256": D.sha(path)}
        loaded = H.load_decoder(entry)
        self.assertEqual(loaded.SCHEMA, D.SCHEMA)
        entry["bytes"] += 1
        with self.assertRaises(ValueError): H.load_decoder(entry)


if __name__ == "__main__":
    unittest.main()
