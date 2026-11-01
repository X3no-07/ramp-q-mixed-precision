"""Synthetic CPU-only tests for a single-forward mixed precision prototype."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

torch.set_num_threads(1)


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load("body32_head16_under_test", ROOT / "scripts/probe_gate1_body32_head16.py")
D = load("frozen_decode_for_prototype_tests", ROOT / "scripts/probe_gate1_decode_precision.py")
SOURCES = [-1] * 23
for index, sources in {12: [-1, 6], 15: [-1, 4], 18: [-1, 13], 21: [-1, 10]}.items(): SOURCES[index] = sources


class BodyLayer(torch.nn.Module):
    def __init__(self, i, f):
        super().__init__(); self.i, self.f = i, deepcopy(f)
        self.conv = torch.nn.Conv2d(2 * len(f) if isinstance(f, list) else 2, 2, 1)
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        if isinstance(value, list): value = torch.cat(value, 1)
        return self.conv(value).sin()


class SyntheticDetect(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.i, self.f = 23, [16, 19, 22]
        self.nc, self.nl, self.reg_max = 10, 3, 4
        self.dynamic = self.export = self.end2end = self.xyxy = False
        self.cv2 = torch.nn.ModuleList([torch.nn.Conv2d(2, 16, 1) for _ in range(3)])
        self.cv3 = torch.nn.ModuleList([torch.nn.Conv2d(2, 10, 1) for _ in range(3)])
        self.dfl = torch.nn.Module(); self.dfl.conv = torch.nn.Conv2d(4, 1, 1, bias=False)
        with torch.no_grad(): self.dfl.conv.weight.copy_(torch.arange(4).view(1, 4, 1, 1))
        self.register_buffer("stride", torch.tensor([8., 16., 32.]))
        self.head_calls = 0

    @property
    def one2many(self): return {"box_head": self.cv2, "cls_head": self.cv3}

    def forward_head(self, features, box_head, cls_head):
        self.head_calls += 1
        return {"boxes": torch.cat([box_head[i](x).view(1, 16, -1) for i, x in enumerate(features)], -1),
                "scores": torch.cat([cls_head[i](x).view(1, 10, -1) for i, x in enumerate(features)], -1), "feats": features}

    def forward(self, *_args, **_kwargs): raise AssertionError("Native head decode must never be used")


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList([BodyLayer(i, f) for i, f in enumerate(SOURCES)] + [SyntheticDetect()])
        self.save = [4, 6, 10, 13, 16, 19, 22]
        self.names = {i: str(i) for i in range(10)}
        self.half().float().eval()

    def forward(self, *_args, **_kwargs): raise AssertionError("Complete native model forward must never be used")


def prepare():
    model = Network()
    model.model[-1].cv2.half(); model.model[-1].cv3.half()
    return model


def anchors(features, strides, offset):
    points, scales = [], []
    for feature, stride in zip(features, strides):
        h, w = feature.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h, dtype=feature.dtype) + offset, torch.arange(w, dtype=feature.dtype) + offset, indexing="ij")
        points.append(torch.stack((x, y), -1).reshape(-1, 2))
        scales.append(torch.full((h * w, 1), float(stride), dtype=feature.dtype))
    return torch.cat(points), torch.cat(scales)


def fixture(stage="pilot3"):
    rows = [{"image_id": f"train/fixture{i:03}", "assigned_split": "debug", "source_split": "train",
             "image_path": f"images/train/fixture{i:03}.jpg"} for i in range(109)]
    for i, image_id in enumerate(P.PILOT_IDS): rows[i].update(image_id=image_id, image_path=f"images/{image_id}.jpg")
    p = {"schema": P.SCHEMA, "runner_sha256": "a" * 64, "precision_contract": deepcopy(P.PRECISION_CONTRACT), "guard_name": P.CHECK_NAME,
         "head_runner": {"sha256": P.HEAD_SHA}, "decoder_runner": {"sha256": P.DECODER_SHA},
         "topology": P.topology_of_model(prepare()), "controls": deepcopy(D.CONTROLS), "raw_tolerances": deepcopy(D.TOLERANCES),
         "qualifier": {"sha256": D.QUALIFIER_SHA}, "checkpoint": {"sha256": D.CHECKPOINT_SHA}, "data_role": "debug",
         "selection_detector_inference": False, "holdout_accessed": False, "historical_equality_required": True,
         "native_reconstruction_required": True, "stage": stage,
         "image_ids": P.PILOT_IDS.copy() if stage == "pilot3" else [r["image_id"] for r in rows],
         "package_versions": {"torch": "synthetic"}, "framework_source_sha256": {"head.py": "b" * 64}}
    return p, rows


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = prepare()
        self.input = (torch.arange(32).reshape(1, 2, 4, 4).float() + 1) / 37

    def test_single_forward_observes_all_fp32_body_modules_and_half_head_logits(self):
        wrapper = P.make_prototype(torch, self.model, D.decode_functional, anchors)
        result = wrapper(self.input)
        self.assertEqual([r["i"] for r in result["body_module_observations"]], list(range(23)))
        self.assertTrue(all(r["dtype"] == "torch.float32" for r in result["body_module_observations"]))
        self.assertTrue(all(m.calls == 1 for m in self.model.model[:-1]))
        self.assertEqual(self.model.model[-1].head_calls, 1)
        self.assertEqual(result["box_logits"].dtype, torch.float16)
        self.assertEqual(result["class_logits"].dtype, torch.float16)
        self.assertTrue(all(v.dtype == torch.float32 for k, v in result.items() if k in ("anchors", "strides", "dfl_bins", "decoded")))
        self.assertEqual(set(result["decoder_stage_dtypes"].values()), {"torch.float32"})

    def test_actual_dag_features_and_endpoint_equal_independent_dispatch(self):
        independent = deepcopy(self.model)
        cache = {-1: self.input}
        for i, module in enumerate(independent.model[:-1]):
            if module.f == -1: value = cache[i - 1]
            elif isinstance(module.f, int): value = cache[module.f]
            else: value = [cache[i - 1] if j == -1 else cache[j] for j in module.f]
            cache[i] = module(value)
        features = [cache[i] for i in (16, 19, 22)]
        head = independent.model[-1]
        expected = head.forward_head([v.half() for v in features], **head.one2many)
        a, s = anchors(features, head.stride, .5)
        decoded = D.decode_functional(torch, expected["boxes"], expected["scores"], a.T, s.T, 4, torch.float32)["decoded"]
        actual = P.make_prototype(torch, self.model, D.decode_functional, anchors)(self.input)
        self.assertTrue(torch.equal(actual["box_logits"], expected["boxes"]))
        self.assertTrue(torch.equal(actual["class_logits"], expected["scores"]))
        self.assertTrue(torch.equal(actual["decoded"], decoded))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(features, actual["features"])))

    def test_repeated_fresh_image_forward_is_deterministic_and_uses_new_pixels(self):
        wrapper = P.make_prototype(torch, self.model, D.decode_functional, anchors)
        first = wrapper(self.input)["decoded"]
        second = wrapper(self.input.clone())["decoded"]
        self.assertTrue(torch.equal(first, second))
        # The first feature changes directly with the fresh input.
        changed = wrapper(self.input + .5)
        self.assertFalse(torch.equal(wrapper(self.input)["features"][0], changed["features"][0]))
        self.assertTrue(all(m.calls == 4 for m in self.model.model[:-1]))

    def test_model_parameters_buffers_and_input_preserved(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        tensor_before = self.input.clone()
        P.make_prototype(torch, self.model, D.decode_functional, anchors)(self.input)
        self.assertTrue(torch.equal(tensor_before, self.input))
        self.assertTrue(all(v.dtype == before[k].dtype and torch.equal(v, before[k]) for k, v in self.model.state_dict().items()))

    def test_rejects_bad_precision_and_training_contracts(self):
        for mode in ("body", "tower", "dfl", "stride", "training", "body_buffer", "tower_buffer", "dfl_buffer"):
            model = deepcopy(self.model)
            if mode == "body": model.model[0].half()
            if mode == "tower": model.model[-1].cv2.float()
            if mode == "dfl": model.model[-1].dfl.half()
            if mode == "stride": model.model[-1].stride = model.model[-1].stride.half()
            if mode == "training": model.model[0].train()
            if mode == "body_buffer": model.model[0].register_buffer("synthetic_buffer", torch.ones(1, dtype=torch.float16))
            if mode == "tower_buffer": model.model[-1].cv2[0].register_buffer("synthetic_buffer", torch.ones(1, dtype=torch.float32))
            if mode == "dfl_buffer": model.model[-1].dfl.register_buffer("synthetic_buffer", torch.ones(1, dtype=torch.float16))
            with self.subTest(mode=mode), self.assertRaises(ValueError): P.make_prototype(torch, model, D.decode_functional, anchors)

    def test_parameter_and_floating_buffer_policy_records_tensor_counts(self):
        self.model.model[0].register_buffer("synthetic_float", torch.ones(2, dtype=torch.float32))
        self.model.model[0].register_buffer("synthetic_counter", torch.ones(2, dtype=torch.int64))
        self.model.model[-1].cv2[0].register_buffer("synthetic_float", torch.ones(3, dtype=torch.float16))
        policy = P.parameter_buffer_policy(self.model)
        self.assertEqual(policy["body_floating_buffers"], {"torch.float32": 1})
        self.assertEqual(policy["head_tower_floating_buffers"], {"torch.float16": 1})
        self.assertEqual(policy["dfl_floating_buffers"], {})
        self.assertEqual(policy["body_parameters"], {"torch.float32": 46})
        self.assertEqual(policy["head_tower_parameters"], {"torch.float16": 12})
        self.assertEqual(policy["dfl_parameters"], {"torch.float32": 1})

    def test_rejects_non_fp32_input_and_non_fp32_decode(self):
        wrapper = P.make_prototype(torch, self.model, D.decode_functional, anchors)
        with self.assertRaises(ValueError): wrapper(self.input.half())
        def half_decode(*args): return {"decoded": torch.ones((1, 14, 48), dtype=torch.float16)}
        with self.assertRaises(ValueError): P.make_prototype(torch, self.model, half_decode, anchors)(self.input)

    def test_forward_hooks_removed_after_failure(self):
        wrapper = P.make_prototype(torch, self.model, D.decode_functional, anchors)
        self.input[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError): wrapper(self.input)
        self.assertTrue(all(not m._forward_hooks for m in self.model.model[:-1]))

    def test_exactly_one_checkpoint_load_and_separate_tower_casts(self):
        saved = Network().half()
        qualifier = SimpleNamespace(NAMES=[str(i) for i in range(10)], validate_unfused_structure=lambda _x: None)
        with patch.object(torch, "load", return_value={"ema": saved}) as mocked:
            model, provenance = P.load_mixed_model(torch, Path("synthetic_checkpoint.pt"), qualifier, device="cpu")
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(provenance["checkpoint_loads"], 1)
        self.assertTrue(all(v.dtype == torch.float32 for m in model.model[:-1] for v in m.parameters()))
        self.assertTrue(all(v.dtype == torch.float16 for tower in (model.model[-1].cv2, model.model[-1].cv3) for v in tower.parameters()))
        self.assertTrue(all(v.dtype == torch.float32 for v in model.model[-1].dfl.parameters()))


class ProtocolTests(unittest.TestCase):
    def test_valid_pilot_and_full(self):
        for stage in ("pilot3", "full109"):
            p, rows = fixture(stage)
            self.assertEqual([r["image_id"] for r in P.validate_protocol(p, "a" * 64, rows, D)], p["image_ids"])

    def test_rejects_changed_precision_helper_and_forbidden_roles(self):
        for key in ("precision_contract", "head_runner", "decoder_runner", "holdout_accessed", "selection_detector_inference", "raw_tolerances", "image_ids"):
            p, rows = fixture()
            if key == "precision_contract": p[key]["saved_frontier_inputs_permitted"] = True
            elif key in ("head_runner", "decoder_runner"): p[key]["sha256"] = "b" * 64
            elif key in ("holdout_accessed", "selection_detector_inference"): p[key] = True
            elif key == "raw_tolerances": p[key]["box_atol"] = 2
            elif key == "image_ids": p[key].reverse()
            with self.subTest(key=key), self.assertRaises(ValueError): P.validate_protocol(p, "a" * 64, rows, D)

    def test_new_check_name_and_branch_labels_preserve_historical_failure(self):
        class Q:
            @staticmethod
            def raw_guard(*_args):
                return {"guard_name": "old", "allclose_pass": False, "raw_geometry": {"fp32": {"a": 1}, "fp16": {"b": 2}}}
        result = P.prototype_guard(Q, np.zeros(1), np.ones(1))
        self.assertEqual(result["guard_name"], P.CHECK_NAME)
        self.assertFalse(result["allclose_pass"])
        self.assertFalse(result["historical_gate1_dtype_guard_replaced"])
        self.assertEqual(set(result["raw_geometry"]), {"historical_fp32", "body32_head16_decode32_prototype"})


if __name__ == "__main__": unittest.main()
