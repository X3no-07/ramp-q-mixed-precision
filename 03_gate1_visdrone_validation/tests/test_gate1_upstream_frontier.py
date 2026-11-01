"""Synthetic CPU DAG tests. No real model, checkpoint, image, or GPU use."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


U = load("upstream_probe_under_test", ROOT / "scripts/probe_gate1_upstream_frontier.py")
D = load("frozen_decoder_for_upstream_tests", ROOT / "scripts/probe_gate1_decode_precision.py")
SOURCES = [-1] * 24
for index, sources in {12: [-1, 6], 15: [-1, 4], 18: [-1, 13], 21: [-1, 10], 23: [16, 19, 22]}.items():
    SOURCES[index] = sources
SAVE = [4, 6, 10, 13, 16, 19, 22]


class Layer(torch.nn.Module):
    def __init__(self, index, sources):
        super().__init__()
        self.i, self.f = index, deepcopy(sources)
        channels = 2 * len(sources) if isinstance(sources, list) else 2
        self.conv = torch.nn.Conv2d(channels, 2, 1)

    def forward(self, value):
        if isinstance(value, list):
            value = torch.cat(value, 1)
        return self.conv(value).sin()


class DAG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList([Layer(i, f) for i, f in enumerate(SOURCES)])
        self.save = SAVE.copy()
        self.half().float().eval()

    def forward(self, value, augment=False):
        cache = {-1: value}
        for i, module in enumerate(self.model):
            # Independent native dispatch reproduces the framework's f syntax.
            if module.f == -1:
                current = cache[i - 1]
            elif isinstance(module.f, int):
                current = cache[module.f]
            else:
                current = [cache[i - 1] if source == -1 else cache[source] for source in module.f]
            cache[i] = module(current)
        return cache[23]


def fixture(stage="pilot3"):
    topology = U.topology_of_model(DAG())
    rows = [{"image_id": f"train/fixture{i:03}", "assigned_split": "debug", "source_split": "train",
             "image_path": f"images/train/fixture{i:03}.jpg"} for i in range(109)]
    for i, image_id in enumerate(U.PILOT_IDS):
        rows[i].update(image_id=image_id, image_path=f"images/{image_id}.jpg")
    p = {"schema": U.SCHEMA, "runner_sha256": "a" * 64, "decoder_runner": {"sha256": U.DECODER_SHA},
         "head_runner": {"sha256": U.HEAD_SHA}, "intervention": deepcopy(U.INTERVENTION), "cuts": U.CUTS.copy(),
         "frontiers": deepcopy(U.FRONTIERS), "topology": topology, "controls": deepcopy(D.CONTROLS),
         "raw_tolerances": deepcopy(D.TOLERANCES), "qualifier": {"sha256": D.QUALIFIER_SHA},
         "checkpoint": {"sha256": D.CHECKPOINT_SHA}, "data_role": "debug", "selection_detector_inference": False,
         "holdout_accessed": False, "historical_equality_required": True, "native_reconstruction_required": True,
         "stage": stage, "image_ids": U.PILOT_IDS.copy() if stage == "pilot3" else [r["image_id"] for r in rows],
         "package_versions": {"torch": "synthetic"}, "framework_source_sha256": {"tasks.py": "b" * 64}}
    return p, rows


class TopologyTests(unittest.TestCase):
    def test_all_dag_crossing_frontiers(self):
        topology = U.topology_of_model(DAG())
        self.assertEqual({str(c): U.frontier_indices(topology, c) for c in U.CUTS}, U.FRONTIERS)
        self.assertEqual(topology["save"], SAVE)

    def test_relative_negative_edges(self):
        topology = {"modules": [{"i": 0, "f": -1, "type": "tensor"}, {"i": 1, "f": -1, "type": "tensor"},
                                {"i": 2, "f": [-1, -2], "type": "concat"}], "save": [0]}
        U.validate_topology(topology)
        self.assertEqual(U.frontier_indices(topology, 0), [0])
        self.assertEqual(U.frontier_indices(topology, 1), [0, 1])

    def test_reject_invalid_dependencies(self):
        for bad in (24, -30, True, []):
            topology = U.topology_of_model(DAG())
            topology["modules"][6]["f"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError): U.validate_topology(topology)

    def test_skip_save_must_cover_non_immediate_dependencies(self):
        topology = U.topology_of_model(DAG())
        topology["save"].remove(4)
        with self.assertRaises(ValueError): U.validate_topology(topology)

    def test_reject_unbounded_or_final_cut(self):
        topology = U.topology_of_model(DAG())
        for cut in (-2, 23, 24):
            with self.assertRaises(ValueError): U.frontier_indices(topology, cut)


class SyntheticReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = DAG()
        self.topology = U.topology_of_model(self.model)
        self.input = (torch.arange(32).reshape(1, 2, 4, 4).float() - 11) / 37

    def test_complete_native_replay_exact_both_dtypes(self):
        for dtype in (torch.float32, torch.float16):
            model = deepcopy(self.model).to(dtype=dtype)
            tensor = self.input.to(dtype=dtype)
            native = model(tensor)
            replay = U.replay_suffix(torch, model, self.topology, -1, {-1: tensor}, dtype)
            self.assertTrue(U.exact(torch, native, replay))

    def test_every_same_dtype_frontier_replay_exact(self):
        for dtype in (torch.float32, torch.float16):
            model = deepcopy(self.model).to(dtype=dtype)
            native, cache, _ = U.capture_native(torch, model, self.input.to(dtype=dtype), self.topology)
            for cut in U.CUTS:
                frontier = {i: cache[i] for i in U.FRONTIERS[str(cut)]}
                replay = U.replay_suffix(torch, model, self.topology, cut, frontier, dtype)
                self.assertTrue(U.exact(torch, native, replay), f"{dtype}:{cut}")

    def test_complete_manual_replay_captures_each_native_prefix_module_exactly(self):
        for dtype in (torch.float32, torch.float16):
            model = deepcopy(self.model).to(dtype=dtype)
            native, cache, _ = U.capture_native(torch, model, self.input.to(dtype=dtype), self.topology)
            captured = {}
            replay = U.replay_suffix(torch, model, self.topology, -1, {-1: cache[-1]}, dtype, captured)
            self.assertEqual(set(captured), set(range(23)))
            self.assertTrue(all(U.exact(torch, value, cache[i]) for i, value in captured.items()))
            self.assertTrue(U.exact(torch, native, replay))

    def test_crossed_replay_casts_all_skip_inputs_and_preserves_state(self):
        models = {"fp32": self.model, "fp16": deepcopy(self.model).half()}
        caches = {s: U.capture_native(torch, m, self.input.to(dtype=next(m.parameters()).dtype), self.topology)[1]
                  for s, m in models.items()}
        before = {s: {i: x.clone() for i, x in cache.items()} for s, cache in caches.items()}
        states = {s: {k: v.clone() for k, v in m.state_dict().items()} for s, m in models.items()}
        for cut in U.CUTS:
            for source, suffix in U.PAIRS:
                dtype = torch.float32 if suffix == "fp32" else torch.float16
                result = U.replay_suffix(torch, models[suffix], self.topology, cut,
                                         {i: caches[source][i] for i in U.FRONTIERS[str(cut)]}, dtype)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(all(U.exact(torch, x, before[s][i]) for s, cache in caches.items() for i, x in cache.items()))
        self.assertTrue(all(U.exact(torch, x, states[s][k]) for s, m in models.items() for k, x in m.state_dict().items()))

    def test_missing_or_extra_frontier_rejected(self):
        _, cache, _ = U.capture_native(torch, self.model, self.input, self.topology)
        with self.assertRaises(ValueError): U.replay_suffix(torch, self.model, self.topology, 10, {10: cache[10]}, torch.float32)
        frontier = {i: cache[i] for i in (4, 6, 10, 11)}
        with self.assertRaises(ValueError): U.replay_suffix(torch, self.model, self.topology, 10, frontier, torch.float32)

    def test_training_dtype_graph_and_nonfinite_rejected(self):
        for mode in ("train", "dtype", "graph", "nonfinite"):
            model, topology, tensor = deepcopy(self.model), deepcopy(self.topology), self.input.clone()
            dtype = torch.float32
            if mode == "train": model.train()
            if mode == "dtype": dtype = torch.float16
            if mode == "graph": topology["modules"][1]["f"] = 0
            if mode == "nonfinite": tensor[0, 0, 0, 0] = float("nan")
            with self.subTest(mode=mode), self.assertRaises(ValueError): U.replay_suffix(torch, model, topology, -1, {-1: tensor}, dtype)

    def test_capture_hooks_removed_after_forward_failure(self):
        class Broken(DAG):
            def forward(self, *args, **kwargs): raise RuntimeError("synthetic failure")
        model = Broken()
        with self.assertRaises(RuntimeError): U.capture_native(torch, model, self.input, U.topology_of_model(model))
        self.assertTrue(all(not module._forward_hooks for module in model.model))

    def test_statistics_and_exactness(self):
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        stats = U.tensor_statistics(tensor)
        self.assertEqual(stats, {"shape": [3], "dtype": "float16", "elements": 3, "min": 1., "max": 3., "mean": 2.,
                                 "sha256": U.array_sha(tensor.numpy())})
        self.assertTrue(U.exact(torch, tensor, tensor.clone()))
        self.assertFalse(U.exact(torch, tensor, tensor.float()))

    def test_repeated_endpoint_aliases_check_actual_bytes(self):
        arrays, reference = {}, {}
        for source, suffix in U.PAIRS:
            for part in ("box_logits", "class_logits", "decoded"):
                value = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
                arrays[U.variant_key(22, source, suffix) + "_" + part] = value.copy()
                reference[f"source_{source}_head_{suffix}_{part}"] = value.copy()
        for part in ("box_logits", "class_logits", "decoded"):
            value = np.arange(4, dtype=np.float16)
            arrays[U.variant_key(-1, "fp32", "fp16") + "_" + part] = value.copy()
            arrays[f"fp16_{part}" if part != "decoded" else "native_fp16_decoded"] = value.copy()
        self.assertTrue(all(U.endpoint_aliases(arrays, reference).values()))
        arrays[U.variant_key(22, "fp32", "fp16") + "_decoded"][0, 0, 0] += 1
        self.assertFalse(U.endpoint_aliases(arrays, reference)["cut22_fp32_to_fp16"])


class ProtocolTests(unittest.TestCase):
    def test_pilot_and_full_valid(self):
        for stage in ("pilot3", "full109"):
            p, rows = fixture(stage)
            self.assertEqual([r["image_id"] for r in U.validate_protocol(p, "a" * 64, rows, D)], p["image_ids"])

    def test_reject_changes_to_frozen_intervention_or_source(self):
        for key in ("runner_sha256", "head_runner", "decoder_runner", "cuts", "frontiers", "intervention", "topology"):
            p, rows = fixture()
            if key == "runner_sha256": p[key] = "b" * 64
            elif key in ("head_runner", "decoder_runner"): p[key]["sha256"] = "b" * 64
            elif key == "cuts": p[key].append(16)
            elif key == "frontiers": p[key]["10"] = [10]
            elif key == "intervention": p[key]["all_comparisons_decode_dtype"] = "fp16"
            elif key == "topology": p[key]["modules"][15]["f"] = [-1, 3]
            with self.subTest(key=key), self.assertRaises(ValueError): U.validate_protocol(p, "a" * 64, rows, D)

    def test_reject_pilot_reordering_and_forbidden_roles(self):
        for key in ("image_ids", "holdout_accessed", "selection_detector_inference", "controls", "raw_tolerances"):
            p, rows = fixture()
            if key == "image_ids": p[key].reverse()
            elif key in ("holdout_accessed", "selection_detector_inference"): p[key] = True
            elif key == "controls": p[key]["tf32"] = True
            elif key == "raw_tolerances": p[key]["box_atol"] = 2
            with self.subTest(key=key), self.assertRaises(ValueError): U.validate_protocol(p, "a" * 64, rows, D)

    def test_exactly_ten_crossed_variants_and_twelve_comparisons(self):
        self.assertEqual(len(U.COMPARISONS), 12)
        self.assertEqual(len(U.PAIRS) * len(U.CUTS), 10)
        self.assertTrue(all(left != right for left, right in U.PAIRS))
        self.assertTrue(all(right.endswith("decoded") for name, (_, right) in U.COMPARISONS.items() if name != "original_fp32_vs_original_fp16"))


if __name__ == "__main__":
    unittest.main()
