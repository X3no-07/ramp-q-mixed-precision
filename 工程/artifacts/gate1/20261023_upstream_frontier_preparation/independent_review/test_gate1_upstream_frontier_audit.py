"""Synthetic DAG/tensor tests only; no real model, image, GPU or annotation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


A = load("upstream_frontier_auditor_under_test", ROOT / "scripts/audit_gate1_upstream_frontier.py")
F = load("frozen_decode_fixture_for_frontier", ROOT / "tests/test_gate1_decode_probe_audit.py")


def topology():
    fs = [-1] * 24
    for i, f in ((12, [-1, 6]), (15, [-1, 4]), (18, [-1, 13]), (21, [-1, 10]), (23, [16, 19, 22])):
        fs[i] = f
    return {"modules": [{"i": i, "f": f, "type": "Detect" if i == 23 else "SyntheticModule"} for i, f in enumerate(fs)],
            "save": [4, 6, 10, 13, 16, 19, 22]}


def fixture():
    decoded, old_meta, history = F.fixture()
    reference = {key: value for key, value in decoded.items() if not key.startswith("source_")}
    arrays = dict(reference)
    for source in ("fp32", "fp16"):
        dtype = np.dtype("float32" if source == "fp32" else "float16")
        reference[f"source_{source}_head_{source}_decoded"] = decoded[f"source_{source}_decode_fp32_decoded"]
        arrays[f"native_{source}_decoded"] = reference[f"source_{source}_head_{source}_decoded"]
        for kind in ("forward", "box_logits", "class_logits"):
            arrays[f"manual_{source}_{kind}"] = arrays[f"{source}_{kind}"].copy()
        for index in (-1, 4, 6, 10, 16, 19, 22):
            value = arrays["input_fp32_cpu"].astype(dtype) if index == -1 else np.full((1, 1, 2, 2), .3333, dtype=dtype)
            if index in (16, 19, 22):
                scale = (16, 19, 22).index(index)
                value = np.zeros((1, 1, (80, 40, 20)[scale], (80, 40, 20)[scale]), dtype=dtype)
                reference[f"{source}_feat_scale{scale}"] = value.copy()
            arrays[f"native_{source}_layer_{index}"] = value
    meta = {"image_id": "train/synthetic", "reg_max": 16, "nc": 10, "nl": 3,
            "normalized_fp32_input_sha256": old_meta["normalized_fp32_input_sha256"],
            "old_guard_replaced": False, "tolerances_changed": False, "runtime_geometry_equal_after_float_lift": True,
            "model_states_unchanged": {"fp32": True, "fp16": True},
            **{key: {"fp32": True, "fp16": True} for key in ("historical_byte_equal", "reference_head_capture_exact", "reference_unified_decode_exact", "native_manual_replay_exact")},
            "native_suffix_replay_exact": {str(c): {"fp32": True, "fp16": True} for c in A.CUTS},
            "native_suffix_replay_sha256": {}, "variant_frontier_casts": {},
            "native_manual_replay_sha256": {}, "native_manual_module_replay_exact": {}, "native_manual_module_replay_sha256": {}, "native_layer_statistics": {},
            "frontier_sha256_before": {}, "frontier_sha256_after": {}}
    for source in ("fp32", "fp16"):
        hashes = {str(i): A.tensor_hash(arrays[f"native_{source}_layer_{i}"]) for i in (-1, 4, 6, 10, 16, 19, 22)}
        meta["frontier_sha256_before"][source] = hashes
        meta["frontier_sha256_after"][source] = dict(hashes)
        meta["native_manual_replay_sha256"][source] = {key: A.tensor_hash(arrays[f"{source}_{key}"]) for key in ("forward", "box_logits", "class_logits")}
        dtype = np.dtype("float32" if source == "fp32" else "float16")
        stats = {str(i): A.tensor_statistics(arrays.get(f"native_{source}_layer_{i}", np.zeros((1, 1, 2, 2), dtype=dtype))) for i in range(-1, 23)}
        meta["native_layer_statistics"][source] = stats
        meta["native_manual_module_replay_exact"][source] = {str(i): True for i in range(23)}
        meta["native_manual_module_replay_sha256"][source] = {str(i): stats[str(i)]["sha256"] for i in range(23)}
    for cut in A.CUTS:
        meta["native_suffix_replay_sha256"][str(cut)] = {source: {**{key: A.tensor_hash(arrays[f"{source}_{key}"]) for key in ("box_logits", "class_logits")},
                                                              "decoded": A.tensor_hash(arrays[f"native_{source}_decoded"])} for source in ("fp32", "fp16")}
    for cut, source, suffix in A.VARIANTS:
        tag = A.prefix(cut, source, suffix)
        for key in ("box_logits", "class_logits"):
            arrays[tag + key] = arrays[f"{suffix}_{key}"]
        arrays[tag + "decoded"] = arrays[f"native_{suffix}_decoded"]
        if cut == 22:
            for key in ("box_logits", "class_logits", "decoded"): reference[f"source_{source}_head_{suffix}_{key}"] = arrays[tag + key].copy()
        dtype = np.dtype("float32" if suffix == "fp32" else "float16")
        meta["variant_frontier_casts"][tag.removesuffix("_")] = {str(i): A.tensor_statistics(arrays[f"native_{source}_layer_{i}"].astype(dtype)) for i in A.EXPECTED_FRONTIERS[str(cut)]}
    meta["endpoint_aliases_exact"] = {"cut22_fp32_to_fp16": True, "cut22_fp16_to_fp32": True, "input_fp32_to_fp16": True}
    meta["comparisons"] = {name: A.D.fixed_guard(arrays[left], arrays[right]) for name, (left, right) in A.COMPARISONS.items()}
    return arrays, meta, reference, history


class GraphTests(unittest.TestCase):
    def test_complete_frontiers_include_every_skip(self):
        self.assertEqual(A.validate_topology(topology()), A.EXPECTED_FRONTIERS)
        self.assertEqual(A.derive_frontier(topology(), 10), [4, 6, 10])

    def test_missing_skip_changes_frontier_and_is_rejected(self):
        graph = topology(); graph["modules"][15]["f"] = -1; graph["save"].remove(4)
        self.assertEqual(A.derive_frontier(graph, 6), [6])
        with self.assertRaises(ValueError): A.validate_topology(graph)

    def test_save_cache_cannot_omit_crossing_dependencies(self):
        graph = topology(); graph["save"].remove(4)
        with self.assertRaises(ValueError): A.derive_frontier(graph, 6)

    def test_forward_invalid_negative_and_noninteger_edges_rejected(self):
        for f in (99, -999, [True], None):
            graph = topology(); graph["modules"][3]["f"] = f
            with self.subTest(f=f), self.assertRaises((ValueError, TypeError)): A.derive_frontier(graph, 4)

    def test_toy_dag_replay_requires_complete_frontier(self):
        fs = [-1, -1, 0, [-1, 1], [-1, 0]]
        graph = {"modules": [{"i": i, "f": f, "type": "SyntheticSum"} for i, f in enumerate(fs)], "save": [0, 1]}
        self.assertEqual(A.derive_frontier(graph, 1), [0, 1])
        full = {-1: np.array([.125], dtype=np.float32)}
        for i, f in enumerate(fs): full[i] = sum(full[j] for j in A.dependencies(i, f)) + i + 1
        replay = {i: full[i].copy() for i in A.derive_frontier(graph, 1)}
        for i in range(2, 5): replay[i] = sum(replay[j] for j in A.dependencies(i, fs[i])) + i + 1
        np.testing.assert_array_equal(replay[4], full[4])
        del replay[0]
        with self.assertRaises(KeyError): sum(replay[j] for j in A.dependencies(4, fs[4]))


class ArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.arrays, cls.meta, cls.reference, cls.history = fixture()

    def check(self, arrays=None, meta=None, reference=None, history=None):
        return A.audit_arrays(arrays or self.arrays, meta or self.meta, reference or self.reference, history or self.history, topology())

    def corrupt(self, key, index, value):
        arrays = dict(self.arrays); arrays[key] = arrays[key].copy(); arrays[key][index] = value
        with self.assertRaises(ValueError): self.check(arrays=arrays)

    def test_all12guards_and10arithmetic_checks_are_recomputed(self):
        result = self.check(); self.assertEqual(len(result["comparisons"]), 12)
        self.assertEqual(len(result["arithmetic_checks"]), 10)
        self.assertTrue(result["all_native_historical_reference_and_cast_bytes_exact"])

    def test_original_history_and_reference_are_byte_exact(self):
        self.corrupt("fp16_forward", (0, 0, 0), .25)
        self.corrupt("fp32_anchors", (0, 0), .25)
        self.corrupt("native_fp32_decoded", (0, 0, 0), .25)

    def test_manual_full_replay_is_not_accepted_from_true_flag(self):
        self.corrupt("manual_fp32_box_logits", (0, 0, 0), .25)

    def test_native_detect_frontier_matches_reference_head_feature(self):
        self.corrupt("native_fp32_layer_16", (0, 0, 0, 0), .25)

    def test_input_frontier_is_exact_branch_input_cast(self):
        self.corrupt("native_fp16_layer_-1", (0, 0, 0, 0), .25)

    def test_native_frontier_hash_before_and_after_bind_bytes(self):
        self.corrupt("native_fp32_layer_4", (0, 0, 0, 0), .25)
        meta = deepcopy(self.meta); meta["frontier_sha256_after"]["fp32"]["4"] = "0" * 64
        with self.assertRaises(ValueError): self.check(meta=meta)

    def test_crossed_cast_hash_and_all_skip_membership_independently_checked(self):
        key = A.prefix(6, "fp16", "fp32").removesuffix("_")
        meta = deepcopy(self.meta); meta["variant_frontier_casts"][key]["4"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): self.check(meta=meta)
        meta = deepcopy(self.meta); del meta["variant_frontier_casts"][key]["4"]
        with self.assertRaises(ValueError): self.check(meta=meta)

    def test_same_dtype_suffix_hash_must_match_native_output(self):
        meta = deepcopy(self.meta); meta["native_suffix_replay_sha256"]["10"]["fp16"]["decoded"] = "0" * 64
        with self.assertRaises(ValueError): self.check(meta=meta)

    def test_all_native_module_reports_and_union_stats_checked(self):
        meta = deepcopy(self.meta); meta["native_manual_module_replay_exact"]["fp32"]["0"] = False
        with self.assertRaises(ValueError): self.check(meta=meta)
        meta = deepcopy(self.meta); meta["native_manual_module_replay_sha256"]["fp16"]["4"] = "0" * 64
        with self.assertRaises(ValueError): self.check(meta=meta)
        meta = deepcopy(self.meta); meta["native_layer_statistics"]["fp16"]["4"]["mean"] = 100.
        with self.assertRaises(ValueError): self.check(meta=meta)

    def test_input_cast_to_fp16_control_requires_native16_output_bytes(self):
        self.corrupt(A.prefix(-1, "fp32", "fp16") + "box_logits", (0, 0, 0), .25)

    def test_cut22_endpoint_reuses_prior_head_intervention_bytes(self):
        self.corrupt(A.prefix(22, "fp16", "fp32") + "box_logits", (0, 0, 0), .25)

    def test_approx_decoder_sanity_checks_class_and_box_logit_relation(self):
        self.corrupt(A.prefix(6, "fp16", "fp32") + "class_logits", (0, 0, 0), .25)
        self.corrupt(A.prefix(6, "fp16", "fp32") + "decoded", (0, 0, 0), 10.)

    def test_dtype_nonfinite_and_guard_tampering_rejected(self):
        key = A.prefix(6, "fp16", "fp32") + "box_logits"
        arrays = dict(self.arrays); arrays[key] = arrays[key].astype(np.float16)
        with self.assertRaises(ValueError): self.check(arrays=arrays)
        self.corrupt(key, (0, 0, 0), np.nan)
        meta = deepcopy(self.meta); meta["comparisons"][next(iter(A.COMPARISONS))]["boxes"]["violations"] = 1
        with self.assertRaises(ValueError): self.check(meta=meta)

    def test_scope_flags_and_native_exactness_are_required(self):
        for key in ("old_guard_replaced", "tolerances_changed"):
            meta = deepcopy(self.meta); meta[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(meta=meta)
        meta = deepcopy(self.meta); meta["native_suffix_replay_exact"]["4"]["fp32"] = False
        with self.assertRaises(ValueError): self.check(meta=meta)


class ScopeSummaryTests(unittest.TestCase):
    def test_scope_and_fixed_intervention_enforced(self):
        cohort = [{"image_id": f"train/synthetic_{i:03}", "assigned_split": "debug", "source_split": "train"} for i in range(109)]
        for row, name in zip(cohort, ["train/0000170_01201_d_0000001", "train/9999990_00000_d_0000024", "train/0000046_00000_d_0000087"]): row["image_id"] = name
        p = {"schema": "rampq.gate1.upstream_frontier_probe_protocol/v1", "cuts": A.CUTS, "frontiers": A.EXPECTED_FRONTIERS, "topology": topology(),
             "decoder_runner": {"sha256": A.DECODE_RUNNER_SHA}, "head_runner": {"sha256": A.HEAD_RUNNER_SHA}, "intervention": A.INTERVENTION,
             "data_role": "debug", "selection_detector_inference": False, "holdout_accessed": False, "raw_tolerances": A.D.RAW_TOLERANCES,
             "historical_equality_required": True, "native_reconstruction_required": True, "stage": "pilot3", "image_ids": [r["image_id"] for r in cohort[:3]]}
        self.assertEqual(len(A.validate_scope(p, cohort)), 109)
        for key, value in (("intervention", {}), ("holdout_accessed", True), ("cuts", [4]), ("raw_tolerances", {})):
            changed = deepcopy(p); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): A.validate_scope(changed, cohort)

    def test_summary_recomputed_with_no_scientific_pass(self):
        arrays, meta, ref, history = fixture(); row = A.audit_arrays(arrays, meta, ref, history, topology()); aggregate = A.aggregate([row])
        summary = {"schema": "rampq.gate1.upstream_frontier_probe_summary/v1", "images": 1, "cuts": A.CUTS, "frontiers": A.EXPECTED_FRONTIERS,
                   **{key: True for key in A.SUMMARY_EXACT}, "all_intervention_decode_dtype": "fp32", "raw_tolerances": A.D.RAW_TOLERANCES,
                   "original_guard_replaced": False, "scientific_gate_pass": False,
                   "comparisons": {name: {"diagnostic_guard_pass_images": actual["pass_images"], "diagnostic_guard_fail_images": actual["fail_images"],
                                          **{section: {key: actual[section][key] for key in ("violations", "max_absolute_error", "max_error_over_tolerance")} for section in ("boxes", "scores")}} for name, actual in aggregate.items()}}
        A.check_summary(aggregate, summary, 1)
        for key, value in (("scientific_gate_pass", True), ("original_guard_replaced", True), ("frontiers", {})):
            changed = deepcopy(summary); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): A.check_summary(aggregate, changed, 1)
        changed = deepcopy(summary); changed["comparisons"][next(iter(A.COMPARISONS))]["boxes"]["violations"] = 1
        with self.assertRaises(ValueError): A.check_summary(aggregate, changed, 1)


if __name__ == "__main__": unittest.main()
