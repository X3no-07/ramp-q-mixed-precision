"""Synthetic saved arrays only; no real model, images, annotations or GPU."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("decode_probe_audit_under_test", ROOT / "scripts/audit_gate1_decode_probe.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
SYNTHETIC_TEMP = ROOT / "artifacts/gate1/20261023_decode_probe_preparation/independent_review/synthetic_temporary"
SYNTHETIC_TEMP.mkdir(parents=True, exist_ok=True)


def fixture():
    arrays = {"input_fp32_cpu": np.zeros((1, 3, 640, 640), dtype=np.float32)}
    for source in ("fp32", "fp16"):
        dtype = np.dtype("float32" if source == "fp32" else "float16")
        arrays[source + "_box_logits"] = np.zeros((1, 64, 8400), dtype=dtype)
        arrays[source + "_class_logits"] = np.zeros((1, 10, 8400), dtype=dtype)
        arrays[source + "_anchors"], arrays[source + "_strides"] = A.expected_geometry(dtype)
        arrays[source + "_dfl_bins"] = np.arange(16, dtype=dtype)
        for decode in ("fp32", "fp16"):
            dt = np.dtype("float32" if decode == "fp32" else "float16")
            p = f"source_{source}_decode_{decode}_"
            arrays[p + "probabilities"] = np.full((1, 16, 4, 8400), .0625, dtype=dt)
            distances = np.full((1, 4, 8400), 7.5, dtype=dt)
            arrays[p + "distances"] = distances
            anchor = arrays[source + "_anchors"].astype(dt)[None]
            xy1, xy2 = anchor - distances[:, :2], anchor + distances[:, 2:]
            xywh = np.concatenate(((xy1 + xy2) / np.array(2, dtype=dt), xy2 - xy1), axis=1)
            arrays[p + "xy1"], arrays[p + "xy2"], arrays[p + "xywh"] = xy1, xy2, xywh
            arrays[p + "boxes"] = xywh * arrays[source + "_strides"].astype(dt)
            arrays[p + "scores"] = np.full((1, 10, 8400), .5, dtype=dt)
            arrays[p + "decoded"] = np.concatenate((arrays[p + "boxes"], arrays[p + "scores"]), axis=1)
        arrays[source + "_forward"] = arrays[f"source_{source}_decode_{source}_decoded"].copy()
    meta = {"image_id": "train/synthetic", "reg_max": 16, "nc": 10, "nl": 3,
            "normalized_fp32_input_sha256": hashlib.sha256(arrays["input_fp32_cpu"].tobytes()).hexdigest(),
            "historical_byte_equal": {"fp32": True, "fp16": True}, "native_reconstruction_exact": {"fp32": True, "fp16": True},
            "runtime_geometry_equal_after_float_lift": True,
            "comparisons": {key: A.fixed_guard(arrays[left], arrays[right]) for key, (left, right) in A.COMPARISONS.items()}}
    history = {branch: arrays[branch + "_forward"].copy() for branch in ("fp32", "fp16")}
    return arrays, meta, history


class GuardTests(unittest.TestCase):
    def test_fixed_guard_uses_full_all_anchor_domain(self):
        raw = np.zeros((1, 14, 8400), dtype=np.float32)
        changed = raw.astype(np.float16)
        changed[0, 1, 8399] = 1
        result = A.fixed_guard(raw, changed)
        self.assertEqual(result["status"], "FAIL_FIXED_TOLERANCES")
        self.assertEqual(result["boxes"]["violations"], 1)
        self.assertEqual(result["boxes"]["elements"], 33600)
        self.assertEqual(result["scores"]["elements"], 84000)

    def test_guard_score_threshold_is_unchanged(self):
        raw = np.zeros((1, 14, 8400), dtype=np.float32)
        changed = raw.astype(np.float16)
        changed[0, 4, 0] = .003
        result = A.fixed_guard(raw, changed)
        self.assertEqual(result["status"], "FAIL_FIXED_TOLERANCES")
        self.assertEqual(result["scores"]["rtol"], .02)
        self.assertEqual(result["scores"]["atol"], .002)

    def test_guard_rejects_shape_and_nonfinite(self):
        raw = np.zeros((1, 14, 8400), dtype=np.float32)
        with self.assertRaises(ValueError):
            A.fixed_guard(raw[..., :10], raw[..., :10])
        raw[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            A.fixed_guard(raw, raw)

    def test_historical_bytes_detect_signed_zero_and_dtype(self):
        a = np.zeros(10, dtype=np.float32)
        b = a.copy(); b[0] = -0.
        self.assertFalse(A.exact_bytes(a, b))
        self.assertFalse(A.exact_bytes(a, a.astype(np.float16)))


class ArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays, cls.meta, cls.history = fixture()

    def test_complete_native_and_intervention_arrays_are_checked(self):
        result = A.audit_arrays(self.arrays, self.meta, self.history)
        self.assertTrue(result["historical_byte_equal"])
        self.assertEqual(set(result["comparisons"]), set(A.COMPARISONS))
        self.assertEqual(len(result["arithmetic_checks"]), 4)
        self.assertTrue(all(item["nonlinear_checks_are_approximate_only"] for item in result["arithmetic_checks"].values()))

    def changed(self, key, index, value):
        arrays = dict(self.arrays)
        arrays[key] = arrays[key].copy()
        arrays[key][index] = value
        with self.assertRaises(ValueError):
            A.audit_arrays(arrays, self.meta, self.history)

    def test_anchor_order_and_stride_level_are_exact(self):
        self.changed("fp16_anchors", (0, 7999), 1)
        self.changed("fp32_strides", (0, 8000), 16)

    def test_actual_dfl_bins_cannot_be_report_only_assertion(self):
        self.changed("fp32_dfl_bins", 15, 14)

    def test_input_normalization_hash_is_checked(self):
        self.changed("input_fp32_cpu", (0, 0, 0, 0), .25)

    def test_logit_mutation_is_detected_by_independent_softmax_equation(self):
        self.changed("fp32_box_logits", (0, 0, 100), .125)

    def test_sigmoid_mutation_is_detected(self):
        self.changed("fp16_class_logits", (0, 0, 100), .125)

    def test_corner_stage_corruption_is_detected(self):
        self.changed("source_fp16_decode_fp32_xy1", (0, 0, 100), 500)

    def test_geometry_reassociation_is_detected(self):
        self.changed("source_fp32_decode_fp16_xywh", (0, 2, 100), 500)

    def test_dfl_distance_mutation_is_detected(self):
        self.changed("source_fp32_decode_fp32_distances", (0, 0, 100), 10)

    def test_probability_corruption_is_detected(self):
        self.changed("source_fp16_decode_fp16_probabilities", (0, 0, 0, 100), .25)

    def test_native_or_historical_reconstruction_false_is_rejected(self):
        for group in ("native_reconstruction_exact", "historical_byte_equal"):
            meta = deepcopy(self.meta); meta[group]["fp32"] = False
            with self.subTest(group=group), self.assertRaises(ValueError):
                A.audit_arrays(self.arrays, meta, self.history)
        history = dict(self.history); history["fp32"] = history["fp32"].copy(); history["fp32"][0, 0, 0] += 1
        with self.assertRaises(ValueError):
            A.audit_arrays(self.arrays, self.meta, history)

    def test_reported_guard_pass_cannot_hide_actual_failure(self):
        meta = deepcopy(self.meta); meta["comparisons"][next(iter(A.COMPARISONS))]["boxes"]["violations"] = 3
        with self.assertRaises(ValueError):
            A.audit_arrays(self.arrays, meta, self.history)


class SummaryTests(unittest.TestCase):
    def make_summary(self):
        arrays, meta, history = fixture()
        row = A.audit_arrays(arrays, meta, history)
        aggregate = A.aggregate([row, row])
        summary = {"schema": "rampq.gate1.decode_precision_probe_summary/v1", "images": 2,
                   "historical_bytes_all_equal": True, "native_reconstruction_all_exact": True,
                   "raw_tolerances": A.RAW_TOLERANCES, "scientific_gate_pass": False, "original_guard_replaced": False,
                   "comparisons": {name: {"diagnostic_guard_pass_images": actual["pass_images"], "diagnostic_guard_fail_images": actual["fail_images"],
                                           **{section: {key: actual[section][key] for key in ("violations", "max_absolute_error", "max_error_over_tolerance")} for section in ("boxes", "scores")}}
                                   for name, actual in aggregate.items()}}
        return aggregate, summary

    def test_resummarizes_every_comparison_and_preserves_scope(self):
        aggregate, summary = self.make_summary()
        A.check_summary(aggregate, summary, 2)
        self.assertEqual(aggregate["original_fp32_vs_original_fp16"]["boxes"]["elements"], 67200)

    def test_summary_tamper_or_scope_expansion_rejected(self):
        aggregate, summary = self.make_summary()
        for key, value in (("images", 3), ("scientific_gate_pass", True), ("original_guard_replaced", True), ("raw_tolerances", {})):
            changed = deepcopy(summary); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                A.check_summary(aggregate, changed, 2)
        changed = deepcopy(summary); changed["comparisons"][next(iter(A.COMPARISONS))]["boxes"]["violations"] = 999
        with self.assertRaises(ValueError):
            A.check_summary(aggregate, changed, 2)


class ManifestTests(unittest.TestCase):
    def test_manifest_rejects_tampering_and_extra_files(self):
        with tempfile.TemporaryDirectory(dir=SYNTHETIC_TEMP) as folder:
            root = Path(folder).resolve(); source = root / "source.txt"; source.write_text("frozen", encoding="utf-8")
            A.write_json(root / "manifest.json", {"files": {"source.txt": {"bytes": source.stat().st_size, "sha256": A.sha(source)}}})
            self.assertEqual(A.verify_manifest(root)["files_verified"], 1)
            source.write_text("edited", encoding="utf-8")
            with self.assertRaises(ValueError):
                A.verify_manifest(root)
            source.write_text("frozen", encoding="utf-8"); (root / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaises(ValueError):
                A.verify_manifest(root)

    def test_safe_paths_reject_traversal_and_absolute(self):
        for value in ("../outside", str(ROOT / "scripts/audit_gate1_decode_probe.py")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                A.safe_file(ROOT, value)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.cohort = [{"image_id": f"train/synthetic_{i:03}", "assigned_split": "debug", "source_split": "train"} for i in range(109)]
        self.protocol = {"schema": "rampq.gate1.decode_precision_probe_protocol/v1", "data_role": "debug",
                         "selection_detector_inference": False, "holdout_accessed": False,
                         "raw_tolerances": A.RAW_TOLERANCES, "historical_equality_required": True,
                         "native_reconstruction_required": True, "stage": "pilot3",
                         "image_ids": [row["image_id"] for row in self.cohort[:3]]}

    def test_pilot_and_full_are_identity_bound_with_no_result_condition(self):
        self.assertEqual(len(A.validate_scope(self.protocol, self.cohort)), 109)
        self.protocol.update(stage="full109", image_ids=[row["image_id"] for row in self.cohort])
        A.validate_scope(self.protocol, self.cohort)

    def test_role_count_order_and_mandatory_exactness_cannot_change(self):
        for key, value in (("holdout_accessed", True), ("data_role", "selection"), ("historical_equality_required", False),
                           ("native_reconstruction_required", False), ("raw_tolerances", {}), ("image_ids", ["train/synthetic_000"])):
            changed = deepcopy(self.protocol); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                A.validate_scope(changed, self.cohort)
        self.protocol.update(stage="full109", image_ids=[row["image_id"] for row in self.cohort][::-1])
        with self.assertRaises(ValueError):
            A.validate_scope(self.protocol, self.cohort)


if __name__ == "__main__":
    unittest.main()
