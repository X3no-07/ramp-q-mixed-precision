"""Synthetic head arrays only. No real model, images, GPU or annotations."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load("head_saved_array_auditor_under_test", ROOT / "scripts/audit_gate1_head_probe.py")
F = load("frozen_decode_synthetic_fixture", ROOT / "tests/test_gate1_decode_probe_audit.py")


def fixture():
    reference, old_meta, historical = F.fixture()
    arrays = {key: value for key, value in reference.items() if key == "input_fp32_cpu" or not key.startswith("source_")}
    for source in ("fp32", "fp16"):
        for head in ("fp32", "fp16"):
            prefix = f"source_{source}_head_{head}_"
            for logit in ("box_logits", "class_logits"):
                arrays[prefix + logit] = reference[head + "_" + logit].copy()
            for stage in ("probabilities", "distances", "xy1", "xy2", "xywh", "boxes", "scores", "decoded"):
                arrays[prefix + stage] = reference[f"source_{source}_decode_fp32_{stage}"].copy()
        for scale, length in enumerate((80, 40, 20)):
            arrays[f"{source}_feat_scale{scale}"] = np.zeros((1, 1, length, length), dtype=np.float32 if source == "fp32" else np.float16)
    meta = {"image_id": old_meta["image_id"], "reg_max": 16, "nc": 10, "nl": 3,
            "normalized_fp32_input_sha256": old_meta["normalized_fp32_input_sha256"], "runtime_geometry_equal_after_float_lift": True,
            "old_guard_replaced": False, "tolerances_changed": False,
            **{key: {"fp32": True, "fp16": True} for key in A.EXACT_REPORT_GROUPS},
            "feature_shapes": {s: [list(arrays[f"{s}_feat_scale{scale}"].shape) for scale in range(3)] for s in ("fp32", "fp16")},
            "feature_sha256_before": {s: [A.feature_hash(arrays[f"{s}_feat_scale{scale}"]) for scale in range(3)] for s in ("fp32", "fp16")},
            "feature_sha256_after": {s: [A.feature_hash(arrays[f"{s}_feat_scale{scale}"]) for scale in range(3)] for s in ("fp32", "fp16")},
            "comparisons": {name: A.D.fixed_guard(arrays[left], arrays[right]) for name, (left, right) in A.COMPARISONS.items()},
            "logit_differences": {}}
    for name, (left, right) in A.COMPARISONS.items():
        if left.endswith("_decoded"):
            meta["logit_differences"][name] = {key: {"elements": arrays[left.removesuffix("decoded") + key].size,
                                                   "unequal_elements": 0, "max_absolute_error": 0., "mean_absolute_error": 0.}
                                               for key in ("box_logits", "class_logits")}
    return arrays, meta, reference, historical


class HeadArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays, cls.meta, cls.reference, cls.history = fixture()

    def check(self, arrays=None, meta=None, reference=None, history=None):
        return A.audit_arrays(arrays or self.arrays, meta or self.meta, reference or self.reference, history or self.history)

    def corrupt(self, key, index, value):
        arrays = dict(self.arrays); arrays[key] = arrays[key].copy(); arrays[key][index] = value
        with self.assertRaises(ValueError):
            self.check(arrays=arrays)

    def test_all_eight_fixed_comparisons_are_resummarized(self):
        row = self.check()
        self.assertEqual(set(row["comparisons"]), set(A.COMPARISONS))
        self.assertEqual(len(row["arithmetic_checks"]), 4)
        self.assertTrue(row["all_native_historical_reference_byte_checks"])

    def test_saved_feature_hash_binds_before_and_after(self):
        self.corrupt("fp32_feat_scale0", (0, 0, 0, 0), .5)
        meta = deepcopy(self.meta); meta["feature_sha256_after"]["fp32"][0] = "0" * 64
        with self.assertRaises(ValueError):
            self.check(meta=meta)

    def test_feature_dtype_and_spatial_shape_checked(self):
        arrays = dict(self.arrays); arrays["fp16_feat_scale2"] = arrays["fp16_feat_scale2"].astype(np.float32)
        with self.assertRaises(ValueError):
            self.check(arrays=arrays)
        arrays = dict(self.arrays); arrays["fp16_feat_scale2"] = arrays["fp16_feat_scale2"][..., :10]
        with self.assertRaises(ValueError):
            self.check(arrays=arrays)

    def test_reference_capture_byte_identity_checked(self):
        self.corrupt("fp16_box_logits", (0, 0, 0), .5)
        self.corrupt("fp32_dfl_bins", 15, 14)

    def test_native_reexecution_logits_are_checked_without_report_trust(self):
        self.corrupt("source_fp32_head_fp32_box_logits", (0, 0, 0), .5)

    def test_cross_head_logit_dtype_checked(self):
        arrays = dict(self.arrays); arrays["source_fp32_head_fp16_box_logits"] = arrays["source_fp32_head_fp16_box_logits"].astype(np.float32)
        with self.assertRaises(ValueError):
            self.check(arrays=arrays)

    def test_cross_head_decode_math_checked(self):
        self.corrupt("source_fp32_head_fp16_class_logits", (0, 0, 0), .5)
        self.corrupt("source_fp16_head_fp32_xywh", (0, 0, 0), .75)

    def test_original_historical_failure_cannot_be_replaced_by_intervention(self):
        history = {key: value.copy() for key, value in self.history.items()}; history["fp16"][0, 0, 0] += 1
        with self.assertRaises(ValueError):
            self.check(history=history)

    def test_every_exactness_group_is_required(self):
        for key in A.EXACT_REPORT_GROUPS:
            meta = deepcopy(self.meta); meta[key]["fp16"] = False
            with self.subTest(group=key), self.assertRaises(ValueError):
                self.check(meta=meta)

    def test_reported_guard_or_logit_statistics_tampering_is_detected(self):
        meta = deepcopy(self.meta); meta["comparisons"][next(iter(A.COMPARISONS))]["boxes"]["violations"] = 1
        with self.assertRaises(ValueError):
            self.check(meta=meta)
        meta = deepcopy(self.meta); meta["logit_differences"]["native_heads_both_decode_fp32"]["box_logits"]["unequal_elements"] = 1
        with self.assertRaises(ValueError):
            self.check(meta=meta)

    def test_scope_flags_stay_false(self):
        for key in ("old_guard_replaced", "tolerances_changed"):
            meta = deepcopy(self.meta); meta[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(meta=meta)

    def test_nonfinite_intervention_is_rejected(self):
        self.corrupt("source_fp16_head_fp32_box_logits", (0, 0, 0), np.nan)


class HeadScopeTests(unittest.TestCase):
    def test_scope_and_intervention_cannot_be_redefined(self):
        cohort = [{"image_id": f"train/synthetic_{i:03}", "assigned_split": "debug", "source_split": "train"} for i in range(109)]
        protocol = {"schema": "rampq.gate1.head_precision_probe_protocol/v1", "data_role": "debug", "selection_detector_inference": False,
                    "holdout_accessed": False, "raw_tolerances": A.D.RAW_TOLERANCES, "historical_equality_required": True,
                    "native_reconstruction_required": True, "stage": "pilot3", "image_ids": [row["image_id"] for row in cohort[:3]],
                    "decoder_runner": {"sha256": A.DECODE_RUNNER_SHA}, "intervention": A.INTERVENTION}
        self.assertEqual(len(A.validate_scope(protocol, cohort)), 109)
        for key, value in (("intervention", {}), ("holdout_accessed", True), ("raw_tolerances", {})):
            changed = deepcopy(protocol); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                A.validate_scope(changed, cohort)


class HeadSummaryTests(unittest.TestCase):
    def make_summary(self):
        arrays, meta, reference, history = fixture()
        row = A.audit_arrays(arrays, meta, reference, history)
        aggregate = A.aggregate([row])
        summary = {"schema": "rampq.gate1.head_precision_probe_summary/v1", "images": 1,
                   **{key: True for key in ("historical_bytes_all_equal", "reference_decode_captures_all_exact", "reference_unified_decode_all_exact", "native_head_logits_all_exact", "features_and_head_states_unchanged")},
                   "raw_tolerances": A.D.RAW_TOLERANCES, "all_intervention_decode_dtype": "fp32", "original_guard_replaced": False, "scientific_gate_pass": False,
                   "comparisons": {name: {"diagnostic_guard_pass_images": actual["pass_images"], "diagnostic_guard_fail_images": actual["fail_images"],
                                           **{section: {key: actual[section][key] for key in ("violations", "max_absolute_error", "max_error_over_tolerance")} for section in ("boxes", "scores")}}
                                   for name, actual in aggregate.items()}}
        return aggregate, summary

    def test_summary_is_independently_resummarized(self):
        aggregate, summary = self.make_summary(); A.check_summary(aggregate, summary, 1)
        self.assertEqual(len(aggregate), 8)

    def test_summary_arithmetic_and_scientific_claims_are_checked(self):
        aggregate, summary = self.make_summary()
        changed = deepcopy(summary); changed["scientific_gate_pass"] = True
        with self.assertRaises(ValueError):
            A.check_summary(aggregate, changed, 1)
        changed = deepcopy(summary); changed["comparisons"]["native_heads_both_decode_fp32"]["boxes"]["violations"] = 1
        with self.assertRaises(ValueError):
            A.check_summary(aggregate, changed, 1)


if __name__ == "__main__":
    unittest.main()
