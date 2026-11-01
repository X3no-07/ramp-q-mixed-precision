"""Synthetic saved tensors only; no model, pixels, CUDA or annotations."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


A = load("body32_head16_auditor_under_test", ROOT / "scripts/audit_gate1_body32_head16.py")
F = load("frozen_decode_synthetic_for_body32_head16", ROOT / "tests/test_gate1_decode_probe_audit.py")


def fixture():
    ref, old_meta, history = F.fixture()
    arrays = {"input_fp32_cpu": ref["input_fp32_cpu"].copy(), "box_logits": ref["fp16_box_logits"].copy(),
              "class_logits": ref["fp16_class_logits"].copy(), "anchors": ref["fp32_anchors"].copy(),
              "strides": ref["fp32_strides"].copy(), "dfl_bins": np.arange(16, dtype=np.float32),
              "decoded": ref["source_fp16_decode_fp32_decoded"].copy()}
    for key in ("box_logits", "class_logits", "decoded"): ref["source_fp32_head_fp16_" + key] = arrays[key].copy()
    meta = {"image_id": "train/synthetic", "reg_max": 16, "nc": 10, "nl": 3, "normalized_fp32_input_sha256": old_meta["normalized_fp32_input_sha256"],
            "endpoint_byte_equal": {key: True for key in ("box_logits", "class_logits", "decoded")}, "reference_feature_exact": [True, True, True],
            "body_module_observations": [{"i": i, "dtype": "torch.float32", "shape": [1, 1, 2, 2]} for i in range(23)],
            "head_input_features": [], "head_cast_features": [], "decoder_stage_dtypes": {key: "torch.float32" for key in A.STAGES},
            "body_state_unchanged": True, "head_state_unchanged": True, "old_guard_replaced": False, "tolerances_changed": False,
            "model_state_unchanged": True, "mixed_forwards": 1, "historical_arrays_opened_after_forward": True,
            "saved_frontier_inference_inputs": False, "reference_input_exact": True, "reference_cast_feature_exact": [True, True, True],
            "comparison": A.D.fixed_guard(history["fp32"], arrays["decoded"])}
    meta["comparison"].update(guard_name=A.CHECK_NAME, historical_gate1_dtype_guard_replaced=False,
                              raw_geometry={name: {"positive_width_height": True, "class_scores_within_0_1": True} for name in ("historical_fp32", "body32_head16_decode32_prototype")},
                              shape_pass=True, finite_pass=True, tolerances_changed=False, old_gate05_guard_replaced=False)
    for scale, (index, length) in enumerate(zip((16, 19, 22), (80, 40, 20))):
        feature = np.full((1, 1, length, length), .3333, dtype=np.float32); ref[f"fp32_feat_scale{scale}"] = feature
        meta["head_input_features"].append(A.statistics(feature)); meta["head_cast_features"].append(A.statistics(feature.astype(np.float16)))
        meta["body_module_observations"][index]["shape"] = list(feature.shape)
    return arrays, meta, ref, history


class PrototypeArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.arrays, cls.meta, cls.reference, cls.history = fixture()

    def check(self, arrays=None, meta=None, reference=None, history=None):
        return A.audit_arrays(arrays or self.arrays, meta or self.meta, reference or self.reference, history or self.history)

    def corrupt(self, key, index, value):
        arrays = dict(self.arrays); arrays[key] = arrays[key].copy(); arrays[key][index] = value
        with self.assertRaises(ValueError): self.check(arrays=arrays)

    def test_complete_prototype_saved_arrays_and_guard_recompute(self):
        actual = self.check(); self.assertTrue(actual["endpoint_bytes_exact"])
        self.assertEqual(actual["observed_body_modules_fp32"], 23)
        self.assertEqual(actual["comparison"]["boxes"]["elements"], 33600)
        self.assertEqual(actual["comparison"]["scores"]["elements"], 84000)

    def test_box_class_and_decoded_endpoint_bytes_mandatory(self):
        for key in ("box_logits", "class_logits", "decoded"):
            with self.subTest(key=key): self.corrupt(key, (0, 0, 0), .25)

    def test_saved_array_inventory_and_shapes_are_fixed(self):
        arrays = dict(self.arrays); arrays["extra_old_frontier"] = np.zeros(1)
        with self.assertRaises(ValueError): self.check(arrays=arrays)
        arrays = dict(self.arrays); arrays["box_logits"] = arrays["box_logits"][..., :1]
        with self.assertRaises(ValueError): self.check(arrays=arrays)

    def test_actual_stored_logits_half_decoded_and_geometry_float32(self):
        for key in ("box_logits", "class_logits", "decoded", "anchors", "dfl_bins"):
            arrays = dict(self.arrays); arrays[key] = arrays[key].astype(np.float32 if key in ("box_logits", "class_logits") else np.float16)
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(arrays=arrays)

    def test_input_hash_domain_and_reference_bytes_checked(self):
        self.corrupt("input_fp32_cpu", (0, 0, 0, 0), 1.5)
        self.corrupt("input_fp32_cpu", (0, 0, 0, 0), .5)
        meta = deepcopy(self.meta); meta["normalized_fp32_input_sha256"] = "0" * 64
        with self.assertRaises(ValueError): self.check(meta=meta)

    def test_every_body_module_precision_observation_required(self):
        for index in (0, 10, 22):
            meta = deepcopy(self.meta); meta["body_module_observations"][index]["dtype"] = "torch.float16"
            with self.subTest(index=index), self.assertRaises(ValueError): self.check(meta=meta)
        meta = deepcopy(self.meta); del meta["body_module_observations"][0]
        with self.assertRaises(ValueError): self.check(meta=meta)

    def test_fresh_feature_hash_and_half_cast_stats_recomputed(self):
        meta = deepcopy(self.meta); meta["head_input_features"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): self.check(meta=meta)
        meta = deepcopy(self.meta); meta["head_cast_features"][1]["mean"] = .3333
        with self.assertRaises(ValueError): self.check(meta=meta)
        meta = deepcopy(self.meta); meta["reference_feature_exact"][2] = False
        with self.assertRaises(ValueError): self.check(meta=meta)

    def test_all_decoder_stage_float32_observations_required(self):
        for key in A.STAGES:
            meta = deepcopy(self.meta); meta["decoder_stage_dtypes"][key] = "torch.float16"
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(meta=meta)

    def test_geometry_order_stride_and_bins_are_exact(self):
        self.corrupt("anchors", (0, 0), 1.5)
        self.corrupt("strides", (0, 0), 16.)
        self.corrupt("dfl_bins", 15, 14.)

    def test_nonfinite_values_and_scope_changes_rejected(self):
        self.corrupt("box_logits", (0, 0, 0), np.nan)
        for key, value in (("old_guard_replaced", True), ("tolerances_changed", True), ("body_state_unchanged", False), ("head_state_unchanged", False)):
            meta = deepcopy(self.meta); meta[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(meta=meta)

    def test_reported_fixed_guard_tampering_rejected(self):
        meta = deepcopy(self.meta); meta["comparison"]["boxes"]["violations"] = 1
        with self.assertRaises(ValueError): self.check(meta=meta)

    def test_prototype_guard_has_new_identity_and_geometry_labels(self):
        for key, value in (("guard_name", "GATE1_SAME_PYTORCH_UNFUSED_FP32_FP16_DTYPE_GUARD"), ("historical_gate1_dtype_guard_replaced", True), ("raw_geometry", {})):
            meta = deepcopy(self.meta); meta["comparison"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(meta=meta)

    def test_fresh_forward_metadata_cannot_claim_saved_frontiers(self):
        for key, value in (("mixed_forwards", 2), ("saved_frontier_inference_inputs", True), ("historical_arrays_opened_after_forward", False)):
            meta = deepcopy(self.meta); meta[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(meta=meta)

    def test_original_failed_guard_remains_separate(self):
        history = {key: value.copy() for key, value in self.history.items()}; history["fp16"][0, 0, 0] += 10
        result = self.check(history=history)
        self.assertTrue(result["comparison"]["allclose_pass"])
        self.assertFalse(result["historical_original_guard"]["allclose_pass"])

    def test_approx_decode_sanity_rejects_changed_box_and_class_logit(self):
        for key in ("box_logits", "class_logits"):
            logits = self.arrays[key].copy(); logits[0, 0, 0] = 10
            with self.subTest(key=key), self.assertRaises(ValueError):
                A.approximate_decode(logits if key == "box_logits" else self.arrays["box_logits"],
                                     logits if key == "class_logits" else self.arrays["class_logits"], self.arrays["decoded"], 16)


class PrototypeScopeSummaryTests(unittest.TestCase):
    def test_frozen_debug_cohort_and_fixed_guard_scope(self):
        cohort = [{"image_id": f"train/synthetic_{i}", "assigned_split": "debug", "source_split": "train"} for i in range(109)]
        for row, name in zip(cohort, A.PILOT_IDS): row["image_id"] = name
        p = {"schema": "rampq.gate1.body32_head16_prototype_protocol/v1", "decoder_runner": {"sha256": A.DECODER_SHA}, "head_runner": {"sha256": A.HEAD_SHA},
             "precision_contract": A.PRECISION_CONTRACT, "guard_name": A.CHECK_NAME,
             "data_role": "debug", "selection_detector_inference": False, "holdout_accessed": False, "raw_tolerances": A.D.RAW_TOLERANCES,
             "historical_equality_required": True, "native_reconstruction_required": True, "stage": "pilot3", "image_ids": A.PILOT_IDS}
        fs = [-1] * 24
        for index, f in ((12, [-1, 6]), (15, [-1, 4]), (18, [-1, 13]), (21, [-1, 10]), (23, [16, 19, 22])): fs[index] = f
        p["topology"] = {"modules": [{"i": i, "f": f, "type": "Detect" if i == 23 else "Synthetic"} for i, f in enumerate(fs)], "save": [4, 6, 10, 13, 16, 19, 22]}
        self.assertEqual(len(A.validate_scope(p, cohort)), 109)
        for key, value in (("holdout_accessed", True), ("selection_detector_inference", True), ("image_ids", A.PILOT_IDS[::-1]), ("raw_tolerances", {}), ("precision_contract", {}), ("guard_name", "old_guard")):
            changed = deepcopy(p); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): A.validate_scope(changed, cohort)

    def test_summary_counts_guard_values_and_nonacceptance_enforced(self):
        arrays, meta, ref, history = fixture(); result = A.audit_arrays(arrays, meta, ref, history); actual = A.aggregate([result])
        summary = {"schema": "rampq.gate1.body32_head16_prototype_summary/v1", "images": 1,
                   "guard_name": A.CHECK_NAME, "historical_gate1_dtype_guard_replaced": False, "precision_contract": A.PRECISION_CONTRACT,
                   "fresh_mixed_forwards": 1, "checkpoint_loads": 1,
                   **{key: True for key in ("endpoint_bytes_all_exact", "all_body_modules_fp32", "all_head_towers_fp16", "all_decoder_stages_fp32", "states_unchanged")},
                   "raw_tolerances": A.D.RAW_TOLERANCES, "old_guard_replaced": False, "original_guard_replaced": False, "scientific_gate_pass": False, "accepted_float_baseline_ready": False,
                   "comparison": {"diagnostic_guard_pass_images": actual["pass_images"], "diagnostic_guard_fail_images": actual["fail_images"],
                                  **{section: {key: actual[section][key] for key in ("violations", "max_absolute_error", "max_error_over_tolerance")} for section in ("boxes", "scores")}}}
        A.check_summary(actual, summary, 1)
        for key, value in (("scientific_gate_pass", True), ("accepted_float_baseline_ready", True), ("all_head_towers_fp16", False)):
            changed = deepcopy(summary); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): A.check_summary(actual, changed, 1)
        changed = deepcopy(summary); changed["comparison"]["boxes"]["violations"] = 1
        with self.assertRaises(ValueError): A.check_summary(actual, changed, 1)

    def test_reported_parameter_and_buffer_dtype_inventory_is_fixed(self):
        p = {"checkpoint": {"sha256": "a" * 64}, "checkpoint_loads": 1, "precision_contract": A.PRECISION_CONTRACT, "topology": {},
             "fusion_called": False, "reference_is_original_fp32_training_state": False,
             "parameter_buffer_policy": {"body_parameters": {"torch.float32": 2}, "body_floating_buffers": {"torch.float32": 2},
                                         "head_tower_parameters": {"torch.float16": 2}, "head_tower_floating_buffers": {"torch.float16": 2},
                                         "dfl_parameters": {"torch.float32": 1}, "dfl_floating_buffers": {}}}
        A.validate_provenance(p, p["checkpoint"], {})
        for group in ("body_parameters", "body_floating_buffers", "head_tower_parameters", "head_tower_floating_buffers", "dfl_parameters"):
            bad = deepcopy(p); bad["parameter_buffer_policy"][group] = {"torch.float64": 1}
            with self.subTest(group=group), self.assertRaises(ValueError): A.validate_provenance(bad, p["checkpoint"], {})


if __name__ == "__main__": unittest.main()
