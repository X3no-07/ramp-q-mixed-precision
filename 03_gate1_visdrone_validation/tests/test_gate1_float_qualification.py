"""Synthetic tests only: no checkpoint load, image data, GPU work, or AP claim."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_gate1_float_baseline.py"
SPEC = importlib.util.spec_from_file_location("gate1_float_qualification_under_test", SCRIPT)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def completion_fixture():
    history = [{"epoch_zero_based": i, "epoch_one_based": i + 1, "fitness": (i // 2) / 100} for i in range(30)]
    receipt = {"schema": "rampq.gate1.float_training_attempt/v1", "mode": "train", "status": Q.FINAL_STATUS,
               "phase": "COMPLETE", "training_started": True, "contract_sha256": Q.CONTRACT_SHA256,
               "contract_manifest_sha256": Q.CONTRACT_MANIFEST_SHA256, "runner_sha256": Q.TRAINING_RUNNER_SHA256,
               "selection_detector_inference": False, "holdout_accessed": False, "recovery_events": [],
               "external_restarts": 0, "effective_batch": 4, "effective_amp": False, "epoch_history": history,
               "checkpoint_history": [{"epoch_zero_based": i, "best_saved_this_epoch": True} for i in range(30)],
               "checkpoint_classes": {str(i): value for i, value in enumerate(Q.NAMES)},
               "checkpoints": {name: {"sha256": "a" * 64, "bytes": 100, "training_epoch_zero_based": 29,
                                      "training_epoch_one_based": 30} for name in ("best.pt", "last.pt")}}
    contract = {"runner_sha256": Q.TRAINING_RUNNER_SHA256, "fixed_args": {"epochs": 30, "amp": False}}
    return receipt, contract


def validate(receipt, contract):
    return Q.validate_completion(receipt, contract, Q.CONTRACT_SHA256, Q.CONTRACT_MANIFEST_SHA256)


def raw():
    value = np.zeros((1, 14, 8400), dtype=np.float32)
    value[:, :4] = 100
    value[:, 4:] = 0.1
    value[:, 4] = 0.4
    return value


def debug_row(identity="train/a"):
    return {"image_id": identity, "assigned_split": "debug", "source_split": "train",
            "image_path": "images/train/a.jpg", "annotation_path": "annotations/train/a.txt",
            "requires_annotation_quality_quarantine": False, "annotation_quality_target_ids": []}


class CompletionTests(unittest.TestCase):
    def test_completed_thirty_epochs_and_later_tie_are_accepted_as_metadata_only(self):
        receipt, contract = completion_fixture()
        check = validate(receipt, contract)
        self.assertEqual(check["best_training_epoch_one_based"], 30)
        self.assertFalse(check["checkpoint_selection_uses_debug"])

    def test_running_interrupted_and_old_contract_are_refused(self):
        for key, value in (("status", "RUNNING_FROZEN_FLOAT_TRAINING"), ("phase", "TRAINING"),
                           ("contract_sha256", "0" * 64), ("runner_sha256", "0" * 64)):
            receipt, contract = completion_fixture()
            receipt[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(receipt, contract)

    def test_missing_duplicate_or_nonfinite_epoch_is_refused(self):
        receipt, contract = completion_fixture()
        for history in (receipt["epoch_history"][:-1], [receipt["epoch_history"][0]] * 30):
            changed = deepcopy(receipt)
            changed["epoch_history"] = history
            with self.assertRaises(ValueError):
                validate(changed, contract)
        receipt["epoch_history"][4]["fitness"] = float("nan")
        with self.assertRaises(ValueError):
            validate(receipt, contract)

    def test_best_epoch_provenance_uses_history_not_serialized_epoch_minus_one(self):
        receipt, contract = completion_fixture()
        receipt["epoch_history"][5]["fitness"] = 0.9
        receipt["epoch_history"][7]["fitness"] = 0.9
        receipt["checkpoints"]["best.pt"].update(training_epoch_zero_based=7, training_epoch_one_based=8)
        self.assertEqual(validate(receipt, contract)["best_training_epoch_one_based"], 8)
        receipt["checkpoints"]["best.pt"]["training_epoch_zero_based"] = -1
        with self.assertRaises(ValueError):
            validate(receipt, contract)

    def test_forbidden_access_recovery_and_changed_effective_batch_are_refused(self):
        for key, value in (("selection_detector_inference", True), ("holdout_accessed", True),
                           ("recovery_events", [{"kind": "OOM"}]), ("external_restarts", 1),
                           ("effective_batch", 2), ("effective_amp", True)):
            receipt, contract = completion_fixture()
            receipt[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(receipt, contract)

    def test_malformed_or_incomplete_checkpoint_identity_is_refused(self):
        receipt, contract = completion_fixture()
        del receipt["checkpoints"]["last.pt"]
        with self.assertRaises(ValueError):
            validate(receipt, contract)
        receipt, contract = completion_fixture()
        receipt["checkpoints"]["best.pt"]["sha256"] = "x" * 64
        with self.assertRaises(ValueError):
            validate(receipt, contract)


class DataRoleTests(unittest.TestCase):
    def test_only_debug_source_train_rows_are_selected(self):
        a, b = debug_row(), debug_row("val/selection")
        b["assigned_split"] = "selection"
        self.assertEqual(Q.select_debug_rows([b, a], expected_count=1), [a])

    def test_debug_role_cannot_spoof_selection_or_quarantine(self):
        for key, value in (("source_split", "val"), ("image_path", "images/val/a.jpg"),
                           ("annotation_path", "annotations/testdev/a.txt"),
                           ("requires_annotation_quality_quarantine", True), ("annotation_quality_target_ids", ["train/a:2"])):
            row = debug_row()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                Q.select_debug_rows([row], expected_count=1)

    def test_duplicate_identity_wrong_count_and_path_traversal_are_refused(self):
        with self.assertRaises(ValueError):
            Q.select_debug_rows([debug_row(), debug_row()], expected_count=2)
        with self.assertRaises(ValueError):
            Q.select_debug_rows([debug_row()])
        with self.assertRaises(ValueError):
            Q.safe_file(Path(__file__).resolve().parent, "../outside.txt")


class NumericGuardTests(unittest.TestCase):
    def test_already_fused_or_changed_architecture_is_refused(self):
        valid = {"Conv2d": 88, "BatchNorm2d": 60, "UltralyticsConv": 60, "UltralyticsConv_with_BatchNorm": 60}
        Q.validate_unfused_structure(valid)
        for key, value in (("Conv2d", 87), ("BatchNorm2d", 0), ("UltralyticsConv_with_BatchNorm", 59)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                Q.validate_unfused_structure(valid | {key: value})

    def test_identical_raw_outputs_pass_without_scientific_acceptance(self):
        result = Q.raw_guard(raw(), raw().astype(np.float16))
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["old_gate05_guard_replaced"])
        self.assertFalse(Q.protocol()["accepted_float_baseline_ready"])

    def test_fixed_asymmetric_tolerance_failure_is_retained(self):
        reference, comparison = raw(), raw()
        comparison[0, 0, 0] = 102  # tolerance is 0.5 + .005 * 100 == 1
        comparison[0, 4, 1] = 0.42  # tolerance is .002 + .02 * .4 == .01
        before = comparison.copy()
        result = Q.raw_guard(reference, comparison)
        self.assertEqual(result["status"], "FAIL_FIXED_TOLERANCES")
        self.assertEqual(result["boxes"]["violations"], 1)
        self.assertEqual(result["scores"]["violations"], 1)
        self.assertFalse(result["tolerances_changed"])
        np.testing.assert_array_equal(comparison, before)

    def test_shape_and_nonfinite_are_separate_abort_reasons(self):
        result = Q.raw_guard(raw()[:, :, :-1], raw())
        self.assertFalse(result["shape_pass"])
        self.assertTrue(result["finite_pass"])
        comparison = raw()
        comparison[0, 4, 0] = np.inf
        result = Q.raw_guard(raw(), comparison)
        self.assertTrue(result["shape_pass"])
        self.assertFalse(result["finite_pass"])
        self.assertFalse(result["allclose_pass"])

    def test_class_argmax_disagreement_is_reported_independently(self):
        reference, comparison = raw(), raw()
        reference[0, 4:6, 0] = [0.4, 0.3999]
        comparison[0, 4:6, 0] = [0.3999, 0.4]
        result = Q.raw_guard(reference, comparison)
        self.assertTrue(result["allclose_pass"])
        self.assertEqual(result["class_argmax"]["all_anchors_equal"], 8399)


class AdapterTests(unittest.TestCase):
    def test_restored_adapter_to_frozen_evaluator_preserves_original_line_identity(self):
        evaluator_path = SCRIPT.with_name("gate1_ignore_aware_evaluation.py")
        self.assertEqual(Q.sha(evaluator_path), Q.EVALUATOR_SHA256)
        spec = importlib.util.spec_from_file_location("frozen_evaluation_adapter_test", evaluator_path)
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        original = evaluator.parse_ground_truth("\n20,30,10,10,1,10,0,0\n", "train/example")
        predictions, _ = Q.restored_predictions(np.array([[20, 35, 25, 40, .25, 9]]), np.array([23]),
                                                {"ratio": (.5, .5), "left": 10, "top": 20}, "train/example", "fp32")
        result = evaluator.match_project_targets(original, predictions, 100, 100, confidence=.25, iou_threshold=.5)
        self.assertEqual(result["matches"][0]["target_id"], "train/example:1")
        self.assertEqual(result["counts"]["hits"], 1)
        self.assertFalse(result["official_AP"])

    def test_letterbox_inverse_preserves_unclipped_raw_box_and_maps_publisher_class(self):
        params = {"ratio": (0.5, 0.5), "left": 10, "top": 20}
        det = np.array([[5, 10, 25, 40, 0.9, 0], [15, 25, 35, 45, 0.8, 9]])
        predictions, check = Q.restored_predictions(det, np.array([11, 22]), params, "train/example", "fp32")
        self.assertEqual(predictions[0]["bbox_xywh"], [-10, -20, 40, 60])
        self.assertEqual([row["category"] for row in predictions], [1, 10])
        self.assertEqual(predictions[0]["raw_anchor_index"], 11)
        self.assertFalse(check["coordinates_clipped_before_ignore_filter"])

    def test_nonpositive_nonfinite_fractional_class_and_anchor_are_refused(self):
        for row, anchor in (([1, 2, 1, 4, 0.8, 0], 1), ([1, 2, np.inf, 4, 0.8, 0], 1),
                            ([1, 2, 3, 4, 0.8, 0.5], 1), ([1, 2, 3, 4, 0.8, 0], 8400)):
            with self.subTest(row=row, anchor=anchor), self.assertRaises(ValueError):
                Q.restored_predictions(np.array([row]), np.array([anchor]), {"ratio": (1, 1), "left": 0, "top": 0}, "train/x", "fp32")

    def test_empty_nms_output_is_valid(self):
        rows, check = Q.restored_predictions(np.empty((0, 6)), np.array([], dtype=int), {"ratio": (1, 1), "left": 0, "top": 0}, "train/x", "fp16")
        self.assertEqual(rows, [])
        self.assertEqual(check["count"], 0)

    def test_same_target_partition_and_identity_mismatch_refusal(self):
        left = {"targets": [{"target_id": f"train/x:{i}", "hit": value} for i, value in enumerate((True, True, False, False))]}
        right = {"targets": [{"target_id": f"train/x:{i}", "hit": value} for i, value in enumerate((True, False, True, False))]}
        result = Q.aggregate_target_comparison(left, right)
        self.assertEqual([result[key] for key in ("both_hit", "fp32_only_hit", "fp16_only_hit", "both_miss")], [1, 1, 1, 1])
        right["targets"][0]["target_id"] = "train/x:99"
        with self.assertRaises(ValueError):
            Q.aggregate_target_comparison(left, right)

    def test_protocol_pins_nms_separately_from_training_and_all_thresholds(self):
        data = Q.protocol()
        self.assertEqual(data["postprocessing"]["max_det"], 1000)
        self.assertEqual(data["postprocessing"]["candidate_confidence_strictly_greater_than"], .001)
        self.assertEqual(data["matching"]["confidence_primary_inclusive"], .25)
        self.assertEqual(data["matching"]["confidence_sensitivity_inclusive"], [.2, .3])
        self.assertEqual(data["matching"]["target_iou"], .5)
        self.assertFalse(data["matching"]["official_AP"])
        self.assertFalse(data["holdout_accessed"])


if __name__ == "__main__":
    unittest.main()
