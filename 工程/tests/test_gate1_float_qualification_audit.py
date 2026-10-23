"""Synthetic fixtures only. No real images, annotations, model or GPU are read."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_TEMP = ROOT / "artifacts" / "gate1" / "20261022_float_qualification_audit_preparation" / "synthetic_temporary"
SYNTHETIC_TEMP.mkdir(parents=True, exist_ok=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load("qualification_audit_under_test", ROOT / "scripts" / "audit_gate1_float_qualification.py")
E = load("frozen_evaluator_for_synthetic_fixtures", ROOT / "scripts" / "gate1_ignore_aware_evaluation.py")


def raw_pair():
    raw = np.zeros((1, 14, 8400), dtype=np.float32)
    raw[:, :4] = 100
    raw[:, 2:4] = 20
    raw[0, 4, 0] = .9
    return raw, raw.astype(np.float16)


def prediction(raw, image_id="train/a", branch="fp32"):
    return {"detection_id": f"{image_id}:{branch}:anchor:0:nms:0", "bbox_xywh": [90., 90., 20., 20.],
            "score": float(raw[0, 4, 0]), "category": 1, "yolo_class_zero_based": 0,
            "raw_anchor_index": 0, "nms_input_order": 0}


PARAMS = {"orig_shape": [640, 640], "new_shape": [640, 640], "ratio": [1., 1.], "new_unpad": [640, 640], "top": 0, "bottom": 0, "left": 0, "right": 0}


def write_manifest(root):
    A.write_json(root / "manifest.json", {"files": {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": A.sha(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"}})


class RawTests(unittest.TestCase):
    def test_exact_dtype_shape_and_asymmetric_bounds(self):
        left, right = raw_pair()
        guard, domain = A.calculate_raw_guard(left, right)
        self.assertEqual(guard["status"], "PASS")
        self.assertEqual(guard["boxes"]["elements"], 33600)
        self.assertEqual(guard["scores"]["elements"], 84000)
        self.assertTrue(domain["dtype_pass"] and domain["domain_pass"])

    def test_single_score_violation_stays_fail_despite_finite(self):
        left, right = raw_pair()
        right[0, 5, 99] = .003
        guard, domain = A.calculate_raw_guard(left, right)
        self.assertTrue(guard["finite_pass"] and domain["domain_pass"])
        self.assertEqual(guard["status"], "FAIL_FIXED_TOLERANCES")
        self.assertEqual(guard["scores"]["violations"], 1)
        self.assertFalse(guard["allclose_pass"])

    def test_shape_nonfinite_dtype_and_domain_are_separate(self):
        left, right = raw_pair()
        guard, _ = A.calculate_raw_guard(left[:, :, :10], right[:, :, :10])
        self.assertEqual(guard["status"], "FAIL_SHAPE_OR_FINITE")
        right[0, 0, 0] = np.inf
        guard, _ = A.calculate_raw_guard(left, right)
        self.assertFalse(guard["finite_pass"])
        left, right = raw_pair()
        _, domain = A.calculate_raw_guard(left, right.astype(np.float32))
        self.assertFalse(domain["dtype_pass"])
        left[0, 2, 0] = -1
        _, domain = A.calculate_raw_guard(left, right)
        self.assertFalse(domain["domain_pass"])


class AdapterTests(unittest.TestCase):
    def test_original_physical_lines_keep_blanks_and_bom(self):
        gt = A.parse_original_annotation("\ufeff90,90,20,20,1,1,0,0\n\n0,0,10,10,0,0,0,0,\n", "train/a")
        self.assertEqual([row["target_id"] for row in gt], ["train/a:0", "train/a:2"])
        self.assertEqual(gt, E.parse_ground_truth("\ufeff90,90,20,20,1,1,0,0\n\n0,0,10,10,0,0,0,0,\n", "train/a"))

    def test_class_mapping_anchor_scores_and_original_coordinates(self):
        raw, _ = raw_pair()
        pred = prediction(raw)
        A.verify_predictions([pred], raw, PARAMS, "train/a", "fp32")
        for key, value in (("category", 0), ("score", .1), ("raw_anchor_index", 8400), ("bbox_xywh", [91, 90, 20, 20])):
            changed = deepcopy(pred)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                A.verify_predictions([changed], raw, PARAMS, "train/a", "fp32")

    def test_aspect_padding_is_recomputed(self):
        A.check_letterbox(PARAMS, 640, 640)
        params = {"orig_shape": [540, 960], "new_shape": [640, 640], "ratio": [2 / 3, 2 / 3], "new_unpad": [640, 360], "top": 140, "bottom": 140, "left": 0, "right": 0}
        A.check_letterbox(params, 960, 540)
        params["top"] += 1
        with self.assertRaises(ValueError):
            A.check_letterbox(params, 960, 540)

    def test_frozen_membership_detects_role_injected_image(self):
        row = {key: key for key in A.DEBUG_IDENTITY_KEYS}
        row["image_id"] = "train/a"
        old = A.canonical_debug_identity([row])
        row["image_sha256"] = "0" * 64
        self.assertNotEqual(old, A.canonical_debug_identity([row]))


class MatchingTests(unittest.TestCase):
    def fixture(self):
        raw, _ = raw_pair()
        preds = [prediction(raw)]
        gt = A.parse_original_annotation("90,90,20,20,1,1,0,0\n150,150,10,10,1,1,0,0\n", "train/a")
        return E.match_project_targets(gt, preds, 640, 640), preds

    def test_recount_hits_fn_fp_and_partitions(self):
        matching, preds = self.fixture()
        counts = A.recount_matching(matching, preds)
        self.assertEqual((counts["eligible_targets"], counts["hits"], counts["misses"], counts["false_positive_detections"]), (2, 1, 1, 0))
        other = deepcopy(matching)
        other["targets"][0]["hit"], other["targets"][1]["hit"] = False, True
        self.assertEqual(A.paired_partition(matching, other), {"eligible_targets": 2, "fp32_hits": 1, "fp16_hits": 1, "both_hit": 0, "fp32_only_hit": 1, "fp16_only_hit": 1, "both_miss": 0})

    def test_forged_count_or_hit_partition_is_rejected(self):
        matching, preds = self.fixture()
        wrong = deepcopy(matching)
        wrong["counts"]["hits"] += 1
        with self.assertRaises(ValueError):
            A.recount_matching(wrong, preds)
        wrong = deepcopy(matching)
        wrong["targets"][0]["hit"] = False
        with self.assertRaises(ValueError):
            A.recount_matching(wrong, preds)

    def test_ignore_absorption_and_class0_drops_are_recounted(self):
        gt = A.parse_original_annotation("100,100,100,100,0,0,0,0\n300,300,50,50,0,1,0,0\n", "train/a")
        preds = [{"detection_id": "drop", "bbox_xywh": [110, 110, 20, 20], "score": .9, "category": 1}, {"detection_id": "absorb", "bbox_xywh": [310, 310, 10, 10], "score": .9, "category": 1}]
        matching = E.match_project_targets(gt, preds, 640, 640)
        counts = A.recount_matching(matching, preds)
        self.assertEqual(counts["category_0_dropped_detections"], 1)
        self.assertEqual(counts["score_0_absorbed_detections"], 1)


class ManifestTests(unittest.TestCase):
    def test_changed_or_unlisted_file_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=SYNTHETIC_TEMP) as temporary:
            root = Path(temporary)
            (root / "a").write_text("abc", encoding="utf-8")
            write_manifest(root)
            self.assertEqual(A.verify_manifest(root)["files_verified"], 1)
            (root / "a").write_text("abd", encoding="utf-8")
            with self.assertRaises(ValueError):
                A.verify_manifest(root)
            write_manifest(root)
            (root / "extra").write_text("abc", encoding="utf-8")
            with self.assertRaises(ValueError):
                A.verify_manifest(root)

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=SYNTHETIC_TEMP) as temporary:
            with self.assertRaises(ValueError):
                A.safe_file(Path(temporary), "../outside")


def completed_synthetic_qualification(root, fail_guard=False):
    for folder in ("debug/images", "debug/annotations", "raw", "per_image"):
        (root / folder).mkdir(parents=True)
    (root / "runner_source.py").write_text("# synthetic source bytes, never executed\n", encoding="utf-8")
    (root / "frozen_training_contract.json").write_text("{}\n", encoding="utf-8")
    (root / "ignore_aware_evaluation_source.py").write_bytes((ROOT / "scripts" / "gate1_ignore_aware_evaluation.py").read_bytes())
    (root / "best.pt").write_bytes(b"synthetic checkpoint bytes, never deserialized")
    checkpoint = root / "best.pt"
    A.write_json(root / "final_training_receipt.json", {"status": "COMPLETE_FLOAT_TRAINING_REQUIRES_SCIENTIFIC_REVIEW", "phase": "COMPLETE", "checkpoints": {"best.pt": {"sha256": A.sha(checkpoint), "bytes": checkpoint.stat().st_size}}})
    A.write_json(root / "qualification_protocol.json", {"data_role": "debug", "expected_images": 109, "split_assignment_sha256": A.ASSIGNMENT_SHA,
                 "raw": {"box_rtol": .005, "box_atol": .5, "score_rtol": .02, "score_atol": .002}, "matching": {"target_iou": .5, "implementation_sha256": A.EVALUATOR_SHA}})
    rows, guards, progress = [], [], []
    summary = {t: {"fp32": A.Counter(), "fp16": A.Counter(), "paired_targets": A.Counter()} for t in A.THRESHOLDS}
    for number in range(109):
        stem = f"synthetic{number:03}"
        image_id = f"train/{stem}"
        image = root / "debug" / "images" / f"{stem}.jpg"
        annotation = root / "debug" / "annotations" / f"{stem}.txt"
        image.write_bytes(b"synthetic bytes, not an image")
        annotation.write_text("90,90,20,20,1,1,0,0\n", encoding="utf-8")
        row = {"image_id": image_id, "image_path": f"images/train/{stem}.jpg", "annotation_path": f"annotations/train/{stem}.txt", "image_sha256": A.sha(image), "annotation_sha256": A.sha(annotation),
               "width": 640, "height": 640, "assigned_split": "debug", "source_split": "train", "valid_target_count": 1, "refined_component": image_id,
               "requires_annotation_quality_quarantine": False, "annotation_quality_target_ids": []}
        rows.append(row)
        left, right = raw_pair()
        if fail_guard and number == 0:
            right[0, 5, 99] = .003
        np.savez_compressed(root / "raw" / f"{stem}.npz", fp32=left, fp16=right)
        guard, _ = A.calculate_raw_guard(left, right)
        guard["image_id"] = image_id
        guards.append(guard)
        gt = A.parse_original_annotation(annotation.read_text(encoding="utf-8"), image_id)
        predictions = {branch: [prediction(raw, image_id, branch)] for branch, raw in (("fp32", left), ("fp16", right))}
        item = {"image_id": image_id, "image_sha256": row["image_sha256"], "annotation_sha256": row["annotation_sha256"], "letterbox_params": PARAMS,
                "preprocessed_uint8_sha256": "a" * 64, "normalized_fp32_input_sha256": "b" * 64, "raw_guard": guard,
                "raw_output_dtypes": {"fp32": "float32", "fp16": "float16"}, "original_ground_truth": gt, "predictions": predictions, "matching": {}, "coordinate_checks": {}}
        for branch in ("fp32", "fp16"):
            item["coordinate_checks"][branch] = {"finite": True, "positive_unclipped_geometry": True, "category_mapping_valid": True, "count": 1,
                                               "coordinates_clipped_before_ignore_filter": False, "max_det_cap": 1000, "max_det_saturated": False, "NMS_time_limit_warning": False}
        for threshold in A.THRESHOLDS:
            matching = {branch: E.match_project_targets(gt, predictions[branch], 640, 640, confidence=float(threshold), iou_threshold=.5) for branch in ("fp32", "fp16")}
            matching["paired_targets"] = A.paired_partition(matching["fp32"], matching["fp16"])
            item["matching"][threshold] = matching
            for branch in ("fp32", "fp16"):
                summary[threshold][branch].update(matching[branch]["counts"])
            summary[threshold]["paired_targets"].update(matching["paired_targets"])
        A.write_json(root / "per_image" / f"{stem}.json", item)
        progress.append({"image_id": image_id, "images_completed": number + 1, "raw_guard_status": guard["status"]})
    for threshold in summary.values():
        for branch in ("fp32", "fp16"):
            threshold[branch].update(project_target_recall=1., project_detection_precision=1.)
    failed = [guard["image_id"] for guard in guards if guard["status"] != "PASS"]
    status = "FAIL_FIXED_TOLERANCES" if failed else "PASS"
    A.write_json(root / "qualification_receipt.json", {"schema": "rampq.gate1.float_qualification_attempt/v1", "status": "COMPLETE_DEBUG_FLOAT_QUALIFICATION_REQUIRES_REVIEW", "phase": "COMPLETE", "images_completed": 109,
             "runner_sha256": A.sha(root / "runner_source.py"), "inference_started": True, "raw_finite_and_shape": True, "debug_project_matching_complete": True,
             "same_backend_dtype_guard": status, "selection_detector_inference": False, "holdout_accessed": False, "accepted_float_baseline_ready": False, "scientific_gate_pass": False, "official_visdrone_ap": False})
    A.write_json(root / "debug_source_manifest.json", {"rows": rows, "selection_or_holdout_copied": False})
    A.write_json(root / "matching_summary.json", {"thresholds": summary, "official_AP": False, "oracle_equivalence_verified": False, "data_role": "debug"})
    A.write_json(root / "raw_dtype_guard.json", {"status": status, "images": 109, "box_rtol": .005, "box_atol": .5, "score_rtol": .02, "score_atol": .002, "guard_failed_images": failed, "guards": guards, "old_gate05_guard_replaced": False})
    A.write_json(root / "raw_dtype_guard_progress.json", {"images_completed": 109, "guards": guards})
    A.write_json(root / "source_after_check.json", {"verified_images_and_original_annotations": 109, "unchanged": True})
    (root / "image_progress.jsonl").write_text("".join(json.dumps(item) + "\n" for item in progress), encoding="utf-8")
    write_manifest(root)
    return rows


class EndToEndTests(unittest.TestCase):
    def run_fixture(self, fail_guard=False, change=None):
        with tempfile.TemporaryDirectory(dir=SYNTHETIC_TEMP) as temporary:
            parent = Path(temporary)
            root, output = parent / "qualification", parent / "audit"
            root.mkdir()
            output.mkdir()
            rows = completed_synthetic_qualification(root, fail_guard)
            if change:
                change(root)
                write_manifest(root)
            with patch.object(A, "QUALIFIER_SHA", A.sha(root / "runner_source.py")), patch.object(A, "CONTRACT_SHA", A.sha(root / "frozen_training_contract.json")), patch.object(A, "DEBUG_CANONICAL_SHA", A.canonical_debug_identity(rows)):
                before = A.sha(root / "manifest.json")
                result = A.audit(root, output)
                self.assertEqual(A.sha(root / "manifest.json"), before)
                self.assertFalse((root / "__pycache__").exists())
                return result

    def test_completed_synthetic_output_passes_without_pixels_or_model(self):
        result = self.run_fixture()
        self.assertEqual(result["status"], "PASS_SAVED_OUTPUT_CONSISTENCY_ONLY")
        self.assertEqual(result["images"], 109)
        self.assertFalse(result["model_executed"] or result["image_pixels_decoded"] or result["scientific_gate_pass"])

    def test_consistent_dtype_failure_is_preserved_not_overridden_by_finite(self):
        result = self.run_fixture(True)
        self.assertEqual(result["status"], "PASS_SAVED_OUTPUT_CONSISTENCY_ONLY")
        self.assertEqual(result["dtype_guard_preserved"], "FAIL_FIXED_TOLERANCES")
        self.assertEqual(result["dtype_failed_images"], ["train/synthetic000"])

    def test_summary_tampering_is_rejected_even_with_new_manifest(self):
        def change(root):
            path = root / "matching_summary.json"
            data = A.read_json(path)
            data["thresholds"]["0.25"]["fp16"]["hits"] += 1
            A.write_json(path, data)
        with self.assertRaisesRegex(ValueError, "Aggregate matching summary"):
            self.run_fixture(change=change)

    def test_running_receipt_is_rejected_before_loading_evaluator(self):
        def change(root):
            path = root / "qualification_receipt.json"
            data = A.read_json(path)
            data["phase"] = "INFERENCE_DEBUG_ONLY"
            A.write_json(path, data)
        with self.assertRaisesRegex(ValueError, "Only completed"):
            self.run_fixture(change=change)


if __name__ == "__main__":
    unittest.main()
