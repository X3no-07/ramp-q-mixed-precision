from collections import Counter
from pathlib import Path
import sys
import unittest
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refine_gate1_splits import allocate_calibration, build_plan


def rows_for(group, count, role="train", source="train"):
    return [{"image_id": f"{source}/{group}_{i:04d}", "prefix_proxy": group,
             "image_sha256": f"unique:{source}:{group}:{i}", "assigned_split": role,
             "source_split": source, "is_content_canonical": True} for i in range(count)]


def fixture():
    rows = []
    for number in range(160):
        role = "calibration_pool" if number < 25 else "debug" if number < 28 else "train"
        rows.extend(rows_for(f"g{number:03d}", 32, role))
    rows.extend(rows_for("big_previous_val", 407, "training_val"))
    rows.extend(rows_for("official_val", 5, "selection", "val"))
    return rows


class RefinedSplits(unittest.TestCase):
    def test_diverse_val_does_not_take_large_component(self):
        rows, stats = build_plan(fixture(), [])
        val = [r for r in rows if r["assigned_split"] == "training_val"]
        counts = Counter(r["refined_component"] for r in val)
        self.assertGreaterEqual(len(val), 400)
        self.assertGreaterEqual(len(counts), 12)
        self.assertLessEqual(max(counts.values()), 64)
        self.assertEqual({r["assigned_split"] for r in rows if r["prefix_proxy"] == "big_previous_val"}, {"train"})
        self.assertEqual(stats["counts"]["selection"], 5)

    def test_near_risk_to_selection_quarantines_whole_train_group(self):
        rows, _ = build_plan(fixture(), [{"a": "train/g120_0000", "b": "val/official_val_0000", "reason": "reviewed near-risk"}])
        self.assertEqual({r["assigned_split"] for r in rows if r["prefix_proxy"] == "g120"}, {"quarantine"})
        self.assertEqual({r["assigned_split"] for r in rows if r["source_split"] == "val"}, {"selection"})

    def test_quality_quarantine_propagates_before_sampling(self):
        original = fixture()
        next(r for r in original if r["image_id"] == "train/g100_0000")["requires_annotation_quality_quarantine"] = True
        rows, _ = build_plan(original, [])
        self.assertEqual({r["assigned_split"] for r in rows if r["prefix_proxy"] == "g100"}, {"quarantine"})

    def test_calibration_three_seeds_have_coverage_cap_and_no_overlap(self):
        pool = {f"p{i}": rows_for(f"p{i}", 16) for i in range(12)}
        selected = allocate_calibration(pool)
        groups = {r["image_id"]: key for key, bucket in pool.items() for r in bucket}
        combined = []
        for ids in selected.values():
            counts = Counter(groups[key] for key in ids)
            self.assertEqual(len(ids), 64)
            self.assertGreaterEqual(len(counts), 12)
            self.assertLessEqual(max(counts.values()), 8)
            combined.extend(ids)
        self.assertEqual(len(set(combined)), 192)

    def test_infeasible_capacity_fails_instead_of_reducing_rules(self):
        with self.assertRaisesRegex(ValueError, "Insufficient|infeasible"):
            allocate_calibration({"only_group": rows_for("only_group", 1000)})

    def test_unknown_risk_edge_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            build_plan(fixture(), [{"a": "testdev/not_allowed", "b": "train/g120_0000", "reason": "bad input"}])

    def test_debug_takes_priority_over_calibration_after_risk_merge(self):
        rows, _ = build_plan(fixture(), [{"a": "train/g000_0000", "b": "train/g025_0000", "reason": "reviewed risk"}])
        self.assertEqual({r["assigned_split"] for r in rows if r["prefix_proxy"] in {"g000", "g025"}}, {"debug"})

    def test_cross_prefix_exact_sha_to_selection_quarantines_train(self):
        original = fixture()
        train = next(r for r in original if r["image_id"] == "train/g120_0000")
        train["image_sha256"] = next(r for r in original if r["image_id"] == "val/official_val_0000")["image_sha256"]
        rows, _ = build_plan(original, [])
        self.assertEqual({r["assigned_split"] for r in rows if r["prefix_proxy"] == "g120"}, {"quarantine"})

    def test_input_order_does_not_change_plan_or_calibration(self):
        original = fixture()
        expected = build_plan(original, [])
        random.Random(17).shuffle(original)
        self.assertEqual(build_plan(original, []), expected)

    def test_deterministic_and_source_not_mutated(self):
        source = fixture()
        first, first_stats = build_plan(source, [])
        second, second_stats = build_plan(source, [])
        self.assertEqual(first, second)
        self.assertEqual(first_stats, second_stats)
        self.assertNotIn("refined_component", source[0])


if __name__ == "__main__":
    unittest.main()
