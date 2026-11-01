"""Synthetic semantic tests only; no dataset, prediction file, or model is read.

These tests verify the declared Python behavior, not MATLAB/Octave equivalence.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gate1_ignore_aware_evaluation.py"
SPEC = importlib.util.spec_from_file_location("gate1_ignore_evaluation_under_test", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def gt(identity="g0", box=(5, 5, 10, 10), *, category=1, score=1, line=0):
    return {"target_id": identity, "physical_line_zero_based": line,
            "bbox_xywh": list(box), "category": category, "score": score,
            "truncation": 0, "occlusion": 0}


def dt(identity="d0", box=(5, 5, 10, 10), *, category=1, score=0.9):
    return {"detection_id": identity, "bbox_xywh": list(box), "category": category, "score": score}


def project(gts, dts, width=100, height=100, **kwargs):
    return EVAL.match_project_targets(gts, dts, width, height, **kwargs)


def test_raw_csv_preserves_physical_lines_and_category_mapping():
    rows = EVAL.parse_ground_truth("\ufeff\n1,2,3,4,1,10,1,2,\n", "synthetic/example")
    assert rows[0]["target_id"] == "synthetic/example:1"
    assert rows[0]["category"] == 10
    assert rows[0]["occlusion"] == 2
    predictions = EVAL.parse_predictions("1,2,3,4,0.8,10,-1,-1\n", "synthetic/example")
    assert predictions[0]["detection_id"] == "synthetic/example:det:0"
    assert predictions[0]["category"] == 10


@pytest.mark.parametrize("text", ["1,2,3,4,nan,1,0,0", "1,2,3,4,2,1,0,0", "1,2,3,4,1,12,0,0", "1,2,3"])
def test_invalid_raw_ground_truth_refused(text):
    with pytest.raises(EVAL.EvaluationInputError):
        EVAL.parse_ground_truth(text, "synthetic")


@pytest.mark.parametrize("value,expected", [(0.5, 1), (1.5, 2), (2.5, 3), (-0.5, -1), (-1.5, -2)])
def test_matlab_half_away_from_zero(value, expected):
    assert EVAL.matlab_round(value) == expected


def test_region_prefilter_drops_both_annotations_and_detections_at_exact_half():
    region = gt("region", (1, 1, 4, 9), category=0, score=0)
    result = EVAL.filter_ignored_regions([region, gt("person", (1, 1, 8, 8), line=1)],
                                        [dt(box=(1, 1, 8, 8))], 20, 20)
    assert result["ground_truth"] == []
    assert result["predictions"] == []
    assert result["removed_predictions"][0]["fraction"] == 0.5
    assert result["removed_ground_truth"][0]["reason"] == "category_0_region_definition"


def test_region_union_counts_overlaps_once_and_keeps_below_half():
    regions = [gt("region1", (1, 1, 3, 9), category=0, score=0),
               gt("region2", (1, 1, 3, 9), category=0, score=0, line=1)]
    result = EVAL.filter_ignored_regions(regions, [dt(box=(1, 1, 8, 8))], 20, 20)
    assert len(result["predictions"]) == 1
    integral = EVAL._ignore_integral(regions, 20, 20)
    assert EVAL.raster_ignore_fraction([1, 1, 8, 8], integral, 20, 20) == 0.375


def test_raster_border_denominator_is_unclipped_rounded_box_area():
    integral = EVAL._ignore_integral([gt("region", (1, 1, 100, 100), category=0, score=0)], 10, 10)
    assert EVAL.raster_ignore_fraction([9, 9, 10, 10], integral, 10, 10) == 0.01


def test_zero_size_original_ignore_region_is_promoted_by_source_rule():
    region = gt("region", (1, 1, 2, 0), category=0, score=0)
    result = EVAL.filter_ignored_regions([region], [dt(box=(1, 1, 1, 1))], 10, 10)
    assert result["removed_predictions"][0]["fraction"] == 1


def test_integral_query_agrees_with_independent_small_pixel_count():
    rng = np.random.default_rng(20261022)
    for _ in range(30):
        boxes = [tuple(map(int, rng.integers(1, 10, size=4))) for _ in range(3)]
        regions = [gt(str(index), box, category=0, score=0, line=index) for index, box in enumerate(boxes)]
        integral = EVAL._ignore_integral(regions, 12, 10)
        pixels = {(x, y) for left, top, width, height in boxes
                  for y in range(top, min(10, top + height) + 1)
                  for x in range(left, min(12, left + width) + 1)}
        query = [float(value) for value in rng.uniform(-1, 14, size=4)]
        x, y, width, height = [max(1, EVAL.matlab_round(value)) for value in query]
        x, y = min(12, x), min(10, y)
        expected = sum((xx, yy) in pixels for yy in range(y + 1, min(10, y + height) + 1)
                       for xx in range(x + 1, min(12, x + width) + 1)) / (width * height)
        assert EVAL.raster_ignore_fraction(query, integral, 12, 10) == expected


def test_score_zero_is_class_specific_iof_absorption_not_global_region():
    ignored = gt("ignored", (1, 1, 80, 80), score=0)
    result = project([ignored], [dt("same", (2, 2, 5, 5)), dt("other", (2, 2, 5, 5), category=2)])
    assert result["counts"]["eligible_targets"] == 0
    assert result["counts"]["score_0_absorbed_detections"] == 1
    assert result["counts"]["false_positive_detections"] == 1
    assert result["counts"]["category_0_dropped_detections"] == 0


def test_positive_match_precedes_score_zero_absorption():
    result = project([gt(), gt("ignored", (1, 1, 80, 80), score=0, line=1)], [dt()])
    assert result["counts"]["hits"] == 1
    assert result["counts"]["score_0_absorbed_detections"] == 0


def test_others_ground_truth_does_not_absorb_predictions_of_evaluated_classes():
    result = project([gt("others", category=11)], [dt(), dt("category11", category=11)])
    assert result["counts"]["eligible_targets"] == 0
    assert result["counts"]["false_positive_detections"] == 1
    assert result["detection_states"][1]["reason"] == "outside_evaluated_categories"


def test_truncation_and_occlusion_do_not_remove_valid_target():
    target = gt()
    target.update(truncation=1, occlusion=2)
    assert project([target], [dt()])["counts"]["hits"] == 1


def test_project_clips_boxes_but_official_style_keeps_raw_coordinates():
    target = gt(box=(-5, 5, 10, 10))
    prediction = dt(box=(0, 5, 5, 10))
    assert project([target], [prediction], iou_threshold=0.75)["counts"]["hits"] == 1
    assert EVAL.match_official_style([target], [prediction], 100, 100, iou_threshold=0.75)["matches"] == []


def test_matching_is_same_class_one_to_one_and_confidence_boundary_inclusive():
    result = project([gt()], [dt("wrongclass", category=2), dt("first", score=0.25), dt("second", score=0.2)])
    assert result["counts"] == {"eligible_targets": 1, "hits": 1, "misses": 0,
                                "false_positive_detections": 1, "score_0_absorbed_detections": 0,
                                "category_0_dropped_detections": 0}
    assert result["matches"][0]["detection_id"] == "first"
    assert result["detection_states"][2]["reason"] == "below_confidence"


def test_duplicate_detections_only_match_once():
    result = project([gt()], [dt("low", score=0.8), dt("high", score=0.9)])
    assert result["counts"]["hits"] == 1
    assert result["counts"]["false_positive_detections"] == 1
    assert result["matches"][0]["detection_id"] == "high"


def test_project_ties_use_later_gt_physical_line_and_detection_input_index():
    result = project([gt("earlier", line=1), gt("later", line=9)], [dt("first"), dt("second")])
    assert result["matches"][0]["target_id"] == "later"
    assert result["matches"][0]["detection_id"] == "second"


def test_project_iou_first_differs_from_official_style_score_first():
    targets = [gt()]
    predictions = [dt("highscore", (6, 5, 10, 10), score=0.95), dt("exact", score=0.8)]
    assert project(targets, predictions)["matches"][0]["detection_id"] == "exact"
    official = EVAL.match_official_style(targets, predictions, 100, 100)
    assert official["matches"][0]["detection_id"] == "highscore"


def test_official_style_input_order_cap_precedes_score_sort_and_class_selection():
    result = EVAL.match_official_style([gt()], [dt("wrong_first", category=2, score=0.1), dt("right_later", score=0.99)],
                                       100, 100, max_detections=1)
    assert result["matches"] == []
    assert result["detection_states"][0]["detection_id"] == "wrong_first"


def test_official_style_detection_ties_preserve_input_order_gt_ties_choose_later():
    result = EVAL.match_official_style([gt("earlier", line=1), gt("later", line=9)], [dt("first"), dt("second")], 100, 100)
    assert result["matches"][0]["detection_id"] == "first"
    assert result["matches"][0]["target_id"] == "later"


def test_official_style_ignored_gt_absorbs_multiple_detections():
    result = EVAL.match_official_style([gt("ignored", (1, 1, 80, 80), score=0)], [dt("one"), dt("two")], 100, 100)
    assert [row["state"] for row in result["detection_states"]] == ["ignored", "ignored"]


def test_empty_inputs_have_zero_counts_without_AP_claim():
    result = project([], [])
    assert result["counts"]["eligible_targets"] == result["counts"]["hits"] == 0
    assert result["official_AP"] is False and result["oracle_equivalence_verified"] is False


@pytest.mark.parametrize("targets,predictions", [([gt(box=(1, 1, 3, 0))], []),
                                               ([], [dt(box=(1, 1, -1, 3))]),
                                               ([gt(), gt("g0", line=1)], []),
                                               ([], [dt("duplicate"), dt("duplicate")]),
                                               ([], [dt(score=float("nan"))]),
                                               ([], [dt(score=True)]),
                                               ([], [dt(box=(1, 1, 1e200, 1e200))])])
def test_invalid_geometry_identity_or_scores_rejected(targets, predictions):
    with pytest.raises(EVAL.EvaluationInputError):
        project(targets, predictions)


def test_noninteger_category_zero_region_is_explicitly_rejected():
    with pytest.raises(EVAL.EvaluationInputError, match="integer region"):
        project([gt("region", (1.5, 1, 3, 3), category=0, score=0)], [])


def test_input_records_are_not_modified():
    targets, predictions = [gt()], [dt()]
    before = repr((targets, predictions))
    project(targets, predictions)
    assert repr((targets, predictions)) == before
