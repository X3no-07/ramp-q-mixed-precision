"""Pure synthetic causal tests; no native oracle results or real data are read."""
import importlib.util
import math
from pathlib import Path
import struct

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gate1_official_ap.py"
SPEC = importlib.util.spec_from_file_location("gate1_official_ap_under_test", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def gt(x=5, y=5, w=10, h=10, *, category=1, ignore=0):
    return [x, y, w, h, ignore, category, 0, 0]


def dt(x=5, y=5, w=10, h=10, *, category=1, score=.9):
    return [x, y, w, h, score, category, -1, -1]


def calculate(gts, dts, *, trace=False):
    return EVAL.calc_accuracy(gts, dts, capture_intermediates=trace)


def at(result, category=1, threshold=.5, cap=500):
    return next(row for row in result["intermediates"] if row["category"] == category
                and row["iou_threshold"] == threshold and row["max_detections"] == cap)


def test_perfect_all_ten_classes_and_table_axes():
    result = calculate([[gt(category=c)] for c in range(1, 11)], [[dt(category=c)] for c in range(1, 11)])
    assert result["AP"].shape == (10, 10)
    assert result["AR"].shape == (10, 10, 4)
    np.testing.assert_array_equal(result["evalClass"], np.arange(1, 11))
    np.testing.assert_array_equal(result["AP"], np.full((10, 10), 100.))
    np.testing.assert_array_equal(result["AR"], np.full((10, 10, 4), 100.))
    assert all(value == 100. for value in result["metrics"].values())


def test_chosen_native_runtime_iou_threshold_float64_hex_constants():
    # Root's pure arithmetic probe used no GT, DT, or AP/matcher output.
    expected = ["3fe0000000000000", "3fe199999999999a", "3fe3333333333333", "3fe4cccccccccccd",
                "3fe6666666666666", "3fe8000000000000", "3fe999999999999a", "3feb333333333334",
                "3feccccccccccccd", "3fee666666666666"]
    assert [struct.pack(">d", value).hex() for value in EVAL.IOU_THRESHOLDS] == expected
    assert "Octave 11.3.0" in EVAL.THRESHOLD_RUNTIME_SCOPE


@pytest.mark.parametrize("gts,dts", [([], []), ([[]], [[]]), ([[]], [[dt()]])])
def test_empty_evaluated_class_keeps_nan_not_fabricated_zero(gts, dts):
    result = calculate(gts, dts)
    assert result["evalClass"].shape == (0,)
    assert all(math.isnan(value) for value in result["metrics"].values())
    assert not result["AP"].any()
    assert not result["AR"].any()


def test_gt_without_detections_has_zero_and_valid_evalclass():
    result = calculate([[gt()]], [[]], trace=True)
    assert all(value == 0. for value in result["metrics"].values())
    assert result["evalClass"].tolist() == [1]
    assert at(result)["by_image"][0]["dt_output"].shape == (0, 6)
    assert at(result)["det_match_score_and_flag"].shape == (0, 2)


def test_score_zero_gt_is_available_class_and_remains_in_denominator():
    mixed = calculate([[gt(), gt(50, 50, ignore=1)]], [[dt()]], trace=True)
    assert mixed["metrics"]["AP_all"] == 50.
    assert mixed["metrics"]["AR_500"] == 50.
    trace = at(mixed)
    assert trace["recall_denominator_including_ignored_gt"] == 2
    np.testing.assert_array_equal(trace["gt_match_flags"], [1, -1])
    ignored_only = calculate([[gt(ignore=1)]], [[dt()]])
    assert ignored_only["evalClass"].tolist() == [1]
    assert all(value == 0. for value in ignored_only["metrics"].values())


def test_ignored_detection_neither_tp_nor_fp_with_ignored_gt_denominator():
    result = calculate([[gt(), gt(40, 40, w=30, h=30, ignore=1)]],
                       [[dt(45, 45, w=5, h=5, score=.99), dt(score=.8)]], trace=True)
    trace = at(result)
    np.testing.assert_array_equal(trace["det_match_score_and_flag"][:, 1], [-1, 1])
    np.testing.assert_array_equal(trace["tp_cumulative"], [0, 1])
    np.testing.assert_array_equal(trace["fp_cumulative"], [0, 0])
    np.testing.assert_array_equal(trace["recall"], [0, .5])
    np.testing.assert_array_equal(trace["precision"], [0, 1])
    assert result["metrics"]["AP_all"] == 50.


def test_class_frequency_weighting_is_not_unique_class_macro_mean():
    result = calculate([[gt()], [gt()], [gt(category=2)]], [[dt()], [], [dt(category=2)]])
    np.testing.assert_array_equal(result["evalClass"], [1, 1, 2])
    np.testing.assert_array_equal(result["AP"][0], np.full(10, 50.))
    np.testing.assert_array_equal(result["AP"][1], np.full(10, 100.))
    assert result["metrics"]["AP_all"] == pytest.approx(200 / 3)
    assert result["metrics"]["AP_all"] != 75.


def test_input_order_cap_applies_across_classes_before_score_sort():
    low_other_class = dt(70, 70, category=2, score=.01)
    high_hit = dt(score=.99)
    capped_out = calculate([[gt()]], [[low_other_class, high_hit]])
    admitted = calculate([[gt()]], [[high_hit, low_other_class]])
    assert capped_out["metrics"]["AR_1"] == 0.
    assert admitted["metrics"]["AR_1"] == 100.
    assert capped_out["metrics"]["AR_10"] == admitted["metrics"]["AR_10"] == 100.


def test_input_order_cap_before_score_sort_within_class_too():
    miss, hit = dt(70, 70, score=.01), dt(score=.99)
    assert calculate([[gt()]], [[miss, hit]])["metrics"]["AR_1"] == 0.
    assert calculate([[gt()]], [[hit, miss]])["metrics"]["AR_1"] == 100.


@pytest.mark.parametrize("hit_index,expected", [(0, [100, 100, 100, 100]), (1, [0, 100, 100, 100]),
                         (10, [0, 0, 100, 100]), (100, [0, 0, 0, 100]), (500, [0, 0, 0, 0])])
def test_each_cap_boundary_and_ap_uses_only_cap500(hit_index, expected):
    rows = [dt(70, 70, category=2, score=.01)] * hit_index + [dt(score=.99)]
    result = calculate([[gt()]], [rows])
    for key, value in zip(("AR_1", "AR_10", "AR_100", "AR_500"), expected):
        assert result["metrics"][key] == value
    assert result["metrics"]["AP_all"] == expected[-1]


def test_cross_image_stable_tie_changes_ap_when_image_order_changes():
    missed_first = calculate([[], [gt()]], [[dt(70, 70, score=.9)], [dt(score=.9)]])
    hit_first = calculate([[gt()], []], [[dt(score=.9)], [dt(70, 70, score=.9)]])
    assert missed_first["metrics"]["AP_all"] == 50.
    assert hit_first["metrics"]["AP_all"] == 100.
    assert "stable" in missed_first["tie_policy"]


def test_within_image_stable_score_ties_preserve_submitted_order():
    miss, hit = dt(70, 70, score=.9), dt(score=.9)
    assert calculate([[gt()]], [[miss, hit]])["metrics"]["AP_all"] == 50.
    assert calculate([[gt()]], [[hit, miss]])["metrics"]["AP_all"] == 100.


def test_voc_envelope_integration_not_raw_precision_average_or_sampling():
    assert EVAL.voc_ap([.5, 1.], [.5, 2 / 3]) == pytest.approx(2 / 3)
    assert EVAL.voc_ap([1 / 3], [1.]) == pytest.approx(1 / 3)
    assert EVAL.voc_ap([], []) == 0.
    result = calculate([[gt(), gt(50, 50)]], [[dt(80, 80, score=.99), dt(score=.9), dt(50, 50, score=.8)]])
    assert result["metrics"]["AP_all"] == pytest.approx(200 / 3)


def test_all_ten_thresholds_and_ap75_slice():
    # IoU=0.6 exactly: x shift 2.5 for 10-wide equal boxes.
    result = calculate([[gt(0, 0)]], [[dt(2.5, 0)]])
    np.testing.assert_array_equal(result["AP"][0], [100, 100, 100, 0, 0, 0, 0, 0, 0, 0])
    assert result["metrics"]["AP_50"] == 100.
    assert result["metrics"]["AP_75"] == 0.
    assert result["metrics"]["AP_all"] == 30.


def test_matcher_score_first_prefers_earlier_higher_score_not_highest_iou():
    gt_out, dt_out = EVAL.eval_res([gt()[:5]], [dt(6, 5, score=.95)[:5], dt(score=.8)[:5]], .5)
    np.testing.assert_array_equal(dt_out[:, 5], [1, 0])
    assert dt_out[0, 0] == 6
    assert gt_out[0, 4] == 1


def test_equal_overlap_gt_tie_prefers_later_nonignored_gt():
    targets = [gt(4, 5)[:5], gt(6, 5)[:5]]
    detections = [dt(5, 5, score=.9)[:5], dt(2, 5, score=.8)[:5]]
    _, matches = EVAL.eval_res(targets, detections, .5)
    _, swapped_matches = EVAL.eval_res(targets[::-1], detections, .5)
    np.testing.assert_array_equal(matches[:, 5], [1, 1])
    np.testing.assert_array_equal(swapped_matches[:, 5], [1, 0])


def test_positive_preferred_to_higher_overlap_ignore_and_ignore_reuse():
    targets = [gt(1, 1, w=80, h=80, ignore=1)[:5], gt()[:5]]
    detections = [dt()[:5], dt(2, 2, w=5, h=5, score=.8)[:5], dt(3, 3, w=5, h=5, score=.7)[:5]]
    gt_out, dt_out = EVAL.eval_res(targets, detections, .5)
    np.testing.assert_array_equal(gt_out[:, 4], [1, -1])
    np.testing.assert_array_equal(dt_out[:, 5], [1, -1, -1])


def test_multiple_flag_mirrors_literal_source_behavior():
    _, dt_out = EVAL.eval_res([gt()[:5]], [dt()[:5], dt(score=.8)[:5]], .5, allow_multiple=True)
    np.testing.assert_array_equal(dt_out[:, 5], [1, -1])


def test_overlap_continuous_geometry_ignore_area_and_empty_dimensions():
    np.testing.assert_array_equal(EVAL.comp_oas([[0, 0, 10, 10]], [[0, 0, 20, 20]]), [[.25]])
    np.testing.assert_array_equal(EVAL.comp_oas([[0, 0, 10, 10]], [[0, 0, 20, 20]], [1]), [[1.]])
    assert EVAL.comp_oas([], [[0, 0, 1, 1]]).shape == (0, 1)
    assert EVAL.comp_oas([[0, 0, 1, 1]], []).shape == (1, 0)
    np.testing.assert_array_equal(EVAL.comp_oas([[0, 0, 0, 0]], [[0, 0, 1, 1]]), [[0.]])
    np.testing.assert_array_equal(EVAL.comp_oas([[0, 0, -1, 5]], [[0, 0, 1, 1]]), [[0.]])
    assert EVAL.comp_oas([[-5, 0, 10, 10]], [[0, 0, 5, 10]])[0, 0] == .5


def test_raw_adapter_filters_regions_before_cap_inverts_score_and_preserves_attributes():
    region, target, ignored = gt(1, 1, w=4, h=9, category=0, ignore=0), gt(1, 1, w=8, h=8, ignore=1), gt(15, 15, ignore=0)
    # Here raw ignore field is publisher score: region0, target1, ignored0.
    ignored[6:] = [2, 1]
    raw_gt = np.asarray([region, target, ignored], dtype=float)
    raw_dt = np.asarray([dt(1, 1, w=8, h=8), dt(15, 15, score=.7)], dtype=float)
    original_gt, original_dt = raw_gt.copy(), raw_dt.copy()
    prepared = EVAL.prepare_raw_image(raw_gt, raw_dt, width=30, height=30)
    assert prepared["filtered_gt"].shape == (1, 8)
    np.testing.assert_array_equal(prepared["filtered_gt"][0], ignored)
    assert prepared["prepared_gt"][0, 4] == 1
    np.testing.assert_array_equal(prepared["prepared_gt"][0, 6:], [2, 1])
    assert len(prepared["filtered_predictions"]) == 1
    np.testing.assert_array_equal(raw_gt, original_gt)
    np.testing.assert_array_equal(raw_dt, original_dt)


def test_raw_adapter_empty_shapes_and_convenience_api():
    prepared = EVAL.prepare_raw_image([], [], width=10, height=10)
    assert all(value.shape == (0, 8) for value in prepared.values())
    result = EVAL.evaluate_raw_images([{"raw_ground_truth": [[5, 5, 10, 10, 1, 1, 0, 0]],
          "raw_predictions": [dt()], "width": 100, "height": 100}], capture_intermediates=False)
    assert result["metrics"]["AP_all"] == 100.


def test_prepared_inputs_and_arrays_are_never_mutated_or_aliased():
    targets = np.asarray([gt(ignore=1), gt()], dtype=float)
    detections = np.asarray([dt(score=.8), dt(score=.9)], dtype=float)
    gt_copy, dt_copy = targets.copy(), detections.copy()
    result = calculate([targets], [detections], trace=True)
    np.testing.assert_array_equal(targets, gt_copy)
    np.testing.assert_array_equal(detections, dt_copy)
    result["intermediates"][0]["by_image"][0]["gt_output"][:] = 999
    np.testing.assert_array_equal(targets, gt_copy)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_is_refused(value):
    target = gt()
    target[0] = value
    with pytest.raises(EVAL.OfficialAPInputError):
        calculate([[target]], [[dt()]])


@pytest.mark.parametrize("bad", [[1, 2, 3], [[1] * 7], np.zeros((0, 7)), np.zeros((1, 8), dtype=bool),
                                 [["1"] * 8], np.zeros((1, 8), dtype=complex)])
def test_bad_matrix_domain_is_refused(bad):
    with pytest.raises(EVAL.OfficialAPInputError):
        calculate([bad], [[]])


@pytest.mark.parametrize("column,value", [(4, 2), (5, -1), (5, 12), (5, 1.5)])
def test_gt_flags_and_category_domain_refused(column, value):
    target = gt()
    target[column] = value
    with pytest.raises(EVAL.OfficialAPInputError):
        calculate([[target]], [[]])


@pytest.mark.parametrize("threshold", [-.1, 1.1, float("nan"), True])
def test_invalid_matching_threshold_refused(threshold):
    with pytest.raises(EVAL.OfficialAPInputError):
        EVAL.eval_res([], [], threshold)


def test_finite_input_overflow_is_refused():
    huge = gt(x=1e308, w=1e308)
    with pytest.raises(EVAL.OfficialAPInputError):
        calculate([[huge]], [[]])


def test_input_sequences_must_have_matching_lengths():
    with pytest.raises(EVAL.OfficialAPInputError):
        calculate([[]], [])


def test_finite_scores_outside_probability_range_are_not_rescaled_or_clipped():
    result = calculate([[gt()]], [[dt(score=-5), dt(70, 70, score=5)]], trace=True)
    trace = at(result)
    np.testing.assert_array_equal(trace["det_match_score_and_flag"][:, 0], [5, -5])
    assert result["metrics"]["AP_all"] == 50.


def test_no_category11_nor_category0_evalclass_but_predictions_consume_cap():
    result = calculate([[gt(category=11)]], [[dt(category=11)]])
    assert not result["evalClass"].size
    assert all(math.isnan(value) for value in result["metrics"].values())


@pytest.mark.parametrize("rec,prec", [([.5, .2], [.5, .5]), ([1.1], [.5]), ([.1], [float("nan")]),
                                     ([.5], [.5, .6]), ([[.5]], [[.5]])])
def test_invalid_voc_vectors_refused(rec, prec):
    with pytest.raises(EVAL.OfficialAPInputError):
        EVAL.voc_ap(rec, prec)
