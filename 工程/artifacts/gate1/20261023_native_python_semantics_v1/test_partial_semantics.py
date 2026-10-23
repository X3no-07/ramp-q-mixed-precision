"""Pure synthetic adapter tests; no native oracle or expected AP values."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("partial_semantics_under_test", HERE / "compare_partial_semantics.py")
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)
EVALUATION = COMPARE.module(COMPARE.PYTHON_SOURCE, "partial_python_source_under_test")
FIXTURE = json.loads((COMPARE.READINESS / "synthetic_fixtures.json").read_text(encoding="utf-8"))


def case(identity):
    return next(row for row in FIXTURE["cases"] if row["id"] == identity)


@pytest.mark.parametrize("row", FIXTURE["cases"], ids=[row["id"] for row in FIXTURE["cases"]])
def test_all_twenty_fixed_cases_adapter_shapes_and_scope(row):
    for index, image in enumerate(row["images"]):
        result = COMPARE.python_intermediates(EVALUATION, row, image, f"{row['id']}.{index}")
        assert all(result[key].ndim == 2 and result[key].shape[1] == 8 for key in ("filtered_gt", "filtered_predictions", "prepared_gt"))
        assert "metrics" not in result
        if row["kind"] == "match":
            assert [value["category"] for value in result["class_matches"]] == list(range(1, 11))
            assert all(value["gt_output"].shape[1] == 5 and value["dt_output"].shape[1] == 6 for value in result["class_matches"])


def test_duplicate_identity_geometry_and_score_zero_are_not_collapsed():
    row = case("M03_ignore_absorbs_multiple_same_class")
    result = COMPARE.python_intermediates(EVALUATION, row, row["images"][0], row["id"])
    assert result["class_matches"][0]["gt_output"][0, 4] == -1
    assert result["class_matches"][0]["dt_output"][:, 5].tolist() == [-1, -1]
    assert result["class_matches"][1]["dt_output"][:, 5].tolist() == [0]


def test_actual_existing_matcher_called_and_project_matcher_never_called(monkeypatch):
    calls = []
    original = EVALUATION.match_official_style

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("The project IoU-first matcher must not be called")

    monkeypatch.setattr(EVALUATION, "match_official_style", observed)
    monkeypatch.setattr(EVALUATION, "match_project_targets", forbidden)
    row = case("M01_score_first_over_iou_first")
    COMPARE.python_intermediates(EVALUATION, row, row["images"][0], row["id"])
    assert calls == [{"iou_threshold": 0.5, "max_detections": 500}]


def test_reconstructed_matcher_raw_coordinates_are_not_clipped():
    row = case("M04_raw_outside_coordinates")
    result = COMPARE.python_intermediates(EVALUATION, row, row["images"][0], row["id"])
    assert result["class_matches"][0]["gt_output"][0, 0] == -5


def test_score_and_ignore_sort_preserves_declared_python_tie_order():
    row = case("M02_tied_score_and_overlap")
    result = COMPARE.python_intermediates(EVALUATION, row, row["images"][0], row["id"])
    gt = result["class_matches"][0]["gt_output"]
    assert gt[:, 0].tolist() == [4, 6]


def test_exact_comparison_does_not_hide_tiny_difference_or_row_order():
    original = np.array([[1.0, 2.0], [3.0, 4.0]])
    altered = original.copy()
    altered[0, 0] = np.nextafter(1.0, 2.0)
    difference = COMPARE.compare_array(original, altered, "synthetic")
    assert not difference["equal"] and difference["differing_elements"] == 1
    assert not COMPARE.compare_array(original, original[::-1], "synthetic")["equal"]


def test_empty_shapes_are_part_of_equality():
    assert COMPARE.compare_array(np.empty((0, 8)), np.empty((0, 8)), "empty")["equal"]
    assert not COMPARE.compare_array(np.empty((0, 8)), np.empty((0, 0)), "empty")["equal"]


def test_source_identity_is_the_existing_pinned_python_source():
    assert COMPARE.sha(COMPARE.PYTHON_SOURCE) == COMPARE.PYTHON_SOURCE_SHA
