"""Fabricated auditor inputs only; no official oracle metric answers are read."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("full_ap_comparator_under_test", HERE / "compare_full_ap.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class Structure:
    @staticmethod
    def cell(value, count, label):
        m.require(isinstance(value, list) and len(value) == count, "Synthetic cell count differs " + label)
        return value

    @staticmethod
    def struct(value, label):
        m.require(isinstance(value, dict), "Synthetic struct differs " + label)
        return value

    @staticmethod
    def text(value, label):
        return str(value)

    @staticmethod
    def scalar(value, label):
        return float(value)

    @staticmethod
    def native_bool(value, label):
        return value


class FabricatedImplementation:
    calls = []

    @classmethod
    def prepare_raw_image(cls, gt, dt, *, width, height):
        cls.calls.append(("prepare", width, height))
        return {"filtered_gt": gt.copy(), "filtered_predictions": dt.copy(), "prepared_gt": gt.copy()}

    @classmethod
    def eval_res(cls, gt, dt, threshold):
        cls.calls.append(("match", threshold))
        return gt.copy(), np.concatenate((dt, np.zeros((len(dt), 1))), axis=1)

    @classmethod
    def calc_accuracy(cls, allgt, alldt, *, capture_intermediates):
        cls.calls.append(("aggregate", capture_intermediates))
        return {"metrics": {name: float(index + .125) for index, name in enumerate(m.METRICS)}}


def fake_inputs():
    cases = []
    results = []
    for index, kind in enumerate(("aggregate", "region", "match")):
        image = {"width": 32, "height": 24, "raw_ground_truth": [[2., 3., 4., 5., 0., 1., 0., 0.]],
                 "raw_predictions": [[2., 3., 4., 5., .75, 1., -1., -1.]]}
        case = {"id": "FABRICATED_" + kind, "kind": kind, "purpose": "Fabricated comparison control only", "images": [image]}
        if kind == "match":
            case["threshold"] = .75
        cases.append(case)
        gt = np.asarray(image["raw_ground_truth"], dtype=np.float64)
        dt = np.asarray(image["raw_predictions"], dtype=np.float64)
        intermediate = {"filtered_gt": gt.copy(), "filtered_predictions": dt.copy(), "prepared_gt": gt.copy()}
        if kind == "match":
            intermediate["class_matches"] = []
            for category in range(1, 11):
                cgt, cdt = gt[gt[:, 5] == category, :5], dt[dt[:, 5] == category, :5]
                intermediate["class_matches"].append({"category": float(category), "gt_output": cgt.copy(),
                    "dt_output": np.concatenate((cdt, np.zeros((len(cdt), 1))), axis=1)})
        record = {"id": case["id"], "kind": kind, "purpose": case["purpose"], "status": "EXECUTED_NATIVE_SOURCE", "intermediates": [intermediate]}
        if kind == "aggregate":
            record["metrics"] = {name: np.asarray([[float(i + .125)]]) for i, name in enumerate(m.METRICS)}
        results.append(record)
    fixture = {"data_role": "synthetic", "expected_oracle_outputs": None, "no_real_data_or_model_access": True, "cases": cases}
    receipt = {"schema": "FABRICATED_NATIVE_SCHEMA", "runtime": "FABRICATED_RUNTIME", "fixture_count": 3., "case_errors": 0.,
               "real_data_read": False, "model_executed": False, "official_AP_equivalence_certified": False,
               "resolved_functions": {name: str(m.SOURCE_ROOT / "utils" / (name + ".m")) for name in m.FUNCTIONS}}
    expected = {"case_count": 3, "case_ids": [c["id"] for c in cases], "kinds": {"aggregate": 1, "region": 1, "match": 1},
                "matrix_count": 29, "metric_count": 7, "native_schema": "FABRICATED_NATIVE_SCHEMA", "native_runtime": "FABRICATED_RUNTIME"}
    return fixture, results, receipt, expected


def run_fake(values=None):
    fixture, results, receipt, expected = fake_inputs() if values is None else values
    FabricatedImplementation.calls = []
    return m.compare_suite(fixture, results, receipt, FabricatedImplementation, Structure, expected)


def test_all_categories_and_cases_covered():
    result = run_fake()
    assert result["matrix_count"] == 29 and result["metric_count"] == 7
    assert [row[0] for row in FabricatedImplementation.calls].count("match") == 10
    assert [row[0] for row in FabricatedImplementation.calls].count("prepare") == 3
    assert ("aggregate", False) in FabricatedImplementation.calls
    assert result["native_internal_AP_AR_tables_compared"] is False


@pytest.mark.parametrize("native,candidate", [(0., 0.), (-3., -3.), (100., 100.), (1./3., 1./3.), (float("nan"), float("nan"))])
def test_scalar_controls(native, candidate):
    assert m.compare_metric(candidate, np.asarray([[native]]), "fabricated")["native_is_nan"] == np.isnan(native)


def test_scalar_reduction_roundoff_permitted():
    assert m.compare_metric(50. + 1e-11, np.asarray([[50.]]), "fabricated")["absolute_error"] > 0


@pytest.mark.parametrize("native,candidate", [(float("nan"), 0.), (0., float("nan")), (float("inf"), float("inf")), (float("-inf"), 0.), (0., 2e-10), (100., 100.00000001)])
def test_scalar_corruptions_fail(native, candidate):
    with pytest.raises(m.ComparisonError):
        m.compare_metric(candidate, np.asarray([[native]]), "fabricated")


@pytest.mark.parametrize("value", [1, np.asarray(1, dtype=np.int64), np.asarray(1., dtype=np.float32), np.asarray([1., 2.]), np.ones((1, 1, 1))])
def test_scalar_dtype_shape_fail(value):
    with pytest.raises(m.ComparisonError):
        m.compare_metric(value, np.asarray([[1.]]), "fabricated")


def test_exact_matrix_disallows_even_tiny_change():
    a = np.asarray([[1., 2., 3., 4., 5.]])
    b = a.copy(); b[0, 4] = np.nextafter(b[0, 4], np.inf)
    with pytest.raises(m.ComparisonError):
        m.compare_matrix(a, b, 5, "fabricated")


@pytest.mark.parametrize("candidate", [np.zeros((0,)), np.zeros((0, 5)), np.zeros((0, 8), dtype=np.float32), np.asarray([[np.nan]*8])])
def test_empty_dimensions_dtype_and_finiteness_fail(candidate):
    with pytest.raises(m.ComparisonError):
        m.compare_matrix(candidate, np.zeros((0, 8)), 8, "fabricated")


def test_empty_matrix_shape_is_preserved():
    assert m.compare_matrix(np.zeros((0, 8)), np.zeros((0, 8)), 8, "fabricated")["shape"] == [0, 8]


@pytest.mark.parametrize("mutation", ["delete_metric", "add_metric", "native_metric_nan", "matrix_value", "delete_case", "reorder_case", "delete_class", "category_order", "wrong_runtime", "native_error", "model_scope", "extra_matrix_field", "dtype", "coverage_count", "fixture_scope", "expected_values", "extra_record_field", "official_shadow", "duplicate_id", "wrong_schema"])
def test_full_suite_corruptions_fail_closed(mutation):
    fixture, results, receipt, expected = fake_inputs()
    if mutation == "delete_metric": results[0]["metrics"].pop(m.METRICS[0])
    elif mutation == "add_metric": results[0]["metrics"]["invented"] = np.asarray([[1.]])
    elif mutation == "native_metric_nan": results[0]["metrics"][m.METRICS[0]] = np.asarray([[np.nan]])
    elif mutation == "matrix_value": results[1]["intermediates"][0]["filtered_gt"][0, 0] += .25
    elif mutation == "delete_case": results.pop()
    elif mutation == "reorder_case": results.reverse()
    elif mutation == "delete_class": results[2]["intermediates"][0]["class_matches"].pop()
    elif mutation == "category_order": results[2]["intermediates"][0]["class_matches"][0]["category"] = 2.
    elif mutation == "wrong_runtime": receipt["runtime"] = "OTHER"
    elif mutation == "native_error": receipt["case_errors"] = 1.
    elif mutation == "model_scope": receipt["model_executed"] = True
    elif mutation == "extra_matrix_field": results[0]["intermediates"][0]["saved_model"] = np.zeros((0, 8))
    elif mutation == "dtype": results[0]["intermediates"][0]["filtered_gt"] = results[0]["intermediates"][0]["filtered_gt"].astype(np.float32)
    elif mutation == "coverage_count": expected["matrix_count"] -= 1
    elif mutation == "fixture_scope": fixture["data_role"] = "selection"
    elif mutation == "expected_values": fixture["expected_oracle_outputs"] = {"metric": 5}
    elif mutation == "extra_record_field": results[0]["oracle_answer_override"] = 9
    elif mutation == "official_shadow": receipt["resolved_functions"]["VOCap"] = str(HERE / "VOCap.m")
    elif mutation == "duplicate_id": fixture["cases"][1]["id"] = fixture["cases"][0]["id"]
    elif mutation == "wrong_schema": receipt["schema"] = "OTHER"
    with pytest.raises((m.ComparisonError, ValueError)):
        run_fake((fixture, results, receipt, expected))


def test_stress_generator_is_deterministic_and_exceeds_all_caps():
    generator = m.load_module(HERE / "generate_stress_fixtures.py", "fabricated_stress_generator_under_test")
    first, second = generator.generate(), generator.generate()
    assert first == second and len(first["cases"]) == 12
    assert first["expected_oracle_outputs"] is None and first["data_role"] == "synthetic"
    assert len(first["cases"][0]["images"][0]["raw_predictions"]) == 523
    assert len(first["cases"][9]["images"][0]["raw_predictions"]) == 620
    assert first["cases"][1]["images"][0]["raw_predictions"] == list(reversed(first["cases"][0]["images"][0]["raw_predictions"]))
