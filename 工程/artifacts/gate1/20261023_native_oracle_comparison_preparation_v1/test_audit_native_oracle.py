"""Fabricated bundles test the auditor, not official AP or native execution.

All metric values constructed below are intentionally dummy audit-test values.
They are not handwritten oracle answers and are never saved as native outputs.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("native_audit_under_test", HERE / "audit_native_oracle.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class FabricatedStruct:
    def __init__(self, **values):
        self._fieldnames = list(values)
        for key, value in values.items():
            setattr(self, key, value)


def native_string(value):
    return np.array([value])


def numeric(value):
    return np.array([[value]], dtype=float)


def cell(values):
    result = np.empty((len(values), 1), dtype=object)
    for index, value in enumerate(values):
        result[index, 0] = value
    return result


def native_json(value):
    if hasattr(value, "_fieldnames"):
        return {key: native_json(getattr(value, key)) for key in value._fieldnames}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [native_json(row) for row in value.ravel(order="F")]
        if value.dtype.kind in "US":
            return str(value.item())
        if value.size == 0:
            return []
        if value.size == 1:
            item = value.item()
            return None if isinstance(item, float) and np.isnan(item) else item
        return np.squeeze(value).tolist()
    raise AssertionError(type(value))


def fabricated_bundle():
    fixture = json.loads((AUDIT.READINESS / "synthetic_fixtures.json").read_text(encoding="utf-8"))
    records = []
    for case in fixture["cases"]:
        intermediates = []
        for image in case["images"]:
            # This is only a legal structural subsequence, not a region-filter oracle.
            gt = AUDIT.rows8(image["raw_ground_truth"])
            gt = gt[gt[:, 5] != 0]
            dt = AUDIT.rows8(image["raw_predictions"])
            prepared = gt.copy()
            prepared[:, 4] = 1 - gt[:, 4]
            fields = dict(filtered_gt=gt, filtered_predictions=dt, prepared_gt=prepared)
            if case["kind"] == "match":
                matches = []
                for category in range(1, 11):
                    gt0 = prepared[prepared[:, 5] == category, :5].copy()
                    gt0 = gt0[np.argsort(gt0[:, 4], kind="stable")]
                    gt0[:, 4] *= -1
                    dt0 = dt[dt[:, 5] == category, :5].copy()
                    dt0 = dt0[np.argsort(-dt0[:, 4], kind="stable")]
                    dt1 = np.column_stack([dt0, np.zeros(len(dt0))])
                    matches.append(FabricatedStruct(category=numeric(category), gt_output=gt0, dt_output=dt1))
                fields["class_matches"] = cell(matches)
            intermediates.append(FabricatedStruct(**fields))
        values = {key: native_string(case[key]) for key in ("id", "kind", "purpose")}
        values.update(intermediates=cell(intermediates), status=native_string("EXECUTED_NATIVE_SOURCE"))
        if case["kind"] == "aggregate":
            values["metrics"] = FabricatedStruct(**{name: numeric(12.345) for name in AUDIT.METRICS})
        records.append(FabricatedStruct(**values))
    results = cell(records)
    resolved = FabricatedStruct(**{name: native_string(str(AUDIT.SOURCES / "official_toolkit/utils" / (name + ".m"))) for name in AUDIT.FUNCTIONS})
    receipt = FabricatedStruct(schema=native_string(AUDIT.NATIVE_SCHEMA), runtime=native_string("fabricated-runtime-for-auditor-tests-only"),
                              computer=native_string("fabricated"), resolved_functions=resolved,
                              fixture_count=numeric(20), case_errors=numeric(0),
                              real_data_read=np.array([[False]]), model_executed=np.array([[False]]),
                              official_AP_equivalence_certified=np.array([[False]]),
                              scope=native_string("Fabricated structure test; never a native oracle receipt"))
    serial = {"receipt": native_json(receipt), "results": native_json(results)}
    return results, receipt, serial, fixture


def validate(bundle):
    return AUDIT.validate_native(*bundle, AUDIT.SOURCES / "official_toolkit")


def reserialize(bundle):
    results, receipt, _, fixture = bundle
    return results, receipt, {"receipt": native_json(receipt), "results": native_json(results)}, fixture


def test_fabricated_twenty_case_structure_only():
    result = validate(fabricated_bundle())
    assert result["fixture_count"] == 20
    assert result["kinds"] == {"aggregate": 11, "region": 4, "match": 5}
    assert all(row["method"] == "live_samefile" for row in result["resolved_function_path_checks"])


@pytest.mark.parametrize("mutation", ["reordered_case", "duplicate_case", "wrong_kind", "wrong_purpose", "case_error", "missing_case", "wrong_count", "scope_flag", "shadowed_function", "metric_inf", "metric_vector", "metric_deleted", "bad_empty_dimensions", "deleted_class", "duplicate_class", "missing_image", "score_flip", "matcher_dt_changed", "matcher_gt_changed", "matcher_flags", "deleted_score_zero_gt"])
def test_structural_corruptions_rejected(mutation):
    bundle = fabricated_bundle()
    results, receipt, serial, fixture = bundle
    if mutation == "reordered_case":
        results[0, 0], results[1, 0] = results[1, 0], results[0, 0]
    elif mutation == "duplicate_case":
        results[1, 0] = deepcopy(results[0, 0])
    elif mutation == "wrong_kind":
        results[0, 0].kind = native_string("region")
    elif mutation == "wrong_purpose":
        results[0, 0].purpose = native_string("changed")
    elif mutation == "case_error":
        receipt.case_errors = numeric(1)
    elif mutation == "missing_case":
        results = results[:-1]
    elif mutation == "wrong_count":
        receipt.fixture_count = numeric(19)
    elif mutation == "scope_flag":
        receipt.official_AP_equivalence_certified = np.array([[True]])
    elif mutation == "shadowed_function":
        receipt.resolved_functions.evalRes = native_string(str(HERE / "fake_evalRes.m"))
    elif mutation == "metric_inf":
        results[0, 0].metrics.AP_all = numeric(np.inf)
    elif mutation == "metric_vector":
        results[0, 0].metrics.AP_all = np.ones((1, 2))
    elif mutation == "metric_deleted":
        results[0, 0].metrics._fieldnames.remove("AP_all")
    elif mutation == "bad_empty_dimensions":
        results[1, 0].intermediates[0, 0].filtered_predictions = np.empty((0, 0))
    elif mutation == "deleted_class":
        results[15, 0].intermediates[0, 0].class_matches = results[15, 0].intermediates[0, 0].class_matches[:-1]
    elif mutation == "duplicate_class":
        results[15, 0].intermediates[0, 0].class_matches[1, 0].category = numeric(1)
    elif mutation == "missing_image":
        results[6, 0].intermediates = results[6, 0].intermediates[:-1]
    elif mutation == "score_flip":
        results[0, 0].intermediates[0, 0].prepared_gt[0, 4] = 1
    elif mutation == "matcher_dt_changed":
        results[15, 0].intermediates[0, 0].class_matches[0, 0].dt_output[0, 0] += 1
    elif mutation == "matcher_gt_changed":
        results[15, 0].intermediates[0, 0].class_matches[0, 0].gt_output[0, 0] += 1
    elif mutation == "matcher_flags":
        results[15, 0].intermediates[0, 0].class_matches[0, 0].gt_output[0, 4] = 2
    elif mutation == "deleted_score_zero_gt":
        results[17, 0].intermediates[0, 0].class_matches[0, 0].gt_output = np.empty((0, 5))
    with pytest.raises(AUDIT.AuditError):
        validate(reserialize((results, receipt, serial, fixture)))


def test_mat_nan_is_preserved_as_null_without_inventing_zero():
    bundle = fabricated_bundle()
    bundle[0][2, 0].metrics.AP_all = numeric(np.nan)
    result = validate(reserialize(bundle))
    observed = result["observations"][2]["native_metrics_observed_not_handwritten_expected"]["AP_all"]
    assert observed == {"value": None, "is_nan": True}


@pytest.mark.parametrize("mutation", ["null_for_finite", "zero_for_nan", "changed_row", "missing_receipt", "wrong_logical", "extra_metric"])
def test_json_corruptions_rejected(mutation):
    results, receipt, serial, fixture = fabricated_bundle()
    if mutation == "null_for_finite":
        serial["results"][0]["metrics"]["AP_all"] = None
    elif mutation == "zero_for_nan":
        results[2, 0].metrics.AP_all = numeric(np.nan)
        serial["results"][2]["metrics"]["AP_all"] = 0
    elif mutation == "changed_row":
        serial["results"][0]["intermediates"][0]["filtered_gt"][0] += 1
    elif mutation == "missing_receipt":
        del serial["receipt"]
    elif mutation == "wrong_logical":
        serial["receipt"]["real_data_read"] = 0
    elif mutation == "extra_metric":
        serial["results"][0]["metrics"]["invented"] = 0
    with pytest.raises(AUDIT.AuditError):
        validate((results, receipt, serial, fixture))


def test_equal_score_row_order_not_forced_to_python_stability():
    bundle = fabricated_bundle()
    class_one = bundle[0][16, 0].intermediates[0, 0].class_matches[0, 0]
    class_one.dt_output = class_one.dt_output[::-1].copy()
    validate(reserialize(bundle))


def test_descending_score_order_is_still_required():
    bundle = fabricated_bundle()
    class_one = bundle[0][15, 0].intermediates[0, 0].class_matches[0, 0]
    class_one.dt_output = class_one.dt_output[::-1].copy()
    with pytest.raises(AUDIT.AuditError, match="score ordering"):
        validate(reserialize(bundle))


@pytest.mark.parametrize("shape", [(0, 5), (0, 6), (0, 8)])
def test_empty_numeric_shape_preserved(shape):
    value = np.empty(shape)
    AUDIT.matrix(value, shape[1], "fabricated empty")
    AUDIT.compare_serialization(value, [], "fabricated empty")


def test_score_zero_gt_and_duplicate_class_occurrences_remain_observed():
    result = validate(fabricated_bundle())
    assert result["observations"][4]["images"][0]["surviving_score_zero_gt"] == 1
    assert len(result["observations"][6]["images"]) == 3


def test_frozen_old_sources_and_all_twelve_readiness_members_unchanged():
    fixture, sources = AUDIT.verify_frozen_inputs()
    assert len(fixture["cases"]) == 20 and len(sources["files"]) == 11


def test_scipy_mat_v7_roundtrip_preserves_zero_dimensions_and_nan(tmp_path):
    sys.path.insert(0, str(HERE / "mat_reader_dependencies"))
    from scipy.io import loadmat, savemat
    target = tmp_path / "fabricated_reader_test_only.mat"
    values = {"empty8": np.empty((0, 8)), "empty5": np.empty((0, 5)), "empty6": np.empty((0, 6)), "nan_value": numeric(np.nan)}
    savemat(target, values, do_compression=True)
    observed = loadmat(target, squeeze_me=False, struct_as_record=False, mat_dtype=True)
    assert observed["empty8"].shape == (0, 8)
    assert observed["empty5"].shape == (0, 5)
    assert observed["empty6"].shape == (0, 6)
    assert observed["nan_value"].shape == (1, 1) and np.isnan(observed["nan_value"].item())


def test_fabricated_nested_bundle_roundtrips_real_mat_reader(tmp_path):
    sys.path.insert(0, str(HERE / "mat_reader_dependencies"))
    from scipy.io import loadmat, savemat
    results, receipt, serial, fixture = fabricated_bundle()

    def savable(value):
        if hasattr(value, "_fieldnames"):
            return {key: savable(getattr(value, key)) for key in value._fieldnames}
        if isinstance(value, np.ndarray) and value.dtype == object:
            result = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                result[index] = savable(value[index])
            return result
        return value

    target = tmp_path / "fabricated_structural_bundle_never_native.mat"
    savemat(target, {"results": savable(results), "receipt": savable(receipt)}, do_compression=True, long_field_names=True)
    loaded = loadmat(target, squeeze_me=False, struct_as_record=False, chars_as_strings=True, mat_dtype=True)
    result = AUDIT.validate_native(loaded["results"], loaded["receipt"], serial, fixture, AUDIT.SOURCES / "official_toolkit")
    assert result["fixture_count"] == 20


def test_approved_isolated_reader_files_equal_official_wheel():
    result = AUDIT.verify_reader_dependency(HERE / "mat_reader_dependencies")
    assert result["verified_extracted_member_count"] == 1535
