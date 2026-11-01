"""Pure metadata/array/target tests; no real images, weights or model execution."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gate1_w8a8_debug_pilot as P


def member(role, index):
    row = {"image_id": f"train/{role}_{index:03d}", "assigned_split": role, "source_split": "train",
        "image_path": f"images/train/{role}_{index:03d}.jpg", "image_sha256": "a" * 64, "width": 640, "height": 640}
    if role == "debug":
        row.update(annotation_path=f"annotations/train/{role}_{index:03d}.txt", annotation_sha256="b" * 64)
    return row


def contract():
    p = {"schema": P.SCHEMA, "configurations": P.CONFIGS[:], "seed": P.SEED,
        "raw_tolerances": dict(P.TOLERANCES), "reference_policy": "strict_elementwise_FP32_identity_abort_on_any_difference",
        "input": {"size": 640, "batch": 1, "AMP": False, "device": "cuda:0", "deterministic": True, "TF32": False},
        "fusion": "BN_unfused_all_other_ops_FP32", "quantization": P.quantization_policy(), "budget": P.budget_policy(),
        "debug": [member("debug", index) for index in range(8)],
        "calibration": [member("calibration_pool", index) for index in range(64)], "postprocessing": P.postprocessing_policy()}
    p["member_order_hashes"] = {key: P.ordered_hash(p[key]) for key in ("debug", "calibration")}
    return p


def test_fixed_metadata_contract_is_accepted():
    P.validate_contract(contract())


@pytest.mark.parametrize("change", ["selection", "traversal", "calib_label", "duplicate", "debug_order", "count", "seed", "config", "amp", "tolerance", "reference", "fusion", "budget", "nms", "calibration_algorithm", "hash"])
def test_contract_drift_or_forbidden_access_rejected(change):
    p = contract()
    if change == "selection": p["debug"][0]["assigned_split"] = "selection"
    elif change == "traversal": p["debug"][0]["image_path"] = "images/train/../../holdout/a.jpg"
    elif change == "calib_label": p["calibration"][0]["annotation_path"] = "annotations/train/a.txt"
    elif change == "duplicate": p["debug"][1] = deepcopy(p["debug"][0])
    elif change == "debug_order": p["debug"].reverse()
    elif change == "count": p["calibration"].pop()
    elif change == "seed": p["seed"] = "picked_after_results"
    elif change == "config": p["configurations"][-1] = "r_only:G9"
    elif change == "amp": p["input"]["AMP"] = True
    elif change == "tolerance": p["raw_tolerances"]["box_atol"] = 3
    elif change == "reference": p["reference_policy"] = "allclose_or_confident_subset"
    elif change == "fusion": p["fusion"] = "fused"
    elif change == "budget": p["budget"]["attempts"] = 2
    elif change == "nms": p["postprocessing"]["NMS"]["max_det"] = 50
    elif change == "calibration_algorithm": p["quantization"]["calibration"] = "optimize_using_debug_labels"
    elif change == "hash": p["member_order_hashes"]["calibration"] = "0" * 64
    with pytest.raises(ValueError): P.validate_contract(p)


def raw():
    value = np.full((1, 14, 8400), .1, dtype=np.float32)
    value[:, :4] = 10
    return value


def test_saved_raw_strict_equality_requires_all_117600_elements():
    reference = raw()
    assert P.strict_reference_check(reference, reference.copy())["elements"] == 117600
    candidate = reference.copy()
    candidate[0, 0, 8399] = np.nextafter(candidate[0, 0, 8399], np.float32(100))
    with pytest.raises(ValueError): P.strict_reference_check(reference, candidate)


@pytest.mark.parametrize("change", ["shape", "dtype", "nan", "negative_width", "score"])
def test_invalid_raw_domain_rejected(change):
    value = raw()
    if change == "shape": value = value[:, :, :1]
    elif change == "dtype": value = value.astype(np.float16)
    elif change == "nan": value[0, 7, 8399] = np.nan
    elif change == "negative_width": value[0, 2, 3] = -1
    elif change == "score": value[0, 13, 9] = 1.1
    with pytest.raises(ValueError): P.check_raw(value)


def target_states():
    hits = {"fp32": {"a", "b", "c", "d"}, "q_all": {"c", "d"},
        "q_only:G1": {"b", "c", "d"}, "r_only:G1": {"a", "c"}}
    return {configuration: {target: target in found for target in ["a", "b", "c", "d", "e"]} for configuration, found in hits.items()}


def test_same_target_rescue_collateral_and_two_way_definitions():
    observed = P.target_summary(target_states())
    assert observed["added_miss_ids_A"] == ["a", "b"]
    assert observed["stable_hit_ids_S"] == ["c", "d"]
    assert observed["rescued_ids"] == observed["two_way_same_target_ids"] == ["a"]
    assert observed["collateral_miss_ids"] == ["d"]
    assert observed["net_rescue"] == 0 and observed["rescue_rate"] == observed["stable_retention"] == .5


def test_zero_denominator_is_insufficient_not_perfect():
    states = {key: {"target": False} for key in P.CONFIGS}
    observed = P.target_summary(states)
    assert observed["rescue_rate"] is observed["net_rescue_rate"] is observed["stable_retention"] is None
    assert observed["evidence_status"] == "INSUFFICIENT_ZERO_DENOMINATOR"


def test_target_identity_drift_rejected():
    states = target_states(); states["r_only:G1"].pop("d")
    with pytest.raises(ValueError): P.target_summary(states)


def test_primitive_targets_must_be_boolean():
    states = target_states(); states["q_all"]["a"] = 1
    with pytest.raises(ValueError): P.target_summary(states)


def test_ordered_cohort_hash_detects_reordering():
    values = [member("calibration_pool", index) for index in range(2)]
    assert P.ordered_hash(values) != P.ordered_hash(list(reversed(values)))


def test_forward_budget_and_expansion_are_outcome_independent():
    budget = P.budget_policy()
    assert sum(budget["forward_breakdown"].values()) == budget["forward_calls"] == 120
    assert budget["raw_arrays_saved"] == 4 * 8
    assert "never rescue" in budget["expansion"]


def test_write_refuses_overwriting_evidence(tmp_path):
    path = tmp_path / "receipt.json"
    P.write(path, {"unchanged": True})
    with pytest.raises(FileExistsError): P.write(path, {"unchanged": False})


def test_finite_calibration_absmax_accumulates_without_label_input():
    import torch
    assert P.observed_maximum(2., torch.tensor([-3., 1.])) == 3.
    assert P.observed_maximum(3., torch.zeros(2)) == 3.


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_calibration_sample_cannot_be_silently_hidden_by_max(value):
    import torch
    with pytest.raises(ValueError): P.observed_maximum(0., torch.tensor([value]))


def test_calibration_hooks_count_observations_and_are_removed():
    import torch
    module = torch.nn.Conv2d(1, 1, 1).eval()
    collector = P.ActivationRanges([("model.0.conv", module, "G1")])
    with torch.no_grad():
        module(torch.tensor([[[[-2.]]]])); module(torch.tensor([[[[3.]]]]))
    collector.close()
    assert collector.max_abs == {"model.0.conv": 3.}
    assert collector.observations == {"model.0.conv": 2}
    with torch.no_grad(): module(torch.tensor([[[[9.]]]]))
    assert collector.max_abs == {"model.0.conv": 3.}
