"""Full extension metadata/array tests; no image, calibration or model run."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gate1_w8a8_debug_full as F
P = F.P


def member(role, index, seed=""):
    identifier = f"{role}_{seed}_{index:04d}"
    row = {"image_id": "train/" + identifier, "assigned_split": role, "source_split": "train",
        "image_path": f"images/train/{identifier}.jpg", "image_sha256": P.ordered_hash(identifier), "width": 640, "height": 640}
    if role == "debug":
        row.update(annotation_path=f"annotations/train/{identifier}.txt", annotation_sha256="b"*64)
    return row


def contract():
    p = {"schema": F.SCHEMA, "seeds": F.SEEDS[:], "configurations": F.CONFIGS[:], "budget": F.budget(), "output_schema": F.output_schema(),
        "fusion": "BN_unfused_all_other_ops_FP32", "quantization": P.quantization_policy(), "postprocessing": P.postprocessing_policy(),
        "raw_tolerances": dict(P.TOLERANCES), "reference_policy": "strict_elementwise_FP32_identity_abort_on_any_difference",
        "input": {"size": 640, "batch": 1, "AMP": False, "device": "cuda:0", "deterministic": True, "TF32": False},
        "debug": [member("debug", i) for i in range(109)],
        "calibration": {seed: [member("calibration_pool", i, seed) for i in range(64)] for seed in F.SEEDS},
        "outcome_independent_extension": True, "scientific_acceptance": False}
    p["member_order_hashes"] = {"debug": P.ordered_hash(p["debug"]),
        "calibration": {seed: P.ordered_hash(p["calibration"][seed]) for seed in F.SEEDS}}
    return p


def receipts():
    binding = {"path": "frozen", "sha256": "a"*64, "bytes": 10}
    completed = {"status": "COMPLETED_BOUNDED_DEBUG_W8A8_SIMULATION_ONLY", "FP32_saved_reference_strict": True,
        "disabled_QDQ_before_after_strict": True, "original_and_wrapper_states_unchanged": True,
        "forward_calls": 120, "raw_arrays": 32, "contract": binding}
    launcher = {"status": "COMPLETED_BOUNDED_WORKER", "returncode": 0, "contract": binding}
    independent = {"status": F.AUDIT_PASS, "contract": binding}
    return completed, launcher, independent


def test_full_fixed_contract_accepted():
    F.validate(contract())


@pytest.mark.parametrize("change", ["seed", "configuration", "count", "order", "duplicate", "role", "calibration_label", "calibration_ID_overlap",
    "calibration_SHA_overlap", "reference", "tolerance", "fusion", "AMP", "budget", "layout", "NMS", "outcome", "scientific", "hash"])
def test_drift_or_forbidden_access_is_rejected(change):
    p = contract()
    if change == "seed": p["seeds"].reverse()
    elif change == "configuration": p["configurations"][-1] = "r_only:G9"
    elif change == "count": p["debug"].pop()
    elif change == "order": p["debug"].reverse()
    elif change == "duplicate": p["debug"][1] = deepcopy(p["debug"][0])
    elif change == "role": p["debug"][0]["assigned_split"] = "selection"
    elif change == "calibration_label": p["calibration"][F.SEEDS[0]][0]["label_path"] = "labels/a.txt"
    elif change == "calibration_ID_overlap": p["calibration"][F.SEEDS[1]][0] = deepcopy(p["calibration"][F.SEEDS[0]][0])
    elif change == "calibration_SHA_overlap": p["calibration"][F.SEEDS[1]][0]["image_sha256"] = p["calibration"][F.SEEDS[0]][0]["image_sha256"]
    elif change == "reference": p["reference_policy"] = "NMS_subset"
    elif change == "tolerance": p["raw_tolerances"]["box_atol"] = 5
    elif change == "fusion": p["fusion"] = "fused"
    elif change == "AMP": p["input"]["AMP"] = True
    elif change == "budget": p["budget"]["forward_calls"] += 1
    elif change == "layout": p["output_schema"]["differences"] = "omit_low_scores"
    elif change == "NMS": p["postprocessing"]["NMS"]["max_det"] = 200
    elif change == "outcome": p["outcome_independent_extension"] = False
    elif change == "scientific": p["scientific_acceptance"] = True
    elif change == "hash": p["member_order_hashes"]["debug"] = "a"*64
    with pytest.raises(ValueError): F.validate(p)


def test_prerequisites_use_technical_receipts_without_rescue_scores():
    assert F.prerequisites(*receipts())["rescue_results_used_to_select_extension"] is False
    c, l, i = receipts(); c["net_rescue"] = -999999
    assert F.prerequisites(c, l, i)["independent_saved_output_PASS"]


@pytest.mark.parametrize("change", ["incomplete", "strict", "state", "budget", "launcher", "exit", "audit", "identity"])
def test_missing_technical_prerequisite_rejected(change):
    c, l, i = receipts()
    if change == "incomplete": c["status"] = "RUNNING"
    elif change == "strict": c["FP32_saved_reference_strict"] = False
    elif change == "state": c["original_and_wrapper_states_unchanged"] = False
    elif change == "budget": c["forward_calls"] = 119
    elif change == "launcher": l["status"] = "FAILED"
    elif change == "exit": l["returncode"] = 1
    elif change == "audit": i["status"] = "PENDING"
    elif change == "identity": i["contract"] = {"sha256": "other"}
    with pytest.raises(ValueError): F.prerequisites(c, l, i)


def test_exact_forward_formula_and_unique_vs_logical_arrays():
    b = F.budget(); d = b["forward_breakdown"]
    assert sum(d["per_seed"].values()) == d["per_seed_total"] == 718
    assert d["shared_FP32_reference"] + d["seeds"]*d["per_seed_total"] == b["forward_calls"] == 2263
    assert b["logical_raw_arrays"] == 3*4*109 == 1308
    assert b["unique_raw_array_files"] == 109 + 3*3*109 == 1090
    assert b["timeout_seconds"] == 3600 and b["attempts"] == 1 and not b["automatic_retry"]


def raw():
    value = np.full((1,14,8400), .1, dtype=np.float32); value[:, :4] = 10
    return value


def test_full_difference_preserves_even_single_low_confidence_anchor():
    ref = raw(); candidate = ref.copy(); candidate[0,13,8399] = np.float32(.10001)
    difference, summary = F.raw_difference(ref, candidate)
    assert difference.shape == ref.shape and difference.dtype == np.float64
    assert summary["scores"]["changed_elements"] == 1 and summary["boxes"]["changed_elements"] == 0
    assert np.count_nonzero(difference) == 1 and difference[0,13,8399] != 0


def test_identical_reference_difference_exact_zero():
    difference, summary = F.raw_difference(raw(), raw())
    assert not np.any(difference) and summary["scores"]["elements"] == 84000


@pytest.mark.parametrize("change", ["dtype", "shape", "nan"])
def test_invalid_raw_difference_input_rejected(change):
    candidate = raw()
    if change == "dtype": candidate = candidate.astype(np.float16)
    elif change == "shape": candidate = candidate[:,:,:1]
    else: candidate[0,0,0] = np.nan
    with pytest.raises(ValueError): F.raw_difference(raw(), candidate)
