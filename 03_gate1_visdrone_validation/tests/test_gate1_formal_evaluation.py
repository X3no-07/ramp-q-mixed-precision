from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_gate1_formal_candidate as E
import qualify_gate1_float_baseline as Q


def protocol():
    rows = {}
    for role, count in (("debug", 109), ("training_val", 414)):
        rows[role] = [{"image_id": f"train/{role}_{i:04d}", "assigned_split": role, "source_split": "train",
                      "image_path": f"images/train/{role}_{i:04d}.jpg", "annotation_path": f"annotations/train/{role}_{i:04d}.txt"}
                     for i in range(count)]
    return {"schema": "rampq.gate1.formal_candidate_evaluation_protocol/v1", "branches": E.BRANCHES[:],
            "candidate_selection": "fixed_final_epoch_30_FP32_EMA", "data_roles": ["debug", "training_val"],
            "raw_tolerances": {"box_atol": .5, "box_rtol": .005, "score_atol": .002, "score_rtol": .02}, "cohorts": rows}


def test_frozen_cohorts_accepted():
    E.validate_protocol(protocol())


@pytest.mark.parametrize("mutation", ["role", "traversal", "count", "duplicate", "order", "tolerance", "candidate", "branch"])
def test_contract_drift_rejected(mutation):
    p = protocol()
    if mutation == "role": p["cohorts"]["debug"][0]["assigned_split"] = "selection"
    if mutation == "traversal": p["cohorts"]["debug"][0]["image_path"] = "images/train/../../selection/x"
    if mutation == "count": p["cohorts"]["debug"].pop()
    if mutation == "duplicate": p["cohorts"]["debug"][1]["image_id"] = p["cohorts"]["debug"][0]["image_id"]
    if mutation == "order": p["cohorts"]["debug"].reverse()
    if mutation == "tolerance": p["raw_tolerances"]["box_atol"] = 2
    if mutation == "candidate": p["candidate_selection"] = "best_AP"
    if mutation == "branch": p["branches"].append("int8")
    with pytest.raises(ValueError): E.validate_protocol(p)


def test_blank_and_trailing_comma_annotations_keep_original_score():
    parsed = E.parse_annotation("1,2,3,4,0,0,0,0,\n\n5,6,7,8,1,2,0,0\n")
    assert parsed.shape == (2, 8) and parsed[:,4].tolist() == [0,1]


@pytest.mark.parametrize("text", ["1,2,3", "1,2,3,4,nan,1,0,0"])
def test_bad_annotation_rejected(text):
    with pytest.raises(ValueError): E.parse_annotation(text)


def test_all_anchor_guard_fails_even_when_all_scores_are_low():
    left = np.zeros((1, 14, 8400), dtype=np.float32)
    left[:,:4] = 10
    right = left.copy()
    right[0,0,8399] = 12
    check = Q.raw_guard(left, right)
    assert check["status"] == "FAIL_FIXED_TOLERANCES" and check["boxes"]["violations"] == 1
    summary = E.guards_summary([{"fp16":check}], "fp16")
    assert summary["fail_images"] == 1 and summary["boxes"]["elements"] == 33600
