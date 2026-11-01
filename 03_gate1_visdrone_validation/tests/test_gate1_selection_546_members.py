"""Pure synthetic fixed-subset guards; no actual image/annotation reads."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate1_selection_546_members as N


def synthetic_inputs():
    selected = []
    for index in range(548):
        identifier = N.EXCLUDED_IDS[index] if index < 2 else f"val/synthetic_{index:04d}"
        content = 0 if index < 2 else index
        row = {"image_id": identifier, "assigned_split": "selection", "source_split": "val",
            "image_path": f"images/val/m_{index}.jpg", "annotation_path": f"annotations/val/m_{index}.txt",
            "image_sha256": f"{content:064x}", "annotation_sha256": f"{index+600:064x}",
            "width": 20, "height": 10, "refined_component": "duplicate" if index < 2 else f"val_component_{index % 74}", "prefix_proxy": "val_prefix"}
        selected.append(row)
    selected.sort(key=lambda row: row["image_id"])
    debug = [{"image_id": f"train/debug_{index:03d}", "assigned_split": "debug", "source_split": "train",
        "image_path": f"images/train/d_{index}.jpg", "annotation_path": f"annotations/train/d_{index}.txt",
        "image_sha256": f"{index+2000:064x}", "annotation_sha256": "f" * 64, "width": 20, "height": 10,
        "refined_component": "debug_component", "prefix_proxy": "debug_prefix"} for index in range(109)]
    calibration = {seed: [{"image_id": f"train/cal_{seed}_{index:03d}", "assigned_split": "calibration_pool", "source_split": "train",
        "image_path": f"images/train/c_{seed}_{index}.jpg", "image_sha256": f"{s*64+index+3000:064x}", "width": 20, "height": 10,
        "refined_component": "cal_component", "prefix_proxy": "cal_prefix"} for index in range(64)] for s, seed in enumerate(N.SEEDS)}
    group = {"image_sha256": selected[0]["image_sha256"], "image_ids": N.EXCLUDED_IDS[:], "component": "duplicate", "member_count": 2}
    original = {"selection": selected, "debug": debug, "calibration": calibration, "prepared_root": "synthetic/root",
        "Selection_ID_imageSHA_order_sha256": "f" * 64, "duplicate_groups": [group]}
    by_id = {row["image_id"]: row for row in selected}
    failure = {"stage": "DUPLICATE_ORIGINAL_ANNOTATION_ROWS", "action": "STOP_PRESERVE_NO_AUTOMATIC_REPAIR", "group": group,
        "members": [{"image_id": identifier, "actual_image_sha256": by_id[identifier]["image_sha256"],
            "actual_annotation_sha256": by_id[identifier]["annotation_sha256"], "refined_component": "duplicate"} for identifier in N.EXCLUDED_IDS],
        "row_differences": [{"synthetic_only": True}]}
    bindings = {"original548_metadata": {"path": "synthetic/oldmeta", "bytes": 1, "sha256": N.OLD_METADATA_SHA},
        "original548_contract": {"path": "synthetic/oldcontract", "bytes": 1, "sha256": N.OLD_CONTRACT_SHA}}
    overlaps = {role: dict.fromkeys(("image_id", "image_sha256", "refined_component", "prefix_proxy"), 0)
                for role in ("train", "training_val", "debug", "calibration_pool", "quarantine")}
    overlaps["quarantine"]["refined_component"] = 24; overlaps["quarantine"]["prefix_proxy"] = 24
    return original, failure, bindings, overlaps


def prepared():
    args = synthetic_inputs()
    with patch.object(N.L, "validate_metadata"):
        return N.build_metadata(*args), args[0]


def test_exact_original548_minus_fixed_two_keeps_remaining_order_and_all_fields():
    new, original = prepared()
    assert new["selection"] == [row for row in original["selection"] if row["image_id"] not in N.EXCLUDED_IDS]
    assert len(new["selection"]) == len({row["image_sha256"] for row in new["selection"]}) == 546
    assert new["selection_metadata_counts"]["refined_components"] == 74
    assert new["debug"] == original["debug"] and new["calibration"] == original["calibration"]


@pytest.mark.parametrize("change", ["wrong_pair", "no_saved_difference", "wrong_stage", "wrong_group", "wrong_annotation_SHA"])
def test_saved_conflict_identity_is_required_before_new_subset(change):
    original, failure, bindings, overlaps = synthetic_inputs()
    if change == "wrong_pair": failure["members"][0]["image_id"] = "val/not_the_conflict"
    elif change == "no_saved_difference": failure["row_differences"] = []
    elif change == "wrong_stage": failure["stage"] = "SOMETHING_ELSE"
    elif change == "wrong_group": failure["group"] = dict(failure["group"], component="other")
    else: failure["members"][0]["actual_annotation_sha256"] = "0" * 64
    with patch.object(N.L, "validate_metadata"), pytest.raises(ValueError):
        N.build_metadata(original, failure, bindings, overlaps)


@pytest.mark.parametrize("change", ["drop_extra", "reinstate_excluded", "reorder", "duplicate_SHA", "wrong_origin", "wrong_policy", "wrong_role", "unsafe_path", "calibration_label", "component_count", "order_hash"])
def test_546_binding_tampering_stops(change):
    new, original = prepared()
    if change == "drop_extra": new["selection"].pop()
    elif change == "reinstate_excluded": new["selection"][0] = deepcopy(original["selection"][0])
    elif change == "reorder": new["selection"].reverse()
    elif change == "duplicate_SHA": new["selection"][1]["image_sha256"] = new["selection"][0]["image_sha256"]
    elif change == "wrong_origin": new["origin548_contract"]["sha256"] = "0" * 64
    elif change == "wrong_policy": new["duplicate_policy"]["expected_duplicate_image_SHA_groups"] = 1
    elif change == "wrong_role": new["selection"][0]["assigned_split"] = "holdout"
    elif change == "unsafe_path": new["selection"][0]["image_path"] = "images/val/../forbidden.jpg"
    elif change == "calibration_label": new["calibration"][N.SEEDS[0]][0]["annotation_path"] = "forbidden"
    elif change == "component_count": new["selection_metadata_counts"]["refined_components"] += 1
    else: new["member_order_hashes"]["selection"] = "0" * 64
    with pytest.raises(ValueError):
        N.validate_metadata(new, original=original)


def test_metadata_only_filter_never_probes_member_paths():
    args = synthetic_inputs()
    with patch.object(N.L, "validate_metadata"), \
         patch.object(Path, "open", side_effect=AssertionError("member file open")), \
         patch.object(Path, "exists", side_effect=AssertionError("member probe")), \
         patch.object(Path, "resolve", side_effect=AssertionError("member resolve")), \
         patch.object(Path, "stat", side_effect=AssertionError("member stat")):
        data = N.build_metadata(*args)
    assert data["Selection_content_files_opened_or_probed"] == 0 and data["duplicate_groups"] == []


def test_new_content_gate_rejects_debug_failure_before_any_content_or_source_IO():
    new, _ = prepared()
    evidence = {"status": "PASS_ALL_109_DEBUG_STRICT_FP32_BEFORE_SELECTION_CONTENT", "images": 108,
        "forward_calls": 109, "all_strict_equal": True, "Selection_content_reads_before_gate": 0, "checks": []}
    with patch.object(N, "verify_metadata_sources", side_effect=AssertionError("source I/O after failedgate")), \
         patch.object(N, "checked_member_path", side_effect=AssertionError("Selection content beforegate")), pytest.raises(ValueError):
        N.validate_content_after_debug_preflight(new, evidence)


def test_calibration_annotation_remains_forbidden_without_touching_files():
    new, _ = prepared()
    with pytest.raises(ValueError):
        N.checked_member_path(new, new["calibration"][N.SEEDS[0]][0], "annotation")


def test_new546_policy_is_not_original548_eval_and_does_not_open_Holdout():
    new, _ = prepared()
    assert new["duplicate_policy"]["subset_is_not_original_548_member_AP"] is True
    assert new["Holdout_access"] == "SEALED" and new["Holdout_member_IDs_or_paths_output"] is False
