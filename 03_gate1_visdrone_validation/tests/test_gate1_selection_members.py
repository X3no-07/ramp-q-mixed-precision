"""Tiny fabricated rows only. No Selection or calibration content reads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate1_selection_members as M


def member(identifier="val/fake", role="selection", image_sha="a" * 64, component="fake_component"):
    split = "val" if role == "selection" else "train"
    stem = identifier.split("/")[-1]
    row = {"image_id": identifier, "assigned_split": role, "source_split": split,
        "image_path": f"images/{split}/{stem}.jpg", "image_sha256": image_sha,
        "width": 20, "height": 10, "refined_component": component, "prefix_proxy": "fake"}
    if role != "calibration_pool":
        row.update(annotation_path=f"annotations/{split}/{stem}.txt", annotation_sha256="b" * 64)
    return row


def debug_metadata():
    return {"debug": [{"image_id": f"train/fake_{index:03d}"} for index in range(109)]}


def good_evidence():
    return {"status": "PASS_ALL_109_DEBUG_STRICT_FP32_BEFORE_SELECTION_CONTENT", "images": 109,
        "all_strict_equal": True, "forward_calls": 109, "Selection_content_reads_before_gate": 0,
        "checks": [{"image_id": row["image_id"], "strict_equal": True} for row in debug_metadata()["debug"]]}


def test_gate_accepts_exact_ordered_109_evidence():
    M.require_debug_gate(debug_metadata(), good_evidence())


@pytest.mark.parametrize("field,value", [("images", 108), ("forward_calls", 110), ("images", True),
    ("all_strict_equal", 1), ("Selection_content_reads_before_gate", 1),
    ("status", "RUNNING"), ("Selection_content_reads_before_gate", False)])
def test_gate_rejects_incomplete_or_nonliteral_fields(field, value):
    evidence = good_evidence(); evidence[field] = value
    with pytest.raises(ValueError):
        M.require_debug_gate(debug_metadata(), evidence)


@pytest.mark.parametrize("change", ["reorder", "omit", "duplicate", "false", "truthy"])
def test_gate_rejects_member_or_result_corruption(change):
    evidence = good_evidence()
    if change == "reorder": evidence["checks"].reverse()
    elif change == "omit": evidence["checks"].pop()
    elif change == "duplicate": evidence["checks"][-1] = dict(evidence["checks"][0])
    elif change == "false": evidence["checks"][4]["strict_equal"] = False
    else: evidence["checks"][4]["strict_equal"] = 1
    with pytest.raises(ValueError):
        M.require_debug_gate(debug_metadata(), evidence)


def test_gate_failure_precedes_source_or_member_IO():
    evidence = good_evidence(); evidence["images"] = 108
    with patch.object(M, "validate_metadata"), patch.object(M, "verify_metadata_sources", side_effect=AssertionError("source I/O forbidden")), \
         patch.object(M, "checked_member_path", side_effect=AssertionError("member I/O forbidden")):
        with pytest.raises(ValueError):
            M.validate_content_after_debug_preflight(debug_metadata(), evidence)


def test_selection_and_debug_member_path_schema_and_hashes():
    M.validate_member(member(), "selection")
    M.validate_member(member("train/fake", "debug"), "debug")


@pytest.mark.parametrize("path", ["images/val/../fake.jpg", "[OMITTED_LOCAL_PATH]", "images/val/a\\fake.jpg", "images/train/fake.jpg"])
def test_escaped_or_wrong_role_member_path_rejected(path):
    row = member(); row["image_path"] = path
    with pytest.raises(ValueError):
        M.validate_member(row, "selection")


def test_calibration_member_excludes_annotation_and_target_fields():
    row = member("train/fake", "calibration_pool"); M.validate_member(row, "calibration_pool")
    for field in ("annotation_path", "label_sha256", "target_count"):
        bad = dict(row); bad[field] = "forbidden"
        with pytest.raises(ValueError):
            M.validate_member(bad, "calibration_pool")


def test_calibration_annotation_access_rejected_before_path_IO():
    row = member("train/fake", "calibration_pool")
    with pytest.raises(ValueError):
        M.checked_member_path({"calibration": {"synthetic": [row]}}, row, "annotation")


def test_metadata_overlap_distinguishes_ID_hash_component():
    left = member()
    right = member("val/other", image_sha="c" * 64)
    assert M.metadata_overlap([left], [right]) == {"image_id": 0, "image_sha256": 0, "refined_component": 1, "prefix_proxy": 1}


def test_duplicate_group_same_component_retains_both_IDs():
    rows = [member(), member("val/second")]
    group = M.duplicate_groups(rows)[0]
    assert group["member_count"] == 2 and group["image_ids"] == ["val/fake", "val/second"]
    rows[1]["refined_component"] = "other"
    with pytest.raises(ValueError):
        M.duplicate_groups(rows)


def test_physical_rows_keep_blanks_all_attributes_and_flags():
    rows = M.parse_physical_annotation(b"1,2,3,4,1,2,0,0,\r\n\r\n0,0,5,5,0,0,1,2\n")
    assert rows[0]["physical_line_zero_based"] == 0
    assert rows[1] == {"physical_line_zero_based": 1, "empty": True}
    assert rows[2]["values"] == [0, 0, 5, 5, 0, 0, 1, 2]
    assert rows == M.parse_physical_annotation(b"\xef\xbb\xbf1,2,3,4,1,2,0,0\n\n0,0,5,5,0,0,1,2\n")


@pytest.mark.parametrize("text", [b"1,2,3", b"nan,2,3,4,1,1,0,0", b"1,2,3,4,0.5,1,0,0",
    b"1,2,3,4,1,12,0,0", b"1,2,-1,4,1,1,0,0", b"1,2,3,4,1,1,4,0"])
def test_invalid_original_gt_stops(text):
    with pytest.raises(ValueError):
        M.parse_physical_annotation(text)


def duplicate_inputs():
    group = {"image_sha256": "a" * 64, "image_ids": ["val/a", "val/b"], "component": "component", "member_count": 2}
    physical = M.parse_physical_annotation(b"1,2,3,4,1,1,0,0\n")
    rows = [{"image_id": identifier, "actual_image_sha256": "a" * 64,
             "actual_annotation_sha256": "b" * 64, "annotation_physical_rows": deepcopy(physical)} for identifier in group["image_ids"]]
    return {"duplicate_groups": [group]}, rows


def test_duplicate_content_equal_physical_rows_passes():
    metadata, rows = duplicate_inputs()
    checks = M.validate_duplicate_content(metadata, rows)
    assert checks[0]["all_original_annotation_physical_rows_equal"] is True


@pytest.mark.parametrize("change", ["geometry", "score", "category", "attributes", "physical_line", "image_sha"])
def test_duplicate_content_any_full_row_or_identity_conflict_stops(change):
    metadata, rows = duplicate_inputs()
    if change == "image_sha": rows[1]["actual_image_sha256"] = "c" * 64
    elif change == "physical_line": rows[1]["annotation_physical_rows"][0]["physical_line_zero_based"] = 1
    else:
        index = {"geometry": 0, "score": 4, "category": 5, "attributes": 6}[change]
        rows[1]["annotation_physical_rows"][0]["values"][index] += 1
    with pytest.raises(ValueError):
        M.validate_duplicate_content(metadata, rows)


def test_canonical_order_hash_changes_on_reorder():
    assert M.canonical_hash(["a", "b"]) != M.canonical_hash(["b", "a"])


def test_duplicate_conflict_carries_complete_physical_evidence():
    metadata, rows = duplicate_inputs()
    rows[1]["annotation_physical_rows"][0]["values"][5] = 2
    with pytest.raises(M.ContentValidationError) as caught:
        M.validate_duplicate_content(metadata, rows)
    evidence = caught.value.evidence
    assert evidence["members"] == rows and evidence["group"] == metadata["duplicate_groups"][0]
    assert evidence["row_differences"][0]["left"]["values"][5] == 1
    assert evidence["row_differences"][0]["right"]["values"][5] == 2
    assert evidence["action"] == "STOP_PRESERVE_NO_AUTOMATIC_REPAIR"


def fake_preparation_sources():
    sources = M.metadata_sources()
    selection = []
    for index in range(548):
        content_index = 0 if index == 547 else index
        sha = hashlib.sha256(f"synthetic_val_{content_index}".encode()).hexdigest()
        component = 0 if index == 547 else index % 75
        selection.append(member(f"val/fake_{index:04d}", image_sha=sha, component=f"val_component_{component}"))
    debug = [member(f"train/debug_{index:04d}", "debug", hashlib.sha256(f"debug{index}".encode()).hexdigest(), "debug_component") for index in range(109)]
    for row in debug:
        row["prefix_proxy"] = "debug_prefix"
    calibration = {}
    for seed in M.SEEDS:
        calibration[seed] = [member(f"train/cal_{seed}_{index:03d}", "calibration_pool", hashlib.sha256(f"cal{seed}{index}".encode()).hexdigest(), "cal_component") for index in range(64)]
        for row in calibration[seed]:
            row["prefix_proxy"] = "cal_prefix"
    quarantine = member("train/q", "quarantine", "c" * 64, "val_component_0")
    quarantine["prefix_proxy"] = "fake"
    all_rows = selection + debug + [row for rows in calibration.values() for row in rows] + [quarantine]
    source_root = "[OMITTED_LOCAL_PATH]"
    full_cal = {seed: [{key: value for key, value in row.items() if key != "prefix_proxy"} for row in rows] for seed, rows in calibration.items()}
    data = {
        sources["assignment"]: "\n".join(json.dumps(row) for row in all_rows),
        sources["split_plan"]: json.dumps({"calibration_image_ids": {seed: [row["image_id"] for row in rows] for seed, rows in calibration.items()}}),
        sources["formal_protocol"]: json.dumps({"prepared_root": source_root, "cohorts": {"debug": debug}}),
        sources["three_seed_contract"]: json.dumps({"calibration": full_cal}),
        sources["formal_inference_manifest"]: json.dumps({"files": {"debug/raw/" + row["image_id"].split("/")[1] + ".npz": {"bytes": 1, "sha256": "a" * 64} for row in debug}}),
        sources["Selection_metadata_diagnostic"]: json.dumps({"members_ordered_metadata_only": [{"ordinal": index, "image_id": row["image_id"], "image_sha256": row["image_sha256"], "refined_component": row["refined_component"]} for index, row in enumerate(selection, 1)]}),
        sources["Selection_list"]: "\n".join(str(Path(source_root) / row["image_path"]).replace("\\", "/") for row in selection),
    }
    for seed, rows in calibration.items():
        data[sources["calibration_list_" + seed]] = "\n".join(str(Path(source_root) / row["image_path"]).replace("\\", "/") for row in rows)
    joint = M.canonical_hash([{"image_id": row["image_id"], "image_sha256": row["image_sha256"]} for row in selection])
    return sources, data, joint


def run_fake_prepare():
    sources, data, joint = fake_preparation_sources()
    def fake_text(path, *args, **kwargs):
        assert path in data, "metadata preparation tried to read a member or unknown source"
        return data[path]
    def fake_identity(path):
        assert path in sources.values(), "metadata preparation tried to hash a member"
        return {"path": str(path), "bytes": 1, "sha256": "f" * 64}
    with patch.object(M, "SELECTION_ORDER_SHA", joint), patch.object(M, "identity", side_effect=fake_identity), \
         patch.object(Path, "read_text", fake_text), patch.object(Path, "open", side_effect=AssertionError("raw open forbidden")), \
         patch.object(Path, "stat", side_effect=AssertionError("member stat forbidden")), \
         patch.object(Path, "exists", side_effect=AssertionError("member probe forbidden")), \
         patch.object(Path, "resolve", side_effect=AssertionError("member resolve forbidden")):
        return M.prepare_metadata(), joint


def test_prepare_reads_only_supplied_metadata_and_never_member_files_or_probes():
    prepared, _ = run_fake_prepare()
    assert prepared["Selection_content_files_opened_or_probed"] == 0
    assert prepared["cross_role_metadata_overlap"]["quarantine"]["refined_component"] == 1
    assert prepared["cross_role_metadata_overlap"]["quarantine"]["image_sha256"] == 0
    assert len(prepared["selection"]) == 548
    assert all(not any("annotation" in key or "label" in key for key in row) for rows in prepared["calibration"].values() for row in rows)


@pytest.mark.parametrize("change", ["selection_role", "selection_source", "annotation_train_path", "image_sha", "ID_order", "debug_order", "duplicate_component"])
def test_bound_metadata_role_sha_order_and_duplicate_risks_rejected(change):
    prepared, joint = run_fake_prepare()
    if change == "selection_role": prepared["selection"][0]["assigned_split"] = "holdout"
    elif change == "selection_source": prepared["selection"][0]["source_split"] = "train"
    elif change == "annotation_train_path": prepared["selection"][0]["annotation_path"] = "annotations/train/fake.txt"
    elif change == "image_sha": prepared["selection"][1]["image_sha256"] = "f" * 64
    elif change == "ID_order": prepared["selection"].reverse()
    elif change == "debug_order": prepared["debug"].reverse()
    else: prepared["selection"][-1]["refined_component"] = "other"
    with patch.object(M, "SELECTION_ORDER_SHA", joint), pytest.raises(ValueError):
        M.validate_metadata(prepared)
