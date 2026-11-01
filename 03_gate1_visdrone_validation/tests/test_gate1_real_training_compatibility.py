"""Synthetic negative cases for the read-only membership/quarantine auditor."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("real_compatibility", ROOT / "scripts/audit_gate1_real_training_compatibility.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def records():
    return [{"image_id": "train/one", "assigned_split": "train", "source_split": "train", "is_content_canonical": True,
             "requires_annotation_quality_quarantine": False, "image_sha256": "a" * 64, "prefix_proxy": "one", "refined_component": "one"},
            {"image_id": "train/two", "assigned_split": "training_val", "source_split": "train", "is_content_canonical": True,
             "requires_annotation_quality_quarantine": False, "image_sha256": "b" * 64, "prefix_proxy": "two", "refined_component": "two"}]


def test_explicit_roles_and_counts():
    rows = records()
    assert audit.check_membership(rows, {"train": 1, "training_val": 1}) == rows


@pytest.mark.parametrize("field", ["image_sha256", "prefix_proxy", "refined_component"])
def test_cross_role_overlap_rejected(field):
    rows = records()
    rows[1][field] = rows[0][field]
    with pytest.raises(ValueError, match="Cross-role overlap"):
        audit.check_membership(rows, {"train": 1, "training_val": 1})


@pytest.mark.parametrize("field,value", [("source_split", "val"), ("is_content_canonical", False),
                                         ("requires_annotation_quality_quarantine", True), ("image_id", "val/one")])
def test_invalid_training_membership_rejected(field, value):
    rows = records()
    rows[0][field] = value
    with pytest.raises(ValueError):
        audit.check_membership(rows, {"train": 1, "training_val": 1})


def test_duplicate_ids_and_changed_counts_rejected():
    rows = records()
    with pytest.raises(ValueError, match="counts"):
        audit.check_membership(rows, {"train": 2})
    rows[1]["image_id"] = rows[0]["image_id"]
    with pytest.raises(ValueError, match="Duplicate assignment"):
        audit.check_membership(rows, {"train": 1, "training_val": 1})


@pytest.mark.parametrize("relative", ["../images/train/x", "images/val/x", "images/train/../x", "images/train/holdout.jpg",
                                      "[OMITTED_LOCAL_PATH]", "images\\train\\x", "/images/train/x"])
def test_forbidden_members_rejected_before_read(tmp_path, relative):
    with pytest.raises(ValueError):
        audit.safe_member(tmp_path, relative, "image")


def test_wrong_file_identity_rejected(tmp_path):
    path = tmp_path / "member.txt"
    path.write_bytes(b"original")
    info = audit.file_info(path)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="identity changed"):
        audit.verified_bytes(info)


def test_quarantine_expands_whole_component_without_mutating_members():
    rows = records()
    sibling = deepcopy(rows[0])
    sibling["image_id"] = "train/sibling"
    rows.append(sibling)
    original = deepcopy(rows)
    result = audit.proposed_quarantine(rows, [{"image_id": "train/one", "issues": [{"reason": "bad"}]},
                                              {"image_id": "train/two", "issues": []}, {"image_id": "train/sibling", "issues": []}])
    assert rows == original and result["applied"] is False
    assert result["component_count"] == 1 and result["directly_incompatible_samples"] == 1
    assert {r["image_id"] for r in result["samples"]} == {"train/one", "train/sibling"}
    assert [r["trigger"] for r in result["samples"]] == ["direct_incompatibility", "same_existing_component"]


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'])
def test_ambiguous_json_rejected(raw):
    with pytest.raises(ValueError):
        audit.strict_json(raw)


def geometry_fixture(width=10.):
    spec = importlib.util.spec_from_file_location("compat_geometry_test", ROOT / "scripts/gate1_ignore_geometry.py")
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    identity = {"image_id": "train/one", "image_sha256": "a" * 64, "sidecar_sha256": "b" * 64}
    region = {"image_id": "train/one", "physical_line_zero_based": 0, "target_id": "train/one:0",
              "score": 0, "category_original": 1, "publisher_ignored": True, "ignored": True,
              "excluded_for_preparation_quality": False, "original_xywh": [10., 5., width, 10.],
              "original_xyxy": [10., 5., 10. + width, 15.], "clipped_xyxy": [10., 5., 10. + width, 15.],
              "ignore_reasons": ["score_0"]}
    return geometry, identity, region


@pytest.mark.parametrize("hw", [(80, 100), (100, 80)])
def test_geometry_runs_non_square_resize_all_flip_branches(hw):
    geometry, identity, region = geometry_fixture()
    assert audit.geometry_check(geometry, np, identity, hw, [region], 64) == {"publisher_regions": 1, "flip_branches": 4}


def test_explicit_empty_ignore_is_valid():
    geometry, identity, _ = geometry_fixture()
    assert audit.geometry_check(geometry, np, identity, (80, 100), [], 64)["publisher_regions"] == 0


def test_zero_area_ignore_rejects_whole_sample():
    geometry, identity, region = geometry_fixture(width=0.)
    with pytest.raises(ValueError, match="nonpositive publisher geometry"):
        audit.geometry_check(geometry, np, identity, (80, 100), [region], 64)


def test_final_float32_collapse_is_detected():
    geometry, identity, region = geometry_fixture(width=1e-9)
    with pytest.raises(ValueError, match="Final FP32 ignore geometry collapsed"):
        audit.geometry_check(geometry, np, identity, (80, 100), [region], 64)
