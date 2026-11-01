"""New546 fixed subset metadata and debug-first content gate.

Never edits the failed548 sources/results. Metadata preparation reads only
saved binding/contract/failure evidence and metadata; real images/annotations
are opened only by the separately authorized runner after109 exact debug checks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import gate1_selection_members as L

ROOT = L.ROOT
OLD = L.OUT
OUT = ROOT / "artifacts/gate1/20261024_selection_546_fixed_G1_execution_v1"
FAILURE_AUDIT = ROOT / "artifacts/gate1/20261024_selection_content_failure_audit_v1"
SCHEMA = "rampq.gate1.fixed_G1_Selection546_member_binding/v1"
EXCLUDED_IDS = ["val/0000022_00000_d_0000004", "val/0000023_00000_d_0000008"]
OLD_METADATA_SHA = "3e0aa59975bc9708c94ce9a9bacac2550c072018bc5c5df1f67a4099ddff3ce9"
OLD_CONTRACT_SHA = "0ca8d06382f4f92a33fbb2c8786d1080322a2f46eaab3b4fecfdc8d3ae16f420"
OLD_HELPER_SHA = "edc65b993586d3bd57ef958b53f2241982f9aa67314399fa7c7293f34817ef0e"
SEEDS = L.SEEDS[:]


def require(value, message):
    L.require(value, message)


def read(path):
    return L.read(path)


def identity(path):
    return L.identity(path)


def canonical_hash(value):
    return L.canonical_hash(value)


def duplicate_policy():
    return {"retain_all_546_fixed_remaining_members": True, "excluded_original_548_IDs": EXCLUDED_IDS,
        "only_saved_conflicting_pair_excluded_no_other_deletions": True,
        "expected_duplicate_image_SHA_groups": 0, "remaining_content_conflict_action": "STOP_PRESERVE_NO_AUTOMATIC_REPAIR",
        "original_548_failed_attempt_and_evidence_preserved": True,
        "subset_is_not_original_548_member_AP": True, "independent_sample_or_significance_claim": False}


def sources():
    result = {"original548_metadata": OLD / "member_metadata.json", "original548_contract": OLD / "contract.json",
        "original548_helper": ROOT / "scripts/gate1_selection_members.py",
        "original548_failure": OLD / "attempt_01/failure_receipt.json",
        "original548_duplicate_evidence": OLD / "attempt_01/content_preflight_failure_evidence.json",
        "original548_terminal_manifest": OLD / "attempt_terminal_manifest.json",
        "original548_failure_audit": FAILURE_AUDIT / "audit_result.json"}
    result.update({"original_metadata_source_" + key: value for key, value in L.metadata_sources().items()})
    return result


def selection_id_sha_hash(rows):
    return canonical_hash([{"image_id": row["image_id"], "image_sha256": row["image_sha256"]} for row in rows])


def build_metadata(original, failure, bindings, cross_role):
    """Pure fixed filtering; inputs are saved metadata, never actual content."""
    L.validate_metadata(original)
    require([row["image_id"] for row in failure["members"]] == EXCLUDED_IDS and
        failure["group"]["image_ids"] == EXCLUDED_IDS and failure["stage"] == "DUPLICATE_ORIGINAL_ANNOTATION_ROWS" and
        failure["action"] == "STOP_PRESERVE_NO_AUTOMATIC_REPAIR" and bool(failure["row_differences"]), "Wrong saved conflict identity/stage")
    require(failure["group"] == original["duplicate_groups"][0], "Saved conflict group differs from old binding")
    old_by_id = {row["image_id"]: row for row in original["selection"]}
    for actual in failure["members"]:
        recorded = old_by_id[actual["image_id"]]
        require(actual["actual_image_sha256"] == recorded["image_sha256"] and
                actual["actual_annotation_sha256"] == recorded["annotation_sha256"] and
                actual["refined_component"] == recorded["refined_component"], "Saved conflicting member SHA/component differs")
    selected = [deepcopy(row) for row in original["selection"] if row["image_id"] not in EXCLUDED_IDS]
    result = {"schema": SCHEMA, "status": "BOUND_NEW546_METADATA_ONLY_NO_CONTENT_READ",
        "prepared_root": original["prepared_root"], "selection": selected, "debug": deepcopy(original["debug"]),
        "calibration": deepcopy(original["calibration"]), "excluded_ids": EXCLUDED_IDS[:],
        "origin548_metadata": bindings["original548_metadata"], "origin548_contract": bindings["original548_contract"],
        "metadata_bindings": bindings, "origin_548_ID_imageSHA_order_sha256": original["Selection_ID_imageSHA_order_sha256"],
        "Selection_ID_imageSHA_order_sha256": selection_id_sha_hash(selected),
        "Selection_ID_order_sha256": canonical_hash([row["image_id"] for row in selected]),
        "member_order_hashes": {"selection": canonical_hash(selected), "debug": canonical_hash(original["debug"]),
            "calibration": {seed: canonical_hash(original["calibration"][seed]) for seed in SEEDS}},
        "selection_metadata_counts": {"members": len(selected), "recorded_unique_image_SHA256": len({row["image_sha256"] for row in selected}),
            "refined_components": len({row["refined_component"] for row in selected}),
            "prefix_proxies": len({row["prefix_proxy"] for row in selected})},
        "cross_role_metadata_overlap": cross_role, "duplicate_groups": L.duplicate_groups(selected),
        "duplicate_policy": duplicate_policy(), "selection_unit": "new fixed546 development subset, not original548-member evaluation",
        "quarantine_component_overlap_policy": "inactive risk component/prefix overlap disclosed; exact ID/SHA disjoint and no quarantined content read",
        "Selection_content_files_opened_or_probed": 0, "calibration_labels_read": False,
        "Holdout_access": "SEALED", "Holdout_member_IDs_or_paths_output": False,
        "human_authorization_verified_by_helper": False,
        "access_requires": "new546 human decision and source-bound frozen runner; this metadata is not authorization"}
    validate_metadata(result, original=original)
    return result


def validate_metadata(metadata, *, original=None):
    require(metadata["schema"] == SCHEMA and metadata["excluded_ids"] == EXCLUDED_IDS, "Wrong546 schema or exclusion pair")
    selected, debug, calibration = metadata["selection"], metadata["debug"], metadata["calibration"]
    require(len(selected) == len({row["image_id"] for row in selected}) == 546 and
        len({row["image_sha256"] for row in selected}) == 546, "Wrong546 member/content identity count")
    require(not any(row["image_id"] in EXCLUDED_IDS for row in selected) and
        [row["image_id"] for row in selected] == sorted(row["image_id"] for row in selected), "Excluded member retained or original remaining order changed")
    require(len(debug) == len({row["image_id"] for row in debug}) == 109 and set(calibration) == set(SEEDS), "Debug/calibration scope changed")
    require([row["image_id"] for row in debug] == sorted(row["image_id"] for row in debug), "Debug order changed")
    for rows, role in ((selected, "selection"), (debug, "debug")):
        for row in rows:
            L.validate_member(row, role)
    combined = []
    for seed in SEEDS:
        require(len(calibration[seed]) == len({row["image_id"] for row in calibration[seed]}) == 64, "Calibration per-seed count changed")
        for row in calibration[seed]:
            L.validate_member(row, "calibration_pool")
        combined.extend(calibration[seed])
    require(len({row["image_id"] for row in combined}) == len({row["image_sha256"] for row in combined}) == 192, "Calibration actual member identities overlap")
    require(metadata["duplicate_groups"] == L.duplicate_groups(selected) == [], "Remaining546 duplicate group discovered")
    expected_counts = {"members": 546, "recorded_unique_image_SHA256": 546,
        "refined_components": len({row["refined_component"] for row in selected}), "prefix_proxies": len({row["prefix_proxy"] for row in selected})}
    require(metadata["selection_metadata_counts"] == expected_counts, "New546 component/count metadata differs")
    require(metadata["member_order_hashes"] == {"selection": canonical_hash(selected), "debug": canonical_hash(debug),
        "calibration": {seed: canonical_hash(calibration[seed]) for seed in SEEDS}}, "New546 member order binding differs")
    require(metadata["Selection_ID_imageSHA_order_sha256"] == selection_id_sha_hash(selected) and
        metadata["Selection_ID_order_sha256"] == canonical_hash([row["image_id"] for row in selected]), "New546 Selection identity/order hash differs")
    for role, overlap in metadata["cross_role_metadata_overlap"].items():
        fields = ("image_id", "image_sha256") if role == "quarantine" else tuple(overlap)
        require(all(overlap[field] == 0 for field in fields), "546 role/content metadata isolation differs")
    require(metadata["duplicate_policy"] == duplicate_policy() and metadata["Holdout_access"] == "SEALED"
        and metadata["Holdout_member_IDs_or_paths_output"] is False, "Subset/duplicate/Holdout policy changed")
    require(metadata["origin548_metadata"]["sha256"] == OLD_METADATA_SHA and metadata["origin548_contract"]["sha256"] == OLD_CONTRACT_SHA,
        "Original failed548 roots differ")
    if original is not None:
        require(selected == [row for row in original["selection"] if row["image_id"] not in EXCLUDED_IDS], "546 is not exact original548-minus-fixed-two in original order")
        require(metadata["prepared_root"] == original["prepared_root"] and debug == original["debug"] and calibration == original["calibration"], "New546 changed shared/debug/calibration sources")


def prepare_metadata():
    paths = sources()
    bindings = {key: identity(path) for key, path in paths.items()}
    require(bindings["original548_metadata"]["sha256"] == OLD_METADATA_SHA and
        bindings["original548_contract"]["sha256"] == OLD_CONTRACT_SHA and
        bindings["original548_helper"]["sha256"] == OLD_HELPER_SHA, "Original548 frozen preparation changed")
    original, failure, old_contract = (read(paths[key]) for key in ("original548_metadata", "original548_duplicate_evidence", "original548_contract"))
    require(old_contract["members_metadata"] == original, "Original failed548 contract/member metadata mismatch")
    require(read(paths["original548_failure"])["status"] == "FAILED_NO_AUTOMATIC_RETRY", "Original548 failure not preserved")
    require(read(paths["original548_failure_audit"])["status"] == "PASS_SAVED_FAILURE_EVIDENCE_AND_109_DEBUG_REPLAY_AUDIT_SELECTION_ATTEMPT_REMAINS_FAILED", "Original548 failure evidence audit missing")
    selected = [row for row in original["selection"] if row["image_id"] not in EXCLUDED_IDS]
    assignments = [json.loads(line) for line in L.metadata_sources()["assignment"].read_text(encoding="utf-8").splitlines()]
    require(not any(row["assigned_split"] in {"holdout", "testdev", "test_dev", "test-dev"} for row in assignments), "Unexpected sealed-role metadata; do not expose")
    overlaps = {role: L.metadata_overlap(selected, [row for row in assignments if row["assigned_split"] == role])
                for role in ("train", "training_val", "debug", "calibration_pool", "quarantine")}
    return build_metadata(original, failure, bindings, overlaps)


def verify_metadata_sources(metadata):
    require(metadata["metadata_bindings"] == {key: identity(path) for key, path in sources().items()}, "546 source metadata/code/failure evidence changed")
    original = read(sources()["original548_metadata"])
    validate_metadata(metadata, original=original)


def require_debug_gate(metadata, evidence):
    L.require_debug_gate(metadata, evidence)


def checked_member_path(metadata, row, kind):
    # Existing role-safe membership and path logic has no count548 assumption.
    return L.checked_member_path(metadata, row, kind)


def validate_content_after_debug_preflight(metadata, evidence):
    validate_metadata(metadata)
    require_debug_gate(metadata, evidence)
    verify_metadata_sources(metadata)
    content = []
    for row in metadata["selection"]:
        image_path = checked_member_path(metadata, row, "image")
        annotation_path = checked_member_path(metadata, row, "annotation")
        image_bytes, annotation_bytes = image_path.read_bytes(), annotation_path.read_bytes()
        require(hashlib.sha256(image_bytes).hexdigest() == row["image_sha256"] and
            hashlib.sha256(annotation_bytes).hexdigest() == row["annotation_sha256"], "546 content changed during read")
        require(L.image_dimensions(image_bytes) == (row["width"], row["height"]), "546 actual image geometry differs")
        physical = L.parse_physical_annotation(annotation_bytes)
        content.append({"image_id": row["image_id"], "actual_image_sha256": row["image_sha256"],
            "actual_annotation_sha256": row["annotation_sha256"], "width": row["width"], "height": row["height"],
            "refined_component": row["refined_component"], "annotation_physical_rows": physical})
    require(len(content) == len({row["actual_image_sha256"] for row in content}) == 546, "546 remaining actual image identities not unique")
    return {"status": "PASS_FIXED_546_SELECTION_CONTENT_IDENTITY_ONLY", "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_binding_sha256": canonical_hash(metadata), "debug_preflight_evidence_sha256": canonical_hash(evidence),
        "members": content, "images": 546, "unique_actual_image_SHA256": 546, "duplicate_checks": [],
        "excluded_original548_IDs": EXCLUDED_IDS, "origin548_contract": metadata["origin548_contract"],
        "all546_fixed_remaining_members_retained": True, "original548_failed_evidence_preserved": True,
        "member_unit": "new546 subset, not original548 official-member evaluation", "model_forward_or_AP_executed_by_helper": False,
        "calibration_labels_read": False, "Holdout_access": "SEALED", "scientific_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare",))
    parser.parse_args()
    data = prepare_metadata()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "member_metadata.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": data["status"], "member_binding": identity(path), "counts": data["selection_metadata_counts"],
        "new_Selection_content_read_or_probed": 0, "model_or_AP_calls": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
