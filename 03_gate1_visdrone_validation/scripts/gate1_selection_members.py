"""Bind fixed Selection metadata; content checks require the debug-first gate.

prepare_metadata reads only explicitly named existing metadata/manifests. It
never stats, resolves, hashes or opens any copied member path. Content access
is performed only by validate_content_after_debug_preflight in the later
source-bound runner; the helper itself grants no human authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gate1/20261024_selection_fixed_G1_execution_v1"
PLAN = ROOT / "artifacts/gate1/20261022_split_refinement/split_plan_v1"
FORMAL = ROOT / "artifacts/gate1/20261024_formal_candidate_evaluation_v1"
FULL = ROOT / "artifacts/gate1/20261024_w8a8_debug_3seed_v1"
META = ROOT / "artifacts/gate1/20261024_metadata_access_readiness_v1"
PROPOSAL = ROOT / "artifacts/gate1/20261024_selection_fixed_G1_proposal_v1"
SEEDS = ["2026102201", "2026102202", "2026102203"]
SCHEMA = "rampq.gate1.fixed_G1_Selection_member_binding/v1"
BASE_FIELDS = ("image_id", "assigned_split", "source_split", "image_path", "image_sha256", "width", "height", "refined_component", "prefix_proxy")
SELECTION_ORDER_SHA = "3caefdadd87d1925be31e3ef43f1d0266c8033ff6a1b2e7d0ebd85f5986487ec"


class ContentValidationError(ValueError):
    """A stop condition with serializable member evidence for the runner."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def metadata_sources():
    paths = {"assignment": PLAN / "split_assignment.jsonl", "split_plan": PLAN / "split_plan.json",
        "split_manifest": PLAN / "manifest.json", "Selection_list": PLAN / "splits/selection.txt",
        "formal_protocol": FORMAL / "protocol.json", "formal_inference_manifest": FORMAL / "inference/manifest.json",
        "three_seed_contract": FULL / "contract.json", "Selection_metadata_diagnostic": META / "result/Selection_order_and_isolation_metadata.json",
        "Selection_duplicate_policy_required": META / "Selection_duplicate_identity_policy_required.json",
        "proposal": PROPOSAL / "recommended_proposal.json", "report60": ROOT / "docs/60_gate1_selection_fixed_g1_development_reproduction_plan.md"}
    for seed in SEEDS:
        paths["calibration_list_" + seed] = PLAN / f"splits/calibration_seed_{seed}.txt"
    return paths


def validate_member(row, role):
    require(row["assigned_split"] == role, "Member role differs")
    split = "val" if role == "selection" else "train"
    require(row["source_split"] == split and row["image_id"].startswith(split + "/"), "Wrong original source role")
    require(type(row["width"]) is int and type(row["height"]) is int and min(row["width"], row["height"]) > 0, "Invalid image dimensions")
    fields = ["image"] + (["annotation"] if role in ("selection", "debug") else [])
    for key in fields:
        path = row[key + "_path"]
        prefix = ("images/" if key == "image" else "annotations/") + split + "/"
        require(isinstance(path, str) and path.startswith(prefix) and "\\" not in path and
                not Path(path).is_absolute() and not any(part in ("..", ".", "") for part in path.split("/")), "Unsafe member path")
        require(bool(re.fullmatch("[0-9a-f]{64}", row[key + "_sha256"])), "Invalid member SHA")
    if role == "calibration_pool":
        require(not any("annotation" in key or "label" in key or "target" in key for key in row), "Calibration labels prohibited")


def metadata_overlap(left, right):
    return {key: len({row[key] for row in left} & {row[key] for row in right})
            for key in ("image_id", "image_sha256", "refined_component", "prefix_proxy")}


def duplicate_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_sha256"]].append(row)
    output = []
    for image_sha, members in sorted(grouped.items()):
        if len(members) > 1:
            require(len({row["refined_component"] for row in members}) == 1, "Same content SHA crosses frozen components")
            output.append({"image_sha256": image_sha, "image_ids": [row["image_id"] for row in members],
                "component": members[0]["refined_component"], "member_count": len(members)})
    return output


def validate_metadata(metadata):
    require(metadata["schema"] == SCHEMA, "Wrong metadata schema")
    selection, debug, calibration = metadata["selection"], metadata["debug"], metadata["calibration"]
    for rows, role, count in ((selection, "selection", 548), (debug, "debug", 109)):
        require(len(rows) == len({row["image_id"] for row in rows}) == count, "Wrong fixed member count")
        require([row["image_id"] for row in rows] == sorted(row["image_id"] for row in rows), "Fixed order differs")
        for row in rows:
            validate_member(row, role)
    require(len({row["image_sha256"] for row in selection}) == 547 and
            len({row["refined_component"] for row in selection}) == 75, "Selection recorded identity count differs")
    require(set(calibration) == set(SEEDS), "Calibration seeds differ")
    all_calibration = [row for seed in SEEDS for row in calibration[seed]]
    require(len({row["image_id"] for row in all_calibration}) ==
            len({row["image_sha256"] for row in all_calibration}) == 192, "Calibration image identities overlap across seeds")
    for seed in SEEDS:
        require(len(calibration[seed]) == len({row["image_id"] for row in calibration[seed]}) == 64, "Calibration count differs")
        for row in calibration[seed]:
            validate_member(row, "calibration_pool")
        require(all(value == 0 for value in metadata_overlap(selection, calibration[seed]).values()), "Selection/calibration metadata isolation differs")
    require(all(value == 0 for value in metadata_overlap(selection, debug).values()), "Selection/debug metadata isolation differs")
    require(metadata["duplicate_groups"] == duplicate_groups(selection) and
            [group["member_count"] for group in metadata["duplicate_groups"]] == [2], "Duplicate metadata policy differs")
    require(metadata["member_order_hashes"] == {"selection": canonical_hash(selection), "debug": canonical_hash(debug),
        "calibration": {seed: canonical_hash(calibration[seed]) for seed in SEEDS}}, "Bound member order hashes differ")
    require(metadata["Selection_ID_imageSHA_order_sha256"] == SELECTION_ORDER_SHA ==
            canonical_hash([{"image_id": row["image_id"], "image_sha256": row["image_sha256"]} for row in selection]), "Selection source joint order differs")
    for role, overlap in metadata["cross_role_metadata_overlap"].items():
        keys = ("image_id", "image_sha256") if role == "quarantine" else tuple(overlap)
        require(all(overlap[key] == 0 for key in keys), "Selection active-role or recorded image identity isolation differs")
    require(metadata["duplicate_policy"] == {"retain_all_548_original_members": True,
        "same_image_SHA_requires_same_original_annotation_physical_rows": True,
        "same_content_cross_component_or_role_action": "STOP_PRESERVE_NO_AUTOMATIC_REPAIR",
        "image_or_annotation_identity_conflict_action": "STOP_PRESERVE_NO_AUTOMATIC_REPAIR",
        "content_dependence_group": "actual verified image SHA; original member IDs remain separate",
        "independent_sample_or_significance_claim": False}, "Duplicate policy changed")
    require(metadata["Holdout_access"] == "SEALED" and metadata["Holdout_member_IDs_or_paths_output"] is False, "Holdout boundary changed")


def prepare_metadata():
    sources = metadata_sources()
    assignments = [json.loads(line) for line in sources["assignment"].read_text(encoding="utf-8").splitlines()]
    require(not any(row["assigned_split"] in {"holdout", "testdev", "test_dev", "test-dev"} for row in assignments), "Unexpected sealed role metadata; no exposure")
    by_id = {row["image_id"]: row for row in assignments}
    require(len(by_id) == len(assignments), "Duplicate metadata member IDs")
    formal, full, plan = (read(sources[key]) for key in ("formal_protocol", "three_seed_contract", "split_plan"))
    diagnostic = read(sources["Selection_metadata_diagnostic"])
    selection = []
    for source in sorted((row for row in assignments if row["assigned_split"] == "selection"), key=lambda row: row["image_id"]):
        selection.append({key: source[key] for key in BASE_FIELDS + ("annotation_path", "annotation_sha256")})
    require([{"ordinal": index, "image_id": row["image_id"], "image_sha256": row["image_sha256"], "refined_component": row["refined_component"]}
        for index, row in enumerate(selection, 1)] == diagnostic["members_ordered_metadata_only"], "Selection metadata diagnostic differs")
    prepared_root = formal["prepared_root"]
    expected_paths = [str(Path(prepared_root) / row["image_path"]).replace("\\", "/") for row in selection]
    require(sources["Selection_list"].read_text(encoding="utf-8").splitlines() == expected_paths, "Selection original list/order differs")
    joint = canonical_hash([{"image_id": row["image_id"], "image_sha256": row["image_sha256"]} for row in selection])
    require(joint == SELECTION_ORDER_SHA, "Original joint Selection hash differs")
    inference = read(sources["formal_inference_manifest"])["files"]
    debug = []
    for source in sorted(formal["cohorts"]["debug"], key=lambda row: row["image_id"]):
        require(source["image_sha256"] == by_id[source["image_id"]]["image_sha256"] and
                source["annotation_sha256"] == by_id[source["image_id"]]["annotation_sha256"], "Debug source metadata differs")
        row = {key: source[key] for key in BASE_FIELDS + ("annotation_path", "annotation_sha256")}
        relative = "debug/raw/" + row["image_id"].split("/")[1] + ".npz"
        row["saved_raw_reference"] = {"path": str(FORMAL / "inference" / relative), **inference[relative]}
        debug.append(row)
    calibration = {}
    for seed in SEEDS:
        rows = [{key: by_id[image_id][key] for key in BASE_FIELDS} for image_id in plan["calibration_image_ids"][seed]]
        require([{key: row[key] for key in BASE_FIELDS if key != "prefix_proxy"} for row in rows] == full["calibration"][seed], "Existing calibration contract differs")
        expected_paths = [str(Path(prepared_root) / row["image_path"]).replace("\\", "/") for row in rows]
        require(sources["calibration_list_" + seed].read_text(encoding="utf-8").splitlines() == expected_paths, "Calibration list/order differs")
        calibration[seed] = rows
    cross_role = {}
    for role in ("train", "training_val", "debug", "calibration_pool", "quarantine"):
        cross_role[role] = metadata_overlap(selection, [row for row in assignments if row["assigned_split"] == role])
        checked_keys = ("image_id", "image_sha256") if role == "quarantine" else tuple(cross_role[role])
        require(all(cross_role[role][key] == 0 for key in checked_keys), "Selection crosses an active role or exact image identity boundary")
    result = {"schema": SCHEMA, "status": "BOUND_METADATA_ONLY_NO_SELECTION_CONTENT_READ",
        "prepared_root": prepared_root, "selection": selection, "debug": debug, "calibration": calibration,
        "member_order_hashes": {"selection": canonical_hash(selection), "debug": canonical_hash(debug),
            "calibration": {seed: canonical_hash(calibration[seed]) for seed in SEEDS}},
        "Selection_ID_imageSHA_order_sha256": joint, "metadata_bindings": {key: identity(path) for key, path in sources.items()},
        "cross_role_metadata_overlap": cross_role,
        "quarantine_component_overlap_policy": "inactive quarantine risk-component/prefix overlaps are recorded; no quarantined content access or role change",
        "duplicate_groups": duplicate_groups(selection),
        "duplicate_policy": {"retain_all_548_original_members": True,
            "same_image_SHA_requires_same_original_annotation_physical_rows": True,
            "same_content_cross_component_or_role_action": "STOP_PRESERVE_NO_AUTOMATIC_REPAIR",
            "image_or_annotation_identity_conflict_action": "STOP_PRESERVE_NO_AUTOMATIC_REPAIR",
            "content_dependence_group": "actual verified image SHA; original member IDs remain separate",
            "independent_sample_or_significance_claim": False},
        "Selection_content_files_opened_or_probed": 0, "calibration_labels_read": False,
        "Holdout_access": "SEALED", "Holdout_member_IDs_or_paths_output": False,
        "human_authorization_verified_by_helper": False,
        "access_requires": "caller source-bound execution contract and direct human decision; this metadata cannot grant access"}
    validate_metadata(result)
    return result


def verify_metadata_sources(metadata):
    require(metadata["metadata_bindings"] == {key: identity(path) for key, path in metadata_sources().items()}, "Frozen source metadata changed")


def require_debug_gate(metadata, evidence):
    require(evidence["status"] == "PASS_ALL_109_DEBUG_STRICT_FP32_BEFORE_SELECTION_CONTENT" and
            type(evidence["images"]) is int and evidence["images"] == 109 and evidence["all_strict_equal"] is True and
            type(evidence["forward_calls"]) is int and evidence["forward_calls"] == 109 and
            type(evidence["Selection_content_reads_before_gate"]) is int and evidence["Selection_content_reads_before_gate"] == 0,
            "109-debug strict reference gate not completed before Selection content")
    checks = evidence["checks"]
    require(isinstance(checks, list) and len(checks) == 109 and
            [row["image_id"] for row in checks] == [row["image_id"] for row in metadata["debug"]] and
            all(row["strict_equal"] is True for row in checks), "Debug gate identities/order/strict results differ")


def checked_member_path(metadata, row, kind):
    require(kind in ("image", "annotation"), "Unknown member content kind")
    role = row["assigned_split"]
    if role == "selection":
        collection = metadata["selection"]
    elif role == "debug":
        collection = metadata["debug"]
    elif role == "calibration_pool":
        require(kind == "image", "Calibration label access forbidden")
        collection = [item for rows in metadata["calibration"].values() for item in rows]
    else:
        raise ValueError("Forbidden content role")
    require(any(row == item for item in collection), "Member absent from fixed binding")
    validate_member(row, role)
    root = Path(metadata["prepared_root"]).resolve(strict=True)
    path = (root / row[kind + "_path"]).resolve(strict=True)
    require(path.is_relative_to(root), "Resolved member path escaped prepared root")
    require(sha(path) == row[kind + "_sha256"], "Actual member content SHA differs: " + row["image_id"] + "/" + kind)
    return path


def parse_physical_annotation(raw_bytes):
    text = raw_bytes.decode("utf-8-sig")
    output = []
    for line_index, text_line in enumerate(text.splitlines()):
        if not text_line.strip():
            output.append({"physical_line_zero_based": line_index, "empty": True})
            continue
        parts = text_line.split(",")
        if len(parts) == 9 and not parts[-1].strip():
            parts.pop()
        require(len(parts) == 8, "Original GT must have eight fields per physical row")
        values = [float(part.strip()) for part in parts]
        require(all(math.isfinite(value) for value in values), "Nonfinite original GT row")
        x, y, width, height, score, category, truncation, occlusion = values
        require(all(value.is_integer() for value in values[4:]) and score in (0, 1) and 0 <= category <= 11
                and truncation in (0, 1, 2, 3) and occlusion in (0, 1, 2), "Original GT flags/category invalid")
        require(math.isfinite(x + width) and math.isfinite(y + height) and math.isfinite(width * height), "GT geometry arithmetic overflow")
        if score == 1 and 1 <= category <= 10:
            require(width > 0 and height > 0, "Evaluated original GT geometry invalid")
        output.append({"physical_line_zero_based": line_index, "values": values})
    return output


def image_dimensions(raw_bytes):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    require(image is not None and image.ndim == 3 and image.shape[2] == 3, "Image decode failed")
    return int(image.shape[1]), int(image.shape[0])


def validate_duplicate_content(metadata, content_rows):
    by_id = {row["image_id"]: row for row in content_rows}
    require(len(by_id) == len(content_rows), "Repeated validated member")
    checked = []
    for group in metadata["duplicate_groups"]:
        rows = [by_id[image_id] for image_id in group["image_ids"]]
        if not all(row["actual_image_sha256"] == group["image_sha256"] for row in rows):
            raise ContentValidationError("Duplicate group actual image identities differ; STOP",
                {"stage": "DUPLICATE_IMAGE_IDENTITY", "group": group, "members": rows, "action": "STOP_PRESERVE_NO_AUTOMATIC_REPAIR"})
        if not all(row["annotation_physical_rows"] == rows[0]["annotation_physical_rows"] for row in rows):
            differences = []
            for other in rows[1:]:
                left, right = rows[0]["annotation_physical_rows"], other["annotation_physical_rows"]
                for position in range(max(len(left), len(right))):
                    first = left[position] if position < len(left) else None
                    second = right[position] if position < len(right) else None
                    if first != second:
                        differences.append({"left_image_id": rows[0]["image_id"], "right_image_id": other["image_id"],
                            "physical_row_position": position, "left": first, "right": second})
            raise ContentValidationError("Same-image original annotation physical-row conflict; STOP",
                {"stage": "DUPLICATE_ORIGINAL_ANNOTATION_ROWS", "group": group, "members": rows,
                 "row_differences": differences, "action": "STOP_PRESERVE_NO_AUTOMATIC_REPAIR"})
        checked.append({"image_sha256": group["image_sha256"], "image_ids": group["image_ids"],
            "component": group["component"], "member_count": len(rows), "all_original_annotation_physical_rows_equal": True,
            "original_annotation_sha256": [row["actual_annotation_sha256"] for row in rows],
            "same_content_is_one_dependence_group_not_independent_samples": True})
    return checked


def validate_content_after_debug_preflight(metadata, evidence):
    # No Selection member path may be inspected before both checks complete.
    validate_metadata(metadata)
    require_debug_gate(metadata, evidence)
    verify_metadata_sources(metadata)
    content_rows = []
    for row in metadata["selection"]:
        image_path = checked_member_path(metadata, row, "image")
        annotation_path = checked_member_path(metadata, row, "annotation")
        image_bytes, annotation_bytes = image_path.read_bytes(), annotation_path.read_bytes()
        require(hashlib.sha256(image_bytes).hexdigest() == row["image_sha256"] and
                hashlib.sha256(annotation_bytes).hexdigest() == row["annotation_sha256"], "Member changed during content read")
        require(image_dimensions(image_bytes) == (row["width"], row["height"]), "Actual image dimensions differ")
        physical = parse_physical_annotation(annotation_bytes)
        content_rows.append({"image_id": row["image_id"], "actual_image_sha256": row["image_sha256"],
            "actual_annotation_sha256": row["annotation_sha256"], "width": row["width"], "height": row["height"],
            "refined_component": row["refined_component"], "annotation_physical_rows": physical})
    duplicate_checks = validate_duplicate_content(metadata, content_rows)
    return {"status": "PASS_FIXED_548_SELECTION_CONTENT_AND_DUPLICATE_ORIGINAL_GT_IDENTITY_ONLY",
        "as_of_utc": datetime.now(timezone.utc).isoformat(), "debug_preflight_evidence_sha256": canonical_hash(evidence),
        "metadata_binding_sha256": canonical_hash(metadata), "members": content_rows, "images": len(content_rows),
        "unique_actual_image_SHA256": len({row["actual_image_sha256"] for row in content_rows}),
        "duplicate_checks": duplicate_checks, "all_548_original_members_retained": True,
        "physical_row_comparison": "UTF8 BOM and newline encoding normalized; blanks preserve positions; all 8 numeric fields and physical indices compare",
        "model_forward_or_AP_executed_by_helper": False, "calibration_labels_read": False,
        "Holdout_access": "SEALED", "scientific_acceptance": False,
        "scope_limit": "current fixed member bytes/GT identity only, not all near-duplicate or physical scene independence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare",))
    parser.parse_args()
    metadata = prepare_metadata()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "member_metadata.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": metadata["status"], "binding": identity(output),
        "Selection_content_reads": 0, "model_or_AP_calls": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
