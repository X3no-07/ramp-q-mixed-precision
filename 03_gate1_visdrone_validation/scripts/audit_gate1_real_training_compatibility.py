"""Read-only compatibility audit of explicitly bound train/training_val members.

This does not lift the Dataset/Trainer's synthetic-only restriction. It checks
existing parsers and proposed restricted geometry, never trains or scores a
model, and only proposes component quarantine without changing assignments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import unicodedata


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {v}")))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def verified_bytes(info):
    path = Path(info["path"])
    raw = path.read_bytes()
    require(len(raw) == info["bytes"] and hashlib.sha256(raw).hexdigest() == info["sha256"],
            f"File identity changed: {path}")
    return raw


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def jsonl(raw):
    return [strict_json(line.encode("utf-8")) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]


def safe_member(root, relative, kind):
    """Reject forbidden namespaces and traversal before touching member bytes."""
    require(isinstance(relative, str) and "\\" not in relative and ":" not in relative, "Invalid relative member path")
    p = PurePosixPath(relative)
    require(not p.is_absolute() and all(v not in {"", ".", ".."} for v in relative.split("/")), "Traversing member path")
    expected = {"image": "images/train/", "label": "labels/train/"}[kind]
    require(relative.startswith(expected), "Only official-train image/label members are allowed")
    normalized = unicodedata.normalize("NFKC", relative).casefold()
    require(not re.search(r"selection|holdout|test[-_]?dev", normalized), "Forbidden member namespace")
    root = Path(root).resolve(strict=True)
    candidate = root.joinpath(*p.parts)
    resolved = candidate.resolve(strict=True)
    require(resolved.is_relative_to(root), "Member escaped prepared root")
    require(not candidate.is_symlink() and not any(parent.is_symlink() for parent in candidate.parents if parent.is_relative_to(root)),
            "Symlink members are forbidden")
    return resolved


def check_membership(rows, expected_counts):
    require(len({r["image_id"] for r in rows}) == len(rows), "Duplicate assignment identity")
    allowed = {"train", "training_val", "calibration_pool", "debug", "selection", "quarantine", "duplicate_excluded"}
    require(all(r["assigned_split"] in allowed for r in rows), "Unknown split role")
    counts = dict(Counter(r["assigned_split"] for r in rows))
    require(counts == expected_counts, "Frozen split counts changed")
    active = {"train", "training_val", "calibration_pool", "debug", "selection"}
    for field in ("image_sha256", "prefix_proxy", "refined_component"):
        owners = defaultdict(set)
        for row in rows:
            if row["assigned_split"] in active:
                owners[row[field]].add(row["assigned_split"])
        require(all(len(v) == 1 for v in owners.values()), f"Cross-role overlap: {field}")
    chosen = [r for r in rows if r["assigned_split"] in {"train", "training_val"}]
    require(bool(chosen), "No permitted members")
    for row in chosen:
        require(row["source_split"] == "train" and re.fullmatch(r"train/[A-Za-z0-9_.-]+", row["image_id"]),
                "Training member is not official-train-derived")
        require(row["is_content_canonical"] is True and row["requires_annotation_quality_quarantine"] is False,
                "Noncanonical or quality-quarantined member reached active audit")
    return chosen


def geometry_check(geometry, np, identity, hw, rows, image_size):
    sample = geometry.from_prepared_rows(identity, hw, rows, dtype=np.float64)
    h, w = hw
    ratio = image_size / max(h, w)
    rh, rw = min(math.ceil(h * ratio), image_size), min(math.ceil(w * ratio), image_size)
    sample = geometry.apply_loader_resize(sample, (rh, rw))
    scale = min(image_size / rh, image_size / rw)
    unpad = (round(rw * scale), round(rh * scale))
    dw, dh = (image_size - unpad[0]) / 2, (image_size - unpad[1]) / 2
    params = {"orig_shape": (rh, rw), "new_shape": (image_size, image_size), "ratio": (scale, scale),
              "new_unpad": unpad, "left": round(dw - .1), "right": round(dw + .1),
              "top": round(dh - .1), "bottom": round(dh + .1)}
    sample = geometry.apply_letterbox_params(sample, params)
    for vertical, horizontal in ((False, False), (True, False), (False, True), (True, True)):
        variant = sample
        for direction, enabled in (("vertical", vertical), ("horizontal", horizontal)):
            variant = geometry.apply_flip_params(variant, {"flip": enabled, "direction": direction,
                "h": image_size, "w": image_size, "flip_idx": None})
        cast = variant["boxes"].astype(np.float32)
        require(bool(np.isfinite(cast).all()) and bool((cast[:, 2:] > cast[:, :2]).all()),
                "Final FP32 ignore geometry collapsed")
    return {"publisher_regions": len(sample["boxes"]), "flip_branches": 4}


def proposed_quarantine(selected, results):
    bad_ids = {r["image_id"] for r in results if r["issues"]}
    bad_components = {r["refined_component"] for r in selected if r["image_id"] in bad_ids}
    proposed = [{"image_id": r["image_id"], "previous_role": r["assigned_split"], "refined_component": r["refined_component"],
                 "trigger": "direct_incompatibility" if r["image_id"] in bad_ids else "same_existing_component"}
                for r in selected if r["refined_component"] in bad_components]
    return {"schema": "rampq.gate1.real_training_quarantine_proposal/v1", "applied": False,
            "component_count": len(bad_components), "directly_incompatible_samples": len(bad_ids), "samples": proposed}


def run(protocol_path):
    import cv2
    import numpy as np
    import torch

    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol = strict_json(protocol_path.read_bytes())
    require(protocol["schema"] == "rampq.gate1.real_training_compatibility_protocol/v1", "Wrong audit protocol")
    require(protocol["training_authorized_by_protocol"] is False, "Audit protocol cannot authorize training")
    output = Path(protocol["output_directory"]).resolve()
    require(output == protocol_path.parent / "result" and not output.exists(), "Audit result must be new beside protocol")
    output.mkdir()
    bindings = protocol["bindings"]
    bound_raw = {key: verified_bytes(info) for key, info in bindings.items()}
    require(bindings["auditor"]["sha256"] == digest(Path(__file__)), "Auditor differs from frozen protocol")
    modules = {}
    for key in ("dataset_source", "geometry_source"):
        spec = importlib.util.spec_from_file_location("compatibility_" + key, bindings[key]["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[key] = module
    dataset, geometry = modules["dataset_source"], modules["geometry_source"]
    require(dataset.GEOMETRY_SHA256 == bindings["geometry_source"]["sha256"], "Dataset geometry dependency drift")
    plan, plan_manifest = strict_json(bound_raw["split_plan"]), strict_json(bound_raw["split_plan_manifest"])
    for name, key in (("split_assignment.jsonl", "assignments"), ("split_plan.json", "split_plan")):
        require(plan_manifest["files"][name] == {k: bindings[key][k] for k in ("bytes", "sha256")}, "Plan manifest binding mismatch")
    require(plan["source_manifest_sha256"] == bindings["source_inventory"]["sha256"] and
            plan["source_file_manifest_sha256"] == bindings["source_file_manifest"]["sha256"], "Prepared metadata binding mismatch")
    root = Path(plan["source_prepared"]).resolve(strict=True)
    require(root == Path(protocol["prepared_root"]).resolve(strict=True), "Prepared root changed")
    selected = check_membership(jsonl(bound_raw["assignments"]), protocol["expected_counts"])
    source_rows = jsonl(bound_raw["source_inventory"])
    require(len({r["image_id"] for r in source_rows}) == len(source_rows), "Duplicate original source identity")
    source_rows = {r["image_id"]: r for r in source_rows}
    for row in selected:
        source = source_rows[row["image_id"]]
        require(set(row) == set(source) | {"previous_assigned_split", "refined_component"}, "Source fields changed")
        require(all(row[k] == value for k, value in source.items() if k != "assigned_split") and
                row["previous_assigned_split"] == source["assigned_split"], "Prepared row changed during split refinement")
    entries = strict_json(bound_raw["source_file_manifest"])["files"]
    require(len({r["path"] for r in entries}) == len(entries), "Duplicate source file path")
    entries = {r["path"]: r for r in entries}
    ignore_name = "ignored_regions_train.jsonl"
    require(Path(bindings["shared_train_ignores"]["path"]).resolve() == root / ignore_name and
            all(entries[ignore_name][k] == bindings["shared_train_ignores"][k] for k in ("bytes", "sha256")), "Train sidecar binding mismatch")
    selected_ids = {r["image_id"] for r in selected}
    ignores = defaultdict(list)
    shared_lines = 0
    for line in bound_raw["shared_train_ignores"].decode("utf-8-sig").splitlines():
        require(bool(line.strip()), "Blank shared sidecar row")
        entry = strict_json(line.encode("utf-8"))
        shared_lines += 1
        require(entry["image_id"].startswith("train/"), "Nontrain entry in official-train sidecar")
        if entry["image_id"] in selected_ids:
            ignores[entry["image_id"]].append(entry)
    results, opened = [], []
    for index, row in enumerate(selected):
        raw_files = {}
        for kind, field, hash_field in (("image", "image_path", "image_sha256"), ("label", "label_path", "label_sha256")):
            relative = row[field]
            path = safe_member(root, relative, kind)
            info = entries[relative]
            require(info["sha256"] == row[hash_field], "Row/file manifest hash mismatch")
            identity = {**info, "path": str(path), "image_id": row["image_id"], "role": row["assigned_split"]}
            raw_files[kind] = verified_bytes(identity)
            opened.append(identity)
        decoded = cv2.imdecode(np.frombuffer(raw_files["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        require(decoded is not None and decoded.dtype == np.uint8 and decoded.ndim == 3 and decoded.shape[2] == 3 and
                tuple(decoded.shape[:2]) == (row["height"], row["width"]), f"Image decode/dimensions mismatch: {row['image_id']}")
        issues = []
        positive_count = None
        try:
            positive_count = len(dataset._positive_labels(raw_files["label"], 10))
            require(positive_count == row["valid_target_count"], "Positive-label count disagrees with source metadata")
        except ValueError as error:
            issues.append({"stage": "positive_label_parser", "reason": str(error)})
        sidecar_rows = ignores[row["image_id"]]
        require(len(sidecar_rows) == row["ignored_target_count"], "Sidecar member count disagrees with source metadata")
        try:
            geometry_result = geometry_check(geometry, np,
                {"image_id": row["image_id"], "image_sha256": row["image_sha256"], "sidecar_sha256": bindings["shared_train_ignores"]["sha256"]},
                (row["height"], row["width"]), sidecar_rows, protocol["image_size"])
        except ValueError as error:
            issues.append({"stage": "ignore_geometry", "reason": str(error)})
            geometry_result = None
        results.append({"image_id": row["image_id"], "role": row["assigned_split"], "refined_component": row["refined_component"],
                        "positive_count": positive_count, "sidecar_rows": len(sidecar_rows), "geometry": geometry_result,
                        "issues": issues, "compatible_with_checked_primitives": not issues})
        if (index + 1) % 500 == 0:
            print(json.dumps({"checked": index + 1, "total": len(selected)}, ensure_ascii=True), flush=True)
    for info in bindings.values():
        require(file_info(info["path"]) == info, "Bound input changed during audit")
    proposal = proposed_quarantine(selected, results)
    role_summary = {}
    for role in ("train", "training_val"):
        subset = [r for r in results if r["role"] == role]
        quarantined = sum(r["previous_role"] == role for r in proposal["samples"])
        role_summary[role] = {"checked": len(subset), "compatible": sum(not r["issues"] for r in subset),
                              "incompatible": sum(bool(r["issues"]) for r in subset), "proposed_component_quarantine": quarantined,
                              "would_remain_after_proposal": len(subset) - quarantined,
                              "issue_counts": dict(Counter(issue["reason"] for r in subset for issue in r["issues"]))}
    summary = {"schema": "rampq.gate1.real_training_compatibility_audit/v1", "status": "COMPLETED_READ_ONLY_COMPATIBILITY_AUDIT_ONLY",
               "as_of_utc": datetime.now(timezone.utc).isoformat(), "protocol": file_info(protocol_path),
               "roles": role_summary, "image_and_label_files_verified": len(opened),
               "image_and_label_bytes_verified": sum(i["bytes"] for i in opened), "shared_official_train_sidecar_rows_parsed": shared_lines,
               "selected_ignore_rows": sum(len(v) for v in ignores.values()), "proposed_quarantine_components": proposal["component_count"],
               "all_bound_inputs_unchanged": True,
               "runtime": {"python": sys.executable, "python_version": sys.version, "numpy": np.__version__, "opencv": cv2.__version__,
                           "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
               "dataset_training_scope_still": "synthetic_only", "cuda_trainer_and_capture_qualified": False,
               "geometry_scope": "declared_source_rows_resize_letterbox_four_flip_branches_FP32_cast_only_not_actual_Dataset_execution",
               "shared_sidecar_note": "Full official-train shared sidecar parsed; only explicit train/training_val members semantically checked.",
               "training_started": False, "model_inference_executed": False, "task_AP_computed": False,
               "selection_pixels_labels_or_val_sidecars_opened": False, "holdout_opened_or_listed": False,
               "source_data_or_assignments_modified": False, "quarantine_proposal_applied": False, "gate1_scientific_pass": False,
               "old_gate1_dtype_guard": "FAIL_FIXED_TOLERANCES", "old_gate05_numeric_guard": "FAILED_UNCHANGED"}
    for name, values in (("per_sample.jsonl", results), ("opened_image_label_files.jsonl", opened)):
        with (output / name).open("w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "quarantine_proposal.json", proposal)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {"files": {name: file_info(output / name) for name in
        ("per_sample.jsonl", "opened_image_label_files.jsonl", "quarantine_proposal.json", "summary.json")}})
    print(json.dumps({"status": summary["status"], "roles": role_summary, "output": str(output)}, ensure_ascii=True), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    run(parser.parse_args().protocol)
