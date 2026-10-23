"""Seal completed frontier diagnosis and the separate direct-image prototype.

Run only after both final metadata reviews are complete. Hash immutable files
once per resolved path, including debug array/image/label bytes without parsing
them. No model, arrays, pixels, annotations, or Holdout archives are decoded.
Only delivery_validation.json and delivery_manifest.json are created, in that
order. Existing evidence and the mutable latest-state pointer are never edited.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "artifacts/gate1"
CHECK_NAME = "GATE1_BODY_FP32_HEAD_TOWERS_FP16_DECODE_FP32_DEBUG_NUMERIC_CHECK"
PINNED_SMALL = {
    "46_Gate1_上游边界精度对照与缓解路径.md": "d9eec5f9be1d9ba65f6c54daaf52f73d43544781abe34d025235bec086011bd3",
    "47_Gate1_混合精度独立推理原型验证.md": "be840401edd050183ef212ad774dd8800197a37fc26e38d8a1a68c39f9fa27e2",
    "artifacts/gate1/current_execution_state_v7.json": "2fbe8bb1b734a07a18133069eb3d88c75af031b6622626253e89a3c14c057e14",
    "artifacts/gate1/current_execution_state_v8.json": "602106167ccf280fcb26705081bd6911ebb78accf6da9f93c3e2817c9ca228f3",
    "artifacts/gate1/20261023_upstream_diagnosis_delivery/decision.json": "c3d18e1e531a9dcff4091aa8235cc75bbe2055595dd9720821f9869cf82be54f",
    "artifacts/gate1/20261023_upstream_diagnosis_delivery/prototype_decision.json": "5ab46c53f60fcfa9ba804b6ba9c3907e47f95ef20bf73b60d918cb55f3e158e7",
}
# (manifest path relative to ROOT, base mode, expected member count, pinned SHA)
NESTED = [
    ("artifacts/gate1/20261023_precision_diagnosis_delivery/delivery_manifest.json", "root", 112, "a7e364da78b4ceacb94dc06e262eeaebd5c2b17e5f8bb1306a809bed28cf0f53"),
    ("artifacts/gate1/20261022_training_qualification_delivery/delivery_manifest.json", "root", 75, "63a5b9f22cef694b193ae2008f32350bd943a6ba74358bf1c5334e46ecdfbcf1"),
    ("artifacts/gate05/20261021_final_acceptance/final_manifest.json", "root", 45, "f40498d62d8eae25b5bf1330437be486306311f77ad65ae2ef5cc1e7efbcc7e1"),
    ("artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification/manifest.json", "local", 454, "77b096b122dc41c5e0ae879d2c50bd10efb52bf021679fae2ed89999ff4368e6"),
    ("artifacts/gate1/20261023_head_precision_full_v1/manifest.json", "local", 228, "5394dbde5881bf2936ac2cc7de049ed717f6d4d8554e60bac761ceb024d15408"),
    ("artifacts/gate1/20261023_upstream_frontier_pilot_v1/manifest.json", "local", 15, "7f36a018a2ed9c2a064a04f5a52cb10e2da01fa878f917f7f36c8ea5a48bb1a6"),
    ("artifacts/gate1/20261023_upstream_frontier_full_v1/manifest.json", "local", 227, "a5e3da8beb8f232267aa924bb1dca02abbcf9fcfb030a6e0a908ed537ad6c1e6"),
    ("artifacts/gate1/20261023_upstream_frontier_pilot_audit_v1/manifest.json", "local", 3, "058f8ccef005a640d16069adf4508d5bb0ea38b636c2d8712708a376d1ae2fe7"),
    ("artifacts/gate1/20261023_upstream_frontier_full_audit_v1/manifest.json", "local", 3, "cc1a0b5393086f6df089303862296a99c136f9cf6298ab90d0c63cf94c8de298"),
    ("artifacts/gate1/20261023_body32_head16_pilot_v1/manifest.json", "local", 15, "55e37e776b03bcd12bbdf7319f5a896b90e27c9c801b0507f7b17de714acda78"),
    ("artifacts/gate1/20261023_body32_head16_full_v1/manifest.json", "local", 227, "b09ed6611e003bbfaabb7b78b71aeda279df61b89973136f940eb4febc54b8aa"),
    ("artifacts/gate1/20261023_body32_head16_pilot_audit_v1/manifest.json", "local", 3, "1762111cf59af09d223762080313a41d2f079c27601d8de0400d43b2cb31bb6a"),
    ("artifacts/gate1/20261023_body32_head16_full_audit_v1/manifest.json", "local", 3, "537d150d3880b68be3d8d3f73bd4fb01f09a3a62ab39455d2500929ad8da3939"),
    ("artifacts/gate1/20261023_upstream_frontier_preparation/independent_review/manifest.json", "local", 8, "9fdadc8b670230533b9abd41e3a57395496deff1ee2b0eb5554a69fb9b876286"),
    ("artifacts/gate1/20261023_body32_head16_prototype_preparation/independent_review/manifest.json", "local", 8, "bbc097d694e493f38cff69bf466710e64531acda68f2275215fd2888151d5971"),
    ("artifacts/gate1/20261023_official_oracle_followup/manifest.json", "local", 4, "2aba5aa78dc431b482e163be23b4849343fe43549c6b01a1c9eb13feac44165a"),
]
CACHE = {}
CACHE_HITS = 0
BYTES_HASHED = 0


def require(condition, message):
    if not condition: raise ValueError(message)


def opaque_holdout_name(relative):
    return any("holdout" in part.casefold() or "test-dev" in part.casefold() or "test_dev" in part.casefold() for part in PurePosixPath(relative.replace("\\", "/")).parts)


def safe(base, relative):
    rel = PurePosixPath(str(relative).replace("\\", "/"))
    require(not rel.is_absolute() and bool(rel.parts) and ".." not in rel.parts and not any(":" in p for p in rel.parts), "Unsafe manifest path")
    require(not opaque_holdout_name(rel.as_posix()), "Holdout/test-dev paths must remain opaque")
    candidate = base
    for part in rel.parts:
        candidate /= part
        require(not candidate.is_symlink() and not candidate.is_junction(), "Reparse links are forbidden")
    candidate = candidate.resolve(strict=True)
    require(candidate.is_relative_to(ROOT) and candidate.is_file(), "Path escapes project or is not a regular file")
    return candidate


def signature(path):
    value = path.stat()
    require(stat.S_ISREG(value.st_mode), "Not a regular file")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def identity(path):
    global CACHE_HITS, BYTES_HASHED
    path = safe(ROOT, path.relative_to(ROOT).as_posix())
    key = str(path).casefold(); before = signature(path)
    if key in CACHE:
        CACHE_HITS += 1
        require(CACHE[key][0] == before, "Previously hashed evidence changed: " + str(path))
        return dict(CACHE[key][1])
    with path.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(before == signature(path), "Evidence changed during hashing: " + str(path))
    result = {"bytes": before[2], "sha256": digest}
    CACHE[key] = (before, result); BYTES_HASHED += before[2]
    return dict(result)


def read_json(path):
    # Only manifest/receipt/protocol/decision/review/state metadata uses JSON parsing.
    identity(path)
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_members(document):
    files = document.get("files")
    if isinstance(files, dict): rows = [(name, value) for name, value in files.items()]
    elif isinstance(files, list): rows = [(entry["path"], {"bytes": entry["bytes"], "sha256": entry["sha256"]}) for entry in files]
    else: raise ValueError("Unsupported nested manifest membership format")
    require(rows and len({str(name).replace("\\", "/").casefold() for name, _ in rows}) == len(rows), "Empty or duplicate manifest members")
    for name, entry in rows:
        require(type(entry.get("bytes")) is int and entry["bytes"] >= 0 and isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64, "Invalid nested member identity")
    return rows


def excluded_dir(name):
    lower = name.casefold()
    return lower == "__pycache__" or ("mpl" in lower and ("cache" in lower or "config" in lower))


def local_files(directory):
    result = set()
    for current, dirs, names in os.walk(directory, followlinks=False):
        for name in list(dirs):
            child = Path(current) / name
            require(not child.is_symlink() and not child.is_junction(), "Linked evidence directory")
            if excluded_dir(name): dirs.remove(name)
            elif opaque_holdout_name(name):
                # No directory listing, file open or traversal beneath this child.
                raise ValueError("Unexpected Holdout/test-dev directory in an explicit metadata/output scope")
        for name in names:
            relative = (Path(current) / name).relative_to(directory).as_posix()
            if not opaque_holdout_name(relative): result.add(relative)
    return result


def verify_nested(relative, mode, expected_count, expected_sha):
    path = safe(ROOT, relative); manifest_identity = identity(path)
    require(manifest_identity["sha256"] == expected_sha, "Frozen nested manifest changed: " + relative)
    document = read_json(path); rows = normalize_members(document)
    require(len(rows) == expected_count, "Nested member count changed: " + relative)
    base = ROOT if mode == "root" else path.parent
    skipped, verified = [], 0
    for name, expected in rows:
        if opaque_holdout_name(name):
            # Preserve the already frozen opaque declaration; do not stat/open/list it.
            skipped.append(str(name)); continue
        member = safe(base, name)
        require(identity(member) == {"bytes": expected["bytes"], "sha256": expected["sha256"]}, "Nested member changed: " + str(name))
        verified += 1
    if mode == "local":
        require(local_files(base) == {str(name).replace("\\", "/") for name, _ in rows if not opaque_holdout_name(name)} | {path.name}, "Nested manifest membership changed: " + relative)
    return {"manifest": relative, "base": "." if mode == "root" else base.relative_to(ROOT).as_posix(), "membership_format": type(document["files"]).__name__,
            "declared_count": len(rows), "verified_count": verified, "opaque_holdout_entries_not_opened": skipped, **manifest_identity}


def no_elevation(document):
    for name in ("scientific_gate_pass", "gate1_scientific_pass", "accepted_float_baseline_ready", "selection_detector_inference", "holdout_accessed", "holdout_opened_or_listed", "new_precision_model_deployed", "new_precision_model_accepted_or_deployed", "official_oracle_verified"):
        if name in document: require(document[name] is False, "Evidence scope was elevated: " + name)


def check_execution(probe_name, audit_name, expected_images, probe_status, audit_status):
    probe, audit = BASE / probe_name, BASE / audit_name
    receipt, summary = read_json(probe / "receipt.json"), read_json(audit / "recomputed_summary.json")
    require(receipt.get("status") == probe_status and receipt.get("phase") == "COMPLETE" and receipt.get("images_completed") == expected_images, "Incomplete probe: " + probe_name)
    require(summary.get("status") == audit_status and summary.get("images_verified") == expected_images, "Incomplete audit: " + audit_name)
    for document in (receipt, summary): no_elevation(document)
    require(summary["protocol_sha256"] == receipt["protocol_sha256"] == identity(probe / "protocol.json")["sha256"], "Probe/audit protocol identity differs")
    require(summary["probe_manifest"]["sha256"] == identity(probe / "manifest.json")["sha256"], "Probe/audit manifest differs")
    return {"probe": probe_name, "audit": audit_name, "images": expected_images, "probe_status": probe_status, "audit_status": audit_status}


def main():
    validation_path, manifest_path = HERE / "delivery_validation.json", HERE / "delivery_manifest.json"
    require(not validation_path.exists() and not manifest_path.exists(), "Delivery already exists; preserve it")
    for relative, expected in PINNED_SMALL.items(): require(identity(safe(ROOT, relative))["sha256"] == expected, "Completed report/state/decision changed: " + relative)
    for name in ("macro_review.json", "prototype_review.json"):
        review = read_json(BASE / "20261023_upstream_final_review" / name)
        require(review.get("status", "").startswith("PASS") and review.get("issues", []) == [], "Final review incomplete: " + name)
        no_elevation(review)
    for relative in ("20261023_upstream_frontier_preparation/independent_review/review.json", "20261023_body32_head16_prototype_preparation/independent_review/review.json"):
        review = read_json(BASE / relative); require(review.get("status", "").startswith("PASS") and review.get("issues", []) == [], "Source review incomplete")
        for source, expected in review["source_identities"].items():
            path = safe(ROOT, Path(source).relative_to(ROOT).as_posix())
            require(identity(path) == {"bytes": expected["bytes"], "sha256": expected["sha256"]}, "Reviewed live source changed")
    for relative in PINNED_SMALL:
        if relative.endswith(".json"): no_elevation(read_json(safe(ROOT, relative)))
    runs = [check_execution("20261023_upstream_frontier_" + stage + "_v1", "20261023_upstream_frontier_" + stage + "_audit_v1", count,
                            "COMPLETE_UPSTREAM_DIAGNOSTIC_ONLY", "PASS_SAVED_UPSTREAM_FRONTIER_ARRAYS_ONLY") for stage, count in (("pilot", 3), ("full", 109))]
    runs += [check_execution("20261023_body32_head16_" + stage + "_v1", "20261023_body32_head16_" + stage + "_audit_v1", count,
                             "COMPLETE_BODY32_HEAD16_DIAGNOSTIC_ONLY", "PASS_SAVED_BODY32_HEAD16_OUTPUTS_ONLY") for stage, count in (("pilot", 3), ("full", 109))]
    full = read_json(BASE / "20261023_body32_head16_full_audit_v1/recomputed_summary.json")
    require(full["guard_name"] == CHECK_NAME and full["comparison"]["pass_images"] == 109 and full["comparison"]["fail_images"] == 0
            and full["comparison"]["boxes"]["violations"] == full["comparison"]["scores"]["violations"] == 0, "Prototype numeric result differs")
    old = full["historical_full_model_comparison_unchanged"]
    require(old["fail_images"] == 109 and old["boxes"]["violations"] == 10849 and old["scores"]["violations"] == 0, "Historical failed check was changed")
    nested = [verify_nested(*descriptor) for descriptor in NESTED]
    top = {safe(ROOT, relative) for relative in PINNED_SMALL}
    top.add(Path(__file__).resolve())
    for relative, mode, _, _ in NESTED:
        path = safe(ROOT, relative); top.add(path)
        if mode == "local" and path.parent.name != "02_float_qualification" and path.parent.name != "20261023_head_precision_full_v1":
            top.update(safe(path.parent, name) for name in local_files(path.parent) if "/" not in name and not name.endswith(".npz"))
    # Explicit metadata-only roots: never recursively scan ROOT or any data scope.
    for relative in ("20261023_upstream_probe_preparation", "20261023_upstream_frontier_preparation", "20261023_body32_head16_prototype_preparation",
                     "20261023_upstream_final_review", "20261023_upstream_diagnosis_delivery", "20261023_official_oracle_followup"):
        directory = BASE / relative
        for name in local_files(directory):
            if name in ("delivery_validation.json", "delivery_manifest.json") or name.endswith(".npz"): continue
            top.add(safe(directory, name))
    for relative in ("scripts/probe_gate1_upstream_frontier.py", "scripts/audit_gate1_upstream_frontier.py", "scripts/probe_gate1_body32_head16.py", "scripts/audit_gate1_body32_head16.py",
                     "tests/test_gate1_upstream_frontier.py", "tests/test_gate1_upstream_frontier_audit.py", "tests/test_gate1_body32_head16.py", "tests/test_gate1_body32_head16_audit.py"):
        top.add(safe(ROOT, relative))
    files = {path.relative_to(ROOT).as_posix(): identity(path) for path in sorted(top)}
    require("artifacts/gate1/latest_execution_state.json" not in files, "Mutable state pointer must not be sealed")
    # Recheck cached stat identities immediately before the only new writes.
    for key, (expected_signature, _) in CACHE.items(): require(signature(Path(key)) == expected_signature, "Evidence changed before sealing")
    validation = {"schema": "rampq.gate1.upstream_and_direct_prototype_delivery_validation/v1", "status": "PASS_COMPLETED_ENGINEERING_EVIDENCE_BINDINGS_ONLY",
                  "client_date": "2026-10-23", "at_utc": datetime.now(timezone.utc).isoformat(), "executions": runs, "nested_manifests": nested,
                  "immutable_small_files_bound": len(files), "unique_paths_hashed": len(CACHE), "total_file_bytes_hashed_once": BYTES_HASHED,
                  "duplicate_path_hash_requests_served_from_cache": CACHE_HITS, "hashing_policy": "Opaque file-byte SHA256 only; no NumPy/model/image/annotation deserialization. Cached resolved path identity is reused only if file stat identity stays unchanged.",
                  "large_arrays_bound_transitively": True, "mutable_latest_state_excluded": True, "temporary_mpl_caches_excluded": True,
                  "holdout_opened_or_listed": False, "selection_detector_inference": False, "gate1_scientific_pass": False,
                  "accepted_float_baseline_ready": False, "old_gate05_guard": "FAILED_UNCHANGED", "original_gate1_dtype_guard": "FAIL_FIXED_TOLERANCES",
                  "prototype_guard_name": CHECK_NAME, "prototype_debug_numeric_pass_images": 109,
                  "limits": ["Delivery integrity and completed engineering checks only; no new numerical computation, official AP, accuracy, deployment, performance or energy proof.", "Debug images and labels are hashed as bytes only, with Holdout/test-dev archive paths left opaque if declared in historical manifests."]}
    with validation_path.open("x", encoding="utf-8") as stream: stream.write(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    files[validation_path.relative_to(ROOT).as_posix()] = identity(validation_path)
    manifest = {"schema": "rampq.gate1.upstream_and_direct_prototype_delivery_manifest/v1", "status": "COMPLETED_BOUNDED_DIAGNOSIS_AND_DIRECT_DEBUG_PROTOTYPE_ONLY",
                "at_utc": datetime.now(timezone.utc).isoformat(), "client_date": "2026-10-23", "file_base": "engineering_project_root",
                "files": dict(sorted(files.items())), "nested_manifests_verified": nested, "gate1_scientific_pass": False,
                "accepted_float_baseline_ready": False, "selection_detector_inference": False, "holdout_opened_or_listed": False,
                "original_gate1_dtype_guard": "FAIL_FIXED_TOLERANCES", "old_gate05_guard": "FAILED_UNCHANGED"}
    # Final write: nothing is added to or modified within the sealed delivery afterward.
    with manifest_path.open("x", encoding="utf-8") as stream: stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": manifest["status"], "files": len(files), "nested_manifests_verified": len(nested),
                      "unique_paths_hashed": len(CACHE), "manifest_sha256": identity(manifest_path)["sha256"]}, ensure_ascii=False))


if __name__ == "__main__": main()
