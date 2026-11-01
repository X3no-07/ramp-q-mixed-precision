"""Bounded ZIP-container/central-directory inventory after explicit new access.

prepare reads named metadata/source files only. No current archive probe is
performed. Authorized worker never calls ZIP member open/read/extract/testzip,
decodes an image, parses a label, or invokes a model, NMS or AP.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gate1/20261025_holdout_inventory_preparation_v1"
READINESS = ROOT / "artifacts/gate1/20261025_independent_acceptance_readiness_v1"
SEAL_PATH = ROOT / "artifacts/gate1/20261022_preparation/sealed_holdout/SEAL.json"
EXPECTED_ARCHIVE_BYTES = 311_251_787
EXPECTED_ARCHIVE_SHA = "b28a94b06dfd9e36ce77ff8155fb82b9d2c030f198a76105933bd56c6ea6a68d"


class InventoryError(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def require(value, message, evidence=None):
    if not value:
        raise InventoryError(message, evidence)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def metadata_sources():
    return {"SEAL": SEAL_PATH, "D_probe": READINESS / "D_output_probe_observation.json",
        "readiness_entry_spec": READINESS / "inventory_only_entry_spec.json", "script": Path(__file__),
        "tests_source": ROOT / "tests/test_gate1_holdout_inventory_v2.py", "tests_receipt": OUT / "tests_receipt.json",
        "assignment_metadata": ROOT / "artifacts/gate1/20261022_split_refinement/split_plan_v1/split_assignment.jsonl"}


def budget():
    return {"attempts": 1, "timeout_seconds": 300, "automatic_retry": False, "maximum_entries": 100000,
        "maximum_inventory_metadata_bytes": 64 * 1024 * 1024, "maximum_total_new_write_bytes": 100_000_000,
        "minimum_output_free_bytes": 100_000_000, "maximum_reported_uncompressed_total_bytes": 64_000_000_000,
        "maximum_reported_entry_expansion_ratio": 1000, "member_payload_reads": 0,
        "extracted_files": 0, "model_forwards": 0, "AP_calls": 0}


def exact_scope():
    return {"role": "SEALED_HOLDOUT_CONTAINER_AND_CENTRAL_DIRECTORY_METADATA_ONLY",
        "outer_archive_byte_count_and_SHA256": True, "ZIP_central_directory_listing": True,
        "ordered_names_sizes_CRC_flags_extensions_and_filename_pairing": True,
        "ID_prefix_overlap_with_existing_saved_metadata_only": True,
        "ZIP_member_payload_open_read_test_or_extract": False, "image_decode_or_view": False,
        "annotation_parse_or_label_statistics": False, "model_NMS_calibration_or_AP": False,
        "candidate_or_configuration_change": False, "scientific_acceptance": False,
        "old_files_moved_deleted_or_modified": False}


def check_metadata_sources(prepared):
    require(prepared["source_bindings"] == {key: fingerprint(path) for key, path in metadata_sources().items()}, "Named metadata/source identity changed")


def build_prepared(seal, probe, bindings):
    # Deliberately do not touch or return seal['archive']['path'] here.
    require(seal["status"] == "SEALED_OPAQUE_ARCHIVE_NOT_EXTRACTED" and
        seal["archive"]["bytes"] == EXPECTED_ARCHIVE_BYTES and seal["archive"]["sha256"] == EXPECTED_ARCHIVE_SHA,
        "Unexpected historical sealed container identity")
    require(probe["status"] == "PASS_SINGLE_NEW_D_DIRECTORY_GENERATED_1MiB_WRITE_READ_ONLY", "New output directory probe missing")
    root = Path(probe["new_absolute_directory"])
    require(root.is_absolute() and root.drive.upper() == "D:", "Expected explicit probed D directory")
    return {"schema": "rampq.gate1.Holdout_inventory_prepared_contract/v1",
        "status": "PREPARED_REVIEWABLE_NOT_AUTHORIZED_NOT_EXECUTED", "created_at_utc": now(),
        "source_bindings": bindings, "scope": exact_scope(), "budget": budget(),
        "expected_container": {"bytes": EXPECTED_ARCHIVE_BYTES, "sha256": EXPECTED_ARCHIVE_SHA,
            "publisher_digest": None, "scope": "historical local outer-container fingerprint only"},
        "archive_path_policy": "exact value from bound SEAL after explicit authorization; no scans or alternate archive",
        "output_directory": str(root / "Holdout_inventory_attempt_01"),
        "output_directory_policy": "new exclusive child; if existing STOP, no overwrite or automatic new attempt",
        "current_Holdout_member_count": None, "current_Holdout_inventory": "UNKNOWN_NOT_ZERO",
        "archive_body_currently_probed": False, "human_access_authorized": False,
        "future_content_or_scientific_execution_included": False,
        "payload_SHA_label_quality_content_scene_leakage": "UNKNOWN; CRC32 and names cannot establish this"}


def prepare():
    require(not (OUT / "prepared_contract.json").exists(), "New preparation already exists")
    sources = metadata_sources()
    seal, probe, tests = read(sources["SEAL"]), read(sources["D_probe"]), read(sources["tests_receipt"])
    require(tests["failures"] == tests["errors"] == 0 and tests["tests"] >= 5 and
        tests["runner"] == fingerprint(__file__) and tests["test_source"] == fingerprint(sources["tests_source"]), "Current pure ZIP guards/tests not bound")
    contract = build_prepared(seal, probe, {key: fingerprint(path) for key, path in sources.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "prepared_contract.json").open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2, allow_nan=False)
    template = {"schema": "rampq.gate1.Holdout_inventory_human_access_decision/v1", "decision": "PENDING_HUMAN_DECISION",
        "actual_access_authorized": False, "human_message_verbatim": None, "conversation_basis": None,
        "scope": exact_scope(), "prepared_contract": fingerprint(OUT / "prepared_contract.json"),
        "execution_requires_source_review_and_new_frozen_authorized_contract": True,
        "further_image_label_content_or_inference_access": False}
    with (OUT / "human_authorization_template.json").open("x", encoding="utf-8") as stream:
        json.dump(template, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return {"status": contract["status"], "prepared_contract": fingerprint(OUT / "prepared_contract.json"),
        "archive_stat_open_listing_or_payload_read": 0, "model_or_AP_calls": 0}


def validate_human_decision(receipt, prepared_binding):
    require(receipt["schema"] == "rampq.gate1.Holdout_inventory_human_access_decision/v1" and
        receipt["decision"] == "AUTHORIZED_CONTAINER_AND_CENTRAL_DIRECTORY_METADATA_ONLY" and
        receipt["actual_access_authorized"] is True and receipt["scope"] == exact_scope(), "No exact inventory-only human decision")
    require(isinstance(receipt["human_message_verbatim"], str) and bool(receipt["human_message_verbatim"].strip()) and
        isinstance(receipt["conversation_basis"], str) and bool(receipt["conversation_basis"].strip()), "Human decision/context must be recorded externally; not inferred by this tool")
    require(receipt["prepared_contract"] == prepared_binding and receipt["execution_requires_source_review_and_new_frozen_authorized_contract"] is True
        and receipt["further_image_label_content_or_inference_access"] is False, "Inventory authorization cannot escalate scope")


def freeze(reviewed_prepared_sha, authorization_sha):
    prepared_path = OUT / "prepared_contract.json"
    require(fingerprint(prepared_path)["sha256"] == reviewed_prepared_sha, "Explicit reviewed prepared SHA required")
    prepared = read(prepared_path); check_metadata_sources(prepared)
    authorization = OUT / "human_authorization.json"
    require(fingerprint(authorization)["sha256"] == authorization_sha, "Explicit externally recorded human decision SHA required")
    validate_human_decision(read(authorization), fingerprint(prepared_path))
    frozen = {"schema": "rampq.gate1.Holdout_inventory_authorized_contract/v1", "status": "FROZEN_INVENTORY_ONLY_AUTHORIZED_NOT_EXECUTED",
        "frozen_at_utc": now(), "prepared": fingerprint(prepared_path), "authorization": fingerprint(authorization),
        "scope": exact_scope(), "budget": budget(), "output_directory": prepared["output_directory"]}
    with (OUT / "authorized_contract.json").open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return {"status": frozen["status"], "authorized_contract": fingerprint(OUT / "authorized_contract.json")}


def safe_member_name(info):
    raw = info.orig_filename
    require(raw == info.filename and isinstance(raw, str) and raw, "Altered/NUL/empty central-directory name", {"name": raw})
    require(not raw.startswith(("/", "\\")) and "\\" not in raw and ":" not in raw,
        "Absolute/drive/ADS/backslash name prohibited", {"name": raw})
    name = raw[:-1] if info.is_dir() else raw
    parts = name.split("/")
    require(all(part not in ("", ".", "..") for part in parts) and
        all(not part.endswith((" ", ".")) and not any(ord(ch) < 32 or ch in '<>"|?*' for ch in part) for part in parts),
        "Unsafe path component", {"name": raw})
    require(not any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part) for part in parts), "Reserved device path component", {"name": raw})
    mode = info.external_attr >> 16
    require(not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR), "Symlink/special entry prohibited", {"name": raw})
    require(not (info.flag_bits & (1 | 64 | 8192)), "Encrypted central-directory entry prohibited", {"name": raw})
    return unicodedata.normalize("NFC", name)


def classify_member(name, is_directory):
    if is_directory:
        return "directory", "", None
    path = PurePosixPath(name)
    extension = path.suffix.casefold()
    parent = path.parent.name.casefold()
    if parent == "images" and extension in (".jpg", ".jpeg", ".png", ".bmp"):
        return "image", extension, path.stem
    if parent == "annotations" and extension == ".txt":
        return "annotation", extension, path.stem
    return "other", extension, None


def central_inventory(stream, container_bytes, limits=None):
    """Works on an outer container handle/BytesIO; never opens member payloads."""
    limits = budget() if limits is None else limits
    records, paths, offsets, logical = [], set(), set(), {"image": {}, "annotation": {}}
    total_bytes = 0
    with zipfile.ZipFile(stream, mode="r") as archive:
        entries = archive.infolist()
        require(len(entries) <= limits["maximum_entries"], "Central-directory entry budget exceeded")
        for entry in entries:
            name = safe_member_name(entry)
            key = name.casefold()
            require(key not in paths, "Duplicate/casefold central-directory path", {"name": entry.orig_filename})
            paths.add(key)
            require(entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "Unsupported compressed metadata", {"name": entry.orig_filename})
            require(type(entry.file_size) is int and type(entry.compress_size) is int and
                0 <= entry.file_size <= limits["maximum_reported_uncompressed_total_bytes"] and 0 <= entry.compress_size <= container_bytes,
                "Invalid declared compressed/uncompressed sizes", {"name": entry.orig_filename})
            require(entry.file_size == 0 or entry.compress_size > 0, "Nonempty payload declared zero compressed size", {"name": entry.orig_filename})
            require(entry.file_size <= max(1, entry.compress_size) * limits["maximum_reported_entry_expansion_ratio"], "Unsafe declared expansion ratio", {"name": entry.orig_filename})
            require(0 <= entry.CRC <= 0xffffffff and 0 <= entry.header_offset < container_bytes and entry.header_offset not in offsets,
                "Invalid CRC/header offset metadata", {"name": entry.orig_filename})
            offsets.add(entry.header_offset)
            total_bytes += entry.file_size
            require(total_bytes <= limits["maximum_reported_uncompressed_total_bytes"], "Declared aggregate uncompressed budget exceeded")
            role, extension, logical_id = classify_member(name, entry.is_dir())
            if logical_id is not None:
                require(logical_id.casefold() not in logical[role], "Duplicate logical image/annotation ID", {"role": role, "logical_id": logical_id})
                logical[role][logical_id.casefold()] = logical_id
            records.append({"archive_name": entry.orig_filename, "normalized_name": name, "is_directory": entry.is_dir(),
                "role": role, "logical_id": logical_id, "extension": extension, "declared_bytes": entry.file_size,
                "declared_compressed_bytes": entry.compress_size, "CRC32_hex": f"{entry.CRC:08x}",
                "compression_type": entry.compress_type, "flag_bits": entry.flag_bits, "external_attr": entry.external_attr,
                "local_header_offset": entry.header_offset, "member_SHA256": None, "payload_opened": False})
    records.sort(key=lambda row: (row["normalized_name"], row["archive_name"]))
    encoded = canonical(records)
    require(len(encoded) <= limits["maximum_inventory_metadata_bytes"], "Inventory metadata byte budget exceeded")
    missing_labels = sorted(set(logical["image"]) - set(logical["annotation"]))
    missing_images = sorted(set(logical["annotation"]) - set(logical["image"]))
    require(not missing_labels and not missing_images, "Image/annotation filename pairing conflict",
        {"images_without_annotation": missing_labels, "annotations_without_image": missing_images})
    return {"records": records, "record_count": len(records), "ordered_inventory_SHA256": hashlib.sha256(encoded).hexdigest(),
        "counts_by_role": dict(Counter(row["role"] for row in records)), "counts_by_extension": dict(Counter(row["extension"] for row in records if not row["is_directory"])),
        "filename_paired_image_annotation_count": len(logical["image"]), "reported_total_uncompressed_bytes": total_bytes,
        "reported_total_compressed_bytes": sum(row["declared_compressed_bytes"] for row in records),
        "member_payload_reads": 0, "member_SHA256_available": False, "payload_integrity_or_label_quality_verified": False,
        "actual_scene_or_content_leakage_or_scientific_sufficiency": "UNKNOWN"}


def verify_authorized(authorized_sha):
    frozen_path = OUT / "authorized_contract.json"
    require(fingerprint(frozen_path)["sha256"] == authorized_sha, "Explicit frozen authorized contract SHA required")
    frozen = read(frozen_path)
    require(frozen["status"] == "FROZEN_INVENTORY_ONLY_AUTHORIZED_NOT_EXECUTED" and frozen["scope"] == exact_scope() and frozen["budget"] == budget(), "Authorized inventory scope changed")
    for binding in (frozen["prepared"], frozen["authorization"]):
        require(fingerprint(binding["path"]) == binding, "Prepared/auth identity changed")
    prepared = read(frozen["prepared"]["path"])
    check_metadata_sources(prepared)
    validate_human_decision(read(frozen["authorization"]["path"]), frozen["prepared"])
    require(frozen["output_directory"] == prepared["output_directory"], "Output destination changed")
    return frozen, prepared


def worker(authorized_sha):
    frozen, prepared = verify_authorized(authorized_sha)
    output = Path(frozen["output_directory"])
    require((output / "launcher_receipt.json").is_file(), "Bounded launcher required")
    started = time.monotonic()
    written_bytes = 0
    def save(name, value):
        nonlocal written_bytes
        data = canonical(value)
        written_bytes += len(data)
        require(written_bytes <= budget()["maximum_total_new_write_bytes"], "Total metadata write cap exceeded")
        with (output / name).open("xb") as stream:
            stream.write(data)
    try:
        seal = read(SEAL_PATH)
        # First and only actual archive path resolution is now behind exact auth.
        archive_path = Path(seal["archive"]["path"]).resolve(strict=True)
        require(archive_path.stat().st_size == EXPECTED_ARCHIVE_BYTES, "Outer container byte count changed")
        digest = hashlib.sha256()
        with archive_path.open("rb") as container:
            while block := container.read(1024 * 1024):
                digest.update(block)
                require(time.monotonic() - started <= 300, "Container hash time budget exceeded")
            require(digest.hexdigest() == EXPECTED_ARCHIVE_SHA, "Outer container historical SHA changed")
            inventory = central_inventory(container, EXPECTED_ARCHIVE_BYTES)
        require(time.monotonic() - started <= 300, "Inventory time budget exceeded")
        # Existing metadata is permitted; no original train/Selection member file access.
        rows = [json.loads(line) for line in metadata_sources()["assignment_metadata"].read_text(encoding="utf-8").splitlines()]
        require(not any(row["assigned_split"] in {"holdout", "testdev", "test_dev", "test-dev"} for row in rows), "Existing assignment unexpectedly contains sealed-role entries")
        image_ids = {row["logical_id"] for row in inventory["records"] if row["role"] == "image"}
        prefixes = {value.split("_")[0] for value in image_ids}
        overlaps = {}
        for role in sorted({row["assigned_split"] for row in rows}):
            cohort = [row for row in rows if row["assigned_split"] == role]
            overlaps[role] = {"basename_ID_intersection": len(image_ids & {row["image_id"].split("/")[-1] for row in cohort}),
                "prefix_proxy_intersection": len(prefixes & {row["prefix_proxy"] for row in cohort}),
                "image_SHA_or_content_scene_leakage": "UNKNOWN"}
        save("central_directory_inventory.json", inventory)
        save("metadata_role_overlap_limits.json", {"metadata_only": True, "overlaps": overlaps, "IDs_prefixes_are_not_content_or_scene_identity": True})
        check_metadata_sources(prepared)
        receipt = {"status": "COMPLETED_AUTHORIZED_CONTAINER_AND_CENTRAL_DIRECTORY_INVENTORY_ONLY", "ended_at_utc": now(),
            "authorized_contract": fingerprint(OUT / "authorized_contract.json"), "worker_pid": os.getpid(),
            "outer_container_bytes": EXPECTED_ARCHIVE_BYTES, "outer_container_SHA256": digest.hexdigest(),
            "outer_container_identity_not_publisher_or_member_payload_verification": True, "central_directory_entries": inventory["record_count"],
            "paired_image_annotation_entries": inventory["filename_paired_image_annotation_count"], "inventory_metadata_bytes_written": written_bytes,
            "CRC_is_not_SHA256": True, "member_SHA256": None, "member_payload_reads": 0, "extracted_files": 0,
            "image_decode_or_annotation_parse": False, "model_NMS_calibration_or_AP": False,
            "actual_GT_content_scene_leakage_or_scientific_sufficiency": "UNKNOWN", "scientific_PASS": False,
            "further_content_or_confirmation_authorized": False}
        save("completion_receipt.json", receipt)
        return receipt
    except BaseException as error:
        save("failure_receipt.json", {"status": "FAILED_PRESERVED_NO_AUTOMATIC_RETRY", "ended_at_utc": now(),
            "error_type": type(error).__name__, "error": str(error), "evidence": getattr(error, "evidence", {}),
            "member_payload_reads": 0, "model_or_AP_calls": 0})
        raise


def run(authorized_sha):
    frozen, _ = verify_authorized(authorized_sha)
    output = Path(frozen["output_directory"])
    expected_parent = Path(read(metadata_sources()["D_probe"])["new_absolute_directory"]).resolve(strict=True)
    require(output.parent.resolve(strict=True) == expected_parent and output.name == "Holdout_inventory_attempt_01", "Unexpected output parent or exclusive attempt name")
    require(shutil.disk_usage(expected_parent).free >= budget()["minimum_output_free_bytes"], "Inventory output reserve below100MB")
    output.mkdir(exist_ok=False)
    receipt = {"status": "RUNNING_AUTHORIZED_INVENTORY_ONLY", "started_at_utc": now(), "timeout_seconds": 300,
        "authorized_contract": fingerprint(OUT / "authorized_contract.json"), "launcher_pid": os.getpid()}
    receipt_path = output / "launcher_receipt.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    try:
        with (output / "worker_console.txt").open("xb") as console:
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--authorized-contract-sha", authorized_sha],
                cwd=ROOT, stdout=console, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            receipt["worker_pid"] = process.pid
            try:
                code = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=30)
                raise
        receipt["returncode"] = code
        require(code == 0 and (output / "completion_receipt.json").is_file(), "Inventory worker failed; preserve one attempt")
        receipt["status"] = "COMPLETED_BOUNDED_INVENTORY_ONLY"
    except BaseException as error:
        receipt.update(status="FAILED_OR_TIMEOUT_PRESERVED_NO_RETRY", error_type=type(error).__name__, error=str(error), automatic_retry=False)
        raise
    finally:
        receipt["ended_at_utc"] = now()
        temporary = output / "launcher_receipt.final.partial"
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, receipt_path)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "freeze", "run", "worker"))
    parser.add_argument("--reviewed-prepared-sha")
    parser.add_argument("--authorization-sha")
    parser.add_argument("--authorized-contract-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare()
    elif args.mode == "freeze":
        result = freeze(args.reviewed_prepared_sha, args.authorization_sha)
    elif args.mode == "run":
        result = run(args.authorized_contract_sha)
    else:
        result = worker(args.authorized_contract_sha)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
