"""Independently audit prepared VisDrone2019-DET train/val data, read-only.

No preparation-script import, model execution, or test-dev access occurs. The
new output directory must be outside the prepared tree and is never reused.
This verifies preparation fidelity; scene independence and official AP remain
outside its scope.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
import zipfile


EXPECTED_IMAGES = {"train": 6471, "val": 548}
ROLES = {"train", "training_val", "calibration_pool", "debug", "selection",
         "quarantine", "duplicate_excluded"}
ACTIVE_ROLES = {"train", "training_val", "calibration_pool", "debug", "selection"}
LABEL_COORDINATE_ATOL = 1e-12  # Only accommodates the declared 12-significant-digit label serialization.


class AuditError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reject_holdout_path(path: Path | str) -> None:
    lowered = str(path).casefold().replace("_", "-")
    require(not any(token in lowered for token in ("test-dev", "testdev", "test-challenge", "holdout")),
            f"Test/Holdout paths are prohibited: {path}")


def prepared_path(root: Path, relative: str) -> Path:
    reject_holdout_path(relative)
    pure = PurePosixPath(relative)
    require(bool(relative) and not pure.is_absolute() and "\\" not in relative and ":" not in relative
            and all(part not in {"", ".", ".."} for part in relative.split("/")),
            f"Unsafe prepared relative path: {relative!r}")
    path = root.joinpath(*pure.parts)
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root), f"Prepared path escapes its root: {relative}")
    require(path.is_file() and not path.is_symlink(), f"Expected a regular prepared file: {relative}")
    return resolved


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank JSONL record in {path.name}:{number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"Expected JSON object in {path.name}:{number}")
            yield value


def path_list(root: Path, relative: str) -> list[Path]:
    result = []
    for line in prepared_path(root, relative).read_text(encoding="utf-8").splitlines():
        require(bool(line.strip()), f"Blank path in {relative}")
        reject_holdout_path(line)
        path = Path(line)
        require(path.is_absolute(), f"Split paths must be absolute: {relative}: {line}")
        path = path.resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), f"Split image escapes prepared root: {line}")
        result.append(path)
    require(len(result) == len(set(result)), f"Duplicate paths in {relative}")
    return result


def annotation_rows(data: bytes, width: int, height: int, image_id: str, split: str) -> list[dict[str, Any]]:
    """Parse the publisher CSV directly; no preparation implementation is reused."""
    result = []
    for physical_line, raw in enumerate(data.decode("utf-8-sig").splitlines()):
        if not raw.strip():
            continue
        fields = raw.split(",")
        if len(fields) == 9 and not fields[-1].strip():
            fields = fields[:-1]
        require(len(fields) == 8, f"Annotation field count: {image_id}:{physical_line}")
        values = [float(field.strip()) for field in fields]
        require(all(math.isfinite(value) for value in values), f"Nonfinite annotation: {image_id}:{physical_line}")
        left, top, box_width, box_height, score, category, truncation, occlusion = values
        require(all(value.is_integer() for value in values[4:]), f"Noninteger flags: {image_id}:{physical_line}")
        require(score in (0, 1) and 0 <= category <= 11 and truncation in (0, 1, 2, 3)
                and occlusion in (0, 1, 2), f"Invalid annotation flags: {image_id}:{physical_line}")
        original = [left, top, left + box_width, top + box_height]
        require(all(math.isfinite(value) for value in original), f"Coordinate overflow: {image_id}:{physical_line}")
        clipped = [min(max(original[0], 0), width), min(max(original[1], 0), height),
                   min(max(original[2], 0), width), min(max(original[3], 0), height)]
        publisher_ignored = score == 0 or category in (0, 11)
        degenerate = box_width <= 0 or box_height <= 0
        quality = degenerate and not publisher_ignored
        require(not (quality and split != "train"), f"Invalid positive geometry in val: {image_id}:{physical_line}")
        reasons = []
        if score == 0:
            reasons.append("score_0")
        if category in (0, 11):
            reasons.append(f"category_{int(category)}")
        if degenerate:
            reasons.append("non_positive_ignored_geometry" if publisher_ignored else "invalid_positive_box_geometry")
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            reasons.append("empty_after_clipping")
        row = {"image_id": image_id, "target_id": f"{image_id}:{physical_line}",
               "physical_line_zero_based": physical_line, "raw_line": raw,
               "original_xywh": values[:4], "original_xyxy": original, "clipped_xyxy": clipped,
               "was_clipped": original != clipped, "score": int(score), "category_original": int(category),
               "class_id": int(category) - 1 if 1 <= category <= 10 else None,
               "truncation": int(truncation), "occlusion": int(occlusion),
               "publisher_ignored": publisher_ignored, "excluded_for_preparation_quality": quality,
               "ignored": bool(reasons), "ignore_reasons": reasons, "yolo_line_zero_based": None}
        result.append(row)
    return result


def compare_yolo(data: str, targets: list[dict[str, Any]], width: int, height: int, image_id: str) -> None:
    valid = [target for target in targets if not target["ignored"]]
    lines = data.splitlines()
    require(len(lines) == len(valid), f"YOLO target count differs: {image_id}")
    for index, (line, target) in enumerate(zip(lines, valid)):
        parts = line.split()
        require(len(parts) == 5, f"YOLO field count: {image_id}:{index}")
        numbers = [float(value) for value in parts]
        require(all(math.isfinite(value) for value in numbers), f"Nonfinite YOLO value: {image_id}:{index}")
        require(numbers[0].is_integer() and int(numbers[0]) == target["class_id"], f"YOLO class mismatch: {image_id}:{index}")
        x1, y1, x2, y2 = target["clipped_xyxy"]
        expected = [(x1 + x2) / width / 2, (y1 + y2) / height / 2,
                    (x2 - x1) / width, (y2 - y1) / height]
        require(all(0 <= number <= 1 for number in numbers[1:]), f"YOLO value outside [0,1]: {image_id}:{index}")
        require(all(abs(actual - wanted) <= LABEL_COORDINATE_ATOL for actual, wanted in zip(numbers[1:], expected)),
                f"YOLO coordinates differ from original clipped annotation: {image_id}:{index}")
        target["yolo_line_zero_based"] = index
        target["normalized_xywh"] = expected


def consume_sidecar(iterator: Iterator[dict[str, Any]], expected: list[dict[str, Any]], name: str) -> None:
    for wanted in expected:
        actual = next(iterator, None)
        require(actual is not None, f"Missing sidecar row: {name}:{wanted['target_id']}")
        require(set(actual) == set(wanted), f"Sidecar schema mismatch: {name}:{wanted['target_id']}")
        for key, value in wanted.items():
            if key == "normalized_xywh":
                require(len(actual[key]) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v)
                        and abs(v - w) <= LABEL_COORDINATE_ATOL for v, w in zip(actual[key], value)),
                        f"Sidecar normalized coordinates differ: {name}:{wanted['target_id']}")
            else:
                require(actual[key] == value, f"Sidecar {key} mismatch: {name}:{wanted['target_id']}")


def audit(prepared: Path, report: dict[str, Any]) -> None:
    def passed(name: str, detail: Any) -> None:
        report["checks"].append({"name": name, "status": "PASS", "detail": detail})
        print(f"AUDIT_CHECK_PASS: {name}", flush=True)

    summary = read_json(prepared_path(prepared, "preparation_summary.json"))
    require(summary["status"] == "PREPARED_NOT_FROZEN_NOT_TRAINED_NOT_SCORED", "Unexpected preparation status")
    require(summary["holdout_opened"] is False and summary["official_val_is_selection_only"] is True,
            "Preparation does not declare the required data-access scope")
    require(summary["sources_unchanged_after_preparation"] is True
            and summary["source_image_and_annotation_bytes_preserved"] is True, "Source preservation not declared")
    require(set(summary["source_archives"]) == {"train", "val"}, "Only train/val sources are permitted")
    source_paths = {}
    for split in ("train", "val"):
        entry = summary["source_archives"][split]
        reject_holdout_path(entry["path"])
        source = Path(entry["path"]).resolve(strict=True)
        require(source.is_file() and "test" not in source.name.casefold(), f"Not a train/val archive: {source}")
        require(source.stat().st_size == entry["bytes"] and sha256(source) == entry["sha256"],
                f"Original {split} ZIP identity differs from summary")
        source_paths[split] = source
    require(source_paths["train"] != source_paths["val"], "Source archives must be distinct")
    passed("source_archive_identities", summary["source_archives"])

    manifest = read_json(prepared_path(prepared, "manifest.json"))
    entries = manifest["files"]
    declared = {entry["path"] for entry in entries}
    require(len(declared) == len(entries) and "manifest.json" not in declared, "Duplicate/self-listed manifest entries")
    actual = {path.relative_to(prepared).as_posix() for path in prepared.rglob("*") if path.is_file()}
    require(actual == declared | {"manifest.json"}, f"Manifest/file-set mismatch: missing={sorted(declared-actual)[:5]}, extra={sorted(actual-declared-{'manifest.json'})[:5]}")
    for index, entry in enumerate(entries, 1):
        path = prepared_path(prepared, entry["path"])
        require(path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"],
                f"Prepared manifest hash/size mismatch: {entry['path']}")
        if index % 5000 == 0:
            print(f"AUDIT_MANIFEST_FILES: {index}/{len(entries)}", flush=True)
    report["counters"]["manifest_files_verified"] = len(entries)
    passed("complete_prepared_file_manifest", {"files": len(entries), "manifest_sha256": sha256(prepared / "manifest.json")})

    rows = list(jsonl_rows(prepared_path(prepared, "image_manifest.jsonl")))
    require(len(rows) == sum(EXPECTED_IMAGES.values()), f"Expected 7019 image records, found {len(rows)}")
    require(Counter(row["source_split"] for row in rows) == Counter(EXPECTED_IMAGES), "Unexpected official train/val image counts")
    require(len({row["image_id"] for row in rows}) == len(rows), "Duplicate image IDs")
    by_path = {}
    for row in rows:
        require(row["assigned_split"] in ROLES and row["source_split"] in ("train", "val"), "Unknown data role")
        path = prepared_path(prepared, row["image_path"])
        require(path not in by_path, f"Duplicate image path: {path}")
        by_path[path] = row
        require((row["source_split"] == "val") == (row["assigned_split"] == "selection"),
                f"Official split/Selection role mismatch: {row['image_id']}")
        require(row["image_id"] == f"{row['source_split']}/{Path(row['filename']).stem}", "Image ID/filename mismatch")
        require(row["prefix_proxy"] == Path(row["filename"]).stem.split("_", 1)[0], "Incorrect filename prefix proxy")
        require(row["prefix_is_verified_scene_identity"] is False, "Filename prefix must not claim verified scene identity")
    for role in sorted(ROLES):
        listed = path_list(prepared, f"splits/{role}.txt")
        expected = [prepared_path(prepared, row["image_path"]) for row in rows if row["assigned_split"] == role]
        require(listed == expected, f"Split list does not exactly match manifest role/order: {role}")
    counts = Counter(row["assigned_split"] for row in rows)
    require(dict(counts) == summary["split_counts"], "Summary split counts differ from actual manifest")
    proposal = read_json(prepared_path(prepared, "split_proposal.json"))
    require(dict(counts) == proposal["counts"], "Proposal split counts differ from actual manifest")
    passed("image_manifest_and_absolute_split_lists", dict(counts))

    # Reconstruct connected components from identities; do not trust the supplied
    # component field when proving quarantine or cross-role isolation.
    parent = {row["image_id"]: row["image_id"] for row in rows}
    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for field in ("prefix_proxy", "image_sha256"):
        first = {}
        for row in rows:
            key = row[field]
            image_id = row["image_id"]
            if key in first:
                a, b = find(first[key]), find(image_id)
                parent[max(a, b)] = min(a, b)
            else:
                first[key] = image_id
    independent_components = {row["image_id"]: find(row["image_id"]) for row in rows}
    components = defaultdict(list)
    by_content = defaultdict(list)
    for row in rows:
        component = independent_components[row["image_id"]]
        require(row["isolation_component"] == component, f"Component field differs from independent grouping: {row['image_id']}")
        components[component].append(row)
        by_content[row["image_sha256"]].append(row)
    conflict_components = set()
    for bucket in by_content.values():
        canonical = min(bucket, key=lambda row: (row["source_split"] != "val", row["image_id"]))
        if len({row["annotation_sha256"] for row in bucket}) > 1:
            conflict_components.add(independent_components[canonical["image_id"]])
        for row in bucket:
            require(row["content_canonical_image_id"] == canonical["image_id"]
                    and row["is_content_canonical"] == (row["image_id"] == canonical["image_id"]),
                    f"Content canonical identity mismatch: {row['image_id']}")
    for component, bucket in components.items():
        mixed = {row["source_split"] for row in bucket} == {"train", "val"}
        if mixed or component in conflict_components:
            require(all(row["source_split"] == "val" or row["assigned_split"] == "quarantine" for row in bucket),
                    f"Cross-val/content-conflict train component was used: {component}")
        for row in bucket:
            if row["source_split"] == "train" and row["assigned_split"] in ACTIVE_ROLES:
                require(row["is_content_canonical"] is True, f"Duplicate train image remains active: {row['image_id']}")
    passed("independently_reconstructed_components_and_duplicate_isolation", {"components": len(components)})

    for field in ("image_sha256", "prefix_proxy", "isolation_component"):
        assignments = defaultdict(set)
        for row in rows:
            if row["assigned_split"] in ACTIVE_ROLES:
                assignments[row[field]].add(row["assigned_split"])
        require(all(len(roles) == 1 for roles in assignments.values()), f"Cross-role leakage by {field}")
    passed("active_role_content_prefix_component_isolation", sorted(ACTIVE_ROLES))

    calibration_lists = []
    seeds = proposal["policy"]["calibration_seeds"]
    require(len(seeds) == 3 and len(set(seeds)) == 3 and proposal["policy"]["calibration_per_seed"] == 64,
            "Expected three unique calibration seeds of 64 images")
    by_id = {row["image_id"]: row for row in rows}
    for seed in seeds:
        ids = proposal["calibration_image_ids"][str(seed)]
        require(len(ids) == 64 and len(set(ids)) == 64, f"Calibration size/uniqueness mismatch: {seed}")
        require(all(key in by_id and by_id[key]["assigned_split"] == "calibration_pool" for key in ids),
                f"Calibration image outside reserved pool: {seed}")
        expected = [prepared_path(prepared, by_id[key]["image_path"]) for key in ids]
        require(path_list(prepared, f"splits/calibration_seed_{seed}.txt") == expected, f"Calibration list mismatch: {seed}")
        calibration_lists.append({by_id[key]["image_sha256"] for key in ids})
    require(sum(map(len, calibration_lists)) == 192 and len(set.union(*calibration_lists)) == 192,
            "Calibration samples overlap by exact image content")
    passed("three_64_image_calibration_sets", {"seeds": seeds, "unique_image_sha256": 192})

    quality_ids = []
    quality_components = set()
    audited = 0
    all_counts = {}
    from PIL import Image
    with ExitStack() as stack:
        archives = {split: stack.enter_context(zipfile.ZipFile(path)) for split, path in source_paths.items()}
        for split in ("train", "val"):
            archive = archives[split]
            names = archive.namelist()
            require(len(names) == len(set(names)), f"Duplicate ZIP member names in {split}")
            root = f"VisDrone2019-DET-{split}"
            require(all(name == root + "/" or name.startswith(root + "/") for name in names),
                    f"Unexpected ZIP root; only {root} is permitted")
            targets = jsonl_rows(prepared_path(prepared, f"targets_{split}.jsonl"))
            ignored = jsonl_rows(prepared_path(prepared, f"ignored_regions_{split}.jsonl"))
            split_rows = sorted((row for row in rows if row["source_split"] == split),
                                key=lambda row: Path(row["filename"]).stem.casefold())
            source_images = {name for name in names if not name.endswith("/") and name.startswith(root + "/images/")}
            source_annotations = {name for name in names if not name.endswith("/") and name.startswith(root + "/annotations/")}
            require(source_images == {row["source_zip_member"] for row in split_rows}, f"ZIP image set differs: {split}")
            require(source_annotations == {row["annotation_zip_member"] for row in split_rows}, f"ZIP annotation set differs: {split}")
            for member in archive.infolist():
                if not member.is_dir() and member.filename not in source_images | source_annotations:
                    extra = archive.read(member)  # Checks CRC for archive extras, too.
                    extra_relative = PurePosixPath(member.filename).relative_to(root).as_posix()
                    require(sha256(prepared_path(prepared, f"archive_extras/{split}/{extra_relative}")) == bytes_sha256(extra),
                            f"Original archive extra differs: {member.filename}")
            counter = Counter(crc_verified_files=sum(not item.is_dir() for item in archive.infolist()))
            counter["images"] = len(split_rows)
            for row in split_rows:
                image_id = row["image_id"]
                require(row["image_path"] == f"images/{split}/{row['filename']}", f"Image path mismatch: {image_id}")
                require(row["source_zip_member"] == f"{root}/images/{row['filename']}", f"Source member mismatch: {image_id}")
                annotation_name = PurePosixPath(row["annotation_zip_member"]).name
                require(row["annotation_zip_member"] == f"{root}/annotations/{annotation_name}"
                        and Path(annotation_name).stem.casefold() == Path(row["filename"]).stem.casefold(),
                        f"Annotation/image pairing mismatch: {image_id}")
                require(row["annotation_path"] == f"annotations/{split}/{annotation_name}"
                        and row["label_path"] == f"labels/{split}/{Path(row['filename']).stem}.txt", f"Annotation/label path mismatch: {image_id}")
                image_data = archive.read(row["source_zip_member"])
                annotation_data = archive.read(row["annotation_zip_member"])
                image_path = prepared_path(prepared, row["image_path"])
                annotation_path = prepared_path(prepared, row["annotation_path"])
                label_path = prepared_path(prepared, row["label_path"])
                require(bytes_sha256(image_data) == row["image_sha256"] == sha256(image_path), f"Original image bytes differ: {image_id}")
                require(bytes_sha256(annotation_data) == row["annotation_sha256"] == sha256(annotation_path), f"Original annotation bytes differ: {image_id}")
                require(sha256(label_path) == row["label_sha256"], f"Label SHA differs: {image_id}")
                with Image.open(io.BytesIO(image_data)) as image:
                    width, height = image.size
                    require((width, height) == (row["width"], row["height"]) and width > 0 and height > 0,
                            f"Original image dimensions differ: {image_id}")
                    require(image.mode == row["image_mode"] and image.getexif().get(274) == row["exif_orientation"],
                            f"Image mode/EXIF differs: {image_id}")
                    require(row["exif_orientation"] in (None, 1), f"Undeclared image rotation: {image_id}")
                parsed = annotation_rows(annotation_data, width, height, image_id, split)
                compare_yolo(label_path.read_text(encoding="utf-8"), parsed, width, height, image_id)
                consume_sidecar(targets, parsed, f"targets_{split}")
                consume_sidecar(ignored, [target for target in parsed if target["ignored"]], f"ignored_regions_{split}")
                quality = [target["target_id"] for target in parsed if target["excluded_for_preparation_quality"]]
                require(row["requires_annotation_quality_quarantine"] == bool(quality)
                        and row["annotation_quality_target_ids"] == quality, f"Quality audit IDs differ: {image_id}")
                if quality:
                    require(split == "train" and row["assigned_split"] == "quarantine", f"Invalid positive geometry is active: {image_id}")
                    quality_ids.extend(quality)
                    quality_components.add(row["isolation_component"])
                valid_count = sum(not target["ignored"] for target in parsed)
                ignored_count = sum(target["ignored"] for target in parsed)
                publisher_count = sum(target["publisher_ignored"] for target in parsed)
                clipped_count = sum(target["was_clipped"] for target in parsed)
                require((row["valid_target_count"], row["ignored_target_count"], row["publisher_ignored_target_count"], row["clipped_target_count"])
                        == (valid_count, ignored_count, publisher_count, clipped_count), f"Per-image target counters differ: {image_id}")
                for target in parsed:
                    counter["annotations_total"] += 1
                    counter["ignored_annotations" if target["ignored"] else "valid_yolo_targets"] += 1
                    if target["publisher_ignored"]:
                        counter["publisher_ignored_annotations"] += 1
                    if target["excluded_for_preparation_quality"]:
                        counter["annotation_quality_excluded_positive_targets"] += 1
                    if target["was_clipped"]:
                        counter["clipped_annotations"] += 1
                    counter.update(target["ignore_reasons"])
                audited += 1
                if audited % 1000 == 0:
                    print(f"AUDIT_ORIGINAL_IMAGES_AND_LABELS: {audited}/7019", flush=True)
            require(next(targets, None) is None and next(ignored, None) is None, f"Unexpected extra sidecar rows: {split}")
            require(dict(counter) == summary["split_annotation_counts"][split], f"Independent annotation counters differ: {split}")
            all_counts[split] = dict(counter)
    passed("original_zip_bytes_dimensions_labels_and_annotation_sidecars", {"images": audited, "annotation_counts": all_counts})

    quarantined = sorted(row["image_id"] for row in rows if row["source_split"] == "train"
                         and row["isolation_component"] in quality_components)
    for row in rows:
        if row["source_split"] == "train" and row["isolation_component"] in quality_components:
            require(row["assigned_split"] == "quarantine" and "component_contains_invalid_positive_box_geometry" in row["quarantine_reasons"],
                    f"Quality component member was used: {row['image_id']}")
    quality_summary = proposal["annotation_quality_quarantine"]
    require(quality_summary["invalid_positive_target_ids"] == sorted(quality_ids)
            and quality_summary["affected_train_component_ids"] == sorted(quality_components)
            and quality_summary["quarantined_train_image_ids"] == quarantined
            and quality_summary["quarantined_train_image_count"] == len(quarantined), "Quality quarantine summary differs")
    require(summary["annotation_quality_policy"] == quality_summary, "Summary quality policy differs from proposal")
    passed("entire_invalid_positive_train_component_quarantined", {"target_ids": sorted(quality_ids), "train_images": len(quarantined)})
    report["counters"].update({"images_verified": audited, "source_split_images": EXPECTED_IMAGES,
                               "assigned_role_images": dict(counts), "annotation_counts": all_counts,
                               "quality_excluded_positive_targets": len(quality_ids), "quality_quarantined_train_images": len(quarantined)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reject_holdout_path(args.prepared)
    reject_holdout_path(args.output)
    prepared = args.prepared.resolve(strict=True)
    require(prepared.is_dir(), "Prepared path must be a directory")
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing any existing output")
    output = args.output.resolve()
    require(not output.is_relative_to(prepared) and not prepared.is_relative_to(output),
            "Output must be outside prepared and cannot be its ancestor")
    require(not output.exists(), "Refusing resolved existing output")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "rampq.gate1.independent_preparation_audit/v1", "status": "RUNNING",
              "prepared": str(prepared), "audit_script_sha256": sha256(Path(__file__)),
              "checks": [], "counters": {}, "limits": {
                  "holdout_or_testdev_read": False, "models_executed": False, "prepared_files_modified": False,
                  "near_duplicate_independence": "NOT_ESTABLISHED",
                  "true_scene_identity_and_independence": "NOT_ESTABLISHED",
                  "ignore_aware_AP_evaluator": "NOT_IMPLEMENTED_BY_THIS_AUDIT",
                  "prefix_grouping": "filename proxy only", "label_coordinate_comparison_atol": LABEL_COORDINATE_ATOL,
                  "preparation_audit_is_not_Gate1_result": True}}
    exit_code = 0
    try:
        audit(prepared, report)
        report["status"] = "PASS_PREPARATION_FIDELITY_ONLY"
    except Exception as error:
        exit_code = 2
        report["status"] = "FAIL"
        report["checks"].append({"name": "audit_failure", "status": "FAIL", "error_type": type(error).__name__, "detail": str(error)})
    finally:
        (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output / "audit.json"), "checks": len(report["checks"])}, ensure_ascii=False), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
