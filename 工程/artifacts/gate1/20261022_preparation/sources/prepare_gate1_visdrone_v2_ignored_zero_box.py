"""Prepare only VisDrone train/val archives; never open a final holdout.

This is data preparation, not an evaluation or a frozen research contract.
The original image/annotation bytes are retained. Converted YOLO labels are
training conveniences; ignored regions remain in sidecars and require an
ignore-aware evaluator. No download, model import, inference, or scoring occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import stat
import uuid
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
         "awning-tricycle", "bus", "motor"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024 * 1024


class PreparationError(ValueError):
    """The supplied archives cannot be prepared without ambiguity."""


@dataclass(frozen=True)
class SplitPolicy:
    seed: int = 20261022
    training_val_target: int = 300
    calibration_pool_target: int = 192
    debug_target: int = 64
    calibration_seeds: tuple[int, ...] = (2026102201, 2026102202, 2026102203)
    calibration_per_seed: int = 64


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validated_members(archive: zipfile.ZipFile, split: str) -> list[zipfile.ZipInfo]:
    """Validate every member before writing any archive content."""
    expected_root = f"VisDrone2019-DET-{split}"
    seen: set[str] = set()
    total = 0
    result = []
    for info in archive.infolist():
        name = info.filename
        if info.orig_filename != name:
            raise PreparationError(f"ZIP member requires unsafe name normalization: {info.orig_filename!r}")
        path = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or path.is_absolute()
                or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
                or any(ord(c) < 32 for c in name)):
            raise PreparationError(f"Unsafe ZIP member: {name!r}")
        if path.parts[0] != expected_root:
            raise PreparationError(f"Expected only {expected_root!r}, found {name!r}")
        if len(path.parts) == 1 and not info.is_dir():
            raise PreparationError(f"ZIP dataset root must be a directory: {name}")
        reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)}
        if any(part.endswith((".", " ")) or part.split(".", 1)[0].upper() in reserved for part in path.parts):
            raise PreparationError(f"Unsafe Windows ZIP filename: {name}")
        key = name.rstrip("/").casefold()
        if key in seen:
            raise PreparationError(f"Duplicate/case-colliding ZIP member: {name}")
        seen.add(key)
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
            raise PreparationError(f"Link or special ZIP member: {name}")
        if info.flag_bits & 1:
            raise PreparationError(f"Encrypted ZIP member: {name}")
        if info.file_size > MAX_MEMBER_BYTES or info.file_size < 0:
            raise PreparationError(f"ZIP member size is outside the limit: {name}")
        total += info.file_size
        if total > MAX_ARCHIVE_BYTES:
            raise PreparationError("ZIP uncompressed size exceeds the preparation limit")
        result.append(info)
    # A regular file must not serve as an ancestor of another path.
    files = {i.filename.casefold() for i in result if not i.is_dir()}
    for info in result:
        for ancestor in PurePosixPath(info.filename).parents:
            if str(ancestor).casefold() in files:
                raise PreparationError(f"ZIP file/directory collision: {info.filename}")
    return sorted(result, key=lambda i: i.filename)


def image_dimensions(data: bytes) -> tuple[int, int, str, int | None]:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 32_000_000:
            raise PreparationError("Image pixel count exceeds preparation limit")
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        orientation = image.getexif().get(274)
        if width <= 0 or height <= 0:
            raise PreparationError("Image dimensions must be positive")
        # No silent EXIF rotation; training and annotations must use the same axes.
        if orientation not in {None, 1}:
            raise PreparationError(f"EXIF orientation {orientation} needs an explicit coordinate policy")
        return width, height, image.mode, orientation


def convert_annotation(data: bytes, width: int, height: int, image_id: str) -> tuple[str, list[dict[str, Any]]]:
    """Preserve zero-based physical line identity, including skipped blank lines."""
    targets = []
    yolo = []
    for physical_line, raw_line in enumerate(data.decode("utf-8-sig").splitlines()):
        if not raw_line.strip():
            continue
        fields = next(csv.reader([raw_line]))
        # Official releases can have a trailing delimiter.
        if len(fields) == 9 and not fields[-1].strip():
            fields.pop()
        if len(fields) != 8:
            raise PreparationError(f"{image_id}:{physical_line}: expected 8 annotation fields")
        try:
            values = [float(v.strip()) for v in fields]
        except ValueError as error:
            raise PreparationError(f"{image_id}:{physical_line}: non-numeric annotation") from error
        if not all(math.isfinite(v) for v in values):
            raise PreparationError(f"{image_id}:{physical_line}: non-finite annotation")
        x, y, bw, bh = values[:4]
        score, category, truncation, occlusion = values[4:]
        if any(v != int(v) for v in values[4:]):
            raise PreparationError(f"{image_id}:{physical_line}: annotation flags must be integers")
        if score not in {0, 1} or not 0 <= category <= 11:
            raise PreparationError(f"{image_id}:{physical_line}: invalid score/category")
        if truncation not in {0, 1, 2, 3} or occlusion not in {0, 1, 2}:
            raise PreparationError(f"{image_id}:{physical_line}: invalid truncation/occlusion")
        non_positive_geometry = bw <= 0 or bh <= 0
        already_ignored = score == 0 or category in {0, 11}
        if non_positive_geometry and not already_ignored:
            raise PreparationError(f"{image_id}:{physical_line}: non-positive box size")
        original = [x, y, x + bw, y + bh]
        if not all(math.isfinite(v) for v in original):
            raise PreparationError(f"{image_id}:{physical_line}: box coordinate arithmetic overflow")
        clipped = [max(0.0, min(float(width), original[0])),
                   max(0.0, min(float(height), original[1])),
                   max(0.0, min(float(width), original[2])),
                   max(0.0, min(float(height), original[3]))]
        reasons = []
        if score == 0:
            reasons.append("score_0")
        if category in {0, 11}:
            reasons.append(f"category_{int(category)}")
        if non_positive_geometry:
            reasons.append("non_positive_ignored_geometry")
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            reasons.append("empty_after_clipping")
        row = {"image_id": image_id, "target_id": f"{image_id}:{physical_line}",
               "physical_line_zero_based": physical_line, "raw_line": raw_line,
               "original_xywh": values[:4], "original_xyxy": original,
               "clipped_xyxy": clipped, "was_clipped": clipped != original,
               "score": int(score), "category_original": int(category),
               "class_id": int(category) - 1 if 1 <= category <= 10 else None,
               "truncation": int(truncation), "occlusion": int(occlusion),
               "ignored": bool(reasons), "ignore_reasons": reasons,
               "yolo_line_zero_based": None}
        if not reasons:
            x1, y1, x2, y2 = clipped
            normalized = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                          (x2 - x1) / width, (y2 - y1) / height]
            row["yolo_line_zero_based"] = len(yolo)
            row["normalized_xywh"] = normalized
            yolo.append(f"{int(category) - 1} " + " ".join(f"{v:.12g}" for v in normalized))
        targets.append(row)
    return "\n".join(yolo) + ("\n" if yolo else ""), targets


def extract_split(zip_path: Path, split: str, stage: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    annotations: dict[str, tuple[Path, str]] = {}
    counts = Counter()
    root = f"VisDrone2019-DET-{split}"
    with zipfile.ZipFile(zip_path) as archive:
        members = validated_members(archive, split)
        for info in members:
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            # read() checks the CRC, including non-image archive extras.
            data = archive.read(info)
            counts["crc_verified_files"] += 1
            if len(parts) == 3 and parts[1] == "images" and PurePosixPath(parts[2]).suffix.lower() in IMAGE_EXTENSIONS:
                name = parts[2]
                stem = PurePosixPath(name).stem.casefold()
                if stem in images:
                    raise PreparationError(f"Duplicate image stem in {split}: {name}")
                width, height, mode, orientation = image_dimensions(data)
                rel = Path("images") / split / name
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                images[stem] = {"source_split": split, "filename": name,
                    "image_id": f"{split}/{PurePosixPath(name).stem}",
                    "prefix_proxy": PurePosixPath(name).stem.split("_", 1)[0],
                    "prefix_is_verified_scene_identity": False,
                    "source_zip_member": info.filename, "image_path": rel.as_posix(),
                    "image_sha256": digest_bytes(data), "width": width, "height": height,
                    "image_mode": mode, "exif_orientation": orientation}
            elif len(parts) == 3 and parts[1] == "annotations" and parts[2].lower().endswith(".txt"):
                stem = PurePosixPath(parts[2]).stem.casefold()
                if stem in annotations:
                    raise PreparationError(f"Duplicate annotation stem in {split}: {parts[2]}")
                rel = Path("annotations") / split / parts[2]
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                annotations[stem] = (rel, info.filename)
            else:
                # Keep all archive extras verbatim, but reject malformed dataset subtrees.
                if len(parts) > 1 and parts[1] in {"images", "annotations"}:
                    raise PreparationError(f"Unexpected dataset member under {root}: {info.filename}")
                rel = Path("archive_extras") / split / Path(*parts[1:])
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
    if not images or set(images) != set(annotations):
        raise PreparationError(f"Image/annotation pairing failed in {split}: "
            f"images_only={sorted(set(images) - set(annotations))[:8]}, "
            f"annotations_only={sorted(set(annotations) - set(images))[:8]}")
    target_path = stage / f"targets_{split}.jsonl"
    ignore_path = stage / f"ignored_regions_{split}.jsonl"
    with target_path.open("w", encoding="utf-8", newline="\n") as all_handle, ignore_path.open("w", encoding="utf-8", newline="\n") as ignored_handle:
        for stem in sorted(images):
            row = images[stem]
            annotation_rel, member = annotations[stem]
            annotation = (stage / annotation_rel).read_bytes()
            label, targets = convert_annotation(annotation, row["width"], row["height"], row["image_id"])
            label_rel = Path("labels") / split / (Path(row["filename"]).stem + ".txt")
            dest = stage / label_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(label, encoding="utf-8", newline="\n")
            for target in targets:
                payload = json.dumps(target, ensure_ascii=False, sort_keys=True) + "\n"
                all_handle.write(payload)
                if target["ignored"]:
                    ignored_handle.write(payload)
                counts["annotations_total"] += 1
                counts["ignored_annotations" if target["ignored"] else "valid_yolo_targets"] += 1
                if target["was_clipped"]:
                    counts["clipped_annotations"] += 1
                for reason in target["ignore_reasons"]:
                    counts[reason] += 1
            row.update(annotation_path=annotation_rel.as_posix(), annotation_zip_member=member,
                       annotation_sha256=digest_bytes(annotation), label_path=label_rel.as_posix(),
                       label_sha256=digest_bytes(label.encode("utf-8")),
                       valid_target_count=sum(not t["ignored"] for t in targets),
                       ignored_target_count=sum(t["ignored"] for t in targets),
                       clipped_target_count=sum(t["was_clipped"] for t in targets))
    counts["images"] = len(images)
    return [images[k] for k in sorted(images)], dict(counts)


def split_records(rows: list[dict[str, Any]], policy: SplitPolicy) -> dict[str, Any]:
    """Group by filename proxy AND exact-content identity, quarantine cross-val links."""
    if min(policy.training_val_target, policy.calibration_pool_target, policy.debug_target,
           policy.calibration_per_seed) < 1:
        raise PreparationError("Split target sizes must be positive")
    if len(set(policy.calibration_seeds)) != len(policy.calibration_seeds):
        raise PreparationError("Calibration seeds must be unique")
    parent = {r["image_id"]: r["image_id"] for r in rows}
    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    by_prefix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prefix[row["prefix_proxy"]].append(row)
        by_sha[row["image_sha256"]].append(row)
    for buckets in (by_prefix, by_sha):
        for bucket in buckets.values():
            for row in bucket[1:]:
                union(bucket[0]["image_id"], row["image_id"])
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        components[find(row["image_id"])].append(row)
    collisions = []
    for kind, buckets in (("prefix_proxy", by_prefix), ("exact_image_sha256", by_sha)):
        for value, bucket in sorted(buckets.items()):
            if {r["source_split"] for r in bucket} == {"train", "val"}:
                collisions.append({"kind": kind, "value": value,
                                   "image_ids": sorted(r["image_id"] for r in bucket),
                                   "policy": "quarantine_entire_connected_train_component"})
    duplicates = []
    conflicting_components = set()
    for sha, bucket in sorted(by_sha.items()):
        canonical = min(bucket, key=lambda r: (r["source_split"] != "val", r["image_id"]))
        conflict = len({r["annotation_sha256"] for r in bucket}) > 1
        if len(bucket) > 1:
            duplicates.append({"image_sha256": sha, "canonical_image_id": canonical["image_id"],
                "image_ids": sorted(r["image_id"] for r in bucket), "annotation_bytes_conflict": conflict})
        if conflict:
            conflicting_components.add(find(canonical["image_id"]))
        for row in bucket:
            row["content_canonical_image_id"] = canonical["image_id"]
            row["is_content_canonical"] = row is canonical
    available: dict[str, list[dict[str, Any]]] = {}
    for component, bucket in sorted(components.items()):
        mixed = {r["source_split"] for r in bucket} == {"train", "val"}
        conflict = component in conflicting_components
        for row in bucket:
            row["isolation_component"] = component
            row["quarantine_reasons"] = (["cross_train_val_prefix_or_content_component"] if mixed else []) + (["duplicate_image_annotation_bytes_conflict"] if conflict else [])
            if row["source_split"] == "val":
                row["assigned_split"] = "selection"
            elif mixed or conflict:
                row["assigned_split"] = "quarantine"
            elif not row["is_content_canonical"]:
                row["assigned_split"] = "duplicate_excluded"
            else:
                row["assigned_split"] = "unassigned"
                available.setdefault(component, []).append(row)
    def rank(key: str, salt: str) -> str:
        return digest_bytes(f"{policy.seed}:{salt}:{key}".encode())
    remaining = dict(available)
    # Reserve whole components, never individual near-sequence filename proxies.
    for label, target in (("training_val", policy.training_val_target),
                          ("calibration_pool", policy.calibration_pool_target),
                          ("debug", policy.debug_target)):
        count = 0
        for component in sorted(list(remaining), key=lambda c: (rank(c, label), c)):
            if count >= target:
                break
            for row in remaining.pop(component):
                row["assigned_split"] = label
                count += 1
        if count < target:
            raise PreparationError(f"Insufficient independent train groups for {label}: {count} < {target}")
    if not remaining:
        raise PreparationError("No training groups remain after reservations")
    for bucket in remaining.values():
        for row in bucket:
            row["assigned_split"] = "train"
    pool = [r for r in rows if r["assigned_split"] == "calibration_pool"]
    required = len(policy.calibration_seeds) * policy.calibration_per_seed
    if len(pool) < required:
        raise PreparationError(f"Calibration pool needs {required} unique images, has {len(pool)}")
    unused = {r["image_id"]: r for r in pool}
    calibration = {}
    for seed in policy.calibration_seeds:
        chosen = sorted(unused, key=lambda k: (digest_bytes(f"{seed}:{k}".encode()), k))[:policy.calibration_per_seed]
        calibration[str(seed)] = chosen
        for key in chosen:
            unused.pop(key)
    # Assert actual content and group isolation, not merely different path strings.
    used = [r for r in rows if r["assigned_split"] in {"train", "training_val", "calibration_pool", "debug", "selection"}]
    for field in ("image_sha256", "prefix_proxy", "isolation_component"):
        assignments: dict[str, set[str]] = defaultdict(set)
        for row in used:
            assignments[row[field]].add(row["assigned_split"])
        if any(len(v) > 1 for v in assignments.values()):
            raise PreparationError(f"Internal leakage assertion failed: {field}")
    return {"status": "PREPARATION_PROPOSAL_NOT_RESEARCH_FREEZE", "policy": policy.__dict__,
        "grouping": "first_underscore_filename_prefix_plus_exact_content_connected_components",
        "prefix_is_verified_scene_identity": False,
        "near_duplicates_checked": False,
        "near_duplicate_and_true_scene_review_required_before_contract_freeze": True,
        "counts": dict(Counter(r["assigned_split"] for r in rows)),
        "unique_content_counts": dict(Counter(r["assigned_split"] for r in rows if r["is_content_canonical"])),
        "component_counts": {label: len({r["isolation_component"] for r in rows if r["assigned_split"] == label})
                             for label in sorted({r["assigned_split"] for r in rows})},
        "cross_train_val_collisions": collisions, "content_duplicates": duplicates,
        "independence_claim_status": "NOT_ESTABLISHED_PENDING_TRUE_SCENE_AND_NEAR_DUPLICATE_REVIEW",
        "cross_split_collision_review_required": bool(collisions),
        "calibration_image_ids": calibration,
        "calibration_sets_pairwise_disjoint_by_image_sha256": True,
        "selection_is_official_val_unscored": True,
        "selection_annotation_conflicts_require_manual_review": any(
            r["source_split"] == "val" and "duplicate_image_annotation_bytes_conflict" in r["quarantine_reasons"] for r in rows),
        "holdout_opened": False}


def prepare(train_zip: Path, val_zip: Path, output: Path, policy: SplitPolicy = SplitPolicy()) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite any existing output: {output}")
    train_zip, val_zip, output = train_zip.resolve(), val_zip.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite any existing output: {output}")
    if train_zip == val_zip:
        raise PreparationError("Train and val archives must be different files")
    for path in (train_zip, val_zip):
        lowered = path.name.casefold()
        if any(marker in lowered for marker in ("test", "holdout", "partial")):
            raise PreparationError("Only complete train/val archives are accepted; holdout/test paths are prohibited")
        if not path.is_file():
            raise FileNotFoundError(path)
        if output == path or output in path.parents:
            raise PreparationError("Output cannot contain either source archive")
    fingerprints = {split: {"path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path)}
                    for split, path in (("train", train_zip), ("val", val_zip))}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / (f".{output.name}.preparing-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        rows = []
        split_counts = {}
        for split, path in (("train", train_zip), ("val", val_zip)):
            split_rows, split_counts[split] = extract_split(path, split, stage)
            rows.extend(split_rows)
        proposal = split_records(rows, policy)
        rows.sort(key=lambda r: r["image_id"])
        splits_dir = stage / "splits"
        splits_dir.mkdir()
        for label in ("train", "training_val", "calibration_pool", "debug", "selection", "quarantine", "duplicate_excluded"):
            paths = [str(output / r["image_path"]).replace("\\", "/") for r in rows if r["assigned_split"] == label]
            (splits_dir / f"{label}.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8", newline="\n")
        by_id = {r["image_id"]: r for r in rows}
        for seed, ids in proposal["calibration_image_ids"].items():
            paths = [str(output / by_id[key]["image_path"]).replace("\\", "/") for key in ids]
            (splits_dir / f"calibration_seed_{seed}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8", newline="\n")
        # JSON quoted strings are valid YAML, including Unicode Windows paths.
        yaml_lines = [f"path: {json.dumps(str(output).replace(chr(92), '/'), ensure_ascii=False)}",
                      'train: "splits/train.txt"', 'val: "splits/training_val.txt"', "names:"]
        yaml_lines.extend(f"  {index}: {json.dumps(name)}" for index, name in enumerate(NAMES))
        (stage / "training.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8", newline="\n")
        write_jsonl(stage / "image_manifest.jsonl", rows)
        write_json(stage / "split_proposal.json", proposal)
        for split, path in (("train", train_zip), ("val", val_zip)):
            if digest_file(path) != fingerprints[split]["sha256"]:
                raise PreparationError(f"Source archive changed during preparation: {split}")
        summary = {"schema": "rampq.gate1.visdrone_preparation/v1", "status": "PREPARED_NOT_FROZEN_NOT_TRAINED_NOT_SCORED",
            "preparation_script_sha256": digest_file(Path(__file__)),
            "source_archives": fingerprints, "sources_unchanged_after_preparation": True,
            "source_image_and_annotation_bytes_preserved": True,
            "source_roots": {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val"},
            "class_mapping": {str(i + 1): {"class_id": i, "name": name} for i, name in enumerate(NAMES)},
            "valid_label_rule": "score == 1 and category in 1..10 and nonempty clipped box",
            "ignored_rule": "score == 0 or category in {0,11} or empty_after_clipping",
            "physical_line_ids": "image_id:zero_based_original_annotation_physical_line",
            "box_policy": "preserve originals; clip xyxy to original image bounds; report all clipped boxes",
            "split_annotation_counts": split_counts, "split_counts": proposal["counts"],
            "ignored_region_sidecars_are_not_an_evaluator": True,
            "plain_yolo_training_ignores_ignored_regions_as_positive_labels_but_does_not_mask_negative_loss": True,
            "official_val_is_selection_only": True, "holdout_opened": False,
            "no_download_no_model_execution_no_training_no_scoring": True,
            "remaining_before_research_freeze": ["review prefix proxy versus true scene identities and near duplicates",
                "review train/val collision quarantine and annotation conflicts", "freeze ignore-aware target/AP evaluation",
                "freeze training/baseline/precision and research acceptance contracts"]}
        (stage / "README.md").write_text(
            "# VisDrone Gate1 preparation\n\nStatus: PREPARED_NOT_FROZEN_NOT_TRAINED_NOT_SCORED.\n\n"
            "Only official train/val supplied archives were opened. Images and original annotations remain byte-exact. "
            "The official val is unscored Selection. training.yaml validates only on reserved training_val.\n\n"
            "splits are a deterministic proposal using filename-prefix/exact-content connected components. "
            "Prefixes are proxies, not proven scene identities. Near-duplicate/scene review remains required. "
            "Cross train/val connected train groups are quarantined. Exact image duplicates are recorded and excluded "
            "from training reservations. Official val records are retained; review any duplicate annotation conflicts.\n\n"
            "YOLO labels include only score=1, original classes 1..10 mapped to 0..9, and nonempty clipped boxes. "
            "All annotation rows retain original physical line IDs in targets_*.jsonl; ignored rows additionally "
            "appear in ignored_regions_*.jsonl. The sidecars do not implement ignore-aware evaluation. "
            "Plain YOLO training still treats ignored areas as background for negative loss; this limitation "
            "must be stated or explicitly addressed before claiming an official ignore-aware baseline.\n",
            encoding="utf-8", newline="\n")
        write_json(stage / "preparation_summary.json", summary)
        files = []
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size,
                              "sha256": digest_file(path)})
        write_json(stage / "manifest.json", {"schema": "rampq.gate1.preparation_manifest/v1", "files": files})
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation; refusing replacement: {output}")
        stage.rename(output)
        return summary
    except Exception as error:
        # Preserve partial artifacts for diagnosis rather than silently reusing or deleting them.
        (stage / "PREPARATION_FAILED.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--val-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = prepare(args.train_zip, args.val_zip, args.output)
    except (PreparationError, FileExistsError, FileNotFoundError, zipfile.BadZipFile) as error:
        parser.exit(2, f"PREPARATION_REFUSED: {error}\n")
    print(json.dumps({"status": summary["status"], "output": str(args.output.resolve()),
                      "split_counts": summary["split_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
