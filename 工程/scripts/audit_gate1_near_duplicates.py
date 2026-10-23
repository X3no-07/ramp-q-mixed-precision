"""Read-only train/val near-duplicate screening; no model, scoring, or split edits.

Only image paths enumerated by a verified prepared image_manifest are read.
Radius-four dHash64 OR pHash64 candidates are a review queue, not proof of
duplicate content, shared scene identity, or independence of unflagged images.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import time
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageOps, __version__ as PILLOW_VERSION

RADIUS = 4
GRAY_SIZE = 64
COLOR_SIZE = 16
PHASH_INPUT_SIZE = 32
MAX_CONTACT_PAIRS = 48
PAIRS_PER_SHEET = 8
MAX_CONTACT_PAIRS_PER_PREFIX_EDGE = 2


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64) + 0.5
    frequencies = np.arange(size, dtype=np.float64)[:, None]
    matrix = np.cos((math.pi / size) * frequencies * positions) * math.sqrt(2.0 / size)
    matrix[0] /= math.sqrt(2.0)
    return matrix


DCT32 = dct_matrix(PHASH_INPUT_SIZE)


def bits_to_int(bits: np.ndarray) -> int:
    result = 0
    for bit in bits.ravel(order="C"):
        result = (result << 1) | bool(bit)
    return result


def describe_image(image: Image.Image) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Independent spatial-gradient and low-frequency DCT hashes."""
    rgb_image = image.convert("RGB")
    gray_image = rgb_image.convert("L")
    small_dhash = np.asarray(gray_image.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8)
    dhash = bits_to_int(small_dhash[:, 1:] > small_dhash[:, :-1])
    small_phash = np.asarray(gray_image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low_dct = (DCT32 @ small_phash @ DCT32.T)[:8, :8]
    # Remove round-off around analytic zero, so constant images do not hash
    # numerical cosine noise. Threshold is in normalized DCT pixel units.
    low_dct[np.abs(low_dct) < 1e-10] = 0.0
    phash = bits_to_int(low_dct > float(np.median(low_dct)))
    gray = np.asarray(gray_image.resize((GRAY_SIZE, GRAY_SIZE), Image.Resampling.LANCZOS), dtype=np.uint8)
    rgb = np.asarray(rgb_image.resize((COLOR_SIZE, COLOR_SIZE), Image.Resampling.LANCZOS), dtype=np.uint8)
    descriptor = {"dhash64": f"{dhash:016x}", "phash64": f"{phash:016x}",
        "gray_mean_0_1": float(gray.mean() / 255.0), "gray_std_0_1": float(gray.std() / 255.0),
        "mean_rgb_0_1": (rgb.mean(axis=(0, 1)) / 255.0).tolist(),
        "std_rgb_0_1": (rgb.std(axis=(0, 1)) / 255.0).tolist()}
    return descriptor, gray, rgb


class BKTree:
    """Exact Hamming-radius index. Repeated hash values retain all indices."""
    def __init__(self) -> None:
        self.root: list[Any] | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value: int, radius: int) -> list[int]:
        if self.root is None:
            return []
        found = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                found.extend(node[1])
            for edge, child in node[2].items():
                if distance - radius <= edge <= distance + radius:
                    pending.append(child)
        return found


def candidate_pairs(descriptors: list[dict[str, Any]]) -> Iterator[tuple[int, int]]:
    dhash_tree, phash_tree = BKTree(), BKTree()
    for current, row in enumerate(descriptors):
        dhash, phash = int(row["dhash64"], 16), int(row["phash64"], 16)
        found = set(dhash_tree.query(dhash, RADIUS)) | set(phash_tree.query(phash, RADIUS))
        for previous in sorted(found):
            yield previous, current
        dhash_tree.add(dhash, current)
        phash_tree.add(phash, current)


def pair_metrics(gray_a: np.ndarray, gray_b: np.ndarray, rgb_a: np.ndarray, rgb_b: np.ndarray) -> dict[str, Any]:
    a, b = gray_a.astype(np.float64) / 255.0, gray_b.astype(np.float64) / 255.0
    centered_a, centered_b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    correlation = float(np.clip(np.sum(centered_a * centered_b) / denominator, -1.0, 1.0)) if denominator > 1e-12 else None
    difference = a - b
    color_a, color_b = rgb_a.astype(np.float64) / 255.0, rgb_b.astype(np.float64) / 255.0
    return {"gray_64_rmse_0_1": float(np.sqrt(np.mean(difference ** 2))),
            "gray_64_mae_0_1": float(np.mean(np.abs(difference))),
            "gray_64_centered_correlation": correlation,
            "rgb_16_rmse_0_1": float(np.sqrt(np.mean((color_a - color_b) ** 2))),
            "mean_rgb_absolute_difference_0_1": float(np.mean(np.abs(color_a.mean(axis=(0, 1)) - color_b.mean(axis=(0, 1))))),
            "minimum_gray_std_0_1": min(float(a.std()), float(b.std()))}


def validate_rows(root: Path, rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], Path]]:
    seen_ids, seen_paths = set(), set()
    result = []
    for row in sorted(rows, key=lambda r: r["image_id"]):
        if row.get("source_split") not in {"train", "val"}:
            raise ValueError("Only manifest train/val image rows may be accessed")
        rel = row["image_path"]
        parts = PurePosixPath(rel).parts
        if ("\\" in rel or ":" in rel or parts != ("images", row["source_split"], row["filename"])
                or any(part in {".", ".."} for part in parts)):
            raise ValueError(f"Image path does not match the allowed train/val schema: {rel}")
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Image path resolves outside the prepared root")
        if row["image_id"] in seen_ids or path in seen_paths:
            raise ValueError("Duplicate image identity/path in the source inventory")
        seen_ids.add(row["image_id"])
        seen_paths.add(path)
        result.append((row, path))
    return result


def method_contract(root: Path, source_manifest_sha: str, image_manifest_sha: str, script_sha: str) -> dict[str, Any]:
    return {"schema": "rampq.gate1.near_duplicate_screen/v1", "status": "SCREENING_ONLY_NO_SPLIT_EDITS",
        "prepared_root": str(root), "source_manifest_sha256": source_manifest_sha,
        "source_image_manifest_sha256": image_manifest_sha, "script_sha256": script_sha,
        "input_boundary": "only listed images/train and images/val; no archive enumeration or holdout access",
        "hashes": {
            "dhash64": "RGB->Pillow L; LANCZOS 9x8; 64 horizontal right>left bits, row-major MSB first",
            "phash64": "RGB->Pillow L; LANCZOS 32x32; NumPy orthonormal DCT-II; top-left 8x8 including DC; abs coefficient<1e-10 set to 0; greater-than median 64 bits row-major MSB first"},
        "candidate_rule": "dHash Hamming<=4 OR pHash Hamming<=4; exact BK-tree union, no approximate nearest-neighbor truncation",
        "saved_pair_scope": "at least one of old assigned role, filename prefix, or old isolation component differs",
        "descriptors": "LANCZOS grayscale64x64 and RGB16x16 normalized by255; centered grayscale correlation, raw intensity RMSE/MAE and RGB color differences are descriptive",
        "high_similarity_contact_priority": {"minimum_gray_centered_correlation": 0.95, "maximum_rgb16_rmse": 0.10,
            "maximum_other_hash_hamming": 10, "maximum_absolute_log_aspect_ratio_difference": 0.05,
            "minimum_gray_std": 0.02, "cross_prefix_required": True,
            "meaning": "only queue ordering for visual review; never an automatic duplicate/scene decision"},
        "contact_sheets": {"max_pairs": MAX_CONTACT_PAIRS, "pairs_per_sheet": PAIRS_PER_SHEET,
                           "max_pairs_per_prefix_edge": MAX_CONTACT_PAIRS_PER_PREFIX_EDGE},
        "exact_duplicate_definition": "identical original compressed image byte SHA256; image content duplicates with re-encoding may not be exact under this definition",
        "limitations": ["Cannot prove absence of near duplicates or shared scenes, including unflagged pairs.",
            "Different crops, perspective, translation, rotation, exposure, or moving objects may escape both hashes.",
            "Low texture/repetitive structures can yield false positives; grayscale hashes lose color information.",
            "Resizing stretches aspect ratios; low-resolution metrics can hide local changes.",
            "Filename prefixes and old components are grouping proxies, not verified scene identities.",
            "No automatic quarantine, split refinement, training, model inference, label inspection, or detection scoring.",
            "No holdout/test-dev access or claim about its relationship to train/val."]}


def create_contact_sheets(root: Path, output: Path, candidates: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {r["image_id"]: r for r in rows}
    ranked = sorted((p for p in candidates if p["cross_prefix"]), key=lambda p: (
        not p["high_similarity_screen"], -(p["gray_64_centered_correlation"] if p["gray_64_centered_correlation"] is not None else -2),
        p["dhash_hamming"] + p["phash_hamming"], p["rgb_16_rmse_0_1"], p["pair_id"]))
    selected, edges = [], Counter()
    for pair in ranked:
        edge = tuple(sorted((pair["a"]["prefix_proxy"], pair["b"]["prefix_proxy"])))
        if edges[edge] >= MAX_CONTACT_PAIRS_PER_PREFIX_EDGE:
            continue
        selected.append(pair)
        edges[edge] += 1
        if len(selected) == MAX_CONTACT_PAIRS:
            break
    indices = []
    thumb_width, thumb_height, row_height, sheet_width = 480, 270, 335, 1000
    for start in range(0, len(selected), PAIRS_PER_SHEET):
        chunk = selected[start:start + PAIRS_PER_SHEET]
        sheet = Image.new("RGB", (sheet_width, 45 + row_height * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), "TRAIN/VAL SCREENING ONLY | visual review required | no split edits / no holdout", fill="black")
        sheet_index = start // PAIRS_PER_SHEET + 1
        sheet_name = f"contact_sheet_{sheet_index:02d}.jpg"
        for offset, pair in enumerate(chunk):
            y = 40 + offset * row_height
            corr = pair["gray_64_centered_correlation"]
            label = f"{pair['pair_id']} d={pair['dhash_hamming']} p={pair['phash_hamming']} corr={corr if corr is None else round(corr, 4)} RGB-RMSE={pair['rgb_16_rmse_0_1']:.4f} high_screen={pair['high_similarity_screen']}"
            draw.text((10, y), label, fill="black")
            for side, x in (("a", 10), ("b", 510)):
                item = pair[side]
                row = by_id[item["image_id"]]
                source = root / row["image_path"]
                data = source.read_bytes()
                if sha_bytes(data) != row["image_sha256"]:
                    raise ValueError("Image changed before contact-sheet creation")
                with Image.open(io.BytesIO(data)) as image:
                    thumb = ImageOps.pad(image.convert("RGB"), (thumb_width, thumb_height), color=(32, 32, 32))
                sheet.paste(thumb, (x, y + 20))
                draw.text((x, y + 294), item["image_id"], fill="black")
                draw.text((x, y + 309), f"old role={item['assigned_split']} prefix={item['prefix_proxy']}", fill="black")
            indices.append({"sheet": sheet_name, "row_zero_based": offset, "pair_id": pair["pair_id"],
                            "a": pair["a"], "b": pair["b"], "high_similarity_screen": pair["high_similarity_screen"],
                            "review_decision": "PENDING_HUMAN_VISUAL_REVIEW"})
        sheet.save(output / sheet_name, quality=88, subsampling=0)
    write_json(output / "contact_sheet_index.json", indices)
    return indices


def run_audit(prepared_root: Path, output: Path, *, expected_images: int = 7019) -> dict[str, Any]:
    started = time.perf_counter()
    root = prepared_root.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing audit output")
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Audit output and preserved source must be separate directories")
    source_manifest_path = root / "manifest.json"
    image_manifest_path = root / "image_manifest.jsonl"
    source_sha, image_sha, script_sha = sha_file(source_manifest_path), sha_file(image_manifest_path), sha_file(Path(__file__))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    inventory_entry = [e for e in source_manifest["files"] if e["path"] == "image_manifest.jsonl"]
    if len(inventory_entry) != 1 or inventory_entry[0]["sha256"] != image_sha or inventory_entry[0]["bytes"] != image_manifest_path.stat().st_size:
        raise ValueError("Source image inventory does not match its original manifest")
    rows = [json.loads(line) for line in image_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_images:
        raise ValueError(f"Expected {expected_images} image rows; found {len(rows)}")
    validated = validate_rows(root, rows)
    rows = [r for r, _ in validated]
    output.mkdir(parents=True)
    contract = method_contract(root, source_sha, image_sha, script_sha)
    write_json(output / "method_contract.json", contract)
    try:
        descriptors, grays, rgbs = [], [], []
        for index, (row, path) in enumerate(validated):
            data = path.read_bytes()
            if sha_bytes(data) != row["image_sha256"]:
                raise ValueError(f"Source image SHA mismatch: {row['image_id']}")
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.size != (row["width"], row["height"]) or image.getexif().get(274) not in {None, 1}:
                    raise ValueError(f"Source image dimensions/orientation changed: {row['image_id']}")
                descriptor, gray, rgb = describe_image(image)
            descriptor.update({k: row[k] for k in ("image_id", "image_path", "image_sha256", "width", "height",
                "source_split", "assigned_split", "prefix_proxy", "isolation_component")})
            descriptors.append(descriptor)
            grays.append(gray)
            rgbs.append(rgb)
            if (index + 1) % 250 == 0 or index + 1 == len(rows):
                print(f"HASHED_IMAGES {index + 1}/{len(rows)}", flush=True)
                write_json(output / "progress.json", {"phase": "image_hashes", "completed": index + 1,
                    "total": len(rows), "elapsed_seconds": time.perf_counter() - started})
        write_jsonl(output / "image_hash_cache.jsonl", descriptors)
        gray_array, rgb_array = np.stack(grays), np.stack(rgbs)
        np.savez_compressed(output / "low_resolution_descriptors.npz", gray64=gray_array, rgb16=rgb_array,
            image_ids=np.asarray([r["image_id"] for r in rows]), source_sha256=np.asarray([r["image_sha256"] for r in rows]))
        exact = defaultdict(list)
        for row in rows:
            exact[row["image_sha256"]].append(row)
        exact_groups = [{"image_sha256": sha, "image_ids": [r["image_id"] for r in bucket],
            "prefixes": sorted({r["prefix_proxy"] for r in bucket}), "old_roles": sorted({r["assigned_split"] for r in bucket}),
            "source_splits": sorted({r["source_split"] for r in bucket}), "count": len(bucket)}
            for sha, bucket in sorted(exact.items()) if len(bucket) > 1]
        write_jsonl(output / "exact_byte_duplicate_groups.jsonl", exact_groups)
        candidates, edges = [], {}
        total_hash_candidates = 0
        for index_a, index_b in candidate_pairs(descriptors):
            total_hash_candidates += 1
            a, b = rows[index_a], rows[index_b]
            cross_role, cross_prefix = a["assigned_split"] != b["assigned_split"], a["prefix_proxy"] != b["prefix_proxy"]
            cross_component = a["isolation_component"] != b["isolation_component"]
            if not (cross_role or cross_prefix or cross_component):
                continue
            metrics = pair_metrics(gray_array[index_a], gray_array[index_b], rgb_array[index_a], rgb_array[index_b])
            dh = (int(descriptors[index_a]["dhash64"], 16) ^ int(descriptors[index_b]["dhash64"], 16)).bit_count()
            ph = (int(descriptors[index_a]["phash64"], 16) ^ int(descriptors[index_b]["phash64"], 16)).bit_count()
            aspect_gap = abs(math.log((a["width"] / a["height"]) / (b["width"] / b["height"])))
            corr = metrics["gray_64_centered_correlation"]
            high = bool(cross_prefix and corr is not None and corr >= 0.95 and metrics["rgb_16_rmse_0_1"] <= 0.10
                and max(dh, ph) <= 10 and aspect_gap <= 0.05 and metrics["minimum_gray_std_0_1"] >= 0.02)
            pair_id = f"PAIR_{len(candidates) + 1:07d}"
            pair = {"pair_id": pair_id, "a": {k: a[k] for k in ("image_id", "assigned_split", "prefix_proxy", "isolation_component", "source_split")},
                "b": {k: b[k] for k in ("image_id", "assigned_split", "prefix_proxy", "isolation_component", "source_split")},
                "dhash_hamming": dh, "phash_hamming": ph, "cross_old_role": cross_role, "cross_prefix": cross_prefix,
                "cross_old_component": cross_component, "cross_official_train_val": a["source_split"] != b["source_split"],
                "exact_image_bytes": a["image_sha256"] == b["image_sha256"], "absolute_log_aspect_ratio_difference": aspect_gap,
                "high_similarity_screen": high, "automatic_duplicate_or_scene_decision": False, **metrics}
            candidates.append(pair)
            key = tuple(sorted(((a["prefix_proxy"], a["isolation_component"], a["assigned_split"]),
                                (b["prefix_proxy"], b["isolation_component"], b["assigned_split"]))))
            if key not in edges:
                edges[key] = {"a_prefix_component_role": list(key[0]), "b_prefix_component_role": list(key[1]),
                    "candidate_pair_count": 0, "high_similarity_pair_count": 0, "minimum_dhash_hamming": dh,
                    "minimum_phash_hamming": ph, "maximum_gray_centered_correlation": corr,
                    "representative_pair_id": pair_id, "screening_edge_only_not_confirmed_scene_link": True}
            edge = edges[key]
            edge["candidate_pair_count"] += 1
            edge["high_similarity_pair_count"] += high
            edge["minimum_dhash_hamming"] = min(edge["minimum_dhash_hamming"], dh)
            edge["minimum_phash_hamming"] = min(edge["minimum_phash_hamming"], ph)
            if corr is not None and (edge["maximum_gray_centered_correlation"] is None or corr > edge["maximum_gray_centered_correlation"]):
                edge["maximum_gray_centered_correlation"] = corr
                edge["representative_pair_id"] = pair_id
        write_jsonl(output / "candidate_pairs.jsonl", candidates)
        write_jsonl(output / "prefix_component_screening_edges.jsonl", [edges[k] for k in sorted(edges)])
        contacts = create_contact_sheets(root, output, candidates, rows)
        # A second complete byte-SHA pass verifies that the read-only audit did
        # not silently inherit later image changes while descriptors were made.
        for index, (row, path) in enumerate(validated):
            if sha_file(path) != row["image_sha256"]:
                raise ValueError(f"Image changed during audit: {row['image_id']}")
            if (index + 1) % 1000 == 0:
                print(f"REVERIFIED_IMAGE_SHA {index + 1}/{len(rows)}", flush=True)
        if sha_file(source_manifest_path) != source_sha or sha_file(image_manifest_path) != image_sha or sha_file(Path(__file__)) != script_sha:
            raise ValueError("Source inventory or audit script changed during execution")
        summary = {"status": "COMPLETE_TRAIN_VAL_SCREENING_PENDING_VISUAL_REVIEW_NO_SPLIT_EDITS", "image_count": len(rows),
            "source_counts": dict(Counter(r["source_split"] for r in rows)), "old_role_counts": dict(Counter(r["assigned_split"] for r in rows)),
            "old_prefix_count": len({r["prefix_proxy"] for r in rows}), "old_component_count": len({r["isolation_component"] for r in rows}),
            "all_undirected_image_pairs": len(rows) * (len(rows) - 1) // 2,
            "radius4_or_hash_candidates_all_groups": total_hash_candidates, "saved_cross_role_prefix_or_component_candidate_pairs": len(candidates),
            "cross_old_role_candidate_pairs": sum(p["cross_old_role"] for p in candidates),
            "cross_prefix_candidate_pairs": sum(p["cross_prefix"] for p in candidates),
            "cross_official_train_val_candidate_pairs": sum(p["cross_official_train_val"] for p in candidates),
            "high_similarity_screen_pairs": sum(p["high_similarity_screen"] for p in candidates),
            "prefix_component_screening_edges": len(edges), "exact_byte_duplicate_groups": len(exact_groups),
            "exact_byte_duplicate_images": sum(g["count"] for g in exact_groups), "contact_sheet_pairs": len(contacts),
            "contact_sheet_count": (len(contacts) + PAIRS_PER_SHEET - 1) // PAIRS_PER_SHEET,
            "all_source_image_bytes_verified_before_and_after": True,
            "source_manifest_sha256": source_sha, "source_image_manifest_sha256": image_sha, "script_sha256": script_sha,
            "method_contract_sha256": sha_file(output / "method_contract.json"),
            "holdout_opened_or_listed": False, "annotations_opened": False, "model_executed": False,
            "split_edits": False, "automatic_quarantine": False, "no_near_duplicate_or_shared_scene_absence_claim": True,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PILLOW_VERSION},
            "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "run_summary.json", summary)
        write_json(output / "progress.json", {"phase": "COMPLETE", "completed": len(rows), "total": len(rows)})
        (output / "README.md").write_text(
            "# Gate1 train/val near-duplicate screening\n\n"
            "This output is a visual-review queue. Both radius-four hash screens are exhaustive within the "
            "listed 7019 train/val images, but they cannot prove absence of near duplicates/shared scenes. "
            "No detector, annotations, holdout, split modification, or automatic quarantine was used.\n\n"
            "candidate_pairs.jsonl stores only cross-old-role/prefix/component candidates. Hashes, standardized "
            "low-resolution grayscale correlation and color/intensity differences are descriptive. "
            "prefix_component_screening_edges.jsonl contains candidate links, not confirmed scene identities. "
            "Inspect contact sheets and retain explicit visual decisions before using any links to refine splits. "
            "Original SHA checks were repeated after the audit. Method details and limitations are in method_contract.json.\n",
            encoding="utf-8")
        files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha_file(p)}
                 for p in sorted(output.iterdir()) if p.is_file()]
        write_json(output / "audit_manifest.json", {"schema": "rampq.gate1.near_duplicate_audit_manifest/v1", "files": files})
        return summary
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "type": type(error).__name__, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_audit(args.prepared_root, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
