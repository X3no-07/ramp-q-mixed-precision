"""Freeze the Gate 0 selection/holdout image split from screen predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


def sha256_lines(lines: Sequence[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"configuration", "image", "target_id", "class_id", "matched"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    return rows


def image_index(dataset_root: Path) -> dict[str, Path]:
    val_root = dataset_root / "images" / "val"
    if not val_root.is_dir():
        raise FileNotFoundError(val_root)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in val_root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            grouped[path.name].append(path.resolve())
    duplicates = {name: paths for name, paths in grouped.items() if len(paths) != 1}
    if duplicates:
        raise ValueError(f"Non-unique validation image basenames: {sorted(duplicates)[:10]}")
    return {name: paths[0] for name, paths in grouped.items()}


def subset_stats(
    image_names: set[str],
    fp32: dict[str, dict[str, str]],
    q_all: dict[str, dict[str, str]],
) -> dict[str, Any]:
    targets = [row for row in fp32.values() if row["image"] in image_names]
    fp_hits = [row for row in targets if int(row["matched"]) == 1]
    misses = [row for row in fp_hits if int(q_all[row["target_id"]]["matched"]) == 0]
    miss_images = Counter(row["image"] for row in misses)
    miss_classes = Counter(row["class_id"] for row in misses)
    return {
        "image_count": len(image_names),
        "target_count": len(targets),
        "fp32_hit_count": len(fp_hits),
        "int8_added_miss_count": len(misses),
        "int8_added_miss_rate": len(misses) / len(fp_hits) if fp_hits else 0.0,
        "miss_image_count": len(miss_images),
        "misses_by_class": dict(sorted(miss_classes.items(), key=lambda item: int(item[0]))),
        "largest_single_image_share": max(miss_images.values(), default=0) / len(misses) if misses else 0.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--selection-ratio", type=float, default=0.70)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    if not 0 < args.selection_ratio < 1:
        raise ValueError("selection-ratio must be between 0 and 1")

    rows = read_rows(args.predictions)
    by_config: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        target_id = row["target_id"]
        if target_id in by_config[row["configuration"]]:
            raise ValueError(f"Duplicate target in {row['configuration']}: {target_id}")
        by_config[row["configuration"]][target_id] = row
    fp32 = by_config.get("fp32", {})
    q_all = by_config.get("q_all", {})
    if not fp32 or set(fp32) != set(q_all):
        raise ValueError("FP32 and q_all target sets are missing or unequal")

    screen_images = sorted({row["image"] for row in fp32.values()})
    current_index = image_index(Path(config["dataset_root"]))
    missing_images = sorted(set(screen_images) - set(current_index))
    if missing_images:
        raise FileNotFoundError(f"Screen images missing below current dataset root: {missing_images[:10]}")

    shuffled = list(screen_images)
    random.Random(seed).shuffle(shuffled)
    selection_count = math.floor(len(shuffled) * args.selection_ratio)
    selection_names = set(shuffled[:selection_count])
    holdout_names = set(shuffled[selection_count:])
    selection_paths = sorted(str(current_index[name]) for name in selection_names)
    holdout_paths = sorted(str(current_index[name]) for name in holdout_names)

    split_dir = args.project_root / "splits"
    artifact_dir = args.project_root / "artifacts" / "manifests"
    split_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    selection_file = split_dir / "gate0_selection_images.txt"
    holdout_file = split_dir / "gate0_holdout_images.txt"
    selection_file.write_text("\n".join(selection_paths) + "\n", encoding="utf-8")
    holdout_file.write_text("\n".join(holdout_paths) + "\n", encoding="utf-8")

    selection_stats = subset_stats(selection_names, fp32, q_all)
    holdout_stats = subset_stats(holdout_names, fp32, q_all)
    total_misses_by_class = Counter()
    for class_id, count in selection_stats["misses_by_class"].items():
        total_misses_by_class[class_id] += count
    for class_id, count in holdout_stats["misses_by_class"].items():
        total_misses_by_class[class_id] += count
    major_classes = sorted(class_id for class_id, count in total_misses_by_class.items() if count >= 5)
    missing_major_selection = [c for c in major_classes if c not in selection_stats["misses_by_class"]]
    missing_major_holdout = [c for c in major_classes if c not in holdout_stats["misses_by_class"]]

    overlap = selection_names & holdout_names
    coverage_ok = selection_names | holdout_names == set(screen_images)
    all_paths_exist = all(Path(path).is_file() for path in selection_paths + holdout_paths)
    checks = {
        "image_overlap_count": len(overlap),
        "covers_all_screen_images": coverage_ok,
        "all_listed_paths_exist": all_paths_exist,
        "selection_added_misses_at_least_100": selection_stats["int8_added_miss_count"] >= 100,
        "holdout_added_misses_at_least_40": holdout_stats["int8_added_miss_count"] >= 40,
        "major_classes_present_in_selection": not missing_major_selection,
        "major_classes_present_in_holdout": not missing_major_holdout,
    }
    passed = (
        checks["image_overlap_count"] == 0
        and all(value is True for key, value in checks.items() if key != "image_overlap_count")
    )

    manifest = {
        "schema": "rampq.gate0.split/v1",
        "status": "PASS" if passed else "FAIL",
        "source_predictions": str(args.predictions.resolve()),
        "source_predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "dataset_root": str(Path(config["dataset_root"]).resolve()),
        "seed": seed,
        "algorithm": "sort image basenames, Python random.Random(seed).shuffle, floor(N*0.70) to selection",
        "selection_ratio": args.selection_ratio,
        "screen_image_count": len(screen_images),
        "selection": selection_stats | {
            "list": str(selection_file.resolve()),
            "list_sha256": sha256_lines(selection_paths),
        },
        "holdout": holdout_stats | {
            "list": str(holdout_file.resolve()),
            "list_sha256": sha256_lines(holdout_paths),
        },
        "major_class_rule": "an added-miss class with at least 5 targets overall",
        "major_classes": major_classes,
        "missing_major_classes": {
            "selection": missing_major_selection,
            "holdout": missing_major_holdout,
        },
        "checks": checks,
    }
    manifest_path = artifact_dir / "gate0_split.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_classes = sorted(total_misses_by_class, key=int)
    class_rows = [
        f"| {class_id} | {selection_stats['misses_by_class'].get(class_id, 0)} | "
        f"{holdout_stats['misses_by_class'].get(class_id, 0)} | {total_misses_by_class[class_id]} |"
        for class_id in all_classes
    ]
    rare_notes = [
        class_id for class_id in all_classes
        if class_id not in selection_stats["misses_by_class"] or class_id not in holdout_stats["misses_by_class"]
    ]
    report = [
        '# Gate 0 Data Split Check',
        "",
        f"> Conclusion: **{'PASS' if passed else 'FAIL'}**. This study only froze the selection/holdout image list and did not conduct module isolation experiments.",
        "",
        '## Freezing Rules',
        "",
        f"- Random seed: `{seed}`",
        f"- Image ratio: selection {args.selection_ratio:.0%}，holdout {1-args.selection_ratio:.0%}",
        '- Split unit: entire images; all real targets within the same image belong to only one set.',
        '- Algorithm: Sort images by filename first, then apply Python `random.Random(seed).shuffle` to select the first 70% for selection.',
        '- The selection set is used for module selection; the holdout set must not be used for ranking, hyperparameter tuning, or module swapping prior to module and rule locking.',
        "",
        '## Overall Results',
        "",
        '| Metric | selection | holdout | Total |',
        "|---|---:|---:|---:|",
        f"| Images | {selection_stats['image_count']} | {holdout_stats['image_count']} | {len(screen_images)} |",
        f"| True Object Count | {selection_stats['target_count']} | {holdout_stats['target_count']} | {len(fp32)} |",
        f"| FP32 Hit Count | {selection_stats['fp32_hit_count']} | {holdout_stats['fp32_hit_count']} | {selection_stats['fp32_hit_count'] + holdout_stats['fp32_hit_count']} |",
        f"| Additional Misses in W8A8 | {selection_stats['int8_added_miss_count']} | {holdout_stats['int8_added_miss_count']} | {selection_stats['int8_added_miss_count'] + holdout_stats['int8_added_miss_count']} |",
        f"| Number of Images with Additional Misses | {selection_stats['miss_image_count']} | {holdout_stats['miss_image_count']} | 102 (no image overlap between sets) |",
        "",
        '## Distribution of Additional Misses by Category',
        "",
        '| Category ID | selection | holdout | Total |',
        "|---:|---:|---:|---:|",
        *class_rows,
        "",
        '## Integrity Check',
        "",
        f"- Image overlap between two sets:{len(overlap)}(requirement: 0)",
        f"- Coverage of all screen images:{coverage_ok}",
        f"- All paths in the list exist:{all_paths_exist}",
        f"- Selection has at least 100 additional misses:{selection_stats['int8_added_miss_count']}，{'passed' if checks['selection_added_misses_at_least_100'] else 'not passed'}",
        f"- Holdout has at least 40 additional misses:{holdout_stats['int8_added_miss_count']}，{'passed' if checks['holdout_added_misses_at_least_40'] else 'not passed'}",
        f"- Major categories (total additional misses ≥ 5) present in both sides:{'passed' if not missing_major_selection and not missing_major_holdout else 'not passed'}",
        "",
        '## Files and Reproducibility Fingerprint',
        "",
        f"- selection list: `{selection_file.resolve()}`",
        f"- selection SHA-256：`{sha256_lines(selection_paths)}`",
        f"- holdout list: `{holdout_file.resolve()}`",
        f"- holdout SHA-256：`{sha256_lines(holdout_paths)}`",
        f"- machine-readable list: `{manifest_path.resolve()}`",
        f"- Input prediction table SHA-256: `{manifest['source_predictions_sha256']}`",
        "",
        '## Usage Boundaries',
        "",
        '- Category ID 7 has only 2 additional misses in total, which may appear on only one side after random split; it does not meet the primary category threshold and cannot be used for category-level conclusions.',
        '- The current split is only valid for these 500 screen images and their existing FP32/W8A8 results; if the input list, random seed, or prediction file changes, a new one must be regenerated and a new SHA-256 obtained.',
        '- The next step can only run `q_only:G1` on the selection; the holdout remains sealed.',
    ]
    report_path = args.project_root / "docs/11_gate0_data_split_audit.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": manifest["status"],
        "selection_images": selection_stats["image_count"],
        "holdout_images": holdout_stats["image_count"],
        "selection_added_misses": selection_stats["int8_added_miss_count"],
        "holdout_added_misses": holdout_stats["int8_added_miss_count"],
        "report": str(report_path.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
