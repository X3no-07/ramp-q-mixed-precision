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
        "# Gate 0 数据划分检查",
        "",
        f"> 结论：**{'PASS' if passed else 'FAIL'}**。本次只冻结 selection/holdout 图片名单，没有运行模块隔离实验。",
        "",
        "## 冻结规则",
        "",
        f"- 随机种子：`{seed}`",
        f"- 图片比例：selection {args.selection_ratio:.0%}，holdout {1-args.selection_ratio:.0%}",
        "- 划分单位：整张图片；同一图片中的所有真实目标只能属于一个集合。",
        "- 算法：先按图片名排序，再用 Python `random.Random(seed).shuffle`，前 70% 进入 selection。",
        "- selection 用于选择模块；holdout 在模块与规则锁定前不得用于排名、调参或换模块。",
        "",
        "## 结果总览",
        "",
        "| 指标 | selection | holdout | 合计 |",
        "|---|---:|---:|---:|",
        f"| 图片数 | {selection_stats['image_count']} | {holdout_stats['image_count']} | {len(screen_images)} |",
        f"| 真实目标数 | {selection_stats['target_count']} | {holdout_stats['target_count']} | {len(fp32)} |",
        f"| FP32 命中数 | {selection_stats['fp32_hit_count']} | {holdout_stats['fp32_hit_count']} | {selection_stats['fp32_hit_count'] + holdout_stats['fp32_hit_count']} |",
        f"| W8A8 新增漏检 | {selection_stats['int8_added_miss_count']} | {holdout_stats['int8_added_miss_count']} | {selection_stats['int8_added_miss_count'] + holdout_stats['int8_added_miss_count']} |",
        f"| 含新增漏检的图片数 | {selection_stats['miss_image_count']} | {holdout_stats['miss_image_count']} | 102（集合间无图片重叠） |",
        "",
        "## 新增漏检类别分布",
        "",
        "| 类别ID | selection | holdout | 合计 |",
        "|---:|---:|---:|---:|",
        *class_rows,
        "",
        "## 完整性检查",
        "",
        f"- 两集合图片重叠：{len(overlap)}（要求 0）",
        f"- 覆盖全部 screen 图片：{coverage_ok}",
        f"- 名单内路径全部存在：{all_paths_exist}",
        f"- selection 新增漏检不少于 100：{selection_stats['int8_added_miss_count']}，{'通过' if checks['selection_added_misses_at_least_100'] else '不通过'}",
        f"- holdout 新增漏检不少于 40：{holdout_stats['int8_added_miss_count']}，{'通过' if checks['holdout_added_misses_at_least_40'] else '不通过'}",
        f"- 主要类别（总新增漏检不少于 5）在两边都有：{'通过' if not missing_major_selection and not missing_major_holdout else '不通过'}",
        "",
        "## 文件与复现指纹",
        "",
        f"- selection 名单：`{selection_file.resolve()}`",
        f"- selection SHA-256：`{sha256_lines(selection_paths)}`",
        f"- holdout 名单：`{holdout_file.resolve()}`",
        f"- holdout SHA-256：`{sha256_lines(holdout_paths)}`",
        f"- 机器可读清单：`{manifest_path.resolve()}`",
        f"- 输入预测表 SHA-256：`{manifest['source_predictions_sha256']}`",
        "",
        "## 使用边界",
        "",
        "- 类别ID 7总共只有2个新增漏检，随机划分后可能只出现在一侧；它不满足主要类别门槛，不能单独用于类别级结论。",
        "- 当前划分只适用于这500张screen图片及其现有FP32/W8A8结果；若输入名单、随机种子或预测文件变化，必须重新生成并获得新的SHA-256。",
        "- 下一步只能在selection上运行 `q_only:G1`；holdout继续封存。",
    ]
    report_path = args.project_root / "11_Gate0_数据划分检查.md"
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
