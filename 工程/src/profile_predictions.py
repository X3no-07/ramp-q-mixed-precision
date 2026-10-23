"""Data-quality and exploratory risk profile for a Gate 0 prediction CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence


REQUIRED = {
    "schema", "configuration", "image", "target_id", "class_id", "area_ratio",
    "local_contrast", "matched", "match_iou", "match_conf", "best_low_iou", "best_low_conf",
}


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def relative_risk(rows: list[dict[str, str]], misses: set[str], field: str, threshold: float) -> dict[str, Any]:
    risky = [row for row in rows if float(row[field]) <= threshold]
    other = [row for row in rows if float(row[field]) > threshold]
    risky_rate = sum(row["target_id"] in misses for row in risky) / len(risky) if risky else 0.0
    other_rate = sum(row["target_id"] in misses for row in other) / len(other) if other else 0.0
    return {
        "field": field,
        "threshold_q25": threshold,
        "risky_count": len(risky),
        "other_count": len(other),
        "risky_miss_rate": risky_rate,
        "other_miss_rate": other_rate,
        "relative_risk": risky_rate / other_rate if other_rate > 0 else None,
    }


def profile(input_csv: Path) -> dict[str, Any]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        rows = list(reader)
    missing_columns = sorted(REQUIRED - columns)
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_config[row["configuration"]].append(row)

    duplicate_keys: dict[str, int] = {}
    target_sets: dict[str, set[str]] = {}
    invalid_rows = Counter()
    for name, config_rows in by_config.items():
        keys = [row["target_id"] for row in config_rows]
        duplicate_keys[name] = len(keys) - len(set(keys))
        target_sets[name] = set(keys)
        for row in config_rows:
            matched = int(row["matched"])
            numeric = {
                key: float(row[key])
                for key in ("area_ratio", "local_contrast", "match_iou", "match_conf", "best_low_iou", "best_low_conf")
            }
            if matched not in {0, 1}:
                invalid_rows["matched_not_binary"] += 1
            if not (0 <= numeric["area_ratio"] <= 1):
                invalid_rows["area_out_of_range"] += 1
            for key in ("local_contrast", "match_iou", "match_conf", "best_low_iou", "best_low_conf"):
                if not (0 <= numeric[key] <= 1):
                    invalid_rows[f"{key}_out_of_range"] += 1
            if matched and (numeric["match_iou"] < 0.5 or numeric["match_conf"] < 0.25):
                invalid_rows["matched_below_rule"] += 1
            if not matched and (numeric["match_iou"] != 0 or numeric["match_conf"] != 0):
                invalid_rows["unmatched_has_primary_assignment"] += 1

    reference_equal = False
    if "fp32" in target_sets and "q_all" in target_sets:
        reference_equal = target_sets["fp32"] == target_sets["q_all"]

    fp_rows = {row["target_id"]: row for row in by_config.get("fp32", [])}
    q_rows = {row["target_id"]: row for row in by_config.get("q_all", [])}
    common = sorted(set(fp_rows) & set(q_rows))
    fp_hits = [fp_rows[target] for target in common if int(fp_rows[target]["matched"]) == 1]
    misses = {
        row["target_id"] for row in fp_hits
        if int(q_rows[row["target_id"]]["matched"]) == 0
    }

    miss_by_image = Counter(fp_rows[target]["image"] for target in misses)
    miss_by_class = Counter(fp_rows[target]["class_id"] for target in misses)
    image_counts = sorted(miss_by_image.values(), reverse=True)
    class_counts = sorted(miss_by_class.values(), reverse=True)
    area_values = [float(row["area_ratio"]) for row in fp_hits]
    contrast_values = [float(row["local_contrast"]) for row in fp_hits]
    area_q25 = quantile(area_values, 0.25)
    contrast_q25 = quantile(contrast_values, 0.25)

    failure_modes = Counter()
    for target in misses:
        qrow = q_rows[target]
        low_iou = float(qrow["best_low_iou"])
        low_conf = float(qrow["best_low_conf"])
        if low_iou >= 0.5 and low_conf < 0.25:
            failure_modes["confidence_threshold"] += 1
        elif low_iou < 0.5 and low_conf >= 0.25:
            failure_modes["localization_or_assignment"] += 1
        elif low_iou < 0.5 and low_conf < 0.25:
            failure_modes["combined_or_no_candidate"] += 1
        else:
            failure_modes["one_to_one_competition"] += 1

    return {
        "schema": "rampq.gate0.data_quality/v1",
        "input": str(input_csv.resolve()),
        "row_count": len(rows),
        "column_count": len(columns),
        "missing_required_columns": missing_columns,
        "configurations": {name: len(config_rows) for name, config_rows in sorted(by_config.items())},
        "duplicate_target_keys_by_configuration": duplicate_keys,
        "fp32_q_all_target_sets_equal": reference_equal,
        "invalid_row_counts": dict(invalid_rows),
        "target_count": len(common),
        "fp32_hit_count": len(fp_hits),
        "int8_added_miss_count": len(misses),
        "int8_added_miss_rate": len(misses) / len(fp_hits) if fp_hits else 0.0,
        "miss_image_count": len(miss_by_image),
        "largest_single_image_share": image_counts[0] / len(misses) if image_counts and misses else 0.0,
        "top_10_image_share": sum(image_counts[:10]) / len(misses) if misses else 0.0,
        "miss_class_count": len(miss_by_class),
        "largest_single_class_share": class_counts[0] / len(misses) if class_counts and misses else 0.0,
        "misses_by_class": dict(sorted(miss_by_class.items())),
        "failure_modes": dict(failure_modes),
        "fp_hit_area_median": median(area_values) if area_values else None,
        "miss_area_median": median(float(fp_rows[target]["area_ratio"]) for target in misses) if misses else None,
        "fp_hit_contrast_median": median(contrast_values) if contrast_values else None,
        "miss_contrast_median": median(float(fp_rows[target]["local_contrast"]) for target in misses) if misses else None,
        "exploratory_relative_risk": {
            "small_area": relative_risk(fp_hits, misses, "area_ratio", area_q25),
            "low_contrast": relative_risk(fp_hits, misses, "local_contrast", contrast_q25),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    blocking = (
        bool(result["missing_required_columns"])
        or any(result["duplicate_target_keys_by_configuration"].values())
        or not result["fp32_q_all_target_sets_equal"]
        or any(result["invalid_row_counts"].values())
    )
    quality = "FAIL" if blocking else "PASS"
    area = result["exploratory_relative_risk"]["small_area"]
    contrast = result["exploratory_relative_risk"]["low_contrast"]
    rr_area = "undefined" if area["relative_risk"] is None else f"{area['relative_risk']:.3f}"
    rr_contrast = "undefined" if contrast["relative_risk"] is None else f"{contrast['relative_risk']:.3f}"
    lines = [
        "# Gate 0 Screen 数据质量检查",
        "",
        f"> 数据质量结论：**{quality}**。本报告检查结果表是否可用于后续模块隔离；风险分层属于探索性描述，不是冻结后的假设检验。",
        "",
        "## 数据集与粒度",
        "",
        f"- 输入：`{result['input']}`",
        f"- 行数：{result['row_count']}；目标数：{result['target_count']}；每行粒度为一个配置下的一个真实目标。",
        f"- FP32 命中：{result['fp32_hit_count']}；全 W8A8 新增漏检：{result['int8_added_miss_count']}（{result['int8_added_miss_rate']:.2%}）。",
        "",
        "## 核心质量检查",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        f"| 必需列缺失 | {result['missing_required_columns']} |",
        f"| 配置内重复目标键 | {result['duplicate_target_keys_by_configuration']} |",
        f"| FP32 与 W8A8 目标集合一致 | {result['fp32_q_all_target_sets_equal']} |",
        f"| 非法值或规则矛盾 | {result['invalid_row_counts']} |",
        "",
        "## 集中性与失效形态",
        "",
        f"- 新增漏检分布在 {result['miss_image_count']} 张图片；单张图片最大占比 {result['largest_single_image_share']:.2%}，前 10 张合计 {result['top_10_image_share']:.2%}。",
        f"- 新增漏检覆盖 {result['miss_class_count']} 个类别；最大单类别占比 {result['largest_single_class_share']:.2%}。",
        f"- 诊断性失效分类：`{result['failure_modes']}`。",
        "",
        "## 探索性风险信号",
        "",
        f"- FP32 命中目标面积中位数：{result['fp_hit_area_median']:.6f}；新增漏检面积中位数：{result['miss_area_median']:.6f}。",
        f"- 面积最低四分位相对风险：{rr_area}。",
        f"- FP32 命中目标局部对比度中位数：{result['fp_hit_contrast_median']:.6f}；新增漏检对比度中位数：{result['miss_contrast_median']:.6f}。",
        f"- 对比度最低四分位相对风险：{rr_contrast}。",
        "",
        "## 使用边界",
        "",
        "- Screen 只证明样本量是否足以进入模块扫描，不能证明某个模块具有因果作用。",
        "- 风险四分位阈值由当前数据产生，属于探索性分析；正式 Gate 的权重公式必须先冻结，再用于独立留出集。",
        "- 模拟 W8A8 不是 TensorRT INT8；这里不能报告 Jetson 延迟、能耗或真实 INT8 精度。",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    result = profile(args.input)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

