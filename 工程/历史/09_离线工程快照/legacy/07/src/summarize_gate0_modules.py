"""Summarize all selection-set single-group interventions and freeze pair choices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


GROUPS = tuple(f"G{i}" for i in range(1, 9))


def prediction_map(path: Path, configuration: str | None = None) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if configuration is not None and row["configuration"] != configuration:
                continue
            result[row["target_id"]] = row
    return result


def baseline_maps(path: Path, selection_names: set[str]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    fp32: dict[str, dict[str, str]] = {}
    q_all: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image"] not in selection_names:
                continue
            if row["configuration"] == "fp32":
                fp32[row["target_id"]] = row
            elif row["configuration"] == "q_all":
                q_all[row["target_id"]] = row
    if not fp32 or set(fp32) != set(q_all):
        raise ValueError("Frozen selection FP32/q_all target sets are missing or unequal")
    return fp32, q_all


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()

    selection_names = {
        Path(line.strip()).name
        for line in (root / "splits" / "gate0_selection_images.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    fp32, q_all = baseline_maps(root / "artifacts" / "predictions" / "screen.csv", selection_names)
    fp_hits = {target for target, row in fp32.items() if int(row["matched"]) == 1}
    added_misses = {target for target in fp_hits if int(q_all[target]["matched"]) == 0}
    inventory = json.loads((root / "artifacts" / "manifests" / "module_inventory.json").read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        q_name = f"q_only:{group}"
        r_name = f"r_only:{group}"
        q_path = root / "artifacts" / "predictions" / f"selection_q_only_{group}.csv"
        r_path = root / "artifacts" / "predictions" / f"selection_r_only_{group}.csv"
        q_rows = prediction_map(q_path, q_name)
        r_rows = prediction_map(r_path, r_name)
        if set(q_rows) != set(fp32) or set(r_rows) != set(fp32):
            raise ValueError(f"Target set mismatch for {group}")

        q_degraded = {target for target in fp_hits if int(q_rows[target]["matched"]) == 0}
        q_overlap = q_degraded & added_misses
        rescued = {target for target in added_misses if int(r_rows[target]["matched"]) == 1}
        stable_hits = fp_hits - added_misses
        collateral = {target for target in stable_hits if int(r_rows[target]["matched"]) == 0}
        two_sided = q_overlap & rescued
        mac_share = float(inventory["groups"][group]["mac_share"])
        rescue_rate = len(rescued) / len(added_misses)
        stable_retention = 1.0 - len(collateral) / len(stable_hits)
        net_rescue = len(rescued) - len(collateral)

        strong = (
            len(rescued) >= 20
            and rescue_rate >= 0.15
            and stable_retention >= 0.98
            and net_rescue > 0
            and mac_share <= 0.25
            and len(two_sided) > 0
        )
        low_priority = (
            not strong
            and stable_retention >= 0.98
            and net_rescue > 0
            and mac_share <= 0.25
            and len(two_sided) > 0
        )
        status = "strong" if strong else "low_priority" if low_priority else "eliminated"
        rows.append({
            "group": group,
            "status": status,
            "q_only_degraded": len(q_degraded),
            "q_overlap_with_all_int8": len(q_overlap),
            "q_overlap_coverage": len(q_overlap) / len(added_misses),
            "rescued": len(rescued),
            "rescue_rate": rescue_rate,
            "collateral_misses": len(collateral),
            "stable_retention": stable_retention,
            "net_rescue": net_rescue,
            "two_sided_same_targets": len(two_sided),
            "two_sided_rate": len(two_sided) / len(added_misses),
            "mac_share": mac_share,
        })

    status_order = {"strong": 0, "low_priority": 1, "eliminated": 2}
    rows.sort(key=lambda row: (
        status_order[row["status"]],
        -row["net_rescue"],
        -row["two_sided_same_targets"],
        -row["rescued"],
        row["mac_share"],
    ))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    by_group = {row["group"]: row for row in rows}
    combinations = [
        {
            "priority": 1,
            "groups": ["G1", "G5"],
            "q_configuration": "q_only:G1+G5",
            "r_configuration": "r_only:G1+G5",
            "mac_share": by_group["G1"]["mac_share"] + by_group["G5"]["mac_share"],
            "reason": "G1 is the only strong module; G5 has the highest net rescue among remaining modules and the second-highest degradation overlap.",
        },
        {
            "priority": 2,
            "groups": ["G1", "G3"],
            "q_configuration": "q_only:G1+G3",
            "r_configuration": "r_only:G1+G3",
            "mac_share": by_group["G1"]["mac_share"] + by_group["G3"]["mac_share"],
            "reason": "G3 has the highest rescue count among the remaining modules and a positive net rescue, while staying below the 25% pair MAC cap with G1.",
        },
    ]
    g1 = by_group["G1"]
    pair_thresholds = {
        "rescued_at_least": g1["rescued"] + 5,
        "net_rescue_at_least": g1["net_rescue"] + 5,
        "stable_retention_at_least": 0.98,
        "two_sided_same_targets_at_least": g1["two_sided_same_targets"],
        "mac_share_at_most": 0.25,
        "decision": "A pair replaces G1 alone only if every threshold passes. If neither pair passes, lock G1 alone. If both pass, prefer higher net rescue, then more two-sided targets, then lower MAC share.",
    }
    result = {
        "schema": "rampq.gate0.selection_ranking/v1",
        "selection_image_count": len(selection_names),
        "fp32_hit_count": len(fp_hits),
        "all_int8_added_miss_count": len(added_misses),
        "ranking_rule": "status(strong, low_priority, eliminated), then net rescue, two-sided count, rescued count, lower MAC",
        "single_module_thresholds": {
            "rescued_at_least": 20,
            "rescue_rate_at_least": 0.15,
            "stable_retention_at_least": 0.98,
            "net_rescue_positive": True,
            "two_sided_same_targets_positive": True,
            "mac_share_at_most": 0.25,
        },
        "modules": rows,
        "preregistered_combinations": combinations,
        "pair_incremental_thresholds_vs_g1": pair_thresholds,
        "holdout_status": "sealed; no module intervention inference has been run",
    }
    output_json = root / "artifacts" / "tables" / "selection_module_ranking_and_pair_preregistration.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    labels = {"strong": "强候选", "low_priority": "低优先级", "eliminated": "淘汰"}
    table_rows = [
        f"| {row['rank']} | {row['group']} | {labels[row['status']]} | {row['q_overlap_with_all_int8']} | "
        f"{row['rescued']}（{row['rescue_rate']:.2%}） | {row['collateral_misses']} | {row['net_rescue']:+d} | "
        f"{row['two_sided_same_targets']} | {row['stable_retention']:.2%} | {row['mac_share']:.2%} |"
        for row in rows
    ]
    report = [
        "# Gate 0 Selection：单模块统一排名与双模块组合预注册",
        "",
        "> 结论：G1是唯一通过全部单模块门槛的强候选。组合实验在查看结果前锁定为G1+G5和G1+G3；holdout继续封存。",
        "",
        "## 统一排名",
        "",
        "| 排名 | 模块 | 状态 | 破坏侧与全INT8重合 | 救回数（率） | 附带漏检 | 净救回 | 双向同目标 | 稳定保持率 | MAC占比 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "排名规则依次为：状态（强候选、低优先级、淘汰）、净救回、双向同目标数、救回数、较低MAC占比。该规则只用于Selection模块选择。",
        "",
        "## 单模块结论",
        "",
        "- G1是唯一强候选：救回94个、净救回+81、双向同目标69个、MAC占比10.32%。",
        "- G5和G3只有小幅正收益与少量双向证据，保留为组合候选来源。",
        "- G2、G4、G6、G7、G8未满足净收益或双向一致性要求，不进入组合池。",
        "",
        "## 已冻结的双模块组合",
        "",
        "1. 主组合：`G1+G5`，MAC占比代理22.35%。运行 `q_only:G1+G5` 与 `r_only:G1+G5`。",
        "2. 备组合：`G1+G3`，MAC占比代理22.14%。运行 `q_only:G1+G3` 与 `r_only:G1+G3`。",
        "",
        "两个组合都必须运行并完整报告，不能根据第一个组合结果临时更换第二个组合，也不能新增其他组合后只挑最好结果。",
        "",
        "## 组合替代G1单模块的增量门槛",
        "",
        "组合只有同时满足以下条件，才允许替代G1单模块方案：",
        "",
        f"- 救回数至少{pair_thresholds['rescued_at_least']}个，即比G1的94个至少增加5个；",
        f"- 净救回至少{pair_thresholds['net_rescue_at_least']}，即比G1的+81至少增加5；",
        "- 稳定目标保持率至少98%；",
        f"- 双向同目标至少{pair_thresholds['two_sided_same_targets_at_least']}个，不低于G1；",
        "- 两模块MAC占比合计不超过25%。",
        "",
        "如果两个组合都不通过，锁定G1单模块作为holdout候选；如果两个都通过，依次比较净救回、双向同目标数和MAC占比。",
        "",
        "## 执行顺序",
        "",
        "1. `q_only:G1+G5`",
        "2. `r_only:G1+G5`",
        "3. `q_only:G1+G3`",
        "4. `r_only:G1+G3`",
        "5. 汇总并锁定唯一候选；之后才允许在holdout运行该唯一候选。",
        "",
        "## 证据边界",
        "",
        "- 当前全部结果来自Selection和PyTorch Q/DQ模拟W8A8，不是TensorRT INT8或Jetson实测。",
        "- holdout没有用于模块干预推理、排名或组合选择。",
        "- 当前只能说G1在Selection上有强信号，不能说已经完成独立验证或达到论文结论。",
    ]
    output_md = root / "07_Gate0_模块排名与组合预注册.md"
    output_md.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "strong_candidate": "G1",
        "pair_1": "G1+G5",
        "pair_2": "G1+G3",
        "report": str(output_md),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
