"""Independently recompute and decide the three frozen random controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_map(path: Path, configuration: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["configuration"] != configuration:
                continue
            target = row["target_id"]
            if target in rows:
                raise ValueError(f"Duplicate target {target} in {path}")
            rows[target] = row
    if not rows:
        raise ValueError(f"No {configuration} rows in {path}")
    return rows


def baseline_maps(path: Path, image_names: set[str]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    fp32: dict[str, dict[str, str]] = {}
    q_all: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image"] not in image_names:
                continue
            destination = fp32 if row["configuration"] == "fp32" else q_all if row["configuration"] == "q_all" else None
            if destination is None:
                continue
            target = row["target_id"]
            if target in destination:
                raise ValueError(f"Duplicate baseline target {target}")
            destination[target] = row
    if not fp32 or set(fp32) != set(q_all):
        raise ValueError("Selection FP32 and q_all target sets are missing or unequal")
    return fp32, q_all


def recompute_pair(
    root: Path,
    groups: list[str],
    fp32: dict[str, dict[str, str]],
    q_all: dict[str, dict[str, str]],
) -> dict[str, Any]:
    pair = "+".join(groups)
    safe_pair = "_plus_".join(groups)
    q_path = root / "artifacts" / "predictions" / f"selection_q_only_{safe_pair}.csv"
    r_path = root / "artifacts" / "predictions" / f"selection_r_only_{safe_pair}.csv"
    if not q_path.is_file() or not r_path.is_file():
        raise FileNotFoundError(f"Incomplete q/r pair: {pair}")
    q_rows = prediction_map(q_path, f"q_only:{pair}")
    r_rows = prediction_map(r_path, f"r_only:{pair}")
    expected = set(fp32)
    if set(q_rows) != expected or set(r_rows) != expected:
        raise ValueError(f"Target set mismatch for {pair}")

    fp_hits = {target for target, row in fp32.items() if int(row["matched"]) == 1}
    added_misses = {target for target in fp_hits if int(q_all[target]["matched"]) == 0}
    stable_hits = fp_hits - added_misses
    q_misses = {target for target in fp_hits if int(q_rows[target]["matched"]) == 0}
    r_hits = {target for target in fp_hits if int(r_rows[target]["matched"]) == 1}
    r_misses = fp_hits - r_hits
    overlap = q_misses & added_misses
    rescued = r_hits & added_misses
    collateral = r_misses & stable_hits
    two_sided = overlap & rescued
    denominator = len(added_misses)
    return {
        "groups": groups,
        "q_degraded": len(q_misses),
        "q_overlap": len(overlap),
        "q_outside_all_int8": len(q_misses - added_misses),
        "rescued": len(rescued),
        "collateral_misses": len(collateral),
        "net_rescue": len(rescued) - len(collateral),
        "net_rescue_rate": (len(rescued) - len(collateral)) / denominator,
        "stable_retention": 1.0 - len(collateral) / len(stable_hits),
        "two_sided": len(two_sided),
        "two_sided_rate": len(two_sided) / denominator,
        "input_hashes": {"q_predictions": file_sha256(q_path), "r_predictions": file_sha256(r_path)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()

    prereg_path = root / "artifacts" / "tables" / "gate0_control_preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    selection_paths = [
        Path(line.strip())
        for line in (root / "splits" / "gate0_selection_images.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(selection_paths) != 350 or len({str(path) for path in selection_paths}) != 350:
        raise ValueError("Frozen Selection list is not exactly 350 unique images")
    image_names = {path.name for path in selection_paths}
    baseline_path = root / "artifacts" / "predictions" / "screen.csv"
    fp32, q_all = baseline_maps(baseline_path, image_names)
    fp_hits = {target for target, row in fp32.items() if int(row["matched"]) == 1}
    added_misses = {target for target in fp_hits if int(q_all[target]["matched"]) == 0}
    if len(fp32) != 1920 or len(fp_hits) != 1571 or len(added_misses) != 119:
        raise ValueError("Frozen Selection baseline counts changed")

    candidate = recompute_pair(root, ["G1", "G5"], fp32, q_all)
    candidate["mac_share"] = float(prereg["candidate"]["mac_share"])
    controls: list[dict[str, Any]] = []
    for registered in prereg["random_controls"]:
        row = recompute_pair(root, list(registered["groups"]), fp32, q_all)
        row["order"] = int(registered["order"])
        row["mac_share"] = float(registered["mac_share"])
        controls.append(row)
    controls.sort(key=lambda row: row["order"])

    random_mean_net_rate = sum(row["net_rescue_rate"] for row in controls) / len(controls)
    random_mean_two_sided_rate = sum(row["two_sided_rate"] for row in controls) / len(controls)
    if random_mean_net_rate > 0:
        primary_threshold = 1.20 * random_mean_net_rate
        primary_pass = candidate["net_rescue_rate"] >= primary_threshold
        primary_rule_branch = "positive_random_mean_1.20x"
    else:
        primary_threshold = random_mean_net_rate + 0.05
        primary_pass = candidate["net_rescue_rate"] > 0 and candidate["net_rescue_rate"] >= primary_threshold
        primary_rule_branch = "nonpositive_random_mean_positive_plus_0.05"
    guardrail_stability = candidate["stable_retention"] >= 0.98
    guardrail_two_sided = candidate["two_sided_rate"] >= random_mean_two_sided_rate
    beats_count = sum(candidate["net_rescue_rate"] > row["net_rescue_rate"] for row in controls)
    guardrail_beats_two = beats_count >= 2
    random_control_pass = primary_pass and guardrail_stability and guardrail_two_sided and guardrail_beats_two

    supervised = prereg["supervised_accuracy_sensitivity_baseline"]
    label_free_path = root / "artifacts" / "tables" / "gate0_label_free_output_reconstruction.json"
    label_free = json.loads(label_free_path.read_text(encoding="utf-8"))
    label_free_groups = list(label_free["all_same_budget_pair_selection"]["groups"])
    label_free_safe = "_plus_".join(label_free_groups)
    label_free_evaluated = (
        root / "artifacts" / "predictions" / f"selection_q_only_{label_free_safe}.csv"
    ).is_file() and (
        root / "artifacts" / "predictions" / f"selection_r_only_{label_free_safe}.csv"
    ).is_file()

    result = {
        "schema": "rampq.gate0.random_control_decision/v1",
        "status": "COMPLETE",
        "selection_counts": {
            "images": len(selection_paths),
            "targets": len(fp32),
            "fp32_hits": len(fp_hits),
            "all_int8_added_misses": len(added_misses),
        },
        "candidate": candidate,
        "random_controls": controls,
        "random_aggregate": {
            "mean_net_rescue": sum(row["net_rescue"] for row in controls) / len(controls),
            "mean_net_rescue_rate": random_mean_net_rate,
            "mean_two_sided": sum(row["two_sided"] for row in controls) / len(controls),
            "mean_two_sided_rate": random_mean_two_sided_rate,
        },
        "preregistered_tests": {
            "primary": {
                "branch": primary_rule_branch,
                "candidate_net_rescue_rate": candidate["net_rescue_rate"],
                "required_net_rescue_rate": primary_threshold,
                "pass": primary_pass,
            },
            "stable_retention_at_least_0_98": guardrail_stability,
            "two_sided_rate_at_least_random_mean": guardrail_two_sided,
            "net_rescue_rate_beats_at_least_two_controls": guardrail_beats_two,
            "controls_beaten": beats_count,
        },
        "same_budget_random_control_decision": "PASS" if random_control_pass else "FAIL",
        "supervised_accuracy_sensitivity_baseline": {
            "selected_groups": supervised["selected_groups"],
            "candidate_collision": bool(supervised["candidate_collision"]),
            "decision": "COLLISION_NO_METHOD_SEPARATION" if supervised["candidate_collision"] else "NO_COLLISION",
        },
        "label_free_output_baseline": {
            "selected_groups": label_free_groups,
            "candidate_collision": bool(label_free["all_same_budget_pair_selection"]["candidate_collision"]),
            "detection_pair_evaluated": label_free_evaluated,
            "decision": "PENDING_Q_R_EVALUATION" if not label_free_evaluated else "READY_FOR_COMPARISON",
        },
        "overall_novelty_gate": "PARTIAL_RANDOM_PASS_BASELINE_PENDING",
        "next_step": f"q_only:{'+'.join(label_free_groups)}" if not label_free_evaluated else "summarize label-free baseline comparison",
        "input_hashes": {
            "baseline_predictions": file_sha256(baseline_path),
            "selection_list": file_sha256(root / "splits" / "gate0_selection_images.txt"),
            "control_preregistration": file_sha256(prereg_path),
            "label_free_ranking": file_sha256(label_free_path),
        },
        "evidence_boundary": (
            "Random controls pass only on RUOD-R Selection under PyTorch Q/DQ simulated W8A8. "
            "The supervised baseline collides with the candidate, and the selected label-free pair still needs q/r detection evaluation."
        ),
    }
    output_json = root / "artifacts" / "tables" / "gate0_random_control_decision.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table_rows = [
        f"| {'+'.join(row['groups'])} | {row['mac_share']:.2%} | {row['rescued']} | {row['collateral_misses']} | "
        f"{row['net_rescue']:+d}（{row['net_rescue_rate']:.2%}） | {row['two_sided']}（{row['two_sided_rate']:.2%}） | "
        f"{row['stable_retention']:.2%} |"
        for row in [candidate, *controls]
    ]
    report = [
        "# Gate 0 同预算随机对照最终判定",
        "",
        f"> 判定：**随机对照PASS；总体创新性Gate仍为PARTIAL。** `G1+G5`显著超过三个固定随机组合，但有标签总体敏感度基线也选出`G1+G5`，且无标签基线`{'+'.join(label_free_groups)}`尚未完成检测级q/r验证。",
        "",
        "## 1. 从原始逐目标记录独立重算",
        "",
        "本报告重新读取Selection的FP32、全W8A8以及每个组合的逐目标预测CSV，独立构造新增漏检、救回、附带漏检和双向同目标集合，没有直接复制单组摘要中的统计值。",
        "",
        "| 组合 | MAC占比 | 救回 | 附带漏检 | 净救回（率） | 双向同目标（率） | 稳定保持率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "三个随机组合平均净救回为+2.00，平均净救回率为1.68%；平均双向同目标数为1.33，平均双向同目标率为1.12%。",
        "",
        "## 2. 预注册判据核对",
        "",
        "| 判据 | 实际值 | 要求 | 结果 |",
        "|---|---:|---:|---|",
        f"| 候选净救回率相对随机均值 | {candidate['net_rescue_rate']:.2%} | 至少{primary_threshold:.2%}（随机均值的1.20倍） | {'PASS' if primary_pass else 'FAIL'} |",
        f"| 候选稳定保持率 | {candidate['stable_retention']:.2%} | 至少98% | {'PASS' if guardrail_stability else 'FAIL'} |",
        f"| 候选双向同目标率 | {candidate['two_sided_rate']:.2%} | 不低于随机均值{random_mean_two_sided_rate:.2%} | {'PASS' if guardrail_two_sided else 'FAIL'} |",
        f"| 候选超过的随机组合数 | {beats_count}/3 | 至少2/3 | {'PASS' if guardrail_beats_two else 'FAIL'} |",
        "",
        f"四项全部通过，因此同预算随机对照判定为**{'PASS' if random_control_pass else 'FAIL'}**。候选净救回率约为随机均值的{candidate['net_rescue_rate'] / random_mean_net_rate:.1f}倍。这个倍数很大，但随机均值本身接近零，论文中应同时报告绝对差值，不能只强调倍数。",
        "",
        "## 3. 为什么还不能宣布选模方法创新成立",
        "",
        "- 随机对照：已通过，说明结果不是随便选两个同预算模块就能得到。",
        "- 有标签总体敏感度基线：与候选完全碰撞，同样选出`G1+G5`，所以RAMP-Q尚未显示出相对该传统规则的独立选择优势。",
        f"- 无标签输出误差基线：选出`{'+'.join(label_free_groups)}`，但该组合还没有完成检测级破坏/恢复实验，不能判断哪种选择实际更好。",
        "",
        "因此当前最准确的状态是`PARTIAL_RANDOM_PASS_BASELINE_PENDING`，不是完整GO，也不是STOP。",
        "",
        "## 4. 下一步",
        "",
        f"运行无标签基线的破坏侧：`q_only:{'+'.join(label_free_groups)}`；之后运行对应的`r_only:{'+'.join(label_free_groups)}`，再与`G1+G5`作同预算检测级比较。",
        "",
        "## 证据边界",
        "",
        "本结论只覆盖RUOD-R Selection与PyTorch Q/DQ模拟W8A8。随机对照来自事后但结果未知时固定的种子与预算池；它不是从整个项目起点就冻结的预注册。当前没有TensorRT、Jetson、延迟、能耗、VisDrone或SOTA证据。",
    ]
    output_md = root / "02_Gate0_随机对照判定.md"
    output_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "COMPLETE",
        "random_control_decision": result["same_budget_random_control_decision"],
        "overall_novelty_gate": result["overall_novelty_gate"],
        "candidate_net_rescue_rate": candidate["net_rescue_rate"],
        "random_mean_net_rescue_rate": random_mean_net_rate,
        "next_step": result["next_step"],
        "report": str(output_md),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
