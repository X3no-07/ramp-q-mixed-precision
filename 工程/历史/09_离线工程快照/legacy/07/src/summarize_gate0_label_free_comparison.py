"""Compare RAMP-Q with the frozen label-free baseline and close Gate 0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_configuration(path: Path, configuration: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["configuration"] != configuration:
                continue
            if row["target_id"] in rows:
                raise ValueError(f"Duplicate target in {path}: {row['target_id']}")
            rows[row["target_id"]] = row
    if not rows:
        raise ValueError(f"No {configuration} rows in {path}")
    return rows


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()

    image_names = {
        Path(line.strip()).name
        for line in (root / "splits" / "gate0_selection_images.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(image_names) != 350:
        raise ValueError("Selection image count changed")
    screen_path = root / "artifacts" / "predictions" / "screen.csv"
    fp32: dict[str, dict[str, str]] = {}
    q_all: dict[str, dict[str, str]] = {}
    with screen_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image"] not in image_names:
                continue
            if row["configuration"] == "fp32":
                fp32[row["target_id"]] = row
            elif row["configuration"] == "q_all":
                q_all[row["target_id"]] = row
    if set(fp32) != set(q_all) or len(fp32) != 1920:
        raise ValueError("Frozen baseline target set changed")
    fp_hits = {target for target, row in fp32.items() if int(row["matched"]) == 1}
    added_misses = {target for target in fp_hits if int(q_all[target]["matched"]) == 0}
    stable_hits = fp_hits - added_misses
    if len(fp_hits) != 1571 or len(added_misses) != 119:
        raise ValueError("Frozen baseline counts changed")

    label_free_path = root / "artifacts" / "tables" / "gate0_label_free_output_reconstruction.json"
    label_free = json.loads(label_free_path.read_text(encoding="utf-8"))
    baseline_groups = list(label_free["all_same_budget_pair_selection"]["groups"])
    if baseline_groups != ["G1", "G2"]:
        raise ValueError("Frozen label-free baseline pair changed")
    inventory = json.loads(
        (root / "artifacts" / "manifests" / "module_inventory.json").read_text(encoding="utf-8")
    )

    configurations = {
        "candidate": ["G1", "G5"],
        "label_free_baseline": baseline_groups,
    }
    metrics: dict[str, dict[str, Any]] = {}
    target_sets: dict[str, dict[str, set[str]]] = {}
    input_hashes: dict[str, str] = {
        "screen_predictions": file_sha256(screen_path),
        "label_free_ranking": file_sha256(label_free_path),
    }
    for label, groups in configurations.items():
        pair = "+".join(groups)
        safe_pair = "_plus_".join(groups)
        q_path = root / "artifacts" / "predictions" / f"selection_q_only_{safe_pair}.csv"
        r_path = root / "artifacts" / "predictions" / f"selection_r_only_{safe_pair}.csv"
        q_rows = read_configuration(q_path, f"q_only:{pair}")
        r_rows = read_configuration(r_path, f"r_only:{pair}")
        if set(q_rows) != set(fp32) or set(r_rows) != set(fp32):
            raise ValueError(f"Target set mismatch for {pair}")
        q_misses = {target for target in fp_hits if int(q_rows[target]["matched"]) == 0}
        r_hits = {target for target in fp_hits if int(r_rows[target]["matched"]) == 1}
        rescued = added_misses & r_hits
        collateral = stable_hits - r_hits
        two_sided = added_misses & q_misses & r_hits
        mac_share = sum(float(inventory["groups"][group]["mac_share"]) for group in groups)
        metrics[label] = {
            "groups": groups,
            "mac_share": mac_share,
            "q_degraded": len(q_misses),
            "q_overlap": len(q_misses & added_misses),
            "rescued": len(rescued),
            "rescue_rate": len(rescued) / len(added_misses),
            "collateral_misses": len(collateral),
            "net_rescue": len(rescued) - len(collateral),
            "net_rescue_rate": (len(rescued) - len(collateral)) / len(added_misses),
            "stable_retention": 1.0 - len(collateral) / len(stable_hits),
            "two_sided": len(two_sided),
            "two_sided_rate": len(two_sided) / len(added_misses),
        }
        target_sets[label] = {"rescued": rescued, "collateral": collateral, "two_sided": two_sided}
        input_hashes[f"{label}_q_predictions"] = file_sha256(q_path)
        input_hashes[f"{label}_r_predictions"] = file_sha256(r_path)

    candidate = metrics["candidate"]
    baseline = metrics["label_free_baseline"]
    candidate_sets = target_sets["candidate"]
    baseline_sets = target_sets["label_free_baseline"]

    rescue_candidate_only = len(candidate_sets["rescued"] - baseline_sets["rescued"])
    rescue_baseline_only = len(baseline_sets["rescued"] - candidate_sets["rescued"])
    two_candidate_only = len(candidate_sets["two_sided"] - baseline_sets["two_sided"])
    two_baseline_only = len(baseline_sets["two_sided"] - candidate_sets["two_sided"])

    utility_candidate: dict[str, int] = {}
    utility_baseline: dict[str, int] = {}
    for target in fp_hits:
        utility_candidate[target] = (
            1 if target in candidate_sets["rescued"] else -1 if target in candidate_sets["collateral"] else 0
        )
        utility_baseline[target] = (
            1 if target in baseline_sets["rescued"] else -1 if target in baseline_sets["collateral"] else 0
        )
    utility_candidate_better = sum(utility_candidate[t] > utility_baseline[t] for t in fp_hits)
    utility_baseline_better = sum(utility_candidate[t] < utility_baseline[t] for t in fp_hits)
    utility_equal = len(fp_hits) - utility_candidate_better - utility_baseline_better

    comparison = {
        "net_rescue_difference_candidate_minus_baseline": candidate["net_rescue"] - baseline["net_rescue"],
        "net_rescue_rate_difference": candidate["net_rescue_rate"] - baseline["net_rescue_rate"],
        "rescued_difference": candidate["rescued"] - baseline["rescued"],
        "collateral_difference": candidate["collateral_misses"] - baseline["collateral_misses"],
        "two_sided_difference": candidate["two_sided"] - baseline["two_sided"],
        "mac_share_difference": candidate["mac_share"] - baseline["mac_share"],
        "rescue_discordance": {
            "candidate_only": rescue_candidate_only,
            "baseline_only": rescue_baseline_only,
            "exploratory_exact_sign_p_two_sided": exact_two_sided_sign_p(
                rescue_candidate_only, rescue_baseline_only
            ),
        },
        "two_sided_discordance": {
            "candidate_only": two_candidate_only,
            "baseline_only": two_baseline_only,
            "exploratory_exact_sign_p_two_sided": exact_two_sided_sign_p(
                two_candidate_only, two_baseline_only
            ),
        },
        "paired_target_utility": {
            "definition": "+1 added miss rescued; -1 stable hit lost; 0 otherwise",
            "candidate_better": utility_candidate_better,
            "baseline_better": utility_baseline_better,
            "equal": utility_equal,
            "exploratory_exact_sign_p_two_sided": exact_two_sided_sign_p(
                utility_candidate_better, utility_baseline_better
            ),
        },
    }

    result = {
        "schema": "rampq.gate0.label_free_detection_comparison/v1",
        "status": "COMPLETE",
        "selection_counts": {
            "images": len(image_names),
            "targets": len(fp32),
            "fp32_hits": len(fp_hits),
            "all_int8_added_misses": len(added_misses),
        },
        "candidate": candidate,
        "label_free_baseline": baseline,
        "comparison": comparison,
        "descriptive_result": "SMALL_CANDIDATE_ADVANTAGE",
        "confirmatory_result": "NOT_ESTABLISHED",
        "selection_method_novelty_gate": "SUPERIORITY_NOT_ESTABLISHED",
        "mechanism_reliability_gate": "GO",
        "reasons": [
            "Candidate was selected on the same Selection set, so its small advantage is selection-biased.",
            "Paired utility is 19 candidate-better versus 14 baseline-better targets; exploratory two-sided sign p is not small.",
            "The supervised accuracy-sensitivity baseline already selects the identical G1+G5 pair.",
            "No clean independent comparison against G1+G2 exists in Gate 0.",
        ],
        "recommended_positioning": (
            "Continue as a target-level quantization failure mechanism and reliability study. Do not lead with a "
            "novel mixed-precision selector unless a clean preregistered Gate 1 benchmark shows separation from both baselines."
        ),
        "next_step": (
            "Close Gate 0 and preregister Gate 1 on VisDrone with candidate, supervised sensitivity, label-free output "
            "reconstruction, and random controls frozen before the validation split is opened."
        ),
        "input_hashes": input_hashes,
        "evidence_boundary": (
            "RUOD-R Selection and PyTorch Q/DQ simulated W8A8 only. Exact sign tests are exploratory and do not "
            "remove selection bias or establish TensorRT/Jetson performance."
        ),
    }
    output_json = root / "artifacts" / "tables" / "gate0_label_free_detection_comparison.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Gate 0 无标签基线检测级比较与收口决策",
        "",
        "> 决策：**总体方向GO；主线不变；选模优越性未证实。** G1+G5在Selection上描述性优于无标签基线G1+G2，但优势较小、来自用于选候选的同一数据，且有标签敏感度规则与G1+G5完全碰撞，因此不能把Gate 0写成选模算法已经胜出。",
        "",
        "## 1. 同预算检测级结果",
        "",
        "| 方案 | MAC占比 | 救回 | 附带漏检 | 净救回（率） | 双向同目标（率） | 稳定保持率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| RAMP-Q候选 G1+G5 | {candidate['mac_share']:.2%} | {candidate['rescued']} | {candidate['collateral_misses']} | {candidate['net_rescue']:+d}（{candidate['net_rescue_rate']:.2%}） | {candidate['two_sided']}（{candidate['two_sided_rate']:.2%}） | {candidate['stable_retention']:.2%} |",
        f"| 无标签输出误差基线 G1+G2 | {baseline['mac_share']:.2%} | {baseline['rescued']} | {baseline['collateral_misses']} | {baseline['net_rescue']:+d}（{baseline['net_rescue_rate']:.2%}） | {baseline['two_sided']}（{baseline['two_sided_rate']:.2%}） | {baseline['stable_retention']:.2%} |",
        "",
        "G1+G5比G1+G2多救回7个目标，附带漏检多2个，因此净救回多5个；双向同目标多10个，FP32 MAC占比反而低0.62个百分点。G1+G2的优势是少2个附带漏检、稳定保持率高0.14个百分点。",
        "",
        "## 2. 同目标配对检查",
        "",
        "| 配对项目 | G1+G5独有 | G1+G2独有 | 探索性双侧符号检验 |",
        "|---|---:|---:|---:|",
        f"| 救回目标 | {rescue_candidate_only} | {rescue_baseline_only} | p={comparison['rescue_discordance']['exploratory_exact_sign_p_two_sided']:.3f} |",
        f"| 双向同目标 | {two_candidate_only} | {two_baseline_only} | p={comparison['two_sided_discordance']['exploratory_exact_sign_p_two_sided']:.3f} |",
        f"| 逐目标净效用更好 | {utility_candidate_better} | {utility_baseline_better} | p={comparison['paired_target_utility']['exploratory_exact_sign_p_two_sided']:.3f} |",
        "",
        f"逐目标净效用定义为：全W8A8漏检被救回记+1，原本稳定命中却被误伤记-1，其余记0。G1+G5在{utility_candidate_better}个目标上更好，G1+G2在{utility_baseline_better}个目标上更好，另有{utility_equal}个目标相同。",
        "",
        "这些p值只是补充诊断：比较规则没有在最初预注册，而且候选正是在同一Selection上选出来的，所以即使p值更小也不能消除选择偏差。本结果只能称为描述性小优势，不能称为独立显著胜出。",
        "",
        "## 3. Gate 0最终解释",
        "",
        "- 核心机制：GO。模块量化会系统性影响具体目标，恢复G1可在独立holdout复现大规模救回与高稳定保持率。",
        "- 同预算随机对照：PASS。G1+G5明显超过三个冻结随机组合。",
        "- 无标签输出误差基线：G1+G5有小幅描述性优势，但不是独立确认结果。",
        "- 有标签总体敏感度基线：COLLISION。它也选出G1+G5，当前没有选模结果差异可作为算法新意。",
        "- 选模算法优越性：NOT ESTABLISHED。Gate 0不足以支持‘新的选择算法优于传统方法’，但目标级归因与恢复研究主线不变。",
        "- 机制与可靠性主线：GO。可以继续研究哪些弱小目标对哪些模块量化敏感、这种敏感性为何产生、以及能否形成可泛化的可靠性指标。",
        "",
        "## 4. 建议的论文定位",
        "",
        "论文不要把‘自动选出G1+G5’作为唯一核心创新。更稳妥的主线是：目标级量化失效归因、破坏—恢复双向验证、小目标可靠性指标，以及在给定计算预算下的风险控制。混合精度选择保留为应用结果或次要贡献。",
        "",
        "## 5. 下一步",
        "",
        "关闭RUOD-R Gate 0，不再在当前Selection追加组合。进入VisDrone Gate 1前，先冻结候选生成规则、传统有标签敏感度、无标签输出误差、随机对照和独立验证划分；只有在未查看验证结果前完成这些冻结，才能重新检验选模算法是否真正有额外价值。",
        "",
        "## 证据边界",
        "",
        "当前仍是RUOD-R上的PyTorch Q/DQ模拟W8A8。没有TensorRT INT8、Jetson延迟/能耗、VisDrone复现或SOTA比较证据。本报告支持保持目标级归因与可靠恢复主线并进入下一Gate，不等于论文方法已经验证完成。",
    ]
    output_md = root / "01_Gate0_无标签基线比较.md"
    output_md.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE",
        "candidate_net_rescue": candidate["net_rescue"],
        "baseline_net_rescue": baseline["net_rescue"],
        "paired_utility": [utility_candidate_better, utility_baseline_better, utility_equal],
        "paired_utility_p": comparison["paired_target_utility"]["exploratory_exact_sign_p_two_sided"],
        "selection_method_novelty_gate": result["selection_method_novelty_gate"],
        "mechanism_reliability_gate": result["mechanism_reliability_gate"],
        "report": str(output_md),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
