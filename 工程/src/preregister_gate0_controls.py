"""Freeze Gate 0 same-budget controls without running new inference."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any, Sequence


GROUPS = tuple(f"G{i}" for i in range(1, 9))
SEED = 20260910
BUDGET_CAP = 0.25
BUDGET_TOLERANCE = 0.02
RANDOM_CONTROL_COUNT = 3
ALREADY_EVALUATED_PAIRS = {("G1", "G3"), ("G1", "G5")}


def eligible_pairs(
    mac_shares: dict[str, float],
    target_budget: float,
) -> list[tuple[str, str, float]]:
    """Return unseen two-group controls within the frozen compute window."""
    pairs: list[tuple[str, str, float]] = []
    for first, second in itertools.combinations(GROUPS, 2):
        pair = (first, second)
        share = mac_shares[first] + mac_shares[second]
        if pair in ALREADY_EVALUATED_PAIRS:
            continue
        if share > BUDGET_CAP:
            continue
        if abs(share - target_budget) > BUDGET_TOLERANCE:
            continue
        pairs.append((first, second, share))
    return pairs


def choose_random_controls(
    pairs: list[tuple[str, str, float]],
    seed: int = SEED,
) -> list[tuple[str, str, float]]:
    """Choose controls deterministically so the result cannot be cherry-picked."""
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < RANDOM_CONTROL_COUNT:
        raise ValueError("Fewer than three eligible same-budget random pairs")
    return shuffled[:RANDOM_CONTROL_COUNT]


def greedy_accuracy_sensitivity_baseline(
    modules: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Rank by total q_only FP32-hit degradation, then greedily fill the cap."""
    ranking = sorted(
        modules,
        key=lambda row: (-int(row["q_only_degraded"]), str(row["group"])),
    )
    selected: list[str] = []
    used = 0.0
    for row in ranking:
        share = float(row["mac_share"])
        if used + share <= BUDGET_CAP:
            selected.append(str(row["group"]))
            used += share
    return [str(row["group"]) for row in ranking], selected


def write_once_or_verify(path: Path, content: str) -> None:
    """Refuse to silently rewrite a preregistration after it has been frozen."""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise RuntimeError(f"Frozen preregistration differs from regenerated content: {path}")
        return
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()

    inventory = json.loads(
        (root / "artifacts" / "manifests" / "module_inventory.json").read_text(encoding="utf-8")
    )
    ranking = json.loads(
        (root / "artifacts" / "tables" / "selection_module_ranking_and_pair_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    decision = json.loads(
        (root / "artifacts" / "tables" / "gate0_final_decision.json").read_text(encoding="utf-8")
    )

    mac_shares = {group: float(inventory["groups"][group]["mac_share"]) for group in GROUPS}
    target_budget = mac_shares["G1"] + mac_shares["G5"]
    pair_pool = eligible_pairs(mac_shares, target_budget)
    random_controls = choose_random_controls(pair_pool)
    sensitivity_ranking, sensitivity_selected = greedy_accuracy_sensitivity_baseline(ranking["modules"])
    sensitivity_budget = sum(mac_shares[group] for group in sensitivity_selected)
    candidate_groups = ["G1", "G5"]
    collision = sensitivity_selected == candidate_groups

    result: dict[str, Any] = {
        "schema": "rampq.gate0.same_budget_control_preregistration/v1",
        "status": "FROZEN",
        "registration_timing": (
            "Post-candidate and post-candidate-holdout; random choices are deterministic and outcome-independent, "
            "but this is not a pristine untouched-holdout comparison."
        ),
        "seed": SEED,
        "candidate": {
            "groups": candidate_groups,
            "q_configuration": "q_only:G1+G5",
            "r_configuration": "r_only:G1+G5",
            "mac_share": target_budget,
            "selection_net_rescue": int(decision["selection"]["g1_g5"]["net_rescue"]),
            "selection_two_sided": int(decision["selection"]["g1_g5"]["two_sided"]),
            "selection_stable_retention": float(decision["selection"]["g1_g5"]["stable_retention"]),
        },
        "budget_rule": {
            "target_mac_share": target_budget,
            "absolute_tolerance": BUDGET_TOLERANCE,
            "hard_cap": BUDGET_CAP,
            "eligible_rule": "two groups; unseen pair; abs(pair MAC - candidate MAC) <= 0.02; pair MAC <= 0.25",
        },
        "eligible_unseen_pairs": [
            {"groups": [first, second], "mac_share": share} for first, second, share in pair_pool
        ],
        "random_controls": [
            {
                "order": index,
                "groups": [first, second],
                "q_configuration": f"q_only:{first}+{second}",
                "r_configuration": f"r_only:{first}+{second}",
                "mac_share": share,
            }
            for index, (first, second, share) in enumerate(random_controls, start=1)
        ],
        "primary_random_control_test": {
            "dataset": "selection",
            "primary_metric": "net_rescue_rate = (rescued - collateral_misses) / all_int8_added_miss_count",
            "candidate_pass_rule": (
                "candidate net_rescue_rate >= 1.20 * mean(random control net_rescue_rate); "
                "if random mean <= 0, candidate must be positive and exceed the mean by at least 0.05"
            ),
            "guardrails": [
                "candidate stable_retention >= 0.98",
                "candidate two_sided_rate >= mean(random control two_sided_rate)",
                "candidate net_rescue_rate beats at least two of three random controls",
            ],
            "risk_weighted_metric_status": (
                "EXPLORATORY_ONLY: target-risk weights were not frozen before observing module results, "
                "so they cannot be the primary confirmatory control metric in Gate 0."
            ),
        },
        "supervised_accuracy_sensitivity_baseline": {
            "definition": (
                "Rank groups by the total number of Selection FP32 hits degraded by q_only:Gj; "
                "break ties by group id; greedily add groups while cumulative MAC <= 0.25."
            ),
            "ranking": sensitivity_ranking,
            "selected_groups": sensitivity_selected,
            "selected_mac_share": sensitivity_budget,
            "candidate_collision": collision,
            "interpretation": (
                "COLLISION: this conventional label-using sensitivity baseline selects G1+G5 too, "
                "so current evidence cannot establish that the RAMP-Q selector itself is novel."
                if collision
                else "NO_COLLISION"
            ),
        },
        "label_free_output_reconstruction_baseline": {
            "status": "PREREGISTERED_NOT_COMPUTED",
            "calibration_images": 64,
            "data_source": "the image list already frozen in artifacts/calibration/screen.json",
            "per_group_intervention": "q_only:Gj",
            "model_output": "raw detector output before confidence filtering and NMS",
            "primary_score": (
                "mean over images of sum((raw_q_only - raw_fp32)^2) / "
                "(sum(raw_fp32^2) + 1e-12)"
            ),
            "secondary_score": "mean cosine distance; used only as a diagnostic, not to change the primary ranking",
            "pair_selection": (
                "Among pairs satisfying the same budget rule, choose the pair with maximum sum of the two "
                "individual primary scores; tie-break by lower MAC, then lexicographic group ids."
            ),
            "anti_leakage": (
                "Freeze code and scores before reading any new random-pair q_only/r_only detection results."
            ),
        },
        "execution_order": [
            "Compute and freeze the label-free output-reconstruction ranking.",
            "selection q_only:G4+G8",
            "selection r_only:G4+G8",
            "selection q_only:G3+G8",
            "selection r_only:G3+G8",
            "selection q_only:G2+G8",
            "selection r_only:G2+G8",
            "Summarize all controls without changing pairs or thresholds.",
        ],
        "novelty_status": (
            "NOVELTY_COLLISION_RISK: the mechanism signal is feasible, but selection-method novelty is not "
            "demonstrated because the supervised accuracy-sensitivity baseline returns the same pair."
        ),
        "evidence_boundary": (
            "RUOD-R Selection and PyTorch Q/DQ simulated W8A8 only. No new inference is performed by this script; "
            "no TensorRT, Jetson, latency, energy, VisDrone, or SOTA claim."
        ),
    }

    json_text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output_json = root / "artifacts" / "tables" / "gate0_control_preregistration.json"

    pool_rows = [
        f"| {first}+{second} | {share:.2%} |" for first, second, share in pair_pool
    ]
    random_rows = [
        f"| {index} | {first}+{second} | {share:.2%} | `q_only:{first}+{second}` | `r_only:{first}+{second}` |"
        for index, (first, second, share) in enumerate(random_controls, start=1)
    ]
    module_rows_by_group = {str(row["group"]): row for row in ranking["modules"]}
    sensitivity_rows = [
        f"| {index} | {group} | {int(module_rows_by_group[group]['q_only_degraded'])} | "
        f"{mac_shares[group]:.2%} |"
        for index, group in enumerate(sensitivity_ranking, start=1)
    ]
    report = [
        "# Gate 0 同预算对照预注册",
        "",
        "> 状态：**FROZEN**。本文件只冻结对照，不运行新推理。最重要的预警是：普通的Selection总体精度敏感度排序也选出G1+G5，因此当前不能宣称RAMP-Q的模块选择算法具有独立创新性。",
        "",
        "## 1. 为什么现在必须做对照",
        "",
        "现有实验已经支持‘某些模块量化会选择性破坏目标，并可通过恢复模块救回目标’。但它还没有回答：目标级方法是否比随机挑模块或普通敏感度排序更好。没有同预算对照，论文最多是可行性观察，不能成为新的选择算法。",
        "",
        "## 2. 候选和预算窗口",
        "",
        f"- 已锁定候选：`G1+G5`，FP32 MAC占比代理为{target_budget:.2%}。",
        f"- 硬上限：{BUDGET_CAP:.0%}。",
        f"- 同预算容差：候选上下各{BUDGET_TOLERANCE:.0%}个绝对百分点。",
        "- 随机池只允许两个模块、此前未运行过、并满足预算窗口的组合。",
        "- 已经看过结果的`G1+G5`和`G1+G3`不进入随机池。",
        "",
        "## 3. 合格但未运行的随机池",
        "",
        "| 组合 | MAC占比 |",
        "|---|---:|",
        *pool_rows,
        "",
        f"使用固定种子`{SEED}`对上述排序后的池进行一次确定性洗牌，冻结前三个：",
        "",
        "| 顺序 | 随机对照 | MAC占比 | 破坏侧 | 恢复侧 |",
        "|---:|---|---:|---|---|",
        *random_rows,
        "",
        "不得因第一个结果不好或太强而换组合。",
        "",
        "## 4. 主要随机对照判据",
        "",
        "主要数据集仍是Selection，因为候选holdout已经看过，不能再把它称为对这些新对照完全未触碰的盲测集。",
        "",
        "- 主指标：`净救回率 = (救回数 - 附带漏检数) / 全W8A8新增漏检数`。",
        "- 通过线：G1+G5净救回率至少达到3个随机对照平均值的1.20倍。",
        "- 若随机均值小于等于0，则候选必须为正且至少高出0.05。",
        "- 护栏一：候选稳定保持率至少98%。",
        "- 护栏二：候选双向同目标率不低于随机对照均值。",
        "- 护栏三：候选净救回率至少超过3个随机对照中的2个。",
        "",
        "风险加权救回只能作为探索性分析。原因是弱小/低对比度权重并未在看到单模块结果前冻结，现在把它升为主指标会构成事后选择。",
        "",
        "## 5. 传统有标签精度敏感度基线",
        "",
        "定义：按每个`q_only:Gj`使Selection中FP32命中目标变漏检的总数，从高到低排序；同分按模块编号；再在25% MAC上限内依序贪心加入。",
        "",
        "| 排名 | 模块 | 总体新增漏检 | MAC占比 |",
        "|---:|---|---:|---:|",
        *sensitivity_rows,
        "",
        f"该规则选出：`{'+'.join(sensitivity_selected)}`，MAC占比{sensitivity_budget:.2%}。它与RAMP-Q候选完全相同，记为**COLLISION**。这不是RAMP-Q获胜，而是说明目标级规则目前没有表现出区别于普通有标签敏感度排序的选模结果。",
        "",
        "## 6. 无标签输出重建基线",
        "",
        "在运行上述随机组合前，先用已冻结的64张校准图片完成以下基线：",
        "",
        "1. 分别运行FP32与8个`q_only:Gj`，读取置信度过滤和NMS之前的原始检测输出。",
        "2. 每张图计算归一化均方误差：`sum((Q-F)^2) / (sum(F^2)+1e-12)`。",
        "3. 每个模块以64张图的平均值为主分数；平均余弦距离只做诊断。",
        "4. 在同一预算池中，选择两个模块主分数之和最大的组合；并列时先选MAC更低者，再按模块编号。",
        "5. 分数、排名和所选组合写入文件后，才允许查看新的随机组合检测结果。",
        "",
        "这是一个不使用目标标签的普通输出误差基线，可检验目标级救回信息是否真的提供额外选择价值。",
        "",
        "## 7. 已冻结执行顺序",
        "",
        "1. 计算并冻结无标签输出重建排名。",
        "2. `q_only:G4+G8`。",
        "3. `r_only:G4+G8`。",
        "4. `q_only:G3+G8`。",
        "5. `r_only:G3+G8`。",
        "6. `q_only:G2+G8`。",
        "7. `r_only:G2+G8`。",
        "8. 不改变组合与阈值，统一汇总。",
        "",
        "## 8. 当前结论",
        "",
        "- 机制可行性：仍为`CONDITIONAL GO`。",
        "- 选择算法创新性：`NOVELTY COLLISION RISK`。普通有标签敏感度基线已经与候选碰撞。",
        "- 若无标签基线也选中相同/等效组合，且随机对照不能显示明确优势，应把论文主线转为‘量化失效机制与小目标可靠性’，不再主打新型混合精度搜索算法。",
        "",
        "## 证据边界",
        "",
        "本预注册晚于候选及其holdout结果；随机组合由固定种子、固定预算池产生，虽然不依赖其尚未产生的结果，但这仍不是从实验最初就冻结的完整预注册。当前仍只有RUOD-R上的PyTorch Q/DQ模拟W8A8证据，没有TensorRT、Jetson、延迟、能耗、VisDrone或SOTA证据。",
    ]
    md_text = "\n".join(report) + "\n"
    output_md = root / "09_Gate0_同预算对照预注册.md"

    write_once_or_verify(output_json, json_text)
    write_once_or_verify(output_md, md_text)

    print(json.dumps({
        "status": "FROZEN",
        "random_controls": [f"{a}+{b}" for a, b, _ in random_controls],
        "accuracy_sensitivity_baseline": "+".join(sensitivity_selected),
        "candidate_collision": collision,
        "next_step": "compute label-free output-reconstruction ranking",
        "first_inference_after_that": "q_only:G4+G8",
        "report": str(output_md),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
