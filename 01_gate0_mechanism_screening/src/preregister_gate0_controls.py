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
        '# Gate 0 Pre-registration Against Budget Control',
        "",
        '> Status: **FROZEN**. This document only freezes the control, does not run new inference. The most critical warning is: the ordinary Selection also ranks G1+G5 in terms of overall accuracy sensitivity, thus it is currently inappropriate to claim that the module selection algorithm of RAMP-Q possesses independent innovation.',
        "",
        '## 1. Why a control is now mandatory',
        "",
        'Existing experiments support the idea that certain modules may selectively destroy targets, which can be recovered by a recovery module. However, it has not yet answered whether target-level methods outperform random module selection or standard sensitivity ranking. Without a budget-controlled comparison, the paper remains a feasibility observation and cannot serve as a new selection algorithm.',
        "",
        '## 2. Candidate and Budget Window',
        "",
        f"- Locked candidate: `G1+G5`, FP32 MAC share proxy is{target_budget:.2%}。",
        f"- Hard upper bound:{BUDGET_CAP:.0%}。",
        f"- Budget tolerance: candidates within ±{BUDGET_TOLERANCE:.0%}absolute percentage points.",
        '- The random pool only allows combinations of two modules, neither of which has been previously run, and both of which satisfy the budget window.',
        '- Candidates `G1+G5` and `G1+G3` that have already been reviewed are excluded from the random pool.',
        "",
        '## 3. Qualified but Not Executed Random Pool',
        "",
        '| Combination | MAC Share |',
        "|---|---:|",
        *pool_rows,
        "",
        f"Use fixed seed `{SEED}` to deterministically shuffle the above-sorted pool, freeze the first three:",
        "",
        '| Order | Random Control | MAC Share | Destruction Side | Recovery Side |',
        "|---:|---|---:|---|---|",
        *random_rows,
        "",
        'No combination should be changed due to a poor or overly strong first result.',
        "",
        '## 4. Primary Random Control Criterion',
        "",
        'The primary dataset remains Selection, as the candidate holdout has already been seen and cannot be considered a completely untouched blind test set for these new controls.',
        "",
        '- Primary metric: `net_recovery_rate = (recovery_count - collateral_miss_count) / all_W8A8_additional_miss_count`.',
        '- Pass criterion: the G1+G5 net_recovery_rate must be at least 1.20 times the mean of the 3 random controls.',
        'If the random mean is less than or equal to 0, the candidate must be positive and at least exceed 0.05.',
        '- Guardrail one: The candidate stability retention rate must be at least 98%.',
        '- Guardrail two: The candidate bidirectional object-level agreement rate must not be lower than the mean of the random control groups.',
        '- Guardrail three: The candidate net_recovery_rate must exceed 3 randomly selected controls in at least 2 cases.',
        "",
        'Risk-weighted recovery is only for exploratory analysis. The reason is that small/low-contrast weights were not frozen before observing per-module results; promoting them as the primary metric would constitute post-hoc selection.',
        "",
        '## 5. Traditional labeled-accuracy sensitivity baseline',
        "",
        'Define: sort in descending order by the total number of FP32-targeted detections in Selection that become missed due to each `q_only:Gj`, with ties broken by module index; then greedily add one by one within the 25% MAC limit.',
        "",
        '| Rank | Module | Overall additional misses | MAC Share |',
        "|---:|---|---:|---:|",
        *sensitivity_rows,
        "",
        f"This rule selects: `{'+'.join(sensitivity_selected)}` with MAC share{sensitivity_budget:.2%}. It is identical to the RAMP-Q candidate, denoted as **COLLISION**. This is not a victory for RAMP-Q, but rather indicates that the object-level rules currently fail to distinguish outcomes from conventional label-sensitive ranking-based selection.",
        "",
        '## 6. Unlabeled output reconstruction baseline',
        "",
        'Before running any random combinations, complete the following baseline using the pre-frozen 64 calibration images:',
        "",
        '1. Run FP32 and the 8 `q_only:Gj` configurations separately, and read the raw detection outputs before confidence filtering and NMS.',
        '2. Compute the normalized mean squared error: `sum((Q-F)^2) / (sum(F^2)+1e-12)` per image.',
        '3. Each module uses the average across the 64 images as its primary score; normalized cosine distance is used only for diagnostic purposes.',
        '4. Within the same budget pool, select the combination with the highest sum of the two primary scores; in case of ties, prefer the one with lower MAC, then by module ID.',
        '5. After writing down the scores, rankings, and selected combination, new random combination detection results are allowed to be viewed.',
        "",
        'This is a baseline based on output error without target labels, which tests whether target-level recovery information provides additional selection value.',
        "",
        '## 7. Frozen Execution Order',
        "",
        '1. Compute and freeze the unlabeled output reconstruction ranking.',
        "2. `q_only:G4+G8`。",
        "3. `r_only:G4+G8`。",
        "4. `q_only:G3+G8`。",
        "5. `r_only:G3+G8`。",
        "6. `q_only:G2+G8`。",
        "7. `r_only:G2+G8`。",
        '8. Do not change the combination or threshold; unify the aggregation.',
        "",
        '## 8. Current Conclusions',
        "",
        '- Mechanism feasibility: Still `CONDITIONAL GO`.',
        '- Algorithm innovation: `NOVELTY COLLISION RISK`. The conventional labeled-accuracy sensitivity baseline has already collided with the candidate.',
        "- If the unlabeled baseline also selects the same or equivalent combination, and the random controls fail to demonstrate clear superiority, the paper's main line should shift to 'quantization failure mechanisms and small-object reliability', and the novel mixed-precision search algorithm should no longer be the primary focus.",
        "",
        '## Evidence Boundaries',
        "",
        "This preregistration occurs after the candidate and its holdout results; the random combination is generated from a fixed seed and a fixed budget pool, although it does not depend on results that have not yet been produced, it still represents only a partial preregistration frozen at the experiment's inception. Currently, there is only evidence from PyTorch Q/DQ simulations of W8A8 on RUOD-R, with no evidence from TensorRT, Jetson, latency, energy consumption, VisDrone, or state-of-the-art performance.",
    ]
    md_text = "\n".join(report) + "\n"
    output_md = root / "docs/09_gate0_budget_matched_control_preregistration.md"

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
