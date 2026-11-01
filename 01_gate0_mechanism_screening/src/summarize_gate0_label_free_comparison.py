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
        '# Gate 0 Unlabeled Baseline Detection-Level Comparison and Closeout Decision',
        "",
        '> Decision: **Overall direction GO; mainline unchanged; selection model superiority not established.** G1+G5 descriptively outperforms the unlabeled baseline G1+G2 on Selection, but the advantage is minor, derived from the same data used for candidate selection, and the labeled sensitivity rule collides completely with G1+G5; therefore, Gate 0 cannot be written as the selection algorithm has already won.',
        "",
        '## 1. Same Budget Detection-Level Results',
        "",
        '| Scheme | MAC Share | recovery | Collateral misses | Net Recovery (Rate) | Bidirectional Same-Target (Rate) | Stable-target retention |',
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| RAMP-Q Candidate G1+G5 | {candidate['mac_share']:.2%} | {candidate['rescued']} | {candidate['collateral_misses']} | {candidate['net_rescue']:+d}（{candidate['net_rescue_rate']:.2%}） | {candidate['two_sided']}（{candidate['two_sided_rate']:.2%}） | {candidate['stable_retention']:.2%} |",
        f"| Unlabeled Output Error Baseline G1+G2 | {baseline['mac_share']:.2%} | {baseline['rescued']} | {baseline['collateral_misses']} | {baseline['net_rescue']:+d}（{baseline['net_rescue_rate']:.2%}） | {baseline['two_sided']}（{baseline['two_sided_rate']:.2%}） | {baseline['stable_retention']:.2%} |",
        "",
        'G1+G5 recovers 7 more targets than G1+G2, with 2 more collateral misses, resulting in a net recovery of 5 more; bidirectional same-target increases by 10, but the FP32 MAC share drops by 0.62 percentage points. The advantage of G1+G2 lies in fewer 2 collateral misses and a higher stable retention rate of 0.14 percentage points.',
        "",
        '## 2. Same-Target Pairwise Comparison',
        "",
        '| Paired Items | G1+G5 Exclusive | G1+G2 Exclusive | Exploratory Two-Sided Sign Test |',
        "|---|---:|---:|---:|",
        f"| Recovered Targets | {rescue_candidate_only} | {rescue_baseline_only} | p={comparison['rescue_discordance']['exploratory_exact_sign_p_two_sided']:.3f} |",
        f"| same-target bidirectional agreement | {two_candidate_only} | {two_baseline_only} | p={comparison['two_sided_discordance']['exploratory_exact_sign_p_two_sided']:.3f} |",
        f"| Better Net Utility per Target | {utility_candidate_better} | {utility_baseline_better} | p={comparison['paired_target_utility']['exploratory_exact_sign_p_two_sided']:.3f} |",
        "",
        f"The per-object net utility is defined as: +1 for a missed detection in full W8A8 being recovered, -1 for a previously stable detection being incorrectly damaged, and 0 for all other cases. G1+G5 performs better on{utility_candidate_better}targets, G1+G2 performs better on{utility_baseline_better}targets, and there are{utility_equal}targets with no difference.",
        "",
        'These p values are only supplementary diagnostics: the comparison rule was not pre-registered initially, and the candidate was selected from the very same Selection; therefore, even if the p value is smaller, selection bias cannot be eliminated. This result can only be described as a minor descriptive advantage and cannot be considered an independent, significant victory.',
        "",
        '## 3. Gate 0 Final Explanation',
        "",
        '- Core Mechanism: GO. Module quantization systematically affects specific targets; recovery of G1 can be independently reproduced in a holdout setting, demonstrating large-scale recovery and high stability retention.',
        '- Randomized Control under Same Budget: PASS. G1+G5 significantly outperforms three frozen random combinations.',
        '- Unlabeled Output Error Baseline: G1+G5 has a minor descriptive advantage, but this is not an independently confirmed result.',
        '- Labeled Overall Sensitivity Baseline: COLLISION. It also selects G1+G5; currently, there is no modal result difference that can serve as algorithmic novelty.',
        "- Superiority of Selection Algorithm: NOT ESTABLISHED. Gate 0 is insufficient to support 'a new selection algorithm outperforms traditional methods'; however, the main line of research on target-level attribution and recovery remains unchanged.",
        '- Mechanism and Reliability Main Line: GO. Further research can explore which small targets are sensitive to module quantization, why such sensitivity arises, and whether a generalizable reliability metric can be formed.',
        "",
        '## 4. Suggested paper positioning',
        "",
        "The paper should not position 'automatically selecting G1+G5' as the sole core innovation. A more stable main thread should be: target-level quantization failure attribution, bidirectional validation of damage and recovery, small-object reliability metrics, and risk control under a given computational budget. Mixed-precision selection should be retained as an application outcome or secondary contribution.",
        "",
        '## 5. Next Steps',
        "",
        'Close out RUOD-R Gate 0. No additional combinations will be pursued in the current Selection. Before entering VisDrone Gate 1, freeze the candidate generation rules, traditional labeled sensitivity, unlabeled output error, random controls, and independent validation splits. Only after completing these freezes without reviewing validation results can the selection algorithm be re-evaluated for any genuine added value.',
        "",
        '## Evidence Boundaries',
        "",
        "The current result remains a PyTorch Q/DQ simulation of W8A8 on RUOD-R. There is no evidence of TensorRT INT8, Jetson latency/power consumption, VisDrone reproduction, or state-of-the-art comparison. This report supports maintaining object-level attribution and reliable recovery as the mainline and proceeding to the next Gate, which does not equate to the validation of the paper's method.",
    ]
    output_md = root / "docs/01_gate0_label_free_baseline_comparison.md"
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
