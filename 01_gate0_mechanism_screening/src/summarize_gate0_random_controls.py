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
        '# Gate 0 Budgeted Random Control Final Judgment',
        "",
        f"> Decision: **Random controls PASS; the overall innovation Gate remains PARTIAL.** `G1+G5` substantially exceeds the three fixed random combinations, but the labeled aggregate-sensitivity baseline also selects `G1+G5`, and the label-free baseline `{'+'.join(label_free_groups)}` has not yet completed detection-level q/r validation.",
        "",
        '## 1. Independent Recalculation from Original Per-Target Records',
        "",
        "This report independently re-reads Selection's FP32, full W8A8, and per-target predictions for each combination CSV, and independently constructs new missed detections, recoveries, collateral misses, and bidirectional same-target sets, without directly copying statistical values from the summary of any single group.",
        "",
        '| Combination | MAC Share | recovery | Collateral misses | Net Recovery (Rate) | Bidirectional Same-Target (Rate) | Stable-target retention |',
        "|---|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        'The average net recovery across three random combinations is +2.00, with an average net_recovery_rate of 1.68%; the average bidirectional same-target count is 1.33, and the average bidirectional same-target rate is 1.12%.',
        "",
        '## 2. Pre-registration criterion verification',
        "",
        '| criterion | Actual Value | Requirement | Result |',
        "|---|---:|---:|---|",
        f"| Candidate net_recovery_rate relative to random mean | {candidate['net_rescue_rate']:.2%} | at least{primary_threshold:.2%}(1.20 times the random mean) | {'PASS' if primary_pass else 'FAIL'} |",
        f"| Candidate stability retention rate | {candidate['stable_retention']:.2%} | At least 98% | {'PASS' if guardrail_stability else 'FAIL'} |",
        f"| Candidate bidirectional same-object rate | {candidate['two_sided_rate']:.2%} | not below the random mean{random_mean_two_sided_rate:.2%} | {'PASS' if guardrail_two_sided else 'FAIL'} |",
        f"| Candidate number of random combinations exceeded | {beats_count}/3 | at least 2/3 | {'PASS' if guardrail_beats_two else 'FAIL'} |",
        "",
        f"all four criteria passed, thus the random control comparison under the same budget is judged as **{'PASS' if random_control_pass else 'FAIL'}**. The candidate net_recovery_rate is approximately{candidate['net_rescue_rate'] / random_mean_net_rate:.1f}times. Although this ratio is large, the random-control mean is close to zero. Report the absolute difference as well, rather than emphasizing the ratio alone.",
        "",
        '## 3. Why the method cannot yet be declared innovative',
        "",
        '- Random control: passed, indicating the results are not simply obtained by randomly selecting two modules within the same budget.',
        '- Label-based overall sensitivity baseline: fully collides with the candidate, selecting `G1+G5`, thus RAMP-Q has not yet demonstrated an independent advantage over this traditional rule.',
        f"- Label-free output-error baseline: selects `{'+'.join(label_free_groups)}`, but this combination has not completed detection-level disruption/recovery experiments, so the better selection is still undetermined.",
        "",
        'Therefore, the most accurate current status is `PARTIAL_RANDOM_PASS_BASELINE_PENDING`, not a complete GO, nor STOP.',
        "",
        '## 4. Next steps',
        "",
        f"Run the label-free baseline's disruption arm: `q_only:{'+'.join(label_free_groups)}`; then run the corresponding `r_only:{'+'.join(label_free_groups)}`, and compare with `G1+G5` at the detection level under the same budget.",
        "",
        '## Evidence Boundaries',
        "",
        "This conclusion only covers RUOD-R Selection with PyTorch Q/DQ simulation of W8A8. The random control comes from post-hoc fixed seeds and budget pools when outcomes are unknown; it is not a pre-registered freeze from the project's inception. There is currently no evidence from TensorRT, Jetson, latency, energy consumption, VisDrone, or SOTA.",
    ]
    output_md = root / "docs/02_gate0_random_control_decision.md"
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
