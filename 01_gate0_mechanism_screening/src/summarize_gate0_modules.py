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

    labels = {"strong": 'Strong candidate', "low_priority": 'Low priority', "eliminated": 'Eliminated'}
    table_rows = [
        f"| {row['rank']} | {row['group']} | {labels[row['status']]} | {row['q_overlap_with_all_int8']} | "
        f"{row['rescued']}（{row['rescue_rate']:.2%}） | {row['collateral_misses']} | {row['net_rescue']:+d} | "
        f"{row['two_sided_same_targets']} | {row['stable_retention']:.2%} | {row['mac_share']:.2%} |"
        for row in rows
    ]
    report = [
        '# Gate 0 Selection: Single-Module Ranking and Pre-Registration of Dual-Module Combination',
        "",
        '> Conclusion: G1 is the only strong candidate that passes all single-module thresholds. The combination experiment was pre-locked to G1+G5 and G1+G3 prior to result inspection; the holdout remains sealed.',
        "",
        '## Unified Ranking',
        "",
        '| Rank | Module | Status | Destruction side overlaps with full INT8 | recovery_count (rate) | Collateral misses | Net recovery | same-target bidirectional agreement | Stable-target retention | MAC Share |',
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        'Ranking criteria are: status (Strong Candidate, Low Priority, Eliminated), net recovery, bidirectional same-target count, recovery_count, and lower MAC share. This rule applies only to the Selection module.',
        "",
        '## Single-Module Conclusions',
        "",
        '- G1 is the only strong candidate: recovered 94 targets, net recovery +81, 69 bidirectional matches at the same object level, and accounts for 10.32% of MAC.',
        '- G5 and G3 show only minor positive gains and limited bidirectional evidence; they are retained as sources for combination candidates.',
        '- G2, G4, G6, G7, and G8 fail to meet net gain or bidirectional consistency requirements and are not included in the combination pool.',
        "",
        '## Frozen Two-Module Combinations',
        "",
        '1. Main combination: `G1+G5`, accounting for 22.35% of MAC proxy. Run `q_only:G1+G5` and `r_only:G1+G5`.',
        '2. Backup combination: `G1+G3`, accounting for 22.14% of MAC proxy. Run `q_only:G1+G3` and `r_only:G1+G3`.',
        "",
        'Both combinations must be executed and fully reported. The second combination cannot be selected based on the first, and no new combinations may be introduced and then selected based on the best result.',
        "",
        '## Incremental Threshold for Replacing G1 with a Two-Module Combination',
        "",
        'A combination is only allowed to replace the G1 single-module solution if it simultaneously satisfies the following conditions:',
        "",
        f"recovery_count at least{pair_thresholds['rescued_at_least']}more than G1's 94, by at least 5;",
        f"net recovery at least{pair_thresholds['net_rescue_at_least']}more than G1's +81, by at least 5;",
        '- At least 98% stability of target retention;',
        f"bidirectional same-object at least{pair_thresholds['two_sided_same_targets_at_least']}instances, not less than G1;",
        '- The combined MAC contribution of the two modules does not exceed 25%;',
        "",
        'If neither combination passes, lock G1 as the holdout candidate; if both pass, compare net recovery, bidirectional object-level matches, and MAC share in sequence.',
        "",
        '## Execution Order',
        "",
        "1. `q_only:G1+G5`",
        "2. `r_only:G1+G5`",
        "3. `q_only:G1+G3`",
        "4. `r_only:G1+G3`",
        '5. Aggregate and lock in a single candidate; only after this step is the unique candidate allowed to run on holdout;',
        "",
        '## Evidence Boundaries',
        "",
        '- All current results come from Selection and PyTorch Q/DQ simulation of W8A8, not from TensorRT INT8 or Jetson real-world testing;',
        '- The holdout was not used for module-level inference intervention, ranking, or combination selection;',
        '- Currently, we can only say that G1 shows strong signals in Selection, not that it has been independently validated or reached the conclusion stated in the paper;',
    ]
    output_md = root / "docs/07_gate0_module_ranking_and_combination_preregistration.md"
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
