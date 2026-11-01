"""Create a bidirectional summary for one preregistered random control pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detail_sets(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["outcome"], set()).add(row["target_id"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pair", required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    groups = args.pair.split("+")
    if len(groups) != 2 or groups[0] == groups[1]:
        raise ValueError("--pair must contain two distinct groups, for example G4+G8")

    prereg_path = root / "artifacts" / "tables" / "gate0_control_preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    registered = next(
        (row for row in prereg["random_controls"] if row["groups"] == groups),
        None,
    )
    if registered is None:
        raise ValueError(f"Pair is not a preregistered random control: {args.pair}")
    registered_order = int(registered["order"])
    next_control = next(
        (row for row in prereg["random_controls"] if int(row["order"]) == registered_order + 1),
        None,
    )
    next_configuration = next_control["q_configuration"] if next_control else "summarize_all_random_controls"

    safe_pair = "_plus_".join(groups)
    q_summary_path = root / "artifacts" / "tables" / f"selection_q_only_{safe_pair}_summary.json"
    r_summary_path = root / "artifacts" / "tables" / f"selection_r_only_{safe_pair}_summary.json"
    q_details_path = root / "artifacts" / "tables" / f"selection_q_only_{safe_pair}_targets.csv"
    r_details_path = root / "artifacts" / "tables" / f"selection_r_only_{safe_pair}_targets.csv"
    for path in (q_summary_path, r_summary_path, q_details_path, r_details_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    q_summary = json.loads(q_summary_path.read_text(encoding="utf-8"))
    r_summary = json.loads(r_summary_path.read_text(encoding="utf-8"))
    if q_summary["configuration"] != registered["q_configuration"]:
        raise ValueError("q_only configuration differs from preregistration")
    if r_summary["configuration"] != registered["r_configuration"]:
        raise ValueError("r_only configuration differs from preregistration")
    if q_summary["image_list_sha256"] != r_summary["image_list_sha256"]:
        raise ValueError("q_only and r_only image lists differ")
    if q_summary["q_all_added_miss_count"] != r_summary["q_all_added_miss_count"]:
        raise ValueError("q_only and r_only baseline added-miss counts differ")

    q_sets = detail_sets(q_details_path)
    r_sets = detail_sets(r_details_path)
    q_overlap = q_sets.get("overlap_q_all_miss", set())
    rescued = r_sets.get("rescued", set())
    two_sided = q_overlap & rescued
    added_misses = int(q_summary["q_all_added_miss_count"])
    candidate = prereg["candidate"]

    result: dict[str, Any] = {
        "schema": "rampq.gate0.random_control_bidirectional/v1",
        "status": "COMPLETE",
        "preregistered_order": registered_order,
        "groups": groups,
        "mac_share": float(registered["mac_share"]),
        "selection_image_count": int(q_summary["image_count"]),
        "all_int8_added_miss_count": added_misses,
        "q_only": {
            "degraded": int(q_summary["intervention_degraded_count"]),
            "overlap_with_all_int8": int(q_summary["overlap_with_q_all_added_misses"]),
            "overlap_rate": len(q_overlap) / added_misses,
            "outside_all_int8": int(q_summary["intervention_misses_outside_q_all_set"]),
        },
        "r_only": {
            "rescued": int(r_summary["rescued_count"]),
            "rescue_rate": float(r_summary["rescue_rate"]),
            "collateral_misses": int(r_summary["collateral_miss_count"]),
            "stable_retention": float(r_summary["stable_hit_retention"]),
            "net_rescue": int(r_summary["net_rescue"]),
            "net_rescue_rate": int(r_summary["net_rescue"]) / added_misses,
        },
        "two_sided_same_targets": len(two_sided),
        "two_sided_rate": len(two_sided) / added_misses,
        "two_sided_target_ids": sorted(two_sided),
        "candidate_reference": {
            "groups": candidate["groups"],
            "mac_share": float(candidate["mac_share"]),
            "net_rescue": int(candidate["selection_net_rescue"]),
            "net_rescue_rate": int(candidate["selection_net_rescue"]) / added_misses,
            "two_sided": int(candidate["selection_two_sided"]),
            "stable_retention": float(candidate["selection_stable_retention"]),
        },
        "input_hashes": {
            "preregistration": file_sha256(prereg_path),
            "q_summary": file_sha256(q_summary_path),
            "r_summary": file_sha256(r_summary_path),
            "q_details": file_sha256(q_details_path),
            "r_details": file_sha256(r_details_path),
        },
        "evidence_boundary": (
            "One preregistered random pair on RUOD-R Selection under PyTorch Q/DQ simulated W8A8. "
            "Overall candidate-versus-random pass/fail requires all three frozen random controls."
        ),
    }

    output_json = root / "artifacts" / "tables" / f"selection_control_{safe_pair}_bidirectional_summary.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        f"# Gate 0 Selection：{args.pair}Bidirectional Randomized Control Summary",
        "",
        f"> Result: restoring{args.pair}recovers{result['r_only']['rescued']}additional misses introduced by full W8A8, but causes{result['r_only']['collateral_misses']}collateral misses, for a net recovery of{result['r_only']['net_rescue']:+d}. This is frozen random control number{registered['order']}among the three; it cannot determine the final outcome alone.",
        "",
        '## Bidirectional Results',
        "",
        '| Metric | Numerical |',
        "|---|---:|",
        f"| FP32 MAC ratio proxy | {registered['mac_share']:.2%} |",
        f"| Total missed targets on the disrupted side | {result['q_only']['degraded']} |",
        f"| Overlap with full W8A8 misses | {result['q_only']['overlap_with_all_int8']}/{added_misses}（{result['q_only']['overlap_rate']:.2%}） |",
        f"| Recovery side recovery | {result['r_only']['rescued']}/{added_misses}（{result['r_only']['rescue_rate']:.2%}） |",
        f"| Collateral misses | {result['r_only']['collateral_misses']} |",
        f"| Net recovery | {result['r_only']['net_rescue']:+d}（{result['r_only']['net_rescue_rate']:.2%}） |",
        f"| same-target bidirectional agreement | {result['two_sided_same_targets']}/{added_misses}（{result['two_sided_rate']:.2%}） |",
        f"| Stable-target retention | {result['r_only']['stable_retention']:.2%} |",
        "",
        '## Comparison with Current Candidate Reference',
        "",
        f"- RAMP-Q candidate `G1+G5`: net recovery{candidate['selection_net_rescue']:+d}, same-target bidirectional agreement{candidate['selection_two_sided']}, MAC share{candidate['mac_share']:.2%}。",
        f"- This randomized control `{args.pair}`: net recovery of{result['r_only']['net_rescue']:+d}, same-target bidirectional agreement{result['two_sided_same_targets']}, MAC share{registered['mac_share']:.2%}。",
        f"- This describes only control number{registered_order}. The preregistered primary criterion uses the mean net_recovery_rate of the three random controls; RAMP-Q cannot be declared to pass at this point.",
        "",
        '## Next Steps',
        "",
        (
            f"Run the disruption arm of the next random control in the frozen order: `{next_configuration}`。"
            if next_control
            else 'All three randomized controls have been completed; the next step is to consolidate results uniformly, with no new combinations to be added.'
        ),
        "",
        '## Evidence Boundaries',
        "",
        'This result comes from Selection and PyTorch Q/DQ simulation of W8A8, not from TensorRT or Jetson real-world measurements. Bidirectional same-object representation means that a labeled target disappears on the destruction side and reappears on the recovery side, but it does not imply that the unique causal path of numerical propagation at the base level has been proven.',
    ]
    pair_names = {
        "G2+G8": "docs/20_gate0_random_control_g2_g8.md",
        "G3+G8": "docs/21_gate0_random_control_g3_g8.md",
        "G4+G8": "docs/22_gate0_random_control_g4_g8.md",
    }
    pair = "+".join(groups)
    if pair not in pair_names:
        raise ValueError(f"No registered report name for random pair {pair}")
    report_path = root / pair_names[pair]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE",
        "pair": args.pair,
        "net_rescue": result["r_only"]["net_rescue"],
        "two_sided": result["two_sided_same_targets"],
        "stable_retention": result["r_only"]["stable_retention"],
        "next_configuration": next_configuration,
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
