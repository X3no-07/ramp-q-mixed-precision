"""Compute the preregistered label-free output-reconstruction baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import gate0


GROUPS = tuple(gate0.GROUP_ORDER)
EPSILON = 1e-12


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_tensor(raw: Any) -> torch.Tensor:
    """Return the decoded detector tensor used immediately before NMS."""
    candidate = raw[0] if isinstance(raw, (tuple, list)) else raw
    if not isinstance(candidate, torch.Tensor):
        raise TypeError(f"Expected detector tensor, got {type(candidate)!r}")
    return candidate.detach().float().reshape(-1)


def reconstruction_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Output shape changed: {reference.shape} versus {candidate.shape}")
    ref = reference.double()
    cand = candidate.double()
    diff = cand - ref
    nmse = float(torch.sum(diff * diff).item() / (torch.sum(ref * ref).item() + EPSILON))
    denominator = float(torch.linalg.vector_norm(ref).item() * torch.linalg.vector_norm(cand).item())
    cosine = float(torch.dot(ref, cand).item() / (denominator + EPSILON))
    cosine_distance = 1.0 - max(-1.0, min(1.0, cosine))
    if not math.isfinite(nmse) or not math.isfinite(cosine_distance):
        raise ValueError("Non-finite output reconstruction metric")
    return nmse, cosine_distance


def choose_pair(
    pairs: Sequence[tuple[str, str]],
    scores: dict[str, float],
    mac_shares: dict[str, float],
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("Empty pair pool")
    rows = []
    for first, second in pairs:
        rows.append({
            "groups": [first, second],
            "score_sum": scores[first] + scores[second],
            "mac_share": mac_shares[first] + mac_shares[second],
        })
    rows.sort(key=lambda row: (-row["score_sum"], row["mac_share"], row["groups"]))
    return rows[0]


def same_budget_pairs(
    mac_shares: dict[str, float],
    target: float,
    tolerance: float,
    cap: float,
) -> list[tuple[str, str]]:
    return [
        (first, second)
        for first, second in itertools.combinations(GROUPS, 2)
        if mac_shares[first] + mac_shares[second] <= cap
        and abs(mac_shares[first] + mac_shares[second] - target) <= tolerance
    ]


def assert_control_results_unseen(root: Path, preregistration: dict[str, Any]) -> None:
    forbidden: list[Path] = []
    for control in preregistration["random_controls"]:
        for key in ("q_configuration", "r_configuration"):
            safe_name = control[key].replace(":", "_").replace("+", "_plus_")
            forbidden.extend([
                root / "artifacts" / "predictions" / f"selection_{safe_name}.csv",
                root / "artifacts" / "tables" / f"selection_{safe_name}_summary.json",
            ])
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(
            "Random-control results already exist; refusing to compute a supposedly pre-control ranking: "
            + ", ".join(existing[:4])
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--control-preregistration", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    calibration_path = args.calibration.resolve()
    prereg_path = args.control_preregistration.resolve()
    config = gate0.load_config(config_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    preregistration = json.loads(prereg_path.read_text(encoding="utf-8"))
    if preregistration.get("status") != "FROZEN":
        raise ValueError("Control preregistration is not FROZEN")
    if preregistration["label_free_output_reconstruction_baseline"]["status"] != "PREREGISTERED_NOT_COMPUTED":
        raise ValueError("Unexpected label-free baseline registration status")
    assert_control_results_unseen(root, preregistration)

    model_path = Path(config["model"]).resolve()
    if file_sha256(model_path) != calibration["model_sha256"]:
        raise ValueError("Model differs from frozen calibration model")
    if int(config["imgsz"]) != int(calibration["imgsz"]):
        raise ValueError("Image size differs from frozen calibration")
    image_paths = [Path(item).resolve() for item in calibration["images"]]
    if len(image_paths) != 64 or len(image_paths) != int(calibration["image_count"]):
        raise ValueError("The preregistered baseline requires exactly 64 frozen calibration images")
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing calibration images: {missing[:5]}")

    inventory = json.loads(
        (root / "artifacts" / "manifests" / "module_inventory.json").read_text(encoding="utf-8")
    )
    mac_shares = {group: float(inventory["groups"][group]["mac_share"]) for group in GROUPS}
    scales = {name: float(value) for name, value in calibration["activation_max_abs"].items()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    log_path = root / "logs" / "gate0_label_free_output_reconstruction.log"
    gate0.append_log(log_path, f"label_free_output_reconstruction device={device} images={len(image_paths)}")
    started = time.perf_counter()
    model = gate0.load_fused_model(model_path, device)
    wrappers = gate0.install_qdq_wrappers(model, scales)

    per_image_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for image_index, path in enumerate(image_paths, start=1):
            tensor, _bgr, _ratio, _left, _top = gate0.image_tensor(path, int(config["imgsz"]), device)
            gate0.enable_configuration(wrappers, "fp32")
            reference = prediction_tensor(model(tensor))
            for group in GROUPS:
                configuration = f"q_only:{group}"
                gate0.enable_configuration(wrappers, configuration)
                candidate = prediction_tensor(model(tensor))
                nmse, cosine_distance = reconstruction_metrics(reference, candidate)
                per_image_rows.append({
                    "image_index": image_index,
                    "image": str(path),
                    "group": group,
                    "configuration": configuration,
                    "output_elements": int(reference.numel()),
                    "normalized_mse": nmse,
                    "cosine_distance": cosine_distance,
                })
            if image_index % 8 == 0 or image_index == len(image_paths):
                gate0.append_log(log_path, f"output reconstruction {image_index}/{len(image_paths)}")

    aggregates: list[dict[str, Any]] = []
    for group in GROUPS:
        group_rows = [row for row in per_image_rows if row["group"] == group]
        mean_nmse = sum(float(row["normalized_mse"]) for row in group_rows) / len(group_rows)
        mean_cosine = sum(float(row["cosine_distance"]) for row in group_rows) / len(group_rows)
        aggregates.append({
            "group": group,
            "mean_normalized_mse": mean_nmse,
            "mean_cosine_distance": mean_cosine,
            "mac_share": mac_shares[group],
        })
    aggregates.sort(key=lambda row: (-row["mean_normalized_mse"], row["group"]))
    for rank, row in enumerate(aggregates, start=1):
        row["rank"] = rank

    scores = {str(row["group"]): float(row["mean_normalized_mse"]) for row in aggregates}
    budget = preregistration["budget_rule"]
    all_pairs = same_budget_pairs(
        mac_shares,
        float(budget["target_mac_share"]),
        float(budget["absolute_tolerance"]),
        float(budget["hard_cap"]),
    )
    unseen_pairs = [tuple(row["groups"]) for row in preregistration["eligible_unseen_pairs"]]
    selected_all = choose_pair(all_pairs, scores, mac_shares)
    selected_unseen = choose_pair(unseen_pairs, scores, mac_shares)
    selected_all["candidate_collision"] = selected_all["groups"] == ["G1", "G5"]

    elapsed = time.perf_counter() - started
    result = {
        "schema": "rampq.gate0.label_free_output_reconstruction/v1",
        "status": "FROZEN",
        "method": "label-free raw detector output reconstruction under each q_only:Gj intervention",
        "device": str(device),
        "image_count": len(image_paths),
        "image_list_sha256": calibration["image_list_sha256"],
        "calibration_sha256": file_sha256(calibration_path),
        "control_preregistration_sha256": file_sha256(prereg_path),
        "model_sha256": calibration["model_sha256"],
        "primary_score": "mean per-image normalized MSE",
        "secondary_score": "mean per-image cosine distance (diagnostic only)",
        "groups": aggregates,
        "all_same_budget_pair_selection": selected_all,
        "unseen_random_pool_pair_selection": selected_unseen,
        "clarification": (
            "The preregistration phrase 'same budget pool' can be read narrowly as the unseen random pool. "
            "Both the literal unseen-pool result and the all-pair diagnostic are reported; neither result is suppressed."
        ),
        "elapsed_seconds": elapsed,
        "labels_read": False,
        "random_control_results_read": False,
        "evidence_boundary": (
            "Calibration-set raw-output reconstruction under PyTorch Q/DQ simulation; this is not detection "
            "accuracy, TensorRT INT8, Jetson latency, or energy evidence."
        ),
    }

    details_path = root / "artifacts" / "tables" / "gate0_label_free_output_reconstruction_per_image.csv"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)

    output_json = root / "artifacts" / "tables" / "gate0_label_free_output_reconstruction.json"
    gate0.write_json(output_json, result)

    table_rows = [
        f"| {row['rank']} | {row['group']} | {row['mean_normalized_mse']:.8e} | "
        f"{row['mean_cosine_distance']:.8e} | {row['mac_share']:.2%} |"
        for row in aggregates
    ]
    all_name = "+".join(selected_all["groups"])
    unseen_name = "+".join(selected_unseen["groups"])
    report = [
        '# Gate 0 Unlabeled Output Reconstruction Error Baseline',
        "",
        f"> Result: The label-output error method selects among all budget-constrained combinations{all_name}`;Select ` from the preregistered, unexecuted random pool{unseen_name}This step did not read labels, nor did it run or read the detection outputs of three random combinations.",
        "",
        '## 1. Frozen Conditions',
        "",
        f"- Calibration images:{len(image_paths)}images, SHA-256 list: `{calibration['image_list_sha256']}`。",
        f"- Model SHA-256: `{calibration['model_sha256']}`。",
        f"- Device: `{device}`。",
        '- Configuration: FP32 is compared against `q_only:G1` to `q_only:G8` before NMS, in terms of detection outputs.',
        'Labels were not read; no random control results that had not yet been generated were accessed.',
        "",
        '## 2. Module-level Output Reconstruction Error Ranking',
        "",
        '| Rank | Module | Average normalized MSE | Average cosine distance | MAC Share |',
        "|---:|---|---:|---:|---:|",
        *table_rows,
        "",
        'The primary ranking uses only the average normalized MSE. Cosine distance is used only for diagnostic purposes and does not affect the ranking. A higher error indicates that the module is more sensitive to quantization according to conventional output reconstruction methods.',
        "",
        '## 3. Budgeted Combination Selection',
        "",
        f"- All budget-compliant two-module combinations: selected `{all_name}` with a total score of{selected_all['score_sum']:.8e}, MAC share{selected_all['mac_share']:.2%}。",
        f"- Only pre-registered, not executed, random pool: selected `{unseen_name}` with a total score of{selected_unseen['score_sum']:.8e}, MAC share{selected_unseen['mac_share']:.2%}。",
        f"- Collision check with RAMP-Q candidate `G1+G5`:{'Yes' if selected_all['candidate_collision'] else 'No'}。",
        "",
        "The term 'same-budget pool' in the preregistered set may be narrowly interpreted as the unexecuted random pool. To avoid selective reporting, both narrow and full combination diagnostics—including candidates—are publicly disclosed here. Subsequent analyses must not retain only the more favorable result.",
        "",
        '## 4. Interpretation',
        "",
        "The output reconstruction error only answers the question of 'which module, when quantized, causes the largest change in the model's original output'—it does not directly address whether any real targets are missed or recovered. It serves as a traditional label-free proxy baseline and is not a conclusion about detection performance.",
        "",
        '## 5. Next Steps',
        "",
        "The first random control's destructive side is executed under the frozen order: `q_only:G4+G8`. This result does not permit reconfiguration of the random combinations.",
        "",
        '## Evidence Boundaries',
        "",
        'Results from 64 calibration images of RUOD-R and PyTorch Q/DQ simulation of W8A8. No target labels were used, yet the images shared with those used for quantization calibration; it is not an independent test set result and does not represent TensorRT, Jetson, latency, or power consumption performance.',
    ]
    report_path = root / "docs/03_gate0_label_free_output_error_baseline.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "schema": "rampq.gate0.run/v1",
        "phase": "label_free_output_reconstruction",
        "status": "COMPLETE_AND_FROZEN",
        "device": str(device),
        "torch": torch.__version__,
        "inputs": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "calibration": str(calibration_path),
            "calibration_sha256": file_sha256(calibration_path),
            "control_preregistration": str(prereg_path),
            "control_preregistration_sha256": file_sha256(prereg_path),
        },
        "outputs": {
            "summary": str(output_json),
            "summary_sha256": file_sha256(output_json),
            "per_image": str(details_path),
            "per_image_sha256": file_sha256(details_path),
            "report": str(report_path),
            "report_sha256": file_sha256(report_path),
        },
        "elapsed_seconds": elapsed,
    }
    manifest_path = root / "artifacts" / "manifests" / "gate0_label_free_output_reconstruction_run.json"
    gate0.write_json(manifest_path, manifest)
    gate0.append_log(log_path, f"complete elapsed={elapsed:.2f}s all_pair={all_name} unseen_pair={unseen_name}")

    print(json.dumps({
        "status": "COMPLETE_AND_FROZEN",
        "all_same_budget_pair": all_name,
        "candidate_collision": selected_all["candidate_collision"],
        "unseen_pool_pair": unseen_name,
        "next_configuration": "q_only:G4+G8",
        "elapsed_seconds": elapsed,
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
