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
        "# Gate 0 无标签输出重建误差基线",
        "",
        f"> 结果：无标签输出误差方法在全部同预算组合中选出`{all_name}`；在预注册的未运行随机池中选出`{unseen_name}`。本步骤没有读取标签，也没有运行或读取三个随机组合的检测结果。",
        "",
        "## 1. 冻结条件",
        "",
        f"- 校准图片：{len(image_paths)}张，名单SHA-256：`{calibration['image_list_sha256']}`。",
        f"- 模型SHA-256：`{calibration['model_sha256']}`。",
        f"- 设备：`{device}`。",
        "- 配置：分别比较FP32与`q_only:G1`至`q_only:G8`的NMS前检测输出。",
        "- 没有读取目标标签；没有读取尚未产生的随机对照结果。",
        "",
        "## 2. 单模块输出重建误差排名",
        "",
        "| 排名 | 模块 | 平均归一化MSE | 平均余弦距离 | MAC占比 |",
        "|---:|---|---:|---:|---:|",
        *table_rows,
        "",
        "主排名只使用平均归一化MSE；余弦距离仅用于诊断，不能改变排名。误差越大，普通输出重建方法认为该模块越敏感。",
        "",
        "## 3. 同预算组合选择",
        "",
        f"- 全部满足预算的两模块组合：选出`{all_name}`，分数和为{selected_all['score_sum']:.8e}，MAC占比{selected_all['mac_share']:.2%}。",
        f"- 仅预注册未运行随机池：选出`{unseen_name}`，分数和为{selected_unseen['score_sum']:.8e}，MAC占比{selected_unseen['mac_share']:.2%}。",
        f"- 与RAMP-Q候选`G1+G5`是否碰撞：{'是' if selected_all['candidate_collision'] else '否'}。",
        "",
        "预注册中的‘同一预算池’可能被狭义理解为未运行随机池。为避免选择性报告，这里同时公开狭义结果与包含候选在内的全组合诊断，后续不得只保留更有利的一项。",
        "",
        "## 4. 如何解释",
        "",
        "输出重建误差只回答‘量化哪个模块会让模型原始输出变化较大’，不直接回答哪些真实目标漏检或被救回。它是传统的无标签代理基线，不是检测效果结论。",
        "",
        "## 5. 下一步",
        "",
        "按照已冻结顺序运行第一个随机对照的破坏侧：`q_only:G4+G8`。不能根据本结果更换随机组合。",
        "",
        "## 证据边界",
        "",
        "结果来自RUOD-R的64张校准图和PyTorch Q/DQ模拟W8A8。没有使用目标标签，但仍共享已用于量化校准的图片；它不是独立测试集结果，也不代表TensorRT、Jetson、延迟或能耗表现。",
    ]
    report_path = root / "03_Gate0_无标签输出误差基线.md"
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
