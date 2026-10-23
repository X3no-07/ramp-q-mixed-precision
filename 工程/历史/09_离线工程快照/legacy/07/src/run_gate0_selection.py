"""Run one frozen Gate 0 intervention on a selection or holdout image list."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import gate0


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def list_sha256(lines: Sequence[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_image_list(path: Path) -> list[Path]:
    raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw) != len(set(raw)):
        raise ValueError(f"Duplicate paths in image list: {path}")
    paths = [Path(line).resolve() for line in raw]
    missing = [str(item) for item in paths if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing intervention images: {missing[:5]}")
    return paths


def read_prediction_map(path: Path, configuration: str | None = None) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if configuration is not None and row["configuration"] != configuration:
                continue
            key = row["target_id"]
            if key in rows:
                raise ValueError(f"Duplicate target {key} for {configuration} in {path}")
            rows[key] = row
    return rows


def baseline_maps(path: Path, image_names: set[str]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    fp32: dict[str, dict[str, str]] = {}
    q_all: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image"] not in image_names:
                continue
            destination = fp32 if row["configuration"] == "fp32" else q_all if row["configuration"] == "q_all" else None
            if destination is not None:
                destination[row["target_id"]] = row
    if not fp32 or set(fp32) != set(q_all):
        raise ValueError("Subset FP32/q_all baselines are missing or unequal")
    return fp32, q_all


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=("selection", "holdout"), default="selection")
    parser.add_argument("--candidate-lock", type=Path)
    args = parser.parse_args(argv)

    enabled_groups = gate0.parse_configuration(args.configuration)
    if not args.configuration.startswith(("q_only:", "r_only:")):
        raise ValueError("Only q_only/r_only interventions are allowed")
    if not enabled_groups <= set(gate0.GROUP_ORDER):
        raise ValueError(args.configuration)

    project_root = args.project_root.resolve()
    subset = args.subset
    if subset == "holdout":
        if args.candidate_lock is None:
            raise ValueError("Holdout runs require --candidate-lock")
        candidate_lock = json.loads(args.candidate_lock.read_text(encoding="utf-8"))
        if candidate_lock.get("status") != "LOCKED":
            raise ValueError("Holdout candidate lock is not LOCKED")
        allowed = set(candidate_lock.get("holdout_allowed_interventions", []))
        if args.configuration not in allowed:
            raise ValueError(f"Configuration is not preregistered for holdout: {args.configuration}")
    config = gate0.load_config(args.config.resolve())
    model_path = Path(config["model"]).resolve()
    dataset_root = Path(config["dataset_root"]).resolve()
    image_paths = read_image_list(args.image_list.resolve())
    canonical_lines = sorted(str(path) for path in image_paths)

    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    expected_hash = split_manifest[subset]["list_sha256"]
    actual_hash = list_sha256(canonical_lines)
    if actual_hash != expected_hash:
        raise ValueError(f"{subset} list hash changed: expected {expected_hash}, got {actual_hash}")
    other_subset = "holdout" if subset == "selection" else "selection"
    other_file = Path(split_manifest[other_subset]["list"])
    other_paths = set(line.strip() for line in other_file.read_text(encoding="utf-8").splitlines() if line.strip())
    overlap = set(canonical_lines) & other_paths
    if overlap:
        raise ValueError(f"Selection/holdout overlap: {sorted(overlap)[:5]}")

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration["model_sha256"] != file_sha256(model_path):
        raise ValueError("Model differs from the frozen screen calibration model")
    if int(calibration["seed"]) != int(config["seed"]) or int(calibration["imgsz"]) != int(config["imgsz"]):
        raise ValueError("Calibration seed/imgsz differs from current config")
    scales = {name: float(value) for name, value in calibration["activation_max_abs"].items()}

    safe_name = args.configuration.replace(":", "_").replace("+", "_plus_")
    log_path = project_root / "logs" / f"{subset}_{safe_name}.log"
    prediction_path = project_root / "artifacts" / "predictions" / f"{subset}_{safe_name}.csv"
    summary_path = project_root / "artifacts" / "tables" / f"{subset}_{safe_name}_summary.json"
    details_path = project_root / "artifacts" / "tables" / f"{subset}_{safe_name}_targets.csv"
    manifest_path = project_root / "artifacts" / "manifests" / f"{subset}_{safe_name}_run.json"
    report_dir = project_root / "archive" / "gate0_auto_summaries"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"Gate0_{subset}_{safe_name}_自动摘要.md"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gate0.append_log(log_path, f"subset={subset} intervention={args.configuration} device={device} images={len(image_paths)}")
    started = time.perf_counter()
    model = gate0.load_fused_model(model_path, device)
    wrappers = gate0.install_qdq_wrappers(model, scales)
    gate0.evaluate_configurations(
        model=model,
        wrappers=wrappers,
        paths=image_paths,
        configurations=[args.configuration],
        dataset_root=dataset_root,
        split="val",
        config=config,
        device=device,
        output_csv=prediction_path,
        log_path=log_path,
    )

    image_names = {path.name for path in image_paths}
    fp32, q_all = baseline_maps(args.baseline.resolve(), image_names)
    intervention = read_prediction_map(prediction_path, args.configuration)
    if set(intervention) != set(fp32):
        raise ValueError(f"Intervention target set differs from frozen {subset} baseline")

    fp_hits = {target for target, row in fp32.items() if int(row["matched"]) == 1}
    added_misses = {target for target in fp_hits if int(q_all[target]["matched"]) == 0}
    stable_hits = {target for target in fp_hits if int(q_all[target]["matched"]) == 1}
    intervention_hits = {target for target in fp_hits if int(intervention[target]["matched"]) == 1}
    intervention_misses = fp_hits - intervention_hits
    summary: dict[str, Any] = {
        "schema": "rampq.gate0.subset_intervention/v1",
        "configuration": args.configuration,
        "subset": subset,
        "image_count": len(image_paths),
        "target_count": len(fp32),
        "fp32_hit_count": len(fp_hits),
        "q_all_added_miss_count": len(added_misses),
        "image_list_sha256": actual_hash,
        "baseline_predictions_sha256": file_sha256(args.baseline.resolve()),
        "intervention_predictions_sha256": file_sha256(prediction_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.configuration.startswith("q_only:"):
        overlap_misses = intervention_misses & added_misses
        distinct_misses = intervention_misses - added_misses
        summary.update({
            "intervention_degraded_count": len(intervention_misses),
            "intervention_degrade_rate_among_fp32_hits": len(intervention_misses) / len(fp_hits) if fp_hits else 0.0,
            "overlap_with_q_all_added_misses": len(overlap_misses),
            "coverage_of_q_all_added_misses": len(overlap_misses) / len(added_misses) if added_misses else 0.0,
            "intervention_misses_outside_q_all_set": len(distinct_misses),
        })
        detail_targets = [
            (target, "overlap_q_all_miss" if target in added_misses else "q_only_distinct_miss")
            for target in sorted(intervention_misses)
        ]
    else:
        rescued = added_misses & intervention_hits
        unrecovered = added_misses - rescued
        collateral_misses = stable_hits & intervention_misses
        rescue_precision_denominator = len(rescued) + len(collateral_misses)
        summary.update({
            "rescued_count": len(rescued),
            "rescue_rate": len(rescued) / len(added_misses) if added_misses else 0.0,
            "unrecovered_count": len(unrecovered),
            "collateral_miss_count": len(collateral_misses),
            "stable_hit_retention": 1.0 - len(collateral_misses) / len(stable_hits) if stable_hits else 1.0,
            "rescue_precision": len(rescued) / rescue_precision_denominator if rescue_precision_denominator else 0.0,
            "net_rescue": len(rescued) - len(collateral_misses),
        })
        detail_targets = (
            [(target, "rescued") for target in sorted(rescued)]
            + [(target, "unrecovered") for target in sorted(unrecovered)]
            + [(target, "collateral_miss") for target in sorted(collateral_misses)]
        )
    gate0.write_json(summary_path, summary)

    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["target_id", "image", "class_id", "outcome", "in_q_all_added_misses", "fp32_conf", "q_all_conf", "intervention_conf"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target, outcome in detail_targets:
            writer.writerow({
                "target_id": target,
                "image": fp32[target]["image"],
                "class_id": fp32[target]["class_id"],
                "outcome": outcome,
                "in_q_all_added_misses": int(target in added_misses),
                "fp32_conf": fp32[target]["match_conf"],
                "q_all_conf": q_all[target]["match_conf"],
                "intervention_conf": intervention[target]["match_conf"],
            })

    manifest = {
        "schema": "rampq.gate0.run/v1",
        "phase": f"{subset}_single_intervention",
        "configuration": args.configuration,
        "device": str(device),
        "torch": torch.__version__,
        "model": str(model_path),
        "model_sha256": file_sha256(model_path),
        "calibration": str(args.calibration.resolve()),
        "image_list": str(args.image_list.resolve()),
        "holdout_read_for_inference": subset == "holdout",
        "outputs": {
            "predictions": str(prediction_path.resolve()),
            "summary": str(summary_path.resolve()),
            "degraded_targets": str(details_path.resolve()),
            "report": str(report_path.resolve()),
        },
        "summary": summary,
    }
    gate0.write_json(manifest_path, manifest)

    lines = [
        f"# Gate 0 {subset.capitalize()}：{args.configuration}",
        "",
        "> 本轮只运行一个模块干预配置，使用PyTorch Q/DQ模拟W8A8；不是TensorRT或Jetson实测。",
        "",
        "## 结果",
        "",
        f"- {subset.capitalize()}图片：{len(image_paths)}",
        f"- FP32原本命中：{len(fp_hits)}",
        f"- 全W8A8新增漏检：{len(added_misses)}",
    ]
    if args.configuration.startswith("q_only:"):
        lines.extend([
            f"- `{args.configuration}` 单独造成的漏检：{len(intervention_misses)}（占FP32命中 {summary['intervention_degrade_rate_among_fp32_hits']:.2%}）",
            f"- 其中与全W8A8新增漏检重合：{summary['overlap_with_q_all_added_misses']}（覆盖全W8A8新增漏检 {summary['coverage_of_q_all_added_misses']:.2%}）",
            f"- 不属于全W8A8新增漏检集合：{summary['intervention_misses_outside_q_all_set']}",
            "",
            "## 本轮能回答什么",
            "",
            "- 只回答：单独量化该模块组是否足以让一部分FP32命中目标消失，以及它与全W8A8漏检是否重合。",
            "- 不能回答该模块恢复FP32后能否救回目标；必须再运行对应的 `r_only` 隔离实验。",
        ])
    else:
        lines.extend([
            f"- 恢复该模块后救回：{summary['rescued_count']}（救回率 {summary['rescue_rate']:.2%}）",
            f"- 仍未救回：{summary['unrecovered_count']}",
            f"- 对原本全W8A8稳定目标造成的附带漏检：{summary['collateral_miss_count']}",
            f"- 稳定目标保持率：{summary['stable_hit_retention']:.2%}",
            f"- 净救回：{summary['net_rescue']}",
            "",
            "## 本轮能回答什么",
            "",
            "- 只回答：从全W8A8出发，恢复该模块组是否能够救回新增漏检，以及是否引入附带漏检。",
            "- 必须和对应 `q_only` 的目标级重合关系联合解释，不能只看总体救回率。",
        ])
    lines.extend([
        "- 单个干预配置不构成Gate 0最终结论，必须按预注册完成对应的双向实验。",
        "",
        "## 数据隔离",
        "",
        f"- {subset.capitalize()}名单SHA-256：`{actual_hash}`",
    ])
    if subset == "selection":
        lines.append("- holdout没有用于推理、排名或参数调整；脚本只读取holdout名单进行零重叠检查。")
    else:
        lines.append("- 配置已通过候选锁定文件校验；没有运行或改换未预注册配置。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.configuration.startswith("q_only:"):
        gate0.append_log(log_path, f"complete degraded={summary['intervention_degraded_count']} overlap_q_all={summary['overlap_with_q_all_added_misses']}")
    else:
        gate0.append_log(log_path, f"complete rescued={summary['rescued_count']} collateral={summary['collateral_miss_count']} net={summary['net_rescue']}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
