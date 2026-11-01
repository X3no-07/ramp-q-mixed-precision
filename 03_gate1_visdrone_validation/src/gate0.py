"""RAMP-Q Gate 0: object-wise quantization failure and module rescue.

This is a deliberately small, inference-only causal screening tool.  It performs
static signed INT8 quantize/dequantize (Q/DQ) simulation around fused Conv2d
operators.  It does *not* claim TensorRT kernel execution or Jetson latency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression


SCHEMA = "rampq.gate0.predictions/v1"
GROUP_ORDER = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
GROUP_DESCRIPTIONS = {
    "G1": "stem_and_early_backbone_top_0_2",
    "G2": "backbone_mid_top_3_4",
    "G3": "backbone_middeep_top_5_6",
    "G4": "backbone_deep_sppf_psa_top_7_10",
    "G5": "neck_topdown_top_13_16",
    "G6": "neck_bottomup_top_17_22",
    "G7": "detect_p3_small_branch",
    "G8": "detect_p4_p5_and_dfl",
}


def sha256_text(lines: Sequence[str]) -> str:
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{stamp}] {message}\n")
    print(message, flush=True)


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = ("model", "dataset_root", "imgsz", "seed")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    return cfg


def group_for_conv(name: str) -> str:
    """Map a fused YOLO11 Conv2d name into one mutually exclusive group."""
    match = re.match(r"model\.(\d+)(?:\.|$)", name)
    if not match:
        raise ValueError(f"Cannot map Conv2d outside model.<index>: {name}")
    top = int(match.group(1))
    if 0 <= top <= 2:
        return "G1"
    if 3 <= top <= 4:
        return "G2"
    if 5 <= top <= 6:
        return "G3"
    if 7 <= top <= 10:
        return "G4"
    if top in {13, 16}:
        return "G5"
    if top in {17, 19, 20, 22}:
        return "G6"
    if top == 23:
        if name.startswith("model.23.cv2.0") or name.startswith("model.23.cv3.0"):
            return "G7"
        return "G8"
    raise ValueError(f"Unexpected Conv2d top-level index in {name}")


def load_fused_model(model_path: Path, device: torch.device) -> nn.Module:
    holder = YOLO(str(model_path))
    model = holder.model
    model.fuse()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def list_conv_modules(model: nn.Module) -> list[tuple[str, nn.Conv2d, str]]:
    rows: list[tuple[str, nn.Conv2d, str]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            rows.append((name, module, group_for_conv(name)))
    if not rows:
        raise RuntimeError("No Conv2d modules found")
    return rows


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = size - new_w, size - new_h
    left, top = int(round(dw / 2 - 0.1)), int(round(dh / 2 - 0.1))
    right, bottom = int(round(dw / 2 + 0.1)), int(round(dh / 2 + 0.1))
    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return canvas, ratio, left, top


def image_tensor(path: Path, size: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray, float, int, int]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    canvas, ratio, left, top = letterbox(bgr, size)
    rgb = canvas[:, :, ::-1].transpose(2, 0, 1)
    array = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).to(device)
    return tensor, bgr, ratio, left, top


def image_paths(dataset_root: Path, split: str) -> list[Path]:
    root = dataset_root / "images" / split
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise RuntimeError(f"No images below {root}")
    return paths


def deterministic_sample(paths: Sequence[Path], count: int, seed: int) -> list[Path]:
    if count >= len(paths):
        return list(paths)
    rng = random.Random(seed)
    return sorted(rng.sample(list(paths), count))


def label_path_for(image_path: Path, dataset_root: Path, split: str) -> Path:
    relative = image_path.relative_to(dataset_root / "images" / split)
    return dataset_root / "labels" / split / relative.parent / f"{image_path.stem}.txt"


@dataclass(frozen=True)
class GroundTruth:
    target_id: str
    class_id: int
    box: tuple[float, float, float, float]
    area_ratio: float
    contrast: float


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def local_contrast(gray: np.ndarray, box: Sequence[float]) -> float:
    height, width = gray.shape
    x1, y1, x2, y2 = box
    x1i, y1i = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    x2i, y2i = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    bw, bh = x2i - x1i, y2i - y1i
    pad_x, pad_y = max(2, bw // 2), max(2, bh // 2)
    rx1, ry1 = max(0, x1i - pad_x), max(0, y1i - pad_y)
    rx2, ry2 = min(width, x2i + pad_x), min(height, y2i + pad_y)
    inner = gray[y1i:y2i, x1i:x2i]
    ring = gray[ry1:ry2, rx1:rx2].copy()
    mask = np.ones(ring.shape, dtype=bool)
    mask[y1i - ry1 : y2i - ry1, x1i - rx1 : x2i - rx1] = False
    background = ring[mask]
    if inner.size == 0 or background.size == 0:
        return 0.0
    return float(abs(float(inner.mean()) - float(background.mean())) / 255.0)


def read_ground_truths(image_path: Path, dataset_root: Path, split: str, bgr: np.ndarray) -> list[GroundTruth]:
    label_path = label_path_for(image_path, dataset_root, split)
    if not label_path.is_file():
        return []
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ground_truths: list[GroundTruth] = []
    for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        cx, cy, bw, bh = (float(value) for value in parts[1:])
        box = (
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        )
        area_ratio = max(0.0, bw) * max(0.0, bh)
        ground_truths.append(
            GroundTruth(
                target_id=f"{image_path.name}:{index}",
                class_id=class_id,
                box=box,
                area_ratio=area_ratio,
                contrast=local_contrast(gray, box),
            )
        )
    return ground_truths


class ActivationCollector:
    def __init__(self, model: nn.Module) -> None:
        self.max_abs: dict[str, float] = defaultdict(float)
        self.handles: list[Any] = []
        for name, module, _group in list_conv_modules(model):
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name: str):
        def observe(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            tensor = inputs[0].detach()
            value = float(tensor.abs().amax().item())
            self.max_abs[name] = max(self.max_abs[name], value)

        return observe

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class QDQConv2d(nn.Module):
    """Fused Conv2d with optional deterministic signed-INT8 Q/DQ simulation."""

    def __init__(self, original: nn.Conv2d, name: str, group: str, activation_max_abs: float) -> None:
        super().__init__()
        self.original = original
        self.name = name
        self.group = group
        self.enabled = False
        activation_scale = max(float(activation_max_abs) / 127.0, 1e-12)
        self.register_buffer("activation_scale", torch.tensor(activation_scale, dtype=torch.float32))
        weight = original.weight.detach()
        dims = tuple(range(1, weight.ndim))
        scale = weight.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12) / 127.0
        qweight = torch.round(weight / scale).clamp(-127, 127) * scale
        self.register_buffer("quantized_weight", qweight)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.original(tensor)
        scale = self.activation_scale.to(dtype=tensor.dtype, device=tensor.device)
        quantized_input = torch.round(tensor / scale).clamp(-127, 127) * scale
        weight = self.quantized_weight.to(dtype=tensor.dtype, device=tensor.device)
        return F.conv2d(
            quantized_input,
            weight,
            self.original.bias,
            self.original.stride,
            self.original.padding,
            self.original.dilation,
            self.original.groups,
        )


def replace_child(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def install_qdq_wrappers(model: nn.Module, scales: dict[str, float]) -> list[QDQConv2d]:
    original_rows = list_conv_modules(model)
    missing = [name for name, _module, _group in original_rows if name not in scales]
    if missing:
        raise RuntimeError(f"Missing activation calibration for {len(missing)} Conv2d modules: {missing[:5]}")
    wrappers: list[QDQConv2d] = []
    for name, module, group in original_rows:
        wrapper = QDQConv2d(module, name, group, scales[name])
        replace_child(model, name, wrapper)
        wrappers.append(wrapper)
    return wrappers


def parse_configuration(configuration: str) -> set[str]:
    if configuration == "fp32":
        return set()
    if configuration == "q_all":
        return set(GROUP_ORDER)
    if configuration.startswith("q_only:"):
        groups = set(configuration.split(":", 1)[1].split("+"))
        return groups
    if configuration.startswith("r_only:"):
        restored = set(configuration.split(":", 1)[1].split("+"))
        return set(GROUP_ORDER) - restored
    raise ValueError(f"Unsupported configuration: {configuration}")


def enable_configuration(wrappers: Sequence[QDQConv2d], configuration: str) -> None:
    enabled_groups = parse_configuration(configuration)
    unknown = enabled_groups - set(GROUP_ORDER)
    if unknown:
        raise ValueError(f"Unknown groups in {configuration}: {sorted(unknown)}")
    for wrapper in wrappers:
        wrapper.enabled = wrapper.group in enabled_groups


def collect_macs(model: nn.Module, size: int, device: torch.device) -> tuple[dict[str, int], dict[str, int]]:
    module_macs: dict[str, int] = {}
    group_macs: dict[str, int] = defaultdict(int)
    handles = []
    for name, module, group in list_conv_modules(model):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor, *, n=name, g=group) -> None:
            batch, out_channels, out_h, out_w = output.shape
            kernel_h, kernel_w = _module.kernel_size
            in_channels = _module.in_channels
            macs = batch * out_channels * out_h * out_w * (in_channels // _module.groups) * kernel_h * kernel_w
            module_macs[n] = module_macs.get(n, 0) + int(macs)
            group_macs[g] += int(macs)

        handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(torch.zeros(1, 3, size, size, device=device))
    for handle in handles:
        handle.remove()
    return module_macs, dict(group_macs)


def calibrate(
    model: nn.Module,
    paths: Sequence[Path],
    size: int,
    device: torch.device,
    log_path: Path,
) -> dict[str, float]:
    collector = ActivationCollector(model)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index, path in enumerate(paths, start=1):
                tensor, _bgr, _ratio, _left, _top = image_tensor(path, size, device)
                model(tensor)
                if index % 16 == 0 or index == len(paths):
                    append_log(log_path, f"calibration {index}/{len(paths)}")
    finally:
        collector.close()
    append_log(log_path, f"calibration finished in {time.perf_counter() - started:.2f}s")
    return dict(collector.max_abs)


def decode_detections(
    prediction: torch.Tensor,
    ratio: float,
    left: int,
    top: int,
    original_width: int,
    original_height: int,
    config: dict[str, Any],
) -> list[Detection]:
    outputs = non_max_suppression(
        prediction,
        conf_thres=float(config["diagnostic_conf"]),
        iou_thres=float(config["nms_iou"]),
        max_det=int(config["max_det"]),
        nc=10,
        max_time_img=2.0,
    )[0]
    detections: list[Detection] = []
    for row in outputs.detach().cpu().tolist():
        x1, y1, x2, y2, confidence, class_id = row[:6]
        x1 = max(0.0, min(original_width, (x1 - left) / ratio))
        y1 = max(0.0, min(original_height, (y1 - top) / ratio))
        x2 = max(0.0, min(original_width, (x2 - left) / ratio))
        y2 = max(0.0, min(original_height, (y2 - top) / ratio))
        detections.append(Detection(int(class_id), float(confidence), (x1, y1, x2, y2)))
    return detections


def match_targets(
    ground_truths: Sequence[GroundTruth],
    detections: Sequence[Detection],
    primary_conf: float,
    match_iou: float,
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, float, int, int]] = []
    for gt_index, gt in enumerate(ground_truths):
        for det_index, detection in enumerate(detections):
            if detection.class_id != gt.class_id or detection.confidence < primary_conf:
                continue
            iou = box_iou(gt.box, detection.box)
            if iou >= match_iou:
                candidates.append((iou, detection.confidence, gt_index, det_index))
    candidates.sort(reverse=True)
    assigned_gt: dict[int, tuple[int, float, float]] = {}
    used_detections: set[int] = set()
    for iou, confidence, gt_index, det_index in candidates:
        if gt_index in assigned_gt or det_index in used_detections:
            continue
        assigned_gt[gt_index] = (det_index, iou, confidence)
        used_detections.add(det_index)

    rows: list[dict[str, Any]] = []
    for gt_index, gt in enumerate(ground_truths):
        same_class = [det for det in detections if det.class_id == gt.class_id]
        best = max(
            ((box_iou(gt.box, det.box), det.confidence) for det in same_class),
            default=(0.0, 0.0),
        )
        assignment = assigned_gt.get(gt_index)
        rows.append(
            {
                "target_id": gt.target_id,
                "class_id": gt.class_id,
                "x1": round(gt.box[0], 4),
                "y1": round(gt.box[1], 4),
                "x2": round(gt.box[2], 4),
                "y2": round(gt.box[3], 4),
                "area_ratio": round(gt.area_ratio, 8),
                "local_contrast": round(gt.contrast, 8),
                "matched": int(assignment is not None),
                "match_iou": round(assignment[1], 6) if assignment else 0.0,
                "match_conf": round(assignment[2], 6) if assignment else 0.0,
                "best_low_iou": round(best[0], 6),
                "best_low_conf": round(best[1], 6),
            }
        )
    return rows


CSV_FIELDS = [
    "schema", "configuration", "image", "image_index", "target_id", "class_id",
    "x1", "y1", "x2", "y2", "area_ratio", "local_contrast", "matched",
    "match_iou", "match_conf", "best_low_iou", "best_low_conf",
]


def evaluate_configurations(
    model: nn.Module,
    wrappers: Sequence[QDQConv2d],
    paths: Sequence[Path],
    configurations: Sequence[str],
    dataset_root: Path,
    split: str,
    config: dict[str, Any],
    device: torch.device,
    output_csv: Path,
    log_path: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    row_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for image_index, path in enumerate(paths):
                tensor, bgr, ratio, left, top = image_tensor(path, int(config["imgsz"]), device)
                ground_truths = read_ground_truths(path, dataset_root, split, bgr)
                for configuration in configurations:
                    enable_configuration(wrappers, configuration)
                    raw = model(tensor)
                    prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
                    detections = decode_detections(
                        prediction,
                        ratio,
                        left,
                        top,
                        bgr.shape[1],
                        bgr.shape[0],
                        config,
                    )
                    for row in match_targets(
                        ground_truths,
                        detections,
                        float(config["primary_conf"]),
                        float(config["match_iou"]),
                    ):
                        writer.writerow(
                            {
                                "schema": SCHEMA,
                                "configuration": configuration,
                                "image": path.name,
                                "image_index": image_index,
                                **row,
                            }
                        )
                        row_count += 1
                if (image_index + 1) % 10 == 0 or image_index + 1 == len(paths):
                    append_log(log_path, f"evaluation {image_index + 1}/{len(paths)} configs={len(configurations)}")
    enable_configuration(wrappers, "fp32")
    append_log(log_path, f"wrote {row_count} target-config rows to {output_csv}")
    append_log(log_path, f"evaluation finished in {time.perf_counter() - started:.2f}s")


def summarize_predictions(input_csv: Path, output_json: Path, output_md: Path) -> dict[str, Any]:
    by_configuration: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_configuration[row["configuration"]][row["target_id"]] = row
    if "fp32" not in by_configuration or "q_all" not in by_configuration:
        raise RuntimeError("Summary requires fp32 and q_all configurations")
    fp32 = by_configuration["fp32"]
    q_all = by_configuration["q_all"]
    common = sorted(set(fp32) & set(q_all))
    fp_hits = {target for target in common if int(fp32[target]["matched"]) == 1}
    added_misses = {target for target in fp_hits if int(q_all[target]["matched"]) == 0}
    stable_hits = {target for target in fp_hits if int(q_all[target]["matched"]) == 1}
    configurations: dict[str, Any] = {}
    for name, rows in sorted(by_configuration.items()):
        if name in {"fp32", "q_all"}:
            continue
        hits = {target for target in added_misses if target in rows and int(rows[target]["matched"]) == 1}
        if name.startswith("r_only:"):
            collateral_misses = {
                target for target in stable_hits if target in rows and int(rows[target]["matched"]) == 0
            }
            rescue_precision_denominator = len(hits) + len(collateral_misses)
            configurations[name] = {
                "rescued": len(hits),
                "rescue_rate": len(hits) / len(added_misses) if added_misses else 0.0,
                "collateral_misses": len(collateral_misses),
                "stable_hit_retention": 1.0 - len(collateral_misses) / len(stable_hits) if stable_hits else 1.0,
                "rescue_precision": len(hits) / rescue_precision_denominator if rescue_precision_denominator else 0.0,
                "net_fp32_hit_change_vs_q_all": len(hits) - len(collateral_misses),
            }
        elif name.startswith("q_only:"):
            degraded = {target for target in fp_hits if target in rows and int(rows[target]["matched"]) == 0}
            configurations[name] = {
                "degraded": len(degraded),
                "degrade_rate": len(degraded) / len(fp_hits) if fp_hits else 0.0,
                "overlap_with_all_int8_misses": len(degraded & added_misses),
            }
    result = {
        "schema": "rampq.gate0.summary/v1",
        "input_csv": str(input_csv.resolve()),
        "target_count": len(common),
        "fp32_hit_count": len(fp_hits),
        "stable_hit_count": len(stable_hits),
        "int8_added_miss_count": len(added_misses),
        "int8_added_miss_rate_among_fp32_hits": len(added_misses) / len(fp_hits) if fp_hits else 0.0,
        "configurations": configurations,
    }
    write_json(output_json, result)
    lines = [
        "# Gate 0 summary",
        "",
        '> This summary only describes the PyTorch Q/DQ simulation quantization results and is not evidence from TensorRT or Jetson.',
        "",
        f"- Number of targets:{result['target_count']}",
        f"- FP32 hits:{result['fp32_hit_count']}",
        f"- Additional misses in full W8A8:{result['int8_added_miss_count']}",
        f"- Additional miss rate:{result['int8_added_miss_rate_among_fp32_hits']:.4%}",
        "",
        '## Module Configuration',
        "",
        '| Configuration | Degradation count | Overlap with full INT8 misses | recovery_count | Recovery rate | Collateral new misses | Stable target retention rate | Net recovery |',
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in configurations.items():
        lines.append(
            f"| {name} | {values.get('degraded', '')} | "
            f"{values.get('overlap_with_all_int8_misses', '')} | "
            f"{values.get('rescued', '')} | "
            f"{values.get('rescue_rate', 0.0):.2%} | "
            f"{values.get('collateral_misses', '')} | "
            f"{values.get('stable_hit_retention', 0.0):.2%} | "
            f"{values.get('net_fp32_hit_change_vs_q_all', '')} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def save_calibration_manifest(
    output_path: Path,
    scales: dict[str, float],
    paths: Sequence[Path],
    config: dict[str, Any],
    model_path: Path,
) -> None:
    payload = {
        "schema": "rampq.gate0.calibration/v1",
        "model": str(model_path.resolve()),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "seed": int(config["seed"]),
        "imgsz": int(config["imgsz"]),
        "image_count": len(paths),
        "image_list_sha256": sha256_text([str(path.resolve()) for path in paths]),
        "images": [str(path.resolve()) for path in paths],
        "activation_max_abs": scales,
    }
    write_json(output_path, payload)


def run_phase(phase: str, config: dict[str, Any], project_root: Path) -> int:
    log_path = project_root / "logs" / f"{phase}.log"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    append_log(log_path, f"phase={phase} device={device} torch={torch.__version__}")
    model_path = Path(config["model"])
    dataset_root = Path(config["dataset_root"])
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = load_fused_model(model_path, device)
    module_rows = list_conv_modules(model)
    module_macs, group_macs = collect_macs(model, int(config["imgsz"]), device)
    total_macs = sum(group_macs.values())
    inventory = {
        "schema": "rampq.gate0.module_inventory/v1",
        "model": str(model_path.resolve()),
        "conv_count": len(module_rows),
        "total_conv_macs": total_macs,
        "groups": {
            group: {
                "description": GROUP_DESCRIPTIONS[group],
                "conv_count": sum(1 for _name, _module, mapped in module_rows if mapped == group),
                "conv_macs": group_macs.get(group, 0),
                "mac_share": group_macs.get(group, 0) / total_macs if total_macs else 0.0,
            }
            for group in GROUP_ORDER
        },
        "modules": [
            {
                "name": name,
                "group": group,
                "weight_shape": list(module.weight.shape),
                "macs": module_macs[name],
            }
            for name, module, group in module_rows
        ],
    }
    write_json(project_root / "artifacts" / "manifests" / "module_inventory.json", inventory)
    append_log(log_path, f"inventory convs={len(module_rows)} total_conv_macs={total_macs}")
    if phase == "inspect":
        for group in GROUP_ORDER:
            info = inventory["groups"][group]
            append_log(log_path, f"{group} convs={info['conv_count']} mac_share={info['mac_share']:.2%}")
        return 0

    if phase == "smoke":
        calibration_count = int(config["calibration_images_smoke"])
        validation_count = int(config["validation_images_smoke"])
        configurations = ["fp32", "q_all", "q_only:G1", "r_only:G1"]
    elif phase == "screen":
        calibration_count = int(config["calibration_images_screen"])
        validation_count = int(config["validation_images_screen"])
        configurations = ["fp32", "q_all"]
    else:
        raise ValueError(phase)

    train_paths = deterministic_sample(image_paths(dataset_root, "train"), calibration_count, int(config["seed"]))
    val_paths = deterministic_sample(image_paths(dataset_root, "val"), validation_count, int(config["seed"]) + 1)
    scales = calibrate(model, train_paths, int(config["imgsz"]), device, log_path)
    calibration_path = project_root / "artifacts" / "calibration" / f"{phase}.json"
    save_calibration_manifest(calibration_path, scales, train_paths, config, model_path)
    wrappers = install_qdq_wrappers(model, scales)
    prediction_path = project_root / "artifacts" / "predictions" / f"{phase}.csv"
    evaluate_configurations(
        model,
        wrappers,
        val_paths,
        configurations,
        dataset_root,
        "val",
        config,
        device,
        prediction_path,
        log_path,
    )
    report_dir = project_root / "archive" / "gate0_auto_summaries"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_predictions(
        prediction_path,
        project_root / "artifacts" / "tables" / f"{phase}_summary.json",
        report_dir / f"Gate0_{phase}_summary.md",
    )
    manifest = {
        "schema": "rampq.gate0.run/v1",
        "phase": phase,
        "device": str(device),
        "torch": torch.__version__,
        "ultralytics": __import__("ultralytics").__version__,
        "config": config,
        "configurations": configurations,
        "validation_image_count": len(val_paths),
        "validation_image_list_sha256": sha256_text([str(path.resolve()) for path in val_paths]),
        "summary": summary,
    }
    write_json(project_root / "artifacts" / "manifests" / f"{phase}_run.json", manifest)
    append_log(log_path, f"phase={phase} complete added_misses={summary['int8_added_miss_count']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inspect", "smoke", "screen"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config.resolve())
    return run_phase(args.phase, config, args.project_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
