"""Frozen, debug-only decoder precision intervention; never changes an acceptance guard.

Imports of torch/Ultralytics are deferred until a newly created output and its
private settings directory exist. No NMS, annotations, Selection, or Holdout are
used. Native reconstructions and historical bytes must match before any result
can be described as a completed probe.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np


SCHEMA = "rampq.gate1.decode_precision_probe_protocol/v1"
QUALIFIER_SHA = "b981869ad658e0619454cdbe2361c8a34bd029adf2a8bbb7f85590853c736f34"
CHECKPOINT_SHA = "cddd14b3ebba294e4c3200bfb50fc1db1a9459c425d8e51d583c0881d71cfcfa"
CONTROLS = {"tf32": False, "float32_matmul_precision": "highest", "autocast_enabled": False,
            "deterministic_algorithms": True, "cudnn_benchmark": False}
TOLERANCES = {"box_rtol": 0.005, "box_atol": 0.5, "score_rtol": 0.02, "score_atol": 0.002}
COMPARISONS = {
    "original_fp32_vs_original_fp16": ("fp32_forward", "fp16_forward"),
    "original_fp32_vs_fp16_logits_decode_fp32": ("fp32_forward", "source_fp16_decode_fp32_decoded"),
    "same_fp32_logits_decode_fp32_vs_fp16": ("source_fp32_decode_fp32_decoded", "source_fp32_decode_fp16_decoded"),
    "same_fp16_logits_decode_fp32_vs_fp16": ("source_fp16_decode_fp32_decoded", "source_fp16_decode_fp16_decoded"),
    "fp32_logits_vs_fp16_logits_both_decode_fp32": ("source_fp32_decode_fp32_decoded", "source_fp16_decode_fp32_decoded"),
}
LIMITS = [
    "Post-hoc arithmetic diagnosis on previously reused training-derived debug images; no Selection or Holdout access.",
    "The saved fp16 EMA lifted to float32 is not original full-precision training EMA.",
    "FP16-logit FP32 decoding changes arithmetic only and is not a validated deployment fix.",
    "Diagnostic comparison PASS never replaces the original unchanged dtype guard or accepts Gate1.",
    "No NMS, target matching, AP, quantization, TensorRT, Orin, speed, or energy measurement.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def bound_identity(entry: dict, label: str) -> Path:
    require(isinstance(entry, dict) and set(entry) >= {"path", "bytes", "sha256"}, f"Missing {label} identity")
    path = Path(entry["path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), f"Invalid {label} path")
    require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], f"Changed {label}")
    return path.resolve(strict=True)


def safe_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    require(not rel.is_absolute() and rel.parts and ".." not in rel.parts, "Unsafe relative path")
    cursor = root
    for part in rel.parts:
        cursor /= part
        require(not cursor.is_symlink(), "Symlink in immutable input")
    result = cursor.resolve(strict=True)
    require(result.is_relative_to(root) and result.is_file(), "Input escapes immutable directory")
    return result


def validate_protocol(p: dict, own_sha: str, rows: list[dict]) -> list[dict]:
    """Pure metadata guard, exercised by synthetic tests before any GPU use."""
    require(p.get("schema") == SCHEMA, "Wrong protocol schema")
    require(p.get("runner_sha256") == own_sha, "Runner not bound to the protocol")
    require(p.get("data_role") == "debug" and p.get("selection_detector_inference") is False
            and p.get("holdout_accessed") is False, "Forbidden data role")
    require(p.get("historical_equality_required") is True and p.get("native_reconstruction_required") is True,
            "Mandatory exactness checks cannot be disabled")
    require(p.get("controls") == CONTROLS and p.get("raw_tolerances") == TOLERANCES,
            "Precision controls or frozen tolerances changed")
    require(p.get("qualifier", {}).get("sha256") == QUALIFIER_SHA, "Unreviewed qualifier")
    require(p.get("checkpoint", {}).get("sha256") == CHECKPOINT_SHA, "Checkpoint changed")
    require(isinstance(rows, list) and len(rows) == 109, "Expected the frozen 109-image debug cohort")
    by_id = {}
    for row in rows:
        image_id = row.get("image_id", "")
        require(row.get("assigned_split") == "debug" and row.get("source_split") == "train"
                and image_id.startswith("train/") and row.get("image_path", "").startswith("images/train/"),
                "Cohort contains a non-debug image")
        require(image_id not in by_id, "Duplicate cohort identity")
        by_id[image_id] = row
    ids = p.get("image_ids")
    require(isinstance(ids, list) and ids and len(ids) == len(set(ids)) and all(i in by_id for i in ids),
            "Requested IDs must be unique members of the frozen debug cohort")
    stage = p.get("stage")
    require((stage == "pilot3" and len(ids) == 3) or
            (stage == "full109" and ids == [row["image_id"] for row in rows]), "Wrong stage/count/order")
    require(isinstance(p.get("package_versions"), dict) and bool(p["package_versions"]), "Missing package identities")
    require(isinstance(p.get("framework_source_sha256"), dict) and bool(p["framework_source_sha256"]), "Missing framework identities")
    return [by_id[i] for i in ids]


def decode_functional(torch: Any, box_logits: Any, class_logits: Any, anchors: Any,
                      strides: Any, reg_max: int, dtype: Any) -> dict[str, Any]:
    """Mirror pinned DFL/Detect arithmetic; never algebraically simplify geometry.

    Precision intervention casts captured logits, geometry, and fixed integral
    bins to the requested dtype. Network forward computation is not rerun.
    """
    require(isinstance(reg_max, int) and reg_max > 1, "DFL reg_max must be > 1")
    require(dtype in (torch.float32, torch.float16), "Unsupported decode dtype")
    require(box_logits.ndim == 3 and box_logits.shape[1] == 4 * reg_max, "Wrong box-logit shape")
    b, _, n = box_logits.shape
    require(class_logits.ndim == 3 and class_logits.shape[0] == b and class_logits.shape[2] == n, "Wrong class-logit shape")
    require(tuple(anchors.shape) == (2, n) and tuple(strides.shape) == (1, n), "Wrong geometry shape")
    require(all(bool(torch.isfinite(x).all()) for x in (box_logits, class_logits, anchors, strides)), "Nonfinite decoder input")
    logits = box_logits.to(dtype=dtype)
    classes = class_logits.to(dtype=dtype)
    geometry = anchors.to(dtype=dtype).unsqueeze(0)
    scale = strides.to(dtype=dtype)
    probabilities = logits.view(b, 4, reg_max, n).transpose(2, 1).softmax(1)
    bins = torch.arange(reg_max, device=logits.device, dtype=dtype).view(1, reg_max, 1, 1)
    distances = torch.nn.functional.conv2d(probabilities, bins).view(b, 4, n)
    lt, rb = distances.chunk(2, 1)
    xy1, xy2 = geometry - lt, geometry + rb
    center = (xy1 + xy2) / 2
    wh = xy2 - xy1
    xywh = torch.cat([center, wh], 1)
    boxes = xywh * scale
    scores = classes.sigmoid()
    decoded = torch.cat((boxes, scores), 1)
    return {"probabilities": probabilities, "distances": distances, "xy1": xy1, "xy2": xy2,
            "xywh": xywh, "boxes": boxes, "scores": scores, "decoded": decoded}


def bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_lifted_states(torch: Any, left: Any, right: Any) -> dict:
    a, b = left.state_dict(), right.state_dict()
    require(a.keys() == b.keys(), "Branch state keys differ")
    counts = Counter()
    for key in a:
        require(a[key].shape == b[key].shape and torch.equal(a[key].float(), b[key].float()),
                f"Branch state values differ after float32 lift: {key}")
        counts["floating_tensors" if a[key].is_floating_point() else "nonfloating_tensors"] += 1
    return {"equal_after_float32_lift": True, "state_tensors": len(a), **counts}


def historical_inputs(protocol: dict, output: Path) -> tuple[Any, Path, list[dict], dict]:
    qualifier_path = bound_identity(protocol["qualifier"], "qualifier")
    qualification = Path(protocol["qualification_dir"]).resolve(strict=True)
    require(qualification.is_dir(), "Invalid qualification root")
    manifest_path = bound_identity(protocol["qualification_manifest"], "qualification manifest")
    require(manifest_path == qualification / "manifest.json", "Qualification manifest/root mismatch")
    manifest = read_json(manifest_path)
    require(isinstance(manifest.get("files"), dict) and bool(manifest["files"]), "Empty qualification manifest")
    # Byte hashes only: annotation content is never parsed or used by the probe.
    for relative, expected in manifest["files"].items():
        path = safe_file(qualification, relative)
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], f"Historical file changed: {relative}")
    receipt = read_json(safe_file(qualification, "qualification_receipt.json"))
    require(receipt.get("status") == "COMPLETE_DEBUG_FLOAT_QUALIFICATION_REQUIRES_REVIEW"
            and receipt.get("images_completed") == 109 and receipt.get("holdout_accessed") is False
            and receipt.get("selection_detector_inference") is False, "Historical qualification incomplete/forbidden")
    rows = read_json(safe_file(qualification, "debug_source_manifest.json"))["rows"]
    selected = validate_protocol(protocol, sha(Path(__file__)), rows)
    checkpoint = bound_identity(protocol["checkpoint"], "checkpoint")
    require(sha(checkpoint) == sha(safe_file(qualification, "best.pt")), "Checkpoint differs from historical raw source")
    environment_path = bound_identity(protocol["runtime_environment"], "runtime environment")
    require(environment_path == qualification / "environment.json", "Wrong historical runtime identity")
    environment = read_json(environment_path)
    require(protocol["package_versions"] == environment["packages"], "Protocol/historical packages differ")
    require(all(protocol["framework_source_sha256"].get(k) == v["sha256"] for k, v in environment["framework_sources"].items()),
            "Protocol omits/changes historical framework source")
    for name in ("frozen_training_contract.json", "source_review.json"):
        shutil.copyfile(safe_file(qualification, name), output / name)
    spec = importlib.util.spec_from_file_location("frozen_gate1_qualifier_for_decode_probe", qualifier_path)
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    return qualifier, qualification, selected, {"checkpoint": checkpoint, "historical_environment": environment,
                                             "manifest_identity": identity(manifest_path)}


def run(protocol: dict, output: Path, receipt: dict) -> None:
    q, historical, rows, verified = historical_inputs(protocol, output)
    receipt["phase"] = "RUNTIME_CONFIGURATION"
    write_json(output / "receipt.json", receipt)
    torch, cv2, LetterBox, _, _, environment = q.configure_runtime(output)
    require(environment["packages"] == protocol["package_versions"], "Live package identities changed")
    for key, expected in CONTROLS.items():
        require(environment.get(key) == expected, f"Live precision control differs: {key}")
    import ultralytics
    framework_root = Path(ultralytics.__file__).resolve().parent
    for relative, expected in protocol["framework_source_sha256"].items():
        require(sha(safe_file(framework_root, relative)) == expected, f"Live framework source changed: {relative}")
    require(environment["cuda"] == verified["historical_environment"]["cuda"]
            and environment["cudnn"] == verified["historical_environment"]["cudnn"]
            and environment["gpu"] == verified["historical_environment"]["gpu"], "CUDA/cuDNN/GPU changed")
    require(environment["executable"]["sha256"] == verified["historical_environment"]["executable"]["sha256"],
            "Python executable changed")
    write_json(output / "environment.json", environment)
    models, provenance = q.load_models(torch, verified["checkpoint"])
    state_check = compare_lifted_states(torch, models["fp32"], models["fp16"])
    heads = {key: value.model[-1] for key, value in models.items()}
    head_properties = {}
    for branch, head in heads.items():
        require(head.nc == 10 and head.nl == 3 and head.reg_max > 1 and not head.dynamic
                and not head.end2end and not head.xyxy and not head.export, "Unsupported frozen detection head")
        require(torch.equal(head.dfl.conv.weight.flatten().float(),
                            torch.arange(head.reg_max, device="cuda:0", dtype=torch.float32)), "DFL weights are not fixed bin indices")
        head_properties[branch] = {"nc": int(head.nc), "nl": int(head.nl), "reg_max": int(head.reg_max),
                                  "saved_shape": list(head.shape) if head.shape is not None else None,
                                  "cached_anchor_dtype": str(head.anchors.dtype), "cached_stride_dtype": str(head.strides.dtype)}
    require(head_properties["fp32"]["reg_max"] == head_properties["fp16"]["reg_max"], "Head DFL dimensions differ")
    reg_max = int(heads["fp32"].reg_max)
    write_json(output / "model_provenance.json", {"checkpoint": identity(verified["checkpoint"]), "models": provenance,
               "branch_state_equality": state_check, "heads": head_properties, "fusion_called": False})
    letterbox = LetterBox((640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32,
                         padding_value=114, interpolation=cv2.INTER_LINEAR)
    raw_dir, per_dir = output / "raw", output / "per_image"
    raw_dir.mkdir(); per_dir.mkdir()
    items = []
    receipt.update(phase="DEBUG_INFERENCE_AND_FUNCTIONAL_DECODE", inference_started=True)
    write_json(output / "receipt.json", receipt)
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for number, row in enumerate(rows, 1):
            stem = Path(row["image_path"]).stem
            old_detail = read_json(safe_file(historical, f"per_image/{stem}.json"))
            require(old_detail["image_id"] == row["image_id"], "Historical image metadata mismatch")
            image_path = safe_file(historical, "debug/images/" + Path(row["image_path"]).name)
            require(sha(image_path) == row["image_sha256"], "Historical debug image changed")
            image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(image is not None and image.shape == (row["height"], row["width"], 3), "Decoded image dimensions differ")
            params = letterbox.get_params({"img": image})
            padded = letterbox.apply_image({"img": image}, params)["img"]
            require(padded.shape == (640, 640, 3), "Letterbox geometry changed")
            rgb_chw = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))
            normalized = torch.from_numpy(rgb_chw).unsqueeze(0).float().div(255)
            uint8_sha = hashlib.sha256(rgb_chw.tobytes()).hexdigest()
            normalized_sha = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
            require(uint8_sha == old_detail["preprocessed_uint8_sha256"]
                    and normalized_sha == old_detail["normalized_fp32_input_sha256"], "Historical preprocessing bytes differ")
            serial_params = {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()}
            require(serial_params == old_detail["letterbox_params"], "Historical letterbox parameters differ")
            tensor = normalized.to("cuda:0")
            captured, arrays = {}, {"input_fp32_cpu": normalized.numpy().copy()}
            for branch, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
                out = models[branch](tensor if branch == "fp32" else tensor.half(), augment=False)
                require(isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict), "Unsupported pinned forward result")
                decoded, logits = out
                require(isinstance(decoded, torch.Tensor) and decoded.dtype == dtype
                        and tuple(decoded.shape) == (1, 14, 8400), "Forward dtype/shape changed")
                require(set(logits) == {"boxes", "scores", "feats"}, "Unexpected forward-head schema")
                require(tuple(logits["boxes"].shape) == (1, 4 * reg_max, 8400)
                        and tuple(logits["scores"].shape) == (1, 10, 8400), "Wrong captured logits")
                head = heads[branch]
                require(head.anchors.dtype == dtype and head.strides.dtype == dtype, "Runtime anchor/stride cache has wrong dtype")
                captured[branch] = {"forward": decoded, "box_logits": logits["boxes"], "class_logits": logits["scores"],
                                    "anchors": head.anchors.clone(), "strides": head.strides.clone()}
                for name, value in captured[branch].items():
                    require(bool(torch.isfinite(value).all()), "Nonfinite forward capture")
                    arrays[f"{branch}_{name}"] = value.detach().cpu().numpy().copy()
                arrays[f"{branch}_dfl_bins"] = head.dfl.conv.weight.detach().flatten().cpu().numpy().copy()
            torch.cuda.synchronize()
            geometry_equal = all(torch.equal(captured["fp32"][key].float(), captured["fp16"][key].float())
                                 for key in ("anchors", "strides"))
            require(geometry_equal, "Runtime branch geometry differs after float32 lift")
            native = {}
            for source in ("fp32", "fp16"):
                for arithmetic, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
                    values = decode_functional(torch, captured[source]["box_logits"], captured[source]["class_logits"],
                                               captured[source]["anchors"], captured[source]["strides"], reg_max, dtype)
                    prefix = f"source_{source}_decode_{arithmetic}"
                    if source == arithmetic:
                        native[source] = bool(torch.equal(values["decoded"], captured[source]["forward"]))
                    for name, value in values.items():
                        require(bool(torch.isfinite(value).all()), "Nonfinite functional decoder output")
                        arrays[f"{prefix}_{name}"] = value.detach().cpu().numpy().copy()
            np.savez_compressed(raw_dir / f"{stem}.npz", **arrays)
            with np.load(safe_file(historical, f"raw/{stem}.npz"), allow_pickle=False) as old_raw:
                old_equal = {branch: bytes_equal(arrays[f"{branch}_forward"], old_raw[branch]) for branch in ("fp32", "fp16")}
            comparisons = {name: q.raw_guard(arrays[left], arrays[right]) for name, (left, right) in COMPARISONS.items()}
            detail = {"image_id": row["image_id"], "image_sha256": row["image_sha256"], "reg_max": reg_max,
                      "array_file": f"raw/{stem}.npz",
                      "nc": 10, "nl": 3, "normalized_fp32_input_sha256": normalized_sha,
                      "preprocessed_uint8_sha256": uint8_sha, "letterbox_params": serial_params,
                      "historical_byte_equal": old_equal, "native_reconstruction_exact": native,
                      "runtime_geometry_equal_after_float_lift": geometry_equal, "comparisons": comparisons,
                      "old_guard_replaced": False, "tolerances_changed": False}
            write_json(per_dir / f"{stem}.json", detail)
            require(all(old_equal.values()), "Historical forward bytes differ; attempt stopped and preserved")
            require(set(native) == {"fp32", "fp16"} and all(native.values()), "Native functional reconstruction differs; attempt stopped and preserved")
            items.append(detail)
            receipt["images_completed"] = number
            write_json(output / "receipt.json", receipt)
            print(f"DECODE_PROBE {number}/{len(rows)} {row['image_id']} NATIVE_AND_HISTORY_EXACT", flush=True)
    # Recheck immutable inputs after processing, without changing the old manifest.
    bound_identity(protocol["qualification_manifest"], "qualification manifest after execution")
    bound_identity(protocol["checkpoint"], "checkpoint after execution")
    bound_identity(protocol["qualifier"], "qualifier after execution")
    bound_identity(protocol["runtime_environment"], "historical runtime after execution")
    summary = {"schema": "rampq.gate1.decode_precision_probe_summary/v1", "stage": protocol["stage"],
               "images": len(items), "historical_bytes_all_equal": True, "native_reconstruction_all_exact": True,
               "comparisons": {}, "raw_tolerances": TOLERANCES, "scientific_gate_pass": False,
               "original_guard_replaced": False, "limits": LIMITS}
    for name in COMPARISONS:
        guards = [item["comparisons"][name] for item in items]
        summary["comparisons"][name] = {"diagnostic_guard_pass_images": sum(g["allclose_pass"] for g in guards),
            "diagnostic_guard_fail_images": sum(not g["allclose_pass"] for g in guards),
            **{section: {"violations": sum(g[section]["violations"] for g in guards),
                         "max_absolute_error": max(g[section]["max_absolute_error"] for g in guards),
                         "max_error_over_tolerance": max(g[section]["max_error_over_tolerance"] for g in guards)}
               for section in ("boxes", "scores")}}
    write_json(output / "summary.json", summary)
    receipt.update(status="COMPLETE_DIAGNOSTIC_ONLY", phase="COMPLETE", images_completed=len(items),
                   historical_bytes_all_equal=True, native_reconstruction_all_exact=True)


def seal(output: Path) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(output).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    write_json(output / "manifest.json", {"schema": "rampq.gate1.decode_precision_probe_manifest/v1", "files": files})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    require(sha(protocol_path) == args.protocol_sha256, "Protocol does not match its pre-execution hash")
    protocol = read_json(protocol_path)
    require(protocol.get("runner_sha256") == sha(Path(__file__)), "Protocol does not bind this exact runner")
    output = args.output.resolve()
    require(not output.exists(), "Output must be a new directory")
    qualification = Path(protocol["qualification_dir"]).resolve(strict=True)
    require(not output.is_relative_to(qualification) and not qualification.is_relative_to(output), "Output overlaps historical qualification")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output / "protocol.json")
    shutil.copyfile(Path(__file__), output / "runner_source.py")
    started = time.monotonic()
    receipt = {"schema": "rampq.gate1.decode_precision_probe_receipt/v1", "status": "RUNNING", "phase": "INPUT_VALIDATION",
               "started_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": args.protocol_sha256,
               "runner_sha256": sha(Path(__file__)), "stage": protocol.get("stage"), "images_requested": len(protocol.get("image_ids", [])),
               "images_completed": 0, "inference_started": False, "data_role": "debug", "selection_detector_inference": False,
               "holdout_accessed": False, "scientific_gate_pass": False, "accepted_float_baseline_ready": False,
               "old_guard_replaced": False, "tolerances_changed": False, "limits": LIMITS}
    write_json(output / "receipt.json", receipt)
    try:
        run(protocol, output, receipt)
    except Exception as error:
        receipt.update(status="FAILED_STOPPED_PRESERVED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "receipt.json", receipt)
        seal(output)


if __name__ == "__main__":
    main()
