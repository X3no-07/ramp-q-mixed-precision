"""Independently audit saved decode-probe arrays with NumPy; no model or pixels.

The original all-anchor guard is recomputed at its unchanged tolerances. NumPy
softmax/DFL checks below are explicitly approximate arithmetic sanity checks,
not bitwise CUDA or implementation-equivalence proofs and not new guard limits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


COMPARISONS = {
    "original_fp32_vs_original_fp16": ("fp32_forward", "fp16_forward"),
    "original_fp32_vs_fp16_logits_decode_fp32": ("fp32_forward", "source_fp16_decode_fp32_decoded"),
    "same_fp32_logits_decode_fp32_vs_fp16": ("source_fp32_decode_fp32_decoded", "source_fp32_decode_fp16_decoded"),
    "same_fp16_logits_decode_fp32_vs_fp16": ("source_fp16_decode_fp32_decoded", "source_fp16_decode_fp16_decoded"),
    "fp32_logits_vs_fp16_logits_both_decode_fp32": ("source_fp32_decode_fp32_decoded", "source_fp16_decode_fp32_decoded"),
}
RAW_TOLERANCES = {"box_rtol": .005, "box_atol": .5, "score_rtol": .02, "score_atol": .002}
LIMITS = [
    "Saved-array consistency only; no model, GPU, image decoding, NMS, or target matching is rerun.",
    "NumPy softmax and DFL checks are approximate arithmetic sanity checks, not bitwise GPU operation proofs.",
    "A failed original FP32/FP16 guard remains failed; no posthoc masking, tolerance revision, official AP, or scientific baseline acceptance.",
    "Debug images are reused engineering data. Selection detector inference and Holdout access remain forbidden.",
    "Interventions identify sensitivity under captured logits; they do not isolate individual convolution or normalization causes.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def safe_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    require(not rel.is_absolute() and bool(rel.parts) and ".." not in rel.parts, "Unsafe relative path")
    cursor = root
    for part in rel.parts:
        cursor /= part
        require(not cursor.is_symlink(), "Symbolic link forbidden")
    result = cursor.resolve(strict=True)
    require(result.is_relative_to(root) and result.is_file(), "File escapes root")
    return result


def verify_manifest(root: Path) -> dict:
    path = safe_file(root, "manifest.json")
    manifest = read_json(path)
    files = manifest["files"]
    require(isinstance(files, dict) and bool(files), "Empty manifest")
    actual = set()
    for item in root.rglob("*"):
        require(not item.is_symlink(), "Symbolic link forbidden")
        if item.is_file() and item != path:
            actual.add(item.relative_to(root).as_posix())
    require(actual == set(files), "Manifest membership mismatch")
    for relative, expected in files.items():
        item = safe_file(root, relative)
        require(item.stat().st_size == expected["bytes"] and sha(item) == expected["sha256"], f"Manifest mismatch: {relative}")
    return {"sha256": sha(path), "files_verified": len(files)}


def exact_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes(order="C") == right.tobytes(order="C")


def fixed_guard(reference: np.ndarray, comparison: np.ndarray) -> dict:
    require(reference.shape == comparison.shape == (1, 14, 8400), "Unexpected decoded shape")
    require(bool(np.isfinite(reference).all() and np.isfinite(comparison).all()), "Nonfinite decoded values")
    left, right = reference.astype(np.float64), comparison.astype(np.float64)
    result = {}
    for name, section, rtol, atol in (("boxes", slice(0, 4), .005, .5), ("scores", slice(4, 14), .02, .002)):
        error = np.abs(right[:, section] - left[:, section])
        bound = atol + rtol * np.abs(left[:, section])
        result[name] = {"rtol": rtol, "atol": atol, "elements": int(error.size), "violations": int((error > bound).sum()),
                        "max_absolute_error": float(error.max()), "max_error_over_tolerance": float((error / bound).max())}
    passed = result["boxes"]["violations"] == result["scores"]["violations"] == 0
    return {"status": "PASS" if passed else "FAIL_FIXED_TOLERANCES", "allclose_pass": passed, **result}


def compare_reported_guard(actual: dict, reported: dict) -> None:
    for key in ("status", "allclose_pass"):
        require(reported.get(key) == actual[key], f"Reported guard differs: {key}")
    for section in ("boxes", "scores"):
        for key, value in actual[section].items():
            require(reported.get(section, {}).get(key) == value, f"Reported guard differs: {section}.{key}")


def validate_scope(protocol: dict, cohort: list[dict]) -> dict[str, dict]:
    require(protocol.get("schema") == "rampq.gate1.decode_precision_probe_protocol/v1", "Wrong protocol schema")
    require(protocol.get("data_role") == "debug" and protocol.get("selection_detector_inference") is False
            and protocol.get("holdout_accessed") is False and protocol.get("raw_tolerances") == RAW_TOLERANCES,
            "Protocol changes evidence scope or tolerances")
    require(protocol.get("historical_equality_required") is True and protocol.get("native_reconstruction_required") is True,
            "Protocol disables exactness")
    by_id = {row["image_id"]: row for row in cohort}
    require(len(by_id) == len(cohort) == 109 and all(row.get("assigned_split") == "debug" and row.get("source_split") == "train"
            and row["image_id"].startswith("train/") for row in cohort), "Invalid debug cohort")
    ids = protocol["image_ids"]
    require(isinstance(ids, list) and len(ids) == len(set(ids)) and all(image_id in by_id for image_id in ids), "Invalid requested IDs")
    require((protocol.get("stage") == "pilot3" and len(ids) == 3)
            or (protocol.get("stage") == "full109" and ids == [row["image_id"] for row in cohort]), "Protocol stage/count/order differs")
    return by_id


def expected_geometry(dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    anchors, strides = [], []
    for length, step in ((80, 8), (40, 16), (20, 32)):
        y, x = np.meshgrid(np.arange(length, dtype=dtype) + .5, np.arange(length, dtype=dtype) + .5, indexing="ij")
        anchors.append(np.stack((x.reshape(-1), y.reshape(-1))))
        strides.append(np.full((1, length * length), step, dtype=dtype))
    return np.concatenate(anchors, axis=1).astype(dtype), np.concatenate(strides, axis=1)


def arithmetic_check(arrays: dict[str, np.ndarray], source: str, decode: str, reg_max: int) -> dict:
    """No model import: check stages and approximately check nonlinear math."""
    dtype = np.dtype("float32" if decode == "fp32" else "float16")
    prefix = f"source_{source}_decode_{decode}_"
    shapes = {"probabilities": (1, reg_max, 4, 8400), "distances": (1, 4, 8400), "xy1": (1, 2, 8400),
              "xy2": (1, 2, 8400), "xywh": (1, 4, 8400), "boxes": (1, 4, 8400),
              "scores": (1, 10, 8400), "decoded": (1, 14, 8400)}
    values = {}
    for key, shape in shapes.items():
        value = arrays[prefix + key]
        require(value.shape == shape and value.dtype == dtype and bool(np.isfinite(value).all()), f"Stage shape/dtype/finite mismatch: {prefix + key}")
        values[key] = value
    anchors = arrays[source + "_anchors"].astype(dtype)[None]
    strides = arrays[source + "_strides"].astype(dtype)
    distances = values["distances"]
    rounding_allowance = 8 * np.finfo(dtype).eps * reg_max
    require(bool((distances >= -rounding_allowance).all() and (distances <= reg_max - 1 + rounding_allowance).all()), "DFL distance outside bin range allowing rounding")
    expected_xy1 = anchors - distances[:, :2]
    expected_xy2 = anchors + distances[:, 2:]
    expected_xywh = np.concatenate(((expected_xy1 + expected_xy2) / np.array(2, dtype=dtype), expected_xy2 - expected_xy1), axis=1)
    # Elementwise addition/subtraction/division/multiplication stages have no
    # reduction; fp16 rounding must match saved native stage order exactly.
    require(exact_bytes(values["xy1"], expected_xy1) and exact_bytes(values["xy2"], expected_xy2), "Corner operation order differs")
    require(exact_bytes(values["xywh"], expected_xywh), "Center/width-height arithmetic differs")
    require(exact_bytes(values["boxes"], expected_xywh * strides), "Stride multiplication differs")
    require(exact_bytes(values["decoded"], np.concatenate((values["boxes"], values["scores"]), axis=1)), "Decoded concatenation differs")
    probs = values["probabilities"].astype(np.float64)
    require(bool((probs >= 0).all() and (probs <= 1).all()), "Invalid probability domain")
    prob_sum_error = float(np.abs(probs.sum(axis=1) - 1).max())
    half = decode == "fp16"
    require(prob_sum_error <= (.002 if half else .000005), "Softmax probabilities fail approximate normalization")
    logits = arrays[source + "_box_logits"].astype(dtype).astype(np.float64).reshape(1, 4, reg_max, 8400).transpose(0, 2, 1, 3)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    expected_probs = exponent / exponent.sum(axis=1, keepdims=True)
    require(bool(np.allclose(probs, expected_probs, rtol=.003 if half else .000003, atol=.000001 if half else .0000002)), "Softmax differs from approximate NumPy equation")
    bins = np.arange(reg_max, dtype=np.float64)[None, :, None, None]
    expected_distances = (probs * bins).sum(axis=1)
    dfl_error = float(np.abs(distances.astype(np.float64) - expected_distances).max())
    require(bool(np.allclose(distances, expected_distances, rtol=.002 if half else .000002, atol=.002 if half else .000002)), "DFL differs from approximate convex weighted sum")
    cls = arrays[source + "_class_logits"].astype(dtype).astype(np.float64)
    # Avoid overflow at large negative logits without modifying saved values.
    expected_scores = np.exp(-np.logaddexp(0, -cls))
    require(bool(np.allclose(values["scores"], expected_scores, rtol=.003 if half else .000003, atol=.000001 if half else .0000002)), "Sigmoid differs from approximate NumPy equation")
    return {"elementwise_stage_checks_exact": True, "probability_normalization_max_error": prob_sum_error,
            "softmax_max_absolute_error_vs_numpy_fp64": float(np.abs(probs - expected_probs).max()),
            "dfl_max_absolute_error_vs_numpy_fp64_from_saved_probs": dfl_error,
            "sigmoid_max_absolute_error_vs_numpy_fp64": float(np.abs(values["scores"].astype(np.float64) - expected_scores).max()),
            "nonlinear_checks_are_approximate_only": True}


def audit_arrays(arrays: dict[str, np.ndarray], metadata: dict, historical: dict[str, np.ndarray]) -> dict:
    reg_max = metadata["reg_max"]
    require(type(reg_max) is int and 2 <= reg_max <= 128 and metadata["nc"] == 10 and metadata["nl"] == 3, "Invalid head layout")
    require(arrays["input_fp32_cpu"].shape == (1, 3, 640, 640) and arrays["input_fp32_cpu"].dtype == np.float32, "Invalid saved input")
    require(bool(np.isfinite(arrays["input_fp32_cpu"]).all()) and bool(((arrays["input_fp32_cpu"] >= 0) & (arrays["input_fp32_cpu"] <= 1)).all()), "Invalid input domain")
    require(hashlib.sha256(arrays["input_fp32_cpu"].tobytes()).hexdigest() == metadata["normalized_fp32_input_sha256"], "Normalized input hash differs")
    for source, dtype in (("fp32", np.dtype("float32")), ("fp16", np.dtype("float16"))):
        for suffix, shape in (("forward", (1, 14, 8400)), ("box_logits", (1, 4 * reg_max, 8400)), ("class_logits", (1, 10, 8400)), ("anchors", (2, 8400)), ("strides", (1, 8400))):
            value = arrays[source + "_" + suffix]
            require(value.dtype == dtype and value.shape == shape and bool(np.isfinite(value).all()), f"Source shape/dtype/finite mismatch: {source}_{suffix}")
        anchor, stride = expected_geometry(dtype)
        bins = arrays[source + "_dfl_bins"]
        require(exact_bytes(bins, np.arange(reg_max, dtype=dtype)), "Saved DFL weight bins differ")
        require(exact_bytes(arrays[source + "_anchors"], anchor) and exact_bytes(arrays[source + "_strides"], stride), "Unexpected actual anchors/strides")
        require(exact_bytes(arrays[source + "_forward"], historical[source]), f"Historical raw bytes differ: {source}")
        require(exact_bytes(arrays[source + "_forward"], arrays[f"source_{source}_decode_{source}_decoded"]), f"Native reconstruction differs: {source}")
        require(metadata.get("historical_byte_equal", {}).get(source) is True and metadata.get("native_reconstruction_exact", {}).get(source) is True, "Reported exact identity absent")
    require(exact_bytes(arrays["fp32_anchors"], arrays["fp16_anchors"].astype(np.float32)) and exact_bytes(arrays["fp32_strides"], arrays["fp16_strides"].astype(np.float32)), "Runtime geometry changes with branch")
    require(metadata["runtime_geometry_equal_after_float_lift"] is True, "Reported runtime geometry identity absent")
    checks = {f"source_{source}_decode_{decode}": arithmetic_check(arrays, source, decode, reg_max) for source in ("fp32", "fp16") for decode in ("fp32", "fp16")}
    comparisons = {name: fixed_guard(arrays[left], arrays[right]) for name, (left, right) in COMPARISONS.items()}
    require(set(metadata["comparisons"]) == set(COMPARISONS), "Comparison inventory differs")
    for name, actual in comparisons.items():
        compare_reported_guard(actual, metadata["comparisons"][name])
    return {"image_id": metadata["image_id"], "historical_byte_equal": True, "native_reconstruction_exact": True,
            "geometry_exact": True, "arithmetic_checks": checks, "comparisons": comparisons}


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for name in COMPARISONS:
        status = Counter(row["comparisons"][name]["status"] for row in rows)
        sections = {}
        for section in ("boxes", "scores"):
            sections[section] = {"elements": sum(row["comparisons"][name][section]["elements"] for row in rows),
                                 "violations": sum(row["comparisons"][name][section]["violations"] for row in rows),
                                 "max_absolute_error": max(row["comparisons"][name][section]["max_absolute_error"] for row in rows),
                                 "max_error_over_tolerance": max(row["comparisons"][name][section]["max_error_over_tolerance"] for row in rows)}
        result[name] = {"images": len(rows), "pass_images": status["PASS"], "fail_images": status["FAIL_FIXED_TOLERANCES"], **sections}
    return result


def check_summary(recomputed: dict, summary: dict, count: int) -> None:
    require(summary.get("schema") == "rampq.gate1.decode_precision_probe_summary/v1" and summary.get("images") == count,
            "Summary schema/count differs")
    require(summary.get("historical_bytes_all_equal") is True and summary.get("native_reconstruction_all_exact") is True,
            "Summary exactness missing")
    require(summary.get("scientific_gate_pass") is False and summary.get("original_guard_replaced") is False
            and summary.get("raw_tolerances") == RAW_TOLERANCES, "Summary changes evidence scope or tolerances")
    require(set(summary.get("comparisons", {})) == set(COMPARISONS), "Summary comparison inventory differs")
    for name, actual in recomputed.items():
        reported = summary["comparisons"][name]
        require(reported.get("diagnostic_guard_pass_images") == actual["pass_images"]
                and reported.get("diagnostic_guard_fail_images") == actual["fail_images"], f"Summary status counts differ: {name}")
        for section in ("boxes", "scores"):
            for key in ("violations", "max_absolute_error", "max_error_over_tolerance"):
                require(reported.get(section, {}).get(key) == actual[section][key], f"Summary value differs: {name}.{section}.{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe, protocol_path, output = args.probe.resolve(strict=True), args.protocol.resolve(strict=True), args.output.resolve()
    require(not output.exists(), "Output directory already exists")
    require(sha(protocol_path) == args.protocol_sha256, "Protocol SHA differs")
    protocol = read_json(protocol_path)
    require(not output.is_relative_to(probe) and not probe.is_relative_to(output), "Audit output overlaps probe input")
    probe_manifest = verify_manifest(probe)
    require(sha(safe_file(probe, "protocol.json")) == args.protocol_sha256, "Saved protocol differs")
    for key in ("qualification_manifest", "checkpoint", "qualifier", "runtime_environment"):
        entry = protocol[key]
        path = Path(entry["path"]).resolve(strict=True)
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], f"Protocol binding differs: {key}")
    historical_root = Path(protocol["qualification_dir"]).resolve(strict=True)
    require(not output.is_relative_to(historical_root) and not historical_root.is_relative_to(output), "Audit output overlaps historical input")
    require(sha(safe_file(historical_root, "manifest.json")) == protocol["qualification_manifest"]["sha256"], "Historical qualification differs")
    runner_path = safe_file(probe, "runner_source.py")
    require(sha(runner_path) == protocol["runner_sha256"], "Runner source differs")
    receipt = read_json(safe_file(probe, "receipt.json"))
    require(receipt.get("status") == "COMPLETE_DIAGNOSTIC_ONLY" and receipt.get("phase") == "COMPLETE"
            and receipt.get("protocol_sha256") == args.protocol_sha256 and receipt.get("runner_sha256") == protocol["runner_sha256"]
            and receipt.get("images_completed") == len(protocol["image_ids"])
            and receipt.get("selection_detector_inference") is False and receipt.get("holdout_accessed") is False
            and receipt.get("scientific_gate_pass") is False and receipt.get("old_guard_replaced") is False
            and receipt.get("tolerances_changed") is False, "Receipt is incomplete or changes evidence scope")
    historical_manifest = read_json(historical_root / "manifest.json")["files"]
    cohort = read_json(safe_file(historical_root, "debug_source_manifest.json"))["rows"]
    by_id = validate_scope(protocol, cohort)
    rows = []
    metadata_paths = sorted((probe / "per_image").glob("*.json"))
    require(len(metadata_paths) == len(protocol["image_ids"]), "Probe image count differs")
    for meta_path in metadata_paths:
        metadata = read_json(meta_path)
        require(metadata.get("old_guard_replaced") is False and metadata.get("tolerances_changed") is False, "Image report changes guard")
        require(metadata["image_id"] in by_id and by_id[metadata["image_id"]]["assigned_split"] == "debug"
                and by_id[metadata["image_id"]]["source_split"] == "train", "Image role differs")
        raw_path = safe_file(probe, metadata["array_file"])
        history_relative = f"raw/{meta_path.stem}.npz"
        require(metadata["array_file"] == history_relative, "Raw file association differs")
        historical_detail_relative = f"per_image/{meta_path.stem}.json"
        historical_detail_path = safe_file(historical_root, historical_detail_relative)
        expected_detail = historical_manifest[historical_detail_relative]
        require(historical_detail_path.stat().st_size == expected_detail["bytes"] and sha(historical_detail_path) == expected_detail["sha256"], "Historical image metadata changed")
        historical_detail = read_json(historical_detail_path)
        for key in ("image_id", "normalized_fp32_input_sha256", "preprocessed_uint8_sha256", "letterbox_params"):
            require(metadata[key] == historical_detail[key], f"Historical input metadata differs: {key}")
        history_path = safe_file(historical_root, history_relative)
        expected = historical_manifest[history_relative]
        require(history_path.stat().st_size == expected["bytes"] and sha(history_path) == expected["sha256"], "Historical raw manifest mismatch")
        with np.load(raw_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        with np.load(history_path, allow_pickle=False) as archive:
            historical = {"fp32": archive["fp32"], "fp16": archive["fp16"]}
        rows.append(audit_arrays(arrays, metadata, historical))
    require(sorted(row["image_id"] for row in rows) == sorted(protocol["image_ids"]) and len({row["image_id"] for row in rows}) == len(rows), "Image identity inventory differs")
    recomputed = aggregate(rows)
    check_summary(recomputed, read_json(safe_file(probe, "summary.json")), len(rows))
    output.mkdir(parents=True)
    summary = {"schema": "rampq.gate1.decode_probe_saved_array_audit/v1", "status": "PASS_SAVED_ARRAY_CONSISTENCY_ONLY",
               "protocol_sha256": args.protocol_sha256, "probe_manifest": probe_manifest, "images_verified": len(rows),
               "comparisons": recomputed, "scientific_gate_pass": False, "old_gate05_guard_replaced": False,
               "tolerances_changed": False, "selection_detector_inference": False, "holdout_accessed": False, "limits": LIMITS,
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "recomputed_summary.json", summary)
    write_json(output / "per_image_checks.json", rows)
    write_json(output / "audit_source_identity.json", identity(Path(__file__).resolve()))
    write_json(output / "manifest.json", {"files": {item.name: {"bytes": item.stat().st_size, "sha256": sha(item)} for item in sorted(output.iterdir()) if item.is_file()}})
    print(json.dumps({"status": summary["status"], "images_verified": len(rows), "comparisons": summary["comparisons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
