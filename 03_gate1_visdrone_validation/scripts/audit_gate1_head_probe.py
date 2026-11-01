"""Audit saved head-crossing arrays; no model, images, GPU, NMS or AP.

This verifies captured-array identities, arithmetic stages and all fixed guards.
It does not independently execute a convolution or certify causal head causes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DECODE_AUDITOR_SHA = "04a7d2256ab0a3d90025b2453ef2bc44b043e8b1493b7866c9acc603838dc1ef"
DECODE_RUNNER_SHA = "f76f3bfd677487fa7d87947f866daaf7407ddd455ad3e4d9e399da719e481ec0"
_HELPER_PATH = ROOT / "scripts/audit_gate1_decode_probe.py"
if hashlib.sha256(_HELPER_PATH.read_bytes()).hexdigest() != DECODE_AUDITOR_SHA:
    raise ValueError("Frozen NumPy decode-auditor helper changed")
_SPEC = importlib.util.spec_from_file_location("frozen_decode_auditor_for_head_saved_arrays", _HELPER_PATH)
D = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D)
COMPARISONS = {
    "original_fp32_vs_original_fp16": ("fp32_forward", "fp16_forward"),
    "native_heads_both_decode_fp32": ("source_fp32_head_fp32_decoded", "source_fp16_head_fp16_decoded"),
    "same_fp32_features_head_fp32_vs_fp16": ("source_fp32_head_fp32_decoded", "source_fp32_head_fp16_decoded"),
    "same_fp16_features_head_fp32_vs_fp16": ("source_fp16_head_fp32_decoded", "source_fp16_head_fp16_decoded"),
    "same_fp32_head_features_fp32_vs_fp16": ("source_fp32_head_fp32_decoded", "source_fp16_head_fp32_decoded"),
    "same_fp16_head_features_fp32_vs_fp16": ("source_fp32_head_fp16_decoded", "source_fp16_head_fp16_decoded"),
    "reference_fp32_vs_fp32_features_fp16_head": ("source_fp32_head_fp32_decoded", "source_fp32_head_fp16_decoded"),
    "reference_fp32_vs_fp16_features_fp32_head": ("source_fp32_head_fp32_decoded", "source_fp16_head_fp32_decoded"),
}
INTERVENTION = {"feature_sources": ["fp32", "fp16"], "head_compute_dtypes": ["fp32", "fp16"],
                "all_comparisons_decode_dtype": "fp32", "native_head_logits_exact_required": True,
                "reference_decode_logits_exact_required": True}
EXACT_REPORT_GROUPS = ("historical_byte_equal", "reference_decode_capture_exact", "reference_unified_decode_exact",
                       "native_head_logits_exact", "features_unchanged_after_intervention", "head_state_unchanged_after_intervention")
LIMITS = [
    "Saved arrays and metadata only; no convolution, BatchNorm, model, pixel decoding, GPU, NMS, annotations, AP, Selection or Holdout computation.",
    "Feature hashes bind saved feature bytes to reported before/after hashes; head-state immutability remains a reviewed-runner assertion, not an independent model-state deserialization.",
    "Nonlinear NumPy decode equations are approximate sanity checks; exact raw identity and fixed guards are separate.",
    "Cross-head logits are independently checked for shape, dtype and finite values but are not regenerated from features and weights by this auditor.",
    "The FP32-feature to FP16-head cast is part of that intervention. A head contrast includes convolution, normalization and activation and does not isolate an individual operation.",
    "Two reported comparison keys alias existing contrasts; they are not additional independent observations.",
    "No favorable diagnostic outcome replaces the original failed dtype/Gate0.5 guards or accepts Gate1.",
]


def validate_scope(protocol: dict, cohort: list[dict]) -> dict:
    D.require(protocol.get("schema") == "rampq.gate1.head_precision_probe_protocol/v1", "Wrong head protocol schema")
    D.require(protocol.get("decoder_runner", {}).get("sha256") == DECODE_RUNNER_SHA and protocol.get("intervention") == INTERVENTION,
              "Head intervention or frozen decoder changed")
    adapted = dict(protocol)
    adapted["schema"] = "rampq.gate1.decode_precision_probe_protocol/v1"
    return D.validate_scope(adapted, cohort)


def feature_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def audit_arrays(arrays: dict, metadata: dict, reference: dict, historical: dict) -> dict:
    reg_max = metadata["reg_max"]
    D.require(type(reg_max) is int and 2 <= reg_max <= 128 and metadata["nc"] == 10 and metadata["nl"] == 3, "Wrong head dimensions")
    D.require(arrays["input_fp32_cpu"].dtype == np.float32 and arrays["input_fp32_cpu"].shape == (1, 3, 640, 640), "Wrong input shape/dtype")
    D.require(feature_hash(arrays["input_fp32_cpu"]) == metadata["normalized_fp32_input_sha256"]
              and D.exact_bytes(arrays["input_fp32_cpu"], reference["input_fp32_cpu"]), "Saved input differs from reference")
    for group in EXACT_REPORT_GROUPS:
        D.require(metadata.get(group) == {"fp32": True, "fp16": True}, f"Missing reported exactness: {group}")
    D.require(metadata.get("runtime_geometry_equal_after_float_lift") is True and metadata.get("old_guard_replaced") is False
              and metadata.get("tolerances_changed") is False, "Image report changes scope")
    features = {}
    for source in ("fp32", "fp16"):
        dtype = np.dtype("float32" if source == "fp32" else "float16")
        for key, shape in (("forward", (1, 14, 8400)), ("box_logits", (1, 4 * reg_max, 8400)), ("class_logits", (1, 10, 8400)),
                           ("anchors", (2, 8400)), ("strides", (1, 8400)), ("dfl_bins", (reg_max,))):
            value = arrays[source + "_" + key]
            D.require(value.shape == shape and value.dtype == dtype and bool(np.isfinite(value).all()), f"Invalid native capture: {source}_{key}")
            D.require(D.exact_bytes(value, reference[source + "_" + key]), f"Reference native capture differs: {source}_{key}")
        D.require(D.exact_bytes(arrays[source + "_forward"], historical[source]), "Historical forward bytes differ")
        anchor, stride = D.expected_geometry(dtype)
        D.require(D.exact_bytes(arrays[source + "_anchors"], anchor) and D.exact_bytes(arrays[source + "_strides"], stride), "Anchor order/strides differ")
        D.require(D.exact_bytes(arrays[source + "_dfl_bins"], np.arange(reg_max, dtype=dtype)), "Actual DFL bins differ")
        hashes, shapes = [], []
        for scale, length in enumerate((80, 40, 20)):
            value = arrays[f"{source}_feat_scale{scale}"]
            D.require(value.ndim == 4 and value.shape[0] == 1 and value.shape[1] > 0 and value.shape[2:] == (length, length)
                      and value.dtype == dtype and bool(np.isfinite(value).all()), "Feature shape/dtype/finite mismatch")
            hashes.append(feature_hash(value)); shapes.append(list(value.shape))
        D.require(metadata["feature_shapes"][source] == shapes, "Reported feature shape differs")
        D.require(metadata["feature_sha256_before"][source] == hashes and metadata["feature_sha256_after"][source] == hashes,
                  "Saved feature bytes differ from before/after hashes")
        features[source] = {"sha256": hashes, "shapes": shapes}
    D.require(features["fp32"]["shapes"] == features["fp16"]["shapes"], "Feature-origin channel shapes differ")
    for key in ("anchors", "strides"):
        D.require(D.exact_bytes(arrays["fp32_" + key], arrays["fp16_" + key].astype(np.float32)), "Native branch geometry differs")
    checks = {}
    for source in ("fp32", "fp16"):
        for head in ("fp32", "fp16"):
            prefix = f"source_{source}_head_{head}_"
            dtype = np.dtype("float32" if head == "fp32" else "float16")
            for logit, count in (("box_logits", 4 * reg_max), ("class_logits", 10)):
                value = arrays[prefix + logit]
                D.require(value.shape == (1, count, 8400) and value.dtype == dtype and bool(np.isfinite(value).all()), "Intervention logit shape/dtype/finite mismatch")
                if source == head:
                    D.require(D.exact_bytes(value, arrays[source + "_" + logit]), "Native head logits differ")
            if source == head:
                D.require(D.exact_bytes(arrays[prefix + "decoded"], reference[f"source_{source}_decode_fp32_decoded"]), "Reference unified FP32 decode differs")
            alias = {"captured_box_logits": arrays[prefix + "box_logits"], "captured_class_logits": arrays[prefix + "class_logits"],
                     "captured_anchors": arrays[source + "_anchors"], "captured_strides": arrays[source + "_strides"]}
            for stage in ("probabilities", "distances", "xy1", "xy2", "xywh", "boxes", "scores", "decoded"):
                alias["source_captured_decode_fp32_" + stage] = arrays[prefix + stage]
            checks[prefix.removesuffix("_")] = D.arithmetic_check(alias, "captured", "fp32", reg_max)
    comparisons = {name: D.fixed_guard(arrays[left], arrays[right]) for name, (left, right) in COMPARISONS.items()}
    D.require(set(metadata["comparisons"]) == set(COMPARISONS), "Comparison inventory differs")
    logit_differences = {}
    for name, actual in comparisons.items():
        D.compare_reported_guard(actual, metadata["comparisons"][name])
        left, right = COMPARISONS[name]
        if not left.endswith("_decoded"):
            continue
        logit_differences[name] = {}
        for logit in ("box_logits", "class_logits"):
            a = arrays[left.removesuffix("decoded") + logit].astype(np.float64)
            b = arrays[right.removesuffix("decoded") + logit].astype(np.float64)
            difference = np.abs(a - b)
            value = {"elements": int(difference.size), "unequal_elements": int(np.count_nonzero(difference)),
                     "max_absolute_error": float(difference.max()), "mean_absolute_error": float(difference.mean())}
            D.require(metadata["logit_differences"][name][logit] == value, "Reported logit differences disagree")
            logit_differences[name][logit] = value
    return {"image_id": metadata["image_id"], "all_native_historical_reference_byte_checks": True,
            "features": features, "arithmetic_checks": checks, "comparisons": comparisons, "logit_differences": logit_differences}


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for name in COMPARISONS:
        guards = [row["comparisons"][name] for row in rows]
        result[name] = {"images": len(rows), "pass_images": sum(row["allclose_pass"] for row in guards),
                        "fail_images": sum(not row["allclose_pass"] for row in guards)}
        for section in ("boxes", "scores"):
            result[name][section] = {"elements": sum(row[section]["elements"] for row in guards),
                                    "violations": sum(row[section]["violations"] for row in guards),
                                    "max_absolute_error": max(row[section]["max_absolute_error"] for row in guards),
                                    "max_error_over_tolerance": max(row[section]["max_error_over_tolerance"] for row in guards)}
    return result


def check_summary(recomputed: dict, summary: dict, count: int) -> None:
    D.require(summary.get("schema") == "rampq.gate1.head_precision_probe_summary/v1" and summary.get("images") == count, "Wrong head summary/count")
    D.require(all(summary.get(key) is True for key in ("historical_bytes_all_equal", "reference_decode_captures_all_exact", "reference_unified_decode_all_exact",
                                                     "native_head_logits_all_exact", "features_and_head_states_unchanged")), "Summary exactness missing")
    D.require(summary.get("raw_tolerances") == D.RAW_TOLERANCES and summary.get("all_intervention_decode_dtype") == "fp32"
              and summary.get("original_guard_replaced") is False and summary.get("scientific_gate_pass") is False, "Summary changes scope")
    D.require(set(summary.get("comparisons", {})) == set(COMPARISONS), "Summary comparison inventory differs")
    for name, actual in recomputed.items():
        reported = summary["comparisons"][name]
        D.require(reported.get("diagnostic_guard_pass_images") == actual["pass_images"] and reported.get("diagnostic_guard_fail_images") == actual["fail_images"], "Summary comparison count differs")
        for section in ("boxes", "scores"):
            D.require(reported.get(section) == {key: actual[section][key] for key in ("violations", "max_absolute_error", "max_error_over_tolerance")}, "Summary comparison values differ")


def bound_identity(entry: dict) -> Path:
    path = Path(entry["path"]).resolve(strict=True)
    D.require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry["bytes"] and D.sha(path) == entry["sha256"], "Bound source identity differs")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe, protocol_path, output = args.probe.resolve(strict=True), args.protocol.resolve(strict=True), args.output.resolve()
    D.require(not output.exists() and D.sha(protocol_path) == args.protocol_sha256, "Output exists or protocol SHA differs")
    p = D.read_json(protocol_path)
    manifest = D.verify_manifest(probe)
    D.require(D.sha(D.safe_file(probe, "protocol.json")) == args.protocol_sha256, "Saved protocol differs")
    for key in ("qualification_manifest", "checkpoint", "qualifier", "runtime_environment", "decoder_runner", "reference_decode_probe_manifest"):
        bound_identity(p[key])
    D.require(D.sha(D.safe_file(probe, "runner_source.py")) == p["runner_sha256"], "Runner source differs")
    historical_root = Path(p["qualification_dir"]).resolve(strict=True)
    reference_root = Path(p["reference_decode_probe_dir"]).resolve(strict=True)
    for root in (probe, historical_root, reference_root):
        D.require(not output.is_relative_to(root) and not root.is_relative_to(output), "Output overlaps immutable input")
    D.require(p["qualification_manifest"]["path"] == str(historical_root / "manifest.json")
              and p["reference_decode_probe_manifest"]["path"] == str(reference_root / "manifest.json"), "Manifest/root association differs")
    reference_manifest = D.verify_manifest(reference_root)
    reference_protocol = D.read_json(reference_root / "protocol.json")
    D.require(reference_protocol["image_ids"] == p["image_ids"] and reference_protocol["checkpoint"]["sha256"] == p["checkpoint"]["sha256"], "Reference cohort/checkpoint differs")
    cohort = D.read_json(D.safe_file(historical_root, "debug_source_manifest.json"))["rows"]
    by_id = validate_scope(p, cohort)
    receipt = D.read_json(D.safe_file(probe, "receipt.json"))
    D.require(receipt.get("status") == "COMPLETE_HEAD_DIAGNOSTIC_ONLY" and receipt.get("phase") == "COMPLETE"
              and receipt.get("protocol_sha256") == args.protocol_sha256 and receipt.get("runner_sha256") == p["runner_sha256"]
              and receipt.get("images_completed") == len(p["image_ids"]) and receipt.get("scientific_gate_pass") is False
              and receipt.get("old_guard_replaced") is False and receipt.get("tolerances_changed") is False
              and receipt.get("selection_detector_inference") is False and receipt.get("holdout_accessed") is False, "Head receipt incomplete or expands evidence")
    historical_manifest = D.read_json(historical_root / "manifest.json")["files"]
    metadata_paths = sorted((probe / "per_image").glob("*.json"))
    D.require(len(metadata_paths) == len(p["image_ids"]), "Head output image count differs")
    rows = []
    for meta_path in metadata_paths:
        metadata = D.read_json(meta_path)
        D.require(metadata["image_id"] in by_id and metadata["array_file"] == f"raw/{meta_path.stem}.npz", "Head image/raw association differs")
        history_meta_relative = f"per_image/{meta_path.stem}.json"
        history_raw_relative = f"raw/{meta_path.stem}.npz"
        for relative in (history_meta_relative, history_raw_relative):
            path = D.safe_file(historical_root, relative); entry = historical_manifest[relative]
            D.require(path.stat().st_size == entry["bytes"] and D.sha(path) == entry["sha256"], "Historical file changed")
        old_metadata = D.read_json(historical_root / history_meta_relative)
        ref_metadata = D.read_json(D.safe_file(reference_root, history_meta_relative))
        for key in ("image_id", "normalized_fp32_input_sha256", "preprocessed_uint8_sha256", "letterbox_params"):
            D.require(metadata[key] == old_metadata[key] == ref_metadata[key], "Historical/reference input metadata differs")
        with np.load(D.safe_file(probe, metadata["array_file"]), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        with np.load(D.safe_file(reference_root, history_raw_relative), allow_pickle=False) as archive:
            reference = {key: archive[key] for key in archive.files}
        with np.load(historical_root / history_raw_relative, allow_pickle=False) as archive:
            historical = {key: archive[key] for key in ("fp32", "fp16")}
        rows.append(audit_arrays(arrays, metadata, reference, historical))
    D.require(sorted(row["image_id"] for row in rows) == sorted(p["image_ids"]) and len({row["image_id"] for row in rows}) == len(rows), "Head image inventory differs")
    recomputed = aggregate(rows)
    check_summary(recomputed, D.read_json(D.safe_file(probe, "summary.json")), len(rows))
    output.mkdir(parents=True)
    summary = {"schema": "rampq.gate1.head_probe_saved_array_audit/v1", "status": "PASS_SAVED_HEAD_PROBE_ARRAYS_ONLY",
               "protocol_sha256": args.protocol_sha256, "probe_manifest": manifest, "reference_decode_manifest": reference_manifest,
               "images_verified": len(rows), "comparisons": recomputed, "scientific_gate_pass": False,
               "old_guard_replaced": False, "tolerances_changed": False, "selection_detector_inference": False, "holdout_accessed": False,
               "limits": LIMITS, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    D.write_json(output / "recomputed_summary.json", summary)
    D.write_json(output / "per_image_checks.json", rows)
    D.write_json(output / "audit_source_identity.json", {"head_auditor": D.identity(Path(__file__).resolve()), "frozen_numpy_decode_helper": D.identity(_HELPER_PATH)})
    D.write_json(output / "manifest.json", {"files": {item.name: {"bytes": item.stat().st_size, "sha256": D.sha(item)} for item in sorted(output.iterdir()) if item.is_file()}})
    print(summary["status"], "images", len(rows))


if __name__ == "__main__":
    main()
