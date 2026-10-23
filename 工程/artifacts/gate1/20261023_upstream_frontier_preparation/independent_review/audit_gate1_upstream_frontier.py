"""Audit saved DAG frontier interventions; never load models, images or CUDA.

Frontiers are independently derived from every dependency crossing the cut.
Saved tensors establish byte/cast/replay consistency, not independently
regenerated convolution arithmetic or unique operation-level causality.
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
HEAD_RUNNER_SHA = "842e16b7b743c6044975ab2faad5725ead19cd8c793575340c9ee81ddaaef23f"
_HELPER = ROOT / "scripts/audit_gate1_decode_probe.py"
if hashlib.sha256(_HELPER.read_bytes()).hexdigest() != DECODE_AUDITOR_SHA:
    raise ValueError("Frozen NumPy decoder auditor changed")
_SPEC = importlib.util.spec_from_file_location("frozen_decode_auditor_for_upstream_frontier", _HELPER)
D = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D)
CUTS = [-1, 4, 6, 10, 22]
EXPECTED_FRONTIERS = {"-1": [-1], "4": [4], "6": [4, 6], "10": [4, 6, 10], "22": [16, 19, 22]}
VARIANTS = [(cut, source, suffix) for cut in CUTS for source, suffix in (("fp32", "fp16"), ("fp16", "fp32"))]
INTERVENTION = {"prefix_sources": ["fp32", "fp16"], "suffix_compute_dtypes": ["fp32", "fp16"],
                "all_comparisons_decode_dtype": "fp32", "all_native_module_replay_exact_required": True,
                "all_same_dtype_cut_replay_exact_required": True, "frontier_policy": "all DAG edges crossing the cut",
                "cut_indices": CUTS, "stored_pairs": [["fp32", "fp16"], ["fp16", "fp32"]]}
SUMMARY_EXACT = ("historical_bytes_all_equal", "reference_head_captures_all_exact", "reference_unified_decode_all_exact",
                 "native_manual_replay_all_exact", "native_suffix_replay_all_exact", "frontiers_and_states_unchanged", "endpoint_aliases_all_exact")
LIMITS = [
    "Saved-array consistency only; no model, convolution, BatchNorm, image decoding, GPU, NMS, annotations, AP, Selection or Holdout computation.",
    "The complete DAG cut frontier is independently derived from frozen module f fields. Topology-to-real-model identity remains a reviewed runner/CPU-inspector binding.",
    "Native and suffix replay arrays are compared byte-for-byte; convolution outputs are not independently regenerated from weights by this auditor.",
    "Saved frontier hashes and casts are checked exactly. Model-state before/after hashes remain reviewed-runner assertions, not independent checkpoint deserialization.",
    "Nonlinear NumPy decoding checks are approximate sanity checks, separate from exact identities and unchanged fixed guards.",
    "Cuts are broad interacting arithmetic interventions, not an additive decomposition or proof of a unique layer cause. Input casts are included explicitly.",
    "No diagnostic contrast replaces historical failed guards, loosens their tolerances, or accepts Gate1. Reused debug images do not establish independent generalization.",
]


def prefix(cut: int, source: str, suffix: str) -> str:
    return f"cut_{cut}_source_{source}_suffix_{suffix}_"


def comparisons() -> dict:
    result = {"original_fp32_vs_original_fp16": ("fp32_forward", "fp16_forward"),
              "native_heads_both_decode_fp32": ("native_fp32_decoded", "native_fp16_decoded")}
    result.update({f"reference_fp32_vs_cut_{cut}_source_{source}_suffix_{suffix}":
                   ("native_fp32_decoded", prefix(cut, source, suffix) + "decoded") for cut, source, suffix in VARIANTS})
    return result


COMPARISONS = comparisons()


def dependencies(index: int, f: int | list[int]) -> list[int]:
    values = [f] if type(f) is int else f
    D.require(isinstance(values, list) and bool(values), "Missing dependency list")
    result = []
    for value in values:
        D.require(type(value) is int, "Noninteger dependency")
        source = index + value if value < 0 else value
        D.require(-1 <= source < index and (source != -1 or index == 0), "Invalid or forward dependency")
        result.append(source)
    return result


def derive_frontier(topology: dict, cut: int) -> list[int]:
    modules = topology.get("modules")
    D.require(isinstance(modules, list) and bool(modules) and type(cut) is int and -1 <= cut < len(modules) - 1, "Invalid cut/topology")
    frontier = set()
    expected_save = set()
    for index, module in enumerate(modules):
        D.require(module.get("i") == index and isinstance(module.get("type"), str) and module["type"], "Invalid module inventory")
        deps = dependencies(index, module.get("f"))
        original = module["f"] if isinstance(module["f"], list) else [module["f"]]
        expected_save.update(dep for dep, value in zip(deps, original) if value != -1)
        if index > cut:
            frontier.update(dep for dep in deps if dep <= cut)
    D.require(topology.get("save") == sorted(expected_save), "Pinned save cache differs from dependency inventory")
    return sorted(frontier)


def validate_topology(topology: dict) -> dict:
    modules = topology.get("modules", [])
    D.require(len(modules) == 24 and modules[-1].get("f") == [16, 19, 22] and modules[-1].get("type", "").endswith("Detect"), "Unexpected YOLO11n topology")
    actual = {str(cut): derive_frontier(topology, cut) for cut in CUTS}
    D.require(actual == EXPECTED_FRONTIERS, "Macro cut frontiers differ")
    return actual


def validate_scope(protocol: dict, cohort: list[dict]) -> dict:
    D.require(protocol.get("schema") == "rampq.gate1.upstream_frontier_probe_protocol/v1", "Wrong frontier schema")
    D.require(protocol.get("cuts") == CUTS and protocol.get("frontiers") == validate_topology(protocol["topology"]), "Cut/frontier protocol differs")
    D.require(protocol.get("decoder_runner", {}).get("sha256") == DECODE_RUNNER_SHA, "Wrong decoder runner")
    D.require(protocol.get("head_runner", {}).get("sha256") == HEAD_RUNNER_SHA and protocol.get("intervention") == INTERVENTION, "Wrong head runner/intervention")
    if protocol.get("stage") == "pilot3":
        D.require(protocol.get("image_ids") == ["train/0000170_01201_d_0000001", "train/9999990_00000_d_0000024", "train/0000046_00000_d_0000087"], "Frozen pilot selection/order differs")
    adapted = dict(protocol); adapted["schema"] = "rampq.gate1.decode_precision_probe_protocol/v1"
    return D.validate_scope(adapted, cohort)


def tensor_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def tensor_statistics(array: np.ndarray) -> dict:
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": tensor_hash(array), "elements": int(array.size),
            "min": float(array.min()), "max": float(array.max()), "mean": float(array.astype(np.float64).mean())}


def approximate_decode_check(box_logits: np.ndarray, class_logits: np.ndarray, decoded: np.ndarray, reg_max: int) -> dict:
    """FP64 NumPy equation sanity only; never a replacement raw-output guard.

    Box allowance covers FP32 softmax/integral and geometry reduction roundoff.
    It is frozen before new execution and cannot create a dtype-guard PASS.
    """
    D.require(decoded.shape == (1, 14, 8400) and decoded.dtype == np.float32 and np.isfinite(decoded).all(), "Wrong decoded shape/dtype/finite")
    logits = box_logits.astype(np.float64).reshape(1, 4, reg_max, 8400).transpose(0, 2, 1, 3)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True)); probs = exp / exp.sum(axis=1, keepdims=True)
    distances = (probs * np.arange(reg_max, dtype=np.float64)[None, :, None, None]).sum(axis=1)
    anchors, strides = D.expected_geometry(np.dtype("float32"))
    xy1, xy2 = anchors[None].astype(np.float64) - distances[:, :2], anchors[None].astype(np.float64) + distances[:, 2:]
    boxes = np.concatenate(((xy1 + xy2) / 2, xy2 - xy1), axis=1) * strides.astype(np.float64)
    scores = np.exp(-np.logaddexp(0, -class_logits.astype(np.float64)))
    box_error, score_error = np.abs(decoded[:, :4].astype(np.float64) - boxes), np.abs(decoded[:, 4:].astype(np.float64) - scores)
    D.require(np.all(box_error <= .002 + .00001 * np.abs(boxes)), "Decoded boxes differ from approximate NumPy FP64 equation")
    D.require(np.all(score_error <= .0000002 + .000003 * np.abs(scores)), "Decoded scores differ from approximate NumPy sigmoid")
    return {"nonlinear_checks_are_approximate_only": True, "box_rtol_sanity_only": .00001, "box_atol_sanity_only": .002,
            "score_rtol_sanity_only": .000003, "score_atol_sanity_only": .0000002,
            "boxes_max_absolute_error_vs_numpy_fp64": float(box_error.max()), "scores_max_absolute_error_vs_numpy_fp64": float(score_error.max())}


def audit_arrays(arrays: dict, metadata: dict, reference: dict, historical: dict, topology: dict) -> dict:
    frontiers = validate_topology(topology)
    reg_max = metadata["reg_max"]
    D.require(type(reg_max) is int and 2 <= reg_max <= 128 and metadata["nc"] == 10 and metadata["nl"] == 3, "Wrong detector dimensions")
    inp = arrays["input_fp32_cpu"]
    D.require(inp.shape == (1, 3, 640, 640) and inp.dtype == np.float32 and np.isfinite(inp).all()
              and ((inp >= 0) & (inp <= 1)).all(), "Wrong input shape/dtype/domain")
    D.require(D.exact_bytes(inp, reference["input_fp32_cpu"]) and tensor_hash(inp) == metadata["normalized_fp32_input_sha256"], "Input identity differs")
    for group in ("historical_byte_equal", "reference_head_capture_exact", "reference_unified_decode_exact", "native_manual_replay_exact"):
        D.require(metadata.get(group) == {"fp32": True, "fp16": True}, "Missing exact replay/capture report: " + group)
    D.require(metadata.get("native_suffix_replay_exact") == {str(cut): {"fp32": True, "fp16": True} for cut in CUTS}, "Missing native suffix exactness")
    D.require(metadata.get("endpoint_aliases_exact") == {"cut22_fp32_to_fp16": True, "cut22_fp16_to_fp32": True, "input_fp32_to_fp16": True}, "Endpoint alias exactness absent")
    D.require(metadata.get("model_states_unchanged") == {"fp32": True, "fp16": True}, "Missing model-state immutability report")
    D.require(metadata.get("runtime_geometry_equal_after_float_lift") is True and metadata.get("old_guard_replaced") is False
              and metadata.get("tolerances_changed") is False, "Metadata changes scope")
    union = sorted({index for values in frontiers.values() for index in values})
    native_frontiers = {}
    for source in ("fp32", "fp16"):
        dtype = np.dtype("float32" if source == "fp32" else "float16")
        module_keys = {str(index) for index in range(23)}
        D.require(metadata["native_manual_module_replay_exact"][source] == {key: True for key in module_keys}
                  and set(metadata["native_manual_module_replay_sha256"][source]) == module_keys, "Incomplete native per-module exact replay")
        stats = metadata["native_layer_statistics"][source]
        D.require(set(stats) == {str(index) for index in range(-1, 23)}, "Incomplete native layer statistic inventory")
        D.require(all(metadata["native_manual_module_replay_sha256"][source][key] == stats[key]["sha256"] for key in module_keys), "Native per-module replay hashes differ")
        for key, shape in (("forward", (1, 14, 8400)), ("box_logits", (1, 4 * reg_max, 8400)), ("class_logits", (1, 10, 8400)),
                           ("anchors", (2, 8400)), ("strides", (1, 8400)), ("dfl_bins", (reg_max,))):
            value = arrays[source + "_" + key]
            D.require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), "Invalid native capture: " + key)
            D.require(D.exact_bytes(value, reference[source + "_" + key]), "Reference capture bytes differ")
            if key in ("forward", "box_logits", "class_logits"):
                D.require(D.exact_bytes(arrays["manual_" + source + "_" + key], value), "Manual full replay bytes differ")
                D.require(metadata["native_manual_replay_sha256"][source][key] == tensor_hash(value), "Manual full replay reported hash differs")
        D.require(D.exact_bytes(arrays[source + "_forward"], historical[source]), "Historical output differs")
        anchors, strides = D.expected_geometry(dtype)
        D.require(D.exact_bytes(arrays[source + "_anchors"], anchors) and D.exact_bytes(arrays[source + "_strides"], strides)
                  and D.exact_bytes(arrays[source + "_dfl_bins"], np.arange(reg_max, dtype=dtype)), "Native geometry/bins differ")
        D.require(D.exact_bytes(arrays["native_" + source + "_decoded"], reference[f"source_{source}_head_{source}_decoded"]), "Unified reference decode differs")
        native_frontiers[source] = {}
        for index in union:
            value = arrays[f"native_{source}_layer_{index}"]
            D.require(value.ndim == 4 and value.shape[0] == 1 and min(value.shape[1:]) > 0 and value.dtype == dtype and np.isfinite(value).all(), "Invalid native frontier tensor")
            if index == -1:
                D.require(D.exact_bytes(value, inp.astype(dtype)), "Input frontier is not exact branch input cast")
            if index in (16, 19, 22):
                D.require(D.exact_bytes(value, reference[f"{source}_feat_scale{(16, 19, 22).index(index)}"]), "Detect frontier differs from historical head feature capture")
            digest = tensor_hash(value)
            D.require(stats[str(index)] == tensor_statistics(value), "Native layer statistics differ from saved frontier bytes")
            D.require(metadata["frontier_sha256_before"][source][str(index)] == digest
                      and metadata["frontier_sha256_after"][source][str(index)] == digest, "Frontier before/after hashes differ")
            native_frontiers[source][str(index)] = {"shape": list(value.shape), "dtype": str(dtype), "sha256": digest}
    for index in union:
        D.require(arrays[f"native_fp32_layer_{index}"].shape == arrays[f"native_fp16_layer_{index}"].shape, "Native frontier branch shapes differ")
    for cut in CUTS:
        for source in ("fp32", "fp16"):
            expected = {kind: tensor_hash(arrays[source + "_" + kind]) for kind in ("box_logits", "class_logits")}
            expected["decoded"] = tensor_hash(arrays["native_" + source + "_decoded"])
            D.require(metadata["native_suffix_replay_sha256"][str(cut)][source] == expected, "Native suffix replay hashes differ from captured/reference bytes")
    arithmetic = {}
    for cut, source, suffix in VARIANTS:
        tag = prefix(cut, source, suffix)
        dtype = np.dtype("float32" if suffix == "fp32" else "float16")
        casts = metadata["variant_frontier_casts"][tag.removesuffix("_")]
        D.require(set(casts) == {str(index) for index in frontiers[str(cut)]}, "Cast frontier membership differs")
        for index in frontiers[str(cut)]:
            D.require(casts[str(index)] == tensor_statistics(arrays[f"native_{source}_layer_{index}"].astype(dtype)), "Crossed frontier report differs from exact source cast")
        for kind, count in (("box_logits", 4 * reg_max), ("class_logits", 10)):
            value = arrays[tag + kind]
            D.require(value.shape == (1, count, 8400) and value.dtype == dtype and np.isfinite(value).all(), "Invalid intervention logits")
            if cut == -1 and suffix == "fp16":
                D.require(D.exact_bytes(value, arrays[suffix + "_" + kind]), "Native/input-control suffix replay logits differ")
        if cut == -1 and suffix == "fp16":
            D.require(D.exact_bytes(arrays[tag + "decoded"], arrays["native_" + suffix + "_decoded"]), "Native suffix unified decode differs")
        if cut == 22:
            for kind in ("box_logits", "class_logits", "decoded"):
                D.require(D.exact_bytes(arrays[tag + kind], reference[f"source_{source}_head_{suffix}_{kind}"]), "Cut22 endpoint differs from existing head intervention")
        arithmetic[tag.removesuffix("_")] = approximate_decode_check(arrays[tag + "box_logits"], arrays[tag + "class_logits"], arrays[tag + "decoded"], reg_max)
    D.require(set(metadata["variant_frontier_casts"]) == {prefix(*v).removesuffix("_") for v in VARIANTS}, "Variant cast inventory differs")
    guards = {name: D.fixed_guard(arrays[left], arrays[right]) for name, (left, right) in COMPARISONS.items()}
    D.require(set(metadata["comparisons"]) == set(guards), "Comparison inventory differs")
    for name, guard in guards.items():
        D.compare_reported_guard(guard, metadata["comparisons"][name])
    return {"image_id": metadata["image_id"], "all_native_historical_reference_and_cast_bytes_exact": True,
            "derived_frontiers": frontiers, "native_frontiers": native_frontiers, "arithmetic_checks": arithmetic, "comparisons": guards}


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for name in COMPARISONS:
        guards = [row["comparisons"][name] for row in rows]
        result[name] = {"images": len(rows), "pass_images": sum(g["allclose_pass"] for g in guards), "fail_images": sum(not g["allclose_pass"] for g in guards)}
        for section in ("boxes", "scores"):
            result[name][section] = {"elements": sum(g[section]["elements"] for g in guards), "violations": sum(g[section]["violations"] for g in guards),
                                     "max_absolute_error": max(g[section]["max_absolute_error"] for g in guards),
                                     "max_error_over_tolerance": max(g[section]["max_error_over_tolerance"] for g in guards)}
    return result


def check_summary(recomputed: dict, summary: dict, count: int) -> None:
    D.require(summary.get("schema") == "rampq.gate1.upstream_frontier_probe_summary/v1" and summary.get("images") == count, "Wrong summary schema/count")
    D.require(all(summary.get(key) is True for key in SUMMARY_EXACT), "Summary exactness absent")
    D.require(summary.get("cuts") == CUTS and summary.get("frontiers") == EXPECTED_FRONTIERS and summary.get("all_intervention_decode_dtype") == "fp32"
              and summary.get("raw_tolerances") == D.RAW_TOLERANCES and summary.get("original_guard_replaced") is False
              and summary.get("scientific_gate_pass") is False, "Summary changes scope")
    D.require(set(summary.get("comparisons", {})) == set(COMPARISONS), "Summary comparison inventory differs")
    for name, actual in recomputed.items():
        reported = summary["comparisons"][name]
        D.require(reported.get("diagnostic_guard_pass_images") == actual["pass_images"] and reported.get("diagnostic_guard_fail_images") == actual["fail_images"], "Summary guard count differs")
        for section in ("boxes", "scores"):
            D.require(reported.get(section) == {key: actual[section][key] for key in ("violations", "max_absolute_error", "max_error_over_tolerance")}, "Summary guard values differ")


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
    for key in ("qualification_manifest", "checkpoint", "qualifier", "runtime_environment", "decoder_runner", "head_runner", "reference_head_probe_manifest"):
        bound_identity(p[key])
    D.require(D.sha(D.safe_file(probe, "runner_source.py")) == p["runner_sha256"], "Runner source differs")
    history_root, reference_root = Path(p["qualification_dir"]).resolve(strict=True), Path(p["reference_head_probe_dir"]).resolve(strict=True)
    for root in (probe, history_root, reference_root):
        D.require(not output.is_relative_to(root) and not root.is_relative_to(output), "Output overlaps immutable input")
    D.require(p["qualification_manifest"]["path"] == str(history_root / "manifest.json") and p["reference_head_probe_manifest"]["path"] == str(reference_root / "manifest.json"), "Manifest association differs")
    reference_manifest = D.verify_manifest(reference_root)
    reference_protocol = D.read_json(reference_root / "protocol.json")
    D.require(set(p["image_ids"]).issubset(set(reference_protocol["image_ids"])) and reference_protocol["checkpoint"]["sha256"] == p["checkpoint"]["sha256"], "Reference cohort/checkpoint differs")
    ref_receipt = D.read_json(reference_root / "receipt.json")
    D.require(ref_receipt.get("status") == "COMPLETE_HEAD_DIAGNOSTIC_ONLY" and ref_receipt.get("images_completed") == 109
              and ref_receipt.get("native_head_logits_all_exact") is True and ref_receipt.get("features_and_head_states_unchanged") is True, "Reference head probe incomplete")
    cohort = D.read_json(D.safe_file(history_root, "debug_source_manifest.json"))["rows"]
    by_id = validate_scope(p, cohort)
    receipt = D.read_json(D.safe_file(probe, "receipt.json"))
    D.require(receipt.get("status") == "COMPLETE_UPSTREAM_DIAGNOSTIC_ONLY" and receipt.get("phase") == "COMPLETE"
              and receipt.get("protocol_sha256") == args.protocol_sha256 and receipt.get("runner_sha256") == p["runner_sha256"]
              and receipt.get("images_completed") == len(p["image_ids"]) and receipt.get("scientific_gate_pass") is False
              and receipt.get("old_guard_replaced") is False and receipt.get("tolerances_changed") is False
              and receipt.get("selection_detector_inference") is False and receipt.get("holdout_accessed") is False, "Incomplete receipt or scope change")
    historical_manifest = D.read_json(history_root / "manifest.json")["files"]
    metadata_paths = sorted((probe / "per_image").glob("*.json"))
    D.require(len(metadata_paths) == len(p["image_ids"]), "Image count differs")
    rows = []
    for path in metadata_paths:
        meta = D.read_json(path)
        raw_relative, meta_relative = f"raw/{path.stem}.npz", f"per_image/{path.stem}.json"
        D.require(meta["image_id"] in by_id and meta["array_file"] == raw_relative, "Image/raw association differs")
        for relative in (meta_relative, raw_relative):
            old = D.safe_file(history_root, relative); expected = historical_manifest[relative]
            D.require(old.stat().st_size == expected["bytes"] and D.sha(old) == expected["sha256"], "Historical file changed")
        old_meta, ref_meta = D.read_json(history_root / meta_relative), D.read_json(reference_root / meta_relative)
        for key in ("image_id", "normalized_fp32_input_sha256", "preprocessed_uint8_sha256", "letterbox_params"):
            D.require(meta[key] == old_meta[key] == ref_meta[key], "Historical/reference input metadata differs")
        with np.load(D.safe_file(probe, raw_relative), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        with np.load(D.safe_file(reference_root, raw_relative), allow_pickle=False) as archive:
            reference = {key: archive[key] for key in archive.files}
        with np.load(history_root / raw_relative, allow_pickle=False) as archive:
            historical = {key: archive[key] for key in ("fp32", "fp16")}
        rows.append(audit_arrays(arrays, meta, reference, historical, p["topology"]))
    D.require(sorted(row["image_id"] for row in rows) == sorted(p["image_ids"]) and len({row["image_id"] for row in rows}) == len(rows), "Image inventory differs")
    recomputed = aggregate(rows)
    check_summary(recomputed, D.read_json(D.safe_file(probe, "summary.json")), len(rows))
    output.mkdir(parents=True)
    summary = {"schema": "rampq.gate1.upstream_frontier_saved_array_audit/v1", "status": "PASS_SAVED_UPSTREAM_FRONTIER_ARRAYS_ONLY",
               "protocol_sha256": args.protocol_sha256, "probe_manifest": manifest, "reference_head_manifest": reference_manifest,
               "images_verified": len(rows), "cuts": CUTS, "derived_frontiers": EXPECTED_FRONTIERS, "comparisons": recomputed,
               "scientific_gate_pass": False, "old_guard_replaced": False, "tolerances_changed": False,
               "selection_detector_inference": False, "holdout_accessed": False, "limits": LIMITS, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    D.write_json(output / "recomputed_summary.json", summary)
    D.write_json(output / "per_image_checks.json", rows)
    D.write_json(output / "audit_source_identity.json", {"frontier_auditor": D.identity(Path(__file__).resolve()), "frozen_numpy_decode_helper": D.identity(_HELPER)})
    D.write_json(output / "manifest.json", {"files": {item.name: {"bytes": item.stat().st_size, "sha256": D.sha(item)} for item in sorted(output.iterdir()) if item.is_file()}})
    print(summary["status"], "images", len(rows))


if __name__ == "__main__":
    main()
