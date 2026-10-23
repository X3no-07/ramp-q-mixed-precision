"""Independent saved-output audit of the direct body32/head16 prototype.

This reads arrays, source identities and runtime observation reports only. It
does not load a model, decode pixels, execute CUDA, or prove layer causality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HELPER_SHA = "04a7d2256ab0a3d90025b2453ef2bc44b043e8b1493b7866c9acc603838dc1ef"
DECODER_SHA = "f76f3bfd677487fa7d87947f866daaf7407ddd455ad3e4d9e399da719e481ec0"
HEAD_SHA = "842e16b7b743c6044975ab2faad5725ead19cd8c793575340c9ee81ddaaef23f"
CHECK_NAME = "GATE1_BODY_FP32_HEAD_TOWERS_FP16_DECODE_FP32_DEBUG_NUMERIC_CHECK"
PRECISION_CONTRACT = {"body_module_indices": list(range(23)), "body_parameters_and_outputs": "fp32",
                      "head_towers": ["cv2", "cv3"], "head_tower_parameters_and_logits": "fp16",
                      "head_input_cast": "fresh FP32 body features explicitly cast to FP16",
                      "geometry_dfl_and_sigmoid": "fp32", "independent_checkpoint_loads": 1,
                      "mixed_forwards_per_image": 1, "saved_frontier_inputs_permitted": False,
                      "endpoint_exact_required": True, "fusion_called": False}
_HELPER = ROOT / "scripts/audit_gate1_decode_probe.py"
if hashlib.sha256(_HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError("Frozen decode auditor source changed")
_SPEC = importlib.util.spec_from_file_location("frozen_decode_auditor_for_direct_body32_head16", _HELPER)
D = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(D)
PILOT_IDS = ["train/0000170_01201_d_0000001", "train/9999990_00000_d_0000024", "train/0000046_00000_d_0000087"]
STAGES = ("probabilities", "distances", "xy1", "xy2", "xywh", "boxes", "scores", "decoded")
LIMITS = [
    "Saved-array and metadata consistency only; no model, convolution, BatchNorm, pixel decoding, GPU, NMS, annotations, AP, Selection or Holdout execution.",
    "The reviewed runner must load the frozen checkpoint and compute from each fresh image; reference NPZ array values may be read only after the new mixed forward for comparison. Earlier integrity hashing reads bytes only.",
    "Runtime body/head/decode precision observations are reviewed-runner reports. This auditor does not independently execute those operators or establish GPU internal accumulation precision.",
    "Fresh feature arrays are deduplicated into SHA/statistics and compared against stored reference features. The receiving FP16 cast statistics are independently recomputed from the reference FP32 tensors.",
    "NumPy FP64 decoding is an approximate arithmetic sanity check, not bitwise CUDA equivalence or a replacement numerical guard.",
    "The repeated head-intervention endpoint is not independent evidence. A diagnostic guard PASS never replaces the frozen full-FP16 FAIL or accepts Gate1, deployment, accuracy or performance.",
]


def tensor_hash(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def statistics(value: np.ndarray) -> dict:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value), "elements": int(value.size),
            "min": float(value.min()), "max": float(value.max()), "mean": float(value.astype(np.float64).mean())}


def approximate_decode(box_logits: np.ndarray, class_logits: np.ndarray, decoded: np.ndarray, reg_max: int) -> dict:
    logits = box_logits.astype(np.float64).reshape(1, 4, reg_max, 8400).transpose(0, 2, 1, 3)
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True)); probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    distances = (probabilities * np.arange(reg_max, dtype=np.float64)[None, :, None, None]).sum(axis=1)
    anchors, strides = D.expected_geometry(np.dtype("float32"))
    xy1, xy2 = anchors.astype(np.float64)[None] - distances[:, :2], anchors.astype(np.float64)[None] + distances[:, 2:]
    boxes = np.concatenate(((xy1 + xy2) / 2, xy2 - xy1), axis=1) * strides.astype(np.float64)
    scores = np.exp(-np.logaddexp(0, -class_logits.astype(np.float64)))
    box_error, score_error = np.abs(decoded[:, :4].astype(np.float64) - boxes), np.abs(decoded[:, 4:].astype(np.float64) - scores)
    D.require(np.all(box_error <= .002 + .00001 * np.abs(boxes)), "FP32 boxes differ from approximate NumPy FP64 decoding")
    D.require(np.all(score_error <= .0000002 + .000003 * np.abs(scores)), "FP32 scores differ from approximate NumPy sigmoid")
    return {"nonlinear_checks_are_approximate_only": True, "box_rtol_sanity_only": .00001, "box_atol_sanity_only": .002,
            "score_rtol_sanity_only": .000003, "score_atol_sanity_only": .0000002,
            "boxes_max_absolute_error_vs_numpy_fp64": float(box_error.max()), "scores_max_absolute_error_vs_numpy_fp64": float(score_error.max())}


def validate_scope(protocol: dict, cohort: list[dict]) -> dict:
    D.require(protocol.get("schema") == "rampq.gate1.body32_head16_prototype_protocol/v1", "Wrong prototype protocol schema")
    D.require(protocol.get("decoder_runner", {}).get("sha256") == DECODER_SHA and protocol.get("head_runner", {}).get("sha256") == HEAD_SHA, "Wrong frozen helper")
    D.require(protocol.get("precision_contract") == PRECISION_CONTRACT and protocol.get("guard_name") == CHECK_NAME, "Prototype precision/check identity changed")
    topology = protocol.get("topology", {})
    modules = topology.get("modules", [])
    expected_f = [-1] * 24
    for index, f in ((12, [-1, 6]), (15, [-1, 4]), (18, [-1, 13]), (21, [-1, 10]), (23, [16, 19, 22])): expected_f[index] = f
    D.require(len(modules) == 24 and all(m.get("i") == i and m.get("f") == expected_f[i] for i, m in enumerate(modules))
              and modules[-1].get("type", "").endswith("Detect") and topology.get("save") == [4, 6, 10, 13, 16, 19, 22], "Pinned YOLO11n DAG differs")
    if protocol.get("stage") == "pilot3":
        D.require(protocol.get("image_ids") == PILOT_IDS, "Pilot IDs/order changed")
    adapted = dict(protocol, schema="rampq.gate1.decode_precision_probe_protocol/v1")
    return D.validate_scope(adapted, cohort)


def audit_arrays(arrays: dict, metadata: dict, reference: dict, historical: dict) -> dict:
    reg_max = metadata["reg_max"]
    D.require(type(reg_max) is int and 2 <= reg_max <= 128 and metadata["nc"] == 10 and metadata["nl"] == 3, "Wrong detector dimensions")
    expected_keys = {"input_fp32_cpu", "box_logits", "class_logits", "anchors", "strides", "dfl_bins", "decoded"}
    D.require(set(arrays) == expected_keys, "Prototype stored array inventory differs")
    shapes = {"input_fp32_cpu": (1, 3, 640, 640), "box_logits": (1, 4 * reg_max, 8400), "class_logits": (1, 10, 8400),
              "anchors": (2, 8400), "strides": (1, 8400), "dfl_bins": (reg_max,), "decoded": (1, 14, 8400)}
    for key in expected_keys:
        dtype = np.dtype("float16" if key in ("box_logits", "class_logits") else "float32")
        D.require(arrays[key].shape == shapes[key] and arrays[key].dtype == dtype and np.isfinite(arrays[key]).all(), "Prototype array shape/dtype/finite differs: " + key)
    inp = arrays["input_fp32_cpu"]
    D.require(((inp >= 0) & (inp <= 1)).all() and D.exact_bytes(inp, reference["input_fp32_cpu"])
              and tensor_hash(inp) == metadata["normalized_fp32_input_sha256"], "Input domain/identity differs")
    anchors, strides = D.expected_geometry(np.dtype("float32"))
    D.require(D.exact_bytes(arrays["anchors"], anchors) and D.exact_bytes(arrays["strides"], strides)
              and D.exact_bytes(arrays["dfl_bins"], np.arange(reg_max, dtype=np.float32)), "FP32 decoder geometry/bins differ")
    D.require(metadata.get("endpoint_byte_equal") == {"box_logits": True, "class_logits": True, "decoded": True}, "Endpoint identity report absent")
    for key in ("box_logits", "class_logits", "decoded"):
        D.require(D.exact_bytes(arrays[key], reference["source_fp32_head_fp16_" + key]), "Prototype differs from declared earlier head endpoint: " + key)
    D.require(metadata.get("reference_feature_exact") == [True, True, True], "Reference feature exactness absent")
    observations = metadata["body_module_observations"]
    D.require(isinstance(observations, list) and len(observations) == 23, "Incomplete body observation inventory")
    for index, observation in enumerate(observations):
        shape = observation.get("shape")
        D.require(observation.get("i") == index and observation.get("dtype") == "torch.float32" and isinstance(shape, list)
                  and len(shape) == 4 and shape[0] == 1 and all(type(v) is int and v > 0 for v in shape), "Body runtime precision/shape observation differs")
    D.require(len(metadata["head_input_features"]) == len(metadata["head_cast_features"]) == 3, "Head feature inventory differs")
    for scale, index in enumerate((16, 19, 22)):
        feature = reference[f"fp32_feat_scale{scale}"]
        D.require(feature.dtype == np.float32 and np.isfinite(feature).all(), "Invalid reference feature")
        D.require(metadata["head_input_features"][scale] == statistics(feature), "Fresh FP32 feature report differs from reference bytes")
        D.require(metadata["head_cast_features"][scale] == statistics(feature.astype(np.float16)), "Head FP16 feature cast differs")
        D.require(observations[index]["shape"] == list(feature.shape), "Body/head feature geometry differs")
    D.require(metadata.get("decoder_stage_dtypes") == {key: "torch.float32" for key in STAGES}, "Decoder runtime stage precision differs")
    D.require(metadata.get("body_state_unchanged") is True and metadata.get("head_state_unchanged") is True, "Runtime state immutability absent")
    D.require(metadata.get("model_state_unchanged") is True and metadata.get("mixed_forwards") == 1
              and metadata.get("historical_arrays_opened_after_forward") is True and metadata.get("saved_frontier_inference_inputs") is False
              and metadata.get("reference_input_exact") is True and metadata.get("reference_cast_feature_exact") == [True, True, True], "Fresh mixed-forward scope/report changed")
    D.require(metadata.get("old_guard_replaced") is False and metadata.get("tolerances_changed") is False, "Prototype changes evidence scope")
    arithmetic = approximate_decode(arrays["box_logits"], arrays["class_logits"], arrays["decoded"], reg_max)
    guard = D.fixed_guard(historical["fp32"], arrays["decoded"])
    D.require(metadata["comparison"].get("guard_name") == CHECK_NAME and metadata["comparison"].get("historical_gate1_dtype_guard_replaced") is False, "Prototype comparison conflates original dtype check")
    D.compare_reported_guard(guard, metadata["comparison"])
    geometry = {name: {"positive_width_height": bool((value[:, 2:4] > 0).all()),
                       "class_scores_within_0_1": bool(((value[:, 4:] >= 0) & (value[:, 4:] <= 1)).all())}
                for name, value in (("historical_fp32", historical["fp32"]), ("body32_head16_decode32_prototype", arrays["decoded"]))}
    D.require(metadata["comparison"].get("raw_geometry") == geometry and metadata["comparison"].get("shape_pass") is True
              and metadata["comparison"].get("finite_pass") is True and metadata["comparison"].get("tolerances_changed") is False
              and metadata["comparison"].get("old_gate05_guard_replaced") is False, "Prototype geometry/scope is mislabeled or differs")
    guard.update(guard_name=CHECK_NAME, historical_gate1_dtype_guard_replaced=False, raw_geometry=geometry)
    original_guard = D.fixed_guard(historical["fp32"], historical["fp16"])
    return {"image_id": metadata["image_id"], "endpoint_bytes_exact": True, "feature_reference_and_cast_reports_match_saved_reference": True,
            "observed_body_modules_fp32": 23, "arithmetic_check": arithmetic, "comparison": guard, "historical_original_guard": original_guard}


def aggregate(rows: list[dict], key: str = "comparison") -> dict:
    guards = [row[key] for row in rows]
    result = {"images": len(rows), "pass_images": sum(g["allclose_pass"] for g in guards), "fail_images": sum(not g["allclose_pass"] for g in guards)}
    for section in ("boxes", "scores"):
        result[section] = {"elements": sum(g[section]["elements"] for g in guards), "violations": sum(g[section]["violations"] for g in guards),
                           "max_absolute_error": max(g[section]["max_absolute_error"] for g in guards),
                           "max_error_over_tolerance": max(g[section]["max_error_over_tolerance"] for g in guards)}
    return result


def check_summary(actual: dict, summary: dict, count: int) -> None:
    D.require(summary.get("schema") == "rampq.gate1.body32_head16_prototype_summary/v1" and summary.get("images") == count, "Wrong prototype summary")
    D.require(summary.get("endpoint_bytes_all_exact") is True and summary.get("all_body_modules_fp32") is True
              and summary.get("all_head_towers_fp16") is True and summary.get("all_decoder_stages_fp32") is True
              and summary.get("states_unchanged") is True, "Summary exactness/runtime observations absent")
    D.require(summary.get("raw_tolerances") == D.RAW_TOLERANCES and summary.get("old_guard_replaced") is False
              and summary.get("original_guard_replaced") is False and summary.get("scientific_gate_pass") is False
              and summary.get("accepted_float_baseline_ready") is False, "Summary changes scientific scope")
    D.require(summary.get("guard_name") == CHECK_NAME and summary.get("historical_gate1_dtype_guard_replaced") is False
              and summary.get("precision_contract") == PRECISION_CONTRACT and summary.get("fresh_mixed_forwards") == count
              and summary.get("checkpoint_loads") == 1, "Summary prototype identity/load/forward scope differs")
    reported = summary["comparison"]
    D.require(reported.get("diagnostic_guard_pass_images") == actual["pass_images"] and reported.get("diagnostic_guard_fail_images") == actual["fail_images"], "Summary guard counts differ")
    for section in ("boxes", "scores"):
        D.require(reported.get(section) == {key: actual[section][key] for key in ("violations", "max_absolute_error", "max_error_over_tolerance")}, "Summary guard values differ")


def bound(entry: dict) -> Path:
    path = Path(entry["path"]).resolve(strict=True)
    D.require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry["bytes"] and D.sha(path) == entry["sha256"], "Bound input source differs")
    return path


def validate_provenance(provenance: dict, checkpoint: dict, topology: dict) -> None:
    D.require(provenance.get("checkpoint") == checkpoint and provenance.get("checkpoint_loads") == 1
              and provenance.get("precision_contract") == PRECISION_CONTRACT and provenance.get("topology") == topology
              and provenance.get("fusion_called") is False and provenance.get("reference_is_original_fp32_training_state") is False,
              "Provenance checkpoint/load/precision identity differs")
    policy = provenance["parameter_buffer_policy"]
    expected = {"body_parameters": "torch.float32", "body_floating_buffers": "torch.float32", "head_tower_parameters": "torch.float16",
                "head_tower_floating_buffers": "torch.float16", "dfl_parameters": "torch.float32", "dfl_floating_buffers": "torch.float32"}
    D.require(set(policy) == set(expected), "Parameter/buffer precision inventory differs")
    for key, dtype in expected.items():
        counts = policy[key]
        D.require(isinstance(counts, dict) and set(counts).issubset({dtype}) and all(type(count) is int and count > 0 for count in counts.values())
                  and (bool(counts) or key == "dfl_floating_buffers"), "Reported parameter/buffer precision differs: " + key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe, protocol_path, output = args.probe.resolve(strict=True), args.protocol.resolve(strict=True), args.output.resolve()
    D.require(not output.exists() and D.sha(protocol_path) == args.protocol_sha256, "Output exists or protocol SHA differs")
    p = D.read_json(protocol_path); probe_manifest = D.verify_manifest(probe)
    D.require(D.sha(D.safe_file(probe, "protocol.json")) == args.protocol_sha256 and D.sha(D.safe_file(probe, "runner_source.py")) == p["runner_sha256"], "Copied protocol/runner differs")
    for key in ("qualification_manifest", "checkpoint", "qualifier", "runtime_environment", "decoder_runner", "head_runner", "reference_head_probe_manifest"):
        bound(p[key])
    history_root, reference_root = Path(p["qualification_dir"]).resolve(strict=True), Path(p["reference_head_probe_dir"]).resolve(strict=True)
    for root in (probe, history_root, reference_root):
        D.require(not output.is_relative_to(root) and not root.is_relative_to(output), "Audit output overlaps frozen input")
    D.require(p["qualification_manifest"]["path"] == str(history_root / "manifest.json") and p["reference_head_probe_manifest"]["path"] == str(reference_root / "manifest.json"), "Reference manifest/root association differs")
    reference_manifest = D.verify_manifest(reference_root)
    ref_p, ref_receipt = D.read_json(reference_root / "protocol.json"), D.read_json(reference_root / "receipt.json")
    D.require(set(p["image_ids"]).issubset(ref_p["image_ids"]) and ref_p["checkpoint"]["sha256"] == p["checkpoint"]["sha256"], "Reference cohort/checkpoint differs")
    D.require(ref_receipt.get("status") == "COMPLETE_HEAD_DIAGNOSTIC_ONLY" and ref_receipt.get("images_completed") == 109
              and ref_receipt.get("native_head_logits_all_exact") is True and ref_receipt.get("features_and_head_states_unchanged") is True, "Reference head intervention incomplete")
    cohort = D.read_json(D.safe_file(history_root, "debug_source_manifest.json"))["rows"]
    by_id = validate_scope(p, cohort)
    validate_provenance(D.read_json(D.safe_file(probe, "model_provenance.json")), p["checkpoint"], p["topology"])
    receipt = D.read_json(D.safe_file(probe, "receipt.json"))
    D.require(receipt.get("status") == "COMPLETE_BODY32_HEAD16_DIAGNOSTIC_ONLY" and receipt.get("phase") == "COMPLETE"
              and receipt.get("protocol_sha256") == args.protocol_sha256 and receipt.get("runner_sha256") == p["runner_sha256"]
              and receipt.get("images_completed") == len(p["image_ids"]) and receipt.get("old_guard_replaced") is False
              and receipt.get("tolerances_changed") is False and receipt.get("scientific_gate_pass") is False
              and receipt.get("selection_detector_inference") is False and receipt.get("holdout_accessed") is False, "Prototype receipt incomplete or changes scope")
    history_manifest = D.read_json(history_root / "manifest.json")["files"]
    paths = sorted((probe / "per_image").glob("*.json")); D.require(len(paths) == len(p["image_ids"]), "Output image count differs")
    rows = []
    for path in paths:
        meta = D.read_json(path); relative_raw, relative_meta = f"raw/{path.stem}.npz", f"per_image/{path.stem}.json"
        D.require(meta["image_id"] in by_id and meta["array_file"] == relative_raw, "Image/raw association differs")
        for relative in (relative_meta, relative_raw):
            old = D.safe_file(history_root, relative); expected = history_manifest[relative]
            D.require(old.stat().st_size == expected["bytes"] and D.sha(old) == expected["sha256"], "Historical raw/metadata changed")
        old_meta, ref_meta = D.read_json(history_root / relative_meta), D.read_json(reference_root / relative_meta)
        for key in ("image_id", "normalized_fp32_input_sha256", "preprocessed_uint8_sha256", "letterbox_params"):
            D.require(meta[key] == old_meta[key] == ref_meta[key], "Historical/reference input metadata differs")
        with np.load(D.safe_file(probe, relative_raw), allow_pickle=False) as archive: arrays = {key: archive[key] for key in archive.files}
        with np.load(D.safe_file(reference_root, relative_raw), allow_pickle=False) as archive: reference = {key: archive[key] for key in archive.files}
        with np.load(history_root / relative_raw, allow_pickle=False) as archive: history = {key: archive[key] for key in ("fp32", "fp16")}
        rows.append(audit_arrays(arrays, meta, reference, history))
    D.require(sorted(row["image_id"] for row in rows) == sorted(p["image_ids"]) and len({r["image_id"] for r in rows}) == len(rows), "Output cohort differs")
    actual = aggregate(rows); original = aggregate(rows, "historical_original_guard")
    check_summary(actual, D.read_json(D.safe_file(probe, "summary.json")), len(rows))
    output.mkdir(parents=True)
    summary = {"schema": "rampq.gate1.body32_head16_saved_output_audit/v1", "status": "PASS_SAVED_BODY32_HEAD16_OUTPUTS_ONLY",
               "images_verified": len(rows), "protocol_sha256": args.protocol_sha256, "probe_manifest": probe_manifest,
               "reference_head_manifest": reference_manifest, "comparison": actual, "historical_full_model_comparison_unchanged": original,
               "guard_name": CHECK_NAME, "historical_gate1_dtype_guard_replaced": False,
               "scientific_gate_pass": False, "accepted_float_baseline_ready": False, "old_guard_replaced": False,
               "selection_detector_inference": False, "holdout_accessed": False, "limits": LIMITS, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    D.write_json(output / "recomputed_summary.json", summary); D.write_json(output / "per_image_checks.json", rows)
    D.write_json(output / "audit_source_identity.json", {"prototype_auditor": D.identity(Path(__file__).resolve()), "frozen_numpy_decode_helper": D.identity(_HELPER)})
    D.write_json(output / "manifest.json", {"files": {path.name: {"bytes": path.stat().st_size, "sha256": D.sha(path)} for path in sorted(output.iterdir()) if path.is_file()}})
    print(summary["status"], "images", len(rows))


if __name__ == "__main__": main()
