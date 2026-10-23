"""Describe differences between saved fixed-image arrays; no model execution."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
SOURCES = {
    "native_raw_outputs": ROOT / "artifacts/gate05/20261019_corrected_qonly_native/post_smoke_evidence/fixed_image_smoke/raw_outputs.npz",
    "native_smoke_result": ROOT / "artifacts/gate05/20261019_corrected_qonly_native/post_smoke_evidence/fixed_image_smoke/smoke_result.json",
    "cpu_raw_outputs": ROOT / "artifacts/gate05/20261019_corrected_qonly_cpu_equivalence/raw_outputs.npz",
    "cpu_equivalence_analysis": ROOT / "artifacts/gate05/20261019_corrected_qonly_cpu_equivalence/analysis.json",
}
CLASS_NAMES = [
    "holothurian", "echinus", "scallop", "starfish", "fish", "corals",
    "diver", "cuttlefish", "turtle", "jellyfish",
]
PRIMARY_NAMES = ["x", "y", "w", "h"] + CLASS_NAMES


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_array(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "all_finite": bool(np.isfinite(array).all()),
        "nonfinite_elements": int(np.count_nonzero(~np.isfinite(array))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def compare(native, cpu, source_channels):
    if native.shape != cpu.shape:
        raise ValueError("Saved output shapes differ")
    if not np.isfinite(native).all() or not np.isfinite(cpu).all():
        raise ValueError("Nonfinite saved output requires explicit separate handling")
    # Convert operands before subtracting; every reduction below is float64.
    signed = native.astype(np.float64) - cpu.astype(np.float64)
    absolute = np.abs(signed)
    flat_index = int(np.argmax(absolute))
    local_index = tuple(int(v) for v in np.unravel_index(flat_index, native.shape))
    global_index = list(local_index)
    global_index[1] = int(source_channels[local_index[1]])
    percentiles = np.percentile(absolute, [50, 95, 99], method="linear")
    different = int(np.count_nonzero(native != cpu))
    return {
        "source_channels": [int(v) for v in source_channels],
        "shape": list(native.shape),
        "elements": int(native.size),
        "native_input": describe_array(native),
        "cpu_original_input": describe_array(cpu),
        "difference_direction": "native_minus_cpu_original",
        "arithmetic_dtype": "float64",
        "max_abs": float(np.max(absolute)),
        "mean_abs": float(np.mean(absolute, dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(signed), dtype=np.float64))),
        "p50_abs": float(percentiles[0]),
        "p95_abs": float(percentiles[1]),
        "p99_abs": float(percentiles[2]),
        "signed_mean": float(np.mean(signed, dtype=np.float64)),
        "signed_min": float(np.min(signed)),
        "signed_max": float(np.max(signed)),
        "different_elements": different,
        "equal_elements": int(native.size) - different,
        "different_fraction": float(different / native.size),
        "worst": {
            "original_array_index": global_index,
            "local_slice_index": list(local_index),
            "native_value": float(native[local_index]),
            "cpu_original_value": float(cpu[local_index]),
            "signed_difference": float(signed[local_index]),
            "absolute_difference": float(absolute[local_index]),
            "tie_rule": "First maximum in C-order within the selected slice",
        },
    }


def cpu_pair_description(original, code_clip):
    same_bytes = bool(
        original.shape == code_clip.shape
        and original.dtype == code_clip.dtype
        and original.tobytes(order="C") == code_clip.tobytes(order="C")
    )
    return {
        "shape": list(original.shape),
        "dtype": str(original.dtype),
        "saved_arrays_byte_identical": same_bytes,
        "different_elements": int(np.count_nonzero(original != code_clip)),
        "max_abs_float64": float(np.max(np.abs(original.astype(np.float64) - code_clip.astype(np.float64)))),
    }


def main():
    before = {name: sha256(path) for name, path in SOURCES.items()}
    cpu_analysis = json.loads(SOURCES["cpu_equivalence_analysis"].read_text(encoding="utf-8"))
    smoke = json.loads(SOURCES["native_smoke_result"].read_text(encoding="utf-8"))
    with np.load(SOURCES["native_raw_outputs"], allow_pickle=False) as saved_native:
        native_primary = saved_native["tensorrt"].copy()
        native_auxiliary = saved_native["auxiliary_output_0"].copy()
        native_keys = list(saved_native.files)
    with np.load(SOURCES["cpu_raw_outputs"], allow_pickle=False) as saved_cpu:
        cpu_primary = saved_cpu["original_0"].copy()
        cpu_auxiliary = saved_cpu["original_1"].copy()
        code_clip_primary = saved_cpu["code_clip_0"].copy()
        code_clip_auxiliary = saved_cpu["code_clip_1"].copy()
        cpu_keys = list(saved_cpu.files)
    if native_primary.shape != (1, 14, 8400) or cpu_primary.shape != (1, 14, 8400):
        raise ValueError("Unexpected primary array shape")
    if native_auxiliary.shape != (1, 128, 20, 20) or cpu_auxiliary.shape != (1, 128, 20, 20):
        raise ValueError("Unexpected auxiliary array shape")
    if cpu_analysis.get("original_complete_model_pytorch_ort_guard") != "FAILED_UNCHANGED":
        raise ValueError("Original full-model guard evidence no longer matches expected retained status")
    primary_channels = []
    for channel, name in enumerate(PRIMARY_NAMES):
        item = compare(native_primary[:, channel:channel + 1, :], cpu_primary[:, channel:channel + 1, :], [channel])
        item.update({"channel": channel, "name": name, "kind": "xywh" if channel < 4 else "class_score"})
        if channel >= 4:
            item["class_id"] = channel - 4
        primary_channels.append(item)
    auxiliary_channels = []
    for channel in range(128):
        item = compare(native_auxiliary[:, channel:channel + 1, :, :], cpu_auxiliary[:, channel:channel + 1, :, :], [channel])
        item["channel"] = channel
        auxiliary_channels.append(item)
    report = {
        "schema": "rampq.gate05.saved_fixed_image_numeric_arrays_recheck/v1",
        "status": "DESCRIPTIVE_SAVED_ARRAY_COMPARISON_COMPLETED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One saved fixed image; CPU arithmetic on existing NPZ arrays only",
        "image": smoke.get("image"),
        "image_sha256_from_native_smoke": smoke.get("image_sha256"),
        "numpy_version": np.__version__,
        "program": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "source_files": {name: str(path.resolve()) for name, path in SOURCES.items()},
        "source_hashes_before": before,
        "archive_contents": {"native_npz_keys": native_keys, "cpu_npz_keys": cpu_keys},
        "source_report_hash_agreement": {
            "native_raw_outputs_match_smoke_result": before["native_raw_outputs"] == smoke.get("raw_outputs_sha256"),
            "cpu_raw_outputs_match_analysis": before["cpu_raw_outputs"] == cpu_analysis.get("raw_outputs_sha256"),
        },
        "array_mapping": {
            "primary": {"native": "tensorrt", "cpu_original": "original_0", "cpu_code_clip": "code_clip_0", "output_name": "output0", "channel_axis": 1},
            "auxiliary": {"native": "auxiliary_output_0", "cpu_original": "original_1", "cpu_code_clip": "code_clip_1", "output_name": "/model.10/m/m.0/attn/pe/conv/Conv_output_0", "channel_axis": 1},
        },
        "statistic_definitions": {
            "difference": "float64(native) - float64(cpu_original)",
            "percentiles": "Absolute differences, numpy.percentile with method=linear, all elements in the stated slice",
            "rmse": "sqrt(mean(square(signed_difference))), float64",
            "different_elements": "Count of saved numeric values for which native != cpu_original; no tolerance",
            "input_ranges": "native_input and cpu_original_input are the two saved-array operands, not the model image input",
            "primary_order": PRIMARY_NAMES,
            "primary_xywh_units": "Raw model output xywh coordinates, before NMS",
        },
        "primary": {
            "overall": compare(native_primary, cpu_primary, range(14)),
            "xywh": compare(native_primary[:, :4, :], cpu_primary[:, :4, :], range(4)),
            "class_scores": compare(native_primary[:, 4:, :], cpu_primary[:, 4:, :], range(4, 14)),
            "channels": primary_channels,
        },
        "auxiliary": {
            "overall": compare(native_auxiliary, cpu_auxiliary, range(128)),
            "channels": auxiliary_channels,
        },
        "saved_cpu_original_vs_code_clip": {
            "primary": cpu_pair_description(cpu_primary, code_clip_primary),
            "auxiliary": cpu_pair_description(cpu_auxiliary, code_clip_auxiliary),
            "scope": "Saved outputs from the existing CPU single-image original/code_clip comparison only",
            "interpretation": "Bitexact saved CPU outputs do not establish native TensorRT parity or original PyTorch/ORT migration",
        },
        "evidence_boundaries": {
            "original_complete_model_pytorch_ort_guard": "FAILED_UNCHANGED",
            "new_tolerances_defined": False,
            "numerical_pass_claim": False,
            "nms_executed": False,
            "ssh_executed": False,
            "gpu_executed": False,
            "model_inference_executed": False,
            "dataset_evaluation_run": False,
            "accuracy_claim": False,
            "performance_claim": False,
            "causal_attribution_allowed": False,
            "limits": "Descriptive values only. One frozen image does not establish a dataset result, accuracy rescue, numerical migration, native parity, or performance. The original complete-model PyTorch/ORT guard remains FAILED_UNCHANGED.",
        },
    }
    after = {name: sha256(path) for name, path in SOURCES.items()}
    report["source_hashes_after"] = after
    report["all_source_hashes_unchanged"] = before == after
    if before != after:
        raise RuntimeError("A read-only source changed while computing the comparison")
    output = OUT_DIR / "numeric_arrays_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256(output), "program_sha256": report["program"]["sha256"], "primary": report["primary"]["overall"], "xywh": report["primary"]["xywh"], "class_scores": report["primary"]["class_scores"], "auxiliary": report["auxiliary"]["overall"], "all_source_hashes_unchanged": before == after}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
