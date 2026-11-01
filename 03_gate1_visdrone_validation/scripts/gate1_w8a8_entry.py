"""Gate1 synthetic-only entry audit for unfused-Conv W8A8 Q/DQ simulation.

This module never loads images, labels, calibration cohorts or a legacy runner.
The old Gate0 quantizer/group pure operations are reused on a separately rebuilt
FP32 model; BN remains unfused, so this is a newly defined intervention.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "artifacts/gate1/20261023_formal_ignore_training_v1/formal_attempt_01/fp32_ema/epoch_0030"
DESTINATION = ROOT / "artifacts/gate1/20261024_w8a8_entry_audit_v1"
FINAL_SHA = "4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27"
CONFIGURATIONS = ["fp32", "q_all", "r_only:G1"]
GROUPS = tuple(f"G{i}" for i in range(1, 9))
SCHEMA = "rampq.gate1.unfused_w8a8_synthetic_entry/v1"
CONTRACT_NAME = "contract_v2.json"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_once(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


@lru_cache(maxsize=1)
def legacy():
    source = ROOT / "src/gate0.py"
    spec = importlib.util.spec_from_file_location("gate1_entry_legacy_pure_ops", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_input(side=64):
    require(type(side) is int and side in (64, 640), "Only frozen synthetic input sizes are admitted")
    if side == 640:
        return torch.zeros(1, 3, side, side, dtype=torch.float32)
    return torch.arange(3 * side * side, dtype=torch.float32).remainder(251).div(250).reshape(1, 3, side, side)


def tensor_info(value):
    value = value.detach().cpu().contiguous()
    require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "Nonfinite tensor")
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def state_info(model):
    return {name: tensor_info(value) for name, value in sorted(model.state_dict().items())}


def output_info(output):
    if isinstance(output, torch.Tensor):
        return {"tensor": tensor_info(output)}
    if isinstance(output, (list, tuple)):
        return {"kind": type(output).__name__, "items": [output_info(value) for value in output]}
    if isinstance(output, dict):
        return {"kind": "dict", "items": {key: output_info(value) for key, value in sorted(output.items())}}
    require(output is None or type(output) in (str, int, bool), "Unsupported output object")
    return output


def make_wrapper(conv, name, maximum):
    require(isinstance(conv, nn.Conv2d) and math.isfinite(float(maximum)) and float(maximum) >= 0,
            "Finite nonnegative synthetic activation maximum required")
    require(conv.weight.dtype == torch.float32 and bool(torch.isfinite(conv.weight).all()), "Finite FP32 Conv weight required")
    if conv.bias is not None:
        require(conv.bias.dtype == torch.float32 and bool(torch.isfinite(conv.bias).all()), "Finite FP32 bias required")
    return legacy().QDQConv2d(conv, name, legacy().group_for_conv(name), float(maximum))


def install(model, maxima):
    rows = legacy().list_conv_modules(model)
    require(set(maxima) == {name for name, _, _ in rows}, "Synthetic scale keys must exactly cover Conv identities")
    wrappers = []
    for name, conv, _ in rows:
        wrapper = make_wrapper(conv, name, maxima[name])
        legacy().replace_child(model, name, wrapper)
        wrappers.append(wrapper)
    return wrappers


def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def capture_helpers():
    sys.path.insert(0, str(ROOT / "scripts"))
    import gate1_fp32_live_capture as capture
    return capture


def source_bindings():
    sources = {"runner": Path(__file__), "tests": ROOT / "tests/test_gate1_w8a8_entry.py",
               "legacy_pure_operations": ROOT / "src/gate0.py",
               "state_loader": ROOT / "scripts/gate1_fp32_reference_state.py",
               "capture_and_architecture_validation": ROOT / "scripts/gate1_fp32_live_capture.py"}
    return {key: binding(path) for key, path in sources.items()}


def freeze(output):
    require(output.resolve() == DESTINATION.resolve(), "Only the dedicated stage A output is admitted")
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / CONTRACT_NAME).exists(), "Contract already frozen")
    capture = capture_helpers()
    state = binding(CAPTURE / "state/reference_state.pt")
    require(state["sha256"] == FINAL_SHA, "Wrong final FP32 candidate")
    contract = {"schema": SCHEMA, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": 'Only synthetic inputs with unfused Conv W8A8 Q/DQ engineering entry; not a real calibration or detection experiment',
        "candidate": state, "architecture": binding(CAPTURE / "architecture.json"),
        "state_manifest": binding(CAPTURE / "state/manifest.json"),
        "state_receipt": binding(CAPTURE / "state/snapshot_receipt.json"),
        "source_bindings": source_bindings(), "runtime_sources": capture.sources(),
        "fusion_policy": 'BN remains unfused; retain the current official FP32 graph structure. This is not an equivalent reproduction of the previous Gate0 fused Conv intervention.',
        "reconstruction_class_policy": 'The original training class IgnoreAwareDetectionModel was solely for training loss adaptation; reconstruct inference using the saved reconstruction_class via DetectionModel; the architecture is field-by-field identical except for the live_class declaration.',
        "precision": 'CPU FP32 floating-point convolution; input and weight undergo QDQ preprocessing, but remain computed in FP32; batch normalization, other operators, and decoding are retained in FP32.',
        "activation": 'Symmetric per tensor input; scale = max(max_abs / 127, 1e-12); torch.round; clamp within [-127, 127]; zero point is 0.',
        "weight": 'Symmetric per output channel; scale = max(absmax, 1e-12) / 127; torch.round; clamp within [-127, 127]; bias remains in FP32.',
        "synthetic_activation_max_abs": 1.0,
        "synthetic_scale_warning": 'The range of 1.0 is solely for packaging status and arithmetic smoke testing and is strictly prohibited as a valid calibration range or conclusion regarding precision.',
        "configurations": CONFIGURATIONS, "group_mapping": 'src/gate0.py::group_for_conv, containing DFL-projected convolutions; the actual list is stored separately.',
        "synthetic_inputs": {"entry": tensor_info(synthetic_input(64)), "macs": tensor_info(synthetic_input(640))},
        "budget": {"candidate_count": 1, "candidate": "fixed_final_epoch_30_FP32_EMA", "real_images": 0,
            "real_calibration_images": 0, "entry_input_forward_configurations": CONFIGURATIONS,
            "MAC": 'Only one 640-zero input is used to accumulate multiply-add counts per convolution via hooks; this is not related to runtime or energy consumption.',
            "device": "cpu", "fixed_G1_only": True, "control_selection": 'No control variant is selected or module ranking is performed in this stage.'},
        "acceptance": ['No quantized new packaging is used, and the quantized output tree is not strictly identical to the original unquantized FP32 synthesis tree on a per-tensor SHA basis.',
            'Quantization applies only to deep copies; the original FP32 weights, batch normalization states, inputs, source files, and the output with quantization disabled are preserved unchanged.',
            'All convolution identities are covered and grouped exclusively; MACs are derived from actual 640-synthesized inputs.',
            'The quantization branch only verifies finite behavior and state isolation; it does not require QDQ outputs to match FP32 outputs, nor does it accept QDQ outputs as valid precision qualifications for the task.'],
        "forbidden": ['Real images or labels', 'Real calibration', '109 images inference', "AP", "Selection/Holdout", 'ONNX or device-level task'],
        "scientific_acceptance": False, "deployment_INT8": False,
        "old_failures_preserved": ["FAIL_FIXED_TOLERANCES", "FAILED_UNCHANGED"]}
    if (output / "contract.json").exists():
        contract["supersedes_unexecuted_contract"] = binding(output / "contract.json")
        contract["supersession_reason"] = 'Static review prior to execution reveals that the original architecture declares live_class as a training subclass; explicitly, only the preserved reconstructed class policy is applied to this declaration, and the old contract model check is not executed.'
    write_once(output / CONTRACT_NAME, contract)
    return {"status": "FROZEN_BEFORE_SYNTHETIC_MODEL_CHECKS", "contract": binding(output / CONTRACT_NAME)}


def audit(output):
    require(output.resolve() == DESTINATION.resolve(), "Only the dedicated stage A output is admitted")
    require(not (output / "result.json").exists(), "Synthetic audit already has a result")
    contract = read(output / CONTRACT_NAME)
    require(contract["schema"] == SCHEMA and contract["candidate"]["sha256"] == FINAL_SHA, "Frozen contract identity differs")
    for key in ("candidate", "architecture", "state_manifest", "state_receipt"):
        require(binding(contract[key]["path"]) == contract[key], "Frozen candidate binding changed")
    require(source_bindings() == contract["source_bindings"], "Frozen stage A source changed")
    configure()
    capture = capture_helpers()
    require(capture.sources() == contract["runtime_sources"], "Runtime source changed")
    from ultralytics.nn.tasks import DetectionModel
    architecture = read(contract["architecture"]["path"])
    saved, metadata = capture.FORMAT.load_snapshot(CAPTURE / "state")
    canonical_architecture = capture.canonical(architecture)
    require(hashlib.sha256(canonical_architecture.encode()).hexdigest() == metadata["architecture_sha256"], "Architecture metadata identity mismatch")
    with torch.random.fork_rng(devices=[]):
        model = DetectionModel(deepcopy(architecture["cfg"]), ch=3, nc=10, verbose=False)
    model.load_state_dict(saved, strict=True)
    model.names = dict(enumerate(architecture["names"]))
    model.cpu().float().eval().requires_grad_(False)
    require(architecture["live_class"] == "gate1_ignore_trainer.IgnoreAwareDetectionModel" and
            architecture["reconstruction_class"] == "ultralytics.nn.tasks.DetectionModel", "Unreviewed training/reconstruction class")
    reconstructed_architecture = capture.architecture(model)
    require({key: value for key, value in reconstructed_architecture.items() if key != "live_class"} ==
            {key: value for key, value in architecture.items() if key != "live_class"}, "Reconstructed graph architecture differs")
    require(sum(isinstance(layer, nn.BatchNorm2d) for layer in model.modules()) > 0, "BN unexpectedly fused")
    original_state = state_info(model)
    require(original_state == {name: tensor_info(value) for name, value in sorted(saved.items())}, "Loaded FP32 tensor identity differs")
    x = synthetic_input(64)
    require(tensor_info(x) == contract["synthetic_inputs"]["entry"], "Synthetic entry input changed")
    with torch.inference_mode():
        baseline = output_info(model(x))
    modules = legacy().list_conv_modules(model)
    inventory = []
    macs, group_macs = legacy().collect_macs(model, 640, torch.device("cpu"))
    require(set(macs) == {name for name, _, _ in modules} and set(group_macs) == set(GROUPS), "Conv/MAC group identity coverage differs")
    for name, conv, group in modules:
        inventory.append({"name": name, "group": group, "weight_shape": list(conv.weight.shape),
            "bias": conv.bias is not None, "stride": list(conv.stride), "padding": list(conv.padding),
            "dilation": list(conv.dilation), "groups": conv.groups,
            "weight_and_bias_parameters": sum(value.numel() for value in conv.parameters()), "MAC_640": macs[name]})
    require(sum(macs.values()) == sum(group_macs.values()) and all(value > 0 for value in macs.values()), "Invalid MAC accounting")
    candidate = deepcopy(model)
    require(all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), candidate.parameters())), "FP32 candidate storage alias")
    wrappers = install(candidate, {name: contract["synthetic_activation_max_abs"] for name, _, _ in modules})
    candidate.eval()
    with torch.inference_mode():
        legacy().enable_configuration(wrappers, "fp32")
        disabled = output_info(candidate(x))
        require(disabled == baseline, "Disabled QDQ changed the complete output tree")
        wrapped_state = state_info(candidate)
        outputs = {"fp32": disabled}
        masks = {}
        for configuration in CONFIGURATIONS:
            legacy().enable_configuration(wrappers, configuration)
            masks[configuration] = [wrapper.name for wrapper in wrappers if wrapper.enabled]
            observed = output_info(candidate(x))
            outputs[configuration] = observed
            require(state_info(candidate) == wrapped_state, "Quantizer execution polluted wrapped FP32/BN/buffer state")
            require(state_info(model) == original_state, "Original FP32 state polluted by quantizer")
        legacy().enable_configuration(wrappers, "fp32")
        after_quantized_replay = output_info(candidate(x))
        original_replay = output_info(model(x))
    require(after_quantized_replay == baseline == original_replay, "Quantization polluted subsequent disabled/original output")
    require(tensor_info(x) == contract["synthetic_inputs"]["entry"], "Synthetic input was mutated")
    require(state_info(model) == original_state and state_info(candidate) == wrapped_state, "Final state fingerprint changed")
    require(source_bindings() == contract["source_bindings"] and capture.sources() == contract["runtime_sources"], "Source changed during audit")
    for key in ("candidate", "architecture", "state_manifest", "state_receipt"):
        require(binding(contract[key]["path"]) == contract[key], "Candidate source changed during audit")
    total = sum(group_macs.values())
    write_once(output / "module_inventory.json", {"schema": SCHEMA, "fusion": "unfused_BN", "Conv_count": len(inventory),
        "groups": {group: {"Conv_count": sum(row["group"] == group for row in inventory),
            "MAC_640": group_macs[group], "MAC_share": group_macs[group] / total} for group in GROUPS},
        "MAC_640_total": total, "MAC_is_not_measured_latency_or_energy": True, "modules": inventory})
    write_once(output / "synthetic_outputs.json", {"complete_output_tree_descriptions": outputs,
        "disabled_after_quantization": after_quantized_replay, "original_after_quantization": original_replay})
    write_once(output / "state_fingerprints.json", {"original_FP32": original_state, "wrapped_FP32_and_QDQ_buffers": wrapped_state,
        "before_after_verified_unchanged": True})
    result = {"schema": SCHEMA, "status": "PASS_SYNTHETIC_UNFUSED_W8A8_ENGINEERING_ENTRY_ONLY",
        "contract": binding(output / CONTRACT_NAME), "candidate": contract["candidate"],
        "Conv_count": len(inventory), "BN_count": sum(isinstance(layer, nn.BatchNorm2d) for layer in model.modules()),
        "group_counts": dict(Counter(row["group"] for row in inventory)), "MAC_640_total": total,
        "G1_MAC_share": group_macs["G1"] / total, "all_disabled_output_tensors_exact": disabled == baseline,
        "original_FP32_parameters_and_BN_state_unchanged": True, "wrapper_buffers_unchanged": True,
        "original_and_wrapped_parameter_storage_disjoint": True, "input_unchanged": True,
        "disabled_replay_after_QDQ_exact": True, "QDQ_configurations_finite": CONFIGURATIONS[1:],
        "enabled_Conv_masks": masks, "quantized_output_equals_FP32_required": False,
        "real_images_or_labels_read": 0, "real_calibration_performed": False, "task_AP_computed": False,
        "Selection_Holdout_accessed": False, "scientific_acceptance": False, "deployment_INT8": False,
        "sources_and_candidate_unchanged": True, "old_failures_preserved": contract["old_failures_preserved"]}
    write_once(output / "result.json", result)
    return {key: result[key] for key in ("status", "Conv_count", "BN_count", "MAC_640_total", "G1_MAC_share")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "audit"))
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    os.environ.update(YOLO_CONFIG_DIR=str(args.output / "settings"), YOLO_OFFLINE="true",
                      PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false")
    try:
        result = freeze(args.output) if args.mode == "freeze" else audit(args.output)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        if args.mode == "audit" and args.output.is_dir() and not (args.output / "failure.json").exists():
            write_once(args.output / "failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
