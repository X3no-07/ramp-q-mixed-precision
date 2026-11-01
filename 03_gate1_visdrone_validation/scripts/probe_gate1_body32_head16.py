"""Fresh-image body-FP32/head-FP16 engineering prototype, FP32 decoding.

One independent checkpoint load and one mixed forward per image. Historical
numerical arrays are loaded only after the forward for comparison; pre-run
integrity verification reads historical file bytes only to compute hashes.
No saved features, NMS, ground truth, Selection or Holdout enter computation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np


SCHEMA = "rampq.gate1.body32_head16_prototype_protocol/v1"
DECODER_SHA = "f76f3bfd677487fa7d87947f866daaf7407ddd455ad3e4d9e399da719e481ec0"
HEAD_SHA = "842e16b7b743c6044975ab2faad5725ead19cd8c793575340c9ee81ddaaef23f"
CHECK_NAME = "GATE1_BODY_FP32_HEAD_TOWERS_FP16_DECODE_FP32_DEBUG_NUMERIC_CHECK"
PILOT_IDS = ["train/0000170_01201_d_0000001", "train/9999990_00000_d_0000024", "train/0000046_00000_d_0000087"]
PRECISION_CONTRACT = {"body_module_indices": list(range(23)), "body_parameters_and_outputs": "fp32",
                      "head_towers": ["cv2", "cv3"], "head_tower_parameters_and_logits": "fp16",
                      "head_input_cast": "fresh FP32 body features explicitly cast to FP16",
                      "geometry_dfl_and_sigmoid": "fp32", "independent_checkpoint_loads": 1,
                      "mixed_forwards_per_image": 1, "saved_frontier_inputs_permitted": False,
                      "endpoint_exact_required": True, "fusion_called": False}
LIMITS = [
    "Engineering prototype on fixed reused train-derived debug images; no independent generalization or scientific baseline acceptance.",
    "Body modules 0..22 are FP32; only final cv2/cv3 towers are FP16; this is not an all-FP16 or INT8 deployment.",
    "A single fresh checkpoint load and mixed forward are used per attempt/image; saved frontiers are never inference inputs.",
    "Historical head-cross endpoint equality verifies implementation consistency, not independent discovery of a new numerical effect.",
    "Saved FP16 EMA is lifted to FP32 for the body; original full-precision training state is not recovered.",
    "No NMS, annotations, target matching, AP, Selection, Holdout, TensorRT, Orin, performance or energy evidence.",
    "Prototype guard results do not replace either unchanged historical failed guard or accept Gate1.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def statistics(value: Any) -> dict:
    array = value.detach().cpu().numpy()
    require(bool(np.isfinite(array).all()), "Nonfinite prototype tensor")
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": array_sha(array), "elements": int(array.size),
            "min": float(array.min()), "max": float(array.max()), "mean": float(array.astype(np.float64).mean())}


def prototype_guard(qualifier: Any, reference: np.ndarray, prototype: np.ndarray) -> dict:
    result = qualifier.raw_guard(reference, prototype)
    result["guard_name"] = CHECK_NAME
    result["historical_gate1_dtype_guard_replaced"] = False
    if "raw_geometry" in result:
        geometry = result["raw_geometry"]
        result["raw_geometry"] = {"historical_fp32": geometry["fp32"], "body32_head16_decode32_prototype": geometry["fp16"]}
    return result


def topology_of_model(model: Any) -> dict:
    return {"modules": [{"i": int(module.i), "f": module.f if isinstance(module.f, int) else list(module.f),
                         "type": type(module).__module__ + "." + type(module).__name__} for module in model.model],
            "save": sorted(int(i) for i in model.save)}


def validate_protocol(p: dict, own_sha: str, rows: list[dict], decoder: Any) -> list[dict]:
    require(p.get("schema") == SCHEMA and p.get("runner_sha256") == own_sha, "Prototype protocol/source binding differs")
    require(p.get("precision_contract") == PRECISION_CONTRACT and p.get("decoder_runner", {}).get("sha256") == DECODER_SHA
            and p.get("head_runner", {}).get("sha256") == HEAD_SHA and p.get("guard_name") == CHECK_NAME,
            "Precision contract/frozen helper/check name differs")
    topology = p.get("topology", {})
    require(len(topology.get("modules", [])) == 24 and topology["modules"][-1]["i"] == 23
            and topology["modules"][-1]["f"] == [16, 19, 22] and topology["modules"][-1]["type"].endswith("Detect"), "Unsupported topology")
    if p.get("stage") == "pilot3":
        require(p.get("image_ids") == PILOT_IDS, "Frozen prototype pilot IDs/order changed")
    return decoder.validate_protocol(dict(p, schema=decoder.SCHEMA, runner_sha256=DECODER_SHA), DECODER_SHA, rows)


def make_prototype(torch: Any, model: Any, decoder: Any, make_anchors: Any) -> Any:
    """Wrap one fresh mixed-precision model; no historical arrays are accepted."""
    require(not model.training and len(model.model) == 24 and model.model[-1].f == [16, 19, 22], "Unsupported eval model")
    require(all(not module.training for module in model.modules()), "A prototype submodule is training")
    require(all(value.dtype == torch.float32 for module in model.model[:-1] for value in module.parameters()), "Body parameters must be FP32")
    require(all(value.dtype == torch.float32 for module in model.model[:-1] for value in module.buffers() if value.is_floating_point()),
            "Body floating buffers must be FP32")
    head = model.model[-1]
    require(all(value.dtype == torch.float16 for tower in (head.cv2, head.cv3) for value in tower.parameters()), "Head tower parameters must be FP16")
    require(all(value.dtype == torch.float16 for tower in (head.cv2, head.cv3) for value in tower.buffers() if value.is_floating_point()),
            "Head tower floating buffers must be FP16")
    require(all(value.dtype == torch.float32 for value in head.dfl.parameters()), "DFL parameters must stay FP32")
    require(all(value.dtype == torch.float32 for value in head.dfl.buffers() if value.is_floating_point()), "DFL floating buffers must stay FP32")
    require(head.stride.dtype == torch.float32, "Head geometry stride must be FP32")

    class Body32Head16(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = model
            self.eval()

        def forward(self, tensor: Any) -> dict:
            require(not self.training and tensor.dtype == torch.float32 and tensor.ndim == 4 and tensor.shape[0] == 1,
                    "Prototype input must be a batch-one FP32 image")
            observations, handles = [], []
            def observe(index: int):
                def hook(_module: Any, _inputs: Any, value: Any) -> None:
                    require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                            "A body module output is not finite FP32")
                    observations.append({"i": index, "dtype": str(value.dtype), "shape": list(value.shape)})
                return hook
            y, current = [], tensor
            try:
                for i, module in enumerate(self.network.model[:-1]):
                    handles.append(module.register_forward_hook(observe(i)))
                for i, module in enumerate(self.network.model[:-1]):
                    require(module.i == i, "Body module index changed")
                    if module.f != -1:
                        current = y[module.f] if isinstance(module.f, int) else [current if j == -1 else y[j] for j in module.f]
                    current = module(current)
                    y.append(current if i in self.network.save else None)
            finally:
                for handle in handles:
                    handle.remove()
            require([row["i"] for row in observations] == list(range(23)), "Body observation inventory differs")
            head = self.network.model[-1]
            features = [y[i] for i in head.f]
            require(all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 for value in features), "Missing FP32 Detect inputs")
            cast_features = [value.half() for value in features]
            logits = head.forward_head(cast_features, **head.one2many)
            require(set(logits) == {"boxes", "scores", "feats"} and logits["boxes"].dtype == logits["scores"].dtype == torch.float16,
                    "Head tower output schema/precision differs")
            anchors, strides = make_anchors(features, head.stride, 0.5)
            anchors, strides = anchors.transpose(0, 1), strides.transpose(0, 1)
            require(anchors.dtype == strides.dtype == torch.float32, "Prototype geometry is not FP32")
            decoded_stages = decoder(torch, logits["boxes"], logits["scores"], anchors, strides, int(head.reg_max), torch.float32)
            require(all(value.dtype == torch.float32 for value in decoded_stages.values()), "A decoder stage is not FP32")
            decoded = decoded_stages["decoded"]
            require(decoded.dtype == torch.float32, "Prototype decode is not FP32")
            return {"box_logits": logits["boxes"], "class_logits": logits["scores"], "anchors": anchors, "strides": strides,
                    "dfl_bins": head.dfl.conv.weight.flatten(), "decoded": decoded,
                    "features": features, "cast_features": cast_features, "body_module_observations": observations,
                    "decoder_stage_dtypes": {key: str(value.dtype) for key, value in decoded_stages.items()}}

    return Body32Head16()


def parameter_buffer_policy(model: Any) -> dict:
    body = list(model.model[:-1]); head = model.model[-1]
    towers = [head.cv2, head.cv3]
    def counts(modules: list, buffers: bool) -> dict:
        tensors = (value for module in modules for value in (module.buffers() if buffers else module.parameters()))
        return dict(Counter(str(value.dtype) for value in tensors if value.is_floating_point()))
    return {"body_parameters": counts(body, False), "body_floating_buffers": counts(body, True),
            "head_tower_parameters": counts(towers, False), "head_tower_floating_buffers": counts(towers, True),
            "dfl_parameters": counts([head.dfl], False), "dfl_floating_buffers": counts([head.dfl], True)}


def load_mixed_model(torch: Any, checkpoint: Path, qualifier: Any, device: str = "cuda:0") -> tuple[Any, dict]:
    """Exactly one checkpoint deserialization; never call the two-branch loader."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = "ema" if isinstance(payload.get("ema"), torch.nn.Module) else "model"
    model = payload.get(key)
    require(isinstance(model, torch.nn.Module), "Checkpoint contains no model")
    saved_dtypes = Counter(str(value.dtype) for value in model.state_dict().values())
    require(set(str(value.dtype) for value in model.parameters()) == {"torch.float16"}, "Expected serialized FP16 EMA parameters")
    require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite saved state")
    require({int(k): v for k, v in model.names.items()} == dict(enumerate(qualifier.NAMES)), "Saved class names changed")
    modules = list(model.modules())
    wrappers = [m for m in modules if type(m).__name__ == "Conv" and type(m).__module__.startswith("ultralytics.nn.modules")]
    structure = {"Conv2d": sum(isinstance(m, torch.nn.Conv2d) for m in modules),
                 "BatchNorm2d": sum(isinstance(m, torch.nn.BatchNorm2d) for m in modules), "UltralyticsConv": len(wrappers),
                 "UltralyticsConv_with_BatchNorm": sum(isinstance(getattr(m, "bn", None), torch.nn.BatchNorm2d) for m in wrappers)}
    qualifier.validate_unfused_structure(structure)
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = True
    model.float().eval().to(device=device, dtype=torch.float32)
    head = model.model[-1]
    require(head.nc == 10 and head.nl == 3 and head.reg_max > 1 and not head.dynamic and not head.export
            and not head.end2end and not head.xyxy, "Unsupported prototype head")
    head.cv2.half(); head.cv3.half()
    return model, {"checkpoint_loads": 1, "checkpoint_key": key, "saved_state_dtypes": dict(saved_dtypes),
                   "unfused_structure": structure, "fusion_called": False, "body_compute_dtype": "torch.float32",
                   "head_cv2_cv3_dtype": "torch.float16", "dfl_geometry_sigmoid_dtype": "torch.float32",
                   "reference_is_original_fp32_training_state": False, "parameter_buffer_policy": parameter_buffer_policy(model)}


def run(p: dict, output: Path, receipt: dict) -> None:
    for key, expected in (("decoder_runner", DECODER_SHA), ("head_runner", HEAD_SHA)):
        entry = p[key]; path = Path(entry["path"])
        require(path.is_file() and path.stat().st_size == entry["bytes"] and sha(path) == expected, "Frozen helper source changed")
    spec = importlib.util.spec_from_file_location("frozen_decoder_for_body32_head16", p["decoder_runner"]["path"])
    decoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(decoder)
    selected = validate_protocol(p, sha(Path(__file__)), read_json(Path(p["qualification_dir"]) / "debug_source_manifest.json")["rows"], decoder)
    q, historical, rows, verified = decoder.historical_inputs(dict(p, schema=decoder.SCHEMA, runner_sha256=DECODER_SHA), output)
    require(selected == rows, "Historical helper selected a different cohort")
    reference = Path(p["reference_head_probe_dir"]).resolve(strict=True)
    reference_manifest = decoder.bound_identity(p["reference_head_probe_manifest"], "reference head manifest")
    require(reference_manifest == reference / "manifest.json", "Reference manifest/root mismatch")
    for relative, entry in read_json(reference_manifest)["files"].items():
        path = decoder.safe_file(reference, relative)
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Reference head output changed")
    ref_receipt, ref_protocol = read_json(reference / "receipt.json"), read_json(reference / "protocol.json")
    require(ref_receipt.get("status") == "COMPLETE_HEAD_DIAGNOSTIC_ONLY" and ref_receipt.get("images_completed") == 109
            and ref_receipt.get("native_head_logits_all_exact") is True and ref_receipt.get("features_and_head_states_unchanged") is True,
            "Reference full109 head probe not complete/exact")
    require(set(p["image_ids"]).issubset(set(ref_protocol["image_ids"])) and p["checkpoint"]["sha256"] == ref_protocol["checkpoint"]["sha256"],
            "Reference cohort/checkpoint differs")
    torch, cv2, LetterBox, _, _, environment = q.configure_runtime(output)
    require(environment["packages"] == p["package_versions"] and all(environment[k] == v for k, v in decoder.CONTROLS.items()), "Runtime control/packages changed")
    import ultralytics
    from ultralytics.utils.tal import make_anchors
    framework_root = Path(ultralytics.__file__).resolve().parent
    require(all(sha(decoder.safe_file(framework_root, path)) == value for path, value in p["framework_source_sha256"].items()), "Framework source changed")
    old_environment = verified["historical_environment"]
    require(all(environment[k] == old_environment[k] for k in ("cuda", "cudnn", "gpu"))
            and environment["executable"]["sha256"] == old_environment["executable"]["sha256"], "Historical runtime identity changed")
    write_json(output / "environment.json", environment)
    model, provenance = load_mixed_model(torch, verified["checkpoint"], q)
    require(topology_of_model(model) == p["topology"], "Live topology differs")
    prototype = make_prototype(torch, model, decoder.decode_functional, make_anchors)
    states = {key: value.clone() for key, value in model.state_dict().items()}
    write_json(output / "model_provenance.json", {**provenance, "checkpoint": decoder.identity(verified["checkpoint"]),
               "precision_contract": PRECISION_CONTRACT, "topology": p["topology"], "reg_max": int(model.model[-1].reg_max)})
    letterbox = LetterBox((640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32,
                         padding_value=114, interpolation=cv2.INTER_LINEAR)
    (output / "raw").mkdir(); (output / "per_image").mkdir()
    items = []
    receipt.update(phase="FRESH_MIXED_FORWARD", inference_started=True)
    write_json(output / "receipt.json", receipt)
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for number, row in enumerate(rows, 1):
            stem = Path(row["image_path"]).stem
            image_path = decoder.safe_file(historical, "debug/images/" + Path(row["image_path"]).name)
            require(sha(image_path) == row["image_sha256"], "Debug image bytes changed")
            image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(image is not None and image.shape == (row["height"], row["width"], 3), "Decoded image dimensions differ")
            params = letterbox.get_params({"img": image})
            rgb = np.ascontiguousarray(letterbox.apply_image({"img": image}, params)["img"][..., ::-1].transpose(2, 0, 1))
            normalized = torch.from_numpy(rgb).unsqueeze(0).float().div(255)
            # Only the newly decoded and normalized image is an inference input.
            mixed = prototype(normalized.to("cuda:0"))
            torch.cuda.synchronize()
            arrays = {key: value.detach().cpu().numpy().copy() for key, value in mixed.items()
                      if key not in ("features", "cast_features", "body_module_observations", "decoder_stage_dtypes")}
            arrays["input_fp32_cpu"] = normalized.numpy().copy()
            require(arrays["decoded"].shape == (1, 14, 8400) and all(np.isfinite(value).all() for value in arrays.values()), "Prototype output shape/finite failure")
            feature_stats = [statistics(value) for value in mixed["features"]]
            cast_stats = [statistics(value) for value in mixed["cast_features"]]
            # Historical arrays are not opened until the independent forward ends.
            old_detail = read_json(decoder.safe_file(historical, f"per_image/{stem}.json"))
            serial_params = {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()}
            require(array_sha(rgb) == old_detail["preprocessed_uint8_sha256"] and array_sha(normalized.numpy()) == old_detail["normalized_fp32_input_sha256"]
                    and serial_params == old_detail["letterbox_params"], "Historical preprocessing differs")
            with np.load(decoder.safe_file(reference, f"raw/{stem}.npz"), allow_pickle=False) as ref:
                endpoint_exact = {key: decoder.bytes_equal(arrays[key], ref["source_fp32_head_fp16_" + key]) for key in ("box_logits", "class_logits", "decoded")}
                reference_feature_exact = [feature_stats[i]["sha256"] == array_sha(ref[f"fp32_feat_scale{i}"])
                                           and feature_stats[i]["shape"] == list(ref[f"fp32_feat_scale{i}"].shape) for i in range(3)]
                cast_feature_exact = [cast_stats[i]["sha256"] == array_sha(ref[f"fp32_feat_scale{i}"].astype(np.float16)) for i in range(3)]
                input_exact = decoder.bytes_equal(arrays["input_fp32_cpu"], ref["input_fp32_cpu"])
            require(all(endpoint_exact.values()) and all(reference_feature_exact) and all(cast_feature_exact) and input_exact,
                    "Fresh prototype differs from frozen endpoint/input/features")
            with np.load(decoder.safe_file(historical, f"raw/{stem}.npz"), allow_pickle=False) as old:
                comparison = prototype_guard(q, old["fp32"], arrays["decoded"])
            state_unchanged = all(value.dtype == states[key].dtype and value.shape == states[key].shape and bool(torch.equal(value, states[key]))
                                  for key, value in model.state_dict().items())
            require(state_unchanged, "Prototype model state mutated")
            body_state_unchanged = all(value.dtype == states[key].dtype and bool(torch.equal(value, states[key]))
                                       for key, value in model.state_dict().items() if not key.startswith("model.23."))
            head_state_unchanged = all(value.dtype == states[key].dtype and bool(torch.equal(value, states[key]))
                                       for key, value in model.state_dict().items() if key.startswith("model.23."))
            detail = {"image_id": row["image_id"], "image_sha256": row["image_sha256"], "array_file": f"raw/{stem}.npz",
                      "reg_max": int(model.model[-1].reg_max), "nc": 10, "nl": 3,
                      "normalized_fp32_input_sha256": array_sha(arrays["input_fp32_cpu"]), "preprocessed_uint8_sha256": array_sha(rgb),
                      "letterbox_params": serial_params, "body_module_observations": mixed["body_module_observations"],
                      "head_input_features": feature_stats, "head_cast_features": cast_stats,
                      "endpoint_byte_equal": endpoint_exact, "reference_feature_exact": reference_feature_exact, "reference_cast_feature_exact": cast_feature_exact,
                      "reference_input_exact": input_exact, "model_state_unchanged": state_unchanged, "mixed_forwards": 1,
                      "historical_arrays_opened_after_forward": True, "saved_frontier_inference_inputs": False,
                      "decoder_stage_dtypes": mixed["decoder_stage_dtypes"], "body_state_unchanged": body_state_unchanged,
                      "head_state_unchanged": head_state_unchanged,
                      "comparison": comparison, "old_guard_replaced": False, "tolerances_changed": False}
            np.savez_compressed(output / "raw" / f"{stem}.npz", **arrays)
            write_json(output / "per_image" / f"{stem}.json", detail)
            items.append(detail)
            receipt["images_completed"] = number
            write_json(output / "receipt.json", receipt)
            print(f"BODY32_HEAD16 {number}/{len(rows)} {row['image_id']} FRESH_FORWARD_ENDPOINT_EXACT", flush=True)
    for key in ("checkpoint", "qualifier", "head_runner", "decoder_runner", "runtime_environment", "qualification_manifest", "reference_head_probe_manifest"):
        decoder.bound_identity(p[key], key + " after execution")
    guards = [row["comparison"] for row in items]
    summary = {"schema": "rampq.gate1.body32_head16_prototype_summary/v1", "images": len(items), "stage": p["stage"],
               "endpoint_all_exact": True, "endpoint_bytes_all_exact": True, "reference_features_and_inputs_all_exact": True,
               "model_state_unchanged": True, "states_unchanged": True, "all_body_modules_fp32": True,
               "all_head_towers_fp16": True, "all_decoder_stages_fp32": True,
               "fresh_mixed_forwards": len(items), "checkpoint_loads": 1, "precision_contract": PRECISION_CONTRACT,
               "raw_tolerances": decoder.TOLERANCES, "diagnostic_guard_pass_images": sum(g["allclose_pass"] for g in guards),
               "guard_name": CHECK_NAME, "historical_gate1_dtype_guard_replaced": False,
               "diagnostic_guard_fail_images": sum(not g["allclose_pass"] for g in guards),
               "comparison": {section: {"violations": sum(g[section]["violations"] for g in guards),
                                          "max_absolute_error": max(g[section]["max_absolute_error"] for g in guards),
                                          "max_error_over_tolerance": max(g[section]["max_error_over_tolerance"] for g in guards)} for section in ("boxes", "scores")},
               "original_guard_replaced": False, "scientific_gate_pass": False, "limits": LIMITS}
    summary["comparison"].update(diagnostic_guard_pass_images=summary["diagnostic_guard_pass_images"],
                                 diagnostic_guard_fail_images=summary["diagnostic_guard_fail_images"])
    summary.update(old_guard_replaced=False, accepted_float_baseline_ready=False)
    write_json(output / "summary.json", summary)
    receipt.update(status="COMPLETE_BODY32_HEAD16_DIAGNOSTIC_ONLY", phase="COMPLETE", endpoint_all_exact=True,
                   reference_features_and_inputs_all_exact=True, model_state_unchanged=True, checkpoint_loads=1,
                   fresh_mixed_forwards=len(items))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.protocol.resolve(strict=True)
    require(sha(path) == args.protocol_sha256, "Protocol pre-execution hash differs")
    p = read_json(path)
    require(p.get("schema") == SCHEMA and p.get("runner_sha256") == sha(Path(__file__)), "Prototype protocol/source binding differs")
    output = args.output.resolve()
    require(not output.exists(), "Output must be a new directory")
    for key in ("qualification_dir", "reference_head_probe_dir"):
        source = Path(p[key]).resolve(strict=True)
        require(not output.is_relative_to(source) and not source.is_relative_to(output), "Output overlaps immutable input")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path, output / "protocol.json"); shutil.copyfile(Path(__file__), output / "runner_source.py")
    started = time.monotonic()
    receipt = {"schema": "rampq.gate1.body32_head16_prototype_receipt/v1", "status": "RUNNING", "phase": "INPUT_VALIDATION",
               "protocol_sha256": args.protocol_sha256, "runner_sha256": sha(Path(__file__)), "started_utc": datetime.now(timezone.utc).isoformat(),
               "stage": p.get("stage"), "images_requested": len(p.get("image_ids", [])), "images_completed": 0, "inference_started": False,
               "data_role": "debug", "selection_detector_inference": False, "holdout_accessed": False, "scientific_gate_pass": False,
               "accepted_float_baseline_ready": False, "old_guard_replaced": False, "tolerances_changed": False, "limits": LIMITS}
    write_json(output / "receipt.json", receipt)
    try:
        run(p, output, receipt)
    except Exception as error:
        receipt.update(status="FAILED_STOPPED_PRESERVED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "receipt.json", receipt)
        files = {file.relative_to(output).as_posix(): {"bytes": file.stat().st_size, "sha256": sha(file)}
                 for file in sorted(output.rglob("*")) if file.is_file() and file.name != "manifest.json"}
        write_json(output / "manifest.json", {"schema": "rampq.gate1.body32_head16_prototype_manifest/v1", "files": files})


if __name__ == "__main__":
    main()
