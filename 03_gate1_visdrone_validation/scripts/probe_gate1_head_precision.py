"""Frozen debug-only 2x2 head precision intervention, with FP32 decoding.

Two original network forwards capture the Detect input features. Their feature
origins are crossed with the two original unfused detection heads' arithmetic
dtypes. No training, NMS, ground truth, Selection, or Holdout is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np


SCHEMA = "rampq.gate1.head_precision_probe_protocol/v1"
DECODER_SHA = "f76f3bfd677487fa7d87947f866daaf7407ddd455ad3e4d9e399da719e481ec0"
PILOT_IDS = ["train/0000170_01201_d_0000001", "train/9999990_00000_d_0000024", "train/0000046_00000_d_0000087"]
INTERVENTION = {"feature_sources": ["fp32", "fp16"], "head_compute_dtypes": ["fp32", "fp16"],
                "all_comparisons_decode_dtype": "fp32", "native_head_logits_exact_required": True,
                "reference_decode_logits_exact_required": True}
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
LIMITS = [
    "Post-hoc arithmetic intervention on reused, train-derived engineering debug images; no independent generalization claim.",
    "The feature-origin contrast includes upstream network arithmetic; the FP32-to-FP16 cast is part of the head-dtype intervention.",
    "The head contrast includes final box/class towers and their convolutions, BatchNorm and activations; it does not isolate an individual operation.",
    "All intervention outputs use explicit FP32 DFL/geometry/sigmoid decoding; this is not an accepted deployment change.",
    "No NMS, target matching, AP, quantization, TensorRT, Orin, performance or energy evidence.",
    "A favorable diagnostic contrast never replaces either unchanged failing numerical guard or accepts Gate1.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_decoder(entry: dict) -> Any:
    require(entry.get("sha256") == DECODER_SHA, "Unreviewed decoder implementation")
    path = Path(entry["path"]).resolve(strict=True)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry["bytes"] and sha(path) == DECODER_SHA,
            "Frozen decoder source changed")
    spec = importlib.util.spec_from_file_location("frozen_decode_probe_for_head_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decoder_adapter(p: dict, decoder: Any) -> dict:
    """Reuse reviewed metadata/runtime preparation without relaxing any controls.

    This local adapter changes only the schema/runner binding expected by the
    frozen decoder helper. The actual head protocol and runner are independently
    SHA-bound by main(), and their original bytes are stored in the output.
    """
    adapted = dict(p)
    adapted.update(schema=decoder.SCHEMA, runner_sha256=DECODER_SHA)
    return adapted


def validate_head_protocol(p: dict, own_sha: str, rows: list[dict], decoder: Any) -> list[dict]:
    require(p.get("schema") == SCHEMA and p.get("runner_sha256") == own_sha, "Head protocol/runner binding differs")
    require(p.get("decoder_runner", {}).get("sha256") == DECODER_SHA, "Wrong frozen decoder")
    require(p.get("intervention") == INTERVENTION, "Head intervention cannot be redefined")
    if p.get("stage") == "pilot3":
        require(p.get("image_ids") == PILOT_IDS, "Head pilot images/order must remain fixed")
    return decoder.validate_protocol(decoder_adapter(p, decoder), DECODER_SHA, rows)


def cross_head(torch: Any, head: Any, features: list[Any], dtype: Any) -> dict:
    """Only cast the captured features and call the original final head towers."""
    require(dtype in (torch.float32, torch.float16), "Wrong head arithmetic dtype")
    require(not head.training and len(features) == head.nl == 3, "Head must be eval with three feature scales")
    require(all(not module.training for module in head.modules()), "A head submodule is in training mode")
    require(all(value.dtype == dtype for value in head.parameters()), "Head parameter dtype differs")
    converted = [value.to(dtype=dtype) for value in features]
    result = head.forward_head(converted, **head.one2many)
    require(isinstance(result, dict) and set(result) == {"boxes", "scores", "feats"}, "Unexpected pinned forward_head schema")
    require(result["boxes"].dtype == result["scores"].dtype == dtype, "Head output dtype changed")
    return result


def tensor_exact(torch: Any, a: Any, b: Any) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))


def run(p: dict, output: Path, receipt: dict) -> None:
    decoder = load_decoder(p["decoder_runner"])
    qualification = Path(p["qualification_dir"]).resolve(strict=True)
    rows = read_json(decoder.safe_file(qualification, "debug_source_manifest.json"))["rows"]
    selected = validate_head_protocol(p, sha(Path(__file__)), rows, decoder)
    q, historical, prepared_rows, verified = decoder.historical_inputs(decoder_adapter(p, decoder), output)
    require([r["image_id"] for r in selected] == [r["image_id"] for r in prepared_rows], "Debug helper selection differs")
    reference = Path(p["reference_decode_probe_dir"]).resolve(strict=True)
    reference_manifest = decoder.bound_identity(p["reference_decode_probe_manifest"], "reference decode probe manifest")
    require(reference_manifest == reference / "manifest.json", "Reference manifest/root mismatch")
    ref_files = read_json(reference_manifest)["files"]
    for relative, entry in ref_files.items():
        file = decoder.safe_file(reference, relative)
        require(file.stat().st_size == entry["bytes"] and sha(file) == entry["sha256"], "Reference decode probe changed")
    ref_receipt = read_json(decoder.safe_file(reference, "receipt.json"))
    require(ref_receipt.get("status") == "COMPLETE_DIAGNOSTIC_ONLY"
            and ref_receipt.get("historical_bytes_all_equal") is True
            and ref_receipt.get("native_reconstruction_all_exact") is True
            and ref_receipt.get("images_completed") == len(selected), "Reference decode probe incomplete or not exact")
    ref_protocol = read_json(decoder.safe_file(reference, "protocol.json"))
    require(ref_protocol["image_ids"] == p["image_ids"] and ref_protocol["checkpoint"]["sha256"] == p["checkpoint"]["sha256"],
            "Reference decode cohort/checkpoint differs")
    write_json(output / "runtime_adapter.json", {"purpose": "reuse frozen decoder input validation and runtime helpers",
               "changes_from_original_protocol": {"schema": decoder.SCHEMA, "runner_sha256": DECODER_SHA},
               "original_head_protocol_sha256": receipt["protocol_sha256"], "tolerances_or_runtime_controls_changed": False})
    torch, cv2, LetterBox, _, _, environment = q.configure_runtime(output)
    require(environment["packages"] == p["package_versions"], "Live packages changed")
    for key, expected in decoder.CONTROLS.items():
        require(environment[key] == expected, f"Runtime precision control changed: {key}")
    import ultralytics
    framework_root = Path(ultralytics.__file__).resolve().parent
    for relative, expected in p["framework_source_sha256"].items():
        require(sha(decoder.safe_file(framework_root, relative)) == expected, "Live framework source changed")
    old_environment = verified["historical_environment"]
    require(all(environment[key] == old_environment[key] for key in ("cuda", "cudnn", "gpu"))
            and environment["executable"]["sha256"] == old_environment["executable"]["sha256"], "Historical runtime identity changed")
    write_json(output / "environment.json", environment)
    models, provenance = q.load_models(torch, verified["checkpoint"])
    state_equality = decoder.compare_lifted_states(torch, models["fp32"], models["fp16"])
    heads = {branch: model.model[-1] for branch, model in models.items()}
    head_state = {branch: {key: value.clone() for key, value in head.state_dict().items()} for branch, head in heads.items()}
    for branch, head in heads.items():
        require(head.nc == 10 and head.nl == 3 and head.reg_max > 1 and not head.training and not head.dynamic
                and not head.end2end and not head.export and not head.xyxy, "Unsupported native Detect head")
        require(torch.equal(head.dfl.conv.weight.flatten().float(), torch.arange(head.reg_max, device="cuda:0").float()), "Wrong DFL bins")
    require(heads["fp32"].reg_max == heads["fp16"].reg_max, "Head DFL dimensions differ")
    reg_max = int(heads["fp32"].reg_max)
    write_json(output / "model_provenance.json", {"models": provenance, "checkpoint": decoder.identity(verified["checkpoint"]),
               "full_model_state_equal_after_float32_lift": state_equality, "head_batchnorm_eval": True,
               "reg_max": reg_max, "nc": 10, "nl": 3, "fusion_called": False})
    letterbox = LetterBox((640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32,
                         padding_value=114, interpolation=cv2.INTER_LINEAR)
    raw_dir, details_dir = output / "raw", output / "per_image"
    raw_dir.mkdir(); details_dir.mkdir()
    items = []
    receipt.update(phase="DEBUG_HEAD_CROSS_INTERVENTION", inference_started=True)
    write_json(output / "receipt.json", receipt)
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for number, row in enumerate(selected, 1):
            stem = Path(row["image_path"]).stem
            historical_detail = read_json(decoder.safe_file(historical, f"per_image/{stem}.json"))
            require(historical_detail["image_id"] == row["image_id"], "Historical image identity differs")
            image_path = decoder.safe_file(historical, "debug/images/" + Path(row["image_path"]).name)
            require(sha(image_path) == row["image_sha256"], "Debug image bytes changed")
            image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(image is not None and image.shape == (row["height"], row["width"], 3), "Decoded dimensions changed")
            params = letterbox.get_params({"img": image})
            padded = letterbox.apply_image({"img": image}, params)["img"]
            require(padded.shape == (640, 640, 3), "Letterbox shape changed")
            rgb_chw = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))
            normalized = torch.from_numpy(rgb_chw).unsqueeze(0).float().div(255)
            uint8_sha = hashlib.sha256(rgb_chw.tobytes()).hexdigest()
            input_sha = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
            require(uint8_sha == historical_detail["preprocessed_uint8_sha256"]
                    and input_sha == historical_detail["normalized_fp32_input_sha256"], "Historical input bytes differ")
            serial_params = {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()}
            require(serial_params == historical_detail["letterbox_params"], "Letterbox parameters differ")
            arrays, captures, feature_copies = {"input_fp32_cpu": normalized.numpy().copy()}, {}, {}
            tensor = normalized.to("cuda:0")
            for source, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
                out = models[source](tensor if source == "fp32" else tensor.half(), augment=False)
                require(isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict), "Unexpected native forward schema")
                forward, native = out
                require(set(native) == {"boxes", "scores", "feats"} and tuple(forward.shape) == (1, 14, 8400)
                        and forward.dtype == dtype, "Native forward semantics changed")
                require(len(native["feats"]) == 3 and all(value.dtype == dtype for value in native["feats"]), "Feature scales/dtypes differ")
                require([tuple(value.shape[-2:]) for value in native["feats"]] == [(80, 80), (40, 40), (20, 20)]
                        and all(value.ndim == 4 and value.shape[0] == 1 for value in native["feats"]), "Unexpected feature geometry")
                require(tuple(native["boxes"].shape) == (1, 4 * reg_max, 8400)
                        and tuple(native["scores"].shape) == (1, 10, 8400), "Unexpected native logits")
                head = heads[source]
                require(head.anchors.dtype == head.strides.dtype == dtype, "Runtime geometry dtype differs")
                captures[source] = {"forward": forward, "box_logits": native["boxes"], "class_logits": native["scores"],
                                    "anchors": head.anchors.clone(), "strides": head.strides.clone(), "features": native["feats"]}
                for key in ("forward", "box_logits", "class_logits", "anchors", "strides"):
                    value = captures[source][key]
                    require(bool(torch.isfinite(value).all()), "Nonfinite native capture")
                    arrays[f"{source}_{key}"] = value.detach().cpu().numpy().copy()
                arrays[f"{source}_dfl_bins"] = head.dfl.conv.weight.detach().flatten().cpu().numpy().copy()
                feature_copies[source] = [value.clone() for value in native["feats"]]
                for scale, value in enumerate(native["feats"]):
                    require(bool(torch.isfinite(value).all()), "Nonfinite native head input feature")
                    arrays[f"{source}_feat_scale{scale}"] = value.detach().cpu().numpy().copy()
            geometry_equal = all(tensor_exact(torch, captures["fp32"][key].float(), captures["fp16"][key].float())
                                 for key in ("anchors", "strides"))
            require(geometry_equal, "Native geometry differs after float32 lift")
            require(all(captures["fp32"]["features"][i].shape == captures["fp16"]["features"][i].shape for i in range(3)),
                    "Feature origin shapes differ")
            native_exact = {}
            for source in ("fp32", "fp16"):
                for arithmetic, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
                    head = heads[arithmetic]
                    logits = cross_head(torch, head, captures[source]["features"], dtype)
                    prefix = f"source_{source}_head_{arithmetic}"
                    for name, value in (("box_logits", logits["boxes"]), ("class_logits", logits["scores"])):
                        require(bool(torch.isfinite(value).all()), "Nonfinite intervention logits")
                        arrays[f"{prefix}_{name}"] = value.detach().cpu().numpy().copy()
                    if source == arithmetic:
                        native_exact[source] = all(tensor_exact(torch, logits[actual], captures[source][name])
                                                  for actual, name in (("boxes", "box_logits"), ("scores", "class_logits")))
                    decoded = decoder.decode_functional(torch, logits["boxes"], logits["scores"],
                                                         captures[source]["anchors"], captures[source]["strides"], reg_max, torch.float32)
                    for name, value in decoded.items():
                        require(bool(torch.isfinite(value).all()), "Nonfinite unified FP32 decoder result")
                        arrays[f"{prefix}_{name}"] = value.detach().cpu().numpy().copy()
            feature_unchanged = {source: all(tensor_exact(torch, value, saved) for value, saved in zip(captures[source]["features"], feature_copies[source]))
                                 for source in ("fp32", "fp16")}
            feature_sha_before = {source: [hashlib.sha256(arrays[f"{source}_feat_scale{scale}"].tobytes()).hexdigest()
                                           for scale in range(3)] for source in ("fp32", "fp16")}
            feature_sha_after = {source: [hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                                          for value in captures[source]["features"]] for source in ("fp32", "fp16")}
            head_state_unchanged = {branch: all(tensor_exact(torch, value, head_state[branch][key])
                                               for key, value in head.state_dict().items()) for branch, head in heads.items()}
            torch.cuda.synchronize()
            np.savez_compressed(raw_dir / f"{stem}.npz", **arrays)
            with np.load(decoder.safe_file(historical, f"raw/{stem}.npz"), allow_pickle=False) as old:
                historical_equal = {branch: decoder.bytes_equal(arrays[f"{branch}_forward"], old[branch]) for branch in ("fp32", "fp16")}
            with np.load(decoder.safe_file(reference, f"raw/{stem}.npz"), allow_pickle=False) as ref:
                reference_equal = {branch: all(decoder.bytes_equal(arrays[f"{branch}_{key}"], ref[f"{branch}_{key}"])
                                                for key in ("forward", "box_logits", "class_logits", "anchors", "strides", "dfl_bins"))
                                   for branch in ("fp32", "fp16")}
                reference_decode_equal = {branch: decoder.bytes_equal(arrays[f"source_{branch}_head_{branch}_decoded"],
                                                                       ref[f"source_{branch}_decode_fp32_decoded"])
                                          for branch in ("fp32", "fp16")}
            comparisons = {name: q.raw_guard(arrays[left], arrays[right]) for name, (left, right) in COMPARISONS.items()}
            logit_differences = {}
            for name, (left, right) in COMPARISONS.items():
                if not left.endswith("_decoded"):
                    continue
                logit_differences[name] = {}
                for logit in ("box_logits", "class_logits"):
                    a = arrays[left.removesuffix("decoded") + logit].astype(np.float64)
                    b = arrays[right.removesuffix("decoded") + logit].astype(np.float64)
                    difference = np.abs(a - b)
                    logit_differences[name][logit] = {"elements": int(difference.size), "unequal_elements": int(np.count_nonzero(difference)),
                                                    "max_absolute_error": float(difference.max()), "mean_absolute_error": float(difference.mean())}
            detail = {"image_id": row["image_id"], "image_sha256": row["image_sha256"], "array_file": f"raw/{stem}.npz",
                      "reg_max": reg_max, "nc": 10, "nl": 3, "normalized_fp32_input_sha256": input_sha,
                      "preprocessed_uint8_sha256": uint8_sha, "letterbox_params": serial_params,
                      "feature_shapes": {s: [list(x.shape) for x in captures[s]["features"]] for s in ("fp32", "fp16")},
                      "feature_cast": "Each captured feature is explicitly cast to the receiving head arithmetic dtype; FP32 decoding casts head logits and geometry to FP32.",
                      "historical_byte_equal": historical_equal, "reference_decode_capture_exact": reference_equal,
                      "reference_unified_decode_exact": reference_decode_equal, "native_head_logits_exact": native_exact,
                      "runtime_geometry_equal_after_float_lift": geometry_equal, "features_unchanged_after_intervention": feature_unchanged,
                      "feature_sha256_before": feature_sha_before, "feature_sha256_after": feature_sha_after,
                      "head_state_unchanged_after_intervention": head_state_unchanged, "comparisons": comparisons,
                      "logit_differences": logit_differences, "old_guard_replaced": False, "tolerances_changed": False}
            write_json(details_dir / f"{stem}.json", detail)
            require(all(historical_equal.values()) and all(reference_equal.values()) and all(reference_decode_equal.values()),
                    "Historical/reference captures differ; stopped and preserved")
            require(all(native_exact.values()) and all(feature_unchanged.values()) and all(head_state_unchanged.values()),
                    "Native head exactness or immutable feature/state check failed; stopped and preserved")
            items.append(detail)
            receipt["images_completed"] = number
            write_json(output / "receipt.json", receipt)
            print(f"HEAD_PROBE {number}/{len(selected)} {row['image_id']} NATIVE_AND_HISTORY_EXACT", flush=True)
    for key in ("checkpoint", "qualifier", "decoder_runner", "runtime_environment", "qualification_manifest", "reference_decode_probe_manifest"):
        decoder.bound_identity(p[key], f"{key} after execution")
    summary = {"schema": "rampq.gate1.head_precision_probe_summary/v1", "stage": p["stage"], "images": len(items),
               "historical_bytes_all_equal": True, "reference_decode_captures_all_exact": True,
               "reference_unified_decode_all_exact": True, "native_head_logits_all_exact": True,
               "features_and_head_states_unchanged": True, "comparisons": {}, "raw_tolerances": decoder.TOLERANCES,
               "all_intervention_decode_dtype": "fp32", "original_guard_replaced": False, "scientific_gate_pass": False, "limits": LIMITS}
    for name in COMPARISONS:
        guards = [item["comparisons"][name] for item in items]
        summary["comparisons"][name] = {"diagnostic_guard_pass_images": sum(g["allclose_pass"] for g in guards),
            "diagnostic_guard_fail_images": sum(not g["allclose_pass"] for g in guards),
            **{section: {"violations": sum(g[section]["violations"] for g in guards),
                         "max_absolute_error": max(g[section]["max_absolute_error"] for g in guards),
                         "max_error_over_tolerance": max(g[section]["max_error_over_tolerance"] for g in guards)}
               for section in ("boxes", "scores")}}
    write_json(output / "summary.json", summary)
    receipt.update(status="COMPLETE_HEAD_DIAGNOSTIC_ONLY", phase="COMPLETE", images_completed=len(items),
                   historical_bytes_all_equal=True, reference_decode_captures_all_exact=True, native_head_logits_all_exact=True,
                   reference_unified_decode_all_exact=True, features_and_head_states_unchanged=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    require(sha(protocol_path) == args.protocol_sha256, "Protocol pre-execution hash differs")
    p = read_json(protocol_path)
    require(p.get("schema") == SCHEMA and p.get("runner_sha256") == sha(Path(__file__)), "Head protocol/source binding differs")
    output = args.output.resolve()
    require(not output.exists(), "Output must be a new directory")
    for key in ("qualification_dir", "reference_decode_probe_dir"):
        source_root = Path(p[key]).resolve(strict=True)
        require(not output.is_relative_to(source_root) and not source_root.is_relative_to(output), "Output overlaps immutable input")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output / "protocol.json")
    shutil.copyfile(Path(__file__), output / "runner_source.py")
    started = time.monotonic()
    receipt = {"schema": "rampq.gate1.head_precision_probe_receipt/v1", "status": "RUNNING", "phase": "INPUT_VALIDATION",
               "started_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": args.protocol_sha256,
               "runner_sha256": sha(Path(__file__)), "stage": p.get("stage"), "images_requested": len(p.get("image_ids", [])),
               "images_completed": 0, "inference_started": False, "data_role": "debug", "selection_detector_inference": False,
               "holdout_accessed": False, "scientific_gate_pass": False, "accepted_float_baseline_ready": False,
               "old_guard_replaced": False, "tolerances_changed": False, "limits": LIMITS}
    write_json(output / "receipt.json", receipt)
    try:
        run(p, output, receipt)
    except Exception as error:
        receipt.update(status="FAILED_STOPPED_PRESERVED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "receipt.json", receipt)
        files = {path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        write_json(output / "manifest.json", {"schema": "rampq.gate1.head_precision_probe_manifest/v1", "files": files})


if __name__ == "__main__":
    main()
