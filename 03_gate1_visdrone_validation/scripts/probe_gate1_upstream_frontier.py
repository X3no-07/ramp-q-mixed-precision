"""Bounded debug-only DAG frontier precision intervention; no acceptance claim.

Two native unfused forwards capture all DAG frontier values.  Every complete
manual replay and same-dtype suffix replay must reproduce the native logits
exactly before crossed-prefix values may be interpreted.  Only ten crossed
variants are stored; repeated controls are represented by exactness and hashes.
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


SCHEMA = "rampq.gate1.upstream_frontier_probe_protocol/v1"
DECODER_SHA = "f76f3bfd677487fa7d87947f866daaf7407ddd455ad3e4d9e399da719e481ec0"
HEAD_SHA = "842e16b7b743c6044975ab2faad5725ead19cd8c793575340c9ee81ddaaef23f"
CUTS = [-1, 4, 6, 10, 22]
FRONTIERS = {"-1": [-1], "4": [4], "6": [4, 6], "10": [4, 6, 10], "22": [16, 19, 22]}
PILOT_IDS = ["train/0000170_01201_d_0000001", "train/9999990_00000_d_0000024", "train/0000046_00000_d_0000087"]
PAIRS = [("fp32", "fp16"), ("fp16", "fp32")]
INTERVENTION = {"prefix_sources": ["fp32", "fp16"], "suffix_compute_dtypes": ["fp32", "fp16"],
                "stored_pairs": [["fp32", "fp16"], ["fp16", "fp32"]],
                "all_comparisons_decode_dtype": "fp32", "all_native_module_replay_exact_required": True,
                "all_same_dtype_cut_replay_exact_required": True, "frontier_policy": "all DAG edges crossing the cut",
                "cut_indices": CUTS}
LIMITS = [
    "Post-hoc bounded arithmetic diagnosis on 109 reused, train-derived debug images; no independent test evidence.",
    "A cut intervention replaces every DAG edge crossing that cut, not a unique operation or layer cause.",
    "The input cut includes input rounding; cut 22 repeats the existing head endpoint and is not independent evidence.",
    "FP16 saved EMA lifted to FP32 is not recovered original FP32 training state.",
    "All crossed outputs use FP32 decoding; this is not an accepted deployment modification.",
    "No Selection, Holdout, annotations, NMS, target matching, AP, quantization, TensorRT, Orin, speed or energy evidence.",
    "Diagnostic comparisons never replace either failing frozen guard or accept Gate1.",
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


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def variant_key(cut: int, source: str, suffix: str) -> str:
    return f"cut_{cut}_source_{source}_suffix_{suffix}"


def comparison_map() -> dict:
    result = {"original_fp32_vs_original_fp16": ("fp32_forward", "fp16_forward"),
              "native_heads_both_decode_fp32": ("native_fp32_decoded", "native_fp16_decoded")}
    for cut in CUTS:
        for source, suffix in PAIRS:
            key = variant_key(cut, source, suffix)
            result[f"reference_fp32_vs_{key}"] = ("native_fp32_decoded", key + "_decoded")
    return result


COMPARISONS = comparison_map()


def endpoint_aliases(arrays: Any, reference: Any) -> dict:
    """Predeclared repeated endpoints must remain byte-identical, not new evidence."""
    def same(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    result = {}
    for source, suffix in PAIRS:
        key = variant_key(22, source, suffix)
        result[f"cut22_{source}_to_{suffix}"] = all(same(arrays[f"{key}_{part}"], reference[f"source_{source}_head_{suffix}_{part}"])
                                                    for part in ("box_logits", "class_logits", "decoded"))
    result["input_fp32_to_fp16"] = all(same(arrays[variant_key(-1, "fp32", "fp16") + "_" + part], arrays[f"fp16_{part}" if part != "decoded" else "native_fp16_decoded"])
                                      for part in ("box_logits", "class_logits", "decoded"))
    return result


def topology_of_model(model: Any) -> dict:
    """Metadata only; caller is responsible for safely loading a CPU model."""
    modules = []
    for expected, module in enumerate(model.model):
        require(module.i == expected, "Top-level module indices are not sequential")
        sources = module.f if isinstance(module.f, int) else list(module.f)
        modules.append({"i": expected, "f": sources, "type": type(module).__module__ + "." + type(module).__name__})
    topology = {"modules": modules, "save": sorted(int(i) for i in model.save)}
    validate_topology(topology)
    return topology


def dependencies(topology: dict) -> dict[int, list[int]]:
    modules = topology.get("modules")
    require(isinstance(modules, list) and len(modules) >= 2, "Invalid module topology")
    result = {}
    for i, module in enumerate(modules):
        require(module.get("i") == i and isinstance(module.get("type"), str), "Invalid topology module identity")
        f = module.get("f")
        sources = [f] if isinstance(f, int) and not isinstance(f, bool) else f
        require(isinstance(sources, list) and bool(sources), "Invalid module sources")
        resolved = []
        for source in sources:
            require(isinstance(source, int) and not isinstance(source, bool), "Noninteger module dependency")
            target = i + source if source < 0 else source
            require(-1 <= target < i and (target != -1 or i == 0), "Forward/out-of-range module dependency")
            resolved.append(target)
        result[i] = resolved
    return result


def frontier_indices(topology: dict, cut: int) -> list[int]:
    deps = dependencies(topology)
    require(isinstance(cut, int) and -1 <= cut < len(deps) - 1, "Cut must precede final module")
    return sorted({source for i, sources in deps.items() if i > cut for source in sources if source <= cut})


def validate_topology(topology: dict) -> None:
    deps = dependencies(topology)
    save = topology.get("save")
    require(isinstance(save, list) and save == sorted(set(save))
            and all(isinstance(i, int) and 0 <= i < len(deps) for i in save), "Invalid model.save")
    # Every non-immediate edge in the native implementation reads model.save.
    required_save = {source for i, sources in deps.items() for source in sources if source >= 0 and source != i - 1}
    require(required_save.issubset(set(save)), "model.save omits a skip dependency")


def validate_protocol(p: dict, own_sha: str, rows: list[dict], decoder: Any) -> list[dict]:
    require(p.get("schema") == SCHEMA and p.get("runner_sha256") == own_sha, "Upstream protocol/source binding differs")
    require(p.get("decoder_runner", {}).get("sha256") == DECODER_SHA and p.get("head_runner", {}).get("sha256") == HEAD_SHA,
            "Frozen helper identity differs")
    require(p.get("intervention") == INTERVENTION and p.get("cuts") == CUTS and p.get("frontiers") == FRONTIERS,
            "Frozen intervention/cuts/frontiers changed")
    topology = p.get("topology", {})
    validate_topology(topology)
    require(len(topology["modules"]) == 24 and topology["modules"][-1]["f"] == [16, 19, 22], "Expected pinned 24-module YOLO11 topology")
    require({str(cut): frontier_indices(topology, cut) for cut in CUTS} == FRONTIERS, "Topology/frontier mismatch")
    if p.get("stage") == "pilot3":
        require(p.get("image_ids") == PILOT_IDS, "Frozen pilot IDs/order changed")
    adapted = dict(p, schema=decoder.SCHEMA, runner_sha256=DECODER_SHA)
    return decoder.validate_protocol(adapted, DECODER_SHA, rows)


def module_input(cache: dict, i: int, f: Any) -> Any:
    def get(source: int) -> Any:
        resolved = i + source if source < 0 else source
        require(resolved in cache, f"Missing DAG dependency {resolved} for module {i}")
        return cache[resolved]
    return get(f) if isinstance(f, int) else [get(source) for source in f]


def replay_suffix(torch: Any, model: Any, topology: dict, cut: int, frontier: dict[int, Any], dtype: Any,
                  captured_outputs: dict | None = None) -> Any:
    """Recompute every suffix module; preserve *all* crossing DAG dependencies."""
    require(dtype in (torch.float32, torch.float16), "Unsupported suffix dtype")
    require(not model.training and all(not m.training for m in model.modules()), "Suffix model must be eval")
    expected = frontier_indices(topology, cut)
    require(set(frontier) == set(expected), "Frontier contains missing or extra DAG inputs")
    require(all(value.is_floating_point() and bool(torch.isfinite(value).all()) for value in frontier.values()), "Invalid frontier tensor")
    require(all(value.dtype == dtype for value in model.parameters()), "Receiving model parameter dtype differs")
    cache = {i: value.to(dtype=dtype) for i, value in frontier.items()}
    for i in range(cut + 1, len(topology["modules"])):
        module = model.model[i]
        require(module.i == i and module.f == topology["modules"][i]["f"], "Runtime module graph changed")
        cache[i] = module(module_input(cache, i, module.f))
        if captured_outputs is not None and i < len(topology["modules"]) - 1:
            require(isinstance(cache[i], torch.Tensor), "Non-tensor replay prefix output")
            captured_outputs[i] = cache[i].clone()
    return cache[len(topology["modules"]) - 1]


def capture_native(torch: Any, model: Any, tensor: Any, topology: dict) -> tuple[Any, dict[int, Any], dict]:
    """Hook all tensor-valued top-level prefix outputs without replacing them."""
    cache, stats, handles = {-1: tensor.clone()}, {}, []
    def hook(index: int):
        def capture(_module: Any, _inputs: Any, result: Any) -> None:
            require(isinstance(result, torch.Tensor), "Non-tensor prefix output")
            cache[index] = result.clone()
        return capture
    try:
        for i, module in enumerate(model.model[:-1]):
            handles.append(module.register_forward_hook(hook(i)))
        result = model(tensor, augment=False)
    finally:
        for handle in handles:
            handle.remove()
    require(set(cache) == set(range(-1, len(topology["modules"]) - 1)), "Native prefix capture incomplete")
    for index, value in cache.items():
        stats[str(index)] = tensor_statistics(value)
    return result, cache, stats


def tensor_statistics(value: Any) -> dict:
    array = value.detach().cpu().numpy()
    require(bool(np.isfinite(array).all()), "Nonfinite tensor capture")
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": array_sha(array), "elements": int(array.size),
            "min": float(array.min()), "max": float(array.max()), "mean": float(array.astype(np.float64).mean())}


def native_parts(torch: Any, result: Any, head: Any, dtype: Any) -> dict:
    require(isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict), "Unexpected native forward output")
    forward, native = result
    require(set(native) == {"boxes", "scores", "feats"} and forward.dtype == dtype and tuple(forward.shape) == (1, 14, 8400),
            "Pinned forward layout/dtype changed")
    require(tuple(native["boxes"].shape) == (1, 4 * int(head.reg_max), 8400)
            and tuple(native["scores"].shape) == (1, 10, 8400), "Native logit shape changed")
    result = {"forward": forward, "box_logits": native["boxes"], "class_logits": native["scores"],
              "anchors": head.anchors.clone(), "strides": head.strides.clone(), "dfl_bins": head.dfl.conv.weight.flatten().clone()}
    require(all(v.dtype == dtype and bool(torch.isfinite(v).all()) for v in result.values()), "Invalid native capture dtype/values")
    return result


def exact(torch: Any, a: Any, b: Any) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))


def run(p: dict, output: Path, receipt: dict) -> None:
    head_path = Path(p["head_runner"]["path"])
    require(head_path.is_file() and head_path.stat().st_size == p["head_runner"]["bytes"] and sha(head_path) == HEAD_SHA,
            "Frozen head helper changed")
    decoder_path = Path(p["decoder_runner"]["path"])
    require(decoder_path.is_file() and decoder_path.stat().st_size == p["decoder_runner"]["bytes"] and sha(decoder_path) == DECODER_SHA,
            "Frozen decoder changed")
    spec = importlib.util.spec_from_file_location("frozen_decoder_for_upstream", decoder_path)
    decoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(decoder)
    rows = read_json(decoder.safe_file(Path(p["qualification_dir"]), "debug_source_manifest.json"))["rows"]
    selected = validate_protocol(p, sha(Path(__file__)), rows, decoder)
    adapted = dict(p, schema=decoder.SCHEMA, runner_sha256=DECODER_SHA)
    q, historical, prepared_rows, verified = decoder.historical_inputs(adapted, output)
    require(selected == prepared_rows, "Historical helper selected a different cohort")
    reference = Path(p["reference_head_probe_dir"]).resolve(strict=True)
    reference_manifest = decoder.bound_identity(p["reference_head_probe_manifest"], "reference head probe manifest")
    require(reference_manifest == reference / "manifest.json", "Reference manifest/root mismatch")
    for relative, entry in read_json(reference_manifest)["files"].items():
        file = decoder.safe_file(reference, relative)
        require(file.stat().st_size == entry["bytes"] and sha(file) == entry["sha256"], "Reference head output changed")
    ref_receipt = read_json(decoder.safe_file(reference, "receipt.json"))
    require(ref_receipt.get("status") == "COMPLETE_HEAD_DIAGNOSTIC_ONLY" and ref_receipt.get("images_completed") == 109
            and ref_receipt.get("native_head_logits_all_exact") is True and ref_receipt.get("features_and_head_states_unchanged") is True,
            "Reference head probe not complete/exact")
    ref_protocol = read_json(decoder.safe_file(reference, "protocol.json"))
    require(ref_protocol["checkpoint"]["sha256"] == p["checkpoint"]["sha256"]
            and set(p["image_ids"]).issubset(set(ref_protocol["image_ids"])), "Reference checkpoint/cohort differs")
    torch, cv2, LetterBox, _, _, environment = q.configure_runtime(output)
    require(environment["packages"] == p["package_versions"], "Live package identities changed")
    require(all(environment[k] == value for k, value in decoder.CONTROLS.items()), "Precision control changed")
    import ultralytics
    framework_root = Path(ultralytics.__file__).resolve().parent
    for relative, expected in p["framework_source_sha256"].items():
        require(sha(decoder.safe_file(framework_root, relative)) == expected, "Live framework source changed")
    previous_environment = verified["historical_environment"]
    require(all(environment[k] == previous_environment[k] for k in ("cuda", "cudnn", "gpu"))
            and environment["executable"]["sha256"] == previous_environment["executable"]["sha256"], "Runtime identity changed")
    write_json(output / "environment.json", environment)
    models, provenance = q.load_models(torch, verified["checkpoint"])
    state_equal = decoder.compare_lifted_states(torch, models["fp32"], models["fp16"])
    require(all(topology_of_model(model) == p["topology"] for model in models.values()), "Live topology differs from frozen metadata")
    heads = {source: model.model[-1] for source, model in models.items()}
    for head in heads.values():
        require(head.nc == 10 and head.nl == 3 and not head.dynamic and not head.export and not head.end2end and not head.xyxy,
                "Unsupported final head")
    require(heads["fp32"].reg_max == heads["fp16"].reg_max, "Branch DFL dimensions differ")
    reg_max = int(heads["fp32"].reg_max)
    states = {source: {k: v.clone() for k, v in model.state_dict().items()} for source, model in models.items()}
    write_json(output / "model_provenance.json", {"models": provenance, "state_equal_after_float32_lift": state_equal,
               "topology": p["topology"], "cuts": CUTS, "frontiers": FRONTIERS, "reg_max": reg_max, "fusion_called": False})
    letterbox = LetterBox((640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32,
                         padding_value=114, interpolation=cv2.INTER_LINEAR)
    raw_dir = output / "raw"; details_dir = output / "per_image"
    raw_dir.mkdir(); details_dir.mkdir()
    union = sorted({i for values in FRONTIERS.values() for i in values})
    items = []
    receipt.update(phase="NATIVE_AND_DAG_REPLAY", inference_started=True)
    write_json(output / "receipt.json", receipt)
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for number, row in enumerate(selected, 1):
            stem = Path(row["image_path"]).stem
            old_detail = read_json(decoder.safe_file(historical, f"per_image/{stem}.json"))
            image_path = decoder.safe_file(historical, "debug/images/" + Path(row["image_path"]).name)
            require(sha(image_path) == row["image_sha256"], "Debug image bytes changed")
            image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(image is not None and image.shape == (row["height"], row["width"], 3), "Decoded image dimensions changed")
            params = letterbox.get_params({"img": image})
            padded = letterbox.apply_image({"img": image}, params)["img"]
            rgb = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))
            normalized = torch.from_numpy(rgb).unsqueeze(0).float().div(255)
            uint8_sha, input_sha = array_sha(rgb), array_sha(normalized.numpy())
            serial_params = {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}
            require(uint8_sha == old_detail["preprocessed_uint8_sha256"] and input_sha == old_detail["normalized_fp32_input_sha256"]
                    and serial_params == old_detail["letterbox_params"], "Historical preprocessing changed")
            arrays, native, caches, layer_stats = {"input_fp32_cpu": normalized.numpy().copy()}, {}, {}, {}
            manual_exact, manual_sha, manual_module_exact, manual_module_sha = {}, {}, {}, {}
            tensor = normalized.to("cuda:0")
            for source, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
                result, cache, statistics = capture_native(torch, models[source], tensor.to(dtype=dtype), p["topology"])
                native[source] = native_parts(torch, result, heads[source], dtype)
                caches[source], layer_stats[source] = cache, statistics
                for name, value in native[source].items():
                    arrays[f"{source}_{name}"] = value.detach().cpu().numpy().copy()
                for index in union:
                    arrays[f"native_{source}_layer_{index}"] = cache[index].detach().cpu().numpy().copy()
                replay_outputs = {}
                manual = native_parts(torch, replay_suffix(torch, models[source], p["topology"], -1, {-1: cache[-1]}, dtype,
                                                           replay_outputs), heads[source], dtype)
                require(set(replay_outputs) == set(range(23)), "Manual replay prefix captures incomplete")
                manual_module_exact[source] = {str(i): exact(torch, value, cache[i]) for i, value in replay_outputs.items()}
                manual_module_sha[source] = {str(i): array_sha(value.detach().cpu().numpy()) for i, value in replay_outputs.items()}
                manual_exact[source] = all(exact(torch, native[source][key], manual[key]) for key in ("forward", "box_logits", "class_logits"))
                manual_sha[source] = {}
                for key in ("forward", "box_logits", "class_logits"):
                    arrays[f"manual_{source}_{key}"] = manual[key].detach().cpu().numpy().copy()
                    manual_sha[source][key] = array_sha(arrays[f"manual_{source}_{key}"])
                require(manual_exact[source] and all(manual_module_exact[source].values()),
                        "Manual complete native replay differs; stopped before interventions")
                decoded = decoder.decode_functional(torch, native[source]["box_logits"], native[source]["class_logits"],
                                                      native[source]["anchors"], native[source]["strides"], reg_max, torch.float32)["decoded"]
                arrays[f"native_{source}_decoded"] = decoded.detach().cpu().numpy().copy()
            geometry_equal = all(exact(torch, native["fp32"][key].float(), native["fp16"][key].float()) for key in ("anchors", "strides", "dfl_bins"))
            require(geometry_equal, "Runtime geometry differs after FP32 lift")
            with np.load(decoder.safe_file(historical, f"raw/{stem}.npz"), allow_pickle=False) as old:
                historical_equal = {source: decoder.bytes_equal(arrays[f"{source}_forward"], old[source]) for source in models}
            with np.load(decoder.safe_file(reference, f"raw/{stem}.npz"), allow_pickle=False) as ref:
                reference_exact = {source: all(decoder.bytes_equal(arrays[f"{source}_{key}"], ref[f"{source}_{key}"])
                                               for key in native[source])
                                   and all(decoder.bytes_equal(arrays[f"native_{source}_layer_{index}"], ref[f"{source}_feat_scale{scale}"])
                                           for scale, index in enumerate((16, 19, 22))) for source in models}
                decode_exact = {source: decoder.bytes_equal(arrays[f"native_{source}_decoded"], ref[f"source_{source}_head_{source}_decoded"])
                                for source in models}
            require(all(historical_equal.values()) and all(reference_exact.values()) and all(decode_exact.values()),
                    "Historical/native reference bytes differ; stopped before interventions")
            before_sha = {source: {str(i): array_sha(arrays[f"native_{source}_layer_{i}"]) for i in union} for source in models}
            same_exact, same_sha, casts = {}, {}, {}
            # Exact same-dtype controls precede every crossed intervention.
            for cut in CUTS:
                same_exact[str(cut)], same_sha[str(cut)] = {}, {}
                for source, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
                    frontier = {i: caches[source][i] for i in FRONTIERS[str(cut)]}
                    result = native_parts(torch, replay_suffix(torch, models[source], p["topology"], cut, frontier, dtype), heads[source], dtype)
                    decoded = decoder.decode_functional(torch, result["box_logits"], result["class_logits"],
                                                          result["anchors"], result["strides"], reg_max, torch.float32)["decoded"]
                    same_exact[str(cut)][source] = all(exact(torch, result[k], native[source][k]) for k in ("forward", "box_logits", "class_logits"))
                    same_exact[str(cut)][source] &= decoder.bytes_equal(decoded.detach().cpu().numpy(), arrays[f"native_{source}_decoded"])
                    same_sha[str(cut)][source] = {key: array_sha(value.detach().cpu().numpy()) for key, value in
                                                 (("box_logits", result["box_logits"]), ("class_logits", result["class_logits"]), ("decoded", decoded))}
                    require(same_exact[str(cut)][source], "Same-dtype cut replay differs; stopped before crossed replay")
                for source, suffix in PAIRS:
                    dtype = torch.float32 if suffix == "fp32" else torch.float16
                    key = variant_key(cut, source, suffix)
                    frontier = {i: caches[source][i] for i in FRONTIERS[str(cut)]}
                    casts[key] = {str(i): tensor_statistics(value.to(dtype=dtype)) for i, value in frontier.items()}
                    result = native_parts(torch, replay_suffix(torch, models[suffix], p["topology"], cut, frontier, dtype), heads[suffix], dtype)
                    decoded = decoder.decode_functional(torch, result["box_logits"], result["class_logits"],
                                                          result["anchors"], result["strides"], reg_max, torch.float32)["decoded"]
                    for name, value in (("box_logits", result["box_logits"]), ("class_logits", result["class_logits"]), ("decoded", decoded)):
                        require(bool(torch.isfinite(value).all()), "Nonfinite crossed output")
                        arrays[key + "_" + name] = value.detach().cpu().numpy().copy()
            after_sha = {source: {str(i): array_sha(caches[source][i].detach().cpu().numpy()) for i in union} for source in models}
            with np.load(decoder.safe_file(reference, f"raw/{stem}.npz"), allow_pickle=False) as ref:
                aliases = endpoint_aliases(arrays, ref)
            require(all(aliases.values()), "A predeclared repeated endpoint differs from frozen reference")
            state_unchanged = {source: all(exact(torch, value, states[source][key]) for key, value in model.state_dict().items())
                               for source, model in models.items()}
            require(before_sha == after_sha and all(state_unchanged.values()), "Frontier/model state mutated by replay")
            comparisons = {name: q.raw_guard(arrays[left], arrays[right]) for name, (left, right) in COMPARISONS.items()}
            detail = {"image_id": row["image_id"], "image_sha256": row["image_sha256"], "array_file": f"raw/{stem}.npz",
                      "reg_max": reg_max, "nc": 10, "nl": 3, "normalized_fp32_input_sha256": input_sha,
                      "preprocessed_uint8_sha256": uint8_sha, "letterbox_params": serial_params,
                      "historical_byte_equal": historical_equal, "reference_head_capture_exact": reference_exact,
                      "reference_unified_decode_exact": decode_exact, "native_manual_replay_exact": manual_exact,
                      "native_manual_replay_sha256": manual_sha, "native_suffix_replay_exact": same_exact,
                      "native_manual_module_replay_exact": manual_module_exact, "native_manual_module_replay_sha256": manual_module_sha,
                      "native_suffix_replay_sha256": same_sha, "frontier_sha256_before": before_sha,
                      "frontier_sha256_after": after_sha, "variant_frontier_casts": casts, "native_layer_statistics": layer_stats,
                      "runtime_geometry_equal_after_float_lift": geometry_equal, "model_states_unchanged": state_unchanged,
                      "endpoint_aliases_exact": aliases,
                      "comparisons": comparisons, "old_guard_replaced": False, "tolerances_changed": False}
            torch.cuda.synchronize()
            np.savez_compressed(raw_dir / f"{stem}.npz", **arrays)
            write_json(details_dir / f"{stem}.json", detail)
            items.append(detail)
            receipt["images_completed"] = number
            write_json(output / "receipt.json", receipt)
            print(f"UPSTREAM_PROBE {number}/{len(selected)} {row['image_id']} ALL_NATIVE_CONTROLS_EXACT", flush=True)
    for key in ("checkpoint", "qualifier", "head_runner", "decoder_runner", "runtime_environment", "qualification_manifest", "reference_head_probe_manifest"):
        decoder.bound_identity(p[key], key + " after execution")
    summary = {"schema": "rampq.gate1.upstream_frontier_probe_summary/v1", "images": len(items), "stage": p["stage"],
               "cuts": CUTS, "frontiers": FRONTIERS, "comparisons": {}, "all_native_controls_exact": True,
               "historical_bytes_all_equal": True, "reference_head_captures_all_exact": True,
               "reference_unified_decode_all_exact": True, "frontiers_and_model_states_unchanged": True,
               "native_manual_replay_all_exact": True, "native_suffix_replay_all_exact": True, "frontiers_and_states_unchanged": True,
               "endpoint_aliases_all_exact": True,
               "raw_tolerances": decoder.TOLERANCES, "old_guard_replaced": False, "original_guard_replaced": False,
               "all_intervention_decode_dtype": "fp32", "scientific_gate_pass": False, "limits": LIMITS}
    for name in COMPARISONS:
        guards = [item["comparisons"][name] for item in items]
        summary["comparisons"][name] = {"diagnostic_guard_pass_images": sum(g["allclose_pass"] for g in guards),
            "diagnostic_guard_fail_images": sum(not g["allclose_pass"] for g in guards),
            **{section: {"violations": sum(g[section]["violations"] for g in guards),
                         "max_absolute_error": max(g[section]["max_absolute_error"] for g in guards),
                         "max_error_over_tolerance": max(g[section]["max_error_over_tolerance"] for g in guards)} for section in ("boxes", "scores")}}
    write_json(output / "summary.json", summary)
    receipt.update(status="COMPLETE_UPSTREAM_DIAGNOSTIC_ONLY", phase="COMPLETE", all_native_controls_exact=True,
                   historical_bytes_all_equal=True, reference_head_captures_all_exact=True,
                   reference_unified_decode_all_exact=True, frontiers_and_model_states_unchanged=True, endpoint_aliases_all_exact=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.protocol.resolve(strict=True)
    require(sha(path) == args.protocol_sha256, "Protocol pre-execution hash differs")
    p = read_json(path)
    require(p.get("schema") == SCHEMA and p.get("runner_sha256") == sha(Path(__file__)), "Protocol/source binding differs")
    output = args.output.resolve()
    require(not output.exists(), "Output must be a new directory")
    for key in ("qualification_dir", "reference_head_probe_dir"):
        source = Path(p[key]).resolve(strict=True)
        require(not output.is_relative_to(source) and not source.is_relative_to(output), "Output overlaps immutable source")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path, output / "protocol.json"); shutil.copyfile(Path(__file__), output / "runner_source.py")
    started = time.monotonic()
    receipt = {"schema": "rampq.gate1.upstream_frontier_probe_receipt/v1", "status": "RUNNING", "phase": "VALIDATION",
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
        files = {file.relative_to(output).as_posix(): {"bytes": file.stat().st_size, "sha256": sha(file)}
                 for file in sorted(output.rglob("*")) if file.is_file() and file.name != "manifest.json"}
        write_json(output / "manifest.json", {"schema": "rampq.gate1.upstream_frontier_probe_manifest/v1", "files": files})


if __name__ == "__main__":
    main()
