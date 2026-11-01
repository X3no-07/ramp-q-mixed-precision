"""Prepare/freeze/run a bounded, explicitly authorized Gate1 W8A8 debug pilot.

prepare reads source files and existing metadata only. Real execution requires
a separately frozen reviewed draft, its explicit SHA, and a fresh attempt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gate1/20261024_w8a8_debug_pilot_v1"
FORMAL = ROOT / "artifacts/gate1/20261024_formal_candidate_evaluation_v1"
PLAN = ROOT / "artifacts/gate1/20261022_split_refinement/split_plan_v1"
ENTRY = ROOT / "artifacts/gate1/20261024_w8a8_entry_audit_v1"
SCHEMA = "rampq.gate1.w8a8_debug_pilot/v1"
CONFIGS = ["fp32", "q_all", "q_only:G1", "r_only:G1"]
SEED = "2026102201"
TOLERANCES = {"box_atol": .5, "box_rtol": .005, "score_atol": .002, "score_rtol": .02}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write(path, payload, *, once=True):
    path = Path(path)
    with path.open("x" if once else "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def ordered_hash(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_member(row, role):
    require(row["assigned_split"] == role and row["source_split"] == "train" and row["image_id"].startswith("train/"), "Forbidden cohort role")
    require(type(row["width"]) is int and type(row["height"]) is int and min(row["width"], row["height"]) > 0, "Invalid dimensions")
    for key, prefix in [("image", "images/train/")] + ([("annotation", "annotations/train/")] if role == "debug" else []):
        relative = row[key + "_path"]
        require(relative.startswith(prefix) and ".." not in Path(relative).parts and not Path(relative).is_absolute(), "Escaped data path")
        value = row[key + "_sha256"]
        require(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), "Invalid member SHA")
    if role == "calibration_pool":
        require(not any("annotation" in key or "label" in key or "target" in key for key in row), "Calibration contract must not carry label fields")


def validate_contract(p):
    require(p["schema"] == SCHEMA and p["configurations"] == CONFIGS and p["seed"] == SEED, "Frozen configuration matrix differs")
    require(p["raw_tolerances"] == TOLERANCES and p["reference_policy"] == "strict_elementwise_FP32_identity_abort_on_any_difference", "Reference/guard weakened")
    require(p["input"] == {"size": 640, "batch": 1, "AMP": False, "device": "cuda:0", "deterministic": True, "TF32": False}, "Input precision differs")
    require(p["fusion"] == "BN_unfused_all_other_ops_FP32" and p["quantization"] == quantization_policy(), "Intervention semantics differ")
    require(p["budget"] == budget_policy(), "Execution budget differs")
    debug, calibration = p["debug"], p["calibration"]
    require(len(debug) == 8 and len(calibration) == 64, "Wrong fixed pilot size")
    for rows, role in ((debug, "debug"), (calibration, "calibration_pool")):
        require(len({row["image_id"] for row in rows}) == len(rows), "Duplicate cohort identity")
        for row in rows:
            validate_member(row, role)
    require([row["image_id"] for row in debug] == sorted(row["image_id"] for row in debug), "Debug order differs")
    require({row["image_id"] for row in debug}.isdisjoint(row["image_id"] for row in calibration), "Calibration/debug overlap")
    require(p["member_order_hashes"] == {"debug": ordered_hash(debug), "calibration": ordered_hash(calibration)}, "Cohort ordering/hash drift")
    require(p["postprocessing"] == postprocessing_policy(), "Thresholds or target semantics differ")


def quantization_policy():
    return {"activation": "Conv input per-tensor signed symmetric scale=max(max_abs/127,1e-12); torch.round ties-to-even; clamp[-127,127]; zero_point0",
            "weight": "per-output-channel axis0 scale=max(absmax,1e-12)/127; torch.round; clamp[-127,127]; biasFP32",
            "calibration": "64 fixed ordered images, original unquantized FP32 model, Conv input absmax over all elements/images; finite required; labels forbidden",
            "scope": "all 88 grouped Conv including DFL projection; other arithmetic FP32; floating Conv QDQ simulation only"}


def budget_policy():
    return {"debug_images": 8, "calibration_images": 64, "configurations": 4, "raw_arrays_saved": 32,
            "forward_calls": 120, "forward_breakdown": {"reference": 8, "calibration": 64, "disabled_before": 8,
                "quantized": 24, "disabled_after": 8, "original_after": 8}, "timeout_seconds": 1800,
            "attempts": 1, "automatic_retry": False, "Selection_Holdout": False,
            "expansion": "not executed here; technical identity/implementation checks only, never rescue score or sign"}


def postprocessing_policy():
    return {"NMS": {"confidence_gt": .001, "IoU": .7, "max_det": 1000, "max_nms": 30000,
            "max_wh": 7680, "multi_label": False, "class_aware": True, "backend": "CPU_FP32_torchvision_clone", "warning_is_failure": True},
            "restore": "subtract recorded integer padding / recorded ratio; do not clip before category0 filter",
            "target_match": {"confidence_ge": .25, "same_class": True, "IoU_ge": .5,
                "identity": "image_id:zero_based_original_physical_line", "matcher": "PROJECT_IGNORE_AWARE_TARGET_MATCHING"},
            "submission": "descending confidence stable NMS-order ties; debug images lexical", "official_AP": False}


def check_raw(raw):
    require(isinstance(raw, np.ndarray) and raw.shape == (1, 14, 8400) and raw.dtype == np.float32, "Raw shape/dtype differs")
    require(bool(np.isfinite(raw).all() and (raw[:, 2:4] > 0).all() and ((raw[:, 4:] >= 0) & (raw[:, 4:] <= 1)).all()), "Raw finite/geometry domain invalid")


def observed_maximum(previous, tensor):
    value = float(tensor.detach().abs().amax().item())
    require(math.isfinite(value) and math.isfinite(previous) and previous >= 0, "Nonfinite/invalid activation observation")
    return max(previous, value)


class ActivationRanges:
    """Same absmax policy as Gate0, but reject nonfinite samples before max()."""
    def __init__(self, rows):
        self.max_abs, self.observations, self.handles = {}, Counter(), []
        for name, module, _ in rows:
            self.max_abs[name] = 0.
            def hook(_module, inputs, n=name):
                require(bool(inputs), "Missing Conv calibration input")
                self.max_abs[n] = observed_maximum(self.max_abs[n], inputs[0])
                self.observations[n] += 1
            self.handles.append(module.register_forward_pre_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def strict_reference_check(reference, candidate):
    check_raw(reference); check_raw(candidate)
    require(np.array_equal(reference, candidate), "FP32 strict saved-array identity failed; stop without substituting reference")
    return {"strict_elementwise_equal": True, "elements": int(reference.size)}


def target_summary(states):
    require(set(states) == set(CONFIGS), "Missing target configuration")
    keys = set(states["fp32"])
    require(all(set(state) == keys and all(type(hit) is bool for hit in state.values()) for state in states.values()), "Eligible target identities/states differ")
    hits = {key: {target for target, hit in state.items() if hit} for key, state in states.items()}
    F, Q, D, R = (hits[key] for key in CONFIGS)
    A, S = F - Q, F & Q
    rescued, collateral = A & R, S - R
    net = len(rescued) - len(collateral)
    return {"eligible_targets": len(keys), "hits": {key: len(value) for key, value in hits.items()},
        "added_miss_ids_A": sorted(A), "stable_hit_ids_S": sorted(S), "rescued_ids": sorted(rescued),
        "collateral_miss_ids": sorted(collateral), "two_way_same_target_ids": sorted(rescued & (F - D)),
        "added_misses": len(A), "rescued": len(rescued), "collateral_misses": len(collateral), "net_rescue": net,
        "rescue_rate": len(rescued) / len(A) if A else None, "net_rescue_rate": net / len(A) if A else None,
        "stable_retention": 1 - len(collateral) / len(S) if S else None,
        "evidence_status": "DEVELOPMENT_DESCRIPTIVE_ONLY" if A and S else "INSUFFICIENT_ZERO_DENOMINATOR",
        "scientific_acceptance": False, "random_or_traditional_superiority": False, "deployment_INT8": False}


def source_paths():
    return {"runner": Path(__file__), "tests": ROOT / "tests/test_gate1_w8a8_debug_pilot.py",
        "entry": ROOT / "scripts/gate1_w8a8_entry.py", "legacy_pure_operations": ROOT / "src/gate0.py",
        "state_loader": ROOT / "scripts/gate1_fp32_reference_state.py", "capture": ROOT / "scripts/gate1_fp32_live_capture.py",
        "formal_runner": ROOT / "scripts/evaluate_gate1_formal_candidate.py", "NMS_restore_guard": ROOT / "scripts/qualify_gate1_float_baseline.py",
        "matcher": ROOT / "scripts/gate1_ignore_aware_evaluation.py", "formal_protocol": FORMAL / "protocol.json",
        "formal_manifest": FORMAL / "inference/manifest.json", "split_plan": PLAN / "split_plan.json",
        "split_assignment": PLAN / "split_assignment.jsonl", "calibration_list": PLAN / f"splits/calibration_seed_{SEED}.txt",
        "entry_contract": ENTRY / "contract_v2.json", "entry_result": ENTRY / "result.json", "entry_inventory": ENTRY / "module_inventory.json"}


def metadata_cohorts():
    """Read existing inventories only; never hash/decode any data image or label."""
    formal, plan = read(FORMAL / "protocol.json"), read(PLAN / "split_plan.json")
    assigned = {row["image_id"]: row for row in (json.loads(line) for line in (PLAN / "split_assignment.jsonl").read_text(encoding="utf-8").splitlines())}
    images_only = ["image_id", "assigned_split", "source_split", "image_path", "image_sha256", "width", "height", "refined_component"]
    calibration = [{key: assigned[identifier][key] for key in images_only} for identifier in plan["calibration_image_ids"][SEED]]
    expected_paths = [str(Path(formal["prepared_root"]) / row["image_path"]).replace("\\", "/") for row in calibration]
    require((PLAN / f"splits/calibration_seed_{SEED}.txt").read_text(encoding="utf-8").splitlines() == expected_paths, "Existing calibration ordering mismatch")
    manifest = read(FORMAL / "inference/manifest.json")["files"]
    debug = []
    for row in sorted(formal["cohorts"]["debug"], key=lambda item: item["image_id"])[:8]:
        member = {key: row[key] for key in images_only + ["annotation_path", "annotation_sha256"]}
        relative = "debug/raw/" + row["image_id"].split("/")[1] + ".npz"
        member["saved_raw_reference"] = {"path": str(FORMAL / "inference" / relative), **manifest[relative]}
        debug.append(member)
    return formal, debug, calibration


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "contract_draft.json").exists(), "Draft already exists; preserve it")
    sys.path.insert(0, str(ROOT / "scripts"))
    import gate1_w8a8_entry as E
    formal, debug, calibration = metadata_cohorts()
    require(read(ENTRY / "result.json")["status"] == "PASS_SYNTHETIC_UNFUSED_W8A8_ENGINEERING_ENTRY_ONLY", "Stage A not passed")
    p = {"schema": SCHEMA, "status": "DRAFT_NOT_FROZEN_NOT_EXECUTED", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": identity(E.CAPTURE / "state/reference_state.pt"), "architecture": identity(E.CAPTURE / "architecture.json"),
        "state_manifest": identity(E.CAPTURE / "state/manifest.json"), "state_receipt": identity(E.CAPTURE / "state/snapshot_receipt.json"),
        "source_bindings": {key: identity(path) for key, path in source_paths().items()}, "runtime_sources": E.capture_helpers().sources(),
        "prepared_root": formal["prepared_root"], "debug": debug, "calibration": calibration,
        "debug_selection_rule": "lexicographic_first_8_of_existing_109_debug_members_before_any_QDQ_output",
        "member_order_hashes": {"debug": ordered_hash(debug), "calibration": ordered_hash(calibration)},
        "members_bound_from_existing_metadata_only": True, "seed": SEED, "configurations": CONFIGS,
        "fusion": "BN_unfused_all_other_ops_FP32", "quantization": quantization_policy(),
        "input": {"size": 640, "batch": 1, "AMP": False, "device": "cuda:0", "deterministic": True, "TF32": False},
        "preprocessing": formal["preprocessing"], "raw_tolerances": TOLERANCES,
        "reference_policy": "strict_elementwise_FP32_identity_abort_on_any_difference",
        "stage_order": ["8 fresh FP32 arrays strictly equal sealed FP32 arrays", "64 image-only FP32 Conv-input absmax calibration",
            "8 disabled-QDQ arrays strictly equal fresh FP32", "3x8 QDQ branches with raw arrays/NMS/target states",
            "8 disabled and 8 original FP32 replay arrays strictly equal initial reference"],
        "postprocessing": postprocessing_policy(), "budget": budget_policy(),
        "calibration_labels_read": False, "Selection_Holdout_accessed": False, "scientific_acceptance": False,
        "task_AP": False, "old_failures_preserved": ["FAIL_FIXED_TOLERANCES", "FAILED_UNCHANGED"]}
    require(p["candidate"]["sha256"] == E.FINAL_SHA, "Wrong formal candidate")
    validate_contract(p)
    write(OUT / "contract_draft.json", p)
    return {"status": p["status"], "draft": identity(OUT / "contract_draft.json"), "real_images_or_labels_read": 0}


def verify_bindings(p):
    require(p["source_bindings"] == {key: identity(path) for key, path in source_paths().items()}, "Bound source inventory differs")
    for value in list(p["source_bindings"].values()) + [p[key] for key in ("candidate", "architecture", "state_manifest", "state_receipt")]:
        require(identity(value["path"]) == value, "Frozen source/candidate changed: " + value["path"])
    formal, debug, calibration = metadata_cohorts()
    require(p["debug"] == debug and p["calibration"] == calibration and p["prepared_root"] == formal["prepared_root"] and
            p["preprocessing"] == formal["preprocessing"] and p["candidate"] == formal["final_state"], "Pilot members/reference differ from original frozen metadata")


def freeze(reviewed_draft_sha):
    draft = OUT / "contract_draft.json"
    require(sha(draft) == reviewed_draft_sha, "Explicit reviewed draft SHA required")
    p = read(draft); validate_contract(p); verify_bindings(p)
    require(p["status"] == "DRAFT_NOT_FROZEN_NOT_EXECUTED", "Only reviewed draft can freeze")
    tests = read(OUT / "tests_receipt.json")
    require(tests["failures"] == tests["errors"] == 0 and tests["tests"] > 0, "Pure/synthetic tests not passed")
    require(tests["runner"] == identity(__file__) and tests["test_source"] == identity(source_paths()["tests"]), "Test source drift")
    p.update(status="FROZEN_REVIEWED_NOT_EXECUTED", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
             reviewed_draft=identity(draft), tests=identity(OUT / "tests_receipt.json"))
    write(OUT / "contract.json", p)
    return {"status": p["status"], "contract": identity(OUT / "contract.json")}


def checked_member_path(p, row, key):
    root = Path(p["prepared_root"]).resolve(strict=True)
    path = (root / row[key + "_path"]).resolve(strict=True)
    require(path.is_relative_to(root) and sha(path) == row[key + "_sha256"], "Data member identity changed")
    return path


def image_input(p, row, torch, cv2, letterbox):
    validate_member(row, row["assigned_split"])
    path = checked_member_path(p, row, "image")
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    require(image is not None and image.shape == (row["height"], row["width"], 3), "Unexpected image dimensions")
    params = letterbox.get_params({"img": image})
    padded = letterbox.apply_image({"img": image}, params)["img"]
    tensor = torch.from_numpy(np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))).unsqueeze(0).float().div(255).to("cuda:0")
    return tensor, params


def worker():
    attempt = OUT / "attempt_01"
    require((attempt / "launcher_receipt.json").is_file(), "Only the bounded launcher may start a worker")
    p = read(OUT / "contract.json"); validate_contract(p); verify_bindings(p)
    require(p["status"] == "FROZEN_REVIEWED_NOT_EXECUTED", "Contract not frozen")
    sys.path.insert(0, str(ROOT / "scripts"))
    import gate1_w8a8_entry as E
    import evaluate_gate1_formal_candidate as F
    import qualify_gate1_float_baseline as Q
    import gate1_ignore_aware_evaluation as M
    torch, cv2 = F.configure()
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    capture = E.capture_helpers()
    require(capture.sources() == p["runtime_sources"], "Runtime source changed")
    saved, metadata = capture.FORMAT.load_snapshot(E.CAPTURE / "state")
    architecture = read(p["architecture"]["path"])
    require(hashlib.sha256(capture.canonical(architecture).encode()).hexdigest() == metadata["architecture_sha256"], "Architecture metadata changed")
    with torch.random.fork_rng(devices=[]):
        model = DetectionModel(deepcopy(architecture["cfg"]), ch=3, nc=10, verbose=False)
    model.load_state_dict(saved, strict=True); model.names = dict(enumerate(architecture["names"]))
    model.to("cuda:0").float().eval().requires_grad_(False)
    original_state = E.state_info(model)
    modules = E.legacy().list_conv_modules(model)
    inventory = read(ENTRY / "module_inventory.json")
    require([(name, group, list(conv.weight.shape)) for name, conv, group in modules] ==
        [(row["name"], row["group"], row["weight_shape"]) for row in inventory["modules"]], "Actual group/Conv map changed")
    require(sum(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules()) == 81, "BN fusion drift")
    letterbox = LetterBox((640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32,
        padding_value=114, interpolation=cv2.INTER_LINEAR)
    (attempt / "raw").mkdir()
    (attempt / "targets").mkdir()
    (attempt / "predictions").mkdir()
    inputs, references, guards, target_maps, counts, forward_calls = {}, {}, [], {key: {} for key in CONFIGS}, {key: Counter() for key in CONFIGS}, 0

    def progress(stage, index, total):
        write(attempt / "live_status.json", {"stage": stage, "images": index, "total": total, "forward_calls": forward_calls}, once=False)
        print(json.dumps({"stage": stage, "images": index, "total": total, "forward_calls": forward_calls}), flush=True)

    def forward(current, tensor):
        nonlocal forward_calls
        forward_calls += 1
        require(forward_calls <= p["budget"]["forward_calls"], "Forward budget exhausted")
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            raw = current(tensor)[0].detach().cpu().numpy().copy()
        check_raw(raw)
        return raw

    def save_scored(configuration, row, raw, params):
        stem = row["image_id"].split("/")[1]
        np.savez_compressed(attempt / "raw" / f"{configuration.replace(':', '_')}__{stem}.npz", raw=raw)
        detection, anchors = Q.guarded_nms(non_max_suppression, torch.from_numpy(raw))
        predictions, check = Q.restored_predictions(detection[0].numpy(), anchors[0].numpy(), params, row["image_id"], configuration)
        predictions = sorted(predictions, key=lambda item: (-item["score"], item["nms_input_order"]))
        annotation = checked_member_path(p, row, "annotation")
        gt = M.parse_ground_truth(annotation.read_text(encoding="utf-8-sig"), row["image_id"])
        matched = M.match_project_targets(gt, predictions, row["width"], row["height"], confidence=.25, iou_threshold=.5)
        name = f"{configuration.replace(':', '_')}__{stem}.json"
        write(attempt / "predictions" / name, {"image_id": row["image_id"], "configuration": configuration,
            "letterbox_params": params, "anchor_indices_in_NMS_order": anchors[0].numpy().reshape(-1).tolist(),
            "predictions": predictions, "restoration_checks": check, "NMS_max_det_saturated": len(predictions) == 1000})
        write(attempt / "targets" / name, matched)
        target_maps[configuration].update({item["target_id"]: bool(item["hit"]) for item in matched["targets"]})
        counts[configuration].update(matched["counts"])

    # Strict reference gate must finish for all eight images before calibration.
    for index, row in enumerate(p["debug"], 1):
        tensor, params = image_input(p, row, torch, cv2, letterbox)
        raw = forward(model, tensor)
        reference = row["saved_raw_reference"]
        require(identity(reference["path"]) == reference, "Sealed FP32 raw reference changed")
        with np.load(reference["path"], allow_pickle=False) as arrays:
            old_fp32 = arrays["fp32"].copy()
        item = {"image_id": row["image_id"], "original_all_anchor_guard": Q.raw_guard(old_fp32, raw),
            "strict_elementwise_equal": bool(np.array_equal(old_fp32, raw)), "no_NMS_subset": True}
        guards.append(item); write(attempt / "reference_checks.json", guards, once=False)
        if not item["strict_elementwise_equal"]:
            np.savez_compressed(attempt / f"failed_reference_{index}.npz", fresh_FP32=raw, sealed_FP32=old_fp32)
        strict_reference_check(old_fp32, raw)
        inputs[row["image_id"]] = (tensor, params, E.tensor_info(tensor))
        references[row["image_id"]] = raw
        progress("FP32_STRICT_REFERENCE", index, 8)
    require(E.state_info(model) == original_state, "FP32 reference stage polluted state")
    collector = ActivationRanges(modules)
    try:
        for index, row in enumerate(p["calibration"], 1):
            tensor, _ = image_input(p, row, torch, cv2, letterbox)
            forward(model, tensor)
            require(all(math.isfinite(value) and value >= 0 for value in collector.max_abs.values()), "Nonfinite activation range")
            progress("FP32_IMAGE_ONLY_CALIBRATION", index, 64)
    finally:
        collector.close()
    maxima = dict(collector.max_abs)
    require(set(maxima) == {name for name, _, _ in modules}, "Calibration did not cover all Conv inputs")
    require(set(collector.observations) == set(maxima) and all(value == 64 for value in collector.observations.values()), "Incomplete Conv calibration observations")
    require(E.state_info(model) == original_state, "Calibration polluted FP32 state")
    write(attempt / "calibration_ranges.json", {"seed": SEED, "max_abs": maxima, "activation_scale": {key: max(value / 127., 1e-12) for key, value in maxima.items()},
        "images": 64, "observations_per_Conv": dict(collector.observations), "labels_read": False,
        "member_order_hash": p["member_order_hashes"]["calibration"]})
    candidate = deepcopy(model)
    require(all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), candidate.parameters())), "FP32/candidate storage aliases")
    wrappers = E.install(candidate, maxima); candidate.eval()
    wrapped_state = E.state_info(candidate)
    E.legacy().enable_configuration(wrappers, "fp32")
    for index, row in enumerate(p["debug"], 1):
        tensor, params, _ = inputs[row["image_id"]]
        strict_reference_check(references[row["image_id"]], forward(candidate, tensor))
        save_scored("fp32", row, references[row["image_id"]], params)
        progress("DISABLED_QDQ_STRICT_IDENTITY", index, 8)
    for configuration in CONFIGS[1:]:
        E.legacy().enable_configuration(wrappers, configuration)
        require({wrapper.group for wrapper in wrappers if wrapper.enabled} == E.legacy().parse_configuration(configuration), "Configuration mask mismatch")
        for index, row in enumerate(p["debug"], 1):
            tensor, params, _ = inputs[row["image_id"]]
            raw = forward(candidate, tensor)
            save_scored(configuration, row, raw, params)
            progress(configuration, index, 8)
        require(E.state_info(candidate) == wrapped_state and E.state_info(model) == original_state, "QDQ polluted FP32/BN/buffer state")
    E.legacy().enable_configuration(wrappers, "fp32")
    for index, row in enumerate(p["debug"], 1):
        tensor, _, tensor_before = inputs[row["image_id"]]
        strict_reference_check(references[row["image_id"]], forward(candidate, tensor))
        strict_reference_check(references[row["image_id"]], forward(model, tensor))
        require(E.tensor_info(tensor) == tensor_before, "Input was mutated")
        progress("STATE_AND_DISABLED_REPLAY", index, 8)
    require(E.state_info(candidate) == wrapped_state and E.state_info(model) == original_state, "Final state polluted")
    verify_bindings(p); require(capture.sources() == p["runtime_sources"], "Runtime changed during pilot")
    require(forward_calls == 120, "Unexpected actual forward budget")
    write(attempt / "target_matrix.json", {"configurations": CONFIGS, "states": target_maps})
    summary = target_summary(target_maps)
    write(attempt / "target_summary.json", summary | {"branch_counts": {key: dict(value) for key, value in counts.items()}})
    receipt = {"status": "COMPLETED_BOUNDED_DEBUG_W8A8_SIMULATION_ONLY", "contract": identity(OUT / "contract.json"),
        "raw_arrays": 32, "forward_calls": forward_calls, "FP32_saved_reference_strict": True,
        "disabled_QDQ_before_after_strict": True, "original_and_wrapper_states_unchanged": True,
        "calibration_images": 64, "debug_images": 8, "calibration_labels_read": False,
        "Selection_Holdout_accessed": False, "official_AP": False, "scientific_acceptance": False,
        "expansion_decision": "TECHNICAL_CHECKS_PASS_ONLY_NO_RESCUE_OUTCOME_SELECTION; expansion requires a new frozen contract"}
    write(attempt / "completion_receipt.json", receipt)
    write(attempt / "manifest.json", {"files": [identity(path) for path in sorted(attempt.rglob("*")) if path.is_file() and path.name not in
        {"launcher_receipt.json", "worker_console.txt", "live_status.json"}]})
    print(json.dumps(receipt), flush=True)


def run(contract_sha):
    path = OUT / "contract.json"
    require(path.is_file() and sha(path) == contract_sha, "Explicit frozen contract SHA required")
    p = read(path); validate_contract(p); verify_bindings(p)
    require(p["status"] == "FROZEN_REVIEWED_NOT_EXECUTED", "Reviewed frozen contract missing")
    attempt = OUT / "attempt_01"; attempt.mkdir(exist_ok=False)
    receipt = {"status": "RUNNING_BOUNDED_WORKER", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": identity(path), "timeout_seconds": p["budget"]["timeout_seconds"], "launcher_pid": os.getpid()}
    write(attempt / "launcher_receipt.json", receipt)
    try:
        with (attempt / "worker_console.txt").open("x", encoding="utf-8") as console:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker"], cwd=ROOT,
                stdout=console, stderr=subprocess.STDOUT, timeout=p["budget"]["timeout_seconds"],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        receipt.update(status="WORKER_EXITED", returncode=completed.returncode,
            completion_receipt_present=(attempt / "completion_receipt.json").is_file())
        require(completed.returncode == 0 and receipt["completion_receipt_present"], "Worker failed; preserve attempt and do not restart")
        receipt["status"] = "COMPLETED_BOUNDED_WORKER"
    except subprocess.TimeoutExpired:
        receipt.update(status="TIMEOUT_PRESERVED_NO_RETRY", timeout=True)
        write(attempt / "timeout_receipt.json", receipt)
        raise
    except Exception as error:
        receipt.update(status="FAILED_PRESERVED_NO_RETRY", error=repr(error))
        raise
    finally:
        receipt["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(attempt / "launcher_receipt.json", receipt, once=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "freeze", "run", "worker"))
    parser.add_argument("--reviewed-draft-sha")
    parser.add_argument("--contract-sha")
    args = parser.parse_args()
    os.environ.update(YOLO_CONFIG_DIR=str(OUT / "settings"), YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false",
        PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if args.mode == "prepare": result = prepare()
    elif args.mode == "freeze": result = freeze(args.reviewed_draft_sha)
    elif args.mode == "run": result = run(args.contract_sha)
    else:
        try:
            result = worker()
        except Exception as error:
            attempt = OUT / "attempt_01"
            if attempt.is_dir() and not (attempt / "failure_receipt.json").exists():
                write(attempt / "failure_receipt.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "error": repr(error)})
            raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
