"""Prepare the outcome-independent 109-debug / three-seed W8A8 extension.

prepare/test read no real images or labels. freeze binds a reviewed draft and
the pilot's completed technical checks plus independent saved-output PASS.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gate1_w8a8_debug_pilot as P

OUT = ROOT / "artifacts/gate1/20261024_w8a8_debug_3seed_v1"
SEEDS = ["2026102201", "2026102202", "2026102203"]
CONFIGS = P.CONFIGS[:]
SCHEMA = "rampq.gate1.w8a8_debug_109_three_seed/v1"
AUDIT_PASS = "PASS_INDEPENDENT_SAVED_W8A8_DEBUG_ALL_ANCHOR_NMS_TARGET_REPLAY_ONLY"


def budget():
    # 109 reference forwards once; per seed 64+109+3*109+109+109.
    return {"debug_images": 109, "calibration_seeds": SEEDS, "calibration_images_per_seed": 64,
        "configuration_count": 4, "logical_raw_arrays": 1308, "unique_raw_array_files": 1090,
        "FP32_reference_storage": "109 shared arrays plus per-seed 109 explicit identity references",
        "forward_calls": 2263, "forward_breakdown": {"shared_FP32_reference": 109,
            "per_seed": {"calibration": 64, "disabled_before": 109, "quantized": 327,
                "disabled_after": 109, "original_after": 109}, "per_seed_total": 718, "seeds": 3},
        "timeout_seconds": 3600, "attempts": 1, "automatic_retry": False,
        "expansion_reason": "pilot technical identity, implementation, state isolation and independent saved-output PASS only; no rescue outcomes",
        "further_expansion": "no automatic additional datasets, methods, candidates or seeds"}


def output_schema():
    return {"attempt": "attempt_01", "shared_FP32": "shared_FP32/raw/<stem>.npz:raw",
        "shared_reference_checks": "shared_FP32/reference_checks.json", "seed_directory": "<seed>",
        "seed_QDQ_arrays": "<seed>/raw/<config_with_colon_replaced_by_underscore>__<stem>.npz:raw",
        "seed_FP32_references": "<seed>/FP32_shared_references.json",
        "differences": "<seed>/differences/<config>__<stem>.npz:difference_FP64",
        "difference_checks": "<seed>/raw_difference_checks.json", "predictions": "<seed>/predictions/<config>__<stem>.json",
        "targets": "<seed>/targets/<config>__<stem>.json", "target_matrix": "<seed>/target_matrix.json",
        "target_summary": "<seed>/target_summary.json", "calibration_ranges": "<seed>/calibration_ranges.json",
        "state_evidence": "<seed>/state_fingerprint_evidence.json", "replay_evidence": "<seed>/strict_replay_evidence.json",
        "progress_file": "live_status.json", "progress_console": "worker_console.txt",
        "progress_fields": ["stage", "seed", "images", "total", "forward_calls"],
        "progress_stages": ["SHARED_FP32_STRICT_REFERENCE", "IMAGE_ONLY_CALIBRATION", "DISABLED_BEFORE_STRICT",
            "q_all", "q_only:G1", "r_only:G1", "DISABLED_AND_ORIGINAL_AFTER_STRICT"],
        "manifest": "manifest.json", "completion": "completion_receipt.json",
        "state_evidence_limit": "tensor hash descriptions from live runner; not saved independent state tensors",
        "raw_file_semantics": "float32[1,14,8400] decoded boxes_xywh+10scores; differences float64 candidate-reference all117600elements"}


def prerequisites(completion, launcher, independent):
    P.require(completion["status"] == "COMPLETED_BOUNDED_DEBUG_W8A8_SIMULATION_ONLY", "Pilot incomplete")
    P.require(all(completion[key] is True for key in ("FP32_saved_reference_strict", "disabled_QDQ_before_after_strict",
        "original_and_wrapper_states_unchanged")), "Pilot technical gates not passed")
    P.require(completion["forward_calls"] == 120 and completion["raw_arrays"] == 32, "Pilot technical budget differs")
    P.require(launcher["status"] == "COMPLETED_BOUNDED_WORKER" and launcher["returncode"] == 0,
        "Pilot process not completed cleanly")
    P.require(independent["status"] == AUDIT_PASS and independent["contract"] == completion["contract"] == launcher["contract"],
        "Pilot independent saved-output PASS/contract missing")
    return {"pilot_FP32_strict": True, "disabled_and_state_isolation": True, "independent_saved_output_PASS": True,
        "rescue_results_used_to_select_extension": False}


def source_paths():
    paths = dict(P.source_paths())
    paths.update({"runner": Path(__file__), "tests": ROOT / "tests/test_gate1_w8a8_debug_full.py",
        "pilot_runner": ROOT / "scripts/run_gate1_w8a8_debug_pilot.py",
        "pilot_tests": ROOT / "tests/test_gate1_w8a8_debug_pilot.py",
        "pilot_frozen_contract": P.OUT / "contract.json",
        "pilot_completion": P.OUT / "attempt_01/completion_receipt.json",
        "pilot_launcher": P.OUT / "attempt_01/launcher_receipt.json",
        "pilot_independent_audit": P.OUT / "independent_saved_audit_result.json"})
    paths.pop("calibration_list")
    for seed in SEEDS:
        paths["calibration_list_" + seed] = P.PLAN / f"splits/calibration_seed_{seed}.txt"
    return paths


def cohort_metadata():
    formal, plan = P.read(P.FORMAL / "protocol.json"), P.read(P.PLAN / "split_plan.json")
    assignments = {row["image_id"]: row for row in (json.loads(line) for line in
        (P.PLAN / "split_assignment.jsonl").read_text(encoding="utf-8").splitlines())}
    image_keys = ["image_id", "assigned_split", "source_split", "image_path", "image_sha256", "width", "height", "refined_component"]
    calibration = {}
    for seed in SEEDS:
        calibration[seed] = [{key: assignments[identifier][key] for key in image_keys} for identifier in plan["calibration_image_ids"][seed]]
        listed = (P.PLAN / f"splits/calibration_seed_{seed}.txt").read_text(encoding="utf-8").splitlines()
        expected = [str(Path(formal["prepared_root"]) / row["image_path"]).replace("\\", "/") for row in calibration[seed]]
        P.require(listed == expected, "Frozen calibration member ordering differs")
    manifest = P.read(P.FORMAL / "inference/manifest.json")["files"]
    debug = []
    for row in sorted(formal["cohorts"]["debug"], key=lambda item: item["image_id"]):
        item = {key: row[key] for key in image_keys + ["annotation_path", "annotation_sha256"]}
        relative = "debug/raw/" + row["image_id"].split("/")[1] + ".npz"
        item["saved_raw_reference"] = {"path": str(P.FORMAL / "inference" / relative), **manifest[relative]}
        debug.append(item)
    return formal, debug, calibration


def validate(p):
    P.require(p["schema"] == SCHEMA and p["seeds"] == SEEDS and p["configurations"] == CONFIGS, "Extension matrix differs")
    P.require(p["budget"] == budget() and p["output_schema"] == output_schema() and p["fusion"] == "BN_unfused_all_other_ops_FP32", "Budget/layout/fusion differs")
    P.require(p["quantization"] == P.quantization_policy() and p["postprocessing"] == P.postprocessing_policy(), "Pilot semantic drift")
    P.require(p["raw_tolerances"] == P.TOLERANCES and p["reference_policy"] == "strict_elementwise_FP32_identity_abort_on_any_difference", "Reference gate weakened")
    P.require(p["input"] == {"size": 640, "batch": 1, "AMP": False, "device": "cuda:0", "deterministic": True, "TF32": False}, "Input precision differs")
    debug = p["debug"]
    P.require(len(debug) == len({row["image_id"] for row in debug}) == 109 and
        [row["image_id"] for row in debug] == sorted(row["image_id"] for row in debug), "Full debug membership/order differs")
    for row in debug:
        P.validate_member(row, "debug")
    P.require(set(p["calibration"]) == set(SEEDS), "Unexpected calibration seed")
    combined = []
    for seed in SEEDS:
        rows = p["calibration"][seed]
        P.require(len(rows) == len({row["image_id"] for row in rows}) == 64, "Wrong per-seed calibration size")
        for row in rows:
            P.validate_member(row, "calibration_pool")
        P.require({row["image_id"] for row in rows}.isdisjoint(row["image_id"] for row in debug), "Calibration/debug overlap")
        combined.extend(rows)
    P.require(len({row["image_id"] for row in combined}) == len({row["image_sha256"] for row in combined}) == 192,
        "Calibration seeds overlap by ID or image SHA")
    P.require(p["member_order_hashes"] == {"debug": P.ordered_hash(debug),
        "calibration": {seed: P.ordered_hash(p["calibration"][seed]) for seed in SEEDS}}, "Cohort order hash differs")
    P.require(p["outcome_independent_extension"] is True and p["scientific_acceptance"] is False,
        "Extension scope changed")


def verify_sources(p):
    P.require(p["source_bindings"] == {key: P.identity(path) for key, path in source_paths().items()}, "Bound source inventory changed")
    for item in [p[key] for key in ("candidate", "architecture", "state_manifest", "state_receipt")]:
        P.require(P.identity(item["path"]) == item, "Candidate source changed")
    formal, debug, calibration = cohort_metadata()
    P.require(p["debug"] == debug and p["calibration"] == calibration and p["prepared_root"] == formal["prepared_root"] and
        p["candidate"] == formal["final_state"] and p["preprocessing"] == formal["preprocessing"], "Metadata-derived cohort/reference drift")
    P.require(p["technical_prerequisites"] == prerequisites(P.read(source_paths()["pilot_completion"]),
        P.read(source_paths()["pilot_launcher"]), P.read(source_paths()["pilot_independent_audit"])), "Technical prerequisite drift")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    import gate1_w8a8_entry as E
    formal, debug, calibration = cohort_metadata()
    technical = prerequisites(P.read(source_paths()["pilot_completion"]), P.read(source_paths()["pilot_launcher"]),
        P.read(source_paths()["pilot_independent_audit"]))
    p = {"schema": SCHEMA, "status": "DRAFT_NOT_FROZEN_NOT_EXECUTED", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": P.identity(E.CAPTURE / "state/reference_state.pt"), "architecture": P.identity(E.CAPTURE / "architecture.json"),
        "state_manifest": P.identity(E.CAPTURE / "state/manifest.json"), "state_receipt": P.identity(E.CAPTURE / "state/snapshot_receipt.json"),
        "source_bindings": {key: P.identity(path) for key, path in source_paths().items()}, "runtime_sources": E.capture_helpers().sources(),
        "technical_prerequisites": technical, "outcome_independent_extension": True, "seeds": SEEDS,
        "configurations": CONFIGS, "prepared_root": formal["prepared_root"], "debug": debug, "calibration": calibration,
        "member_order_hashes": {"debug": P.ordered_hash(debug), "calibration": {seed: P.ordered_hash(rows) for seed, rows in calibration.items()}},
        "debug_selection": "all existing109 lexical; no outcomes used; no extra members",
        "quantization": P.quantization_policy(), "fusion": "BN_unfused_all_other_ops_FP32", "preprocessing": formal["preprocessing"],
        "input": {"size": 640, "batch": 1, "AMP": False, "device": "cuda:0", "deterministic": True, "TF32": False},
        "postprocessing": P.postprocessing_policy(), "raw_tolerances": P.TOLERANCES,
        "reference_policy": "strict_elementwise_FP32_identity_abort_on_any_difference", "budget": budget(), "output_schema": output_schema(),
        "output_policy": "109 shared FP32 arrays plus each seed109 FP32 references and327 QDQ arrays; NMS/target states for all4x109x3; full elementwise differences saved",
        "calibration_labels_read": False, "Selection_Holdout_accessed": False, "task_AP": False, "scientific_acceptance": False,
        "deployment_INT8": False, "random_or_traditional_superiority": False,
        "old_failures_preserved": ["FAIL_FIXED_TOLERANCES", "FAILED_UNCHANGED"]}
    P.require(p["candidate"]["sha256"] == E.FINAL_SHA, "Wrong candidate")
    validate(p); verify_sources(p)
    P.write(OUT / "contract_draft.json", p)
    return {"status": p["status"], "draft": P.identity(OUT / "contract_draft.json"), "real_images_or_labels_read": 0}


def freeze(reviewed_sha):
    P.require(P.sha(OUT / "contract_draft.json") == reviewed_sha, "Explicit reviewed draft SHA required")
    p = P.read(OUT / "contract_draft.json"); validate(p); verify_sources(p)
    tests = P.read(OUT / "tests_receipt.json")
    P.require(tests["failures"] == tests["errors"] == 0 and tests["tests"] > 0 and
        tests["runner"] == P.identity(__file__) and tests["test_source"] == P.identity(source_paths()["tests"]), "Synthetic tests/source drift")
    P.require(p["status"] == "DRAFT_NOT_FROZEN_NOT_EXECUTED", "Only reviewed draft freezes")
    p.update(status="FROZEN_REVIEWED_NOT_EXECUTED", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        reviewed_draft=P.identity(OUT / "contract_draft.json"), tests=P.identity(OUT / "tests_receipt.json"))
    P.write(OUT / "contract.json", p)
    return {"status": p["status"], "contract": P.identity(OUT / "contract.json")}


def raw_difference(reference, candidate):
    P.check_raw(reference); P.check_raw(candidate)
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return difference, {name: {"elements": int(difference[:, section].size),
        "changed_elements": int(np.count_nonzero(difference[:, section])),
        "max_absolute_error": float(np.max(np.abs(difference[:, section])))}
        for name, section in (("boxes", slice(0, 4)), ("scores", slice(4, 14)))}


def worker():
    attempt = OUT / "attempt_01"
    P.require((attempt / "launcher_receipt.json").is_file(), "Bounded launcher required")
    p = P.read(OUT / "contract.json"); validate(p); verify_sources(p)
    P.require(p["status"] == "FROZEN_REVIEWED_NOT_EXECUTED", "Contract not frozen")
    import gate1_w8a8_entry as E
    import evaluate_gate1_formal_candidate as F
    import qualify_gate1_float_baseline as Q
    import gate1_ignore_aware_evaluation as M
    torch, cv2 = F.configure()
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    capture = E.capture_helpers()
    P.require(capture.sources() == p["runtime_sources"], "Runtime changed")
    saved, metadata = capture.FORMAT.load_snapshot(E.CAPTURE / "state")
    architecture = P.read(p["architecture"]["path"])
    P.require(hashlib.sha256(capture.canonical(architecture).encode()).hexdigest() == metadata["architecture_sha256"], "Architecture metadata drift")
    with torch.random.fork_rng(devices=[]):
        model = DetectionModel(deepcopy(architecture["cfg"]), ch=3, nc=10, verbose=False)
    model.load_state_dict(saved, strict=True); model.names = dict(enumerate(architecture["names"]))
    model.to("cuda:0").float().eval().requires_grad_(False)
    original_state = E.state_info(model)
    modules = E.legacy().list_conv_modules(model)
    inventory = P.read(P.ENTRY / "module_inventory.json")
    P.require([(name, group, list(conv.weight.shape)) for name, conv, group in modules] ==
        [(row["name"], row["group"], row["weight_shape"]) for row in inventory["modules"]], "Actual Conv/group map drift")
    P.require(sum(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()) == 81, "BN fusion drift")
    letterbox = LetterBox((640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32,
        padding_value=114, interpolation=cv2.INTER_LINEAR)
    shared = attempt / "shared_FP32"; shared.mkdir(); (shared / "raw").mkdir()
    reference_checks, inputs, references, forward_calls = [], {}, {}, 0

    def progress(stage, index, total, seed=None):
        P.write(attempt / "live_status.json", {"stage": stage, "seed": seed, "images": index, "total": total,
            "forward_calls": forward_calls}, once=False)
        print(json.dumps({"stage": stage, "seed": seed, "images": index, "total": total, "forward_calls": forward_calls}), flush=True)

    def forward(current, tensor):
        nonlocal forward_calls
        forward_calls += 1
        P.require(forward_calls <= p["budget"]["forward_calls"], "Forward budget exhausted")
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            raw = current(tensor)[0].detach().cpu().numpy().copy()
        P.check_raw(raw)
        return raw

    for index, row in enumerate(p["debug"], 1):
        tensor, params = P.image_input(p, row, torch, cv2, letterbox)
        raw = forward(model, tensor); old = row["saved_raw_reference"]
        P.require(P.identity(old["path"]) == old, "Sealed FP32 reference changed")
        with np.load(old["path"], allow_pickle=False) as data:
            old_raw = data["fp32"].copy()
        reference_checks.append({"image_id": row["image_id"], "strict_equal": bool(np.array_equal(old_raw, raw)),
            "original_all_anchor_guard": Q.raw_guard(old_raw, raw)})
        P.write(shared / "reference_checks.json", reference_checks, once=False)
        if not reference_checks[-1]["strict_equal"]:
            np.savez_compressed(shared / f"failed_{index}.npz", fresh_FP32=raw, sealed_FP32=old_raw)
        P.strict_reference_check(old_raw, raw)
        stem = row["image_id"].split("/")[1]
        np.savez_compressed(shared / "raw" / f"{stem}.npz", raw=raw)
        inputs[row["image_id"]] = (tensor, params, E.tensor_info(tensor))
        references[row["image_id"]] = raw
        progress("SHARED_FP32_STRICT_REFERENCE", index, 109)
    P.require(E.state_info(model) == original_state, "Reference state pollution")
    seed_receipts = []
    for seed in SEEDS:
        folder = attempt / seed; folder.mkdir()
        for name in ("raw", "differences", "predictions", "targets"):
            (folder / name).mkdir()
        collector = P.ActivationRanges(modules)
        try:
            for index, row in enumerate(p["calibration"][seed], 1):
                tensor, _ = P.image_input(p, row, torch, cv2, letterbox)
                forward(model, tensor)
                progress("IMAGE_ONLY_CALIBRATION", index, 64, seed)
        finally:
            collector.close()
        maxima = dict(collector.max_abs)
        P.require(set(maxima) == {name for name, _, _ in modules} and set(collector.observations) == set(maxima) and
            all(count == 64 for count in collector.observations.values()), "Incomplete calibration coverage")
        P.require(E.state_info(model) == original_state, "Calibration state pollution")
        P.write(folder / "calibration_ranges.json", {"seed": seed, "max_abs": maxima,
            "activation_scale": {name: max(value / 127., 1e-12) for name, value in maxima.items()},
            "observations_per_Conv": dict(collector.observations), "labels_read": False,
            "member_order_hash": p["member_order_hashes"]["calibration"][seed]})
        candidate = deepcopy(model)
        P.require(all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), candidate.parameters())), "Parameter storage aliases")
        wrappers = E.install(candidate, maxima); candidate.eval(); wrapped_state = E.state_info(candidate)
        targets, counts, diffs, shared_refs, replay = {key: {} for key in CONFIGS}, {key: Counter() for key in CONFIGS}, [], [], []

        def save_outputs(configuration, row, raw, params):
            stem = row["image_id"].split("/")[1]; name = f"{configuration.replace(':', '_')}__{stem}"
            if configuration == "fp32":
                shared_refs.append({"image_id": row["image_id"], "raw": P.identity(shared / "raw" / f"{stem}.npz"),
                    "strict_disabled_recompute_passed": True})
            else:
                np.savez_compressed(folder / "raw" / f"{name}.npz", raw=raw)
            difference, summary = raw_difference(references[row["image_id"]], raw)
            np.savez_compressed(folder / "differences" / f"{name}.npz", difference_FP64=difference)
            diffs.append({"image_id": row["image_id"], "configuration": configuration, "full_difference": f"differences/{name}.npz",
                "summary": summary, "original_all_anchor_guard_descriptive": Q.raw_guard(references[row["image_id"]], raw)})
            det, anchors = Q.guarded_nms(non_max_suppression, torch.from_numpy(raw))
            restored, checks = Q.restored_predictions(det[0].numpy(), anchors[0].numpy(), params, row["image_id"], configuration)
            restored = sorted(restored, key=lambda item: (-item["score"], item["nms_input_order"]))
            annotation = P.checked_member_path(p, row, "annotation")
            gt = M.parse_ground_truth(annotation.read_text(encoding="utf-8-sig"), row["image_id"])
            matched = M.match_project_targets(gt, restored, row["width"], row["height"], confidence=.25, iou_threshold=.5)
            P.write(folder / "predictions" / f"{name}.json", {"image_id": row["image_id"], "configuration": configuration,
                "letterbox_params": params, "anchor_indices_in_NMS_order": anchors[0].numpy().reshape(-1).tolist(),
                "predictions": restored, "restoration_checks": checks, "NMS_max_det_saturated": len(restored) == 1000})
            P.write(folder / "targets" / f"{name}.json", matched)
            targets[configuration].update({item["target_id"]: bool(item["hit"]) for item in matched["targets"]})
            counts[configuration].update(matched["counts"])

        E.legacy().enable_configuration(wrappers, "fp32")
        for index, row in enumerate(p["debug"], 1):
            tensor, params, _ = inputs[row["image_id"]]
            raw = forward(candidate, tensor); P.strict_reference_check(references[row["image_id"]], raw)
            replay.append({"image_id": row["image_id"], "stage": "disabled_before",
                "reference_sha256": hashlib.sha256(references[row["image_id"]].tobytes()).hexdigest(),
                "replay_sha256": hashlib.sha256(raw.tobytes()).hexdigest(), "strict_equal": True})
            save_outputs("fp32", row, references[row["image_id"]], params)
            progress("DISABLED_BEFORE_STRICT", index, 109, seed)
        P.write(folder / "FP32_shared_references.json", shared_refs)
        for configuration in CONFIGS[1:]:
            E.legacy().enable_configuration(wrappers, configuration)
            P.require({wrapper.group for wrapper in wrappers if wrapper.enabled} == E.legacy().parse_configuration(configuration), "QDQ mask differs")
            for index, row in enumerate(p["debug"], 1):
                tensor, params, _ = inputs[row["image_id"]]
                save_outputs(configuration, row, forward(candidate, tensor), params)
                progress(configuration, index, 109, seed)
            P.require(E.state_info(model) == original_state and E.state_info(candidate) == wrapped_state, "QDQ state pollution")
        E.legacy().enable_configuration(wrappers, "fp32")
        for index, row in enumerate(p["debug"], 1):
            tensor, _, original_input = inputs[row["image_id"]]
            disabled_raw, original_raw = forward(candidate, tensor), forward(model, tensor)
            P.strict_reference_check(references[row["image_id"]], disabled_raw)
            P.strict_reference_check(references[row["image_id"]], original_raw)
            for stage, observed in (("disabled_after", disabled_raw), ("original_after", original_raw)):
                replay.append({"image_id": row["image_id"], "stage": stage,
                    "reference_sha256": hashlib.sha256(references[row["image_id"]].tobytes()).hexdigest(),
                    "replay_sha256": hashlib.sha256(observed.tobytes()).hexdigest(), "strict_equal": True})
            P.require(E.tensor_info(tensor) == original_input, "Input mutation")
            progress("DISABLED_AND_ORIGINAL_AFTER_STRICT", index, 109, seed)
        P.require(E.state_info(model) == original_state and E.state_info(candidate) == wrapped_state, "Final per-seed state pollution")
        P.write(folder / "strict_replay_evidence.json", {"checks": replay, "all_exact": True,
            "scope": "live recomputation hashes; these replay tensors are not separately saved"})
        P.write(folder / "state_fingerprint_evidence.json", {"original_before": original_state, "original_after": E.state_info(model),
            "wrapped_before": wrapped_state, "wrapped_after": E.state_info(candidate),
            "parameter_storage_disjoint": True, "source_tensor_check_scope": "live runner tensor descriptions, not separately saved state tensors"})
        P.write(folder / "raw_difference_checks.json", diffs)
        P.write(folder / "target_matrix.json", {"configurations": CONFIGS, "states": targets})
        P.write(folder / "target_summary.json", P.target_summary(targets) | {"seed": seed,
            "branch_counts": {key: dict(value) for key, value in counts.items()}})
        seed_receipt = {"seed": seed, "status": "COMPLETED_FIXED_DEBUG_SIMULATION_SEED_ONLY", "forwards": 718,
            "calibration_images": 64, "debug_images": 109, "raw_arrays_QDQ": 327, "FP32_shared_references": 109,
            "full_difference_arrays": 436, "disabled_before_after_exact": True, "original_replay_exact": True,
            "states_and_inputs_unchanged": True, "scientific_acceptance": False}
        P.write(folder / "completion_receipt.json", seed_receipt); seed_receipts.append(seed_receipt)
        del candidate, wrappers
    verify_sources(p); P.require(capture.sources() == p["runtime_sources"], "Runtime source drift")
    P.require(forward_calls == 2263 and E.state_info(model) == original_state, "Final budget/state failure")
    receipt = {"status": "COMPLETED_FIXED_109_DEBUG_THREE_SEED_W8A8_SIMULATION_ONLY", "contract": P.identity(OUT / "contract.json"),
        "forward_calls": forward_calls, "unique_raw_array_files": 1090, "logical_raw_arrays": 1308,
        "full_difference_arrays": 1308, "seed_receipts": seed_receipts, "strict_shared_FP32_saved_reference": True,
        "original_FP32_and_quantized_states_unpolluted": True, "calibration_labels_read": False,
        "Selection_Holdout_accessed": False, "task_AP": False, "scientific_acceptance": False,
        "random_or_traditional_superiority": False, "deployment_INT8": False, "outcome_independent_extension": True}
    P.write(attempt / "completion_receipt.json", receipt)
    P.write(attempt / "manifest.json", {"files": [P.identity(path) for path in sorted(attempt.rglob("*")) if path.is_file() and path.name not in
        {"launcher_receipt.json", "worker_console.txt", "live_status.json"}]})
    print(json.dumps(receipt), flush=True)


def run(contract_sha):
    P.require(P.sha(OUT / "contract.json") == contract_sha, "Explicit reviewed frozen contract SHA required")
    p = P.read(OUT / "contract.json"); validate(p); verify_sources(p)
    P.require(p["status"] == "FROZEN_REVIEWED_NOT_EXECUTED", "Frozen contract missing")
    attempt = OUT / "attempt_01"; attempt.mkdir(exist_ok=False)
    receipt = {"status": "RUNNING_BOUNDED_WORKER", "contract": P.identity(OUT / "contract.json"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": 3600, "launcher_pid": os.getpid()}
    P.write(attempt / "launcher_receipt.json", receipt)
    try:
        with (attempt / "worker_console.txt").open("x", encoding="utf-8") as console:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker"], cwd=ROOT,
                stdout=console, stderr=subprocess.STDOUT, timeout=3600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        receipt.update(returncode=result.returncode, completion_receipt_present=(attempt / "completion_receipt.json").is_file())
        P.require(result.returncode == 0 and receipt["completion_receipt_present"], "Worker failed; preserve attempt and no retry")
        receipt["status"] = "COMPLETED_BOUNDED_WORKER"
    except subprocess.TimeoutExpired:
        receipt.update(status="TIMEOUT_PRESERVED_NO_RETRY", timeout=True)
        P.write(attempt / "timeout_receipt.json", receipt)
        raise
    except Exception as error:
        receipt.update(status="FAILED_PRESERVED_NO_RETRY", error=repr(error))
        raise
    finally:
        receipt["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        P.write(attempt / "launcher_receipt.json", receipt, once=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "freeze", "run", "worker"))
    parser.add_argument("--reviewed-draft-sha"); parser.add_argument("--contract-sha")
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
                P.write(attempt / "failure_receipt.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "error": repr(error)})
            raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
