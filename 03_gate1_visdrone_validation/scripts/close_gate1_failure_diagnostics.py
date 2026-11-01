"""Independent saved-output arithmetic/provenance checks and append-only closeout."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from diagnose_gate1_full_comparison_failure import ROOT, OLD, OUT, ARMS, read, write, identity, now
import gate1_ignore_aware_evaluation as E

PROBE = ROOT / "artifacts/gate1/20261027_train_gradient_probe_v1"
REPORT = ROOT / "docs/87_gate1_failure_mechanism_review_and_guarded_correction_plan.md"


def main():
    assert not (OUT / "analysis_validation.json").exists(), "Do not overwrite a completed closeout"
    c = read(OLD / "contract.json")
    diagnosis = read(OUT / "result.json")
    probe = read(PROBE / "result.json")
    assert probe["forward_images"] == 64 and probe["weight_updates"] == 0
    assert probe["exact_original_loss_parity_batches"] == 16 and probe["checkpoint_states_unchanged"]
    source_rows = read(OUT / "diagnostic_contract.json")["sources"] + read(PROBE / "contract.json")["sources"]
    assert all(identity(row["path"]) == row for row in source_rows)
    assert identity(E.__file__) == c["source_bindings"]["gate1_ignore_aware_evaluation.py"]
    # Direct counts at the reported cutoffs; independent of grouped-curve code.
    points = 0
    for arm, result in diagnosis["rank_diagnostic"].items():
        with np.load(OLD / "attempt_01/evaluation" / arm / "official_IoU05_rank_curve.npz", allow_pickle=False) as a:
            for key in ("at_baseline_FP_budget", "at_baseline_TP_target"):
                p = result[key]
                if p.get("achievable", True):
                    selected = a["score"] >= p["threshold"]
                    assert int(np.count_nonzero(selected & (a["flag"] == 1))) == p["TP"]
                    assert int(np.count_nonzero(selected & (a["flag"] == 0))) == p["FP"]
                    points += 1
    # Recheck every lost target via the original scalar box helper, not the
    # new vectorized geometry implementation; no model, NMS or AP rerun.
    rows_by_arm = {a: {r["image_id"]: r for r in read(OLD / "attempt_01/evaluation" / a / "saved_predictions_and_targets.json")} for a in ARMS[1:]}
    meta_by_id = {m["image_id"]: m for m in c["evaluation_members"]}
    root = Path(read(c["evaluation_protocol"]["path"])["prepared_root"])
    gt_cache, pred_cache = {}, {}
    checked = Counter()
    for detail in read(OUT / "lost_target_diagnostics.json"):
        image, arm = detail["image_id"], detail["arm"]
        meta = meta_by_id[image]
        if image not in gt_cache:
            records = E.parse_ground_truth((root / meta["annotation_path"]).read_text(encoding="utf-8-sig"), image)
            gt_cache[image] = {r["target_id"]: r for r in records}
        if (arm, image) not in pred_cache:
            row = rows_by_arm[arm][image]
            states = {r["detection_id"]: r["reason"] for r in row["target_matching"]["detection_states"]}
            pred_cache[arm, image] = [p for p in row["predictions"] if states[p["detection_id"]] not in
                ("category_0_raster_coverage_ge_0.5", "outside_evaluated_categories", "empty_after_clipping")]
        gt = gt_cache[image][detail["target_id"]]
        box = E._xyxy(gt["bbox_xywh"], meta["width"], meta["height"])
        available = [p["score"] for p in pred_cache[arm, image] if p["category"] == gt["category"] and
                     E.overlap(E._xyxy(p["bbox_xywh"], meta["width"], meta["height"]), box) >= .5]
        best = max(available, default=None)
        assert best == detail["best_candidate_score"] and len(available) == detail["candidate_count_IoU05"]
        label = ("no_same_class_IoU05_candidate_in_saved_postNMS_pool" if best is None else
                 "candidate_available_only_below_fixed_threshold" if best < .25 else
                 "candidate_above_fixed_threshold_but_unmatched")
        assert label == detail["diagnostic"]
        checked[arm] += 1
    assert dict(checked) == {ARMS[1]: 861, ARMS[2]: 1148}
    clipping = {}
    for arm in ARMS[:2]:
        trace = [json.loads(line) for line in (OLD / "attempt_01/training" / arm / "batch_trace.jsonl").read_text("utf-8").splitlines()]
        norms = np.array([r["grad_norm_before_clip"] for r in trace if r["update"]], np.float64)
        assert len(norms) == 2840
        clipping[arm] = {"optimizer_steps": len(norms), "exceed_clip10": int(np.count_nonzero(norms > 10)),
                         "fraction": float(np.mean(norms > 10)), "mean_clip_factor": float(np.minimum(1, 10 / (norms + 1e-6)).mean())}
    assert clipping[ARMS[0]]["exceed_clip10"] == 45 and clipping[ARMS[1]]["exceed_clip10"] == 2598
    gram_checks = 0
    for arm in ARMS[:2]:
        content = read(PROBE / f"{arm}_observations.json")
        assert content["checkpoint_state_unchanged"]
        for r in content["observations"]:
            gram = np.array(r["gram_float64"], np.float64)
            assert np.isfinite(gram).all() and np.allclose(gram, gram.T, rtol=1e-12, atol=1e-10)
            assert np.linalg.eigvalsh(gram).min() >= -1e-8 * max(1., np.linalg.eigvalsh(gram).max())
            norms = np.sqrt(np.maximum(0, np.diag(gram)))
            assert np.array_equal(norms, r["norms"])
            for i, j, name in ((0, 2, "cosine_base_auxloc"), (0, 3, "cosine_base_auxerror"),
                                (1, 2, "cosine_classification_auxloc"), (1, 3, "cosine_classification_auxerror")):
                if norms[i] * norms[j] > 0:
                    assert abs(float(gram[i, j] / (norms[i] * norms[j])) - r[name]) <= 1e-12
                else:
                    assert r[name] is None
            gram_checks += 1
    assert gram_checks == 20
    observation = read(OUT / "process_observation.json")
    assert observation["active_diagnostic_workers"] == []
    write(PROBE / "terminal_receipt.json", {"observed_at_utc": now(), "pid": read(PROBE / "started.json")["pid"],
          "exec_session_id": 68693, "reported_exit_code": 0, "source": "exec tool terminal result plus saved result and live process observation",
          "active_worker_observation": identity(OUT / "process_observation.json")})
    write(OUT / "training_clipping_summary.json", clipping)
    plan = {
        "schema": "rampq.gate1.proposed_guarded_correction_experiment/v1", "status": "DESIGN_ONLY_NOT_LAUNCHED_NOT_EXECUTION_CONTRACT",
        "created_at_utc": now(), "hypothesis": "Bound auxiliary dominance and remove local extra conflicts; test intervention selection beyond equally guarded ordinary hard-example selection.",
        "initialization_all_arms": identity(OLD / "attempt_01/training/ordinary_qat/epoch_10_fp32.pt"),
        "proposed_arms": ["ordinary_qat_continuation", "guarded_matched_hard_examples", "guarded_intervention_examples"],
        "gradient_rule": {
            "b": "native detection total gradient", "c": "native classification gradient", "a": "existing weighted localization plus overconfidence auxiliary gradient",
            "projection": "argmin_u ||u-a||^2 subject to u dot b >= 0 and u dot c >= 0; exact two-halfspace projection, defined zero-vector cases",
            "cap": "u <- u*min(1,0.25*||b||/||u||); return b+u; zero when b or u norm is zero",
            "applied_at": "same 4-microbatch accumulation-window boundary, before frozen clip10; never scale optimizer moments separately",
            "claim_boundary": "Only local auxiliary first-order constraints; no guarantee for TP, AP, held-out data, finite step or SGD momentum"
        },
        "selection_control": "Keep existing intervention selection; ordinary hard selection matches positive class/size and negative class/score-bin/geometry counts using the same saved R teacher pool. Exact deterministic control must be implemented and frozen before running.",
        "proposed_budget": {"training_members": 4530, "epochs_each": 3, "batch": 4, "accumulate": 4,
            "optimizer_steps_each": 852, "training_seed": c["training_seed"], "final_epoch": 3,
            "training_and_evaluation_forward_images": 42012, "reconstruction_forward_images_separate": 9,
            "AP_jobs": 3, "new_teacher_forwards": 0, "new_candidate_count": 3, "automatic_retry": False},
        "optimizer": "Reset all three equally; SGD lr1e-4 momentum0.9 weight_decay1e-4, clip10; AMPoff EMAoff; BN/DFL/scales frozen. Training compute overhead must be measured, not claimed equal from equal steps.",
        "development_rule": "AP increases, fixed confidence0.25 PROJECT hits do not decrease and FP do not increase versus continued ordinaryQAT; separately compare against guarded matched hard control. No best epoch/seed/threshold selection.",
        "before_execution": ["Implementation and deterministic selection frozen with source/data hashes", "Projection boundary and zero-vector tests",
            "Zero-auxiliary ordinary QAT equivalence", "Ignore and loss parity", "Accumulation tail and save/rebuild tests", "Bounded measured runtime and explicit launcher contract"],
        "stop": "One finite round; retain all arms and negative results. No automatic budget extension or pass redefinition.",
        "Selection_payload_access": False, "Holdout_payload_access": False, "actual_INT8": False, "scientific_acceptance": False,
        "prior_art": ["https://arxiv.org/abs/2001.06782", "https://arxiv.org/abs/2408.02561"]}
    write(OUT / "next_experiment_plan.json", plan)
    validation = {"status": "PASS_DIAGNOSTIC_ARITHMETIC_AND_PROVENANCE_ONLY", "at_utc": now(),
                  "source_bindings_checked_unchanged": len(source_rows), "direct_threshold_points_checked": points,
                  "lost_target_scalar_geometry_checks": dict(checked), "gradient_gram_checks": gram_checks,
                  "objective_parity_already_measured_batches": 16, "new_model_forward_images_this_validation": 0,
                  "new_AP_jobs": 0, "independent_gradient_recomputation_claimed": False,
                  "original_joint_rule_still_failed": True, "new_training_started": False,
                  "process_observation": identity(OUT / "process_observation.json")}
    assert all(identity(row["path"]) == row for row in source_rows)
    write(OUT / "analysis_validation.json", validation)
    decision = {"status": "DIAGNOSED_PROPOSE_ONE_GUARDED_CORRECTION_CONTROLLED_ROUND", "at_utc": now(),
                "current_method_win": False, "frozen_previous_failures_preserved": True,
                "new_training_started": False, "local_gradient_hypothesis_only": True,
                "next_plan": identity(OUT / "next_experiment_plan.json"), "report": identity(REPORT),
                "new_monitor_needed_now": False, "Selection_payload_access": False, "Holdout_payload_access": False}
    write(OUT / "machine_decision.json", decision)
    latest = ROOT / "artifacts/gate1/latest_execution_state.json"
    previous = read(latest)
    assert previous["snapshot"] == "artifacts/gate1/current_execution_state_v50.json", "Another stage advanced; do not overwrite pointer"
    assert identity(ROOT / previous["snapshot"])["sha256"] == previous["snapshot_sha256"]
    snapshot = ROOT / "artifacts/gate1/current_execution_state_v51.json"
    write(snapshot, {"schema": "rampq.gate1.execution_state/v51", "at_utc": now(),
          "status": "FAILURE_DIAGNOSIS_COMPLETE_GUARDED_TRAINING_PLAN_NOT_STARTED", "previous_pointer": previous,
          "report": identity(REPORT), "diagnostic_result": identity(OUT / "result.json"),
          "gradient_probe_result": identity(PROBE / "result.json"), "analysis_validation": identity(OUT / "analysis_validation.json"),
          "next_experiment_plan": identity(OUT / "next_experiment_plan.json"),
          "new_model_forward_images_this_phase": 64, "weight_updates_this_phase": 0, "AP_jobs_this_phase": 0,
          "current_training_comparison_joint_rule_met": False, "new_training_started": False,
          "scientific_acceptance": False, "actual_INT8": False, "Selection_payload_access": False, "Holdout_payload_access": False,
          "next_action": "Implement and qualify the one described guarded three-arm finite plan before any new training; no active monitor or training. Preserve all previous failures."})
    members = [identity(p) for folder in (OUT, PROBE) for p in sorted(folder.rglob("*")) if p.is_file()]
    members += [identity(REPORT), identity(snapshot)]
    members += [identity(ROOT / "scripts" / name) for name in ("diagnose_gate1_full_comparison_failure.py", "probe_gate1_loss_gradients.py", "close_gate1_failure_diagnostics.py")]
    write(OUT / "delivery_manifest.json", {"files": members})
    assert all(identity(item["path"]) == item for item in members)
    delivery = {"status": "PASS_DIAGNOSIS_AND_PROPOSAL_DELIVERY_NOT_METHOD_PASS", "at_utc": now(), "members": len(members),
                "manifest": identity(OUT / "delivery_manifest.json"), "validation": identity(OUT / "analysis_validation.json"), "report": identity(REPORT)}
    write(OUT / "delivery_validation.json", delivery)
    pointer = {"schema": "rampq.gate1.mutable_latest_state_pointer/v1", "snapshot": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
               "snapshot_sha256": identity(snapshot)["sha256"], "delivery_validation": identity(OUT / "delivery_validation.json"),
               "note": "Report87: saved-output failure diagnosis and64-forward train-only gradient probe complete; old joint FAIL retained. One guarded three-arm plan designed; no new training, AP or monitor started."}
    temporary = latest.with_name("latest_execution_state.v51.tmp")
    write(temporary, pointer)
    assert read(latest) == previous
    temporary.replace(latest)
    print(json.dumps({"delivery": delivery, "state": identity(snapshot), "checks": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
