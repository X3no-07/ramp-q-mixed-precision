"""Finite three-arm continuation: ordinary QAT, guarded hard, guarded intervention."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gate1/20261028_guarded_training_comparison_v1"
OLD = ROOT / "artifacts/gate1/20261027_full_training_comparison_v1"
PLAN = ROOT / "artifacts/gate1/20261027_failure_mechanism_diagnostics_v1/next_experiment_plan.json"
os.environ["YOLO_CONFIG_DIR"] = str(OUT / "settings")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import gate1_intervention_training_v2 as P
import gate1_full_training_comparison as W
import gate1_gradient_guard as G
os.environ["YOLO_CONFIG_DIR"] = str(OUT / "settings")
N = G.N
read, write, identity, require, sha, now = P.read, P.write, P.identity, P.require, P.sha, P.now
ARMS = ("ordinary_qat_continuation", "guarded_matched_hard_examples", "guarded_intervention_examples")
EPOCHS, TRAIN_COUNT, VAL_COUNT, BATCH, ACCUMULATE = 3, 4530, 414, 4, 4
SCHEMA = "rampq.gate1.guarded_three_arm_continuation/v1"
VALUE_SHA = W.value_sha


def source_bindings():
    result = W.source_bindings()
    for name in ("gate1_gradient_guard.py", "gate1_guarded_training_comparison.py", "audit_gate1_guarded_training_comparison.py", "launch_gate1_guarded_training_comparison.py"):
        result[name] = identity(ROOT / "scripts" / name)
    name = "test_gate1_guarded_training_comparison.py"
    result[name] = identity(ROOT / "tests" / name)
    return result


def geometry_sources(record):
    if "training_to_original_params" in record:
        return record["training_to_original_params"], []
    native = record["reused_record"]
    require(identity(native["path"]) == native, "Reused native supervision changed")
    original = read(native["path"])["original_supervision"]
    require(identity(original["path"]) == original, "Reused original geometry changed")
    return read(original["path"])["training_to_original_params"], [native, original]


def derive_controls(record, r, gt, meta, params):
    """Control priorities use only the same R output/GT; selected quotas may overlap."""
    selected = deepcopy(record["native_intervention_correction"])
    predictions = P.predictions(r, params, meta["image_id"], "r")
    matched = P.MATCH.match_project_targets(gt, predictions, meta["width"], meta["height"], confidence=.25, iou_threshold=.5)
    require(matched["counts"] == record["r_counts"], "Saved R teacher/membership replay differs")
    eligible = {x["target_id"] for x in matched["targets"]}
    positives = [g for g in gt if g["score"] == 1 and 1 <= g["category"] <= 10]
    boxes = r[0, :4].T.copy()
    boxes[:, :2] -= boxes[:, 2:] / 2; boxes[:, 2:] += boxes[:, :2]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - params["left"]) / params["ratio"][0]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - params["top"]) / params["ratio"][1]
    overlaps = P.box_iou_matrix(boxes, P.xyxy([g["bbox_xywh"] for g in positives]))
    pool = []
    for i, g in enumerate(positives):
        if g["target_id"] not in eligible:
            continue
        area = g["bbox_xywh"][2] * g["bbox_xywh"][3] * params["ratio"][0] * params["ratio"][1]
        pool.append({"gt_index": i, "category": g["category"] - 1,
                     "size_bin": 0 if area < 32**2 else 1 if area < 96**2 else 2,
                     "quality": float((overlaps[:, i] * r[0, 3 + g["category"]]).max())})
    indexed = {r["gt_index"]: r for r in pool}
    require(all(i in indexed for i in selected["positive_gt"]), "Selected positive missing from eligible R pool")
    treatment_pos = [indexed[i] for i in selected["positive_gt"]]
    hardpos = P.match_counts_to_controls(treatment_pos, pool, lambda x: (x["category"], x["size_bin"]), lambda x: (x["quality"], x["gt_index"]))
    positive_eval = [g for g in gt if g["target_id"] in eligible]
    clipped_gt = np.array([P.MATCH._xyxy(g["bbox_xywh"], meta["width"], meta["height"]) for g in positive_eval]).reshape(-1, 4)
    classes = np.array([g["category"] for g in positive_eval])
    fp = {s["detection_id"] for s in matched["detection_states"] if s["reason"] == "false_positive"}
    errors = []
    for d in predictions:
        if d["detection_id"] not in fp:
            continue
        iou = P.box_iou_matrix(np.array([P.MATCH._xyxy(d["bbox_xywh"], meta["width"], meta["height"])]), clipped_gt)[0]
        errors.append({"anchor": d["raw_anchor_index"], "category": d["category"] - 1, "score": d["score"],
                       "score_bin": min(2, int((d["score"] - .25) / .25)), "kind": N.error_kind(iou, classes, d["category"]),
                       "maximum_GT_IoU": float(iou.max()) if len(iou) else 0.})
    indexed_errors = {(x["anchor"], x["category"]): x for x in errors}
    for e in selected["errors"]:
        actual = indexed_errors[e["anchor"], e["category"]]
        require(all(e[k] == actual[k] for k in actual), "Original intervention error identity changed")
    hardneg = P.match_counts_to_controls(selected["errors"], errors,
                  lambda x: (x["category"], x["score_bin"], x["kind"]), lambda x: (-x["score"], x["anchor"]))
    return {"image_id": meta["image_id"], "selection": {ARMS[0]: {"positive_gt": [], "errors": []},
            ARMS[1]: {"positive_gt": [x["gt_index"] for x in hardpos], "errors": hardneg}, ARMS[2]: selected},
            "positive_pool": pool, "error_pool": errors, "R_counts": matched["counts"],
            "positive_overlap": len(set(selected["positive_gt"]) & {x["gt_index"] for x in hardpos}),
            "error_overlap": len({(x["anchor"], x["category"]) for x in selected["errors"]} & {(x["anchor"], x["category"]) for x in hardneg}),
            "control_uses_Q_outcomes_beyond_matching_quota": False, "quotas_are_not_effective_gradient_count_guarantees": True}


def qualify():
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "qualification.json").exists() and not (OUT / "contract.json").exists(), "Qualification already exists")
    sources = source_bindings()
    argv = [sys.executable, "-B", "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider",
            str(ROOT / "tests/test_gate1_guarded_training_comparison.py"),
            str(ROOT / "tests/test_gate1_native_target_correction.py"),
            "--junitxml=" + str(OUT / "qualification_tests.xml")]
    started = time.monotonic()
    with (OUT / "qualification_console.txt").open("x", encoding="utf-8") as stream:
        code = subprocess.call(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    tree = ET.parse(OUT / "qualification_tests.xml")
    suites = list(tree.getroot().iter("testsuite"))
    counts = {k: sum(int(s.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    passed = code == 0 and counts["tests"] > 0 and all(counts[k] == 0 for k in ("failures", "errors", "skipped"))
    write(OUT / "qualification.json", {"status": "PASS" if passed else "FAIL", "exit_code": code, "counts": counts,
          "source_bindings": sources, "sources_unchanged": sources == source_bindings(), "elapsed_seconds": time.monotonic() - started,
          "model_test_images": "Synthetic only; saved real teacher replay uses no model forward", "at_utc": now()})
    require(passed and sources == source_bindings(), "Qualification failed; preserve attempt")
    print(json.dumps(read(OUT / "qualification.json"), ensure_ascii=True), flush=True)


def prepare():
    require(not (OUT / "contract.json").exists() and not (OUT / "attempt_01").exists(), "Existing stage")
    qualification = read(OUT / "qualification.json")
    require(qualification["status"] == "PASS" and qualification["source_bindings"] == source_bindings(), "Sources changed after qualification")
    old = read(OLD / "contract.json"); plan = read(PLAN)
    require(identity(OLD / "contract.json")["sha256"] == "fc57b2428ca53ef65dbfd58c987cdbfd70b60cfde6cd9c8eb8092dc890c31791", "Old contract changed")
    previous = read(ROOT / "artifacts/gate1/latest_execution_state.json")
    delivery = read(OLD / "delivery_validation.json")
    require(identity(delivery["manifest"]["path"]) == delivery["manifest"], "Previous manifest changed")
    sealed = {f["path"]: f for f in read(delivery["manifest"]["path"])["files"]}
    teachers = []
    for i, meta in enumerate(old["training_annotation_metadata"]):
        path = OLD / "attempt_01/training_supervision" / f"{i:04d}.json"
        binding = identity(path); require(binding == sealed[str(path)], "Teacher record changed")
        row = read(path); require(row["image_id"] == meta["image_id"], "Teacher member order changed")
        params, geometry = geometry_sources(row)
        require(identity(row["raw_source"]["path"]) == row["raw_source"], "Teacher array changed")
        teachers.append({"image_id": row["image_id"], "record": binding, "raw": row["raw_source"], "params": params, "geometry_sources": geometry})
        if (i + 1) % 256 == 0: print(json.dumps({"stage": "BIND_COMPLETED_TEACHERS", "done": i + 1, "total": TRAIN_COUNT}), flush=True)
    initial = plan["initialization_all_arms"]
    require(identity(initial["path"]) == initial and initial == sealed[initial["path"]], "Common initialization changed")
    c = deepcopy(old)
    c.update(schema=SCHEMA, created_at_utc=now(), authorization='User: Alright, get going; execute the three-arm constrained error correction comparison as agreed in report 87, prepared for this phase on 2026-10-28',
             previous_state=identity(ROOT / previous["snapshot"]), prior_full_contract=identity(OLD / "contract.json"),
             proposed_plan=identity(PLAN), initial_candidate=initial, qualification=identity(OUT / "qualification.json"),
             source_bindings=source_bindings(), teacher_inputs=teachers, arms=list(ARMS), evaluation_order=list(ARMS),
             epochs=EPOCHS, optimizer_steps_per_arm=852, final_candidate="epoch_03_only", new_teacher_forwards=0,
             training_forward_images=40770, evaluation_forward_images=1242, real_image_forward_budget=42012,
             reconstruction_real_image_forwards_separate=9, AP_jobs=3,
             auxiliary_gradient_rule=plan["gradient_rule"], control_selection="Frozen R teacher: lowest IoU*class-score positives matched class/size, highest-score false positives matched class/score-bin/geometry; exact per-image counts; intervention selection unchanged",
             training="Start ALL arms from same ordinaryQAT epoch10; reset identical SGD lr1e-4 momentum0.9 decay1e-4; 3epochs; batch4 accumulate4; frozen BN/DFL/scales; no augmentation/AMP/EMA. Project accumulated auxiliary before clip10. Ordinary one backward; guarded three component backwards: not equal training FLOPs.",
             primary_development_rule="AP_all increases AND fixed0.25 PROJECT hits do not decrease AND FP do not increase relative to continued ordinaryQAT. Compare guarded intervention separately against guarded matched hard. No threshold/epoch/seed selection.",
             candidate_budget={"new_detector_candidates": 3, "training_seeds": 1, "new_scorer_fits": 0, "frozen_score_transfer_branches": 0},
             scorer_transfer="Disabled; old scorer binding retained only because unchanged evaluation helper reads metadata; never applied",
             comparability_limit="Development continuation budget, not convergence; floating QDQ, not actualINT8. Gradient protection not novel by itself. Effective supervision and training wall time may differ; record them.",
             gradient_checks={"float64_projection_relative": G.DOUBLE_FEASIBILITY, "float32_cast_relative": G.FLOAT32_FEASIBILITY,
                              "distinct_from_original_all_anchor_output_tolerances": True},
             minimum_free_disk_bytes=8*1024**3, timeout_seconds=21600, automatic_retry=False)
    require(c["training_seed"] == plan["proposed_budget"]["training_seed"] and len(teachers) == 4530, "Plan differs")
    write(OUT / "contract.json", c)
    print(json.dumps({"status": "FROZEN", "contract": identity(OUT / "contract.json"), "new_forwards": 42012}, ensure_ascii=True), flush=True)


class Worker(W.Worker):
    def __init__(self):
        self.contract = read(OUT / "contract.json")
        self.attempt = OUT / "attempt_01"; self.attempt.mkdir(exist_ok=False)
        self.started = time.monotonic()
        self.live = {"schema": SCHEMA, "pid": os.getpid(), "parent_pid": os.getppid(), "started_at_utc": now(),
                     "contract": identity(OUT / "contract.json"), "forward_images": 0, "reconstruction_forward_images": 0}
        self.progress("STARTING", 0, 1)

    def progress(self, stage, done, total, **extra):
        require(shutil.disk_usage(OUT).free >= self.contract["minimum_free_disk_bytes"], "Disk below frozen minimum")
        P.Worker.progress(self, stage, done, total, elapsed_seconds=round(time.monotonic()-self.started, 3), **extra)

    def verify(self):
        c = self.contract
        require(c["schema"] == SCHEMA and c["source_bindings"] == source_bindings(), "Frozen source changed")
        require(c["framework"] == P.verify_framework_sources(), "Framework changed")
        keys = ("candidate", "architecture", "formal_contract", "train_manifest", "training_val_manifest", "evaluation_protocol",
                "training_annotation_assignment", "ranges", "scorer_model", "previous_state", "initial_candidate", "prior_full_contract", "proposed_plan", "qualification")
        for key in keys: require(identity(c[key]["path"]) == c[key], "Input changed:" + key)
        for row in c["teacher_inputs"]:
            for b in [row["record"], row["raw"], *row["geometry_sources"]]:
                require(identity(b["path"]) == b, "Reused teacher input changed")
        for row in c["FP32_saved_reference_subset"]:
            require(identity(row["raw"]["path"]) == row["raw"], "FP32 reference changed")

    def supervision(self):
        folder = self.attempt / "training_supervision"; folder.mkdir()
        root = Path(read(self.contract["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
        records = []; counts = Counter()
        self.progress("DERIVE_MATCHED_TRAIN_ONLY_CONTROLS_FROM_SAVED_ARRAYS", 0, TRAIN_COUNT)
        for i, (meta, source) in enumerate(zip(self.contract["training_annotation_metadata"], self.contract["teacher_inputs"])):
            require(meta["assigned_split"] == "train" and meta["image_id"] == source["image_id"], "Training role mismatch")
            ann = (root / meta["annotation_path"]).resolve(strict=True)
            require(ann.is_relative_to(root) and sha(ann) == meta["annotation_sha256"], "Training annotation changed")
            gt = P.MATCH.parse_ground_truth(ann.read_text(encoding="utf-8-sig"), meta["image_id"])
            with np.load(source["raw"]["path"], allow_pickle=False) as d: r = d["r"].copy()
            row = derive_controls(read(source["record"]["path"]), r, gt, meta, source["params"])
            row["source"] = source
            write(folder / f"{i:04d}.json", row); records.append(row)
            counts.update(positives=len(row["selection"][ARMS[2]]["positive_gt"]), errors=len(row["selection"][ARMS[2]]["errors"]),
                          positive_overlap=row["positive_overlap"], error_overlap=row["error_overlap"])
            if (i+1) % 16 == 0 or i+1 == TRAIN_COUNT:
                self.progress("DERIVE_MATCHED_TRAIN_ONLY_CONTROLS_FROM_SAVED_ARRAYS", i+1, TRAIN_COUNT)
        require(counts["positives"] == 5094 and counts["errors"] == 7144, "Frozen intervention budgets changed")
        write(folder / "summary.json", {"images": TRAIN_COUNT, "counts": dict(counts), "new_teacher_forwards": 0, "exact_stratum_budgets": True})
        return records

    def train(self, model, dataset, records, arm, maxima):
        c = self.contract; P.seed_all(c["training_seed"])
        model.args.epochs = EPOCHS
        folder = self.attempt / "training" / arm; folder.mkdir(parents=True)
        initial = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        initial_hash = VALUE_SHA(initial)
        model.train(); P.freeze_bn(model)
        criterion = G.GuardedCriterion(model)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=1e-4, momentum=.9, weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        generator = np.random.Generator(np.random.PCG64(c["training_seed"]))
        orders = []; batches = steps = 0; epochs = []; start = time.monotonic(); accumulation = None
        with (folder / "batch_trace.jsonl").open("x", encoding="utf-8") as trace, (folder / "gradient_guard_trace.jsonl").open("x", encoding="utf-8") as guard_trace:
            for epoch in range(EPOCHS):
                order = generator.permutation(len(dataset)).tolist(); orders.append(order); losses = []
                for b, pos in enumerate(range(0, len(order), BATCH)):
                    ids = order[pos:pos+BATCH]; batch = P.gpu_batch([dataset[j] for j in ids])
                    self.live["forward_images"] += len(ids)
                    require(self.live["forward_images"] <= c["real_image_forward_budget"], "Forward budget exceeded")
                    pred = model(batch["img"]); denominator, update = W.accumulation_window(b, c["batches_per_epoch"])
                    if arm == ARMS[0]:
                        loss = criterion.compute(pred, batch, [], "ordinary_qat")
                        (loss / denominator).backward()
                        exact_parity = None
                    else:
                        specs = [records[j]["selection"][arm] for j in ids]
                        base, classification, auxiliary = criterion.components(pred, batch, specs)
                        loss = base + auxiliary; exact_parity = None
                        if b == 0:
                            reference = criterion.compute(pred, batch, [{"native_intervention_correction": s} for s in specs], "native_intervention_correction")
                            require(torch.equal(loss.detach(), reference.detach()), "Frozen native loss parity differs")
                            exact_parity = True; del reference
                        grads = [G.flat_gradient(x / denominator, parameters, retain_graph=i < 2) for i, x in enumerate((base, classification, auxiliary))]
                        if accumulation is None: accumulation = grads
                        else:
                            for old, new in zip(accumulation, grads): old.add_(new)
                        del grads, base, classification, auxiliary
                    require(bool(torch.isfinite(loss)), "Nonfinite loss")
                    scalar = float(loss.detach()); losses.append(scalar); batches += 1; norm = None
                    if update:
                        if arm != ARMS[0]:
                            combined, measurement = G.guarded_gradient(*accumulation)
                            G.assign_gradient(parameters, combined)
                            guard_trace.write(json.dumps({"epoch": epoch+1, "step": steps+1, "end_batch": batches, **measurement}, allow_nan=False) + "\n")
                            del combined, measurement; accumulation = None
                        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), "Missing/nonfinite optimizer gradient")
                        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True))
                        optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
                    trace.write(json.dumps({"epoch": epoch+1, "epoch_batch": b+1, "batch": batches, "indices": ids,
                        "loss": scalar, "optimizer_step": steps, "update": update, "accumulation_denominator": denominator,
                        "grad_norm_before_clip": norm, "original_loss_parity": exact_parity, **criterion.last_aux}, allow_nan=False) + "\n")
                    if batches % 16 == 0 or b+1 == c["batches_per_epoch"]:
                        trace.flush(); guard_trace.flush()
                        self.progress("TRAINING_"+arm, batches, EPOCHS*c["batches_per_epoch"], epoch=epoch+1, optimizer_steps=steps, latest_loss=scalar)
                    del batch, pred, loss
                require(accumulation is None, "Unflushed accumulated gradient")
                state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
                require(all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point()), "Nonfinite checkpoint")
                torch.save({"state_dict": state, "epoch": epoch+1, "arm": arm, "contract_sha256": self.live["contract"]["sha256"]}, folder / f"epoch_{epoch+1:02d}_fp32.pt")
                epochs.append({"epoch": epoch+1, "mean_batch_loss": float(np.mean(losses)), "optimizer_steps": steps}); del state
        require(steps == 852 and batches == 3399, "Training budget differs")
        frozen = [n for n in initial if "running_" in n or "num_batches_tracked" in n or n.endswith("_scale") or ".dfl." in n]
        require(all(torch.equal(initial[n], model.state_dict()[n].cpu()) for n in frozen), "Frozen state changed")
        changed = sum(int((model.state_dict()[n].cpu() != v).sum()) for n, v in initial.items() if n.endswith("weight"))
        require(changed > 0, "No weight updates")
        model.eval(); x = P.gpu_batch([dataset[0]])["img"]
        with torch.inference_mode(): reference = model(x)[0].detach().cpu()
        saved = torch.load(folder / "epoch_03_fp32.pt", map_location="cpu", weights_only=True)
        rebuilt = P.new_model(maxima); rebuilt.args.epochs = EPOCHS; rebuilt.load_state_dict(saved["state_dict"], strict=True)
        with torch.inference_mode(): replay = rebuilt(x)[0].detach().cpu()
        require(torch.equal(reference, replay), "Independent CUDA save/rebuild differs")
        rebuilt.cpu()
        with torch.inference_mode(): cpu = rebuilt(x.cpu())[0].detach().cpu()
        require(bool(torch.isfinite(cpu).all()), "CPU rebuild nonfinite")
        self.live["reconstruction_forward_images"] += 3
        write(folder / "epoch_orders.json", orders)
        write(folder / "training_receipt.json", {"arm": arm, "epochs": EPOCHS, "batches": batches, "optimizer_steps": steps,
              "initial_state_value_sha256": initial_hash, "final_state_value_sha256": VALUE_SHA(model.state_dict()),
              "changed_weight_elements": changed, "frozen_BN_DFL_scales_unchanged": True, "epoch_summaries": epochs,
              "epoch_orders": identity(folder / "epoch_orders.json"), "final_candidate": identity(folder / "epoch_03_fp32.pt"),
              "CUDA_reconstruction_exact": True, "CPU_CUDA_output_max_abs": float((reference-cpu).abs().max()),
              "CPU_CUDA_equivalence_certified": False, "reconstruction_real_image_forwards_separate": 3,
              "training_elapsed_seconds": time.monotonic()-start, "gradient_guard_windows": steps if arm != ARMS[0] else 0})
        criterion._capture_handle.remove()
        del rebuilt, criterion, optimizer, parameters, initial, saved, x; torch.cuda.empty_cache()
        return model

    def run(self):
        P.configure(); self.verify(); require(torch.cuda.is_available(), "CUDA unavailable")
        write(self.attempt / "runtime.json", {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "GPU": torch.cuda.get_device_name(0), "backend": P.FT.backend_settings()})
        c = self.contract; maxima = read(c["ranges"]["path"])["max_abs"]
        dataset = P.SelectedTrainingDataset(c["train_manifest"], c["training_members"])
        records = self.supervision(); jobs = []
        initial = torch.load(c["initial_candidate"]["path"], map_location="cpu", weights_only=True)
        for arm in ARMS:
            model = P.new_model(maxima); P.precision(model, "r_only:G1")
            model.load_state_dict(initial["state_dict"], strict=True)
            model = self.train(model, dataset, records, arm, maxima)
            jobs.extend(self.evaluate(model, arm)); del model; torch.cuda.empty_cache()
        require(self.live["forward_images"] == 42012 and self.live["reconstruction_forward_images"] == 9, "Forward count incomplete")
        self.score_jobs(jobs)
        metrics = {a: {"official_source_python": read(self.attempt / "evaluation" / a / "metrics.json"),
                       "project": read(self.attempt / "evaluation" / a / "project_metrics.json")} for a in ARMS}
        def compare(a, b):
            ap = metrics[a]["official_source_python"]["metrics_percent"]["AP_all"] - metrics[b]["official_source_python"]["metrics_percent"]["AP_all"]
            x, y = (metrics[t]["project"]["conf025_counts"] for t in (a, b))
            tp, fp = x["hits"] - y["hits"], x["false_positive_detections"] - y["false_positive_detections"]
            return {"AP_delta_percentage_points": ap, "TP_delta_conf025": tp, "FP_delta_conf025": fp, "joint_rule_met": ap > 0 and tp >= 0 and fp <= 0}
        result = {"status": "COMPLETE_FIXED_GUARDED_THREE_ARM_DEVELOPMENT", "metrics": metrics,
                  "intervention_vs_ordinary": compare(ARMS[2], ARMS[0]), "hard_vs_ordinary": compare(ARMS[1], ARMS[0]),
                  "intervention_vs_matched_hard": compare(ARMS[2], ARMS[1]), "scientific_acceptance": False, "actual_INT8": False,
                  "single_seed": True, "Selection_payload_access": False, "Holdout_payload_access": False, "historical_failures_preserved": True}
        write(self.attempt / "comparison_result.json", result)
        self.verify()
        files = [identity(p) for p in sorted(self.attempt.rglob("*")) if p.is_file() and p.name not in {"live_status.json", "events.jsonl", "manifest.json", "completion_receipt.json"}]
        write(self.attempt / "manifest.json", {"files": files})
        self.progress(result["status"], 1, 1)
        write(self.attempt / "completion_receipt.json", {**self.live, "status": result["status"], "elapsed_seconds": time.monotonic()-self.started,
              "comparison": identity(self.attempt / "comparison_result.json"), "manifest": identity(self.attempt / "manifest.json")})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("qualify", "prepare", "run")); args = parser.parse_args()
    if args.mode == "qualify": qualify(); return
    if args.mode == "prepare": prepare(); return
    worker = Worker(); awake = False
    try:
        if os.name == "nt":
            import ctypes
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        worker.run()
    except BaseException as error:
        worker.live.update(status="FAILED_ATTEMPT_PRESERVED", error_type=type(error).__name__, error=str(error), updated_at_utc=now())
        write(worker.attempt / "live_status.json", worker.live, once=False)
        write(worker.attempt / "failure_receipt.json", {**worker.live, "traceback": traceback.format_exc()})
        raise
    finally:
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__": main()
