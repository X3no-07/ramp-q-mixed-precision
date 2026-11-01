"""Finite, saved-output-only diagnosis. No model, fitting, AP job or data expansion."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/gate1/20261027_full_training_comparison_v1"
OUT = ROOT / "artifacts/gate1/20261027_failure_mechanism_diagnostics_v1"
ARMS = ("ordinary_qat", "native_intervention_correction", "ordinary_qat_frozen_scorer")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def grouped(scores, flags):
    assert scores.ndim == flags.ndim == 1 and len(scores) == len(flags)
    assert len(scores) and np.isfinite(scores).all() and np.isin(flags, [-1, 0, 1]).all()
    order = np.argsort(-scores, kind="stable")
    s, f = scores[order], flags[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    return dict(threshold=s[ends], TP=np.cumsum(f == 1)[ends], FP=np.cumsum(f == 0)[ends])


def point(curve, index):
    return {k: float(v[index]) if k == "threshold" else int(v[index]) for k, v in curve.items()}


def fp_point(curve, budget):
    good = np.flatnonzero(curve["FP"] <= budget)
    return point(curve, good[-1]) if len(good) else dict(TP=0, FP=0, threshold=None)


def tp_point(curve, target):
    good = np.flatnonzero(curve["TP"] >= target)
    return {"achievable": True, **point(curve, good[0])} if len(good) else {"achievable": False}


def summaries(values):
    a = np.asarray(values, dtype=np.float64)
    assert a.ndim == 1 and np.isfinite(a).all()
    if not len(a):
        return {"count": 0}
    return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.quantile(a, .1)), "p90": float(np.quantile(a, .9))}


def clip(box, width, height):
    x, y, w, h = box
    return np.minimum(np.maximum([x, y, x + w, y + h], 0), [width, height, width, height])


def ious(boxes, target):
    if not len(boxes):
        return np.empty(0)
    boxes = np.asarray(boxes, dtype=np.float64)
    inter = np.maximum(np.minimum(boxes[:, 2:], target[2:]) - np.maximum(boxes[:, :2], target[:2]), 0).prod(1)
    union = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1) + np.maximum(target[2:] - target[:2], 0).prod() - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def prepare():
    assert not OUT.exists(), "Preserve existing diagnostic; do not overwrite or rerun"
    assert identity(OLD / "contract.json")["sha256"] == "fc57b2428ca53ef65dbfd58c987cdbfd70b60cfde6cd9c8eb8092dc890c31791"
    delivery = read(OLD / "delivery_validation.json")
    assert identity(delivery["manifest"]["path"]) == delivery["manifest"]
    sealed = {str(Path(f["path"]).resolve()): f for f in read(delivery["manifest"]["path"])["files"]}
    c = read(OLD / "contract.json")
    assert identity(c["evaluation_protocol"]["path"]) == c["evaluation_protocol"]
    prepared = Path(read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
    assert len(c["evaluation_members"]) == 414
    sources = [identity(OLD / "contract.json"), identity(OLD / "delivery_validation.json"), identity(delivery["manifest"]["path"]),
               identity(ROOT / "artifacts/gate1/current_execution_state_v50.json"), identity(__file__), c["evaluation_protocol"]]
    for arm in ARMS:
        for name in ("official_IoU05_rank_curve.npz", "saved_predictions_and_targets.json", "project_metrics.json", "metrics.json"):
            item = identity(OLD / "attempt_01/evaluation" / arm / name)
            assert item == sealed[item["path"]], "Saved evaluation differs from sealed delivery"
            sources.append(item)
    for arm in ARMS[:2]:
        item = identity(OLD / "attempt_01/training" / arm / "batch_trace.jsonl")
        assert item == sealed[item["path"]]
        sources.append(item)
    for meta in c["evaluation_members"]:
        assert meta["assigned_split"] == "training_val"
        ann = (prepared / meta["annotation_path"]).resolve(strict=True)
        assert ann.is_relative_to(prepared)
        item = identity(ann)
        assert item["sha256"] == meta["annotation_sha256"]
        sources.append(item)
    protocol = {
        "schema": "rampq.gate1.saved_failure_diagnosis/v1", "frozen_at_utc": now(), "sources": sources,
        "data": "Only the existing 414 training_val saved predictions/matches/rank curves and annotations; two existing training traces.",
        "new_model_forwards": 0, "new_training_steps": 0, "new_AP_jobs": 0, "new_candidates": 0,
        "analyses": [
            "Rebuild tie-aware official-source IoU0.5 rank curve exactly from saved scores/flags. Compare all three branches at ordinary QAT's own official-style confidence0.25 FP budget and TP target.",
            "Same fixed PROJECT confidence0.25 target identities: shared/lost/gained, per class, shared-target confidence/IoU changes.",
            "Partition lost targets using unchanged saved post-NMS predictions and saved category0 exclusion states: same-class IoU>=0.5 candidate above threshold but unmatched; only below threshold; no such candidate in saved pool.",
            "Summarize all 11330 recorded batches per arm, and per epoch: base/auxiliary scalar losses, kept/active error channels, any-class foreground counters. No gradient attribution from scalar loss magnitudes."
        ],
        "threshold_policy": "Computed matched-budget thresholds are retrospective diagnostic oracles only. No deployed threshold, calibration fitting or replacement of the frozen joint failure rule.",
        "limitations": ["Official rank matcher and PROJECT IoU-first target matcher remain separate.",
                        "Saved candidate availability is not proof of recoverability after one-to-one matching, nor absence in all raw anchors.",
                        "Repeated-use development data, single seed, no causal ablation, no novelty or independent science conclusion."],
        "Selection_payload_access": False, "Holdout_payload_access": False,
    }
    OUT.mkdir(parents=True)
    write(OUT / "diagnostic_contract.json", protocol)
    print(json.dumps({"status": "FROZEN", "source_count": len(sources), "contract": identity(OUT / "diagnostic_contract.json")}, ensure_ascii=False))


def run():
    protocol = read(OUT / "diagnostic_contract.json")
    assert not (OUT / "result.json").exists(), "Do not repeat completed diagnosis"
    assert all(identity(x["path"]) == x for x in protocol["sources"]), "Input changed"
    # Independent corner cases: score ties must enter together; ignored DT adds no FP;
    # a TP target beyond the saved candidate pool must be reported unreachable.
    z = grouped(np.array([.5, .5, .4, .9]), np.array([1, 0, -1, 1]))
    assert fp_point(z, 0) == dict(threshold=.9, TP=1, FP=0)
    assert tp_point(z, 2) == dict(achievable=True, threshold=.5, TP=2, FP=1)
    assert tp_point(z, 3) == dict(achievable=False)
    assert ious([[0, 0, 2, 2], [2, 2, 4, 4]], np.array([0, 0, 2, 2])).tolist() == [1., 0.]
    print("Qualified tied-score/ignored-DT/unreachable-target/geometry corner cases", flush=True)
    curves, fixed, saved = {}, {}, {}
    for arm in ARMS:
        folder = OLD / "attempt_01/evaluation" / arm
        with np.load(folder / "official_IoU05_rank_curve.npz", allow_pickle=False) as archive:
            curves[arm] = grouped(archive["score"], archive["flag"])
            assert all(np.array_equal(value, archive[key]) for key, value in curves[arm].items())
            mask = archive["score"] >= .25
            fixed[arm] = {"TP": int(np.sum(mask & (archive["flag"] == 1))), "FP": int(np.sum(mask & (archive["flag"] == 0)))}
        saved[arm] = read(folder / "saved_predictions_and_targets.json")
        assert len(saved[arm]) == 414 and all(x["frozen_confidence"] == .25 for x in saved[arm])
        counts = Counter()
        for row in saved[arm]:
            counts.update(row["target_matching"]["counts"])
        assert dict(counts) == read(folder / "project_metrics.json")["conf025_counts"]
    reference = fixed[ARMS[0]]
    ranking = {arm: {"official_style_at_confidence025": fixed[arm],
                     "at_baseline_FP_budget": fp_point(curves[arm], reference["FP"]),
                     "at_baseline_TP_target": tp_point(curves[arm], reference["TP"])} for arm in ARMS}
    write(OUT / "rank_diagnostic.json", {"reference": reference, "retrospective_oracle_only": True, "branches": ranking})
    print("Rank curves checked and matched-budget summaries saved", flush=True)
    c = read(OLD / "contract.json")
    prepared = Path(read(c["evaluation_protocol"]["path"])["prepared_root"])
    transitions = {arm: Counter() for arm in ARMS[1:]}
    per_class = {arm: defaultdict(Counter) for arm in ARMS[1:]}
    shared = {arm: {"confidence_delta": [], "IoU_delta": []} for arm in ARMS[1:]}
    target_details = []
    for index, meta in enumerate(c["evaluation_members"]):
        records = {arm: saved[arm][index] for arm in ARMS}
        assert all(row["image_id"] == meta["image_id"] and row["GT_source_sha256"] == meta["annotation_sha256"] for row in records.values())
        gt = {}
        for lineno, line in enumerate((prepared / meta["annotation_path"]).read_text(encoding="utf-8-sig").splitlines()):
            if line.strip():
                values = [float(v) for v in line.strip().strip(",").split(",")]
                assert len(values) == 8
                gt[f"{meta['image_id']}:{lineno}"] = {"box": clip(values[:4], meta["width"], meta["height"]), "category": int(values[5])}
        base = records[ARMS[0]]["target_matching"]
        base_hits = {m["target_id"]: m for m in base["matches"]}
        base_targets = {t["target_id"] for t in base["targets"]}
        for arm in ARMS[1:]:
            row = records[arm]
            matching = row["target_matching"]
            assert {t["target_id"] for t in matching["targets"]} == base_targets
            hits = {m["target_id"]: m for m in matching["matches"]}
            common = set(base_hits) & set(hits)
            lost = set(base_hits) - set(hits)
            gained = set(hits) - set(base_hits)
            transitions[arm].update(shared=len(common), lost=len(lost), gained=len(gained), neither=len(base_targets - set(base_hits) - set(hits)))
            for label, ids in (("shared", common), ("lost", lost), ("gained", gained)):
                for target in ids:
                    per_class[arm][str(gt[target]["category"])][label] += 1
            for target in common:
                shared[arm]["confidence_delta"].append(hits[target]["confidence"] - base_hits[target]["confidence"])
                shared[arm]["IoU_delta"].append(hits[target]["iou"] - base_hits[target]["iou"])
            states = {s["detection_id"]: s["reason"] for s in matching["detection_states"]}
            assert len(states) == len(row["predictions"])
            pred_by_class = defaultdict(list)
            for p in row["predictions"]:
                if states[p["detection_id"]] not in ("category_0_raster_coverage_ge_0.5", "outside_evaluated_categories", "empty_after_clipping"):
                    pred_by_class[p["category"]].append(p)
            for target in sorted(lost):
                truth = gt[target]
                candidates = pred_by_class[truth["category"]]
                overlap = ious([clip(p["bbox_xywh"], meta["width"], meta["height"]) for p in candidates], truth["box"])
                good = np.flatnonzero(overlap >= .5)
                score = max((candidates[k]["score"] for k in good), default=None)
                if score is None:
                    reason = "no_same_class_IoU05_candidate_in_saved_postNMS_pool"
                elif score < .25:
                    reason = "candidate_available_only_below_fixed_threshold"
                else:
                    reason = "candidate_above_fixed_threshold_but_unmatched"
                transitions[arm][reason] += 1
                target_details.append({"image_id": meta["image_id"], "arm": arm, "target_id": target,
                                       "category": truth["category"], "diagnostic": reason,
                                       "best_candidate_score": score, "candidate_count_IoU05": len(good),
                                       "baseline_match": base_hits[target]})
    target_summary = {arm: {"counts": dict(transitions[arm]),
                            "shared_target_changes": {k: summaries(v) for k, v in shared[arm].items()},
                            "per_category": {k: dict(v) for k, v in sorted(per_class[arm].items())}} for arm in ARMS[1:]}
    for arm in ARMS[1:]:
        total = transitions[arm]
        assert total["shared"] + total["lost"] == 10596
        assert total["lost"] == sum(v for k, v in total.items() if k.startswith("candidate_") or k.startswith("no_same_class_"))
    write(OUT / "target_transition_summary.json", target_summary)
    write(OUT / "lost_target_diagnostics.json", target_details)
    print("414-image saved target transitions and candidate availability classified", flush=True)
    losses = {}
    for arm in ARMS[:2]:
        rows = [json.loads(line) for line in (OLD / "attempt_01/training" / arm / "batch_trace.jsonl").read_text("utf-8").splitlines()]
        assert len(rows) == 11330 and sum(bool(r["update"]) for r in rows) == 2840
        groups = {"all": rows, **{str(e): [r for r in rows if r["epoch"] == e] for e in range(1, 11)}}
        result = {}
        for key, group in groups.items():
            counter = Counter({k: sum(r[k] for r in group) for k in ("positive_objects", "selected_error_channels_kept", "active_error_channels", "ignored_error_channels", "error_anchors_currently_foreground")})
            means = {k: float(np.mean([r[k] for r in group])) for k in ("loss", "localization_loss", "overconfidence_loss", "auxiliary_loss")}
            assert all(np.isclose(r["auxiliary_loss"], .25 * (r["localization_loss"] + r["overconfidence_loss"]), atol=2e-7) for r in group)
            result[key] = {"batches": len(group), "batch_mean_scalars": means,
                           "base_scalar_from_difference": means["loss"] - means["auxiliary_loss"],
                           "localization_share_of_auxiliary_scalar": means["localization_loss"] / (means["localization_loss"] + means["overconfidence_loss"]) if means["auxiliary_loss"] else None,
                           "repeated_channel_occurrences": dict(counter),
                           "gradient_norm_before_clip": summaries([r["grad_norm_before_clip"] for r in group if r["update"]])}
        losses[arm] = result
    write(OUT / "training_loss_diagnostic.json", losses)
    unchanged = all(identity(x["path"]) == x for x in protocol["sources"])
    assert unchanged, "A bound source changed during diagnosis"
    result = {"status": "PASS_FINITE_SAVED_DIAGNOSIS_NOT_METHOD_ACCEPTANCE", "finished_at_utc": now(),
              "contract": identity(OUT / "diagnostic_contract.json"), "sources_checked_unchanged": len(protocol["sources"]),
              "rank_diagnostic": ranking, "target_transitions": target_summary,
              "training_scalar_summary": {arm: losses[arm]["all"] for arm in ARMS[:2]},
              "new_model_forwards": 0, "new_training_steps": 0, "new_AP_jobs": 0,
              "original_joint_rule_still_failed": True, "Selection_payload_access": False, "Holdout_payload_access": False}
    write(OUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
