"""Two frozen checkpoint gradient probes; 32 existing train images, no updates."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time
import traceback

import numpy as np
from diagnose_gate1_full_comparison_failure import ROOT, OLD, read, write, identity, now

OUT = ROOT / "artifacts/gate1/20261027_train_gradient_probe_v1"
ARMS = ("ordinary_qat", "native_intervention_correction")


def prepare():
    assert not OUT.exists(), "Preserve any earlier attempt"
    c = read(OLD / "contract.json")
    assert identity(OLD / "contract.json")["sha256"] == "fc57b2428ca53ef65dbfd58c987cdbfd70b60cfde6cd9c8eb8092dc890c31791"
    delivery = read(OLD / "delivery_validation.json")
    assert identity(delivery["manifest"]["path"]) == delivery["manifest"]
    sealed = {str(Path(f["path"]).resolve()): f for f in read(delivery["manifest"]["path"])["files"]}
    sources = [identity(__file__), identity(Path(__file__).with_name("diagnose_gate1_full_comparison_failure.py")),
               identity(OLD / "contract.json"), identity(delivery["manifest"]["path"])]
    sources += list(c["source_bindings"].values())
    sources += [c[k] for k in ("ranges", "candidate", "architecture", "formal_contract", "train_manifest")]
    orderpath = OLD / "attempt_01/training/ordinary_qat/epoch_orders.json"
    assert identity(orderpath) == sealed[str(orderpath)]
    sources.append(identity(orderpath))
    indices = read(orderpath)[0][:32]
    assert len(indices) == len(set(indices)) == 32
    rows = [c["training_members"][i] for i in indices]
    header = read(c["train_manifest"]["path"])
    dataroot = (Path(c["train_manifest"]["path"]).parent / header["data_root"]).resolve(strict=True)
    for i, row in zip(indices, rows):
        assert row["split_role"] == "train"
        for key in ("image", "labels", "sidecar"):
            bound = row[key]
            path = (dataroot / bound["path"]).resolve(strict=True)
            assert path.is_relative_to(dataroot)
            item = identity(path)
            assert item["sha256"] == bound["sha256"] and item["bytes"] == bound["bytes"]
            sources.append(item)
        path = OLD / "attempt_01/training_supervision" / f"{i:04d}.json"
        assert identity(path) == sealed[str(path)]
        sources.append(identity(path))
    for arm in ARMS:
        for name in ("epoch_10_fp32.pt", "training_receipt.json"):
            path = OLD / "attempt_01/training" / arm / name
            assert identity(path) == sealed[str(path)]
            sources.append(identity(path))
    assert all(identity(f["path"]) == f for f in sources)
    OUT.mkdir(parents=True)
    write(OUT / "contract.json", {
        "schema": "rampq.gate1.train_gradient_probe/v1", "frozen_at_utc": now(), "sources": sources,
        "member_rule": "First32 indices of the already sealed ordinary QAT epoch1 permutation; no mining or selection by result.",
        "indices_in_full_train": indices, "members": rows, "checkpoints": list(ARMS),
        "new_model_forward_images": 64, "batch_size": 4, "accumulation_window": 4,
        "measured_gradients": ["native_base_total", "native_base_classification", "weighted_aux_localization", "weighted_aux_overconfidence"],
        "objective_parity": "Reconstructed components must reproduce original frozen NativeCorrectiveLoss total exactly on the same forward prediction; scalar parity is checked each batch.",
        "checks": "Per-batch and 4-microbatch averaged gradient dot products, norms and cosine; frozen clip10 instantaneous effects. No optimizer, momentum application or checkpoint save.",
        "budget": {"train_steps": 0, "new_candidates": 0, "new_AP_jobs": 0, "timeout_seconds": 600, "retries": 0},
        "limitations": "32 train images at final checkpoints only; local first-order diagnostics, not causal attribution across the historical trajectory or proof of AP improvement.",
        "Selection_payload_access": False, "Holdout_payload_access": False,
    })
    print({"status": "FROZEN", "contract": identity(OUT / "contract.json"), "source_count": len(sources)}, flush=True)


def geometry(gram):
    g = np.asarray(gram, np.float64)
    assert g.shape == (4, 4) and np.isfinite(g).all()
    norm = np.sqrt(np.maximum(np.diag(g), 0))
    def cosine(i, j):
        den = norm[i] * norm[j]
        return float(np.clip(g[i, j] / den, -1, 1)) if den > 0 else None
    # Components 2 and 3 already include the frozen 0.25 multiplier.
    total2 = float(g[np.ix_([0, 2, 3], [0, 2, 3])].sum())
    combined = np.sqrt(max(total2, 0))
    combined_clip = min(1., 10. / (combined + 1e-6))
    base_clip = min(1., 10. / (norm[0] + 1e-6))
    return {"norms": norm.tolist(), "cosine_base_auxloc": cosine(0, 2), "cosine_base_auxerror": cosine(0, 3),
            "cosine_classification_auxloc": cosine(1, 2), "cosine_classification_auxerror": cosine(1, 3),
            "combined_norm": float(combined), "combined_clip_scale": float(combined_clip),
            "base_alone_clip_scale": float(base_clip),
            "first_order_base_projection_relative_to_base_only": float(combined_clip * (g[0, 0] + g[0, 2] + g[0, 3]) / (base_clip * g[0, 0])) if g[0, 0] > 0 else None,
            "first_order_classification_descent_dot_before_clip": float(g[1, 0] + g[1, 2] + g[1, 3])}


def run():
    contract = read(OUT / "contract.json")
    assert not (OUT / "started.json").exists(), "No repeat or automatic retry"
    assert all(identity(f["path"]) == f for f in contract["sources"])
    # Load Ultralytics with NEW writable settings before historical modules assign
    # their own legacy environment variables; no legacy config is opened for write.
    os.environ["YOLO_CONFIG_DIR"] = str(OUT / "settings")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import gate1_intervention_training_v2 as P
    import gate1_native_target_correction as N
    os.environ["YOLO_CONFIG_DIR"] = str(OUT / "settings")
    P.configure()
    c = read(OLD / "contract.json")
    assert P.verify_framework_sources() == c["framework"]
    assert torch.cuda.is_available()
    assert geometry(np.array([[1, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]], float))["cosine_base_auxloc"] == -1.
    started = time.monotonic()
    write(OUT / "started.json", {"at_utc": now(), "pid": os.getpid(), "parent_pid": os.getppid(), "GPU": torch.cuda.get_device_name(0)})

    class Components(N.NativeCorrectiveLoss):
        def components(self, preds, batch, records):
            self.native_targets = None
            assignment, base, _ = P.IgnoreAwareDetectionLoss.get_assigned_targets_and_loss(self, preds, batch)
            fg, gt_index, _, anchors, stride = assignment
            fg = fg.bool()
            scores = preds["scores"].permute(0, 2, 1).contiguous()
            assert self.native_targets is not None and self.native_targets.shape == scores.shape
            keep = N.build_negative_keep_mask((anchors * stride).detach(), fg, batch["ignore_boxes_xyxy"], tuple(batch["img"].shape[-2:]))
            boxes = self.bbox_decode(anchors, preds["boxes"].permute(0, 2, 1).contiguous()) * stride
            zero = scores.sum() * 0
            loc, errors = [], []
            for b, record in enumerate(records):
                spec = record["native_intervention_correction"]
                ids = torch.where(batch["batch_idx"] == b)[0]
                for local_gt in spec["positive_gt"]:
                    selected = fg[b] & (gt_index[b] == local_gt)
                    if not selected.any():
                        continue
                    t = batch["bboxes"][ids[local_gt]] * 640
                    truth = torch.cat((t[:2] - t[2:] / 2, t[:2] + t[2:] / 2))
                    loc.append(self.hyp.box * (1 - N.aligned_iou(boxes[b, selected], truth)).mean())
                for e in spec["errors"]:
                    anchor, cls = e["anchor"], e["category"]
                    if bool(keep[b, anchor]):
                        errors.append(self.hyp.cls * N.overconfidence_divergence(scores[b, anchor, cls], self.native_targets[b, anchor, cls]))
            loc_loss = torch.stack(loc).mean() if loc else zero
            error_loss = torch.stack(errors).mean() if errors else zero
            return (base.sum(), base[1], .25 * loc_loss, .25 * error_loss), base.sum() + .25 * (loc_loss + error_loss)

    def state_sha(model):
        h = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            a = value.detach().cpu().contiguous().numpy()
            h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.hexdigest()

    def measure(vectors):
        # Dot products in float64, only the 4x4 Gram matrix is retained.
        a = torch.stack(vectors).to(torch.float64)
        g = (a @ a.T).cpu().numpy()
        return {"gram_float64": g.tolist(), **geometry(g)}

    dataset = P.SelectedTrainingDataset(c["train_manifest"], contract["members"])
    records = [read(OLD / "attempt_01/training_supervision" / f"{i:04d}.json") for i in contract["indices_in_full_train"]]
    assert all(r["image_id"] == meta["image_id"] for r, meta in zip(records, contract["members"]))
    maxima = read(c["ranges"]["path"])["max_abs"]
    observations, forwards = [], 0
    for arm in ARMS:
        P.seed_all(c["training_seed"])
        model = P.new_model(maxima)
        P.precision(model, "r_only:G1")
        checkpoint = torch.load(OLD / "attempt_01/training" / arm / "epoch_10_fp32.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 10 and checkpoint["arm"] == arm
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.args.epochs = 10
        before = state_sha(model)
        assert before == read(OLD / "attempt_01/training" / arm / "training_receipt.json")["final_state_value_sha256"]
        model.train(); P.freeze_bn(model)
        criterion = Components(model)
        params = [p for p in model.parameters() if p.requires_grad]
        accumulated = None
        armrows = []
        for batch_index in range(8):
            assert time.monotonic() - started < 600, "Probe timeout; preserve attempt"
            ids = list(range(4 * batch_index, 4 * batch_index + 4))
            batch = P.gpu_batch([dataset[i] for i in ids])
            pred = model(batch["img"]); forwards += 4
            assert forwards <= 64
            values, combined = criterion.components(pred, batch, [records[i] for i in ids])
            reference = criterion.compute(pred, batch, [records[i] for i in ids], "native_intervention_correction")
            assert torch.equal(combined.detach(), reference.detach()), "Exact frozen-loss parity failure"
            gradients = []
            for value in values:
                gs = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                vector = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])
                assert bool(torch.isfinite(vector).all())
                gradients.append(vector)
                del gs
            item = {"checkpoint": arm, "kind": "microbatch", "batch": batch_index + 1,
                    "full_train_indices": [contract["indices_in_full_train"][i] for i in ids],
                    "loss_components": [float(v.detach()) for v in values], "exact_original_loss_parity": True,
                    "native_auxiliary_counts": criterion.last_aux, **measure(gradients)}
            armrows.append(item)
            if accumulated is None:
                accumulated = [v / 4 for v in gradients]
            else:
                for i, v in enumerate(gradients):
                    accumulated[i].add_(v / 4)
            if (batch_index + 1) % 4 == 0:
                armrows.append({"checkpoint": arm, "kind": "four_microbatch_window", "window": (batch_index + 1) // 4,
                                **measure(accumulated)})
                accumulated = None
            print({"checkpoint": arm, "batch": batch_index + 1, "forwards": forwards,
                   "cos_base_auxloc": item["cosine_base_auxloc"], "cos_cls_auxloc": item["cosine_classification_auxloc"]}, flush=True)
            del batch, pred, values, combined, reference, gradients, vector
        criterion._capture_handle.remove()
        assert state_sha(model) == before, "Probe changed state"
        assert all(p.grad is None for p in params), "Unexpected accumulated .grad"
        write(OUT / f"{arm}_observations.json", {"checkpoint_state_unchanged": True, "observations": armrows})
        observations += armrows
        del criterion, model, params, checkpoint
        torch.cuda.empty_cache()
    assert forwards == contract["new_model_forward_images"] == 64
    assert all(identity(f["path"]) == f for f in contract["sources"]), "Source changed"
    summary = {}
    for arm in ARMS:
        rows = [r for r in observations if r["checkpoint"] == arm and r["kind"] == "microbatch"]
        windows = [r for r in observations if r["checkpoint"] == arm and r["kind"] == "four_microbatch_window"]
        summary[arm] = {"batches": len(rows), "base_auxloc_negative_cosine_batches": sum(r["cosine_base_auxloc"] is not None and r["cosine_base_auxloc"] < 0 for r in rows),
                        "classification_auxloc_negative_cosine_batches": sum(r["cosine_classification_auxloc"] is not None and r["cosine_classification_auxloc"] < 0 for r in rows),
                        "classification_auxerror_negative_cosine_batches": sum(r["cosine_classification_auxerror"] is not None and r["cosine_classification_auxerror"] < 0 for r in rows),
                        "windows": windows}
    write(OUT / "result.json", {"status": "PASS_LOCAL_GRADIENT_MEASUREMENT_NOT_CAUSAL_OR_METHOD_PASS", "at_utc": now(),
                                "forward_images": forwards, "weight_updates": 0, "sources_unchanged": len(contract["sources"]),
                                "exact_original_loss_parity_batches": 16, "checkpoint_states_unchanged": True,
                                "summary": summary, "elapsed_seconds": time.monotonic() - started,
                                "Selection_payload_access": False, "Holdout_payload_access": False})
    print({"status": "COMPLETE", "forward_images": forwards, "elapsed_seconds": time.monotonic() - started}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    try:
        prepare() if args.action == "prepare" else run()
    except BaseException as error:
        if args.action == "run" and OUT.exists() and not (OUT / "failure_receipt.json").exists():
            write(OUT / "failure_receipt.json", {"at_utc": now(), "error": repr(error), "traceback": traceback.format_exc()})
        raise
