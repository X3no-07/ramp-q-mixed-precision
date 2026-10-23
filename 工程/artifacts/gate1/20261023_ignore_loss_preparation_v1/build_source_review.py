"""Read source/contract metadata only; write a bounded synthetic preparation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PKG = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "ultralytics"
CONTRACT = ROOT / "artifacts/gate1/20261022_split_refinement/training_contract_v3"


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write(name: str, obj: object) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    require_new = [OUT / name for name in ("source_gap_review.json", "synthetic_counterexample.json", "sources")]
    if any(path.exists() for path in require_new):
        raise RuntimeError("New preparation outputs already exist; do not overwrite frozen evidence")
    source_dir = OUT / "sources"
    source_dir.mkdir()
    helper_path = ROOT / "scripts/gate1_ignore_negative_loss.py"
    spec = importlib.util.spec_from_file_location("ignore_loss_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sources = {}
    ranges = {
        "utils/loss.py": [(337, 373), (402, 467)],
        "utils/tal.py": [(147, 175), (271, 309)],
        "data/dataset.py": [(274, 333), (376, 438)],
        "data/augment.py": [(754, 771), (838, 851), (1203, 1235), (1237, 1269), (1573, 1593), (1737, 1776), (2812, 2854)],
        "nn/tasks.py": [],
    }
    contract = json.loads((CONTRACT / "contract.json").read_text(encoding="utf-8"))
    pins = contract["environment_fingerprint"]["framework_source_sha256"]
    for name, spans in ranges.items():
        original = PKG / name
        snapshot = source_dir / name.replace("/", "__")
        shutil.copyfile(original, snapshot)
        lines = original.read_text(encoding="utf-8").splitlines()
        pin = pins.get(name)
        sources[name] = {"original_path": str(original), "snapshot": snapshot.relative_to(OUT).as_posix(),
                         **identity(snapshot), "historical_training_contract_sha256": pin,
                         "historical_pin_matches_current": identity(snapshot)["sha256"] == pin if pin else None,
                         "code_evidence": [{"line_start": a, "line_end": b,
                                            "lines": [{"line": n, "text": lines[n - 1]} for n in range(a, b + 1)]}
                                           for a, b in spans]}
    # Deliberately do not load a model, dataset, image, real annotation or sidecar.
    logits = torch.zeros((1, 3, 2), dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[0., 0.], [1., 0.], [0., 0.]]], dtype=torch.float64)
    fg = torch.tensor([[False, True, False]])
    anchors = torch.tensor([[1., 1.], [2., 2.], [8., 8.]], dtype=torch.float64)
    ignored = [torch.tensor([[0., 0., 4., 4.]], dtype=torch.float64)]
    baseline = F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum() / target.sum().clamp_min(1)
    result = helper.masked_classification_bce(logits, target, fg, anchors, ignored, (10, 10))
    baseline_grad = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
    masked_grad = torch.autograd.grad(result["loss"], logits)[0]
    receipt = {
        "schema": "rampq.gate1.ignore_negative_synthetic_counterexample/v1",
        "status": "PASS_SYNTHETIC_FIXED_LOGIT_GRADIENT_PROPERTIES_ONLY",
        "device": "cpu", "torch": torch.__version__, "python": sys.version,
        "fixture": {"invented_data_only": True, "image_hw": [10, 10], "logits": logits.detach().tolist(),
                    "target_scores": target.tolist(), "fg_mask": fg.tolist(), "anchor_centers_xy": anchors.tolist(),
                    "ignore_boxes_xyxy": [x.tolist() for x in ignored]},
        "ordinary_loss": baseline.item(), "masked_loss": result["loss"].item(),
        "denominator": result["denominator"].item(), "ordinary_gradient": baseline_grad.tolist(),
        "masked_gradient": masked_grad.tolist(), "keep_mask": result["keep_mask"].tolist(),
        "ignored_negative_gradient_zero": bool((masked_grad[:, 0] == 0).all()),
        "positive_all_class_gradient_bitexact": torch.equal(masked_grad[:, 1], baseline_grad[:, 1]),
        "retained_background_gradient_bitexact": torch.equal(masked_grad[:, 2], baseline_grad[:, 2]),
        "real_data_read": False, "weights_loaded": False, "gpu_used": False,
        "training_run": False, "training_pipeline_integrated": False,
        "limits": ["Only a fixed-logit classification primitive, not a trained-network result.",
                   "Equal retained-logit gradients do not imply equal shared-parameter gradients or future training trajectories."]}
    assert all(receipt[k] for k in ("ignored_negative_gradient_zero", "positive_all_class_gradient_bitexact", "retained_background_gradient_bitexact"))
    write("synthetic_counterexample.json", receipt)
    write("source_gap_review.json", {
        "schema": "rampq.gate1.ignore_negative_loss_source_gap_review/v1",
        "status": "GAP_CONFIRMED_CURRENT_SOURCE_AND_FROZEN_CONTRACT_PRIMITIVE_ONLY",
        "sources": sources,
        "workspace_sources": {str(p.relative_to(ROOT).as_posix()): identity(p) for p in [
            CONTRACT / "contract.json", CONTRACT / "runner_source.py", CONTRACT / "framework_defaults.json",
            ROOT / "scripts/prepare_gate1_visdrone.py", helper_path,
            ROOT / "tests/test_gate1_ignore_negative_loss.py", Path(__file__)]},
        "evidence_chain": [
            {"source": "frozen training contract limits", "finding": "Explicitly acknowledges ordinary YOLO loss does not mask ignored negative regions."},
            {"source": "prepare_gate1_visdrone.py:161-204,255-271", "finding": "Publisher score=0/category 0 or 11 rows are excluded from YOLO positives; ignored rows also remain in sidecars. Preparation-quality exclusions are distinguished."},
            {"source": "frozen runner_source.py:308-332", "finding": "Only ordinary image/YOLO label copies are exposed in data.yaml; no ignore geometry is provided to criterion."},
            {"source": "utils/tal.py:300-307", "finding": "Unassigned anchor target scores are zero; zero BCE targets still incur background loss."},
            {"source": "utils/loss.py:411,426-441", "finding": "Assign from regular GT, calculate BCE for all anchors/classes, optionally apply class weights, sum every entry with original target-score denominator."},
            {"source": "utils/loss.py:444-455", "finding": "Box and DFL branches operate on fg_mask positives; the proposed primitive never changes them."},
            {"source": "data/augment.py:759-771,1203-1235,1573-1593;data/dataset.py:409-438", "finding": "Ordinary transform/collation supports specific fields; arbitrary ignore sidecars are not automatically transformed, merged or indexed."}],
        "historical_pin_limit": "loss.py and tal.py were not separately pinned in training_contract_v3. This review pins their current installed bytes only. Dataset/augment/tasks pins are checked; historical contract explicitly acknowledges the same gap, but no retrospective byte identity is claimed for unpinned files.",
        "proposed_policy": helper.POLICY,
        "ignore_source_policy_proposed_not_loaded": {
            "include": "Nonempty clipped publisher_ignored regions (score=0 or category in {0,11}) for authorized train rows only.",
            "exclude": "Do not silently convert preparation-quality invalid positive geometry into publisher-ignore regions; fail/quarantine through a separately reviewed policy.",
            "class_semantics": "All such regions would suppress every class at unassigned anchors. This deliberately stronger class-agnostic policy is an engineering proposal, not official class-specific score=0 matching semantics.",
            "selection_holdout": "Neither may enter training loss preparation or tuning."},
        "augmentation_integration_required": [
            "Carry ignore geometry and source identities separately from positive Instances; never add ignored boxes to TAL GT.",
            "Use the exact sampled resize/pad/crop/mosaic/flip parameters applied to the image, not an independently sampled second transform.",
            "Mosaic must merge every contributing source ignore set, map placement/crop, and clip to final canvas; arbitrary dict fields are currently dropped by final_labels reconstruction.",
            "Use a loss-specific clipped region validity rule, not positive box_candidates area/aspect filtering; dropping tiny ignore remnants silently reintroduces negative gradients.",
            "Current frozen defaults have degrees/shear/perspective=0, mixup/cutmix/copy_paste=0. A first integration must assert these settings or implement separately reviewed polygon/paste semantics. AABB after rotation can ignore extra background.",
            "Photometric transforms need no geometry change, but any spatial Albumentations must expose/replay geometry or be rejected.",
            "Format and collate must preserve ragged per-image ignore boxes and explicit batch identity; criterion receives final-canvas pixel coordinates after multi-scale preprocessing, if enabled.",
            "Unit tests of this primitive do not verify these unimplemented data-flow requirements."],
        "normalization_and_noninterference": {
            "keep": "fg_mask OR NOT anchor_center_inside_union_ignore",
            "numerator": "sum(where(keep[...,None], ordinary_elementwise_BCE_after_optional_class_weights, 0))",
            "denominator": "target_scores.sum().clamp_min(1), identical original positive normalization",
            "unchanged_at_fixed_inputs": ["TAL assignment", "positive all-class BCE terms and gradients", "retained negative BCE terms and gradients", "box/DFL inputs and normalization"],
            "not_claimed": "Shared-parameter gradients and future training trajectories necessarily can change after removing negative terms; no training efficacy or official semantics is established."},
        "data_access": {"source_code_and_contract_metadata_only": True, "real_images_or_labels_parsed": False,
                        "real_ignored_rows_parsed": False, "selection_accessed": False, "holdout_accessed": False,
                        "models_or_weights_loaded": False, "gpu_used": False, "training_started": False},
        "complete_ignore_aware_training_solved": False, "official_ap_implemented_or_certified": False,
        "original_scientific_gates_changed": False})


if __name__ == "__main__":
    main()
