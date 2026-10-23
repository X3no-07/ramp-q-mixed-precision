"""Synthetic-only CPU checks of the installed model/loss/Trainer integration."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate1_ignore_trainer as integration
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss


TINY_CFG = {
    "nc": 3, "depth_multiple": 1.0, "width_multiple": 1.0,
    "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [16, 3, 2]],
                 [-1, 1, "Conv", [24, 3, 2]]],
    "head": [[[2], 1, "Detect", ["nc"]]],
}


@pytest.fixture(autouse=True)
def cpu_single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model(weighted=False):
    torch.manual_seed(106)
    m = integration.IgnoreAwareDetectionModel(deepcopy(TINY_CFG), ch=3, nc=3, verbose=False).float().train()
    m.args = deepcopy(DEFAULT_CFG)
    if weighted:
        m.class_weights = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
    return m


def batch(with_gt=True, *, full_ignore=False):
    image = torch.linspace(0, 1, 2 * 3 * 64 * 64).reshape(2, 3, 64, 64).requires_grad_()
    return {
        "img": image,
        "batch_idx": torch.tensor([0., 1.]) if with_gt else torch.empty((0,)),
        "cls": torch.tensor([[0.], [2.]]) if with_gt else torch.empty((0, 1)),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]) if with_gt else torch.empty((0, 4)),
        "ignore_boxes_xyxy": [torch.tensor([[0., 0., 64., 64.]]) if full_ignore else torch.empty((0, 4)) for _ in range(2)],
        "image_identities": tuple({"image_id": f"train/synthetic_{i}", "image_sha256": "0" * 64,
                                    "sidecar_sha256": "1" * 64} for i in range(2)),
    }


def assert_all_gradients_equal(left, right):
    for (ln, lp), (rn, rp) in zip(left.named_parameters(), right.named_parameters()):
        assert ln == rn
        assert (lp.grad is None) == (rp.grad is None), ln
        if lp.grad is not None:
            assert torch.equal(lp.grad, rp.grad), ln


def test_pinned_framework_sources_are_current():
    assert {k: v["sha256"] for k, v in integration.verify_framework_sources().items()} == integration.FRAMEWORK_PINS


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("with_gt", [False, True])
def test_actual_random_model_empty_ignore_loss_and_all_gradients_bitexact(weighted, with_gt):
    original, changed = model(weighted), model(weighted)
    old_batch, new_batch = batch(with_gt), batch(with_gt)
    old_prediction, new_prediction = original(old_batch["img"]), changed(new_batch["img"])
    old_criterion, new_criterion = v8DetectionLoss(original), integration.IgnoreAwareDetectionLoss(changed)
    old_loss, old_parts = old_criterion(old_prediction, old_batch)
    new_loss, new_parts = new_criterion(new_prediction, new_batch)
    assert torch.equal(old_loss, new_loss)
    assert tuple(old_parts) == tuple(new_parts)
    assert all(torch.equal(old_parts[k], new_parts[k]) for k in old_parts)
    old_loss.sum().backward()
    new_loss.sum().backward()
    assert_all_gradients_equal(original, changed)
    assert torch.equal(old_batch["img"].grad, new_batch["img"].grad)
    assert new_criterion.last_mask_receipt["ignored_negative_anchors"] == 0
    assert new_criterion.last_mask_receipt["positive_rows_preserved"] is True


def test_actual_tal_positive_rows_preserved_and_box_dfl_unchanged():
    ordinary_model, masked_model = model(True), model(True)
    ordinary_batch, masked_batch = batch(), batch(full_ignore=True)
    old_prediction, new_prediction = ordinary_model(ordinary_batch["img"]), masked_model(masked_batch["img"])
    old_prediction["scores"].retain_grad()
    new_prediction["scores"].retain_grad()
    old = v8DetectionLoss(ordinary_model)
    new = integration.IgnoreAwareDetectionLoss(masked_model)
    old_assignment, old_parts, _ = old.get_assigned_targets_and_loss(old_prediction, ordinary_batch)
    new_assignment, new_parts, _ = new.get_assigned_targets_and_loss(new_prediction, masked_batch)
    assert all(torch.equal(a, b) for a, b in zip(old_assignment, new_assignment))
    fg = old_assignment[0]
    assert bool(fg.any()) and bool((~fg).any())
    assert torch.equal(old_parts[[0, 2]], new_parts[[0, 2]])
    assert 0 < new_parts[1] < old_parts[1]
    old_parts.sum().backward()
    new_parts.sum().backward()
    old_score_grads = old_prediction["scores"].grad.permute(0, 2, 1)
    new_score_grads = new_prediction["scores"].grad.permute(0, 2, 1)
    assert torch.equal(old_score_grads[fg], new_score_grads[fg])
    assert torch.equal(new_score_grads[~fg], torch.zeros_like(new_score_grads[~fg]))
    assert new.last_mask_receipt["assigned_positive_anchors"] > 0
    assert new.last_mask_receipt["ignored_negative_anchors"] > 0
    assert new.last_mask_receipt["positive_rows_preserved"] is True


def test_all_ignore_empty_gt_zero_loss_and_score_gradient():
    m, b = model(), batch(False, full_ignore=True)
    preds = m(b["img"])
    preds["scores"].retain_grad()
    loss, _ = integration.IgnoreAwareDetectionLoss(m)(preds, b)
    assert torch.equal(loss, torch.zeros_like(loss))
    loss.sum().backward()
    assert torch.equal(preds["scores"].grad, torch.zeros_like(preds["scores"].grad))


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b.pop("ignore_boxes_xyxy"), "explicit ragged"),
    (lambda b: b.pop("image_identities"), "explicit ragged"),
    (lambda b: b.update(image_identities=()), "one identity"),
    (lambda b: b.update(ignore_boxes_xyxy=[torch.empty(0, 4)]), "one ragged"),
    (lambda b: b.update(ignore_boxes_xyxy=[torch.tensor([[0., 0., 70., 64.]])] * 2), "clipped"),
])
def test_malformed_sidecar_batch_fails_closed(mutation, match):
    m, b = model(), batch()
    preds = m(b["img"])
    mutation(b)
    with pytest.raises(ValueError, match=match):
        integration.IgnoreAwareDetectionLoss(m)(preds, b)


@pytest.mark.parametrize("weights", [torch.ones(4), torch.tensor([1., -1., 1.]),
                                     torch.tensor([1., float("nan"), 1.]), torch.ones(3, dtype=torch.float64),
                                     torch.ones(3, requires_grad=True)])
def test_invalid_class_weights_rejected(weights):
    m = model()
    m.class_weights = weights
    with pytest.raises(ValueError, match="class weights"):
        integration.IgnoreAwareDetectionLoss(m)


def test_half_model_rejected_and_model_factory_selects_custom_criterion():
    m = model()
    assert isinstance(m.init_criterion(), integration.IgnoreAwareDetectionLoss)
    with pytest.raises(ValueError, match="FP32 model"):
        integration.IgnoreAwareDetectionLoss(m.half())


@pytest.mark.parametrize("key,value", [("amp", True), ("device", "0"), ("epochs", 2), ("resume", True),
    ("pretrained", True), ("mosaic", 1.0), ("multi_scale", 0.5), ("scale", 0.1),
    ("degrees", 1.0), ("translate", 0.1), ("augmentations", []), ("compile", True),
    ("val", True), ("profile", True), ("classes", [0]), ("fraction", 0.5),
    ("model", "yolov8n.pt"), ("workers", 2), ("batch", 0.5), ("nbs", 64),
    ("imgsz", 63), ("fliplr", 1.1), ("cls_pw", -0.1), ("unknown_knob", 1)])
def test_restricted_training_options_reject_drift(key, value):
    with pytest.raises(ValueError):
        integration.validate_synthetic_overrides({key: value})


def test_explicit_safe_options_and_flips_allowed():
    args = integration.validate_synthetic_overrides({"fliplr": 1.0, "flipud": 1.0, "project": "synthetic"})
    assert args["amp"] is False and args["multi_scale"] == 0
    assert args["fliplr"] == args["flipud"] == 1.0
    assert args["batch"] == args["nbs"] == 2


def contract(tmp_path):
    entries = {}
    for role in ("train", "training_val"):
        p = tmp_path / f"{role}.json"
        p.write_text(json.dumps({"schema": "rampq.gate1.ignore_dataset_manifest/v1", "data_role": "synthetic_only"}), encoding="utf-8")
        entries[role] = {"manifest_path": str(p.resolve()), "manifest_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return {"schema": "rampq.gate1.synthetic_ignore_trainer_contract/v1", "synthetic_only": True,
            "class_count": 3, "names": ["a", "b", "c"], "datasets": entries,
            "restricted_options": deepcopy(integration.RESTRICTED_OPTIONS)}


def test_contract_scope_and_bound_hashes_fail_closed(tmp_path):
    c = contract(tmp_path)
    assert integration._validate_contract(c) == c
    c["synthetic_only"] = False
    with pytest.raises(ValueError, match="synthetic"):
        integration._validate_contract(c)
    c["synthetic_only"] = True
    Path(c["datasets"]["train"]["manifest_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        integration._validate_contract(c)


def fake_trainer():
    trainer = integration.RestrictedIgnoreTrainer.__new__(integration.RestrictedIgnoreTrainer)
    trainer.args = SimpleNamespace(multi_scale=0.0, amp=False)
    trainer.device = torch.device("cpu")
    trainer.integration_events = []
    return trainer


def test_native_preprocess_normalizes_and_moves_ragged_sidecars():
    t = fake_trainer()
    b = batch()
    b["img"] = torch.full((2, 3, 64, 64), 255, dtype=torch.uint8)
    out = t.preprocess_batch(b)
    assert out["img"].dtype == torch.float32 and bool((out["img"] == 1).all())
    assert len(out["ignore_boxes_xyxy"]) == 2
    assert all(x.dtype == torch.float32 and x.device.type == "cpu" for x in out["ignore_boxes_xyxy"])
    assert out["image_identities"] == b["image_identities"]


def test_preprocess_rejects_geometry_dtype_drift_before_cast_can_hide_it():
    t = fake_trainer()
    b = batch()
    b["img"] = torch.zeros((2, 3, 64, 64), dtype=torch.uint8)
    b["ignore_boxes_xyxy"] = [torch.empty((0, 4), dtype=torch.float16)] * 2
    with pytest.raises(ValueError, match="detached FP32"):
        t.preprocess_batch(b)


def test_synthetic_validator_and_final_eval_explicitly_have_no_ap():
    t = fake_trainer()
    assert t.get_validator().metrics.keys == []
    assert t.validate() == ({}, 0.0)
    t.final_eval()
    assert len(t.integration_events) == 3
    assert all(e["status"] == "SMOKE_NO_TASK_AP" for e in t.integration_events)


def test_save_callback_is_before_framework_half_serialization(monkeypatch):
    t = fake_trainer()
    events = []
    t.epoch = 0
    t._handle_nan_recovery = lambda epoch: False
    t.fp32_capture_callback = lambda trainer: events.append("capture")
    monkeypatch.setattr(integration.DetectionTrainer, "save_model", lambda self: events.append("framework") or True)
    assert t.save_model() is True
    assert events == ["capture", "framework"]


def test_nan_recovery_fails_instead_of_loading_any_checkpoint():
    t = fake_trainer()
    t.loss = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="recovery forbidden"):
        t._handle_nan_recovery(0)


def test_oom_retry_cannot_rebuild_or_reduce_batch():
    t = fake_trainer()
    t._oom_retries = 1
    with pytest.raises(ValueError, match="reduction is forbidden"):
        t._build_train_pipeline()


def test_get_model_rejects_checkpoint_and_setup_uses_only_bound_dict():
    t = fake_trainer()
    t.data = {"nc": 3}
    t.synthetic_model_cfg = deepcopy(TINY_CFG)
    t.args.verbose = False
    assert t.setup_model() is None
    assert isinstance(t.model, integration.IgnoreAwareDetectionModel)
    assert t.resume is False
    with pytest.raises(ValueError, match="weight loading"):
        t.get_model(TINY_CFG, "old.pt", False)
    with pytest.raises(ValueError, match="dictionary"):
        t.get_model("any.yaml", None, False)

