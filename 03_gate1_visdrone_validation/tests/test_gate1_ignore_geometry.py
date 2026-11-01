"""CPU-only synthetic sidecars and in-memory images; no real samples/weights."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gate1/20261023_ignore_geometry_preparation_v1"
os.environ["YOLO_CONFIG_DIR"] = str(OUT / "runtime_config")
(OUT / "runtime_config").mkdir(parents=True, exist_ok=True)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g = load("gate1_ignore_geometry")
loss_helper = load("gate1_ignore_negative_loss")


def identity(name="synthetic_a", image_digest="a", sidecar_digest="b"):
    return {"image_id": f"train/{name}", "image_sha256": image_digest * 64, "sidecar_sha256": sidecar_digest * 64}


def row(*, image_id="train/synthetic_a", line=0, xywh=(1., 2., 3., 4.), score=0, category=1, hw=(10, 20)):
    x, y, w, h = xywh
    original = [x, y, x+w, y+h]
    ih, iw = hw
    clipped = np.clip(original, [0., 0., 0., 0.], [iw, ih, iw, ih]).tolist()
    reasons = [s for flag, s in ((score == 0, "score_0"), (category == 0, "category_0"), (category == 11, "category_11")) if flag]
    return {"image_id": image_id, "target_id": f"{image_id}:{line}", "physical_line_zero_based": line,
            "score": score, "category_original": category, "publisher_ignored": score == 0 or category in {0, 11},
            "ignored": bool(reasons), "excluded_for_preparation_quality": False, "original_xywh": list(xywh),
            "original_xyxy": original, "clipped_xyxy": clipped, "ignore_reasons": reasons}


def prepared(rows=None, hw=(10, 20), ident=None, dtype=np.float64):
    return g.from_prepared_rows(identity() if ident is None else ident, hw, [row(hw=hw)] if rows is None else rows, dtype=dtype)


def letter_params(original_hw=(10, 20), final_hw=(40, 40), scaleup=True):
    h, w = original_hw
    r = min(final_hw[0]/h, final_hw[1]/w)
    if not scaleup:
        r = min(r, 1.)
    unpad = (round(w*r), round(h*r))
    dw, dh = (final_hw[1]-unpad[0])/2, (final_hw[0]-unpad[1])/2
    return {"orig_shape": original_hw, "new_shape": final_hw, "ratio": (r, r), "new_unpad": unpad,
            "top": round(dh-.1), "bottom": round(dh+.1), "left": round(dw-.1), "right": round(dw+.1)}


def transformed(sample=None):
    sample = prepared() if sample is None else sample
    sample = g.apply_loader_resize(sample, sample["canvas_hw"])
    return g.apply_letterbox_params(sample, letter_params(sample["canvas_hw"]))


@pytest.mark.parametrize("score,category,reasons", [(0, 4, ("score_0",)), (1, 0, ("category_0",)), (1, 11, ("category_11",)), (0, 0, ("score_0", "category_0")), (0, 11, ("score_0", "category_11"))])
def test_publisher_source_categories_remain_distinct(score, category, reasons):
    sample = prepared([row(score=score, category=category)])
    assert sample["records"][0]["source_reasons"] == reasons
    assert sample["records"][0]["score"] == score
    assert sample["records"][0]["category_original"] == category


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_outside_clipped_tiny_remnants_are_retained(dtype):
    sample = prepared([row(xywh=(-1, 2, 1.00001, 0.0001))], dtype=dtype)
    assert len(sample["boxes"]) == 1
    assert sample["boxes"][0, 0] == 0
    assert sample["boxes"][0, 2] > 0


@pytest.mark.parametrize("bad", [(2, 2, 3, 0), (2, 2, 0, 3), (2, 2, -1, 3), (30, 2, 2, 2)])
def test_nonpositive_or_empty_publisher_never_silently_dropped(bad):
    with pytest.raises(ValueError, match="quarantine"):
        prepared([row(xywh=bad)])


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.update(image_id="train/other"), "image identity"),
    (lambda r: r.update(target_id="train/synthetic_a:8"), "target line"),
    (lambda r: r.update(physical_line_zero_based=True), "line identity"),
    (lambda r: r.update(score=True), "score/category"),
    (lambda r: r.update(category_original=12), "score/category"),
    (lambda r: r.update(publisher_ignored=False), "publisher flag"),
    (lambda r: r.update(excluded_for_preparation_quality=True), "separate quarantine"),
    (lambda r: r.update(ignored=False), "publisher-ignore"),
    (lambda r: r.update(original_xyxy=[1., 2., 9., 6.]), "representations"),
    (lambda r: r.update(clipped_xyxy=[1., 2., 9., 6.]), "clipped geometry"),
    (lambda r: r.update(ignore_reasons=["category_0"]), "ignore reasons"),
    (lambda r: r.update(original_xywh=[1., 2., float("nan"), 3.]), "finite"),
])
def test_sidecar_corruption_fails_closed(mutate, match):
    r = row()
    mutate(r)
    with pytest.raises(ValueError, match=match):
        prepared([r])


def test_quality_exclusion_not_converted_to_publisher_ignore():
    r = row(score=1, category=3)
    r.update(excluded_for_preparation_quality=True, ignored=True, ignore_reasons=["invalid_positive_box_geometry"])
    with pytest.raises(ValueError, match="separate quarantine"):
        prepared([r])


def test_duplicate_target_identity_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        prepared([row(), row()])


@pytest.mark.parametrize("bad", ["val/synthetic", "test-dev/synthetic", "selection/a", "train/../a", "train/", "train/a:b"])
def test_identity_is_explicit_train_only(bad):
    value = identity()
    value["image_id"] = bad
    with pytest.raises(ValueError, match="train image"):
        prepared([], ident=value)


@pytest.mark.parametrize("bad", ["", "A"*64, "g"*64, "a"*63])
def test_digest_identity_malformed_rejected(bad):
    value = identity()
    value["sidecar_sha256"] = bad
    with pytest.raises(ValueError, match="SHA256"):
        prepared([], ident=value)


def test_loader_resize_uses_two_actual_axes_not_assumed_scalar():
    hw = (33, 101)
    source = prepared([row(hw=hw, xywh=(5, 7, 16, 12))], hw=hw)
    actual = g.apply_loader_resize(source, (21, 64))
    expected = source["boxes"] * np.array([64/101, 21/33, 64/101, 21/33])
    assert np.array_equal(actual["boxes"], expected)
    assert not np.array_equal(actual["boxes"], source["boxes"] * (64/101))


@pytest.mark.parametrize("hw", [(33, 101), (101, 33), (9, 11)])
def test_synthetic_installed_loader_to_letterbox_flip_crosscheck(monkeypatch, hw):
    """Exercise installed loader arithmetic with imread replaced by an array.

    There is no dataset construction/scanning and no real file decoding. The
    normalized synthetic Instances follow the ordinary positive-box path,
    while the sidecar uses original-canvas absolute coordinates.
    """
    from ultralytics.data import base as base_module
    from ultralytics.data.augment import LetterBox, RandomFlip
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.instance import Instances
    h, w = hw
    array = np.zeros((h, w, 3), dtype=np.uint8)
    calls = []

    def in_memory_imread(path, flags):
        calls.append((path, flags))
        assert path == "synthetic-memory-only"
        return array.copy()

    monkeypatch.setattr(base_module, "imread", in_memory_imread)
    obj = YOLODataset.__new__(YOLODataset)
    obj.ims, obj.im_files, obj.npy_files = [None], ["synthetic-memory-only"], [OUT / "must_not_exist.npy"]
    assert not obj.npy_files[0].exists()
    obj.imgsz, obj.cv2_flag, obj.augment = 64, 1, False
    img, ori_hw, actual_hw = base_module.BaseDataset.load_image(obj, 0)
    assert calls == [("synthetic-memory-only", 1)]
    assert ori_hw == hw
    assert actual_hw == img.shape[:2]
    source = prepared([row(hw=hw, xywh=(w/8, h/8, w/2, h/2))], hw=hw)
    actual = g.apply_loader_resize(source, actual_hw)
    normalized = source["boxes"] / np.array([w, h, w, h], dtype=np.float64)
    labels = {"img": img, "instances": Instances(normalized.copy(), bbox_format="xyxy", normalized=True)}
    letterbox = LetterBox(new_shape=(64, 64), auto=False, scale_fill=False, center=True)
    params = letterbox.get_params(labels)
    labels = letterbox.apply_image(labels, params)
    labels = letterbox.apply_instances(labels, params)
    actual = g.apply_letterbox_params(actual, params)
    for direction in ("vertical", "horizontal"):
        flip = RandomFlip(p=1., direction=direction)
        params = flip.get_params(labels)
        labels = flip.apply_image(labels, params)
        labels = flip.apply_instances(labels, params)
        actual = g.apply_flip_params(actual, params)
    labels["instances"].convert_bbox("xyxy")
    np.testing.assert_allclose(actual["boxes"], labels["instances"].bboxes, rtol=1e-12, atol=1e-12)
    assert actual["canvas_hw"] == labels["img"].shape[:2] == (64, 64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("hw,final_hw,scaleup", [((31, 67), (64, 64), True), ((67, 31), (64, 64), True), ((31, 67), (96, 128), False), ((32, 33), (65, 66), True), ((1, 5), (8, 8), True)])
def test_letterbox_matches_real_shared_get_params_apply_instances(dtype, hw, final_hw, scaleup):
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.instance import Instances
    h, w = hw
    sample = prepared([row(hw=hw, xywh=(w/8, h/8, w/2, h/2))], hw=hw, dtype=dtype)
    resized = g.apply_loader_resize(sample, hw)
    real = LetterBox(new_shape=final_hw, auto=False, scale_fill=False, center=True, scaleup=scaleup)
    labels = {"img": np.zeros((*hw, 3), dtype=np.uint8),
              "instances": Instances(resized["boxes"].copy(), bbox_format="xyxy", normalized=False)}
    params = real.get_params(labels)
    expected = real.apply_image(labels, params)
    expected = real.apply_instances(expected, params)
    actual = g.apply_letterbox_params(resized, params)
    assert expected["img"].shape[:2] == actual["canvas_hw"] == final_hw
    assert np.array_equal(expected["instances"].bboxes, actual["boxes"])


@pytest.mark.parametrize("direction", ["vertical", "horizontal"])
@pytest.mark.parametrize("flip", [False, True])
def test_flip_reuses_real_params_and_preserves_xyxy_geometry(direction, flip):
    from ultralytics.data.augment import RandomFlip
    from ultralytics.utils.instance import Instances
    sample = transformed()
    real = RandomFlip(p=1. if flip else 0., direction=direction)
    labels = {"img": np.zeros((*sample["canvas_hw"], 3), dtype=np.uint8),
              "instances": Instances(sample["boxes"].copy(), bbox_format="xyxy", normalized=False)}
    params = real.get_params(labels)
    expected = real.apply_image(labels, params)
    expected = real.apply_instances(expected, params)
    expected["instances"].convert_bbox("xyxy")
    actual = g.apply_flip_params(sample, params)
    assert np.array_equal(actual["boxes"], expected["instances"].bboxes)


@pytest.mark.parametrize("changes,match", [({"ratio": (2., 3.)}, "stretch"), ({"top": 0}, "padding"), ({"orig_shape": (11, 20)}, "canvas mismatch"), ({"new_unpad": (40, 19)}, "rounding"), ({"ratio": (3., 3.)}, "standard LetterBox")])
def test_letterbox_invalid_contract_rejected(changes, match):
    params = letter_params()
    params.update(changes)
    with pytest.raises(ValueError, match=match):
        g.apply_letterbox_params(g.apply_loader_resize(prepared(), (10, 20)), params)


def test_transform_order_and_normalized_flip_rejected():
    with pytest.raises(ValueError, match="follow actual"):
        g.apply_letterbox_params(prepared(), letter_params())
    sample = transformed()
    with pytest.raises(ValueError, match="first and only"):
        g.apply_loader_resize(sample, (40, 40))
    with pytest.raises(ValueError, match="normalized"):
        g.apply_flip_params(sample, {"direction": "horizontal", "flip": True, "h": 1, "w": 1})


def test_no_random_sampling_input_mutation_or_box_dropping():
    original = prepared([row(line=0), row(line=1, xywh=(-1, 0, 2, 1))])
    before = copy.deepcopy(original)
    final = transformed(original)
    params = {"flip": True, "direction": "horizontal", "h": 40, "w": 40, "flip_idx": None}
    final = g.apply_flip_params(final, params)
    assert np.array_equal(original["boxes"], before["boxes"])
    assert original["trace"] == ()
    assert original["records"] == before["records"] == final["records"]
    assert len(final["boxes"]) == 2
    with pytest.raises(ValueError, match="duplicate flip"):
        g.apply_flip_params(final, params)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ragged_collation_reordered_batch_stays_identified_and_loss_compatible(dtype):
    a = transformed()
    b = transformed(prepared([], ident=identity("synthetic_b", "c", "d")))
    batch = g.collate_ignores([b, a], [b["identity"], a["identity"]], (40, 40), torch_dtype=dtype)
    assert batch["batch_indices"] == (0, 1)
    assert batch["ignore_boxes_xyxy"][0].shape == (0, 4)
    assert batch["ignore_boxes_xyxy"][1].shape == (1, 4)
    # Actual loss primitive verifies half-open anchor boundaries and positive priority.
    # Source row maps from [1,2,4,6] to [2,14,8,22].
    centers = torch.tensor([[2., 14.], [8., 14.], [4., 18.]], dtype=dtype)
    fg = torch.tensor([[False, False, False], [False, False, True]])
    keep = loss_helper.build_negative_keep_mask(centers, fg, batch["ignore_boxes_xyxy"], batch["image_hw"])
    assert torch.equal(keep, torch.tensor([[True, True, True], [False, True, True]]))


@pytest.mark.parametrize("mutation", ["order", "image_hash", "sidecar_hash", "canvas", "untransformed"])
def test_batch_identity_or_canvas_corruption_rejected(mutation):
    a = transformed()
    b = transformed(prepared([], ident=identity("synthetic_b", "c", "d")))
    samples, expected, hw = [a, b], [dict(a["identity"]), dict(b["identity"])], (40, 40)
    if mutation == "order":
        expected.reverse()
    elif mutation == "image_hash":
        expected[0]["image_sha256"] = "e"*64
    elif mutation == "sidecar_hash":
        expected[0]["sidecar_sha256"] = "e"*64
    elif mutation == "canvas":
        hw = (32, 32)
    else:
        samples[0] = prepared()
        hw = (10, 20)
    with pytest.raises(ValueError):
        g.collate_ignores(samples, expected, hw)


def test_torch_half_and_geometry_underflow_rejected():
    with pytest.raises(ValueError, match="float32/float64"):
        g.collate_ignores([transformed()], [identity()], (40, 40), torch_dtype=torch.float16)
    with pytest.raises(ValueError, match="float32/float64"):
        prepared(dtype=np.float16)
    tiny = prepared([row(xywh=(0., 0., 1e-50, 1e-50), hw=(40, 40))], hw=(40, 40))
    tiny = transformed(tiny)
    with pytest.raises(ValueError, match="cast produced"):
        g.collate_ignores([tiny], [identity()], (40, 40), torch_dtype=torch.float32)


def test_existing_loss_source_unchanged():
    assert hashlib.sha256((ROOT / "scripts/gate1_ignore_negative_loss.py").read_bytes()).hexdigest() == "67fa0f7a171d26b1f8d2ebd922333f30bb97ffb94b523a58f6f784ed572b3c1f"


def test_explicit_chain_options_accept_only_complete_reviewed_contract():
    assert g.validate_restricted_options(dict(g.RESTRICTED_OPTIONS)) == g.RESTRICTED_OPTIONS
    with pytest.raises(ValueError, match="exact explicit-chain"):
        g.validate_restricted_options({"mosaic": 0.})


@pytest.mark.parametrize("key", list(g.RESTRICTED_OPTIONS))
def test_nonzero_or_unreviewed_spatial_options_rejected(key):
    options = dict(g.RESTRICTED_OPTIONS)
    options[key] = not options[key] if type(options[key]) is bool else .5
    with pytest.raises(ValueError, match="unsupported"):
        g.validate_restricted_options(options)
