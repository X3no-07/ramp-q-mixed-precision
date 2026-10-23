"""CPU-only invented tensors; never read real images, labels or weights."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "gate1_ignore_negative_loss.py"
spec = importlib.util.spec_from_file_location("gate1_ignore_negative_loss", SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def fixture():
    logits = torch.tensor([[[0.1, -1.2], [2.1, -0.7], [-0.2, 0.4]]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[[0.0, 0.0], [0.8, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    fg = torch.tensor([[False, True, False]])
    centers = torch.tensor([[1.0, 1.0], [2.0, 2.0], [8.0, 8.0]], dtype=torch.float64)
    boxes = [torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float64)]
    return logits, targets, fg, centers, boxes, (10, 10)


def run(args, **kwargs):
    return helper.masked_classification_bce(*args, **kwargs)


def ordinary(logits, targets, weights=None):
    elements = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weights is not None:
        elements = elements * weights
    return elements.sum() / targets.sum().clamp_min(1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_no_ignore_value_and_gradient_bitexact(dtype, weighted):
    args = list(fixture())
    args[0] = args[0].detach().to(dtype).requires_grad_()
    args[1] = args[1].to(dtype)
    args[4] = [torch.empty((0, 4), dtype=dtype)]
    weights = torch.tensor([1.3, 0.7], dtype=dtype) if weighted else None
    actual = run(args, class_weights=weights)["loss"]
    expected = ordinary(args[0], args[1], weights)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, args[0], retain_graph=True)[0],
                       torch.autograd.grad(expected, args[0])[0])


def test_ignored_negative_gradients_zero_positive_entire_row_unchanged():
    args = fixture()
    actual = run(args)
    expected = ordinary(args[0], args[1])
    actual_grad = torch.autograd.grad(actual["loss"], args[0], retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, args[0])[0]
    assert actual["keep_mask"].tolist() == [[False, True, True]]
    assert torch.equal(actual_grad[:, 0], torch.zeros_like(actual_grad[:, 0]))
    assert torch.equal(actual_grad[:, 1:], expected_grad[:, 1:])
    assert actual_grad[0, 1, 1] != 0  # negative class at a positive anchor survives
    assert torch.equal(actual["denominator"], args[1].sum().clamp_min(1))
    assert expected_grad[0, 0, 0] > 0


def test_all_ignored_no_positive_denominator_one_zero_loss_zero_gradient():
    logits = torch.zeros((1, 3, 2), dtype=torch.float64, requires_grad=True)
    result = run((logits, torch.zeros_like(logits), torch.zeros((1, 3), dtype=torch.bool),
                  torch.tensor([[1., 1.], [2., 2.], [8., 8.]]),
                  [torch.tensor([[0., 0., 10., 10.]])], (10, 10)))
    assert result["denominator"].item() == 1
    assert result["loss"].item() == 0
    assert torch.equal(torch.autograd.grad(result["loss"], logits)[0], torch.zeros_like(logits))


def test_ragged_batch_isolation_and_union():
    centers = torch.tensor([[1., 1.], [3., 3.], [5., 5.], [8., 8.]])
    fg = torch.tensor([[False, False, False, False], [False, True, False, False], [False] * 4])
    boxes = [torch.empty((0, 4)), torch.tensor([[0., 0., 4., 4.], [2., 2., 6., 6.]]),
             torch.tensor([[7., 7., 10., 10.]])]
    result = helper.build_negative_keep_mask(centers, fg, boxes, (10, 10))
    assert result.tolist() == [[True] * 4, [False, True, False, True], [True, True, True, False]]


def test_half_open_geometry_left_top_included_right_bottom_excluded():
    centers = torch.tensor([[0., 0.], [1., 1.], [2., 1.], [1., 2.], [2., 2.]])
    result = helper.build_negative_keep_mask(centers, torch.zeros((1, 5), dtype=torch.bool),
                                             [torch.tensor([[0., 0., 2., 2.]])], (3, 3))
    assert result.tolist() == [[False, False, True, True, True]]


def test_no_inputs_modified():
    args = fixture()
    weights = torch.tensor([1.3, .7], dtype=torch.float64)
    originals = [value.detach().clone() for value in args[:4]] + [args[4][0].clone(), weights.clone()]
    run(args, class_weights=weights)["loss"].backward()
    current = list(args[:4]) + [args[4][0], weights]
    assert all(torch.equal(a, b) for a, b in zip(originals, current))


@pytest.mark.parametrize("box", [
    [0., 0., 0., 1.], [2., 0., 1., 1.], [0., 2., 1., 1.],
    [-1., 0., 1., 1.], [0., 0., 11., 1.], [0., 0., 1., 11.],
    [0., 0., float("nan"), 1.], [0., 0., float("inf"), 1.],
])
def test_invalid_geometry_rejected(box):
    args = list(fixture())
    args[4] = [torch.tensor([box], dtype=torch.float64)]
    with pytest.raises(ValueError):
        run(args)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("which", [0, 1, 3])
def test_nonfinite_logits_targets_and_centers_rejected(bad, which):
    args = list(fixture())
    value = args[which].detach().clone()
    value.flatten()[0] = bad
    args[which] = value
    with pytest.raises(ValueError):
        run(args)


@pytest.mark.parametrize("kind", ["ragged_count", "box_shape", "fg_dtype", "fg_shape", "target_shape",
                                  "target_nonzero_negative", "target_above_one", "image_bool", "image_zero",
                                  "geometry_grad", "target_grad", "anchor_outside", "empty_anchors"])
def test_contract_errors_rejected(kind):
    args = list(fixture())
    if kind == "ragged_count": args[4] = []
    elif kind == "box_shape": args[4] = [torch.zeros((1, 3))]
    elif kind == "fg_dtype": args[2] = args[2].float()
    elif kind == "fg_shape": args[2] = torch.zeros((1, 2), dtype=torch.bool)
    elif kind == "target_shape": args[1] = torch.zeros((1, 2, 2), dtype=torch.float64)
    elif kind == "target_nonzero_negative": args[1][0, 0, 0] = .1
    elif kind == "target_above_one": args[1][0, 1, 0] = 1.1
    elif kind == "image_bool": args[5] = (True, 10)
    elif kind == "image_zero": args[5] = (0, 10)
    elif kind == "geometry_grad": args[4][0].requires_grad_()
    elif kind == "target_grad": args[1].requires_grad_()
    elif kind == "anchor_outside": args[3][0, 0] = 10
    elif kind == "empty_anchors": args[3] = torch.empty((0, 2))
    with pytest.raises(ValueError):
        run(args)


@pytest.mark.parametrize("weights", [torch.tensor([-1., 1.]), torch.tensor([float("nan"), 1.]),
                                      torch.ones(3), torch.ones(2, requires_grad=True)])
def test_bad_class_weights_rejected(weights):
    with pytest.raises(ValueError):
        run(fixture(), class_weights=weights.to(torch.float64))


def test_retained_gradient_with_large_original_denominator_and_class_weights():
    args = list(fixture())
    args[0] = args[0].expand(2, -1, -1).clone().detach().requires_grad_()
    args[1] = args[1].expand(2, -1, -1).clone()
    args[2] = args[2].expand(2, -1).clone()
    args[4] = args[4] * 2
    weights = torch.tensor([[[.2, 1.7]]], dtype=torch.float64)
    result = run(args, class_weights=weights)
    expected = ordinary(args[0], args[1], weights)
    assert result["denominator"].item() == 1.6
    g = torch.autograd.grad(result["loss"], args[0], retain_graph=True)[0]
    base_g = torch.autograd.grad(expected, args[0])[0]
    assert torch.equal(g[:, 1:], base_g[:, 1:])
    assert torch.equal(g[:, :1], torch.zeros_like(g[:, :1]))
