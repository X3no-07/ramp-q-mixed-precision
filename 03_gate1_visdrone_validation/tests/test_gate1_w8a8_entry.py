"""Meaningful scalar/tiny-Conv checks; no real data or detector execution."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate1_w8a8_entry as E


def scalar_conv(weight=1.0):
    conv = nn.Conv2d(1, 1, 1, bias=False).eval()
    with torch.no_grad():
        conv.weight.fill_(weight)
    return conv


def test_round_ties_even_and_signed_narrow_range_saturation():
    wrapper = E.make_wrapper(scalar_conv(), "model.0.conv", 15.875)
    wrapper.enabled = True
    units = torch.tensor([-300., -128., -127.5, -127., -2.5, -1.5, -.5, 0., .5, 1.5, 2.5, 127., 128., 300.])
    expected_codes = torch.tensor([-127., -127., -127., -127., -2., -2., 0., 0., 0., 2., 2., 127., 127., 127.])
    observed = wrapper((units * .125).reshape(1, 1, 1, -1)).reshape(-1)
    torch.testing.assert_close(observed, expected_codes * .125, rtol=0, atol=0)


def test_weight_per_output_channel_scale_axis_and_bias_float():
    conv = nn.Conv2d(2, 2, 1, bias=True).eval()
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([127., 63.5, 1.27, .635]).reshape(2, 2, 1, 1))
        conv.bias.copy_(torch.tensor([.1234567, -.7654321]))
    wrapper = E.make_wrapper(conv, "model.3.conv", 127.)
    expected = torch.tensor([127., 64., 1.27, .64]).reshape(2, 2, 1, 1)
    torch.testing.assert_close(wrapper.quantized_weight, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(wrapper.original.bias, conv.bias)


def test_zero_ranges_are_finite_and_do_not_introduce_signal():
    wrapper = E.make_wrapper(scalar_conv(0.), "model.0.conv", 0.)
    wrapper.enabled = True
    assert wrapper.activation_scale.item() > 0
    assert torch.isfinite(wrapper.quantized_weight).all()
    assert torch.equal(wrapper(torch.ones(1, 1, 2, 2)), torch.zeros(1, 1, 2, 2))


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_activation_maximum_rejected(value):
    with pytest.raises(ValueError):
        E.make_wrapper(scalar_conv(), "model.0.conv", value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_weight_rejected(value):
    with pytest.raises(ValueError):
        E.make_wrapper(scalar_conv(value), "model.0.conv", 1.)


def test_non_FP32_weight_rejected():
    with pytest.raises(ValueError):
        E.make_wrapper(scalar_conv().half(), "model.0.conv", 1.)


@pytest.mark.parametrize("name,group", [("model.0.conv", "G1"), ("model.2.cv1.conv", "G1"),
    ("model.3.conv", "G2"), ("model.5.conv", "G3"), ("model.10.m.0.attn.pe.conv", "G4"),
    ("model.13.cv1.conv", "G5"), ("model.22.cv1.conv", "G6"),
    ("model.23.cv2.0.0.conv", "G7"), ("model.23.cv3.0.0.conv", "G7"),
    ("model.23.cv2.1.0.conv", "G8"), ("model.23.dfl.conv", "G8")])
def test_group_map_keeps_legacy_semantic_boundaries(name, group):
    assert E.legacy().group_for_conv(name) == group


@pytest.mark.parametrize("name", ["outside.conv", "model.11.conv", "model.24.conv"])
def test_unreviewed_group_identity_rejected(name):
    with pytest.raises(ValueError):
        E.legacy().group_for_conv(name)


def test_configuration_masks_do_not_quantize_restored_G1():
    wrappers = [E.make_wrapper(scalar_conv(), f"model.{index}.conv", 1.) for index in (0, 3, 5)]
    for config, expected in [("fp32", [False, False, False]), ("q_all", [True, True, True]),
                             ("r_only:G1", [False, True, True]), ("q_only:G1", [True, False, False])]:
        E.legacy().enable_configuration(wrappers, config)
        assert [wrapper.enabled for wrapper in wrappers] == expected
    with pytest.raises(ValueError):
        E.legacy().enable_configuration(wrappers, "q_only:G9")


def test_quantized_clone_does_not_pollute_original_or_later_disabled_output():
    original = scalar_conv(.37)
    with torch.no_grad():
        x = torch.tensor([-.6, -.1, .1, .6]).reshape(1, 1, 1, 4)
        before = original(x).clone()
        state = E.state_info(original)
        wrapper = E.make_wrapper(deepcopy(original), "model.0.conv", .25)
        assert wrapper.original.weight.data_ptr() != original.weight.data_ptr()
        assert torch.equal(wrapper(x), before)
        wrapper.enabled = True
        changed = wrapper(x)
        assert not torch.equal(changed, before)
        assert E.state_info(original) == state and torch.equal(original(x), before)
        wrapper.enabled = False
        assert torch.equal(wrapper(x), before)


def test_synthetic_input_contract_rejects_real_image_dimensions():
    with pytest.raises(ValueError):
        E.synthetic_input(1080)
    assert E.synthetic_input(64).shape == (1, 3, 64, 64)
    assert E.synthetic_input(640).count_nonzero() == 0


def test_scale_inventory_cannot_silently_omit_conv():
    model = nn.Module()
    model.model = nn.ModuleList([scalar_conv()])
    with pytest.raises(ValueError):
        E.install(model, {})


def test_write_once_refuses_overwrite(tmp_path):
    path = tmp_path / "new.json"
    E.write_once(path, {"frozen": True})
    with pytest.raises(FileExistsError):
        E.write_once(path, {"frozen": False})
