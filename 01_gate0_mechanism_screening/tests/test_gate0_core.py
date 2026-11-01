from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "gate0.py"
SPEC = importlib.util.spec_from_file_location("gate0", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate0
SPEC.loader.exec_module(gate0)


def test_group_mapping_is_mutually_exclusive() -> None:
    assert gate0.group_for_conv("model.0.conv") == "G1"
    assert gate0.group_for_conv("model.10.m.0.attn.qkv.conv") == "G4"
    assert gate0.group_for_conv("model.16.cv1.conv") == "G5"
    assert gate0.group_for_conv("model.23.cv2.0.0.conv") == "G7"
    assert gate0.group_for_conv("model.23.cv3.2.2") == "G8"


def test_configuration_masks() -> None:
    assert gate0.parse_configuration("fp32") == set()
    assert gate0.parse_configuration("q_all") == set(gate0.GROUP_ORDER)
    assert gate0.parse_configuration("q_only:G2") == {"G2"}
    assert "G2" not in gate0.parse_configuration("r_only:G2")
    assert len(gate0.parse_configuration("r_only:G2")) == 7


def test_iou() -> None:
    assert gate0.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert gate0.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_one_to_one_matching_does_not_reuse_prediction() -> None:
    ground_truths = [
        gate0.GroundTruth("x:0", 0, (0, 0, 10, 10), 0.01, 0.1),
        gate0.GroundTruth("x:1", 0, (1, 1, 11, 11), 0.01, 0.1),
    ]
    detections = [gate0.Detection(0, 0.9, (0, 0, 10, 10))]
    rows = gate0.match_targets(ground_truths, detections, primary_conf=0.25, match_iou=0.5)
    assert sum(row["matched"] for row in rows) == 1


def test_qdq_wrapper_is_identity_when_disabled() -> None:
    import torch

    conv = torch.nn.Conv2d(3, 4, 3, padding=1).eval()
    wrapper = gate0.QDQConv2d(conv, "model.0.conv", "G1", activation_max_abs=1.0)
    tensor = torch.randn(1, 3, 8, 8)
    expected = conv(tensor)
    actual = wrapper(tensor)
    assert torch.equal(actual, expected)
