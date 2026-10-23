from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "src" / "compute_gate0_output_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("compute_gate0_output_reconstruction", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_reconstruction_metrics_identity() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    nmse, cosine_distance = module.reconstruction_metrics(reference, reference.clone())
    assert nmse == 0.0
    assert abs(cosine_distance) < 1e-12


def test_choose_pair_uses_score_then_lower_mac() -> None:
    pairs = [("G1", "G2"), ("G3", "G4")]
    scores = {"G1": 3.0, "G2": 1.0, "G3": 2.0, "G4": 2.0}
    macs = {"G1": 0.10, "G2": 0.13, "G3": 0.11, "G4": 0.11}
    assert module.choose_pair(pairs, scores, macs)["groups"] == ["G3", "G4"]
