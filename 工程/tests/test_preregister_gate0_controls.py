from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "preregister_gate0_controls.py"
SPEC = importlib.util.spec_from_file_location("preregister_gate0_controls", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
controls = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controls
SPEC.loader.exec_module(controls)


MAC_SHARES = {
    "G1": 0.1031739602404368,
    "G2": 0.1265047552696813,
    "G3": 0.11820936148150549,
    "G4": 0.13287211807974594,
    "G5": 0.12028320992854943,
    "G6": 0.12857860371672528,
    "G7": 0.1778325043340192,
    "G8": 0.09254548694933654,
}


def test_random_control_choice_is_frozen() -> None:
    target = MAC_SHARES["G1"] + MAC_SHARES["G5"]
    pool = controls.eligible_pairs(MAC_SHARES, target)
    chosen = controls.choose_random_controls(pool)
    assert [(first, second) for first, second, _ in chosen] == [
        ("G4", "G8"),
        ("G3", "G8"),
        ("G2", "G8"),
    ]


def test_accuracy_sensitivity_baseline_collides_with_candidate() -> None:
    degraded = {"G1": 100, "G2": 9, "G3": 11, "G4": 7, "G5": 21, "G6": 3, "G7": 16, "G8": 10}
    modules = [
        {"group": group, "q_only_degraded": degraded[group], "mac_share": MAC_SHARES[group]}
        for group in controls.GROUPS
    ]
    ranking, selected = controls.greedy_accuracy_sensitivity_baseline(modules)
    assert ranking == ["G1", "G5", "G7", "G3", "G8", "G2", "G4", "G6"]
    assert selected == ["G1", "G5"]
