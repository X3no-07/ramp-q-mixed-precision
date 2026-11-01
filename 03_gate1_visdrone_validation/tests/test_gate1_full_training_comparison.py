"""Partial-window data accounting and unchanged teacher-policy qualification."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gate1_full_training_comparison as W
P = W.P


def test_partial_epoch_flush_has_no_lost_gradient_or_sample():
    windows = [W.accumulation_window(b, 1133) for b in range(1133)]
    assert sum(update for _, update in windows) == 284
    assert windows[-1] == (1, True)
    # Constant scalar losses must sum to exactly one gradient per update,
    # including the final one-batch accumulation window.
    x = torch.tensor(1., requires_grad=True)
    gradients = []
    for width, update in windows:
        (x / width).backward()
        if update:
            gradients.append(x.grad.item()); x.grad = None
    assert gradients == [1.] * 284
    assert sum(min(4, 4530 - b * 4) for b in range(1133)) == 4530


def test_full_teacher_extension_preserves_previous_frozen_policy():
    # A saved real training member, CPU replay only: no model forward or fit.
    old = P.read(P.OUT / 'contract.json')
    meta = old['training_annotation_metadata'][0]
    folder = P.OUT / 'attempt_01/training_supervision'
    original = P.read(folder / '0000.json')
    native = P.read(W.N.OUT / 'attempt_01/training_supervision/0000.json')
    root = Path(P.read(old['evaluation_protocol']['path'])['prepared_root'])
    gt = P.MATCH.parse_ground_truth((root / meta['annotation_path']).read_text(encoding='utf-8-sig'), meta['image_id'])
    with np.load(folder / '0000_raw.npz', allow_pickle=False) as d:
        new = W.mine_spec(d['q'].copy(), d['r'].copy(), gt, meta, original['training_to_original_params'])
    assert new[W.ARMS[1]] == native[W.ARMS[1]]
    assert new['q_counts'] == native['q_counts'] and new['r_counts'] == native['r_counts']
    assert new['rescued_targets'] == native['rescued_targets']
    assert new['collateral_targets'] == native['collateral_targets']
