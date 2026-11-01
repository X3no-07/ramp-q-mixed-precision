"""Exercise the new Trainer's own ancestor dispatch on generated images."""
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate1_formal_training as F


def test_new_trainer_preprocess_uses_actual_detection_ancestor():
    trainer = F.BoundCUDATrainer.__new__(F.BoundCUDATrainer)
    trainer.args = SimpleNamespace(multi_scale=0.0, amp=False)
    trainer.device = torch.device("cpu")
    batch = {"img": torch.full((2, 3, 64, 64), 255, dtype=torch.uint8),
             "ignore_boxes_xyxy": [torch.tensor([[1., 2., 3., 4.]]), torch.empty((0, 4))],
             "image_identities": ({"image_id": "train/generated_a"}, {"image_id": "train/generated_b"})}
    result = trainer.preprocess_batch(batch)
    assert result["img"].dtype == torch.float32 and bool((result["img"] == 1).all())
    assert [list(v.shape) for v in result["ignore_boxes_xyxy"]] == [[1, 4], [0, 4]]


def test_one_epoch_fixture_has_no_warmup_even_when_requested():
    trainer = F.BoundCUDATrainer.__new__(F.BoundCUDATrainer)
    trainer.args = SimpleNamespace(warmup_epochs=3.0)
    trainer.epochs = 1
    assert trainer._get_warmup_iterations(32) == 0
    trainer.epochs = 30
    assert trainer._get_warmup_iterations(1133) == 3399
