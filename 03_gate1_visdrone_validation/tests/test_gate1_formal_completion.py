from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_gate1_formal_completion as A


def events():
    result = []
    step = 0
    for epoch in (1, 2):
        result.append({"event": "epoch_started", "epoch": epoch})
        for batch in (1, 2):
            step += 1
            result.append({"event": "training_batch", "epoch": epoch, "batch": batch, "loss": 1.,
                "optimizer_steps": step, "ema_updates": step,
                "mask": {"positive_rows_preserved": True, "batch_size": 2 if batch == 2 else 4, "ignored_negative_anchors": 3}})
        for name in ("save_before_FP32_capture", "FP32_reference_committed", "framework_compatibility_checkpoint_saved", "epoch_completed_with_FP32_reference"):
            result.append({"event": name, "epoch": epoch})
    return result


def test_complete_synthetic_event_order():
    assert A.validate_events(events(), epochs=2, batches=2, updates=4)["batches"] == 4


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "positive", "batch_size", "updates", "order"])
def test_invalid_saved_events_rejected(mutation):
    rows = deepcopy(events())
    if mutation == "missing": rows.pop(2)
    if mutation == "duplicate": rows.insert(2, deepcopy(rows[1]))
    if mutation == "nan": rows[1]["loss"] = float("nan")
    if mutation == "positive": rows[1]["mask"]["positive_rows_preserved"] = False
    if mutation == "batch_size": rows[2]["mask"]["batch_size"] = 4
    if mutation == "updates": rows[1]["ema_updates"] = 2
    if mutation == "order": rows[3], rows[4] = rows[4], rows[3]
    with pytest.raises(ValueError):
        A.validate_events(rows, epochs=2, batches=2, updates=4)


def csv_text():
    header = "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2\n"
    return header + "".join(f"{e},10,1,2,3,0,0,0,0.01,0.01,0.01\n" for e in range(1,30)) + "30,10,1,2,3,0.001,0.001,0.001\n"


def test_short_final_csv_does_not_turn_lr_into_validation_metrics():
    result = A.normalize_csv(csv_text())
    assert result[-1]["lr"] == [.001]*3 and result[-1]["task_validation_metrics"] is None
    assert result[-1]["raw_field_count"] == 8


def test_unexpected_csv_truncation_rejected():
    with pytest.raises(ValueError):
        A.normalize_csv(csv_text().replace("1,10,1,2,3,0,0,0,0.01,0.01,0.01", "1,10,1"))
