"""Fresh random CPU SGD+EMA example; no checkpoint, image, or dataset input."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
(HERE / "ultralytics_config").mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(HERE / "ultralytics_config")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import torch
torch.set_num_threads(1)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA

spec = importlib.util.spec_from_file_location("fp32_live_capture", ROOT / "scripts/gate1_fp32_live_capture.py")
capture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture_module)
from types import SimpleNamespace

cfg = {"nc": 2, "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [16, 3, 2]],
                              [-1, 1, "Conv", [32, 3, 2]]], "head": [[[0, 1, 2], 1, "Detect", ["nc"]]]}
contract = {"schema": "rampq.gate1.fp32_live_capture_component_example/v1", "seed": 713, "amp": False,
            "device": "cpu", "sgd_steps": 2, "ema_updates": 2, "loss": "first_parameter.square().mean()",
            "cfg": cfg, "scope": "SYNTHETIC_COMPONENT_ONLY_NOT_DATASET_TRAINING"}
capture_module.write_json(HERE / "synthetic_training_contract.json", contract)
contract_sha = capture_module.sha(HERE / "synthetic_training_contract.json")
torch.manual_seed(713)
model = DetectionModel(cfg=deepcopy(cfg), ch=3, nc=2, verbose=False).cpu().float()
model.names = {0: "synthetic_zero", 1: "synthetic_one"}
ema = ModelEMA(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []
for _ in range(2):
    optimizer.zero_grad()
    loss = next(model.parameters()).square().mean()
    losses.append(float(loss.detach()))
    loss.backward()
    optimizer.step()
    ema.update(model)
trainer = SimpleNamespace(args=SimpleNamespace(amp=False), amp=False, epoch=0, ema=ema)
pins = capture_module.sources()
capture_module.write_json(HERE / "runtime_source_fingerprints.json", pins)
capture = capture_module.LiveFP32Capture(HERE / "synthetic_capture", contract_sha, [1], expected_source_sha256=pins)
receipt = capture(trainer)
capture_module.write_json(HERE / "synthetic_example_receipt.json", {
    "status": "PASS_TWO_SYNTHETIC_SGD_EMA_STEPS_AND_CONTROLLED_CPU_CAPTURE_ONLY",
    "losses": losses, "ema_updates": ema.updates, "capture_status": receipt["status"],
    "training_contract_sha256": contract_sha, "dataset_training_executed": False,
    "old_weights_read": False, "real_images_read": False, "scientific_reference_accepted": False})
print(json.dumps({"status": "PASS_COMPONENT_EXAMPLE", "ema_updates": ema.updates, "capture": receipt["status"]}))
