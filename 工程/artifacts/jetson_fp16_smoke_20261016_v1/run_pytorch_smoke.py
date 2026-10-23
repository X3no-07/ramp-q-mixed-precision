"""One-image CUDA FP32 pipeline check before a separate TensorRT FP16 build."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != info["sha256"]:
            raise RuntimeError("File checksum mismatch: " + name)
    print("FILES_SHA256_OK", flush=True)
    environment = {"python": platform.python_version(), "machine": platform.machine()}
    for package in ("torch", "torchvision", "ultralytics", "tensorrt", "onnx", "numpy"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = "metadata not found"
    environment["onnx_module_found"] = importlib.util.find_spec("onnx") is not None
    print(json.dumps(environment, ensure_ascii=True, indent=2), flush=True)

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this check must run on the Jetson GPU")
    environment["gpu"] = torch.cuda.get_device_name(0)
    settings = manifest["pytorch_smoke_settings"]
    model = YOLO(str(ROOT / "best.pt"), task="detect")
    result = model.predict(source=str(ROOT / "000020.jpg"), **settings)[0]
    torch.cuda.synchronize()
    device = str(model.predictor.model.device)
    if not device.startswith("cuda"):
        raise RuntimeError("Inference did not use CUDA: " + device)
    if result.boxes is None or not bool(torch.isfinite(result.boxes.data).all()):
        raise RuntimeError("Missing or non-finite detection output")
    output = ROOT / "results" / datetime.now(timezone.utc).strftime("pytorch_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "one-image PyTorch CUDA FP32 smoke test; not a TensorRT or accuracy benchmark",
        "environment": environment,
        "device": device,
        "settings": settings,
        "model_sha256": manifest["files"]["best.pt"]["sha256"],
        "image_sha256": manifest["files"]["000020.jpg"]["sha256"],
        "image": "000020.jpg",
        "class_names": result.names,
        "detections_xyxy_conf_cls": result.boxes.data.detach().cpu().tolist(),
    }
    (output / "prediction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    result.save(filename=str(output / "prediction.jpg"))
    print("DETECTIONS:", len(result.boxes))
    print("RESULT_DIR:", output)
    print("PYTORCH_MODEL_SMOKE_OK")


if __name__ == "__main__":
    main()
