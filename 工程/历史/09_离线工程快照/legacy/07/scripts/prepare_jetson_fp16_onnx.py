"""Export one fixed-shape FP32 ONNX for a later on-Jetson FP16 build."""
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "jetson_trt_fp16_20261016_v1"
WORK = ROOT / "artifacts" / "jetson_trt_fp16_export_20261016_v1"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_CONFIG_DIR"] = str(WORK / ".ultralytics")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    WORK.mkdir(parents=True, exist_ok=False)
    (WORK / ".ultralytics" / "Ultralytics").mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config/gate0.json").read_text(encoding="utf-8"))
    record = json.loads((ROOT / "artifacts/manifests/selection_r_only_G1_run.json").read_text(encoding="utf-8"))
    source = Path(cfg["model"])
    assert sha(source) == record["model_sha256"]
    weights = WORK / "model.pt"
    shutil.copy2(source, weights)
    image = Path((ROOT / "splits/gate0_selection_images.txt").read_text(encoding="utf-8-sig").splitlines()[0])
    assert image.name == "000020.jpg"

    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    torch.set_num_threads(4)
    export_settings = dict(format="onnx", imgsz=640, batch=1, device="cpu", opset=13,
                           simplify=False, dynamic=False, half=False, nms=False)
    exported = Path(YOLO(str(weights), task="detect").export(**export_settings))
    model_onnx = onnx.load(str(exported))
    onnx.checker.check_model(model_onnx, full_check=True)
    assert len(model_onnx.graph.input) == 1 and len(model_onnx.graph.output) == 1
    shape = [d.dim_value for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]
    assert shape == [1, 3, 640, 640]
    assert [(o.domain, o.version) for o in model_onnx.opset_import] == [("", 13)]
    assert not any(t.external_data for t in model_onnx.graph.initializer)

    bgr = cv2.imread(str(image))
    assert bgr is not None
    padded = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=bgr)
    array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    reference = YOLO(str(weights), task="detect").model.float().eval()
    reference.fuse()
    with torch.inference_mode():
        pt = reference(torch.from_numpy(array))
        pt = (pt[0] if isinstance(pt, (tuple, list)) else pt).cpu().numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(exported), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: array})[0]
    assert actual.shape == pt.shape == (1, 14, 8400)
    assert np.isfinite(actual).all() and np.isfinite(pt).all()
    # Predefined numerical checks on this one input, not a dataset accuracy claim.
    np.testing.assert_allclose(actual[:, :4], pt[:, :4], rtol=1e-4, atol=0.02)
    np.testing.assert_allclose(actual[:, 4:], pt[:, 4:], rtol=1e-3, atol=2e-5)
    destination = OUT / "model.onnx"
    assert not destination.exists()
    shutil.copy2(exported, destination)
    report = {
        "scope": "Windows CPU one-image ONNX export check only; TensorRT/Jetson not tested here",
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                     "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
        "source_weights": str(source), "model_sha256": sha(source), "onnx_sha256": sha(destination),
        "image": str(image), "image_sha256": sha(image), "selection_rule": "first existing Selection entry",
        "export_settings": export_settings, "onnx_ir_version": model_onnx.ir_version,
        "input": {"name": session.get_inputs()[0].name, "shape": shape, "dtype": "float32"},
        "output": {"name": session.get_outputs()[0].name, "shape": list(actual.shape), "dtype": "float32"},
        "class_names": reference.names, "checker": "PASS", "numerical_check": "PASS",
        "box_tolerance": {"rtol": 1e-4, "atol": 0.02}, "score_tolerance": {"rtol": 1e-3, "atol": 2e-5},
        "max_abs_box_difference": float(np.max(np.abs(actual[:, :4] - pt[:, :4]))),
        "max_abs_score_difference": float(np.max(np.abs(actual[:, 4:] - pt[:, 4:]))),
        "original_source_unchanged": sha(source) == record["model_sha256"],
    }
    (OUT / "export_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print("LOCAL_ONNX_EXPORT_CHECK_OK")


if __name__ == "__main__":
    main()
