"""Build one FP16-enabled TensorRT 8.5 engine on the target Jetson; no evaluation."""
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"command": command, "returncode": result.returncode,
                "output": result.stdout + result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if sha(ROOT / name) != info["sha256"]:
            raise RuntimeError("File checksum mismatch: " + name)
    print("FILES_SHA256_OK", flush=True)
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Run this script on the Jetson, not the Windows computer")
    import tensorrt as trt
    if not trt.__version__.startswith("8.5."):
        raise RuntimeError("This build was prepared for TensorRT 8.5.x, found " + trt.__version__)
    executable = Path("/usr/src/tensorrt/bin/trtexec")
    help_info = query([str(executable), "--help"])
    help_text = help_info.get("output", "")
    # 8.5 calls this --buildOnly; later releases call it --skipInference.
    build_only = "--buildOnly" if "--buildOnly" in help_text else "--skipInference"
    for option in (build_only, "--memPoolSize", "--profilingVerbosity", "--noTF32"):
        if option not in help_text:
            raise RuntimeError("trtexec help lacks required option: " + option + "\n" + str(help_info))

    run_dir = ROOT / "results" / datetime.now(timezone.utc).strftime("build_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    engine_path = run_dir / "model_fp16.engine"
    command = [str(executable), "--onnx=" + str(ROOT / "model.onnx"),
               "--saveEngine=" + str(engine_path), "--fp16", "--noTF32",
               "--memPoolSize=workspace:1024", "--profilingVerbosity=detailed", build_only]
    environment = {
        "tensorrt": trt.__version__, "python": platform.python_version(), "machine": platform.machine(),
        "power_mode": query(["/usr/sbin/nvpmodel", "-q"]),
        "memory": query(["free", "-h"]),
        "disk": query(["df", "-h", str(ROOT)]),
        "jtop": query(["systemctl", "show", "jtop.service", "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts"]),
        "fan_service": query(["systemctl", "is-active", "nvfancontrol.service"]),
        "jetson_linux": Path("/etc/nv_tegra_release").read_text(encoding="utf-8"),
    }
    write_json(run_dir / "environment.json", environment)
    (run_dir / "trtexec_help.txt").write_text(help_text, encoding="utf-8")
    write_json(run_dir / "build_request.json", {"command": command, "onnx_sha256": sha(ROOT / "model.onnx"),
        "precision": "FP16 enabled with FP32 fallback, TF32 disabled", "workspace_mib": 1024,
        "note": "workspace limit is not a cap on total process memory; build only, no detection or benchmark"})
    print("RESULT_DIR:", run_dir, flush=True)
    print("TensorRT:", trt.__version__, flush=True)
    print("POWER_MODE_QUERY:", environment["power_mode"], flush=True)
    print("Building one 1x3x640x640 engine; progress below is elapsed time, not percent.", flush=True)
    log_path = run_dir / "trtexec_build.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    print("BUILD_RUNNING_SECONDS:", int(time.monotonic() - start), flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    warnings = [line for line in log_text.splitlines() if "[W]" in line or "[E]" in line]
    if warnings:
        print("BUILD_WARNINGS_OR_ERRORS_LAST_12:\n" + "\n".join(warnings[-12:]), flush=True)
    if code != 0 or not engine_path.is_file():
        print("\n".join(log_text.splitlines()[-45:]), flush=True)
        write_json(run_dir / "build_result.json", {"status": "FAILED", "returncode": code})
        raise RuntimeError("Engine build failed; see " + str(log_path))

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Engine was saved but could not be deserialized")
    bindings = [{"name": engine.get_binding_name(i), "input": bool(engine.binding_is_input(i)),
                 "shape": list(engine.get_binding_shape(i)), "dtype": str(engine.get_binding_dtype(i))}
                for i in range(engine.num_bindings)]
    inputs = [b for b in bindings if b["input"]]
    outputs = [b for b in bindings if not b["input"]]
    if len(inputs) != 1 or inputs[0]["shape"] != [1, 3, 640, 640]:
        raise RuntimeError("Unexpected engine inputs: " + str(inputs))
    if len(outputs) != 1 or outputs[0]["shape"] != [1, 14, 8400]:
        raise RuntimeError("Unexpected engine outputs: " + str(outputs))
    inspector = engine.create_engine_inspector()
    layer_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    (run_dir / "engine_layers.json").write_text(layer_text, encoding="utf-8")
    report = {"status": "BUILT_AND_DESERIALIZED", "engine_sha256": sha(engine_path),
              "onnx_sha256": sha(ROOT / "model.onnx"), "engine_bytes": engine_path.stat().st_size,
              "bindings": bindings, "engine_layers": engine.num_layers,
              "fp16_enabled": True, "layer_precision_review": "pending review of engine_layers.json",
              "detection_inference": "not run", "benchmark": "not run"}
    write_json(run_dir / "build_result.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("RESULT_DIR:", run_dir, flush=True)
    print("TENSORRT_FP16_BUILD_OK", flush=True)


if __name__ == "__main__":
    main()
