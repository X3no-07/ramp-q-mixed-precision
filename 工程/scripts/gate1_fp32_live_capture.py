"""Controlled CPU training-thread EMA capture before framework FP16 serialization.

This hook is an engineering component, not scientific reference acceptance. Its
lock coordinates this hook only; arbitrary concurrent writers are prohibited by
the caller protocol and cannot be excluded merely by a Python lock or hashes.
No model object pickle, old weights, external YAML, or data images are loaded.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import threading

import torch

_FORMAT_PATH = Path(__file__).with_name("gate1_fp32_reference_state.py")
_SPEC = importlib.util.spec_from_file_location("gate1_fp32_capture_format", _FORMAT_PATH)
FORMAT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FORMAT)
SCHEMA = "rampq.gate1.fp32_live_capture/v1"
PHASE = "SAVE_MODEL_ENTRY_BEFORE_SUPER_FP16_SERIALIZATION"
ALLOWED_CFG_MODULES = {"Conv", "C2f", "C3k2", "SPPF", "Concat", "Detect", "Bottleneck", "C2PSA", "nn.Upsample", "nn.Identity"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sources():
    """Pin the complete installed Python implementation plus both local writers."""
    import ultralytics
    root = Path(ultralytics.__file__).resolve().parent
    members = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*.py"))}
    require(bool(members), "Empty Ultralytics Python source inventory")
    return {"ultralytics_python_tree_sha256": hashlib.sha256(canonical(members).encode()).hexdigest(),
            "ultralytics_python_files": len(members), "ultralytics_version": str(ultralytics.__version__),
            "torch_version": str(torch.__version__), "capture_sha256": sha(__file__),
            "state_format_sha256": sha(_FORMAT_PATH)}


def deterministic_synthetic_input(shape=(1, 3, 64, 64)):
    require(isinstance(shape, (list, tuple)) and len(shape) == 4 and all(type(x) is int and x > 0 for x in shape), "Invalid synthetic shape")
    require(shape[0] == 1 and shape[1] == 3 and shape[2] % 32 == 0 and shape[3] % 32 == 0 and max(shape[2:]) <= 256,
            "Only bounded one-image RGB synthetic shapes divisible by 32 are allowed")
    size = 1
    for value in shape:
        size *= value
    return torch.arange(size, dtype=torch.float32).remainder(251).div(250).reshape(tuple(shape))


def _json_primitives(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        require(torch.isfinite(torch.tensor(value)).item(), "Nonfinite architecture scalar")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_primitives(v) for v in value]
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), "Architecture keys must be strings")
        return {k: _json_primitives(v) for k, v in value.items()}
    raise ValueError("Architecture must contain JSON primitives only")


def architecture(model):
    from ultralytics.nn.tasks import DetectionModel
    require(isinstance(model, DetectionModel), "Only DetectionModel-compatible inference graphs are supported")
    cfg = _json_primitives(deepcopy(model.yaml))
    require(type(cfg) is dict and type(cfg.get("nc")) is int and cfg["nc"] > 0, "Invalid architecture nc")
    require("activation" not in cfg, "Custom activation expressions are not reviewed")
    require(type(cfg.get("channels", 3)) is int and cfg.get("channels", 3) == 3, "Only three-channel architecture supported")
    require(set(cfg) <= {"nc", "backbone", "head", "channels", "scale", "scales", "yaml_file", "inplace",
                         "depth_multiple", "width_multiple", "max_channels"}, "Unreviewed architecture configuration key")
    require(all(k in cfg for k in ("backbone", "head")), "Missing architecture graph")
    for layer in cfg["backbone"] + cfg["head"]:
        require(isinstance(layer, list) and len(layer) == 4 and type(layer[2]) is str and layer[2] in ALLOWED_CFG_MODULES,
                "Unsupported/unreviewed graph module")
    nc = cfg["nc"]
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    require(type(names) is dict and set(names) == set(range(nc)) and all(type(v) is str for v in names.values()), "Names/nc mismatch")
    require(getattr(model.model[-1], "nc", None) == nc, "Head class count mismatch")
    return {"reconstruction_class": "ultralytics.nn.tasks.DetectionModel", "cfg": cfg,
            "names": [names[i] for i in range(nc)], "nc": nc, "channels": cfg.get("channels", 3),
            "live_class": type(model).__module__ + "." + type(model).__qualname__,
            "inference_only_reconstruction": True}


def _tensor_description(value):
    require(isinstance(value, torch.Tensor), "Expected tensor")
    value = value.detach().cpu().contiguous()
    require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "Nonfinite output")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def describe_output(output):
    if isinstance(output, torch.Tensor):
        return {"kind": "tensor", "tensor": _tensor_description(output)}
    if isinstance(output, (list, tuple)):
        return {"kind": type(output).__name__, "items": [describe_output(v) for v in output]}
    if isinstance(output, dict):
        require(all(type(k) is str for k in output), "Output dict keys differ")
        return {"kind": "dict", "items": {k: describe_output(v) for k, v in sorted(output.items())}}
    if output is None or type(output) in (str, int, float, bool):
        return {"kind": "scalar", "value": _json_primitives(output)}
    raise ValueError("Unsupported synthetic output structure")


def reconstruct_and_verify(state_directory, architecture_contract, expected_sources, frozen_live_copy, synthetic_input):
    """Restricted state-only load into a freshly initialized local cfg graph."""
    require(sources() == expected_sources, "Runtime source identity changed before reconstruction")
    from ultralytics.nn.tasks import DetectionModel
    saved, metadata = FORMAT.load_snapshot(state_directory)  # weights_only=True inside frozen writer.
    require(hashlib.sha256(canonical(architecture_contract).encode()).hexdigest() == metadata["architecture_sha256"],
            "Architecture identity differs")
    cfg = deepcopy(architecture_contract["cfg"])
    # Isolate constructor randomness from training RNG; no path/weights argument is supplied.
    with torch.random.fork_rng(devices=[]):
        rebuilt = DetectionModel(cfg=cfg, ch=architecture_contract["channels"], nc=architecture_contract["nc"], verbose=False)
    loaded = rebuilt.load_state_dict(saved, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, "Strict state reconstruction differs")
    rebuilt.names = dict(enumerate(architecture_contract["names"]))
    rebuilt = rebuilt.cpu().float().eval()
    require(FORMAT.describe_state(FORMAT.freeze_state(rebuilt.state_dict())) == FORMAT.describe_state(saved), "Reconstructed state differs")
    with torch.no_grad():
        left = describe_output(frozen_live_copy(synthetic_input.clone()))
        right = describe_output(rebuilt(synthetic_input.clone()))
    require(left == right, "CPU synthetic outputs differ after architecture reconstruction")
    return {"status": "PASS_STRICT_STATE_AND_SYNTHETIC_CPU_OUTPUT_RECONSTRUCTION_ONLY", "restricted_weights_only_load": True,
            "strict_load_state_dict": True, "input": _tensor_description(synthetic_input), "output": right,
            "output_values_dtypes_shapes_exact": True, "model_object_deserialized": False,
            "external_architecture_path_or_weights_used": False, "optimizer_resume_reconstruction": False}


class LiveFP32Capture:
    """Callable for trainer.fp32_capture_callback, invoked before super.save_model.

    Constructor and every call must occur on the same main training thread. Only
    selected epochs save a separate exact EMA reference; this never selects best.
    Device scope is CPU in v1. The caller must forbid all concurrent EMA writers.
    """
    def __init__(self, output_root, training_contract_sha256, selected_epochs, *, input_shape=(1, 3, 64, 64), expected_source_sha256):
        require(isinstance(training_contract_sha256, str) and re.fullmatch("[0-9a-f]{64}", training_contract_sha256), "Invalid training contract SHA")
        require(isinstance(selected_epochs, (list, tuple)) and selected_epochs and len(set(selected_epochs)) == len(selected_epochs)
                and all(type(i) is int and i > 0 for i in selected_epochs), "Selected epochs must be unique positive ints")
        require(threading.current_thread() is threading.main_thread(), "Capture must be configured on main training thread")
        require(type(expected_source_sha256) is dict and expected_source_sha256 == sources(), "Source identity must be pre-pinned")
        self.output_root = Path(output_root).resolve()
        self.training_contract_sha256 = training_contract_sha256
        self.selected_epochs = tuple(selected_epochs)
        self.expected_sources = deepcopy(expected_source_sha256)
        self.synthetic_input = deterministic_synthetic_input(input_shape)
        self.thread_ident = threading.get_ident()
        self._lock = threading.Lock()
        self.receipts = []

    def __call__(self, trainer):
        require(threading.current_thread() is threading.main_thread() and threading.get_ident() == self.thread_ident,
                "Capture must run on its original main training thread")
        require(self._lock.acquire(blocking=False), "Concurrent or reentrant capture is forbidden")
        try:
            return self._capture(trainer)
        finally:
            self._lock.release()

    @torch.no_grad()
    def _capture(self, trainer):
        require(getattr(trainer.args, "amp", None) is False and getattr(trainer, "amp", None) is False, "AMP must be explicitly disabled")
        require(type(trainer.epoch) is int and trainer.epoch >= 0, "Invalid trainer epoch")
        epoch = trainer.epoch + 1
        if epoch not in self.selected_epochs:
            return None
        require(sources() == self.expected_sources, "Runtime source identity changed before capture")
        ema = trainer.ema
        require(ema is not None and getattr(ema, "enabled", False) is True, "Enabled live EMA required")
        require(type(ema.updates) is int and ema.updates > 0, "EMA must have completed at least one update")
        require(not ema.ema.training, "Live EMA must already be in eval mode")
        live = ema.ema
        require(all(v.device.type == "cpu" for v in live.state_dict().values()), "v1 capture is qualified on CPU only")
        before_updates = ema.updates
        before = FORMAT.describe_state(FORMAT.freeze_state(live.state_dict()))
        arch = architecture(live)
        arch_sha = hashlib.sha256(canonical(arch).encode()).hexdigest()
        metadata = {"training_contract_sha256": self.training_contract_sha256, "architecture_sha256": arch_sha,
                    "epoch_one_based": epoch, "source_stage": FORMAT.STAGE}
        final = self.output_root / f"epoch_{epoch:04d}"
        pending = self.output_root / f"epoch_{epoch:04d}.pending"
        require(not final.exists() and not pending.exists(), "Capture epoch directory must be new")
        pending.mkdir(parents=True, exist_ok=False)
        # An isolated copy preserves live forward caches as well as all state tensors.
        frozen_live_copy = deepcopy(live).cpu().eval()
        require(FORMAT.describe_state(FORMAT.freeze_state(frozen_live_copy.state_dict())) == before, "Live copy changed state")
        write_json(pending / "architecture.json", arch)
        FORMAT.write_snapshot(live.state_dict(), pending / "state", metadata)
        reconstruction = reconstruct_and_verify(pending / "state", arch, self.expected_sources, frozen_live_copy, self.synthetic_input)
        after = FORMAT.describe_state(FORMAT.freeze_state(live.state_dict()))
        require(ema.updates == before_updates and before == after, "EMA state/update counter changed during capture")
        require(sources() == self.expected_sources, "Runtime source identity changed during capture")
        receipt = {"schema": SCHEMA, "status": "PASS_CONTROLLED_LIVE_FP32_CAPTURE_AND_CPU_RECONSTRUCTION_ONLY",
                   "phase": PHASE, "epoch_one_based": epoch, "ema_updates_before": before_updates, "ema_updates_after": ema.updates,
                   "training_contract_sha256": self.training_contract_sha256, "architecture_sha256": arch_sha,
                   "sources": self.expected_sources, "amp": False, "device_scope": "CPU_ONLY",
                   "floating_source_already_fp32": True, "half_lift_performed": False,
                   "live_state_before_sha256": hashlib.sha256(canonical(before).encode()).hexdigest(),
                   "live_state_after_sha256": hashlib.sha256(canonical(after).encode()).hexdigest(),
                   "state_and_update_counter_unchanged": True, "main_training_thread_only": True, "nonreentrant_hook_lock": True,
                   "arbitrary_external_concurrent_writers_excluded_by_lock": False,
                   "caller_protocol_requires_no_concurrent_ema_writers": True,
                   "synthetic_input_recipe": "arange(numel).float32().remainder(251).div(250), NCHW",
                   "reconstruction": reconstruction, "old_checkpoint_read_or_recovered": False,
                   "scientific_reference_accepted": False, "optimizer_or_resume_state_saved": False,
                   "gpu_capture_qualified": False, "real_dataset_or_holdout_read": False}
        write_json(pending / "capture_receipt.json", receipt)
        members = {p.relative_to(pending).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                   for p in sorted(pending.rglob("*")) if p.is_file()}
        write_json(pending / "manifest.json", {"schema": SCHEMA, "files": members})
        pending.rename(final)  # Same-parent commit only after all checks; pending failures are not success evidence.
        self.receipts.append(receipt)
        return receipt


class FP32LiveCaptureMixin:
    """Alternative MRO adapter; do not also install the callable callback."""
    def configure_fp32_capture(self, output_root, training_contract_sha256, selected_epochs, *,
                               input_shape=(1, 3, 64, 64), expected_source_sha256):
        require(getattr(self, "_fp32_live_capture", None) is None, "Capture already configured")
        require(getattr(self, "fp32_capture_callback", None) is None, "Choose either mixin or callback adapter")
        self._fp32_live_capture = LiveFP32Capture(output_root, training_contract_sha256, selected_epochs,
                                                 input_shape=input_shape, expected_source_sha256=expected_source_sha256)
        return self._fp32_live_capture

    def save_model(self):
        require(getattr(self, "_fp32_live_capture", None) is not None, "Configure controlled capture before framework save")
        require(getattr(self, "fp32_capture_callback", None) is None, "Double capture adapters are forbidden")
        self._fp32_live_capture(self)
        return super().save_model()

