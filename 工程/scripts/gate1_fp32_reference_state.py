"""Lossless FP32 state snapshot format; not a training or checkpoint selector.

This future-writer component never lifts a half checkpoint or infers provenance
from dtype. It cannot recover the old FP16 EMA's discarded mantissa. A reviewed
live-training hook, architecture contract and independent qualification remain
necessary before a saved state can be an accepted scientific reference.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import torch


SCHEMA = "rampq.gate1.fp32_reference_state_format/v1"
REQUIRED_METADATA = {"training_contract_sha256", "architecture_sha256", "epoch_one_based", "source_stage"}
STAGE = "LIVE_FP32_STATE_BEFORE_ANY_FP16_SERIALIZATION"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_metadata(metadata: dict) -> dict:
    require(type(metadata) is dict and set(metadata) == REQUIRED_METADATA, "Metadata contract keys differ")
    require(all(isinstance(metadata[k], str) and re.fullmatch("[0-9a-f]{64}", metadata[k])
                for k in ("training_contract_sha256", "architecture_sha256")), "Missing SHA256 identity")
    require(type(metadata["epoch_one_based"]) is int and metadata["epoch_one_based"] > 0, "Invalid epoch")
    require(metadata["source_stage"] == STAGE, "Post-save dtype lifting is not an allowed source declaration")
    # This declaration is bound, not independently proved by this format writer.
    return dict(metadata)


def freeze_state(state: dict) -> dict:
    require(isinstance(state, dict) and bool(state), "Empty/non-dict state")
    result = {}
    floating = 0
    for name, value in state.items():
        require(isinstance(name, str) and name and isinstance(value, torch.Tensor), "State must contain named tensors only")
        require(value.layout == torch.strided and not value.is_quantized and not value.is_complex(), "Unsupported tensor representation")
        if value.is_floating_point():
            floating += 1
            require(value.dtype == torch.float32, "Floating source tensors must already be FP32: " + name)
            require(bool(torch.isfinite(value).all()), "Nonfinite state: " + name)
        else:
            require(value.dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), "Unsupported nonfloating state")
        result[name] = value.detach().to(device="cpu").contiguous().clone()
    require(floating > 0, "State has no floating tensors")
    return result


def describe_state(state: dict) -> dict:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()}
            for name, value in sorted(state.items())}


def write_snapshot(state: dict, output: Path, metadata: dict) -> dict:
    metadata = validate_metadata(metadata)
    frozen = freeze_state(state)
    described = describe_state(frozen)
    output = Path(output).resolve()
    require(not output.exists(), "Snapshot output must be a new directory")
    output.mkdir(parents=True, exist_ok=False)
    payload = {"schema": SCHEMA, "metadata": metadata, "state_dict": frozen,
               "training_origin_independently_proven": False, "accepted_scientific_reference": False}
    destination = output / "reference_state.pt"
    torch.save(payload, destination)
    # A restricted roundtrip tests actual serialized values, not just dtype.
    restored = torch.load(destination, map_location="cpu", weights_only=True)
    require(restored["schema"] == SCHEMA and restored["metadata"] == metadata
            and describe_state(restored["state_dict"]) == described, "Serialized state changed")
    require(all(torch.equal(frozen[name], restored["state_dict"][name]) for name in frozen), "Roundtrip values differ")
    # Retain non-interference of the caller's live tensors throughout the write.
    require(describe_state(freeze_state(state)) == described, "Caller state changed during snapshot")
    receipt = {"schema": SCHEMA, "status": "PASS_FP32_STATE_FORMAT_ROUNDTRIP_ONLY",
               "at_utc": datetime.now(timezone.utc).isoformat(), "metadata": metadata,
               "torch": str(torch.__version__), "tensors": described,
               "payload": {"path": destination.name, "bytes": destination.stat().st_size, "sha256": sha(destination)},
               "floating_source_was_already_fp32": True, "half_lift_performed": False,
               "restricted_weights_only_load": True, "all_values_and_dtypes_exact": True,
               "caller_state_unchanged": True, "training_origin_independently_proven": False,
               "live_training_hook_integrated": False, "architecture_reconstruction_tested": False,
               "accepted_scientific_reference": False, "old_checkpoint_recovered": False,
               "optimizer_or_resume_state_saved": False}
    (output / "snapshot_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
             for path in sorted(output.iterdir()) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps({"schema": SCHEMA, "files": files}, indent=2)+"\n", encoding="utf-8")
    return receipt


def load_snapshot(output: Path) -> tuple[dict, dict]:
    output = Path(output).resolve(strict=True)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == SCHEMA and set(manifest.get("files", {})) == {"reference_state.pt", "snapshot_receipt.json"}, "Snapshot manifest differs")
    for name, expected in manifest["files"].items():
        path = output / name
        require(not path.is_symlink() and path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], "Snapshot member changed")
    payload = torch.load(output / "reference_state.pt", map_location="cpu", weights_only=True)
    require(payload.get("schema") == SCHEMA and payload.get("accepted_scientific_reference") is False
            and payload.get("training_origin_independently_proven") is False, "Snapshot scope differs")
    metadata = validate_metadata(payload["metadata"])
    state = freeze_state(payload["state_dict"])
    receipt = json.loads((output / "snapshot_receipt.json").read_text(encoding="utf-8"))
    require(receipt["status"] == "PASS_FP32_STATE_FORMAT_ROUNDTRIP_ONLY" and receipt["metadata"] == metadata
            and receipt["tensors"] == describe_state(state), "Snapshot receipt differs")
    return state, metadata
