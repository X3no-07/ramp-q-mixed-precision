"""Seal this new native-runtime preparation after alias cleanup and audits.

Runtime files are bound transitively by their existing complete manifest;
metadata, native outputs, download archive and signature are bound directly.
No runtime execution or software/environment modification occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def row(path: Path) -> dict:
    return {"path": path.relative_to(HERE).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def main() -> None:
    destination = HERE / "preparation_manifest.json"
    assert not destination.exists(), "Refusing to replace sealed preparation manifest"
    final_path = HERE / "preparation_final_receipt.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["status"] == "NATIVE_SYNTHETIC_ORACLE_EXECUTED_PREPARATION_COMPLETE"
    assert final["temporary_alias"]["removed"] is True
    runtime_manifest_path = HERE / "runtime_manifest.json"
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    assert runtime_manifest["schema"] == "rampq.gate1.octave_portable_runtime_files/v1"
    for item in runtime_manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        assert path.is_relative_to(HERE / "runtime")
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], str(path)
    files = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == destination or path.is_relative_to(HERE / "runtime"):
            continue
        assert not path.name.endswith(".partial"), "Retained partial must be accounted for before sealing"
        files.append(row(path))
    manifest = {"schema": "rampq.gate1.native_runtime_preparation_manifest/v1", "path_base": "manifest_directory",
                "status": "SEALED_NATIVE_SYNTHETIC_ORACLE_EXECUTION_ONLY",
                "files": files, "file_count": len(files),
                "transitive_runtime_manifest": {**row(runtime_manifest_path), "path_base": "workspace",
                                                "bound_runtime_file_count": len(runtime_manifest["files"])},
                "official_AP_equivalence_certified": False, "scientific_gate1_passed": False,
                "real_data_read": False, "model_executed": False,
                "selection_accessed": False, "holdout_accessed": False,
                "notes": "source_plan.json is a preserved historical planning snapshot; preparation_final_receipt.json gives the final state. Runtime files are bound transitively, not duplicated in the top-level list."}
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in files:
        path = HERE / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    print(json.dumps({"status": manifest["status"], "files": len(files), "runtime_files": len(runtime_manifest["files"]),
                      "manifest_sha256": sha(destination)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
