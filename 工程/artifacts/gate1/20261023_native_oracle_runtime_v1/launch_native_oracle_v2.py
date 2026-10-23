"""Run frozen official synthetic MATLAB-source functions using a pinned Octave.

This wrapper never installs, extracts, translates, edits or replaces native
functions.  The native input consists only of the already frozen 20 synthetic
fixtures.  Its new output must stay within this preparation directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
READINESS = ROOT / "artifacts/gate1/20261023_official_evaluator_readiness"
SOURCES = ROOT / "artifacts/gate1/20261022_split_refinement/evaluation_sources"
ARCHIVE_SHA256 = "fd3cf0e885467a15211b8ceded42800432a5462481324d49a896ef257e05d1a0"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(path: Path) -> dict:
    canonical = path.resolve()
    return {"path": canonical.relative_to(ROOT).as_posix(),
            "bytes": canonical.stat().st_size, "sha256": sha(canonical)}


def quoted(path: str | Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--octave-executable", type=Path, required=True)
    parser.add_argument("--mean2-directory", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--project-alias", type=Path,
                        help="Existing temporary ASCII SUBST root referring to this exact workspace")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(HERE) and output != HERE
    assert not output.exists(), "New execution directory required"
    runtime_manifest = args.runtime_manifest.resolve()
    assert runtime_manifest.is_relative_to(HERE) and runtime_manifest.is_file()
    exe = args.octave_executable.resolve()
    mean2_dir = args.mean2_directory.resolve()
    assert exe.is_relative_to(HERE) and exe.is_file(), "Runtime executable must be inside this preparation directory"
    assert mean2_dir.is_relative_to(HERE) and (mean2_dir / "mean2.m").is_file()
    runtime_contract = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert runtime_contract["schema"] == "rampq.gate1.octave_portable_runtime_files/v1"
    assert runtime_contract["path_base"] == "workspace"
    assert runtime_contract["archive_sha256"] == ARCHIVE_SHA256
    verified_runtime_files = {}
    for row in runtime_contract["files"]:
        path = (ROOT / row["path"]).resolve()
        assert path.is_relative_to(HERE / "runtime") and path.is_file()
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], str(path)
        verified_runtime_files[str(path).casefold()] = row
    assert str(exe).casefold() in verified_runtime_files
    assert str(mean2_dir / "mean2.m").casefold() in verified_runtime_files

    source_manifest_path = SOURCES / "source_manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    checked = []
    for row in manifest["files"]:
        path = SOURCES / row["path"]
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], str(path)
        checked.append(identity(path))
    contract_path = READINESS / "fixture_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for row in contract["bound_local_files"]:
        assert sha(READINESS / row["path"]) == row["sha256"], row["path"]
    fixture_path = READINESS / "synthetic_fixtures.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["data_role"] == "synthetic" and fixture["expected_oracle_outputs"] is None
    assert len(fixture["cases"]) == 20

    alias = None
    if args.project_alias is not None:
        native_root = args.project_alias
        assert native_root.is_absolute() and os.path.samefile(native_root, ROOT)
        assert str(native_root).isascii(), "Alias must avoid the documented non-ASCII runtime-path risk"
        alias = {"native_prefix": str(native_root), "real_prefix": str(ROOT)}
    else:
        native_root = ROOT

    def native(path: Path) -> Path:
        return native_root / path.resolve().relative_to(ROOT)

    output.mkdir(parents=True, exist_ok=False)
    subst_result = subprocess.run([str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/subst.exe")],
                                  capture_output=True, check=False)
    (output / "subst_query.txt").write_bytes(subst_result.stdout + subst_result.stderr)
    preflight_path = native(output) / "runtime_preflight.json"
    expression = (
        "addpath(" + quoted(native(mean2_dir)) + ",'-begin');"
        "assert(strcmp(canonicalize_file_name(which('mean2')),canonicalize_file_name(" + quoted(native(mean2_dir) / "mean2.m") + ")));"
        "assert(exist('jsondecode','builtin') || exist('jsondecode','file'));"
        "assert(exist('jsonencode','builtin') || exist('jsonencode','file'));"
        "p=struct('version',version,'computer',computer,'mean2_resolved_path',which('mean2'),"
        "'jsondecode_resolved_path',which('jsondecode'),'jsonencode_resolved_path',which('jsonencode'));"
        "f=fopen(" + quoted(preflight_path) + ",'w');assert(f>=0);"
        "fwrite(f,jsonencode(p),'char');fclose(f);"
        "addpath(" + quoted(native(READINESS)) + ",'-begin');"
        "run_synthetic_oracle(" + quoted(native(output)) + ");"
    )
    command = [str(native(exe)), "--no-gui", "--quiet", "--no-history", "--no-init-file",
               "--no-site-file", "--eval", expression]
    receipt = {
        "schema": "rampq.gate1.native_synthetic_oracle_launcher/v2",
        "source_commit": manifest["commit"], "source_manifest_sha256": sha(source_manifest_path),
        "verified_sources": checked, "fixture_sha256": sha(fixture_path),
        "fixture_contract_sha256": sha(contract_path), "fixture_count": len(fixture["cases"]),
        "runner_sha256": sha(READINESS / "run_synthetic_oracle.m"),
        "wrapper": identity(Path(__file__)), "runtime_manifest": identity(runtime_manifest),
        "runtime_file_count_verified": len(verified_runtime_files),
        "runtime": {"name": "octave", "executable": str(native(exe)),
                    "exe_sha256": sha(exe), "version": None, "command": command},
        "alias": alias, "subst_query": identity(output / "subst_query.txt"),
        "python": sys.version, "real_data_read": False, "model_executed": False,
        "selection_accessed": False, "holdout_accessed": False,
        "official_sources_edited": False, "custom_mean2_used": False,
        "official_oracle_executed": False, "official_AP_equivalence_certified": False,
        "status": "PREPARED_NOT_EXECUTED", "returncode": None,
        "mean2_resolved_path": None,
    }
    receipt_path = output / "launcher_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        with (output / "native_console.txt").open("wb") as console:
            result = subprocess.run(command, stdout=console, stderr=subprocess.STDOUT, check=False,
                                    cwd=str(native(output)))
        receipt["returncode"] = result.returncode
        if (output / "runtime_preflight.json").exists():
            runtime_info = json.loads((output / "runtime_preflight.json").read_text(encoding="utf-8"))
            assert runtime_info["version"] == "11.3.0", "Unexpected native runtime version"
            receipt["runtime"]["version"] = runtime_info["version"]
            receipt["mean2_resolved_path"] = runtime_info["mean2_resolved_path"]
        receipt["official_oracle_executed"] = (output / "oracle_results.mat").is_file()
        native_json = output / "oracle_results.json"
        native_ok = False
        if native_json.is_file():
            oracle = json.loads(native_json.read_text(encoding="utf-8"))
            native_ok = oracle["receipt"]["case_errors"] == 0 and oracle["receipt"]["fixture_count"] == 20
        receipt["status"] = ("EXECUTED_NATIVE_SYNTHETIC_ORACLE_UNCERTIFIED"
                             if result.returncode == 0 and native_ok else "NATIVE_RUNTIME_OR_CASE_FAILURE")
    except Exception as error:
        receipt["status"] = "NATIVE_LAUNCH_FAILURE"
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["outputs"] = {name: identity(output / name) for name in
                              ("native_console.txt", "runtime_preflight.json", "oracle_results.mat", "oracle_results.json")
                              if (output / name).is_file()}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(receipt["status"], flush=True)
    if receipt["returncode"] != 0 or not native_ok:
        raise SystemExit(receipt["returncode"] or 1)


if __name__ == "__main__":
    main()
