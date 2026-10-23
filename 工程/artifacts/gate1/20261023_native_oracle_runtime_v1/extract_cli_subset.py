"""Extract a root-approved explicit member list after a live disk-budget check.

No install/post-install/native executable is launched.  Use only after the
separate archive inventory and selected-member list have been reviewed.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = HERE / "downloads/octave-11.3.0-w64.7z"
EXPECTED = "fd3cf0e885467a15211b8ceded42800432a5462481324d49a896ef257e05d1a0"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def free_bytes() -> int:
    available = ctypes.c_ulonglong()
    function = ctypes.windll.kernel32.GetDiskFreeSpaceExW
    function.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulonglong),
                         ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong)]
    function.restype = ctypes.c_int
    ok = function(str(HERE), ctypes.byref(available), None, None)
    if not ok:
        raise ctypes.WinError()
    return available.value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-subset-sha256", required=True)
    parser.add_argument("--reserve-bytes", type=int, default=1610612736)
    args = parser.parse_args()
    assert args.reserve_bytes >= 1610612736
    subset_path = HERE / "proposed_cli_subset.json"
    assert sha(subset_path) == args.approved_subset_sha256
    assert sha(ARCHIVE) == EXPECTED
    inventory = json.loads((HERE / "archive_inventory.json").read_text(encoding="utf-8"))
    assert inventory["status"] == "LISTED_READ_ONLY_NO_EXTRACTION"
    selected = json.loads(subset_path.read_text(encoding="utf-8"))
    assert selected and len({row["path"] for row in selected}) == len(selected)
    expanded = sum(row["bytes"] for row in selected)
    # Allow extra filesystem metadata/rounding without spending the reserve.
    allowance = 64 * 1024 * 1024 + len(selected) * 4096
    before = free_bytes()
    assert before - expanded - allowance >= args.reserve_bytes, "Insufficient live free space for approved subset and reserve"
    runtime = HERE / "runtime"
    runtime.mkdir(exist_ok=False)
    members_path = HERE / "approved_cli_members.txt"
    assert not members_path.exists()
    for row in selected:
        assert row["mode"].startswith("-") and row["path"].isascii()
        assert not row["path"].startswith("/") and ":" not in row["path"]
        assert ".." not in row["path"].split("/") and "\n" not in row["path"]
    members_path.write_text("".join(row["path"] + "\n" for row in selected), encoding="ascii")
    command = ["C:/Windows/system32/tar.exe", "-xf", str(ARCHIVE), "-C", str(runtime), "-T", str(members_path)]
    with (HERE / "cli_extraction_console.txt").open("wb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    receipt = {"schema": "rampq.gate1.octave_cli_subset_extraction/v1", "returncode": result.returncode,
               "command": command, "archive_sha256": EXPECTED,
               "approved_subset_sha256": args.approved_subset_sha256, "selected_files": len(selected),
               "selected_expanded_bytes": expanded, "preflight_available_bytes": before,
               "filesystem_allowance_bytes": allowance, "required_remaining_reserve_bytes": args.reserve_bytes,
               "post_available_bytes": free_bytes(), "post_install_executed": False,
               "native_runtime_executed": False, "system_path_modified": False, "registry_modified": False}
    (HERE / "cli_extraction_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    assert result.returncode == 0, "Partial extraction retained for inspection"
    expected_files = {row["path"]: row for row in selected}
    actual = {p.relative_to(runtime).as_posix(): p for p in runtime.rglob("*") if p.is_file()}
    assert actual.keys() == expected_files.keys(), "Unexpected or missing extracted members"
    files = []
    for name in sorted(actual):
        path = actual[name]
        assert path.stat().st_size == expected_files[name]["bytes"], name
        files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = {"schema": "rampq.gate1.octave_portable_runtime_files/v1", "path_base": "workspace",
                "archive_sha256": EXPECTED, "archive_source": "https://github.com/gnu-octave/octave/releases/expanded_assets/release-11-3-0",
                "selection_identity": {"path": subset_path.relative_to(ROOT).as_posix(),
                                       "bytes": subset_path.stat().st_size, "sha256": sha(subset_path)},
                "subset_only": True, "post_install_executed": False, "files": files}
    (HERE / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("CLI_SUBSET_EXTRACTED_HASHED_NO_NATIVE_EXECUTION", flush=True)
    print(json.dumps({"files": len(files), "expanded_bytes": expanded, "free_bytes_after": receipt["post_available_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
