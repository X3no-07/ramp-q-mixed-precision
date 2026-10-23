"""Read-only inventory and conservative CLI-subset budget for pinned Octave.

This script lists members; it never extracts or executes archive content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "downloads/octave-11.3.0-w64.7z"
EXPECTED = "fd3cf0e885467a15211b8ceded42800432a5462481324d49a896ef257e05d1a0"
TAR = Path("C:/Windows/system32/tar.exe")


def main() -> None:
    h = hashlib.sha256()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == EXPECTED
    log = HERE / "archive_verbose_listing.txt"
    assert not log.exists(), "Inventory output must be new"
    command = [str(TAR), "-tvf", str(ARCHIVE)]
    result = subprocess.run(command, capture_output=True, check=False)
    log.write_bytes(result.stdout)
    (HERE / "archive_listing_stderr.txt").write_bytes(result.stderr)
    receipt = {"schema": "rampq.gate1.octave_archive_inventory/v1", "command": command,
               "returncode": result.returncode, "archive_sha256": EXPECTED,
               "archive_bytes": ARCHIVE.stat().st_size, "extracted": False,
               "native_runtime_executed": False}
    if result.returncode:
        receipt["status"] = "BSDTAR_CANNOT_LIST_PINNED_7Z"
        (HERE / "archive_inventory.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise SystemExit(result.returncode)
    pattern = re.compile(r"^(?P<mode>\S+)\s+(?P<links>\d+)\s+(?P<owner>\S+)\s+(?P<group>\S+)\s+(?P<size>\d+)\s+\S+\s+\d+\s+[\d:]+\s+(?P<path>.+)$")
    files, unparsed = [], []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            unparsed.append(line)
            continue
        value = match.groupdict()
        name = value["path"].replace("\\", "/")
        if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in name.split("/"):
            raise AssertionError("Unsafe archive member: " + name)
        files.append({"path": name, "bytes": int(value["size"]), "mode": value["mode"]})
    assert not unparsed, repr(unparsed[:10])
    regular = [row for row in files if row["mode"].startswith("-")]
    assert regular and all(not row["mode"].startswith("l") for row in files), "Review symlinks before any extraction"
    roots = sorted({row["path"].split("/")[0] for row in files})
    assert len(roots) == 1
    prefix = roots[0] + "/"
    selected = []
    for row in regular:
        name = row["path"][len(prefix):]
        reason = None
        if re.fullmatch(r"mingw64/bin/[^/]+\.dll", name, flags=re.I):
            reason = "All bin DLLs conservatively retain native runtime dependencies"
        elif re.fullmatch(r"mingw64/bin/octave[^/]*\.exe", name, flags=re.I):
            reason = "Octave executables including CLI"
        elif name.startswith(("mingw64/lib/octave/", "mingw64/libexec/octave/", "mingw64/share/octave/11.3.0/")):
            reason = "Core native modules, executable support and built-in m scripts"
        elif re.match(r"mingw64/share/octave/packages/image-[^/]+/", name):
            reason = "Bundled unmodified image package including mean2"
        elif "license" in name.lower() or re.search(r"(^|/)(copying|copyright)(\.|$)", name, re.I):
            reason = "Original license/copyright notices"
        elif name.lower() in ("post-install.bat", "octave-launch.exe"):
            reason = "Preparation script for review and launcher (not executed)"
        if reason:
            selected.append({**row, "reason": reason})
    by_section = {}
    for row in regular:
        parts = row["path"].split("/")
        section = "/".join(parts[:min(len(parts), 4)])
        by_section[section] = by_section.get(section, 0) + row["bytes"]
    receipt.update({"status": "LISTED_READ_ONLY_NO_EXTRACTION", "root": roots[0],
                    "member_count": len(files), "regular_file_count": len(regular),
                    "expanded_regular_file_bytes": sum(row["bytes"] for row in regular),
                    "conservative_cli_subset_file_count": len(selected),
                    "conservative_cli_subset_bytes": sum(row["bytes"] for row in selected),
                    "sections_by_bytes": sorted(by_section.items(), key=lambda item: -item[1]),
                    "selection_is_provisional_not_executability_proof": True})
    (HERE / "archive_inventory.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (HERE / "archive_members.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    (HERE / "proposed_cli_subset.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "sections_by_bytes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
