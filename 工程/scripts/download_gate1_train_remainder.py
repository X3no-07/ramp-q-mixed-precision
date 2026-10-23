"""Continue the pinned public training archive using verified HTTP byte ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlsplit

import requests


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    assets_path = root / "sources/selected_release_assets.json"
    asset = json.loads(assets_path.read_text(encoding="utf-8"))["VisDrone2019-DET-train.zip"]
    if asset["id"] != 178786188 or asset["size"] != 1549875511:
        raise ValueError("Unexpected release asset identity")
    prefix = root / "downloads/VisDrone2019-DET-train.zip.partial"
    target = root / "downloads/VisDrone2019-DET-train.zip"
    if target.exists():
        raise FileExistsError(target)
    initial = prefix.stat().st_size
    if not 0 < initial < asset["size"]:
        raise ValueError("Expected a stopped incomplete prefix download")
    headers_path = root / "sources/train_asset_redirect_headers.txt"
    cdn_url = next(line.split(":", 1)[1].strip() for line in headers_path.read_text(encoding="utf-8").splitlines() if line.lower().startswith("location:"))
    parsed = urlsplit(cdn_url)
    if parsed.scheme != "https" or parsed.hostname != "release-assets.githubusercontent.com":
        raise ValueError("Unexpected public asset CDN redirect")
    part_dir = root / "downloads/train_ranges_v2"
    part_dir.mkdir(exist_ok=False)
    count = 8
    step = (asset["size"] - initial + count - 1) // count
    ranges = [(start, min(start + step - 1, asset["size"] - 1)) for start in range(initial, asset["size"], step)]
    record = {"status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "reason": "Intentional stop of slow original session; preserve its exact prefix and fetch only missing byte ranges from same asset ID",
              "source_asset_metadata_sha256": digest(assets_path), "asset": asset,
              "cdn_redirect_headers_sha256": digest(headers_path),
              "prefix_bytes": initial, "prefix_sha256": digest(prefix), "ranges": ranges,
              "publisher_digest_available": asset.get("digest") is not None}
    receipt_path = part_dir / "download_receipt.json"
    receipt_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def fetch(bounds: tuple[int, int]) -> dict:
        start, end = bounds
        path = part_dir / f"{start}-{end}.partial"
        expected = end - start + 1
        completed = 0
        attempts = []
        began = time.monotonic()
        for attempt in range(1, 4):
            offset = start + completed
            if completed == expected:
                break
            url = cdn_url
            try:
                with requests.get(url, headers={"Accept": "application/octet-stream", "Accept-Encoding": "identity", "Range": f"bytes={offset}-{end}"}, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    expected_range = f"bytes {offset}-{end}/{asset['size']}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError(f"Range response mismatch: {response.status_code}, {response.headers.get('Content-Range')}, expected {expected_range}")
                    with path.open("ab") as handle:
                        last_progress = time.monotonic()
                        for block in response.iter_content(1024 * 1024):
                            if not block:
                                continue
                            if completed + len(block) > expected:
                                raise ValueError("Server supplied bytes beyond requested range")
                            handle.write(block)
                            completed += len(block)
                            if time.monotonic() - last_progress >= 20:
                                handle.flush()
                                print(json.dumps({"range_start": start, "completed": completed, "total": expected}), flush=True)
                                last_progress = time.monotonic()
                    if completed != expected:
                        raise IOError(f"Short range body: {completed}/{expected}")
                attempts.append({"attempt": attempt, "from": offset, "status": "COMPLETE"})
            except (requests.RequestException, IOError) as error:
                # Append only to this validated range; retain interrupted bytes.
                completed = path.stat().st_size if path.exists() else 0
                attempts.append({"attempt": attempt, "from": offset, "error": str(error)})
                if attempt == 3:
                    raise
        if completed != expected or path.stat().st_size != expected:
            raise IOError("Incomplete range after bounded attempts")
        result = {"start": start, "end": end, "bytes": expected, "path": str(path), "sha256": digest(path), "elapsed_seconds": time.monotonic() - began, "attempts": attempts}
        path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"range_complete": start, "bytes": expected}), flush=True)
        return result

    results = []
    try:
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [executor.submit(fetch, bounds) for bounds in ranges]
            for future in as_completed(futures):
                results.append(future.result())
                record["completed_ranges"] = sorted(results, key=lambda r: r["start"])
                receipt_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if prefix.stat().st_size != initial or digest(prefix) != record["prefix_sha256"]:
            raise RuntimeError("Original prefix changed; it must have no active writer")
        merged = target.with_suffix(".zip.assembling")
        if merged.exists():
            raise FileExistsError(merged)
        with merged.open("xb") as stream:
            with prefix.open("rb") as source:
                shutil.copyfileobj(source, stream, 1024 * 1024)
            next_start = initial
            for result in sorted(results, key=lambda r: r["start"]):
                if result["start"] != next_start:
                    raise RuntimeError("Non-contiguous range receipts")
                path = Path(result["path"])
                if digest(path) != result["sha256"]:
                    raise RuntimeError("Range changed before assembly")
                with path.open("rb") as source:
                    shutil.copyfileobj(source, stream, 1024 * 1024)
                next_start = result["end"] + 1
        if merged.stat().st_size != asset["size"] or next_start != asset["size"]:
            raise RuntimeError("Assembled archive size mismatch")
        final_sha = digest(merged)
        merged.rename(target)
        record.update({"status": "COMPLETE_SIZE_AND_RANGE_VALIDATED", "archive_path": str(target), "archive_sha256": final_sha,
                       "archive_bytes": target.stat().st_size, "integrity_scope": "Pinned public asset ID and size; exact verified byte ranges plus local SHA. Publisher digest is unavailable. ZIP CRC/content verification is a separate preparation step."})
    except BaseException as error:
        record.update({"status": "FAILED_PARTS_PRESERVED", "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "archive_sha256": record["archive_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
