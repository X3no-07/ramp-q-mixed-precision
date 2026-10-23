"""Download one pinned official portable Octave archive, retaining failed bytes.

No extraction, installation, PATH change, registry edit or native execution.
Only a checksum matching the official release asset is accepted.  The detached
signature is retained, but is not a claim of a verified PGP trust chain.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = (
    ("octave-11.3.0-w64.7z", "fd3cf0e885467a15211b8ceded42800432a5462481324d49a896ef257e05d1a0"),
    ("octave-11.3.0-w64.7z.sig", "333a02ab7a7cbfb697e525a0c9b2af2e83b14e3d0a3b82299e5634b7f54042ce"),
)
BASE = "https://github.com/gnu-octave/octave/releases/download/release-11-3-0/"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    downloads = HERE / "downloads"
    downloads.mkdir(exist_ok=True)
    log = HERE / "download_events.jsonl"
    accepted = []
    for name, expected in ASSETS:
        target = downloads / name
        partial = downloads / (name + ".partial")
        if target.exists():
            assert digest(target) == expected, "Existing completed asset identity differs"
            accepted.append({"name": name, "bytes": target.stat().st_size, "sha256": expected})
            continue
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "RAMPQ-native-oracle-portable-preparation/1", "Cache-Control": "no-cache"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        # Cache-busting avoids an expired cached GitHub signed redirect.  The
        # canonical source and expected SHA remain fixed above.
        url = BASE + name + "?rampq_request=" + str(time.time_ns())
        event = {"name": name, "canonical_url": BASE + name, "resume_from": offset,
                 "timestamp_unix": time.time(), "pgp_signature_verified": False}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                status = response.status
                if offset and status != 206:
                    raise RuntimeError("Resume was not honored; partial file preserved, no overwrite")
                event.update({"http_status": status, "content_length": response.headers.get("Content-Length"),
                              "content_range": response.headers.get("Content-Range"),
                              "response_host": urllib.parse.urlparse(response.url).hostname})
                received = offset
                checkpoint = received
                with partial.open("ab" if offset else "xb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        received += len(block)
                        if received - checkpoint >= 32 * 1024 * 1024:
                            stream.flush()
                            print(f"DOWNLOAD_BYTES {name}: {received}", flush=True)
                            checkpoint = received
            actual = digest(partial)
            assert actual == expected, "Downloaded asset SHA-256 mismatch; partial retained"
            partial.rename(target)
            event.update({"status": "DOWNLOADED_SHA256_VERIFIED", "bytes": received, "sha256": actual})
            accepted.append({"name": name, "bytes": received, "sha256": actual})
        except Exception as error:
            event.update({"status": "FAILED_PARTIAL_PRESERVED", "error": repr(error),
                          "partial_bytes": partial.stat().st_size if partial.exists() else 0})
            raise
        finally:
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    receipt = {"schema": "rampq.gate1.octave_portable_download/v1", "assets": accepted,
               "official_checksum_source": "https://github.com/gnu-octave/octave/releases/expanded_assets/release-11-3-0",
               "pgp_signature_verified": False, "extracted": False, "installed": False,
               "system_path_modified": False, "registry_modified": False, "native_runtime_executed": False}
    output = HERE / "download_receipt.json"
    assert not output.exists(), "Refusing to overwrite receipt"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("OFFICIAL_ASSETS_SHA256_VERIFIED_NO_EXTRACTION", flush=True)


if __name__ == "__main__":
    main()
