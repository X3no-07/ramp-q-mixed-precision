"""Fetch the one approved official wheel; unpack only inside this new tree."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
URL = "https://files.pythonhosted.org/packages/39/e7/979fd14e75008623df31ba70d6bb144700f68feadcea042021c06a05bf82/scipy-1.18.1-cp312-cp312-win_amd64.whl"
SHA256 = "5e4d44984abc0020154ea81b247adeddcc3ac5527b975ff798bd1ba0adc513c2"
EXPECTED_BYTES = 36658278


def main():
    wheel = HERE / URL.rsplit("/", 1)[-1]
    assert not wheel.exists(), "New wheel path required"
    with urllib.request.urlopen(URL, timeout=60) as source, wheel.open("xb") as destination:
        while block := source.read(1024 * 1024):
            destination.write(block)
    assert wheel.stat().st_size == EXPECTED_BYTES
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == SHA256
    target = HERE / "mat_reader_dependencies"
    assert not target.exists(), "New isolated dependency target required"
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        total = sum(member.file_size for member in members)
        receipt = {"schema": "rampq.gate1.isolated_mat_reader_dependency/v1", "official_metadata_url": "https://pypi.org/pypi/scipy/json",
                   "url": URL, "wheel_bytes": EXPECTED_BYTES, "wheel_sha256": SHA256, "member_count": len(members),
                   "total_uncompressed_bytes": total, "environment_modified": False, "dependencies_downloaded": False,
                   "status": "VERIFIED_WHEEL_PENDING_UNPACK_SIZE_GATE"}
        (HERE / "mat_reader_wheel_listing.json").write_text(json.dumps({**receipt, "members": [{"path": m.filename, "bytes": m.file_size} for m in members]}, indent=2) + "\n", encoding="utf-8")
        assert total <= 200 * 1024 ** 2, "Approved 200 MiB unpack limit exceeded"
        for member in members:
            resolved = (target / member.filename).resolve()
            assert resolved.is_relative_to(target.resolve()), "Unsafe wheel member path"
        target.mkdir()
        archive.extractall(target)
    receipt.update(status="VERIFIED_SINGLE_WHEEL_ISOLATED_UNPACK_ONLY", target=str(target),
                   scope="CPU MAT-v7 reading only; no packages installed into existing interpreters")
    (HERE / "mat_reader_dependency_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
