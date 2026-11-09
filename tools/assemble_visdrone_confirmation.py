"""Restore and verify the VisDrone prediction archive from its release parts."""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


def check_predictions(archive_path, manifest_path):
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"results/" + record["path"]: record for record in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate image output path in the manifest")
    seen = set()
    with tarfile.open(archive_path, "r:") as archive:
        for entry in archive:
            if entry.isdir():
                continue
            if not entry.isfile() or entry.name not in expected or entry.name in seen:
                raise ValueError(f"Unexpected archive member {entry.name}")
            record = expected[entry.name]
            with archive.extractfile(entry) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if entry.size != record["bytes"] or digest != record["sha256"]:
                raise ValueError(f"Image output failed verification: {entry.name}")
            seen.add(entry.name)
    if seen != set(expected):
        raise ValueError("Prediction archive is incomplete")
    return len(seen)


def assemble(parts_directory, output, release_manifest, image_manifest):
    archive = release_manifest["archive"]
    temporary = output.with_name(output.name + ".assembling")
    if output.exists():
        raise FileExistsError(output)
    archive_hash = hashlib.sha256()
    total_bytes = 0
    with temporary.open("xb") as destination:
        for part in release_manifest["parts"]:
            name = part["name"]
            if not re.fullmatch(r"visdrone-20261108-predictions\.tar\.part\d{3}", name):
                raise ValueError(f"Invalid release part name: {name}")
            part_hash = hashlib.sha256()
            part_bytes = 0
            with (parts_directory / name).open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    destination.write(block)
                    part_hash.update(block)
                    archive_hash.update(block)
                    part_bytes += len(block)
            if part_bytes != part["bytes"] or part_hash.hexdigest() != part["sha256"]:
                raise ValueError(f"Release part failed verification: {name}")
            total_bytes += part_bytes
    if total_bytes != archive["bytes"] or archive_hash.hexdigest() != archive["sha256"]:
        raise ValueError("Assembled archive failed verification")
    count = check_predictions(temporary, image_manifest)
    temporary.rename(output)
    return {"archive": str(output), "sha256": archive_hash.hexdigest(), "verified_images": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = Path(__file__).resolve().parents[1] / "results/visdrone_testdev_20261108"
    manifest = json.loads((args.parts_directory / "prediction_archive.json").read_text(encoding="utf-8"))
    result = assemble(args.parts_directory, args.output, manifest, results / "image_output_manifest.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
