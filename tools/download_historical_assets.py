"""Download and verify historical ZIPs, including archives stored as segments."""
import argparse
import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def matches(path, size, digest):
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def check_name(value):
    if not isinstance(value, str) or not value.isascii() or Path(value).name != value or '/' in value or '\\' in value:
        raise ValueError('Unsafe artifact filename')
    if not re.fullmatch(r'rampq-results-\d{8}-part\d{3}\.zip(?:\.seg\d{3}-of-\d{3})?', value):
        raise ValueError('Unexpected artifact filename: ' + value)
    return value


def validate(manifest):
    if not manifest.get('planned_archives') or manifest.get('available_archives') != manifest.get('planned_archives'):
        raise ValueError('The archive inventory is incomplete')
    seen = set()
    for archive in manifest['assets']:
        name = check_name(archive['name'])
        if name in seen:
            raise ValueError('Duplicate archive name')
        seen.add(name)
        if not re.fullmatch('[0-9a-f]{64}', archive['expected_sha256_from_historical_manifest']):
            raise ValueError('Invalid archive digest')
        if archive['download_mode'] == 'direct':
            continue
        if archive['download_mode'] != 'segmented' or not archive.get('segments'):
            raise ValueError('Unsupported download mode')
        total = 0
        for index, part in enumerate(archive['segments'], 1):
            check_name(part['name'])
            if part['index'] != index or part['offset'] != total or part['bytes'] <= 0:
                raise ValueError('Invalid segment order or size')
            if not part['name'].startswith(name + '.seg') or not re.fullmatch('[0-9a-f]{64}', part['sha256']):
                raise ValueError('Invalid segment identity')
            total += part['bytes']
        if total != archive['expected_bytes']:
            raise ValueError('Segment sizes do not reconstruct the ZIP size')


def download(url, destination, size, digest):
    if destination.exists():
        if matches(destination, size, digest):
            return
        raise FileExistsError('Existing file has different contents: ' + str(destination))
    prefix = 'https://github.com/X3no-07/ramp-q-mixed-precision/releases/download/'
    if not url.startswith(prefix):
        raise ValueError('Unexpected download host or repository')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.', suffix='.partial', delete=False) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(url, headers={'User-Agent': 'RAMP-Q-artifact-downloader'})
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if not matches(temporary, size, digest):
            raise ValueError('Downloaded file failed size/SHA256 validation: ' + destination.name)
        if destination.exists():
            raise FileExistsError('Destination appeared during download: ' + str(destination))
        temporary.rename(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore(archive, output_dir):
    name = check_name(archive['name'])
    destination = output_dir / name
    size = archive['expected_bytes']
    digest = archive['expected_sha256_from_historical_manifest']
    if destination.exists():
        if matches(destination, size, digest):
            print('Verified existing ZIP: ' + name)
            return
        raise FileExistsError('Existing ZIP has different contents: ' + str(destination))
    if archive['download_mode'] == 'direct':
        download(archive['download_url'], destination, size, digest)
    else:
        cache = output_dir / '.segments' / name
        for segment in archive['segments']:
            download(segment['download_url'], cache / check_name(segment['name']), segment['bytes'], segment['sha256'])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output_dir, prefix=name + '.', suffix='.assembling', delete=False) as output:
                temporary = Path(output.name)
                for segment in archive['segments']:
                    with (cache / segment['name']).open('rb') as source:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            if not matches(temporary, size, digest):
                raise ValueError('Reassembled ZIP failed the original SHA256 check: ' + name)
            if destination.exists():
                raise FileExistsError('Destination appeared during assembly: ' + str(destination))
            temporary.rename(destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print('Verified ZIP: ' + name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'export/ARTIFACT_AVAILABILITY.json')
    parser.add_argument('--output-dir', type=Path, default=Path('.release-downloads'))
    parser.add_argument('--part', action='append', type=int, help='Download only this part number; repeat to select multiple ZIPs.')
    parser.add_argument('--list', action='store_true', help='Show available archives without downloading.')
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    validate(manifest)
    selected = [a for a in manifest['assets'] if not args.part or a['part'] in args.part]
    if not selected or (args.part and set(args.part) - {a['part'] for a in selected}):
        parser.error('Requested archive part is not in the manifest')
    if args.list:
        for archive in selected:
            print(f"{archive['part']:03d}  {archive['expected_bytes']} bytes  {archive['download_mode']}  {archive['name']}")
        return 0
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / '.rampq-download.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write('RAMP-Q download in progress. Inspect before removing a stale lock.\n')
    try:
        for archive in selected:
            restore(archive, output_dir)
    finally:
        lock.unlink()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
