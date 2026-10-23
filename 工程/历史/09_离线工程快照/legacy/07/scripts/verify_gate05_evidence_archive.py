"""Verify a received board ZIP and every receipt file before any native audit.

Default operation is read-only. --extract-to creates a fresh directory only;
existing input/evidence files are never overwritten. The expected ZIP SHA256
must come from the board collector's recorded output, not from this command.
This verifies file integrity, not process termination, engine precision, INT8
tactics, successful inference, numerical migration, or any scientific criterion.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import unicodedata
import zipfile


RECEIPT_SCHEMA = 'rampq.remote_file_receipt/v1'
DEVICE_NAMES = {'CON', 'PRN', 'AUX', 'NUL'} | {
    prefix + str(number) for prefix in ('COM', 'LPT') for number in range(1, 10)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_stream(handle):
    digest = hashlib.sha256()
    size = 0
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            return digest.hexdigest(), size
        digest.update(block)
        size += len(block)


def file_digest(path):
    with path.open('rb') as handle:
        return sha_stream(handle)


def safe_name(name, directory=False):
    require(isinstance(name, str) and bool(name), 'Empty/non-string ZIP or receipt path')
    require('\\' not in name and not any(ord(char) < 32 for char in name), 'Unsafe path: ' + repr(name))
    normalized = name[:-1] if directory and name.endswith('/') else name
    path = PurePosixPath(normalized)
    require(normalized and not path.is_absolute() and str(path) == normalized and
            all(part not in ('', '.', '..') for part in path.parts), 'Non-canonical relative path: ' + repr(name))
    for part in path.parts:
        require(not any(char in '<>:"|?*' for char in part) and not part.endswith((' ', '.')),
                'Windows-unsafe path component: ' + repr(name))
        require(part.split('.')[0].upper() not in DEVICE_NAMES, 'Reserved Windows device path: ' + repr(name))
    return path.as_posix()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON object key: ' + repr(key))
        result[key] = value
    return result


def inventory(archive):
    entries, folded = {}, {}
    for info in archive.infolist():
        name = safe_name(info.filename, info.is_dir())
        collision = unicodedata.normalize('NFC', name).casefold()
        require(collision not in folded, 'Duplicate/case-colliding ZIP member: ' + name)
        folded[collision] = name
        require(not info.flag_bits & 1, 'Encrypted ZIP members are unsupported')
        kind = stat.S_IFMT(info.external_attr >> 16)
        allowed = (0, stat.S_IFDIR) if info.is_dir() else (0, stat.S_IFREG)
        require(kind in allowed, 'ZIP links or special files are forbidden: ' + name)
        if not info.is_dir():
            entries[name] = info
    file_names = {unicodedata.normalize('NFC', name).casefold() for name in entries}
    for name in entries:
        require(not any(unicodedata.normalize('NFC', parent.as_posix()).casefold() in file_names
                        for parent in PurePosixPath(name).parents
                        if parent.as_posix() != '.'), 'ZIP file is also a parent directory: ' + name)
    return entries


def verify_archive(path, expected_sha256, expected_bytes=None, receipt_name='remote_receipt.json'):
    require(re.fullmatch('[0-9a-fA-F]{64}', expected_sha256) is not None, 'Expected a recorded ZIP SHA256')
    actual_sha256, actual_bytes = file_digest(path)
    require(actual_sha256 == expected_sha256.lower(), 'Received ZIP SHA256 differs from the recorded board value')
    if expected_bytes is not None:
        require(type(expected_bytes) is int and expected_bytes >= 0 and expected_bytes == actual_bytes,
                'Received ZIP byte size differs from the recorded board value')
    receipt_name = safe_name(receipt_name)
    with zipfile.ZipFile(path) as archive:
        entries = inventory(archive)
        require(receipt_name in entries, 'Receipt is absent from the archive')
        require(entries[receipt_name].file_size <= 16 * 1024 * 1024, 'Receipt exceeds the supported 16 MiB metadata size')
        receipt_bytes = archive.read(receipt_name)
        receipt = json.loads(receipt_bytes.decode('utf-8-sig'), object_pairs_hook=unique_object)
        require(receipt.get('schema') in (None, RECEIPT_SCHEMA), 'Unsupported receipt schema')
        require(isinstance(receipt.get('files'), dict) and bool(receipt['files']), 'Receipt has no file inventory')
        parent = PurePosixPath(receipt_name).parent
        required, verification = {}, []
        for relative, info in receipt['files'].items():
            relative = safe_name(relative)
            member = (parent / relative).as_posix()
            require(member != receipt_name, 'A receipt cannot hash itself')
            require(isinstance(info, dict) and type(info.get('bytes')) is int and info['bytes'] >= 0,
                    'Invalid receipt byte size: ' + relative)
            require(isinstance(info.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', info['sha256']) is not None,
                    'Invalid receipt SHA256: ' + relative)
            required[member] = info
        require(set(entries) == set(required) | {receipt_name},
                'ZIP file inventory differs from receipt; missing=' + repr(sorted(set(required) - set(entries))) +
                '; unlisted=' + repr(sorted(set(entries) - set(required) - {receipt_name})))
        for member, info in sorted(required.items()):
            require(entries[member].file_size == info['bytes'], 'ZIP/receipt byte size differs: ' + member)
            with archive.open(entries[member]) as handle:
                digest, size = sha_stream(handle)
            require(size == info['bytes'] and digest == info['sha256'], 'Receipt content hash differs: ' + member)
            verification.append({'path': member, 'bytes': size, 'sha256': digest})
    return {'schema': 'rampq.gate05.received_archive_verification/v1',
            'status': 'ARCHIVE_AND_EXACT_RECEIPT_FILE_INVENTORY_VERIFIED',
            'archive': str(path.resolve()), 'archive_bytes': actual_bytes, 'archive_sha256': actual_sha256,
            'receipt_path': receipt_name, 'receipt_schema': receipt.get('schema'),
            'receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
            'verified_file_count': len(verification), 'verified_files': verification,
            'legacy_receipt_schema_missing': receipt.get('schema') is None,
            'native_process_termination_verified': False, 'native_precision_audit_run': False,
            'strict_int8_tactics_verified': False, 'inference_run': False,
            'numerical_migration_claim': False, 'dataset_evaluation_run': False,
            'limits': 'File integrity only. Root must independently confirm terminal native processes, '
                      'audit actual engine/source precision, and verify any fixed-image execution. '
                      'Formats-only evidence is not strict integer-tactic verification.'}


def extract_verified(path, destination, report):
    destination = destination.absolute()
    require(not destination.exists() and not destination.is_symlink(), 'Extraction requires a fresh destination')
    require(destination.parent.is_dir(), 'Extraction parent directory must already exist')
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + '.verified-', dir=str(destination.parent)))
    # Failed extraction leaves its fresh temporary directory for inspection. No
    # recursive deletion or overwrite of an existing evidence directory occurs.
    with zipfile.ZipFile(path) as archive:
        entries = inventory(archive)
        expected = {row['path']: row for row in report['verified_files']}
        expected[report['receipt_path']] = {'sha256': report['receipt_sha256']}
        require(set(entries) == set(expected), 'ZIP contents changed after verification')
        for name, info in sorted(entries.items()):
            output = temporary.joinpath(*PurePosixPath(name).parts)
            require(temporary.resolve() in output.resolve().parents, 'Extraction path escaped the fresh directory')
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output.open('xb') as target:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
            digest, size = file_digest(output)
            require(digest == expected[name]['sha256'] and
                    ('bytes' not in expected[name] or size == expected[name]['bytes']),
                    'Extracted content differs: ' + name)
    require(file_digest(path) == (report['archive_sha256'], report['archive_bytes']),
            'Source archive changed during verification/extraction')
    require(not destination.exists(), 'Destination appeared during extraction; preserve both directories')
    temporary.rename(destination)
    return str(destination.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True, help='ZIP SHA256 recorded by the native board collector')
    parser.add_argument('--expected-bytes', type=int, help='Optional ZIP byte count recorded by the board collector')
    parser.add_argument('--receipt', default='remote_receipt.json')
    parser.add_argument('--extract-to', type=Path, help='Optional fresh extraction directory; omitted means read-only verification')
    parser.add_argument('--report', type=Path, help='Optional new local verification JSON; never overwrites evidence')
    args = parser.parse_args()
    if args.report is not None:
        require(not args.report.exists() and args.report.parent.is_dir(), 'Report requires a fresh file in an existing directory')
    report = verify_archive(args.archive, args.expected_sha256, args.expected_bytes, args.receipt)
    if args.extract_to is not None:
        report['extracted_to'] = extract_verified(args.archive, args.extract_to, report)
    if args.report is not None:
        with args.report.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'verified_files'}, indent=2))


if __name__ == '__main__':
    main()
