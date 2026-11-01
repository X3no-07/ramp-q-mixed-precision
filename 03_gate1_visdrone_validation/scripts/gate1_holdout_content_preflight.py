"""Prepare a bounded content-quality preflight; actual content needs new consent.

Preparation reads only the completed authorized central-directory inventory and
named saved metadata. Actual future execution decodes images and parses original
annotations inside the ZIP, without extraction, models, NMS, matching or AP.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/gate1/20261025_holdout_content_preflight_preparation_v1'
INVENTORY_PREP = ROOT / 'artifacts/gate1/20261025_holdout_inventory_preparation_v1'
ASSIGNMENT = ROOT / 'artifacts/gate1/20261022_split_refinement/split_plan_v1/split_assignment.jsonl'
SEAL = ROOT / 'artifacts/gate1/20261022_preparation/sealed_holdout/SEAL.json'
ARCHIVE_SHA = 'b28a94b06dfd9e36ce77ff8155fb82b9d2c030f198a76105933bd56c6ea6a68d'
ARCHIVE_BYTES = 311251787


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def info(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value).decode() + '\n')


def budget():
    return {'attempts': 1, 'timeout_seconds': 1800, 'automatic_retry': False,
        'fixed_image_annotation_pairs': 1610, 'maximum_image_member_bytes': 32000000,
        'maximum_annotation_member_bytes': 8000000, 'maximum_decoded_pixels_per_image': 50000000,
        'maximum_decompressed_member_bytes_total': 400000000, 'maximum_new_output_bytes': 100000000,
        'minimum_free_output_bytes': 2000000000, 'outer_container_hash_passes': 2,
        'extracted_or_copied_original_members': 0, 'model_calibration_NMS_matching_AP_calls': 0}


def scope():
    return {'role': 'FIXED_1610_HOLDOUT_CONTENT_IDENTITY_AND_ANNOTATION_QUALITY_ONLY',
        'ZIP_member_image_and_original_annotation_reads': True, 'image_decode_for_dimensions_and_pixel_identity': True,
        'original_annotation_structure_category_score_ignore_geometry_validation': True,
        'cross_role_image_byte_SHA_and_ID_prefix_checks': True, 'within_cohort_byte_and_decoded_pixel_duplicate_checks': True,
        'member_extraction_copy_or_image_display': False, 'model_calibration_NMS_matching_AP': False,
        'automatic_member_exclusion_relabeling_or_choosing_duplicate_annotation': False,
        'scientific_or_deployment_acceptance': False, 'further_model_execution_authorized': False}


def fixed_pairs(inventory):
    require(inventory['member_payload_reads'] == 0 and inventory['filename_paired_image_annotation_count'] == 1610,
        'Only completed fixed1610 metadata inventory admitted')
    images, annotations = {}, {}
    for row in inventory['records']:
        require(row['payload_opened'] is False and row['member_SHA256'] is None, 'Unexpected prior payload access claim')
        if row['role'] in ('image', 'annotation'):
            name = row['archive_name']
            require(not name.startswith(('/', '\\')) and '\\' not in name and ':' not in name
                and all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe fixed member name')
            target = images if row['role'] == 'image' else annotations
            key = row['logical_id']
            require(key not in target, 'Duplicate fixed logical identity')
            target[key] = row
    require(set(images) == set(annotations) and len(images) == 1610, 'Fixed full member pairing changed')
    return [{'image_id': 'testdev/' + key, 'image': images[key], 'annotation': annotations[key]}
            for key in sorted(images)]


def sources():
    authorized = read(INVENTORY_PREP / 'authorized_contract.json')
    inventory_out = Path(authorized['output_directory'])
    return {'runner': Path(__file__), 'tests_source': ROOT / 'tests/test_gate1_holdout_content_preflight.py',
        'tests_receipt': OUT / 'tests_receipt.json', 'SEAL': SEAL, 'assignment_metadata': ASSIGNMENT,
        'completed_inventory': inventory_out / 'central_directory_inventory.json',
        'inventory_completion': inventory_out / 'completion_receipt.json',
        'inventory_saved_audit': ROOT / 'artifacts/gate1/20261025_holdout_inventory_delivery_v1/inventory_saved_audit.json',
        'original_annotation_parser': ROOT / 'scripts/gate1_selection_members.py'}


def prepare():
    paths = sources()
    tests = read(paths['tests_receipt'])
    require(tests['errors'] == tests['failures'] == 0 and tests['runner'] == info(__file__)
        and tests['test_source'] == info(paths['tests_source']), 'Current pure tests/source binding required')
    completion = read(paths['inventory_completion'])
    require(completion['status'] == 'COMPLETED_AUTHORIZED_CONTAINER_AND_CENTRAL_DIRECTORY_INVENTORY_ONLY'
        and completion['outer_container_SHA256'] == ARCHIVE_SHA and completion['outer_container_bytes'] == ARCHIVE_BYTES,
        'Completed fixed inventory identity differs')
    pairs = fixed_pairs(read(paths['completed_inventory']))
    previous = read(INVENTORY_PREP / 'authorized_contract.json')
    parent = Path(previous['output_directory']).parent
    contract = {'schema': 'rampq.gate1.Holdout_content_quality_preparation/v1',
        'status': 'PREPARED_NOT_AUTHORIZED_NO_CONTENT_READ', 'created_at_utc': now(),
        'source_bindings': {name: info(path) for name, path in paths.items()}, 'scope': scope(), 'budget': budget(),
        'fixed_pairs': pairs, 'member_order_sha256': hashlib.sha256(canonical(pairs)).hexdigest(),
        'outer_container': {'bytes': ARCHIVE_BYTES, 'sha256': ARCHIVE_SHA, 'publisher_digest': None},
        'output_directory': str(parent / 'Holdout_content_preflight_attempt_01'),
        'duplicate_policy': 'Retain original full1610 identities/order; identical content is a dependency group. Any byte identity leakage across roles or annotation conflict for identical bytes/pixels STOP; never auto-exclude or relabel.',
        'image_decode': 'Pillow header/EXIF guard then OpenCV IMREAD_COLOR|IMREAD_IGNORE_ORIENTATION; decoded BGR uint8 dimensions+bytes hash',
        'scene_independence_limit': 'Cross-role decoded-pixel/near-duplicate semantic comparison not covered; SHA/IDs/prefixes are risk checks only',
        'scientific_effect_threshold_or_candidate_selection': None,
        'current_member_content_access_authorized': False, 'current_model_execution_authorized': False,
        'actual_original_image_or_label_reads': 0}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'prepared_contract.json', contract)
    write(OUT / 'human_authorization_template.json', {'schema': 'rampq.gate1.Holdout_content_quality_human_decision/v1',
        'decision': 'PENDING', 'actual_access_authorized': False, 'human_message_verbatim': None,
        'conversation_basis': None, 'scope': scope(), 'prepared': info(OUT / 'prepared_contract.json'),
        'further_model_scoring_access': False})
    return {'status': contract['status'], 'prepared': info(OUT / 'prepared_contract.json'), 'actual_member_content_reads': 0}


def validate_decision(decision, prepared):
    require(decision['schema'] == 'rampq.gate1.Holdout_content_quality_human_decision/v1'
        and decision['decision'] == 'AUTHORIZED_FIXED1610_CONTENT_QUALITY_ONLY'
        and decision['actual_access_authorized'] is True and decision['scope'] == scope()
        and decision['prepared'] == prepared and decision['further_model_scoring_access'] is False,
        'Exact new content-quality consent required; inventory consent cannot substitute')
    require(isinstance(decision['human_message_verbatim'], str) and bool(decision['human_message_verbatim'].strip())
        and isinstance(decision['conversation_basis'], str) and bool(decision['conversation_basis'].strip()),
        'Human decision must be recorded externally')


def freeze(prepared_sha, authorization_sha):
    prepared = info(OUT / 'prepared_contract.json')
    authorization = info(OUT / 'human_authorization.json')
    require(prepared['sha256'] == prepared_sha and authorization['sha256'] == authorization_sha, 'Reviewed identities required')
    validate_decision(read(authorization['path']), prepared)
    contract = read(prepared['path'])
    require(contract['source_bindings'] == {name: info(path) for name, path in sources().items()}, 'Prepared sources changed')
    value = {'status': 'FROZEN_AUTHORIZED_CONTENT_QUALITY_ONLY_NOT_EXECUTED', 'prepared': prepared,
        'authorization': authorization, 'budget': budget(), 'scope': scope(), 'frozen_at_utc': now()}
    write(OUT / 'authorized_contract.json', value)
    return info(OUT / 'authorized_contract.json')


def content_identity(image_bytes, annotation_bytes):
    import cv2
    import numpy as np
    from PIL import Image
    import gate1_selection_members as parser
    with Image.open(io.BytesIO(image_bytes)) as header:
        width, height = header.size
        orientation = header.getexif().get(274)
        require(0 < width * height <= budget()['maximum_decoded_pixels_per_image'] and orientation in (None, 1),
            'Decoded-size or nontrivial EXIF orientation guard failed')
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    require(image is not None and image.dtype == np.uint8 and image.shape == (height, width, 3), 'Image decoder shape mismatch')
    physical = parser.parse_physical_annotation(annotation_bytes)
    return {'width': width, 'height': height, 'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
        'decoded_BGR_pixel_sha256': hashlib.sha256(canonical([height, width, 3]) + image.tobytes()).hexdigest(),
        'annotation_sha256': hashlib.sha256(annotation_bytes).hexdigest(),
        'annotation_physical_rows_sha256': hashlib.sha256(canonical(physical)).hexdigest(),
        'annotation_nonempty_rows': sum('values' in row for row in physical), 'EXIF_orientation': orientation}


def check_duplicate(seen, key, item):
    if key in seen:
        previous = seen[key]
        require(previous['annotation_physical_rows_sha256'] == item['annotation_physical_rows_sha256'],
            'Identical image bytes/pixels have conflicting original physical annotation rows: '
            + previous['image_id'] + ' and ' + item['image_id'])
    else:
        seen[key] = item


def worker(authorized_sha):
    require(info(OUT / 'authorized_contract.json')['sha256'] == authorized_sha, 'Exact authorized contract SHA required')
    authorized = read(OUT / 'authorized_contract.json')
    require(authorized['status'] == 'FROZEN_AUTHORIZED_CONTENT_QUALITY_ONLY_NOT_EXECUTED'
        and authorized['budget'] == budget() and authorized['scope'] == scope(), 'Frozen scope changed')
    for binding in (authorized['prepared'], authorized['authorization']):
        require(info(binding['path']) == binding, 'Authorized source changed')
    validate_decision(read(authorized['authorization']['path']), authorized['prepared'])
    p = read(authorized['prepared']['path'])
    require(p['source_bindings'] == {name: info(path) for name, path in sources().items()}, 'Frozen preparation changed')
    output = Path(p['output_directory'])
    require((output / 'launcher_receipt.json').is_file(), 'Bounded launcher required')
    seen_bytes, seen_pixels, rows = {}, {}, []
    try:
        archive_path = Path(read(SEAL)['archive']['path'])
        require(info(archive_path)['sha256'] == ARCHIVE_SHA and archive_path.stat().st_size == ARCHIVE_BYTES, 'Outer archive changed')
        assignments = [json.loads(line) for line in ASSIGNMENT.read_text(encoding='utf-8').splitlines()]
        prior_sha = {row['image_sha256'] for row in assignments}
        total_read = 0
        with zipfile.ZipFile(archive_path) as archive, (output / 'member_fingerprints.jsonl').open('x', encoding='utf-8') as stream:
            for index, pair in enumerate(p['fixed_pairs'], 1):
                blobs = {}
                for kind, cap in (('image', 'maximum_image_member_bytes'), ('annotation', 'maximum_annotation_member_bytes')):
                    member = pair[kind]
                    current = archive.getinfo(member['archive_name'])
                    require(current.file_size == member['declared_bytes'] and current.CRC == int(member['CRC32_hex'], 16)
                        and current.file_size <= budget()[cap], 'Declared member identity or size changed')
                    with archive.open(current) as handle:
                        blob = handle.read(budget()[cap] + 1)
                    require(len(blob) == current.file_size, 'Member exceeds cap or decompressed size differs')
                    total_read += len(blob)
                    require(total_read <= budget()['maximum_decompressed_member_bytes_total'], 'Member read budget exceeded')
                    blobs[kind] = blob
                row = {'image_id': pair['image_id'], **content_identity(blobs['image'], blobs['annotation'])}
                require(row['image_sha256'] not in prior_sha, 'Cross-role actual image SHA leakage: ' + pair['image_id'])
                check_duplicate(seen_bytes, row['image_sha256'], row)
                check_duplicate(seen_pixels, row['decoded_BGR_pixel_sha256'], row)
                rows.append(row)
                stream.write(canonical(row).decode() + '\n')
                if index % 50 == 0:
                    stream.flush()
                    require(shutil.disk_usage(output).free >= budget()['minimum_free_output_bytes'], 'Output reserve exhausted')
                    print(json.dumps({'stage': 'CONTENT_IDENTITY_QUALITY_ONLY', 'images': index, 'total': 1610}), flush=True)
        require(len(rows) == 1610 and info(archive_path)['sha256'] == ARCHIVE_SHA, 'Full cohort or final archive identity differs')
        require(p['source_bindings'] == {name: info(path) for name, path in sources().items()}, 'Sources changed during preflight')
        completion = {'status': 'COMPLETED_FIXED1610_CONTENT_IDENTITY_QUALITY_ONLY', 'authorized_contract': info(OUT / 'authorized_contract.json'),
            'members': len(rows), 'unique_image_bytes': len(seen_bytes), 'unique_decoded_pixel_images': len(seen_pixels),
            'decompressed_member_bytes_read': total_read, 'original_members_extracted_or_copied': 0,
            'cross_role_raw_image_SHA_intersections': 0, 'cross_role_near_duplicate_or_semantic_scene_isolation': 'UNKNOWN_NOT_VERIFIED',
            'model_calibration_NMS_matching_AP_calls': 0, 'scientific_acceptance': False, 'further_model_execution_authorized': False,
            'member_fingerprints': info(output / 'member_fingerprints.jsonl'), 'ended_at_utc': now()}
        write(output / 'completion_receipt.json', completion)
        return completion
    except Exception as error:
        write(output / 'failure_receipt.json', {'status': 'FAILED_CONTENT_QUALITY_PRESERVED_NO_RETRY', 'error_type': type(error).__name__,
            'error': str(error), 'completed_members': len(rows), 'model_AP_calls': 0, 'ended_at_utc': now()})
        raise


def run(authorized_sha):
    require(info(OUT / 'authorized_contract.json')['sha256'] == authorized_sha, 'Exact authorized contract SHA required')
    auth = read(OUT / 'authorized_contract.json')
    validate_decision(read(auth['authorization']['path']), auth['prepared'])
    p = read(auth['prepared']['path'])
    output = Path(p['output_directory'])
    require(output.name == 'Holdout_content_preflight_attempt_01' and output.parent == Path(read(INVENTORY_PREP / 'authorized_contract.json')['output_directory']).parent,
        'Bound new output directory required')
    require(shutil.disk_usage(output.parent).free >= budget()['minimum_free_output_bytes'], 'Output reserve too low')
    output.mkdir(exist_ok=False)
    receipt = {'status': 'RUNNING_CONTENT_QUALITY_ONLY', 'authorized_contract': info(OUT / 'authorized_contract.json'), 'started_at_utc': now()}
    write(output / 'launcher_receipt.json', receipt)
    try:
        with (output / 'worker_console.txt').open('x', encoding='utf-8') as console:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--authorized-sha', authorized_sha],
                cwd=ROOT, stdout=console, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                code = process.wait(timeout=budget()['timeout_seconds'])
            except subprocess.TimeoutExpired:
                if os.name == 'nt' and process.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                else:
                    process.kill()
                process.wait(timeout=30)
                raise
        receipt['returncode'] = code
        require(code == 0 and (output / 'completion_receipt.json').is_file(), 'Preflight did not complete; no automatic retry')
        require(sum(path.stat().st_size for path in output.iterdir() if path.is_file()) <= budget()['maximum_new_output_bytes'], 'New metadata output cap exceeded')
        receipt['status'] = 'COMPLETED_BOUNDED_CONTENT_QUALITY_ONLY'
    except Exception as error:
        receipt.update(status='FAILED_OR_TIMEOUT_CONTENT_QUALITY_NO_RETRY', error=repr(error))
        raise
    finally:
        receipt['ended_at_utc'] = now()
        with (output / 'launcher_receipt.json').open('w', encoding='utf-8') as stream:
            stream.write(canonical(receipt).decode() + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'run', 'worker'))
    parser.add_argument('--prepared-sha'); parser.add_argument('--authorization-sha'); parser.add_argument('--authorized-sha')
    args = parser.parse_args()
    if args.mode == 'prepare': result = prepare()
    elif args.mode == 'freeze': result = freeze(args.prepared_sha, args.authorization_sha)
    elif args.mode == 'run': result = run(args.authorized_sha)
    else: result = worker(args.authorized_sha)
    print(json.dumps(result, ensure_ascii=False), flush=True)
