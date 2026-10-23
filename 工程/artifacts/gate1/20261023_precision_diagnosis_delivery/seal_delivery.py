"""Seal reviewed reports and nested completed probe evidence, preserving old files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
BASE = ROOT / 'artifacts/gate1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ident(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def new(path, data):
    assert not path.exists(), path
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verify(manifest, workspace_relative=False):
    files = read(manifest)['files']
    entries = files.items() if isinstance(files, dict) else ((row['path'], row) for row in files)
    for name, row in entries:
        path = ROOT / name if workspace_relative else manifest.parent / name
        assert path.is_file() and ident(path) == {'bytes': row['bytes'], 'sha256': row['sha256']}, (manifest, name)
    return {'path': manifest.relative_to(ROOT).as_posix(), 'manifest_sha256': sha(manifest), 'files_verified': len(files)}


def main():
    assert not (OUT / 'delivery_manifest.json').exists()
    decision = read(OUT / 'decision.json')
    assert decision['status'] == 'COMPLETED_LAYERED_ENGINEERING_DIAGNOSIS_ONLY'
    assert decision['original_gate1_dtype_guard'] == 'FAIL_FIXED_TOLERANCES' and not decision['gate1_scientific_pass']
    nested_dirs = [
        '20261023_decode_precision_pilot_v1', '20261023_decode_precision_pilot_audit_v1',
        '20261023_decode_precision_pilot_audit_v2', '20261023_decode_precision_full_v1',
        '20261023_decode_precision_full_audit_v1', '20261023_head_precision_pilot_v1',
        '20261023_head_precision_pilot_audit_v1', '20261023_head_precision_full_v1',
        '20261023_head_precision_full_audit_v1', '20261023_official_evaluator_readiness',
        '20261023_decode_probe_preparation/independent_review',
        '20261023_head_probe_preparation/independent_review',
    ]
    checks = [verify(BASE / name / 'manifest.json') for name in nested_dirs]
    checks += [verify(BASE / '20261022_training_qualification_delivery/delivery_manifest.json', True),
               verify(ROOT / 'artifacts/gate05/20261021_final_acceptance/final_manifest.json', True),
               verify(BASE / '20261022_posttraining_checks_v1/02_float_qualification/manifest.json')]
    for row in decision['evidence']:
        assert ident(ROOT / row['path']) == {'bytes': row['bytes'], 'sha256': row['sha256']}
    pointer = read(BASE / 'latest_execution_state.json')
    assert pointer['snapshot'] == 'artifacts/gate1/current_execution_state_v6.json'
    assert sha(ROOT / pointer['snapshot']) == pointer['snapshot_sha256']
    report = ROOT / '45_Gate1_精度误差分层诊断与评测准备.md'
    review = read(BASE / '20261023_precision_final_review/final_review.json')
    assert review['issues'] == [] and review['status'].startswith('PASS'), review
    # Review must bind the current report and state, using its explicit source map.
    for name, row in review['sources'].items():
        assert ident(ROOT / name) == {'bytes': row['bytes'], 'sha256': row['sha256']}, name
    normalized_review_names = {name.replace('\\', '/') for name in review['sources']}
    assert '45_Gate1_精度误差分层诊断与评测准备.md' in normalized_review_names
    assert 'artifacts/gate1/current_execution_state_v6.json' in normalized_review_names
    source_pins = {
        'scripts/probe_gate1_decode_precision.py': 'f76f3bfd677487fa7d87947f866daaf7407ddd455ad3e4d9e399da719e481ec0',
        'scripts/audit_gate1_decode_probe.py': '04a7d2256ab0a3d90025b2453ef2bc44b043e8b1493b7866c9acc603838dc1ef',
        'scripts/probe_gate1_head_precision.py': '842e16b7b743c6044975ab2faad5725ead19cd8c793575340c9ee81ddaaef23f',
        'scripts/audit_gate1_head_probe.py': 'cadac8b9c00ce4cb6c8063ceeb496ce224530a2c5c0fbdd988007d03995991a9',
    }
    for name, expected in source_pins.items():
        assert sha(ROOT / name) == expected, name
    links = []
    md_paths = [report, BASE / '20261023_official_evaluator_readiness/README.md']
    for md in md_paths:
        for target in re.findall(r'\]\(([^\n)]+)\)', md.read_text(encoding='utf-8')):
            if target.startswith(('https://', 'http://', '#')):
                continue
            path = Path(target.strip('<>'))
            if not path.is_absolute():
                path = md.parent / path
            assert path.exists(), (md, target)
            links.append({'source': md.relative_to(ROOT).as_posix(), 'target': str(path)})
    new(OUT / 'verification.json', {
        'schema': 'rampq.gate1.layered_precision_delivery_verification/v1',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_INTEGRITY_LINKS_AND_REVIEW_ONLY',
        'nested_manifests': checks, 'scripts_verified': source_pins,
        'local_links_checked': len(links), 'local_links': links,
        'mutable_pointer_verified': True, 'original_guard_unchanged': 'FAIL_FIXED_TOLERANCES',
        'scientific_gate_pass': False, 'new_model_calls_by_this_verification': 0,
        'scope': 'Hash existing files and check report/metadata links; no model, image decoding, archive listing or new data scoring.'
    })
    members = {report, BASE / 'current_execution_state_v6.json'}
    for directory in [OUT, BASE / '20261023_decode_probe_preparation', BASE / '20261023_head_probe_preparation',
                      BASE / '20261023_official_evaluator_readiness', BASE / '20261023_saved_decode_trace',
                      BASE / '20261023_precision_final_review']:
        members.update(p for p in directory.rglob('*') if p.is_file())
    for name in nested_dirs:
        members.add(BASE / name / 'manifest.json')
    # Large raw arrays remain transitively pinned by the checked nested manifests.
    for name in ('20261023_decode_precision_pilot_v1', '20261023_decode_precision_full_v1',
                 '20261023_head_precision_pilot_v1', '20261023_head_precision_full_v1',
                 '20261023_decode_precision_pilot_audit_v1', '20261023_decode_precision_pilot_audit_v2',
                 '20261023_decode_precision_full_audit_v1', '20261023_head_precision_pilot_audit_v1',
                 '20261023_head_precision_full_audit_v1'):
        members.update(p for p in (BASE / name).glob('*.json') if p.is_file())
    members.update(ROOT / p for p in source_pins)
    members.update(ROOT / ('tests/' + name) for name in (
        'test_gate1_decode_precision.py', 'test_gate1_decode_probe_audit.py',
        'test_gate1_head_precision.py', 'test_gate1_head_probe_audit.py'))
    files = {p.relative_to(ROOT).as_posix(): ident(p) for p in sorted(members)}
    new(OUT / 'delivery_manifest.json', {
        'schema': 'rampq.gate1.layered_precision_diagnosis_delivery_manifest/v1',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'status': decision['status'],
        'files': files, 'large_probe_arrays_transitively_bound': True,
        'mutable_latest_state_pointer_not_sealed': True,
        'scientific_gate_pass': False, 'accepted_float_baseline_ready': False,
        'original_gate1_dtype_guard': 'FAIL_FIXED_TOLERANCES', 'old_gate05_numeric_guard': 'FAILED_UNCHANGED',
        'selection_detector_inference': False, 'holdout_accessed': False,
        'scope': 'Completed debug decode/head arithmetic crosses and independent saved-array audits plus unexecuted native-oracle fixture preparation. No scientific baseline or deployment acceptance.'
    })
    for name, row in read(OUT / 'delivery_manifest.json')['files'].items():
        assert ident(ROOT / name) == row
    print(json.dumps({'status': 'SEALED_AND_RECHECKED', 'files': len(files), 'nested_manifests_checked': len(checks),
                      'links_checked': len(links), 'manifest_sha256': sha(OUT / 'delivery_manifest.json')}))


if __name__ == '__main__':
    main()
