"""Verify inherited evidence and seal this completed candidate/check delivery."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def write_new(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def resolve_member(manifest_path, name, row):
    named = Path(name)
    if named.is_absolute():
        return named
    if name.replace('\\', '/').startswith(('artifacts/', 'scripts/', 'tests/')):
        return ROOT / named
    local = manifest_path.parent / named
    if local.is_file():
        return local
    return ROOT / named


def verify_manifest(relative):
    path = ROOT / relative
    doc = read(path)
    count = 0
    for field in ('files', 'external_source_files'):
        for name, row in doc.get(field, {}).items():
            member = resolve_member(path, name, row)
            assert member.is_file(), (relative, name, member)
            assert member.stat().st_size == row['bytes'], (relative, name, 'bytes')
            assert sha(member) == row['sha256'], (relative, name, 'sha256')
            count += 1
    return {'manifest': relative, 'sha256': sha(path), 'members_verified': count}


def main():
    assert not (OUT / 'delivery_manifest.json').exists()
    manifests = [
        'artifacts/gate05/20261021_final_acceptance/final_manifest.json',
        'artifacts/gate1/20261022_preparation/preparation_delivery_manifest.json',
        'artifacts/gate1/20261022_split_refinement/refinement_and_training_launch_delivery_manifest.json',
        'artifacts/gate1/20261022_split_refinement/static_prelaunch_evidence_index_v3.json',
        'artifacts/gate1/20261022_training_progress_qualification_preparation/delivery_manifest.json',
        'artifacts/gate1/20261022_float_qualification_preparation/preparation_manifest.json',
        'artifacts/gate1/20261022_float_qualification_preparation/source_review/manifest.json',
        'artifacts/gate1/20261022_float_qualification_audit_preparation/manifest.json',
        'artifacts/gate1/20261022_pipeline_readiness_review/manifest.json',
        'artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification/manifest.json',
        'artifacts/gate1/20261022_posttraining_checks_v1/03_qualification_output_audit/manifest.json',
        'artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review/manifest.json',
    ]
    verification = [verify_manifest(name) for name in manifests]
    protocol = read(ROOT / 'artifacts/gate1/20261022_posttraining_execution_protocol/protocol.json')
    for name, row in protocol['source_identities'].items():
        assert identity(ROOT / name) == {'bytes': row['bytes'], 'sha256': row['sha256']}
    pointer = read(ROOT / 'artifacts/gate1/latest_execution_state.json')
    assert pointer['snapshot'] == 'artifacts/gate1/current_execution_state_v5.json'
    assert sha(ROOT / pointer['snapshot']) == pointer['snapshot_sha256']
    decision = read(OUT / 'decision.json')
    for row in decision['evidence']:
        assert identity(ROOT / row['path']) == {'bytes': row['bytes'], 'sha256': row['sha256']}
    for row in decision['checkpoints'].values():
        assert identity(Path(row['path'])) == {'bytes': row['bytes'], 'sha256': row['sha256']}
    review = read(ROOT / 'artifacts/gate1/20261022_posttraining_final_review/delivery_consistency_review_v2.json')
    assert 'PASS' in review['status']
    previous_review_path = ROOT / review['previous_review']['path']
    assert sha(previous_review_path) == review['previous_review']['sha256']
    for name, row in read(previous_review_path)['sources'].items():
        if name == review['current_report']['path']:
            continue  # v2 binds the report with the newly reviewed scope paragraph.
        assert identity(ROOT / name) == {'bytes': row['bytes'], 'sha256': row['sha256']}
    for row in [review['current_report'], *review['new_links_checked']]:
        assert identity(ROOT / row['path']) == {'bytes': row['bytes'], 'sha256': row['sha256']}

    links = []
    report_paths = [ROOT / '44_Gate1_首轮训练与浮点资格检查结果.md']
    report_paths += sorted((ROOT / 'artifacts/gate1/20261022_posttraining_final_review').glob('*.md'))
    report_paths += sorted((ROOT / 'artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review').glob('*.md'))
    for report in report_paths:
        text = report.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^\n)]+)\)', text):
            target = target.strip('<>')
            if target.startswith(('https://', 'http://', '#')):
                continue
            member = Path(target)
            if not member.is_absolute():
                member = report.parent / member
            assert member.exists(), (report, target)
            links.append({'report': report.relative_to(ROOT).as_posix(), 'target': str(member)})

    qa_path = OUT / 'delivery_verification.json'
    write_new(qa_path, {
        'schema': 'rampq.gate1.training_qualification_delivery_verification/v1',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_EVIDENCE_INTEGRITY_AND_REPORT_LINKS_ONLY',
        'verified_manifests': verification, 'protocol_scripts_verified': len(protocol['source_identities']),
        'decision_evidence_identities_verified': len(decision['evidence']),
        'local_link_occurrences_verified': len(links), 'local_links': links,
        'latest_pointer_snapshot_hash_verified': True,
        'dtype_guard_unchanged': decision['same_backend_dtype_guard'],
        'old_gate05_guard_unchanged': decision['old_gate05_numeric_guard'],
        'gate1_scientific_pass': False,
        'data_scope': 'Hash selected existing evidence files and already-materialized debug data only; no ZIP member listing, model execution or image decoding.',
        'limits': 'The 21077-member prepared-data manifest is bound by prior evidence, not rehashed in full here; completed-training audit and qualification audit already verify their used subsets.'
    })

    members = {ROOT / '44_Gate1_首轮训练与浮点资格检查结果.md', ROOT / 'artifacts/gate1/current_execution_state_v5.json'}
    for directory in [
        OUT,
        ROOT / 'artifacts/gate1/20261022_posttraining_final_review',
        ROOT / 'artifacts/gate1/20261022_dtype_guard_diagnostics',
        ROOT / 'artifacts/gate1/20261022_posttraining_execution_protocol',
        ROOT / 'artifacts/gate1/20261022_float_qualification_audit_preparation',
        ROOT / 'artifacts/gate1/20261022_pipeline_readiness_review',
        ROOT / 'artifacts/gate1/20261022_posttraining_checks_v1/01_training_audit',
    ]:
        members.update(p for p in directory.rglob('*') if p.is_file())
    base = ROOT / 'artifacts/gate1/20261022_posttraining_checks_v1'
    members.update(p for p in base.glob('*.txt') if p.is_file())
    members.update(ROOT / name for name in manifests)
    attempt = ROOT / 'artifacts/gate1/20261022_split_refinement/float_training_v2'
    members.update(p for p in attempt.glob('*') if p.is_file())
    members.update(attempt / p for p in ('run/results.csv', 'run/args.yaml', 'run/weights/best.pt', 'run/weights/last.pt'))
    members.update(p for p in (ROOT / 'artifacts/gate1/20261022_split_refinement/float_training_launch_v2').glob('*') if p.is_file())
    members.update(ROOT / name for name in protocol['source_identities'])
    members.add(ROOT / 'tests/test_gate1_float_qualification_audit.py')
    frozen = {p.relative_to(ROOT).as_posix(): identity(p) for p in sorted(members)}
    manifest = OUT / 'delivery_manifest.json'
    write_new(manifest, {
        'schema': 'rampq.gate1.training_qualification_delivery_manifest/v1',
        'at_utc': datetime.now(timezone.utc).isoformat(),
        'status': decision['status'],
        'scope': 'Completed 30-epoch float candidate, real 109-image debug qualification, independent saved-output consistency audit and bounded raw-only diagnosis; not an accepted scientific baseline.',
        'files': frozen,
        'nested_qualification_manifest': 'artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification/manifest.json',
        'nested_qualification_members_verified': 454,
        'mutable_latest_state_pointer_not_sealed': True,
        'same_backend_dtype_guard': 'FAIL_FIXED_TOLERANCES',
        'old_gate05_numeric_guard': 'FAILED_UNCHANGED',
        'gate1_scientific_pass': False,
        'selection_detector_inference': False, 'holdout_accessed': False,
        'later_precision_probe_executed': False,
        'note': 'Earlier report/state/review snapshots retain their original identities. The v2 consistency review binds the final report with the scope-addendum links.'
    })
    for name, row in read(manifest)['files'].items():
        assert identity(ROOT / name) == row, name
    print(json.dumps({'status': 'SEALED_AND_RECHECKED', 'delivery_files': len(frozen),
                      'manifest_sha256': sha(manifest), 'links_checked': len(links),
                      'nested_manifests_checked': len(verification),
                      'dtype_status': decision['same_backend_dtype_guard']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
