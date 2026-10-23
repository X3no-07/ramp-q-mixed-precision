"""Freeze direct-image mixed prototype contracts before any new forward."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/gate1'
OUT = Path(__file__).parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ident(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def new(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    runner = ROOT / 'scripts/probe_gate1_body32_head16.py'
    spec = importlib.util.spec_from_file_location('protocol_constants_only_body32_head16', runner)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    review_root = OUT / 'independent_review'
    review = read(review_root / 'review.json')
    assert review['status'].startswith('PASS') and review.get('issues', []) == []
    # The preparation receipt binds the final source and tests. Read no models/pixels here.
    reviewed_sources = read(review_root / 'manifest.json')
    assert reviewed_sources.get('files')
    inherited = read(BASE / '20261023_upstream_probe_preparation/full109_protocol.json')
    macro_audit = read(BASE / '20261023_upstream_frontier_full_audit_v1/recomputed_summary.json')
    assert macro_audit['status'] == 'PASS_SAVED_UPSTREAM_FRONTIER_ARRAYS_ONLY'
    assert macro_audit['images_verified'] == 109
    keys = ['qualifier', 'qualification_dir', 'qualification_manifest', 'checkpoint', 'runtime_environment',
            'framework_source_sha256', 'package_versions', 'controls', 'data_role', 'selection_detector_inference',
            'holdout_accessed', 'historical_equality_required', 'native_reconstruction_required', 'raw_tolerances',
            'old_gate05_guard', 'historical_gate1_dtype_guard', 'head_reg_max_policy', 'automatic_retries',
            'checkpoint_selection', 'scientific_gate_pass', 'accepted_float_baseline_ready', 'decoder_runner',
            'head_runner', 'reference_head_probe_dir', 'reference_head_probe_manifest', 'topology', 'observed_topology_identity']
    common = {key: inherited[key] for key in keys}
    common.update(schema=helper.SCHEMA, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        runner_sha256=sha(runner), guard_name=helper.CHECK_NAME, precision_contract=helper.PRECISION_CONTRACT,
        parameter_buffer_policy='One saved FP16 EMA loaded once; lift to FP32, keep body 0..22 and DFL FP32; only Detect.cv2/cv3 towers and their floating BN buffers become FP16.',
        scope='Fresh direct-image engineering prototype at the already diagnosed head endpoint; not new independent precision selection or acceptance.',
        historical_reconstruction_policy='Fresh input, features, cast features, logits and FP32-decoded endpoint must exactly match the frozen full109 head intervention endpoint; no reference feature is an inference input.',
        new_inference_policy='Exactly one CPU checkpoint deserialization per attempt and one fresh mixed forward per image. Historical numerical arrays are loaded only after forward for comparison. Earlier integrity hashing reads bytes only.',
        comparison_policy='Use every candidate position and unchanged raw tolerances against the historical FP32 reference. Preserve both original failed guards.',
        limits=helper.LIMITS)
    assert common['decoder_runner']['sha256'] == helper.DECODER_SHA
    assert common['head_runner']['sha256'] == helper.HEAD_SHA
    paths = []
    for stage, ids in [('pilot3', helper.PILOT_IDS), ('full109', inherited['image_ids'])]:
        protocol = dict(common, stage=stage, image_ids=ids)
        protocol['execution_condition'] = ('Run these three fixed IDs first; stop and preserve on any source/runtime/input/endpoint mismatch.' if stage == 'pilot3' else
            'Extend only after the pilot completes and an independent saved-output audit confirms identities, endpoint equality and summary consistency. Expand all109 regardless of favorable or unfavorable numeric-check counts; no automatic retries or tolerance changes.')
        path = OUT / f'{stage}_protocol.json'
        new(path, protocol); paths.append(path)
    new(OUT / 'protocol_preparation_receipt.json', {
        'schema': 'rampq.gate1.body32_head16_protocol_preparation/v1',
        'status': 'BOTH_FIXED_PROTOCOLS_FROZEN_BEFORE_FRESH_INFERENCE',
        'runner': ident(runner), 'auditor': ident(ROOT / 'scripts/audit_gate1_body32_head16.py'),
        'source_review_manifest': ident(review_root / 'manifest.json'),
        'protocols': [ident(path) for path in paths], 'real_model_or_image_opened': False,
        'guard_favorability_is_expansion_condition': False})
    print(json.dumps({'protocols': [ident(path) for path in paths]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
