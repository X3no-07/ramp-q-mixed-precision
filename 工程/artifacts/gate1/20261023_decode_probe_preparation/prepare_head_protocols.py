"""Freeze a second, distinct feature-source x head-arithmetic diagnostic."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ident(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def new(path, data):
    assert not path.exists(), path
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    source = ROOT / 'scripts/probe_gate1_head_precision.py'
    decoder = ROOT / 'scripts/probe_gate1_decode_precision.py'
    assert sha(decoder) == 'f76f3bfd677487fa7d87947f866daaf7407ddd455ad3e4d9e399da719e481ec0'
    paths = []
    for stage, reference in [('pilot3', '20261023_decode_precision_pilot_v1'), ('full109', '20261023_decode_precision_full_v1')]:
        protocol = read(OUT / f'{stage}_protocol.json')
        ref = ROOT / 'artifacts/gate1' / reference
        receipt = read(ref / 'receipt.json')
        assert receipt['status'] == 'COMPLETE_DIAGNOSTIC_ONLY'
        assert receipt['native_reconstruction_all_exact'] and receipt['historical_bytes_all_equal']
        assert receipt['images_completed'] == len(protocol['image_ids'])
        for name, row in read(ref / 'manifest.json')['files'].items():
            p = ref / name
            assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], name
        protocol.update({
            'schema': 'rampq.gate1.head_precision_probe_protocol/v1',
            'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'runner_sha256': sha(source),
            'decoder_runner': ident(decoder), 'reference_decode_probe_dir': str(ref),
            'reference_decode_probe_manifest': ident(ref / 'manifest.json'),
            'intervention': {'feature_sources': ['fp32', 'fp16'], 'head_compute_dtypes': ['fp32', 'fp16'],
                             'all_comparisons_decode_dtype': 'fp32', 'native_head_logits_exact_required': True,
                             'reference_decode_logits_exact_required': True},
            'scope': 'Second post-hoc debug diagnostic: captured three-scale head-input feature source x head compute dtype, all compared under the same FP32 decoder. Not scientific baseline acceptance or generalization.',
            'interventions': ['feature_fp32_head_fp32', 'feature_fp32_head_fp16', 'feature_fp16_head_fp32', 'feature_fp16_head_fp16'],
            'execution_condition': ('Execute only after completed exact native/historical decode pilot and full109 confirm predecode differences. No historical results overwritten.' if stage == 'pilot3'
                                    else 'Execute after head pilot completes with exact native logit and decode-reference checks; extend to all109 regardless of favorable/unfavorable intervention counts. No method/module selection.'),
            'upstream_scope': 'Everything before captured Detect input features, including input cast, backbone and neck; feature half casting in intervention is explicitly included.',
            'head_scope': 'Frozen Detect forward_head cv2/cv3 branches, including its convolution/BN/activation sequence; same saved weights lifted to the requested dtype.',
            'nonadditivity': 'Same-feature head contrasts and same-head feature-source contrasts may interact; do not add violation counts or claim a unique layer root cause.'
        })
        path = OUT / f'head_{stage}_protocol.json'
        new(path, protocol)
        paths.append(path)
    new(OUT / 'head_preparation_receipt.json', {
        'schema': 'rampq.gate1.head_probe_preparation/v1',
        'status': 'PROTOCOLS_FROZEN_BEFORE_HEAD_INTERVENTION', 'runner': ident(source),
        'protocols': [ident(p) for p in paths], 'new_head_intervention_executed_by_preparation': False,
        'scope': 'Conditional full extension is fixed before pilot intervention results; frozen data/weights/old guards remain unchanged.'
    })
    print(json.dumps({'protocols': [ident(p) for p in paths]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
