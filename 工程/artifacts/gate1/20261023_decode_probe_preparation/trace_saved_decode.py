"""Describe fixed failed-anchor counterfactuals; never replace the old guard."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / 'artifacts/gate1/20261023_decode_precision_full_v1'
OUT = ROOT / 'artifacts/gate1/20261023_saved_decode_trace'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir()
    summary = json.loads((PROBE / 'summary.json').read_text(encoding='utf-8'))
    assert summary['historical_bytes_all_equal'] and summary['native_reconstruction_all_exact'] and summary['images'] == 109
    protocol = json.loads((PROBE / 'protocol.json').read_text(encoding='utf-8'))
    old_description = json.loads((ROOT / 'artifacts/gate1/20261022_dtype_guard_diagnostics/raw_failure_description.json').read_text(encoding='utf-8'))
    selected = old_description['worst_elements'][:3]
    selected_by_image = {}
    for item in selected:
        selected_by_image.setdefault(item['image_id'], []).append(item)
    counters, traces = Counter(), []
    for image_id in protocol['image_ids']:
        path = PROBE / 'raw' / (image_id.split('/')[1] + '.npz')
        with np.load(path, allow_pickle=False) as arrays:
            ref = arrays['fp32_forward'][0, :4].astype(np.float64)
            orig = arrays['fp16_forward'][0, :4].astype(np.float64)
            intervention = arrays['source_fp16_decode_fp32_decoded'][0, :4].astype(np.float64)
            tolerance = .5 + .005 * abs(ref)
            old_bad = abs(ref - orig) > tolerance
            new_bad = abs(ref - intervention) > tolerance
            counters.update({'old_violations': int(old_bad.sum()), 'fp16_logits_fp32_decode_violations': int(new_bad.sum()),
                             'shared_violations': int((old_bad & new_bad).sum()),
                             'old_only_violations': int((old_bad & ~new_bad).sum()),
                             'new_only_violations': int((new_bad & ~old_bad).sum())})
            for item in selected_by_image.get(image_id, []):
                anchor = item['anchor']
                record = {k: item[k] for k in ('image_id', 'anchor', 'channel', 'fixed_tolerance', 'error_over_tolerance')}
                record['raw_npz_sha256'] = sha(path)
                record['stride'] = float(arrays['fp32_strides'][0, anchor])
                record['anchor_xy'] = arrays['fp32_anchors'][:, anchor].tolist()
                record['branches'] = {}
                for source in ('fp32', 'fp16'):
                    prefix = f'source_{source}_decode_fp32'
                    logits = arrays[f'{source}_box_logits'][0, :, anchor].reshape(4, -1).astype(np.float64)
                    probs = arrays[f'{prefix}_probabilities'][0, :, :, anchor].T.astype(np.float64)
                    record['branches'][source + '_logits_fp32_decode'] = {
                        'logits_ltrb_bins': logits.tolist(), 'probabilities_ltrb_bins': probs.tolist(),
                        'dfl_distances_ltrb': arrays[f'{prefix}_distances'][0, :, anchor].astype(float).tolist(),
                        'decoded_xywh': arrays[f'{prefix}_boxes'][0, :, anchor].astype(float).tolist(),
                        'original_forward_xywh': arrays[f'{source}_forward'][0, :4, anchor].astype(float).tolist(),
                    }
                record['max_box_logit_absolute_source_difference'] = float(abs(arrays['fp32_box_logits'][0, :, anchor].astype(np.float64) - arrays['fp16_box_logits'][0, :, anchor]).max())
                traces.append(record)
    assert counters['old_violations'] == 10849 and counters['fp16_logits_fp32_decode_violations'] == 8992
    assert counters['shared_violations'] + counters['old_only_violations'] == counters['old_violations']
    assert counters['shared_violations'] + counters['new_only_violations'] == counters['fp16_logits_fp32_decode_violations']
    result = {'schema': 'rampq.gate1.saved_decode_counterfactual_trace/v1',
              'scope': 'Post-hoc raw-only explanation of previously selected worst elements; not new criteria, unique-layer attribution or generalization.',
              'source_manifest_sha256': sha(PROBE / 'manifest.json'), 'script_sha256': sha(Path(__file__)),
              'counterfactual_set_overlap': dict(counters), 'worst_element_traces': traces,
              'original_guard_unchanged': 'FAIL_FIXED_TOLERANCES', 'model_or_image_executed': False,
              'limits': ['Source logit differences include all upstream computation and input casting.',
                         'Violation-set differences are not additive or uniquely attributable components of error.',
                         'Selected worst anchors are post-hoc diagnostic examples, not unbiased validation examples.']}
    target = OUT / 'trace.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'overlap': dict(counters), 'traces': len(traces)}))


if __name__ == '__main__':
    main()
