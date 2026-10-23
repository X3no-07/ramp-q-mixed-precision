"""Describe fixed-guard failures from saved tensors; never rescore the guard."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification'
OUT = Path(__file__).parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    saved = json.loads((SOURCE / 'raw_dtype_guard.json').read_text(encoding='utf-8'))
    counters = Counter()
    buckets, heads, selected = {}, {}, {}
    worst, images = [], []
    for row in saved['guards']:
        image_id = row['image_id']; stem = image_id.split('/')[1]
        with np.load(SOURCE / 'raw' / (stem + '.npz'), allow_pickle=False) as archive:
            a, b = archive['fp32'].astype(np.float64)[0], archive['fp16'].astype(np.float64)[0]
        assert a.shape == b.shape == (14, 8400) and np.isfinite(a).all() and np.isfinite(b).all()
        error = np.abs(a[:4] - b[:4]); tolerance = .5 + .005 * np.abs(a[:4])
        bad = error > tolerance; bad_anchor = bad.any(axis=0)
        score_bad = np.abs(a[4:] - b[4:]) > .002 + .02 * np.abs(a[4:])
        assert int(bad.sum()) == row['boxes']['violations']
        assert int(score_bad.sum()) == row['scores']['violations']
        score_a, score_b = a[4:].max(axis=0), b[4:].max(axis=0)
        union = np.maximum(score_a, score_b)
        counters.update({'images': 1, 'box_elements': bad.size, 'box_violations': int(bad.sum()),
                         'score_elements': score_bad.size, 'score_violations': int(score_bad.sum()),
                         'all_anchors': 8400, 'box_violating_anchors': int(bad_anchor.sum())})
        for index, channel in enumerate(('center_x', 'center_y', 'width', 'height')):
            counters['box_violations_' + channel] += int(bad[index].sum())
        for name, mask in {
            'union_score_le_0.001': union <= .001,
            'union_score_gt_0.001_lt_0.2': (union > .001) & (union < .2),
            'union_score_ge_0.2_lt_0.25': (union >= .2) & (union < .25),
            'union_score_ge_0.25_lt_0.3': (union >= .25) & (union < .3),
            'union_score_ge_0.3': union >= .3,
        }.items():
            buckets.setdefault(name, Counter()).update({'anchors': int(mask.sum()), 'box_violating_anchors': int((bad_anchor & mask).sum()),
                                                      'box_violations': int(bad[:, mask].sum())})
        for name, lo, hi in [('stride8', 0, 6400), ('stride16', 6400, 8000), ('stride32', 8000, 8400)]:
            heads.setdefault(name, Counter()).update({'anchors': hi - lo, 'box_violating_anchors': int(bad_anchor[lo:hi].sum()),
                                                    'box_violations': int(bad[:, lo:hi].sum())})
        item = json.loads((SOURCE / 'per_image' / (stem + '.json')).read_text(encoding='utf-8'))
        branch_sets = {}
        for branch in ('fp32', 'fp16'):
            predictions = item['predictions'][branch]
            by_id = {p['detection_id']: p for p in predictions}
            masks = {
                'post_nms_all': {p['raw_anchor_index'] for p in predictions},
                'post_nms_conf_ge_0.25': {p['raw_anchor_index'] for p in predictions if p['score'] >= .25},
                'matched_true_positive_conf_0.25': {by_id[m['detection_id']]['raw_anchor_index'] for m in item['matching']['0.25'][branch]['matches']},
            }
            branch_sets[branch] = masks
            for name, anchors in masks.items():
                idx = sorted(anchors)
                selected.setdefault(branch + '_' + name, Counter()).update({'anchors': len(idx), 'box_violating_anchors': int(bad_anchor[idx].sum()),
                                                                           'box_violations': int(bad[:, idx].sum())})
        ratios = error / tolerance
        for flat in np.argsort(ratios.reshape(-1))[-3:]:
            channel, anchor = np.unravel_index(flat, ratios.shape)
            worst.append({'image_id': image_id, 'channel': ('center_x', 'center_y', 'width', 'height')[channel], 'anchor': int(anchor),
                          'fp32': float(a[channel, anchor]), 'fp16': float(b[channel, anchor]), 'absolute_error': float(error[channel, anchor]),
                          'fixed_tolerance': float(tolerance[channel, anchor]), 'error_over_tolerance': float(ratios[channel, anchor]),
                          'fp32_max_score': float(score_a[anchor]), 'fp16_max_score': float(score_b[anchor]),
                          'post_nms_fp32': anchor in branch_sets['fp32']['post_nms_all'], 'post_nms_fp16': anchor in branch_sets['fp16']['post_nms_all']})
        images.append({'image_id': image_id, 'box_violations': int(bad.sum()), 'box_violating_anchors': int(bad_anchor.sum()),
                       'score_violations': int(score_bad.sum())})
    assert sum(v['box_violations'] for v in buckets.values()) == counters['box_violations']
    assert sum(v['box_violations'] for v in heads.values()) == counters['box_violations']
    assert sum(v['anchors'] for v in buckets.values()) == counters['all_anchors']
    result = {'schema': 'rampq.gate1.saved_raw_dtype_failure_description/v1', 'at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Post-hoc description of the unchanged all-anchor guard, using saved data only; no new inference or pass criterion',
              'source_manifest_sha256': sha(SOURCE / 'manifest.json'), 'source_raw_guard_sha256': sha(SOURCE / 'raw_dtype_guard.json'),
              'analysis_source_sha256': sha(Path(__file__)), 'guard_status_unchanged': saved['status'], 'totals': dict(counters),
              'box_violation_fraction': counters['box_violations'] / counters['box_elements'], 'score_buckets': buckets,
              'head_levels': heads, 'postprocessing_anchor_subsets': selected,
              'worst_elements': sorted(worst, key=lambda x: x['error_over_tolerance'], reverse=True)[:20], 'per_image': images,
              'limits': ['Score and anchor subsets are diagnostic only and never replace the frozen all-anchor guard.',
                         'Decoded outputs do not identify which intermediate layer or arithmetic operation caused the difference.',
                         'Debug target agreement is not official AP, causal quantization evidence, or independent Holdout confirmation.']}
    target = OUT / 'raw_failure_description.json'; assert not target.exists()
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('guard_status_unchanged', 'totals', 'box_violation_fraction', 'postprocessing_anchor_subsets')}))


if __name__ == '__main__':
    main()
