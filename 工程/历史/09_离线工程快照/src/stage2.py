"""Offline evidence verification and saved-output comparison. No device access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def inside(root, name):
    path = (root / name).resolve()
    require(root.resolve() in path.parents, 'Path outside project: ' + name)
    return path


def verify(root=ROOT, source_root=None):
    manifest = read(root / 'config/source_manifest.json')
    errors = []
    for name, info in manifest['files'].items():
        try:
            path = inside(root, name)
            require(path.is_file(), 'Missing file: ' + name)
            require(path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                    'Snapshot hash/size mismatch: ' + name)
            if source_root is not None:
                original = inside(source_root, info['source_relative_path'])
                require(original.is_file() and original.stat().st_size == info['bytes']
                        and sha(original) == info['sha256'], 'Original changed: ' + name)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    return {'status': 'VERIFIED' if not errors else 'FAILED',
            'checked_files': len(manifest['files']), 'errors': errors,
            'scope': 'Copied file integrity only; not model correctness or Gate acceptance.',
            'originals_checked': source_root is not None}


def baseline(root=ROOT):
    result = verify(root)
    require(result['status'] == 'VERIFIED', '; '.join(result['errors']))
    config = read(root / 'config/baseline.json')
    closure = read(inside(root, config['closure']))
    require(closure['gate05_status'] == 'INCOMPLETE', 'Unexpected historical Gate status')
    require(closure['dataset_evaluation_run'] is False, 'Unexpected historical dataset evaluation')
    require(closure['numerical_parity_original_failure'] == 'FAILED_UNCHANGED',
            'Original numerical failure must be retained')
    names = [row['configuration'] for row in closure['matrix']]
    require(len(names) == 6 and set(names) == set(config['cases']), 'Expected six frozen cases')
    return config, closure


def status(root=ROOT):
    config, closure = baseline(root)
    return {'source_client_at': closure['client_at'], 'gate05_status': closure['gate05_status'],
            'matrix': closure['matrix'], 'original_numerical_guard': 'FAILED_UNCHANGED',
            'dataset_evaluation_run': False, 'live_board_checked': False,
            'next_protocol_status': read(root / 'config/next_protocol.json')['status'],
            'scope': 'Historical snapshot; no new inference or board health claim.'}


def load_output(path, key, expected_shape):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        require(key in archive.files, 'Missing output array: ' + key)
        array = archive[key].copy()
    require(array.shape == tuple(expected_shape), 'Unexpected output shape: ' + str(array.shape))
    require(array.dtype == np.dtype('float32'), 'Expected FLOAT external output')
    require(bool(np.isfinite(array).all()), 'Nonfinite output: ' + str(path))
    return array


def difference(reference, candidate):
    import numpy as np
    require(reference.shape == candidate.shape, 'Array shapes differ')
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
            'bit_exact': reference.dtype == candidate.dtype and reference.tobytes() == candidate.tobytes()}


def compare(root=ROOT):
    import numpy as np
    config, closure = baseline(root)
    reference = config['cases']['fp16_control']
    ref_array = load_output(inside(root, reference['raw']), reference['key'], [1, 14, 8400])
    records = {row['configuration']: row for row in closure['matrix']}
    rows = []
    for name, case in config['cases'].items():
        report = read(inside(root, case['report']))
        require(report['image_sha256'] == config['fixed_image_sha256'], 'Image differs: ' + name)
        require(report['nms'] == config['nms'], 'Frozen NMS differs: ' + name)
        require(report['engine_sha256'] == records[name]['engine_sha256'], 'Engine differs: ' + name)
        require(report['execution_success'] is True, 'Execution missing: ' + name)
        array = load_output(inside(root, case['raw']), case['key'], [1, 14, 8400])
        aux = load_output(inside(root, case['raw']), 'auxiliary_output_0', [1, 128, 20, 20])
        predictions = report[case['detections_key']]
        require(len(predictions) == records[name]['detections'], 'Saved detection count differs')
        scores = array[:, 4:, :]
        rows.append({'configuration': name, 'shape': list(array.shape),
                     'finite_primary_and_auxiliary': bool(np.isfinite(aux).all()),
                     'score_min': float(scores.min()), 'score_max': float(scores.max()),
                     'anchors_at_or_above_0_25_before_nms': int((scores.max(axis=1) >= .25).sum()),
                     'saved_detection_count': len(predictions),
                     'boxes_vs_fp16_descriptive': difference(ref_array[:, :4], array[:, :4]),
                     'scores_vs_fp16_descriptive': difference(ref_array[:, 4:], array[:, 4:])})
    old = config['prior_fp16']
    prior_main = load_output(inside(root, old['raw']), old['key'], [1, 14, 8400])
    prior_aux = load_output(inside(root, old['raw']), 'auxiliary_output_0', [1, 128, 20, 20])
    end_aux = load_output(inside(root, reference['raw']), 'auxiliary_output_0', [1, 128, 20, 20])
    cpu = config['qonly_cpu']
    cpu_path = inside(root, cpu['raw'])
    pt = load_output(cpu_path, 'pytorch_qonly_g1', [1, 14, 8400])
    ort = load_output(cpu_path, 'ort_qonly_g1', [1, 14, 8400])
    native_case = config['cases']['q_only_G1']
    native = load_output(inside(root, native_case['raw']), native_case['key'], [1, 14, 8400])
    return {'schema': 'rampq.stage2.saved_comparison/v1', 'gate05_status': 'INCOMPLETE',
            'new_inference_run': False, 'dataset_evaluation_run': False,
            'scope': 'Descriptive only. CPU FP32/native FP16/fusions confound cross-backend comparisons.',
            'rows': rows,
            'fp16_repeat_main': difference(prior_main, ref_array),
            'fp16_repeat_aux': difference(prior_aux, end_aux),
            'qonly_cpu_vs_native': {'pytorch_score_max': float(pt[:, 4:].max()),
                                    'ort_score_max': float(ort[:, 4:].max()),
                                    'native_score_max': float(native[:, 4:].max()),
                                    'pt_ort_scores': difference(pt[:, 4:], ort[:, 4:]),
                                    'ort_native_scores': difference(ort[:, 4:], native[:, 4:])},
            'acceptance_decision': 'NOT_EVALUATED_NO_NEW_TOLERANCE'}


def save_or_print(result, output):
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(payload)
        print(str(path.resolve()))
    else:
        print(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'status', 'compare', 'plan'])
    parser.add_argument('--source-root', type=Path, help='Optional original 07 directory; verify only')
    parser.add_argument('--output', type=Path, help='New JSON path; existing files are never overwritten')
    args = parser.parse_args()
    require(args.source_root is None or args.command == 'verify', '--source-root is for verify only')
    if args.command == 'verify':
        result = verify(source_root=args.source_root)
    elif args.command == 'status':
        result = status()
    elif args.command == 'compare':
        result = compare()
    else:
        result = read(ROOT / 'config/next_protocol.json')
    save_or_print(result, args.output)
    return 1 if result.get('status') == 'FAILED' else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(2)
