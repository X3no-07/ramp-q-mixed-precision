"""Stage existing calibration and Holdout data for a deployment migration check."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    cal_source = root / 'artifacts/calibration/screen.json'
    holdout_source = root / 'splits/gate0_holdout_images.txt'
    calibration = json.loads(cal_source.read_text(encoding='utf-8-sig'))
    train = [Path(p) for p in calibration['images']]
    evaluation = [Path(p) for p in holdout_source.read_text(encoding='utf-8-sig').splitlines() if p.strip()]
    assert len(train) == 64 and len(evaluation) == 150
    assert len({p.name for p in train}) == 64 and len({p.name for p in evaluation}) == 150
    expected = json.loads((root / 'artifacts/manifests/gate0_split.json').read_text(encoding='utf-8-sig'))
    # Gate 0 hashes canonical UTF-8 path lines with LF, not Windows file bytes.
    holdout_list_hash = hashlib.sha256(''.join(str(p.resolve()) + '\n' for p in evaluation).encode('utf-8')).hexdigest()
    # Calibration's original sha256_text omits the final newline.
    calibration_list_hash = hashlib.sha256('\n'.join(str(p.resolve()) for p in train).encode('utf-8')).hexdigest()
    assert holdout_list_hash == expected['holdout']['list_sha256']
    assert calibration_list_hash == calibration['image_list_sha256']
    destination = root / 'artifacts/gate05/data_v1'
    if destination.exists():
        raise RuntimeError('Data bundle exists; inspect before replacing')
    groups = {'calibration': [], 'evaluation': []}
    targets = 0
    for kind, paths in [('calibration', train), ('evaluation', evaluation)]:
        for image in paths:
            assert image.is_file(), str(image)
            item = {'source_image': str(image), 'image': kind + '/images/' + image.name,
                    'image_sha256': sha(image), 'image_bytes': image.stat().st_size}
            if kind == 'evaluation':
                label = image.parents[2] / 'labels' / image.parent.name / (image.stem + '.txt')
                assert label.is_file(), str(label)
                count = 0
                for line in label.read_text().splitlines():
                    if not line.strip():
                        continue
                    values = [float(v) for v in line.split()]
                    assert len(values) == 5 and values[0].is_integer() and 0 <= values[0] < 10
                    assert all(0 <= v <= 1 for v in values[1:])
                    count += 1
                targets += count
                item.update(source_label=str(label), label=kind + '/labels/' + label.name,
                            label_sha256=sha(label), target_count=count)
            groups[kind].append(item)
    cal_hashes = {item['image_sha256'] for item in groups['calibration']}
    eval_hashes = {item['image_sha256'] for item in groups['evaluation']}
    assert len(cal_hashes) == 64 and len(eval_hashes) == 150 and not cal_hashes & eval_hashes
    assert targets == expected['holdout']['target_count'] == 865
    manifest = {'schema': 'rampq.gate05.data/v1', 'prepared_client_date': '2026-10-18',
                'purpose': 'Fixed deployment migration diagnostic using the existing Gate 0 Holdout',
                'evidence_boundary': 'This is previously used Holdout data, not a fresh independent method-confirmation set.',
                'source_calibration_manifest_sha256': sha(cal_source),
                'source_holdout_file_bytes_sha256': sha(holdout_source),
                'canonical_holdout_list_sha256': holdout_list_hash,
                'canonical_calibration_list_sha256': calibration_list_hash,
                'weights_sha256': calibration['model_sha256'], 'calibration_count': 64,
                'evaluation_count': 150, 'evaluation_target_count': targets, 'image_content_overlap': 0,
                'data': groups, 'dataset_inference_run': False}
    destination.mkdir(parents=True, exist_ok=False)
    for items in groups.values():
        for item in items:
            for category in ('image', 'label'):
                if category not in item:
                    continue
                output = destination / item[category]
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item['source_' + category], output)
                assert sha(output) == item[category + '_sha256']
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    (destination / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + p.relative_to(destination).as_posix() + '\n' for p in files), encoding='ascii')
    print(json.dumps({'status': 'DATA_STAGED_ONLY', 'calibration': 64, 'evaluation': 150,
                      'targets': targets, 'content_overlap': 0,
                      'bytes': sum(p.stat().st_size for p in files)}))


if __name__ == '__main__':
    main()
