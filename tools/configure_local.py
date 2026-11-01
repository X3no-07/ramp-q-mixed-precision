"""Create ignored local path configuration without changing frozen artifacts."""
import argparse
import json
from pathlib import Path

STAGES = ('01_gate0_mechanism_screening', '02_gate05_orin_deployment', '03_gate1_visdrone_validation')

def main():
    parser = argparse.ArgumentParser(description='Configure paths for a locally supplied model and RUOD-R. Generated local files are ignored by Git.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    data = args.data_root.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if not data.is_dir() or not model.is_file():
        parser.error('data-root must be an existing directory, and model must be an existing file.')
    root = Path(__file__).resolve().parents[1]
    for stage in STAGES:
        phase = root / stage
        cfg = json.loads((phase / 'config/gate0.json').read_text(encoding='utf-8'))
        cfg.update(model=str(model), dataset_root=str(data))
        (phase / 'config/gate0.local.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        for source in (phase / 'splits').glob('gate0_*_images.txt'):
            members = [line.strip() for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
            if any(Path(value).is_absolute() or '..' in Path(value).parts for value in members):
                raise ValueError('Expected dataset-relative membership list')
            local = source.with_name(source.stem+'.local.txt')
            local.write_text(''.join(str(data / value)+'\n' for value in members), encoding='utf-8')
    print('Local configurations and image-path lists were generated for three stages. No experiment was started, and no frozen manifest was created or replaced.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
