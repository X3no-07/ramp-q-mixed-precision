"""Verify pinned local sources and run native synthetic oracle if already installed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = ROOT / 'artifacts/gate1/20261022_split_refinement/evaluation_sources'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quoted(path):
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='A new directory inside this readiness directory')
    parser.add_argument('--runtime', choices=('auto', 'matlab', 'octave'), default='auto')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        raise SystemExit('Output must be a new child of this readiness directory')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((SOURCES / 'source_manifest.json').read_text(encoding='utf-8'))
    checked = []
    for row in manifest['files']:
        path = SOURCES / row['path']
        actual = sha(path)
        if actual != row['sha256'] or path.stat().st_size != row['bytes']:
            raise SystemExit('Pinned source identity mismatch: ' + row['path'])
        checked.append({'path': str(path.relative_to(ROOT)), 'sha256': actual, 'bytes': path.stat().st_size})
    fixture = json.loads((HERE / 'synthetic_fixtures.json').read_text(encoding='utf-8'))
    contract = json.loads((HERE / 'fixture_contract.json').read_text(encoding='utf-8'))
    for row in contract['bound_local_files']:
        if sha(HERE / row['path']) != row['sha256']:
            raise SystemExit('Bound preparation source changed: ' + row['path'])
    if fixture['data_role'] != 'synthetic' or fixture['expected_oracle_outputs'] is not None:
        raise SystemExit('Fixture scope changed')
    candidates = [('matlab', shutil.which('matlab')), ('octave', shutil.which('octave-cli') or shutil.which('octave'))]
    if args.runtime != 'auto':
        candidates = [item for item in candidates if item[0] == args.runtime]
    available = next(((name, path) for name, path in candidates if path), None)
    receipt = {'schema': 'rampq.gate1.synthetic_oracle_launcher/v1', 'python': sys.version,
               'source_commit': manifest['commit'], 'verified_sources': checked,
               'source_manifest_sha256': sha(SOURCES / 'source_manifest.json'),
               'fixture_sha256': sha(HERE / 'synthetic_fixtures.json'), 'fixture_count': len(fixture['cases']),
               'native_runtime_probe': candidates, 'real_data_read': False, 'model_executed': False,
               'official_oracle_executed': False, 'official_AP_equivalence_certified': False}
    if available is None:
        receipt['status'] = 'NOT_EXECUTED_NO_NATIVE_RUNTIME_ON_PATH'
        receipt['next_action'] = 'Use an already installed MATLAB or Octave with mean2 and JSON support; no installation is performed.'
        (output / 'launcher_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(receipt['status'])
        return
    name, executable = available
    expression = 'addpath(' + quoted(HERE) + ');run_synthetic_oracle(' + quoted(output) + ');'
    command = [executable, '-batch', expression] if name == 'matlab' else [executable, '--no-gui', '--quiet', '--eval', expression]
    receipt['runtime'] = {'name': name, 'executable': executable, 'command': command}
    with (output / 'native_console.txt').open('w', encoding='utf-8') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, encoding='utf-8', check=False)
    receipt['returncode'] = process.returncode
    receipt['official_oracle_executed'] = (output / 'oracle_results.mat').exists()
    receipt['status'] = 'EXECUTED_NATIVE_SYNTHETIC_ORACLE_UNCERTIFIED' if process.returncode == 0 else 'NATIVE_RUNTIME_OR_CASE_FAILURE'
    (output / 'launcher_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(receipt['status'])
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
