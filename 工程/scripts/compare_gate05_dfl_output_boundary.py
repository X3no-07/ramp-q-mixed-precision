"""Compare saved DFL outputs across terminal-Reshape and Conv-output cuts."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_native/native_evidence'
PROBABILITY_SHA = '5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()}


def requests(rows):
    selected = []
    for row in rows:
        if not row['precision_is_set'].get('value') and not any(v['output_type_is_set'].get('value') for v in row['output_types']):
            continue
        selected.append(dict(name=row['name'], precision_set=row['precision_is_set'], precision=row.get('precision'),
            outputs=[dict(index=v['index'], is_set=v['output_type_is_set'], requested_type=v.get('requested_output_type')) for v in row['output_types']]))
    return sorted(selected, key=lambda row: row['name'])


def compare(a, b):
    assert a.shape in ((1, 4, 8400), (1, 1, 4, 8400))
    assert b.shape in ((1, 4, 8400), (1, 1, 4, 8400))
    assert a.dtype == b.dtype == np.float32 and a.size == b.size == 33600
    shapes = [list(a.shape), list(b.shape)]
    first, second = a.reshape(-1), b.reshape(-1)
    finite = np.isfinite(first) & np.isfinite(second)
    delta = np.abs(first[finite].astype(np.float64) - second[finite].astype(np.float64))
    return dict(original_shapes=shapes, correspondence='C-order raw array bytes after explicit shape/dtype/count checks',
        all_pairs_finite=bool(finite.all()), numeric_equal=bool(np.array_equal(first, second)),
        byte_identical=a.tobytes(order='C') == b.tobytes(order='C'),
        different_elements=int(np.count_nonzero(first != second)), max_abs=float(delta.max()) if delta.size else None,
        mean_abs=float(delta.mean()) if delta.size else None,
        rms_difference=float(np.sqrt(np.mean(delta * delta))) if delta.size else None, descriptive_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cut-raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    new, output = args.cut_raw.resolve(), args.output.resolve()
    assert not output.exists() and output.parent.exists() and new not in output.parents and OLD not in output.parents
    before = {str(folder): inventory(folder) for folder in (OLD, new)}
    rows, arrays, configs, constraints, references = {}, {}, {}, {}, {}
    for kind, folder, expected_shape in (('reshaped_output', OLD, (1, 4, 8400)), ('conv_output', new, (1, 1, 4, 8400))):
        receipt = read(folder / 'remote_receipt.json')
        assert set(before[str(folder)]) == set(receipt['files']) | {'remote_receipt.json'}
        for relative, expected in receipt['files'].items():
            path = folder / relative
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
        spec = read(folder / 'spec.json')
        assert spec['input_file_sha256'] == PROBABILITY_SHA
        assert sha(folder / spec['input_file']) == PROBABILITY_SHA
        expected_path = folder / spec['cpu_expected_outputs_file']
        assert sha(expected_path) == spec['cpu_expected_outputs_sha256']
        with np.load(expected_path, allow_pickle=False) as saved:
            references[kind] = saved['code_clip_output'].copy()
        assert references[kind].shape == expected_shape and references[kind].dtype == np.float32
        runs = list((folder / 'results').glob('probe_*'))
        assert len(runs) == 1
        run = runs[0]
        for flag in ('none', 'obey'):
            case = 'code_clip_' + flag
            key, child = kind + '_' + flag, run / case
            build = read(child / 'build_result.json')
            request = read(child / 'parsed_network_request.json')
            configs[key] = request['flags']['get_flag']
            constraints[key] = requests(read(child / 'parsed_constraints_before.json'))
            assert request['workspace_bytes_actual'] == 1073741824 and request['avg_timing_iterations_actual'] == 1
            assert request['no_layer_precision_or_output_type_setters_or_resets']
            record = dict(context=kind, flag=flag, input_sha256=spec['input_file_sha256'], build=build,
                engine_present=(child / 'model.engine').exists(), raw_present=(child / 'raw_outputs.npz').exists(),
                actual_flags=configs[key], named_Q_DQ_constraints=constraints[key], expected_shape=list(expected_shape))
            if (child / 'result.json').exists():
                result = read(child / 'result.json')
                record.update(cpu_difference=result['comparison_to_frozen_cpu_ort'], repeat=result['repeat_comparison'], actual_convolutions=read(child / 'conv_rows.json'))
                with np.load(child / 'raw_outputs.npz', allow_pickle=False) as saved:
                    arrays[key] = saved['first'].copy()
                assert arrays[key].shape == expected_shape
            else:
                record['error_lines'] = [line for line in (child / 'builder.log').read_text(encoding='utf-8', errors='replace').splitlines() if '[E]' in line]
            rows[key] = record
    for key, flags in configs.items():
        assert [name for name in configs['reshaped_output_none'] if flags[name] != configs['reshaped_output_none'][name]] == (['OBEY_PRECISION_CONSTRAINTS'] if key.endswith('_obey') else [])
    assert all(value == constraints['reshaped_output_none'] for value in constraints.values())
    reference_comparison = compare(references['reshaped_output'], references['conv_output'])
    assert reference_comparison['byte_identical']
    pairs = {}
    for first, second in [('reshaped_output_none', 'conv_output_none'), ('reshaped_output_obey', 'conv_output_obey'), ('conv_output_none', 'conv_output_obey')]:
        pairs[first + '_vs_' + second] = compare(arrays[first], arrays[second]) if first in arrays and second in arrays else dict(status='UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED')
    assert before == {str(folder): inventory(folder) for folder in (OLD, new)}
    report = dict(schema='rampq.gate05.dfl_saved_output_boundary_comparison/v1', client_at=datetime.now().astimezone().isoformat(),
        status='SAVED_DFL_OUTPUT_BOUNDARIES_COMPARED', cases=rows, raw_output_comparisons=pairs,
        same_native_probability_input_file=True, CPU_reference_comparison=reference_comparison,
        common_named_Q_DQ_requests_equal=True, common_config_flags_equal_except_OBEY=True,
        raw_trees_unchanged=True, new_inference=False, original_complete_model_guard='FAILED_UNCHANGED', gate05_status='INCOMPLETE',
        limits='Removing Constant/Reshape also changes external output shape and native format/fusion boundary; differing outcomes cannot uniquely identify a deleted node as full-model cause or establish repair. CPU probabilities are a fixed local fixture, not validated native upstream outputs.')
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], cases={key: value['build']['status'] for key, value in rows.items()}, comparisons=pairs)))


if __name__ == '__main__':
    main()
