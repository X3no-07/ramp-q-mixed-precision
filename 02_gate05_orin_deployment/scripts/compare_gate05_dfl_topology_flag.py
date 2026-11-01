"""Compare saved original/code-Clip DFL cases; never execute an engine."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/gate05/20261020_dfl_context_native/native_evidence'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()}


def comparison(a, b):
    assert a.shape == b.shape == (1, 4, 8400) and a.dtype == b.dtype == np.float32
    valid = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[valid].astype(np.float64) - b[valid].astype(np.float64))
    return dict(all_pairs_finite=bool(valid.all()), numeric_equal=bool(np.array_equal(a, b)),
                byte_identical=a.tobytes() == b.tobytes(), different_elements=int(np.count_nonzero(a != b)),
                max_abs=float(d.max()) if d.size else None, mean_abs=float(d.mean()) if d.size else None,
                rms_difference=float(np.sqrt(np.mean(d * d))) if d.size else None, descriptive_only=True)


def constraint_values(rows):
    # Graph derivatives can rename intermediate tensors and place constants at
    # different indices. Match only the explicit named Q/DQ constraint requests.
    selected = []
    for row in rows:
        if not row['precision_is_set'].get('value') and not any(v['output_type_is_set'].get('value') for v in row['output_types']):
            continue
        assert row['precision_is_set']['supported'] and 'precision_value_error' not in row
        outputs = []
        for v in row['output_types']:
            assert v['output_type_is_set']['supported'] and 'requested_output_type_error' not in v
            outputs.append(dict(index=v['index'], is_set=v['output_type_is_set']['value'],
                                requested_type=v.get('requested_output_type') if v['output_type_is_set']['value'] else None))
        selected.append(dict(name=row['name'], precision_is_set=row['precision_is_set']['value'],
                             precision=row.get('precision') if row['precision_is_set']['value'] else None, outputs=outputs))
    return sorted(selected, key=lambda v: v['name'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    new, output = args.original_raw.resolve(), args.output.resolve()
    assert not output.exists() and output.parent.exists() and new not in output.parents and OLD not in output.parents
    initial = {str(folder): snapshot(folder) for folder in (OLD, new)}
    assert sha(OLD / 'protocol.json') == 'b2723549e6acee51c86aeba3b72c8451601ba4add6088f46f7443536a1ab74ee'
    assert sha(new / 'protocol.json') == '81200cd4900a35f0533c45c7542f6dacc353203e8338a09b47cd24465cc0f6ed'
    specs = {str(folder): read(folder / 'spec.json') for folder in (OLD, new)}
    for folder in (OLD, new):
        receipt = read(folder / 'remote_receipt.json')
        assert set(initial[str(folder)]) == set(receipt['files']) | {'remote_receipt.json'}
        for rel, expected in receipt['files'].items():
            path = folder / rel
            assert sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes']
        spec = specs[str(folder)]
        assert spec['input_file_sha256'] == 'cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab'
        assert spec['cpu_expected_outputs_sha256'] == 'fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511'
        assert sha(folder / spec['input_file']) == spec['input_file_sha256']
        assert spec['conv_names'] == ['/model.23/dfl/conv/Conv']
    rows, arrays, requests, constraint_sets = {}, {}, {}, {}
    for topology, folder in (('code_clip', OLD), ('original', new)):
        runs = list((folder / 'results').glob('probe_*'))
        assert len(runs) == 1
        run = runs[0]
        summary = read(run / 'summary.json')
        for flag in ('none', 'obey'):
            name = topology + '_' + flag
            out = run / name
            build = read(out / 'build_result.json')
            request = read(out / 'parsed_network_request.json')
            requests[name] = request
            constraint_sets[name] = constraint_values(read(out / 'parsed_constraints_before.json'))
            row = dict(topology=topology, flag=flag, stage_root=folder.relative_to(ROOT).as_posix(),
                       build_status=build['status'], builder_seconds_not_benchmark=build.get('build_seconds_not_benchmark'),
                       engine_present=(out / 'model.engine').exists(), raw_output_present=(out / 'raw_outputs.npz').exists(),
                       clip_attributes=request['clip_attributes'], constraint_requests=constraint_sets[name],
                       actual_config_flags=request['flags']['get_flag'], run_outcome=summary['cases'][name]['status'])
            if (out / 'result.json').exists():
                result = read(out / 'result.json')
                row.update(engine_sha256=result['engine_sha256'], cpu_difference=result['comparison_to_frozen_cpu_ort'],
                           repeat=result['repeat_comparison'], actual_convolutions=read(out / 'conv_rows.json'))
                with np.load(out / 'raw_outputs.npz', allow_pickle=False) as saved:
                    arrays[name] = saved['first'].copy()
            else:
                lines=(out / 'builder.log').read_text(encoding='utf-8', errors='replace').splitlines()
                row['error_lines'] = [v for v in lines if '[E]' in v]
            rows[name] = row
    base_flags = requests['code_clip_none']['flags']['get_flag']
    for name, request in requests.items():
        diffs = [k for k in base_flags if base_flags[k] != request['flags']['get_flag'][k]]
        assert diffs == (['OBEY_PRECISION_CONSTRAINTS'] if name.endswith('_obey') else [])
        assert request['workspace_bytes_actual'] == 1073741824 and request['avg_timing_iterations_actual'] == 1
        assert request['no_layer_precision_or_output_type_setters_or_resets']
    assert all(v == constraint_sets['code_clip_none'] for v in constraint_sets.values())
    pairs = {}
    for a, b in [('original_none', 'original_obey'), ('original_none', 'code_clip_none'), ('original_obey', 'code_clip_obey')]:
        pairs[a + '_vs_' + b] = comparison(arrays[a], arrays[b]) if a in arrays and b in arrays else dict(status='UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED')
    assert initial == {str(folder): snapshot(folder) for folder in (OLD, new)}
    report = dict(schema='rampq.gate05.dfl_local_topology_flag_comparison/v1', client_at=datetime.now().astimezone().isoformat(),
                  status='FOUR_SAVED_LOCAL_CASES_COMPARED', cases=rows, raw_output_comparisons=pairs,
                  same_fixture_and_saved_CPU_reference=True, same_Q_DQ_explicit_constraint_values=True,
                  all_common_config_flags_equal_except_OBEY=True, original_Clip_attributes_unmodified=True,
                  corrected_Clip_attributes_declared_correction=True, raw_trees_unchanged=True,
                  new_CPU_or_native_inference=False, original_complete_model_guard='FAILED_UNCHANGED', gate05_status='INCOMPLETE',
                  limits='Four single observations across two bounded stages on one diagnostic corrected-CPU fixture; original topology and code-Clip/import-correction derivative form a combined factor. Not a randomized/repeated study, full-model repair or unique failing-node identification.')
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=report['status'], cases={k: v['build_status'] for k,v in rows.items()}, pairs=pairs)))


if __name__ == '__main__':
    main()
