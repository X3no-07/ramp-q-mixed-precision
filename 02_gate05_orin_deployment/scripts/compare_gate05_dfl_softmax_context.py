"""Saved-file comparison of corrected DFL with and without explicit Softmax."""
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


def inventory(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()}


def requests(rows):
    selected = []
    for r in rows:
        if not r['precision_is_set'].get('value') and not any(v['output_type_is_set'].get('value') for v in r['output_types']):
            continue
        selected.append(dict(name=r['name'], precision_set=r['precision_is_set'], precision=r.get('precision'),
                             outputs=[dict(index=v['index'], is_set=v['output_type_is_set'], requested_type=v.get('requested_output_type')) for v in r['output_types']]))
    return sorted(selected, key=lambda r: r['name'])


def compare(a, b):
    assert a.shape == b.shape == (1, 4, 8400) and a.dtype == b.dtype == np.float32
    finite = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    return dict(all_pairs_finite=bool(finite.all()), numeric_equal=bool(np.array_equal(a, b)), byte_identical=a.tobytes() == b.tobytes(),
                different_elements=int(np.count_nonzero(a != b)), max_abs=float(d.max()) if d.size else None,
                mean_abs=float(d.mean()) if d.size else None, rms_difference=float(np.sqrt(np.mean(d*d))) if d.size else None,
                descriptive_only=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cut-raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    new, output = args.cut_raw.resolve(), args.output.resolve()
    assert not output.exists() and output.parent.exists() and new not in output.parents and OLD not in output.parents
    before = {str(f): inventory(f) for f in (OLD, new)}
    rows, arrays, configs, constraints = {}, {}, {}, {}
    for kind, folder in (('softmax_inside', OLD), ('probability_input_cut', new)):
        receipt = read(folder / 'remote_receipt.json')
        assert set(before[str(folder)]) == set(receipt['files']) | {'remote_receipt.json'}
        for rel, expected in receipt['files'].items():
            f = folder / rel
            assert sha(f) == expected['sha256'] and f.stat().st_size == expected['bytes']
        spec = read(folder / 'spec.json')
        assert spec['cpu_expected_outputs_sha256'] == 'fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511'
        if kind == 'softmax_inside':
            assert spec['input_file_sha256'] == 'cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab'
        else:
            assert spec['input_file_sha256'] == '5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520'
            assert read(folder / 'cpu_diagnostic.json')['byte_identical_saved_corrected_DFL_output']
        runs = list((folder / 'results').glob('probe_*'))
        assert len(runs) == 1
        run = runs[0]
        for flag in ('none', 'obey'):
            case = 'code_clip_' + flag
            key, f = kind + '_' + flag, run / case
            build = read(f / 'build_result.json')
            request = read(f / 'parsed_network_request.json')
            configs[key] = request['flags']['get_flag']
            constraints[key] = requests(read(f / 'parsed_constraints_before.json'))
            assert request['workspace_bytes_actual'] == 1073741824 and request['avg_timing_iterations_actual'] == 1
            assert request['no_layer_precision_or_output_type_setters_or_resets']
            record = dict(context=kind, flag=flag, input_sha256=spec['input_file_sha256'], build=build,
                          engine_present=(f / 'model.engine').exists(), raw_present=(f / 'raw_outputs.npz').exists(),
                          actual_flags=configs[key], named_Q_DQ_constraints=constraints[key])
            if (f / 'result.json').exists():
                result = read(f / 'result.json')
                record.update(cpu_difference=result['comparison_to_frozen_cpu_ort'], repeat=result['repeat_comparison'], actual_convolutions=read(f / 'conv_rows.json'))
                with np.load(f / 'raw_outputs.npz', allow_pickle=False) as saved:
                    arrays[key] = saved['first'].copy()
            else:
                record['error_lines'] = [v for v in (f / 'builder.log').read_text(encoding='utf-8', errors='replace').splitlines() if '[E]' in v]
            rows[key] = record
    for key, flags in configs.items():
        assert [k for k in configs['softmax_inside_none'] if flags[k] != configs['softmax_inside_none'][k]] == (['OBEY_PRECISION_CONSTRAINTS'] if key.endswith('_obey') else [])
    assert all(v == constraints['softmax_inside_none'] for v in constraints.values())
    pairs = {}
    for a,b in [('softmax_inside_none', 'probability_input_cut_none'), ('softmax_inside_obey', 'probability_input_cut_obey'), ('probability_input_cut_none', 'probability_input_cut_obey')]:
        pairs[a+'_vs_'+b] = compare(arrays[a], arrays[b]) if a in arrays and b in arrays else dict(status='UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED')
    assert before == {str(f): inventory(f) for f in (OLD, new)}
    result = dict(schema='rampq.gate05.dfl_softmax_saved_context_comparison/v1', client_at=datetime.now().astimezone().isoformat(),
                  status='SAVED_DFL_SOFTMAX_CONTEXTS_COMPARED', cases=rows, raw_output_comparisons=pairs,
                  shared_CPU_output_reference=True, CPU_composition_byte_equivalent=True, common_named_Q_DQ_requests_equal=True,
                  common_config_flags_equal_except_OBEY=True, raw_trees_unchanged=True, new_inference=False,
                  original_complete_model_guard='FAILED_UNCHANGED', gate05_status='INCOMPLETE',
                  limits='The cut also changes the external boundary, native fusion context and use of CPU probabilities. Input arrays are not the same: old consumes logits, new consumes CPU probabilities. Changed outcomes do not identify Softmax as the unique full-model failing node or prove full-model repair.')
    with output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],cases={k:v['build']['status'] for k,v in rows.items()},comparisons=pairs)))


if __name__ == '__main__':
    main()
