"""Read-only saved-evidence audit for the frozen DFL context cases."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype == np.float32
    finite = np.isfinite(a) & np.isfinite(b)
    delta = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    return dict(all_pairs_finite=bool(finite.all()), nonfinite_pair_count=int((~finite).sum()),
                numeric_equal=bool(np.array_equal(a, b)), byte_identical=a.tobytes() == b.tobytes(),
                different_elements=int(np.count_nonzero(a != b)), max_abs=float(delta.max()) if delta.size else None,
                mean_abs=float(delta.mean()) if delta.size else None,
                rms_difference=float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
                descriptive_only_no_new_pass_criterion=True)


def validate_stats(host, native):
    assert set(host) == set(native)
    if 'status' in host:
        assert host == native
        return {}
    reductions = ('mean_abs', 'rms_difference')
    for field in set(host) - set(reductions):
        assert host[field] == native[field], field
    return {k: dict(host_recomputed=host[k], native_reported=native[k],
                    host_minus_native=host[k] - native[k] if host[k] is not None and native[k] is not None else None)
            for k in reductions}


def constraints(rows):
    return [{k: row[k] for k in ('index', 'name', 'precision_is_set', 'precision', 'precision_value_error', 'output_types') if k in row}
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.output.resolve()
    assert output.parent.exists() and not output.exists() and raw not in output.parents
    initial = {p.relative_to(raw).as_posix(): sha(p) for p in raw.rglob('*') if p.is_file()}
    receipt = read(raw / 'remote_receipt.json')
    assert set(initial) == set(receipt['files']) | {'remote_receipt.json'}
    for rel, expected in receipt['files'].items():
        path = (raw / rel).resolve()
        assert raw in path.parents and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
    for rel, expected in read(raw / 'manifest.json')['files'].items():
        path = (raw / rel).resolve()
        assert raw in path.parents and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
    assert sha(raw / 'protocol.json') == '9b2ea26ff12ecd5a4b5a699b09c511a3ea6c2187125d037117bb85f57b7b1170'
    protocol, spec = read(raw / 'protocol.json'), read(raw / 'spec.json')
    assert spec['conv_names'] == protocol['source_conv_names']
    runs = list((raw / 'results').glob('probe_*'))
    assert len(runs) == 1
    run = runs[0]
    summary = read(run / 'summary.json')
    assert summary['requested_case_order'] == protocol['case_order']
    with np.load(raw / spec['cpu_expected_outputs_file'], allow_pickle=False) as saved:
        expected = saved['weight_bypass_output'].copy()
    assert expected.shape == tuple(protocol['output_shape']) and expected.dtype == np.float32
    cases, arrays, requests, parsed = {}, {}, {}, {}
    for case in spec['cases']:
        name = case['id']
        folder = run / name
        if not folder.exists():
            cases[name] = dict(status='NOT_RUN_AFTER_ABORT')
            continue
        result = dict(status=summary['cases'][name]['status'], engine_present=(folder / 'model.engine').exists())
        cleanup = read(folder / 'child_cleanup.json')
        assert cleanup['direct_python_api_child_terminated']
        result['child_cleanup'] = cleanup
        if (folder / 'parsed_network_request.json').exists():
            request = read(folder / 'parsed_network_request.json')
            requests[name] = request
            assert request['case'] == case and request['spec_sha256'] == sha(raw / 'spec.json')
            assert request['model_sha256'] == spec['variants']['weight_bypass']['model_sha256']
            assert request['workspace_bytes_actual'] == 1024 * 1024 * 1024 and request['avg_timing_iterations_actual'] == 1
            assert request['no_layer_precision_or_output_type_setters_or_resets']
            flags = request['flags']['get_flag']
            for flag, enabled in [('INT8', True), ('FP16', True), ('TF32', False), ('PREFER_PRECISION_CONSTRAINTS', False), ('OBEY_PRECISION_CONSTRAINTS', name.endswith('_obey'))]:
                assert flags[flag] == dict(supported=True, enabled=enabled)
            assert len(request['clip_attributes']) == 1
            assert all(c['after'] == dict(alpha=-127.0, beta=127.0) and c['explicit_importer_correction'] for c in request['clip_attributes'])
            before, after = read(folder / 'parsed_constraints_before.json'), read(folder / 'parsed_constraints_after.json')
            assert constraints(before) == constraints(after)
            parsed[name] = before
            precision = [r for r in before if r['precision_is_set'].get('value')]
            outputs = [v for r in before for v in r['output_types'] if v['output_type_is_set'].get('value')]
            errors = [r['name'] for r in before if not r['precision_is_set'].get('supported') or 'precision_value_error' in r or any(not v['output_type_is_set'].get('supported') or 'requested_output_type_error' in v or 'tensor_query_error' in v for v in r['output_types'])]
            assert not errors
            result['parser_constraints'] = dict(parsed_layers=len(before), precision_set_count=len(precision), output_type_set_count=len(outputs), metadata_errors=errors)
            result['actual_flags'] = flags
        build = read(folder / 'build_result.json')
        result['build'] = build
        if (folder / 'raw_outputs.npz').exists():
            saved_result = read(folder / 'result.json')
            assert build['status'] == 'DFL_WEIGHT_BYPASS_ENGINE_SAVED'
            assert sha(folder / 'model.engine') == build['engine_sha256'] == saved_result['engine_sha256']
            assert sha(folder / 'raw_outputs.npz') == saved_result['raw_outputs_sha256']
            assert read(folder / 'engine_layers.json') == read(folder / 'runner_engine_layers.json')
            inspector = read(folder / 'engine_layers.json')
            rows = [r for r in inspector['Layers'] if 'Convolution' in r.get('LayerType', '')]
            mapped, used = [], []
            for source in spec['conv_names']:
                indices = [i for i, r in enumerate(rows) if source in r['Name']]
                record = dict(source=source, matched_actual_indices=indices, unique=len(indices) == 1)
                if len(indices) == 1:
                    index = indices[0]
                    used.append(index)
                    row = rows[index]
                    record.update(actual_row=row, int8_input_weight=bool(row['Inputs']) and all('int8' in v.get('Format/Datatype', '').lower() for v in row['Inputs']) and row['Weights']['Type'].lower() == 'int8',
                                  int8_output=bool(row['Outputs']) and all('int8' in v.get('Format/Datatype', '').lower() for v in row['Outputs']),
                                  named_integer_tactic=bool(re.search('i8|s8|int8|imma', row.get('TacticName', ''), re.I)))
                mapped.append(record)
            result['actual_conv_mapping'] = dict(actual_count=len(rows), source_mappings=mapped, one_unique_source_row=len(rows) == 1 and len(set(used)) == 1 and len(used) == 1)
            bindings = read(folder / 'runtime_bindings.json')
            assert len(bindings) == 2 and all(b['dtype'] == 'DataType.FLOAT' and b['format'] == 'TensorFormat.LINEAR' for b in bindings)
            with np.load(folder / 'raw_outputs.npz', allow_pickle=False) as saved:
                first, repeat = saved['first'].copy(), saved['repeat'].copy()
            assert first.shape == repeat.shape == expected.shape and first.dtype == repeat.dtype == np.float32
            result['repeat'], result['cpu_difference'] = compare(first, repeat), compare(first, expected)
            result['descriptive_reduction_comparisons'] = dict(repeat=validate_stats(result['repeat'], saved_result['repeat_comparison']), cpu=validate_stats(result['cpu_difference'], saved_result['comparison_to_frozen_cpu_ort']))
            arrays[name] = first
        else:
            result['inference_run'] = False
            result['failure'] = read(folder / 'build_failure.json') if (folder / 'build_failure.json').exists() else None
        cases[name] = result
    pair_name = 'weight_bypass_none_vs_weight_bypass_obey'
    pair = compare(arrays['weight_bypass_none'], arrays['weight_bypass_obey']) if len(arrays) == 2 else dict(status='UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED')
    reductions = validate_stats(pair, summary['cross_native_comparisons'][pair_name])
    if len(requests) == 2:
        a, b = requests['weight_bypass_none'], requests['weight_bypass_obey']
        assert constraints(parsed['weight_bypass_none']) == constraints(parsed['weight_bypass_obey'])
        assert [k for k in a['flags']['get_flag'] if a['flags']['get_flag'][k] != b['flags']['get_flag'][k]] == ['OBEY_PRECISION_CONSTRAINTS']
        assert a['model_sha256'] == b['model_sha256']
    monitor, guard = read(raw / 'monitor/status.json'), read(raw / 'supplemental_group_guard.json')
    assert monitor['status'] != 'RUNNING' and guard['status'] == 'COMPLETE' and guard['owned_group_fully_empty'] and not guard['final_group_members']
    assert initial == {p.relative_to(raw).as_posix(): sha(p) for p in raw.rglob('*') if p.is_file()}
    report = dict(schema='rampq.gate05.dfl_weight_bypass_saved_audit/v1', client_at=datetime.now().astimezone().isoformat(), status='DFL_WEIGHT_BYPASS_SAVED_EVIDENCE_AUDITED',
                  receipt_files=len(receipt['files']), successful_output_cases=len(arrays), cases=cases, cross_native_comparison=pair, derived_cross_reduction_comparison=reductions,
                  monitor_status=monitor['status'], monitor_seconds=monitor['elapsed_seconds'], guard_intervention=guard['intervention_performed'], guard_group_empty=True, raw_tree_unchanged=True,
                  original_complete_model_guard='FAILED_UNCHANGED', full_model_retry=False, full_model_failure_resolved=False, dataset_evaluation_run=False, new_tolerances_defined=False, gate05_status='INCOMPLETE')
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], successful_output_cases=len(arrays), cases={k:v['status'] for k,v in cases.items()}, pair=pair), ensure_ascii=False))


if __name__ == '__main__':
    main()
