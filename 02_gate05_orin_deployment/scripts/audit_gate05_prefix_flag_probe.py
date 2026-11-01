"""Audit saved four-case prefix outputs and actual native formats, without execution."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE_NAMES = ['original_none','original_obey','code_clip_none','code_clip_obey']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(a,b):
    assert a.shape == b.shape and a.dtype == b.dtype == np.float32
    finite = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[finite].astype(np.float64)-b[finite].astype(np.float64))
    return {'all_pairs_finite':bool(finite.all()),'nonfinite_pair_count':int((~finite).sum()),
            'numeric_equal':bool(np.array_equal(a,b)),
            'byte_identical':a.tobytes()==b.tobytes(),'different_elements':int(np.count_nonzero(a!=b)),
            'max_abs':float(d.max()) if d.size else None,'mean_abs':float(d.mean()) if d.size else None,
            'rms_difference':float(np.sqrt(np.mean(d*d))) if d.size else None,
            'descriptive_only_no_new_pass_criterion':True}


def constraint_signature(rows):
    return [{k:r[k] for k in ('index','name','precision_is_set','precision','precision_value_error','output_types') if k in r}
            for r in rows]


def audit_statistics(recomputed, native):
    """Keep exact raw facts separate from platform-dependent FP64 reductions.

    This audit defines no numerical acceptance tolerance. Mean and RMS are
    descriptive and both saved values are retained, including their delta.
    """
    assert set(recomputed) == set(native)
    reductions = ('mean_abs', 'rms_difference')
    for field in set(recomputed) - set(reductions):
        assert recomputed[field] == native[field], field
    return {field: {'host_recomputed': recomputed[field], 'native_reported': native[field],
                    'host_minus_native': (recomputed[field] - native[field]
                                          if recomputed[field] is not None and native[field] is not None else None)}
            for field in reductions}


def constraints_summary(rows):
    precision = [r for r in rows if r['precision_is_set'].get('value') is True]
    outputs = [v for r in rows for v in r['output_types'] if v['output_type_is_set'].get('value') is True]
    errors = []
    for r in rows:
        if not r['precision_is_set'].get('supported'): errors.append(r['name']+':precision_is_set')
        if 'precision_value_error' in r: errors.append(r['name']+':precision')
        for v in r['output_types']:
            if not v['output_type_is_set'].get('supported') or 'requested_output_type_error' in v:
                errors.append(r['name']+':output:'+str(v['index']))
    return {'parsed_layers':len(rows),'precision_set_count':len(precision),'output_type_set_count':len(outputs),
            'metadata_errors':errors}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.raw.resolve();output=args.output.resolve()
    assert output.parent.exists() and not output.exists() and raw not in output.parents
    initial={p.relative_to(raw).as_posix():sha(p) for p in raw.rglob('*') if p.is_file()}
    receipt=read(raw/'remote_receipt.json')
    assert set(initial)==set(receipt['files'])|{'remote_receipt.json'}
    for rel,rec in receipt['files'].items():
        p=(raw/rel).resolve();assert raw in p.parents
        assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],rel
    manifest=read(raw/'manifest.json')
    for rel,rec in manifest['files'].items():
        p=(raw/rel).resolve();assert raw in p.parents
        assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],rel
    assert sha(raw/'protocol.json')=='c7fdcada89c685ed52f8f65d8318854982d58f52f16cfce44f3129cdd17a307d'
    spec=read(raw/'spec.json')
    assert spec['conv_count']==1 and spec['graph_node_count_per_variant']==11
    assert spec['original_complete_model_numerical_guard']=='FAILED_UNCHANGED'
    runs=list((raw/'results').glob('probe_*'));assert len(runs)==1
    run=runs[0];summary=read(run/'summary.json')
    assert summary['requested_case_order']==CASE_NAMES
    expected={}
    with np.load(raw/spec['cpu_expected_outputs_file'],allow_pickle=False) as saved:
        expected={k:saved[k].copy() for k in saved.files}
    arrays={};cases={};requests={};constraint_rows={}
    for name in CASE_NAMES:
        case=next(v for v in spec['cases'] if v['id']==name)
        p=run/name
        if not p.exists():
            cases[name]={'status':'NOT_RUN_AFTER_SAFE_ABORT'};continue
        request_path=p/'parsed_network_request.json'
        record={'status':summary['cases'][name]['status']}
        if request_path.exists():
            req=read(request_path);requests[name]=req
            assert req['case']==case and req['spec_sha256']==sha(raw/'spec.json')
            assert req['model_sha256']==spec['variants'][case['variant']]['model_sha256']
            assert req['workspace_bytes_actual']==1024*1024*1024 and req['avg_timing_iterations_actual']==1
            flags=req['flags']['get_flag']
            for key,value in [('INT8',True),('FP16',True),('TF32',False),
                              ('PREFER_PRECISION_CONSTRAINTS',False),('OBEY_PRECISION_CONSTRAINTS',case['precision_constraint_flag']=='obey')]:
                assert flags[key]=={'supported':True,'enabled':value},(name,key)
            assert req['no_layer_precision_or_output_type_setters_or_resets']
            for c in req['clip_attributes']:
                assert c['explicit_importer_correction']==case['apply_native_clip_attribute_correction']
                if case['variant']=='code_clip':assert c['after']=={'alpha':-127.,'beta':127.}
                else:assert c['before']==c['after']
            before,after=read(p/'parsed_constraints_before.json'),read(p/'parsed_constraints_after.json')
            assert constraint_signature(before)==constraint_signature(after)
            constraint_rows[name]=before
            record['parser_constraints']=constraints_summary(before)
            record['actual_config_flags']=req['flags']
            record['clip_attributes']=req['clip_attributes']
        build=read(p/'build_result.json') if (p/'build_result.json').exists() else {}
        record['build_status']=build.get('status');record['failure_phase']=build.get('phase')
        record['engine_present']=(p/'model.engine').exists()
        if (p/'child_cleanup.json').exists():
            cleanup=read(p/'child_cleanup.json');assert cleanup['direct_python_api_child_terminated']
            record['child_cleanup']=cleanup
        if (p/'raw_outputs.npz').exists():
            result=read(p/'result.json')
            assert build['status']=='PREFIX_FLAG_ENGINE_SAVED'
            assert sha(p/'model.engine')==build['engine_sha256']==result['engine_sha256']
            assert sha(p/'raw_outputs.npz')==result['raw_outputs_sha256']
            bindings=read(p/'runtime_bindings.json');assert len(bindings)==2
            assert all(v['dtype']=='DataType.FLOAT' and v['format']=='TensorFormat.LINEAR' for v in bindings)
            inspector=read(p/'engine_layers.json')
            assert inspector==read(p/'runner_engine_layers.json')
            convs=[v for v in inspector['Layers'] if 'Convolution' in v.get('LayerType','')]
            assert len(convs)==1 and spec['conv_name'] in convs[0]['Name']
            conv=convs[0]
            assert conv['Weights']['Count']==432 and conv['Groups']==1 and conv['OutMaps']==16
            formats=[v.get('Format/Datatype','') for v in conv['Inputs']]
            outputs=[v.get('Format/Datatype','') for v in conv['Outputs']]
            native_int8=(conv['Weights']['Type'].lower() in ('int8','int8_t','signed int8','signed byte') and
                         bool(formats) and all('int8' in v.lower() for v in formats))
            integer_tactic=bool(re.search('i8|s8|int8|imma',conv.get('TacticName',''),re.I))
            record['actual_conv']={'input_formats':formats,'weights':conv['Weights'],'output_formats':outputs,
                                  'activation':conv.get('Activation'),'tactic_name':conv.get('TacticName'),
                                  'int8_input_weight_formats_verified':native_int8,
                                  'named_integer_tactic_verified':native_int8 and integer_tactic,
                                  'output_int8':bool(outputs) and all('int8' in v.lower() for v in outputs),
                                  'pointwise_expression_operations':conv.get('PointWiseExpressionOperations')}
            with np.load(p/'raw_outputs.npz',allow_pickle=False) as saved:
                first,repeat=saved['first'].copy(),saved['repeat'].copy()
            assert first.shape==repeat.shape==(1,16,320,320)
            assert first.dtype==repeat.dtype==np.float32
            key=spec['variants'][case['variant']]['expected_output_npz_keys']['rampq_model_1_conv_a_dq']
            record['all_outputs_finite']=bool(np.isfinite(first).all() and np.isfinite(repeat).all())
            record['repeat']=difference(first,repeat)
            record['cpu_difference']=difference(first,expected[key])
            record['derived_reduction_comparison']={
                'repeat':audit_statistics(record['repeat'],result['repeat_comparison']),
                'cpu_difference':audit_statistics(record['cpu_difference'],result['comparison_to_frozen_cpu_ort'])}
            arrays[name]=first
        else:
            record['inference_run']=False
            if (p/'builder.log').exists():
                lines=(p/'builder.log').read_text(errors='replace').splitlines()
                record['error_lines']=[v for v in lines if '[E]' in v or 'RuntimeError:' in v or 'AssertionError' in v]
        cases[name]=record
    pairs={};cross_reductions={}
    for a,b in [('original_none','original_obey'),('code_clip_none','code_clip_obey'),
                ('original_none','code_clip_none'),('original_obey','code_clip_obey')]:
        key=a+'_vs_'+b
        pairs[key]=difference(arrays[a],arrays[b]) if a in arrays and b in arrays else {'status':'UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED'}
        cross_reductions[key]=audit_statistics(pairs[key],summary['cross_native_comparisons'][key])
    flag_controls={}
    for variant in ('original','code_clip'):
        a,b=variant+'_none',variant+'_obey'
        if a in constraint_rows and b in constraint_rows:
            # Full constraint rows also include tensor shapes/dtypes. Raw layer
            # wrapper metadata may expose ActivationType after Clip rebinding.
            flag_controls[variant]={'same_graph_sha256':requests[a]['model_sha256']==requests[b]['model_sha256'],
                'parser_constraints_byte_semantics_equal':constraint_signature(constraint_rows[a])==constraint_signature(constraint_rows[b]),
                'only_config_flag_changed':[k for k in requests[a]['flags']['get_flag'] if requests[a]['flags']['get_flag'][k]!=requests[b]['flags']['get_flag'][k]]}
            assert flag_controls[variant]['only_config_flag_changed']==['OBEY_PRECISION_CONSTRAINTS']
            assert flag_controls[variant]['parser_constraints_byte_semantics_equal']
    status,guard=read(raw/'monitor/status.json'),read(raw/'supplemental_group_guard.json')
    assert status['status']!='RUNNING' and guard['status']=='COMPLETE' and guard['owned_group_fully_empty']
    assert guard['final_group_members']==[]
    assert initial=={p.relative_to(raw).as_posix():sha(p) for p in raw.rglob('*') if p.is_file()}
    report={'schema':'rampq.gate05.prefix_flag_saved_native_audit/v1','client_at':datetime.now().astimezone().isoformat(),
            'status':'PREFIX_FLAG_SAVED_EVIDENCE_AUDITED','raw_receipt_files':len(receipt['files']),
            'run_relative_path':run.relative_to(raw).as_posix(),'monitor_status':status['status'],
            'monitor_seconds':status.get('elapsed_seconds'),'cases':cases,'cross_native_comparisons':pairs,
            'derived_cross_reduction_comparison':cross_reductions,
            'native_reported_cross_statistics':summary['cross_native_comparisons'],
            'derived_reduction_policy':'Mean and RMS are descriptive host/native reductions; both retained with delta, no acceptance tolerance or guard amendment. Raw facts remain exact.',
            'same_topology_flag_controls':flag_controls,'successful_output_cases':len(arrays),
            'guard_fully_empty':True,'guard_intervention':guard['intervention_performed'],
            'raw_tree_bytes_unchanged':True,'original_complete_model_guard':'FAILED_UNCHANGED',
            'full_model_failure_resolved':False,'new_tolerances_defined':False,'dataset_evaluation_run':False,
            'accuracy_causal_recovery_or_performance_pass_claim':False,'gate05_status':'INCOMPLETE',
            'limits':'One Conv context only. Different small-case outcomes localize this prefix; all-case success does not prove the failed complete q_all graph builds or executes.'}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'case_statuses':{k:v['status'] for k,v in cases.items()},
                      'successful_output_cases':len(arrays),'flag_controls':flag_controls},ensure_ascii=False))


if __name__=='__main__':
    main()
