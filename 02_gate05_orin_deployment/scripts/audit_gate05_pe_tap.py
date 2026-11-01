"""Verify collected originals and summarize one full PE-tap diagnostic checkpoint."""
import hashlib
import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/gate05/20261018_int8_pe_tap'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    global RUN
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--client-start-date',default='2026-10-18')
    parser.add_argument('--local-output-check',type=Path,
                        default=ROOT/'artifacts/gate05/20261018_int8_pe_tap/candidate_cpu_output_check.json')
    args=parser.parse_args()
    RUN=args.run.resolve()
    receipt=read(RUN/'remote_receipt.json')
    for name,info in receipt['files'].items():
        path=RUN/name
        assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],name
    monitor=read(RUN/'monitor/status.json')
    assert monitor['status']!='RUNNING'
    samples=[json.loads(line) for line in (RUN/'monitor/resources.jsonl').read_text().splitlines() if line.strip()]
    build=read(RUN/'build/build_result.json') if (RUN/'build/build_result.json').exists() else None
    precision=read(RUN/'build/precision_audit.json') if (RUN/'build/precision_audit.json').exists() else None
    smoke=read(RUN/'smoke/smoke_result.json') if (RUN/'smoke/smoke_result.json').exists() else None
    candidate=read(RUN/'inputs/candidate.json')
    local=read(args.local_output_check)
    assert candidate['candidate_sha256']==local['candidate_sha256']
    assert read(RUN/'inputs/preflight.json')['status']=='LOCAL_QDQ_PREFLIGHT_FAILED'
    built=bool(build and build['status']=='BUILT_AND_DESERIALIZED')
    executed=bool(smoke and smoke['execution_success'] and smoke['finite_output'])
    if built:
        assert sha(RUN/'build/model_q_all.engine')==build['engine_sha256']
        assert build['onnx_sha256']==candidate['candidate_sha256']
    if precision:
        assert precision['layer_file_sha256']==sha(RUN/'build/engine_layers.json')
        assert precision['source_onnx_sha256']==candidate['candidate_sha256']
    if executed:
        assert built and smoke['engine_sha256']==build['engine_sha256']
        assert smoke['boot_id']==monitor['boot_id']
        assert all(smoke['all_outputs_finite'].values())
    resources={'samples':len(samples),'monitor_elapsed_seconds':monitor['elapsed_seconds'],
        'max_temperature_c':max(max(s['temperatures_c'].values()) for s in samples),
        'min_available_mib':min(s['MemAvailable_kib'] for s in samples)/1024,
        'max_swap_used_mib':max(s['SwapTotal_kib']-s['SwapFree_kib'] for s in samples)/1024}
    kernel=(RUN/'monitor/kernel.log').read_text(errors='replace')
    events=[line for line in kernel.splitlines() if line.strip() and not line.startswith('-- Logs begin')]
    timeout=monitor['status']=='STOPPED_BY_MONITOR' and monitor['reason']=='Build supervision time limit reached'
    report={'schema':'rampq.gate05.pe_tap_checkpoint/v1','client_start_date':args.client_start_date,
        'client_completed_at':datetime.now().astimezone().isoformat(),
        'status':'FULL_QDQ_PE_TAP_EXECUTED_DIAGNOSTIC' if executed else
                 ('FULL_QDQ_PE_TAP_BUILT_EXECUTION_PENDING' if built else
                  ('FULL_QDQ_PE_TAP_BUILD_TIMEOUT_INCONCLUSIVE' if timeout else 'FULL_QDQ_PE_TAP_BUILD_NOT_COMPLETED')),
        'configuration':'q_all','gate05_status':'INCOMPLETE',
        'board_original_files_verified':len(receipt['files']),
        'boot_id':monitor['boot_id'],'monitor_status':monitor['status'],'monitor_stop_reason':monitor['reason'],
        'resources':resources,'kernel_events':events,'build':build,
        'full_model_engine_built':built,'full_model_single_image_executed':executed,
        'precision_audit_status':precision['status'] if precision else 'NOT_RUN',
        'int8_format_source_conv_count':precision['convs_with_int8_input_and_weight_formats'] if precision else 0,
        'named_integer_tactic_source_conv_count':precision['verified_int8_convs'] if precision else 0,
        'source_convs_without_tactic_name':precision['convs_without_tactic_name'] if precision else None,
        'local_pytorch_onnx_parity':'FAILED_UNCHANGED',
        'candidate_main_cpu_output_check':local['status'],
        'candidate_cpu_check_reference':str(args.local_output_check.resolve()),
        'candidate_cpu_check_reference_sha256':sha(args.local_output_check),
        'single_image':smoke,
        'limits':'One hardware diagnostic configuration; no fixed-dataset accuracy, recovery configurations, causal comparison, or performance benchmark.'}
    quick=RUN/'diagnostic_bundle_v4_pe_tap_avg1.zip'
    if quick.exists():
        report['next_retry']={'status':'STAGED_NOT_BUILT_OR_EXECUTED','bundle_sha256':sha(quick),
            'kernel_selection_avg_timing':1,'max_build_seconds':1800,
            'graph_sha256':candidate['candidate_sha256']}
    (RUN/'audit_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (RUN/'checkpoint.json').write_text(json.dumps({key:value for key,value in report.items() if key not in ('build','single_image')},indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in RUN.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '.ultralytics' not in p.parts)
    (RUN/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(RUN).as_posix()+'\n' for p in files),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('build','single_image')},indent=2))


if __name__=='__main__':
    main()
