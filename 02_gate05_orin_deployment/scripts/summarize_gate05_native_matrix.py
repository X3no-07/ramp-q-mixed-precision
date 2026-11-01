"""Verify raw board receipts and summarize the frozen native diagnostic matrix."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/gate05/20261019_precision_native'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_receipt(folder):
    receipt=read(folder/'remote_receipt.json')
    for name,info in receipt['files'].items():
        path=folder/name
        assert path.resolve().is_relative_to(folder.resolve())
        assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],name
    return len(receipt['files'])


def main():
    fp=RUN/'fp16_control/post_smoke_evidence'
    mixed=RUN/'mixed_matrix'
    fp_count,mixed_count=verify_receipt(fp),verify_receipt(mixed)
    build=read(fp/'build/build_result.json')
    precision=read(fp/'build/precision_audit.json')
    smoke=read(fp/'smoke/smoke_result.json')
    assert build['configuration']=='fp16_control' and build['status']=='BUILT_AND_DESERIALIZED'
    assert sha(fp/'build/model.engine')==build['engine_sha256']==precision['engine_sha256']==smoke['engine_sha256']
    assert precision['requested_conv_formats_complete'] and precision['verified_fp16_format_convs']==88
    assert precision['verified_int8_format_convs']==0
    assert precision['layer_file_sha256']==sha(fp/'build/engine_layers.json')
    assert smoke['status']=='DIAGNOSTIC_FIXED_IMAGE_EXECUTION_OK' and smoke['execution_success']
    assert all(smoke['all_outputs_finite'].values()) and len(smoke['all_outputs_finite'])==2
    assert smoke['raw_outputs_sha256']==sha(fp/'smoke/raw_outputs.npz')
    assert smoke['dataset_evaluation_run'] is False and smoke['accuracy_pass_claim'] is False
    rows=[{'configuration':'fp16_control','source_conv_count':88,'expected_int8':0,'expected_fp16':88,
           'build_status':build['status'],'engine_sha256':build['engine_sha256'],
           'actual_requested_formats_verified':True,'single_image_executed':True,
           'all_outputs_finite':True,'detections':smoke['detections']}]
    queue=read(mixed/'queue/status.json')
    assert queue['status']=='FROZEN_BUILD_QUEUE_FINISHED_AUDIT_AND_SMOKE_PENDING'
    expected={'r_only_G1':(82,6),'r_only_G1_G5':(74,14),'r_only_G1_G2':(77,11),'q_only_G1':(6,82)}
    assert [r['configuration'] for r in queue['completed_builds']]==list(expected)
    for config,(int8_count,fp16_count) in expected.items():
        folder=mixed/config
        result=read(folder/'build/build_result.json')
        spec=read(folder/'inputs/spec.json')
        local_bundle=ROOT/'artifacts/gate05/20261019_precision_runtime_bundles'/config
        assert sha(folder/'inputs/spec.json')==sha(local_bundle/'spec.json')==result['spec_sha256']
        assert result['onnx_sha256']==spec['graph_sha256']==sha(local_bundle/'model.onnx')
        assert (spec['quantized_conv_count'],spec['fp16_conv_count'])==(int8_count,fp16_count)
        monitor=read(mixed/('queue/monitor_'+config+'/status.json'))
        assert monitor['status']=='BUILD_SCRIPT_FAILED' and monitor['reason'] is None
        log=(folder/'build/trtexec_build.log').read_text(errors='replace')
        assert result['status']=='BUILD_OR_CONTRACT_VALIDATION_FAILED'
        assert 'filterQDQFormats' in log and 'candidateRequirements.empty()' in log
        assert not (folder/'build/model.engine').exists()
        rows.append({'configuration':config,'source_conv_count':88,'expected_int8':int8_count,
            'expected_fp16':fp16_count,'build_status':result['status'],
            'build_script_seconds':result['elapsed_seconds'],'error_class':'TRT_INTERNAL_FILTER_QDQ_FORMATS',
            'actual_requested_formats_verified':False,'single_image_executed':False,
            'monitor_safety_stop':False})
    qal=read(ROOT/'artifacts/gate05/20261019_int8_avg1/audit_summary.json')
    assert qal['status']=='FULL_QDQ_PE_TAP_BUILD_TIMEOUT_INCONCLUSIVE'
    rows.insert(1,{'configuration':'q_all','source_conv_count':88,'expected_int8':88,'expected_fp16':0,
        'build_status':qal['status'],'actual_requested_formats_verified':False,'single_image_executed':False,
        'monitor_seconds':qal['resources']['monitor_elapsed_seconds']})
    report={'schema':'rampq.gate05.fixed_matrix_checkpoint/v1','client_at':datetime.now().astimezone().isoformat(),
        'status':'FROZEN_OBEY_MATRIX_COMPLETE_PREFER_RETRIES_PENDING',
        'gate05_status':'INCOMPLETE','all_six_frozen_configurations_attempted':True,
        'successful_native_full_engines':1,'successful_new_fixed_image_executions':1,
        'verified_original_files':{'fp16_control_post_smoke':fp_count,'mixed_matrix':mixed_count},
        'configurations':rows,'original_q_all_numerical_guard':'FAILED_UNCHANGED',
        'dataset_inference_run':False,'dataset_acceptance_rules_frozen':False,
        'causal_or_recovery_claim_established':False,
        'next_required_work':'Use the demonstrated prefer request for the full frozen mixed configurations; retain exact actual-format standards and all original failures.'}
    (RUN/'matrix_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='configurations'},indent=2))


if __name__=='__main__':
    main()
