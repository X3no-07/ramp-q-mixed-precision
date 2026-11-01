"""Adapt the unchanged two-call d157 helper to audited matched-family engines.

All input checks are stdlib-only. The actual profiled runner must have a private
PID=PGID=SID session (profile wrapper should use setsid --wait). It authenticates
an independent escaped-child guard and waits for its acknowledgement before any
GPU import. This is fixed Selection-image diagnosis, never dataset inference.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
EXECUTION_HELPER_SHA = 'd1576ea834bd48ad78df3bf920b315726c1f0da8fe388eb7c3450dda8d93e0ca'
BUILD_CONTRACT_SHA = '4d2ee7d283abf6cd75b6d76e58870f79f11392fa06eaef37936fab6d7bafca34'
MATCHED_AUDITOR_SHA = '97e031e6ed16f18634483fa3f0f1fa98903f39e12551b94390feab64a8a66d30'
OWNED_GUARD_SHA = '4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4'
CONFIGURATIONS = ('fp16_control','r_only_G1','r_only_G1_G5','r_only_G1_G2','q_only_G1')
MODULES = {}

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))

def write_new(path, obj):
    with path.open('x',encoding='utf-8') as out:
        json.dump(obj,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())

def load_exact(path, expected):
    path=path.resolve();require(sha(path)==expected,'Tool identity differs: '+path.name)
    key=str(path)
    if key not in MODULES:
        spec=importlib.util.spec_from_file_location('rampq_fixed_'+hashlib.sha256(key.encode()).hexdigest()[:16],path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);MODULES[key]=module
    return MODULES[key]

def validate_inputs(args):
    root,build,work=args.root.resolve(),args.build.resolve(),args.work_root.resolve()
    require(root.is_dir() and build.is_dir() and root in build.parents and work.is_dir() and work!=root and
            root not in work.parents and work not in root.parents,'Use distinct member and fixed-input work roots')
    protocol_path,audit_path=args.protocol.resolve(),args.audit.resolve()
    require(protocol_path.parent==work and audit_path.parent==work and
            sha(protocol_path)==args.protocol_sha256 and sha(audit_path)==args.audit_sha256,'Work protocol/audit pins differ')
    protocol=read(protocol_path)
    require(protocol['schema']=='rampq.gate05.matched_control_fixed_input_protocol/v1','Wrong matched smoke protocol')
    require(protocol['status']=='FROZEN_BEFORE_MATCHED_FIXED_INPUT_EXECUTION' or
            (args.check_input_only and protocol['status']=='INPUT_ONLY_COMPATIBILITY_FIXTURE_NO_GPU'),
            'Only a frozen protocol may execute; compatibility fixture permits input check only')
    require(protocol['adapter_sha256']==sha(Path(__file__).resolve()) and protocol['execution_helper_sha256']==EXECUTION_HELPER_SHA and
            protocol['build_contract_sha256']==BUILD_CONTRACT_SHA and protocol['matched_auditor_sha256']==MATCHED_AUDITOR_SHA,
            'Frozen execution tools differ')
    helper=load_exact(work/'jetson_gate05_control_fixed_input.py',EXECUTION_HELPER_SHA)
    contract=load_exact(work/'jetson_build_matched_control.py',BUILD_CONTRACT_SHA)
    load_exact(work/'jetson_owned_group_guard.py',OWNED_GUARD_SHA)
    helper.verify_manifest(work)
    require(sha(work/'jetson_matched_profile_child_guard.py')==protocol['child_guard_sha256'],'Child guard differs')
    spec,parent,_native=contract.validate_inputs(root)
    count=spec['quantized_conv_count']
    require(spec['configuration'] in CONFIGURATIONS and spec['configuration']==protocol['configuration'] and
            protocol['member_spec_sha256']==sha(root/'spec.json') and protocol['member_manifest_sha256']==sha(root/'manifest.json') and
            protocol['source_graph_sha256']==sha(root/'model.onnx')==spec['graph_sha256'] and
            protocol['parent_spec_sha256']==sha(root/'parent_bundle/spec.json') and
            protocol['parent_manifest_sha256']==sha(root/'parent_bundle/manifest.json'),'Member/parent lineage differs')
    require(protocol['executions']==2 and protocol['nms']==helper.NMS and protocol['nvtx_range']==helper.NVTX_RANGE and
            protocol['supervision']['max_seconds']==300 and protocol['supervision']['private_process_group'] is True,
            'Two-call fixed-input supervision scope differs')
    require(protocol['dataset_evaluation_run'] is False and protocol['new_CPU_model_executions']==0 and
            protocol['new_builds']==0 and protocol['new_tolerances_defined'] is False and
            protocol['original_complete_model_numerical_guard']=='FAILED_UNCHANGED' and protocol['gate05_status']=='INCOMPLETE',
            'Protected scope differs')
    require(protocol['expected_boot_id']=='692c7544-b8bd-415b-939c-b11cdbd9f7f0' and
            protocol['monitor_build_script']=='run_fixed.py' and protocol['child_guard_required_before_GPU'] is True,
            'Runner/monitor safety contract missing')
    require(protocol['original_image_shape']==[1080,1920] and protocol['fixed_image_sha256']==helper.FIXED_IMAGE_SHA256,
            'Original fixed image identity/geometry differs')
    result_path,request_path=build/'build_result.json',build/'build_request.json'
    result,request=read(result_path),read(request_path)
    require(result['schema']=='rampq.gate05.matched_control_build_result/v1' and
            result['status']=='BUILT_AND_DESERIALIZED_BUILD_ONLY' and result['configuration']==spec['configuration'] and
            result['native_inference_executions']==0 and result['engine_execution_context_created'] is False,'Engine is not build-only matched member')
    engine=build/'model.engine'
    require(sha(engine)==result['engine_sha256']==args.engine_sha256==protocol['engine_sha256'] and
            sha(result_path)==protocol['build_result_sha256'] and sha(request_path)==protocol['build_request_sha256'] and
            sha(build/'engine_layers.json')==protocol['engine_layers_sha256'],'Frozen build/engine identity differs')
    identity=result['tool_identity']
    require(identity==read(build/'tool_hashes.json')==request['tool_identity'] and
            identity['builder_sha256']==BUILD_CONTRACT_SHA and identity['spec_sha256']==sha(root/'spec.json') and
            identity['manifest_sha256']==sha(root/'manifest.json') and identity['graph_sha256']==spec['graph_sha256'] and
            identity['parent_spec_sha256']==sha(root/'parent_bundle/spec.json') and
            identity['parent_manifest_sha256']==sha(root/'parent_bundle/manifest.json'),'Native tool/build source lineage differs')
    require(request['desired_source_requests']==spec['requests'] and request['runtime_precision_constraints']=='none' and
            request['explicit_layer_precision_constraint_count']==request['explicit_layer_output_type_constraint_count']==request['layer_constraint_reset_count']==0,
            'Source mask or parser preservation differs')
    for name,enabled in (('INT8',count>0),('FP16',True),('TF32',False),('OBEY_PRECISION_CONSTRAINTS',False),('PREFER_PRECISION_CONSTRAINTS',False)):
        require(request['flags']['get_flag'][name]=={'supported':True,'enabled':enabled},'Actual build flag differs: '+name)
    expected=spec['expected_binding_contract']
    require(helper.binding_signature(expected)==helper.binding_signature(protocol['expected_binding_contract'])==helper.binding_signature(result['bindings']) and
            len(expected)==3 and all(r['format']=='TensorFormat.LINEAR' for r in result['bindings']),'Three FLOAT/LINEAR bindings differ')
    audit=read(audit_path)
    require(audit['audit_tool_identity']['auditor_sha256']==MATCHED_AUDITOR_SHA and
            audit['configuration']==spec['configuration'] and audit['native_formats_tactics_and_fusion_criteria_changed'] is False and
            audit['FP16_output_requirement_changed'] is False,'Actual audit implementation or rules differ')
    for key,value in {'engine_sha256':args.engine_sha256,'onnx_sha256':spec['graph_sha256'],
          'source_spec_sha256':sha(root/'spec.json'),'manifest_sha256':sha(root/'manifest.json'),
          'parent_spec_sha256':sha(root/'parent_bundle/spec.json'),'parent_manifest_sha256':sha(root/'parent_bundle/manifest.json'),
          'layer_file_sha256':sha(build/'engine_layers.json'),'build_result_sha256':sha(result_path),'build_request_sha256':sha(request_path),
          'cpu_expected_outputs_sha256':spec['cpu_expected_outputs_sha256']}.items():
        require(audit[key]==value,'Actual audit lineage differs: '+key)
    require(protocol['precision_audit_sha256']==args.audit_sha256 and
            (not args.diagnostic_formats_only or protocol['diagnostic_formats_only_allowed'] is True),'Diagnostic audit authorization differs')
    # The legacy verifier receives a view only; on-disk own spec remains exact.
    view=dict(spec);view['fp16_conv_count']=88-count
    strict,gaps=helper.verify_requested_precision(audit,view,args.diagnostic_formats_only)
    input_path=helper.contained(root,spec['input_file']);cpu_path=helper.contained(root,spec['cpu_expected_outputs_file'])
    require(sha(input_path)==spec['input_file_sha256']==protocol['input_sha256']==helper.INPUT_SHA256 and
            protocol['input_file']==spec['input_file'],'Exact saved Selection input differs')
    with input_path.open('rb') as f:header,offset=helper.npy_header(f)
    require(header['shape']==(1,3,640,640) and input_path.stat().st_size==offset+4*math.prod(header['shape']),'Saved input header differs')
    reference=protocol['cpu_reference']
    require(reference['file']==spec['cpu_expected_outputs_file'] and reference['sha256']==sha(cpu_path)==spec['cpu_expected_outputs_sha256'] and
            reference['output_npz_keys']==spec['expected_output_npz_keys'] and set(reference['output_npz_keys'].values())=={'main_output','pe_output'},
            'Own full CPU reference identity/keys differ')
    desired={r['name']:r for r in expected if not r['input']}
    require(set(reference['output_npz_keys'])==set(desired),'CPU output coverage differs')
    with zipfile.ZipFile(cpu_path) as z:
        for name,key in reference['output_npz_keys'].items():
            with z.open(key+'.npy') as f:header,offset=helper.npy_header(f)
            require(list(header['shape'])==desired[name]['shape'] and z.getinfo(key+'.npy').file_size==offset+4*math.prod(header['shape']),
                    'CPU raw header differs: '+name)
    return {'root':root,'build_dir':build,'work_root':work,'protocol':protocol,'spec':view,'build_result':result,
            'build_request':request,'engine_path':engine,'audit':audit,'input_path':input_path,'cpu_path':cpu_path,
            'strict_integer_tactics_verified':strict,'strict_integer_gaps':gaps,'helper':helper,
            'lineage':{'protocol_sha256':args.protocol_sha256,'precision_audit_sha256':args.audit_sha256,'engine_sha256':args.engine_sha256,
                       'member_spec_sha256':sha(root/'spec.json'),'member_manifest_sha256':sha(root/'manifest.json'),
                       'source_graph_sha256':spec['graph_sha256'],'parent_spec_sha256':sha(root/'parent_bundle/spec.json'),
                       'build_result_sha256':sha(result_path),'build_request_sha256':sha(request_path),
                       'engine_layers_sha256':sha(build/'engine_layers.json'),'input_sha256':sha(input_path),
                       'cpu_reference_sha256':sha(cpu_path),'execution_helper_sha256':EXECUTION_HELPER_SHA}}

def interrupt(_sig,_frame): raise KeyboardInterrupt('Matched profiled runner interrupted')

def start_guard(args,verified):
    work=verified['work_root'];protocol=verified['protocol']
    require(os.getpgrp()==os.getpid() and os.getsid(0)==os.getpid(),
            'Profile wrapper must create private PID=PGID=SID session, e.g. setsid --wait, before adapter')
    module=load_exact(work/'jetson_owned_group_guard.py',OWNED_GUARD_SHA)
    expected_boot=protocol['expected_boot_id'];require(module.boot_id()==expected_boot,'Boot differs')
    parent_pid=os.getppid();libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(1,int(signal.SIGTERM),0,0,0)==0,'PR_SET_PDEATHSIG failed')
    require(os.getppid()==parent_pid and parent_pid>1,'Parent disappeared during parent-death setup')
    status_path=work/'monitor/status.json';status,stamp=module.read_status(status_path)
    require(status['root']==str(work) and status['boot_id']==expected_boot and status['max_seconds']==300 and
            status['status']=='RUNNING' and status['build_script']==str(work/protocol['monitor_build_script']),
            'Original monitor identity/budget differs')
    monitor=module.process_info(status['monitor_pid'])
    require(monitor is not None and str(work/'jetson_monitored_build.py') in monitor['argv'] and str(work) in monitor['argv'],
            'Original live monitor process differs')
    me=module.process_info(os.getpid());require(me is not None and str(Path(__file__).resolve()) in me['argv'],'Own actual argv differs')
    identity={'schema':'rampq.gate05.profile_runner_identity/v1','runner':me,'parent_pid_before_prctl':parent_pid,
              'PR_SET_PDEATHSIG':'SIGTERM','boot_id':expected_boot,'adapter_sha256':sha(Path(__file__)),
              'adapter_path':str(Path(__file__).resolve()),'protocol_sha256':args.protocol_sha256,'work_root':str(work),
              'monitor_identity':{k:status[k] for k in ('root','monitor_pid','build_pid','max_seconds','build_script','boot_id')},
              'monitor_process_identity':monitor,'monitor_status_stamp':stamp,'written_before_GPU_import':True,
              'engine_executions_allowed':2,'new_CPU_model_executions':0}
    identity_path=work/'profile_runner_identity.json';write_new(identity_path,identity)
    console=(work/'profile_child_guard_console.log').open('xb',buffering=0)
    guard=subprocess.Popen([sys.executable,'-u',str(work/'jetson_matched_profile_child_guard.py'),
            '--work-root',str(work),'--protocol',str(args.protocol.resolve()),'--protocol-sha256',args.protocol_sha256,
            '--identity',str(identity_path),'--identity-sha256',sha(identity_path)],cwd=str(work),
            stdin=subprocess.DEVNULL,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
    console.close()
    ack=work/'profile_child_guard.json';deadline=time.monotonic()+10
    while not ack.exists() and guard.poll() is None and time.monotonic()<deadline:time.sleep(.1)
    require(ack.exists() and guard.poll() is None,'Child guard did not acknowledge before GPU import')
    observed=read(ack)
    require(observed['status']=='WATCHING' and observed['guard_pid']==guard.pid and
            observed['runner_identity_sha256']==sha(identity_path) and observed['owned_pgid']==me['pid'] and
            observed['leader_start_ticks']==me['start_ticks'] and observed['protocol_sha256']==args.protocol_sha256,
            'Child guard acknowledgement differs')
    live_guard=module.process_info(guard.pid)
    require(live_guard is not None and live_guard['pid']==live_guard['pgid']==live_guard['sid'] and
            str(work/'jetson_matched_profile_child_guard.py') in live_guard['argv'],'Guard lacks independent authenticated session')
    return identity,guard

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','build','work-root','audit','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('engine-sha256','audit-sha256','protocol-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--diagnostic-formats-only',action='store_true');p.add_argument('--check-input-only',action='store_true')
    args=p.parse_args();verified=validate_inputs(args);work=verified['work_root'];output=args.output.resolve()
    require(output.parent==work and output.name=='fixed_outputs' and not output.exists(),'Fresh fixed_outputs directly inside work root required')
    if args.check_input_only:
        print(json.dumps({'status':'MATCHED_FIXED_INPUTS_VERIFIED_NO_EXECUTION','native_runtime_imported':False,
             'native_inference_executions':0,'new_CPU_model_executions':0,'executions_planned':2,
             'configuration':verified['spec']['configuration'],'strict_integer_gap_count':len(verified['strict_integer_gaps']),
             'lineage':verified['lineage'],'child_guard_handshake_performed':False},indent=2));return
    require(platform.system()=='Linux' and platform.machine()=='aarch64','Real execution requires Orin Linux')
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    marker=verified['root']/('matched_fixed_input_attempt_'+args.protocol_sha256+'.json')
    write_new(marker,{'protocol_sha256':args.protocol_sha256,'engine_sha256':args.engine_sha256,'pid':os.getpid(),
                     'maximum_engine_executions':2,'started_UTC':datetime.now(timezone.utc).isoformat()})
    output.mkdir(exist_ok=False)
    state={'schema':'rampq.gate05.matched_fixed_adapter/v1','runner_sha256':sha(Path(__file__)),
           'adapter_sha256':sha(Path(__file__)),'execution_helper_sha256':EXECUTION_HELPER_SHA,
           'lineage':verified['lineage'],'phase':'BEFORE_GUARD_ACK','executions_attempted':0,'executions_completed':0,
           'native_inference_executions':0,'dataset_evaluation_run':False,'gate05_status':'INCOMPLETE',
           'original_complete_model_numerical_guard':'FAILED_UNCHANGED'}
    helper=verified['helper'];helper.validate_inputs=validate_inputs
    try:
        identity,guard=start_guard(args,verified)
        state.update(child_guard_pid=guard.pid,profile_runner_identity_sha256=sha(work/'profile_runner_identity.json'),
                     child_guard_ack_before_GPU_import=True,phase='GUARD_ACKNOWLEDGED')
        def guard_exited(_sig,_frame):
            if guard.poll() is not None:
                raise KeyboardInterrupt('Authenticated child guard exited before profiled runner')
        signal.signal(signal.SIGCHLD,guard_exited)
        require(guard.poll() is None,'Child guard exited after acknowledgement')
        helper.write(output/'execution_state.json',state)
        # Only this unchanged helper function imports GPU modules and executes.
        helper.execute(args,verified,output,state)
    except BaseException as error:
        state.update(status='MATCHED_FIXED_INPUT_FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        helper.write(output/'smoke_failure.json',state);helper.write(output/'execution_state.json',state)
        raise
    finally:
        files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'}
        helper.write(output/'manifest.json',{'schema':'rampq.gate05.matched_fixed_output_files/v1','files':files})
        write_new(work/'profile_runner_completion.json',{'schema':'rampq.gate05.profile_runner_completion/v1','pid':os.getpid(),
              'protocol_sha256':args.protocol_sha256,'status':state.get('status'),'executions_attempted':state['executions_attempted'],
              'executions_completed':state['executions_completed'],'adapter_finalization_reached':True,
              'all_attempted_calls_synchronized':state['executions_completed']==state['executions_attempted'],
              'two_engine_calls_completed':state['executions_completed']==2,'original_guard':'FAILED_UNCHANGED'})
    # Do not wait for the guard here: it must observe our process/group exit.
    # The profiling wrapper waits for childguard COMPLETE before collection.
    print('MATCHED_FIXED_TWO_CALLS_SAVED_NO_GATE_PASS',flush=True)

if __name__=='__main__':main()
