"""Saved-only new-run NVTX/CUDA/process/maps/authenticated SM87 SASS audit.

Does not execute a model, CUDA tool, disassembler, or SSH command. Old tactic
audits remain unchanged. Exact runtime identity plus authenticated static MAC
body is a distinct evidence branch, not a hardware instruction trace.
"""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import zipfile

ROOT=Path(__file__).resolve().parents[3]
PLAN=ROOT/'artifacts/gate05/20261021_gate05_completion_plan'
# Each record authenticates a distinct already-collected engine/run. No trace reuse.
CASES={
 'r_only_G1':{'engine':'7a5a04ec5243e52ec20a1f34802bce3a785cdcaa71b2569e0a47c4b91b57c357','archive':'e796f88c274cbefbc0cd37cf6050c118db729a6f31efb02b87bdb8145cc3bcf0','receipt':'bdcd55371809fa91a886fc930076036bcae937b074abe5ff8aa3a00c13e7f201','protocol':'919f1e5d5b6c782ae1053e214179a54266a7633287239e59df8717b08aea1884','int8':82,'named':75},
 'r_only_G1_G5':{'engine':'2f240d1debc2f875c9aa5998ebca9c59b3d89e357939360123bc57b4382de811','archive':'d782bcf32122e8928491b18dffde815665d541f884953c82aa3896ca7d4f212e','receipt':'1544d6a412d4d4d7202ea4a4b1f1a316ad8f9e4f7d675a29943e506873b027ac','protocol':'d53d36490fdf44742106f0102ef0ab72ba0f8e31452046aeafe6033bb5f3743f','int8':74,'named':67},
 'r_only_G1_G2':{'engine':'5d09f7a132abad2b1108a82c3410bad660a77a4c95ea2e7e96a21eabde65846b','archive':'5dc2b1e98dece56843d7bb040027aebdc77c105d624655ce453c263b1ce87247','receipt':'9666d80533638a4112a5815ef4079370e4b35d0d9888d4d5a667ec12795c94bc','protocol':'7c5d4ba2d4e61c35a1aef5bb08e9a094bae06cc0578e7d377f9218be496c0af8','int8':77,'named':70},
}
BASE_READER_SHA='7815a395a679924f31d15069d04c446226af189c1e5b22a05663cb73021965fe' 
LIBRARY='4536788450d9a11f98701aa05021116ef13efd99f9318a03c01a2bc014d1c6ca'
SASS_SHA='c3c8715e70a5150b5ebb9a7334dcef999b357c7a5b86841ff8495870b5e5f6ea'
CAPTURE='RAMPQ_MATCHED_QDQ_FIXED_IMAGE'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))

def saved_hashes(root):
    return {str(p.relative_to(root)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()}

def parse_sass(path):
    lines=path.read_text(encoding='utf-8').splitlines()
    starts=[i for i,line in enumerate(lines) if re.search(r'Function\s*:',line)]
    out={}
    for n,begin in enumerate(starts):
        end=starts[n+1] if n+1<len(starts) else len(lines)
        symbol=lines[begin].split(':',1)[1].strip()
        arch=[line.strip() for line in lines[:begin] if re.search(r'code for sm_\d+',line)][-1]
        instructions=[]
        for i in range(begin,end):
            m=re.search(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;',lines[i])
            if m:instructions.append({'line':i+1,'pc':'0x'+m.group(1),'text':m.group(2).strip()})
        idp=[r for r in instructions if re.search(r'\bIDP\.4A\.S8\.S8\b',r['text'])]
        chains=[];converts=[];defs={}
        # A local textual chain is illustrative; it is not a full CFG proof.
        for r in instructions:
            m=re.search(r'IDP\.4A\.S8\.S8\s+(R\d+),\s*([^,]+),\s*([^,]+),\s*([^;]+)',r['text'])
            if m:
                dest,a,b,acc=[s.strip().replace('.reuse','') for s in m.groups()]
                if acc in defs:chains.append({'producer':defs[acc],'consumer':r,'accumulator_register':acc})
                defs[dest]=r
            else:
                m=re.search(r'I2FP\.F32\.S32\s+(R\d+),\s*(R\d+)',r['text'])
                if m and m.group(2) in defs:converts.append({'integer_dot_producer':defs[m.group(2)],'signed32_to_float':r})
                # Clear a simple destination overwritten by an unrelated op;
                # branch/packed-wide definitions remain a stated limitation.
                dest=re.search(r'(?:^|\s)(?:[A-Z][\w.]*)(?:\s+)(R\d+),',r['text'])
                if dest:defs.pop(dest.group(1),None)
        out[symbol]={'symbol':symbol,'architecture_marker':arch,'function_start_line':begin+1,'signed8_IDP4A_count':len(idp),'IDP_examples':idp[:2]+idp[-2:],'local_accumulation_examples':chains[:3],'signed32_to_float_examples':converts[:3],'floating_FFMA_present':any(re.search(r'\bFFMA',r['text']) for r in instructions),'loads_present':any(re.search(r'\bLDG|\bLDS',r['text']) for r in instructions),'full_CFG_or_hardware_instruction_trace':False,'body_sha256':hashlib.sha256('\n'.join(lines[begin:end]).encode()).hexdigest()}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configuration',choices=sorted(CASES),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();case=CASES[a.configuration]
    parent=ROOT/'artifacts/gate05/20261021_matched_fixed_native'/a.configuration
    raw=(parent/'native_evidence').resolve();output=a.output.resolve()
    a.archive=parent/'native_evidence.zip';a.archive_sha256=case['archive'];a.receipt_sha256=case['receipt']
    ENGINE=case['engine']
    execution=read(raw/'execution_spec.json')
    build_name=execution['build_root'].split('/')[-1]
    BUILDROOT=ROOT/'artifacts/gate05/20261021_matched_control_family_native'/a.configuration/'native_evidence'
    BUILD=BUILDROOT/'results'/build_name
    assert not output.exists() and raw not in output.parents and output!=raw
    before=saved_hashes(raw);checks=[]
    def c(name,passed):checks.append({'name':name,'passed':bool(passed)})
    c('exact_new_archive',sha(a.archive)==a.archive_sha256)
    with zipfile.ZipFile(a.archive) as z:
        c('new_archive_CRC',z.testzip() is None)
        infos=[r for r in z.infolist() if not r.is_dir()]
        c('exact_archive_inventory_and_no_duplicates',len(infos)==len(before) and {r.filename for r in infos}==set(before))
        for info in infos:c('archive_member:'+info.filename,{'bytes':info.file_size,'sha256':hashlib.sha256(z.read(info.filename)).hexdigest()}==before.get(info.filename))
    c('exact_new_receipt',sha(raw/'remote_receipt.json')==a.receipt_sha256)
    receipt=read(raw/'remote_receipt.json')['files']
    c('exact_raw_inventory',len(receipt)==51 and set(before)==set(receipt)|{'remote_receipt.json'})
    for rel,row in receipt.items():c('receipt:'+rel,before[rel]==row)
    smoke_dir=raw/'fixed_outputs'
    # Paths are the frozen wrapper's work-root outputs, not the historical run.
    smoke=read(smoke_dir/'smoke_result.json');environment=read(smoke_dir/'environment.json')
    c('new_smoke_engine_identity',smoke['lineage']['engine_sha256']==ENGINE and sha(BUILD/'model.engine')==ENGINE)
    c('two_actual_calls',smoke['native_inference_executions']==smoke['executions_completed']==2 and smoke['executions_attempted']==2)
    c('same_new_runner_process_and_SM87',environment['CUDA_capability']==[8,7] and environment['device_index']==0 and environment['pid']>0)
    c('same_boot',environment['boot_id']=='692c7544-b8bd-415b-939c-b11cdbd9f7f0')
    c('protected_scope',smoke['gate05_status']=='INCOMPLETE' and smoke['original_complete_model_numerical_guard']=='FAILED_UNCHANGED' and smoke['strict_integer_gap_count']==7 and smoke['strict_integer_tactics_verified'] is False)
    layers=read(BUILD/'engine_layers.json')['Layers'];runtime_layers=read(smoke_dir/'runtime_engine_layers.json')['Layers']
    c('runtime_inspector_identical_frozen_engine',runtime_layers==layers)
    audit=read(raw/'precision_audit.json')
    gaps=[r for r in audit['mapping'] if r['precision']=='INT8' and not r['strict_int8_tactic_verified']]
    targets={r['engine_layer'] if 'engine_layer' in r else r['native_layer_name']:r for r in gaps}
    # The outer audit currently uses engine_layers, so select by full source
    # membership if no single engine_layer field is exposed in a mapping row.
    if len(targets)!=7 or any(name not in {r['Name'] for r in layers} for name in targets):
        targets={r['Name']:[g for g in gaps if g['onnx_node'] in r['Name']] for r in layers if r.get('LayerType')=='CudaDepthwiseConvolution'}
    c('exact_seven_current_depthwise_targets',len(targets)==len(gaps)==7)
    target_rows=[r for r in layers if r['Name'] in targets]
    c('all_seven_INT8_input_weights_no_tactic',all(r['Weights']['Type'].lower()=='int8' and any('INT8' in v['Format/Datatype'].upper() for v in r['Inputs']) and not r.get('TacticName') for r in target_rows))
    protocol=read(raw/'protocol.json')
    c('original_reader_exact',sha(ROOT/'artifacts/gate05/20261021_qall_none_fixed_runtime/audit_saved_runtime_sass.py')==BASE_READER_SHA)
    c('own_frozen_protocol_exact',sha(raw/'protocol.json')==case['protocol'] and protocol['configuration']==a.configuration and protocol['engine_sha256']==ENGINE)
    c('own_audit_identity',sha(raw/'precision_audit.json')==protocol['precision_audit_sha256']==smoke['lineage']['precision_audit_sha256'] and audit['engine_sha256']==ENGINE)
    c('own_build_result_identity',sha(BUILD/'build_result.json')==protocol['build_result_sha256']==smoke['lineage']['build_result_sha256'])
    c('own_build_request_identity',sha(BUILD/'build_request.json')==protocol['build_request_sha256']==smoke['lineage']['build_request_sha256'])
    c('own_inspector_byte_identity',sha(BUILD/'engine_layers.json')==sha(smoke_dir/'runtime_engine_layers.json')==protocol['engine_layers_sha256']==smoke['lineage']['engine_layers_sha256'])
    c('own_source_and_member_identity',all(smoke['lineage'][k]==protocol[k] for k in ('member_spec_sha256','member_manifest_sha256','source_graph_sha256','parent_spec_sha256')))
    c('own_actual_format_audit',audit['requested_conv_formats_complete'] and audit['errors']==[] and audit['unmapped_engine_convs']==[] and audit['quantized_conv_count']==audit['verified_int8_format_convs']==case['int8'] and audit['verified_fp16_format_convs']==88-case['int8'] and audit['named_integer_tactic_source_convs']==case['named'] and audit['source_convs_without_tactic_name']==7)
    c('gap_source_unique_actual_layer',all(sum(g['onnx_node'] in r['Name'] for r in target_rows)==1 for g in gaps) and all(sum(g['onnx_node'] in r['Name'] for g in gaps)==1 for r in target_rows))
    c('each_gap_weighted_actual_formats',all(g['requested_formats_verified'] and g['weights_type'].lower()=='int8' and g['engine_layer_type']=='CudaDepthwiseConvolution' for g in gaps))
    c('own_frozen_tools',all(sha(raw/f)==protocol[k] for f,k in [('jetson_matched_fixed_input_adapter.py','adapter_sha256'),('jetson_matched_profile_child_guard.py','child_guard_sha256'),('jetson_gate05_control_fixed_input.py','execution_helper_sha256'),('jetson_build_matched_control.py','build_contract_sha256'),('run_fixed.py','profile_wrapper_sha256')]))
    ident=read(raw/'profile_runner_identity.json');runner=ident['runner'];completion=read(raw/'profile_runner_completion.json')
    child=read(raw/'profile_child_guard.json');outer=read(raw/'supplemental_group_guard.json');terminal=read(raw/'terminal_before_collection.json');monitor=read(raw/'monitor/status.json')
    c('runner_identity_exact_before_GPU',sha(raw/'profile_runner_identity.json')==smoke['profile_runner_identity_sha256']==child['runner_identity_sha256'] and ident['written_before_GPU_import'] and smoke['child_guard_ack_before_GPU_import'] and ident['adapter_sha256']==protocol['adapter_sha256'])
    c('runner_PID_PGID_SID_own_boot',runner['pid']==runner['pgid']==runner['sid']==environment['pid']==environment['pgid'] and ident['boot_id']==environment['boot_id']==child['boot_id']==outer['boot_id']==terminal['boot_id'])
    c('runner_parent_death_and_budget',ident['PR_SET_PDEATHSIG']=='SIGTERM' and ident['parent_pid_before_prctl']==runner['ppid'] and ident['engine_executions_allowed']==2 and ident['new_CPU_model_executions']==0)
    c('child_guard_exact_runner_identity',child['owned_pgid']==runner['pgid'] and child['leader_start_ticks']==runner['start_ticks'] and child['adapter_sha256']==protocol['adapter_sha256'] and child['guard_sha256']==protocol['child_guard_sha256'] and child['protocol_sha256']==case['protocol'] and child['guard_pid']==smoke['child_guard_pid'])
    c('both_groups_truly_empty_no_intervention',all(g['status']=='COMPLETE' and g['intervention_performed'] is False and g['final_group_members']==[] and g['owned_group_fully_empty'] is True for g in (child,outer)) and terminal['wrapper_members']==terminal['escaped_runner_members']==[] and terminal['protective_intervention'] is False)
    c('five_saved_terminal_PID_observations',len(terminal['five_processes'])==5 and all(r['process'] is None for r in terminal['five_processes'].values()) and terminal['five_processes']['profile_runner']['pid']==runner['pid'] and terminal['five_processes']['profile_child_guard']['pid']==child['guard_pid'])
    c('bounded_monitor_success',monitor['status']=='BUILD_SCRIPT_OK' and monitor['build_returncode']==0 and monitor['max_seconds']==300 and monitor['elapsed_seconds']<300 and monitor['monitor_pid']==ident['monitor_identity']['monitor_pid'])
    c('own_two_synchronized_calls_and_no_dataset',completion['pid']==runner['pid'] and completion['all_attempted_calls_synchronized'] and completion['executions_attempted']==completion['executions_completed']==2 and completion['two_engine_calls_completed'] and terminal['completed_engine_calls']==2 and terminal['new_CPU_model_calls']==terminal['dataset_calls']==0)
    c('profile_and_CPU_export_only_succeeded',read(raw/'profile_exit.json')['returncode']==0 and read(raw/'profile_export.json')['returncode']==0)
    maps_proof=[]
    for label in ('before_execution','after_execution'):
        maps_text=(smoke_dir/('maps_'+label+'.txt')).read_text(encoding='utf-8')
        identities=read(smoke_dir/('loaded_libraries_'+label+'.json'))['libraries']
        candidates=[r for r in identities if r.get('real_path','').endswith('/libnvinfer.so.8.5.2') and r.get('sha256')==LIBRARY]
        c('own_loaded_nvinfer_exact:'+label,len(candidates)==1 and candidates[0].get('hash_available') is True)
        if not candidates:continue
        lib=candidates[0]
        regions=[]
        for line in maps_text.splitlines():
            f=line.split(None,5)
            if len(f)==6 and f[5]==lib['mapped_path']:regions.append({'address':f[0],'permissions':f[1],'offset':f[2],'device':f[3],'inode':f[4]})
        c('own_maps_regions_exact:'+label,regions==lib['mapping_regions'] and bool(regions))
        c('own_maps_device_inode_exact:'+label,all((int(r['device'].split(':')[0],16),int(r['device'].split(':')[1],16),int(r['inode']))==(lib['device_major'],lib['device_minor'],lib['inode']) for r in regions))
        maps_proof.append({'label':label,'mapped_library':lib})
    c('same_library_before_after',len(maps_proof)==2 and maps_proof[0]['mapped_library']['sha256']==maps_proof[1]['mapped_library']['sha256']==LIBRARY)
    static=read(PLAN/'sass_static_independent_review.json')
    c('static_review_exact',sha(PLAN/'sass_static_independent_review.json')=='40eb7919d25d795aec84df57f6f8c9cb7d54061672500ebd81b32fdb45dd850c')
    sass_path=PLAN/'sass_read_evidence/library0.sass.txt'
    c('exact_static_library_and_SASS',sha(sass_path)==SASS_SHA and static['libraries'][0]['library_sha256']==LIBRARY)
    functions=parse_sass(sass_path)
    for name,f in functions.items():c('SM87_signed_MAC_body:'+name,f['architecture_marker']=='code for sm_87' and f['signed8_IDP4A_count']>0 and bool(f['local_accumulation_examples']) and bool(f['signed32_to_float_examples']) and f['loads_present'])
    database=raw/'profile/fixed_input.sqlite'
    con=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    con.execute('PRAGMA query_only=ON')
    try:
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        c('new_trace_tables',{'StringIds','NVTX_EVENTS','CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_KERNEL','PROCESSES'}<=tables)
        strings=dict(con.execute('SELECT id,value FROM StringIds'))
        ranges=[]
        for row in con.execute('SELECT rowid AS sqlite_rowid,* FROM NVTX_EVENTS'):
            r=dict(row);r['resolved_text']=r.get('text') if r.get('text') is not None else strings.get(r.get('textId'))
            r['resolved_json']=r.get('jsonText') if r.get('jsonText') is not None else strings.get(r.get('jsonTextId'));ranges.append(r)
        captures=[r for r in ranges if r['resolved_text']==CAPTURE and r['end'] is not None]
        c('one_new_capture',len(captures)==1)
        capture=captures[0]
        runtime=[dict(r) for r in con.execute('SELECT rowid AS sqlite_rowid,* FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
        kernels=[dict(r) for r in con.execute('SELECT rowid AS sqlite_rowid,* FROM CUPTI_ACTIVITY_KIND_KERNEL')]
        processes=[dict(r) for r in con.execute('SELECT * FROM PROCESSES')]
        by_api=defaultdict(list);by_kernel=defaultdict(list)
        for r in runtime:by_api[r['correlationId']].append(r)
        for r in kernels:by_kernel[r['correlationId']].append(r)
        native_names={r['Name'] for r in layers}
        all_layers=[r for r in ranges if r['resolved_text'] in native_names and r['end'] is not None]
        results=[]
        for target in target_rows:
            selected=[r for r in all_layers if r['resolved_text']==target['Name'] and r['globalTid']==capture['globalTid'] and capture['start']<=r['start'] and r['end']<=capture['end']]
            c('two_exact_layer_NVTX:'+target['Name'],len(selected)==2)
            evidence=[]
            for repeat,event in enumerate(sorted(selected,key=lambda r:r['start'])):
                prefix=target['Name']+':'+str(repeat)
                c('registered_JSON_identity:'+prefix,bool(event['resolved_json']) and json.loads(event['resolved_json'])==target)
                c('NVTX_thread_identity:'+prefix,event.get('endGlobalTid') in (None,event['globalTid']))
                launch_chains=[]
                for api in runtime:
                    if api['globalTid']!=event['globalTid'] or not(event['start']<=api['start'] and api['end']<=event['end']):continue
                    linked=by_kernel[api['correlationId']]
                    if not linked:continue
                    containing=[r for r in all_layers if r['globalTid']==api['globalTid'] and r['start']<=api['start'] and api['end']<=r['end']]
                    c('unique_successful_API:'+prefix+':'+str(api['correlationId']),len(by_api[api['correlationId']])==1 and api['returnValue']==0 and len(containing)==1 and containing[0]['sqlite_rowid']==event['sqlite_rowid'])
                    ks=[]
                    for kernel in linked:
                        k=dict(kernel);k['mangled_name_text']=strings.get(k['mangledName']);k['demangled_name_text']=strings.get(k['demangledName']);k['short_name_text']=strings.get(k['shortName'])
                        prows=[r for r in processes if r['globalPid']==k['globalPid']]
                        c('exact_runner_process:'+prefix+':'+str(api['correlationId']),len(prows)==1 and prows[0]['pid']==environment['pid'] and k['deviceId']==environment['device_index'])
                        f=functions.get(k['mangled_name_text'])
                        c('exact_mangled_authenticated_SM87_body:'+prefix+':'+str(api['correlationId']),f is not None)
                        ks.append({'kernel':k,'process_rows':prows,'authenticated_static_MAC_body':f})
                    launch_chains.append({'API':dict(api,resolved_name=strings.get(api['nameId'])),'kernels':ks})
                c('correlated_kernel_exists:'+prefix,bool(launch_chains))
                evidence.append({'repeat':repeat,'NVTX':event,'launch_chains':launch_chains})
            results.append({'engine_layer':target,'source_gap_rows':[g for g in gaps if g['onnx_node'] in target['Name']],'two_execution_evidence':evidence,'TacticName_filled':False})
    finally:con.close()
    after=saved_hashes(raw);c('all_new_saved_evidence_unchanged',before==after)
    passed=all(r['passed'] for r in checks)
    report={'schema':'rampq.gate05.matched_own_runtime_authenticated_SASS_independent_review/v1','client_at':datetime.datetime.now().astimezone().isoformat(),'status':'NEW_RUNTIME_IDENTITIES_AND_AUTHENTICATED_STATIC_SIGNED8_MAC_BRANCH_VERIFIED' if passed else 'RUNTIME_SASS_EVIDENCE_INCOMPLETE','check_count':len(checks),'checks':checks,'gaps':[r['name'] for r in checks if not r['passed']],'engine_sha256':ENGINE,'archive_sha256':a.archive_sha256,'receipt_sha256':a.receipt_sha256,'sqlite_sha256':sha(database),'own_runner_environment':environment,'own_loaded_module_identity':maps_proof,'capture':capture,'target_depthwise_layers':len(target_rows),'layer_evidence':results,'authenticated_static_functions':functions,'evidence_branch':'RUNTIME_AUTHENTICATED_KERNEL_PLUS_STATIC_SIGNED8_MAC_EVIDENCE','branch_satisfied':passed,'old_named_tactic_status':'INTEGER_TACTIC_REVIEW_REQUIRED','configuration':a.configuration,'source_graph_sha256':protocol['source_graph_sha256'],'protocol_sha256':case['protocol'],'old_named_integer_tactic_count':case['named'],'old_missing_tactic_names':7,'old_tactic_audit_changed':False,'no_missing_TacticName_invented':True,'hardware_executed_instruction_trace_claim':False,'full_CFG_and_parameter_abi_operand_proof_claim':False,'TensorCore_claim':False,'interpretation':'Same new engine source/actual INT8 weighted operation -> exact same-thread NVTX/registered inspector JSON -> unique successful CUDA launch correlation -> own process exact kernel symbol -> own SM87 device and loaded authenticated libnvinfer binary -> exact SM87 signed8 dot/accumulator SASS body. This is the separately declared runtime-authenticated static arithmetic branch, stronger than names alone; not a per-instruction hardware trace or old named-tactic PASS. FP32 conversion/epilogue coexists legitimately.','new_CPU_graph_native_SSH_or_disassembly_runs':0,'saved_evidence_unchanged':before==after,'reviewer_script_sha256':sha(Path(__file__))}
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'checks':len(checks),'gaps':report['gaps'],'sha256':sha(output),'branch_satisfied':passed}))

if __name__=='__main__':main()
