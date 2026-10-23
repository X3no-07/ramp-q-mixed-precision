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
BUILDROOT=ROOT/'artifacts/gate05/20261021_qall_none_native/native_evidence'
BUILD=BUILDROOT/'results/build_20261018T002834_795020Z'
ENGINE='c3d9e75d61452c69b97097d1320221d5d56d7ad0e2fc1370480cf3e09eb3a41e'
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
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--archive-sha256',required=True);p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.raw.resolve();output=a.output.resolve()
    assert not output.exists() and raw not in output.parents and output!=raw
    before=saved_hashes(raw);checks=[]
    def c(name,passed):checks.append({'name':name,'passed':bool(passed)})
    c('exact_new_archive',sha(a.archive)==a.archive_sha256)
    with zipfile.ZipFile(a.archive) as z:c('new_archive_CRC',z.testzip() is None)
    c('exact_new_receipt',sha(raw/'remote_receipt.json')==a.receipt_sha256)
    receipt=read(raw/'remote_receipt.json')['files']
    c('exact_raw_inventory',set(before)==set(receipt)|{'remote_receipt.json'})
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
    audit=read(ROOT/'artifacts/gate05/20261021_qall_none_native/precision_audit.json')
    gaps=[r for r in audit['mapping'] if not r['strict_int8_tactic_verified']]
    targets={r['engine_layer'] if 'engine_layer' in r else r['native_layer_name']:r for r in gaps}
    # The outer audit currently uses engine_layers, so select by full source
    # membership if no single engine_layer field is exposed in a mapping row.
    if len(targets)!=7 or any(name not in {r['Name'] for r in layers} for name in targets):
        targets={r['Name']:[g for g in gaps if g['onnx_node'] in r['Name']] for r in layers if r.get('LayerType')=='CudaDepthwiseConvolution'}
    c('exact_seven_current_depthwise_targets',len(targets)==len(gaps)==7)
    target_rows=[r for r in layers if r['Name'] in targets]
    c('all_seven_INT8_input_weights_no_tactic',all(r['Weights']['Type'].lower()=='int8' and any('INT8' in v['Format/Datatype'].upper() for v in r['Inputs']) and not r.get('TacticName') for r in target_rows))
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
    report={'schema':'rampq.gate05.new_runtime_authenticated_SASS_independent_review/v1','client_at':datetime.datetime.now().astimezone().isoformat(),'status':'NEW_RUNTIME_IDENTITIES_AND_AUTHENTICATED_STATIC_SIGNED8_MAC_BRANCH_VERIFIED' if passed else 'RUNTIME_SASS_EVIDENCE_INCOMPLETE','check_count':len(checks),'checks':checks,'gaps':[r['name'] for r in checks if not r['passed']],'engine_sha256':ENGINE,'archive_sha256':a.archive_sha256,'receipt_sha256':a.receipt_sha256,'sqlite_sha256':sha(database),'own_runner_environment':environment,'own_loaded_module_identity':maps_proof,'capture':capture,'target_depthwise_layers':len(target_rows),'layer_evidence':results,'authenticated_static_functions':functions,'evidence_branch':'RUNTIME_AUTHENTICATED_KERNEL_PLUS_STATIC_SIGNED8_MAC_EVIDENCE','branch_satisfied':passed,'old_named_tactic_status':'INTEGER_TACTIC_REVIEW_REQUIRED','old_named_integer_tactic_count':81,'old_missing_tactic_names':7,'old_tactic_audit_changed':False,'no_missing_TacticName_invented':True,'hardware_executed_instruction_trace_claim':False,'full_CFG_and_parameter_abi_operand_proof_claim':False,'TensorCore_claim':False,'interpretation':'Same new engine source/actual INT8 weighted operation -> exact same-thread NVTX/registered inspector JSON -> unique successful CUDA launch correlation -> own process exact kernel symbol -> own SM87 device and loaded authenticated libnvinfer binary -> exact SM87 signed8 dot/accumulator SASS body. This is the separately declared runtime-authenticated static arithmetic branch, stronger than names alone; not a per-instruction hardware trace or old named-tactic PASS. FP32 conversion/epilogue coexists legitimately.','new_CPU_graph_native_SSH_or_disassembly_runs':0,'saved_evidence_unchanged':before==after,'reviewer_script_sha256':sha(Path(__file__))}
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'checks':len(checks),'gaps':report['gaps'],'sha256':sha(output),'branch_satisfied':passed}))

if __name__=='__main__':main()
