"""Independent stdlib-only source and protocol compatibility review; never prepare."""
import ast
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
source=ROOT/'scripts/prepare_gate05_matched_control_family.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
checks=[]
def c(name,value):checks.append({'name':name,'passed':bool(value)})
text=source.read_text(encoding='utf-8')
c('prepare_source_exact',sha(source)=='819d8a89baefe7f32f73e77a581f8dfeaa721ce0b82739ee29e3c865c561cf5a')
ast.parse(text)
spec=importlib.util.spec_from_file_location('reviewed_prepare_not_executed',source)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
c('no_model_runtime_imports_on_source_import',not any(name in sys.modules for name in ('numpy','onnx','onnxruntime','torch','tensorrt')))
for cfg in mod.CONFIGURATIONS:
    inventory,protocol=mod.validate_protocol(mod.PROTOCOL,cfg)
    row=inventory['configurations'][cfg]
    paths=mod.check_parent(ROOT/row['parent_directory'],row)
    c('full_parent_inventory:'+cfg,len(paths)==11)
    for p in paths:c('source_file_present:'+cfg+'/'+p.name,p.is_file())
c('total_fixed_CPU_runs_nine',sum(protocol['CPU_run_budget_by_configuration'].values())==9)
c('five_new_members_plus_existing_qall',tuple(mod.CONFIGURATIONS)==('fp16_control','r_only_G1','r_only_G1_G5','r_only_G1_G2','q_only_G1'))
c('FP16_one_and_four_rewrites_two_each',protocol['CPU_run_budget_by_configuration']==dict(zip(mod.CONFIGURATIONS,(1,2,2,2,2))))
c('same_three_named_binding_contracts',all(row['expected_binding_contract']==inventory['configurations']['fp16_control']['expected_binding_contract'] for row in inventory['configurations'].values()))
c('newnative_not_authorized_in_preparation',"'native_build_authorized':False,'maximum_native_builds':0,'max_build_seconds':None" in text)
c('exact_unapproved_node_and_initializer_bytes_checked',"node.SerializeToString() == expected_nodes.get" in text and "value.SerializeToString() == expected_constants.get" in text)
c('shell_IO_bytes_checked',"old_shell.SerializeToString() == new_shell.SerializeToString()" in text)
c('zero_mask_copies_model_file_bytes',"shutil.copy2(parent/'model.onnx',output/'model.onnx')" in text)
c('fixed_CPU_only_provider',"providers=['CPUExecutionProvider']" in text and "session.get_providers()==['CPUExecutionProvider']" in text)
c('CPU_disabled_optimization_four_threads',"ort.GraphOptimizationLevel.ORT_DISABLE_ALL" in text and 'intra_op_num_threads=4' in text)
c('saved_raw_byte_identity_both_outputs',"old[index].tobytes(order='C')==array.tobytes(order='C')" in text and "['main_output','pe_output']" in text)
c('source_before_after_unchanged',"require(before==after,'Historical sources changed during member preparation')" in text)
c('fresh_destination_no_retry',"not output.exists() and not output.with_suffix('.zip').exists()" in text and "'no_retry_if_interrupted':True" in text)
c('CPU_attempt_marker_precedes_calls',text.index("write(output/'cpu_attempt_started.json'")<text.index("old=execute_cpu_once("))
c('CPU_failure_blocks_seal',"require(passed,'Member CPU equivalence failed; preserve raw outputs, do not retry/seal/start native')" in text)
c('exception_preserves_failure',"'status':'FAILED_PRESERVE_NO_RETRY_RESELECTION_SEAL_OR_NATIVE'" in text)
c('seal_requires_separate_native_protocol',"A separately frozen native-family protocol is required" in text and "native_protocol['family_preparation_protocol_sha256']==PROTOCOL_SHA" in text)
c('all_kernel_build_native_exec_absent',not any(isinstance(n,ast.Attribute) and n.attr in ('build_serialized_network','execute_v2','execute_async_v2') for n in ast.walk(ast.parse(text))))
c('only_CPU_model_calls_two_sites',sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='execute_cpu_once' for n in ast.walk(ast.parse(text)))==2)
c('prepare_nine_budget_is_orchestrated_not_automatic_loop',"if args.prepare_only:" in text and "prepare(output,args.configuration,args.protocol.resolve())" in text)
for key,item in inventory['shared_sources'].items():c('shared_source:'+key,mod.info(ROOT/item['file'])=={k:item[k] for k in ('bytes','sha256')})
report={'schema':'rampq.gate05.matched_family_prepare_source_review/v1','client_at':datetime.datetime.now().astimezone().isoformat(),'status':'READ_ONLY_PREPARE_SOURCE_PROTOCOL_COMPATIBILITY_VERIFIED' if all(x['passed'] for x in checks) else 'BLOCKED','check_count':len(checks),'checks':checks,'launch_blockers':[x['name'] for x in checks if not x['passed']],'prepare_source_sha256':sha(source),'protocol_sha256':sha(mod.PROTOCOL),'source_inventory_sha256':sha(mod.INVENTORY),'new_CPU_model_native_or_SSH_runs':0,'actual_rewrite_CPU_and_prepared_arrays_not_yet_reviewed':True,'source_scope':'Only parent/static/protocol and source execution-path review. Root alone may call one preparation per planned member; no run performed here.','native_scope':'No native sealing/build until separately frozen native-family protocol; restored precision desired only until actual audit.'}
out=HERE/'matched_family_prepare_source_review.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'checks':len(checks),'blockers':report['launch_blockers'],'sha256':sha(out)}))
