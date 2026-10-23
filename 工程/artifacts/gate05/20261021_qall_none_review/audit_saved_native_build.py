"""Independent exact saved receipt, parser state and actual weighted format review."""
import hashlib
import json
from pathlib import Path
import math
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/gate05/20261021_qall_none_native'
RAW=OUT/'native_evidence'
BUILD=RAW/'results/build_20261018T002834_795020Z'
SOURCE=RAW/'source_bundle'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
checks=[]
def c(name,value):checks.append({'name':name,'passed':bool(value)})

c('native_zip_exact',sha(OUT/'native_evidence.zip')=='e78c4bd251e01e33508052538a285590af974d659fdf5052174a46774322e2c1' and (OUT/'native_evidence.zip').stat().st_size==15058198)
with zipfile.ZipFile(OUT/'native_evidence.zip') as z:c('zip_CRC',z.testzip() is None)
c('receipt_SHA',sha(RAW/'remote_receipt.json')=='a4c7903be19e02b3edbe2d2a4377792541eec187c108b459e1e09850b4a7a5f7')
receipt=read(RAW/'remote_receipt.json')['files']
c('receipt_84_files',len(receipt)==84)
for rel,row in receipt.items():
    p=RAW/rel;c('receipt:'+rel,p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'])
c('exact_extracted_inventory',{str(p.relative_to(RAW)).replace('\\','/') for p in RAW.rglob('*') if p.is_file()}==set(receipt)|{'remote_receipt.json'})
for folder in (RAW,SOURCE):
    manifest=read(folder/'manifest.json')
    for rel,row in manifest['files'].items():
        p=folder/rel;c('immutable_manifest:'+str(folder.relative_to(RAW))+':'+rel,p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'])
source_spec=read(SOURCE/'spec.json');control=read(RAW/'control_spec.json')
request=read(BUILD/'build_request.json');result=read(BUILD/'build_result.json')
expected_pins={'builder_file':'jetson_build_corrected_qall_none.py','builder_sha256':'4e5b9b71be93e72f5310273f477c0cbf7256fe2249f66c3a01f9ad3d96844aaa','helper_sha256':'d8431345da90482f0d5798e2d6524bf03591ecef653edfad5ef3e28a509eb19f','protocol_sha256':'572120502a6db021945c63cdd781d81e2a4b1771f5d9a13c3f58d0da7278eac1','control_spec_sha256':sha(RAW/'control_spec.json'),'outer_manifest_sha256':sha(RAW/'manifest.json'),'legacy_source_spec_sha256':sha(SOURCE/'spec.json'),'legacy_source_manifest_sha256':sha(SOURCE/'manifest.json')}
for key,value in expected_pins.items():c('builder_identity:'+key,result['tool_identity'][key]==request['tool_identity'][key]==value)
c('source_full_graph_exact',sha(SOURCE/'model.onnx')==source_spec['graph_sha256']==request['graph_sha256']==result['graph_sha256']=='755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee')
c('old_OBEY_new_NONE_explicit',source_spec['precisionConstraints']=='obey' and request['legacy_source_precision_constraints']=='obey' and request['runtime_precision_constraints']=='none')
c('build_only_success',result['status']=='BUILT_AND_DESERIALIZED_BUILD_ONLY' and result['native_inference_executions']==0 and result['engine_execution_context_created'] is False)
c('engine_exact',sha(BUILD/'model.engine')==result['engine_sha256']=='c3d9e75d61452c69b97097d1320221d5d56d7ad0e2fc1370480cf3e09eb3a41e' and (BUILD/'model.engine').stat().st_size==result['engine_bytes']==7123120)
c('no_numerical_or_gate_upgrade',result['original_complete_model_numerical_guard']=='FAILED_UNCHANGED' and result['gate05_status']=='INCOMPLETE' and not any(result[k] for k in ('numerical_pass_claim','accuracy_pass_claim','performance_pass_claim')))
for key,value in {'INT8':True,'FP16':True,'TF32':False,'OBEY_PRECISION_CONSTRAINTS':False,'PREFER_PRECISION_CONSTRAINTS':False}.items():
    c('actual_getflag:'+key,request['flags']['get_flag'][key]=={'supported':True,'enabled':value})
c('workspace_avg_and_no_setters',request['workspace_bytes_actual']==1024*1024*1024 and request['avg_timing_iterations_actual']==1 and request['explicit_layer_precision_constraint_count']==request['explicit_layer_output_type_constraint_count']==request['layer_constraint_reset_count']==0)
def binding_map(rows):return {r['name']:(r['input'],tuple(r['shape']),r['dtype']) for r in rows}
c('three_FLOAT_LINEAR_bindings',len(result['bindings'])==3 and binding_map(result['bindings'])==binding_map(source_spec['expected_binding_contract']) and all(r['dtype']=='DataType.FLOAT' and r['format']=='TensorFormat.LINEAR' for r in result['bindings']))
snapshots=[read(BUILD/name) for name in ('parsed_constraints_before.json','parsed_constraints_after.json','parsed_constraints_after_build.json')]
def constraints(rows):
    return [(r['index'],r['name'],r['precision_is_set'],r.get('precision'),[(o['index'],o['output_type_is_set'],o.get('requested_output_type')) for o in r['output_types']]) for r in rows]
c('all_parser_request_snapshots_unchanged',constraints(snapshots[0])==constraints(snapshots[1])==constraints(snapshots[2]))
precision=[r for r in snapshots[0] if r['precision_is_set']['value']]
output_requests=[(r,o) for r in snapshots[0] for o in r['output_types'] if o['output_type_is_set']['value']]
c('intrinsic_precision_177',len(precision)==177)
c('intrinsic_output_177',len(output_requests)==177)
c('176DQ_INT8_plus_shape_INT32',sum(r.get('precision')=='DataType.INT8' and r['layer_type']=='LayerType.DEQUANTIZE' for r in precision)==176 and sum(r.get('precision')=='DataType.INT32' and r['layer_type']=='LayerType.SHAPE' for r in precision)==1)
c('176Q_INT8_plus_shape_INT32',sum(o['requested_output_type']=='DataType.INT8' and r['layer_type']=='LayerType.QUANTIZE' for r,o in output_requests)==176 and sum(o['requested_output_type']=='DataType.INT32' and r['layer_type']=='LayerType.SHAPE' for r,o in output_requests)==1)
before=read(BUILD/'parsed_clip_attributes_before_correction.json');after=read(BUILD/'parsed_clip_attributes_after_correction.json')
bm={r['name']:r for r in before['layers']};am={r['name']:r for r in after['layers']}
c('88_exact_clip_names',before['count']==after['count']==len(bm)==len(am)==88 and set(bm)==set(am)==set(source_spec['clip_names']))
for name,row in am.items():c('clip_bounds:'+name,row['before']=={k:bm[name][k] for k in ('alpha','beta')} and row['after']=={'alpha':-127.0,'beta':127.0})
inspector=read(BUILD/'engine_layers.json')['Layers'];convs=read(BUILD/'conv_rows.json')
c('271_raw_layers_88_raw_Conv',len(inspector)==271 and len(convs)==88 and [r for r in inspector if r.get('ParameterType')=='Convolution']==convs)
mapping=read(BUILD/'conv_source_mapping.json')['mapping']
c('all_88_unique_source_row_mappings',len(mapping)==len({m['source_conv_name'] for m in mapping})==88 and all(m['status']=='UNIQUE_SOURCE_MAPPING' and len(m['actual_row_indices'])==1 for m in mapping) and {m['actual_row_indices'][0] for m in mapping}==set(range(88)))
requests={r['onnx_node']:r for r in source_spec['requests']}
sys.path.insert(0,str(ROOT/'scripts'))
import audit_gate05_int8_precision as base
c('old_metadata_classifier_unchanged',sha(ROOT/'scripts/audit_gate05_int8_precision.py')=='4ef240b00932ff350ec13576e783798d3d9ab0f78255e5ea3ffc8d69475ba06b')
c('old_base_auditor_unchanged',sha(ROOT/'scripts/audit_gate05_precision_candidate.py')=='40420902d6fb3fd5d6b6278d64bbaa4b44c839bdae69da73dbaeb3e15cf66f6a')
import onnx
graph=onnx.load(str(SOURCE/'model.onnx'))
initializers={v.name:v for v in graph.graph.initializer}
c('88_source_Conv_names',{n.name for n in graph.graph.node if n.op_type=='Conv'}==set(requests) and len(requests)==88)
observations=[]
for m in mapping:
    name=m['source_conv_name'];row=convs[m['actual_row_indices'][0]];req=requests[name]
    p=base.inspect_conv_precision(row)
    count=math.prod(initializers[req['weight_input']].dims)
    c('source_mapping_identity:'+name,m['source_request']==req and req['weight_input'] in row['Name'] and name in row['Name'])
    c('actual_INT8_weight_activation:'+name,p['int8_input_and_weight_formats'] and row['Weights']['Count']==count)
    observations.append({'source':name,'group':req['group'],'engine_layer':row['Name'],'weight_type':p['weights_type'],'weight_count':count,'activation_input_index':p['convolution_activation_input_index'],'input_formats':p['input_formats'],'output_formats':p['output_formats'],'tactic':p['tactic'],'tactic_value':p['tactic_value'],'named_integer_tactic':p['verified_int8_kernel']})
gaps=[r for r in observations if not r['named_integer_tactic']]
c('81_named_7_missing_unchanged',len(gaps)==7 and sum(r['named_integer_tactic'] for r in observations)==81 and all(not r['tactic'] for r in gaps))
audit=read(OUT/'precision_audit.json')
c('root_audit_identity',sha(OUT/'precision_audit.json')=='52d9f59be4b379817610c1f5d4ef82f73f7aafb23ead8060bfeef59082c9a4a5')
c('root_format_audit_agrees',audit['status']=='REQUESTED_CONV_FORMATS_VERIFIED' and audit['verified_int8_format_convs']==88 and not audit['errors'] and not audit['unmapped_engine_convs'] and audit['strict_int8_status']=='INTEGER_TACTIC_REVIEW_REQUIRED' and audit['source_convs_without_tactic_name']==7)
c('gap_identity_matches_root_audit',{r['source'] for r in gaps}=={r['onnx_node'] for r in audit['mapping'] if not r['strict_int8_tactic_verified']})
monitor=read(RAW/'monitor/status.json');guard=read(RAW/'supplemental_group_guard.json');end=read(RAW/'end_board_status.json')
c('monitor_success_898_53s',monitor['status']=='BUILD_SCRIPT_OK' and monitor['build_returncode']==0 and monitor['elapsed_seconds']==898.53 and monitor['max_seconds']==7200)
c('adapter_identity_received',sha(RAW/'build_entry.py')=='f26303efe459519b43aff0163fb63ff8b4e95c1699b5fc9b143d76bf4a12d1e5' and monitor['build_script'].endswith('/build_entry.py'))
c('guard_complete_no_intervention_empty',guard['status']=='COMPLETE' and guard['intervention_performed'] is False and guard['owned_group_fully_empty'] is True and guard['final_group_members']==[])
c('same_boot_chain',monitor['boot_id']==guard['boot_id']==end['boot_id']=='692c7544-b8bd-415b-939c-b11cdbd9f7f0')
c('end_monitor_build_absent',all(not row['stat'] for row in end['owned_process_checks'].values()))
c('one_native_build_attempt_marker',read(RAW/'native_build_attempt.json')['status']=='ONE_NATIVE_BUILD_ATTEMPT_CLAIMED' and len(list((RAW/'results').iterdir()))==1)
c('first_entry_failure_kept_separate',(ROOT/'artifacts/gate05/20261021_qall_none_entry_failure/native_evidence.zip').is_file())
report={'schema':'rampq.gate05.qall_none_independent_saved_build_review/v1','status':'SAVED_BUILD_FORMATS_AND_PROVENANCE_VERIFIED_STRICT_INTEGER_GAPS_REMAIN' if all(r['passed'] for r in checks) else 'BLOCKED','check_count':len(checks),'checks':checks,'blockers':[r['name'] for r in checks if not r['passed']],'observed_actual_source_convs':observations,'strict_integer_gaps':gaps,'parsed_constraint_summary':{'parser_layers':len(snapshots[0]),'precision_request_count':177,'output_request_count':177,'precision_INT8_DQ':176,'output_INT8_Q':176,'each_additional_shape_INT32':1,'before_after_afterbuild_identical':True},'build_seconds_not_benchmark':result['build_seconds_not_benchmark'],'monitor_seconds':monitor['elapsed_seconds'],'native_engine_sha256':sha(BUILD/'model.engine'),'scope':'One new NONE full graph built and deserialized with all88 INT8 input/weights source mappings. No execution/numerical/accuracy/floating-control or Gate pass; 81 named integer tactic proxy and7 missing names remain separate. New runtime+SASS evidence requires the new engine own process trace/maps and authenticated arithmetic review.','new_CPU_graph_native_or_SSH_runs':0,'static_ONNX_parse_only':True,'root_auditor_algorithm_used_only_for_existing_precision_metadata_classifier':True,'reviewer_script_sha256':sha(Path(__file__))}
report['local_review_revision']={'previous_report':'independent_native_review.json','previous_report_sha256':sha(OUT/'independent_native_review.json'),'previous_two_failed_local_expectations':['parser field is precision, not requested_precision','404209... pins audit_gate05_precision_candidate.py; inspect_conv_precision classifier is the separately unchanged audit_gate05_int8_precision.py4ef240...'],'native_raw_source_or_engine_changed':False,'new_model_execution':False}
out=OUT/'independent_native_review_v2.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'checks':len(checks),'blockers':report['blockers'],'report_sha256':sha(out),'parser_layers':len(snapshots[0]),'strict_integer_gaps':len(gaps)}))
