"""Own input-only schema compatibility check; no child/guard/model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
source=ROOT/'scripts/jetson_matched_fixed_input_adapter.py'
spec=importlib.util.spec_from_file_location('matched_adapter_check_only',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
records=[]
for cfg in ('fp16_control','r_only_G1'):
    work=HERE/('adapter_input_fixture_'+cfg);work.mkdir(exist_ok=False)
    root=ROOT/'artifacts/gate05/20261021_matched_control_family_native'/cfg/'native_evidence'
    build=next((root/'results').iterdir());member=read(root/'spec.json');result=read(build/'build_result.json')
    audit_source=root.parent/'precision_audit.json'
    for name in ('jetson_matched_fixed_input_adapter.py','jetson_matched_profile_child_guard.py','jetson_gate05_control_fixed_input.py','jetson_build_matched_control.py','jetson_owned_group_guard.py'):
        shutil.copy2(ROOT/'scripts'/name,work/name)
    shutil.copy2(audit_source,work/'precision_audit.json')
    protocol={'schema':'rampq.gate05.matched_control_fixed_input_protocol/v1','status':'INPUT_ONLY_COMPATIBILITY_FIXTURE_NO_GPU',
      'configuration':cfg,'adapter_sha256':sha(source),'execution_helper_sha256':module.EXECUTION_HELPER_SHA,
      'build_contract_sha256':module.BUILD_CONTRACT_SHA,'matched_auditor_sha256':module.MATCHED_AUDITOR_SHA,
      'child_guard_sha256':sha(work/'jetson_matched_profile_child_guard.py'),'child_guard_required_before_GPU':True,
      'member_spec_sha256':sha(root/'spec.json'),'member_manifest_sha256':sha(root/'manifest.json'),
      'source_graph_sha256':sha(root/'model.onnx'),'parent_spec_sha256':sha(root/'parent_bundle/spec.json'),
      'parent_manifest_sha256':sha(root/'parent_bundle/manifest.json'),'executions':2,
      'nms':{'conf_thres':.25,'iou_thres':.7,'max_det':1000,'nc':10},'nvtx_range':'RAMPQ_MATCHED_QDQ_FIXED_IMAGE',
      'supervision':{'max_seconds':300,'private_process_group':True},'dataset_evaluation_run':False,
      'new_CPU_model_executions':0,'new_builds':0,'new_tolerances_defined':False,
      'original_complete_model_numerical_guard':'FAILED_UNCHANGED','gate05_status':'INCOMPLETE',
      'expected_boot_id':'692c7544-b8bd-415b-939c-b11cdbd9f7f0','monitor_build_script':'run_fixed.py',
      'original_image_shape':[1080,1920],'fixed_image_sha256':'494bbd9d09de29f3a992d434da03c252b4f6cf1f28099fd5a443d0c797f5ec52',
      'engine_sha256':sha(build/'model.engine'),'build_result_sha256':sha(build/'build_result.json'),
      'build_request_sha256':sha(build/'build_request.json'),'engine_layers_sha256':sha(build/'engine_layers.json'),
      'expected_binding_contract':member['expected_binding_contract'],'precision_audit_sha256':sha(audit_source),
      'diagnostic_formats_only_allowed':True,'input_sha256':member['input_file_sha256'],'input_file':member['input_file'],
      'cpu_reference':{'file':member['cpu_expected_outputs_file'],'sha256':member['cpu_expected_outputs_sha256'],
                       'output_npz_keys':member['expected_output_npz_keys']}}
    with (work/'protocol.json').open('x',encoding='utf-8') as f:json.dump(protocol,f,indent=2)
    files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in work.iterdir() if p.is_file()}
    with (work/'manifest.json').open('x',encoding='utf-8') as f:json.dump({'files':files},f,indent=2)
    args=SimpleNamespace(root=root,build=build,work_root=work,protocol=work/'protocol.json',audit=work/'precision_audit.json',
      protocol_sha256=sha(work/'protocol.json'),audit_sha256=sha(audit_source),engine_sha256=sha(build/'model.engine'),
      diagnostic_formats_only=True,check_input_only=True)
    verified=module.validate_inputs(args)
    records.append({'configuration':cfg,'status':'ACTUAL_SAVED_MEMBER_INPUTS_VALIDATED_ONLY',
                    'derived_fp16_conv_count':verified['spec']['fp16_conv_count'],
                    'strict_named_tactics_verified':verified['strict_integer_tactics_verified'],
                    'strict_named_tactic_gaps':len(verified['strict_integer_gaps']),
                    'fixture_protocol_sha256':sha(work/'protocol.json'),'fixture_gpu_execution_authorized':False})
report={'schema':'rampq.gate05.matched_smoke_adapter_input_compatibility/v1','status':'TWO_ACTUAL_SAVED_MEMBER_SCHEMAS_PASSED_NO_EXECUTION',
 'adapter_sha256':sha(source),'child_guard_sha256':sha(ROOT/'scripts/jetson_matched_profile_child_guard.py'),'records':records,
 'forbidden_runtime_modules_imported':[name for name in ('numpy','cv2','onnx','onnxruntime','torch','tensorrt','ultralytics') if name in sys.modules],
 'guard_handshake_or_private_session_tested':False,'new_model_CPU_native_SSH_or_child_process_runs':0,
 'fixtures_scope':'Input-only compatibility fixtures, explicitly rejected by adapter/guard for execution. Root must freeze a separate execution protocol and reviewed profiling wrapper.'}
assert not report['forbidden_runtime_modules_imported']
out=HERE/'matched_smoke_adapter_input_compatibility.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'adapter_sha256':report['adapter_sha256'],'child_guard_sha256':report['child_guard_sha256'],'review_sha256':sha(out)}))
