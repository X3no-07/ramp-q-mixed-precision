"""Seal an audited member's new two-call profiled work; no model execution."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'artifacts/gate05/20261021_matched_control_family_native'
DEST=ROOT/'artifacts/gate05/20261021_matched_fixed'
ORDER=('fp16_control','r_only_G1','r_only_G1_G5','r_only_G1_G2','q_only_G1')
PINS={
 'jetson_matched_fixed_input_adapter.py':'5651c2b55da5af498637fa73f28e79ac6117c50b01e4f2dc7cd08e502327a1d5',
 'jetson_matched_profile_child_guard.py':'ebdb2ea93ace365250fcaed1a7ee71e99bad858445e475295ba43bf0dc666347',
 'jetson_gate05_control_fixed_input.py':'d1576ea834bd48ad78df3bf920b315726c1f0da8fe388eb7c3450dda8d93e0ca',
 'jetson_build_matched_control.py':'4d2ee7d283abf6cd75b6d76e58870f79f11392fa06eaef37936fab6d7bafca34',
 'jetson_owned_group_guard.py':'4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4',
 'jetson_monitored_build.py':'71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1',
 'jetson_collect_isolated_numeric_probe.py':'af9b7f823e4f889813fb88b6e8a529a6b1a6423b0c7ead0d5704513b517775eb',
 'jetson_run_matched_fixed_profile.py':'8938ead8199f4a638396c622db61048f2fa659caaa8e4a11f4416d6518945d5a'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def info(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--configuration',choices=ORDER,required=True);a=parser.parse_args()
    cfg=a.configuration;member=NATIVE/cfg/'native_evidence';audit_path=NATIVE/cfg/'precision_audit.json'
    spec=read(member/'spec.json');audit=read(audit_path);builds=list((member/'results').glob('build_*'));assert len(builds)==1
    build=builds[0];result=read(build/'build_result.json')
    assert audit['requested_conv_formats_complete'] and not audit['errors'] and not audit['unmapped_engine_convs']
    assert audit['engine_sha256']==result['engine_sha256']==sha(build/'model.engine')
    assert audit['configuration']==spec['configuration']==cfg and audit['source_spec_sha256']==sha(member/'spec.json')
    assert audit['layer_file_sha256']==sha(build/'engine_layers.json') and audit['build_request_sha256']==sha(build/'build_request.json')
    for name,value in PINS.items():assert sha(ROOT/'scripts'/name)==value,name
    out=DEST/cfg;DEST.mkdir(exist_ok=True);assert not out.exists() and not out.with_suffix('.zip').exists();out.mkdir()
    for name in PINS:shutil.copy2(ROOT/'scripts'/name,out/('run_fixed.py' if name=='jetson_run_matched_fixed_profile.py' else name))
    shutil.copy2(audit_path,out/'precision_audit.json')
    board_member='/home/nvidia/rampq/gate05_matched_native_queue_20261021/gate05_matched_'+cfg+'_20261021'
    board_build=board_member+'/'+build.relative_to(member).as_posix()
    archive_root='gate05_matched_fixed_'+cfg+'_20261021';work='/home/nvidia/rampq/gate05_matched_fixed_stage_20261021/'+archive_root
    p={'schema':'rampq.gate05.matched_control_fixed_input_protocol/v1','status':'FROZEN_BEFORE_MATCHED_FIXED_INPUT_EXECUTION','client_at':datetime.now().astimezone().isoformat(),
       'configuration':cfg,'adapter_sha256':PINS['jetson_matched_fixed_input_adapter.py'],'child_guard_sha256':PINS['jetson_matched_profile_child_guard.py'],
       'profile_wrapper_sha256':PINS['jetson_run_matched_fixed_profile.py'],'execution_helper_sha256':PINS['jetson_gate05_control_fixed_input.py'],
       'build_contract_sha256':PINS['jetson_build_matched_control.py'],'matched_auditor_sha256':'97e031e6ed16f18634483fa3f0f1fa98903f39e12551b94390feab64a8a66d30',
       'member_spec_sha256':sha(member/'spec.json'),'member_manifest_sha256':sha(member/'manifest.json'),'source_graph_sha256':sha(member/'model.onnx'),
       'parent_spec_sha256':sha(member/'parent_bundle/spec.json'),'parent_manifest_sha256':sha(member/'parent_bundle/manifest.json'),
       'engine_sha256':sha(build/'model.engine'),'precision_audit_sha256':sha(audit_path),'build_result_sha256':sha(build/'build_result.json'),
       'build_request_sha256':sha(build/'build_request.json'),'engine_layers_sha256':sha(build/'engine_layers.json'),'expected_binding_contract':spec['expected_binding_contract'],
       'input_file':spec['input_file'],'input_sha256':spec['input_file_sha256'],'cpu_reference':{'file':spec['cpu_expected_outputs_file'],'sha256':spec['cpu_expected_outputs_sha256'],'output_npz_keys':spec['expected_output_npz_keys']},
       'primary_output_name':'output0','executions':2,'nms':{'conf_thres':0.25,'iou_thres':0.7,'max_det':1000,'nc':10},'nvtx_range':'RAMPQ_MATCHED_QDQ_FIXED_IMAGE',
       'original_image_shape':[1080,1920],'fixed_image_sha256':'494bbd9d09de29f3a992d434da03c252b4f6cf1f28099fd5a443d0c797f5ec52',
       'diagnostic_formats_only_allowed':True,'child_guard_required_before_GPU':True,'expected_boot_id':'692c7544-b8bd-415b-939c-b11cdbd9f7f0',
       'monitor_build_script':'run_fixed.py','supervision':{'max_seconds':300,'private_process_group':True,'independent_escaped_runner_guard':True},
       'dataset_evaluation_run':False,'new_CPU_model_executions':0,'new_builds':0,'new_tolerances_defined':False,'original_complete_model_numerical_guard':'FAILED_UNCHANGED','gate05_status':'INCOMPLETE',
       'adapter_static_review_sha256':'dada16b6c60dafa37564c2958e425b2f625526361e7d2492670030da3e87cd69',
       'wrapper_static_review_sha256':'d9cc3342eb1486fa4d1b53b09ea0682d0577a70afbc6c1565a3d94949ef9df3f',
       'limits':'Fixed Selection input,exactly2newenginecalls,raw descriptive only. Old named-tactic gaps remain; new authenticated own-runtime/SASS branch is a separate audit. No150-image or speed/accuracy/Gate claim.'}
    write(out/'protocol.json',p)
    args=[]
    for key,value in {'--work-root':work,'--protocol':work+'/protocol.json','--protocol-sha256':sha(out/'protocol.json'),'--engine-sha256':p['engine_sha256'],
                      '--audit':work+'/precision_audit.json','--audit-sha256':p['precision_audit_sha256'],'--output':work+'/fixed_outputs','--root':board_member,'--build':board_build}.items():args.extend([key,value])
    args.append('--diagnostic-formats-only')
    write(out/'execution_spec.json',{'schema':'rampq.gate05.matched_fixed_profile_execution/v1','configuration':cfg,'protocol_sha256':sha(out/'protocol.json'),
          'engine_executions_allowed':2,'native_builds':0,'adapter_arguments':args,'member_root':board_member,'build_root':board_build})
    write(out/'manifest.json',{'schema':'rampq.gate05.matched_fixed_work_bundle/v1','configuration':cfg,'files':{p.name:info(p) for p in sorted(out.iterdir()) if p.is_file()}})
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):z.write(path,archive_root+'/'+path.name)
    with zipfile.ZipFile(out.with_suffix('.zip')) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={archive_root+'/'+path.name for path in out.iterdir()}
        for path in out.iterdir():assert hashlib.sha256(z.read(archive_root+'/'+path.name)).hexdigest()==sha(path)
    print(json.dumps({'configuration':cfg,'archive':str(out.with_suffix('.zip')),'archive_sha256':sha(out.with_suffix('.zip')),'archive_bytes':out.with_suffix('.zip').stat().st_size,'protocol_sha256':sha(out/'protocol.json'),'manifest_sha256':sha(out/'manifest.json'),'work_root':work,'new_model_executions':0}))

if __name__=='__main__':main()
