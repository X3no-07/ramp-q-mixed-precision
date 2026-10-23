"""Freeze final raw and target contracts after six-model technical review.

Reads only identities and qualification reports; never executes a model or
reads dataset model outputs. The original draft and every old failure remain.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sys
import tempfile
import traceback

sys.dont_write_bytecode=True

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/gate05'
PLAN=BASE/'20261021_gate05_completion_plan'
ORDER=['fp16_control','q_all','r_only_G1','r_only_G1_G5','r_only_G1_G2','q_only_G1']
RAW_SHA='ef7e2780203c04442ee3378b7cea31093dafa40eb29c4cea9c012cdb0cd4d26b'
EVAL_SHA='b5ac60351e17761144a8897300ef8c8005f14e87e119b74dc1be2907be4cb895'
POST_SHA='d887b9f43fbecf1fec6a9dd3d1663160049cf8b36df57fe167f5d66bb372798d'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
    with p.open('x',encoding='utf-8') as f:
        json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

@contextmanager
def unpublished_output(target):
    """Publish the complete pair once; preserve failed partial work separately."""
    require(target.parent==BASE and not target.exists() and not target.is_symlink(),'Fresh direct Gate0.5 contract folder required')
    partial=Path(tempfile.mkdtemp(prefix='.'+target.name+'.UNPUBLISHED-',dir=str(target.parent)))
    try:
        (partial/'qualification').mkdir()
        yield partial
        receipt=read(partial/'freeze_receipt.json')
        require(sha(partial/'raw_contract.json')==receipt['raw_contract_sha256'] and
                sha(partial/'acceptance_contract.json')==receipt['acceptance_contract_sha256'],
                'Final contract bytes changed before publication')
        for name,info in receipt['files'].items():
            path=partial/name
            require(path.is_file() and path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],'Frozen output changed before publication')
        require(not target.exists() and not target.is_symlink(),'Final destination appeared; preserve unpublished work')
        partial.rename(target)
    except BaseException as error:
        if partial.is_dir():
            write(partial/'freeze_failure.json',{'status':'UNPUBLISHED_PARTIAL_NOT_A_FROZEN_CONTRACT_PAIR',
                  'target':str(target),'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
                  'model_calls':0,'dataset_outputs_read':False,'automatic_retry':False})
        raise

def validate_payloads(out,raw,accept):
    """Reuse the pure metadata evaluator validator; never load outputs/runtime."""
    require(list(raw['configurations'])==ORDER and [r['name'] for r in accept['configurations']]==ORDER,'Configuration order differs')
    expected=[{'configuration':cfg,'backend':backend} for backend in ('trt','ort') for cfg in ORDER]
    require(accept['postprocess_execution']['stream_order']==expected and len(expected)==12,'Wrong twelve-stream dictionary order')
    require(accept['raw_contract_sha256']==sha(out/'raw_contract.json') and
            accept['input_manifest_sha256']==raw['input_manifest']['sha256'],'Contract pair/input identities differ')
    for path in ('run_gate05_frozen_raw_dataset_v2.py','evaluate_gate05_saved_target_matrix_v2.py'):
        spec=importlib.util.spec_from_file_location('freeze_metadata_'+Path(path).stem,ROOT/'scripts'/path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if path.startswith('run_'):
            module.binding_signature(raw['bindings'])
            for cfg in ORDER:
                for backend in ('ort','trt'):
                    name=raw['configurations'][cfg][backend]['attempt_state_root']
                    require((backend=='ort' and Path(name).is_absolute()) or
                            (backend=='trt' and PurePosixPath(name).is_absolute() and name.startswith('/home/nvidia/rampq/')),
                            'Attempt store must be absolute on its execution platform')
        else:
            module.validate_frozen(accept,sha(out/'acceptance_contract.json'),out/'acceptance_contract.json',ROOT/'src/gate0.py')
    for cfg in ORDER:
        row=raw['configurations'][cfg]
        require(row['auxiliary_policy_difference_disclosed'] is (cfg=='q_only_G1'),'Auxiliary policy role differs')
        for backend in ('ort','trt'):
            for evidence in row[backend]['required_evidence']:
                path=out/evidence['file']
                require(out in path.resolve().parents and path.stat().st_size==evidence['bytes'] and
                        sha(path)==evidence['sha256'],'Qualification copy not bound to this pair')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--technical-decision',type=Path,required=True);p.add_argument('--technical-decision-sha256',required=True)
    p.add_argument('--float-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();decision=read(a.technical_decision)
    require(sha(a.technical_decision)==a.technical_decision_sha256 and decision['status']=='ALL_SELECTED_SIX_TECHNICALLY_ELIGIBLE_FOR_CONDITIONAL_DEPLOYMENT_EVALUATION','Root technical adjudication incomplete')
    require(decision['new150_model_calls']==0 and decision['new150_outputs_read'] is False,'This freeze must precede new150 outcomes')
    require(decision['float_fusion_audit_sha256']==sha(a.float_audit),'Float review identity differs')
    table_path=PLAN/'selected_deployment_identity_table.json';table=read(table_path)
    require(sha(table_path)=='542cfb974e32e81fa1acbb6c117a8902044ee574c86993135917b4eaa8800eeb' and [m['name'] for m in table['members']]==ORDER,'Selected deployment family changed')
    for name,value in [('run_gate05_frozen_raw_dataset_v2.py',RAW_SHA),('evaluate_gate05_saved_target_matrix_v2.py',EVAL_SHA),('postprocess_gate05_frozen_raw_matrix_v2.py',POST_SHA)]:require(sha(ROOT/'scripts'/name)==value,'Final tool changed: '+name)
    env=read(PLAN/'postprocess_environment_v2.json');require(env['postprocess_identity']['implementation_sha256']==POST_SHA and env['model_calls']==env['real_NMS_calls']==0,'Wrong pre-outcome common CPU environment')
    shared=BASE/'20261021_dataset_matrix/common_inputs';input_manifest=read(shared/'manifest.json')
    require(sha(shared/'manifest.json')=='05c6509f5afcf6a715a2328aac9581ca2f9534ee5de6fcffc670a3fd745f4f80','Shared inputs changed')
    for row in input_manifest['images']:require(sha(shared/row['input_file'])==row['input_sha256'],'Common input changed')
    selected={m['name']:m for m in table['members']}
    require(set(decision['configuration_evidence'])==set(ORDER),'Missing selected configuration qualification')
    protected={a.technical_decision.resolve(),a.float_audit.resolve(),table_path,PLAN/'deployment_scope_interpretation_policy.json',
               PLAN/'auxiliary_qonly_selection_policy.json',PLAN/'dataset_acceptance_contract_draft.json',
               PLAN/'postprocess_environment_v2.json',shared/'manifest.json',ROOT/'src/gate0.py',Path(__file__).resolve()}
    protected.update(ROOT/'scripts'/name for name in ('run_gate05_frozen_raw_dataset_v2.py',
                     'evaluate_gate05_saved_target_matrix_v2.py','postprocess_gate05_frozen_raw_matrix_v2.py'))
    protected.update(shared/row['input_file'] for row in input_manifest['images'])
    for cfg,m in selected.items():
        require(m['independent_ORT_reference_graph_sha256']==m['source_graph_sha256'],'ORT must use this exact full source graph')
        require(m['runtime_build_policy']==('AUX_PREFER_WITH82_FP16_COMPUTE_AND_OUTPUT_REQUESTS' if cfg=='q_only_G1' else 'NONE_NO_LAYER_SETTERS'),
                'Undisclosed selected primary/auxiliary policy difference')
        require(isinstance(decision['configuration_evidence'][cfg],list),'Configuration supplements must be an explicit list')
        if m['missing_named_integer_tactics']:
            require(bool(decision['configuration_evidence'][cfg]),'Missing authenticated arithmetic supplement for named-tactic gaps')
        for key in ('local_model','local_engine','local_precision_audit'):
            protected.add(Path(m[key]))
        build=Path(m['local_build'])
        for name,key in (('build_request.json','build_request_sha256'),('build_result.json','build_result_sha256'),('engine_layers.json','engine_layers_sha256')):
            require(sha(build/name)==m[key],'Selected build provenance changed: '+name);protected.add(build/name)
        for item in decision['configuration_evidence'][cfg]:
            path=Path(item['file']);require(sha(path)==item['sha256'],'Qualified supplement changed');protected.add(path)
    inputs_before={str(x):sha(x) for x in protected}
    target=a.output.resolve()
    with unpublished_output(target) as out:
        copies={}
        def add_evidence(path,name):
            path=Path(path).resolve();dst=out/'qualification'/name;require(not dst.exists(),'Duplicate evidence name')
            shutil.copy2(path,dst);require(sha(path)==sha(dst),'Evidence copy changed');copies[name]={'file':'qualification/'+name,'sha256':sha(dst),'bytes':dst.stat().st_size};return copies[name]
        common=[add_evidence(a.technical_decision,'technical_decision.json'),add_evidence(a.float_audit,'float_fusion_comparability.json'),
                add_evidence(PLAN/'deployment_scope_interpretation_policy.json','deployment_scope_policy.json'),
                add_evidence(PLAN/'auxiliary_qonly_selection_policy.json','auxiliary_qonly_selection_policy.json'),add_evidence(table_path,'selected_identity_table.json')]
        raw={'schema':'rampq.gate05.frozen_raw_dataset_contract/v1','status':'FROZEN_BEFORE_DATASET_INFERENCE','client_at':datetime.now().astimezone().isoformat(),
             'runner_sha256':RAW_SHA,'input_manifest':{'schema':'rampq.gate05.shared_preprocessed_inputs/v1','sha256':sha(shared/'manifest.json'),'count':150,'ordered_ids':[r['image'] for r in input_manifest['images']]},
             'attempt_namespace':'gate05_selected_deployment_20261021_v1','maximum_attempts_per_backend_configuration':1,'warmup_calls':0,'extra_inference_calls':0,'new_builds':0,'no_NMS':True,
             'dataset_accuracy_evaluation_by_runner':False,'new_tolerances_defined':False,'original_complete_model_numerical_guard':'FAILED_UNCHANGED','gate05_status':'INCOMPLETE',
             'bindings':read(Path(selected['fp16_control']['local_source'])/'spec.json')['expected_binding_contract'],
             'output_npz_keys':{'output0':'main_output','/model.10/m/m.0/attn/pe/conv/Conv_output_0':'pe_output'},'configurations':{},
             'execution_order':ORDER,'native_total_model_calls':900,'CPU_ORT_total_model_calls':900,
             'nonfinite_policy':'Retain this member full150raw outputs and statistics. Stop queue after current member; do not retry or issue a PASS. True execution/shape/resource errors stop immediately and retain partial records.',
             'maximum_seconds_per_backend_member':1800,'native_serial_only':True,'local_ORT_serial_only':True,'automatic_replay_or_resume':False}
        for cfg,m in selected.items():
            audit=Path(m['local_precision_audit']);require(sha(audit)==m['native_precision_audit_sha256'],'Actual audit changed')
            ae=read(audit);require(ae['requested_conv_formats_complete'] and not ae['errors'] and not ae['unmapped_engine_convs'],'Selected weighted precision invalid')
            require(sha(Path(m['local_model']))==m['source_graph_sha256'] and sha(Path(m['local_engine']))==m['engine_sha256'],'Selected graph/engine changed')
            evidence=[add_evidence(audit,cfg+'_precision_audit.json')]
            for i,item in enumerate(decision['configuration_evidence'][cfg]):
                path=Path(item['file']);require(sha(path)==item['sha256'],'Qualified supplement changed')
                evidence.append(add_evidence(path,cfg+'_supplement_'+str(i)+'.json'))
            raw['configurations'][cfg]={'full_model':True,'graph_sha256':m['source_graph_sha256'],
                'ort':{'authorized':True,'calls':150,'provider':'CPUExecutionProvider','graph_optimization':'ORT_DISABLE_ALL','threads':4,
                       'attempt_state_root':str((BASE/'20261021_dataset_attempts').resolve()),'required_evidence':common+evidence,
                       'environment':{'python':'3.12.14','onnxruntime':'1.29.0','numpy':'2.5.3','providers':['CPUExecutionProvider'],'graph_optimization':'ORT_DISABLE_ALL','intra_op_threads':4,'inter_op_threads':1,'execution_mode':'ORT_SEQUENTIAL'}},
                'trt':{'authorized':True,'calls':150,'engine_sha256':m['engine_sha256'],'attempt_state_root':'/home/nvidia/rampq/gate05_dataset_attempts_20261021',
                       'required_loaded_libraries':[{'basename':'libnvinfer.so.8.5.2','sha256':'4536788450d9a11f98701aa05021116ef13efd99f9318a03c01a2bc014d1c6ca'}],
                       'required_evidence':common+evidence,'environment':{'python':'3.8.10','torch':'2.0.0+nv23.05','tensorrt':'8.5.2.2','numpy':'1.24.4','GPU':'Orin','compute_capability':[8,7]}},
                'actual_build_policy':m['runtime_build_policy'],'auxiliary_policy_difference_disclosed':cfg=='q_only_G1'}
        write(out/'raw_contract.json',raw)
        draft=PLAN/'dataset_acceptance_contract_draft.json';accept=read(draft)
        accept.update(schema='rampq.gate05.dataset_acceptance_contract/v1',status='FROZEN_BEFORE_DATASET_INFERENCE',client_at=datetime.now().astimezone().isoformat(),
            author_role='Root final pre-outcome conditional-deployment adjudication; referenced independent technical checks; previously used holdout is not new independent preregistration.',
            draft_parent_sha256=sha(draft),raw_contract_sha256=sha(out/'raw_contract.json'),input_manifest_sha256=sha(shared/'manifest.json'),
            evaluator_identity={'file_sha256':EVAL_SHA,'matching_source_sha256':'f3c4d80658b79ae685e569b37c5ea89af9e6e9dd4a63ce86aff2806a7fcc5adb'},postprocess_identity=env['postprocess_identity'])
        accept['freeze_rule']['no_freeze_timestamp_or_approval_in_this_draft']=False
        accept['freeze_rule']['all_null_artifact_identities_must_be_bound_or_explicitly_justified_before_execution']=True
        accept['postprocess_execution']={'maximum_NMS_calls':1800,'maximum_attempts':1,'automatic_retry':False,'warning_fail_closed':True,
            'matrix_root':str((BASE/'20261021_dataset_matrix').resolve()),'attempt_root':str((BASE/'20261021_postprocess_attempts').resolve()),
            'stream_order':[{'configuration':cfg,'backend':backend} for backend in ('trt','ort') for cfg in ORDER]}
        for row in accept['configurations']:
            m=selected[row['name']]
            for key in ('source_graph_sha256','engine_sha256','build_request_sha256','native_precision_audit_sha256','independent_ORT_reference_graph_sha256'):row[key]=m[key]
            row.update(float_fusion_comparability_audit_sha256=sha(a.float_audit),binding_status='FROZEN_COMPLETE_SELECTED_ENGINE_AND_REFERENCE_IDENTITIES',actual_build_policy=m['runtime_build_policy'])
        accept['family_build_and_format_contract']['recommended_common_policy']='Actual selected primary5 NONE/FP16true/TF32false; INT8false forFP16baseline andtrue forquantized. Auxiliaryqonly exactsameONNX has PREFER+82FP16compute/outputrequests. Complete actual precision checked, all differences disclosed; not identicalflags or purearithmeticcause.'
        accept['family_build_and_format_contract']['same_flags_even_if_unused']='Original draft recommendation was not adopted; actual pre-outcome policy is precisely bound to selected identities and root deployment_scope/auxiliaryselection policies.'
        accept['floating_and_fusion_comparability']['must_match_or_be_bridged']='Final scope follows frozen deployment_scope_interpretation_policy: authenticate unchanged source/ranges, complete weighted precision/fusion mapping and report realized floating differences as conditional deployment effects. Do not claim pureG1 arithmetic control. Original draft strict_float_match analysis remains preserved separately.'
        accept['floating_and_fusion_comparability']['interpretation_policy_sha256']=sha(PLAN/'deployment_scope_interpretation_policy.json')
        accept['floating_and_fusion_comparability']['auxiliary_selection_policy_sha256']=sha(PLAN/'auxiliary_qonly_selection_policy.json')
        accept['floating_and_fusion_comparability']['actual_complete_audit_sha256']=sha(a.float_audit)
        accept['common_input_postprocess_and_matching']['NMS']['implementation_and_version_sha256']=env['postprocess_identity']
        accept['unbound_prerequisites_before_freeze']=[]
        accept['technical_preflight_decision_sha256']=a.technical_decision_sha256
        accept['new150_model_calls_before_freeze']=0
        accept['integer_arithmetic_evidence']['selected_configuration_evidence']=decision['configuration_evidence']
        accept['selected_family_roles']={'primary_core_five_NONE':ORDER[:5],'auxiliary_different_PREFER_policy':['q_only_G1'],
            'rejected_qonly_NONE_engine_sha256':read(PLAN/'auxiliary_qonly_selection_policy.json')['rejected_new_NONE_control']['engine_sha256'],
            'rejected_qonly_NONE_precision_failure_preserved':True,'pure_G1_arithmetic_causality_claim':False}
        accept['optional_provenance_nulls_justified']='Original sources.*.lines/json_pointer null means a whole-file source citation rather than a missing artifact identity. All execution, model, audit, evaluator, input and postprocessing identities are bound separately.'
        def check_no_unbound_identity(value,path=''):
            if isinstance(value,dict):
                for key,item in value.items():check_no_unbound_identity(item,path+'.'+key)
            elif isinstance(value,list):
                for index,item in enumerate(value):check_no_unbound_identity(item,path+'['+str(index)+']')
            elif value is None:
                require(path.startswith('.sources.') and (path.endswith('.lines') or path.endswith('.json_pointer')),
                        'Unbound final contract field: '+path)
        check_no_unbound_identity(accept)
        write(out/'acceptance_contract.json',accept)
        validate_payloads(out,raw,accept)
        require(inputs_before=={name:sha(Path(name)) for name in inputs_before},'Protected qualification changed during freeze')
        write(out/'freeze_receipt.json',{'raw_contract_sha256':sha(out/'raw_contract.json'),'acceptance_contract_sha256':sha(out/'acceptance_contract.json'),
              'technical_decision_sha256':a.technical_decision_sha256,'float_audit_sha256':sha(a.float_audit),'model_calls_by_freezer':0,
              'dataset_outputs_read_by_freezer':False,'original_draft_and_failures_unchanged':True,
              'protected_input_hashes_before_and_after':inputs_before,'publication':'COMPLETE_PAIR_THEN_ATOMIC_RENAME_TO_REQUESTED_FINAL_DIRECTORY',
              'files':{p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps(read(target/'freeze_receipt.json')|{'files':'SEE_SAVED_RECEIPT'},indent=2))

if __name__=='__main__':main()
