"""Static v2 chain audit plus synthetic cases only; never open predictions."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def c(name,value):checks.append({'name':name,'passed':bool(value)})
def load(name):
    path=ROOT/'scripts'/name
    spec=importlib.util.spec_from_file_location('review_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path,path.read_text(encoding='utf-8')
def funcs(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
def same_function(a,b,name):return ast.dump(funcs(a)[name],include_attributes=False)==ast.dump(funcs(b)[name],include_attributes=False)
raw,rp,rt=load('run_gate05_frozen_raw_dataset_v2.py')
ev,ep,et=load('evaluate_gate05_saved_target_matrix_v2.py')
post,pp,pt=load('postprocess_gate05_frozen_raw_matrix_v2.py')
pins={rp.name:'ef7e2780203c04442ee3378b7cea31093dafa40eb29c4cea9c012cdb0cd4d26b',
      ep.name:'b5ac60351e17761144a8897300ef8c8005f14e87e119b74dc1be2907be4cb895',
      pp.name:'d887b9f43fbecf1fec6a9dd3d1663160049cf8b36df57fe167f5d66bb372798d'}
for p in (rp,ep,pp):c('v2_exact_pin:'+p.name,sha(p)==pins[p.name]);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
old_raw=(ROOT/'scripts/run_gate05_frozen_raw_dataset.py').read_text(encoding='utf-8')
old_ev=(ROOT/'scripts/evaluate_gate05_saved_target_matrix.py').read_text(encoding='utf-8')
old_post=(ROOT/'scripts/postprocess_gate05_frozen_raw_matrix.py').read_text(encoding='utf-8')
for name in ('make_ort_driver','make_trt_driver','run','binding_signature'):
    c('raw_execution_core_unchanged_AST:'+name,same_function(rt,old_raw,name))
for name in ('legacy_matching','match_saved','score_stream','validate_frozen','evaluate'):
    c('evaluator_core_unchanged_AST:'+name,same_function(et,old_ev,name))
old_summary,new_summary=copy.deepcopy(funcs(old_ev)['summarize']),copy.deepcopy(funcs(et)['summarize'])
old_summary.body=old_summary.body[:-1];new_summary.body=new_summary.body[:-1]
c('primary_verdict_metric_calculation_unchanged_AST',ast.dump(old_summary,include_attributes=False)==ast.dump(new_summary,include_attributes=False))
for name in ('source_decode','captured_call','load_environment'):
    c('postprocess_numeric_core_unchanged_AST:'+name,same_function(pt,old_post,name))
c('postprocessor_pins_new_evaluator',post.EVALUATOR.name==ep.name and post.EVALUATOR_SHA==sha(ep))
c('raw_vs_acceptance_contract_link_distinct',"index['raw_contract_sha256'] == contract['raw_contract_sha256']" in pt and "lineage['contract_sha256'] == index['raw_contract_sha256']" in pt and "index['contract_sha256'] == args.contract_sha256" in pt)
c('native_vs_ORT_identity_selection',"'source_graph_sha256' if stream['backend']=='trt' else" in pt and "lineage['engine_sha256'] == binding['engine_sha256']" in pt)
c('raw_exact150_default_no_model',"if not args.execute:" in rt and "backend['calls'] == 150" in rt and "contract['warmup_calls'] == 0" in rt and "contract['extra_inference_calls'] == 0" in rt)
c('one_full_ORT_call_site',rt.count("session.run(names, {'images': array})")==1)
c('one_TRT_call_site_and_host_clone',rt.count('context.execute_v2(')==1 and '.detach().cpu().numpy().copy()' in rt)
c('attempt_exclusive_and_no_resume',"verified['marker'].open('x'" in rt and 'no resume or retry' in rt)
c('nonfinite_outputs_preserved_not_passed','nonfinite_images.append' in rt and "row['all_outputs_finite'] is True" in pt)
c('post1800_once_warning_fail_closed',"policy['maximum_NMS_calls'] == 1800" in pt and "policy['maximum_attempts'] == 1" in pt and "policy['warning_fail_closed'] is True" in pt and "require(capture['complete']" in pt)
c('authenticated_input_manifest_each_row',"match_frozen_input(row, frozen_rows[image_index])" in pt and "row['input_array_sha256'] == frozen['array_sha256']" in pt)
c('input_manifest_remains_bound_at_finish',"sha(input_manifest)==contract['input_manifest_sha256']" in pt)
c('common_CPU_NMS_geometry_old_AST_not_scale_boxes',"require(str(prediction.device)=='cpu'" in pt and 'ops.scale_boxes(' not in pt)
c('saved_raw_copy_before_NMS',"primary=raw['main_output'].copy()" in pt and "source_raw_identities" in pt)
c('only_full150x12_inventory_allowed',"len(streams) == 12" in pt and "len(rows)==150" in pt and "calls==1800" in pt)
c('engineering_E_receipt_exact_six_IDs',"evidence['configuration_identities'] == expected_evidence" in et and "evidence['status'] == 'ELIGIBLE' and evidence['all_six_configuration_audits_eligible'] is True" in et)
c('primary_and_auxiliary_engineering_priority',"if not eligible:" in et and "'INCONCLUSIVE_ENGINEERING_OR_REFERENCE_GAP' if not eligible else" in et)
c('old_guard_and_no_pure_causal_upgrade',"original_complete_model_guard='FAILED_UNCHANGED'" in et and "'pure_G1_causality_proven': False" in et)
# Independent path fixtures call only the actual stdlib validator; no files are
# created, no raw tensors or images are loaded, and no backend is initialized.
anchor=(HERE/'__STATIC_UNCREATED_PATH_FIXTURES__').resolve()
inputs=anchor/'inputs';state=anchor/'state';out=anchor/'outputs';graph=anchor/'models/model.onnx'
def args(state_path=state,out_path=out):return SimpleNamespace(state_root=state_path,output=out_path,graph=graph,engine=None)
backend={'attempt_state_root':str(state)}
c('raw_path_accepts_exact_frozen_store',raw.validate_output_paths(args(),backend,inputs)==(state,out))
for name,a in [('alternate_state_root',args(anchor/'other_state')),('inside_inputs',args(out_path=inputs/'new')),
               ('inside_source_model_dir',args(out_path=graph.parent/'new')),('ancestor_of_source',args(out_path=anchor)),
               ('inside_attempt_store',args(out_path=state/'new')),('output_equals_state',args(out_path=state))]:
    try:raw.validate_output_paths(a,backend,inputs)
    except ValueError:c('raw_path_rejects:'+name,True)
    else:c('raw_path_rejects:'+name,False)
captures={}
for name,fn in [('evaluator17',lambda:ev.self_test(ROOT/'src/gate0.py')),('post_fake_NMS',post.self_test)]:
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):fn()
    captures[name]=json.loads(stream.getvalue())
c('synthetic_evaluator17_pass',len(captures['evaluator17']['checks'])==17 and all(r['passed'] for r in captures['evaluator17']['checks']))
c('synthetic_postprocess_no_real_NMS',captures['post_fake_NMS']['real_NMS_calls']==captures['post_fake_NMS']['model_calls']==captures['post_fake_NMS']['dataset_prediction_streams_read']==0)
c('no_backend_or_NMS_runtime_imports',not any(name in sys.modules for name in ('numpy','torch','torchvision','ultralytics','onnx','onnxruntime','tensorrt')))
report={'schema':'rampq.gate05.dataset_toolchain_v2_static_independent_review/v1','status':'STATIC_SOURCE_AND_SYNTHETIC_BOUNDARIES_VERIFIED_PENDING_FROZEN_INPUT_PREFLIGHT' if all(r['passed'] for r in checks) else 'BLOCKED',
 'check_count':len(checks),'checks':checks,'blockers':[r['name'] for r in checks if not r['passed']],'v2_tools':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in (rp,ep,pp)},
 'synthetic_checks':captures,'reviewer_modified_evaluator_disclosure':'Reviewer authored v2 auxiliary eligibility-priority patch; original evaluator author separately reviews that diff. Raw and postprocessing v2 remain other-agent authored and independently reviewed here.',
 'new_real_NMS_model_CPU_graph_native_SSH_or_150_output_reads':0,'native_and_ORT_outputs_not_present_or_scored_in_this_review':True,
 'remaining_pre_execution_requirements':['Root must freeze both noncircular contracts and bind exact six engine/source/reference/qualification hashes before any new150-image outputs.','Raw contract backend.attempt_state_root must be the intended separate canonical store on each host.','Acceptance postprocess_execution.matrix_root/attempt_root must be intended absolute common matrix and separate store, not protected inputs or original backend raw directories.','Refresh postprocess --describe-environment from the new v2 code; old619/b36 environment identity is not transferable.','Full source-only authenticated contract/input/evidence preflight is still required once final contracts and12raw manifests exist; synthetic source review is not that preflight.'],
 'numeric_reference_limits':'Independent whole ORT reference and target-level outcomes do not pass the original PyTorch/ORT guard. Saved paired raw descriptive discrepancy analysis remains a separate final-report deliverable; no large tolerance was introduced.',
 'old_tool_identities':{name:sha(ROOT/'scripts'/name) for name in ('run_gate05_frozen_raw_dataset.py','evaluate_gate05_saved_target_matrix.py','postprocess_gate05_frozen_raw_matrix.py')},
 'review_script_sha256':sha(Path(__file__))}
out=HERE/'dataset_toolchain_v2_static_review.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'checks':len(checks),'blockers':report['blockers'],'review_sha256':sha(out)}))
