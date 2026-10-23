"""Authenticate an ended smoke, then call its unchanged receipt collector once."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
cfg=sys.argv[1]
assert cfg in ('fp16_control','r_only_G1','r_only_G1_G5','r_only_G1_G2')
root=Path('/home/nvidia/rampq/gate05_matched_fixed_stage_20261021')/('gate05_matched_fixed_'+cfg+'_20261021')
archive=Path('/home/nvidia/rampq')/('gate05_matched_fixed_'+cfg+'_evidence_20261021.zip')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not archive.exists()
assert sha(root/'jetson_owned_group_guard.py')=='4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4'
assert sha(root/'jetson_collect_isolated_numeric_probe.py')=='af9b7f823e4f889813fb88b6e8a529a6b1a6423b0c7ead0d5704513b517775eb'
spec=importlib.util.spec_from_file_location('matched_terminal_owned',root/'jetson_owned_group_guard.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
protocol=read(root/'protocol.json');status=read(root/'monitor/status.json');guard=read(root/'supplemental_group_guard.json')
identity=read(root/'profile_runner_identity.json');child=read(root/'profile_child_guard.json');done=read(root/'profile_runner_completion.json')
boot=m.boot_id();assert boot==protocol['expected_boot_id']=='692c7544-b8bd-415b-939c-b11cdbd9f7f0'
assert status['boot_id']==boot and status['root']==str(root) and status['max_seconds']==300 and status['build_script']==str(root/'run_fixed.py')
assert status['status']=='BUILD_SCRIPT_OK' and status['build_returncode']==0 and status['reason'] is None
assert guard['boot_id']==boot and guard['root']==str(root) and guard['status']=='COMPLETE' and guard['owned_group_fully_empty'] and not guard['intervention_performed']
assert all(guard['monitor_identity'][k]==status[k] for k in ('root','monitor_pid','build_pid','max_seconds','build_script'))
assert identity['boot_id']==boot and identity['work_root']==str(root) and identity['protocol_sha256']==sha(root/'protocol.json')
assert identity['adapter_sha256']==protocol['adapter_sha256']==sha(root/'jetson_matched_fixed_input_adapter.py')
runner=identity['runner'];assert runner['pid']==runner['pgid']==runner['sid'] and runner['pid']>1
assert child['status']=='COMPLETE' and child['boot_id']==boot and child['work_root']==str(root) and not child['intervention_performed']
assert child['owned_pgid']==runner['pid'] and child['leader_start_ticks']==runner['start_ticks'] and child['owned_group_fully_empty']
assert child['runner_identity_sha256']==sha(root/'profile_runner_identity.json') and child['protocol_sha256']==sha(root/'protocol.json')
assert child['guard_sha256']==protocol['child_guard_sha256']==sha(root/'jetson_matched_profile_child_guard.py')
wrapper=m.OwnedGroup(status['build_pid'],boot,root/'run_fixed.py').snapshot()
escaped=m.OwnedGroup(runner['pid'],boot,Path(identity['adapter_path'])).snapshot()
assert not wrapper and not escaped
processes={}
for name,pid in {'monitor':status['monitor_pid'],'wrapper':status['build_pid'],'supplemental_guard':guard['guard_pid'],'profile_runner':runner['pid'],'profile_child_guard':child['guard_pid']}.items():
    process=m.process_info(pid);assert process is None or process['state'] in ('Z','X'),(name,process)
    processes[name]={'pid':pid,'process':process}
assert done['protocol_sha256']==sha(root/'protocol.json') and done['executions_attempted']==done['executions_completed']==2 and done['two_engine_calls_completed'] and done['all_attempted_calls_synchronized']
assert read(root/'profile_exit.json')['returncode']==0 and read(root/'profile_exit.json')['error'] is None
assert read(root/'profile_export.json')['returncode']==0 and (root/'profile/fixed_input.sqlite').is_file()
terminal=read(root/'profile_child_terminal_verification.json')
assert terminal['runner_group_fully_empty'] and terminal['child_guard_process_nonlive'] and not terminal['child_guard_intervention']
result=read(root/'fixed_outputs/smoke_result.json')
assert result['configuration']==cfg and result['execution_success'] and result['executions_completed']==result['executions_attempted']==2
assert result['new_CPU_model_executions']==0 and result['dataset_evaluation_run'] is False and all(result['all_outputs_finite'].values())
assert sha(root/'fixed_outputs/raw_outputs.npz')==result['raw_outputs_sha256']
assert len(result['output_npz_keys'])==2 and all(set(keys)=={'first','repeat'} for keys in result['output_npz_keys'].values())
kernel=(root/'monitor/kernel.log').read_text(errors='replace')
assert not any(marker in kernel for marker in ('controller error detected','HC died;','unregister_netdevice: waiting','Out of memory:','oom-kill:','Kernel panic'))
observation={'schema':'rampq.gate05.matched_fixed_terminal_before_collection/v1','configuration':cfg,'boot_id':boot,
             'monitor_status':status,'supplemental_guard_status':guard['status'],'child_guard_status':child['status'],
             'wrapper_members':wrapper,'escaped_runner_members':escaped,'five_processes':processes,
             'completed_engine_calls':2,'new_CPU_model_calls':0,'dataset_calls':0,'protective_intervention':False,
             'source_protocol_sha256':sha(root/'protocol.json'),'raw_outputs_sha256':result['raw_outputs_sha256'],
             'original_numerical_guard':'FAILED_UNCHANGED','gate05_status':'INCOMPLETE'}
with (root/'terminal_before_collection.json').open('x') as f:json.dump(observation,f,indent=2)
collected=subprocess.run([sys.executable,str(root/'jetson_collect_isolated_numeric_probe.py'),'--root',str(root),'--archive',str(archive)],capture_output=True,text=True,timeout=180)
assert collected.returncode==0,collected.stdout+collected.stderr
print(json.dumps({'terminal':observation,'collector':json.loads(collected.stdout)}),flush=True)
