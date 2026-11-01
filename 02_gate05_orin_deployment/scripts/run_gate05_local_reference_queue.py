"""One bounded serial complete-ORT reference queue; no NMS or scoring.

Default only validates all frozen inputs with the pinned raw runner. Explicit
--execute claims one queue attempt. No GPU backend, warmup, resume or retry.
"""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ORDER=['fp16_control','q_all','r_only_G1','r_only_G1_G5','r_only_G1_G2','q_only_G1']
RUNNER_SHA='ef7e2780203c04442ee3378b7cea31093dafa40eb29c4cea9c012cdb0cd4d26b'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    require(not p.is_symlink(),'Queue status became symlink')
    tmp=p.with_suffix('.tmp')
    require(not tmp.is_symlink(),'Queue status temporary became symlink')
    with tmp.open('w',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)

def interrupt(_sig,_frame):raise KeyboardInterrupt('Local reference queue interrupted; stop exact owned child')

def disjoint(a,b):return a!=b and a not in b.parents and b not in a.parents

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args();plan=read(a.plan)
    require(sha(a.plan)==a.plan_sha256 and plan['schema']=='rampq.gate05.local_reference_queue_plan/v1' and plan['queue_script_sha256']==sha(Path(__file__)),'Wrong queue plan/tool')
    require(plan['configuration_order']==ORDER and [x['configuration'] for x in plan['members']]==ORDER and plan['maximum_model_calls']==900 and plan['max_seconds_per_member']==1800,'Queue order/budget differs')
    runner=Path(plan['runner_file']);contract=Path(plan['contract_file']);base=Path(plan['run_root'])
    require(all(path.is_absolute() and path.resolve()==path and not path.is_symlink() for path in (runner,contract,base)) and
            sha(runner)==RUNNER_SHA and sha(contract)==plan['contract_sha256'] and base.is_dir(),'Runner/contract/root changed')
    raw=read(contract);require(raw['schema']=='rampq.gate05.frozen_raw_dataset_contract/v1' and raw['runner_sha256']==RUNNER_SHA and
            raw['maximum_attempts_per_backend_configuration']==1 and raw['new_builds']==raw['warmup_calls']==raw['extra_inference_calls']==0 and
            raw['no_NMS'] is True and raw['dataset_accuracy_evaluation_by_runner'] is False,'Raw scope differs')
    require(not (base/'attempt.json').exists() and not (base/'status.json').exists(),'No queue replay/resume')
    inputs=Path(plan['inputs_root'])
    require(inputs.is_absolute() and inputs.resolve()==inputs and inputs.is_dir(),'Noncanonical shared input root')
    protected={inputs,runner.parent,contract.parent}
    for item in plan['members']:
        graph=Path(item['graph_file']);require(graph.is_absolute() and graph.resolve()==graph and graph.is_file() and not graph.is_symlink(),'Noncanonical source graph')
        protected.add(graph.parent)
    require(all(disjoint(base,path) for path in protected),'Queue logs/markers must be outside protected source/contract/input directories')
    require(len({str(Path(item['output']).resolve()) for item in plan['members']})==6,'Duplicate reference output directories')
    commands=[]
    for item in plan['members']:
        cfg=item['configuration'];out=Path(item['output'])
        require(out.is_absolute() and out.resolve()==out and base in out.parents and not out.exists() and not out.is_symlink(),'Fresh reference output must be inside frozen run root')
        require(not (base/(cfg+'_console.log')).exists(),'Reference console already exists; no queue replay')
        require(raw['configurations'][cfg]['ort']['authorized'] and raw['configurations'][cfg]['ort']['calls']==150,'Reference not authorized')
        argv=[sys.executable,'-u','-B',str(runner),'--contract',str(contract),'--contract-sha256',plan['contract_sha256'],'--configuration',cfg,'--backend','ort',
              '--inputs',plan['inputs_root'],'--graph',item['graph_file'],'--state-root',raw['configurations'][cfg]['ort']['attempt_state_root'],
              '--output',item['output'],'--evidence-root',str(contract.parent)]
        result=subprocess.run(argv+['--check-input-only'],capture_output=True,text=True,timeout=120)
        require(result.returncode==0,'Reference pure-input preflight failed: '+result.stdout+result.stderr)
        check=json.loads(result.stdout);require(check['status']=='FROZEN150_RAW_DATASET_INPUTS_VERIFIED_NO_MODEL_EXECUTION' and
              check['model_calls']==0 and check['planned_calls']==150 and check['native_runtime_imported'] is False and
              check['new_builds']==0 and check['NMS_run'] is False and check['lineage']['configuration']==cfg and
              check['lineage']['backend']=='ort' and check['lineage']['contract_sha256']==plan['contract_sha256'],
              'Unexpected runtime or wrong member scope in pure preflight')
        commands.append((item,argv))
    if not a.execute:print(json.dumps({'status':'ALL_SIX_ORT_INPUTS_VERIFIED_NO_MODEL_EXECUTION','model_calls':0,'native_runtime_imported':False,'markers_created':0}));return
    require(sha(a.plan)==a.plan_sha256 and sha(runner)==RUNNER_SHA and sha(contract)==plan['contract_sha256'],'Plan/runner/contract changed during preflight')
    with (base/'attempt.json').open('x',encoding='utf-8') as f:
        json.dump({'plan_sha256':a.plan_sha256,'pid':os.getpid(),'maximum_calls':900},f,indent=2);f.flush();os.fsync(f.fileno())
    state={'schema':'rampq.gate05.local_reference_queue_status/v1','status':'RUNNING','pid':os.getpid(),'plan_sha256':a.plan_sha256,'members':[],'model_calls_completed':0,'NMS_calls':0}
    save(base/'status.json',state);active=None
    def cleanup():
        if active is None:return
        try:
            if active.poll() is None:
                active.terminate()
                try:active.wait(timeout=10)
                except subprocess.TimeoutExpired:active.kill();active.wait(timeout=5)
            state['last_owned_child_pid']=active.pid;state['last_owned_child_returncode']=active.poll()
        except BaseException as error:state['owned_child_cleanup_error']=repr(error)
    atexit.register(cleanup)
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,interrupt)
    try:
        for item,argv in commands:
            require(sha(a.plan)==a.plan_sha256 and sha(runner)==RUNNER_SHA and sha(contract)==plan['contract_sha256'],'Frozen queue inputs drifted before member')
            row={'configuration':item['configuration'],'output':item['output'],'status':'RUNNING'};state['members'].append(row)
            with (base/(item['configuration']+'_console.log')).open('xb',buffering=0) as log:
                active=subprocess.Popen(argv+['--execute'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=active.pid;save(base/'status.json',state)
                active.wait(timeout=1800)
            require(active.returncode==0,'Complete ORT reference process failed; preserve partial outputs')
            out=Path(item['output']);rs=read(out/'run_state.json');rm=read(out/'raw_manifest.json')
            require(rs['status']==rm['status']=='ALL150_RAW_OUTPUTS_SERIALIZED_NO_SCORING' and rs['phase']=='COMPLETE' and
                    rs['model_calls_attempted']==rs['model_calls_completed']==rs['serialized_images']==len(rm['images'])==150,'Incomplete150reference')
            require(rs['all_outputs_finite'] is True and not rs['nonfinite_images'] and not rm['nonfinite_images'],'Nonfinite reference; stop queue and preserve full150')
            for lineage in (rs['lineage'],rm['lineage']):
                require(lineage['contract_sha256']==plan['contract_sha256'] and lineage['backend']=='ort' and
                        lineage['configuration']==item['configuration'] and lineage['runner_sha256']==RUNNER_SHA and
                        lineage['graph_sha256']==raw['configurations'][item['configuration']]['graph_sha256'] and
                        lineage['engine_sha256'] is None and lineage['input_manifest_sha256']==raw['input_manifest']['sha256'],
                        'Reference contract/configuration/source/input lineage changed')
            require([r['image_index'] for r in rm['images']]==list(range(150)) and
                    [r['image'] for r in rm['images']]==raw['input_manifest']['ordered_ids'] and
                    all(r['all_outputs_finite'] is True for r in rm['images']),'Reference order or finite inventory differs')
            for image in rm['images']:
                relative=Path(image['raw_file']);saved=out/relative
                require(not relative.is_absolute() and '..' not in relative.parts and out in saved.resolve().parents and
                        saved.is_file() and not saved.is_symlink() and saved.stat().st_size==image['raw_bytes'] and
                        sha(saved)==image['raw_sha256'],'Saved reference tensor file changed or escaped output')
            require(rs['NMS_run'] is False and rs['accuracy_pass_claim'] is False and rm['NMS_run'] is False and
                    rm['target_matching_run'] is False and rm['dataset_accuracy_evaluated'] is False,'Unexpected scoring/NMS')
            row.update(status='SUCCESS_COMPLETE_FULL_ORT150_NO_SCORING',model_calls=150,raw_manifest_sha256=sha(out/'raw_manifest.json'))
            state['model_calls_completed']+=150;save(base/'status.json',state);active=None
        require(state['model_calls_completed']==900 and sha(a.plan)==a.plan_sha256 and sha(runner)==RUNNER_SHA and
                sha(contract)==plan['contract_sha256'],'Final queue count or frozen identity differs')
        state['status']='ALL_SIX_FULL_ORT_REFERENCES_COMPLETE_NO_SCORING';save(base/'status.json',state)
    except BaseException as error:
        cleanup()
        state.update(status='STOPPED_PRESERVE_NO_RETRY',error=repr(error));save(base/'status.json',state);raise
    print(json.dumps(state,indent=2))

if __name__=='__main__':main()
