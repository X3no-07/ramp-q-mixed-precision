"""No-argument entry for one frozen150-call raw-output job; no GPU import here."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    root=Path(__file__).resolve().parent
    spec=read(root/'execution_spec.json')
    if spec['schema']!='rampq.gate05.raw_work_execution/v1':raise RuntimeError('Wrong raw execution spec')
    if spec['wrapper_sha256']!=sha(Path(__file__)):raise RuntimeError('Unpinned wrapper')
    manifest=read(root/'manifest.json')
    for name,expected in manifest['files'].items():
        path=root/name
        if path.resolve().parent!=root and root not in path.resolve().parents:raise RuntimeError('Manifest path escapes work')
        if path.is_symlink() or not path.is_file() or sha(path)!=expected['sha256'] or path.stat().st_size!=expected['bytes']:raise RuntimeError('Work input changed: '+name)
    protocol=read(root/'raw_contract.json');runner=root/'run_gate05_frozen_raw_dataset_v2.py'
    if sha(root/'raw_contract.json')!=spec['contract_sha256'] or sha(runner)!=protocol['runner_sha256']:raise RuntimeError('Wrong frozen contract or runner')
    cfg=spec['configuration'];backend=spec['backend'];bc=protocol['configurations'][cfg][backend]
    if bc['calls']!=150 or spec['max_seconds']!=1800 or spec['extra_model_calls']!=0:raise RuntimeError('Frozen budget differs')
    arguments=['--contract',str(root/'raw_contract.json'),'--contract-sha256',spec['contract_sha256'],
               '--configuration',cfg,'--backend',backend,'--inputs',spec['inputs_root'],'--graph',spec['graph_file'],
               '--state-root',bc['attempt_state_root'],'--output',str(root/'raw_outputs'),'--evidence-root',str(root)]
    if backend=='trt':arguments+=['--engine',spec['engine_file']]
    # Input-only validation never imports a model runtime. One execution child
    # inherits the monitored wrapper group; there is no profiler/setsid escape.
    check=subprocess.run([sys.executable,'-B',str(runner)]+arguments+['--check-input-only'],capture_output=True,text=True,timeout=120)
    with (root/'pure_input_check.json').open('x',encoding='utf-8') as out:json.dump({'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr},out,indent=2)
    if check.returncode!=0:raise RuntimeError('Frozen raw input check failed: '+check.stdout+check.stderr)
    result=subprocess.run([sys.executable,'-u','-B',str(runner)]+arguments+['--execute'])
    with (root/'raw_runner_exit.json').open('x',encoding='utf-8') as out:json.dump({'returncode':result.returncode,'maximum_model_calls':150,'new_builds':0,'NMS_calls':0},out,indent=2)
    if result.returncode:raise RuntimeError('Raw runner failed; preserve once marker and partial outputs')
    state=read(root/'raw_outputs/run_state.json')
    if state['model_calls_attempted']!=150 or state['model_calls_completed']!=150 or state['serialized_images']!=150:raise RuntimeError('Incomplete150-call output')
    print('FROZEN150_RAW_CALLS_COMPLETE_NO_SCORING',flush=True)

if __name__=='__main__':main()
