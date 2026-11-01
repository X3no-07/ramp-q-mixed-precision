"""Profile one frozen two-call matched smoke; invoked without arguments.

The immutable monitor supervises this wrapper. An independent authenticated
child guard supervises nsys's escaped private application process group.
No model build or retry is implemented. All failed reports are preserved.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HELPER_SHA='d1576ea834bd48ad78df3bf920b315726c1f0da8fe388eb7c3450dda8d93e0ca'
GROUP_SHA='4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4'

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,obj):
    with path.open('x',encoding='utf-8') as out:json.dump(obj,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())

def load(path):
    spec=importlib.util.spec_from_file_location('matched_profile_'+path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify_child_terminal(work,group_module):
    identity_path=work/'profile_runner_identity.json'
    if not identity_path.exists():
        return {'runner_identity_written':False,'GPU_import_acknowledgement_absent':True}
    identity=read(identity_path);runner=identity['runner'];guard_path=work/'profile_child_guard.json'
    require(sha(work/'jetson_matched_fixed_input_adapter.py')==identity['adapter_sha256'],'Adapter identity changed')
    require(runner['pid']==runner['pgid']==runner['sid'] and group_module.boot_id()==identity['boot_id'],'Runner/boot changed')
    group=group_module.OwnedGroup(runner['pid'],identity['boot_id'],Path(identity['adapter_path']))
    deadline=time.monotonic()+35
    while time.monotonic()<deadline:
        members=group.snapshot()
        guard=read(guard_path) if guard_path.exists() else None
        guard_process=group_module.process_info(guard['guard_pid']) if guard else None
        guard_live=guard_process is not None and guard_process['state'] not in ('Z','X','x')
        if not members and not guard_live and guard and guard['status'] in ('COMPLETE','COMPLETE_NONLIVE_MEMBERS_AWAIT_REAP'):
            require(guard['runner_identity_sha256']==sha(identity_path) and
                    guard['owned_pgid']==runner['pid'] and guard['leader_start_ticks']==runner['start_ticks'] and
                    guard['protocol_sha256']==identity['protocol_sha256'] and not guard['intervention_performed'],
                    'Child guard identity or intervention requires inspection')
            return {'runner_identity_written':True,'identity_sha256':sha(identity_path),'runner_pid':runner['pid'],
                    'runner_group_fully_empty':True,'child_guard_process_nonlive':True,'child_guard_status':guard['status'],
                    'guard_terminal_members_rechecked_not_assumed':True,'child_guard_intervention':False}
        time.sleep(.2)
    raise RuntimeError('Profile child/guard did not reach authenticated physically empty terminal state')

def main():
    work=Path(__file__).resolve().parent
    require(sha(work/'jetson_gate05_control_fixed_input.py')==HELPER_SHA and sha(work/'jetson_owned_group_guard.py')==GROUP_SHA,'Frozen helper changed')
    helper=load(work/'jetson_gate05_control_fixed_input.py');helper.verify_manifest(work)
    protocol=read(work/'protocol.json');spec=read(work/'execution_spec.json')
    require(protocol['profile_wrapper_sha256']==sha(Path(__file__)) and protocol['status']=='FROZEN_BEFORE_MATCHED_FIXED_INPUT_EXECUTION','Unpinned wrapper/protocol')
    require(spec['schema']=='rampq.gate05.matched_fixed_profile_execution/v1' and spec['protocol_sha256']==sha(work/'protocol.json') and
            spec['configuration']==protocol['configuration'] and spec['engine_executions_allowed']==2 and spec['native_builds']==0,'Execution scope differs')
    args=spec['adapter_arguments'];require(isinstance(args,list) and all(isinstance(x,str) for x in args),'Invalid argv')
    expected={'--work-root':str(work),'--protocol':str(work/'protocol.json'),'--protocol-sha256':sha(work/'protocol.json'),
              '--engine-sha256':protocol['engine_sha256'],'--audit':str(work/'precision_audit.json'),
              '--audit-sha256':protocol['precision_audit_sha256'],'--output':str(work/'fixed_outputs'),
              '--root':spec['member_root'],'--build':spec['build_root']}
    expected_args=[]
    for k,v in expected.items():expected_args.extend([k,v])
    if protocol['diagnostic_formats_only_allowed']:expected_args.append('--diagnostic-formats-only')
    require(args==expected_args,'Frozen exact adapter argument list differs')
    profile=work/'profile';profile.mkdir(exist_ok=False)
    prefix=profile/'fixed_input'
    argv=['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--output',str(prefix),
          '/usr/bin/setsid','--wait',sys.executable,'-u',str(work/'jetson_matched_fixed_input_adapter.py')]+args
    group_module=load(work/'jetson_owned_group_guard.py');started=time.monotonic()
    result=None;failure=None
    try:
        result=subprocess.run(argv,cwd=str(work),stdin=subprocess.DEVNULL,timeout=240)
    except BaseException as error:
        failure=repr(error)
    finally:
        save(work/'profile_exit.json',{'argv':argv,'returncode':None if result is None else result.returncode,
             'error':failure,'elapsed_seconds':time.monotonic()-started,'engine_executions_allowed':2,'new_builds':0})
        # No export or collection until the escaped application and its own
        # guard are confirmed terminal. The existing monitor/child guard handle
        # exceptional shutdown; this wrapper never signals guessed PIDs.
        terminal=verify_child_terminal(work,group_module);save(work/'profile_child_terminal_verification.json',terminal)
    require(failure is None and result is not None and result.returncode==0,'nsys/runner failed; preserve evidence, no retry')
    require(terminal.get('runner_group_fully_empty') is True,'No authenticated application execution receipt')
    completed=read(work/'profile_runner_completion.json')
    require(completed['protocol_sha256']==spec['protocol_sha256'] and completed['two_engine_calls_completed'] and
            completed['all_attempted_calls_synchronized'],'Exactly two complete synchronized calls required')
    reports=list(profile.glob('*.nsys-rep'))+list(profile.glob('*.qdrep'))
    require(len(reports)==1,'Expected one native profile report')
    export=subprocess.run(['/usr/local/bin/nsys','export','--type','sqlite','--output',str(prefix)+'.sqlite',str(reports[0])],
                          capture_output=True,text=True,timeout=35)
    save(work/'profile_export.json',{'returncode':export.returncode,'stdout':export.stdout,'stderr':export.stderr})
    require(export.returncode==0 and Path(str(prefix)+'.sqlite').is_file(),'CPU-only profile export failed; do not rerun inference')
    print('MATCHED_TWO_EXECUTIONS_PROFILE_AND_ESCAPED_CHILD_TERMINAL_VERIFIED',flush=True)

if __name__=='__main__':main()
