"""Authenticate and guard the private nsys application process group.

Stdlib-only; never executes a model. Uses the immutable OwnedGroup signalling
implementation. Ack is durable before GPU import; monitor terminal/stale/budget
conditions stop only the authenticated escaped runner session. Normal child
exit is observed without requiring that child to wait for its own guard.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode=True
OWNED_GUARD_SHA='4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4'

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def interrupted(_sig,_frame):raise KeyboardInterrupt('Profile child guard interrupted; cleanup authenticated runner')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True);p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--identity-sha256',required=True);args=p.parse_args()
    work=args.work_root.resolve();protocol_path=args.protocol.resolve();identity_path=args.identity.resolve()
    require(protocol_path.parent==identity_path.parent==work and identity_path.name=='profile_runner_identity.json',
            'Guard inputs must be directly inside work root')
    require(sha(protocol_path)==args.protocol_sha256 and sha(identity_path)==args.identity_sha256,'Guard lineage changed')
    protocol,identity=read(protocol_path),read(identity_path)
    require(protocol['schema']=='rampq.gate05.matched_control_fixed_input_protocol/v1' and
            protocol['status']=='FROZEN_BEFORE_MATCHED_FIXED_INPUT_EXECUTION' and
            protocol['child_guard_sha256']==sha(Path(__file__).resolve()) and protocol['child_guard_required_before_GPU'] is True,
            'Unpinned/unauthorized child guard')
    require(identity['schema']=='rampq.gate05.profile_runner_identity/v1' and identity['work_root']==str(work) and
            identity['protocol_sha256']==args.protocol_sha256 and identity['adapter_sha256']==protocol['adapter_sha256'] and
            identity['engine_executions_allowed']==2 and identity['written_before_GPU_import'] is True,'Runner receipt differs')
    adapter=Path(identity['adapter_path']);require(adapter.parent==work and sha(adapter)==protocol['adapter_sha256'],'Adapter source changed')
    helper=work/'jetson_owned_group_guard.py';require(sha(helper)==OWNED_GUARD_SHA,'Immutable group guard changed')
    spec=importlib.util.spec_from_file_location('immutable_profile_owned_group',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(os.getpid()==os.getpgrp()==os.getsid(0),'Child guard needs independent private session')
    require(module.boot_id()==identity['boot_id']==protocol['expected_boot_id'],'Boot changed')
    runner=identity['runner'];require(runner['pid']==runner['pgid']==runner['sid'] and runner['pid']>1,'Runner is not private PID=PGID=SID')
    live=module.process_info(runner['pid'])
    require(live is not None and all(live[k]==runner[k] for k in ('pid','pgid','sid','start_ticks','argv','ppid')),
            'Runner process identity changed before acknowledgement')
    group=module.OwnedGroup(runner['pid'],identity['boot_id'],adapter)
    require(group.leader_start_ticks==runner['start_ticks'],'Guard leader identity differs')
    status_path=work/'monitor/status.json';status,stamp=module.read_status(status_path)
    frozen=identity['monitor_identity']
    require(frozen['root']==str(work) and frozen['max_seconds']==protocol['supervision']['max_seconds']==300 and
            frozen['build_script']==str(work/protocol['monitor_build_script']),'Monitor path or native budget differs')
    module.verify_status(status,frozen,identity['boot_id'])
    monitor=module.process_info(frozen['monitor_pid']);expected_monitor=identity['monitor_process_identity']
    require(monitor is not None and monitor['start_ticks']==expected_monitor['start_ticks'] and monitor['argv']==expected_monitor['argv'],
            'Original monitor process changed')
    output=work/'profile_child_guard.json';require(not output.exists(),'Refuse guard receipt overwrite')
    record={'schema':'rampq.gate05.profile_child_guard/v1','status':'WATCHING','guard_pid':os.getpid(),'guard_pgid':os.getpgrp(),
            'guard_sid':os.getsid(0),'boot_id':identity['boot_id'],'work_root':str(work),'owned_pgid':group.group,
            'leader_start_ticks':group.leader_start_ticks,'runner_identity_sha256':args.identity_sha256,
            'protocol_sha256':args.protocol_sha256,'adapter_sha256':protocol['adapter_sha256'],'guard_sha256':sha(Path(__file__)),
            'monitor_identity':frozen,'termination_actions':[],'intervention_performed':False,'initial_group_members':group.initial_members,
            'stale_seconds':15,'budget_grace_seconds':20,'poll_seconds':.5,'no_model_or_GPU_import':True}
    # Claim exclusively, then atomically publish a complete ack. A visible '{}'
    # placeholder would make the adapter race into a false startup failure.
    with (work/'profile_child_guard.claim').open('x') as f:
        json.dump({'guard_pid':os.getpid(),'runner_identity_sha256':args.identity_sha256},f)
    save=lambda:module.atomic_save(output,record)
    # Initial durable WATCHING is the ack the adapter needs before GPU import.
    save();signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    now=time.monotonic();initial_age=max(0,time.time()-stamp['mtime_wall_seconds'])
    last_update=now-initial_age;last_stamp=stamp
    deadline=now+max(0,300+20-stamp['elapsed_seconds']-initial_age)
    zombie_since=None
    try:
        while True:
            require(sha(identity_path)==args.identity_sha256 and sha(protocol_path)==args.protocol_sha256 and
                    sha(adapter)==protocol['adapter_sha256'],'Authenticated profile inputs changed')
            observed,observed_stamp=module.read_status(status_path)
            module.verify_status(observed,frozen,identity['boot_id'])
            require(observed_stamp['elapsed_seconds']>=last_stamp['elapsed_seconds'],'Monitor clock moved backward')
            if (observed_stamp['mtime_ns'],observed_stamp['sha256'])!=(last_stamp['mtime_ns'],last_stamp['sha256']):last_update=time.monotonic()
            last_stamp=observed_stamp
            members=group.snapshot();now=time.monotonic()
            record.update(last_monitor_status=observed,last_group_members=members,status_update_age_seconds=now-last_update)
            if not members:
                record.update(status='COMPLETE',cleanup_trigger='AUTHENTICATED_RUNNER_GROUP_GONE',owned_group_fully_empty=True,
                              owned_group_non_zombie_empty=True,final_group_members=[],intervention_performed=False);save();return 0
            if not module.live_members(members):
                if zombie_since is None:zombie_since=now
                if now-zombie_since>=10:
                    record.update(status='COMPLETE_NONLIVE_MEMBERS_AWAIT_REAP',cleanup_trigger='RUNNER_NONLIVE_PENDING_PARENT_REAP',
                                  owned_group_fully_empty=False,owned_group_non_zombie_empty=True,final_group_members=members);save();return 0
            reason=None
            if observed['status'] in module.TERMINAL_STATES:reason='ORIGINAL_MONITOR_TERMINAL'
            elif now-last_update>15:reason='ORIGINAL_MONITOR_STALE_OVER_15_SECONDS'
            elif now>=deadline:reason='ORIGINAL_MONITOR_BUDGET_PLUS_20_SECONDS'
            else:
                current_monitor=module.process_info(frozen['monitor_pid'])
                if current_monitor is None or current_monitor['start_ticks']!=expected_monitor['start_ticks'] or current_monitor['argv']!=expected_monitor['argv']:
                    reason='ORIGINAL_MONITOR_PROCESS_GONE_OR_IDENTITY_CHANGED'
                leader=module.process_info(runner['pid'])
                if leader is not None and leader['state'] not in ('Z','X') and leader['ppid']!=runner['ppid']:
                    reason='PROFILE_RUNNER_PARENT_GONE_OR_CHANGED'
            if reason:
                record['cleanup_trigger']=reason;save()
                remaining=module.terminate_owned(group,record,save)
                record.update(status='COMPLETE' if not module.live_members(remaining) else 'FAILED_RUNNER_MEMBERS_REMAIN',
                              final_group_members=remaining,owned_group_fully_empty=not remaining,
                              owned_group_non_zombie_empty=not module.live_members(remaining),
                              intervention_performed=any(r['dispatch']=='SENT' for r in record['termination_actions']))
                save();return 0 if not remaining else 2
            save();time.sleep(.5)
    except BaseException as error:
        record.update(status='GUARD_EXCEPTION_AUTHENTICATED_CLEANUP',error=repr(error));save()
        try:
            remaining=module.terminate_owned(group,record,save)
            record.update(final_group_members=remaining,owned_group_fully_empty=not remaining,
                          owned_group_non_zombie_empty=not module.live_members(remaining),
                          intervention_performed=any(r['dispatch']=='SENT' for r in record['termination_actions']))
        except BaseException as cleanup_error:record['cleanup_error']=repr(cleanup_error)
        record['status']='FAILED_REQUIRES_INSPECTION';save();raise

if __name__=='__main__':sys.exit(main())
