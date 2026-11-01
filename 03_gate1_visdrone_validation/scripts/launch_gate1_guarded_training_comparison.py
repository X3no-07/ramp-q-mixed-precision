"""Exclusive outer launcher -> worker/audit supervisor -> immutable final delivery."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/gate1/20261028_guarded_training_comparison_v1'


def now():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def identity(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest}


def bounded_wait(child,seconds,stage):
    try:return child.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        # The live Popen handle must still refer to our child before terminating
        # its known descendants (venv redirector plus any AP process).
        if child.poll() is None:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,text=True)
            else:child.kill()
        child.wait(timeout=30)
        write(OUT/(stage+'_timeout_receipt.json'),{'at_utc':now(),'pid':child.pid,'seconds':seconds,'exit_code':child.returncode})
        raise TimeoutError(stage+' frozen timeout; no retry')


def supervise():
    if (OUT/'attempt_01').exists() or (OUT/'worker_launcher_receipt.json').exists():raise ValueError('Duplicate worker prohibited')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    c=read(OUT/'contract.json')
    with (OUT/'console_stdout.txt').open('x',encoding='utf-8') as stdout,(OUT/'console_stderr.txt').open('x',encoding='utf-8') as stderr:
        argv=[sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/gate1_guarded_training_comparison.py'),'run']
        child=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=flags)
        write(OUT/'worker_launcher_receipt.json',{'at_utc':now(),'launcher_pid':os.getpid(),'launched_pid':child.pid,'argv':argv,'automatic_retry':False})
        code=bounded_wait(child,c['timeout_seconds']+120,'worker')
    write(OUT/'worker_terminal_receipt.json',{'at_utc':now(),'pid':child.pid,'exit_code':code,'exit_code_directly_observed':True,
          'completion_receipt_exists':(OUT/'attempt_01/completion_receipt.json').exists(),'failure_receipt_exists':(OUT/'attempt_01/failure_receipt.json').exists()})
    audit_code=None
    if code==0:
        with (OUT/'audit_console.txt').open('x',encoding='utf-8') as log:
            argv=[sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/audit_gate1_guarded_training_comparison.py')]
            auditor=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            write(OUT/'audit_launcher_receipt.json',{'at_utc':now(),'pid':auditor.pid,'parent_pid':os.getpid(),'argv':argv})
            audit_code=bounded_wait(auditor,7200,'audit')
        write(OUT/'audit_terminal_receipt.json',{'at_utc':now(),'pid':auditor.pid,'exit_code':audit_code,'exit_code_directly_observed':True})
    write(OUT/'supervisor_stage_receipt.json',{'at_utc':now(),'supervisor_pid':os.getpid(),'worker_exit_code':code,'audit_exit_code':audit_code})
    return code if code else (audit_code or 0)


def finalize():
    terminal=read(OUT/'supervisor_terminal_receipt.json')
    if terminal['exit_code']!=0 or not terminal['exit_code_directly_observed']:raise ValueError('Supervisor exit0 not observed')
    for name in ('worker_terminal_receipt.json','audit_terminal_receipt.json'):
        if read(OUT/name)['exit_code']!=0:raise ValueError('Stage exit0 absent')
    audit=read(OUT/'independent_saved_audit.json')
    if audit['full_raw_NMS_replays']!=1242 or audit['gradient_guard_windows_KKT_checked']!=1704:raise ValueError('Incomplete audit')
    for item in read(OUT/'attempt_01/manifest.json')['files']:
        if identity(item['path'])!=item:raise ValueError('Post-audit output changed')
    reports=[ROOT/'docs/88_gate1_guarded_three_arm_comparison_launch.md',ROOT/'docs/89_gate1_guarded_three_arm_continuation_results.md']
    excluded={'delivery_manifest.json','delivery_validation.json','outer_launcher_console.txt','outer_launcher_stderr.txt','live_status.json','events.jsonl'}
    members=[identity(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in excluded]
    members.extend(identity(p) for p in reports)
    write(OUT/'delivery_manifest.json',{'files':members,'excluded_mutable_runtime_files':sorted(excluded)})
    if not all(identity(x['path'])==x for x in members):raise ValueError('Final delivery mismatch')
    write(OUT/'delivery_validation.json',{'status':'PASS_COMPLETE_GUARDED_THREE_ARM_DELIVERY','at_utc':now(),'members':len(members),
          'manifest':identity(OUT/'delivery_manifest.json'),'supervisor_terminal':identity(OUT/'supervisor_terminal_receipt.json'),
          'audit':identity(OUT/'independent_saved_audit.json'),'report':identity(reports[1])})
    previous=read(ROOT/'artifacts/gate1/latest_execution_state.json')
    state=read(ROOT/previous['snapshot'])
    if state.get('active_contract',{}).get('sha256')!=identity(OUT/'contract.json')['sha256']:
        raise ValueError('Latest stage changed; preserve delivery without overwriting pointer')
    number=53
    while (ROOT/f'artifacts/gate1/current_execution_state_v{number}.json').exists():number+=1
    path=ROOT/f'artifacts/gate1/current_execution_state_v{number}.json'
    decision=read(OUT/'machine_decision.json')
    write(path,{'schema':f'rampq.gate1.execution_state/v{number}','status':'COMPLETE_GUARDED_THREE_ARM_AUDITED_TERMINAL_VERIFIED',
          'at_utc':now(),'previous_pointer':previous,'active_contract':identity(OUT/'contract.json'),
          'report':identity(reports[1]),'decision':decision,'delivery_validation':identity(OUT/'delivery_validation.json'),
          'supervisor_exit0_directly_observed':True,'new_candidates_exhausted':True,'scientific_acceptance':False,
          'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
          'next_action':'Report this fixed result, preserve all arms, delete finite monitor. No automatic candidate/epoch/threshold/seed extension.'})
    latest=ROOT/'artifacts/gate1/latest_execution_state.json';tmp=latest.with_name('latest_execution_state.guarded.tmp')
    write(tmp,{'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':str(path.relative_to(ROOT)).replace('\\','/'),
          'snapshot_sha256':identity(path)['sha256'],'delivery_validation':identity(OUT/'delivery_validation.json'),
          'note':'Guarded three-arm continuation completed and terminal/audit verified; read report89 and retain the fixed joint decision. No automatic extension.'})
    if read(latest)!=previous:raise ValueError('Concurrent latest change; preserve snapshot')
    tmp.replace(latest)
    print(json.dumps({'status':'COMPLETE_TERMINAL_AND_DELIVERY','both_rules':decision['meets_both_development_joint_rules']},ensure_ascii=True),flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='supervise':return supervise()
    claim=OUT/'exclusive_launcher_claim.json'
    write(claim,{'pid':os.getpid(),'at_utc':now(),'contract':identity(OUT/'contract.json'),'automatic_retry':False})
    if (OUT/'attempt_01').exists():raise ValueError('Existing attempt; do not restart')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (OUT/'supervisor_console.txt').open('x',encoding='utf-8') as stdout,(OUT/'supervisor_stderr.txt').open('x',encoding='utf-8') as stderr:
        argv=[sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),'supervise']
        child=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=flags)
        write(OUT/'supervisor_launcher_receipt.json',{'at_utc':now(),'pid':child.pid,'parent_pid':os.getpid(),'argv':argv})
        code=bounded_wait(child,read(OUT/'contract.json')['timeout_seconds']+7500,'supervisor')
    write(OUT/'supervisor_terminal_receipt.json',{'at_utc':now(),'pid':child.pid,'exit_code':code,'exit_code_directly_observed':True,'outer_pid':os.getpid()})
    if code==0:finalize()
    return code


if __name__=='__main__':
    try:result=main()
    except BaseException as e:
        name='supervisor_failure_receipt.json' if len(sys.argv)>1 else 'outer_failure_receipt.json'
        if not (OUT/name).exists():write(OUT/name,{'at_utc':now(),'error':repr(e),'traceback':traceback.format_exc()})
        raise
    raise SystemExit(result)
