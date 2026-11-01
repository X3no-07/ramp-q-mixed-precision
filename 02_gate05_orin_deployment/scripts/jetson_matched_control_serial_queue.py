"""One bounded, serial five-member build queue; standard library only, Python3.8.

The plan pins every input before GPU work. Each immutable member builder creates
one attempt marker. No inference, candidate retry, precision pass or scheduler.
"""
import argparse
import atexit
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ORDER = ['fp16_control', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1']
NATIVE_PROTOCOL = '12ce0b79c1e765f3293cc529f01023e6a58671e8ad688f0884fcf216931a3459'
BUILDER = '4d2ee7d283abf6cd75b6d76e58870f79f11392fa06eaef37936fab6d7bafca34'
HELPERS = {'jetson_monitored_build.py':'71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1',
           'jetson_owned_group_guard.py':'4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4',
           'jetson_collect_isolated_numeric_probe.py':'af9b7f823e4f889813fb88b6e8a529a6b1a6423b0c7ead0d5704513b517775eb'}
BAD_KERNEL = ('controller error detected', 'HC died;', 'unregister_netdevice: waiting', 'Out of memory:', 'oom-kill:', 'Kernel panic')
MONITOR_EXIT_GRACE_SECONDS = 90

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, obj):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as out:
        json.dump(obj, out, indent=2, allow_nan=False)
        out.write('\n'); out.flush(); os.fsync(out.fileno())
    tmp.replace(path)

def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def verify_member(item, base):
    root = Path(item['root'])
    require(root.parent == base and root.resolve() == root and not root.is_symlink(), 'Unexpected member directory')
    require(sha(root/'manifest.json') == item['manifest_sha256'] and sha(root/'spec.json') == item['spec_sha256'], 'Member manifest/spec changed')
    manifest = read(root/'manifest.json')
    require(manifest['configuration'] == item['configuration'], 'Member identity differs')
    for name, identity in manifest['files'].items():
        path = root/name
        require(root in path.resolve().parents and not path.is_symlink() and path.is_file(), 'Unsafe manifest entry')
        require(path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'], 'Member bytes changed: '+name)
    require(sha(root/'jetson_build_matched_control.py') == BUILDER and sha(root/'native_protocol.json') == NATIVE_PROTOCOL, 'Builder/protocol differs')
    for name, value in HELPERS.items():
        require(sha(root/name) == value, 'Helper differs: '+name)
    require(not (root/'monitor').exists() and not (root/'latest_build.json').exists() and
            not (root/'native_build_attempt.json').exists() and not (root/'results').exists(), 'Member already attempted')
    spec = read(root/'spec.json')
    require(spec['configuration'] == item['configuration'] and spec['maximum_native_builds'] == 1 and
            spec['max_build_seconds'] == 7200 and spec['native_inference_executions'] == 0, 'Member identity or budget differs')
    result = subprocess.run([sys.executable, str(root/'jetson_build_matched_control.py'), '--check-input-only'], cwd=str(root), capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, 'Pure input check failed: '+result.stdout+result.stderr)
    check = json.loads(result.stdout)
    require(check['native_runtime_imported'] is False and check['native_inference_run'] is False and
            check['native_build_run'] is False and check['new_CPU_execution_runs'] == 0,
            'Input check imported native runtime or executed an engine')
    return {'configuration':item['configuration'], 'root':str(root), 'input_check_stdout':result.stdout}

def wait_path(path, child, seconds):
    deadline = time.monotonic()+seconds
    while not path.exists() and child.poll() is None and time.monotonic()<deadline:
        time.sleep(.2)
    require(path.exists(), 'Missing child receipt: '+str(path))
    return read(path)

def interrupt(_signum, _frame):
    raise KeyboardInterrupt('Queue interrupted; clean up authenticated current group')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--check-input-only',action='store_true')
    args=p.parse_args(); plan=read(args.plan)
    require(sha(args.plan)==args.plan_sha256, 'Plan identity differs')
    require(plan['queue_script_sha256']==sha(Path(__file__)), 'Queue source identity differs from plan')
    require(plan['schema']=='rampq.gate05.matched_control_serial_queue_plan/v1' and plan['configuration_order']==ORDER, 'Queue order differs')
    require(plan['native_protocol_sha256']==NATIVE_PROTOCOL and plan['native_build_attempts']==5 and plan['native_inference_executions']==0, 'Queue scope differs')
    require([x['configuration'] for x in plan['members']]==ORDER, 'Members differ')
    base=Path(plan['base']); require(base.parent==Path('/home/nvidia/rampq') and base.resolve()==base, 'Unsafe queue base')
    require(boot()==plan['expected_boot_id'], 'Board boot changed')
    receipts=[verify_member(item,base) for item in plan['members']]
    if args.check_input_only:
        print(json.dumps({'status':'ALL_FIVE_PURE_INPUT_CHECKS_OK','native_runtime_imported':False,'members':receipts}));return
    marker=base/'queue_attempt.json'
    with marker.open('x') as out:
        json.dump({'pid':os.getpid(),'plan_sha256':args.plan_sha256,'boot_id':boot(),'started':datetime.datetime.now().astimezone().isoformat()},out,indent=2)
    # No second native workload may already be active when queue starts.
    conflicts=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
        try: words=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if any(Path(w.decode(errors='replace')).name.startswith(('jetson_build_', 'jetson_gate05_control_fixed_input', 'trtexec')) for w in words if w): conflicts.append(int(proc.name))
    require(not conflicts, 'Other native job exists: '+str(conflicts))
    state={'schema':'rampq.gate05.matched_control_serial_queue_status/v1','pid':os.getpid(),'plan_sha256':args.plan_sha256,'boot_id':boot(),'status':'RUNNING','members':[],'native_inference_executions':0}
    status_path=base/'queue_status.json';save(status_path,state)
    active={'monitor':None,'guard':None,'owned':None,'module':None,'root':None,'cleaned':False}
    def cleanup():
        if active['root'] is None or active['cleaned']: return
        evidence={'status':'QUEUE_EXCEPTION_CLEANUP','termination_actions':[]}
        dest=active['root']/'queue_exception_cleanup.json'
        persist=lambda:save(dest,evidence)
        # A startup failure can occur after monitor spawn and before the first
        # status/OwnedGroup. Recover only an authenticated current child/group;
        # never skip cleanup merely because OwnedGroup was not assigned yet.
        try:
            module=active['module'];monitor=active['monitor'];owned=active['owned']
            if owned is None and module is not None and monitor is not None and boot()==state['boot_id']:
                status_file=active['root']/'monitor/status.json'
                if status_file.is_file():
                    observed=read(status_file)
                    require(observed['monitor_pid']==monitor.pid and observed['root']==str(active['root']) and
                            observed['boot_id']==state['boot_id'] and observed['build_script']==str(active['root']/'jetson_build_matched_control.py'),
                            'Cannot authenticate early monitor status for cleanup')
                    owned=module.OwnedGroup(observed['build_pid'],state['boot_id'],active['root']/'jetson_build_matched_control.py')
                else:
                    candidates=[]
                    for path in Path('/proc').iterdir():
                        if not path.name.isdigit(): continue
                        child=module.process_info(int(path.name))
                        if child is not None and child['ppid']==monitor.pid and child['pid']==child['pgid']==child['sid'] and \
                                str(active['root']/'jetson_build_matched_control.py') in child['argv']:
                            candidates.append(child)
                    require(len(candidates)<=1,'Ambiguous early builder identity; refuse guessed group signal')
                    if candidates:
                        evidence['early_builder_identity']=candidates[0]
                        owned=module.OwnedGroup(candidates[0]['pid'],state['boot_id'],active['root']/'jetson_build_matched_control.py')
                active['owned']=owned
            if owned is not None:
                remaining=module.terminate_owned(owned,evidence,persist)
                evidence['owned_group_final_members']=remaining
                evidence['owned_group_fully_empty']=not remaining
        except BaseException as exc:
            evidence['owned_cleanup_error']=repr(exc)
        # Popen owns these exact child identities. Signal the monitor even if
        # no child status was ever published, so its immutable finally can stop
        # any builder/collectors already created. Each wait remains bounded.
        for label,grace in (('monitor',45),('guard',12)):
            proc=active[label]
            if proc is None: continue
            try:
                if proc.poll() is None:
                    if label=='monitor': proc.send_signal(signal.SIGINT)
                    try:proc.wait(timeout=grace)
                    except subprocess.TimeoutExpired:
                        evidence[label+'_grace_timeout']=True
                        proc.send_signal(signal.SIGINT)
                        try:proc.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            proc.kill();proc.wait(timeout=5)
                evidence[label+'_returncode']=proc.poll()
            except BaseException as exc:
                evidence[label+'_cleanup_error']=repr(exc)
        if active['owned'] is not None:
            try:
                remaining=active['owned'].snapshot()
                evidence.update(owned_group_final_members=remaining,owned_group_fully_empty=not remaining)
            except BaseException as exc:evidence['final_group_snapshot_error']=repr(exc)
        active['cleaned']=True
        persist()
    atexit.register(cleanup)
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    try:
        for item, receipt in zip(plan['members'],receipts):
            require(boot()==state['boot_id'], 'Boot changed between members')
            root=Path(item['root']);verify_member(item,base)
            save(root/'queue_input_receipt.json',dict(receipt,plan_sha256=args.plan_sha256,queue_script_sha256=sha(Path(__file__))))
            state['current_configuration']=item['configuration'];save(status_path,state)
            modspec=importlib.util.spec_from_file_location('matched_guard',root/'jetson_owned_group_guard.py');module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
            active.update(monitor=None,guard=None,owned=None,module=module,root=root,cleaned=False)
            console=(root/'monitor_console.log').open('xb',buffering=0)
            member_started=time.monotonic()
            active['monitor']=subprocess.Popen([sys.executable,'-u',str(root/'jetson_monitored_build.py'),'--root',str(root),'--run',str(root/'monitor'),'--max-seconds','7200','--build-script','jetson_build_matched_control.py'],cwd=str(root),stdin=subprocess.DEVNULL,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
            monitor=active['monitor']
            ms=wait_path(root/'monitor/status.json',monitor,12)
            require(ms['monitor_pid']==monitor.pid and ms['root']==str(root) and ms['max_seconds']==7200 and
                    ms['boot_id']==state['boot_id'] and ms['build_script']==str(root/'jetson_build_matched_control.py'), 'Monitor identity differs')
            owned=module.OwnedGroup(ms['build_pid'],state['boot_id'],root/'jetson_build_matched_control.py');active.update(owned=owned,module=module)
            guard_log=(root/'supplemental_guard_console.log').open('xb',buffering=0)
            active['guard']=subprocess.Popen([sys.executable,'-u',str(root/'jetson_owned_group_guard.py'),'--root',str(root),'--expected-monitor-pid',str(monitor.pid)],cwd=str(root),stdin=subprocess.DEVNULL,stdout=guard_log,stderr=subprocess.STDOUT,start_new_session=True)
            guard=active['guard']
            gs=wait_path(root/'supplemental_group_guard.json',guard,8)
            require(gs['guard_pid']==guard.pid and gs['owned_pgid']==ms['build_pid'] and gs['boot_id']==state['boot_id'] and gs['root']==str(root), 'Guard identity differs')
            require(all(gs['monitor_identity'][key]==ms[key] for key in ('root','monitor_pid','build_pid','max_seconds','build_script')),
                    'Guard monitor/source identity differs')
            require(gs['status']=='WATCHING' or (gs['status']=='COMPLETE' and gs['owned_group_fully_empty'] and not gs['intervention_performed']),
                    'Guard startup requires inspection')
            row={'configuration':item['configuration'],'root':str(root),'monitor_pid':monitor.pid,'private_group':ms['build_pid'],'guard_pid':guard.pid,'status':'RUNNING'}
            state['members'].append(row);save(status_path,state)
            while monitor.poll() is None:
                require(boot()==state['boot_id'],'Boot changed')
                require(time.monotonic()-member_started<7200+MONITOR_EXIT_GRACE_SECONDS,
                        'Monitor process exceeded native budget plus bounded transport-exit grace')
                observed_guard=read(root/'supplemental_group_guard.json')
                require(observed_guard['status'] in ('WATCHING','COMPLETE') and not observed_guard.get('intervention_performed',False),
                        'Guard failed or performed protective intervention')
                require(guard.poll() is None or (observed_guard['status']=='COMPLETE' and observed_guard['owned_group_fully_empty']), 'Guard failed')
                latest=read(root/'monitor/status.json')
                require(all(latest[key]==ms[key] for key in ('root','monitor_pid','build_pid','max_seconds','build_script','boot_id')),
                        'Monitor identity changed during queue member')
                row['latest_monitor']=latest;save(status_path,state);time.sleep(2)
            guard.wait(timeout=45);console.close();guard_log.close()
            terminal=read(root/'monitor/status.json');gs=read(root/'supplemental_group_guard.json')
            require(boot()==state['boot_id'] and gs['status']=='COMPLETE' and gs['owned_group_fully_empty'] and not gs['intervention_performed'],'Guard/boot cleanup requires inspection')
            require(not owned.snapshot(), 'Private group still has processes')
            require(terminal['status'] in ('BUILD_SCRIPT_OK','BUILD_SCRIPT_FAILED') and terminal.get('reason') is None,'Monitor protective stop requires inspection')
            require(type(terminal.get('build_returncode')) is int and terminal['build_returncode']>=0 and
                    ((terminal['status']=='BUILD_SCRIPT_OK')==(terminal['build_returncode']==0)),
                    'Signal-terminated or inconsistent builder outcome requires inspection')
            require((root/'native_build_attempt.json').is_file() and (root/'latest_build.json').is_file(),
                    'Entry failed before an authenticated native attempt; stop queue for inspection')
            latest_build=read(root/'latest_build.json')
            require(latest_build.get('error_type') not in ('KeyboardInterrupt','MemoryError','TimeoutError'),
                    'Interrupted or resource-failed builder requires inspection')
            require(not any(x in (root/'monitor/kernel.log').read_text(errors='replace') for x in BAD_KERNEL),'Kernel fault requires inspection')
            archive=Path('/home/nvidia/rampq')/('gate05_matched_'+item['configuration']+'_evidence_20261021.zip')
            result=subprocess.run([sys.executable,str(root/'jetson_collect_isolated_numeric_probe.py'),'--root',str(root),'--archive',str(archive)],capture_output=True,text=True,timeout=180)
            require(result.returncode==0,'Collection failed: '+result.stdout+result.stderr)
            row.update(status=terminal['status'],terminal_monitor=terminal,archive=json.loads(result.stdout),guard_complete_no_intervention=True)
            row.pop('latest_monitor',None);save(status_path,state)
            active.update(monitor=None,guard=None,owned=None,module=None,root=None,cleaned=False)
        state['status']='ALL_FIVE_BUILD_ATTEMPTS_TERMINAL_AND_COLLECTED_NO_INFERENCE';save(status_path,state)
    except BaseException as exc:
        cleanup()
        state.update(status='STOPPED_REQUIRES_INSPECTION_NO_AUTO_RETRY',error=repr(exc));save(status_path,state);raise
    print(json.dumps(state),flush=True)

if __name__=='__main__':
    main()
