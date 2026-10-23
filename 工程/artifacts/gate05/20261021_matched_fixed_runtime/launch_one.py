"""One-shot transport launcher for an already sealed matched fixed work.

Run by stdin on the board. No user credentials are stored. Components enforce
the frozen two-call budget; every launch and cleanup identity is retained.
"""
import atexit
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import signal
import subprocess
import sys
import time
import zipfile

cfg,archive_sha,manifest_sha,review_sha=sys.argv[1:]
assert cfg in ('fp16_control','r_only_G1','r_only_G1_G5','r_only_G1_G2','q_only_G1')
base=Path('/home/nvidia/rampq/gate05_matched_fixed_stage_20261021')
root=base/('gate05_matched_fixed_'+cfg+'_20261021');archive=base/(cfg+'.zip')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
assert not root.exists() and sha(archive)==archive_sha
review=base/('pre_native_review_'+('qonly' if cfg=='q_only_G1' else 'first4')+'.json')
assert sha(review)==review_sha
review_record=read(review)
assert not review_record.get('launch_blockers') and review_record.get('checks') and all(x['pass'] is True for x in review_record['checks'])
assert cfg in review_record['configurations_in_order']
# Serialize all launchers in this staging namespace. The fresh process scan
# below also protects against a previous launcher that already returned.
launch_lock=(base/'.matched_fixed_launch.lock').open('a')
fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=z.namelist();assert len(names)==len(set(names)) and names
    for info in z.infolist():
        name=PurePosixPath(info.filename)
        assert not name.is_absolute() and '..' not in name.parts and len(name.parts)==2 and name.parts[0]==root.name
        assert not info.is_dir() and ((info.external_attr>>16)&0o170000)!=0o120000
        assert root in (base/info.filename).resolve().parents
    z.extractall(base)
assert sha(root/'manifest.json')==manifest_sha
manifest=read(root/'manifest.json')
for name,info in manifest['files'].items():
    assert Path(name).name==name and name not in ('.','..')
    p=root/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==info['bytes'] and sha(p)==info['sha256']
assert {x.name for x in root.iterdir()}==set(manifest['files'])|{'manifest.json'}
protocol=read(root/'protocol.json');spec=read(root/'execution_spec.json')
assert spec['protocol_sha256']==sha(root/'protocol.json') and protocol['configuration']==cfg
assert protocol['status']=='FROZEN_BEFORE_MATCHED_FIXED_INPUT_EXECUTION' and protocol['executions']==2
assert protocol['supervision']['max_seconds']==300 and protocol['child_guard_required_before_GPU']
assert sha(root/'run_fixed.py')==protocol['profile_wrapper_sha256']=='8938ead8199f4a638396c622db61048f2fa659caaa8e4a11f4416d6518945d5a'
assert sha(root/'jetson_owned_group_guard.py')=='4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4'
assert sha(root/'jetson_monitored_build.py')=='71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1'
assert Path('/usr/bin/setsid').is_file() and Path('/usr/local/bin/nsys').is_file()
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==protocol['expected_boot_id']
sys.path.insert(0,str(root));import jetson_owned_group_guard as module
previous=read(Path('/home/nvidia/rampq/gate05_matched_native_queue_20261021/queue_status.json'))
assert previous['status']=='ALL_FIVE_BUILD_ATTEMPTS_TERMINAL_AND_COLLECTED_NO_INFERENCE' and previous['boot_id']==boot
assert len(previous['members'])==5 and all(x['status']=='BUILD_SCRIPT_OK' and x['guard_complete_no_intervention'] for x in previous['members'])
assert {x['configuration'] for x in previous['members']}=={'fp16_control','r_only_G1','r_only_G1_G5','r_only_G1_G2','q_only_G1'}
queue_process=module.process_info(previous['pid'])
assert queue_process is None or queue_process['state'] in ('Z','X'), 'Previous native queue still live'
queue_empty=[]
for item in previous['members']:
    prior_root=Path(item['root'])
    assert prior_root==Path('/home/nvidia/rampq/gate05_matched_native_queue_20261021')/('gate05_matched_'+item['configuration']+'_20261021')
    terminal=read(prior_root/'monitor/status.json');prior_guard=read(prior_root/'supplemental_group_guard.json')
    assert terminal['status']=='BUILD_SCRIPT_OK' and terminal['boot_id']==boot and terminal['root']==str(prior_root)
    assert terminal['monitor_pid']==item['monitor_pid'] and terminal['build_pid']==item['private_group'] and terminal['max_seconds']==7200
    assert terminal['build_script']==str(prior_root/'jetson_build_matched_control.py')
    assert prior_guard['guard_pid']==item['guard_pid'] and prior_guard['root']==str(prior_root) and prior_guard['boot_id']==boot
    assert prior_guard['status']=='COMPLETE' and prior_guard['owned_group_fully_empty'] and not prior_guard['intervention_performed']
    assert all(prior_guard['monitor_identity'][k]==terminal[k] for k in ('root','monitor_pid','build_pid','max_seconds','build_script'))
    prior_owned=module.OwnedGroup(item['private_group'],boot,prior_root/'jetson_build_matched_control.py')
    assert not prior_owned.snapshot(), 'Previous native build group is not physically empty'
    for label in ('monitor_pid','guard_pid'):
        observed=module.process_info(item[label]);assert observed is None or observed['state'] in ('Z','X'), 'Previous native supervisor still live'
    queue_empty.append({'configuration':item['configuration'],'private_group':item['private_group'],'fresh_group_members':[],'monitor_and_guard_nonlive':True})
busy=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:words=[Path(w.decode(errors='replace')).name for w in (p/'cmdline').read_bytes().split(b'\0') if w]
    except OSError:continue
    if any(w.startswith(('jetson_build_','jetson_matched_fixed_input_adapter','jetson_matched_profile_child_guard','run_gate05_frozen_raw_dataset','jetson_matched_control_serial_queue','trtexec')) or
           w in ('run_fixed.py','nsys','nsys-launcher','nsys-agent') for w in words):busy.append(int(p.name))
assert not busy,busy
available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
assert available>=3*1024*1024
fan=subprocess.run(['systemctl','is-active','nvfancontrol.service'],capture_output=True,text=True,timeout=10)
assert fan.returncode==0 and fan.stdout.strip()=='active'
check=subprocess.run([sys.executable,str(root/'jetson_matched_fixed_input_adapter.py')]+spec['adapter_arguments']+['--check-input-only'],cwd=str(root),capture_output=True,text=True,timeout=60)
assert check.returncode==0,check.stdout+check.stderr
identity={'schema':'rampq.gate05.matched_fixed_launch/v1','configuration':cfg,'root':str(root),'boot_id':boot,
          'board_time':datetime.datetime.now().astimezone().isoformat(),'archive_sha256':archive_sha,'manifest_sha256':manifest_sha,
          'protocol_sha256':sha(root/'protocol.json'),'pre_native_review_sha256':review_sha,
          'native_input_check_stdout':check.stdout,'fresh_available_memory_kib':available,'fan':'active','new_builds':0,'engine_executions_allowed':2}
identity['previous_native_queue_fresh_empty_groups']=queue_empty
with (root/'native_input_receipt.json').open('x') as f:json.dump(identity,f,indent=2)
monitor,guard,owned=None,None,None;launch_ok=False
def interrupt(_sig,_frame):raise KeyboardInterrupt('Unconfirmed matched fixed launch interrupted')
def cleanup():
    if launch_ok:return
    failure={'status':'LAUNCH_UNCONFIRMED_CLEANUP','termination_actions':[]}
    save=lambda:(root/'launch_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
    recovered=owned
    try:
        if recovered is None and monitor is not None:
            p=root/'monitor/status.json'
            if p.is_file():
                observed=read(p)
                assert observed['monitor_pid']==monitor.pid and observed['root']==str(root) and observed['boot_id']==boot and observed['build_script']==str(root/'run_fixed.py')
                if 'build_pid' in observed:recovered=module.OwnedGroup(observed['build_pid'],boot,root/'run_fixed.py')
            if recovered is None:
                candidates=[]
                for p in Path('/proc').iterdir():
                    if not p.name.isdigit():continue
                    child=module.process_info(int(p.name))
                    if child is not None and child['ppid']==monitor.pid and child['pid']==child['pgid']==child['sid'] and str(root/'run_fixed.py') in child['argv']:candidates.append(child)
                assert len(candidates)<=1, 'Ambiguous early wrapper identity; refuse guessed signals'
                if candidates:
                    failure['early_wrapper_identity']=candidates[0]
                    recovered=module.OwnedGroup(candidates[0]['pid'],boot,root/'run_fixed.py')
        if recovered is not None:failure['final_owned_wrapper_members']=module.terminate_owned(recovered,failure,save)
    except BaseException as error:failure['owned_wrapper_cleanup_error']=repr(error)
    # Each exact Popen is handled separately, even after a prior cleanup error.
    for label,process,grace in (('monitor',monitor,45),('guard',guard,12)):
        if process is None:continue
        try:
            if process.poll() is None:
                if label=='monitor':process.send_signal(signal.SIGINT)
                try:process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    failure[label+'_grace_timeout']=True;process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=8)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            failure[label+'_returncode']=process.poll()
        except BaseException as error:failure[label+'_cleanup_error']=repr(error)
    # The independent application group can outlive the wrapper. Authenticate
    # its exact protocol/adapter/leader receipt and clean it separately.
    try:
        p=root/'profile_runner_identity.json'
        if p.exists():
            r=read(p);runner=r['runner']
            assert r['schema']=='rampq.gate05.profile_runner_identity/v1' and r['work_root']==str(root) and r['boot_id']==boot
            assert r['protocol_sha256']==sha(root/'protocol.json') and r['adapter_path']==str(root/'jetson_matched_fixed_input_adapter.py')
            assert r['adapter_sha256']==protocol['adapter_sha256']==sha(Path(r['adapter_path']))
            assert runner['pid']==runner['pgid']==runner['sid'] and runner['pid']>1
            group=module.OwnedGroup(runner['pid'],boot,Path(r['adapter_path']))
            assert group.leader_start_ticks is None or group.leader_start_ticks==runner['start_ticks']
            failure['escaped_runner_final_members']=module.terminate_owned(group,failure,save)
            failure['escaped_runner_fully_empty']=not failure['escaped_runner_final_members']
    except BaseException as error:failure['escaped_runner_cleanup_error']=repr(error)
    try:
        p=root/'profile_child_guard.json'
        if p.exists() and (root/'profile_runner_identity.json').exists():
            child_guard=read(p)
            if child_guard and 'guard_pid' in child_guard:
                assert child_guard['work_root']==str(root) and child_guard['boot_id']==boot
                assert child_guard['runner_identity_sha256']==sha(root/'profile_runner_identity.json') and child_guard['protocol_sha256']==sha(root/'protocol.json')
                assert child_guard['guard_sha256']==protocol['child_guard_sha256']==sha(root/'jetson_matched_profile_child_guard.py')
                child_pid=child_guard['guard_pid'];deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    current=module.process_info(child_pid)
                    if current is None or current['state'] in ('Z','X'):break
                    assert current['pid']==current['pgid']==current['sid'] and str(root/'jetson_matched_profile_child_guard.py') in current['argv'] and str(root) in current['argv']
                    time.sleep(.25)
                current=module.process_info(child_pid)
                failure['escaped_child_guard_final_process']=current
                failure['escaped_child_guard_nonlive']=current is None or current['state'] in ('Z','X')
    except BaseException as error:failure['escaped_child_guard_observation_error']=repr(error)
    try:
        if recovered is not None:failure['final_owned_wrapper_members']=recovered.snapshot()
    except BaseException as error:failure['final_wrapper_snapshot_error']=repr(error)
    save()
atexit.register(cleanup)
signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
console=(root/'monitor_console.log').open('xb',buffering=0)
monitor=subprocess.Popen([sys.executable,'-u',str(root/'jetson_monitored_build.py'),'--root',str(root),'--run',str(root/'monitor'),'--max-seconds','300','--build-script','run_fixed.py'],cwd=str(root),stdin=subprocess.DEVNULL,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
console.close()
monitor_live=module.process_info(monitor.pid)
assert monitor_live is not None and monitor_live['pid']==monitor_live['pgid']==monitor_live['sid'] and str(root/'jetson_monitored_build.py') in monitor_live['argv'] and str(root) in monitor_live['argv']
deadline=time.monotonic()+12;status_path=root/'monitor/status.json'
while not status_path.exists() and monitor.poll() is None and time.monotonic()<deadline:time.sleep(.2)
assert status_path.exists()
status=read(status_path);assert status['monitor_pid']==monitor.pid and status['max_seconds']==300 and status['root']==str(root) and status['boot_id']==boot and status['build_script']==str(root/'run_fixed.py')
# The first atomic status may precede creation of the wrapper. Wait for its
# authenticated build PID before attaching its immutable group guard.
while 'build_pid' not in status and monitor.poll() is None and time.monotonic()<deadline:
    time.sleep(.2);status=read(status_path)
assert 'build_pid' in status
assert all(status[k]==v for k,v in {'monitor_pid':monitor.pid,'max_seconds':300,'root':str(root),'boot_id':boot,'build_script':str(root/'run_fixed.py')}.items())
owned=module.OwnedGroup(status['build_pid'],boot,root/'run_fixed.py')
guard_log=(root/'supplemental_guard_console.log').open('xb',buffering=0)
guard=subprocess.Popen([sys.executable,'-u',str(root/'jetson_owned_group_guard.py'),'--root',str(root),'--expected-monitor-pid',str(monitor.pid)],cwd=str(root),stdin=subprocess.DEVNULL,stdout=guard_log,stderr=subprocess.STDOUT,start_new_session=True)
guard_log.close()
deadline=time.monotonic()+8;p=root/'supplemental_group_guard.json'
g=None
while time.monotonic()<deadline:
    if p.exists():
        candidate=read(p)
        # The immutable helper exclusively creates {} before its first atomic
        # complete publication. Existence alone is not acknowledgement.
        if candidate and 'status' in candidate:g=candidate;break
    if guard.poll() is not None:break
    time.sleep(.2)
assert g is not None, 'Guard did not publish a complete acknowledgement'
assert g['guard_pid']==guard.pid and g['root']==str(root) and g['boot_id']==boot and g['owned_pgid']==status['build_pid']
assert all(g['monitor_identity'][k]==status[k] for k in ('root','monitor_pid','build_pid','max_seconds','build_script'))
assert g['leader_start_ticks']==owned.leader_start_ticks
assert (g['status']=='WATCHING' and guard.poll() is None) or (g['status']=='COMPLETE' and g['owned_group_fully_empty'] and not g['intervention_performed'] and not owned.snapshot())
if guard.poll() is None:
    guard_live=module.process_info(guard.pid)
    assert guard_live is not None and guard_live['pid']==guard_live['pgid']==guard_live['sid'] and str(root/'jetson_owned_group_guard.py') in guard_live['argv'] and str(root) in guard_live['argv']
identity.update(status='TWO_CALL_PROFILE_LAUNCHED_UNDER_WRAPPER_AND_ESCAPED_CHILD_GUARDS',monitor_pid=monitor.pid,private_group=status['build_pid'],guard_pid=guard.pid)
identity.update(monitor_process_identity=monitor_live,initial_owned_wrapper_members=owned.initial_members,
                initial_owned_wrapper_leader_start_ticks=owned.leader_start_ticks,initial_supplemental_guard_status=g,
                wrapper_guard_and_pre_GPU_child_guard_chain_required=True)
with (root/'launch.json').open('x') as f:
    json.dump(identity,f,indent=2);f.flush();os.fsync(f.fileno())
launch_ok=True;print(json.dumps(identity),flush=True)
