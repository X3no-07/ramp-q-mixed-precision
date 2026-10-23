"""Read-only source review plus fake-child cleanup tests. Creates no processes."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
source=ROOT/'scripts/jetson_matched_control_serial_queue.py'
text=source.read_text(encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tree=ast.parse(text,feature_version=(3,8));checks=[]
def c(name,value):checks.append({'name':name,'passed':bool(value)})
c('Python38_AST',tree is not None)
c('frozen_native_protocol_pin','12ce0b79c1e765f3293cc529f01023e6a58671e8ad688f0884fcf216931a3459' in text)
c('frozen_builder_pin',sha(ROOT/'scripts/jetson_build_matched_control.py')=='4d2ee7d283abf6cd75b6d76e58870f79f11392fa06eaef37936fab6d7bafca34')
c('queue_self_pin_required',"plan['queue_script_sha256']==sha(Path(__file__))" in text)
c('strict_five_order',"ORDER = ['fp16_control', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1']" in text)
c('no_inference_or_profile_code',not any(isinstance(n,ast.Attribute) and n.attr in ('execute_v2','execute_async_v2','build_serialized_network') for n in ast.walk(tree)) and 'nsys' not in text)
c('plan_zero_inference_five_build',"plan['native_build_attempts']==5 and plan['native_inference_executions']==0" in text)
c('permember_pureinput_timeout60','text=True, timeout=60' in text)
c('real_inputcheck_schema',"check['native_inference_run'] is False" in text and "check['native_build_run'] is False" in text and "check['native_runtime_imported'] is False" in text and "check['native_inference_executions']" not in text)
c('default_noarg_builder',"parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)" in (ROOT/'scripts/jetson_build_matched_control.py').read_text())
c('monitor_noarg_builder_argv',"'--build-script','jetson_build_matched_control.py'" in text)
c('one_attempt_each_no_retry_loop',"for item, receipt in zip(plan['members'],receipts):" in text and text.count("active['monitor']=subprocess.Popen(")==1)
c('fresh_member_root_results_attempts',"not (root/'native_build_attempt.json').exists() and not (root/'results').exists()" in text)
c('guard_module_available_before_spawn',text.index("active.update(monitor=None,guard=None,owned=None,module=module,root=root,cleaned=False)")<text.index("active['monitor']=subprocess.Popen("))
c('bounded_early_status12_and_guard8',"wait_path(root/'monitor/status.json',monitor,12)" in text and "wait_path(root/'supplemental_group_guard.json',guard,8)" in text)
c('same_wrapper_own_group',"module.OwnedGroup(ms['build_pid'],state['boot_id'],root/'jetson_build_matched_control.py')" in text)
c('guard_complete_identity',"gs['root']==str(root)" in text and "gs['monitor_identity'][key]==ms[key]" in text)
c('bounded_monitor_wall_exit_grace',"MONITOR_EXIT_GRACE_SECONDS = 90" in text and 'time.monotonic()-member_started<7200+MONITOR_EXIT_GRACE_SECONDS' in text)
c('no_native_budget_extension',"'--max-seconds','7200'" in text and "spec['max_build_seconds'] == 7200" in text)
c('protective_guard_stops_queue',"not observed_guard.get('intervention_performed',False)" in text)
c('negative_signal_exit_stops',"terminal['build_returncode']>=0" in text)
c('ordinary_positive_failure_can_continue',"terminal['status'] in ('BUILD_SCRIPT_OK','BUILD_SCRIPT_FAILED')" in text)
c('prebuild_entry_failure_stops',"Entry failed before an authenticated native attempt" in text)
c('memory_interrupt_exception_stops',"('KeyboardInterrupt','MemoryError','TimeoutError')" in text)
c('monitor_protective_and_kernel_stop',"terminal.get('reason') is None" in text and "for x in BAD_KERNEL" in text)
c('archive_after_terminal_group_proofs',text.index("require(not owned.snapshot()")<text.index("'--archive',str(archive)"))
c('archive_timeout180','text=True,timeout=180' in text)
c('exception_cleanup_before_record',"except BaseException as exc:\n        cleanup()\n        state.update" in text)
c('builder_helpers_unchanged',all(sha(ROOT/'scripts'/name)==expected for name,expected in {'jetson_monitored_build.py':'71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1','jetson_owned_group_guard.py':'4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4'}.items()))

# Execute only the nested cleanup function against fake Popen/module objects;
# there is no import or call of the queue main, subprocess, /proc or GPU work.
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
cleanup=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='cleanup')
class FakeChild:
    def __init__(self,hang=False):self.alive=True;self.hang=hang;self.returncode=None;self.actions=[]
    def poll(self):return self.returncode
    def send_signal(self,sig):
        self.actions.append(('signal',int(sig)))
        if not self.hang:self.alive=False;self.returncode=-int(sig)
    def wait(self,timeout):
        self.actions.append(('wait',timeout))
        if self.alive and self.hang:raise subprocess.TimeoutExpired('FAKE',timeout)
        self.alive=False
        if self.returncode is None:self.returncode=0
        return self.returncode
    def kill(self):self.actions.append(('kill',));self.alive=False;self.returncode=-9
def fake_cleanup(active):
    saved=[]
    env={'active':active,'state':{'boot_id':'FAKE'},'save':lambda dest,obj:saved.append(copy.deepcopy(obj)),
         'signal':signal,'subprocess':subprocess,'Path':Path,'boot':lambda:'FAKE','read':lambda p:{},'require':lambda ok,msg:None if ok else (_ for _ in ()).throw(RuntimeError(msg))}
    code=ast.Module(body=[copy.deepcopy(cleanup)],type_ignores=[]);ast.fix_missing_locations(code)
    exec(compile(code,'fake-cleanup-only','exec'),env)
    env['cleanup']();env['cleanup']()
    return saved
def active(child,module=None,owned=None,guard=None):return {'root':Path('MOCK_NO_FILES_WRITTEN'),'monitor':child,'guard':guard,'module':module,'owned':owned,'cleaned':False}
early=FakeChild();record=fake_cleanup(active(early))
c('fake_early_missing_owned_still_stops_monitor',not early.alive and ('signal',int(signal.SIGINT)) in early.actions and len(record)==1)
def broken(*args):raise RuntimeError('FAKE owned identity failure')
owned=SimpleNamespace(snapshot=lambda:[])
mon=FakeChild();guard=FakeChild();record=fake_cleanup(active(mon,SimpleNamespace(terminate_owned=broken),owned,guard))
c('fake_owned_error_does_not_skip_monitor_guard',not mon.alive and not guard.alive and 'owned_cleanup_error' in record[-1])
hang=FakeChild(hang=True);record=fake_cleanup(active(hang))
c('fake_hung_monitor_bounded_escalation',not hang.alive and ('kill',) in hang.actions and [a[1] for a in hang.actions if a[0]=='wait']==[45,8,5])

report={'schema':'rampq.gate05.serial_queue_source_review/v1','status':'STATIC_QUEUE_AND_FAKE_CLEANUP_PATHS_VERIFIED' if all(r['passed'] for r in checks) else 'BLOCKED','checks':checks,'check_count':len(checks),'blockers':[r['name'] for r in checks if not r['passed']],'queue_script_sha256':sha(source),'modified_file':'scripts/jetson_matched_control_serial_queue.py only','changes':['Cleanup no longer returns when owned group is not yet established; recover only authenticated early group, always signal exact owned monitor child, bounded waits and independent guard cleanup.','Import pinned guard before spawn; record Popen children immediately.','Require whole guard/monitor/script/boot identity, actual pureinput schema, and fresh noattempt markers.','Native7200s unchanged; outer monitor-process exit deadline7200+90s prevents unbounded queue wait.','Signal, protective guard, kernel, interrupted/resource and prebuild entry failures halt; ordinary positive authenticated builder failures may be archived and remaining independent configurations continue.','Plan must pin queue_script_sha256.'],'new_real_processes_GPU_CPU_model_or_SSH':0,'fake_child_tests_only':3,'plan_not_yet_frozen':'Root must bind final queue hash before launch. No queue or member builder executed in this review.'}
out=HERE/'serial_queue_source_review.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'checks':len(checks),'blockers':report['blockers'],'queue_sha256':sha(source),'review_sha256':sha(out)}))
