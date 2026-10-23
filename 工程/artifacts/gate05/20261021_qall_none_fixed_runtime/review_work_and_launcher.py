"""Independent static work-bundle/argv/owned-monitor review; no execution."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
WORK=ROOT/'artifacts/gate05/20261021_qall_none_fixed_work'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
checks=[]
def c(name,value):checks.append({'name':name,'passed':bool(value)})
archive=WORK.with_suffix('.zip')
c('ZIP_exact',sha(archive)=='e99343fe84f7f14533d55639760d19624ab052906cc42fe6dd7227e851b6d357' and archive.stat().st_size==34898)
with zipfile.ZipFile(archive) as z:
    c('ZIP_CRC',z.testzip() is None)
    expected={str(p.relative_to(WORK)).replace('\\','/') for p in WORK.rglob('*') if p.is_file()}
    names={name.partition('/')[2] for name in z.namelist() if not name.endswith('/')}
    c('ZIP_exact_work_inventory',names==expected)
    for name in z.namelist():
        if not name.endswith('/'):
            rel=name.partition('/')[2]
            c('ZIP_bytes:'+rel,z.read(name)==(WORK/rel).read_bytes())
manifest=read(WORK/'manifest.json');spec=read(WORK/'execution_spec.json');protocol=read(WORK/'protocol.json')
for rel,row in manifest['files'].items():
    p=WORK/rel;c('manifest:'+rel,p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'])
c('manifest_exact_payload_inventory',set(manifest['files'])|{'manifest.json'}==expected)
c('execution_spec_exact',sha(WORK/'execution_spec.json')=='7667abfd6cbcfc7d0e6af4eab8fa7025fcc7978ef8a6a6f3433fd09d0469733e')
c('runner_exact_and_unchanged',sha(WORK/'jetson_gate05_control_fixed_input.py')==sha(ROOT/'scripts/jetson_gate05_control_fixed_input.py')==spec['runner_sha256']=='d1576ea834bd48ad78df3bf920b315726c1f0da8fe388eb7c3450dda8d93e0ca')
c('protocol_exact',sha(WORK/'protocol.json')==spec['protocol_sha256']=='857f296be657da8936ddd320f7ade6a0d8b239c19a6ea953a6ba737970f3f29a')
c('audit_exact',sha(WORK/'precision_audit.json')==spec['precision_audit_sha256']=='52d9f59be4b379817610c1f5d4ef82f73f7aafb23ead8060bfeef59082c9a4a5')
c('two_calls_zero_build_CPU_300s',spec['max_engine_executions']==protocol['executions']==2 and spec['new_builds']==spec['new_CPU_model_executions']==0 and spec['max_seconds']==protocol['supervision']['max_seconds']==300)
args=spec['runner_args'];opts={args[i]:args[i+1] for i in range(0,len(args)-1,2)}
c('explicit_diagnostic_gap_branch',args[-1]=='--diagnostic-formats-only' and spec['diagnostic_formats_only'] is True and protocol['diagnostic_formats_only_allowed'] is True)
remote='/home/nvidia/rampq/gate05_qall_none_entryfix1_20261021/gate05_qall_none_control_20261021'
c('runner_root_and_build_correct',opts['--root']==remote and opts['--build']==remote+'/results/build_20261018T002834_795020Z')
c('runner_all_sha_argv_pins',opts['--engine-sha256']==protocol['engine_sha256']=='c3d9e75d61452c69b97097d1320221d5d56d7ad0e2fc1370480cf3e09eb3a41e' and opts['--audit-sha256']==sha(WORK/'precision_audit.json') and opts['--protocol-sha256']==sha(WORK/'protocol.json'))
c('runner_work_placeholder_paths',opts['--audit']=='{WORK}/precision_audit.json' and opts['--protocol']=='{WORK}/protocol.json' and opts['--output']=='{WORK}/fixed_outputs')
wrapper=(WORK/'run_fixed.py').read_text(encoding='utf-8');launch=(HERE/'launch_native.py').read_text(encoding='utf-8')
for name,text in [('wrapper',wrapper),('launcher',launch),('runner',(WORK/'jetson_gate05_control_fixed_input.py').read_text(encoding='utf-8'))]:
    c(name+'_py38_AST',ast.parse(text,feature_version=(3,8)) is not None)
c('wrapper_exact',sha(WORK/'run_fixed.py')=='175c97d4506100ce68da5d9fec57b9c04cfa8517f1080376415e2db889bac0ff')
tree=ast.parse(wrapper)
arg_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument']
c('wrapper_no_required_args',len(arg_calls)==1 and ast.literal_eval(arg_calls[0].args[0])=='--check-input-only' and not any(k.arg=='required' and ast.literal_eval(k.value) for k in arg_calls[0].keywords))
c('wrapper_resolves_own_parent',"root=Path(__file__).resolve().parent" in wrapper)
c('wrapper_input_check_returns_before_nsys',wrapper.index('if args.check_input_only:')<wrapper.index("profile=root/'profile'") and "subprocess.run(command+['--check-input-only']);raise SystemExit(result.returncode)" in wrapper)
c('wrapper_expands_only_WORK_placeholders',"[v.replace('{WORK}',str(root)) for v in spec['runner_args']]" in wrapper)
c('one_profile_no_auto_replay',wrapper.count('result=subprocess.run(argv)')==1 and "if result.returncode:raise SystemExit(result.returncode)" in wrapper and not any(isinstance(n,(ast.For,ast.While)) for n in tree.body[tree.body.index(next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='argv' for t in n.targets))):]))
c('nsys_full_process_capture',"'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none'" in wrapper and '--capture-range' not in wrapper)
c('export_CPU_only_after_profile',wrapper.index("'export','--type','sqlite'")>wrapper.index('if result.returncode:raise SystemExit(result.returncode)') and 'timeout=60' in wrapper)
c('wrapper_fresh_profile_dir',"profile.mkdir(exist_ok=False)" in wrapper)
c('launcher_four_sha_argv','archive_sha, spec_sha, manifest_sha, review_sha = sys.argv[1:]' in launch)
c('launcher_fresh_stage_and_root',"gate05_qall_none_fixed_stage_20261021" in launch and "root = staging / 'gate05_qall_none_fixed_20261021'" in launch and 'assert not root.exists()' in launch)
c('launcher_checks_spec_manifest_protocol_review',"sha(root / 'execution_spec.json') == spec_sha" in launch and "sha(root / 'manifest.json') == manifest_sha" in launch and protocol['runner_sha256']==spec['runner_sha256'] and "sha(staging / 'pre_native_review.json') == review_sha" in launch)
c('launcher_wrapper_check_without_root_arg',"subprocess.run([sys.executable, str(root / 'run_fixed.py'), '--check-input-only']" in launch)
c('monitor_noarg_entry_compatible',"'--build-script', 'run_fixed.py'" in launch and "[sys.executable, '-u', str(build_script)]" in (WORK/'jetson_monitored_build.py').read_text())
c('owned_identity_same_wrapper',"OwnedGroup(status['build_pid'], identity['boot_id'], root / 'run_fixed.py')" in launch)
c('monitor_guard_same_source_PID_and_budget',"'--expected-monitor-pid', str(monitor.pid)" in launch and "('root', 'monitor_pid', 'build_pid', 'max_seconds', 'build_script')" in launch and "status['max_seconds'] == 300" in launch)
c('native_budget_identity_two_zero_build','native_build_attempts=0, engine_executions_allowed=2' in launch)
c('fresh_fan_and_memory','available_kib >= 3 * 1024 * 1024' in launch and "fan.returncode == 0 and fan.stdout.strip() == 'active'" in launch)
prior=(ROOT/'artifacts/gate05/20261021_qall_none_entryfix1_runtime/launch_native.py').read_text(encoding='utf-8')
def fn(text,name):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
c('safe_cleanup_AST_same_mature_launcher',fn(launch,'cleanup_unconfirmed_launch')==fn(prior,'cleanup_unconfirmed_launch'))
c('safe_cleanup_before_spawn',launch.index('atexit.register(cleanup_unconfirmed_launch)')<launch.index('monitor = subprocess.Popen('))
c('sameboot_pinned','692c7544-b8bd-415b-939c-b11cdbd9f7f0' in launch)
c('guardscript_and_monitor_hashes_unchanged',sha(WORK/'jetson_owned_group_guard.py')=='4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4' and sha(WORK/'jetson_monitored_build.py')=='71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1')
c('limits_preserve_failed_guard_no_gate',protocol['original_complete_model_numerical_guard']=='FAILED_UNCHANGED' and protocol['gate05_status']=='INCOMPLETE' and not protocol['dataset_evaluation_run'])
report={'schema':'rampq.gate05.fixed_work_pre_native_independent_review/v1','client_at':datetime.datetime.now().astimezone().isoformat(),'status':'FROZEN_WORK_ENTRY_AND_GUARDED_TWO_EXECUTION_SCOPE_VERIFIED' if all(v['passed'] for v in checks) else 'BLOCKED','check_count':len(checks),'checks':checks,'launch_blockers':[v['name'] for v in checks if not v['passed']],'work_zip_sha256':sha(archive),'manifest_sha256':sha(WORK/'manifest.json'),'execution_spec_sha256':sha(WORK/'execution_spec.json'),'runner_sha256':sha(WORK/'jetson_gate05_control_fixed_input.py'),'wrapper_sha256':sha(WORK/'run_fixed.py'),'launcher_sha256':sha(HERE/'launch_native.py'),'prior_source_input_compatibility_check':'Exact d157... runner with pinned857f... protocol/52d9... audit/c3d9... engine passed purestdlib --check-input-only on saved local inputs; no runtime imports. Remote wrapper will use its correct absolute root and work placeholders.','entry_contract':'Monitor invokes wrapper without arguments, supported by optional check-only parser. Wrapper launches exactly one nsys-profiled runner; runner owns two engine calls. OwnedGroup/guard authenticate wrapper PID/argv and inherited private group. SQLite export is CPU-only after successful GPU run, never a GPU retry.','new_CPU_model_native_or_SSH_runs':0,'old_sources_or_raw_modified':False,'work_creation_metadata_typo_scope':'Root disclosed post-ZIP metadata-print st_size typo; exact closed ZIP/CRC/8payload+manifest bytes verified here. No GPU execution or repackaging inferred from that client-print failure.','strict_integer_gaps':'Seven named tactic gaps retained; profiling/maps enable future authenticated SASS review, not automatic arithmetic PASS.','reviewer_source_sha256':sha(Path(__file__))}
out=HERE/'pre_native_review.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'checks':len(checks),'blockers':report['launch_blockers'],'report_sha256':sha(out),'launcher_sha256':report['launcher_sha256'],'manifest_sha256':report['manifest_sha256']}))
