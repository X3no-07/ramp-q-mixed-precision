"""Saved-evidence/static-only independent entry-contract review; no builder execution."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'artifacts/gate05/20261021_qall_none_entry_failure'
RAW = OLD / 'native_evidence'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
checks = []
def check(name, value):
    checks.append({'name': name, 'passed': bool(value)})

check('first_archive_exact', sha(OLD/'native_evidence.zip') == '46a720c43998a8c7ee5940c2272b24e0c44a04c60a9618b2d0c9ddda91130328')
with zipfile.ZipFile(OLD/'native_evidence.zip') as z:
    check('first_archive_crc', z.testzip() is None)
receipt = read(RAW/'remote_receipt.json')['files']
check('first_receipt_identity', sha(RAW/'remote_receipt.json') == '08309de11cc97e0fc7a587040b8371519be4fabc4e1cfa71f11eebac09527e61')
check('first_receipt_61', len(receipt) == 61)
for rel, row in receipt.items():
    p = RAW/rel
    check('receipt:'+rel, p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'])
check('first_exact_inventory', {str(p.relative_to(RAW)).replace('\\','/') for p in RAW.rglob('*') if p.is_file()} == set(receipt)|{'remote_receipt.json'})
status = read(RAW/'monitor/status.json')
check('first_argparse_rc2_not_buildfailure', status['status'] == 'BUILD_SCRIPT_FAILED' and status['build_returncode'] == 2 and status['elapsed_seconds'] == 3.04)
check('first_required_root_diagnostic', 'the following arguments are required: --root' in (RAW/'monitor/build_console.log').read_text())
check('first_zero_attempt_markers_and_results', not (RAW/'native_build_attempt.json').exists() and not (RAW/'latest_build.json').exists() and not (RAW/'results').exists())
check('first_group_complete_empty', read(RAW/'supplemental_group_guard.json')['status']=='COMPLETE' and read(RAW/'supplemental_group_guard.json')['owned_group_fully_empty'])
builder = ROOT/'scripts/jetson_build_corrected_qall_none.py'
check('sealed_builder_unchanged', sha(builder) == sha(RAW/builder.name) == '4e5b9b71be93e72f5310273f477c0cbf7256fe2249f66c3a01f9ad3d96844aaa')
builder_source = builder.read_text(encoding='utf-8')
check('argparse_precedes_runtime_import_and_attempt', builder_source.index('args = parser.parse_args()') < builder_source.index("marker = root / 'native_build_attempt.json'") < builder_source.index('import tensorrt as trt'))
adapter = (HERE/'build_entry.py').read_text(encoding='utf-8')
check('adapter_exact_identity', sha(HERE/'build_entry.py') == 'f26303efe459519b43aff0163fb63ff8b4e95c1699b5fc9b143d76bf4a12d1e5')
check('adapter_py38_ast', ast.parse(adapter, feature_version=(3,8)) is not None)
check('adapter_root_resolves_own_parent', "root = Path(__file__).resolve().parent" in adapter)
check('adapter_sets_required_argv_only', "sys.argv = [str(builder), '--root', str(root)]" in adapter)
check('adapter_same_pid_runpy_main', "runpy.run_path(str(builder), run_name='__main__')" in adapter and 'subprocess' not in adapter and 'os.exec' not in adapter)
check('adapter_no_native_import_flags_or_model_mutation', {n.name for n in ast.walk(ast.parse(adapter)) if isinstance(n, ast.Import) for n in n.names} == {'runpy','sys'} and '--check-input-only' not in adapter)
monitor = (RAW/'jetson_monitored_build.py').read_text(encoding='utf-8')
check('monitor_actual_noargs_contract', "subprocess.Popen([sys.executable, '-u', str(build_script)]" in monitor)
launch = (HERE/'launch_native.py').read_text(encoding='utf-8')
prior = (ROOT/'artifacts/gate05/20261021_qall_none_runtime/launch_native.py').read_text(encoding='utf-8')
check('launcher_exact_identity', sha(HERE/'launch_native.py') == '0fdab0b4c1ea53b9ee0fbf075eb96defa2467a1d6369a27c563a376ec4ebfdbb')
check('launcher_py38_ast', ast.parse(launch, feature_version=(3,8)) is not None)
check('fresh_root_no_overwrite', "gate05_qall_none_entryfix1_20261021" in launch and 'assert not root.exists()' in launch and ".open('x')" in launch)
check('same_archive_protocol_pin', "gate05_qall_none_control_20261021.zip" in launch and "572120502a6db021945c63cdd781d81e2a4b1771f5d9a13c3f58d0da7278eac1" in launch)
check('adapter_and_review_authenticated', "sha(staging / 'build_entry.py') == 'f26303efe459519b43aff0163fb63ff8b4e95c1699b5fc9b143d76bf4a12d1e5'" in launch and "sha(staging / 'entry_adapter_review.json') == entry_review_sha" in launch)
check('five_argv_contract', 'archive_sha, spec_sha, manifest_sha, review_sha, entry_review_sha = sys.argv[1:]' in launch)
check('monitor_and_owned_group_same_adapter', "'--build-script', 'build_entry.py'" in launch and "OwnedGroup(status['build_pid'], identity['boot_id'], root / 'build_entry.py')" in launch)
def fn(src, name):
    return ast.dump(next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name), include_attributes=False)
check('safe_cleanup_exact_prior_AST', fn(launch,'cleanup_unconfirmed_launch') == fn(prior,'cleanup_unconfirmed_launch'))
check('safe_cleanup_registered_before_monitor', launch.index('atexit.register(cleanup_unconfirmed_launch)') < launch.index('monitor = subprocess.Popen('))
check('same_deadline_and_native_budget', "'--max-seconds', '7200'" in launch and 'native_attempts=1' in launch and 'automatic_inference=False' in launch)
check('fresh_memory_and_fan_still_enforced', 'available_kib >= 3 * 1024 * 1024' in launch and "fan.returncode == 0 and fan.stdout.strip() == 'active'" in launch)
check('guard_same_pid_and_group_authentication', "'--expected-monitor-pid', str(monitor.pid)" in launch and "('owned_pgid', status['build_pid'])" in launch and "('root', 'monitor_pid', 'build_pid', 'max_seconds', 'build_script')" in launch)
check('pinned_monitor_guard_unchanged', sha(RAW/'jetson_monitored_build.py') == '71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1' and sha(RAW/'jetson_owned_group_guard.py') == '4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4')
report = {'schema':'rampq.gate05.entry_adapter_independent_review/v1','client_at':datetime.datetime.now().astimezone().isoformat(),'status':'STATIC_ENTRY_CONTRACT_AND_SAVED_ZERO_ATTEMPT_VERIFIED' if all(c['passed'] for c in checks) else 'BLOCKED','checks':checks,'check_count':len(checks),'launch_blockers':[c['name'] for c in checks if not c['passed']],'no_CPU_graph_native_SSH_or_builder_execution':True,'builder_argv':['sealed_builder','--root','same_resolved_bundle_root'],'process_identity_note':'runpy executes builder in adapter PID. sys.argv is Python state; /proc argv remains build_entry.py and matches monitor/OwnedGroup/guard script identity.','first_launch':'argparse required --root failure before helper/native imports and native attempt marker; no TensorRT build retry','unchanged_native_budget':1,'unchanged_archive_sha256':'e5eb1ed576349f5f3a173cb7a8cfbd2ffe6368450c7618a82ad7f8357659f60d','adapter_sha256':sha(HERE/'build_entry.py'),'launcher_sha256':sha(HERE/'launch_native.py'),'source_reviewer_script_sha256':sha(Path(__file__))}
with (HERE/'entry_adapter_review.json').open('x',encoding='utf-8') as f:
    json.dump(report,f,ensure_ascii=False,indent=2)
    f.write('\n')
print(json.dumps({'checks':len(checks),'blockers':report['launch_blockers'],'report_sha256':sha(HERE/'entry_adapter_review.json')}))
