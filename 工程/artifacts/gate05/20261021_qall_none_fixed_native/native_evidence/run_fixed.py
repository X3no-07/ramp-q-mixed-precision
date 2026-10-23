"""Run the frozen two-call fixed-input profile under the immutable supervisor."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--check-input-only',action='store_true');args=parser.parse_args()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=read(root/'execution_spec.json');manifest=read(root/'manifest.json')
for rel,v in manifest['files'].items():
 p=(root/rel).resolve();assert root in p.parents and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],rel
assert spec['max_engine_executions']==2 and spec['max_seconds']==300 and spec['new_builds']==0
runner=root/spec['runner_file'];assert sha(runner)==spec['runner_sha256']
command=[sys.executable,'-u',str(runner)]+[v.replace('{WORK}',str(root)) for v in spec['runner_args']]
if args.check_input_only:
 result=subprocess.run(command+['--check-input-only']);raise SystemExit(result.returncode)
profile=root/'profile';profile.mkdir(exist_ok=False)
argv=['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--output',str(profile/'fixed_input')]+command
start=time.monotonic();result=subprocess.run(argv)
record=dict(argv=argv,returncode=result.returncode,elapsed_seconds=time.monotonic()-start,engine_execution_budget=2)
(root/'profile_exit.json').write_text(json.dumps(record,indent=2)+'\n')
if result.returncode:raise SystemExit(result.returncode)
reports=list(profile.glob('*.qdrep'))+list(profile.glob('*.nsys-rep'));assert len(reports)==1,[str(p) for p in reports]
try:
 exported=subprocess.run(['/usr/local/bin/nsys','export','--type','sqlite','--output',str(profile/'fixed_input.sqlite'),str(reports[0])],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
 export=dict(returncode=exported.returncode,stdout=exported.stdout,stderr=exported.stderr)
except subprocess.TimeoutExpired as error:export=dict(status='SQLITE_EXPORT_TIMEOUT_PRESERVED',error=str(error))
(root/'profile_export.json').write_text(json.dumps(export,indent=2)+'\n')
print('FROZEN_TWO_EXECUTION_FIXED_INPUT_PROFILE_FINISHED',flush=True)
