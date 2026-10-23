"""Execute predeclared synthetic stress through unchanged official functions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
AUDIT=ROOT/'artifacts/gate1/20261023_full_ap_oracle_audit_v1'
RUNTIME=ROOT/'artifacts/gate1/20261023_native_oracle_runtime_v1'
SOURCES=ROOT/'artifacts/gate1/20261022_split_refinement/evaluation_sources'
PINS={
 'inputs_manifest.json':'1d77ec04575579a1233e230f30ab65eb0d82537938ae5ce347459d90bae7fb91',
 'comparison_policy.json':'ff75048e986f5ec4bd2e956fca41a2370eb62b525b62b386cf3bd07f192f0b70',
 'stress_fixtures.json':'279653d1647e8e9c3bd74b44d6cdc9de91ad22d54275bdeeeb00406cbf4fc7b6',
 'run_full_ap_stress_oracle.m':'8617c5462e7cf86d3cf3ab53e182d3be4cfea915e04477db76ea4316e4ee7a98'}


def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def ident(path):return {'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def quote(path):return "'"+str(path).replace('\\','/').replace("'","''")+"'"


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
 for name,digest in PINS.items():assert sha(AUDIT/name)==digest,name
 source_manifest=SOURCES/'source_manifest.json'
 assert sha(source_manifest)=='1d45d3ecaabf5f3f8a8885436a0a22718a890a17f8ad9d5ec4f9c274c3c079ae'
 rows=read(source_manifest)['files'];assert len(rows)==11
 for row in rows:
  path=(SOURCES/row['path']).resolve(strict=True)
  assert path.is_relative_to(SOURCES) and not path.is_symlink()
  assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
 runtime_manifest=RUNTIME/'runtime_manifest.json'
 assert sha(runtime_manifest)=='2490915f6f61adfde6fe85a5ba7cd249d77877a4461b7c42e418fe05d656d9e2'
 runtime_rows=read(runtime_manifest)['files'];assert len(runtime_rows)==2890
 for row in runtime_rows:
  path=(ROOT/row['path']).resolve(strict=True)
  assert path.is_relative_to(RUNTIME/'runtime') and not path.is_symlink()
  assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
 executable=RUNTIME/'runtime/octave-11.3.0-w64/mingw64/bin/octave-cli.exe'
 assert sha(executable)=='387ac823fb5b01b3407fb34edda6ab1cdba0c9619841a0f6dee323d05a850f2d'
 mean2=RUNTIME/'runtime/octave-11.3.0-w64/mingw64/share/octave/packages/image-2.20.0/mean2.m'
 assert sha(mean2)=='a1d766f961fdb245edbf52a7f1a4d20eced94b56fc803a99f7094f0a089f773a'
 fixture=read(AUDIT/'stress_fixtures.json');assert fixture['data_role']=='synthetic' and len(fixture['cases'])==12
 output=AUDIT/'native_stress_execution_v1';output.mkdir(exist_ok=False)
 expression=('addpath('+quote(mean2.parent)+",'-begin');assert(strcmp(canonicalize_file_name(which('mean2')),canonicalize_file_name("+quote(mean2)+")));"
   +"assert(strcmp(version,'11.3.0'));addpath("+quote(AUDIT)+",'-begin');run_full_ap_stress_oracle("+quote(output)+');')
 command=[str(executable),'--no-gui','--quiet','--no-history','--no-init-file','--no-site-file','--eval',expression]
 receipt={'schema':'rampq.gate1.full_ap_stress_launcher/v1','status':'RUNNING','at_utc':datetime.now(timezone.utc).isoformat(),
  'launcher':ident(Path(__file__)),'inputs':{name:ident(AUDIT/name) for name in PINS},
  'official_source_manifest':ident(source_manifest),'official_sources_verified':11,
  'runtime_manifest':ident(runtime_manifest),'runtime_files_verified':2890,
  'executable':ident(executable),'mean2':ident(mean2),'runtime_version':'11.3.0','command':command,
  'returncode':None,'fixture_count':12,'real_data_read':False,'model_executed':False,
  'selection_or_holdout_accessed':False,'official_sources_modified':False,'system_configuration_modified':False,
  'official_AP_arbitrary_input_equivalence_certified':False,'gate1_scientific_pass':False}
 write(output/'launcher_receipt.json',receipt)
 try:
  with (output/'native_console.txt').open('wb') as console:
   result=subprocess.run(command,cwd=output,stdout=console,stderr=subprocess.STDOUT,check=False)
  receipt['returncode']=result.returncode
  oracle=read(output/'oracle_results.json') if (output/'oracle_results.json').is_file() else None
  assert result.returncode==0,'Native source or case failure retained'
  assert oracle and oracle['receipt']['case_errors']==0 and oracle['receipt']['fixture_count']==12
  assert all(row['status']=='EXECUTED_NATIVE_SOURCE' for row in oracle['results'])
  assert [row['id'] for row in oracle['results']]==[row['id'] for row in fixture['cases']]
  assert oracle['receipt']['runtime']=='11.3.0'
  receipt.update(status='EXECUTED_12_NATIVE_STRESS_CASES_ONLY',native_case_errors=0)
 except Exception as error:
  receipt.update(status='FAILED_STOPPED_PRESERVED',error=repr(error));raise
 finally:
  receipt['outputs']={name:ident(output/name) for name in ('oracle_results.mat','oracle_results.json','native_console.txt') if (output/name).is_file()}
  receipt['finished_utc']=datetime.now(timezone.utc).isoformat();write(output/'launcher_receipt.json',receipt)
  files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.iterdir()) if p.is_file()}
  write(output/'manifest.json',{'schema':'rampq.gate1.full_ap_native_stress_output_manifest/v1','files':files})
 print(receipt['status'],flush=True)


if __name__=='__main__':main()
