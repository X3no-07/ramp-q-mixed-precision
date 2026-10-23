"""Preserve v1, bind the process-isolated v2 runner and a NEW attempt path."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    assert not p.exists(), f'Never overwrite frozen evidence: {p}'
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
old=json.loads((HERE/'preparation_manifest.json').read_text(encoding='utf-8'))
for rel,info in old['files'].items():
    p=ROOT/rel
    assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],rel
contract=deepcopy(json.loads((HERE/'training_contract.json').read_text(encoding='utf-8')))
contract['overrides']['project']=str(HERE/'attempt_02')
contract['process_isolation']={'external_integration_registration':'explicit_process_local_noop_before_Trainer_ctor',
 'only_framework_base_callbacks_before_custom_hooks':True,'framework_files_modified':False,
 'network_isolation_claim':'NO_INTENDED_NETWORK_CODE_PATHS_NOT_OS_SANDBOX_PROOF'}
write(HERE/'training_contract_v2.json',contract)
files={rel:info for rel,info in old['files'].items() if not rel.endswith(('/run_training.py','/training_contract.json','/prepare_training.py'))}
for p in (HERE/'prepare_training_v2.py',HERE/'run_training_v2.py',HERE/'training_contract_v2.json',HERE/'preparation_manifest.json'):
    files[p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
write(HERE/'preparation_manifest_v2.json',{'schema':'rampq.gate1.synthetic_training_preparation/v2','files':files,
 'prior_functional_attempt':'attempt_01','prior_attempt_superseded_for_final_execution_acceptance':True,
 'reason':'v1 disabled sync/loggers but did not suppress framework external platform callback registration',
 'does_not_establish_external_connection_or_upload_occurred_in_v1':True})
print(json.dumps({'status':'FROZEN_PROCESS_ISOLATED_SYNTHETIC_TRAINER_V2','files':len(files),
 'contract_sha256':sha(HERE/'training_contract_v2.json'),
 'manifest_sha256':sha(HERE/'preparation_manifest_v2.json')},indent=2))
