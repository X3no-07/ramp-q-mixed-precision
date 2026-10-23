"""Preserve v1/v2, enforce observed CPU thread settings after Trainer construction."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    assert not p.exists(),f'Do not overwrite frozen evidence: {p}'
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
old=json.loads((HERE/'preparation_manifest_v2.json').read_text(encoding='utf-8'))
for rel,info in old['files'].items():
    p=ROOT/rel
    assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],rel
contract=deepcopy(json.loads((HERE/'training_contract_v2.json').read_text(encoding='utf-8')))
contract['overrides']['project']=str(HERE/'attempt_03')
contract['effective_CPU_settings_policy']={
 'intraop_threads':1,'interop_threads':1,
 'reapply_intraop_after_actual_Trainer_constructor':True,
 'observe_and_assert_at':['after_constructor','on_train_start','each_optimizer_step','save_before_capture','final_receipt'],
 'record_actual_backend_flags':True,'cross_thread_output_equivalence_claimed':False}
write(HERE/'training_contract_v3.json',contract)
files={rel:info for rel,info in old['files'].items() if not rel.endswith(('/run_training_v2.py','/training_contract_v2.json','/prepare_training_v2.py'))}
for p in (HERE/'prepare_training_v3.py',HERE/'run_training_v3.py',HERE/'training_contract_v3.json',HERE/'preparation_manifest_v2.json'):
    files[p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
write(HERE/'preparation_manifest_v3.json',{'schema':'rampq.gate1.synthetic_training_preparation/v3','files':files,
 'prior_attempts_preserved':['attempt_01','attempt_02'],
 'prior_attempt_02_final_execution_acceptance':False,
 'reason':'Trainer select_device(cpu) reset intraop threads from requested1 to8; v2 hardcoded thread receipt was inaccurate; exact independent1-thread output mismatched but8-thread matched',
 'exact_output_guard_relaxed':False})
print(json.dumps({'status':'FROZEN_EFFECTIVE_THREAD_ENFORCED_TRAINER_V3','files':len(files),
 'contract_sha256':sha(HERE/'training_contract_v3.json'),
 'manifest_sha256':sha(HERE/'preparation_manifest_v3.json')},indent=2))
