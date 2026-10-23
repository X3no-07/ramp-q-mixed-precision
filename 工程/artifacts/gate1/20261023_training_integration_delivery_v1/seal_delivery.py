"""Seal the new delivery and verify all explicitly listed new member bindings."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REVIEW=ROOT/'artifacts/gate1/20261023_training_integration_review_v1'
MANIFEST=HERE/'delivery_manifest.json'
VALIDATION=HERE/'delivery_validation.json'
assert not MANIFEST.exists() and not VALIDATION.exists(),'Never overwrite a sealed delivery'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def require(v,msg):
    if not v:raise ValueError(msg)
def identity(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}

prior=ROOT/'artifacts/gate1/20261023_full_ap_delivery_v1/delivery_manifest.json'
require(sha(prior)=='44e8a7f163445320f26a19adb855970dcab4451a9f8ef62379c30360d6ef4c81','Previous manifest changed')
prior_records=read(prior)['files']
for rel,expected in prior_records.items():
    require(identity(ROOT/rel)==expected,f'Older frozen evidence changed: {rel}')

members={}
listed_manifests=[]
def add(path):
    path=Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(),'Only explicit workspace files are sealed')
    members[path.relative_to(ROOT).as_posix()]=identity(path)

def verify_and_add_manifest(path,base):
    path=Path(path)
    data=read(path)
    add(path)
    count=0
    for rel,expected in data['files'].items():
        target=base/rel
        require(identity(target)==expected,f'Nested manifest mismatch: {path.name}: {rel}')
        add(target)
        count+=1
    for key in ('external_sources','workspace_source_bindings'):
        for rel,expected in data.get(key,{}).items():
            require(identity(ROOT/rel)==expected,f'Source binding mismatch: {rel}')
            add(ROOT/rel)
    listed_manifests.append({'path':path.relative_to(ROOT).as_posix(),'members_checked':count,'sha256':sha(path)})

component_dirs=[ROOT/'artifacts/gate1'/name for name in (
 '20261023_ignore_dataset_integration_v1','20261023_ignore_trainer_integration_v1','20261023_fp32_live_capture_v1')]
for directory in component_dirs:
    verify_and_add_manifest(directory/'manifest.json',directory)
    # Their nested capture/state manifests are separately checked, not inferred
    # from the outer manifest's identity alone.
    data=read(directory/'manifest.json')
    for rel in data['files']:
        p=directory/rel
        if p.name=='manifest.json' and isinstance(read(p).get('files'),dict):
            verify_and_add_manifest(p,p.parent)

for version in ('','_v2','_v3'):
    verify_and_add_manifest(HERE/f'preparation_manifest{version}.json',ROOT)
for attempt in ('attempt_01','attempt_02','attempt_03'):
    directory=HERE/attempt
    verify_and_add_manifest(directory/'manifest.json',directory)
    capture=directory/'fp32_capture/epoch_0001'
    verify_and_add_manifest(capture/'manifest.json',capture)
    verify_and_add_manifest(capture/'state/manifest.json',capture/'state')

for p in HERE.rglob('*'):
    if p.is_file() and p not in (MANIFEST,VALIDATION):add(p)
for directory in (REVIEW,*(ROOT/'artifacts/gate1'/f'20261023_training_runner_source_review_v{v}' for v in (1,2,3))):
    for p in directory.rglob('*'):
        if p.is_file():add(p)
for p in REVIEW.rglob('manifest.json'):
    if isinstance(read(p).get('files'),dict):verify_and_add_manifest(p,p.parent)

for p in (prior,ROOT/'50_Gate1_合成训练管线与FP32保存接入验证.md',ROOT/'artifacts/gate1/current_execution_state_v11.json'):
    add(p)
pointer=read(ROOT/'artifacts/gate1/latest_execution_state.json')
require(pointer['snapshot']=='artifacts/gate1/current_execution_state_v11.json'
        and pointer['snapshot_sha256']==sha(ROOT/pointer['snapshot']),'Latest pointer mismatch')
require(read(REVIEW/'synthetic_output_audit_v3/audit_result.json')['status'].startswith('PASS'),'Independent output audit failed')
require(read(REVIEW/'final_metadata_review.json')['status'].startswith('PASS'),'Final metadata review failed')
manifest={'schema':'rampq.gate1.synthetic_training_integration_delivery/v1',
 'at_utc':datetime.now(timezone.utc).isoformat(),
 'status':'COMPLETED_ACTUAL_SYNTHETIC_CPU_TRAINER_IGNORE_AND_PRE_HALF_FP32_CAPTURE_ONLY',
 'accepted_attempt':'attempt_03','file_base':'engineering_project_root','files':dict(sorted(members.items())),
 'nested_manifests_checked':listed_manifests,
 'previous_79_member_delivery_unchanged':True,'previous_manifest_sha256':sha(prior),
 'mutable_latest_state_pointer_excluded_from_seal':True,
 'real_data_training_or_AP_accepted':False,'gate1_scientific_pass':False,
 'original_gate1_dtype_guard':'FAIL_FIXED_TOLERANCES','old_gate05_guard':'FAILED_UNCHANGED',
 'holdout_opened_or_listed':False,'board_or_CUDA_capture_qualified':False}
write(MANIFEST,manifest)
for rel,expected in manifest['files'].items():
    require(identity(ROOT/rel)==expected,f'Final delivery changed during sealing: {rel}')
write(VALIDATION,{'schema':'rampq.gate1.synthetic_training_delivery_validation/v1',
 'status':'PASS_EXPLICIT_DELIVERY_FILES_NESTED_BINDINGS_AND_PRIOR_DELIVERY_UNCHANGED_ONLY',
 'files_verified':len(members),'nested_manifests_checked':len(listed_manifests),
 'total_bytes':sum(v['bytes'] for v in members.values()),'prior_files_reverified':len(prior_records),
 'manifest_sha256':sha(MANIFEST),'pointer_snapshot_sha256':pointer['snapshot_sha256'],
 'acceptance_scope':'actual synthetic CPU integration and independent synthetic FP32 reconstruction only',
 'metadata_integrity_does_not_establish_scientific_or_board_acceptance':True})
print(json.dumps({'status':'SEALED','files':len(members),'nested_manifests':len(listed_manifests),
 'manifest_sha256':sha(MANIFEST),'validation_sha256':sha(VALIDATION)},indent=2))
