"""Freeze a hardware feasibility diagnostic without changing the failed parity test."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / 'artifacts/gate05/20261018_int8'
    source = run / 'input_v1'
    bundle = run / 'diagnostic_bundle_v1'
    preflight = json.loads((source/'preflight.json').read_text(encoding='utf-8'))
    isolated = json.loads((run/'isolated_quantizer_audit.json').read_text(encoding='utf-8'))
    assert preflight['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert isolated['weight_qdq_all_88_exact'] and len(isolated['same_input_quantizer']) == 88
    assert all(r['different'] == 0 for r in isolated['same_input_quantizer'])
    bundle.mkdir(parents=True,exist_ok=False)
    for path in source.iterdir():
        if path.is_file(): shutil.copy2(path,bundle/path.name)
    for name in ('isolated_quantizer_audit.json','local_divergence_diagnostic.json'):
        shutil.copy2(run/name,bundle/name)
    for name in ('jetson_build_qdq.py','jetson_monitored_build.py'):
        path=ROOT/'scripts'/name
        ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
        shutil.copy2(path,bundle/name)
    scope={'schema':'rampq.gate05.int8_diagnostic_scope/v1',
           'status':'DIAGNOSTIC_BUILD_ONLY_FULL_PARITY_FAILED','client_date':'2026-10-18',
           'failed_preflight_sha256':sha(bundle/'preflight.json'),
           'isolated_audit_sha256':sha(bundle/'isolated_quantizer_audit.json'),
           'reason':'All 88 same-input activation quantizers and all weight Q/DQ tensors agree exactly. '
                    'First FP32 Conv backend error is 1.621246337890625e-5; the next quantizer has 3 differing elements. '
                    'Investigate real INT8 kernel feasibility while retaining failed full-model parity.',
           'permitted_conclusion':'Only hardware build, actual kernel precision, and fixed-image execution facts',
           'not_established':'Numerical migration parity, dataset accuracy, recovery mechanism, or performance',
           'original_tolerances_changed':False,'dataset_evaluation_run':False}
    (bundle/'diagnostic_scope.json').write_text(json.dumps(scope,indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in bundle.iterdir() if p.is_file() and p.name!='manifest.json')
    manifest={'schema':'rampq.gate05.qdq_inputs/v1','configuration':'q_all','diagnostic_only':True,
              'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(run/'diagnostic_bundle_v1.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.iterdir()):
            if path.is_file(): archive.write(path,'gate05_int8_diagnostic_20261018_v1/'+path.name)
    print(json.dumps({'status':scope['status'],'bundle_sha256':sha(run/'diagnostic_bundle_v1.zip'),
                      'bytes':(run/'diagnostic_bundle_v1.zip').stat().st_size,'files':len(manifest['files'])},indent=2))


if __name__=='__main__': main()
