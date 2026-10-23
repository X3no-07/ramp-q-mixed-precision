"""Stage, but do not run, the same candidate with one kernel timing average."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=ROOT/'artifacts/gate05/20261018_int8/diagnostic_bundle_v3_pe_tap'
    out=ROOT/'artifacts/gate05/20261018_int8_pe_tap/diagnostic_bundle_v4_pe_tap_avg1'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():
        assert (source/name).stat().st_size==info['bytes'] and sha(source/name)==info['sha256'],name
    out.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.name not in ('manifest.json','build_revision.json','jetson_build_qdq.py'):
            shutil.copy2(path,out/path.name)
    script=ROOT/'scripts/jetson_build_qdq.py'
    ast.parse(script.read_text(encoding='utf-8'),feature_version=(3,8))
    shutil.copy2(script,out/script.name)
    revision={'revision':'v4_pe_tap_avg1','prepared_client_at':datetime.now().astimezone().isoformat(),
        'status':'STAGED_NOT_BUILT_OR_EXECUTED','configuration':'q_all',
        'change':'Kernel selection averaging --avgTiming=1 instead of native default 8',
        'source_onnx_sha256':sha(out/'model_q_all.onnx'),
        'graph_weights_calibration_quantization_and_numerical_tolerances_unchanged':True,
        'reason':'Previous candidate reached 1800-second supervision budget without engine or compatibility error',
        'max_build_seconds':1800,
        'limits':'Kernel selection may differ. This is only a hardware feasibility diagnostic; no performance or causal comparison claim. Speedup is not established.'}
    (out/'build_revision.json').write_text(json.dumps(revision,indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in out.iterdir() if p.is_file())
    manifest.update(kernel_selection_avg_timing=1,
        files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    destination=out.parent/(out.name+'.zip')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            archive.write(path,'gate05_int8_diagnostic_20261019_v4_pe_tap_avg1/'+path.name)
    print(json.dumps({'status':revision['status'],'sha256':sha(destination),'bytes':destination.stat().st_size,
        'graph_sha256':sha(out/'model_q_all.onnx')},indent=2))


if __name__=='__main__':
    main()
