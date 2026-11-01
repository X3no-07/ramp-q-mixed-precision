"""Collect immutable evidence after the one PE-tap build/optional smoke ends."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import time
import zipfile

BASE=Path('/home/nvidia/rampq')
ROOT=BASE/'gate05_int8_diagnostic_20261018_v3_pe_tap'
MONITOR=BASE/'gate05_int8_monitor3_pe_tap_20261018_c4d8fcff'
SMOKE=BASE/'gate05_int8_pe_tap_smoke_20261018_c4d8fcff'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--monitor',type=Path,default=MONITOR)
    parser.add_argument('--smoke',type=Path,default=SMOKE)
    parser.add_argument('--prefix',default='gate05_pe_tap')
    parser.add_argument('--allow-other-builds',action='store_true',
                        help='Collect a terminal target while a different queued configuration builds')
    args=parser.parse_args()
    root,monitor_dir,smoke_dir=(p.resolve() for p in (args.root,args.monitor,args.smoke))
    assert root.parent==BASE
    assert BASE in monitor_dir.parents and BASE in smoke_dir.parents
    assert args.prefix.replace('_','').isalnum()
    monitor=json.loads((monitor_dir/'status.json').read_text())
    assert monitor['status']!='RUNNING'
    active=subprocess.run(['pgrep','-ax','trtexec'],stdout=subprocess.PIPE,universal_newlines=True)
    assert active.returncode in (0,1)
    if args.allow_other_builds:
        assert str(root)+'/' not in active.stdout, 'Target trtexec is still running'
    else:
        assert active.returncode==1
    latest=json.loads((root/'latest_build.json').read_text())
    build=Path(latest['run_dir']).resolve()
    assert build.parent==root/'results'
    files={}
    for label,folder in (('build',build),('monitor',monitor_dir),('smoke',smoke_dir)):
        if folder.exists():
            for path in sorted(folder.rglob('*')):
                if path.is_file() and '.ultralytics' not in path.parts:
                    files[label+'/'+path.relative_to(folder).as_posix()]=path
    input_manifest=json.loads((root/'manifest.json').read_text())
    for name,info in input_manifest['files'].items():
        path=root/name
        assert path.resolve().parent==root
        assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],name
        # Retain the exact executable request files; large model tensors are already
        # frozen by the copied manifest and the separately hashed input archive.
        if path.suffix in ('.json','.py'):
            files['inputs/'+name]=path
    files['inputs/manifest.json']=root/'manifest.json'
    for name in ('jetson_trt_int8_smoke.py','audit_gate05_int8_precision.py',
                 'jetson_precision_candidate_smoke.py','audit_gate05_precision_candidate.py'):
        if (BASE/name).exists():
            files['tools/'+name]=BASE/name
    end={'board_time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
         'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'build':latest,'monitor':monitor,'other_trtexec_processes':active.stdout,
         'other_builds_allowed_during_collection':args.allow_other_builds}
    for label,command in [('memory',['free','-m']),('jtop',['systemctl','show','jtop.service','-p','ActiveState','-p','NRestarts']),
                          ('fan',['systemctl','is-active','nvfancontrol.service'])]:
        p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=15)
        end[label]={'command':command,'returncode':p.returncode,'output':p.stdout}
    end_path=BASE/(args.prefix+'_end_status.json')
    end_path.write_text(json.dumps(end,indent=2)+'\n')
    files['end_board_status.json']=end_path
    receipt={'schema':'rampq.remote_file_receipt/v1',
        'files':{name:{'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in files.items()}}
    destination=BASE/(args.prefix+'_evidence.zip')
    with zipfile.ZipFile(str(destination),'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in files.items():archive.write(str(path),name)
        archive.writestr('remote_receipt.json',json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'path':str(destination),'bytes':destination.stat().st_size,
        'sha256':sha(destination),'files':len(files)}))


if __name__=='__main__':
    main()
