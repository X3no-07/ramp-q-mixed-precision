"""Sequential, time-limited builds of the exact failed depthwise Conv slice."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--only',choices=('conv_only','conv_add','conv_add_tap'))
    parser.add_argument('--timeout',type=int,default=90)
    parser.add_argument('--result-name',default='results')
    args=parser.parse_args()
    root=args.root
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    assert subprocess.run(['pgrep','-x','trtexec'],stdout=subprocess.PIPE).returncode==1
    import tensorrt as trt
    assert trt.__version__.startswith('8.5.')
    result=(root/args.result_name).resolve()
    assert result.parent==root.resolve() and 0<args.timeout<=180
    result.mkdir(exist_ok=False)
    rows=[]
    logger=trt.Logger(trt.Logger.WARNING); trt.init_libnvinfer_plugins(logger,'')
    runtime=trt.Runtime(logger)
    for name in ((args.only,) if args.only else ('conv_only','conv_add','conv_add_tap')):
        engine_path=result/(name+'.engine'); log_path=result/(name+'.log')
        command=['/usr/src/tensorrt/bin/trtexec','--onnx='+str(root/(name+'.onnx')),'--saveEngine='+str(engine_path),
                 '--int8','--fp16','--noTF32','--memPoolSize=workspace:1024','--profilingVerbosity=detailed','--buildOnly']
        started=time.monotonic(); timed_out=False
        with log_path.open('w') as log:
            try:
                code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code=None; timed_out=True
        row={'variant':name,'command':command,'returncode':code,'timeout':timed_out,'max_seconds':args.timeout,'seconds':round(time.monotonic()-started,3)}
        if code==0 and engine_path.is_file():
            engine=runtime.deserialize_cuda_engine(engine_path.read_bytes()); assert engine is not None
            inspector=engine.create_engine_inspector()
            text=inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            (result/(name+'_layers.json')).write_text(text)
            row['convolutions']=[l for l in json.loads(text)['Layers'] if 'Convolution' in l.get('LayerType','')]
            del inspector,engine
        else:
            row['errors']=[line for line in log_path.read_text(errors='replace').splitlines() if '[E]' in line][-8:]
        rows.append(row)
        (result/'summary.json').write_text(json.dumps({'tensorrt':trt.__version__,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'results':rows},indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='convolutions'}),flush=True)
    print('PE_MINIMAL_REPRO_DONE',flush=True)


if __name__=='__main__':main()
