"""Build one small native FP16 precision/output-binding adoption probe."""
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2)+'\n')


def main():
    import tensorrt as trt
    assert platform.system()=='Linux' and platform.machine()=='aarch64'
    assert trt.__version__.startswith('8.5.')
    spec=json.loads((ROOT/'spec.json').read_text())
    assert sha(ROOT/'model.onnx')==spec['model_sha256']
    assert subprocess.run(['pgrep','-x','trtexec'],stdout=subprocess.PIPE).returncode==1
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    assert parser.parse((ROOT/'model.onnx').read_bytes()), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    names=[network.get_layer(i).name for i in range(network.num_layers)]
    assert names.count(spec['conv_name'])==1
    conv_name=spec['conv_name']
    command=['/usr/src/tensorrt/bin/trtexec','--onnx='+str(ROOT/'model.onnx'),
        '--saveEngine='+str(ROOT/'model.engine'),'--fp16','--noTF32','--precisionConstraints=obey',
        '--layerPrecisions='+conv_name+':fp16','--layerOutputTypes='+conv_name+':fp16',
        '--inputIOFormats=fp32:chw','--outputIOFormats=fp32:chw,fp32:chw',
        '--avgTiming=1','--memPoolSize=workspace:1024','--profilingVerbosity=detailed','--buildOnly']
    save('request.json',{'command':command,'max_build_seconds':180,'spec_sha256':sha(ROOT/'spec.json'),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'board_time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'scope':'Small build/adoption diagnostic only; no model inference or performance evidence'})
    started=time.monotonic()
    with (ROOT/'trtexec.log').open('w') as handle:
        # Inherit the monitor's private process group so a safety stop also stops trtexec.
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
        try:
            code=p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGKILL):
                if p.poll() is not None:break
                p.send_signal(sig)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            save('result.json',{'status':'PROBE_TIMEOUT_INCONCLUSIVE','elapsed_seconds':time.monotonic()-started})
            return
    if code:
        save('result.json',{'status':'PROBE_BUILD_FAILED','returncode':code,'elapsed_seconds':time.monotonic()-started})
        return
    runtime=trt.Runtime(logger)
    engine=runtime.deserialize_cuda_engine((ROOT/'model.engine').read_bytes())
    assert engine is not None
    bindings=[{'name':engine.get_binding_name(i),'input':bool(engine.binding_is_input(i)),
        'shape':list(engine.get_binding_shape(i)),'dtype':str(engine.get_binding_dtype(i))} for i in range(engine.num_bindings)]
    layers=json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    save('engine_layers.json',layers)
    convs=[layer for layer in layers['Layers'] if 'Convolution' in layer.get('LayerType','') and conv_name in layer['Name']]
    checks={'all_external_named_bindings_exact':sorted(bindings,key=lambda b:b['name'])==sorted(spec['expected_bindings'],key=lambda b:b['name']),
        'one_native_conv':len(convs)==1}
    if len(convs)==1:
        conv=convs[0]
        checks.update(weights_half=conv.get('Weights',{}).get('Type')=='Half',
            inputs_half=bool(conv.get('Inputs')) and all('FP16' in v.get('Format/Datatype','').upper() for v in conv['Inputs']),
            outputs_half=bool(conv.get('Outputs')) and all('FP16' in v.get('Format/Datatype','').upper() for v in conv['Outputs']))
    save('result.json',{'status':'NATIVE_FP16_CONSTRAINT_ADOPTION_PASS' if all(checks.values()) else 'NATIVE_FP16_ADOPTION_NOT_CONFIRMED',
        'checks':checks,'bindings':bindings,'convolution_layers':convs,'engine_sha256':sha(ROOT/'model.engine'),
        'elapsed_seconds':time.monotonic()-started,'tensorrt':trt.__version__,
        'inference_run':False})
    print((ROOT/'result.json').read_text(),flush=True)


if __name__=='__main__':
    main()
