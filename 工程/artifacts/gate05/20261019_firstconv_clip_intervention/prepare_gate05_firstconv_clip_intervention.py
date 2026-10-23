"""Prepare one exact first-Conv QDQ copy and its isolated identity-Clip control.

The complete model is never run or changed. CPU executes only the two small
Conv+SiLU graphs on the same already frozen input, with ORT_DISABLE_ALL.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXED_IMAGE_SHA256 = '494bbd9d09de29f3a992d434da03c252b4f6cf1f28099fd5a443d0c797f5ec52'
ARCHIVE_ROOT = 'gate05_firstconv_clip_intervention_20261019'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def message_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')


def source_record(paths):
    return {str(path.resolve().relative_to(ROOT)).replace('\\','/'): info(path) for path in sorted(set(paths))}


def summary(value):
    import numpy as np
    finite = np.isfinite(value)
    selected = value[finite]
    return {'shape':list(value.shape),'dtype':str(value.dtype),'all_finite':bool(finite.all()),
            'nonfinite_count':int((~finite).sum()),'min':float(selected.min()) if selected.size else None,
            'max':float(selected.max()) if selected.size else None,
            'raw_float32_bytes_sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/gate05/20261019_firstconv_clip_intervention')
    args=parser.parse_args()
    output=args.output.resolve()
    require(not output.exists() and not output.with_suffix('.zip').exists(),'Choose a fresh output')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort

    previous=ROOT/'artifacts/gate05/20261019_firstconv_numeric_probe'
    previous_spec=read(previous/'spec.json')
    previous_manifest=read(previous/'manifest.json')
    sources=[previous/'manifest.json',Path(__file__).resolve()]
    for name,expected in previous_manifest['files'].items():
        path=previous/name
        require(path.resolve().parent==previous.resolve() and not path.is_symlink() and info(path)==expected,
                'Frozen previous probe source differs: '+name)
        sources.append(path)
    require(previous_spec['schema']=='rampq.gate05.firstconv_numeric_probe/v1' and
            previous_spec['original_complete_model_numerical_guard']=='FAILED_UNCHANGED','Previous probe scope differs')
    original_image=ROOT/'artifacts/jetson_fp16_smoke_20261016_v1/000020.jpg'
    require(sha(original_image)==previous_spec['source_image_sha256']==FIXED_IMAGE_SHA256,'Fixed source image differs')
    sources.append(original_image)
    source_path=previous/previous_spec['variants']['qdq']['model_file']
    require(sha(source_path)==previous_spec['variants']['qdq']['model_sha256'],'Original small QDQ graph differs')
    require(sha(previous/'preprocessed_input.npy')==previous_spec['input_file_sha256'],'Frozen input differs')
    before=source_record(sources)
    model=onnx.load(str(source_path))
    by_name={node.name:node for node in model.graph.node}
    producers={name:node for node in model.graph.node for name in node.output}
    constants={value.name:value for value in model.graph.initializer}
    conv=by_name[previous_spec['selected_conv_name']]
    dq=producers[conv.input[0]]
    q=producers[dq.input[0]]
    clip=producers[q.input[0]]
    require(conv.op_type=='Conv' and dq.op_type=='DequantizeLinear' and q.op_type=='QuantizeLinear' and
            clip.op_type=='Clip' and len(clip.input)==3,'Frozen first Conv activation path differs')
    require(len(model.graph.node)==8 and len(model.graph.initializer)==8,'Expected exact prior 8-node/8-constant micrograph')
    input_name=model.graph.input[0].name
    require(len(model.graph.input)==1 and clip.input[0]==input_name==previous_spec['input_binding']['name'],
            'Original activation Clip does not act directly on the fixed graph input')
    removed_constants=set(clip.input[1:])
    require(removed_constants<=set(constants) and len(removed_constants)==2,'Unexpected Clip bound inventory')
    require(not any(name in node.input for node in model.graph.node if node.name!=clip.name for name in removed_constants),
            'Clip bounds are used by another node')
    no_clip=copy.deepcopy(model)
    kept_nodes=[copy.deepcopy(node) for node in model.graph.node if node.name!=clip.name]
    new_q=next(node for node in kept_nodes if node.name==q.name)
    new_q.input[0]=input_name
    del no_clip.graph.node[:]
    no_clip.graph.node.extend(kept_nodes)
    kept_constants=[copy.deepcopy(value) for value in model.graph.initializer if value.name not in removed_constants]
    del no_clip.graph.initializer[:]
    no_clip.graph.initializer.extend(kept_constants)
    require(len(no_clip.graph.node)==7 and len(no_clip.graph.initializer)==6,'Unexpected no-Clip graph inventory')
    require(all(node.SerializeToString()==by_name[node.name].SerializeToString() for node in no_clip.graph.node if node.name!=q.name),
            'A retained node other than the Q input edge changed')
    require(all(value.SerializeToString()==constants[value.name].SerializeToString() for value in no_clip.graph.initializer),
            'A retained weight/bias/quantizer constant changed')
    require([value.SerializeToString() for value in model.graph.input]==[value.SerializeToString() for value in no_clip.graph.input] and
            [value.SerializeToString() for value in model.graph.output]==[value.SerializeToString() for value in no_clip.graph.output],
            'Input/output declarations or order changed')
    # Undo just the declared deletion/rewire, then compare the entire ModelProto.
    restored=copy.deepcopy(no_clip)
    restored_nodes={node.name:copy.deepcopy(node) for node in restored.graph.node}
    restored_nodes[q.name].input[0]=q.input[0]
    restored_nodes[clip.name]=copy.deepcopy(clip)
    del restored.graph.node[:]
    restored.graph.node.extend([restored_nodes[node.name] for node in model.graph.node])
    restored_constants={value.name:copy.deepcopy(value) for value in restored.graph.initializer}
    restored_constants.update({name:copy.deepcopy(constants[name]) for name in removed_constants})
    del restored.graph.initializer[:]
    restored.graph.initializer.extend([restored_constants[value.name] for value in model.graph.initializer])
    require(restored.SerializeToString()==model.SerializeToString(),'Undeclared model changes exist beyond the Clip intervention')
    onnx.checker.check_model(model,full_check=True)
    onnx.checker.check_model(no_clip,full_check=True)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    shutil.copy2(previous/'preprocessed_input.npy',output/'preprocessed_input.npy')
    shutil.copy2(previous/'original_failed_preflight.json',output/'original_failed_preflight.json')
    shutil.copy2(source_path,output/'model_firstconv_clip.onnx')
    onnx.save(no_clip,str(output/'model_firstconv_no_clip.onnx'))
    require((output/'model_firstconv_clip.onnx').read_bytes()==source_path.read_bytes(),'Clip graph is not a byte-for-byte copy')
    require((output/'preprocessed_input.npy').read_bytes()==(previous/'preprocessed_input.npy').read_bytes(),'Input copy differs')
    write(output/'source_hashes_before.json',before)
    x=np.load(output/'preprocessed_input.npy',allow_pickle=False)
    lower=numpy_helper.to_array(constants[clip.input[1]])
    upper=numpy_helper.to_array(constants[clip.input[2]])
    require(x.dtype==np.float32 and x.shape==(1,3,640,640) and x.flags.c_contiguous and bool(np.isfinite(x).all()) and
            float(x.min())>=0 and float(x.max())<=1,'Input is not the frozen finite [0,1] array')
    require(lower.dtype==upper.dtype==np.float32 and lower.ndim==upper.ndim==0 and float(lower)==-1 and float(upper)==1,
            'Original exact [-1,+1] Clip bounds differ')
    identity=np.clip(x,lower,upper).tobytes()==x.tobytes()
    bindings=copy.deepcopy(previous_spec['output_bindings_in_order'])
    require(len(bindings)==2 and [value.name for value in model.graph.output]==[row['name'] for row in bindings],
            'Original pre/post SiLU output order differs')
    actual,variants,execution={},{},{}
    for variant in ('clip','no_clip'):
        filename='model_firstconv_'+variant+'.onnx'
        options=ort.SessionOptions()
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads=4
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        session=ort.InferenceSession(str(output/filename),sess_options=options,providers=['CPUExecutionProvider'])
        require(session.get_providers()==['CPUExecutionProvider'] and [value.name for value in session.get_inputs()]==[input_name] and
                [value.name for value in session.get_outputs()]==[row['name'] for row in bindings],'CPU provider/I/O differs')
        values=session.run([row['name'] for row in bindings],{input_name:x})
        execution[variant]={'providers':session.get_providers(),'graph_optimization':'ORT_DISABLE_ALL','runs':1,'outputs':{}}
        expected_keys={}
        for row,value in zip(bindings,values):
            key=variant+'_'+row['role']
            # Save raw arrays even if a later exact/finite check fails.
            np.save(output/('cpu_actual_'+key+'.npy'),value)
            actual[key]=value.copy()
            expected_keys[row['name']]=key
            execution[variant]['outputs'][row['name']]=dict(summary(value),npz_key=key,
                                                           raw_file='cpu_actual_'+key+'.npy')
        variants[variant]={'model_file':filename,'model_sha256':sha(output/filename),
                           'nodes':8 if variant=='clip' else 7,'initializers':8 if variant=='clip' else 6,
                           'expected_output_npz_keys':expected_keys}
        del session
    np.savez_compressed(output/'cpu_expected_outputs.npz',**actual)
    checks={'fixed_input_clip_byte_identity':identity,
            'clip_model_file_byte_identical_to_prior':sha(output/'model_firstconv_clip.onnx')==sha(source_path),
            'no_undeclared_model_changes_after_reversing_intervention':True,
            'post_silu_cpu_bit_exact':actual['clip_post_silu'].tobytes()==actual['no_clip_post_silu'].tobytes(),
            'pre_silu_cpu_bit_exact':actual['clip_pre_silu'].tobytes()==actual['no_clip_pre_silu'].tobytes(),
            'all_outputs_expected_float32_shape_finite':all(value.dtype==np.float32 and value.shape==(1,16,320,320) and
                                                           bool(np.isfinite(value).all()) for value in actual.values())}
    with np.load(previous/previous_spec['cpu_expected_outputs_file'],allow_pickle=False) as reference:
        prior_replay={role:actual['clip_'+role].tobytes()==reference['qdq_'+role].tobytes() for role in ('pre_silu','post_silu')}
    write(output/'cpu_diagnostic.json',{'schema':'rampq.gate05.firstconv_clip_intervention_cpu/v1',
                                       'status':'PAIRED_CPU_BITEXACT_EXPECTATIONS_SAVED' if all(checks.values()) else 'CPU_REVIEW_REQUIRED',
                                       'checks':checks,'prior_original_clip_cpu_replay_bit_exact':prior_replay,
                                       'onnx':onnx.__version__,'onnxruntime':ort.__version__,'numpy':np.__version__,
                                       'fixed_input':summary(x),'cpu_execution':execution,
                                       'complete_detector_executions':0,'cuda_used':False,'native_execution':False,
                                       'original_complete_model_numerical_guard':'FAILED_UNCHANGED'})
    change={'removed_clip_node':clip.name,'removed_initializer_names':[clip.input[1],clip.input[2]],
            'rewired_q_node':q.name,'rewired_input_index':0,'original_q_input':q.input[0],'new_q_input':input_name,
            'original_q_serialized_sha256':message_sha(q),'rewired_q_serialized_sha256':message_sha(new_q),
            'unchanged_remaining_node_serialized_sha256':{node.name:message_sha(node) for node in no_clip.graph.node if node.name!=q.name},
            'unchanged_remaining_initializer_serialized_sha256':{value.name:message_sha(value) for value in no_clip.graph.initializer},
            'input_output_proto_bytes_and_order_unchanged':True,'all_other_ModelProto_fields_unchanged':True}
    write(output/'model_change_audit.json',change)
    probe_spec={'schema':'rampq.gate05.firstconv_clip_intervention/v1','prepared_client_at':datetime.now().astimezone().isoformat(),
                'diagnostic_only':True,'source_configuration':'q_only_G1',
                'source_firstconv_probe_spec_sha256':sha(previous/'spec.json'),
                'source_firstconv_probe_manifest_sha256':sha(previous/'manifest.json'),
                'source_original_qdq_micrograph_sha256':sha(source_path),
                'source_full_qonly_graph_sha256':previous_spec['source_qonly_graph_sha256'],
                'source_image':'000020.jpg','source_image_sha256':FIXED_IMAGE_SHA256,
                'input_file':'preprocessed_input.npy','input_file_sha256':sha(output/'preprocessed_input.npy'),
                'input_array_original_bytes_copied':True,'input_binding':copy.deepcopy(previous_spec['input_binding']),
                'output_bindings_in_order':bindings,'variants':variants,
                'selected_conv_name':conv.name,'selected_silu_node_names':previous_spec['selected_silu_node_names'],
                'selected_clip_node_name':clip.name,'selected_activation_q_node_name':q.name,
                'first_conv_source_mapping':copy.deepcopy(previous_spec['first_conv_source_mapping']),
                'cpu_expected_outputs_file':'cpu_expected_outputs.npz','cpu_expected_outputs_sha256':sha(output/'cpu_expected_outputs.npz'),
                'cpu_diagnostic_sha256':sha(output/'cpu_diagnostic.json'),'model_change_audit_sha256':sha(output/'model_change_audit.json'),
                'original_failed_preflight_sha256':sha(output/'original_failed_preflight.json'),
                'fixed_input_clip_identity_proved':identity,'paired_cpu_outputs_bit_exact':checks['pre_silu_cpu_bit_exact'] and checks['post_silu_cpu_bit_exact'],
                'original_complete_model_numerical_guard':'FAILED_UNCHANGED','calibration_recomputed':False,
                'original_tolerances_changed':False,'new_complete_model_criterion_defined':False,'dataset_evaluation_run':False,
                'native_build_or_execution_run_by_preparer':False,'pre_silu_output_tap_may_change_native_fusion':True,
                'limits':'Experimental paired first-Conv micrographs only. no_clip removes an identity only on this fixed input; '
                         'it does not modify or repair the original complete candidate, establish full-model numerical migration, '
                         'strict native arithmetic, target rescue, accuracy, causality or performance.'}
    write(output/'spec.json',probe_spec)
    after=source_record(sources)
    write(output/'source_hashes_after.json',after)
    require(before==after,'Frozen source files changed during preparation')
    (output/'README.md').write_text('''# 首Conv恒等Clip配对干预

clip版为上一首Conv QDQ微图的完整文件原字节副本；no_clip仅删除首Clip和两个未用
边界常量，把原ActivationQ的input[0]改为images。其余Conv、SiLU、weight/bias、
scale/zero、输出声明和顺序完全不变。反向补回三项改动可恢复整个原ModelProto字节。
所有具体改动、原/新Q节点和保留节点/常量SHA均记录在model_change_audit.json。

原固定000020输入npy逐字节复制，FLOAT images[1,3,640,640]。
原Clip上下界严格为[-1,+1]，该输入属于[0,1]，Clip前后原字节相同。
主post-SiLU和辅助pre-SiLU输出保持原名字，均FLOAT[1,16,320,320]。
CPUExecutionProvider、ORT_DISABLE_ALL各执行一个小图，先保存原始数组，再核对
两图两个输出逐元素bitexact。CPU expected NPZ键和接口在spec.json。

只为验证同native选项下去除恒等Clip是否改变这一孤立首Conv的数值，不能表述为
完整模型修复、数值迁移通过、strict算术证明或准确率/性能结果。没有重校准、
更改原标准、运行完整检测器、CUDA、SSH或150图评估。原完整数值失败仍保留。
''',encoding='utf-8')
    files={path.name:info(path) for path in sorted(output.iterdir()) if path.is_file()}
    write(output/'manifest.json',{'schema':'rampq.gate05.firstconv_clip_intervention_bundle/v1','files':files})
    (output/'SHA256SUMS').write_text(''.join(value['sha256']+'  '+name+'\n' for name,value in files.items()),encoding='utf-8')
    with zipfile.ZipFile(output.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():archive.write(path,ARCHIVE_ROOT+'/'+path.name)
    require(all(checks.values()),'CPU paired exact checks failed; original arrays and diagnostics are retained')
    print(json.dumps({'status':'FIRSTCONV_CLIP_INTERVENTION_PREPARED_CPU_BITEXACT','output':str(output),
                      'archive':str(output.with_suffix('.zip')),'archive_bytes':output.with_suffix('.zip').stat().st_size,
                      'archive_sha256':sha(output.with_suffix('.zip')),'spec_sha256':sha(output/'spec.json'),
                      'variants':variants,'checks':checks,'prior_clip_cpu_replay_bit_exact':prior_replay,
                      'source_hashes_unchanged':True,'complete_detector_executions':0,'native_execution':False},indent=2))


if __name__=='__main__':
    main()
