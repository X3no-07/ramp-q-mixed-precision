"""Prepare one member of a frozen matched native-NONE control family.

Only approved activation FLOAT-Clip/Q pairs and their two bound initializers
are rewritten; all Conv, weight, scale, zero, remaining node and model-shell
bytes are retained. The FP16 control graph is copied without any graph change.
Preparation performs only the protocol's fixed CPU count and never runs native
code. Sealing is separate, exclusive, and unavailable after preparation failure.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'artifacts/gate05/20261021_matched_control_family_source_inventory.json'
INVENTORY_SHA = '3088b68899698fe4bcd2f7f61d368651e63a5da0bc1a72311b2bdc72f008e70b'
PROTOCOL = ROOT / 'artifacts/gate05/20261021_matched_control_family_protocol/protocol.json'
PROTOCOL_SHA = '27af69e97a8c4220653294381c69fbf8fa412044bb09e94d7d888d2355cee98c'
OUTPUT = ROOT / 'artifacts/gate05/20261021_matched_control_family'
CONFIGURATIONS = ('fp16_control', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1')
EXPECTED_QUANTIZED = (0, 82, 74, 77, 6)
INPUT_SHA = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
HELPER_PINS = {
    'jetson_build_precision_candidate.py': 'd8431345da90482f0d5798e2d6524bf03591ecef653edfad5ef3e28a509eb19f',
    'jetson_monitored_build.py': '71ebc77843137c93cf2de6558f5234996330fab54dd900e251912b7b1ddae2a1',
    'jetson_owned_group_guard.py': '4f69d15df8c395af81ceabecdd5fe0da7406df2339095046e55200a388013ae4',
    'jetson_collect_isolated_numeric_probe.py': 'af9b7f823e4f889813fb88b6e8a529a6b1a6423b0c7ead0d5704513b517775eb',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def message_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def hashes(paths):
    result = {}
    for path in sorted(set(paths)):
        require(path.resolve().is_relative_to(ROOT) and path.is_file() and not path.is_symlink(), 'Unsafe protected source')
        result[path.resolve().relative_to(ROOT).as_posix()] = info(path)
    return result


def validate_sources(record):
    for name, expected in record.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and info(path) == expected,
                'Frozen source became stale: ' + name)


def check_parent(folder, row):
    require(sha(folder / 'manifest.json') == row['manifest']['sha256'] and
            sha(folder / 'spec.json') == row['spec']['sha256'] and sha(folder / 'model.onnx') == row['model']['sha256'],
            'Frozen parent identity differs')
    manifest = read(folder / 'manifest.json')
    require({p.name for p in folder.iterdir()} == set(manifest['files']) | {'manifest.json', 'SHA256SUMS'} and
            all(p.is_file() and not p.is_symlink() for p in folder.iterdir()), 'Parent file set changed')
    for name, expected in manifest['files'].items():
        path = folder / name
        require(path.resolve().parent == folder.resolve() and info(path) == expected, 'Parent manifest member changed: ' + name)
    sums = {}
    for line in (folder / 'SHA256SUMS').read_text(encoding='utf-8-sig').splitlines():
        digest, name = line.split('  ', 1)
        require(name not in sums and Path(name).name == name, 'Malformed parent SHA256SUMS')
        sums[name] = digest
    # These original runtime bundles checksum their nine payload files only;
    # their manifest identity is independently pinned by the frozen inventory.
    require(set(sums) == set(manifest['files']) and
            all(sha(folder / name) == digest for name, digest in sums.items()), 'Parent checksum record differs')
    return list(folder.iterdir())


def validate_protocol(protocol_path, configuration):
    require(len(PROTOCOL_SHA) == 64 and sha(protocol_path) == PROTOCOL_SHA, 'Root-frozen family protocol not pinned or changed')
    require(sha(INVENTORY) == INVENTORY_SHA, 'Frozen family inventory changed')
    inventory, protocol = read(INVENTORY), read(protocol_path)
    require(inventory['schema'] == 'rampq.gate05.matched_control_family_source_inventory/v1' and
            protocol['schema'] == 'rampq.gate05.matched_control_family_protocol/v1' and
            protocol['source_inventory_sha256'] == INVENTORY_SHA, 'Wrong family identity')
    require(protocol['configuration_order'] == list(CONFIGURATIONS), 'Frozen family order changed')
    require(protocol['total_new_CPU_model_runs'] == 9 and
            protocol['CPU_run_budget_by_configuration'] == {name: 1 if index == 0 else 2
                for index, name in enumerate(CONFIGURATIONS)}, 'Frozen CPU count differs')
    require(protocol['source_configuration_identities'] == inventory['configurations'] and
            protocol['common_source_provenance'] == inventory['common_source_provenance'] and
            protocol['shared_source_identities'] == inventory['shared_sources'] and
            protocol['quantized_conv_counts'] == dict(zip(CONFIGURATIONS, EXPECTED_QUANTIZED)) and
            protocol['restored_fp16_conv_counts'] == {name: 88-count for name, count in zip(CONFIGURATIONS, EXPECTED_QUANTIZED)},
            'Frozen family source or precision partition differs')
    require(protocol['fixed_input_sha256'] == INPUT_SHA and protocol['input_shape'] == [1, 3, 640, 640] and
            protocol['CPU_provider'] == 'CPUExecutionProvider' and protocol['CPU_graph_optimization'] == 'ORT_DISABLE_ALL' and
            protocol['CPU_intra_op_threads'] == 4 and protocol['original_source_spec_precision_constraints'] == 'obey' and
            protocol['new_family_precision_constraints'] == 'none' and
            protocol['original_source_bundle_directory_and_bytes_unchanged'] is True and
            protocol['protected_status']['original_complete_model_guard'] == 'FAILED_UNCHANGED' and
            protocol['protected_status']['overall_gate_status'] == 'INCOMPLETE', 'Frozen preparation scope differs')
    native = protocol['common_native_settings']
    require(native['precision_constraints'] == 'none' and native['FP16'] is True and native['TF32'] is False and
            native['PREFER_PRECISION_CONSTRAINTS'] is False and native['OBEY_PRECISION_CONSTRAINTS'] is False and
            native['layer_precision_or_output_type_setters_or_resets'] is False and native['workspace_mib'] == 1024 and
            native['avg_timing_iterations'] == 1 and native['profiling_verbosity'] == 'DETAILED' and
            native['all_external_bindings'] == 'FLOAT/LINEAR', 'Frozen planned native family settings differ')
    require(configuration in CONFIGURATIONS, 'Unexpected family member')
    return inventory, protocol


def rewrite(source, source_spec, module_map, count):
    """Construct exact approved code-Clip rewrite; no runtime imports or calls."""
    import numpy as np
    import onnx
    from onnx import numpy_helper
    candidate = copy.deepcopy(source)
    source_nodes = {node.name: node for node in source.graph.node}
    source_constants = {value.name: value for value in source.graph.initializer}
    producers = {name: node for node in source.graph.node for name in node.output}
    consumers = {}
    for node in source.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node.name)
    mapping = {row['onnx_node']: row for row in module_map['mapping']}
    requests = source_spec['requests']
    require(len(requests) == len(mapping) == 88 and {node.name for node in source.graph.node if node.op_type == 'Conv'} == set(mapping),
            'Expected unchanged 88 Conv source identities')
    int8_requests = [row for row in requests if row['precision'] == 'INT8']
    require(len(int8_requests) == count and sum(row['precision'] == 'FP16' for row in requests) == 88-count,
            'Frozen request partition differs')
    patched_nodes = {node.name: node for node in candidate.graph.node}
    patched_constants = {value.name: value for value in candidate.graph.initializer}
    used_names = set(source_constants) | set(producers) | {v.name for v in source.graph.input}
    used_names.update(v.name for v in list(source.graph.output) + list(source.graph.value_info))
    rewrites, clip_to_q, expected_nodes, expected_constants = [], {}, {}, {}
    for request in int8_requests:
        conv, frozen = source_nodes[request['onnx_node']], mapping[request['onnx_node']]
        require(request['module'] == frozen['pytorch_module'] and request['group'] == frozen['group'] and
                request['weight_input'] == frozen['weight_input'], 'Frozen module map differs')
        dq = producers[conv.input[0]]
        q = producers[dq.input[0]]
        clip = producers[q.input[0]]
        require(dq.op_type == 'DequantizeLinear' and q.op_type == 'QuantizeLinear' and clip.op_type == 'Clip' and
                len(q.input) == len(dq.input) == len(clip.input) == 3 and
                clip.input[0] == request['original_activation_input'] and not q.attribute and not dq.attribute and not clip.attribute and
                list(q.input[1:]) == list(dq.input[1:]) and consumers[clip.output[0]] == [q.name] and
                consumers[q.output[0]] == [dq.name], 'Expected isolated scalar FLOAT Clip/Q/DQ source')
        scale = numpy_helper.to_array(source_constants[q.input[1]])
        zero = numpy_helper.to_array(source_constants[q.input[2]])
        lower = numpy_helper.to_array(source_constants[clip.input[1]])
        upper = numpy_helper.to_array(source_constants[clip.input[2]])
        require(scale.dtype == np.float32 and scale.ndim == 0 and float(scale) > 0 and
                zero.dtype == np.int8 and zero.ndim == 0 and int(zero) == 0 and
                scale.tobytes() == np.asarray(frozen['activation_scale'], dtype=np.float32).tobytes() and
                lower.dtype == upper.dtype == np.float32 and lower.ndim == upper.ndim == 0 and
                lower.tobytes() == np.float32(scale*np.float32(-127)).tobytes() and
                upper.tobytes() == np.float32(scale*np.float32(127)).tobytes() and
                int(np.rint(np.float32(lower/scale))) == -127 and int(np.rint(np.float32(upper/scale))) == 127,
                'Source bounds must be exact FLOAT32 product and nearest-even signed codes; no division identity assumption')
        new_name = q.output[0] + '_before_code_clip'
        require(new_name not in used_names, 'Code tensor name collision')
        used_names.add(new_name)
        new_q, new_clip = patched_nodes[q.name], patched_nodes[clip.name]
        new_q.input[0], new_q.output[0] = clip.input[0], new_name
        new_clip.input[0], new_clip.output[0] = new_name, q.output[0]
        expected_q, expected_clip = copy.deepcopy(q), copy.deepcopy(clip)
        expected_q.input[0], expected_q.output[0] = clip.input[0], new_name
        expected_clip.input[0], expected_clip.output[0] = new_name, q.output[0]
        expected_nodes[q.name], expected_nodes[clip.name] = expected_q, expected_clip
        for name, bound in zip(clip.input[1:], (-127,127)):
            require(name not in expected_constants, 'Shared bound initializer not permitted')
            replacement = numpy_helper.from_array(np.asarray(bound,dtype=np.int8),name)
            patched_constants[name].CopyFrom(replacement)
            expected_constants[name] = replacement
        clip_to_q[clip.name] = q.name
        rewrites.append({'module': request['module'], 'conv_name':conv.name, 'group':request['group'],
            'clip_name':clip.name, 'q_name':q.name, 'dq_name_unchanged':dq.name,
            'scale_initializer_unchanged':q.input[1], 'scale_float32':float(scale),
            'zero_initializer_unchanged':q.input[2], 'zero_int8':int(zero),
            'parent_float_clip_bounds':[float(lower),float(upper)], 'code_clip_bounds':[-127,127],
            'bound_initializer_names':list(clip.input[1:]), 'dq_input_and_output_unchanged':True,
            'parent_clip_serialized_sha256':message_sha(clip), 'parent_q_serialized_sha256':message_sha(q),
            'candidate_clip_serialized_sha256':message_sha(new_clip), 'candidate_q_serialized_sha256':message_sha(new_q)})
    reordered = []
    q_names = set(clip_to_q.values())
    for node in source.graph.node:
        if node.name in clip_to_q:
            reordered.extend((copy.deepcopy(patched_nodes[clip_to_q[node.name]]),copy.deepcopy(patched_nodes[node.name])))
        elif node.name not in q_names:
            reordered.append(copy.deepcopy(patched_nodes[node.name]))
    del candidate.graph.node[:]
    candidate.graph.node.extend(reordered)
    require(len(expected_nodes) == len(expected_constants) == 2*count and len(rewrites) == count and
            len(candidate.graph.node) == len(source.graph.node) and len(candidate.graph.initializer) == len(source.graph.initializer),
            'Approved rewrite inventory differs')
    for node in candidate.graph.node:
        require(node.SerializeToString() == expected_nodes.get(node.name,source_nodes[node.name]).SerializeToString(),
                'Unexpected node byte difference: '+node.name)
    for value in candidate.graph.initializer:
        require(value.SerializeToString() == expected_constants.get(value.name,source_constants[value.name]).SerializeToString(),
                'Unexpected initializer byte difference: '+value.name)
    old_shell,new_shell = copy.deepcopy(source),copy.deepcopy(candidate)
    for shell in (old_shell,new_shell):
        del shell.graph.node[:]
        del shell.graph.initializer[:]
    require(old_shell.SerializeToString() == new_shell.SerializeToString(), 'Graph/model shell or I/O changed')
    onnx.checker.check_model(candidate,full_check=True)
    audit={'schema':'rampq.gate05.matched_control_family_rewrite/v1', 'status':'EXACT_ALLOWED_REWRITE_VERIFIED',
        'quantized_convs':count,'restored_FP16_desired_only':88-count,'changed_node_count':2*count,
        'changed_bound_initializer_count':2*count,'rewrites':rewrites,'clip_names':[r['clip_name'] for r in rewrites],
        'all_Conv_weight_bias_scale_zero_and_unapproved_node_initializer_bytes_preserved':True,
        'all_model_shell_and_IO_bytes_preserved':True,
        'parent_nodes_serialized_sha256':{n.name:message_sha(n) for n in source.graph.node},
        'candidate_nodes_serialized_sha256':{n.name:message_sha(n) for n in candidate.graph.node},
        'parent_initializers_serialized_sha256':{v.name:message_sha(v) for v in source.graph.initializer},
        'candidate_initializers_serialized_sha256':{v.name:message_sha(v) for v in candidate.graph.initializer},
        'no_calibration_recompute':True,'no_native_build_or_inference':True}
    return candidate,audit


def execute_cpu_once(model_path,input_array,output_names):
    import onnxruntime as ort
    options=ort.SessionOptions()
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads=4
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    require(session.get_providers()==['CPUExecutionProvider'] and [v.name for v in session.get_inputs()]==['images'] and
            [v.name for v in session.get_outputs()]==output_names,'CPU provider or binding order differs')
    return session.run(output_names,{'images':input_array})


def array_info(array):
    import numpy as np
    finite=np.isfinite(array); selected=array[finite]
    return {'shape':list(array.shape),'dtype':str(array.dtype),'elements':int(array.size),'all_finite':bool(finite.all()),
        'min':float(selected.min()) if selected.size else None,'max':float(selected.max()) if selected.size else None,
        'raw_array_sha256':hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def prepare(output,configuration,protocol_path):
    require(not output.exists() and not output.with_suffix('.zip').exists(),'Fresh member destination required; no CPU retry')
    inventory,protocol=validate_protocol(protocol_path,configuration)
    row=inventory['configurations'][configuration]; parent=ROOT/row['parent_directory']
    paths=check_parent(parent,row)
    for item in inventory['shared_sources'].values():
        path=ROOT/item['file'];require(info(path)=={k:item[k] for k in ('bytes','sha256')},'Frozen shared source differs');paths.append(path)
    paths.extend([INVENTORY,protocol_path,Path(__file__).resolve()])
    before=hashes(paths); source_spec=read(parent/'spec.json')
    count=EXPECTED_QUANTIZED[CONFIGURATIONS.index(configuration)]
    require(source_spec['requests']==row['requests'] and source_spec['expected_binding_contract']==row['expected_binding_contract'] and
            source_spec['quantized_conv_count']==count and source_spec['fp16_conv_count']==88-count,
            'Source request partition changed')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import onnx
    import onnxruntime as ort
    source=onnx.load(str(parent/'model.onnx'))
    candidate,audit=rewrite(source,source_spec,read(parent/'module_map.json'),count)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(parent,output/'parent_bundle')
    for original,name in ((INVENTORY,'source_inventory.json'),(protocol_path,'family_protocol.json'),
                           (Path(__file__).resolve(),Path(__file__).name)):
        shutil.copy2(original,output/name)
    for key,item in inventory['shared_sources'].items():
        if key=='fixed_input':name='preprocessed_input.npy'
        else:name='source_'+key+Path(item['file']).suffix
        shutil.copy2(ROOT/item['file'],output/name)
    write(output/'source_hashes_before.json',before)
    if count:
        onnx.save(candidate,str(output/'model.onnx'))
    else:
        require(candidate.SerializeToString()==source.SerializeToString(),'Unquantized control graph must not change')
        shutil.copy2(parent/'model.onnx',output/'model.onnx')
    audit.update({'configuration':configuration,'parent_graph_sha256':row['model']['sha256'],
        'candidate_graph_sha256':sha(output/'model.onnx'),'family_protocol_sha256':PROTOCOL_SHA})
    write(output/'semantic_rewrite_audit.json',audit)
    expected_runs=1 if count==0 else 2
    write(output/'cpu_execution_plan.json',{'schema':'rampq.gate05.matched_control_family_CPU_plan/v1',
        'status':'FROZEN_BEFORE_ANY_MEMBER_CPU_EXECUTION','configuration':configuration,
        'source_CPU_runs':0 if count==0 else 1,'candidate_CPU_runs':1,'total_new_CPU_runs':expected_runs,
        'entire_family_maximum_new_CPU_runs':9,'input_sha256':INPUT_SHA,'provider':'CPUExecutionProvider',
        'graph_optimization':'ORT_DISABLE_ALL','threads':4,'failure_policy':'PRESERVE_NO_RETRY_RESELECTION_SEAL_OR_NATIVE',
        'FP16_control_same_graph_single_execution_reference':count==0,'no_dataset_or_calibration':True})
    input_array=np.load(output/'preprocessed_input.npy',allow_pickle=False)
    require(sha(output/'preprocessed_input.npy')==INPUT_SHA and input_array.dtype==np.float32 and
            input_array.shape==(1,3,640,640) and np.isfinite(input_array).all(),'Frozen fixed input invalid')
    outputs=[r for r in source_spec['expected_binding_contract'] if not r['input']]
    output_names=[r['name'] for r in outputs];keys=['main_output','pe_output']
    require(len(outputs)==2 and output_names==['output0','/model.10/m/m.0/attn/pe/conv/Conv_output_0'], 'Unexpected output set')
    write(output/'cpu_attempt_started.json',{'status':'ONLY_FROZEN_CPU_ATTEMPT_STARTED','configuration':configuration,
        'maximum_runs':expected_runs,'no_retry_if_interrupted':True,'client_at':datetime.now().astimezone().isoformat()})
    old=None
    if count:
        old=execute_cpu_once(output/'parent_bundle/model.onnx',input_array,output_names)
        np.savez_compressed(output/'cpu_source_outputs.npz',**dict(zip(keys,old)))
    actual=execute_cpu_once(output/'model.onnx',input_array,output_names)
    np.savez_compressed(output/'cpu_candidate_outputs.npz',**dict(zip(keys,actual)))
    rows=[]
    for index,(name,contract,array) in enumerate(zip(output_names,outputs,actual)):
        valid=array.dtype==np.float32 and list(array.shape)==contract['shape'] and bool(np.isfinite(array).all())
        equal=True if old is None else (old[index].dtype==array.dtype and old[index].shape==array.shape and
            old[index].tobytes(order='C')==array.tobytes(order='C'))
        rows.append({'name':name,'expected_npz_key':keys[index],'candidate':array_info(array),
            'source':array_info(old[index]) if old is not None else None,'contract_valid':valid,
            'byte_exact_source':equal,'identity_proof_if_no_source_CPU':'Exact same model file bytes' if old is None else None})
    passed=all(r['contract_valid'] and r['byte_exact_source'] for r in rows)
    diagnostic={'schema':'rampq.gate05.matched_control_family_CPU_equivalence/v1',
        'status':'MEMBER_CPU_EQUIVALENCE_AND_REFERENCE_PASSED' if passed else 'MEMBER_CPU_EQUIVALENCE_FAILED',
        'configuration':configuration,'new_CPU_model_runs':expected_runs,'source_CPU_runs':0 if old is None else 1,
        'candidate_CPU_runs':1,'source_graph_and_candidate_graph_file_byte_identical':sha(output/'model.onnx')==row['model']['sha256'],
        'provider':'CPUExecutionProvider','graph_optimization':'ORT_DISABLE_ALL','threads':4,
        'onnx':onnx.__version__,'onnxruntime':ort.__version__,'numpy':np.__version__,'outputs':rows,
        'source_raw_file':'cpu_source_outputs.npz' if old is not None else None,'candidate_raw_file':'cpu_candidate_outputs.npz',
        'original_complete_model_guard':'FAILED_UNCHANGED','new_tolerances_defined':False,'native_or_dataset_run':False}
    write(output/'cpu_equivalence.json',diagnostic)
    after=hashes(paths);write(output/'source_hashes_after.json',after)
    require(before==after,'Historical sources changed during member preparation')
    require(passed,'Member CPU equivalence failed; preserve raw outputs, do not retry/seal/start native')
    shutil.copy2(output/'cpu_candidate_outputs.npz',output/'cpu_expected_outputs.npz')
    native_policy=copy.deepcopy(protocol['common_native_settings'])
    native_policy['INT8']=count>0
    spec={'schema':'rampq.gate05.matched_control_family_candidate/v1','configuration':configuration,
        'prepared_client_at':datetime.now().astimezone().isoformat(),'diagnostic_only':True,
        'family_protocol_file':'family_protocol.json','family_protocol_sha256':PROTOCOL_SHA,
        'source_inventory_file':'source_inventory.json','source_inventory_sha256':INVENTORY_SHA,
        'parent_bundle_directory':'parent_bundle','parent_graph_sha256':row['model']['sha256'],
        'parent_spec_sha256':row['spec']['sha256'],'parent_manifest_sha256':row['manifest']['sha256'],
        'graph_file':'model.onnx','graph_sha256':sha(output/'model.onnx'),
        'input_file':'preprocessed_input.npy','input_file_sha256':INPUT_SHA,
        'requests':copy.deepcopy(source_spec['requests']),'quantized_conv_count':count,'desired_restored_fp16_conv_count':88-count,
        'requests_are_desired_precision_masks_not_manual_layer_constraints':True,
        'restored_FP16_adoption_requires_independent_actual_audit':True,
        'expected_binding_contract':copy.deepcopy(source_spec['expected_binding_contract']),
        'clip_names':audit['clip_names'],'requires_native_clip_importer_correction':count>0,
        'activation_quantizer_topology':'Q_INT8_CODE_CLIP_DQ' if count else 'NO_QDQ_FLOAT_CONTROL',
        'semantic_rewrite_audit_file':'semantic_rewrite_audit.json','semantic_rewrite_audit_sha256':sha(output/'semantic_rewrite_audit.json'),
        'legacy_source_precision_constraints':source_spec['precisionConstraints'],'runtime_precision_constraints':'none',
        'native_build_policy':native_policy,'explicit_layer_precision_or_output_type_setters_or_resets':False,
        'native_build_authorized':False,'maximum_native_builds':0,'max_build_seconds':None,
        'native_preparation_only_policy':protocol['native_preparation_only_policy'],
        'native_inference_executions':0,'new_CPU_execution_runs_this_member':expected_runs,'family_total_new_CPU_runs':9,
        'cpu_equivalence_file':'cpu_equivalence.json','cpu_equivalence_sha256':sha(output/'cpu_equivalence.json'),
        'cpu_expected_outputs_file':'cpu_expected_outputs.npz','cpu_expected_outputs_sha256':sha(output/'cpu_expected_outputs.npz'),
        'expected_output_npz_keys':dict(zip(output_names,keys)),
        'original_complete_model_numerical_guard':'FAILED_UNCHANGED','gate05_status':'INCOMPLETE',
        'common_source_provenance':copy.deepcopy(inventory['common_source_provenance']),
        'new_tolerances_defined':False,'weights_or_calibration_changed':False,'dataset_evaluation_run':False,
        'native_build_or_inference_run_by_preparer':False,
        'limits':'One family preparation and own fixed-image CPU rewrite check. Global FP16/NONE flags do not guarantee desired restored precision. Native actual formats/tactics/fusion must be audited; no numerical migration or Gate pass.'}
    write(output/'spec_draft.json',spec)
    write(output/'prepared_content_manifest.json',{'schema':'rampq.gate05.matched_control_family_prepared/v1',
        'status':'MEMBER_CPU_EQUIVALENCE_PASSED_NOT_SEALED','configuration':configuration,
        'files':{p.relative_to(output).as_posix():info(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'status':'MEMBER_CPU_EQUIVALENCE_PASSED_NOT_SEALED','configuration':configuration,
        'output':str(output),'model_sha256':spec['graph_sha256'],'expected_sha256':spec['cpu_expected_outputs_sha256'],
        'new_CPU_runs':expected_runs,'source_unchanged':True},indent=2))


def seal(output,configuration,native_builder,expected_builder_sha,native_protocol_path,native_protocol_sha):
    require(native_protocol_path is not None and native_protocol_sha is not None and len(native_protocol_sha)==64 and
            sha(native_protocol_path)==native_protocol_sha,
            'A separately frozen native-family protocol is required; preparation authorizes no native sealing/build')
    native_protocol=read(native_protocol_path)
    require(native_protocol['schema']=='rampq.gate05.matched_control_family_native_protocol/v1' and
            native_protocol['family_preparation_protocol_sha256']==PROTOCOL_SHA and
            native_protocol['configuration_order']==list(CONFIGURATIONS) and
            native_protocol['native_inference_executions']==0,
            'Wrong separate native-family control scope')
    native_budget=native_protocol['native_builds'][configuration]
    require(native_budget['attempts']==1 and isinstance(native_budget['max_seconds'],int) and
            native_budget['max_seconds']>0,'One separately budgeted build is required')
    require(output.is_dir() and not output.with_suffix('.zip').exists() and not (output/'manifest.json').exists() and
            not (output/'spec.json').exists() and not (output/'preparation_failure.json').exists(), 'Successful unsealed member required')
    prepared=read(output/'prepared_content_manifest.json')
    require(prepared['status']=='MEMBER_CPU_EQUIVALENCE_PASSED_NOT_SEALED' and prepared['configuration']==configuration,
            'Wrong prepared member')
    for name,expected in prepared['files'].items():
        path=(output/name).resolve();require(output in path.parents and path.is_file() and info(path)==expected,
                                           'Prepared file became stale: '+name)
    sources=read(output/'source_hashes_before.json')
    require(sources==read(output/'source_hashes_after.json'),'Source preservation records differ');validate_sources(sources)
    require(sha(output/'family_protocol.json')==PROTOCOL_SHA and
            read(output/'cpu_equivalence.json')['status']=='MEMBER_CPU_EQUIVALENCE_AND_REFERENCE_PASSED','Frozen member proof invalid')
    require(native_builder is not None and expected_builder_sha is not None and native_builder.parent==ROOT/'scripts' and
            native_builder.is_file() and not native_builder.is_symlink() and len(expected_builder_sha)==64 and
            sha(native_builder)==expected_builder_sha,'Final new native builder identity required')
    native_paths=[native_builder,native_protocol_path]+[ROOT/'scripts'/name for name in HELPER_PINS]
    for name,expected in HELPER_PINS.items():require(sha(ROOT/'scripts'/name)==expected,'Original helper changed')
    before=hashes(native_paths)
    for path in native_paths:shutil.copy2(path,output/('native_protocol.json' if path==native_protocol_path else path.name))
    spec=read(output/'spec_draft.json');spec.update(native_builder_file=native_builder.name,
        native_builder_sha256=expected_builder_sha,sealed_client_at=datetime.now().astimezone().isoformat(),
        native_protocol_file='native_protocol.json',native_protocol_sha256=native_protocol_sha,
        native_build_authorized=True,maximum_native_builds=1,max_build_seconds=native_budget['max_seconds'])
    write(output/'spec.json',spec)
    write(output/'seal_native_source_hashes_before.json',before)
    after=hashes(native_paths);write(output/'seal_native_source_hashes_after.json',after)
    require(before==after,'Native source changed while sealing');validate_sources(sources)
    files={p.relative_to(output).as_posix():info(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'manifest.json',{'schema':'rampq.gate05.matched_control_family_bundle/v1','configuration':configuration,'files':files})
    sums=dict(files);sums['manifest.json']=info(output/'manifest.json')
    (output/'SHA256SUMS').write_text(''.join(row['sha256']+'  '+name+'\n' for name,row in sorted(sums.items())),encoding='utf-8')
    archive_path=output.with_suffix('.zip');archive_root='gate05_matched_'+configuration+'_20261021'
    with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():archive.write(path,archive_root+'/'+path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None,'ZIP CRC failed')
        for path in output.rglob('*'):
            if path.is_file():require(hashlib.sha256(archive.read(archive_root+'/'+path.relative_to(output).as_posix())).hexdigest()==sha(path),
                                      'ZIP member bytes differ')
    print(json.dumps({'status':'MATCHED_CONTROL_MEMBER_SEALED','configuration':configuration,'archive':str(archive_path),
        'archive_sha256':sha(archive_path),'archive_bytes':archive_path.stat().st_size,'spec_sha256':sha(output/'spec.json'),
        'manifest_sha256':sha(output/'manifest.json'),'new_CPU_runs_during_seal':0},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',choices=CONFIGURATIONS,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    stage=parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--prepare-only',action='store_true');stage.add_argument('--seal-only',action='store_true')
    parser.add_argument('--native-builder',type=Path);parser.add_argument('--native-builder-sha256')
    parser.add_argument('--native-protocol',type=Path);parser.add_argument('--native-protocol-sha256')
    args=parser.parse_args();output=(args.output if args.output else OUTPUT/args.configuration).resolve()
    require(output.is_relative_to((ROOT/'artifacts/gate05').resolve()),'Unsafe member destination')
    if args.prepare_only:
        existed=output.exists()
        try:prepare(output,args.configuration,args.protocol.resolve())
        except BaseException as error:
            if not existed and output.is_dir() and not (output/'preparation_failure.json').exists():
                write(output/'preparation_failure.json',{'status':'FAILED_PRESERVE_NO_RETRY_RESELECTION_SEAL_OR_NATIVE',
                    'configuration':args.configuration,'error_type':type(error).__name__,'error':str(error),
                    'traceback':traceback.format_exc(),'original_complete_model_guard':'FAILED_UNCHANGED'})
            raise
    else:seal(output,args.configuration,args.native_builder.resolve() if args.native_builder else None,args.native_builder_sha256,
              args.native_protocol.resolve() if args.native_protocol else None,args.native_protocol_sha256)


if __name__=='__main__':
    main()
