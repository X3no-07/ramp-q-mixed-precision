"""Find the first numerical divergence in the local Q/DQ preparation check."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'artifacts/gate05/20261018_int8/input_v1/.ultralytics')


def main():
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort
    import torch
    sys.path.insert(0,str(ROOT/'src'))
    import gate0
    torch.set_num_threads(4)
    root=ROOT/'artifacts/gate05/20261018_int8/input_v1'
    mapping=json.loads((root/'module_map.json').read_text(encoding='utf-8'))['mapping']
    cal=json.loads((root/'calibration.json').read_text(encoding='utf-8'))
    model=gate0.load_fused_model(ROOT/'artifacts/jetson_fp16_smoke_20261016_v1/best.pt',torch.device('cpu'))
    wrappers=gate0.install_qdq_wrappers(model,cal['activation_max_abs'])
    gate0.enable_configuration(wrappers,'q_all')
    recorded={}
    handles=[]
    for wrapper in wrappers:
        def hook(mod,inputs,output):
            recorded[mod.name]={'output':output.detach().numpy().copy(), 'input':inputs[0].detach().numpy().copy(),
                                'input_qdq':(torch.round(inputs[0]/mod.activation_scale).clamp(-127,127)*mod.activation_scale).detach().numpy().copy(),
                                'weight_qdq':mod.quantized_weight.numpy().copy()}
        handles.append(wrapper.register_forward_hook(hook))
    x,*_=gate0.image_tensor(ROOT/'artifacts/jetson_fp16_smoke_20261016_v1/000020.jpg',640,torch.device('cpu'))
    with torch.inference_mode(): model(x)
    for handle in handles: handle.remove()
    graph=onnx.load(str(root/'model_q_all.onnx'))
    by_name={n.name:n for n in graph.graph.node}
    names=[]
    for row in mapping:
        names.extend([by_name[row['onnx_node']].output[0],row['qdq_prefix']+'_a_dq',row['qdq_prefix']+'_w_dq'])
    for name in names: graph.graph.output.append(helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None))
    options=ort.SessionOptions(); options.intra_op_num_threads=4; options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    outputs=dict(zip(names,session.run(names,{'images':x.numpy()})))
    rows=[]
    for row in mapping:
        mod=row['pytorch_module']; expected=recorded[mod]
        names_for_row={'output':by_name[row['onnx_node']].output[0], 'input_qdq':row['qdq_prefix']+'_a_dq','weight_qdq':row['qdq_prefix']+'_w_dq'}
        stats={}
        for field,name in names_for_row.items():
            a,b=outputs[name],expected[field]; d=np.abs(a-b)
            stats[field]={'shape':list(a.shape),'max_abs':float(d.max()),'mean_abs':float(d.mean()),
                          'unequal':int(np.count_nonzero(d)),'above_1e_4':int(np.count_nonzero(d>1e-4))}
        rows.append({'module':mod,'group':row['group'],**stats})
    (root.parent/'local_divergence_diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    # Same-input isolation distinguishes quantizer semantics from accumulated backend drift.
    probe=helper.make_model(helper.make_graph([
        helper.make_node('Clip',['x','lo','hi'],['clipped']),
        helper.make_node('QuantizeLinear',['clipped','s','z'],['q']),
        helper.make_node('DequantizeLinear',['q','s','z'],['y']),
    ],'same_input_quantizer',
        [helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,shape) for n,shape in
         [('x',[None,None,None,None]),('s',[]),('lo',[]),('hi',[])]],
        [helper.make_tensor_value_info('y',onnx.TensorProto.FLOAT,[None,None,None,None])],
        [numpy_helper.from_array(np.asarray(0,dtype=np.int8),name='z')]),opset_imports=[helper.make_opsetid('',13)])
    probe.ir_version=graph.ir_version
    quant_session=ort.InferenceSession(probe.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    isolated=[]
    for row in mapping:
        rec=recorded[row['pytorch_module']]
        scale=np.asarray(row['activation_scale'],dtype=np.float32)
        bound=np.asarray(scale*np.float32(127),dtype=np.float32)
        observed=quant_session.run(None,{'x':rec['input'],'s':scale,'lo':np.asarray(-bound,dtype=np.float32),'hi':bound})[0]
        expected=rec['input_qdq']; mask=observed!=expected
        detail={'module':row['pytorch_module'],'elements':int(observed.size),'different':int(mask.sum())}
        if mask.any():
            units=np.abs(observed[mask]-expected[mask])/scale
            normalized=rec['input'][mask]/scale
            distance=np.abs(np.abs(normalized-np.rint(normalized))-0.5)
            detail.update(max_difference_in_quant_steps=float(units.max()),max_distance_from_rounding_boundary=float(distance.max()))
            assert units.max()<=1.0001 and distance.max()<=2e-5, detail
        isolated.append(detail)
    result={'status':'ISOLATED_QUANTIZER_SEMANTICS_VERIFIED',
            'scope':'Same inputs only; does not pass or replace the failed full-model numerical parity check',
            'weight_qdq_all_88_exact':all(r['weight_qdq']['unequal']==0 for r in rows),
            'first_conv_output_max_abs_error':rows[0]['output']['max_abs'],
            'second_conv_input_different_elements':rows[1]['input_qdq']['unequal'],
            'same_input_quantizer':isolated,'original_full_model_preflight':'LOCAL_QDQ_PREFLIGHT_FAILED'}
    assert result['weight_qdq_all_88_exact']
    (root.parent/'isolated_quantizer_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('ISOLATED_QUANTIZER_SEMANTICS_VERIFIED',len(isolated),'convs',sum(x['different'] for x in isolated),'boundary disagreements')
    print(json.dumps(rows[:2],indent=2))
    print('WEIGHT_MISMATCHES',[(r['module'],r['weight_qdq']) for r in rows if r['weight_qdq']['above_1e_4']])


if __name__=='__main__': main()
