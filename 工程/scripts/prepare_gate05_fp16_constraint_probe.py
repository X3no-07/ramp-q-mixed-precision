"""Stage an exact-source PE Conv/Sigmoid/Mul two-output FP16 adoption probe."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    import onnx
    from onnx import helper
    source=ROOT/'artifacts/jetson_trt_fp16_20261016_v1/model.onnx'
    original=onnx.load(str(source))
    name='/model.10/m/m.0/attn/pe/conv/Conv'
    conv=copy.deepcopy(next(n for n in original.graph.node if n.name==name))
    weights=[copy.deepcopy(v) for v in original.graph.initializer if v.name in conv.input[1:]]
    assert len(weights)==2
    conv.input[0]='images'
    tensor=conv.output[0]
    nodes=[conv,helper.make_node('Sigmoid',[tensor],['sigmoid'],name='probe/Sigmoid'),
        helper.make_node('Mul',[tensor,'sigmoid'],['output0'],name='probe/SiLU')]
    graph=helper.make_graph(nodes,'fp16_constraint_adoption_probe',
        [helper.make_tensor_value_info('images',onnx.TensorProto.FLOAT,[1,128,20,20])],
        [helper.make_tensor_value_info('output0',onnx.TensorProto.FLOAT,[1,128,20,20]),
         helper.make_tensor_value_info(tensor,onnx.TensorProto.FLOAT,[1,128,20,20])],weights)
    model=helper.make_model(graph,opset_imports=copy.deepcopy(original.opset_import))
    model.ir_version=original.ir_version
    onnx.checker.check_model(model,full_check=True)
    out=ROOT/'artifacts/gate05/20261019_fp16_constraint_probe'
    out.mkdir(exist_ok=False)
    onnx.save(model,str(out/'model.onnx'))
    spec={'status':'STAGED_NOT_BUILT','source_onnx_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256((out/'model.onnx').read_bytes()).hexdigest(),
        'conv_name':name,'desired_conv_precision':'FP16 weight/input/output',
        'desired_external_dtype':'FLOAT for both named outputs, linear format',
        'expected_bindings':[{'name':'images','input':True,'shape':[1,128,20,20],'dtype':'DataType.FLOAT'},
            {'name':'output0','input':False,'shape':[1,128,20,20],'dtype':'DataType.FLOAT'},
            {'name':tensor,'input':False,'shape':[1,128,20,20],'dtype':'DataType.FLOAT'}],
        'scope':'Only native constraint/auxiliary binding adoption, synthetic external input graph, no model inference or accuracy evidence.'}
    (out/'spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(spec,indent=2))


if __name__=='__main__':
    main()
