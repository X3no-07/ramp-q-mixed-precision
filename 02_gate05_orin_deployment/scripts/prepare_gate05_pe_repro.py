"""Extract the exact failing PE depthwise Conv, with controlled fusion variants."""
import copy
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    import onnx
    from onnx import helper
    root = Path(__file__).resolve().parents[1]
    run = root / 'artifacts/gate05/20261018_int8'
    output = run / 'pe_repro_v1'
    output.mkdir(exist_ok=False)
    source = run / 'diagnostic_bundle_v2/model_q_all.onnx'
    graph = onnx.load(str(source))
    prefix = 'rampq_model_10_m_0_attn_pe_conv'
    conv = next(n for n in graph.graph.node if n.name == '/model.10/m/m.0/attn/pe/conv/Conv')
    original_weight = 'model.10.m.0.attn.pe.conv.weight'
    constants = [copy.deepcopy(t) for t in graph.graph.initializer
                 if t.name.startswith(prefix + '_') or t.name in (original_weight, 'model.10.m.0.attn.pe.conv.bias')]
    base = [copy.deepcopy(n) for n in graph.graph.node if n.name.startswith(prefix + '_')]
    assert len(base) == 5 and len(constants) == 8
    base[0].input[0] = 'x'
    base.append(copy.deepcopy(conv))
    for name in ('conv_only', 'conv_add', 'conv_add_tap'):
        nodes = copy.deepcopy(base)
        inputs = [helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1,128,20,20])]
        outputs = [helper.make_tensor_value_info(conv.output[0], onnx.TensorProto.FLOAT, [1,128,20,20])]
        if name != 'conv_only':
            inputs.append(helper.make_tensor_value_info('residual', onnx.TensorProto.FLOAT, [1,128,20,20]))
            nodes.append(helper.make_node('Add', ['residual',conv.output[0]], ['out'], name='/model.10/m/m.0/attn/Add'))
            outputs = [helper.make_tensor_value_info('out', onnx.TensorProto.FLOAT, [1,128,20,20])]
            if name == 'conv_add_tap':
                outputs.append(helper.make_tensor_value_info(conv.output[0], onnx.TensorProto.FLOAT, [1,128,20,20]))
        model = helper.make_model(helper.make_graph(nodes,name,inputs,outputs,constants),
                                  opset_imports=[helper.make_opsetid('',13)])
        model.ir_version = graph.ir_version
        onnx.checker.check_model(model,full_check=True)
        onnx.save(model,str(output/(name+'.onnx')))
    protocol = {'schema':'rampq.gate05.pe_minimal_repro/v1','client_date':'2026-10-18',
                'source_onnx_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'module':'model.10.m.0.attn.pe.conv','group':'G4','group_count':128,'input_shape':[1,128,20,20],
                'weights_scales_bias_and_qdq':'Copied exactly from the failed whole-model graph',
                'variants':['conv_only','conv_add','conv_add_tap'],
                'tap_change':'Expose the pre-Add Conv output as an additional graph output; mathematical values unchanged',
                'max_seconds_per_variant':90,'scope':'Single-node build and precision feasibility only; not full model or dataset evidence'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    files = sorted(p for p in output.iterdir() if p.is_file())
    (output/'manifest.json').write_text(json.dumps({'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
    archive = run/'pe_repro_v1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):z.write(p,'gate05_pe_repro_20261018_v1/'+p.name)
    print('PE_REPRO_ZIP_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
