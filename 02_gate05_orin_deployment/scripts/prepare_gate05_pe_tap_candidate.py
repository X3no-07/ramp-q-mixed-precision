"""Stage a full-graph fusion workaround; does not build or evaluate an engine."""
import hashlib
import json
from pathlib import Path


def main():
    import onnx
    root = Path(__file__).resolve().parents[1]
    run = root / 'artifacts/gate05/20261018_int8'
    source = run / 'diagnostic_bundle_v2/model_q_all.onnx'
    output = run / 'pe_tap_full_graph_candidate'
    output.mkdir(exist_ok=False)
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    value_name = '/model.10/m/m.0/attn/pe/conv/Conv_output_0'
    info = next(value for value in graph.graph.value_info if value.name == value_name)
    graph.graph.output.append(info)
    # Shape inference adds metadata only. All arithmetic nodes and initializers stay byte-identical.
    original = onnx.load(str(source))
    assert [n.SerializeToString() for n in original.graph.node] == [n.SerializeToString() for n in graph.graph.node]
    assert [n.SerializeToString() for n in original.graph.initializer] == [n.SerializeToString() for n in graph.graph.initializer]
    assert graph.graph.output[0].SerializeToString() == original.graph.output[0].SerializeToString()
    onnx.checker.check_model(graph,full_check=True)
    destination = output / 'model_q_all_pe_tap.onnx'
    onnx.save(graph,str(destination))
    report = {'status':'STAGED_UNBUILT_FULL_GRAPH_CANDIDATE', 'client_date':'2026-10-18',
              'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'candidate_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
              'change':'Expose only the failed G4 PE Conv pre-Add tensor as a second graph output',
              'all_arithmetic_nodes_and_initializers_byte_identical':True,
              'quantization_groups_weights_ranges_and_main_output_unchanged':True,
              'extra_output':value_name,'extra_output_shape':[1,128,20,20],
              'whole_model_build_run':False,'whole_model_inference_run':False,
              'limits':'Graph output changes can affect fusion and precision. A new full engine and per-layer audit '
                       'are required. For target-level comparisons control floating precision/output structure across engines. '
                       'This candidate does not pass the failed numerical migration guard or Gate 0.5.'}
    (output/'candidate.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
