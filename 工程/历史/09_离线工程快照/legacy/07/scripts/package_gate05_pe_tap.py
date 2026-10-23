"""Freeze one full-model PE-output fusion workaround for diagnostic building."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/gate05/20261018_int8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def main():
    import onnx
    source = RUN / 'diagnostic_bundle_v2'
    candidate_dir = RUN / 'pe_tap_full_graph_candidate'
    candidate = json.loads((candidate_dir/'candidate.json').read_text(encoding='utf-8'))
    candidate_path = candidate_dir/'model_q_all_pe_tap.onnx'
    assert sha(source/'model_q_all.onnx') == candidate['original_sha256']
    assert sha(candidate_path) == candidate['candidate_sha256']
    original = onnx.load(str(source/'model_q_all.onnx'))
    graph = onnx.load(str(candidate_path))
    onnx.checker.check_model(graph, full_check=True)
    assert [n.SerializeToString() for n in original.graph.node] == [n.SerializeToString() for n in graph.graph.node]
    assert [n.SerializeToString() for n in original.graph.initializer] == [n.SerializeToString() for n in graph.graph.initializer]
    assert graph.graph.output[0].SerializeToString() == original.graph.output[0].SerializeToString()
    assert len(graph.graph.output) == 2
    contract = []
    for is_input, values in ((True, graph.graph.input), (False, graph.graph.output)):
        for value in values:
            assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
            contract.append({'name':value.name, 'input':is_input,
                'shape':[d.dim_value for d in value.type.tensor_type.shape.dim], 'dtype':'DataType.FLOAT'})
    bundle = RUN/'diagnostic_bundle_v3_pe_tap'
    bundle.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name not in ('manifest.json','model_q_all.onnx','build_revision.json'):
            shutil.copy2(path, bundle/path.name)
    shutil.copy2(candidate_path, bundle/'model_q_all.onnx')
    shutil.copy2(candidate_dir/'candidate.json', bundle/'candidate.json')
    for name in ('jetson_build_qdq.py','jetson_monitored_build.py'):
        path=ROOT/'scripts'/name
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3,8))
        shutil.copy2(path,bundle/name)
    assert json.loads((bundle/'preflight.json').read_text(encoding='utf-8'))['status']=='LOCAL_QDQ_PREFLIGHT_FAILED'
    write(bundle/'build_revision.json', {
        'revision':'v3_pe_tap', 'client_date':'2026-10-18',
        'change':'Expose exactly one G4 pre-Add PE Conv output; accommodate exact named extra output binding',
        'source_onnx_sha256':candidate['original_sha256'], 'candidate_sha256':sha(candidate_path),
        'quantization_or_numerical_tolerances_changed':False, 'configuration':'q_all',
        'purpose':'One full-model hardware build and fixed-image diagnostic, no dataset evaluation',
        'limits':'Output structure changes fusion and possibly floating precision; old FP16 comparison is descriptive only. Full parity remains failed.'})
    files = sorted(p for p in bundle.iterdir() if p.is_file())
    manifest = {'schema':'rampq.gate05.qdq_inputs/v2', 'configuration':'q_all', 'diagnostic_only':True,
        'expected_binding_contract':contract,
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
    write(bundle/'manifest.json',manifest)
    destination=RUN/'diagnostic_bundle_v3_pe_tap.zip'
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.iterdir()):
            archive.write(path,'gate05_int8_diagnostic_20261018_v3_pe_tap/'+path.name)
    print(json.dumps({'bundle':str(destination),'sha256':sha(destination),'bytes':destination.stat().st_size,
        'files':len(manifest['files']),'expected_binding_contract':contract},indent=2))


if __name__=='__main__':
    main()
