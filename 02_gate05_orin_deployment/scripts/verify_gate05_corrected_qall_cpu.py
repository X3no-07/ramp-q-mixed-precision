"""Verify one fixed-input original/code-Clip q_all CPU equivalence; no native run."""
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    import numpy as np
    import onnxruntime as ort
    parent = ROOT/'artifacts/gate05/20261019_precision_runtime_bundles/q_all'
    candidate = ROOT/'artifacts/gate05/20261019_corrected_qall_candidate'
    out = ROOT/'artifacts/gate05/20261019_corrected_qall_cpu_equivalence'
    out.mkdir(exist_ok=False)
    paths = [parent/'model.onnx',parent/'spec.json',candidate/'model.onnx',candidate/'spec.json',
             candidate/'manifest.json',candidate/'semantic_rewrite_audit.json',candidate/'preprocessed_input.npy',
             candidate/'reference_contract.json',ROOT/'artifacts/gate05/20261018_int8/input_v1/preflight.json',
             ROOT/'artifacts/gate05/20261018_int8/input_v1/local_raw_outputs.npz',Path(__file__).resolve()]
    before = {p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
    write(out/'source_hashes_before.json',before)
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    spec = read(candidate/'spec.json')
    assert spec['configuration']=='q_all' and spec['quantized_conv_count']==88 and spec['fp16_conv_count']==0
    assert spec['graph_sha256']==sha(candidate/'model.onnx')
    assert spec['parent_graph_sha256']==sha(parent/'model.onnx')=='2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d'
    assert spec['native_diagnostic_reference_contract_sha256']==sha(candidate/'reference_contract.json')=='87fcbf04e88fd311784736cf77cddfdd82fc2608ed3de19882467e58707fb430'
    assert read(candidate/'reference_contract.json')['not_a_gate_amendment'] is True
    assert read(ROOT/'artifacts/gate05/20261018_int8/input_v1/preflight.json')['status']=='LOCAL_QDQ_PREFLIGHT_FAILED'
    for relative, record in read(candidate/'manifest.json')['files'].items():
        p=(candidate/relative).resolve()
        assert candidate.resolve() in p.parents
        assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
    x=np.load(candidate/'preprocessed_input.npy',allow_pickle=False)
    assert sha(candidate/'preprocessed_input.npy')=='24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
    assert x.dtype==np.float32 and x.shape==(1,3,640,640) and np.isfinite(x).all()
    options=ort.SessionOptions();options.intra_op_num_threads=4
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    contract={r['name']:r for r in spec['expected_binding_contract'] if not r['input']}
    results, arrays={},{}
    for label,folder in [('original',parent),('code_clip',candidate)]:
        session=ort.InferenceSession(str(folder/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        names=[v.name for v in session.get_outputs()]
        assert set(names)==set(contract)
        results[label]=dict(zip(names,session.run(None,{'images':x})))
    comparisons={}
    for i,name in enumerate(contract):
        a,b=results['original'][name],results['code_clip'][name]
        assert a.shape==b.shape==tuple(contract[name]['shape'])
        assert a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all()
        delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
        comparisons[name]={'shape':list(a.shape),'dtype':str(a.dtype),'elements':int(a.size),
                           'different_elements':int(np.count_nonzero(a!=b)),
                           'byte_identical':a.tobytes()==b.tobytes(),'max_abs':float(delta.max())}
        arrays['original_'+str(i)]=a;arrays['code_clip_'+str(i)]=b
    with np.load(ROOT/'artifacts/gate05/20261018_int8/input_v1/local_raw_outputs.npz',allow_pickle=False) as old:
        baseline=results['original']['output0'];old_ort=old['onnx_qdq']
        historical_primary_same=baseline.tobytes()==old_ort.tobytes()
    np.savez_compressed(out/'raw_outputs.npz',**arrays)
    after={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
    assert after==before
    write(out/'source_hashes_after.json',after)
    exact=all(r['byte_identical'] for r in comparisons.values())
    report={'schema':'rampq.gate05.corrected_qall_cpu_equivalence/v1','client_at':datetime.now().astimezone().isoformat(),
            'status':'FULL_FIXED_INPUT_CPU_REWRITE_BYTE_EQUIVALENCE_VERIFIED' if exact else 'REWRITE_EQUIVALENCE_FAILED_STOP_NATIVE',
            'versions':{'python':sys.version,'onnxruntime':ort.__version__,'numpy':np.__version__},
            'providers':['CPUExecutionProvider'],'graph_optimizations':'ORT_DISABLE_ALL','intra_op_threads':4,
            'source_and_candidate_outputs':comparisons,'old_no_pe_tap_primary_ort_output_byte_identical':historical_primary_same,
            'input_npy_sha256':sha(candidate/'preprocessed_input.npy'),'candidate_onnx_sha256':sha(candidate/'model.onnx'),
            'parent_onnx_sha256':sha(parent/'model.onnx'),'candidate_spec_sha256':sha(candidate/'spec.json'),
            'candidate_manifest_sha256':sha(candidate/'manifest.json'),'reference_contract_sha256':sha(candidate/'reference_contract.json'),
            'raw_outputs_sha256':sha(out/'raw_outputs.npz'),'all_original_sources_hash_unchanged':True,
            'original_complete_model_pytorch_ort_guard':'FAILED_UNCHANGED','new_tolerances_defined':False,
            'dataset_evaluation_run':False,'native_execution_run':False,'numerical_migration_pass_claim':False,
            'accuracy_pass_claim':False,'performance_pass_claim':False,
            'limits':'One existing fixed image. Compares original ONNX CPU with the new derivative only; does not compare or replace original PyTorch/ORT guard.'}
    write(out/'analysis.json',report)
    write(out/'manifest.json',{'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps(report,ensure_ascii=False))
    assert exact,'Stop before native build: original/rewrite outputs differ'


if __name__=='__main__':
    main()
