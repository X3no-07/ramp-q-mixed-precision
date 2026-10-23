"""Check the candidate's main CPU output against the saved original ONNX run."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/gate05/20261018_int8_pe_tap'
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'.ultralytics')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import onnxruntime as ort
    import torch
    sys.path.insert(0,str(ROOT/'src'))
    import gate0
    OUT.mkdir(exist_ok=True)
    source=ROOT/'artifacts/gate05/20261018_int8/input_v1'
    graph=ROOT/'artifacts/gate05/20261018_int8/pe_tap_full_graph_candidate/model_q_all_pe_tap.onnx'
    image=ROOT/'artifacts/jetson_fp16_smoke_20261016_v1/000020.jpg'
    preflight=json.loads((source/'preflight.json').read_text(encoding='utf-8'))
    assert sha(image)==preflight['image_sha256']
    x,*_=gate0.image_tensor(image,640,torch.device('cpu'))
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
    names=[o.name for o in session.get_outputs()]
    assert names==['output0','/model.10/m/m.0/attn/pe/conv/Conv_output_0']
    actual,aux=session.run(names,{'images':x.numpy()})
    old=np.load(str(source/'local_raw_outputs.npz'),allow_pickle=False)['onnx_qdq']
    assert actual.shape==old.shape==(1,14,8400) and aux.shape==(1,128,20,20)
    assert np.isfinite(actual).all() and np.isfinite(aux).all()
    difference=float(np.abs(actual-old).max())
    exact=bool(np.array_equal(actual,old))
    report={'status':'CANDIDATE_CPU_MAIN_OUTPUT_EXACT' if exact else 'CANDIDATE_CPU_MAIN_OUTPUT_CHANGED',
        'candidate_sha256':sha(graph),'source_raw_outputs_sha256':sha(source/'local_raw_outputs.npz'),
        'image_sha256':sha(image),'max_abs_main_output_difference':difference,
        'primary_output_exact':exact,'auxiliary_finite':True,'auxiliary_shape':list(aux.shape),
        'original_pytorch_onnx_preflight_status':preflight['status'],
        'limits':'Extra output did not change this CPU ONNX main output if exact. Does not repair PyTorch-ONNX parity or prove TensorRT equivalence.',
        'onnxruntime':ort.__version__}
    (OUT/'candidate_cpu_output_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    assert exact


if __name__=='__main__':
    main()
