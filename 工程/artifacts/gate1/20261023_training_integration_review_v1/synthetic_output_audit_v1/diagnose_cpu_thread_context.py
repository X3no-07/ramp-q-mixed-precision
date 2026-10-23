"""Reference-only CPU thread diagnosis; never accepts a new thread contract."""
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).parent
NEW=OUT/"thread_diagnosis"
assert not NEW.exists()
NEW.mkdir()
(NEW/"settings").mkdir()
os.environ.update(YOLO_CONFIG_DIR=str(NEW/"settings"),YOLO_OFFLINE="true",YOLO_AUTOINSTALL="false",CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import NUM_THREADS
from ultralytics.utils.torch_utils import select_device
before=torch.get_num_threads()
select_device("cpu",verbose=False)
after=torch.get_num_threads()
print(json.dumps({"before_select_device":before,"after_select_device":after,"framework_NUM_THREADS":NUM_THREADS}))
base=ROOT/"artifacts/gate1/20261023_training_integration_delivery_v1/attempt_02/fp32_capture/epoch_0001"
arch=json.loads((base/"architecture.json").read_text())
capture=json.loads((base/"capture_receipt.json").read_text())
state=torch.load(base/"state/reference_state.pt",map_location="cpu",weights_only=True)["state_dict"]
shape=(1,3,64,64)
x=torch.arange(1*3*64*64,dtype=torch.float32).remainder(251).div(250).reshape(shape)
def desc(value):
    if isinstance(value,torch.Tensor):
        y=value.detach().cpu().contiguous()
        return {"kind":"tensor","tensor":{"shape":list(y.shape),"dtype":str(y.dtype),"sha256":hashlib.sha256(y.numpy().tobytes()).hexdigest()}}
    if isinstance(value,(list,tuple)):return {"kind":type(value).__name__,"items":[desc(v) for v in value]}
    if isinstance(value,dict):return {"kind":"dict","items":{k:desc(v) for k,v in sorted(value.items())}}
    return {"kind":"scalar","value":value}
results=[]
for threads in (1,after):
    torch.set_num_threads(threads)
    with torch.random.fork_rng(devices=[]):
        model=DetectionModel(cfg=deepcopy(arch["cfg"]),ch=arch["channels"],nc=arch["nc"],verbose=False)
    model.load_state_dict(state,strict=True)
    model.cpu().float().eval()
    with torch.no_grad():output=desc(model(x.clone()))
    results.append({"threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),
                    "exact_output_matches_capture":output==capture["reconstruction"]["output"],"output":output})
    print(json.dumps({k:v for k,v in results[-1].items() if k!="output"}))
result={"schema":"rampq.gate1.cpu_thread_context_diagnosis/v1","status":"DIAGNOSTIC_ONLY_OLD_THREAD_DECLARATION_NOT_ACCEPTED",
        "before_select_device":before,"after_select_device":after,"framework_NUM_THREADS":NUM_THREADS,
        "results":results,"ordinary_model_checkpoints_deserialized":False,"scientific_gate_pass":False,
        "conclusion":"CPU select_device overrides prior torch.set_num_threads; exact saved CPU output depends on effective thread context. Caller must reassert and record actual thread context after trainer construction."}
(NEW/"diagnosis.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
