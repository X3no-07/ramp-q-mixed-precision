"""Independent read-only audit of a NEW synthetic Trainer reference capture.

The ordinary best/last checkpoints are HASHED ONLY. The sole deserialized file
is the NEW state-only reference payload, using weights_only=True on CPU. This
attempt-01 audit is functional reference evidence; strict offline isolation was
not established by that training runner and is explicitly not accepted here.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
DELIVERY=ROOT/"artifacts/gate1/20261023_training_integration_delivery_v1"


def require(condition,message):
    if not condition:raise ValueError(message)


def ident(path):
    raw=Path(path).read_bytes()
    return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)


def verify_manifest(path,base):
    obj=read(path)
    require(isinstance(obj.get("files"),dict),"expected explicit manifest file mapping")
    for rel,expected in obj["files"].items():
        member=(base/rel).resolve(strict=True)
        require(member.is_relative_to(base.resolve()),"manifest escaped base")
        require(ident(member)==expected,f"manifest identity differs: {rel}")
    return {"path":Path(path).relative_to(ROOT).as_posix(),**ident(path),"files_verified":len(obj["files"])}


def run_audit(attempt,preparation,out,*,strict_offline=False):
    attempt,preparation,out=Path(attempt).resolve(),Path(preparation).resolve(),Path(out).resolve()
    require(not (out/"audit_result.json").exists(),"audit result must be new")
    settings=out/"settings"
    settings.mkdir(parents=True,exist_ok=False)
    os.environ.update(YOLO_CONFIG_DIR=str(settings),YOLO_OFFLINE="true",YOLO_AUTOINSTALL="false",
                      CUDA_VISIBLE_DEVICES="",WANDB_MODE="disabled",COMET_MODE="DISABLED",
                      PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    prep=verify_manifest(preparation,ROOT)
    outputs=verify_manifest(attempt/"manifest.json",attempt)
    require(prep["files_verified"]==31,"preparation member count differs")
    require(outputs["files_verified"]==16,"attempt member count differs")
    nested=[verify_manifest(attempt/rel,(attempt/rel).parent) for rel in
            ("fp32_capture/epoch_0001/manifest.json","fp32_capture/epoch_0001/state/manifest.json")]
    receipt=read(attempt/"training_receipt.json")
    contract=read(DELIVERY/"training_contract.json")
    require(receipt["preparation_manifest_sha256"]==prep["sha256"],"preparation identity not bound by receipt")
    require(receipt["training_contract_sha256"]==ident(DELIVERY/"training_contract.json")["sha256"],"contract identity differs")
    require(receipt["actual_train_call_returned"] is True and receipt["epochs"]==1 and receipt["device"]=="cpu"
            and receipt["amp"] is False and receipt["threads"]==1,"training scope differs")
    split_ids,split_hashes={},{}
    verified_synthetic_files=0
    for role in ("train","training_val"):
        binding=contract["datasets"][role]
        manifest_path=Path(binding["manifest_path"])
        require(ident(manifest_path)["sha256"]==binding["manifest_sha256"],"split manifest identity differs")
        manifest=read(manifest_path)
        require(manifest["data_role"]=="synthetic_only" and manifest["split_role"]==role,"split role differs")
        ids,hashes=set(),set()
        for row in manifest["samples"]:
            require(row["split_role"]==role and row["image_id"] not in ids,"split ID membership differs")
            ids.add(row["image_id"])
            for field in ("image","labels","sidecar"):
                info=row[field]
                p=(manifest_path.parent/manifest["data_root"]/info["path"]).resolve()
                require(p.is_relative_to(manifest_path.parent.resolve()),"split member escaped role root")
                require(ident(p)=={k:info[k] for k in ("bytes","sha256")},"synthetic bytes differ")
                verified_synthetic_files+=1
            hashes.add(ident(manifest_path.parent/manifest["data_root"]/row["image"]["path"])["sha256"])
        require(len(ids)==len(hashes)==(4 if role=="train" else 2),"split count or duplicate image hash differs")
        split_ids[role]=ids;split_hashes[role]=hashes
    require(split_ids["train"].isdisjoint(split_ids["training_val"]) and split_hashes["train"].isdisjoint(split_hashes["training_val"]),"split leakage")
    events=[json.loads(line) for line in (attempt/"events.jsonl").read_text().splitlines()]
    require([v["sequence"] for v in events]==list(range(len(events))),"event sequence differs")
    optimizer=[v for v in events if v["event"]=="actual_optimizer_step_completed"]
    require(len(optimizer)==2 and len(receipt["optimizer_steps"])==2 and receipt["ema_updates"]==2,"optimizer/EMA count differs")
    for i,(event,record) in enumerate(zip(optimizer,receipt["optimizer_steps"])):
        require({k:v for k,v in event.items() if k not in {"sequence","event"}}==record,"optimizer event/receipt differs")
        require(record["step"]==i+1 and record["ema_before"]==i and record["ema_after"]==i+1
                and record["first_trainable_parameter_changed_elements"]>0 and math.isfinite(record["gradient_absmax"])
                and record["gradient_absmax"]>0 and record["actual_framework_optimizer_step_returned"] is True,"optimizer evidence invalid")
    batches=[v for v in events if v["event"]=="on_train_batch_end"]
    require([v["mask_receipt"] for v in batches]==receipt["actual_criterion_batches"],"criterion events/receipt differ")
    require(len(batches)==2,"expected two actual criterion batches")
    for v in batches:
        b=v["mask_receipt"]
        require(all(math.isfinite(loss) for loss in v["loss"]),"nonfinite recorded loss")
        require(b["assigned_positive_anchors"]>0 and b["ignored_negative_anchors"]>0 and b["positive_rows_preserved"] is True,"required ignore/positive branch not exercised")
        require(b["kept_anchors"]+b["ignored_negative_anchors"]==b["batch_size"]*b["num_anchors_per_image"],"keep/mask counts differ")
        require(0<=b["classification_before_gain"]<b["ordinary_classification_before_gain"] and b["target_score_denominator"]>=1,"masked classification receipt inconsistent")
    entry=next(v for v in events if v["event"]=="save_model_entry_before_capture")
    returned=next(v for v in events if v["event"]=="save_model_return_after_capture_and_framework_serialization")
    saved=next(v for v in events if v["event"]=="on_model_save")
    require(entry["sequence"]<returned["sequence"]<saved["sequence"] and entry["normal_last_exists"] is False
            and returned["capture_receipts"]==1 and returned["normal_last_exists"] is True and saved["capture_receipt_exists"] is True,
            "actual MRO capture/save event order differs")
    require(saved["framework_half_checkpoint_sha256"]==ident(attempt/"train/weights/last.pt")["sha256"],"ordinary saved checkpoint hash differs")
    capture_dir=attempt/"fp32_capture/epoch_0001"
    capture=read(capture_dir/"capture_receipt.json")
    snapshot=read(capture_dir/"state/snapshot_receipt.json")
    architecture=read(capture_dir/"architecture.json")
    require(capture["phase"]=="SAVE_MODEL_ENTRY_BEFORE_SUPER_FP16_SERIALIZATION" and capture["ema_updates_before"]==capture["ema_updates_after"]==2,"capture phase/counter differs")
    arch_sha=hashlib.sha256(canonical(architecture).encode()).hexdigest()
    require(arch_sha==capture["architecture_sha256"]==snapshot["metadata"]["architecture_sha256"],"architecture hash differs")
    require(capture["training_contract_sha256"]==receipt["training_contract_sha256"]==snapshot["metadata"]["training_contract_sha256"],"capture training origin binding differs")
    # Only now import CPU tensor/model facilities, into private offline settings.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel
    ultralytics.settings.update({k:False for k in ("sync","wandb","comet","clearml","mlflow","dvc","raytune","tensorboard")})
    state_path=(capture_dir/"state/reference_state.pt").resolve()
    load_calls=[]
    original_load=torch.load
    def guarded_load(path,*args,**kwargs):
        require(Path(path).resolve()==state_path,"only the new state-only synthetic reference may deserialize")
        require(kwargs.get("weights_only") is True and kwargs.get("map_location")=="cpu","require explicit CPU weights_only load")
        load_calls.append(str(Path(path).relative_to(ROOT)))
        return original_load(path,*args,**kwargs)
    torch.load=guarded_load
    payload=torch.load(state_path,map_location="cpu",weights_only=True)
    require(payload["schema"]=="rampq.gate1.fp32_reference_state_format/v1" and payload["accepted_scientific_reference"] is False
            and payload["training_origin_independently_proven"] is False,"restricted state scope differs")
    state=payload["state_dict"]
    def tensor_description(value):
        require(isinstance(value,torch.Tensor) and value.device.type=="cpu","tensor must be CPU")
        value=value.detach().contiguous()
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()),"nonfinite tensor")
        return {"shape":list(value.shape),"dtype":str(value.dtype),"sha256":hashlib.sha256(value.numpy().tobytes()).hexdigest()}
    actual_state={name:tensor_description(value) for name,value in sorted(state.items())}
    require(actual_state==snapshot["tensors"],"new reference actual tensor values/dtypes/shapes differ")
    require(all(not value.is_floating_point() or value.dtype==torch.float32 for value in state.values()),"source state not FP32")
    state_sha=hashlib.sha256(canonical(actual_state).encode()).hexdigest()
    require(state_sha==capture["live_state_before_sha256"]==capture["live_state_after_sha256"],"capture source state differs")
    require(payload["metadata"]==snapshot["metadata"],"state payload metadata differs")
    with torch.random.fork_rng(devices=[]):
        rebuilt=DetectionModel(cfg=deepcopy(architecture["cfg"]),ch=architecture["channels"],nc=architecture["nc"],verbose=False)
    strict=rebuilt.load_state_dict(state,strict=True)
    require(not strict.missing_keys and not strict.unexpected_keys,"strict reconstruction keys differ")
    rebuilt.names=dict(enumerate(architecture["names"]))
    rebuilt=rebuilt.cpu().float().eval()
    require({name:tensor_description(value) for name,value in sorted(rebuilt.state_dict().items())}==actual_state,"rebuilt state differs")
    shape=(1,3,64,64)
    analytic=torch.arange(math.prod(shape),dtype=torch.float32).remainder(251).div(250).reshape(shape)
    def describe_output(value):
        if isinstance(value,torch.Tensor):return {"kind":"tensor","tensor":tensor_description(value)}
        if isinstance(value,(tuple,list)):return {"kind":type(value).__name__,"items":[describe_output(v) for v in value]}
        if isinstance(value,dict):return {"kind":"dict","items":{k:describe_output(v) for k,v in sorted(value.items())}}
        if value is None or type(value) in (str,int,float,bool):return {"kind":"scalar","value":value}
        raise ValueError("unreviewed output structure")
    require(tensor_description(analytic)==capture["reconstruction"]["input"],"analytic input differs")
    with torch.no_grad():actual_output=describe_output(rebuilt(analytic))
    require(actual_output==capture["reconstruction"]["output"],"independent CPU output differs")
    require(len(load_calls)==1,"unexpected checkpoint load count")
    initial=read(attempt/"initial_model_state_description.json")
    final=read(attempt/"final_model_state_description.json")
    changed=read(attempt/"changed_parameters.json")
    params=dict(rebuilt.named_parameters())
    require(set(initial)==set(final)==set(state),"model state keys differ")
    require(set(changed)==set(params),"changed parameter key coverage differs")
    for name,description in initial.items():
        require(description["shape"]==final[name]["shape"] and description["dtype"]==final[name]["dtype"],"state shape/dtype changed")
    for name,count in changed.items():
        require(type(count) is int and 0<=count<=params[name].numel(),"changed element count out of shape bounds")
        require((count>0)==(initial[name]["sha256"]!=final[name]["sha256"]),"changed parameter count disagrees with state hash change")
    require(sum(count>0 for count in changed.values())==receipt["changed_parameter_tensors"]==238,"changed tensor sum differs")
    require(sum(changed.values())==receipt["changed_parameter_elements"]==2209411,"changed element sum differs")
    require(sum(p.numel() for p in params.values())==receipt["model_parameters"]==2591790,"model parameter count differs")
    framework_root=Path(ultralytics.__file__).resolve().parent
    inventory={p.relative_to(framework_root).as_posix():ident(p)["sha256"] for p in sorted(framework_root.rglob("*.py"))}
    require(hashlib.sha256(canonical(inventory).encode()).hexdigest()==capture["sources"]["ultralytics_python_tree_sha256"],"runtime source inventory differs")
    require(capture["sources"]==contract["runtime_sources"],"runtime pins differ")
    result={"schema":"rampq.gate1.synthetic_training_independent_output_audit/v1",
            "status":"PASS_FUNCTIONAL_REFERENCE_AUDIT_ONLY_STRICT_OFFLINE_NOT_PROVEN" if not strict_offline else "PENDING_STRICT_OFFLINE_ADDITIONAL_REVIEW",
            "issues":[],"attempt":attempt.relative_to(ROOT).as_posix(),"preparation":prep,"outputs":outputs,"nested_manifests":nested,
            "synthetic_file_bytes_verified":verified_synthetic_files,"split_ids_disjoint":True,"split_actual_image_hashes_disjoint":True,
            "actual_optimizer_events_verified":2,"ema_updates":2,
            "criterion_positive_counts":[v["mask_receipt"]["assigned_positive_anchors"] for v in batches],
            "criterion_ignored_negative_counts":[v["mask_receipt"]["ignored_negative_anchors"] for v in batches],
            "criterion_positive_rows_preserved":True,"capture_actual_MRO_precedes_super_serialization":True,
            "independent_new_state_weights_only_load_count":len(load_calls),"deserialized_files":load_calls,
            "ordinary_best_and_last_only_hashed_not_deserialized":True,"strict_fresh_architecture_reconstruction":True,
            "new_reference_state_exact_against_saved_description":True,"analytic_cpu_input_exact":True,
            "independent_analytic_cpu_output_exact":True,"independent_output_description":actual_output,
            "changed_parameter_hash_correspondence":True,"changed_parameter_tensors":238,"changed_parameter_elements":2209411,
            "scope_limits":["Attempt 01 lacked an explicit integration-callback registration blocker; strict offline isolation was not proven and this audit does not accept it as final integration evidence.",
                            "Per-element changed counts are checked for sums, shape bounds and hash-change correspondence; initial/final parameter arrays were not retained, so exact counts cannot be independently recomputed.",
                            "Per-batch TAL/ignore counts are source-backed event/receipt cross-checks; this audit does not rerun optimizer steps or independently recompute historical TAL assignments.",
                            "Only the freshly captured state payload is deserialized. Neither normal model-object checkpoints nor old weights are loaded.",
                            "No real-data training, official AP, clean evaluation, scientific reference/Gate1, GPU capture, quantization or deployment is accepted."],
            "scientific_gate_pass":False,"real_dataset_or_holdout_read":False,"GPU_forward_used":False,"strict_offline_training_accepted":False}
    (out/"audit_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"strict_cpu_output_exact":True,"state_tensors":len(state),
                      "changed_parameter_tensors":238,"changed_elements":2209411,"result":ident(out/"audit_result.json")}))
    return result


if __name__=="__main__":
    run_audit(DELIVERY/"attempt_01",DELIVERY/"preparation_manifest.json",HERE,strict_offline=False)
