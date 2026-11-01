"""A finite, separately versioned G1 corrective-QAT DEVELOPMENT pilot.

Historical files are read-only. No Selection or Holdout payload is accessed.
This is floating fake-quant training, not an INT8 deployment certification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gate1/20261026_intervention_training_pilot_v2"
PRIOR = ROOT / "artifacts/gate1/20261026_intervention_training_pilot_v1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "settings"))
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import make_anchors

import gate1_formal_training as FT
import gate1_ignore_dataset as DS
import gate1_ignore_aware_evaluation as MATCH
import gate1_official_ap as AP
import gate1_fp32_reference_state as STATE
import gate1_w8a8_entry as ENTRY
import qualify_gate1_float_baseline as EVAL
from gate1_ignore_trainer import IgnoreAwareDetectionLoss, verify_framework_sources

SCHEMA = "rampq.gate1.intervention_corrective_qat_development/v2"
ARMS = ("ordinary_qat", "hard_example", "intervention")
SEED = 2026102601
CALIBRATION_SEED = "2026102201"
TRAIN_COUNT, VAL_COUNT, EPOCHS, BATCH, ACCUMULATE = 512, 128, 3, 4, 4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data, once=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if once:
        with path.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
    else:
        FT.atomic_json(path, data)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        require(torch.get_num_interop_threads() == 1, "Interop thread drift")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    seed_all()


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        scaled = value / scale
        ctx.save_for_backward(scaled)
        return scaled.round().clamp(-127, 127) * scale

    @staticmethod
    def backward(ctx, gradient):
        (scaled,) = ctx.saved_tensors
        return gradient * ((scaled >= -127) & (scaled <= 127)), None


class TrainableQDQConv(nn.Conv2d):
    def __init__(self, conv, name, maximum):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation, conv.groups,
                         conv.bias is not None, conv.padding_mode,
                         device=conv.weight.device, dtype=torch.float32)
        require(conv.padding_mode == "zeros", "Unsupported padding mode")
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if self.bias is not None:
                self.bias.copy_(conv.bias)
        self.name = name
        self.group = ENTRY.legacy().group_for_conv(name)
        self.enabled = self.group != "G1"
        self.register_buffer("activation_scale", torch.tensor(max(float(maximum)/127, 1e-12), device=conv.weight.device))
        self.register_buffer("weight_scale", conv.weight.detach().abs().amax((1,2,3), keepdim=True).clamp_min(1e-12)/127)
        # Scales are fixed at initialization for this finite pilot; saved with weights.
        # DFL's fixed expectation projection must not be optimized.
        if ".dfl." in name:
            self.requires_grad_(False)

    def forward(self, x):
        if self.enabled:
            x = RoundSTE.apply(x, self.activation_scale)
            weight = RoundSTE.apply(self.weight, self.weight_scale)
        else:
            weight = self.weight
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def install(model, maxima):
    rows = ENTRY.legacy().list_conv_modules(model)
    require(len(rows) == 88 and set(maxima) == {n for n, _, _ in rows}, "Conv identity mismatch")
    for name, conv, _ in rows:
        ENTRY.legacy().replace_child(model, name, TrainableQDQConv(conv, name, maxima[name]))
    return model


def precision(model, group):
    require(group in {"fp32", "q_all", "r_only:G1"}, "Unregistered configuration")
    for module in model.modules():
        if isinstance(module, TrainableQDQConv):
            module.enabled = group != "fp32" and not (group == "r_only:G1" and module.group == "G1")


def freeze_bn(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval().requires_grad_(False)
        if name.endswith(".dfl.conv"):
            module.requires_grad_(False)


def new_model(maxima=None, device="cuda:0"):
    state, _ = STATE.load_snapshot(ENTRY.CAPTURE / "state")
    arch = read(ENTRY.CAPTURE / "architecture.json")
    with torch.random.fork_rng(devices=[]):
        model = DetectionModel(deepcopy(arch["cfg"]), ch=3, nc=10, verbose=False)
    model.load_state_dict(state, strict=True)
    model.names = dict(enumerate(arch["names"]))
    model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, "epochs": EPOCHS, "amp": False})
    model.float().to(device)
    if maxima is not None:
        install(model, maxima)
    freeze_bn(model)
    return model.eval()


def hash_subset(rows, count, tag):
    require(len(rows) >= count and len({r["image_id"] for r in rows}) == len(rows), "Membership count/uniqueness")
    chosen = sorted(rows, key=lambda r: hashlib.sha256(f"{SEED}:{tag}:{r['image_id']}".encode()).hexdigest())[:count]
    return sorted(chosen, key=lambda r: r["image_id"])


class SelectedTrainingDataset(DS.RestrictedIgnoreDataset):
    """Lazy subset of the existing bound training manifest; member hashing retained."""
    def __init__(self, binding, rows):
        self.manifest_path = Path(binding["path"]).resolve(strict=True)
        require(identity(self.manifest_path) == binding, "Train manifest changed")
        header = read(self.manifest_path)
        require(header["schema"] == FT.SCHEMA and header["data_role"] == "real_training_v1" and
                header["split_role"] == "train" and header["policy"] == FT.POLICY, "Training role/policy mismatch")
        indexed = {r["image_id"]: r for r in header["samples"]}
        require(all(indexed.get(r["image_id"]) == r for r in rows), "Subset member changed")
        self.manifest_sha256 = binding["sha256"]
        self.root = (self.manifest_path.parent / header["data_root"]).resolve(strict=True)
        require(self.root.is_relative_to(self.manifest_path.parent), "Training root escaped")
        DS._reject_forbidden_absolute(self.root)
        self.entries = deepcopy(rows)
        self._entry_json = tuple(json.dumps(r, sort_keys=True, allow_nan=False) for r in rows)
        self.split_role, self.imgsz, self.class_count = "train", 640, 10
        self.augment, self.fliplr, self.flipud, self.seed, self.epoch = False, 0., 0., SEED, 0
        self.geometry = DS._load_geometry()


def gpu_batch(samples):
    batch = DS.RestrictedIgnoreDataset.collate_fn(samples)
    for key in ("img", "cls", "bboxes", "batch_idx"):
        batch[key] = batch[key].to("cuda:0")
    batch["img"] = batch["img"].float()/255.
    batch["ignore_boxes_xyxy"] = [x.to("cuda:0") for x in batch["ignore_boxes_xyxy"]]
    return batch


def box_iou_matrix(a, b, intersection_over_min=False):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1,4), np.asarray(b, dtype=np.float64).reshape(-1,4)
    if not len(a) or not len(b):
        return np.zeros((len(a),len(b)))
    inter = np.maximum(0, np.minimum(a[:,None,2:], b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    aa, bb = np.maximum(0,a[:,2:]-a[:,:2]).prod(1), np.maximum(0,b[:,2:]-b[:,:2]).prod(1)
    den = np.minimum(aa[:,None],bb[None,:]) if intersection_over_min else aa[:,None]+bb[None,:]-inter
    return inter / np.maximum(den,1e-12)


def xyxy(rows):
    a = np.asarray(rows, dtype=np.float64).reshape(-1,4).copy()
    a[:,2:] += a[:,:2]
    return a


def canvas_gt(sample):
    boxes = sample["bboxes"].numpy().astype(np.float64)*640
    boxes[:,:2] -= boxes[:,2:]/2
    records = []
    for i, (box, cls) in enumerate(zip(boxes, sample["cls"].flatten().tolist())):
        records.append({"target_id":f"gt:{i}", "physical_line_zero_based":i, "bbox_xywh":box.tolist(),
                        "score":1,"category":int(cls)+1,"truncation":0,"occlusion":0})
    # Training sidecars intentionally provide class-agnostic ignore geometry.
    ignores = sample["ignore_sample"]["boxes_xyxy"] if "boxes_xyxy" in sample["ignore_sample"] else None
    if ignores is None:
        collated = DS.RestrictedIgnoreDataset.collate_fn([sample])
        ignores = collated["ignore_boxes_xyxy"][0].numpy()
    for box in np.asarray(ignores).reshape(-1,4):
        i=len(records)
        records.append({"target_id":f"ignore:{i}","physical_line_zero_based":i,
            "bbox_xywh":[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])],
            "score":0,"category":0,"truncation":0,"occlusion":0})
    return records


def predictions(raw, params, image_id, label):
    EVAL.require(raw.shape == (1,14,8400) and np.isfinite(raw).all(), "Raw shape/finite mismatch")
    det, anchors = EVAL.guarded_nms(non_max_suppression, torch.from_numpy(raw))
    pred, _ = EVAL.restored_predictions(det[0].numpy(), anchors[0].numpy(), params, image_id, label)
    return sorted(pred, key=lambda r:(-r["score"],r["nms_input_order"]))


def match_counts_to_controls(selected, pool, key, priority):
    """Exact per-image/category/bin budget; controls may overlap treatment naturally."""
    counts = Counter(key(r) for r in selected)
    result=[]
    for group, count in sorted(counts.items()):
        candidates=sorted((r for r in pool if key(r)==group), key=priority)
        require(len(candidates)>=count, "Insufficient matched hard-example controls")
        result.extend(candidates[:count])
    require(Counter(key(r) for r in result)==counts, "Auxiliary budget mismatch")
    return result


def original_training_geometry(sample, source_gt, width, height):
    """Keep publisher raster ignores at their ORIGINAL integer coordinates."""
    ratio_y,ratio_x=sample["ratio_pad"][0]
    left,top=sample["ratio_pad"][1]
    params={"ratio":[ratio_x,ratio_y],"left":left,"top":top}
    positive=[g for g in source_gt if g["score"]==1 and 1<=g["category"]<=10]
    require(len(positive)==len(sample["cls"]),"Original GT/positive training label count differs")
    require([g["category"]-1 for g in positive]==sample["cls"].flatten().long().tolist(),"Original GT order/class differs")
    expected=xyxy([g["bbox_xywh"] for g in positive])
    expected[:,[0,2]]=np.clip(expected[:,[0,2]],0,width)*ratio_x+left
    expected[:,[1,3]]=np.clip(expected[:,[1,3]],0,height)*ratio_y+top
    observed=sample["bboxes"].numpy().astype(np.float64)*640
    observed[:,:2]-=observed[:,2:]/2;observed[:,2:]+=observed[:,:2]
    require(np.allclose(expected,observed,atol=.002,rtol=1e-5),"Original GT and transformed training label geometry differs")
    MATCH._ignore_integral(source_gt,width,height)
    return params


def mine_record(sample, q_raw, r_raw, source_gt, width, height, params):
    image_id=sample["image_identities"]["image_id"]
    gt=deepcopy(source_gt)
    qp=predictions(q_raw,params,image_id,"q")
    rp=predictions(r_raw,params,image_id,"r")
    qm=MATCH.match_project_targets(gt,qp,width,height,confidence=.25,iou_threshold=.5)
    rm=MATCH.match_project_targets(gt,rp,width,height,confidence=.25,iou_threshold=.5)
    qhits={x["target_id"] for x in qm["matches"]}
    rhits={x["target_id"] for x in rm["matches"]}
    positive_gt=[r for r in gt if r["score"]==1 and 1<=r["category"]<=10]
    rboxes=r_raw[0,:4].T.copy()
    rboxes[:,:2]-=rboxes[:,2:]/2
    rboxes[:,2:]+=rboxes[:,:2]
    rboxes[:,[0,2]]=(rboxes[:,[0,2]]-params["left"])/params["ratio"][0]
    rboxes[:,[1,3]]=(rboxes[:,[1,3]]-params["top"])/params["ratio"][1]
    overlaps=box_iou_matrix(rboxes,xyxy([r["bbox_xywh"] for r in positive_gt]))
    eligible={r["target_id"] for r in qm["targets"]} & {r["target_id"] for r in rm["targets"]}
    pool=[]
    for i,g in enumerate(positive_gt):
        if g["target_id"] not in eligible:
            continue
        cls=g["category"]-1
        quality=float((overlaps[:,i]*r_raw[0,4+cls]).max())
        area=g["bbox_xywh"][2]*g["bbox_xywh"][3]*params["ratio"][0]*params["ratio"][1]
        pool.append({"gt_index":i,"category":cls,"size_bin":0 if area<32**2 else (1 if area<96**2 else 2),
                     "quality":quality,"target_id":g["target_id"]})
    changed=qhits.symmetric_difference(rhits)
    chosen=sorted((r for r in pool if r["target_id"] in changed),key=lambda r:(r["quality"],r["gt_index"]))[:8]
    hardpos=match_counts_to_controls(chosen,pool,lambda r:(r["category"],r["size_bin"]),lambda r:(r["quality"],r["gt_index"]))
    rfp={x["detection_id"] for x in rm["detection_states"] if x["reason"]=="false_positive"}
    qfp={x["detection_id"] for x in qm["detection_states"] if x["reason"]=="false_positive"}
    all_gt_boxes=xyxy([r["bbox_xywh"] for r in gt])
    stride_centers=np.concatenate([np.stack(np.meshgrid(np.arange(640//s)+.5,np.arange(640//s)+.5),-1).reshape(-1,2)*s for s in (8,16,32)])
    stride_centers[:,0]=(stride_centers[:,0]-params["left"])/params["ratio"][0]
    stride_centers[:,1]=(stride_centers[:,1]-params["top"])/params["ratio"][1]
    negpool=[]
    for r in rp:
        if r["detection_id"] not in rfp:
            continue
        rb=xyxy([r["bbox_xywh"]])
        # Exclude any plausible object/ignore overlap, including a large box enclosing a tiny GT.
        if len(all_gt_boxes) and box_iou_matrix(rb,all_gt_boxes,True).max()>=.1:
            continue
        center=stride_centers[r["raw_anchor_index"]]
        if any(b[0]<=center[0]<b[2] and b[1]<=center[1]<b[3] for b in all_gt_boxes):
            continue
        prior=[v for v in qp if v["detection_id"] in qfp and v["category"]==r["category"]]
        already=bool(prior) and bool((box_iou_matrix(rb,xyxy([v["bbox_xywh"] for v in prior]))>=.5).any())
        negpool.append({"anchor":r["raw_anchor_index"],"category":r["category"]-1,"score":r["score"],
                        "score_bin":min(2,int((r["score"]-.25)/.25)),"new_after_restore":not already})
    negatives=sorted((r for r in negpool if r["new_after_restore"]),key=lambda r:(-r["score"],r["anchor"]))[:8]
    hardneg=match_counts_to_controls(negatives,negpool,lambda r:(r["category"],r["score_bin"]),lambda r:(-r["score"],r["anchor"]))
    def packed(pos,neg):
        return {"positive_gt":[r["gt_index"] for r in pos],"negative":[[r["anchor"],r["category"]] for r in neg]}
    return {"image_id":image_id,"q_counts":qm["counts"],"r_counts":rm["counts"],
            "rescued_targets":sorted(rhits-qhits),"collateral_targets":sorted(qhits-rhits),
            "intervention":packed(chosen,negatives),"hard_example":packed(hardpos,hardneg),
            "matched_positive_strata":dict(Counter(str((r["category"],r["size_bin"])) for r in chosen)),
            "matched_negative_strata":dict(Counter(str((r["category"],r["score_bin"])) for r in negatives)),
            "positives":chosen,"new_background_errors":negatives,"other_FP_not_used_as_background":True,
            "records_are_training_supervision_not_official_AP":True,"matching_coordinate_system":"original_publisher_integer_ignore_regions",
            "training_to_original_params":params}


class CorrectiveLoss(IgnoreAwareDetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.last_aux={}

    def compute(self,preds,batch,records,arm):
        assignments, base, _ = super().get_assigned_targets_and_loss(preds,batch)
        fg, gt_index, target_boxes, anchor_points, stride = assignments
        fg=fg.bool()
        scores=preds["scores"].permute(0,2,1).contiguous()
        distribution=preds["boxes"].permute(0,2,1).contiguous()
        box=self.bbox_decode(anchor_points,distribution)*stride
        zero=scores.sum()*0
        positive_terms=[]; negative_terms=[]; effective_positive=0; effective_negative=0
        if arm != "ordinary_qat":
            for b,record in enumerate(records):
                spec=record[arm]
                for local_gt in spec["positive_gt"]:
                    selected=fg[b] & (gt_index[b]==local_gt)
                    if not selected.any():
                        continue
                    indices=torch.where(batch["batch_idx"]==b)[0]
                    target=batch["bboxes"][indices[local_gt]]*640
                    truth=torch.cat((target[:2]-target[2:]/2,target[:2]+target[2:]/2))
                    pbox=box[b,selected]
                    inter=(torch.minimum(pbox[:,2:],truth[2:])-torch.maximum(pbox[:,:2],truth[:2])).clamp_min(0).prod(1)
                    union=(pbox[:,2:]-pbox[:,:2]).clamp_min(0).prod(1)+(truth[2:]-truth[:2]).prod()-inter
                    iou=inter/union.clamp_min(1e-7)
                    cls=int(batch["cls"][indices[local_gt]].item())
                    # Quality-aware target preserves localization/score relation; GT remains authority.
                    cls_term=F.binary_cross_entropy_with_logits(scores[b,selected,cls],iou.detach())
                    positive_terms.append(self.hyp.cls*cls_term+self.hyp.box*(1-iou).mean())
                    effective_positive+=1
                for anchor,cls in spec["negative"]:
                    require(not bool(fg[b,anchor]),"Mined background became a positive assignment")
                    negative_terms.append(self.hyp.cls*F.softplus(scores[b,anchor,cls]))
                    effective_negative+=1
        pos=torch.stack(positive_terms).mean() if positive_terms else zero
        neg=torch.stack(negative_terms).mean() if negative_terms else zero
        aux=.25*(pos+neg)
        self.last_aux={"positive_objects":effective_positive,"negative_anchors":effective_negative,
                       "positive_loss":float(pos.detach()),"negative_loss":float(neg.detach()),"auxiliary_loss":float(aux.detach())}
        # Use per-image normalized loss and explicit accumulation, consistently for every arm.
        return base.sum()+aux


def source_bindings():
    paths=[Path(__file__),ROOT/"tests/test_gate1_intervention_training_v2.py"]
    paths += [ROOT/"scripts"/name for name in (
        "gate1_formal_training.py","gate1_ignore_dataset.py","gate1_ignore_geometry.py",
        "gate1_ignore_trainer.py","gate1_ignore_negative_loss.py","gate1_fp32_reference_state.py",
        "gate1_fp32_live_capture.py","gate1_w8a8_entry.py","qualify_gate1_float_baseline.py",
        "gate1_ignore_aware_evaluation.py","gate1_official_ap.py","gate1_official_prefilter.py")]
    paths.append(ROOT/"src/gate0.py")
    return {p.name:identity(p) for p in paths if p.is_file()}


def prepare():
    """Create a repair contract reusing completed v1 stages, with no parameter search."""
    require(not (OUT/"contract.json").exists(),"v2 contract already frozen")
    previous=read(PRIOR/"contract.json")
    failure=read(PRIOR/"attempt_01/failure_receipt.json")
    require(failure["error"]=="Official category-0 raster assignment requires integer region coordinates","Unexpected v1 failure")
    require(not (PRIOR/"attempt_01/training").exists(),"v1 already trained; review before continuing")
    for binding in previous["source_bindings"].values():
        require(identity(binding["path"])==binding,"v1 frozen source changed")
    assignment=ROOT/"artifacts/gate1/20261022_split_refinement/split_plan_v1/split_assignment.jsonl"
    selected={r["image_id"]:r for r in previous["training_members"]}
    originals={}
    for line in assignment.read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        if row["image_id"] in selected:
            require(row["assigned_split"]=="train" and row["image_sha256"]==selected[row["image_id"]]["image"]["sha256"],"Original training role/hash differs")
            originals[row["image_id"]]=row
    require(set(originals)==set(selected),"Original training annotation metadata incomplete")
    paths=[]
    attempt=PRIOR/"attempt_01"
    for folder in (attempt/"fp32_reference",*[attempt/"evaluation"/label for label in ("legacy_q","legacy_r","mse_q","mse_r")]):
        require((folder/"metrics.json").is_file(),"Completed baseline receipt missing")
        paths.extend(p for p in folder.rglob("*") if p.is_file())
    paths += [attempt/"mse_ranges.json",attempt/"calibration_histograms.npz"]
    reuse_raw=[]
    for index in range(TRAIN_COUNT):
        raw=attempt/"training_supervision"/f"{index:04d}_raw.npz"
        if raw.is_file():
            require((raw.with_name(f"{index:04d}.json")).is_file(),"Uncommitted prior raw")
            reuse_raw.append({"index":index,"image_id":previous["training_members"][index]["image_id"],"raw":identity(raw)})
    c=deepcopy(previous)
    c.update(schema=SCHEMA,created_at_utc=now(),source_bindings=source_bindings(),
        previous_contract=identity(PRIOR/"contract.json"),previous_failure=identity(attempt/"failure_receipt.json"),
        previous_source_bindings=previous["source_bindings"],reused_stage_root=str(attempt.resolve()),
        reused_files=[identity(p) for p in sorted(paths)],reused_training_raw=reuse_raw,
        training_annotation_metadata=[originals[r["image_id"]] for r in previous["training_members"]],
        training_annotation_assignment=identity(assignment),
        repair_reason="v1 sent fractional transformed ignore regions to the original-image raster evaluator. v2 restores detections to original pixels and uses unchanged publisher integer annotation regions; no rounding or scorer/tolerance change.",
        training_label_mapping_tolerance="atol0.002 canvas pixel + rtol1e-5; identity bridge only, not historical raw output guard",
        real_image_forward_budget=2*(TRAIN_COUNT-len(reuse_raw))+3*EPOCHS*TRAIN_COUNT+3*VAL_COUNT,
        reused_completed_forward_images=704+2*len(reuse_raw),
        calibration_and_baseline_inference_repeated=False,
        automatic_retry=False)
    write(OUT/"contract.json",c)
    print(json.dumps({"status":"FROZEN_COORDINATE_BRIDGE_REPAIR","contract":identity(OUT/"contract.json"),"new_forwards":c["real_image_forward_budget"]},ensure_ascii=True))


def prepare_original_unused():
    require(not (OUT/"contract.json").exists(),"Pilot contract already frozen")
    formal=ROOT/"artifacts/gate1/20261023_formal_ignore_training_v1/formal_contract.json"
    f=read(formal)
    train_path=Path(f["datasets"]["train"]["manifest_path"])
    val_path=Path(f["datasets"]["training_val"]["manifest_path"])
    train, val=read(train_path),read(val_path)
    FT.validate_membership(train,val)
    source_protocol=ROOT/"artifacts/gate1/20261024_formal_candidate_evaluation_v1/protocol.json"
    evaluation=read(source_protocol)
    old_contract=ROOT/"artifacts/gate1/20261024_selection_546_fixed_G1_execution_v1/contract.json"
    old=read(old_contract)
    ranges=old_contract.parent/"attempt_01"/CALIBRATION_SEED/"calibration_ranges.json"
    chosen_train=hash_subset(train["samples"],TRAIN_COUNT,"train")
    chosen_val=hash_subset(evaluation["cohorts"]["training_val"],VAL_COUNT,"training_val")
    t_hashes={r["image"]["sha256"] for r in chosen_train}
    require(t_hashes.isdisjoint(r["image_sha256"] for r in chosen_val),"Train/development overlap")
    vmap={r["image_id"]:r for r in val["samples"]}
    require(all(vmap[r["image_id"]]["image"]["sha256"]==r["image_sha256"] for r in chosen_val),"Eval membership mismatch")
    contract={"schema":SCHEMA,"created_at_utc":now(),"authorization":'User 2026-10-26: Yes. Begin; execute the limited G1 three-arm development pilot as suggested in the previous entry',
        "candidate":identity(ENTRY.CAPTURE/"state/reference_state.pt"),"architecture":identity(ENTRY.CAPTURE/"architecture.json"),
        "formal_contract":identity(formal),"train_manifest":identity(train_path),"training_val_manifest":identity(val_path),
        "evaluation_protocol":identity(source_protocol),"calibration_contract":identity(old_contract),"old_ranges":identity(ranges),
        "training_members":chosen_train,"evaluation_members":chosen_val,"calibration_members":old["calibration"][CALIBRATION_SEED],
        "train_count":TRAIN_COUNT,"evaluation_count":VAL_COUNT,"epochs":EPOCHS,"batch":BATCH,"accumulate":ACCUMULATE,
        "arms":list(ARMS),"training_seed":SEED,"calibration_seed":CALIBRATION_SEED,"calibration_labels_read":False,
        "calibration_method":"fixed 4096-bin absolute histogram; deterministic <=32768 regularly spaced samples per Conv/input; MSE search33 clipping fractions0.2..1.0; existing64image calibration cohort",
        "initial_candidate_fixed":"formal_epoch30_FP32_EMA","precision":"G1 FP32; other Conv signed symmetric W8A8 fake quant; FP32 compute; BN frozen; DFL projection frozen",
        "quantizer":"clip-aware STE; activation and per-output-channel weight scales fixed at initial values; buffers saved",
        "training":"SGD lr0.0001 momentum0.9 weight_decay0.0001; clipping10; no AMP; no EMA; fixed final epoch3; no flips/augmentation; identical order and initialization",
        "auxiliary":"0.25*(mean positive GT classification-quality and IoU loss + mean safe-background BCE); <=8positive GT and8negative anchors per image; hard arm exact category/size and category/confidence-bin budgets",
        "evaluation":"128 training_val development only; fixed conf0.25 target counts, official Python mirror AP using frozen preprocessing/NMS; all-anchor historical guard reported separately",
        "evaluation_order":["legacy_q","legacy_r","mse_q","mse_r",*ARMS],
        "threshold_diagnostics":[.2,.25,.3],"historical_fp32_reference":"Historical training_val retained raw hashes but not raw arrays; reconstruct128 once for NEW all-anchor comparisons, require exact historical input/raw hashes, reuse historical predictions for128 subset FP32 AP",
        "reference_bindings":[identity(Path(evaluation["inference_output"])/"training_val/predictions"/f"{r['image_id'].split('/')[-1]}.json") for r in chosen_val],
        "real_image_forward_budget":64+2*TRAIN_COUNT+8*VAL_COUNT+3*EPOCHS*TRAIN_COUNT,
        "optimizer_steps_per_arm":EPOCHS*TRAIN_COUNT//(BATCH*ACCUMULATE),
        "reconstruction_forwards_separate":3*3,"CUDA_synthetic_qualification_separate":True,
        "timeout_seconds":7200,"automatic_retry":False,"failure":"preserve attempt; no batch reduction or tolerance relaxation",
        "Selection_payload_access":False,"Holdout_payload_access":False,"scientific_acceptance":False,"actual_INT8_execution":False,
        "single_seed_pilot_not_final_method_comparison":True,"source_bindings":source_bindings(),"framework":verify_framework_sources()}
    require(contract["candidate"]["sha256"]==ENTRY.FINAL_SHA,"Wrong initial model")
    write(OUT/"contract.json",contract)
    print(json.dumps({"status":"FROZEN_NEW_DEVELOPMENT_PILOT","contract":identity(OUT/"contract.json"),"budget":contract["real_image_forward_budget"]},ensure_ascii=True))


class Worker:
    def __init__(self):
        self.contract=read(OUT/"contract.json")
        self.attempt=OUT/"attempt_01"
        self.attempt.mkdir(exist_ok=False)
        self.started=time.monotonic()
        self.live={"schema":SCHEMA,"pid":os.getpid(),"parent_pid":os.getppid(),"started_at_utc":now(),
                   "contract":identity(OUT/"contract.json"),"forward_images":0,"status":"STARTING"}
        self.progress("STARTING",0,1)

    def progress(self,stage,done,total,**extra):
        require(time.monotonic()-self.started<self.contract["timeout_seconds"],"Frozen pilot time budget exhausted")
        require(not (self.attempt/"stop_request.json").exists(),"Stop request observed")
        self.live.update(status=stage,done=done,total=total,updated_at_utc=now(),**extra)
        write(self.attempt/"live_status.json",self.live,once=False)
        with (self.attempt/"events.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(self.live,ensure_ascii=False,allow_nan=False)+"\n")
        print(json.dumps({k:self.live[k] for k in ("status","done","total","forward_images")},ensure_ascii=True),flush=True)

    def forward(self,model,tensor):
        self.live["forward_images"]+=len(tensor)
        require(self.live["forward_images"]<=self.contract["real_image_forward_budget"],"Forward budget exceeded")
        with torch.inference_mode():
            value=model(tensor)[0].detach().cpu().numpy().copy()
        require(value.shape[1:]==(14,8400) and np.isfinite(value).all(),"Invalid inference output")
        return value

    def verify(self):
        c=self.contract
        require(c["schema"]==SCHEMA and c["arms"]==list(ARMS),"Unsupported contract")
        require(c["source_bindings"]==source_bindings(),"Frozen source changed")
        require(c["framework"]==verify_framework_sources(),"Framework changed")
        for name in ("candidate","architecture","formal_contract","train_manifest","training_val_manifest","evaluation_protocol","calibration_contract","old_ranges"):
            require(identity(c[name]["path"])==c[name],f"Source binding changed: {name}")
        for binding in c["reference_bindings"]:
            require(identity(binding["path"])==binding,"Historical prediction/reference hash file changed")
        for key in ("previous_contract","previous_failure","training_annotation_assignment"):
            require(identity(c[key]["path"])==c[key],"Prior record/metadata changed")
        for binding in list(c["previous_source_bindings"].values())+c["reused_files"]+[r["raw"] for r in c["reused_training_raw"]]:
            require(identity(binding["path"])==binding,"Reused completed-stage artifact changed")

    def reference(self,model):
        protocol=read(self.contract["evaluation_protocol"]["path"])
        root=Path(protocol["prepared_root"]).resolve(strict=True)
        out=self.attempt/"fp32_reference";out.mkdir()
        checks=[];gt_all=[];dt_all=[];counts=Counter()
        for index,(row,binding) in enumerate(zip(self.contract["evaluation_members"],self.contract["reference_bindings"]),1):
            old=read(binding["path"])
            require(old["image_id"]==row["image_id"],"Historical reference order mismatch")
            tensor,_=self.image_input(root,row)
            require(hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()==old["input_sha256"],"Historical FP32 input differs")
            raw=self.forward(model,tensor)
            require(hashlib.sha256(raw.tobytes()).hexdigest()==old["raw_hashes"]["fp32"],"Historical FP32 full raw hash differs")
            stem=row["image_id"].split("/")[-1]
            np.savez_compressed(out/f"{stem}.npz",raw=raw)
            annotation=root/row["annotation_path"]
            require(sha(annotation)==row["annotation_sha256"],"Reference annotation changed")
            gt=MATCH.parse_ground_truth(annotation.read_text(encoding="utf-8-sig"),row["image_id"])
            gt8=np.asarray([g["bbox_xywh"]+[g["score"],g["category"],g["truncation"],g["occlusion"]] for g in gt],dtype=np.float64).reshape(-1,8)
            dt8=np.asarray(old["prediction_rows8"]["fp32"],dtype=np.float64).reshape(-1,8)
            prep=AP.prepare_raw_image(gt8,dt8,width=row["width"],height=row["height"])
            gt_all.append(prep["prepared_gt"]);dt_all.append(prep["filtered_predictions"])
            pred=[{"detection_id":f"reference:{k}","bbox_xywh":r[:4].tolist(),"score":float(r[4]),"category":int(r[5])} for k,r in enumerate(dt8)]
            counts.update(MATCH.match_project_targets(gt,pred,row["width"],row["height"],confidence=.25,iou_threshold=.5)["counts"])
            checks.append({"image_id":row["image_id"],"historical_input_and_full_raw_SHA_exact":True})
            if index%16==0:self.progress("RECONSTRUCT_NEW_PILOT_REFERENCE",index,VAL_COUNT)
        write(out/"historical_hash_checks.json",checks)
        self.progress("OFFICIAL_PYTHON_AP_FP32_SUBSET_REFERENCE",0,1)
        ap=AP.calc_accuracy(gt_all,dt_all,capture_intermediates=False)
        np.savez_compressed(out/"AP_AR_arrays.npz",AP=ap["AP"],AR=ap["AR"],evalClass=ap["evalClass"])
        result={"branch":"fp32_reference","images":VAL_COUNT,"metrics_percent":ap["metrics"],"project_counts_conf025":dict(counts),
                "historical_predictions_reused":True,"historical_all_anchor_hashes_exact":True,"scientific_acceptance":False}
        write(out/"metrics.json",result)
        return result

    def calibration(self,model,maxima):
        c=self.contract
        prior=read(c["calibration_contract"]["path"])
        root=Path(read(c["evaluation_protocol"]["path"])["prepared_root"])
        hist={n:np.zeros(4096,dtype=np.int64) for n in maxima}
        observations=Counter();excess=Counter();handles=[]
        def hook(name):
            def capture(_module,args):
                values=args[0].detach().reshape(-1)
                step=max(1,math.ceil(values.numel()/32768))
                sampled=values[::step].abs().float().cpu().numpy().astype(np.float64)
                require(np.isfinite(sampled).all(),"Nonfinite activation during MSE calibration")
                maximum=max(float(maxima[name]),1e-12)
                excess[name]+=int((sampled>maximum*(1+1e-6)).sum())
                bins=np.minimum((sampled/maximum*4096).astype(np.int64),4095)
                hist[name][:]+=np.bincount(bins,minlength=4096)
                observations[name]+=1
            return capture
        for name,conv,_ in ENTRY.legacy().list_conv_modules(model):
            handles.append(conv.register_forward_pre_hook(hook(name)))
        try:
            for i,row in enumerate(c["calibration_members"],1):
                tensor,_=self.image_input(root,row)
                self.forward(model,tensor)
                if i%8==0:self.progress("MSE_IMAGE_ONLY_CALIBRATION",i,64)
        finally:
            for h in handles:h.remove()
        require(set(observations)==set(maxima) and all(v==64 for v in observations.values()),"Calibration coverage mismatch")
        require(not any(excess.values()),"Previously frozen absmax fails to cover this calibration input")
        selected={};details={}
        for name,counts in hist.items():
            maximum=max(float(maxima[name]),1e-12)
            centers=(np.arange(4096)+.5)*maximum/4096
            errors=[]
            for ratio in np.linspace(.2,1,33):
                scale=max(maximum*ratio/127,1e-12)
                q=np.clip(np.rint(centers/scale),-127,127)*scale
                errors.append(float(np.sum(counts*(centers-q)**2)/max(1,counts.sum())))
            best=min(range(33),key=lambda k:(errors[k],-k))
            ratio=float(np.linspace(.2,1,33)[best])
            selected[name]=maximum*ratio
            details[name]={"ratio":ratio,"MSE":errors[best],"maxabs_MSE":errors[-1],"sampled_values":int(counts.sum())}
        np.savez_compressed(self.attempt/"calibration_histograms.npz",**hist)
        write(self.attempt/"mse_ranges.json",{"max_abs":selected,"details":details,"images":64,"labels_read":False,"approximate_histogram_MSE":True})
        return selected

    def image_input(self,root,row):
        path=(root/row["image_path"]).resolve(strict=True)
        require(path.is_relative_to(root) and sha(path)==row["image_sha256"],"Image source changed")
        image=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        require(image is not None and image.shape==(row["height"],row["width"],3),"Image decode mismatch")
        letterbox=LetterBox((640,640),auto=False,scale_fill=False,scaleup=True,center=True,stride=32,padding_value=114,interpolation=cv2.INTER_LINEAR)
        params=letterbox.get_params({"img":image})
        padded=letterbox.apply_image({"img":image},params)["img"]
        # Historical reference normalized on CPU before transfer; preserve its exact rounding.
        tensor=torch.from_numpy(np.ascontiguousarray(padded[...,::-1].transpose(2,0,1))).unsqueeze(0).float().div(255).to("cuda:0")
        return tensor,params

    def evaluate(self,model,label):
        protocol=read(self.contract["evaluation_protocol"]["path"])
        root=Path(protocol["prepared_root"]).resolve(strict=True)
        output=self.attempt/"evaluation"/label;output.mkdir(parents=True,exist_ok=False)
        rawdir=output/"raw";rawdir.mkdir()
        gt_all=[];dt_all=[];counts=Counter();curves={str(t):Counter() for t in self.contract["threshold_diagnostics"]};guards=[]
        records=[]
        model.eval()
        for i,row in enumerate(self.contract["evaluation_members"],1):
            tensor,params=self.image_input(root,row)
            raw=self.forward(model,tensor)
            stem=row["image_id"].split("/")[-1]
            np.savez_compressed(rawdir/f"{stem}.npz",raw=raw)
            saved=Path(self.contract["reused_stage_root"])/"fp32_reference"/f"{stem}.npz"
            with np.load(saved,allow_pickle=False) as data:reference=data["raw"].copy()
            guards.append({"image_id":row["image_id"],"guard":EVAL.raw_guard(reference,raw)})
            pred=predictions(raw,params,row["image_id"],label)
            annotation=(root/row["annotation_path"]).resolve(strict=True)
            require(annotation.is_relative_to(root) and sha(annotation)==row["annotation_sha256"],"Development annotation identity mismatch")
            text=annotation.read_text(encoding="utf-8-sig")
            gt=MATCH.parse_ground_truth(text,row["image_id"])
            gt8=np.asarray([g["bbox_xywh"]+[g["score"],g["category"],g["truncation"],g["occlusion"]] for g in gt],dtype=np.float64).reshape(-1,8)
            dt8=np.asarray([d["bbox_xywh"]+[d["score"],d["category"],-1,-1] for d in pred],dtype=np.float64).reshape(-1,8)
            prepared=AP.prepare_raw_image(gt8,dt8,width=row["width"],height=row["height"])
            gt_all.append(prepared["prepared_gt"]);dt_all.append(prepared["filtered_predictions"])
            matching={}
            for threshold in self.contract["threshold_diagnostics"]:
                result=MATCH.match_project_targets(gt,pred,row["width"],row["height"],confidence=threshold,iou_threshold=.5)
                curves[str(threshold)].update(result["counts"])
                if threshold==.25:
                    counts.update(result["counts"]);matching=result
            records.append({"image_id":row["image_id"],"predictions":pred,"target_matching":matching,"GT_source_sha256":row["annotation_sha256"],"letterbox":params})
            if i%16==0:self.progress("EVALUATION_"+label,i,VAL_COUNT)
        write(output/"saved_predictions_and_targets.json",records)
        write(output/"all_anchor_guard.json",guards)
        np.savez_compressed(output/"prepared_matrices.npz",**{f"gt_{i}":v for i,v in enumerate(gt_all)},**{f"dt_{i}":v for i,v in enumerate(dt_all)})
        self.progress("OFFICIAL_PYTHON_AP_"+label,0,1)
        ap=AP.calc_accuracy(gt_all,dt_all,capture_intermediates=False)
        np.savez_compressed(output/"AP_AR_arrays.npz",AP=ap["AP"],AR=ap["AR"],evalClass=ap["evalClass"])
        result={"branch":label,"images":VAL_COUNT,"role":"training_val_development_subset","metrics_percent":ap["metrics"],
                "project_counts_conf025":dict(counts),"threshold_curve":{k:dict(v) for k,v in curves.items()},
                "scientific_acceptance":False,"native_current_submission":False,"real_INT8":False}
        write(output/"metrics.json",result)
        return result

    def train(self,model,dataset,records,arm,maxima):
        seed_all()
        folder=self.attempt/"training"/arm;folder.mkdir(parents=True,exist_ok=False)
        initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        initial_sha=hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        model.train();freeze_bn(model)
        criterion=CorrectiveLoss(model)
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1e-4,momentum=.9,weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        generator=np.random.Generator(np.random.PCG64(SEED))
        steps=0;trace=[];batches=0
        for epoch in range(EPOCHS):
            order=generator.permutation(len(dataset)).tolist()
            for position in range(0,len(order),BATCH):
                indices=order[position:position+BATCH]
                samples=[dataset[int(j)] for j in indices]
                batch=gpu_batch(samples)
                self.live["forward_images"]+=len(indices)
                require(self.live["forward_images"]<=self.contract["real_image_forward_budget"],"Training forward budget exceeded")
                pred=model(batch["img"])
                loss=criterion.compute(pred,batch,[records[j] for j in indices],arm)
                require(bool(torch.isfinite(loss)),"Nonfinite loss; no recovery")
                (loss/ACCUMULATE).backward()
                batches+=1
                if batches%ACCUMULATE==0:
                    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
                    trace.append({"epoch":epoch+1,"batch":batches,"step":steps,"loss":float(loss.detach()),"grad_norm_before_clip":float(norm),**criterion.last_aux})
                if batches%8==0:
                    self.progress("TRAINING_"+arm,batches,EPOCHS*TRAIN_COUNT//BATCH,epoch=epoch+1,optimizer_steps=steps,latest_loss=float(loss.detach()))
                del batch,pred,loss
            snapshot={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            require(all(torch.isfinite(v).all() for v in snapshot.values() if v.is_floating_point()),"Nonfinite state")
            torch.save({"state_dict":snapshot,"epoch":epoch+1,"arm":arm,"contract_sha256":self.live["contract"]["sha256"]},folder/f"epoch_{epoch+1:02d}_fp32.pt")
        require(steps==self.contract["optimizer_steps_per_arm"],"Optimizer budget incomplete")
        changed=sum(int((model.state_dict()[n].cpu()!=v).sum()) for n,v in initial.items() if n.endswith("weight"))
        require(changed>0,"Training did not modify weights")
        # Frozen BN state and all quantization scales remain fixed across training.
        frozen=[n for n in initial if "running_" in n or "num_batches_tracked" in n or n.endswith("_scale")]
        require(all(torch.equal(initial[n],model.state_dict()[n].cpu()) for n in frozen),"Frozen BN/quantizer state changed")
        model.eval()
        x=gpu_batch([dataset[0]])["img"]
        with torch.inference_mode():reference=model(x)[0].detach().cpu()
        restored=torch.load(folder/f"epoch_{EPOCHS:02d}_fp32.pt",map_location="cpu",weights_only=True)
        independent=new_model(maxima)
        independent.load_state_dict(restored["state_dict"],strict=True)
        with torch.inference_mode(): replay=independent(x)[0].detach().cpu()
        require(torch.equal(reference,replay),"Independent CUDA checkpoint reconstruction differs")
        independent.cpu()
        with torch.inference_mode():cpu=independent(x.cpu())[0].detach().cpu()
        require(torch.isfinite(cpu).all(),"CPU checkpoint reconstruction nonfinite")
        write(folder/"training_receipt.json",{"arm":arm,"initial_state_value_sha256":initial_sha,"epochs":EPOCHS,"batches":batches,
            "optimizer_steps":steps,"changed_weight_elements":changed,"trace":trace,"frozen_BN_and_scales_unchanged":True,
            "final_candidate":identity(folder/f"epoch_{EPOCHS:02d}_fp32.pt"),"CUDA_reconstruction_exact":True,
            "CPU_CUDA_output_max_abs":float((reference-cpu).abs().max()),"CPU_CUDA_comparison_is_not_old_guard_pass":True,
            "reconstruction_real_image_forwards_outside_training_budget":3})
        del independent,optimizer,initial,criterion
        torch.cuda.empty_cache()
        return model

    def run(self):
        configure();self.verify()
        require(torch.cuda.is_available(),"CUDA required")
        write(self.attempt/"runtime.json",{"python":sys.version,"executable":sys.executable,"torch":str(torch.__version__),"cuda":torch.version.cuda,
            "GPU":torch.cuda.get_device_name(0),"backend":FT.backend_settings()})
        prior=Path(self.contract["reused_stage_root"])
        maxima=read(prior/"mse_ranges.json")["max_abs"]
        metrics=[read(prior/"fp32_reference/metrics.json")]+[read(prior/"evaluation"/label/"metrics.json") for label in ("legacy_q","legacy_r","mse_q","mse_r")]
        reuse={r["index"]:r for r in self.contract["reused_training_raw"]}
        original_root=Path(read(self.contract["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
        dataset=SelectedTrainingDataset(self.contract["train_manifest"],self.contract["training_members"])
        model=new_model(maxima)
        records=[];mining=self.attempt/"training_supervision";mining.mkdir()
        for index in range(len(dataset)):
            sample=dataset[index]
            meta=self.contract["training_annotation_metadata"][index]
            annotation=(original_root/meta["annotation_path"]).resolve(strict=True)
            require(annotation.is_relative_to(original_root) and sha(annotation)==meta["annotation_sha256"],"Original training annotation changed")
            gt=MATCH.parse_ground_truth(annotation.read_text(encoding="utf-8-sig"),meta["image_id"])
            params=original_training_geometry(sample,gt,meta["width"],meta["height"])
            if index in reuse:
                with np.load(reuse[index]["raw"]["path"],allow_pickle=False) as data:q,r=data["q"].copy(),data["r"].copy()
            else:
                tensor=gpu_batch([sample])["img"]
                precision(model,"q_all");q=self.forward(model,tensor)
                precision(model,"r_only:G1");r=self.forward(model,tensor)
            record=mine_record(sample,q,r,gt,meta["width"],meta["height"],params);records.append(record)
            np.savez_compressed(mining/f"{index:04d}_raw.npz",q=q,r=r)
            write(mining/f"{index:04d}.json",record)
            if (index+1)%16==0:self.progress("MINE_TRAINING_ONLY_INTERVENTIONS",index+1,len(dataset))
        write(mining/"summary.json",{"images":len(records),"positive_counts":sum(len(r["intervention"]["positive_gt"]) for r in records),
              "negative_counts":sum(len(r["intervention"]["negative"]) for r in records),"matched_budget":True,
              "different_records":sum(r["intervention"]!=r["hard_example"] for r in records),"training_members_only":True})
        require(any(r["intervention"]!=r["hard_example"] for r in records),"Intervention and hard-example controls are identical")
        require(sum(len(r["intervention"]["negative"]) for r in records)>0,"No eligible new background errors; stop this pilot")
        del model;torch.cuda.empty_cache()
        for arm in ARMS:
            model=new_model(maxima)
            model=self.train(model,dataset,records,arm,maxima)
            metrics.append(self.evaluate(model,arm))
            del model;torch.cuda.empty_cache()
        initials=[read(self.attempt/"training"/arm/"training_receipt.json")["initial_state_value_sha256"] for arm in ARMS]
        require(len(set(initials))==1,"Training arms did not share initialization")
        self.verify()
        require(self.live["forward_images"]==self.contract["real_image_forward_budget"],"Forward budget not exactly completed")
        write(self.attempt/"comparison_result.json",{"status":"COMPLETE_SINGLE_SEED_DEVELOPMENT_PILOT_NOT_SCIENTIFIC_ACCEPTANCE",
            "metrics":metrics,"training_arms_same_initialization":True,"arms":list(ARMS),"old_failures_preserved":True,
            "Holdout_accessed":False,"Selection_accessed":False,"all_inference_scores_are_development_only":True})
        manifest=[identity(p) for p in sorted(self.attempt.rglob("*")) if p.is_file() and p.name not in {"events.jsonl","live_status.json","manifest.json","completion_receipt.json"}]
        write(self.attempt/"manifest.json",{"files":manifest})
        self.progress("COMPLETE_SINGLE_SEED_DEVELOPMENT_PILOT",1,1)
        write(self.attempt/"completion_receipt.json",{**self.live,"finished_at_utc":now(),"elapsed_seconds":time.monotonic()-self.started,
              "comparison":identity(self.attempt/"comparison_result.json"),"manifest":identity(self.attempt/"manifest.json")})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","run"))
    args=parser.parse_args()
    if args.mode=="prepare":
        prepare();return
    worker=Worker()
    sleep_prevented=False
    try:
        if os.name=="nt":
            import ctypes
            sleep_prevented=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        worker.run()
    except BaseException as e:
        worker.live.update(status="FAILED_ATTEMPT_PRESERVED",error_type=type(e).__name__,error=str(e),updated_at_utc=now())
        write(worker.attempt/"live_status.json",worker.live,once=False)
        write(worker.attempt/"failure_receipt.json",{**worker.live,"traceback":traceback.format_exc()})
        raise
    finally:
        if sleep_prevented:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=="__main__":
    main()
