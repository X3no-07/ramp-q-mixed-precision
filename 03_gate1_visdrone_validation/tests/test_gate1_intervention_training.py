"""Meaningful checks for STE, supervision boundaries, fair budgets and actual updates."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import gate1_intervention_training as P


def test_ste_forward_round_ties_clip_and_backward():
    x=torch.tensor([-130.,-127.,-.5,.5,1.5,127.,130.],requires_grad=True)
    y=P.RoundSTE.apply(x,torch.tensor(1.))
    assert torch.equal(y,torch.tensor([-127.,-127.,0.,0.,2.,127.,127.]))
    y.sum().backward()
    assert torch.equal(x.grad,torch.tensor([0.,1.,1.,1.,1.,1.,0.]))


def test_parameter_gradient_and_fixed_scales():
    conv=torch.nn.Conv2d(2,3,1,bias=True)
    q=P.TrainableQDQConv(conv,"model.3.conv",2.)
    x=torch.randn(1,2,8,8).clamp(-1,1).requires_grad_()
    before=q.weight_scale.clone()
    q(x).square().sum().backward()
    assert q.weight.grad is not None and q.weight.grad.abs().sum()>0
    assert x.grad.abs().sum()>0
    assert not q.weight_scale.requires_grad and torch.equal(before,q.weight_scale)


def test_g1_is_restored_and_dfl_remains_frozen():
    c=torch.nn.Conv2d(1,1,1)
    assert P.TrainableQDQConv(c,"model.0.conv",1.).enabled is False
    dfl=P.TrainableQDQConv(c,"model.23.dfl.conv",1.)
    assert dfl.enabled and not dfl.weight.requires_grad


def test_subset_selection_is_metadata_only_deterministic():
    rows=[{"image_id":str(i)} for i in range(100)]
    assert P.hash_subset(rows,10,"train")==P.hash_subset(list(reversed(rows)),10,"train")
    assert P.hash_subset(rows,10,"train")!=P.hash_subset(rows,10,"validation")


def test_hard_control_exact_budget_and_no_best_outcome_selection():
    pool=[{"group":0,"difficulty":i} for i in range(4)]+[{"group":1,"difficulty":i} for i in range(3)]
    selected=[pool[3],pool[4],pool[6]]
    out=P.match_counts_to_controls(selected,pool,lambda r:r["group"],lambda r:r["difficulty"])
    assert out==[pool[0],pool[4],pool[5]]
    with pytest.raises(ValueError):
        P.match_counts_to_controls([{"group":4}],pool,lambda r:r["group"],lambda r:0)


def test_large_box_enclosing_small_target_is_not_safe_background():
    assert P.box_iou_matrix([[0,0,1000,1000]],[[1,1,3,3]],True)[0,0]==1
    assert P.box_iou_matrix([[0,0,1000,1000]],[[1,1,3,3]])[0,0]<.1


def synthetic_sample():
    return {"image_identities":{"image_id":"synthetic/only"},
        "bboxes":torch.tensor([[.25,.25,.05,.05],[.65,.65,.04,.04]]),
        "cls":torch.tensor([[0.],[1.]]),
        "ignore_sample":{"boxes_xyxy":np.array([[400.,50.,500.,150.]])}}


def test_miner_preserves_gt_identity_and_rejects_ignore_false_negatives():
    q=np.zeros((1,14,8400),np.float32)
    q[:,2:4,:]=10
    r=q.copy()
    # Newly recovered true target, new clean background FP, and ignored prediction.
    r[0,:4,0]=[160,160,32,32];r[0,4,0]=.9
    r[0,:4,7000]=[40,500,10,10];r[0,4,7000]=.7
    r[0,:4,6000]=[450,100,10,10];r[0,4,6000]=.8
    record=P.mine_record(synthetic_sample(),q,r)
    assert record["rescued_targets"]==["gt:0"]
    assert record["intervention"]["positive_gt"]==[0]
    assert record["intervention"]["negative"]==[[7000,0]]
    assert len(record["hard_example"]["negative"])==1
    assert len(record["hard_example"]["positive_gt"])==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA synthetic qualification is mandatory before launch")
def test_actual_cuda_equivalence_gradient_update_and_snapshot(tmp_path):
    P.configure()
    old=P.read(P.ROOT/"artifacts/gate1/20261024_selection_546_fixed_G1_execution_v1/attempt_01/2026102201/calibration_ranges.json")["max_abs"]
    original=P.new_model()
    legacy=P.deepcopy(original)
    wrappers=P.ENTRY.install(legacy,old)
    P.ENTRY.legacy().enable_configuration(wrappers,"r_only:G1")
    trainable=P.new_model(old)
    x=torch.zeros(2,3,640,640,device="cuda:0")
    with torch.inference_mode():
        before=legacy(x[:1])[0]
        replay=trainable(x[:1])[0]
    assert torch.equal(before,replay)
    state_before={n:v.detach().clone() for n,v in trainable.state_dict().items()}
    trainable.train();P.freeze_bn(trainable)
    loss=P.CorrectiveLoss(trainable)
    batch={"img":x,"cls":torch.tensor([[0.],[1.]],device="cuda:0"),
        "bboxes":torch.tensor([[.25,.25,.05,.05],[.6,.6,.08,.08]],device="cuda:0"),
        "batch_idx":torch.tensor([0.,1.],device="cuda:0"),
        "ignore_boxes_xyxy":[torch.tensor([[400.,50.,500.,150.]],device="cuda:0") for _ in range(2)],
        "image_identities":[{"image_id":"synthetic/a"},{"image_id":"synthetic/b"}]}
    record={"ordinary_qat":{},"intervention":{"positive_gt":[0],"negative":[]},"hard_example":{"positive_gt":[0],"negative":[]}}
    pred=trainable(x)
    l=loss.compute(pred,batch,[record,record],"intervention")
    assert torch.isfinite(l)
    l.backward()
    parameters=[p for p in trainable.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert any(m.weight.grad.abs().sum()>0 for m in trainable.modules() if isinstance(m,P.TrainableQDQConv) and m.enabled and m.weight.requires_grad)
    assert loss.last_mask_receipt["positive_rows_preserved"]
    assert loss.last_mask_receipt["ignored_negative_anchors"]>0
    torch.nn.utils.clip_grad_norm_(parameters,10,error_if_nonfinite=True)
    torch.optim.SGD(parameters,lr=1e-4).step()
    assert any(not torch.equal(state_before[n],v) for n,v in trainable.state_dict().items() if n.endswith("weight"))
    assert all(torch.equal(state_before[n],v) for n,v in trainable.state_dict().items() if "running_" in n or n.endswith("_scale"))
    trainable.eval()
    destination=tmp_path/"synthetic_state.pt"
    torch.save({"state_dict":{n:v.detach().cpu() for n,v in trainable.state_dict().items()}},destination)
    independent=P.new_model(old)
    independent.load_state_dict(torch.load(destination,map_location="cpu",weights_only=True)["state_dict"],strict=True)
    with torch.inference_mode():
        assert torch.equal(trainable(x[:1])[0],independent(x[:1])[0])
