from collections import Counter
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import gate1_native_target_correction as N
P=N.P


def test_overconfidence_gradient_never_pushes_score_up_and_targets_detached():
    z=torch.tensor([-20.,-2.,0.,2.,20.],requires_grad=True)
    t=torch.tensor([0.,.5,.5,.5,1.],requires_grad=True)
    loss=N.overconfidence_divergence(z,t)
    assert torch.isfinite(loss).all() and (loss>=0).all()
    loss.sum().backward()
    expected=torch.where(z.detach().sigmoid()>t.detach(),z.detach().sigmoid()-t.detach(),0.)
    assert torch.allclose(z.grad,expected,atol=1e-6) and (z.grad>=0).all()
    assert t.grad is None


def test_no_penalty_for_underconfident_true_positive():
    z=torch.tensor([0.],requires_grad=True)
    loss=N.overconfidence_divergence(z,torch.tensor([.8]))
    loss.sum().backward()
    assert loss.item()==0 and z.grad.item()==0


def test_old_IoU_target_can_conflict_with_native_classifier_and_new_one_aligns():
    z=torch.logit(torch.tensor([.2])).requires_grad_()
    old=F.binary_cross_entropy_with_logits(z,torch.tensor([.7]))
    old_gradient=torch.autograd.grad(old,z,retain_graph=True)[0]
    native=F.binary_cross_entropy_with_logits(z,torch.tensor([.05]))
    native_gradient=torch.autograd.grad(native,z,retain_graph=True)[0]
    new_gradient=torch.autograd.grad(N.overconfidence_divergence(z,torch.tensor([.05])).sum(),z)[0]
    assert old_gradient.item()<0<native_gradient.item()
    assert torch.allclose(new_gradient,native_gradient)


@pytest.mark.parametrize('ious,classes,expected',[
    ([.5],[1],N.KINDS[0]),([.6],[2],N.KINDS[1]),([.1],[1],N.KINDS[2]),([.09],[1],N.KINDS[3]),
    ([],[],N.KINDS[3])])
def test_error_partition_includes_overlap_errors(ious,classes,expected):
    assert N.error_kind(np.asarray(ious),np.asarray(classes),1)==expected


def test_matched_control_cannot_swap_localization_for_background():
    pool=[{'category':0,'score_bin':0,'kind':k,'score':s,'anchor':i} for i,(k,s) in enumerate([
        (N.KINDS[2],.3),(N.KINDS[2],.4),(N.KINDS[3],.45)])]
    out=P.match_counts_to_controls([pool[0]],pool,lambda r:(r['category'],r['score_bin'],r['kind']),lambda r:-r['score'])
    assert out==[pool[1]]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA qualification required')
def test_actual_native_targets_gradient_ignore_protection_update_and_checkpoint(tmp_path):
    P.configure()
    ranges=P.read(P.PRIOR/'attempt_01/mse_ranges.json')['max_abs']
    model=P.new_model(ranges);model.train();P.freeze_bn(model)
    x=torch.zeros(2,3,640,640,device='cuda:0')
    batch={'img':x,'cls':torch.tensor([[0.],[1.]],device='cuda:0'),
           'bboxes':torch.tensor([[.25,.25,.05,.05],[.6,.6,.08,.08]],device='cuda:0'),
           'batch_idx':torch.tensor([0.,1.],device='cuda:0'),
           'ignore_boxes_xyxy':[torch.tensor([[400.,50.,500.,150.]],device='cuda:0') for _ in range(2)],
           'image_identities':[{'image_id':'synthetic/a'},{'image_id':'synthetic/b'}]}
    pred=model(x)
    ordinary=P.IgnoreAwareDetectionLoss(model)
    assigned,base,_=ordinary.get_assigned_targets_and_loss(pred,batch)
    criterion=N.NativeCorrectiveLoss(model)
    empty=[{arm:{'positive_gt':[],'errors':[]} for arm in N.ARMS} for _ in range(2)]
    unchanged=criterion.compute(pred,batch,empty,'ordinary_qat')
    assert torch.equal(unchanged,base.sum())
    fg=assigned[0].bool()
    assert fg[0].any() and fg[1].any()
    records=[]
    for b in range(2):
        foreground=int(torch.where(fg[b])[0][0])
        spec={'positive_gt':[0],'errors':[{'anchor':855,'category':b},{'anchor':7000,'category':b},{'anchor':foreground,'category':b}]}
        records.append({arm:spec for arm in N.ARMS})
    loss=criterion.compute(pred,batch,records,'native_intervention_correction')
    extra_gradient=torch.autograd.grad(loss-base.sum(),pred['scores'],retain_graph=True)[0]
    assert (extra_gradient>=-1e-7).all()
    assert torch.count_nonzero(extra_gradient[:,:,855])==0
    assert criterion.last_aux['ignored_error_channels']==2
    assert criterion.last_aux['error_anchors_currently_foreground']>=2
    assert criterion.native_targets[fg].sum()>0
    assert criterion.last_mask_receipt['positive_rows_preserved']
    assert not criterion.last_aux['extra_positive_classification_BCE']
    initial={n:v.detach().clone() for n,v in model.state_dict().items()}
    loss.backward()
    parameters=[p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    torch.nn.utils.clip_grad_norm_(parameters,10,error_if_nonfinite=True)
    torch.optim.SGD(parameters,lr=1e-4).step()
    assert any(not torch.equal(initial[n],v) for n,v in model.state_dict().items() if n.endswith('weight'))
    assert all(torch.equal(initial[n],v) for n,v in model.state_dict().items() if 'running_' in n or n.endswith('_scale'))
    model.eval();path=tmp_path/'native_target_state.pt'
    torch.save(model.state_dict(),path)
    rebuild=P.new_model(ranges)
    rebuild.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    with torch.inference_mode():assert torch.equal(model(x[:1])[0],rebuild(x[:1])[0])
