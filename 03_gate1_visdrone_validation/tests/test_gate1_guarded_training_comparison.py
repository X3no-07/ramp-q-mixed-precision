from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate1_guarded_training_comparison as W
G, P = W.G, W.P


@pytest.mark.parametrize("a,b,c,expected", [
    ([-1., -2.], [1., 0.], [0., 1.], [0., 0.]),
    ([-1., 2.], [1., 0.], [0., 1.], [0., 2.]),
    ([3., 2.], [1., 0.], [-1., 0.], [0., 2.]),
    ([-2., 3.], [1., 0.], [2., 0.], [0., 3.]),
    ([-2., 3.], [0., 0.], [1., 0.], [0., 3.]),
    ([1., 2.], [0., 0.], [0., 0.], [1., 2.]),
])
def test_projection_degenerate_and_active_boundaries(a, b, c, expected):
    result, _ = G.project_two_halfspaces(*(torch.tensor(v, dtype=torch.float64) for v in (a, b, c)))
    assert torch.allclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_projection_independent_constrained_solver_and_near_opposite_normals():
    rng = np.random.default_rng(2026102701)
    def independent_dykstra(a, b, c):
        # Independent iterative projection, not the production active-set solver.
        x=a.copy();corrections=[np.zeros_like(a),np.zeros_like(a)]
        for _ in range(30000):
            previous=x.copy()
            for i,n in enumerate((b,c)):
                y=x+corrections[i]
                x=y-min(0.,float(y@n))/(n@n)*n
                corrections[i]=y-x
            if np.linalg.norm(x-previous)<1e-13 and min(x@b,x@c)>=-1e-12:
                return x
        raise AssertionError('Independent reference did not converge')
    for _ in range(32):
        a, b, c = rng.normal(size=(3, 9))
        expected = independent_dykstra(a,b,c)
        actual, _ = G.project_two_halfspaces(*(torch.tensor(v, dtype=torch.float64) for v in (a,b,c)))
        assert np.allclose(actual.numpy(), expected, atol=2e-8, rtol=2e-8)
    a=torch.tensor([-1., -1., 4.],dtype=torch.float64)
    b=torch.tensor([1., 0., 0.],dtype=torch.float64)
    c=torch.tensor([-1., 1e-8, 0.],dtype=torch.float64)
    u,_=G.project_two_halfspaces(a,b,c)
    assert np.allclose(u.numpy(), [0.,0.,4.],atol=1e-8)


def test_guard_float32_cap_zero_cases_and_local_constraints():
    from audit_gate1_guarded_training_comparison import check_guard
    rng = np.random.default_rng(8)
    for _ in range(32):
        b,c,a=(torch.tensor(v,dtype=torch.float32) for v in rng.normal(size=(3,29)))
        out,receipt=G.guarded_gradient(b,c,100*a)
        u=(out-b).double()
        assert float(torch.linalg.vector_norm(u)) <= .25*float(torch.linalg.vector_norm(b))+2e-6
        assert float(u@b.double()) >= -2e-6 and float(u@c.double()) >= -2e-6
        assert np.isfinite(receipt['gram_float64']).all()
        check_guard(receipt)
        same,_=G.guarded_gradient(b,c,torch.zeros_like(a));assert torch.equal(same,b)
        zero,_=G.guarded_gradient(torch.zeros_like(b),c,a);assert torch.count_nonzero(zero)==0
    with pytest.raises(ValueError):G.project_two_halfspaces(torch.tensor([float('nan')]),torch.ones(1),torch.ones(1))


def test_zero_auxiliary_matches_ordinary_SGD_with_momentum_decay_and_partial_window():
    torch.manual_seed(7)
    baseline=torch.nn.Linear(3,2);guarded=deepcopy(baseline)
    ob=torch.optim.SGD(baseline.parameters(),lr=1e-4,momentum=.9,weight_decay=1e-4)
    og=torch.optim.SGD(guarded.parameters(),lr=1e-4,momentum=.9,weight_decay=1e-4)
    params=list(guarded.parameters()); accumulated=None; updates=0
    for i in range(5):
        x=torch.randn(4,3);y=torch.randn(4,2)
        denominator,update=W.W.accumulation_window(i,5)
        lb=(baseline(x)-y).square().mean()/denominator;lb.backward()
        lg=(guarded(x)-y).square().mean()/denominator
        v=G.flat_gradient(lg,params,retain_graph=False)
        if accumulated is None:accumulated=v
        else:accumulated.add_(v)
        if update:
            final,_=G.guarded_gradient(accumulated,accumulated,torch.zeros_like(accumulated))
            G.assign_gradient(params,final)
            torch.nn.utils.clip_grad_norm_(baseline.parameters(),10)
            torch.nn.utils.clip_grad_norm_(params,10)
            ob.step();og.step();ob.zero_grad(set_to_none=True);og.zero_grad(set_to_none=True)
            assert all(torch.equal(a,b) for a,b in zip(baseline.parameters(),guarded.parameters()))
            accumulated=None;updates+=1
    assert updates==2
    assert sum(W.W.accumulation_window(i,1133)[1] for i in range(1133))==284


def test_new_control_membership_reuses_both_old512_and_new4018_sources():
    c=P.read(W.OLD/'contract.json');reused={r['image_id'] for r in c['reused_teachers']}
    indices=[next(i for i,r in enumerate(c['training_members']) if (r['image_id'] in reused)==kind) for kind in (True,False)]
    root=Path(P.read(c['evaluation_protocol']['path'])['prepared_root'])
    for i in indices:
        meta=c['training_annotation_metadata'][i]
        record=P.read(W.OLD/'attempt_01/training_supervision'/f'{i:04d}.json')
        params,_=W.geometry_sources(record)
        gt=P.MATCH.parse_ground_truth((root/meta['annotation_path']).read_text(encoding='utf-8-sig'),meta['image_id'])
        with np.load(record['raw_source']['path'],allow_pickle=False) as d:r=d['r'].copy()
        out=W.derive_controls(record,r,gt,meta,params)
        assert out['selection'][W.ARMS[2]]==record['native_intervention_correction']
        assert len(out['selection'][W.ARMS[1]]['positive_gt'])==len(out['selection'][W.ARMS[2]]['positive_gt'])
        assert len(out['selection'][W.ARMS[1]]['errors'])==len(out['selection'][W.ARMS[2]]['errors'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA qualification required')
def test_cuda_original_loss_parity_ignore_guard_update_and_save_rebuild(tmp_path):
    P.configure(); ranges=P.read(P.PRIOR/'attempt_01/mse_ranges.json')['max_abs']
    model=P.new_model(ranges);model.train();P.freeze_bn(model)
    batch={'img':torch.zeros(2,3,640,640,device='cuda'), 'cls':torch.tensor([[0.],[1.]],device='cuda'),
           'bboxes':torch.tensor([[.25,.25,.05,.05],[.6,.6,.08,.08]],device='cuda'),
           'batch_idx':torch.tensor([0.,1.],device='cuda'),
           'ignore_boxes_xyxy':[torch.tensor([[400.,50.,500.,150.]],device='cuda') for _ in range(2)],
           'image_identities':[{'image_id':'synthetic/a'},{'image_id':'synthetic/b'}]}
    pred=model(batch['img']);criterion=G.GuardedCriterion(model)
    specs=[{'positive_gt':[0],'errors':[{'anchor':855,'category':i},{'anchor':7000,'category':i}]} for i in range(2)]
    base,classification,aux=criterion.components(pred,batch,specs)
    ours_aux=deepcopy(criterion.last_aux)
    reference=criterion.compute(pred,batch,[{'native_intervention_correction':s} for s in specs],'native_intervention_correction')
    assert torch.equal(base+aux,reference) and ours_aux==criterion.last_aux
    assert ours_aux['ignored_error_channels']==2 and criterion.last_mask_receipt['positive_rows_preserved']
    score_grad=torch.autograd.grad(aux,pred['scores'],retain_graph=True)[0]
    assert torch.count_nonzero(score_grad[:,:,855])==0
    parameters=[p for p in model.parameters() if p.requires_grad]
    gradients=[G.flat_gradient(x,parameters,retain_graph=i<2) for i,x in enumerate((base,classification,aux))]
    final,receipt=G.guarded_gradient(*gradients);G.assign_gradient(parameters,final)
    before={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    torch.nn.utils.clip_grad_norm_(parameters,10,error_if_nonfinite=True)
    torch.optim.SGD(parameters,lr=1e-4,momentum=.9,weight_decay=1e-4).step()
    assert any(not torch.equal(before[n],v.cpu()) for n,v in model.state_dict().items() if n.endswith('weight'))
    assert all(torch.equal(before[n],v.cpu()) for n,v in model.state_dict().items() if 'running_' in n or n.endswith('_scale') or '.dfl.' in n)
    model.eval();path=tmp_path/'guarded_synthetic.pt';torch.save(model.state_dict(),path)
    rebuilt=P.new_model(ranges);rebuilt.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    with torch.inference_mode():assert torch.equal(model(batch['img'][:1])[0],rebuilt(batch['img'][:1])[0])
    criterion._capture_handle.remove()
