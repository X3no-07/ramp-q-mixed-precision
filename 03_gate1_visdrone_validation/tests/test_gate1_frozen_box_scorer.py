from pathlib import Path
import inspect
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import gate1_frozen_box_scorer as S


def synthetic(n):
    raw=np.full((1,14,n),.01,np.float32)
    raw[0,:4]=np.array([[160],[160],[32],[32]])
    raw[0,4]=np.linspace(.9,.3,n)
    rows=[{"raw_anchor_index":i,"category":1,"score":float(raw[0,4,i])} for i in range(n)]
    return raw,rows


@pytest.mark.parametrize('n',[1,4,10,13])
def test_feature_dimensions_no_advanced_indexing_transpose_ambiguity(n):
    raw,rows=synthetic(n)
    x=S.visible_features(raw,rows)
    assert x.shape==(n,22) and np.isfinite(x).all()
    assert np.array_equal(x[:,12],np.ones(n))
    assert np.allclose(x[:,0],S.logit([r['score'] for r in rows]))
    assert np.allclose(x[:,8],np.log1p(n-1))
    if n>1:assert x[0,9]==0 and np.all(x[1:,9]==1)
    assert list(inspect.signature(S.visible_features).parameters)==['raw','predictions']


def test_gt_or_Q_dependent_fields_do_not_change_features():
    raw,rows=synthetic(4)
    a=S.visible_features(raw,rows)
    for row in rows:row.update(GT_label='fake',target_id='untrusted',q_difference=999)
    b=S.visible_features(raw,rows)
    assert np.array_equal(a,b)
    wrong=[dict(r) for r in rows];wrong[0]['category']=2
    with pytest.raises(ValueError,match='category'):S.visible_features(raw,wrong)


def test_training_normalization_and_identity_residual():
    x=np.arange(66,dtype=float).reshape(3,22)
    mean,scale=S.standardize_training(x)
    extreme=np.ones((5,22))*1e6
    before=mean.copy()
    model={'mean':mean.tolist(),'scale':scale.tolist(),'weight':[0.]*22,'bias':0.}
    p=np.array([.1,.5,.9])
    assert np.allclose(S.score_rows(x,p,model),p,atol=1e-15,rtol=0)
    S.score_rows(extreme,np.ones(5)*.5,model)
    assert np.array_equal(mean,before)


def test_tied_scores_cannot_be_split_to_claim_FP_budget_success():
    scores=np.array([.9,.9,.8,.7]);flags=np.array([1,0,1,0])
    curve=S.grouped_curve(scores,flags)
    assert curve['TP'].tolist()==[1,2,2] and curve['FP'].tolist()==[1,1,2]
    assert S.at_fp_budget(curve,0)['TP']==0
    assert S.at_fp_budget(curve,1)['TP']==2
    assert S.at_tp_target(curve,2)['FP']==1
    assert not S.at_tp_target(curve,3)['achievable']


def test_score_curve_ignores_absorbed_detections_and_empty_domain():
    curve=S.grouped_curve(np.array([.9,.8,.7]),np.array([-1,1,0]))
    assert curve['TP'].tolist()==[0,1,1] and curve['FP'].tolist()==[0,0,1]
    empty=S.grouped_curve(np.empty(0),np.empty(0))
    assert S.at_fp_budget(empty,10)['TP']==0


def test_actual_scorer_training_is_deterministic_and_saved_state_replays(tmp_path):
    S.P.configure()
    x=np.zeros((12,22));x[:,0]=np.r_[np.ones(6),-np.ones(6)]
    y=np.r_[np.ones(6)*.9,np.zeros(6)];p=np.ones(12)*.5;image=np.tile(np.arange(3),4)
    a=S.fit_scorer(x,y,p,image,[0,1],[10,11],'score_intervention')
    b=S.fit_scorer(x,y,p,image,[0,1],[10,11],'score_intervention')
    assert a==b and a['steps']==100
    scores=S.score_rows(x,p,a)
    assert scores[:6].mean()>.7 and scores[6:].mean()<.2
    dest=tmp_path/'scorer.json';S.write(dest,a)
    assert np.array_equal(scores,S.score_rows(x,p,S.read(dest)))
    assert a['trace'][-1]['loss']<a['trace'][0]['loss']


def test_official_rank_cap_remains_across_classes():
    gt=np.array([[10,10,10,10,0,1,0,0],[30,30,10,10,0,2,0,0]],float)
    dt=np.array([[10,10,10,10,.9,1,-1,-1]]*500+[[30,30,10,10,.8,2,-1,-1]],float)
    score,flag=S.official_rank_entries([gt],[dt])
    assert len(score)==500 and np.count_nonzero(flag==1)==1
