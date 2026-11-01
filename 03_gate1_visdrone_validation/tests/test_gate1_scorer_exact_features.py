from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import gate1_scorer_exact_features as O
S=O.S


def synthetic(n):
    rng=np.random.Generator(np.random.PCG64(2026102607+n))
    raw=rng.uniform(0,1,(1,14,n)).astype(np.float32)
    raw[0,:2]=rng.uniform(-20,660,(2,n));raw[0,2:4]=rng.uniform(.01,800,(2,n))
    if n>=4:
        raw[0,:4,:4]=np.array([[160],[160],[32],[32]])
        raw[0,4:,:4]=.1;raw[0,4,:4]=.8
    rows=[{'raw_anchor_index':i,'category':int(raw[0,4:,i].argmax())+1,'score':float(raw[0,4:,i].max()),'nms_input_order':i} for i in range(n)]
    return raw,rows


@pytest.mark.parametrize('n',[0,1,4,10,127,128,129,257,1000])
def test_fixed_block_features_bitwise_equal_all_boundary_sizes(n):
    raw,pred=synthetic(n)
    assert np.array_equal(O.visible_features(raw,pred),S.visible_features(raw,pred))


def test_original_order_tie_rule_and_frozen_score_are_exact():
    raw,pred=synthetic(257);x=O.visible_features(raw,pred);old=S.visible_features(raw,pred)
    model={'mean':[.1]*22,'scale':[.8]*22,'weight':np.linspace(-.1,.1,22).tolist(),'bias':-.2}
    p=np.array([r['score'] for r in pred])
    assert np.array_equal(S.score_rows(x,p,model),S.score_rows(old,p,model))
    tie_raw,tie_pred=synthetic(4);tie=O.visible_features(tie_raw,tie_pred)
    assert tie[0,9]==0 and tie[1,9]>=.999999


def test_optimized_features_retain_domain_checks():
    raw,pred=synthetic(5);pred[0]['category']=11
    with pytest.raises(ValueError):O.visible_features(raw,pred)
    raw,pred=synthetic(5);pred[1]['raw_anchor_index']=0
    with pytest.raises(ValueError):O.visible_features(raw,pred)
