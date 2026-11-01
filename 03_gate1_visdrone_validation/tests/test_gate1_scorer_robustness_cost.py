from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import gate1_scorer_robustness_cost as D


def test_group_leave_one_out_exposes_pooled_advantage_concentration():
    rows=[{'prefix_proxy':'a','TP_delta':10,'FP_delta':-20},{'prefix_proxy':'b','TP_delta':1,'FP_delta':5}]
    r=D.group_sensitivity(rows)
    assert r['pooled_TP_delta']==11 and r['pooled_FP_delta']==-15 and r['pooled_joint_observation']
    assert r['leave_one_prefix_joint_count']==1 and not r['all_leave_one_prefix_joint_observation']
    assert not r['leave_one_prefix_out'][0]['remaining_joint_observation']


def test_same_prefix_images_are_removed_as_one_group():
    rows=[{'prefix_proxy':'a','TP_delta':3,'FP_delta':-2},{'prefix_proxy':'a','TP_delta':4,'FP_delta':-3},{'prefix_proxy':'b','TP_delta':0,'FP_delta':0}]
    r=D.group_sensitivity(rows)
    assert r['prefix_proxies']==2 and r['leave_one_prefix_out'][0]['omitted_images']==2
    assert r['leave_one_prefix_out'][0]['remaining_TP_delta']==0 and r['leave_one_prefix_out'][0]['remaining_FP_delta']==0
    assert r['all_leave_one_prefix_joint_observation']


@pytest.mark.parametrize('ious,classes,want',[
    ([.5],[1],D.KINDS[0]),([.5],[2],D.KINDS[1]),([.1],[2],D.KINDS[2]),([.099],[1],D.KINDS[3]),([],[],D.KINDS[3])])
def test_FP_geometry_partition_boundaries_preserved(ious,classes,want):
    assert D.partition(np.array(ious),np.array(classes),1)==want


def test_timing_summary_preserves_units_and_rejects_invalid_samples():
    r=D.summarize_timing([1.,2.,3.,4.])
    assert r['samples']==4 and r['median_ms']==2.5 and r['maximum_ms']==4 and r['minimum_ms']==1
    assert r['p95_ms']==np.percentile([1.,2.,3.,4.],95)
    with pytest.raises(ValueError):D.summarize_timing([float('nan')])
    with pytest.raises(ValueError):D.summarize_timing([])


def test_current_feature_score_sort_benchmark_path_preserves_exact_output():
    raw=np.full((1,14,4),.01,np.float32);raw[0,:4]=np.array([[160.],[160.],[32.],[32.]])
    raw[0,4]=np.array([.9,.8,.7,.6])
    pred=[{'raw_anchor_index':i,'category':1,'score':float(raw[0,4,i]),'nms_input_order':i} for i in range(4)]
    x=D.S.visible_features(raw,pred)
    model={'mean':[0.]*22,'scale':[1.]*22,'weight':[0.]*22,'bias':0.}
    p=np.array([d['score'] for d in pred]);scores=D.S.score_rows(x,p,model)
    assert np.allclose(scores,p,rtol=0,atol=1e-15)
    order=sorted(range(len(pred)),key=lambda i:(-float(scores[i]),pred[i]['nms_input_order']))
    assert order==[0,1,2,3] and x.shape==(4,22)
