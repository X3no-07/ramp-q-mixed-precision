from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import gate1_separate_threshold as T


def member(i,prefix,sha=None):return {'image_id':f'train/{prefix}_{i}','image':{'sha256':sha or str(i)}}


def test_calibration_uses_all_and_only_fit_prefix_disjoint_members():
    train=[member(0,'a'),member(1,'a'),member(2,'b'),member(3,'c')]
    selected=T.choose_calibration_members(train,train[:1])
    assert selected==train[2:]
    assert T.choose_calibration_members(list(reversed(train)),train[:1])==selected


def test_duplicate_and_content_identity_overlaps_rejected():
    a=member(0,'a');b=member(1,'b',sha='0')
    with pytest.raises(ValueError,match='hash overlap'):T.choose_calibration_members([a,b],[a])
    with pytest.raises(ValueError,match='Duplicate'):T.choose_calibration_members([a,a],[a])


def test_threshold_budget_matches_actual_detection_flags_and_preserves_ties():
    rs=np.array([.8,.5,.2]);rf=np.array([1,0,0]);ss=np.array([.8,.6,.6,.4]);sf=np.array([1,0,0,1])
    result=T.select_threshold(rs,rf,ss,sf)
    assert result['reference_calibration_counts']['FP']==1
    assert result['threshold']==.8 and result['chosen_calibration_counts']['FP']==0
    assert result['calibration_TP_delta']==0 and not result['uses_training_val_labels']


def test_no_feasible_nonempty_group_returns_fixed_abstention():
    result=T.select_threshold(np.array([.8]),np.array([1]),np.array([.9,.9]),np.array([0,1]))
    assert result['threshold']==1. and result['chosen_calibration_counts']['FP']==0 and result['chosen_calibration_counts']['TP']==0


def test_ignored_detections_never_consume_false_positive_budget():
    result=T.select_threshold(np.array([.8,.4,.3]),np.array([1,0,-1]),np.array([.8,.7,.6,.5]),np.array([1,-1,0,1]))
    assert result['threshold']==.5 and result['chosen_calibration_counts']=={'TP':2,'FP':1,'ignored':1}


def test_threshold_selection_does_not_mutate_inputs_and_no_dev_argument():
    import inspect
    args=[np.array([.8,.5,.2]),np.array([1,0,0]),np.array([.9,.6,.3]),np.array([1,0,0])]
    before=[x.copy() for x in args];T.select_threshold(*args)
    assert all(np.array_equal(x,b) for x,b in zip(args,before))
    assert 'development' not in str(inspect.signature(T.select_threshold))


def test_frozen_threshold_counts_use_inclusive_whole_group_cutoff():
    result=T.fixed_counts(np.array([.5,.5,.4]),np.array([1,0,-1]),.5)
    assert result=={'TP':1,'FP':1,'ignored':0}
    with pytest.raises(ValueError):T.fixed_counts(np.array([float('nan')]),np.array([1]),.5)


def test_small_saved_matrix_match_flags_and_threshold_apply_without_AP_rerun():
    gt=[np.array([[10,10,10,10,0,1,0,0]],float)]
    r=[np.array([[10,10,10,10,.7,1,-1,-1],[80,80,10,10,.5,1,-1,-1]],float)]
    s=[np.array([[10,10,10,10,.8,1,-1,-1],[80,80,10,10,.3,1,-1,-1]],float)]
    rs,rf=T.J.rank_entries(gt,r);ss,sf=T.J.rank_entries(gt,s)
    result=T.select_threshold(rs,rf,ss,sf)
    assert result['threshold']==.3 and result['chosen_calibration_counts']['TP']==1 and result['chosen_calibration_counts']['FP']==1
