from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import gate1_group_balanced_scorer as G


def test_prefix_weights_equalize_groups_and_images_without_candidate_bias():
    image=np.array([0,0,1,2,2,2,2]);prefix=['a','a','b']
    w=G.row_weights(image,prefix,True)
    assert np.isclose(w.sum(),1) and np.isclose(w[image==0].sum(),.25) and np.isclose(w[image==1].sum(),.25)
    assert np.isclose(w[image==2].sum(),.5)
    ordinary=G.row_weights(image,prefix,False)
    assert all(np.isclose(ordinary[image==i].sum(),1/3) for i in range(3))


def test_no_missing_fitted_image_silently_dropped():
    with pytest.raises(ValueError):G.row_weights(np.array([0,0,2]),['a','b','c'],True)


def test_weighting_is_permutation_invariant_and_preserves_zero_leakage():
    i=np.array([0,0,1,2]);prefix=['a','a','b'];order=np.array([3,0,2,1])
    assert np.array_equal(G.row_weights(i,prefix,True)[order],G.row_weights(i[order],prefix,True))


def tiny_training():
    rng=np.random.Generator(np.random.PCG64(2026102608));x=rng.normal(size=(18,22))
    y=np.r_[np.ones(9)*.8,np.zeros(9)];p=np.full(18,.5);image=np.tile(np.arange(3),6)
    return x,y,p,image,['a','a','b']


def test_quality8_never_learns_removed_identity_or_absolute_geometry_coefficients():
    G.P.configure();x,y,p,i,prefix=tiny_training()
    m,w=G.fit(x,y,p,i,prefix,'prefix_balanced_quality8')
    excluded=sorted(set(range(22))-set(G.QUALITY_INDICES))
    assert np.array_equal(np.array(m['weight'])[excluded],np.zeros(len(excluded)))
    assert m['active_parameter_slots']==9 and m['steps']==100 and not m['prefix_at_inference']
    changed=x.copy();changed[:,excluded]+=1000
    assert np.array_equal(G.S.score_rows(x,p,m),G.S.score_rows(changed,p,m))


def test_new_training_state_is_deterministic_and_save_replay_exact(tmp_path):
    G.P.configure();args=tiny_training()
    a,w=G.fit(*args,'prefix_balanced_full22');b,v=G.fit(*args,'prefix_balanced_full22')
    assert a==b and np.array_equal(w,v)
    path=tmp_path/'model.json';G.write(path,a)
    assert np.array_equal(G.S.score_rows(args[0],args[2],a),G.S.score_rows(args[0],args[2],G.read(path)))


def test_equal_size_groups_make_prefix_and_image_weighting_identical():
    x,y,p,i,_=tiny_training();prefix=['a','b','c']
    assert np.array_equal(G.row_weights(i,prefix,False),G.row_weights(i,prefix,True))


def test_uncorrected_allimagefull22_control_reproduces_existing_generic_fit():
    G.P.configure();x,y,p,i,_=tiny_training()
    old=G.S.fit_scorer(x,y,p,i,[0],[17],'score_base')
    # With one prefix, prefix balancing is precisely the old image-balanced objective.
    new,w=G.fit(x,y,p,i,['a','a','a'],'prefix_balanced_full22')
    assert np.array_equal(np.array(old['weight']),np.array(new['weight'])) and old['bias']==new['bias']
    assert old['mean']==new['mean'] and old['scale']==new['scale']
