import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import gate1_visual_evidence_probe as V


def test_constant_context_has_zero_contrast():
    v=V.local_visual(torch.ones((1,64,80,80)),np.array([[320,320,20,20]],np.float32))
    assert v.shape==(1,128) and np.array_equal(v[0,:64],np.ones(64)) and np.array_equal(v[0,64:],np.zeros(64))


def test_canvas_feature_center_alignment():
    f=torch.arange(80,dtype=torch.float32)[None,None,None,:].expand(1,64,80,80).clone()
    v=V.local_visual(f,np.array([[4,4,8,8],[636,636,8,8],[320,320,8,8]],np.float32))
    assert np.allclose(v[:,0],[0,79,39.5],atol=1e-5)


def test_border_replication_and_finite_outside():
    f=torch.full((1,64,80,80),2.)
    v=V.local_visual(f,np.array([[-10,800,100,100]],np.float32))
    assert np.allclose(v[0,:64],2) and np.allclose(v[0,64:],0)


def test_empty_pool():
    assert V.local_visual(torch.zeros((1,64,80,80)),np.empty((0,4))).shape==(0,128)


def test_invalid_width_rejected():
    with pytest.raises(ValueError):V.local_visual(torch.zeros((1,64,80,80)),np.array([[0,0,0,8]],np.float32))


def test_wrong_layer_shape_rejected():
    with pytest.raises(ValueError):V.local_visual(torch.zeros((1,128,80,80)),np.array([[10,10,8,8]],np.float32))
