from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import gate1_scorer_extension as E
import gate1_cached_ap_job as J


def row(k,prefix):return {'image_id':str(k),'image_sha256':str(k),'prefix_proxy':prefix}


def test_new_and_new_prefix_members_are_selected_by_metadata_only():
    full=[row(0,'a'),row(1,'b'),row(2,'a'),row(3,'c'),row(4,'d')]
    result=E.partition_members(full,full[:2])
    assert [r['image_id'] for r in result['new286']]==['2','3','4']
    assert [r['image_id'] for r in result['new_prefix47']]==['3','4']
    assert result['old128']==full[:2] and result['full414']==full


def test_partition_rejects_duplicates_and_content_overlap():
    with pytest.raises(ValueError):E.partition_members([row(0,'a'),row(0,'a')],[row(0,'a')])
    bad=[row(0,'a'),{**row(1,'b'),'image_sha256':'0'}]
    with pytest.raises(ValueError,match='hash overlap'):E.partition_members(bad,[bad[0]])


def test_training_only_threshold_never_splits_a_tied_negative_group():
    original=np.array([.9,.9,.1,.1]);target=np.array([1.,0.,0.,1.]);score=np.array([.8,.7,.7,.5])
    result=E.select_training_threshold(original,target,score)
    assert result['nominal_train_FP_budget']==1 and result['threshold']==.8
    assert result['chosen_train_point']['FP']==0 and not result['uses_development_labels']


def test_threshold_is_derived_from_training_scores_without_model_fit():
    original=np.array([.6,.5,.2,.1]);target=np.array([.8,0.,0.,.7]);score=np.array([.8,.4,.2,.3])
    a=E.select_training_threshold(original,target,score)
    b=E.select_training_threshold(original.copy(),target.copy(),score.copy())
    assert a==b and a['chosen_train_point']['FP']<=a['nominal_train_FP_budget']


def test_state_fingerprint_detects_mutation_and_preserves_no_grad_inference():
    model=torch.nn.Conv2d(1,2,1)
    before=E.state_value_fingerprint(model)
    with torch.inference_mode():model(torch.ones(1,1,8,8))
    assert E.state_value_fingerprint(model)==before
    with torch.no_grad():model.weight.add_(.01)
    assert E.state_value_fingerprint(model)!=before


def test_lightweight_AP_helper_whole_ties_equal_existing_curve_definition():
    score=np.array([.9,.9,.5,.4]);flag=np.array([1,0,-1,1])
    a=J.curve_from_entries(score,flag);b=E.S.grouped_curve(score,flag)
    assert all(np.array_equal(a[k],b[k]) for k in a)


def test_actual_lightweight_AP_process_handles_bound_matrix_and_never_imports_torch(tmp_path):
    gt=np.array([[10,10,10,10,0,1,0,0]],float)
    dt=np.array([[10,10,10,10,.9,1,-1,-1],[100,100,10,10,.8,1,-1,-1],[10,10,10,10,.7,1,-1,-1]],float)
    inp=tmp_path/'matrices.npz';np.savez_compressed(inp,gt_0=gt,dt_0=dt)
    out=tmp_path/'out';out.mkdir();job=tmp_path/'job.json'
    E.write(job,{'cohort':'synthetic','branch':'fixed','images':1,'matrices':E.identity(inp),'output':str(out)})
    run=subprocess.run([sys.executable,'-B','-X','utf8',str(E.ROOT/'scripts/gate1_cached_ap_job.py'),str(job)],
                       capture_output=True,text=True,encoding='utf-8',timeout=30,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert run.returncode==0,run.stderr
    result=E.read(out/'metrics.json')
    assert not result['torch_imported'] and result['metrics_percent']['AP_all']==100.
    with np.load(out/'official_IoU05_rank_curve.npz',allow_pickle=False) as d:
        assert d['flag'].tolist()==[1,0,0] and d['TP'].tolist()==[1,1,1] and d['FP'].tolist()==[0,1,2]


def test_AP_helper_rejects_changed_input_digest_without_overwriting_results(tmp_path):
    inp=tmp_path/'matrices.npz';np.savez_compressed(inp,gt_0=np.empty((0,8)),dt_0=np.empty((0,8)))
    binding=E.identity(inp);inp.write_bytes(b'changed')
    out=tmp_path/'out';out.mkdir();job=tmp_path/'job.json'
    E.write(job,{'cohort':'synthetic','branch':'fixed','images':1,'matrices':binding,'output':str(out)})
    with pytest.raises(ValueError,match='binding differs'):J.run(job)
    assert not (out/'metrics.json').exists()
