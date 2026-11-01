"""Only generated synthetic PNG/YOLO/JSONL bytes; no real sample/model/GPU."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("gate1_ignore_dataset",ROOT/"scripts/gate1_ignore_dataset.py")
d=importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture
def fx(tmp_path_factory):
    parent=tmp_path_factory.mktemp("dataset_fixture")
    return d.build_synthetic_fixture(parent/"new",train_count=4,val_count=2)


def construct(fx,**kwargs):
    return d.RestrictedIgnoreDataset(fx["train_manifest"],fx["train_manifest_sha256"],image_size=64,**kwargs)


def revise(fx, fn):
    path=Path(fx["train_manifest"])
    value=json.loads(path.read_text(encoding="utf-8"))
    fn(value)
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.write_bytes(raw)
    fx["train_manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    return value


def replace_member(fx,key,raw):
    def change(value):
        info=value["samples"][0][key]
        path=Path(fx["train_manifest"]).parent/value["data_root"]/info["path"]
        path.write_bytes(raw)
        info.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return revise(fx,change)


def test_synthetic_six_images_unique_and_split_memberships_disjoint(fx):
    train=construct(fx)
    val=d.RestrictedIgnoreDataset(fx["val_manifest"],fx["val_manifest_sha256"],split_role="training_val",image_size=64)
    assert len(train)==4 and len(val)==2
    assert {r["image_id"] for r in train.entries}.isdisjoint({r["image_id"] for r in val.entries})
    assert len({r["image"]["sha256"] for r in train.entries+val.entries})==6
    assert train.rect is False and len(train.labels)==4
    np.testing.assert_array_equal(train.labels[0]["cls"],[[0],[1]])


@pytest.mark.parametrize("horizontal,vertical",[(0.,0.),(1.,0.),(0.,1.),(1.,1.)])
def test_image_positive_ignore_share_exact_params(fx,horizontal,vertical):
    dataset=construct(fx,augment=True,fliplr=horizontal,flipud=vertical)
    sample=dataset[0]
    h0,w0=dataset.entries[0]["original_hw"]
    trace=sample["transform_trace"]
    rh,rw=trace["loader_resized_hw"]
    params=trace["letterbox"]
    expected=np.array([[.15*w0,.15*h0,.35*w0,.35*h0],[.65*w0,.65*h0,.85*w0,.85*h0]])
    expected*=np.array([rw/w0,rh/h0,rw/w0,rh/h0])
    expected*=params["ratio"][0]
    expected+=np.array([params["left"],params["top"],params["left"],params["top"]])
    if vertical: expected[:,[1,3]]=64-expected[:,[3,1]]
    if horizontal: expected[:,[0,2]]=64-expected[:,[2,0]]
    boxes=sample["bboxes"].numpy()*64
    actual=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),axis=1)
    np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=0)
    assert sample["img"].dtype==torch.uint8 and tuple(sample["img"].shape)==(3,64,64)
    # Independently transform source ignore rows, including actual ceil axes.
    source=np.array([[w0*.05,h0*.05,w0*.95,h0*.95],[w0*.45,h0*.2,w0*.65,h0*.45],[w0*.7,h0*.6,w0*.9,h0*.9]])
    source*=np.array([rw/w0,rh/h0,rw/w0,rh/h0])
    source*=params["ratio"][0]
    source+=np.array([params["left"],params["top"],params["left"],params["top"]])
    if vertical: source[:,[1,3]]=64-source[:,[3,1]]
    if horizontal: source[:,[0,2]]=64-source[:,[2,0]]
    np.testing.assert_allclose(sample["ignore_sample"]["boxes"],source,atol=1e-12,rtol=0)
    # Nonpadding RGB first image pixel follows the same flips as source BGR.
    src=d.cv2.imdecode(np.fromfile(dataset.root/dataset.entries[0]["image"]["path"],dtype=np.uint8),d.cv2.IMREAD_COLOR)
    img=d.cv2.resize(src,(rw,rh),interpolation=d.cv2.INTER_LINEAR)
    img=d.cv2.copyMakeBorder(img,params["top"],params["bottom"],params["left"],params["right"],d.cv2.BORDER_CONSTANT,value=(114,114,114))
    if vertical: img=img[::-1]
    if horizontal: img=img[:,::-1]
    np.testing.assert_array_equal(sample["img"].numpy(),img[:,:,::-1].transpose(2,0,1))


def test_ragged_batch_regular_contract_and_source_reasons(fx):
    ds=construct(fx)
    samples=[ds[0],ds[1]]
    batch=ds.collate_fn(samples)
    assert tuple(batch["img"].shape)==(2,3,64,64)
    assert tuple(batch["cls"].shape)==(4,1) and tuple(batch["bboxes"].shape)==(4,4)
    assert batch["batch_idx"].tolist()==[0,0,1,1]
    assert len(batch["ignore_boxes_xyxy"])==2 and batch["ignore_boxes_xyxy"][0].dtype==torch.float32
    assert [r["source_reasons"] for r in batch["source_records"][0]]==[("score_0",),("category_0",),("category_11",)]
    assert batch["sample_identities"][0]["label_sha256"]==ds.entries[0]["labels"]["sha256"]
    assert batch["image_hw"]==(64,64)


def test_explicit_empty_ignore_and_empty_positive_not_silently_invented(fx):
    replace_member(fx,"sidecar",b"")
    replace_member(fx,"labels",b"")
    ds=construct(fx)
    batch=ds.collate_fn([ds[0],ds[1]])
    assert tuple(batch["ignore_boxes_xyxy"][0].shape)==(0,4)
    assert batch["batch_idx"].tolist()==[1,1]
    assert len(ds.labels[0]["cls"])==0


@pytest.mark.parametrize("key",["image","labels","sidecar"])
def test_member_hash_changed_after_init_rejected(fx,key):
    ds=construct(fx)
    p=ds.root/ds.entries[0][key]["path"]
    p.write_bytes(p.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="byte identity"):
        ds[0]


def test_manifest_hash_changed_after_init_rejected(fx):
    ds=construct(fx)
    p=Path(fx["train_manifest"])
    p.write_bytes(p.read_bytes()+b" ")
    with pytest.raises(ValueError,match="manifest changed"):
        ds[0]


def test_manifest_wrong_digest_before_member_read(fx,monkeypatch):
    monkeypatch.setattr(d,"_read_member",lambda *a: (_ for _ in ()).throw(AssertionError("should not read member")))
    with pytest.raises(ValueError,match="manifest SHA"):
        d.RestrictedIgnoreDataset(fx["train_manifest"],"0"*64,image_size=64)


@pytest.mark.parametrize("role",["selection","holdout","test-dev","val"])
def test_forbidden_role_before_member_read(fx,monkeypatch,role):
    monkeypatch.setattr(d,"_read_member",lambda *a: (_ for _ in ()).throw(AssertionError("should not read member")))
    with pytest.raises(ValueError,match="roles"):
        construct(fx,split_role=role)


@pytest.mark.parametrize("path",["selection/000.png","holdout/000.png","test-dev/000.png","../000.png","[OMITTED_LOCAL_PATH]"])
def test_forbidden_member_path_before_any_member_read(fx,monkeypatch,path):
    revise(fx,lambda m:m["samples"][3]["image"].update(path=path))
    monkeypatch.setattr(d,"_read_member",lambda *a: (_ for _ in ()).throw(AssertionError("should not read any member")))
    with pytest.raises(ValueError,match="forbidden"):
        construct(fx)


def test_train_namespace_does_not_replace_explicit_role_membership(fx,monkeypatch):
    revise(fx,lambda m:m["samples"][3].update(split_role="training_val"))
    monkeypatch.setattr(d,"_read_member",lambda *a: (_ for _ in ()).throw(AssertionError("should not read member")))
    with pytest.raises(ValueError,match="not a member"):
        construct(fx)


def test_unlisted_file_not_read_and_unlisted_index_rejected(fx,monkeypatch):
    real=d._read_member
    reads=[]
    def spy(root,info):
        reads.append(info["path"])
        return real(root,info)
    monkeypatch.setattr(d,"_read_member",spy)
    extra=Path(fx["train_manifest"]).parent/"files"/"unlisted.png"
    extra.write_bytes(b"THIS_UNLISTED_FILE_MUST_NOT_BE_READ")
    ds=construct(fx)
    assert "unlisted.png" not in reads and len(reads)==12
    with pytest.raises(ValueError,match="membership"):
        ds[4]


def test_mutating_entry_cannot_grant_unlisted_membership(fx):
    ds=construct(fx)
    ds.entries[0]["image_id"]="train/unlisted_but_prefix_looks_valid"
    with pytest.raises(ValueError,match="membership"):
        ds[0]


@pytest.mark.parametrize("key,value",[("original_hw",[99,99]),("image_id","train/duplicate")])
def test_dimensions_and_identity_fail_closed(fx,key,value):
    if key=="image_id":
        revise(fx,lambda m:[s.update(image_id=value) for s in m["samples"][:2]])
        match="duplicate"
    else:
        revise(fx,lambda m:m["samples"][0].update({key:value}))
        match="dimensions"
    with pytest.raises(ValueError,match=match):
        construct(fx)


@pytest.mark.parametrize("raw",[b"0 .5 .5 0 .2\n",b"10 .5 .5 .2 .2\n",b"0 nan .5 .2 .2\n",b"0 .9 .5 .4 .2\n",b"\n",b"0 .5 .5 .2 .2\n0 .5 .5 .2 .2\n"])
def test_invalid_positive_labels_reject_whole_sample(fx,raw):
    replace_member(fx,"labels",raw)
    with pytest.raises(ValueError):
        construct(fx)


@pytest.mark.parametrize("mutate",[
    lambda row:row.update(image_id="train/wrong"),
    lambda row:row.update(original_xywh=[1,1,0,2],original_xyxy=[1,1,1,3],clipped_xyxy=[1,1,1,3]),
    lambda row:row.update(original_xywh=[999,1,1,2],original_xyxy=[999,1,1000,3],clipped_xyxy=[101,1,101,3]),
    lambda row:row.update(excluded_for_preparation_quality=True),
])
def test_invalid_source_sidecar_rejects_whole_sample(fx,mutate):
    manifest=json.loads(Path(fx["train_manifest"]).read_text())
    info=manifest["samples"][0]["sidecar"]
    path=Path(fx["train_manifest"]).parent/"files"/info["path"]
    rows=[json.loads(v) for v in path.read_text().splitlines()]
    mutate(rows[0])
    replace_member(fx,"sidecar",("\n".join(json.dumps(v) for v in rows)+"\n").encode())
    with pytest.raises(ValueError):
        construct(fx)


def test_manifest_real_data_role_rejected(fx):
    revise(fx,lambda m:m.update(data_role="real_training"))
    with pytest.raises(ValueError,match="synthetic fixtures only"):
        construct(fx)


def test_deterministic_epoch_flips_and_no_global_rng_dependency(fx):
    ds=construct(fx,augment=True,fliplr=.5,flipud=.5,seed=19)
    a=ds[0]
    np.random.seed(999)
    b=ds[0]
    assert torch.equal(a["img"],b["img"])
    assert a["transform_trace"]==b["transform_trace"]
    ds.set_epoch(2)
    c=ds[0]
    ds.set_epoch(0)
    assert a["transform_trace"]==ds[0]["transform_trace"]
    assert ds.epoch==0 and c["img"].shape==a["img"].shape


def test_batch_order_and_identity_mix_rejected(fx):
    ds=construct(fx)
    a,b=ds[0],ds[1]
    batch=ds.collate_fn([b,a])
    assert batch["image_identities"][0]["image_id"]==b["image_identities"]["image_id"]
    a["image_identities"]=copy.deepcopy(b["image_identities"])
    with pytest.raises(ValueError):
        ds.collate_fn([a,b])


def test_same_geometry_after_constructor_labels_mutation(fx):
    ds=construct(fx)
    expected=ds[0]["bboxes"].clone()
    ds.labels[0]["bboxes"][:]=0
    assert torch.equal(expected,ds[0]["bboxes"])


def test_plain_torch_dataloader_actual_batch(fx):
    ds=construct(fx,augment=True,fliplr=1.,flipud=1.)
    loader=torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0,shuffle=False,collate_fn=ds.collate_fn)
    batches=list(loader)
    assert len(batches)==2 and sum(len(b["image_identities"]) for b in batches)==4
    assert torch.isfinite(batches[0]["bboxes"]).all()


@pytest.mark.parametrize("kwargs",[{"split_role":"training_val","augment":True,"fliplr":1.}, {"augment":False,"fliplr":1.}, {"image_size":False},{"fliplr":float("nan")}])
def test_unsupported_constructor_options_fail_closed(fx,kwargs):
    opts={"image_size":64,**kwargs}
    with pytest.raises(ValueError):
        d.RestrictedIgnoreDataset(fx["train_manifest"],fx["train_manifest_sha256"],**opts)
