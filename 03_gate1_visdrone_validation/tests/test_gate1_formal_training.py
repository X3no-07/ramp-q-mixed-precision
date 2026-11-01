"""Generated fixtures only; no original dataset/weights or GPU training here."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate1_ignore_dataset as D
import gate1_formal_training as F


@pytest.fixture
def bound_fixture(tmp_path):
    fixture = D.build_synthetic_fixture(tmp_path / "generated", train_count=4, val_count=2)
    manifests = {}
    for role, key in (("train", "train"), ("training_val", "val")):
        path = Path(fixture[key + "_manifest"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(schema=F.SCHEMA, data_role="synthetic_cuda_qualification_v1", policy=deepcopy(F.POLICY))
        path.write_text(json.dumps(value), encoding="utf-8")
        manifests[role] = {"path": path, "value": value, "sha": F.info(path)["sha256"]}
    return manifests


def construct(fixture, **kwargs):
    binding = fixture["train"]
    return F.BoundIgnoreDataset(binding["path"], binding["sha"], split_role="train",
        data_role="synthetic_cuda_qualification_v1", image_size=64, **kwargs)


def test_new_header_frozen_loader_and_collation(bound_fixture):
    dataset = construct(bound_fixture, augment=True, fliplr=1., flipud=1., seed=7)
    assert len(dataset) == 4
    sample = dataset[0]
    batch = dataset.collate_fn([sample, dataset[1]])
    assert batch["img"].shape == (2, 3, 64, 64) and batch["img"].dtype == torch.uint8
    assert batch["ignore_metadata"]["policy"] == F.POLICY
    assert batch["ignore_metadata"]["data_role"] == "synthetic_cuda_qualification_v1"
    assert all(len(v) == 3 for v in batch["ignore_boxes_xyxy"])
    dataset.set_epoch(2)
    assert dataset.epoch == 2


def test_old_synthetic_dataset_still_rejects_new_role(bound_fixture):
    binding = bound_fixture["train"]
    with pytest.raises(ValueError):
        D.RestrictedIgnoreDataset(binding["path"], binding["sha"], image_size=64)


@pytest.mark.parametrize("role", ["real_training_v1", "synthetic_only", "selection", "holdout"])
def test_declared_role_cannot_be_relabelled_by_caller(bound_fixture, role):
    binding = bound_fixture["train"]
    with pytest.raises(ValueError):
        F.BoundIgnoreDataset(binding["path"], binding["sha"], split_role="train", data_role=role, image_size=64)


def test_cross_manifest_id_and_content_leakage_rejected(bound_fixture):
    train = bound_fixture["train"]["value"]
    val = deepcopy(bound_fixture["training_val"]["value"])
    F.validate_membership(train, val)
    val["samples"][0]["image_id"] = train["samples"][0]["image_id"]
    with pytest.raises(ValueError, match="ID leakage"):
        F.validate_membership(train, val)
    val = deepcopy(bound_fixture["training_val"]["value"])
    val["samples"][0]["image"]["sha256"] = train["samples"][0]["image"]["sha256"]
    with pytest.raises(ValueError, match="content leakage"):
        F.validate_membership(train, val)


def test_member_corruption_is_not_silently_dropped(bound_fixture):
    dataset = construct(bound_fixture)
    path = dataset.root / dataset.entries[0]["image"]["path"]
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="byte identity mismatch"):
        dataset[0]


def test_duplicate_positive_label_still_rejects_entire_sample(bound_fixture):
    binding = bound_fixture["train"]
    value = binding["value"]
    labels = value["samples"][0]["labels"]
    path = binding["path"].parent / value["data_root"] / labels["path"]
    path.write_bytes(path.read_bytes() * 2)
    labels.update({k: v for k, v in F.info(path).items() if k != "path"})
    binding["path"].write_text(json.dumps(value), encoding="utf-8")
    binding["sha"] = F.info(binding["path"])["sha256"]
    with pytest.raises(ValueError, match="duplicate positive labels"):
        construct(bound_fixture)


def test_mixed_execution_roles_cannot_be_collated(bound_fixture):
    dataset = construct(bound_fixture)
    first, second = dataset[0], dataset[1]
    second["ignore_metadata"]["data_role"] = "real_training_v1"
    with pytest.raises(ValueError, match="cannot mix"):
        dataset.collate_fn([first, second])
