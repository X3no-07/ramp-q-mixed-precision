"""Preparation safety/semantics tests; synthetic images only, no model or holdout."""
import importlib.util
import io
import json
from pathlib import Path
import stat
import struct
import sys
import zipfile

from PIL import Image
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_gate1_visdrone.py"
SPEC = importlib.util.spec_from_file_location("gate1_preparation_under_test", SCRIPT)
PREP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREP
SPEC.loader.exec_module(PREP)


def png(index=0):
    stream = io.BytesIO()
    Image.new("RGB", (20, 10), (index % 256, index // 256, 51)).save(stream, format="PNG")
    return stream.getvalue()


def archive(path, split, count, start=0, annotation="1,2,5,4,1,4,0,0\n"):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as output:
        for i in range(count):
            stem = f"{split}{i // 2:05d}_{i:05d}_d_{i:07d}"
            output.writestr(f"VisDrone2019-DET-{split}/images/{stem}.png", png(start + i))
            output.writestr(f"VisDrone2019-DET-{split}/annotations/{stem}.txt", annotation)


def record(image_id, split, prefix, sha, annotation_sha="label"):
    return {"image_id": image_id, "source_split": split, "prefix_proxy": prefix,
            "image_sha256": sha, "annotation_sha256": annotation_sha}


def policy():
    return PREP.SplitPolicy(training_val_target=2, calibration_pool_target=4,
                            debug_target=2, calibration_per_seed=1)


def rows():
    return [record(f"train/{i}", "train", f"t{i // 2}", f"sha{i}") for i in range(30)] + [
        record("val/a", "val", "v1", "valsha")]


@pytest.mark.parametrize("bad", ["../outside", "/outside", "C:/outside", "VisDrone2019-DET-train/../outside",
    "VisDrone2019-DET-train/images\\outside", "VisDrone2019-DET-train/images/NUL.txt",
    "VisDrone2019-DET-train/images/a.jpg.", "VisDrone2019-DET-test-dev/images/secret.jpg"])
def test_unsafe_zip_members_rejected(tmp_path, bad):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as output:
        output.writestr(bad, b"unsafe")
    if "\\" in bad:
        # Windows ZipInfo normalizes separators when writing; patch both local
        # and central raw names to exercise a genuinely unsafe source archive.
        path.write_bytes(path.read_bytes().replace(bad.replace("\\", "/").encode(), bad.encode()))
    with zipfile.ZipFile(path) as source, pytest.raises(PREP.PreparationError):
        PREP.validated_members(source, "train")
    assert not (tmp_path / "outside").exists()


def test_duplicate_case_colliding_and_symlink_members_rejected(tmp_path):
    for kind in ("case", "link"):
        path = tmp_path / f"{kind}.zip"
        with zipfile.ZipFile(path, "w") as output:
            if kind == "case":
                output.writestr("VisDrone2019-DET-train/images/a.jpg", b"a")
                output.writestr("VisDrone2019-DET-train/images/A.JPG", b"b")
            else:
                member = zipfile.ZipInfo("VisDrone2019-DET-train/images/link.jpg")
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(member, b"../../outside")
        with zipfile.ZipFile(path) as source, pytest.raises(PREP.PreparationError):
            PREP.validated_members(source, "train")


def test_annotation_ignore_semantics_physical_ids_and_clipping():
    data = ("\n1,2,5,4,1,1,0,0\n1,2,5,4,0,4,0,0\n"
            "1,2,5,4,1,0,0,0\n1,2,5,4,1,11,0,0\n"
            "-2,1,6,4,1,10,0,0\n30,1,6,4,1,2,0,0\n").encode()
    labels, targets = PREP.convert_annotation(data, 20, 10, "train/example")
    assert labels.splitlines() == ["0 0.175 0.4 0.25 0.4", "9 0.1 0.3 0.2 0.4"]
    assert targets[0]["target_id"] == "train/example:1"
    assert targets[1]["ignore_reasons"] == ["score_0"]
    assert targets[2]["ignore_reasons"] == ["category_0"]
    assert targets[3]["ignore_reasons"] == ["category_11"]
    assert targets[4]["original_xywh"] == [-2, 1, 6, 4]
    assert targets[4]["clipped_xyxy"] == [0, 1, 4, 5]
    assert targets[4]["was_clipped"] is True
    assert targets[5]["ignore_reasons"] == ["empty_after_clipping"]


def test_non_positive_ignored_geometry_preserved_without_yolo_targets():
    data = b"1008,374,3,0,0,0,0,0\n1,2,0,0,1,11,0,0\n"
    labels, targets = PREP.convert_annotation(data, 1920, 1080, "train/ignored_geometry")
    assert labels == ""
    assert len(targets) == 2
    assert targets[0]["target_id"] == "train/ignored_geometry:0"
    assert targets[0]["raw_line"] == "1008,374,3,0,0,0,0,0"
    assert targets[0]["original_xywh"] == [1008, 374, 3, 0]
    assert targets[1]["original_xywh"] == [1, 2, 0, 0]
    for target in targets:
        assert target["ignored"] is True
        assert target["yolo_line_zero_based"] is None
        assert "non_positive_ignored_geometry" in target["ignore_reasons"]
    with pytest.raises(PREP.PreparationError, match="non-positive box size"):
        PREP.convert_annotation(b"1,2,3,0,1,1,0,0\n", 20, 10, "train/valid_class_zero_height")


def test_explicit_train_invalid_positive_preserved_and_entire_component_quarantined(tmp_path):
    invalid = b"611,158,4,0,1,4,0,0\n"
    labels, targets = PREP.convert_annotation(invalid, 1920, 1080, "train/bad",
                                             preserve_invalid_positive_geometry=True)
    assert labels == ""
    target = targets[0]
    assert target["raw_line"] == "611,158,4,0,1,4,0,0"
    assert target["score"] == 1 and target["category_original"] == 4
    assert target["publisher_ignored"] is False
    assert target["excluded_for_preparation_quality"] is True
    assert "invalid_positive_box_geometry" in target["ignore_reasons"]
    train_zip = tmp_path / "train.zip"
    with zipfile.ZipFile(train_zip, "w") as output:
        output.writestr("VisDrone2019-DET-train/images/quality_00001.png", png(200))
        output.writestr("VisDrone2019-DET-train/annotations/quality_00001.txt", invalid)
    extracted, counts = PREP.extract_split(train_zip, "train", tmp_path / "stage")
    assert extracted[0]["requires_annotation_quality_quarantine"] is True
    assert counts["annotation_quality_excluded_positive_targets"] == 1
    source = rows() + extracted + [record("train/quality_neighbor", "train", "quality", "unrelated_neighbor_sha")]
    result = PREP.split_records(source, policy())
    quality_rows = [row for row in source if row["prefix_proxy"] == "quality"]
    assert len(quality_rows) == 2
    assert all(row["assigned_split"] == "quarantine" for row in quality_rows)
    assert all("component_contains_invalid_positive_box_geometry" in row["quarantine_reasons"] for row in quality_rows)
    assert result["annotation_quality_quarantine"]["quarantined_train_image_count"] == 2
    assert result["annotation_quality_quarantine"]["invalid_positive_target_ids"] == ["train/quality_00001:0"]
    # Adding the wholly isolated quality group does not perturb eligible groups.
    baseline = rows()
    PREP.split_records(baseline, policy())
    assert {row["image_id"]: row["assigned_split"] for row in source if row["prefix_proxy"] != "quality"} == {
        row["image_id"]: row["assigned_split"] for row in baseline}


def test_val_invalid_positive_geometry_strictly_refused(tmp_path):
    val_zip = tmp_path / "val.zip"
    archive(val_zip, "val", 2, annotation="1,2,4,0,1,4,0,0\n")
    with pytest.raises(PREP.PreparationError, match="non-positive box size"):
        PREP.extract_split(val_zip, "val", tmp_path / "stage")


@pytest.mark.parametrize("bad", ["1,2,0,4,1,4,0,0", "1,2,5,4,1,12,0,0", "1,2,5,4,2,4,0,0",
    "nan,2,5,4,1,4,0,0", "1,2,5,4,1.5,4,0,0", "1,2,5,4,1,4,9,0", "1,2",
    "1e308,2,1e308,4,1,4,0,0"])
def test_bad_annotations_rejected(bad):
    with pytest.raises(PREP.PreparationError):
        PREP.convert_annotation(bad.encode(), 20, 10, "train/a")


def test_grouped_split_deterministic_and_calibrations_disjoint():
    one, two = rows(), rows()
    a = PREP.split_records(one, policy())
    b = PREP.split_records(list(reversed(two)), policy())
    assert a == b
    assignment = {r["image_id"]: r["assigned_split"] for r in one}
    assert assignment == {r["image_id"]: r["assigned_split"] for r in two}
    for field in ("prefix_proxy", "image_sha256", "isolation_component"):
        mapping = {}
        for row in one:
            mapping.setdefault(row[field], set()).add(row["assigned_split"])
        assert all(len(values) == 1 for values in mapping.values())
    lists = list(a["calibration_image_ids"].values())
    assert len(set(sum(lists, []))) == 3
    assert all(len(items) == 1 for items in lists)
    assert a["near_duplicates_checked"] is False


def test_cross_val_proxy_and_content_components_quarantine_train():
    source = rows()
    source += [record("train/shared_prefix", "train", "v1", "fresh"),
               record("train/samebytes", "train", "extra", "valsha"),
               record("train/same_component", "train", "extra", "morebytes")]
    result = PREP.split_records(source, policy())
    by_id = {r["image_id"]: r for r in source}
    for key in ("train/shared_prefix", "train/samebytes", "train/same_component"):
        assert by_id[key]["assigned_split"] == "quarantine"
    assert by_id["val/a"]["assigned_split"] == "selection"
    assert {c["kind"] for c in result["cross_train_val_collisions"]} == {"prefix_proxy", "exact_image_sha256"}
    assert result["cross_split_collision_review_required"] is True
    assert result["independence_claim_status"].startswith("NOT_ESTABLISHED")


def test_same_content_deduplicated_and_annotation_conflicts_quarantined():
    source = rows()
    source += [record("train/duplicate", "train", "otherprefix", "sha0"),
               record("train/conflict", "train", "conflictgroup", "sha2", "different_label")]
    result = PREP.split_records(source, policy())
    by_id = {r["image_id"]: r for r in source}
    assert by_id["train/duplicate"]["assigned_split"] == "duplicate_excluded"
    for key in ("train/2", "train/3", "train/conflict"):
        assert by_id[key]["assigned_split"] == "quarantine"
    assert len(result["content_duplicates"]) == 2


def test_full_synthetic_preparation_and_existing_output_refusal(tmp_path):
    train_zip, val_zip = tmp_path / "train.zip", tmp_path / "val.zip"
    archive(train_zip, "train", 30)
    archive(val_zip, "val", 4, start=100)
    fingerprints = [PREP.digest_file(train_zip), PREP.digest_file(val_zip)]
    output = tmp_path / "prepared"
    summary = PREP.prepare(train_zip, val_zip, output, policy())
    assert summary["holdout_opened"] is False
    assert summary["split_annotation_counts"]["train"]["images"] == 30
    assert [PREP.digest_file(train_zip), PREP.digest_file(val_zip)] == fingerprints
    assert "splits/training_val.txt" in (output / "training.yaml").read_text(encoding="utf-8")
    assert "selection" not in (output / "training.yaml").read_text(encoding="utf-8")
    image_rows = [json.loads(line) for line in (output / "image_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in image_rows:
        assert PREP.digest_file(output / row["image_path"]) == row["image_sha256"]
        assert PREP.digest_file(output / row["annotation_path"]) == row["annotation_sha256"]
        assert PREP.digest_file(output / row["label_path"]) == row["label_sha256"]
    for path in (output / "splits").glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert Path(line).is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        assert PREP.digest_file(output / entry["path"]) == entry["sha256"]
    with pytest.raises(FileExistsError):
        PREP.prepare(train_zip, val_zip, output, policy())


def test_pairing_mismatch_refused(tmp_path):
    path = tmp_path / "train.zip"
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("VisDrone2019-DET-train/images/a.png", png())
    with pytest.raises(PREP.PreparationError, match="pairing"):
        PREP.extract_split(path, "train", tmp_path / "stage")


def test_corrupted_crc_refused(tmp_path):
    path = tmp_path / "train.zip"
    archive(path, "train", 2)
    payload = bytearray(path.read_bytes())
    name_len, extra_len = struct.unpack_from("<HH", payload, 26)
    offset = 30 + name_len + extra_len
    payload[offset] ^= 1
    path.write_bytes(payload)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        PREP.extract_split(path, "train", tmp_path / "stage")


def test_holdout_named_input_rejected_without_opening(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No holdout path should be opened")
    monkeypatch.setattr(PREP, "digest_file", forbidden)
    with pytest.raises(PREP.PreparationError, match="holdout/test"):
        PREP.prepare(tmp_path / "test-dev.zip", tmp_path / "val.zip", tmp_path / "out")
