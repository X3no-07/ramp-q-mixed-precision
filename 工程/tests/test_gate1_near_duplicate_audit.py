"""Synthetic-only semantics for hashes, exact radius search and access boundary."""
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "audit_gate1_near_duplicates.py"
SPEC = importlib.util.spec_from_file_location("gate1_near_duplicate_audit_test", SOURCE)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_independent_dhash_direction_and_phash_determinism():
    array = np.tile(np.arange(90, dtype=np.uint8) * 2, (80, 1))
    image = Image.fromarray(array)
    descriptor, gray, rgb = AUDIT.describe_image(image)
    assert descriptor["dhash64"] == "ffffffffffffffff"
    assert len(descriptor["phash64"]) == 16
    assert descriptor == AUDIT.describe_image(image)[0]
    assert gray.shape == (64, 64) and rgb.shape == (16, 16, 3)
    assert gray.dtype == np.uint8 and rgb.dtype == np.uint8


def test_phash_dct_normalization_against_independent_direct_coefficient():
    matrix = AUDIT.DCT32
    assert np.allclose(matrix @ matrix.T, np.eye(32), atol=1e-12)
    pixels = np.arange(1024, dtype=np.float64).reshape(32, 32) % 177
    transformed = matrix @ pixels @ matrix.T
    for u, v in ((0, 0), (1, 2), (7, 7)):
        direct = 0.0
        scale_u = 1 / np.sqrt(32) if u == 0 else np.sqrt(2 / 32)
        scale_v = 1 / np.sqrt(32) if v == 0 else np.sqrt(2 / 32)
        for y in range(32):
            for x in range(32):
                direct += pixels[y, x] * np.cos(np.pi * (y + 0.5) * u / 32) * np.cos(np.pi * (x + 0.5) * v / 32) * scale_u * scale_v
        assert transformed[u, v] == pytest.approx(direct, abs=1e-9)


def test_bktree_union_matches_exhaustive_hamming_pairs():
    rng = random.Random(20261022)
    hashes = [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(80)]
    hashes += [(hashes[0][0] ^ 3, hashes[0][1] ^ 31), hashes[0],
               (hashes[1][0] ^ 255, hashes[1][1] ^ 7)]
    descriptors = [{"dhash64": f"{d:016x}", "phash64": f"{p:016x}"} for d, p in hashes]
    expected = {(i, j) for i in range(len(hashes)) for j in range(i + 1, len(hashes))
                if (hashes[i][0] ^ hashes[j][0]).bit_count() <= 4 or (hashes[i][1] ^ hashes[j][1]).bit_count() <= 4}
    assert set(AUDIT.candidate_pairs(descriptors)) == expected


def test_similarity_metrics_handle_exact_and_flat_arrays():
    gray = np.arange(4096, dtype=np.uint16).reshape(64, 64).astype(np.uint8)
    rgb = np.arange(768, dtype=np.uint16).reshape(16, 16, 3).astype(np.uint8)
    result = AUDIT.pair_metrics(gray, gray, rgb, rgb)
    assert result["gray_64_centered_correlation"] == pytest.approx(1)
    assert result["gray_64_rmse_0_1"] == 0 and result["rgb_16_rmse_0_1"] == 0
    flat = np.zeros((64, 64), dtype=np.uint8)
    assert AUDIT.pair_metrics(flat, flat, rgb, rgb)["gray_64_centered_correlation"] is None


@pytest.mark.parametrize("split,path", [("test-dev", "images/test-dev/secret.jpg"), ("train", "../secret.jpg"),
    ("val", "images/val/../secret.jpg"), ("train", "images/train/secret\\file.jpg")])
def test_manifest_cannot_access_holdout_or_escaped_paths(tmp_path, split, path):
    row = {"image_id": "a", "source_split": split, "image_path": path, "filename": Path(path).name}
    with pytest.raises(ValueError):
        AUDIT.validate_rows(tmp_path, [row])


def test_small_end_to_end_audit_preserves_sources_and_refuses_overwrite(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    pixels = np.random.default_rng(31).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    rows = []
    for index, split in enumerate(("train", "val", "train")):
        relative = f"images/{split}/prefix{index}_00001.png"
        image_path = root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(image_path)
        rows.append({"image_id": f"{split}/prefix{index}_00001", "image_path": relative,
            "image_sha256": AUDIT.sha_file(image_path), "filename": image_path.name, "source_split": split,
            "assigned_split": "selection" if split == "val" else "train", "prefix_proxy": f"prefix{index}",
            "isolation_component": f"component{index}", "width": 96, "height": 64})
    inventory = root / "image_manifest.jsonl"
    inventory.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    AUDIT.write_json(root / "manifest.json", {"files": [{"path": "image_manifest.jsonl", "bytes": inventory.stat().st_size,
                                                        "sha256": AUDIT.sha_file(inventory)}]})
    fingerprints = {p.relative_to(root).as_posix(): AUDIT.sha_file(p) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "audit"
    summary = AUDIT.run_audit(root, output, expected_images=3)
    assert summary["image_count"] == 3
    assert summary["exact_byte_duplicate_groups"] == 1
    assert summary["saved_cross_role_prefix_or_component_candidate_pairs"] == 3
    assert summary["cross_official_train_val_candidate_pairs"] == 2
    assert summary["contact_sheet_pairs"] == 3
    assert summary["holdout_opened_or_listed"] is False and summary["split_edits"] is False
    assert {p.relative_to(root).as_posix(): AUDIT.sha_file(p) for p in root.rglob("*") if p.is_file()} == fingerprints
    with pytest.raises(FileExistsError):
        AUDIT.run_audit(root, output, expected_images=3)
