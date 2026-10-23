"""Read-only CPU inspection of frozen raw arrays; no model or image loading."""
from pathlib import Path
import hashlib
import json
import numpy as np


BASE = Path(__file__).resolve().parents[2]
QUAL = BASE / "20261022_posttraining_checks_v1" / "02_float_qualification"
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((QUAL / "manifest.json").read_text(encoding="utf-8"))["files"]
    accessed = {}
    def read_json(name):
        path = QUAL / name
        before = digest(path)
        expected = manifest.get(name)
        if expected:
            assert before == expected["sha256"] and path.stat().st_size == expected["bytes"]
        accessed[name] = before
        return json.loads(path.read_text(encoding="utf-8"))
    protocol = read_json("qualification_protocol.json")
    environment = read_json("environment.json")
    provenance = read_json("model_provenance.json")
    receipt = read_json("qualification_receipt.json")
    read_json("raw_dtype_guard.json")
    refs = []
    for name, data in environment["framework_sources"].items():
        if name not in ("nn/tasks.py", "nn/modules/head.py", "nn/modules/block.py", "nn/modules/conv.py", "utils/tal.py"):
            continue
        path = Path(data["path"])
        assert digest(path) == data["sha256"]
        refs.append({"relative": name, "path": str(path), "sha256": data["sha256"]})
    a_tol = np.asarray([.5] * 4 + [.002] * 10).reshape(1, 14, 1)
    r_tol = np.asarray([.005] * 4 + [.02] * 10).reshape(1, 14, 1)
    assert (protocol["raw"]["box_atol"], protocol["raw"]["box_rtol"], protocol["raw"]["score_atol"], protocol["raw"]["score_rtol"]) == (.5, .005, .002, .02)
    tops_ratio, tops_absolute, cast_counts = [], [], []
    for path in sorted((QUAL / "raw").glob("*.npz")):
        relative = path.relative_to(QUAL).as_posix()
        assert digest(path) == manifest[relative]["sha256"] and path.stat().st_size == manifest[relative]["bytes"]
        accessed[relative] = manifest[relative]["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert archive.files == ["fp32", "fp16"]
            assert archive["fp32"].dtype == np.float32 and archive["fp16"].dtype == np.float16
            a = archive["fp32"].astype(np.float64)
            b = archive["fp16"].astype(np.float64)
        assert a.shape == b.shape == (1, 14, 8400)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        item = read_json("per_image/" + path.stem + ".json")
        assert item["image_id"] == "train/" + path.stem
        tol = a_tol + r_tol * np.abs(a)
        err = np.abs(b - a)
        ratio = err / tol
        final_cast = a.astype(np.float16).astype(np.float64)
        cast_err = np.abs(final_cast - a)
        cast_counts.append({"image_id": item["image_id"],
                            "box_violations": int((cast_err[:, :4] > tol[:, :4]).sum()),
                            "score_violations": int((cast_err[:, 4:] > tol[:, 4:]).sum()),
                            "max_box_absolute_error": float(cast_err[:, :4].max()),
                            "max_score_absolute_error": float(cast_err[:, 4:].max()),
                            "max_error_over_tolerance": float((cast_err / tol).max())})
        for criterion, storage in ((ratio, tops_ratio), (err, tops_absolute)):
            for flat in np.argpartition(criterion.ravel(), -3)[-3:]:
                _, channel, anchor = np.unravel_index(flat, a.shape)
                level = 0 if anchor < 6400 else 1 if anchor < 8000 else 2
                offset = int(anchor - [0, 6400, 8000][level])
                side = [80, 40, 20][level]
                storage.append({"image_id": item["image_id"], "channel": int(channel),
                                "channel_name": ["x", "y", "w", "h"][channel] if channel < 4 else "class_" + str(channel - 4),
                                "anchor": int(anchor), "head_stride": [8, 16, 32][level],
                                "grid_row": offset // side, "grid_col": offset % side,
                                "fp32": float(a[0, channel, anchor]), "fp16": float(b[0, channel, anchor]),
                                "absolute_error": float(err[0, channel, anchor]), "fixed_tolerance": float(tol[0, channel, anchor]),
                                "error_over_tolerance": float(ratio[0, channel, anchor]),
                                "fp32_box_xywh": a[0, :4, anchor].tolist(), "fp16_box_xywh": b[0, :4, anchor].tolist(),
                                "fp32_max_class_score": float(a[0, 4:, anchor].max()),
                                "fp16_max_class_score": float(b[0, 4:, anchor].max()),
                                "fp32_final_cast_only": float(final_cast[0, channel, anchor]),
                                "fp32_final_cast_only_absolute_error": float(cast_err[0, channel, anchor]),
                                "normalized_input_sha256": item["normalized_fp32_input_sha256"],
                                "raw_npz_sha256": accessed[relative]})
    assert len(cast_counts) == receipt["images_completed"] == 109
    assert all(digest(QUAL / name) == value for name, value in accessed.items())
    result = {"schema": "rampq.gate1.raw_dtype_independent_review/v1",
              "status": "READ_ONLY_DIAGNOSTIC_NOT_GUARD_REPLACEMENT", "images": len(cast_counts),
              "qualification_directory": str(QUAL), "source_guard_status": receipt["same_backend_dtype_guard"],
              "framework_sources_verified": refs, "model_provenance": provenance,
              "top_three_by_error_over_fixed_tolerance": sorted(tops_ratio, key=lambda x: -x["error_over_tolerance"])[:3],
              "top_three_by_absolute_error": sorted(tops_absolute, key=lambda x: -x["absolute_error"])[:3],
              "counterfactual_final_storage_cast_only": {
                  "description": "Frozen FP32 decoded raw -> NumPy FP16 -> float64, without running any model/operator. This is not a replacement FP16 forward pass or decode-only intervention.",
                  "fixed_tolerances_identical_to_original_guard": True,
                  "images_with_zero_violations": sum(x["box_violations"] == x["score_violations"] == 0 for x in cast_counts),
                  "total_box_violations": sum(x["box_violations"] for x in cast_counts),
                  "total_score_violations": sum(x["score_violations"] for x in cast_counts),
                  "max_box_absolute_error": max(x["max_box_absolute_error"] for x in cast_counts),
                  "max_score_absolute_error": max(x["max_score_absolute_error"] for x in cast_counts),
                  "max_error_over_tolerance": max(x["max_error_over_tolerance"] for x in cast_counts),
                  "per_image": cast_counts},
              "accessed_source_files_unchanged": True, "accessed_source_file_count": len(accessed),
              "model_loaded": False, "images_or_annotations_loaded": False, "gpu_intervention": False,
              "selection_detector_inference": False, "holdout_accessed": False,
              "scientific_gate1_pass": False, "old_gate05_guard": "FAILED_UNCHANGED",
              "source_files": accessed, "script_sha256": digest(Path(__file__))}
    destination = OUT / "raw_array_diagnostics.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key in ("images", "status", "accessed_source_file_count", "accessed_source_files_unchanged")}, ensure_ascii=False))
    print(json.dumps({key: value for key, value in result["counterfactual_final_storage_cast_only"].items() if key != "per_image"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
