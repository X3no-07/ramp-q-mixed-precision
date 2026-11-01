"""Bound post-training inference and official-source-faithful development AP.

No Selection/Holdout input is admitted. Raw dtype checks precede NMS and cover
all anchors. Saved predictions, arrays and geometry allow independent replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

BRANCHES = ["fp32", "fp16", "body32_head16_decode32"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def info(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_protocol(p):
    require(p["schema"] == "rampq.gate1.formal_candidate_evaluation_protocol/v1" and p["branches"] == BRANCHES, "Unsupported protocol/branch matrix")
    require(p["candidate_selection"] == "fixed_final_epoch_30_FP32_EMA" and p["data_roles"] == ["debug", "training_val"], "Forbidden data role/selection")
    require(p["raw_tolerances"] == {"box_atol": .5, "box_rtol": .005, "score_atol": .002, "score_rtol": .02}, "Frozen tolerances changed")
    for role, count in (("debug", 109), ("training_val", 414)):
        rows = p["cohorts"][role]
        require(len(rows) == count and len({r["image_id"] for r in rows}) == count, "Cohort count/identity mismatch")
        require([r["image_id"] for r in rows] == sorted(r["image_id"] for r in rows), "Explicit lexicographic image order required")
        for row in rows:
            require(row["assigned_split"] == role and row["source_split"] == "train" and row["image_id"].startswith("train/"), "Forbidden cohort member")
            for name, prefix in (("image", "images/train/"), ("annotation", "annotations/train/")):
                relative = row[name + "_path"]
                require(isinstance(relative, str) and relative.startswith(prefix) and ".." not in Path(relative).parts and
                        not Path(relative).is_absolute(), "Forbidden cohort member file")
    require({r["image_id"] for r in p["cohorts"]["debug"]}.isdisjoint(r["image_id"] for r in p["cohorts"]["training_val"]), "Evaluation cohort overlap")


def configure():
    import cv2
    import torch
    import torchvision
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    require(torch.cuda.is_available(), "CUDA required")
    return torch, cv2


def parse_annotation(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.strip().rstrip(",").split(",")
        require(len(fields) == 8, "Annotation must contain eight fields")
        rows.append([float(value) for value in fields])
    array = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
    require(bool(np.isfinite(array).all()), "Nonfinite GT")
    return array


def guards_summary(rows, branch):
    selected = [r[branch] for r in rows]
    return {"images": len(selected), "pass_images": sum(g["allclose_pass"] for g in selected),
            "fail_images": sum(not g["allclose_pass"] for g in selected),
            "boxes": {"elements": sum(g["boxes"]["elements"] for g in selected), "violations": sum(g["boxes"]["violations"] for g in selected),
                "max_absolute_error": max(g["boxes"]["max_absolute_error"] for g in selected),
                "max_error_over_tolerance": max(g["boxes"]["max_error_over_tolerance"] for g in selected)},
            "scores": {"elements": sum(g["scores"]["elements"] for g in selected), "violations": sum(g["scores"]["violations"] for g in selected),
                "max_absolute_error": max(g["scores"]["max_absolute_error"] for g in selected),
                "max_error_over_tolerance": max(g["scores"]["max_error_over_tolerance"] for g in selected)},
            "old_guard_replaced": False}


def infer(protocol_path):
    p = read(protocol_path)
    validate_protocol(p)
    out = Path(p["inference_output"])
    require(not out.exists(), "Inference result directory must be new")
    out.mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(out / "settings"), YOLO_OFFLINE="true", PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    torch, cv2 = configure()
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.nms import non_max_suppression
    import gate1_fp32_live_capture as C
    import probe_gate1_body32_head16 as P
    import probe_gate1_decode_precision as D
    import qualify_gate1_float_baseline as Q
    for binding in p["source_bindings"].values():
        require(info(binding["path"]) == binding, "Bound source changed before inference")
    require(C.sources() == p["runtime_sources"], "Framework source changed")
    epoch_dir = Path(p["final_capture_directory"])
    state, metadata = C.FORMAT.load_snapshot(epoch_dir / "state")
    require(info(epoch_dir / "state/reference_state.pt") == p["final_state"], "Final FP32 state changed")
    arch = read(epoch_dir / "architecture.json")
    models = {}
    for branch in BRANCHES:
        with torch.random.fork_rng(devices=[]):
            model = DetectionModel(deepcopy(arch["cfg"]), ch=3, nc=10, verbose=False)
        model.load_state_dict(state, strict=True)
        model.names = dict(enumerate(arch["names"]))
        model = model.to("cuda:0").float().eval()
        require(int(model.model[-1].reg_max) == 16 and len(model.model) == 24, "Measured Detect architecture differs")
        if branch == "fp16":
            model.half()
        elif branch == "body32_head16_decode32":
            model.model[-1].cv2.half()
            model.model[-1].cv3.half()
            model = P.make_prototype(torch, model, D.decode_functional, make_anchors)
        models[branch] = model
    letterbox = LetterBox((640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32,
                         padding_value=114, interpolation=cv2.INTER_LINEAR)
    root = Path(p["prepared_root"]).resolve(strict=True)
    all_summaries = {}
    for role in p["data_roles"]:
        folder = out / role
        folder.mkdir()
        (folder / "raw").mkdir()
        (folder / "predictions").mkdir()
        guards, submitted, saturations = [], [], Counter()
        for index, row in enumerate(p["cohorts"][role], 1):
            files = {}
            for key in ("image", "annotation"):
                path = (root / row[key + "_path"]).resolve(strict=True)
                require(path.is_relative_to(root) and sha(path) == row[key + "_sha256"], "Cohort file identity mismatch")
                files[key] = path
            image = cv2.imdecode(np.frombuffer(files["image"].read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(image is not None and image.shape == (row["height"], row["width"], 3), "Image dimensions/channels differ")
            params = letterbox.get_params({"img": image})
            padded = letterbox.apply_image({"img": image}, params)["img"]
            tensor = torch.from_numpy(np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))).unsqueeze(0).float().div(255).to("cuda:0")
            arrays = {}
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                for branch, model in models.items():
                    output = model(tensor.half() if branch == "fp16" else tensor)
                    decoded = output["decoded"] if branch == "body32_head16_decode32" else output[0]
                    require(decoded.shape == (1, 14, 8400) and decoded.dtype == (torch.float16 if branch == "fp16" else torch.float32), "Decoded output shape/dtype differs")
                    arrays[branch] = decoded.detach().cpu().numpy().copy()
            stem = row["image_id"].split("/")[1]
            if role == "debug":
                np.savez_compressed(folder / "raw" / f"{stem}.npz", **arrays)
            checks = {b: Q.raw_guard(arrays["fp32"], arrays[b]) for b in BRANCHES[1:]}
            require(all(g["shape_pass"] and g["finite_pass"] and all(v["positive_width_height"] and v["class_scores_within_0_1"]
                for v in g["raw_geometry"].values()) for g in checks.values()), "Invalid raw finite/geometry domain")
            guards.append({"image_id": row["image_id"], **checks})
            item = {"image_id": row["image_id"], "role": role, "letterbox_params": params,
                    "input_sha256": hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest(),
                    "raw_dtypes": {b: str(a.dtype) for b, a in arrays.items()},
                    "raw_hashes": {b: hashlib.sha256(a.tobytes()).hexdigest() for b, a in arrays.items()},
                    "prediction_rows8": {}, "anchor_indices": {}}
            for branch in BRANCHES:
                # guarded_nms clones raw input; qualification always precedes NMS.
                detection, anchor = Q.guarded_nms(non_max_suppression, torch.from_numpy(arrays[branch].astype(np.float32)))
                restored, _ = Q.restored_predictions(detection[0].numpy(), anchor[0].numpy(), params, row["image_id"], branch)
                # Stable confidence order is the explicitly frozen submission order.
                order = sorted(range(len(restored)), key=lambda i: (-restored[i]["score"], i))
                item["prediction_rows8"][branch] = [[*restored[i]["bbox_xywh"], restored[i]["score"], restored[i]["category"], 0, 0] for i in order]
                item["anchor_indices"][branch] = [restored[i]["raw_anchor_index"] for i in order]
                saturations[branch] += len(restored) == 1000
            write(folder / "predictions" / f"{stem}.json", item)
            submitted.append({"image_id": row["image_id"], "file": f"predictions/{stem}.json"})
            write(out / "live_status.json", {"status": "INFERENCE", "role": role, "images": index, "total": len(p["cohorts"][role]), "AP_computed": False})
            if index % 25 == 0 or index == len(p["cohorts"][role]):
                print(json.dumps({"role": role, "inferred": index, "total": len(p["cohorts"][role])}), flush=True)
        all_summaries[role] = {b: guards_summary(guards, b) for b in BRANCHES[1:]}
        write(folder / "raw_guards.json", {"per_image": guards, "summary": all_summaries[role]})
        write(folder / "submission_order.json", submitted)
        write(folder / "NMS_saturations.json", dict(saturations))
    for binding in p["source_bindings"].values():
        require(info(binding["path"]) == binding, "Bound source changed during inference")
    result = {"status": "COMPLETED_FRESH_FIXED_COHORT_INFERENCE_ONLY", "protocol": info(protocol_path),
        "guard_summary": all_summaries, "AP_computed": False, "final_state": p["final_state"],
        "source_bindings_unchanged": True, "data_roles": p["data_roles"], "scientific_acceptance": False,
        "training_val_raw_arrays_saved": False, "debug_raw_arrays_saved": True,
        "selection_or_holdout_input_accessed": False, "original_failures_preserved": True}
    write(out / "inference_receipt.json", result)
    write(out / "manifest.json", {"files": {path.relative_to(out).as_posix(): {k: v for k, v in info(path).items() if k != "path"}
          for path in sorted(out.rglob("*")) if path.is_file() and "settings" not in path.parts}})
    print(json.dumps(result), flush=True)


def score(protocol_path, branch):
    p = read(protocol_path)
    validate_protocol(p)
    require(branch in BRANCHES, "Unknown scoring branch")
    import gate1_official_ap as A
    for name in ("AP_mirror", "prefilter"):
        binding = p["source_bindings"][name]
        require(info(binding["path"]) == binding, "Scoring implementation changed")
    inferred = Path(p["inference_output"])
    require((inferred / "inference_receipt.json").is_file(), "Inference incomplete")
    output = Path(p["scoring_output"]) / branch
    require(not output.exists(), "Scoring branch output must be new")
    output.mkdir(parents=True)
    role = "training_val"
    prepared_gt, prepared_dt, input_index = [], [], []
    inference_manifest = read(inferred / "manifest.json")["files"]
    for row in p["cohorts"][role]:
        annotation = Path(p["prepared_root"]) / row["annotation_path"]
        require(sha(annotation) == row["annotation_sha256"], "Scoring GT identity mismatch")
        stem = row["image_id"].split("/")[1]
        prediction_file = inferred / role / "predictions" / f"{stem}.json"
        expected = inference_manifest[prediction_file.relative_to(inferred).as_posix()]
        require(all(info(prediction_file)[k] == expected[k] for k in ("bytes", "sha256")), "Scoring predictions changed")
        predictions = np.asarray(read(prediction_file)["prediction_rows8"][branch], dtype=np.float64).reshape(-1, 8)
        prefiltered = A.prepare_raw_image(parse_annotation(annotation.read_text(encoding="utf-8-sig")), predictions,
                                         width=row["width"], height=row["height"])
        prepared_gt.append(prefiltered["prepared_gt"])
        prepared_dt.append(prefiltered["filtered_predictions"])
        input_index.append({"image_id": row["image_id"], "annotation_sha256": row["annotation_sha256"],
                            "predictions": info(prediction_file), "GT_rows_after_filter": len(prepared_gt[-1]),
                            "DT_rows_after_filter": len(prepared_dt[-1])})
    result = A.calc_accuracy(prepared_gt, prepared_dt, capture_intermediates=False)
    np.savez_compressed(output / "AP_AR_arrays.npz", AP=result["AP"], AR=result["AR"], evalClass=result["evalClass"])
    summary = {"status": "COMPLETED_OFFICIAL_SOURCE_FAITHFUL_PYTHON_DEVELOPMENT_AP_ONLY", "branch": branch,
        "role": "training_val", "images": 414, "metrics_percent": result["metrics"], "official_commit": A.OFFICIAL_COMMIT,
        "submission_order": p["postprocessing"]["submission_order"], "official_native_real_dataset_comparison_executed": False,
        "prior_synthetic_native_agreement": p["AP_mirror_native_validation"], "scientific_acceptance": False,
        "independent_generalization_claim": False, "selection_or_holdout_accessed": False}
    write(output / "input_index.json", input_index)
    write(output / "metrics.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--action", choices=["infer", "score"], required=True)
    parser.add_argument("--branch", choices=BRANCHES)
    args = parser.parse_args()
    if args.action == "infer":
        infer(args.protocol)
    else:
        score(args.protocol, args.branch)
