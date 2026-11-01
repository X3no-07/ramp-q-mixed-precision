"""Explicit-member, hash-bound restricted Dataset for a NEW training contract.

Only loader resize, centered LetterBox and shared flips are implemented. No
directory scan, dataset cache, old model, selection image or holdout is read.
The source implements a loader, not evidence that real-data training succeeded.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import unicodedata

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

GEOMETRY_SHA256 = "4ecb3eedc4c5597f64e2290d73de4d6a4d40f4cd265e06c0dbb0ca03371dfd56"
POLICY = {
    "schema": "rampq.gate1.ignore_dataset_policy/v1",
    "membership": "exact_list_in_caller_hash_bound_single_role_manifest_no_prefix_inference",
    "allowed_roles": ["train", "training_val"],
    "empty_sidecar": "explicit_zero_rows_allowed",
    "bad_or_empty_source_region": "reject_whole_sample_never_silent_drop",
    "positive_labels": "independent_normalized_yolo_xywh_not_publisher_ignore_rows",
    "image_decode": "verified_bytes_cv2_imdecode_color_no_unbound_cache",
    "resize": "longest_side_to_image_size_ceil_both_axes_cv2_linear",
    "letterbox": "centered_fixed_square_auto_false_scale_fill_false_padding_114",
    "flips": "same_explicit_per_image_epoch_seed_params_for_image_positive_ignore",
    "color": "BGR_decode_to_RGB_CHW_uint8",
    "hash_recheck": "manifest_and_three_member_files_each_getitem",
    "selection_or_holdout": "roles_and_paths_rejected_before_member_file_read",
    "complex_transforms": False,
    "official_training_rule": False,
    "real_data_training_verified": False,
    "current_execution_data_role": "synthetic_only",
}
MANIFEST_SCHEMA = "rampq.gate1.ignore_dataset_manifest/v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"train/[A-Za-z0-9_.-]+")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON number: {value}")))


def _safe_path_text(text):
    _require(isinstance(text, str) and text and "\\" not in text, "member paths use explicit nonempty POSIX relative paths")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    components = re.split(r"[/\\]", normalized)
    _require(not any(re.search(r"selection|holdout|test[-_]?dev", p) or p in {"test", "testset"} for p in components),
             "Selection/Holdout/test paths are forbidden")
    p = PurePosixPath(text)
    _require(not p.is_absolute() and all(v not in {"", ".", ".."} for v in text.split("/")) and ":" not in text,
             "absolute or traversing member path is forbidden")


def _reject_forbidden_absolute(path):
    normalized = unicodedata.normalize("NFKC", str(path)).casefold()
    components = re.split(r"[/\\]", normalized)
    _require(not any(re.search(r"selection|holdout|test[-_]?dev", p) or p in {"test", "testset"} for p in components),
             "Selection/Holdout/test paths are forbidden")


def _read_member(root, info):
    _require(isinstance(info, dict) and set(info) == {"path", "bytes", "sha256"}, "member requires exact path,bytes,sha256 fields")
    _safe_path_text(info["path"])
    _require(type(info["bytes"]) is int and info["bytes"] >= 0 and isinstance(info["sha256"], str)
             and _DIGEST.fullmatch(info["sha256"]), "invalid file identity")
    path = root.joinpath(*PurePosixPath(info["path"]).parts)
    _reject_forbidden_absolute(path)
    resolved = path.resolve(strict=True)
    _require(resolved.is_relative_to(root), "member escaped the explicit data root")
    _require(not path.is_symlink() and all(not p.is_symlink() for p in path.parents if p.is_relative_to(root)),
             "symlink members are forbidden")
    raw = resolved.read_bytes()
    _require(len(raw) == info["bytes"] and _sha(raw) == info["sha256"], f"member byte identity mismatch: {info['path']}")
    return path, raw


def _positive_labels(raw, class_count):
    rows = []
    for line in raw.decode("utf-8-sig").splitlines():
        _require(bool(line.strip()), "blank positive-label rows are forbidden")
        fields = line.split()
        _require(len(fields) == 5, "positive labels require class and four normalized xywh values")
        values = np.array([float(v) for v in fields], dtype=np.float64)
        _require(bool(np.isfinite(values).all()), "positive labels must be finite")
        cls, x, y, w, h = values
        _require(cls == math.floor(cls) and 0 <= cls < class_count, "positive class outside contract")
        _require(0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0 and
                 x - w / 2 >= -1e-7 and y - h / 2 >= -1e-7 and x + w / 2 <= 1 + 1e-7 and y + h / 2 <= 1 + 1e-7,
                 "positive geometry must be nonempty and normalized inside original canvas")
        rows.append(values)
    out = np.array(rows, dtype=np.float64).reshape(-1, 5)
    _require(len({tuple(v) for v in out.tolist()}) == len(out), "duplicate positive labels require explicit correction")
    return out


def _load_geometry():
    path = Path(__file__).with_name("gate1_ignore_geometry.py")
    _require(_sha(path.read_bytes()) == GEOMETRY_SHA256, "frozen geometry source changed")
    spec = importlib.util.spec_from_file_location("gate1_ignore_geometry_dataset_dependency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RestrictedIgnoreDataset(Dataset):
    """DetectionTrainer-compatible Dataset with an explicit single-role manifest.

    Manifest samples are the membership authority. Every entry has its own role,
    image ID, actual image dimensions and three file identities. The image ID's
    train/ namespace alone grants no membership. No invalid entry is dropped.
    """
    def __init__(self, manifest_path, manifest_sha256, *, split_role="train", image_size=640,
                 class_count=10, augment=False, fliplr=0., flipud=0., seed=0, epoch=0):
        _require(split_role in {"train", "training_val"}, "only train/training_val roles are supported")
        _require(type(image_size) is int and image_size > 0, "image_size must be positive integer")
        _require(type(class_count) is int and class_count > 0, "class_count must be positive integer")
        _require(type(augment) is bool and all(type(p) in {float, int} and 0 <= p <= 1 for p in (fliplr, flipud)), "invalid flip options")
        _require((augment and split_role == "train") or (fliplr == 0 and flipud == 0), "validation/nonaugment flips must be zero")
        _require(type(seed) is int and seed >= 0 and type(epoch) is int and epoch >= 0, "seed and epoch require nonnegative integers")
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        _reject_forbidden_absolute(self.manifest_path)
        _require(isinstance(manifest_sha256, str) and _DIGEST.fullmatch(manifest_sha256), "manifest requires lowercase SHA256")
        self.manifest_sha256 = manifest_sha256
        raw = self.manifest_path.read_bytes()
        _require(_sha(raw) == manifest_sha256, "manifest SHA256 mismatch")
        manifest = _json(raw)
        _require(set(manifest) == {"schema", "data_role", "split_role", "class_count", "data_root", "policy", "samples"}, "manifest requires exact contract fields")
        _require(manifest["schema"] == MANIFEST_SCHEMA and manifest["policy"] == POLICY, "manifest schema/policy mismatch")
        _require(manifest["data_role"] == "synthetic_only", "current dataset integration permits synthetic fixtures only")
        _require(manifest["split_role"] == split_role and type(manifest["class_count"]) is int and manifest["class_count"] == class_count, "manifest role/class contract mismatch")
        _safe_path_text(manifest["data_root"])
        self.root = (self.manifest_path.parent / manifest["data_root"]).resolve(strict=True)
        _require(self.root.is_relative_to(self.manifest_path.parent), "data root escaped manifest directory")
        _reject_forbidden_absolute(self.root)
        _require(isinstance(manifest["samples"], list) and manifest["samples"], "explicit nonempty sample membership required")
        self.entries = copy.deepcopy(manifest["samples"])
        ids, paths = set(), set()
        # Validate the complete membership/path contract before any member read.
        for row in self.entries:
            _require(isinstance(row, dict) and set(row) == {"image_id", "split_role", "original_hw", "image", "labels", "sidecar"}, "sample requires exact fields")
            _require(isinstance(row["image_id"], str) and _IMAGE_ID.fullmatch(row["image_id"]) and row["image_id"] not in ids, "invalid or duplicate explicit image identity")
            _safe_path_text(row["image_id"])
            ids.add(row["image_id"])
            _require(row["split_role"] == split_role, "sample is not a member of the requested split role")
            _require(isinstance(row["original_hw"], list) and len(row["original_hw"]) == 2 and all(type(v) is int and v > 0 for v in row["original_hw"]), "invalid declared image dimensions")
            for key in ("image", "labels", "sidecar"):
                _require(isinstance(row[key], dict) and set(row[key]) == {"path", "bytes", "sha256"}, "invalid member identity fields")
                _safe_path_text(row[key]["path"])
                _require(row[key]["path"] not in paths, "each sample member path must be unique")
                paths.add(row[key]["path"])
        self.split_role, self.imgsz, self.class_count = split_role, image_size, class_count
        self.augment, self.fliplr, self.flipud, self.seed, self.epoch = augment, float(fliplr), float(flipud), seed, epoch
        self.geometry = _load_geometry()
        self.rect = False
        self.mosaic = False
        self.use_segments = self.use_keypoints = self.use_obb = False
        self.data = {"nc": class_count, "channels": 3, "names": {i: str(i) for i in range(class_count)}}
        self.labels, self.im_files, self.label_files = [], [], []
        self._entry_json = tuple(json.dumps(row, sort_keys=True, allow_nan=False) for row in self.entries)
        for entry in self.entries:
            _, labels, _, paths = self._load_sample(entry)
            self.im_files.append(str(paths["image"]))
            self.label_files.append(str(paths["labels"]))
            self.labels.append({"im_file": str(paths["image"]), "shape": tuple(entry["original_hw"]),
                                "cls": labels[:, :1].astype(np.float32), "bboxes": labels[:, 1:].astype(np.float32),
                                "segments": [], "keypoints": None, "normalized": True, "bbox_format": "xywh"})

    def __len__(self):
        return len(self.entries)

    def set_epoch(self, epoch):
        _require(type(epoch) is int and epoch >= 0, "epoch requires nonnegative integer")
        self.epoch = epoch

    def close_mosaic(self, hyp=None):
        """The restricted chain never contains mosaic; no transform rebuild occurs."""
        _require(self.mosaic is False, "mosaic cannot be enabled")

    def _load_sample(self, entry):
        _require(_sha(self.manifest_path.read_bytes()) == self.manifest_sha256, "manifest changed after construction")
        _require(json.dumps(entry, sort_keys=True, allow_nan=False) in self._entry_json, "sample is not in original explicit membership")
        files = {key: _read_member(self.root, entry[key]) for key in ("image", "labels", "sidecar")}
        image = cv2.imdecode(np.frombuffer(files["image"][1], dtype=np.uint8), cv2.IMREAD_COLOR)
        _require(image is not None and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3,
                 "image did not decode as uint8 three-channel color")
        _require(tuple(image.shape[:2]) == tuple(entry["original_hw"]), "actual image dimensions disagree with manifest")
        labels = _positive_labels(files["labels"][1], self.class_count)
        rows = []
        for line in files["sidecar"][1].decode("utf-8-sig").splitlines():
            _require(bool(line.strip()), "blank sidecar source rows are forbidden")
            rows.append(_json(line.encode("utf-8")))
        identity = {"image_id": entry["image_id"], "image_sha256": entry["image"]["sha256"], "sidecar_sha256": entry["sidecar"]["sha256"]}
        ignore = self.geometry.from_prepared_rows(identity, entry["original_hw"], rows, dtype=np.float64)
        return image, labels, ignore, {key: pair[0] for key, pair in files.items()}

    def __getitem__(self, index):
        _require(type(index) is int and 0 <= index < len(self), "index is outside explicit sample membership")
        entry = self.entries[index]
        image, labels, ignore, paths = self._load_sample(entry)
        h0, w0 = image.shape[:2]
        ratio = self.imgsz / max(h0, w0)
        rw, rh = min(math.ceil(w0 * ratio), self.imgsz), min(math.ceil(h0 * ratio), self.imgsz)
        if (rh, rw) != (h0, w0):
            image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        ignore = self.geometry.apply_loader_resize(ignore, (rh, rw))
        # Positive labels and ignores follow identical explicit sampled geometry.
        xywh = labels[:, 1:].copy() * np.array([w0, h0, w0, h0])
        positive = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
        positive *= np.array([rw/w0, rh/h0, rw/w0, rh/h0])
        r = min(self.imgsz/rh, self.imgsz/rw)
        unpad = (round(rw*r), round(rh*r))
        dw, dh = (self.imgsz-unpad[0])/2, (self.imgsz-unpad[1])/2
        params = {"orig_shape": (rh, rw), "new_shape": (self.imgsz, self.imgsz), "ratio": (r, r), "new_unpad": unpad,
                  "left": round(dw-.1), "right": round(dw+.1), "top": round(dh-.1), "bottom": round(dh+.1)}
        if image.shape[1::-1] != unpad:
            image = cv2.resize(image, unpad, interpolation=cv2.INTER_LINEAR)
        image = cv2.copyMakeBorder(image, params["top"], params["bottom"], params["left"], params["right"], cv2.BORDER_CONSTANT, value=(114,114,114))
        positive *= r
        positive += np.array([params["left"], params["top"], params["left"], params["top"]])
        ignore = self.geometry.apply_letterbox_params(ignore, params)
        rng_digest = _sha(f"{self.seed}:{self.epoch}:{entry['image_id']}".encode("utf-8"))
        rng = np.random.Generator(np.random.PCG64(int(rng_digest[:32], 16)))
        flips = []
        for direction, probability in (("vertical", self.flipud), ("horizontal", self.fliplr)):
            flip = bool(self.augment and rng.random() < probability)
            fp = {"flip": flip, "direction": direction, "h": self.imgsz, "w": self.imgsz, "flip_idx": None}
            if flip:
                image = np.flip(image, axis=0 if direction == "vertical" else 1)
                a,b = (1,3) if direction == "vertical" else (0,2)
                positive[:, [a,b]] = self.imgsz - positive[:, [b,a]]
            ignore = self.geometry.apply_flip_params(ignore, fp)
            flips.append(fp)
        positive = np.clip(positive, 0, self.imgsz)
        _require(bool(np.isfinite(positive).all()) and bool((positive[:, 2:] > positive[:, :2]).all()), "transformed positive labels are invalid")
        final_xywh = np.concatenate(((positive[:, :2] + positive[:, 2:])/2, positive[:, 2:] - positive[:, :2]), axis=1) / self.imgsz
        im = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2,0,1)))
        identity = copy.deepcopy(ignore["identity"])
        return {"img": im, "cls": torch.tensor(labels[:, :1], dtype=torch.float32),
                "bboxes": torch.tensor(final_xywh, dtype=torch.float32), "batch_idx": torch.zeros(len(labels), dtype=torch.float32),
                "im_file": str(paths["image"]), "ori_shape": (h0,w0), "resized_shape": (self.imgsz,self.imgsz),
                "ratio_pad": ((rh/h0*r, rw/w0*r), (params["left"],params["top"])),
                "ignore_sample": ignore, "image_identities": identity,
                "sample_identities": {**identity, "label_sha256": entry["labels"]["sha256"], "split_role": self.split_role, "manifest_sha256": self.manifest_sha256},
                "transform_trace": {"loader_resized_hw": (rh,rw), "letterbox": params, "flips": tuple(flips)},
                "ignore_metadata": {"policy": copy.deepcopy(POLICY), "manifest_sha256": self.manifest_sha256, "split_role": self.split_role}}

    @staticmethod
    def collate_fn(samples):
        _require(isinstance(samples, (list,tuple)) and len(samples)>0, "collate requires nonempty explicit samples")
        geometry = _load_geometry()
        hw = tuple(samples[0]["img"].shape[-2:])
        _require(all(tuple(s["img"].shape) == (3,*hw) and s["img"].dtype == torch.uint8 for s in samples), "all batch images require same uint8 CHW RGB canvas")
        _require(all(s["ignore_metadata"] == samples[0]["ignore_metadata"] for s in samples), "cannot mix role, manifest or policy in one batch")
        identities = [s["image_identities"] for s in samples]
        _require(len({s["image_id"] for s in identities}) == len(identities), "duplicate identities in batch require explicit sampler policy")
        ignore = geometry.collate_ignores([s["ignore_sample"] for s in samples], identities, hw, torch_dtype=torch.float32)
        result = {"img": torch.stack([s["img"] for s in samples]), "cls": torch.cat([s["cls"] for s in samples]),
                  "bboxes": torch.cat([s["bboxes"] for s in samples]),
                  "batch_idx": torch.cat([s["batch_idx"] + i for i,s in enumerate(samples)]),
                  "ignore_boxes_xyxy": ignore["ignore_boxes_xyxy"], "image_identities": ignore["image_identities"],
                  "source_records": ignore["source_records"], "image_hw": hw,
                  "ignore_metadata": copy.deepcopy(samples[0]["ignore_metadata"])}
        for key in ("im_file", "ori_shape", "resized_shape", "ratio_pad", "sample_identities", "transform_trace"):
            result[key] = tuple(s[key] for s in samples)
        return result


def build_synthetic_fixture(root, *, train_count=4, val_count=2, class_count=10):
    """Write NEW synthetic-only cv2 PNG/YOLO/JSONL files, never scan input data.

    The directory must not exist. Return identities for the two explicit role
    manifests. This helper is also the fixture entry point for a caller's NEW
    synthetic Trainer run; it never imports a model, trainer, or GPU runtime.
    """
    root = Path(root)
    _reject_forbidden_absolute(root)
    _require(not root.exists(), "synthetic fixture directory must be new")
    _require(all(type(n) is int and n > 0 for n in (train_count,val_count,class_count)) and class_count >= 2,
             "synthetic fixture requires positive counts and at least two classes")
    root.mkdir(parents=True)
    result = {"names": {i: str(i) for i in range(class_count)}, "data_role": "synthetic_only"}
    for role, count in (("train",train_count),("training_val",val_count)):
        role_dir = root / role
        files = role_dir / "files"
        files.mkdir(parents=True)
        samples=[]
        for i in range(count):
            global_i = i + (0 if role == "train" else train_count)
            h,w = ((33,101),(101,33),(47,67),(64,64)) [global_i % 4]
            y,x = np.indices((h,w))
            image = np.stack(((x*7+global_i*13)%256,(y*11+global_i*31)%256,((x+y)*3+global_i*47)%256),axis=2).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            _require(ok, "synthetic PNG encoder failed")
            image_id = f"train/synthetic_{role}_{i:03d}"
            labels = "0 0.25 0.25 0.2 0.2\n1 0.75 0.75 0.2 0.2\n".encode("utf-8")
            rows=[]
            for line,(score,category,xywh) in enumerate(((0,2,(w*.05,h*.05,w*.9,h*.9)),(1,0,(w*.45,h*.2,w*.2,h*.25)),(1,11,(w*.7,h*.6,w*.2,h*.3)))):
                bx,by,bw,bh = xywh
                original=[bx,by,bx+bw,by+bh]
                reasons=[name for flag,name in ((score==0,"score_0"),(category==0,"category_0"),(category==11,"category_11")) if flag]
                rows.append({"image_id":image_id,"target_id":f"{image_id}:{line}","physical_line_zero_based":line,
                             "score":score,"category_original":category,"publisher_ignored":True,"ignored":True,
                             "excluded_for_preparation_quality":False,"original_xywh":list(xywh),"original_xyxy":original,
                             "clipped_xyxy":original,"ignore_reasons":reasons})
            sidecar=("\n".join(json.dumps(row,sort_keys=True,allow_nan=False) for row in rows)+"\n").encode("utf-8")
            members={}
            for key,name,raw in (("image",f"{i:03d}.png",encoded.tobytes()),("labels",f"{i:03d}.txt",labels),("sidecar",f"{i:03d}.ignore.jsonl",sidecar)):
                (files/name).write_bytes(raw)
                members[key]={"path":name,"bytes":len(raw),"sha256":_sha(raw)}
            samples.append({"image_id":image_id,"split_role":role,"original_hw":[h,w],**members})
        manifest={"schema":MANIFEST_SCHEMA,"data_role":"synthetic_only","split_role":role,"class_count":class_count,
                  "data_root":"files","policy":copy.deepcopy(POLICY),"samples":samples}
        path=role_dir/"manifest.json"
        raw=(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
        path.write_bytes(raw)
        prefix="train" if role=="train" else "val"
        result[f"{prefix}_manifest"]=str(path.resolve())
        result[f"{prefix}_manifest_sha256"]=_sha(raw)
    return result
