"""Check all already selected training members without model inference."""
import sys
from pathlib import Path

import gate1_intervention_training_v2 as P


def main():
    P.configure()
    c = P.read(P.PRIOR / "contract.json")
    dataset = P.SelectedTrainingDataset(c["train_manifest"], c["training_members"])
    assignment = P.ROOT / "artifacts/gate1/20261022_split_refinement/split_plan_v1/split_assignment.jsonl"
    selected = {r["image_id"]: r for r in c["training_members"]}
    originals = {}
    for line in assignment.read_text(encoding="utf-8").splitlines():
        r = P.json.loads(line)
        if r["image_id"] in selected:
            P.require(r["assigned_split"] == "train", "Wrong member role")
            P.require(r["image_sha256"] == selected[r["image_id"]]["image"]["sha256"], "Image differs")
            originals[r["image_id"]] = r
    P.require(set(originals) == set(selected), "Missing original training metadata")
    root = Path(P.read(c["evaluation_protocol"]["path"])["prepared_root"]).resolve(strict=True)
    checks = []
    for i, row in enumerate(c["training_members"]):
        meta = originals[row["image_id"]]
        annotation = (root / meta["annotation_path"]).resolve(strict=True)
        P.require(annotation.is_relative_to(root) and P.sha(annotation) == meta["annotation_sha256"], "Annotation identity differs")
        gt = P.MATCH.parse_ground_truth(annotation.read_text(encoding="utf-8-sig"), meta["image_id"])
        sample = dataset[i]
        params = P.original_training_geometry(sample, gt, meta["width"], meta["height"])
        checks.append({"image_id": meta["image_id"], "annotation": P.identity(annotation),
                       "positive_targets": len(sample["cls"]), "params": params, "PASS": True})
        if (i + 1) % 128 == 0:
            print(f"Geometry checked {i + 1}/512", flush=True)
    P.write(P.OUT / "geometry_preflight.json", {
        "status": "PASS_ORIGINAL_TRAINING_GEOMETRY_ALL_512", "created_at_utc": P.now(),
        "script": P.identity(__file__), "pilot_source": P.identity(P.__file__),
        "previous_contract": P.identity(P.PRIOR / "contract.json"), "assignment": P.identity(assignment),
        "images": len(checks), "checks": checks, "model_inference": False,
        "Selection_payload_access": False, "Holdout_payload_access": False})
    print("PASS geometry identity bridge for all 512 selected training members", flush=True)


if __name__ == "__main__":
    main()
