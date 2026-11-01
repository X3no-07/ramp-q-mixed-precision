"""Create a new split plan without changing prepared images or old evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path


POLICY = {
    "seed": 20261022,
    "training_val_target": 400,
    "training_val_min_components": 12,
    "training_val_max_component_images": 64,
    "calibration_pool_target": 640,
    "calibration_pool_min_components": 24,
    "calibration_seeds": [2026102201, 2026102202, 2026102203],
    "calibration_per_seed": 64,
    "calibration_min_components_per_seed": 12,
    "calibration_max_images_per_component_per_seed": 8,
    "minimum_train_images": 3000,
    "minimum_train_components": 60,
}
ACTIVE = {"train", "training_val", "calibration_pool", "debug", "selection"}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rank(salt: str, value: str) -> str:
    return hashlib.sha256(f"{POLICY['seed']}:{salt}:{value}".encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def allocate_calibration(pool: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Reserve multi-group coverage, then solve a small integer max-flow."""
    seeds = POLICY["calibration_seeds"]
    minimum = POLICY["calibration_min_components_per_seed"]
    quota = POLICY["calibration_per_seed"]
    cap = POLICY["calibration_max_images_per_component_per_seed"]
    core = sorted((key for key, rows in pool.items() if len(rows) >= len(seeds)), key=lambda key: (rank("calibration_core", key), key))[:minimum]
    require(len(core) == minimum, "Insufficient multi-image components for every calibration seed")
    chosen = {str(seed): [] for seed in seeds}
    remaining = {key: list(rows) for key, rows in pool.items()}
    for key in core:
        items = sorted(remaining[key], key=lambda r: (rank("calibration_reserve", r["image_id"]), r["image_id"]))
        reserved = {item["image_id"] for item in items[:len(seeds)]}
        remaining[key] = [r for r in remaining[key] if r["image_id"] not in reserved]
        for seed, item in zip(seeds, items):
            chosen[str(seed)].append(item["image_id"])

    capacities: dict[str, dict[str, int]] = defaultdict(dict)
    adjacency: dict[str, list[str]] = defaultdict(list)
    def edge(a: str, b: str, value: int) -> None:
        capacities[a][b] = value
        capacities[b][a] = 0
        adjacency[a].append(b)
        adjacency[b].append(a)
    source, sink = "SOURCE", "SINK"
    keys = sorted(pool, key=lambda key: (rank("calibration_flow", key), key))
    for key in keys:
        edge(source, "G:" + key, len(remaining[key]))
        for seed in seeds:
            edge("G:" + key, "S:" + str(seed), cap - int(key in core))
    for seed in seeds:
        edge("S:" + str(seed), sink, quota - minimum)
    total = 0
    while True:
        previous = {source: None}
        queue = deque([source])
        while queue and sink not in previous:
            node = queue.popleft()
            for other in adjacency[node]:
                if other not in previous and capacities[node][other] > 0:
                    previous[other] = node
                    queue.append(other)
        if sink not in previous:
            break
        amount = quota
        node = sink
        while previous[node] is not None:
            amount = min(amount, capacities[previous[node]][node])
            node = previous[node]
        node = sink
        while previous[node] is not None:
            before = previous[node]
            capacities[before][node] -= amount
            capacities[node][before] += amount
            node = before
        total += amount
    require(total == len(seeds) * (quota - minimum), "Calibration group coverage/capacity is infeasible")
    for key in keys:
        unused = {r["image_id"] for r in remaining[key]}
        for seed in seeds:
            count = capacities["S:" + str(seed)]["G:" + key]
            items = sorted(unused, key=lambda value: (hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(), value))[:count]
            require(len(items) == count, "Calibration image allocation failed")
            chosen[str(seed)].extend(items)
            unused.difference_update(items)
    by_id = {r["image_id"]: key for key, rows in pool.items() for r in rows}
    all_ids = []
    for seed in seeds:
        ids = chosen[str(seed)]
        counts = Counter(by_id[value] for value in ids)
        require(len(ids) == quota and len(set(ids)) == quota, "Calibration size/uniqueness failure")
        require(len(counts) >= minimum and max(counts.values()) <= cap, "Calibration component coverage failure")
        all_ids.extend(ids)
    require(len(set(all_ids)) == len(all_ids), "Calibration seeds overlap by image")
    return chosen


def build_plan(source_rows: list[dict], risk_edges: list[dict]) -> tuple[list[dict], dict]:
    rows = [dict(row, previous_assigned_split=row["assigned_split"]) for row in source_rows]
    by_id = {row["image_id"]: row for row in rows}
    require(len(by_id) == len(rows), "Duplicate image IDs")
    parent = {key: key for key in by_id}
    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    for field in ("prefix_proxy", "image_sha256"):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row[field]].append(row["image_id"])
        for bucket in buckets.values():
            for key in bucket[1:]:
                union(bucket[0], key)
    for edge in risk_edges:
        a, b = edge["a"], edge["b"]
        require(a in by_id and b in by_id and a != b, "Unknown/self near-risk edge")
        require(edge.get("reason"), "Near-risk edge must carry an explicit reason")
        union(a, b)
    components = defaultdict(list)
    for row in rows:
        components[find(row["image_id"])].append(row)
    available = {}
    calibration = {}
    for key, bucket in sorted(components.items()):
        old_roles = {r["previous_assigned_split"] for r in bucket}
        val_present = any(r["source_split"] == "val" for r in bucket)
        quality_present = any(r.get("requires_annotation_quality_quarantine", False) for r in bucket)
        for row in bucket:
            row["refined_component"] = key
            if row["source_split"] == "val":
                row["assigned_split"] = "selection"
            elif val_present or quality_present or "quarantine" in old_roles:
                row["assigned_split"] = "quarantine"
            elif not row["is_content_canonical"]:
                row["assigned_split"] = "duplicate_excluded"
            elif "debug" in old_roles:
                row["assigned_split"] = "debug"
            elif "calibration_pool" in old_roles:
                row["assigned_split"] = "calibration_pool"
                calibration.setdefault(key, []).append(row)
            else:
                require(row["source_split"] == "train", "Only official train can be allocated")
                row["assigned_split"] = "unassigned"
                available.setdefault(key, []).append(row)
    def pool_ready() -> bool:
        return (sum(map(len, calibration.values())) >= POLICY["calibration_pool_target"]
                and len(calibration) >= POLICY["calibration_pool_min_components"]
                and sum(len(v) >= 3 for v in calibration.values()) >= POLICY["calibration_min_components_per_seed"]
                and sum(min(len(v), 3 * POLICY["calibration_max_images_per_component_per_seed"]) for v in calibration.values()) >= 192)
    eligible = sorted((key for key, bucket in available.items() if len(bucket) <= POLICY["training_val_max_component_images"]), key=lambda key: (rank("additional_calibration", key), key))
    for key in eligible:
        if pool_ready():
            break
        calibration[key] = available.pop(key)
        for row in calibration[key]:
            row["assigned_split"] = "calibration_pool"
    require(pool_ready(), "Cannot create diverse calibration pool without splitting components")
    validation = {}
    eligible = sorted((key for key, bucket in available.items() if len(bucket) <= POLICY["training_val_max_component_images"]), key=lambda key: (rank("diverse_training_val", key), key))
    for key in eligible:
        if sum(map(len, validation.values())) >= POLICY["training_val_target"] and len(validation) >= POLICY["training_val_min_components"]:
            break
        validation[key] = available.pop(key)
        for row in validation[key]:
            row["assigned_split"] = "training_val"
    require(sum(map(len, validation.values())) >= POLICY["training_val_target"] and len(validation) >= POLICY["training_val_min_components"], "Cannot create diverse training validation")
    require(sum(map(len, available.values())) >= POLICY["minimum_train_images"] and len(available) >= POLICY["minimum_train_components"], "Too little training data after isolation")
    for bucket in available.values():
        for row in bucket:
            row["assigned_split"] = "train"
    for field in ("image_sha256", "prefix_proxy", "refined_component"):
        roles = defaultdict(set)
        for row in rows:
            if row["assigned_split"] in ACTIVE:
                roles[row[field]].add(row["assigned_split"])
        require(all(len(value) == 1 for value in roles.values()), f"Active-role isolation failed for {field}")
    chosen = allocate_calibration(calibration)
    stats = {"status": "REFINED_SPLIT_PLAN_NOT_SCIENTIFIC_GATE_PASS", "policy": POLICY,
             "counts": dict(Counter(r["assigned_split"] for r in rows)),
             "component_counts": {role: len({r["refined_component"] for r in rows if r["assigned_split"] == role}) for role in sorted({r["assigned_split"] for r in rows})},
             "training_val_component_sizes": {key: len(value) for key, value in validation.items()},
             "calibration_image_ids": chosen, "risk_edge_count": len(risk_edges),
             "scene_identity_proven": False, "holdout_accessed": False,
             "limits": ["Filename prefixes and screened near-risk edges are conservative grouping proxies, not proven camera/scene identity", "No absence-of-near-duplicates guarantee", "No Selection model scoring or Holdout access"]}
    return sorted(rows, key=lambda r: r["image_id"]), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--risk-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepared = args.prepared.resolve(strict=True)
    review_path = args.risk_review.resolve(strict=True)
    output = args.output.resolve()
    require(not output.exists() and not args.output.is_symlink(), "Refusing existing output")
    require(not output.is_relative_to(prepared), "Split plan must be outside preserved preparation")
    manifest_path = prepared / "image_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    require(review["source_manifest_sha256"] == sha(manifest_path), "Near-risk review source identity mismatch")
    require(review["holdout_accessed"] is False, "Holdout is outside this preparation scope")
    rows, stats = build_plan(rows, review["risk_edges"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "splits").mkdir()
    by_id = {r["image_id"]: r for r in rows}
    for role in ("train", "training_val", "calibration_pool", "debug", "selection", "quarantine", "duplicate_excluded"):
        paths = [str(prepared / r["image_path"]).replace("\\", "/") for r in rows if r["assigned_split"] == role]
        (output / "splits" / f"{role}.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    for seed, ids in stats["calibration_image_ids"].items():
        paths = [str(prepared / by_id[key]["image_path"]).replace("\\", "/") for key in ids]
        (output / "splits" / f"calibration_seed_{seed}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    (output / "split_assignment.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    stats.update({"source_prepared": str(prepared), "source_manifest_sha256": sha(manifest_path), "source_file_manifest_sha256": sha(prepared / "manifest.json"), "risk_review_path": str(review_path), "risk_review_sha256": sha(review_path), "runner_sha256": sha(Path(__file__).resolve())})
    (output / "split_plan.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = {str(f.relative_to(output)).replace("\\", "/"): {"bytes": f.stat().st_size, "sha256": sha(f)} for f in sorted(output.rglob("*")) if f.is_file()}
    (output / "manifest.json").write_text(json.dumps({"files": files}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: stats[key] for key in ("status", "counts", "component_counts", "risk_edge_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
