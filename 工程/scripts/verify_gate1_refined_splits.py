"""Independent saved-data verification of the Gate1 split_plan_v1.

No import of the split producer; no models, image decoding, or test-dev access.
All 21,077 files enumerated by the original prepared manifest are byte-checked.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

SOURCE_MANIFEST_SHA = "7e42fb1aa0725c99864ef9197dc65bb07bb00b5de8eab1a54675f5ea7df7c15f"
SOURCE_INVENTORY_SHA = "e657bc24d4cdf16b868bf91111647a933bec95d90385b7fc8ceab483020f8048"
EXPECTED_COUNTS = {"train": 4534, "training_val": 414, "calibration_pool": 689,
                   "debug": 109, "quarantine": 725, "selection": 548}
EXPECTED_COMPONENT_COUNTS = {"train": 92, "training_val": 58, "calibration_pool": 24,
                             "debug": 6, "quarantine": 27, "selection": 75}
EXPECTED_POLICY = {"seed": 20261022, "training_val_target": 400, "training_val_min_components": 12,
    "training_val_max_component_images": 64, "calibration_pool_target": 640, "calibration_pool_min_components": 24,
    "calibration_seeds": [2026102201, 2026102202, 2026102203], "calibration_per_seed": 64,
    "calibration_min_components_per_seed": 12, "calibration_max_images_per_component_per_seed": 8,
    "minimum_train_images": 3000, "minimum_train_components": 60}
ACTIVE = {"train", "training_val", "calibration_pool", "debug", "selection"}
ALL_ROLES = ACTIVE | {"quarantine", "duplicate_excluded"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def safe_member(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if (not parts or PurePosixPath(relative).is_absolute() or "\\" in relative or ":" in relative
            or any(p in {".", ".."} for p in relative.split("/"))
            or any(marker in part.casefold() for part in parts
                   for marker in ("test-dev", "testdev", "test-challenge", "sealed_holdout"))):
        raise ValueError(f"Forbidden or escaped manifest path: {relative}")
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest member resolves outside root: {relative}")
    return resolved


class Components:
    """Union by size with iterative path compression; semantic names are minima."""
    def __init__(self, identifiers):
        self.parent = {key: key for key in identifiers}
        self.size = dict.fromkeys(self.parent, 1)

    def find(self, key):
        trail = []
        while self.parent[key] != key:
            trail.append(key)
            key = self.parent[key]
        for previous in trail:
            self.parent[previous] = key
        return key

    def connect(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def order_for(salt: str, key: str) -> tuple[str, str]:
    return hashlib.sha256(f"20261022:{salt}:{key}".encode()).hexdigest(), key


def verify(prepared: Path, plan_root: Path, review_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    prepared, plan_root, review_path = prepared.resolve(strict=True), plan_root.resolve(strict=True), review_path.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Independent audit output already exists")
    output = output.resolve()
    if output.is_relative_to(prepared) or output.is_relative_to(plan_root):
        raise ValueError("Audit output must be outside both preserved roots")
    producer = Path(__file__).resolve().with_name("refine_gate1_splits.py")
    watched = {"source_manifest": prepared / "manifest.json", "source_inventory": prepared / "image_manifest.jsonl",
        "plan_manifest": plan_root / "manifest.json", "plan_summary": plan_root / "split_plan.json",
        "plan_assignments": plan_root / "split_assignment.jsonl", "review": review_path,
        "producer_script": producer, "independent_verifier_script": Path(__file__).resolve()}
    pre = {name: digest(path) for name, path in watched.items()}
    output.mkdir(parents=True)
    checks = []
    def check(condition, name, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(name)
    try:
        check(pre["source_manifest"] == SOURCE_MANIFEST_SHA and pre["source_inventory"] == SOURCE_INVENTORY_SHA,
              "original_prepared_source_identities_match_prior_verified_pins")
        source_manifest = read_json(watched["source_manifest"])
        entries = source_manifest["files"]
        check(len(entries) == 21077 and len({e["path"] for e in entries}) == 21077,
              "all_21077_original_prepared_manifest_entries_unique")
        with (output / "source_file_byte_checks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for index, entry in enumerate(entries):
                path = safe_member(prepared, entry["path"])
                actual_bytes, actual_sha = path.stat().st_size, digest(path)
                if actual_bytes != entry["bytes"] or actual_sha != entry["sha256"]:
                    raise AssertionError(f"Original source bytes changed: {entry['path']}")
                handle.write(json.dumps({"path": entry["path"], "bytes": actual_bytes, "sha256": actual_sha, "verified": True}) + "\n")
                if (index + 1) % 5000 == 0:
                    print(f"ORIGINAL_FILES_VERIFIED {index + 1}/21077", flush=True)
        check(True, "all_21077_original_files_bytes_and_SHA256_unchanged")
        manifest = read_json(watched["plan_manifest"])["files"]
        for relative, metadata in manifest.items():
            path = safe_member(plan_root, relative)
            check(path.stat().st_size == metadata["bytes"] and digest(path) == metadata["sha256"], "plan_member_integrity", relative)
        actual_plan_files = {p.relative_to(plan_root).as_posix() for p in plan_root.rglob("*") if p.is_file()}
        check(actual_plan_files == set(manifest) | {"manifest.json"}, "plan_manifest_exact_file_coverage")
        source_rows = jsonl(watched["source_inventory"])
        rows = jsonl(watched["plan_assignments"])
        original = {r["image_id"]: r for r in source_rows}
        by_id = {r["image_id"]: r for r in rows}
        check(len(source_rows) == len(original) == len(rows) == len(by_id) == 7019 and original.keys() == by_id.keys(),
              "all_7019_records_preserved_once_with_no_new_IDs")
        for identifier, old in original.items():
            new = by_id[identifier]
            check(set(new) == set(old) | {"previous_assigned_split", "refined_component"}, "source_field_set_preserved", identifier)
            if any(new[key] != value for key, value in old.items() if key != "assigned_split"):
                raise AssertionError(f"Original record field changed: {identifier}")
            if new["previous_assigned_split"] != old["assigned_split"]:
                raise AssertionError(f"Original role history changed: {identifier}")
        check(True, "every_original_field_except_assigned_role_preserved_exactly")
        summary, review = read_json(watched["plan_summary"]), read_json(review_path)
        check(summary["policy"] == EXPECTED_POLICY, "frozen_policy_matches_independent_constants")
        check(summary["source_manifest_sha256"] == pre["source_inventory"]
            and summary["source_file_manifest_sha256"] == pre["source_manifest"]
            and summary["risk_review_sha256"] == pre["review"]
            and summary["runner_sha256"] == pre["producer_script"], "plan_source_review_and_producer_identity_chain")
        check(Path(summary["source_prepared"]).resolve() == prepared and Path(summary["risk_review_path"]).resolve() == review_path,
              "plan_source_and_review_absolute_paths_match_actual_inputs")
        check(review["source_manifest_sha256"] == pre["source_inventory"] and review["holdout_accessed"] is False
            and review["model_outputs_accessed"] is False, "review_scope_and_source_identity")
        evidence_root = review_path.parent / "near_duplicate_audit"
        for name, expected in review["bound_evidence"].items():
            check(digest(safe_member(evidence_root, name)) == expected, "review_bound_evidence_integrity", name)
        audit_manifest = read_json(evidence_root / "audit_manifest.json")
        for entry in audit_manifest["files"]:
            path = safe_member(evidence_root, entry["path"])
            check(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "near_duplicate_audit_member_integrity", entry["path"])
        candidates = {r["pair_id"]: r for r in jsonl(evidence_root / "candidate_pairs.jsonl")}
        decisions = {r["pair_id"]: r for r in review["pair_decisions"]}
        check(len(decisions) == len(review["pair_decisions"]) == len(candidates) == 5 and decisions.keys() == candidates.keys(),
              "all_five_saved_near_candidate_pairs_have_review_decisions")
        for pair_id, decision in decisions.items():
            pair = candidates[pair_id]
            check((decision["a"], decision["b"]) == (pair["a"]["image_id"], pair["b"]["image_id"]),
                  "review_pair_IDs_match_saved_candidate", pair_id)
        check(len(review["risk_edges"]) == summary["risk_edge_count"] == 1, "one_explicit_reviewed_near_risk_edge")
        structure = Components(original)
        for field in ("prefix_proxy", "image_sha256"):
            first = {}
            for row in source_rows:
                value = row[field]
                if value in first:
                    structure.connect(first[value], row["image_id"])
                else:
                    first[value] = row["image_id"]
        for edge in review["risk_edges"]:
            a, b = edge["a"], edge["b"]
            check(a in original and b in original and a != b, "risk_edge_source_IDs_exist")
            check(edge["a_image_sha256"] == original[a]["image_sha256"] and edge["b_image_sha256"] == original[b]["image_sha256"],
                  "risk_edge_source_image_SHAs_match_preserved_inventory")
            decision = decisions[edge["source_pair_id"]]
            check(decision["decision"] == "ADD_CONSERVATIVE_RISK_EDGE" and (decision["a"], decision["b"]) == (a, b)
                  and bool(edge["reason"]), "risk_edge_is_explicitly_accepted_review_pair")
            structure.connect(a, b)
        groups = defaultdict(list)
        for identifier, row in original.items():
            groups[structure.find(identifier)].append(row)
        components = {min(r["image_id"] for r in bucket): bucket for bucket in groups.values()}
        component_by_id = {r["image_id"]: key for key, bucket in components.items() for r in bucket}
        check(all(by_id[key]["refined_component"] == component_by_id[key] for key in original),
              "independent_DSU_rebuild_matches_every_refined_component")
        expected_roles, free, calibration = {}, {}, {}
        quality_ids = []
        for component, bucket in components.items():
            old_roles = {r["assigned_split"] for r in bucket}
            has_val = any(r["source_split"] == "val" for r in bucket)
            has_quality = any(r.get("requires_annotation_quality_quarantine", False) for r in bucket)
            if has_quality:
                quality_ids.extend(r["image_id"] for r in bucket if r["source_split"] == "train")
            for row in bucket:
                identifier = row["image_id"]
                if row["source_split"] == "val":
                    expected_roles[identifier] = "selection"
                elif has_val or has_quality or "quarantine" in old_roles:
                    expected_roles[identifier] = "quarantine"
                elif not row["is_content_canonical"]:
                    expected_roles[identifier] = "duplicate_excluded"
                elif "debug" in old_roles:
                    expected_roles[identifier] = "debug"
                elif "calibration_pool" in old_roles:
                    expected_roles[identifier] = "calibration_pool"
                    calibration.setdefault(component, []).append(row)
                else:
                    free.setdefault(component, []).append(row)
        def enough_calibration():
            return (sum(len(bucket) for bucket in calibration.values()) >= 640 and len(calibration) >= 24
                and sum(len(bucket) >= 3 for bucket in calibration.values()) >= 12
                and sum(min(len(bucket), 24) for bucket in calibration.values()) >= 192)
        for key in sorted((key for key, bucket in free.items() if len(bucket) <= 64), key=lambda key: order_for("additional_calibration", key)):
            if enough_calibration():
                break
            calibration[key] = free.pop(key)
            for row in calibration[key]:
                expected_roles[row["image_id"]] = "calibration_pool"
        check(enough_calibration(), "independent_diverse_calibration_pool_feasibility")
        validation = {}
        for key in sorted((key for key, bucket in free.items() if len(bucket) <= 64), key=lambda key: order_for("diverse_training_val", key)):
            if sum(map(len, validation.values())) >= 400 and len(validation) >= 12:
                break
            validation[key] = free.pop(key)
            for row in validation[key]:
                expected_roles[row["image_id"]] = "training_val"
        for bucket in free.values():
            for row in bucket:
                expected_roles[row["image_id"]] = "train"
        check(set(expected_roles) == set(original) and all(by_id[key]["assigned_split"] == role for key, role in expected_roles.items()),
              "all_roles_match_independent_priority_and_deterministic_group_allocation")
        counts = dict(Counter(r["assigned_split"] for r in rows))
        component_counts = {role: len({r["refined_component"] for r in rows if r["assigned_split"] == role}) for role in counts}
        check(counts == EXPECTED_COUNTS == summary["counts"], "all_actual_role_counts_match_plan_and_expected_delivery", counts)
        check(component_counts == EXPECTED_COMPONENT_COUNTS == summary["component_counts"], "all_actual_component_counts_match_delivery", component_counts)
        for field in ("image_sha256", "prefix_proxy", "refined_component"):
            roles = defaultdict(set)
            for row in rows:
                if row["assigned_split"] in ACTIVE:
                    roles[row[field]].add(row["assigned_split"])
            check(all(len(v) == 1 for v in roles.values()), "complete_active_role_isolation", field)
        official_val = [r for r in rows if r["source_split"] == "val"]
        check(len(official_val) == 548 and all(r["assigned_split"] == "selection" for r in official_val), "all_548_official_val_remain_only_Selection")
        for role in ("train", "training_val", "calibration_pool", "debug"):
            check(all(r["source_split"] == "train" for r in rows if r["assigned_split"] == role), "nonSelection_active_role_uses_only_official_train", role)
        quality = read_json(prepared / "preparation_summary.json")["annotation_quality_policy"]
        check(len(quality["quarantined_train_image_ids"]) == 79 and set(quality["quarantined_train_image_ids"]) == set(quality_ids)
            and all(by_id[k]["assigned_split"] == "quarantine" for k in quality_ids), "all_79_original_quality_component_images_still_quarantined")
        validation_sizes = Counter(r["refined_component"] for r in rows if r["assigned_split"] == "training_val")
        check(len(validation_sizes) >= 12 and max(validation_sizes.values()) <= 64 and sum(validation_sizes.values()) >= 400
            and dict(validation_sizes) == summary["training_val_component_sizes"], "training_val_414_images_58_complete_small_components", dict(validation_sizes))
        check(counts["train"] >= 3000 and component_counts["train"] >= 60, "training_data_minimum_size_and_diversity")
        for role in sorted(ALL_ROLES):
            expected_paths = [str(prepared / r["image_path"]).replace("\\", "/") for r in sorted(rows, key=lambda r: r["image_id"]) if r["assigned_split"] == role]
            actual = (plan_root / "splits" / f"{role}.txt").read_text(encoding="utf-8").splitlines()
            check(actual == expected_paths and len(set(actual)) == len(actual), "split_list_exact_original_absolute_paths_and_order", role)
        calibration_proofs, combined_ids = {}, []
        selections = summary["calibration_image_ids"]
        check(set(selections) == {"2026102201", "2026102202", "2026102203"}, "three_fixed_calibration_seeds")
        for seed, identifiers in selections.items():
            check(len(identifiers) == len(set(identifiers)) == 64 and all(k in by_id and by_id[k]["assigned_split"] == "calibration_pool" for k in identifiers),
                  "calibration_seed_64_unique_pool_members", seed)
            coverage = Counter(component_by_id[k] for k in identifiers)
            check(len(coverage) >= 12 and max(coverage.values()) <= 8, "calibration_seed_group_diversity_and_max8", seed)
            expected = [str(prepared / original[k]["image_path"]).replace("\\", "/") for k in identifiers]
            check((plan_root / "splits" / f"calibration_seed_{seed}.txt").read_text(encoding="utf-8").splitlines() == expected,
                  "calibration_seed_list_paths_and_order", seed)
            calibration_proofs[seed] = {"images": len(identifiers), "components": len(coverage), "maximum_per_component": max(coverage.values()), "component_counts": dict(coverage)}
            combined_ids.extend(identifiers)
        check(len(combined_ids) == len(set(combined_ids)) == len({original[k]["image_sha256"] for k in combined_ids}) == 192,
              "all_192_calibration_images_disjoint_by_ID_and_content_SHA")
        post = {name: digest(path) for name, path in watched.items()}
        check(pre == post, "all_source_plan_review_producer_and_verifier_identities_unchanged_pre_post")
        result = {"status": "PASS_INDEPENDENT_SAVED_SPLIT_PLAN_VERIFICATION", "source_records": len(source_rows),
            "all_original_source_files_verified": len(entries), "plan_files_verified": len(manifest), "counts": counts,
            "component_counts": component_counts, "calibration_seed_checks": calibration_proofs,
            "source_fields_preserved_except_role": True, "all_role_priorities_independently_reconstructed": True,
            "all_active_roles_isolated_by_prefix_content_and_reviewed_graph": True,
            "original_quality_images_retained_in_quarantine": len(quality_ids),
            "source_before_sha256": pre, "source_after_sha256": post, "checks_passed": len(checks),
            "holdout_or_testdev_ZIP_read_or_listed": False, "producer_imported": False,
            "model_executed": False, "old_source_or_plan_modified": False,
            "does_not_prove_scene_identity_or_absence_of_unflagged_near_duplicates": True,
            "elapsed_seconds": time.perf_counter() - started}
        save_json(output / "verification.json", result)
        save_json(output / "checks.json", {"checks": checks})
        save_json(output / "audit_manifest.json", {"files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(output.iterdir()) if p.is_file()]})
        return result
    except Exception as error:
        save_json(output / "failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "error": str(error), "checks": checks,
            "pre_sha256": pre, "post_sha256": {name: digest(path) for name, path in watched.items()}})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--risk-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.prepared, args.plan, args.risk_review, args.output)
    print(json.dumps({k: result[k] for k in ("status", "all_original_source_files_verified", "checks_passed", "counts", "component_counts", "elapsed_seconds")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
