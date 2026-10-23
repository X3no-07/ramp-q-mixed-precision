"""Bind the fixed auditor's completed native-output audit without editing it."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    preparation = json.loads((HERE / "preparation_manifest.json").read_text(encoding="utf-8"))
    for row in preparation["files"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"] and identity(path)["sha256"] == row["sha256"]
    audit_path = HERE / "native_output_audit_v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS_NATIVE_SYNTHETIC_OUTPUT_STRUCTURE_IDENTITIES_ONLY"
    assert all(row["method"] == "live_samefile" for row in audit["resolved_function_path_checks"])
    runtime_dir = ROOT / "artifacts/gate1/20261023_native_oracle_runtime_v1"
    supplemental = runtime_dir / "subst_query_elevated_active.txt"
    members = [HERE / "preparation_manifest.json", HERE / "seal_native_audit.py", audit_path]
    if supplemental.exists():
        members.append(supplemental)
    manifest = {"schema": "rampq.gate1.native_synthetic_oracle_saved_output_audit_manifest/v1",
                "path_base": "workspace", "file_count": len(members), "files": [identity(path) for path in members],
                "native_execution_inputs_outputs_and_runtime_bound_via_audit_and_launcher": True,
                "six_official_functions_verified_live_samefile": True,
                "AP_equivalence_or_gate1_scientific_acceptance_claimed": False,
                "native_nan_case_ids_observed": ["A03_empty_both", "A04_predictions_without_gt"]}
    with (HERE / "native_output_audit_manifest_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(identity(HERE / "native_output_audit_manifest_v1.json")))


if __name__ == "__main__":
    main()
