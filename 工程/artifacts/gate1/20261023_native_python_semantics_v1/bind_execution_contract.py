"""Bind passed native output identities to the already frozen comparison rules."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONDITIONS_SHA = "dcde553318db54b14d7f837434e26ae147250d8ef8b8ca1a62c3e36adadea84f"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def main():
    conditions_path = HERE / "predeclared_conditions.json"
    assert sha(conditions_path) == CONDITIONS_SHA
    conditions = json.loads(conditions_path.read_text(encoding="utf-8"))
    for field in ("comparison_source", "test_source", "test_result", "python_source", "fixture"):
        row = conditions[field]
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], field
    audit_path = ROOT / "artifacts/gate1/20261023_native_oracle_comparison_preparation_v1/native_output_audit_v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS_NATIVE_SYNTHETIC_OUTPUT_STRUCTURE_IDENTITIES_ONLY"
    contract = {"schema": "rampq.gate1.native_python_partial_comparison_contract/v1",
                "predeclared_conditions": identity(conditions_path),
                **{key: conditions[key] for key in ("case_ids", "matrix_fields", "absolute_tolerance", "relative_tolerance", "comparison_source", "python_source", "fixture")},
                "native_structure_audit": identity(audit_path),
                "native_mat": identity(ROOT / "artifacts/gate1/20261023_native_oracle_runtime_v1/execution_01/oracle_results.mat"),
                "auditor_source": identity(ROOT / "artifacts/gate1/20261023_native_oracle_comparison_preparation_v1/audit_native_oracle.py"),
                "all_twenty_cases_and_all_matching_categories_required": True,
                "AP_metrics_compared": False, "full_Python_AP_implemented": False,
                "official_AP_equivalence_certified": False, "gate1_scientific_pass": False}
    path = HERE / "execution_contract_v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(identity(path)))


if __name__ == "__main__":
    main()
