"""Read-only saved-evidence review of the root's actual synthetic Trainer v3, with observed CPU thread controls."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DELIVERY = ROOT / "artifacts/gate1/20261023_training_integration_delivery_v1"
ATTEMPT = DELIVERY / "attempt_03"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}

def verify_manifest(path, base):
    manifest = read(path)
    for name, expected in manifest["files"].items():
        member = base / name
        assert member.is_file() and member.stat().st_size == expected["bytes"] and sha(member) == expected["sha256"], name
    return {**identity(path), "listed_files_verified": len(manifest["files"])}

source_review = read(HERE / "source_review.json")
for rel, expected in source_review["sources"].items():
    assert sha(ROOT / rel) == expected["sha256"]
verified = [verify_manifest(DELIVERY / "preparation_manifest_v3.json", ROOT),
            verify_manifest(ATTEMPT / "manifest.json", ATTEMPT),
            verify_manifest(ATTEMPT / "fp32_capture/epoch_0001/manifest.json", ATTEMPT / "fp32_capture/epoch_0001"),
            verify_manifest(ATTEMPT / "fp32_capture/epoch_0001/state/manifest.json", ATTEMPT / "fp32_capture/epoch_0001/state")]
contract = read(DELIVERY / "training_contract_v3.json")
receipt = read(ATTEMPT / "training_receipt.json")
assert receipt["training_contract_sha256"] == sha(DELIVERY / "training_contract_v3.json")
assert receipt["preparation_manifest_sha256"] == sha(DELIVERY / "preparation_manifest_v3.json")
role_ids, role_shas, member_count = {}, {}, 0
for role, binding in contract["datasets"].items():
    path = Path(binding["manifest_path"])
    assert sha(path) == binding["manifest_sha256"]
    manifest = read(path)
    assert manifest["data_role"] == "synthetic_only" and manifest["split_role"] == role
    role_ids[role] = {row["image_id"] for row in manifest["samples"]}
    role_shas[role] = {row["image"]["sha256"] for row in manifest["samples"]}
    for row in manifest["samples"]:
        assert row["split_role"] == role
        for key in ("image", "labels", "sidecar"):
            member = path.parent / manifest["data_root"] / row[key]["path"]
            assert member.stat().st_size == row[key]["bytes"] and sha(member) == row[key]["sha256"]
            member_count += 1
assert len(role_ids["train"]) == 4 and len(role_ids["training_val"]) == 2
assert not role_ids["train"] & role_ids["training_val"] and not role_shas["train"] & role_shas["training_val"]
split_preflight = read(DELIVERY / "split_identity_preflight.json")
assert split_preflight["identity_intersection"] == [] and split_preflight["image_sha256_intersection"] == []

events = [json.loads(line) for line in (ATTEMPT / "events.jsonl").read_text().splitlines()]
assert [e["sequence"] for e in events] == list(range(len(events)))
names = [e["event"] for e in events]
assert names[:4] == ["external_integration_registration_disabled", "effective_cpu_threads_reapplied_after_Trainer_constructor", "base_callbacks_verified_before_custom_hooks", "actual_DetectionTrainer_train_call_enter"]
assert names[-1] == "actual_DetectionTrainer_train_call_return"
callback_inventory = read(ATTEMPT / "base_callback_inventory.json")
assert callback_inventory and all(name.startswith("ultralytics.utils.callbacks.base.") for callbacks in callback_inventory.values() for name in callbacks)
assert receipt["external_integration_registration_disabled_in_process"] and receipt["base_callback_inventory_verified_before_custom_hooks"]
runtime_events = [e for e in events if "runtime_cpu_settings" in e]
assert len(runtime_events) == 3
for event in runtime_events:
    settings = event["runtime_cpu_settings"]
    assert settings["intraop_threads"] == settings["interop_threads"] == 1
reapply = next(e for e in events if e["event"] == "effective_cpu_threads_reapplied_after_Trainer_constructor")
assert reapply["constructor_intraop_threads"] == 8
capture_runtime = next(e["runtime_cpu_settings"] for e in events if e["event"] == "save_model_entry_before_capture")
assert capture_runtime["deterministic_algorithms"] and capture_runtime["deterministic_warn_only"]
assert capture_runtime["mkldnn_enabled"] and capture_runtime["float32_matmul_precision"] == "highest"
assert receipt["threads"] == receipt["runtime_cpu_settings"]["intraop_threads"] == receipt["runtime_cpu_settings"]["interop_threads"] == 1
# BaseTrainer.train finally calls unset_deterministic; record capture and teardown separately.
assert receipt["runtime_cpu_settings"]["deterministic_algorithms"] is False
assert receipt["runtime_cpu_settings"]["deterministic_warn_only"] is False
assert next(e["runtime_cpu_settings"] for e in events if e["event"] == "on_train_start") == capture_runtime
assert sha(HERE / "result_review_v2.json") == "62ce160e533b19ebf2c75d75f5fa79cabef4fb0a25664e0541cdee9d699d0042"
runner_text = (DELIVERY / "run_training_v3.py").read_text(encoding="utf-8")
assert "framework_callbacks.add_integration_callbacks=disable_external_integration_registration" in runner_text
assert "require_cpu_settings()" in runner_text and "constructor_intraop_threads=torch.get_num_threads()" in runner_text
assert "dataset_factory=" not in runner_text  # Actual constructor takes the reviewed default Dataset branch.
assert receipt["status"] == "PASS_ACTUAL_SYNTHETIC_CPU_TRAINER_WITH_IGNORE_AND_PRE_HALF_FP32_CAPTURE_ONLY"
assert receipt["actual_train_call_returned"] and receipt["epochs"] == 1 and receipt["device"] == "cpu" and receipt["amp"] is False
assert receipt["ema_updates"] == 2 and len(receipt["optimizer_steps"]) == 2
for i, step in enumerate(receipt["optimizer_steps"]):
    assert step["step"] == i + 1 and step["ema_before"] == i and step["ema_after"] == i + 1
    assert step["gradient_absmax"] > 0 and step["first_trainable_parameter_changed_elements"] > 0 and step["actual_framework_optimizer_step_returned"]
step_events = [e for e in events if e["event"] == "actual_optimizer_step_completed"]
assert [{k: v for k, v in e.items() if k not in {"sequence", "event"}} for e in step_events] == receipt["optimizer_steps"]
mask_batches = receipt["actual_criterion_batches"]
assert len(mask_batches) == 2
for batch in mask_batches:
    assert batch["positive_rows_preserved"] and batch["assigned_positive_anchors"] > 0 and batch["ignored_negative_anchors"] > 0
    assert batch["kept_anchors"] + batch["ignored_negative_anchors"] == batch["batch_size"] * batch["num_anchors_per_image"]
    assert 0 < batch["classification_before_gain"] < batch["ordinary_classification_before_gain"]
assert [e["mask_receipt"] for e in events if e["event"] == "on_train_batch_end"] == mask_batches
changed = read(ATTEMPT / "changed_parameters.json")
assert sum(changed.values()) == receipt["changed_parameter_elements"] == 2209385
assert sum(v > 0 for v in changed.values()) == receipt["changed_parameter_tensors"] == 238
initial = read(ATTEMPT / "initial_model_state_description.json")
final = read(ATTEMPT / "final_model_state_description.json")
for key, count in changed.items():
    assert initial[key]["shape"] == final[key]["shape"] and initial[key]["dtype"] == final[key]["dtype"]
    assert (initial[key]["sha256"] != final[key]["sha256"]) == (count > 0)

capture_directory = ATTEMPT / "fp32_capture/epoch_0001"
capture = read(capture_directory / "capture_receipt.json")
architecture = read(capture_directory / "architecture.json")
assert hashlib.sha256(canonical(architecture).encode()).hexdigest() == capture["architecture_sha256"]
assert capture["sources"] == contract["runtime_sources"]
assert capture["training_contract_sha256"] == receipt["training_contract_sha256"]
assert capture["ema_updates_before"] == capture["ema_updates_after"] == 2
assert capture["live_state_before_sha256"] == capture["live_state_after_sha256"] and capture["state_and_update_counter_unchanged"]
assert capture["reconstruction"]["restricted_weights_only_load"] and capture["reconstruction"]["strict_load_state_dict"]
assert capture["reconstruction"]["output_values_dtypes_shapes_exact"]
assert capture["gpu_capture_qualified"] is False and capture["scientific_reference_accepted"] is False
assert capture["phase"] == "SAVE_MODEL_ENTRY_BEFORE_SUPER_FP16_SERIALIZATION"
entry = next(e for e in events if e["event"] == "save_model_entry_before_capture")
returned = next(e for e in events if e["event"] == "save_model_return_after_capture_and_framework_serialization")
assert entry["normal_last_exists"] is False and returned["normal_last_exists"] is True and returned["capture_receipts"] == 1
assert names.index("save_model_entry_before_capture") < names.index("save_model_return_after_capture_and_framework_serialization") < names.index("on_model_save")
assert next(e for e in receipt["integration_events"] if e["event"] == "framework_half_checkpoint_serialized")["epoch"] == 0

# Restricted state-only load is independent of model reconstruction/forward.
spec = importlib.util.spec_from_file_location("reviewed_fp32_state_format", ROOT / "scripts/gate1_fp32_reference_state.py")
state_format = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state_format)
state, metadata = state_format.load_snapshot(capture_directory / "state")
assert metadata["architecture_sha256"] == capture["architecture_sha256"] and metadata["training_contract_sha256"] == receipt["training_contract_sha256"]
assert hashlib.sha256(canonical(state_format.describe_state(state)).encode()).hexdigest() == capture["live_state_before_sha256"]
assert all(t.dtype == state_format.torch.float32 for t in state.values() if t.is_floating_point())
synthetic = state_format.torch.arange(1 * 3 * 64 * 64, dtype=state_format.torch.float32).remainder(251).div(250)
assert hashlib.sha256(synthetic.numpy().tobytes()).hexdigest() == capture["reconstruction"]["input"]["sha256"]

audit_directory = HERE / "synthetic_output_audit_v3"
assert sha(audit_directory / "audit_result.json") == "98ae7dca0d7d72154da9e62162084a6728b280b093bb064ca8ab37a30ef977e8"
verified.append(verify_manifest(audit_directory / "manifest.json", audit_directory))
independent_audit = read(audit_directory / "audit_result.json")
assert independent_audit["status"] == "PASS_ACTUAL_SYNTHETIC_CPU_RECORDS_AND_INDEPENDENT_NEW_FP32_RECONSTRUCTION_ONLY"
assert independent_audit["issues"] == [] and independent_audit["independent_analytic_cpu_output_exact"]
assert independent_audit["independent_replay_cpu_settings"] == independent_audit["effective_saved_cpu_settings"] == capture_runtime
assert independent_audit["independent_output_description"] == capture["reconstruction"]["output"]
assert independent_audit["outputs"]["sha256"] == sha(ATTEMPT / "manifest.json")
assert independent_audit["new_reference_state_exact_against_saved_description"] and len(state) == 499

result = {
    "schema": "rampq.gate1.actual_synthetic_trainer_result_review/v1",
    "status": "PASS_V3_THREAD_CONTROL_SAVED_STATE_AND_BOUND_INDEPENDENT_FORWARD_AUDIT_ONLY",
    "issues": [], "blocking_issue_count": 0, "reviewed_attempt": "attempt_03", "reviewed_runner": identity(DELIVERY / "run_training_v3.py"),
    "source_review": identity(HERE / "source_review.json"), "training_contract": identity(DELIVERY / "training_contract_v3.json"),
    "verified_manifests": verified, "actual_training_receipt": identity(ATTEMPT / "training_receipt.json"),
    "read_only_review": True, "model_forward_or_training_rerun": False,
    "state_only_restricted_load_and_tensor_hashes_independently_recomputed": True,
    "synthetic_input_recipe_sha_independently_recomputed": True,
    "architecture_output_equal_independently_replayed_by_other_agent_bound_audit": True,
    "train_images": 4, "training_val_images": 2, "explicit_member_bytes_verified": member_count,
    "image_identity_and_content_hash_intersections": [], "default_reviewed_dataset_factory_used": True,
    "observed_optimizer_steps": 2, "observed_ema_updates": 2,
    "parameter_change_receipt_sum_consistent": 2209385, "parameter_hash_change_sets_consistent": 238,
    "exact_changed_element_count_not_recomputed_from_initial_tensors": True,
    "positive_anchor_counts": [b["assigned_positive_anchors"] for b in mask_batches],
    "ignored_negative_anchor_counts": [b["ignored_negative_anchors"] for b in mask_batches],
    "positive_rows_preserved": True, "classification_reduced_by_ignore_mask": True,
    "capture_before_framework_half_serialization": "supported by bound actual MRO source plus entry/return events, dedicated capture receipt and serialization event",
    "external_integration_registration_disabled_before_actual_train": True,
    "only_base_callbacks_present_before_custom_hooks": True,
    "runtime_registration_function_intentionally_patched_in_process": True,
    "framework_on_disk_sources_unchanged": True,
    "network_traffic_absence_independently_measured": False,
    "attempt_01": "Preserved functional success; not selected for final external-integration-registration assurance or correct thread declaration",
    "attempt_02": "Superseded: receipt declared one thread, but Trainer constructor reset effective intraop threads to eight; independent fresh-forward audit exposed the resulting hash mismatch",
    "prior_v2_review_limit_disclosed": "The saved-metadata review did not detect the runtime thread-setting provenance error. Its original text remains unchanged and must not be used as final acceptance",
    "prior_v2_review_identity": identity(HERE / "result_review_v2.json"),
    "actual_constructor_intraop_threads_before_reapply": reapply["constructor_intraop_threads"],
    "observed_capture_runtime_cpu_settings": capture_runtime,
    "observed_after_train_teardown_runtime_cpu_settings": receipt["runtime_cpu_settings"],
    "step_thread_checks": "Pinned v3 optimizer_step asserts effective intraop/inter-op threads before each real step; full backend dictionaries are recorded at start and capture, not at every step",
    "fresh_independent_forward_audit": identity(audit_directory / "audit_result.json"),
    "fresh_independent_forward_audit_manifest": identity(audit_directory / "manifest.json"),
    "fresh_same_context_output_replay_verified_by_separate_bound_audit": True,
    "state_tensor_count": len(state),
    "final_numeric_output_acceptance_provided_by_this_review": False,

    "scientific_gate1_passed": False, "official_AP_qualified": False, "GPU_or_board_capture_qualified": False,
    "old_gate05_numeric_guard": "FAILED_UNCHANGED", "old_gate1_dtype_guard": "FAIL_FIXED_TOLERANCES",
    "scope_limits": ["CPU synthetic one epoch; no real data training or selection/holdout certification", "No optimizer/resume reconstruction",
                     "No arbitrary external-writer exclusion guarantee", "Cross-role split and default-factory obligations checked only for this bound fixture",
                     "This metadata reviewer independently recomputed state hashes and did not rerun model forward. A separate bound audit independently reproduced fresh outputs; gradients and element-change counts remain bounded runner observations rather than rerun training"]}
destination = HERE / "result_review_v3.json"
assert not destination.exists()
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"status": result["status"], "sha256": sha(destination), "verified_members": sum(v["listed_files_verified"] for v in verified)}))
