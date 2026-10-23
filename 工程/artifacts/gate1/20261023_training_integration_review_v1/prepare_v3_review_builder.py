"""Derive a new review builder; keep the v2 review and builder unchanged."""
from pathlib import Path

here = Path(__file__).resolve().parent
text = (here / "build_result_review.py").read_text(encoding="utf-8")
text = text.replace("Trainer v2.", "Trainer v3, with observed CPU thread controls.")
text = text.replace("attempt_02", "attempt_03").replace("_v2", "_v3").replace("2209411", "2209385")
text = text.replace('assert names[:3] == ["external_integration_registration_disabled", "base_callbacks_verified_before_custom_hooks", "actual_DetectionTrainer_train_call_enter"]',
                    'assert names[:4] == ["external_integration_registration_disabled", "effective_cpu_threads_reapplied_after_Trainer_constructor", "base_callbacks_verified_before_custom_hooks", "actual_DetectionTrainer_train_call_enter"]')
marker = 'runner_text = (DELIVERY / "run_training_v3.py").read_text(encoding="utf-8")'
addition = '''runtime_events = [e for e in events if "runtime_cpu_settings" in e]
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
'''
assert marker in text
text = text.replace(marker, addition + marker)
text = text.replace('assert "dataset_factory=" not in runner_text',
                    'assert "require_cpu_settings()" in runner_text and "constructor_intraop_threads=torch.get_num_threads()" in runner_text\nassert "dataset_factory=" not in runner_text')
text = text.replace('"PASS_BOUND_SAVED_EVIDENCE_AND_ACTUAL_CPU_TRAINER_INTEGRATION_ONLY"',
                    '"PASS_V3_THREAD_OBSERVATIONS_AND_SAVED_STATE_ONLY_FRESH_FORWARD_AUDIT_PENDING"')
text = text.replace('"attempt_01": "Preserved functional success; not selected for final external-integration-registration assurance",',
'''"attempt_01": "Preserved functional success; not selected for final external-integration-registration assurance or correct thread declaration",
    "attempt_02": "Superseded: receipt declared one thread, but Trainer constructor reset effective intraop threads to eight; independent fresh-forward audit exposed the resulting hash mismatch",
    "prior_v2_review_limit_disclosed": "The saved-metadata review did not detect the runtime thread-setting provenance error. Its original text remains unchanged and must not be used as final acceptance",
    "prior_v2_review_identity": identity(HERE / "result_review_v2.json"),
    "actual_constructor_intraop_threads_before_reapply": reapply["constructor_intraop_threads"],
    "observed_capture_runtime_cpu_settings": capture_runtime,
    "observed_after_train_teardown_runtime_cpu_settings": receipt["runtime_cpu_settings"],
    "step_thread_checks": "Pinned v3 optimizer_step asserts effective intraop/inter-op threads before each real step; full backend dictionaries are recorded at start and capture, not at every step",
    "fresh_independent_forward_audit": "PENDING_ROOT_AGGREGATION_DO_NOT_CLAIM_REPRODUCED_OUTPUT_FROM_THIS_METADATA_REVIEW",
    "final_numeric_output_acceptance_provided_by_this_review": False,
''')
destination = here / "build_result_review_v3.py"
assert not destination.exists()
destination.write_text(text, encoding="utf-8")
print(destination.name)
