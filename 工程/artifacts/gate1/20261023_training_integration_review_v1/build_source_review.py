"""Independent read-only integration source review; no images/model run."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PINS = {
    "scripts/gate1_ignore_dataset.py": "f2ae752a0f8ad0549ac33e750071659de16f3b9a572f6b071ae005afcdb7ebe7",
    "tests/test_gate1_ignore_dataset.py": "90865122778dd45d1572c7ae7881f65d4159bfce845b1d55901510418fe6d971",
    "scripts/gate1_ignore_trainer.py": "f68d3163b7c8a0cff42432a470b4813a57f4dca737e558d6fbdabdf03a0659bf",
    "tests/test_gate1_ignore_trainer.py": "ccdb5e93370f8ecd17f35d0a7458924a7309ed46142a3f7ee8ba688be7d1fe06",
    "scripts/gate1_ignore_geometry.py": "4ecb3eedc4c5597f64e2290d73de4d6a4d40f4cd265e06c0dbb0ca03371dfd56",
    "scripts/gate1_ignore_negative_loss.py": "67fa0f7a171d26b1f8d2ebd922333f30bb97ffb94b523a58f6f784ed572b3c1f",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert all(sha(ROOT / name) == expected for name, expected in PINS.items())
verified_components = []
for rel in ("artifacts/gate1/20261023_ignore_dataset_integration_v1/manifest.json",
            "artifacts/gate1/20261023_ignore_trainer_integration_v1/manifest.json"):
    path = ROOT / rel
    manifest = json.loads(path.read_text())
    for name, expected in manifest["files"].items():
        member = path.parent / name
        assert member.stat().st_size == expected["bytes"] and sha(member) == expected["sha256"]
    for key in ("workspace_source_bindings", "workspace_sources", "external_sources"):
        for name, expected in manifest.get(key, {}).items():
            member = ROOT / name
            assert member.stat().st_size == expected["bytes"] and sha(member) == expected["sha256"]
    verified_components.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha(path),
                                "listed_members_verified": len(manifest["files"])})
review = {
    "schema": "rampq.gate1.training_integration_source_review/v1",
    "status": "PASS_SOURCE_REVIEW_FOR_FROZEN_SYNTHETIC_CPU_PROTOCOL_WITH_SCOPE_LIMITS",
    "reviewer": "fp32_live_capture_agent_independent_of_dataset_and_trainer_authors",
    "mutations_to_reviewed_sources": False,
    "blocking_issue_count_for_current_protocol": 0,
    "scope": "Source review plus component-manifest integrity only; no new model or Dataset execution by this reviewer",
    "sources": {name: {"bytes": (ROOT / name).stat().st_size, "sha256": expected} for name, expected in PINS.items()},
    "component_manifests": verified_components,
    "findings": [
        {"area": "actual_framework_lifecycle", "status": "SOURCE_PATH_SUPPORTED_PENDING_ROOT_EXECUTION",
         "evidence": "setup_model/get_model use only supplied cfg dict and reject weights; get_dataset avoids framework dataset scanning; default build_dataset binds explicit role/hash; get_validator/validate/final_eval declare SMOKE_NO_TASK_AP; on_train_start has initialized CPU EMA; save_model precedes framework deepcopy(ema).half().",
         "limit": "The FP32 mixin runs before RestrictedIgnoreTrainer.save_model. That trainer's save_model_before_fp32_callback event refers to its optional direct callback, and is not evidence of MRO capture order. Root must observe actual capture and framework serialization independently."},
        {"area": "byte_identity_and_single_role_membership", "status": "SOURCE_PATH_SUPPORTED",
         "evidence": "Manifest SHA checked at construction and sample access; each PNG, positive-label file and JSONL sidecar is read as bound bytes and hashed before parse/decode; actual decoded dimensions agree with manifest. Complete entry role/path metadata is validated before member content reads; entry train/ prefix alone does not grant membership."},
        {"area": "same_geometric_parameters", "status": "SOURCE_PATH_SUPPORTED",
         "evidence": "Image, positive boxes and ignore boxes share actual ceil loader-resize axes, explicit LetterBox ratio/pad and the same deterministic per-image/epoch horizontal and vertical flip decisions. Complex transforms and multiscale are disabled. This is a new restricted engineering policy, not a claim to reproduce the ordinary training augmentation distribution."},
        {"area": "TAL_box_DFL_and_normalization", "status": "SOURCE_MATCHES_PINNED_FRAMEWORK_PATH",
         "evidence": "One TAL call, detached classification/decoded inputs, unchanged target-score sum denominator and unchanged box/DFL call arguments/gains. Inherited loss() retains batch multiplier. Only classification elementwise BCE is masked at unassigned anchor centers in ignore geometry; every channel at a TAL-positive anchor is preserved."},
        {"area": "empty_gt_and_class_weights", "status": "SOURCE_PATH_SUPPORTED_WITH_COMPONENT_TEST_EVIDENCE",
         "evidence": "Empty-GT float fg_mask is converted to bool only for strict masking helper while the original assignment and empty regression graph branch remain. class_weights are validated finite/detached/nonnegative/FP32/device and retain the pinned normalization. Component tests exercise empty/nonempty GT, weighted/unweighted empty-ignore bit-exact loss and gradients, and all-ignore empty-GT zero loss."},
        {"area": "actual_optimizer_and_EMA_updates", "status": "PENDING_ROOT_EXECUTION",
         "evidence": "epochs=1, AMP=False, CPU, batch=nbs and warmup_epochs=0 permit one optimizer/EMA update per batch in pinned BaseTrainer. Root must retain observed positive update counts and actual loss/backward/step evidence; source feasibility alone is not completed training."},
    ],
    "issues": [
        {"severity": "medium", "id": "CROSS_ROLE_DISJOINTNESS_IS_CONTROLLER_OBLIGATION", "blocking_for_current_frozen_protocol": False,
         "description": "Each Dataset enforces one role manifest, but Trainer does not itself compare train/training_val image IDs or image SHA sets across the two manifests. Both single-role manifests could individually be valid yet overlap.",
         "required_current_run_condition": "Root preflight must verify and bind zero intersection for both explicit image IDs and image-content hashes in its actual generated six-image fixture.",
         "future_requirement": "Before real-data or independent-validation claims, bind and enforce split-disjointness in the controller's frozen protocol rather than infer it from prefixes."},
        {"severity": "medium", "id": "CUSTOM_FACTORY_IS_TRUSTED_EXTENSION", "blocking_for_current_frozen_protocol": False,
         "description": "Optional dataset_factory can return an arbitrary object without the default RestrictedIgnoreDataset hash/membership checks being revalidated by Trainer.",
         "required_current_run_condition": "Use dataset_factory=None/default reviewed Dataset. If instrumentation supplies a factory, independently verify its returned Dataset identity and bound manifests.",
         "future_requirement": "Do not describe arbitrary factory use as a universally fail-closed data boundary."},
        {"severity": "info", "id": "SYNTHETIC_SINGLE_EPOCH_ONLY", "blocking_for_current_frozen_protocol": False,
         "description": "Dataset epoch flips can be set manually; Trainer currently fixes one epoch and does not promise multi-epoch set_epoch synchronization. Dataset policy rejects real data. Loss observed_batches and ordinary-BCE diagnostics are engineering trace features, not official AP."},
    ],
    "required_root_acceptance_evidence": ["frozen source/contract identities unchanged", "six synthetic images and train/val ID plus hash disjointness", "default reviewed dataset factory",
                                          "two observed nonzero SGD/EMA updates for four training images at batch two", "finite actual losses and preserved positives plus ignored backgrounds",
                                          "capture completes before framework half serialization under actual MRO", "FP32 state and synthetic CPU reconstruction exact", "no real data, selection, holdout, old weights or GPU training"],
    "scientific_gate1_accepted": False,
    "official_AP_or_deployment_proven": False,
    "root_actual_trainer_result_reviewed": False,
}
destination = HERE / "source_review.json"
assert not destination.exists()
destination.write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"status": review["status"], "review_sha256": sha(destination), "blockers": 0}))
