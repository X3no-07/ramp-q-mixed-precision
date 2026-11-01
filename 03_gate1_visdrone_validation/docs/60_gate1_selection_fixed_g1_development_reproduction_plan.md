# Gate1: Fixed G1 Development Reproduction Execution Recommendation



**It is recommended to proceed with only the reproduction of the fixed G1 configuration on Selection, maintaining the original 548 members and their order.** The current scheme status is `RECOMMENDED_PENDING_SELECTION_CONTENT_ACCESS`, and Selection remains at `METADATA_ONLY`; this report does not open any content, perform inference, or alter any previously sealed contracts. Holdout remains at `SEALED`.

Report 59 has completed consistency validation of FP32 native scores for 414 images and independent saved audit verification. This phase has been closed out, and the scope of subsequent work is strictly limited. It uses the existing model, four precision configurations, and three calibration seeds, without new module scanning or new combinations; it does not assert superiority of random/standard controls, real INT8, or independent scientific validation.

# # Fixed Scope

| Item | This Recommendation |
|---|---|
| Model Identity | YOLO11n formal round 30 of FP32 EMA; SHA256 `4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27` |
| Model Candidate | 1 fixed weights, no training or weights selected via Selection |
| Precision Configuration | F: FP32; Q: Full Convolutional floating-point W8A8 QDQ; D: Only G1 QDQ; R: Full QDQ with G1 restored to FP32 |
| Calibration range | Three existing seeds: 2026102201, 2026102202, 2026102203, each with 64 calibration_pool images and label access set to 0; each seed is recalibrated once, and the three quantization configurations within the same seed share the calibration range |
| Selection members | Original 548 members, original order; list file SHA256 `2e1046a50278ad979007b6cceebffe5c07236d4dcf84b743b970b98ddb11f2bf`; joint order of ID and record image SHA256 `3caefdadd87d1925be31e3ef43f1d0266c8033ff6a1b2e7d0ebd85f5986487ec` |
| Input and QDQ semantics | 640,batch1,AMP disabled, CUDA determinism, TF32 disabled; retaining unfused BN, 88 Conv grouping, activation per-tensor and weight per-output-channel symmetric QDQ, convolution actual operation is FP32 |
| Original post-processing | CPU FP32 NMS: conf > 0.001, IoU0.7, max_det1000; coordinate restoration and ignore rules inherit from the frozen source; target matching criterion: conf ≥ 0.25, same category, IoU ≥ 0.5 |
| Maximum attempt limit | Suggested limit of 7200 seconds; abnormal states are saved on-site, no automatic retries, batch changes, or range expansion; prior to formal execution, the adapted code, complete source, and machine contract must be frozen |
| Holdout / Device | Holdout is not accessed in this run; no Orin, TensorRT, or integer kernel execution is performed |

The four configurations here are predefined interventions on the same model; three calibration ranges generate nine quantization configuration instances. FP32 is computed and saved only once, shared across three seed references. The Selection does not have an existing formal candidate array for the old FP32; this approach does not guarantee equivalence to a non-existent Selection history reference.

# # Execution order and budget

After content is made open, new execution source code and machine contract must be frozen first; authorization fields must be filled based on explicit human access decisions, not by script-driven auto-filling.

1. Verify the identity of all sources and available disk space. First, run FP32 on 109 debug images using the new adaptation path, strictly validate the original sealed complete arrays; if any element differs, halt immediately—Selection content is not read at this stage.
2. Validate the record SHA, original label structure, fixed categories, and ignore semantics of the 548 Selection images, as well as consistency of repeated identities in the following section; members are not filtered based on detection results.
3. Save a single complete FP32 output for the 548 images. Each seed independently performs 64-image calibration to establish a separate range; sequentially execute: QDQ kernel validation (disabled), three quantization configurations (Q/D/R), QDQ disabled, and original FP32 replay. Complete outputs, state fingerprints, and input identities are all saved or recorded in a form that allows for reproducibility.
4. Use the saved outputs to perform fixed post-processing, ignore-aware target matching, and descriptive statistics; all three sub-results and adverse outcomes are fully reported. The output arrays are independently verified before results are released.

| Forward pass portion | Count |
|---|---:|
| Debugging of the original FP32 has been opened for strict verification | 109 |
| Selection shares FP32 | 548 |
| Per seed calibration | 64 |
| Verification before disabling QDQ per seed | 548 |
| Three quantization configurations per seed | 1644 |
| Verification after disabling QDQ per seed | 548 |
| Replay of original FP32 per seed | 548 |
| Aggregation of Selection and calibration components | 10604 |
| Forward pass budget over all real images | **10713** |

The forward pass for stride-based synthetic data generated by the model constructor is provided as a separate source and does not count as a real image forward pass. The independent original arrays are estimated at 5480, with logical configuration outputs at 6576. Shared FP32 is referenced by explicit identity and is not redundantly counted or fabricated into three independent baselines. All 6576 full-element differences and post-processing results are preserved in the existing format.

AP uses a verified, officially sourced Python image with fixed origin, and computes only from the saved prediction and ground truth rows of this run. The shared FP32 is used once, and each of the nine quantization instances is evaluated once, resulting in a total of 10 scoring calls, with seven instances of AP/AR each saved; it must not be stated that each submission has undergone independent native Octave comparison. This approach does not initiate a native scoring matrix, nor does it establish a new scientific PASS threshold on Selection.

# # Policy on Duplicate Members

Existing metadata includes 548 members, 547 image records by SHA, and 75 risk-isolation components. Currently, no re-validation of Selection content has been performed. It is recommended to retain the original 548 members and their order. The AP development should continue using the original member set, without silently changing to 547 or selecting representatives based on results.

The first access to content must verify each member's current image bytes against the recorded SHA, then check the physical row-by-row consistency of actual images and original annotations, including class, score, and ignore fields. If identity changes, annotation conflicts occur, or dependency groups cross existing component isolation boundaries, the scene must be preserved and the issue must be re-explained; no automatic annotation selection, image deletion, or rights modification is allowed.

The main description at the object level continues to be recorded using the original member set, reporting new missed detections, recovered targets, side effects, net recovery, bidirectional same-target cases, and all hit/false positive changes. The distribution of duplicate identities and components is also disclosed. Duplicate members and three sub-duplicate targets do not increase independent sample counts, and no significance testing is performed treating them as independent samples. Any specification of deduplication or weighting results of content identity must be pre-frozen and explicitly stated if needed in the future; such modifications are not silently introduced in this scheme.

75 components are risk-isolation proxies, not proven 75 independent real-world scenarios; the calibration seed images are non-overlapping but share components and are not treated as three independent datasets. The best seed is not selected, and no image augmentation or configuration expansion is performed.

# # Numerical and Termination Criteria

The original complete 8400 bounding box threshold is maintained: box `0.5+0.005×|FP32_reference|`, score `0.002+0.02×|FP32_reference|`. All numerical differences between quantization and FP32 are preserved and reported as-is, without modification via confidence levels or NMS subsets. The existing records from nine debug quantization branches, each with 0/109 passes, and historical data `FAIL_FIXED_TOLERANCES` and `FAILED_UNCHANGED` are preserved.

Inconsistencies in the original debug reference, non-identity when QDQ is disabled, changes in source or member identity, non-finite values, abnormal output shapes/domains, input or state contamination, label or identity conflicts, budget overruns, or disk space insufficiency will trigger termination and scene preservation. Numerical differences between quantization intervention and FP32 thresholds are still recorded as failures. This round only provides a clear description of QDQ development and does not claim numerical qualification, cross-backend equivalence, or deployment pass based on task completion or AP scores.

# # Current Executable Status and Required Access Decision

This round only completes the scheme and pure synthetic budget/status validation; actual model forward pass, calibration, scoring, and selection image/label reading are all 0. The previous 11 item pending template, duplicate policy appendix, report 59 and v20 remain unchanged; this scheme is a new recommendation and does not reverse historical preregistration.

Technical parameters and termination conditions have been defined within the above-limited scope. Before formal execution, only the following needs to be clarified: **Are the original Selection's 548 images and original labels being opened for access, to be used for fixed G1 development inference, preservation, and scoring under this scheme, while maintaining the holdout seal?** This constitutes a change to the existing `Selection=METADATA_ONLY` boundary; mere preparation materials cannot be used to justify content access.

The scheme, pure synthetic validation, and independent scope review are preserved in `../artifacts/gate1/20261024_selection_fixed_G1_proposal_v1`. No execution tasks have been run in this phase; the previously deleted scheduled monitoring remains stopped.
