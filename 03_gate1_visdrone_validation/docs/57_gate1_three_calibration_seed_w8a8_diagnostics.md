# Gate1: W8A8 Three Calibration Seed Development Diagnosis



The current phase has completed fixed 109 debug images, 3 existing calibration seeds, and 4 configuration variants of floating-point QDQ simulation, all independently validated for result consistency. All three seeds observed recovery of G1 object-level targets, but all quantization configurations failed the original full candidate box numerical threshold; recovery of G1 did not reach FP32 total detections, and false positives increased. The conclusions are limited to development diagnostics and cannot be stated as real INT8 success, deployment success, or independent scientific acceptance.

# # 1. Freeze Scope and Execution Completion

- Candidates are fixed at the 30th round of FP32 EMA from formal training; SHA256 is `4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27`. The fact of training completion remains valid.
- The new contract SHA256 is `78e0c4b14cc7d1d6fd3f77ef96ead16aa3538f11108c49785e2feca0700c7103`, frozen prior to real data forward pass. All 109 images are existing debug members, executed in fixed lexicographical order, from 6 isolated components; the same batch of 9,059 evaluable targets is used three times, and thus cannot be treated as three independent test sets.
- The calibration seed is fixed at `2026102201/2026102202/2026102203`, with 64 calibration_pool images per seed, totaling 192 distinct members; calibration labels are not read. For each seed, the calibration range is reset to zero and the same range is used to execute all configurations for that seed.
All 109 new FP32 outputs are strictly element-wise equal to the previously stored array before calibration begins. The actual forward pass count is `109 + 3 × (64 + 109 + 327 + 109 + 109) = 2263`, with a single-pass upper limit of 3, 600 seconds, and only 1 attempts, no automatic retries. Execution occurred from Beijing Time 2026-10-24 10:42:20 to 10:45:44, with a launcher return code of 0, and an actual completion acknowledgment is present.
- The model retains 81 BN layers that are not fused; 88 Conv layers use fixed grouping. W8A8 refers only to the QDQ rules applied to weights and activations in this simulation; convolution and other arithmetic operations remain in floating-point. No actual integer kernel, latency, power consumption, or board-level performance measurements are conducted.

The following configurations are fixed: `F=fp32` corresponds to original FP32; `Q=q_all` applies QDQ to all Conv layers; `D=q_only:G1` applies QDQ only to the 6 Conv layers in G1; `R=r_only:G1` restores G1 to FP32 while keeping all other Conv layers in QDQ. Activation calibration uses symmetric absmax per tensor, and weight quantization uses symmetric per-output-channel; all other operations remain in FP32, with no change to BN fusion policy.

# # 2. Object-level results across all seeds

Object-level diagnosis uses a fixed ignore-aware matching criterion: confidence `≥0.25`, same category, IoU `≥0.5`, with target identity defined by the physical row of the original annotation in the image. This is not the official AP. Each cell in the table shows 'hits / false positives'; each row has 9,059 evaluable targets.

| calibration seed | F:FP32 | Q: Full Conv QDQ | D: QDQ only on G1 | R: Recovery of G1 |
|---|---:|---:|---:|---:|
| 2026102201 | 3310 / 2283 | 3246 / 2489 | 3262 / 2205 | 3291 / 2492 |
| 2026102202 | 3310 / 2283 | 3232 / 2361 | 3274 / 2232 | 3248 / 2451 |
| 2026102203 | 3310 / 2283 | 3266 / 2563 | 3274 / 2232 | 3302 / 2598 |

Let F/Q/D/R denote the sets of detected targets for each configuration. Additional misses are `A=F\Q`; recoveries are `A∩R`; collateral misses adjacent to stable targets are `(F∩Q)\R`; net recovery is defined as recovery_count minus the number of collateral misses adjacent to stable targets; bidirectional same-target recovery is `(A∩R)∩(F\D)`. All seeds are listed, without result filtering.

| calibration seed | Q additional misses A | R recovery | collateral misses adjacent to stable targets | net recovery | same-target bidirectional agreement | recovery / A |
|---|---:|---:|---:|---:|---:|---:|
| 2026102201 | 204 | 125 | 69 | +56 | 44 | 61.27% |
| 2026102202 | 219 | 118 | 82 | +36 | 43 | 53.88% |
| 2026102203 | 186 | 114 | 71 | +43 | 44 | 61.29% |

Net recovery only counts targets that were already detected in the original FP32 configuration, and does not equal the difference between total detections across two configurations. The actual total detection increment of R relative to Q is `+45/+16/+36`; the total detection drop of Q relative to F is `64/78/44`. R still misses `19/62/8` targets relative to F, and false positives increase by `209/168/315`. Therefore, local recovery alone cannot be used to claim overall improvement in detection quality.

The intersection of additional misses across three seeds is 51 targets, and the union is 383 targets. Among the 51 targets with common additional misses, R recovers `23/19/24` targets. The intersection of targets recovered across all three seeds is 9, and the intersection of targets satisfying bidirectional description across all three seeds is 5. These represent repeated observations of the same targets under different calibration ranges. Seeds × targets cannot be merged into independent samples, and thus G1's general causal advantage cannot be declared based on this observation.

# # 3. Original candidate box threshold: all quantization configurations fail

Each image retains all 8,400 candidate boxes with their `[1,14,8400]` FP32 decoded arrays. Numerical checks do not first subset by confidence or NMS; the original tolerance remains at box `0.5 + 0.005×|FP32|` and score `0.002 + 0.02×|FP32|`. For each configuration and each seed, a total of 3,662,400 box coordinate elements and 9,156,000 score elements are checked. The maximum box error is measured in pixels within a 640-input coordinate system, and score error is absolute value.

| seed | Configuration | number of bounding box outliers | number of score outliers | maximum bounding box error | maximum score error |
|---|---|---:|---:|---:|---:|
| 2026102201 | Q | 1,749,248 | 111,873 | 475.596931 | 0.632207 |
| 2026102201 | D | 1,335,381 | 77,419 | 408.806610 | 0.435593 |
| 2026102201 | R | 1,692,865 | 98,875 | 380.776245 | 0.537998 |
| 2026102202 | Q | 1,872,619 | 111,272 | 470.219391 | 0.687356 |
| 2026102202 | D | 1,465,045 | 76,741 | 407.206451 | 0.451499 |
| 2026102202 | R | 1,705,587 | 99,390 | 440.717728 | 0.474063 |
| 2026102203 | Q | 1,759,990 | 109,921 | 421.481461 | 0.760213 |
| 2026102203 | D | 1,302,063 | 76,701 | 384.061462 | 0.453660 |
| 2026102203 | R | 1,585,606 | 96,850 | 413.734512 | 0.652794 |

Q/D/R passes `0/109` images per seed under the original threshold. The range of maximum bounding box error across the nine-arm experiments is `380.776245–475.596931` pixels, and the range of maximum score error is `0.435593–0.760213`; the highest error-to-tolerance ratios are approximately bounding box `784.5441` and score `296.9325`. For each seed, 109 images have strictly equal F-values, with zero error and zero outlier count.

Saved results are consistent, intervention paths are executable as per contract, yet floating-point qualification still fails. This report does not relax thresholds, nor does it convert failures into passes by citing target-level hits, NMS subsets, or low-confidence scores. Historical `FAIL_FIXED_TOLERANCES` and `FAILED_UNCHANGED` are preserved unchanged.

# # 4. Actual Range of Independent Result Verification

The independent audit result reflects the true state as `PASS_INDEPENDENT_SAVED_109_DEBUG_THREE_SEED_ALL_ANCHOR_NMS_TARGET_REPLAY_ONLY`, bound to the same contract SHA. The audit before and after verification covers 5, 040 members in the checklist hashes, confirming 109 shared FP32 arrays strictly consistent with the previous sealed reference; reading 1, 090 independent original arrays, inspecting 9, 156, 000 bounding boxes, and 128, 184, 000 elements' shapes, finite values, and geometric domains; precisely replaying 1, 308 full-element FP64 differences, logical post-processing, and target outputs, and verifying 1, 936 progress records against 2, 263 forward passes.

CPU post-processing is independently constructed by the auditor, using the same frozen torchvision NMS kernel. Target matching re-invokes a hash-fixed project matcher, and target set statistics are independently reconstructed. The audit does not re-run the model or calibration. 981 closure summaries of QDQ/original model replay match shared saved arrays, and the saved pre- and post-state fingerprints are consistent; however, the un-saved state tensors and replayed tensors are not independently reconstructed, and thus runtime tensor audit is not claimed to be complete.

This evaluation extends under existing technical pass criteria without basing expansion on recovery direction. Cross-seed summaries are compiled after results are finalized; during the preparation of the summarizer, the header of seed1 file was reviewed to confirm JSON structure, but no changes were made to experiments, seeds, metrics, or scope. Therefore, this is not an independent pre-registration validation of unseen results.

# # 5. Current Conclusion and Evidence Entry

Existing development evidence indicates that G1 intervention is associated with some object-level changes, and all three calibration seeds exhibit local recovery; the effect varies with calibration, accompanied by increased side effects and false positives. No randomized or traditional mixed-precision comparisons have been conducted, and true INT8 performance gains, speed improvements, or board-level acceptability have not been established. Development samples from 6 components do not provide independent scientific validation; official AP results are not available in this phase.

Selection uses only existing metadata; Holdout remains sealed and is not read for images, labels, or detection results. Formal training is complete; the standalone FP32 original official training_val comparison must be fully native-filtered, aggregated over 414 images, and compared across seven metrics before any conclusion can be drawn. This report does not prematurely claim native consistency is achieved, nor does it reframe an incomplete comparison as a training failure.

- [Frozen Contract](../../docs/DATA_AND_ARTIFACTS.md) and [Complete Execution Receipt](../../docs/DATA_AND_ARTIFACTS.md).
- [Independent Result Audit](../../docs/DATA_AND_ARTIFACTS.md).
- [Per-seed Description Summary](../../docs/DATA_AND_ARTIFACTS.md), [Cross-seed Target Set](../../docs/DATA_AND_ARTIFACTS.md), and [Complete Candidate Box Numerical Summary](../../docs/DATA_AND_ARTIFACTS.md).

Original per-image results, full-element differences, target identity sets, calibration ranges, and state fingerprints for each seed are stored within the corresponding seed directory of the complete execution receipt. Previous reports and lists are preserved in their original form.
