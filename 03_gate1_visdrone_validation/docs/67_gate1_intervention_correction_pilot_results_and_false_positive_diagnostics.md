# Gate1 Precision Intervention Correction Pilot: Completion Results and False Positive Diagnosis



Date: 2026-10-26. Conclusion: **This version of correction training fails to achieve the goal of preserving multiple detections while reducing false positives, and thus does not support expanding training or claiming the method is effective.**

All three arms completed 512 training images, 3 epochs, and 96 optimizer updates each, using the fixed final epoch 3. The worker terminated with exit code 0. Continuation after the coordinate fix took 670.6 seconds, approximately 11 minutes 11 seconds, excluding the references, calibration, and baseline evaluation completed in the first version and reused here. Training completion and result validity are separate conclusions.

# # Development Results

All of the following are from the pre-fixed **128-image subset of training_val**, not the previous 414-image total score. AP is computed using the officially validated Python image; no additional native official scoring script was run for this submission. Detection and false positives are counts derived from per-object matching at a fixed confidence level of 0.25 and IoU of 0.5, and do not equate to the internal counts used in the official AP metric.

| Branch | AP50–95(%) | Detected true targets | false positives |
|---|---:|---:|---:|
| Original FP32 reference | 12.507049 | 3463 | 2414 |
| Original max-value calibration Q | 11.593180 | 3382 | 2591 |
| Original max-value calibration recovery G1 | 11.865606 | 3446 | 2679 |
| MSE-range calibration Q | 11.971447 | 3449 | 2597 |
| MSE-range calibration recovery G1, common starting point for all three arms | 11.986118 | 3469 | 2603 |
| Standard QAT | 12.054024 | 3462 | 2621 |
| Standard hard example training | 12.013040 | 3438 | 2708 |
| Precision intervention correction training | **11.823727**|**3539**|**3120** |

Relative to the common starting point, the correction group detects 70 additional targets and increases false positives by 517, resulting in a drop of **0.162391 percentage points** in AP. Relative to standard QAT, it detects 77 more targets and increases false positives by 499, leading to a drop of 0.230297 percentage points in AP; relative to the standard hard example group, it shows +101 detections, +412 false positives, and a decrease of −0.189313 percentage points in AP. None of these perspectives demonstrate the required overall advantage.

The three pre-fixed confidence levels are retained; no more favorable thresholds were selected.

| threshold | Detection at baseline / False positives | Detection by correction group / False positives | Change in detection / Change in false positives |
|---|---:|---:|---:|
| 0.20 | 3708 / 3504 | 3750 / 4124 | +42 / +620 |
| 0.25 | 3469 / 2603 | 3539 / 3120 | +70 / +517 |
| 0.30 | 3227 / 1943 | 3313 / 2371 | +86 / +428 |

Additionally, under MSE calibration, the AP gain of G1 relative to Q is only 0.014671 percentage points; under maximum value calibration, it is 0.272426 percentage points. This suggests that the general calibration baseline significantly influences the assessment of 'how valuable the recovery module is'. It is not sufficient to compare contributions against a weak calibration baseline.

# # Cross-check of results for the same target

Within the common baseline, G1 recovers 108 specific targets relative to Q. Among them, standard QAT preserves 56, the hard example group preserves 50, and the correction group preserves 70. The correction group adds 223 detections, loses 153 original detections, resulting in a net gain of 70.

Thus, there is a descriptive phenomenon of 'some originally recovered targets being preserved more', but this coexists with a significant increase in false positives. Therefore, we cannot claim that the new method is effective based solely on this. All corresponding target IDs and per-image gains and losses have been saved, without modifying the model's forward pass or re-scoring.

The training supervision is set to 601 targets affected by intervention and 120 newly added safe background counterexamples. The number of matched categories and scales, as well as confidence interval matches between the hard example group and the correction group; there are 59 positive matches and 81 negative matches. This refers to the pre-defined supervision budget. The dynamic positive sample allocation and actual wall-clock cost during model training may still differ, and we do not claim that the measured computational cost is exactly equal.

# # Geometric diagnosis of net increase in false positives

This is a post-hoc descriptive diagnosis added after observing training results. It only reads the saved false positive annotations and the authorized 128 original annotations. It does not alter the original scores, candidates, thresholds, or acceptance criteria. Using cropped bounding boxes and their IoU with the valid ground truth, false positives are categorized in the following priority order; this is not an official error type nor a causal proof.

| False positive geometric type | Common baseline | Correction group | Net change |
|---|---:|---:|---:|
| Same-target IoU ≥ 0.5: overlapping or duplicated candidates | 477 | 654 | +177 |
| No overlap of same-target type, but other-category IoU ≥ 0.5 | 822 | 894 | +72 |
| Maximum IoU in [0.1, 0.5): localization candidates from nearby targets | 990 | 1238 | +248 |
| Maximum IoU < 0.1: residual cases | 314 | 334 | +20 |
| Total | 2603 | 3120 | +517 |

The net increase in false positives is primarily concentrated in erroneous boxes around the true targets. The residual cases with low overlap may still include regions ignored by labeling, missed targets, etc., and cannot be considered entirely background; the classification difference does not imply that each newly generated false positive box has been individually tracked.

This version only treats safe background false positives as negative cases, deliberately avoiding the hasty classification of localization errors or misclassifications near true targets as background. This protective boundary is reasonable but does not resolve these errors. The next step worth investigating is: how to design correction supervision for localization quality, redundant responses, and misclassification, while preserving standard quality-aware training and hard example methods as controls. The current results do not prove that any new loss is effective. Increasing negative examples, more training epochs, or switching to larger GPUs cannot replace rigorous validation.

# # Integrity and Numerical Boundaries

- 10 pre-checks passed; coordinate bridging between original images and training annotations was verified for 512 images.
A new attempt: 6014 forward passes within budget; first version 708 original forward passes recorded, totaling 6722, of which 2 failed and were not saved. Each of the three candidates has 3 saved reconstruction forward passes listed separately, totaling 9.
- Verified 1438 new saved members, 680 reused/legacy source bindings, and 2368 authorized image/annotation/sidecar input members.
- 9 models in the per-epoch model files have floating-point states of FP32; the initial states across the three arms are strictly identical, with BN and quantization ranges held frozen, and weights are actually updated in each group.
- Independent CUDA reconstruction of the three final states exactly matches the in-memory outputs. CPU outputs are finite, but the respective CPU/CUDA maximum absolute differences are approximately 34.9083, 29.9385, and 42.0507. **Cross-device numerical consistency has not been certified.**
- The seven non-FP32 branches each re-evaluate the full 128 image set's complete 8400 candidate box thresholds from saved arrays, all achieving **0/128 passed**. The comparison of the trained model against the prior FP32 weights remains FAIL, without threshold modification, without substituting NMS subsets, and without treating saved integrity PASS as floating-point qualification PASS.
- Seven AP/AR aggregations were independently recalculated from saved arrays and are consistent; no full matching or model inference repetitions occurred.

All historical failures, first-version interface errors, and first-version frozen sources and outputs are preserved. The previous official 30-epoch training is still completed; the outcome of this new method development does not alter the judgment of prior training failures.

# # Stage Closure

This round is a pilot with a single training seed and limited epochs. 128 development images cover 37 file prefix proxies; proxies do not equate to independently validated scenarios. Selection images, annotations, and detection results were not loaded; holdout data remains sealed. Floating-point QDQ, labeled fine-tuning, official-scoring development benchmarks, independent scientific validation, and real INT8/board-end validation are strictly separated.

The current budget has been fully executed and sealed; no automatic extension of seeds, epochs, or cloud costs is applied. This local continuation took approximately 11 minutes, and computational power is not the primary bottleneck in determining the validity of this version. Stronger cloud GPUs can accelerate subsequent GPU training; the overall workflow includes CPU matching and scoring, and performance should be measured under identical data, configurations, and final outcomes.

The primary evidence resides in `../artifacts/gate1/20261026_intervention_training_pilot_v2`:

- `attempt_01/completion_receipt.json`, `process_exit_observation.json`: actual completion and termination records.
- `saved_delivery_audit.json`: verification of source, saved state, budget, and array aggregation.
- `machine_decision.json`: complete difference values and conclusion boundaries for each comparison object.
- `saved_target_diagnostics.json`: hit/miss outcomes for the same target across saved runs.
- `fp_diagnostic_contract.json`, `saved_false_positive_diagnostics.json`: post-hoc false positive grouping rules and per-bounding-box results.
- `delivery_validation.json`: final delivery seal check; entry status uses `../artifacts/gate1/latest_execution_state.json`.
