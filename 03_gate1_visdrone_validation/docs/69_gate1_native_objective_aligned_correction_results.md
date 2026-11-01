# Gate1 Native Training Objective Alignment Correction: Completion Results



2026-10-26, Beijing time.

**The new version reduces false positives while simultaneously decreasing true target detections; it does not achieve the joint objective of preserving more targets and reducing false positives.**

For the group of typical hard examples with matched intervention, no joint advantage in AP, detection, or false positive rates is observed. This round cannot claim that intervention information provides additional contribution based on this result.

# # Fixed Development Results

The following are fixed 128 images from the training_val subset, which cannot be conflated with the original 414-image total set. AP uses the official Python image with native consistency validated against real-world data; the new submission does not include re-running the native official program. Target counting is based on the project matcher's results at a fixed confidence level of 0.25 and IoU0.5, and is independent of the AP internal count.

| Branch | AP50–95(%) | Detected true targets | false positives |
|---|---:|---:|---:|
| Original FP32 reference | 12.507049 | 3463 | 2414 |
| Original max-value calibration Q | 11.593180 | 3382 | 2591 |
| Original max-value calibration recovery G1 | 11.865606 | 3446 | 2679 |
| MSE Calibration Q | 11.971447 | 3449 | 2597 |
| MSE Calibration Recovery G1, common starting point | 11.986118 | 3469 | 2603 |
| Existing Ordinary QAT | 12.054024 | 3462 | 2621 |
| Previous Version Ordinary Hard Examples | 12.013040 | 3438 | 2708 |
| Previous Version Intervention Correction | 11.823727 | 3539 | 3120 |
| New: Only improves localization | 12.039152 | 3461 | 2544 |
| New: Native target correction for ordinary hard examples | 12.024435 | 3218 | 2021 |
| New: Native target intervention correction | 11.908322 | 3269 | 2131 |

Each new arm trained on 512 images for 3 epochs and 96 optimizer updates, using the fixed final epoch 3 without selecting the best epoch or seed. Initialization, training members, batch order, optimizer, and range calibration are identical. Matching the number of additional supervision items does not establish identical measured FLOPs or wall-clock time.

# # Difference between the Correction Intervention Group and the Primary Control

| Comparison Object | AP Change (in percentage points) | Detection Change | False Positive Change |
|---|---:|---:|---:|
| MSE Calibration Recovery G1, common starting point | -0.077796 | -200 | -472 |
| Existing Ordinary QAT | -0.145702 | -193 | -490 |
| Previous Version Intervention Correction | +0.084595 | -270 | -989 |
| New: Only improves localization | -0.130830 | -192 | -413 |
| New: Native target correction for ordinary hard examples | -0.116113 | +51 | +110 |
| Original FP32 reference | -0.598727 | -194 | -283 |

All three pre-specified thresholds are retained and cannot be selectively chosen as the final winner:

| threshold | Baseline: Detection / False Positive | New Ordinary-Hard Examples: Detection / False Positive | New Intervention: Detection / False Positive |
|---|---:|---:|---:|
| 0.2 | 3708 / 3504 | 3492 / 2779 | 3541 / 2933 |
| 0.25 | 3469 / 2603 | 3218 / 2021 | 3269 / 2131 |
| 0.3 | 3227 / 1943 | 2969 / 1453 | 3032 / 1562 |

# # What Was Changed in This Round

The original version used an additional positive example classification with IoU-based supervision, which potentially conflicted with the detector's native TAL soft-target strategy. The new version removes this extra positive classification and instead only adds IoU localization loss for selected ground truths; the additional classification term for erroneous boxes retains the current TAL target from the same iteration, penalizing only the parts with scores above that target.

The active region introduces an extra gradient on logits that is non-zero only when p−t≥0, and zero when below the target, meaning the target does not participate in gradient computation. Testing confirms this does not directly boost detection scores; shared weights are still influenced by other losses, so this cannot guarantee that every output decreases after training. This mathematical property is not a novel scientific contribution.

The three new arms are localization ablation, ordinary hard example correction, and intervention correction. The two correction groups are matched by category, confidence level, and error geometric type; each image contains at most 8 error channels. The native ignore mask and positive sample priority are retained; localization errors, misclassification, or boxes currently assigned as foreground are not arbitrarily labeled as pure background.

Fixed 601 intervention ground truths; newly added erroneous channels total 882, distributed as follows:

| Training Error Type | Count |
|---|---:|
| Same-class Overlap | 315 |
| Cross-class Overlap | 154 |
| Low IoU with Nearby Objects | 280 |
| Low Overlap Residual | 133 |

59 erroneous channels overlap with GTs selected by intervention/hard examples, and 437 erroneous channels overlap with each other. The sample pool, types, and quantities are fixed before training, and all three groups' results are preserved.

# # Actual Supervision Usage Records

Count the cumulative occurrences of GT or channel across batches over three epochs, not the total number of distinct targets. Whether erroneous channels are active or are assigned positive labels by TAL varies with the model, and the pre-matched count cannot be equated to identical effective gradients at each step.

| Group | Number of Actual Assigned Localization GTs | Number of Retained Erroneous Channels | Number of Active Penalty Events | Number of Ignored Channels | Number of Erroneous Anchors Currently in Foreground |
|---|---:|---:|---:|---:|---:|
| New: Only improves localization | 1762 | 0 | 0 | 0 | 0 |
| New: Native target correction for ordinary hard examples | 1417 | 2631 | 2239 | 15 | 2023 |
| New: Native target intervention correction | 1760 | 2625 | 2144 | 21 | 1995 |

# # Post-hoc Diagnosis of the Same Target and False Positives in Geometry

Under the original MSE, G1 recovered 108 specific targets. The following table continues to track the same target ID; this is a descriptive result and does not replace overall AP or independent acceptance criteria.

| Group | Original G1 Recovered Target | Relative Startpoint New Hits | Relative Startpoint Lost Hits |
|---|---:|---:|---:|
| Existing Ordinary QAT | 56 | 143 | 150 |
| Previous Version Intervention Correction | 70 | 223 | 153 |
| New: Only improves localization | 63 | 160 | 168 |
| New: Native target correction for ordinary hard examples | 35 | 80 | 331 |
| New: Native target intervention correction | 44 | 94 | 294 |

The following follows the previously defined post-hoc geometric grouping: first, same-class IoU ≥ 0.5; then, other classes with IoU ≥ 0.5; next, maximum IoU in [0.1, 0.5); finally, the remainder with IoU < 0.1. Grouping is based only on existing false positive annotations, without modifying matches or AP. This grouping is not an official error classification or causal explanation; low overlap remainder does not imply all are purely background.

| Branch | Same-class Overlap | Cross-class Overlap | Localization Neighborhood | Low Overlap Residual |
|---|---:|---:|---:|---:|
| MSE Calibration Recovery G1, common starting point | 477 | 822 | 990 | 314 |
| Existing Ordinary QAT | 509 | 815 | 987 | 310 |
| Previous Version Intervention Correction | 654 | 894 | 1238 | 334 |
| New: Only improves localization | 488 | 785 | 987 | 284 |
| New: Native target correction for ordinary hard examples | 363 | 700 | 741 | 217 |
| New: Native target intervention correction | 395 | 707 | 799 | 230 |

# # Completion, Saving, and Numerical Qualification

Final state completion acknowledgment is complete; the actual process has exited, with stderr of 0 bytes; the main workflow took 635.0 seconds. The process exited after bounded waiting, and the numeric exit code was not directly captured, thus it is not claimed that exit0; on-site records are null. A total of 4992 forward passes on new images were conducted within budget, with independently saved reconstructions counted separately as 9; the forward passes on the new teacher model were 0. Reference, calibration, baseline, and old model inferences were all reused; the matching of saved teacher arrays was constructed for re-computing new supervision and separately saved.

10 checkpoints passed pre-check; audit re-verified 932 newly saved members, 2219 reused bindings, and 2368 authorized input members. The floating-point state of 9 iterative model iterations was all FP32; quantization range and BN remained frozen, with three arms of real-weight updates; final states were strictly equal upon independent CUDA reconstruction. The seven AP/AR aggregations in 11 branch were recalculated from saved arrays, and the complete 8400 candidate box thresholds for three new groups were recalculated from saved raw data, without any forward pass or full AP matching re-execution.

| New Group | Full Candidate Threshold Passed Image Count | Maximum absolute difference between CPU and CUDA |
|---|---:|---:|
| New: Only improves localization | 0 / 128 | 28.212372 |
| New: Native target correction for ordinary hard examples | 0 / 128 | 65.828903 |
| New: Native target intervention correction | 0 / 128 | 45.800537 |

Pass for integrity preservation and CUDA reconstruction equality does not imply full candidate numerical qualification; numerical consistency across CPU/CUDA is unverified. The original numerical threshold remains unchanged, and neither confidence scores nor NMS subsets are used as substitutes. A formal 30-epoch training process, with real-world evaluation on 414 images at the native level, maintains consistency with prior FAIL cases and the first version of interface failures, all of which are preserved as-is.

# # Research Judgment and Next Steps

The new version reduces false positives while decreasing true object detection; it fails to achieve the joint objective of preserving more targets and reducing false positives.

For the group of typical hard examples with matched intervention, no joint advantage in AP, detection, or false positive rates is observed. This round cannot claim that intervention information provides additional contribution based on this result.

If proceeding further, it is essential first to verify whether improvements in localization and candidate ranking can genuinely preserve targets under the same false positive rate. Reducing scores or extending training duration alone cannot establish this. No automatic seed, epoch, or threshold search will be conducted in this round, nor will cloud costs be incurred.

Quality-aware scoring and its coordination with classification and localization have prior work, such as [GFL](https://arxiv.org/abs/2006.04388); task imbalance in quantization training has also been discussed in [HQOD](https://arxiv.org/abs/2408.02561). This round only draws upon the problem awareness and does not fully reproduce these methods; thus, this approach itself is not claimed as a novel contribution.

This round remains a floating-point W8A8 QDQ development with labeled, non-fused Conv layers, not real INT8 deployment. Only one training seed is used, and these development images were previously used for diagnosis and route adjustments. Therefore, they cannot be considered an unseen test set or independently validated for scientific acceptance. The filename prefix proxy is not a validated independent scenario. Selection load is not read, holdout remains sealed; no new timing checks or cloud resources are deployed.

Evidence directory: `../artifacts/gate1/20261026_native_target_correction_pilot_v1`. Key files include the contract, completion acknowledgment, actual exit observations, `saved_delivery_audit.json`, `machine_decision.json`, `saved_result_diagnostics.json`, and the final `delivery_validation.json`. The latest entry is `../artifacts/gate1/latest_execution_state.json`.
