# Gate1: Frozen Ordinary Scorer - Extended Validation Results for 286 Images



2026-10-26.

**The frozen scorer achieves the target AP and meets the same false positive budget condition on the additional 286 images, new prefix 47 images, and merged 414 images, reproducing the extended development results.**

The fixed training threshold does not simultaneously satisfy 'detection count not reduced' and 'false positive count not increased' on the new 286 images. Therefore, curve improvement cannot be directly converted into a commitment to using this fixed threshold.

This round remains training_val extended development, not independent scientific validation, Gate1 full acceptance, or real INT8/board-end validation. All ranges and three fixed branches are preserved.

# # Complete Results of AP and Same False Positive Budget

| Scope | Branch | AP50–95(%) | Maximum TP under the same reference FP budget | Actual FP | Minimum FP to reach reference TP |
|---|---|---:|---:|---:|---:|
| New 286 images (primary) | Original mixed-precision R | 11.976789 | 7089 | 5214 | 5212 |
| New 286 images (primary) | R with frozen baseline scorer | 12.057128 | 7144 | 5214 | 5015 |
| New 286 images (primary) | Existing baseline QAT | 12.055115 | 7129 | 5214 | 5113 |
| New prefix 47 images (auxiliary) | Original mixed-precision R | 11.340946 | 1072 | 953 | 952 |
| New prefix 47 images (auxiliary) | R with frozen baseline scorer | 11.420621 | 1076 | 953 | 933 |
| New prefix 47 images (auxiliary) | Existing baseline QAT | 11.079887 | 1058 | 953 | 1021 |
| Merged 414 images (auxiliary) | Original mixed-precision R | 11.869288 | 10556 | 7819 | 7818 |
| Merged 414 images (auxiliary) | R with frozen baseline scorer | 11.940007 | 10649 | 7819 | 7480 |
| Merged 414 images (auxiliary) | Existing baseline QAT | 11.921909 | 10582 | 7819 | 7772 |

For each range, the false positive budget is independently calculated based on the official IoU0.5/cap500 FP count at score ≥ 0.25 of the original R, fixed prior to new inference. The original R itself also takes the maximum TP under the same budget across the full curve, avoiding comparison with only a single unfavorable reference point; all candidates with the same score are included. The following differences are listed in full, without selecting favorable ranges:

| Scope | Reference FP budget | AP scorer − R (percentage points) | TP scorer under same budget − R | AP scorer − baseline QAT (percentage points) | TP scorer under same budget − baseline QAT |
|---|---:|---:|---:|---:|---:|
| New 286 images (primary) | 5214 | +0.080339 | +55 | +0.002013 | +15 |
| New prefix 47 images (auxiliary) | 953 | +0.079675 | +4 | +0.340734 | +18 |
| Merged 414 images (auxiliary) | 7819 | +0.070719 | +93 | +0.018098 | +67 |

In the previous 128-image set, the AP of the baseline scorer increased from 11.986118% to 12.083086%, with 37 additional detections under the same official false positive budget of 2605. This 128-image result was used to select the current frozen scorer, and thus cannot be combined with the new 286-image set to claim an independent 414-image test set.

# # Actual checks with fixed training-set thresholds

The scorer threshold **0.17119779099758897** is derived solely from the original 512-training-image cache and the frozen scorer. Both original R and baseline QAT use the existing 0.25 threshold; no adjustment of scores is made based on newly developed images. The training threshold ensures that the nominal FP under the original training labels does not exceed the FP count of R at 0.25, but this does not guarantee exact FP equality on new images.

| Scope | Branch | Threshold used | Actual detections by project | Actual false positives by project |
|---|---|---:|---:|---:|
| New 286 images (primary) | Original mixed-precision R | 0.250000000 | 7090 | 5213 |
| New 286 images (primary) | R with frozen baseline scorer | 0.171197791 | 7184 | 5345 |
| New 286 images (primary) | Existing baseline QAT | 0.250000000 | 7123 | 5192 |
| New prefix 47 images (auxiliary) | Original mixed-precision R | 0.250000000 | 1071 | 954 |
| New prefix 47 images (auxiliary) | R with frozen baseline scorer | 0.171197791 | 1088 | 1013 |
| New prefix 47 images (auxiliary) | Existing baseline QAT | 0.250000000 | 1059 | 983 |
| Merged 414 images (auxiliary) | Original mixed-precision R | 0.250000000 | 10559 | 7816 |
| Merged 414 images (auxiliary) | R with frozen baseline scorer | 0.171197791 | 10714 | 8010 |
| Merged 414 images (auxiliary) | Existing baseline QAT | 0.250000000 | 10585 | 7813 |

The fixed training threshold does not simultaneously satisfy 'detection count not reduced' and 'false positive count not increased' on the new 286 images. Therefore, curve improvement cannot be directly converted into a commitment to using this fixed threshold.

Target matching is prioritized by IoU, while official sorting diagnosis uses score priority, with cross-category cap500 applied in the latter; these two counts are not interchangeable. The optimal threshold derived from development annotations is used solely for performance evaluation and is not used to replace the training-set threshold.

Simultaneously preserves the project counts from all branches under the original 0.25 threshold:

| Scope | Branch | Original 0.25 detection | Original 0.25 false positives |
|---|---|---:|---:|
| New 286 images (primary) | Original mixed-precision R | 7090 | 5213 |
| New 286 images (primary) | R with frozen baseline scorer | 6314 | 2970 |
| New 286 images (primary) | Existing baseline QAT | 7123 | 5192 |
| New prefix 47 images (auxiliary) | Original mixed-precision R | 1071 | 954 |
| New prefix 47 images (auxiliary) | R with frozen baseline scorer | 945 | 526 |
| New prefix 47 images (auxiliary) | Existing baseline QAT | 1059 | 983 |
| Merged 414 images (auxiliary) | Original mixed-precision R | 10559 | 7816 |
| Merged 414 images (auxiliary) | R with frozen baseline scorer | 9391 | 4546 |
| Merged 414 images (auxiliary) | Existing baseline QAT | 10585 | 7813 |

# # Data Boundaries and Correlations

The newly added 286 images are all members of the original training_val minus the old 128 images, and their image SHA values do not overlap with the complete 4530 training list. Among 286, 239 images share the same file name prefixes as those in the old 128, while 47 images come from 21 new prefixes; the original 128 covers 37 prefixes, and the new 286 covers 53 prefixes, of which 32 are overlapping. The members of the new prefixes 47 images are determined solely based on existing metadata prior to this detection round, and no selection is made based on the current detection results.

File name prefixes are not validated independent scene identities. The 286, 47, and 414 sets overlap with each other and do not represent three independent scientific experiments; these training_val images have also been used in earlier floating-point development checks. Even if all three ranges show improvement, this can only be described as an extended development reproducibility and cannot be claimed as independent testing or statistically significant.

This round only validates the reproducibility of the standard scorer. The previous intervention sampling did not outperform the standard scorer, and thus the benefits of the standard scorer cannot be rebranded as G1 intervention contributions. The quality scoring itself is part of an existing research direction, and originality still requires additional mechanisms and strong control evidence.

# # Execution and Independent Audit Verification

The frozen 23 detector parameters, features, normalizations, and both detectors were not retrained. Each new 286 image runs two existing detectors, completing 572 forward passes on images; the old 128 outputs are directly reused. The complete state fingerprints of both detectors remain identical before and after inference, and the initial values of ordinary QAT are consistent with the saved files from the existing round 3. The new model is trained for 0 epochs, and the activation range is recalibrated 0 times.

8 pre-checks passed; the audit verified 952 members of the current outputs, 521 reused bindings, 572 original arrays, and 572 new image/annotation inputs. The model-visible features and scorer outputs for the 286 images were independently reconstructed, confirming that the scorer did not modify bounding boxes, categories, or candidate members; all new GT/DT filtering matrices were matched against original annotations and outputs with zero tolerance, and the matrices from the original 128 were verified against the predictions.

All 9 aggregated tasks actually reported exit code 0, with only lightweight NumPy processes handling cached matrices. The audit recalculated 63 aggregated metrics from saved AP/AR arrays and re-computed IoU0.5 matching flags and full group tie curves; no detector re-execution or full AP pipeline was performed. The training threshold was independently recalculated from cached data and model parameters, and was consistent. The actual threshold matching for the fixed 32 new images was also recalculated and verified.

The first version of the saved audit terminated at the end of the 9th group due to an undefined 'now' name at the final report timestamp, and the PASS file was not generated. The original frozen script, contract, and error receipts were preserved as-is; an independent audit_v2 was implemented using P.now, completing a read-only verification and saving the results. This error does not involve model inference, scorer training, or the 9 formal AP tasks, and does not alter their completion status.

The new 286 images did not reconstruct the original FP32 full candidate reference, and thus do not claim numerical qualification; the historical complete 8400 candidate values remain as FAIL. Frozen parameters, saved consistency, AP or ranking improvements cannot substitute for original numerical qualification or board-level acceptance.

# # Stage Conclusion and Future Boundaries

The frozen scorer achieved the predefined AP and identical false positive budget conditions across the new 286 images, the new prefix 47 images, and the merged 414 images, demonstrating reproducibility of extended development results.

The fixed training threshold does not simultaneously satisfy 'detection count not reduced' and 'false positive count not increased' on the new 286 images. Therefore, curve improvement cannot be directly converted into a commitment to using this fixed threshold.

The current budget has been fully executed and sealed; all ranges have reported results. No additional threshold search, candidate expansion, training rounds, or feature filtering will be performed to meet a 'must succeed' requirement. If further evidence is needed for publication, a specific problem must first be frozen, and a data access scheme not previously used in repeated development must be established, along with supplementary stability, strong baselines, and actual computational overhead measurements.

Selection load was not read, Holdout remains sealed; no cloud resources or scheduled tasks were created. The evidence directory is `../artifacts/gate1/20261026_scorer_extension_286_v1`, with primary files including `contract.json`, training threshold calibration records, 9-group saved matrices/scoring curves, `attempt_01/comparison_result.json`, completion and actual exit receipts, `saved_delivery_audit.json`, `machine_decision.json`, and final delivery verification.
