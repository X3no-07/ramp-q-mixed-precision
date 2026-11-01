# Gate1 Scorer Group Balance and Feature Ablation: Complete Results



2026-10-26.

**Some candidates meet the joint observation condition on the main 286-image set, but no candidate satisfies all three scopes and all leave-one-prefix group conditions simultaneously, indicating that stability remains unresolved. Relative to the original baseline scorer, no group achieves AP improvement, detection stability (TP not decreasing), and false positive control (FP not increasing) on the main 286-image set. This round fails to demonstrate any additional value from the new method.**

This round only includes the three pre-defined new scorers, all of which are retained. All data are from training_val repeatedly used in development and are not independent test sets; group balancing and feature selection are not considered original contributions.

# # All Ranges and Fixed Threshold Endpoints

| Scope | Branch | AP50–95(%) | Fixed Threshold | Official TP | Official FP | TP Relative to R | FP Relative to R |
|---|---|---:|---:|---:|---:|---:|---:|
| New 286 images (primary) | Original R | 11.976789 | 0.250000000 | 7089 | 5214 | 0 | 0 |
| New 286 images (primary) | Original Baseline Scorer | 12.057128 | 0.174738524 | 7140 | 5202 | +51 | -12 |
| New 286 images (primary) | Baseline QAT (with existing cache) | 12.055115 | 0.250000000 | 7122 | 5193 | +33 | -21 |
| New 286 images (primary) | Group equal-weighting + complete 22-dimensional | 12.028789 | 0.176691036 | 7136 | 5202 | +47 | -12 |
| New 286 images (primary) | Image equal-weighting + relative quality 8-dimensional | 12.065538 | 0.172741813 | 7180 | 5247 | +91 | +33 |
| New 286 images (primary) | Group equal-weighting + relative quality 8-dimensional | 12.051016 | 0.176880981 | 7157 | 5230 | +68 | +16 |
| New prefix 47 images (auxiliary) | Original R | 11.340946 | 0.250000000 | 1072 | 953 | 0 | 0 |
| New prefix 47 images (auxiliary) | Original Baseline Scorer | 11.420621 | 0.174738524 | 1083 | 988 | +11 | +35 |
| New prefix 47 images (auxiliary) | Baseline QAT (with existing cache) | 11.079887 | 0.250000000 | 1061 | 981 | -11 | +28 |
| New prefix 47 images (auxiliary) | Group equal-weighting + complete 22-dimensional | 11.406191 | 0.176691036 | 1079 | 960 | +7 | +7 |
| New prefix 47 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | 11.407748 | 0.172741813 | 1096 | 967 | +24 | +14 |
| New prefix 47 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | 11.413755 | 0.176880981 | 1092 | 961 | +20 | +8 |
| Merged 414 images (auxiliary) | Original R | 11.869288 | 0.250000000 | 10556 | 7819 | 0 | 0 |
| Merged 414 images (auxiliary) | Original Baseline Scorer | 11.940007 | 0.174738524 | 10641 | 7792 | +85 | -27 |
| Merged 414 images (auxiliary) | Baseline QAT (with existing cache) | 11.921909 | 0.250000000 | 10580 | 7818 | +24 | -1 |
| Merged 414 images (auxiliary) | Group equal-weighting + complete 22-dimensional | 11.928868 | 0.176691036 | 10633 | 7813 | +77 | -6 |
| Merged 414 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | 11.949112 | 0.172741813 | 10693 | 7857 | +137 | +38 |
| Merged 414 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | 11.937548 | 0.176880981 | 10665 | 7834 | +109 | +15 |

Official endpoint policy follows score-priority IoU0.5 matching, cross-category cap500, original submission order, and ignore rules. AP is computed using a fixed candidate pool; actual endpoints use unique scalar thresholds pre-frozen across 178 calibration images for each model. No threshold re-tuning is performed based on the development set.

# # Additional Value Over the Original Baseline Scorer

| Scope | New branch | AP Difference (in percentage points) | Fixed endpoint TP difference | Fixed endpoint FP difference |
|---|---|---:|---:|---:|
| New 286 images (primary) | Group equal-weighting + complete 22-dimensional | -0.028339 | -4 | +0 |
| New 286 images (primary) | Image equal-weighting + relative quality 8-dimensional | +0.008410 | +40 | +45 |
| New 286 images (primary) | Group equal-weighting + relative quality 8-dimensional | -0.006112 | +17 | +28 |
| New prefix 47 images (auxiliary) | Group equal-weighting + complete 22-dimensional | -0.014430 | -4 | -28 |
| New prefix 47 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | -0.012873 | +13 | -21 |
| New prefix 47 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | -0.006865 | +9 | -27 |
| Merged 414 images (auxiliary) | Group equal-weighting + complete 22-dimensional | -0.011139 | -8 | +21 |
| Merged 414 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | +0.009105 | +52 | +65 |
| Merged 414 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | -0.002459 | +24 | +42 |

Among the main 286 images, there are no candidates that simultaneously improve AP over the original baseline scorer, maintain or increase fixed endpoint detections, and reduce false positives. This is merely a list of results from registration criteria, not a post-hoc selection of the best model, nor does it establish causality or originality.

# # Comparison with Standard QAT Baseline

| Scope | New branch | AP Difference (in percentage points) | Fixed endpoint TP difference | Fixed endpoint FP difference |
|---|---|---:|---:|---:|
| New 286 images (primary) | Group equal-weighting + complete 22-dimensional | -0.026326 | +14 | +9 |
| New 286 images (primary) | Image equal-weighting + relative quality 8-dimensional | +0.010423 | +58 | +54 |
| New 286 images (primary) | Group equal-weighting + relative quality 8-dimensional | -0.004099 | +35 | +37 |
| New prefix 47 images (auxiliary) | Group equal-weighting + complete 22-dimensional | +0.326304 | +18 | -21 |
| New prefix 47 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | +0.327861 | +35 | -14 |
| New prefix 47 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | +0.333869 | +31 | -20 |
| Merged 414 images (auxiliary) | Group equal-weighting + complete 22-dimensional | +0.006959 | +53 | -5 |
| Merged 414 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | +0.027203 | +113 | +39 |
| Merged 414 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | +0.015639 | +85 | +16 |

Standard QAT reuses previously saved results from the same development scope and the original 0.25 endpoint. This round verifies the curves, counts, and AP sources without retraining or re-scoring. The calibration methods for different branches are explicitly stated in the table; endpoint differences cannot be directly interpreted as equivalent false positive rates or equivalent computational advantages. Real INT8 runtime performance was not measured in this round.

# # All Joint Conditions and Group Sensitivity

| New branch | Main 286-image joint condition | Three-range joint condition | Three-range joint condition including all leave-one prefix groups |
|---|---|---|---|
| Group equal-weighting + complete 22-dimensional | Yes | No | No |
| Image equal-weighting + relative quality 8-dimensional | No | No | No |
| Group equal-weighting + relative quality 8-dimensional | No | No | No |

The main joint condition requires AP to exceed R, at least R@0.25 in fixed threshold TP, and no more than R@0.25 in FP. Additional robustness checks are performed on the three predefined ranges and the count of each prefix group removal. All results are preserved and no favorable subset is selected. Prefix groups are not validated independent scenarios; leaving out one group is not an independent repetition or statistical confidence interval.

| Scope | New branch | Project hit change | Project false positive change | Number of joint improvements after leave-one prefix group removal |
|---|---|---:|---:|---:|
| New 286 images (primary) | Group equal-weighting + complete 22-dimensional | +49 | -14 | 51/53 |
| New 286 images (primary) | Image equal-weighting + relative quality 8-dimensional | +93 | +31 | 0/53 |
| New 286 images (primary) | Group equal-weighting + relative quality 8-dimensional | +69 | +15 | 0/53 |
| New prefix 47 images (auxiliary) | Group equal-weighting + complete 22-dimensional | +7 | +7 | 1/21 |
| New prefix 47 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | +24 | +14 | 0/21 |
| New prefix 47 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | +20 | +8 | 1/21 |
| Merged 414 images (auxiliary) | Group equal-weighting + complete 22-dimensional | +79 | -8 | 52/58 |
| Merged 414 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | +139 | +36 | 0/58 |
| Merged 414 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | +110 | +14 | 3/58 |

# # Only two factors are changed

The existing image-equal-weight 22-dimensional model is reused directly; the three new models fill the other cells of the corresponding 2×2 ablation. Group-equal weighting gives all 58 training-prefix groups the same loss weight, followed by equal image weighting within each group and equal candidate weighting within each image. This prevents a prefix group with 50 images from naturally receiving 50 times the weight of a group with 1 image. Prefix identity is not provided to the inference model.

The 8-dimensional quality model retains 4 score-quality quantities and 4 relative-neighbor quantities, removing absolute width/height, center location, and class one-hot features. Removed coefficients are exactly 0 in the common 22-dimensional slot, leaving 9 effective parameters; the complete 22-dimensional model has 23. This design tests source associations without assuming all source features are wrong or interpreting the three-arm differences as a complete causal explanation.

Three models use native 512 images, 213716 cached candidate soft labels, without reading new images or training the detector. Normalization fits only the original training cache, with zero-residual initialization, fixed 100 CPU float64 Adam updates, lr0.03, betas0.9/0.999, eps1e-8, L2 coefficients 0.0005. No hard examples or intervention auxiliary losses are added; no best epoch, group weights, seeds, or regularization are selected.

# # The threshold is fixed once on the calibration dataset

| Branch | Unique threshold | Calibration TP relative to R | Calibration FP relative to R |
|---|---:|---:|---:|
| Group equal-weighting + complete 22-dimensional | 0.17669103591957433 | +81 | +0 |
| Image equal-weighting + relative quality 8-dimensional | 0.17274181308902065 | +83 | +0 |
| Group equal-weighting + relative quality 8-dimensional | 0.17688098105034686 | +64 | +0 |

These 178 training images were not involved in the scorer fitting and have no image ID, SHA, or prefix proxy overlap with the 512 fitting images or the 414 development images; they remain part of the original detector training data and do not constitute an independent test set for the detector. Each group maintains the same official FP budget and is processed uniformly across the group. All three thresholds are frozen only after being applied to the development outputs; the calibration budget does not guarantee that the number of false positives on new images will not increase.

# # Ranking diagnosis under the same false positive budget

| Scope | New branch | R reference FP budget | TP within new group budget | Actual FP |
|---|---|---:|---:|---:|
| New 286 images (primary) | Group equal-weighting + complete 22-dimensional | 5214 | 7139 | 5214 |
| New 286 images (primary) | Image equal-weighting + relative quality 8-dimensional | 5214 | 7173 | 5214 |
| New 286 images (primary) | Group equal-weighting + relative quality 8-dimensional | 5214 | 7153 | 5214 |
| New prefix 47 images (auxiliary) | Group equal-weighting + complete 22-dimensional | 953 | 1078 | 953 |
| New prefix 47 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | 953 | 1093 | 953 |
| New prefix 47 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | 953 | 1090 | 953 |
| Merged 414 images (auxiliary) | Group equal-weighting + complete 22-dimensional | 7819 | 10633 | 7819 |
| Merged 414 images (auxiliary) | Image equal-weighting + relative quality 8-dimensional | 7819 | 10686 | 7819 |
| Merged 414 images (auxiliary) | Group equal-weighting + relative quality 8-dimensional | 7819 | 10660 | 7819 |

The table evaluates the ranking along the entire curve derived from development labels; the thresholds obtained from this curve are not deployable thresholds. It does not replace the previously fixed threshold endpoints and does not allow using it to mask failures of the fixed endpoints.

# # Integrity and interpretability boundaries

The main workflow observed an actual exit code 0, with all three new evaluators completing 100 updates; all nine newly submitted AP tasks achieved exit0. 7 pre-checks were passed. Independent audit verified 280 newly saved members and 1489 existing cached bindings, recalculated loss weights and normalizations per image/group, removed coefficients strictly set to 0, three preserved states, 178 image calibration matrices and unique thresholds, all scoring, ranking, candidate identity, and filtering matrices for 414 images, and aggregated seven-metric summaries along with full tie-curve computations. Project matches were consistently recalculated for 32 fixed development probes per group.

No model re-fitting, detector image inference, or full AP scoring is performed. The previous engineering optimizations for bit-level equivalence and CPU timing are retained. No additional board or CPU latency tests are conducted in this round. It is not valid to claim that the entire feature computation is free simply because the effective feature dimension is reduced.

This design is influenced by prior diagnostic developments, and in conjunction with the previous 3 evaluators, it accumulates a new set of 6 candidates, all results are retained. The data is reused throughout development and is not considered a standalone scientific validation. Prior work on prefix balancing and group robust learning has already been established, such as in [Group DRO paper](https://arxiv.org/abs/1911.08731); this round only implements group equal-weight ERM and does not claim to reproduce the complete algorithm or present it as a new contribution.

Some candidates meet the joint observation conditions on the main 286 images, but no candidate satisfies all three ranges and the full leave-one-prefix-group condition simultaneously, and stability remains unsolved. Compared to the previous ordinary scorer, no group achieves improved AP, unchanged detection count, and no increase in false positives on the main 286 images. This round does not prove the additional value of the new method.

After budget exhaustion, no further attempts are made to pursue 'success' by selecting features, adjusting group weights, changing thresholds, or removing groups. If the current ablation is insufficient, the paper requires a new, distinguishable mechanism and independent validation, rather than merely renaming an ordinary scorer. Selection load is not read, holdout data is sealed, true INT8/Jetson and Gate1 are not passed; no new cloud resources or scheduled tasks are available.

Evidence directory: `../artifacts/gate1/20261026_group_balanced_scorer_v1`. All models, loss weights, thresholds, 9-group AP results, curves, item matches, leave-one-group records, independent audit, machine decisions, and final delivery verification are preserved.
