# Gate1: Frozen Scorer Standalone Threshold Calibration: 178-Image Calibration and Fixed Endpoint Validation



2026-10-26.

**After separate calibration, the fixed threshold increases detections and reduces false positives on the main additional 286-image set and some aggregate scopes. False positives still increase on the 47 images with new prefixes; the conditions do not pass across every scope.**

This round only validates the actual threshold endpoint, without altering the scorer, prediction boxes, categories, candidate members, ranking, or mAP. Development pass does not equate to paper contribution, independent scientific validation, Gate1 overall pass, or INT8/board-level pass.

# # Actual Results After Fixed Threshold

The unique threshold is **0.17473852372359241**. It was first computed from 178 training images independently fitted to the scorer and written into an immutable record, then applied to the existing 414 development outputs. The original R and standard QAT continue to use 0.25.

The following adopts project-level target matching (confidence fixed, IoU0.5, IoU priority):

| Scope | Branch | Threshold used | True target hits | false positives |
|---|---|---:|---:|---:|
| New 286 images (primary) | Original mixed-precision R | 0.250000000 | 7090 | 5213 |
| New 286 images (primary) | R with frozen baseline scorer | 0.174738524 | 7143 | 5199 |
| New 286 images (primary) | Existing baseline QAT | 0.250000000 | 7123 | 5192 |
| New prefix 47 images (auxiliary) | Original mixed-precision R | 0.250000000 | 1071 | 954 |
| New prefix 47 images (auxiliary) | R with frozen baseline scorer | 0.174738524 | 1082 | 989 |
| New prefix 47 images (auxiliary) | Existing baseline QAT | 0.250000000 | 1059 | 983 |
| Merged 414 images (auxiliary) | Original mixed-precision R | 0.250000000 | 10559 | 7816 |
| Merged 414 images (auxiliary) | R with frozen baseline scorer | 0.174738524 | 10646 | 7787 |
| Merged 414 images (auxiliary) | Existing baseline QAT | 0.250000000 | 10585 | 7813 |

Difference in project matching between the scorer and original R:

| Scope | Hit variation | False Positive Change | Detection not degraded and false positives not increased |
|---|---:|---:|---|
| New 286 images (primary) | +53 | -14 | Yes |
| New prefix 47 images (auxiliary) | +11 | +35 | No |
| Merged 414 images (auxiliary) | +87 | -29 | Yes |

The primary scope is the pre-fixed additional 286 images. The merged 414-image set includes the 128 images previously used for scorer selection, and cannot mask the increase in false positives within the 47-image auxiliary range using a merged average. The 47 images contain 21 new filename prefixes, but these prefixes are not already validated independent scenarios, and thus cannot be considered 21 independent scientific samples.

# # Official Counting Under the Same Fixed Threshold

The main evaluation metric comes from the officially validated Python image IoU0.5 matching, preserving score priority, original submission order, and cross-category cap500. Each score threshold is fixed in advance and not selected dynamically along the development curve. This count differs from project-level IoU priority matching; all differences are reported truthfully.

| Scope | Branch | Official TP | Official FP | Ignored items |
|---|---|---:|---:|---:|
| New 286 images (primary) | Original mixed-precision R | 7089 | 5214 | 0 |
| New 286 images (primary) | R with frozen baseline scorer | 7140 | 5202 | 0 |
| New 286 images (primary) | Existing baseline QAT | 7122 | 5193 | 0 |
| New prefix 47 images (auxiliary) | Original mixed-precision R | 1072 | 953 | 0 |
| New prefix 47 images (auxiliary) | R with frozen baseline scorer | 1083 | 988 | 0 |
| New prefix 47 images (auxiliary) | Existing baseline QAT | 1061 | 981 | 0 |
| Merged 414 images (auxiliary) | Original mixed-precision R | 10556 | 7819 | 0 |
| Merged 414 images (auxiliary) | R with frozen baseline scorer | 10641 | 7792 | 0 |
| Merged 414 images (auxiliary) | Existing baseline QAT | 10580 | 7818 | 0 |

| Scope | Official TP relative to R | Official FP relative to R | Official TP relative to standard QAT | Official FP relative to standard QAT |
|---|---:|---:|---:|---:|
| New 286 images (primary) | +51 | -12 | +18 | +9 |
| New prefix 47 images (auxiliary) | +11 | +35 | +22 | +7 |
| Merged 414 images (auxiliary) | +85 | -27 | +61 | -26 |

# # Calibration Generates Only One Candidate

Starting from the original 4530 training images, metadata exclude filename prefixes associated with the scorer fitting set of 512 images. All remaining 178 images across 34 prefixes are used for calibration. Calibration images do not overlap the scorer fitting set or 414 development images by image ID, content SHA, or prefix proxy. No new predictions or annotations were used to select members.

These calibration images belong to the detector's original training set, so the term 'separate' refers only to parameter fitting for the scorer, and does not constitute an unseen test set for the detector. No gradient information for the scorer is provided for the 178 calibration images; the 23 parameters of the scorer, normalization, features, and detector range are maintained at their original values.

On all 178 calibration images, the official FP count at R@0.25 from the original detector defines the budget. Using the same official method as the scorer, a curve is matched and the lowest feasible score threshold is selected such that FP does not exceed the budget. All boxes with the same score are grouped together; if no non-empty feasible group exists, the threshold is fixed at 1. No search for safety margins, class thresholds, seeds, additional calibration members, or a second threshold is conducted.

The calibration reference endpoint is TP=6003, FP=3892; the selected threshold's calibration endpoint is TP=6098, FP=3892. This is only a budget constraint on the calibration data and does not guarantee that the number of false positives on new images will not increase.

Threshold optimization and model fitting data should be separated; this principle is also described in [scikit-learn threshold tuning documentation](https://scikit-learn.org/stable/modules/classification_threshold.html). In this round, a custom detection-matching endpoint is used, rather than reproducing the full classification tuning method.

# # AP remains unchanged

Changing the acceptance threshold does not alter the sorting of the previously fixed candidate pool; the full AP is still taken from the original stored result. The full AP was not re-computed in this round, and the truncated results after thresholding were not misrepresented as an AP improvement. The original scorer's AP data is still listed as follows:

| Scope | Original R AP (%) | Frozen Scorer AP (%) | Ordinary QAT AP (%) |
|---|---:|---:|---:|
| New 286 images (primary) | 11.976789 | 12.057128 | 12.055115 |
| New prefix 47 images (auxiliary) | 11.340946 | 11.420621 | 11.079887 |
| Merged 414 images (auxiliary) | 11.869288 | 11.940007 | 11.921909 |

The AP difference between the new scorer and ordinary QAT on an additional 286 images is only about 0.002 percentage points, and thus cannot be claimed to have a significant accuracy advantage. The ordinary scorer represents the existing baseline form; prior interventions in sampling did not exceed the ordinary scorer, and therefore the benefits from this calibration cannot be rebranded as contributions from the intervention method.

# # Actual Completion and Saved Validation

The current round of work completed with exit code 0, and all return receipts and 178 forward-pass budgets were fully realized. The new scorer was fitted 0 times, inference on development images occurred 0 times, and full AP re-evaluation occurred 0 times. Per-target IoU matching and new threshold projects are new analyses in this round and are each independently saved.

8 pre-checks passed. Independent audit saved 541 new output members, 30 existing development bindings, and 770 authorized image/annotation inputs; all 178 raw images, 178 independently visible feature and scorer outputs, original annotations to filtering matrices, per-target IoU matching, grouped curves, and unique thresholds were verified.

The freeze threshold event occurs prior to the application of the development threshold, and the new threshold function interface only accepts calibration score and match flag. Three development scopes, three paths, collectively account for 9 official endpoints counted independently from saved curves, with 32 fixed development images recalculated for per-object match consistency. Validation confirms no duplicate model inference, model fitting, or full AP.

The detector state fingerprint remains unchanged before and after calibration inference, and the scorer file SHA remains frozen. The original 8400 candidate values in raw data were not reclassified as FAIL; no FP32_reference was constructed for the 178 new images, and thus no floating-point qualification is authenticated. A saved consistency PASS only indicates evidence completeness and cannot substitute for effectiveness, scientific validation, or equipment acceptance.

# # Progress and Remaining Issues

After separate calibration, the fixed threshold achieves detection increases and false positive reductions on the majority of the newly added 286 images and on some aggregated ranges; however, the new prefix of 47 images still increases false positives, and cross-range conditions have not been fully satisfied.

The first stage has already provided actual evidence of improvement from the fixed threshold on the main development data, but false positive control on the new prefix subset remains unsolved, and thus 'all cases completed' is not marked. This round includes only one registered threshold, and no further adjustment is made based on the results of the 47-image subset.

The next stage of work should shift toward independently verifiable contributions and robustness: clearly distinguishing mechanisms from ordinary quality scoring, strong baselines and ablation studies, truly independent data not involved in development, and additional computational costs. Continuing to adjust thresholds on the same development images cannot prove such contributions.

Holdout content remains unopened, and Selection load has not been read. A true independent data phase requires specific data access decisions and frozen contracts; equipment phase requires actual connections and INT8 execution evidence; neither has been claimed to have passed. No new cloud resources or scheduled checks have been added.

Evidence Directory: `../artifacts/gate1/20261026_separate_threshold178_v1`. Primary documents include `contract.json`, `attempt_01/frozen_threshold.json`, calibration prediction and filtering matrix, `attempt_01/comparison_result.json`, independent audit, actual exit confirmation, machine decision, and final delivery verification.
