# Gate1 Frozen Scorer: 178-image individual threshold calibration initiated



2026-10-26, at 20:55:41 CST, the pilot phase is initiated in accordance with user-provided instructions to address the limitations of previously using threshold-based scoring models fitted to calibration data. The decision of acceptance or rejection remains governed by fixed rules and complete results.

# # Data Split

From the original 4530 training images, exclude 512 images whose file name prefixes are fitted by the scorer, leaving 178 images and 34 prefixes entirely used for this standalone calibration. Members are determined solely by existing metadata and are sorted by image_id, without selection based on prediction performance. This set does not overlap with the scorer-fitted members, nor with 414 images of training_val in terms of image ID, image SHA, and prefix proxy.

These 178 images belong to the detector's original training data, and thus are independent of the scorer fitting, but are **not independent of the detector training**. The file name prefixes are also not validated as independent scene identities. Therefore, this round cannot constitute a new scientific test set.

The previous scorer parameters and old thresholds are retained. The frozen 23-parameter baseline scorer is not re-fitted, and the detector weights and ranges remain unchanged. Only the existing R model is run on these 178 calibration images, with 178 forward passes; all development outputs are reused without re-inference.

# # Unique Threshold Rule

First, perform official mirror IoU0.5 and cross-category cap500 matching between the calibration R model and the re-scored candidate pool. Let B be the number of calibration false positives (FPs) from the original R model at score ≥ 0.25. From the complete group-wise scoring curve of the scorer, select the lowest feasible score threshold such that FP ≤ B; if no non-empty feasible group exists, fix the threshold at 1 and reject all outputs. Group-wise matching is not split, and TP/FP mismatches are ignored.

This round employs only this threshold rule. No safety margin search, category-specific tuning, multiple calibration seeds, or repeated attempts are conducted. After obtaining the threshold, an immutable frozen_threshold.json is written, then the development comparison results are read and the threshold is applied. The threshold is not adjusted based on false positives in development images.

The threshold calibration uses the same score-based IoU0.5/cap500 counting as the official endpoint, improving upon the previous inconsistency with 'nominal FP count from the fitting set project labels'. Project-level target matching is still reported as a side-diagnostic, and is not mixed with official counting.

# # Scope and Pre-fixed Conditions

The primary comparison is with the existing additional 286 training_val images: under the frozen threshold scorer, the official TP must be at least as high as the original R@0.25, and the official FP must not exceed the original R@0.25. The project-level target matching is separately reported to satisfy the same conditions. The auxiliary scopes of the new prefix 47 images and the merged 414 images are fully preserved. All existing ordinary QAT@0.25 comparisons are also reported.

The 414 images were previously used extensively in development and influenced the selection of the baseline scorer. Thus, this is a fixed-threshold development check and cannot be considered an independent validation. The Selection load is not read, and the Holdout is sealed; true independent validation still requires separate confirmation of data access and experimental contracts.

A monotonic threshold does not change ranking or AP. Complete AP recomputations in this round total 0; previously archived AP values are reused. The old 8400-candidate numerical FAIL remains unchanged. No FP32 all-candidate reference is rebuilt for the 178 images, and no new numerical qualification is certified. This stage also makes no passing claim for real INT8 execution or device performance.

# # Pre-check, Sources, and Records

8 tests passed, covering prefix and content identity isolation, inclusion of all metadata members, same-budget handling, no feasible group fixed rejection, ignoring marks, no input modification, threshold closed interval handling, and actual matrix preservation matching.

Scorer training and threshold calibration should use different data—this principle is also stated in the [scikit-learn threshold tuning documentation](https://scikit-learn.org/stable/modules/classification_threshold.html).
This custom detection endpoint uses the existing official mirror matching and does not claim to reproduce the classification threshold method from that library.

Directory: `../artifacts/gate1/20261026_separate_threshold178_v1`

Contract SHA256: `57d4151fdccd0897c3d79b20367b7c76853ea64861f2040514ae3b0bf8193408`

The one-time launcher records the actual exit code propagation, without automatic retries. On recovery, it verifies the live state, actual process creation time, parent-child relationships, command line, and final state receipts, without treating process disappearance as completion.

This round does not lease cloud resources or add scheduled checks. All original code, contracts, reports, weights, and previous failures are preserved, and new outputs are separately sealed.
