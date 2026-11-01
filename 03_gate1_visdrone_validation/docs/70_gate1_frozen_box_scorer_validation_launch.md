# Gate1: Freeze the bounding box scorer — limited development validation initiated



2026-10-26, at 18:59:56 CST, the pilot commenced. Users agreed to 'try it out', with the expectation of achieving meaningful results; however, pass/fail status remains determined by pre-defined comparison rules and complete results, not by selecting thresholds, candidates, or metrics to achieve 'pass'.

# # This Round's Validation

Fix the existing MSE calibration and recovery G1 of the detection model, reusing the outputs from 512 training images and 128 development images already saved. The existing NMS candidate pool consists of conf0.001, IoU0.7, with a maximum of 1000 boxes; no changes are made to the boxes, categories, or candidate members, and NMS is not rerun. Only the candidate scores and the official submission ranking are modified.

The scorer's 22 inputs are model-visible information: current score and second-category score, category gap and entropy, width and height with center position, number of neighboring boxes of the same category in terms of score and overlap, maximum overlap with other categories, and one-hot category encoding. Inputs do not include ground truth annotations, GT distances, or Q-path outputs; precision intervention information is used only during offline sample selection. If this scorer is deployed, additional costs for neighbor computation and ranking must still be separately evaluated, and such additional computation cannot be considered free.

Three fixed candidates use the same linear structure, 23 parameters, identical zero initialization, and 100 CPU float64 full-batch Adam updates:

1. score_base: loss from common training data.
2. score_hard: common loss with positive and negative supervision from typical hard examples.
3. score_intervention: common loss with positive and negative supervision from previously selected precision intervention samples.

The last two groups perform per-image matching of positive examples by class and scale, and negative examples by class, confidence interval, and error type. Only samples still available in a fixed R candidate pool are selected; missing samples are not artificially generated; the actual count is reported by cache response. Base supervision consists of all training candidates not ignored; TP targets are based on existing GT matches with IoU, while FP targets are 0. Labels are used solely for training targets and sample selection. Normalization parameters are fitted only from training features; development annotations do not enter training or input features.

The scoring form is the original logit score plus a linear residual, with parameters starting from 0; training is conducted for 100 fixed updates without selecting the best iteration. The common loss is weighted equally per training image, the sampling loss is split equally between positive and negative groups, with an additional fixed weight of 0.25, and the L2 coefficient fixed at 0.0005. No search is performed on network architecture, features, learning rate, random seed, or auxiliary weights.

# # Comparison Rules

Preserve the official Python AP and fixed 0.25 threshold project-level target count. Additionally, introduce a new diagnostic focused on ranking: construct a global score curve using the official IoU0.5, cap500 matching. The false positive budget B is defined as the official number of false positives at score 0.25 in the original R; the target count T is taken at this reference point. For candidate reports, when FP ≤ B, the maximum TP is capped, and the minimum FP required to reach T is reported. All co-ranked candidates are included, and splitting tied scores to gain favorable outcomes is prohibited.

These curve points belong to the development set performance evaluation. The score thresholds derived from the curve are based on development annotations and cannot be treated as learned deployment thresholds. The project-level matching count is defined differently from this official ranking diagnostic and must not be mixed or conflated. Candidates must show improvements in both detection and AP under a fixed FP budget; the additional value of the intervention signal must also be compared against the same scorer's standard hard examples and standard training. Without independent data or multiple lines of evidence, the contribution of the paper is not established.

The original complete AP of R is not re-run and is directly reused; the newly introduced IoU matching is used solely for this new ranking curve diagnostic. A full new submission score is run only for the three new scorers. The forward pass of detector images is 0, calibration is 0, and the original detector weights training is 0.

# # Pre-check and identity

All 10 tests passed, covering 22-dimensional feature indexing, invariance to additional GT/Q fields, training-only normalization, retention of original scores when the residual is 0, tied-group budget handling, exclusion of ignored items from TP/FP, deterministic training and saved-state reproduction of the actual scorer, and the official cap applied across classes.

Contract directory: `../artifacts/gate1/20261026_frozen_box_scorer_pilot_v1`.

Contract SHA256: `9dc0d3256183dd5e44aa980e9a1e9c8c599210bf149947cdec26c8d0c982c778`.

The launcher adds a one-time parent process wait, records the exit code of actual worker processes, and does not automatically fail and restart. Upon task recovery, it verifies live status, parent-child relationships, creation time, command line, and completion/failure receipts.

This round remains limited to the 128 training_val images repeatedly used in development and cannot serve as an independent test set. Selection load is not read, and Holdout remains sealed. Original full bounding box numerical FAILs, previous training, scoring, and failure histories are preserved; consistency of scorer saving, ranking/AP improvement, and real INT8/board acceptance are separately described. No cloud resources or scheduled tasks are created.
