Gate1 Scorer: Fixed three-arm development experiments on prefix balancing and feature ablation



2026-10-26. With user progress under authorization, this round of fixed three-arm small scorer comparisons is conducted based on existing diagnostics of revenue concentration and training group imbalance, without guarantee of favorable outcomes.

Problem and Controls

The original scorer was trained on 512 images covering 58 prefix groups, with group sizes ranging from 1 to 50 images; conventional training uses image-level equal weighting. Another fact is that model inputs include absolute width and height, center coordinates, and class identity—information that may vary across data sources. This round evaluates weights and features separately, without pre-determining either as a cause of false positive growth.

| Training weights | Full 22-feature | Relative quality 8-feature |
|---|---|---|
| Image-level equal weighting | Old standard scorer, directly reused | New image_balanced_quality8 |
| Prefix-group equal weighting | New prefix_balanced_full22 | New prefix_balanced_quality8 |

The 8-dimensional quality model retains original score, second-highest score, class margin, and entropy, plus 4 relative-neighbor quantities. It removes absolute width/height, absolute center coordinates, and 10-class one-hot features. It uses a common 22-dimensional parameter slot, but coefficients of removed features must be exactly 0. Its effective parameters are 8 coefficients and 1 bias; the complete model has 22 coefficients and 1 bias.

Prefix-group equal weighting is defined as: all 58 training prefix groups have equal total weights; images within each group are equally weighted; candidate rows within each image are equally weighted. Prefix identity is not provided as input during inference; true labels are used only for original training soft targets. The prefix serves only as a filename proxy and does not represent a validated scenario or domain.

Frozen budget

All three new evaluators are initialized from zero-residual, reusing the same 512 image, 213716 training candidates and GT IoU soft targets, with normalization fitted only from this cache. A fixed 100 full-batch CPU float64 Adam updates, lr0.03, betas0.9/0.999, eps1e-8, L2 coefficients 0.0005. No hard examples or additional losses are introduced, and no hyperparameter search over iteration count, regularization, learning rate, or group weights is performed.

The threshold is still determined solely from the existing 178 scorers calibrated on external validation images. Each new model generates exactly one scalar threshold, determined in the official R@0.25 manner using IoU0.5/cap500 false positive budget, and all bounding boxes in the group are processed simultaneously. All three thresholds are written into immutable records before being applied to development data; the thresholds are not adjusted based on new development results.

Reuses the entire 414 image R candidate pool and saved features, keeping prediction boxes, classes, and candidate identities unchanged, while only modifying scores and submission order. The new detector performs inference on 0, image decoding on 0; three new models are deployed within the predefined 286 / 47 / 414 ranges, resulting in a total of 9 new AP aggregation tasks, with at most two lightweight CPU tasks running in parallel. All previous R, original scorer, and ordinary QAT are fully reused.

Including the previous 3 scorers, the cumulative number of new scorer candidates after this round is 6. These adaptive development attempts cannot be presented as one predefined independent experiment.

# # Pre-fixed Comparison

The primary scope is the newly added 286 images, evaluating the AP improvement of the new groups relative to R, under fixed thresholds after calibration, with official TP not decreasing and FP not increasing. Complete reports are provided for the new 47-image prefix and the merged 414-image group. Robust observation also requires that the number of items in each scope and all leave-one-prefix groups meet joint improvement. This requirement may not be satisfied, and if so, it is retained truthfully.

Each group is also compared against the original baseline scorer under the original standalone-calibrated threshold of 0.17473852372359241. If a new model is only stronger than the original R but still does not exceed the old scorer, it cannot be considered an additional gain from this modification. Group ablation and feature ablation are both retained; no single group is selected post-hoc to replace the full set of results.

7 pre-checks passed: group weight conservation, group-wise image and candidate normalization, input order invariance, missing image rejection, removal of feature coefficients with value 0, deterministic training and reproducibility, and strict equivalence to previous standard training when operating on a single group.

Group-wise robust learning is a well-established research direction, e.g., [Group DRO paper](https://arxiv.org/abs/1911.08731). This round only conducts a fixed-group equal-weight ERM comparison and does not reproduce the complete Group DRO algorithm from the paper, and thus it is not considered a new contribution.

Directory: `../artifacts/gate1/20261026_group_balanced_scorer_v1`

Contract SHA256: `e51d3c5941f0faf41b2e5ec8e65e32a4798888d1a33696c9e6885e9ec52d4500`

training_val has been used across multiple development rounds and cannot be considered an independent test set; the Selection load has not been read, and Holdout is sealed. No real INT8/Jetson validation has occurred; the original full-candidate numerical result FAIL is retained. The original source files and contract do not cover this work; no cloud resources or scheduled checks are available.
