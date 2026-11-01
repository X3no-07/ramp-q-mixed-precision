# Gate1: Frozen bounding box scorer: results and fixed false positive budget check



2026-10-26.

**The standard scorer pre-defined in the prior list meets the joint development observation criteria relative to the original R: AP improvement and no reduction in detection under the same false positive budget. This is a development signal, not a pass of Gate1 or independent scientific acceptance.**

The intervention-selected scorer fails to meet the predefined joint development observation criteria relative to the original R. The intervention-selected scorer does not demonstrate a shared advantage over the standard scorer and standard hard examples, and thus its additional contribution remains unproven.

These judgments apply only to the 128 training_val images repeatedly used in development; they do not constitute independent scientific acceptance, real INT8, or board-level approval.

# # Scoring Results

| Branch | AP50–95(%) | Fixed 0.25 threshold project detection | Fixed 0.25 threshold project false positives |
|---|---:|---:|---:|
| Original mixed-precision R | 11.986118 | 3469 | 2603 |
| Standard scorer | 12.083086 | 3077 | 1576 |
| Standard hard example scorer | 10.763443 | 2626 | 1890 |
| Intervention-selected scorer | 11.800868 | 3652 | 3117 |

The conventional QAT model on the same 128-image dataset achieves an AP of 12.054024%. It serves as a development counterpart of another detection model and its prediction boxes are not mixed into the fixed R candidate pool in this study.

Project-level matching prioritizes IoU over the official scoring method, which differs from the standard score-based matching definition. After score adjustment, the 0.25 threshold count is also affected by the score scaling, thus it must be combined with the following identical false positive budget check and complete AP evaluation.

# # Same False Positive Budget Ranking Check

The reference rule was fixed before executing new candidates: B is the original R false-positive count under official-style IoU0.5 and cap 500 at score≥0.25, namely **2605**; T is the **3467** true detections at that point. For a fair comparison, the original R also uses its maximum TP over the complete curve subject to FP≤B, rather than only its point at 0.25. Tied candidates are included as complete groups without splitting ties.

| Branch | Maximum TP under FP ≤ B | Corresponding FP | Minimum FP required to reach T |
|---|---:|---:|---:|
| Original mixed-precision R | 3467 | 2605 | 2604 |
| Standard scorer | 3504 | 2605 | 2468 |
| Standard hard example scorer | 2980 | 2605 | 4111 |
| Intervention-selected scorer | 3498 | 2605 | 2511 |

These curve points are derived from development annotations for performance evaluation, and the resulting score thresholds are not directly deployable. Final deployment thresholds must be established independently on development or calibration data separate from the validation set. The optimal point on the curve is not used as justification for modifying the original 0.25 threshold.

# # Intervention Group Relative to Controls

| Comparison Object | AP Difference (in percentage points) | TP Difference under Same FP Budget |
|---|---:|---:|
| Original R | -0.185250 | +31 |
| Standard scorer | -0.282219 | -6 |
| Standard hard example scorer | +1.037425 | +518 |

# # Input and Supervision Boundaries

The scorer has only 22 models visible inputs and 23 trainable parameters. Features include original and second-highest class scores, class gaps and entropy, bounding box width and height, center coordinates, overlap and scores with neighboring instances of the same class, and overlap with other classes, as well as one-hot class encoding. Ground truth box distances, matching IoU, true/false labels, and Q-path outputs are not included as inputs to the scorer. Ground truth is used solely for training soft labels and offline sampling.

512 images from the existing training dataset generate 213716 unignored candidate training rows. The intervention group selects 536 positive routine instances and 882 negative routine instances; the conventional hard example group matches the same number of instances by category/scale and by category/confidence interval/error type. The original 601 intervention ground truth labels and 882 erroneous channels are retained only for members still available within a fixed candidate pool, without synthesizing missing data. The two groups show overlapping selections: 121 positive examples and 401 negative examples.

All scorers are initialized from the same zero-residual state and undergo 100 fixed full-batch Adam updates. The common loss assigns equal weight to each training image, with an additional balanced binary cross-entropy (BCE) loss of 0.25 weight for both positive and negative examples in the two sampling groups; an L2 coefficient of 0.0005. No optimal iteration, random seed, feature, threshold, or parameter is selected. Mean and variance normalization is fitted solely from training features, and development images are not provided with labels or loss during training.

# # Verified Completeness

10 items passed pre-check. Audit verified 1304 current saved members and 1669 reused bindings, independently reconstructed 128 image development features and 32 deterministic training probe features, confirmed consistency in bounding boxes, categories, and identities across three saved scorers and 384 development candidate pools, as well as all seven AP/AR aggregations. One IoU-matching annotation from the official curve was recalculated from the saved matrix, and the entire group of tied curves was independently re-computed and found consistent; no model retraining or detector forward passes were performed.

This round performs 0 new detector image forwards, 0 updates to existing weights, and 0 calibrations. Training features use the fixed NMS pool from saved R arrays; the development pool is reused directly. NMS is not rerun after scoring and no boxes are added; only scores and submission order change. Old raw outputs and every historical all-candidate numerical FAIL remain unchanged. These scoring outputs do not certify original-tensor numerical qualification or device qualification.

All workflow completion receipts are complete, and the launcher observed a propagation exit code of 0. An additional final complete delivery hash check is performed, and the saved consistency PASS is not equated with method effectiveness PASS.

# # Explanation and Termination Criteria

The standard scorer, pre-specified in advance, meets the joint observation condition for this development cycle: relative to the original R, it improves AP while maintaining the same false positive budget and achieving or better detection performance. This constitutes a development signal, not a full scientific acceptance or gate-level approval of Gate1 or any independent component.

The intervention sampling has not demonstrated a relative advantage over the standard scorer when matched against hard examples, and thus its additional contribution remains unproven.

If the standard scorer is effective, it indicates that model-visible features possess a meaningful ranking value. Only if the intervention sampling consistently and stably outperforms both the standard training and the standard hard example setup under the same architecture, does it justify further investigation of the selection information. Currently, there is only one fixed development cohort and one deterministic linear structure, and these development graphs have already been involved in prior diagnostic analyses. Any observed improvements require independent and repeated validation with newly registered data.

Linear scoring, quality-aware scoring, and hard sample training have already been extensively studied in prior work. This round does not treat the scorer itself as a novel contribution. The standard scorer serves as a pre-registered control and is not rebranded post-hoc as an intervention method. Additional neighbor computation and ranking incur real costs; this round does not measure end-to-end GPU or board-level latency, nor does it claim such costs are negligible.

All three candidates are fully retained. After the budget ends, the criteria will not be altered or the screening process continued simply due to user requests for 'pass'. The selection load has not been read, and the holdout set remains sealed. No cloud resources are available, and no new timing checks are scheduled.

Evidence Directory: `../artifacts/gate1/20261026_frozen_box_scorer_pilot_v1`. Primary files: `contract.json`, `attempt_01/comparison_result.json`, `attempt_01/FP_budget_reference.json`, saved curves, three scorers, `saved_delivery_audit.json`, `machine_decision.json`, launcher final state, and final delivery validation.
