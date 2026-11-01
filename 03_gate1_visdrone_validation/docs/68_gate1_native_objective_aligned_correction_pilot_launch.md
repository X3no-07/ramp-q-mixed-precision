# Gate1 Native Scoring Target Alignment Correction Pilot: Launch Record



2026-10-26, 18:28:26 UTC time started. Upon user 'continue' authorization, a new limited development comparison is initiated based on the failure clues from the previous round. This report only records design and startup; completion and outcome conclusions will be addressed in follow-up verification and additional results reports.

# # The Question to Be Tested

The previous version used IoU as the target for the additional positive-classification BCE, while the original detector's classification loss used soft targets assigned by native TAL. These two targets may differ: for example, a score of 0.2, TAL target of 0.05, and IoU target of 0.7 would require the original classification term to reduce the score, whereas the additional classification term would require increasing it. This conflict has been reproduced via gradient testing; it is an interpretable design risk, but has not yet been proven to be the sole cause of the increased false positives in the previous round.

The new version removes the additional positive-classification BCE and only adds IoU localization loss for selected ground truth (GT) boxes. The additional classification supervision for incorrect boxes continues to use the native TAL soft target from the same training iteration, and only activates when the score exceeds this target:

`Lerror = 1[p > t] · (BCE(logit, t) − H(t))`

wherein t is completely detached from gradients; within the active region, the gradient with respect to the logits is `p−t≥0`, and gradient descent can only directly reduce the score of that channel. When below the target, this term is zero. This does not guarantee that the overall network output scores will necessarily decrease, because shared weights are also influenced by the base loss, localization loss, and other samples.

# # Fixed Three Arms and Controls

| New Training Group | Additional Localization Supervision | Additional Score Correction |
|---|---|---|
| localization_only | The 601 GTs selected in the original intervention group | None |
| native_hard_correction | Standard hard examples (GTs) with the same category and scale count as before | Standard hard example incorrect boxes |
| native_intervention_correction | The 601 GTs selected in the original intervention group | Incorrect boxes newly introduced after Q→recovery of G1 |

The last two groups' erroneous boxes are matched by category, confidence score, and error geometry type on a per-image basis, with a maximum of 8 matches per image. Coverage includes same-category overlaps, cross-category overlaps, localization neighborhoods, and low-overlap residuals; category-level explanations remain for project diagnostics and do not equate to official error types. All errors retain their original native TAL targets and ignore masks: localization discrepancies or currently assigned positive boxes are not directly treated as pure background.

All existing ordinary QAT, old hard examples, old intervention, and four calibration Q/R baselines, as well as the FP32 reference, are reused and preserved for each group. Only the localization-only group is used to observe changes resulting from the removal of additional positive-classification supervision; the hard example and intervention correction groups are used to evaluate whether sample selection adds additional value. Each comparison remains limited development evidence and cannot serve as a complete causal ablation or independent scientific validation.

# # Budget and Pre-check

The training and development team still uses a fixed number of 512/128 images, 3 epochs, batch size 4, accumulated gradients 4, with each group undergoing 96 optimizer updates; identical initial FP32 states, MSE calibration range, sequence, learning rate, and final rule at epoch 3. The three new groups collectively perform a total of 4992 forward passes, with the final model independently saved and reconstructed for 3 separate forward passes. The new version does not perform teacher model forward passes, does not repeat calibration or baseline inference, and instead reconstructs new supervision solely from the previously saved Q/R array. Recomputing matches of the saved array will be recorded in the new output and cannot be considered a repetition of prior model experiments.

10 test cases passed, including finite behavior at the 0/1 endpoint for single-side terms, gradient direction, lack of suppression on low-confidence true positives, counterexamples to the original target conflict, geometric grouping boundaries, error type matching budget, actual CUDA gradients, ignore region protection, weight updates, and model saving and reconstruction.

This work directory: `../artifacts/gate1/20261026_native_target_correction_pilot_v1`

Contract SHA256: `d9eb73fb9059ef1c4c106d825247e49df51cc4136463d757e3c736cf8dc7d33b`.

Launcher PID 49096, actual working process PID 45096; parent-child relationship, creation time, and command line have been verified. Upon recovery, the live file and actual process must be re-read; old PIDs or vanished processes cannot substitute for completion acknowledgment.

# # Literature and Contribution Boundaries

Collaborative alignment between localization quality and classification scoring has already been established in [GFL](https://arxiv.org/abs/2006.04388) and similar works; imbalance in classification and regression during quantization training has also been discussed in [HQOD](https://arxiv.org/abs/2408.02561). This version inherits this awareness but does not reproduce the full methods of these papers, and does not claim the side-correction or quality scoring mechanisms as novel contributions. The remaining open question remains whether intervention information provides additional value when matched to typical hard examples.

This round remains a labeled floating-point QDQ development training. The 128 training_val images serve only as development evidence; the original full 8400 candidate threshold is preserved separately, all previous FAILs are retained; Selection and Holdout loads are not read. No cloud resource leasing, data upload, or new scheduled tasks have been initiated.
