# Failure Mechanism Review and Next-Phase Plan for Gate1



2026-10-27, Beijing time.

**The next round of priority validation focuses on the training modification that limits auxiliary correction strength while protecting normal object detection. The current original correction scheme still fails. This round completes failure diagnosis and small-scale gradient measurement, without training any new candidate models.**

# # 1. What happened to the lost targets

On the existing 414 training_val images, using the original project matcher and fixed 0.25 threshold, standard QAT detects 10596 targets and the correction arm 10010. Identity-level checks find 9735 shared detections,861 targets lost by correction,275 gained, and a net decrease of 586.

An inspection of the original sample-preserved post-processing candidate pool for the 861 lost targets reveals:

| Candidate preservation status | Number of targets |
|---|---:|
| There are still same-class boxes with IoU ≥ 0.5, but all have scores below 0.25 | 724 |
| There are boxes with scores reaching 0.25, but no one-to-one matching occurs | 24 |
| No same-class boxes with IoU ≥ 0.5 are preserved in the post-processing candidate pool | 113 |

724 out of 861 (approximately 84.1%) fall into the first category. The average score of jointly detected targets decreases by 0.03018, and the average IoU change is −0.000428. This supports the hypothesis that score degradation is a key indicator of fixed-endpoint detection failures. However, it cannot be concluded that the 724 cases would necessarily be recovered by lowering the threshold, nor that the 113 cases represent complete absence of original anchor targets. Candidate competition, class, localization, NMS, and candidate truncation remain distinct factors. This round does not perform further causal attribution on these aspects.

# # 2. Can lowering the threshold resolve the issue?

The ranking curves are reconstructed from saved official-style IoU0.5 scores and match flags, including tied scores together, and agree elementwise with the old curves. All comparisons below use the **official-style 7751 false-positive** budget of standard QAT at 0.25, avoiding confusion with the project matcher's 7747 false positives.

| Pathway | Same 7751 false positive detections detected | Relative to standard QAT |
|---|---:|---:|
| Standard QAT | 10592 | — |
| Native intervention correction | 10541 | −51 |
| Standard QAT with original frozen scorer | 10656 | +64 |

Even after aligning the number of false positives, the original correction group does not exceed the standard QAT. The original scorer shows only minor improvement in ranking; its AP50–95 is 12.106640226%, which is about 0.047608 percentage points higher than standard QAT, but the original fixed 0.25 combined rule still fails.

These thresholds are derived post-hoc from the same repeatedly used development set and represent only diagnostic ideal operating points. They are not written back to deployment parameters, no success announcement is made after development set tuning, and no new scorer is fitted. The minor gains of the native scorer cannot prove the contribution of the intervention selection or independent generalization.

# # 3. Empirical Evidence on Training Intensity and Direction

Among 2840 full training updates in the original training, conventional QAT had 45 instances where the gradient norm exceeded a fixed clipping value of 10, approximately 1.58%; the correction group had 2598 instances, about 91.48%. This is a descriptive statistical summary from existing logs, and clipping alone does not imply training errors.

The batch mean of the additional loss in the correction group is 0.692532. The localization component accounts for approximately 97.22% of the scalar value. **The loss value does not directly represent gradient contribution**, and thus an alternative measurement was frozen: take the first 32 images from the pre-arranged training set at the old epoch 1, and measure once at each of two fixed checkpoints at epoch 10, totaling 64 image forward passes and 16 batches, without optimizer updates or weight modifications.

The total reconstructed loss value for each batch matched strictly with the native correction implementation; the hash values of the model states before and after the measurement were identical. No re-runs of model inference or AP results were performed, and Selection/Holdout sets were not accessed.

The measurement revealed that:

- At each checkpoint, 2 out of 8 mini-batches exhibited additional localization gradients opposing the direction of normal detection gradients.
- At the correction checkpoint, 3 out of 8 mini-batches had additional localization gradients opposing normal classification gradients, and 6 out of 8 had additional suppression gradients opposing normal classification gradients.
- Under the original 4-batch accumulation scheme, the correction model's additional localization-gradient norms in two windows are 1.07 and 1.97 times the normal total detection-gradient norm. These ratios already include the original 0.25 weight.
- 4 additional localization and normal detection gradient cosine values are positive across cumulative windows, and thus small-batch local conflicts cannot be construed as the entire training process having been performing backward updates.

Therefore, the next round of validation will prioritize checking the scaling and interference of auxiliary terms, with clear measurement support; however, the first-order local quantities from 32 training images and the final checkpoint cannot be used to prove that they caused the entire historical AP decline, nor do they include the full historical effect of SGD momentum.

# # 4. Only One Modification Will Be Validated in the Next Round

The 10th-round training of the conventional QAT will be retained as the common starting point for the three groups. Drawing on gradient protection concepts from prior multi-object optimization, two actions will be applied to the additional correction component:

1. Remove components of the additional gradient that conflict with the total normal detection and normal classification objectives; use a well-defined bi-constrained minimum distance projection to avoid reintroducing conflicts through sequential projections.
2. Limit the norm of the processed additional gradient to no more than 25% of the norm of the normal detection gradient. This ratio is a pre-defined engineering assumption for the next round and is not the optimal value derived from a search over the development set.

This constraint applies only to the first-order auxiliary direction of the current batch and does not guarantee that the actual update of every true target, the final AP, or the momentum term will not deteriorate. The original TAL objective, ignore rules, precision mask, quantization range, and inference structure remain unchanged.

The three groups are: conventional QAT continues training, difficult example selection with the same protection rule, and intervention-based selection with the same protection rule. The latter two groups match category and scale bins with error type and confidence bins; no guarantee is made that the number of effective gradients or wall-clock time per step is identical, and each step must be individually recorded. This allows us to distinguish whether the 'protection rule is useful' from whether 'our selection information has additional value'.

Recommendation for limited budget: 4530 existing training images, three arms with 3 epochs each, 852 optimizer updates per arm, fixed at epoch 3; same starting point, same sequence, batch4/ cumulative 4, same optimization settings, with individual optimizer resets. No retraining of teacher model, no selection of best epoch. Total of 42012 forward passes through training/evaluation images, with reconstruction saved for 9 additional instances, three AP tasks. This is a **proposed and qualification-check scheme**, not yet frozen into an executable training contract, and no training has been initiated yet.

Prior to training entry, the following conditions must be met: projection within mathematical boundaries, replay of original loss, consistency with the ordinary QAT under zero auxiliary conditions, ignore protection, final batch accumulation, and saved reconstruction checks; only a single pre-registration comparison is conducted, without automatically adding candidates or relaxing original thresholds. The conditions for continuing training of the conventional QAT are: AP improvement, no reduction in the fixed 0.25 detection rate, and no increase in false positives; if only improvement in ranking under the same false positive budget is achieved, it must be classified as partial development gain. Only when outperforming the same protection rule in the difficult example group can the value of intervention information be further discussed.

If the experiment fails, the expansion of this corrective training path will be halted; if it passes, independent data, sufficient training validation, a published strong baseline, and real backend verification are still required. The current 414-image development set is insufficient to support independent scientific validation.

# # 5. Relationship to prior work

[Gradient Surgery for Multi-Task Learning / PCGrad](https://arxiv.org/abs/2001.06782) Existing gradient projection ideas; the protective rules mentioned above are adapted and referenced, not a novel contribution introduced from scratch, nor a complete reimplementation of PCGrad. [HQOD](https://arxiv.org/abs/2408.02561) Prior research on classification-localization coordination in quantized object detection. "Limiting interference" and "coordinating the two tasks" alone are insufficient to establish novelty.

This project must still demonstrate that the target selection provided by precision intervention delivers stable and worthwhile gains under the same protection rules, reasonable hard examples, and task-aware baseline. If this point is not established, the primary contribution should be abandoned rather than merely rephrased.

# # 6. Evidence and Boundaries

Saved diagnosis: `../artifacts/gate1/20261027_failure_mechanism_diagnostics_v1`; training gradient measurement: `../artifacts/gate1/20261027_train_gradient_probe_v1`. New reports and status use newly generated files; previous reports, contracts, weights, and delivery lists remain unmodified. Saved diagnosis passed, local measurement passed, method joint failure, development AP, actual INT8, and independent scientific acceptance are separately recorded.
