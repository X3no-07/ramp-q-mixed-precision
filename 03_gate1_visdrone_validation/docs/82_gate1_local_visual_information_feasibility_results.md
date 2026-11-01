# Gate1 Local Visual Information Feasibility Diagnosis Results



2026-10-26.

**The fixed local visual linear probe in this round does not simultaneously satisfy the diagnostic condition of 'improving TP under the same false positive budget as the original scorer and reducing soft quality BCE'; therefore, this candidate cannot be claimed to have improved performance based on this result.**

This validation addresses whether adding local visual features from the model can supplement information beyond the original score and bounding box statistics. Even if the results are favorable, it does not directly answer whether the 414 development images, an independent test set, originality, or real INT8 performance have been improved.

# # Results on Fixed 178 Images

| Branch | Same R False Positive Budget TP | Actual FP | Per-Image Weighted Soft Quality BCE |
|---|---:|---:|---:|
| Official R@0.25 | 6003 | 3892 | 0.150666550 |
| Original 22-Dimensional Standard Scorer | 6098 | 3892 | 0.146720612 |
| New 150-Dimensional Local Visual Probe | 6042 | 3892 | 0.145613182 |

The relative performance of the new probe compared to the original scorer shows a change in TP within budget of **-56**, and a change in BCE of **-0.001107430**. The preconditions require the former to be strictly greater than 0 and the latter to be strictly less than 0; these conditions were not simultaneously satisfied in this case.

The false positive budget is fixed at the official IoU0.5/cap500 FP count from the original R@0.25, with all groups treated equally and not split by score. The curve uses the existing official matching and ignores submission rules. This is a ranking curve diagnosis using calibrated labels, and the selected points do not equate to validated deployable thresholds. No new deployment threshold is frozen in this round, no full AP score is added, and the TP numbers in the table cannot be interpreted as development set AP or independent scientific validation.

# # What Actually Changed

The detector was completely frozen, remaining in its original 30-epoch FP32 state with the MSE-range R path (G1 retained floating-point precision, all other fused Conv layers were kept in floating-point QDQ). A single forward pass at the 16th layer P3 output produced a 64×80×80 tensor. The center of each candidate bounding box was sampled using bilinear interpolation to extract a 64-dimensional vector. Additionally, an 8-point average was computed around the bounding box with a 0.75 scaling of width and height to form a 64-dimensional center-minus-neighborhood mean. The resulting vector was concatenated with the original 22-dimensional features, forming a linear scorer with 151 parameters. The surrounding features are not verified pure background; thus, they cannot be used as prior knowledge to eliminate targets.

Retained the original 512 images, 213716 candidate soft quality labels, with only one 100-step CPU float64 Adam fitting. Equal-weight per image, normalization only applied to the image itself, zero-residual initialization, lr0.03, betas0.9/0.999, eps1e-8, L2 coefficients 0.0005; no hard example or precision intervention auxiliary loss added, no candidate selection, feature layout, or optimal epoch tuning. Combined with the previous 6 scoring candidate models, a total of 7 candidates are now accumulated, and all previous candidates are retained.

690 new forward passes were used to extract previously un-saved P3 features, without claiming any zero-forward cache experiment. The two original preprocessing pipelines—on the 512 fitting images and 178 calibration images—were preserved. The complete 8400-candidate raw arrays for all 690 images were bitwise identical to historical R. The detector's saved state hash remained consistent before and after. Therefore, the observed score differences stem solely from the new scorer, not from modifications to bounding boxes, classes, the detector, NMS, or image membership.

# # Data and Interpretation Boundaries

The 178 calibration training images and 512 fitting images have no overlapping IDs, content SHA, or filename prefixes; however, both sets are drawn from the original training data. The 178 images were already used in the prior scorer calibration, and this round is assumed to be influenced by existing development diagnostics. This can only be described as 'prefix-non-overlapping information diagnostics'—it cannot be called a fully independent test. The prefix is also not a validated independent scenario identity.

This round does not load the 414 training/val development images, their labels, or detection results, nor does it compute a new AP. The Selection and Holdout loads were not read. The original all-anchor qualification failures and historical failure records were preserved as-is. The consistency of the original arrays in this round does not constitute a reclassification of these failures as passes.

Local visual quality scoring belongs to an established research direction: [IoU-Net](https://arxiv.org/abs/1807.11590) learns localization quality, [GFLV2](https://arxiv.org/abs/2011.12885) studies distribution statistics and quality ranking. This work is not a complete reproduction of these methods, nor does it package ordinary local feature linear probing as an original contribution.

# # Independent Review and Delivery

6 items passed pre-check. The main task exited with exit code 0, no stderr, no failure acknowledgment, and the actual process terminated. Independent audit verified 873 new saved members and 2070 existing cached bindings; all 690 local features were re-computed using independent NumPy bilinear interpolation, with a maximum absolute difference of 2.09251429e-05; element-wise comparisons were performed against the previously frozen atol=1e-4 and rtol=2e-5. This is only the CPU-based tolerance for newly added features and does not alter the original full 8400 candidate value threshold.

An independent audit was conducted on the fitting normalization, 150-dimensional scores for the 178 images, original 22-dimensional reference, original candidate identities, order, and soft quality labels, BCE loss, ignore filtering matrix, IoU0.5 matching, the full same-score curve, and the original R false positive budget. No model re-fitting, detector execution, or full AP computation was performed during the audit. The audit passing only indicates that the saved computations are reproducible.

This fixed local visual linear probing does not simultaneously satisfy the diagnostic condition of 'outperforming the previous scorer under the same false positive budget and achieving lower quality BCE'. Therefore, this candidate cannot be claimed to have improved based on this result.

This fixed candidate's budget has ended. There is no evidence supporting the direct addition of a larger classification head, more visual features, or a seed-based pursuit to pass. A clear, distinguishable research mechanism and independent validation scope must be re-established before proceeding. Independent scientific acceptance, true INT8, Jetson, and Gate1 as a whole have not been achieved.

Evidence Directory: `../artifacts/gate1/20261026_local_visual_information_probe_v1`.
