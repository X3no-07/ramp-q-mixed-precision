# Gate1 Standard Scorer: 286-Image Frozen Extension Validation Initiated



2026-10-26, at 19:57:18 CST, the pilot launch is initiated. User request for continued validation. Success or failure shall be determined by fixed rules of this round of empirical testing, without threshold adjustment to ensure consistency.

# # Frozen Objects

This experiment freezes score_base, the ordinary-control scorer from the previous fixed candidate, without refitting its 23 parameters or changing features or normalization. The existing untrained, MSE-range-calibrated G1-restored detector R and the existing epoch 3 standard-QAT model serve as controls. The three paths are R, R with the frozen scorer, and standard QAT.

New computations are performed on 286 training_val images. Each image runs only the existing detectors R and standard QAT, resulting in a total of 572 forward passes. The scorer reads the existing NMS candidate pool from R, preserving bounding boxes, categories, and membership, but only modifies scores and submission order. The outputs from the original 128 images are directly reused without re-running inference. The complete state fingerprint of both detectors before and after inference must remain identical.

# # Data and Interpretation Boundaries

| Scope | Count | Purpose |
|---|---:|---|
| new286 | 286 | Primary Expansion Development Comparison |
| new_prefix47 | 47 | 21 filenames with prefixes not present in the original 128-image metadata, pre-fixed auxiliary observations |
| full414 | 414 | Original 128 and new 286 images are merged in the original protocol order, as auxiliary observations |
| old128 | 128 | Previous development results are preserved as reported, without recalculating their full AP |

The new 286 images have no SHA overlap with the training members, and the original isolation rules for training/validation lists have been verified. Among the new 286 images, 239 still share prefixes from the original 128; an additional 47 come from 21 new prefixes. Therefore, the 286 images cannot all be considered as new scenarios, and the prefixes do not constitute independently validated scenario identities.

The ordinary-control scorer was selected using the earlier 128-image results, and these 286 images were also used in earlier floating-point development work. All 414 images remain training_val development data. The three evaluation scopes overlap and are not three independent scientific experiments. No Selection payload was read, and Holdout remains sealed.

# # A Practical Fixed-Threshold Check

The previous round's false positive budget curve was from a development evaluation, and the score threshold on that curve cannot be directly used as a deployment threshold. In this round, another set of scoring training candidates from the original 512 training images is used, with 213716 candidates. A fixed budget of nominal training false positives at 0.25 is applied, and the lowest feasible threshold is selected from the scoring curves of all groups with the same score.

The resulting frozen score threshold is **0.17119779099758897**. This calculation only reads the training cache and the frozen scorer, without accessing new development annotations or retraining any parameters. It forms a nominal budget based on the training data and cannot guarantee that the false positive count in new images will be exactly equal.

The new images will simultaneously report: the number of project matches at original R at 0.25, the scorer at this training threshold, and the ordinary QAT at 0.25. A standalone check for 'detection count not reduced and false positives not increased' is performed; if it fails, the threshold is not re-adjusted.

# # Fixed Comparison Rules and Budget

The primary rule compares the frozen scorer with the original R on new286: AP50–95 must improve, and the entire curve must maintain or improve detection count under the official false positive budget of the reference R. The false positive budget is fixed by counting FP at score ≥0.25 in the original R, using IoU0.5/cap500; the original R itself also takes the optimal TP under this budget across the full curve. All candidates at the same score level are included as a group. The threshold point remains part of performance evaluation and is not converted into a deployment threshold.

The same rule is reported on new_prefix47 and full414; the judgment of 'joint improvement across three ranges' requires all three ranges to satisfy the condition, without selecting a favorable subset. The relative AP and ranking differences compared to ordinary QAT are also reported, not just compared against a weaker baseline.

Three arms multiplied by three predefined ranges yield a total of 9 new aggregation scores. The scoring input matrix is bound to members, ranking, and source, with at most two lightweight CPU processes performing scoring, no additional model training, and no automatic failure retries.

# # Pre-check and Record

8 items passed pre-check, covering metadata splitting, image hash overlap rejection, batch-wise threshold alignment during training, frozen weight fingerprinting, lightweight scoring curve consistency, actual independent AP process not importing Torch, and scoring rejection under input variations.

Contract directory: `../artifacts/gate1/20261026_scorer_extension_286_v1`.

Contract SHA256: `29c5540942a167c698da6ceeb865fa3f404ea0fcca5dce7b9495af56f418957d`.

LauncherPID48680, actual working processPID52284, actual parent processPID41936. This entry serves only as a launch identity record; upon recovery, the process creation time, parent-child relationship, command line, live status, and final state acknowledgment must be re-verified. The one-time parent process is responsible for observing the actual working process and propagating its exit code.

The original FP32_reference for the additional 286 images was not re-constructed in this round; their full-candidate score qualifications are not authenticated. The earlier full 8400 candidate scores that failed are preserved as-is, and are not replaced by ranking or AP improvements. Model training completion, consistency in saving, development effectiveness, independent scientific acceptance, and real INT8 and board-end acceptance remain separate. No new cloud resources or scheduled checks are added.
