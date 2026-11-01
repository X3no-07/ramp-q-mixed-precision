# Gate1 Constrained Correction Three-Arm Training Results



Generated time: 2026-10-28T08:14:23.316863+00:00 (UTC).

Three arms start from the same ordinary QAT state at iteration 10, each continuing for 3 iterations, with identical data, seed, and update steps; the candidate fixed arm is set at iteration 3. Two correction arms match the sample group count and employ the same gradient protection. The ordinary QAT computes backward pass once, while the constrained arms compute gradients for three components, and thus no claim is made regarding identical training FLOPs or wall-clock time.

| Branch | AP50–95(%) | Fixed detection threshold of 0.25 | Fixed false detection threshold of 0.25 |
|---|---:|---:|---:|
| ordinary_qat_continuation | 11.985899444 | 10552 | 7728 |
| guarded_matched_hard_examples | 11.918156643 | 10486 | 7494 |
| guarded_intervention_examples | 11.877794522 | 10443 | 7494 |

Joint condition for intervention group versus standard QAT: False; joint condition for intervention group versus hard examples under same protection: False.

Completed 9 instances of numerical/KKT validation for FP32 states and 1704 protection windows, 1242 full raw/NMS, 1242 prediction/object matching, 2484 AP input matrix zero-tolerance validations, and 3 AP job exits 0. Passing the validation does not equate to method acceptance.

The protection constraint applies only to the current additional gradient; it does not guarantee outcomes for each ground-truth target, SGD momentum history, or AP after finite updates. The 414 training_val images are repeatedly reused for development. Single-seed continuation for 3 epochs is neither convergence nor independent scientific acceptance. HQOD has not been reproduced, and actual INT8/TensorRT/Orin execution has not been validated.

The original all 8400-candidate numerical gate is rechecked only on the 128 images with old FP32 arrays. The other 286 images have no additional FP32 inference, so complete numerical equivalence over 414 images cannot be claimed. Selection/Holdout remain sealed, and historical failures are retained.

This round does not automatically append additional training, thresholds, seeds, or candidates. The final official state must simultaneously read supervisor_terminal_receipt.json and delivery_validation.json; judgment based solely on this report or process termination is invalid.
