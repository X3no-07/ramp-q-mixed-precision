# Gate1: Formal Training Completion and FP32 State Verification



**The formal 30-round training has been completed, and all 30 live FP32 EMA states have been successfully saved. The final state from the 30th round was strictly reconstructed in an independent process, with CPU and CUDA parsing of synthesized inputs and outputs matching exactly.** This passes the engineering verification of fixed training budget and FP32 state preservation chain, but does not constitute task AP, floating-point qualification, or Gate1 scientific acceptance.

Trained from Beijing Time 2026-10-23 23:43 to 2026-10-24 02:41, totaling 10673.11 seconds, approximately 2 hours 58 minutes. The user has authorized continued operation after training completion, and thus floating-point comparisons and training_val scoring have been advanced under the new freeze protocol.

# # Verified Results

| Item | Verified Results |
|---|---|
| Training Contract | Fixed single YOLO11n candidate, 30 rounds, 640, batch=4, nbs=64, AMP off |
| Data | 4530 train images and 414 training_val images; sizes and SHA digests of the 14832 image/positive-label/per-image-sidecar files in the new copy were rechecked |
| Actual Training Records | 33990 batches, with 1133 per epoch; the last batch in each epoch has 2 images, and all others have 4 |
| Update | 2557 optimizer and EMA updates completed; update count matches batch count only when warm-up and gradient accumulation are considered, resulting in fewer updates than batch count |
| classification ignore | 29707 batches actually mask at least one background anchor; all recorded batches retain the entire positive classification row |
| FP32_reference | 30/30 sets, 150 saved list members verified; all state-only snapshots validated under restricted loading |
| Final reconstruction | Strict state_dict loading at round 30; CPU/CUDA synthetic outputs and saved references match exactly |
| Runtime exception | No failure report; stderr is empty; source file hashes remain consistent; no automatic recovery or batch reduction |

Mask count and update events originate from the actual, frozen runner; audit confirms coverage, order, finite values, and mutual consistency; no full retraining or per-batch TAL is performed, thus event checks are not considered independent reproduction of each assignment. Data members and isolation relationships are verified against the original frozen split; no Selection detection outputs or Holdout content are used.

The 30th round row in the original training CSV has only 8 fields, while the header and first 29 rows have 11 fields. This is because the final framework validate returned empty metrics and did not write into the three originally unused validation placeholder fields. The original CSV is preserved; in the new normalized record, the last three fields are explicitly parsed as learning rate and are not treated as validation loss or AP; historical files are not modified.

[Training Completion Report](../../docs/DATA_AND_ARTIFACTS.md); [Full Audit Result](../../docs/DATA_AND_ARTIFACTS.md); [Normalized Training History](../../docs/DATA_AND_ARTIFACTS.md)

# # Fixed Final Candidates

Candidates are fixed at the 30th round FP32 EMA prior to training, not selected from debug, training_val scores, or framework best. The current save represents the actual FP32 training state; standard framework half-precision checkpoint is not considered a scientific reference. Only inference state, architecture, and source are saved; full optimizer recovery training is not validated based on this.

[Final FP32 state](../../docs/DATA_AND_ARTIFACTS.md) is of size 10579289 bytes, SHA-256 is `4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27`. [Round 30 capture acknowledgment](../../docs/DATA_AND_ARTIFACTS.md) records the CUDA synchronization prior to saving, FP32 origin, unchanged source state/number of updates, unchanged Torch RNG, and reconstruction checks between two backends at save time.

New training data handling, constrained augmentation, and saving mechanisms differ from the old 30-round candidate, thus new and old training metrics cannot be treated as a control experiment varying only one precision factor. The precision lost in the previous FP16 EMA is not recovered.

# # Post-Training Checks

Subsequent evaluation will use the [New Frozen Evaluation Protocol](../../docs/DATA_AND_ARTIFACTS.md), with a fixed number of 109 debug images and 414 training_val images/submission order, comparing the full FP32 and full FP16 results under the same final state against previously established paths involving 'subject-level FP32', 'detection tower FP16', and 'decoder FP32'. All 8400 candidate bounding boxes' original numerical thresholds will be compared; NMS and task AP are handled as independent subsequent steps. training_val provides development validation evidence only and is not considered a standalone out-of-sample validation.

The previous full FP16 check remains `FAIL_FIXED_TOLERANCES`; the previous Gate0.5 numerical record remains `FAILED_UNCHANGED`. This round does not read Holdout, does not perform Selection inference, and does not include INT8, TensorRT, Orin speed, memory, or power consumption conclusions.
