# Gate1: Formal Floating-Point Candidate Check and Development Set AP



**The three floating-point paths of the final epoch 30 FP32 EMA completed actual inference on the fixed 109 debug and 414 training_val images. The FP32 body / FP16 detection towers / FP32 decoding path meets the original numerical gate on all 523 images; full FP16 fails on all 523 images.** Complete Python scoring on the 414 training_val images is finished, with FP32 AP50–95 of 12.440710%. These are development-validation results; Gate1 scientific acceptance has not passed.

The independent reconstruction results of 30 epochs of training and the final weights are presented in [Report 53](53_gate1_formal_training_completion_and_fp32_state_audit.md). The model, image members, preprocessing, all bounding box tolerances, NMS, submission order, and candidate budget were first written into [Evaluation Protocol](../../docs/DATA_AND_ARTIFACTS.md) before execution, and no adjustments were made to pathways, thresholds, or data based on the current results.

# # Full bounding box numerical results

The reference is the FP32 EMA saved from the new training. All three models are strictly reconstructed from restricted state-only files; the full FP16 branch converts the same state to half-precision, while the mixed-precision pathway uses a previously implemented fixed tiling scheme. No optimal layer search was performed in this round. None of the three branches fuse the network or load historical saved features as inference inputs.

Compare the 8400 positions of 4 bounding box estimates against 10 class scores. Box tolerance is `0.5 + 0.005*abs(FP32_reference)`, score tolerance is `0.002 + 0.02*abs(FP32_reference)`, and no low-confidence candidates are removed.

| Data Role | Reference Pathway | Number of Passed Images | Bounding box element exceeds tolerance | Scoring element exceedance |
|---|---|---:|---:|---:|
| debug (109 images) | Full FP16 | 0/109 | 6769/3662400 | 138/9156000 |
| debug (109 images) | Main FP32, Detection Tower FP16, Decoding FP32 | 109/109 | 0 | 0 |
| training_val (414 images) | Full FP16 | 0/414 | 31788/13910400 | 258/34776000 |
| training_val (414 images) | Main FP32, Detection Tower FP16, Decoding FP32 | 414/414 | 0 | 0 |

The mixed path has maximum absolute box differences of 0.398315 and 0.437622 pixels on debug/training_val, respectively; full FP16 has differences of 23.011322 and 30.217361 pixels. Passing the mixed path does not constitute acceptance of full FP16 and does not change the full-FP16 failure record.

The original three-branch arrays from the 109 debug images have been saved. An independent program has re-computed all tolerance counts, and 327 branch/image CPU NMS and coordinate reconstructions have been replayed from the saved arrays, with full consistency in outputs and anchor identities. For training_val, only numerical verification records and predictions are saved; the original arrays are not preserved. The original threshold statistics for training_val come from a frozen runner and backfill; the independent array recalculations in this round do not cover these 414 images.

[Inference Backfill and Summary](../../docs/DATA_AND_ARTIFACTS.md); [debug independent arrays/NMS validation](../../docs/DATA_AND_ARTIFACTS.md)

# # AP on 414 Development Validation Images

Scoring uses a complete Python image mirror verified against the official commit `005445782213e20cb91bc50a597db3dd949e749a`, preserving original uncropped GT, category0 pixel ignoring filtering, GT score inversion, cross-image sorting, and official aggregation behavior. Inputs are processed in image_id dictionary order, with per-image predictions sorted by confidence descending and NMS order preserved for ties. Predicted categories 0 to 9 are mapped to official categories 1 to 10.

| Pathway | AP50–95(%) | AP50(%) | AP75(%) | AR500(%) |
|---|---:|---:|---:|---:|
| FP32 | 12.440710 | 23.782556 | 11.467493 | 21.546395 |
| Full FP16 | 12.382601 | 23.732494 | 11.363852 | 21.473534 |
| Main FP32, Detection Tower FP16, Decoding FP32 | 12.438218 | 23.780935 | 11.465271 | 21.537962 |

Mixed-precision pathway shows a relative AP50–95 drop of −0.002492 points compared to FP32, while full FP16 shows a drop of −0.058109 points. The small AP differences do not invalidate the failure of the full bounding box numerical thresholds; these differences lack statistical significance, cross-seed, or independent generalization meaning.

NMS with fixed confidence > 0.001, IoU = 0.7, and a maximum of 1000 detections per image; three branches each have 26 training_val images reaching this limit. The official scoring first truncates the top 500 full-category predictions by submission order before evaluating per class, without changing to 500 per class. Confidence, NMS, and the upper limit are fixed evaluation settings for this candidate and cannot be claimed to cover any arbitrary submission strategy.

[FP32 score](../../docs/DATA_AND_ARTIFACTS.md); [full FP16 score](../../docs/DATA_AND_ARTIFACTS.md); [mixed-precision path score](../../docs/DATA_AND_ARTIFACTS.md).

The Python image previously passed through a 32-combined native baseline. In this round, a new [native baseline contract](../../docs/DATA_AND_ARTIFACTS.md) was frozen for 414 real training_val images with the same FP32 predictions, using the official functions and the verified Octave 11.3.0 runtime. **At the time of this report, the native baseline is still running and has not yet produced or been compared against its final results; therefore, the native consistency on real data for this round cannot be stated as passed.** The per-pixel ignore region integration loop in the native source code is computationally intensive, and the background task is still ongoing; subsequent conclusions will be based on independent native comparisons.

# # Descriptive diagnosis for same-targets

Additionally, a descriptive matching contract was frozen prior to execution for the current saved predictions: training_val, confidence ≥ 0.25, and same-class IoU ≥ 0.5, using the project's existing IoU-priority matcher. This matcher is not the official AP algorithm; this does not affect the original numerical thresholds.

| Pathway | Assessable targets | hits | False positives |
|---|---:|---:|---:|
| FP32 | 27989 | 10590 | 7254 |
| Full FP16 | 27989 | 10584 | 7287 |
| Mixed-precision path | 27989 | 10591 | 7258 |

After per-target alignment, there are 57 cases where FP32 detections are matched but FP16 misses them, of which 55 are detected by the mixed-precision path. The mixed-precision path still misses 4 FP32-confirmed targets and incorrectly detects 5 FP32-missed targets. This is a floating-point path diagnosis on the same development set, not an INT8-added miss or recovery experiment, nor a conclusion about the G1 module causality.

[Per-target diagnosis and contract](../../docs/DATA_AND_ARTIFACTS.md).

# # Current Boundary and Next Step

Training completion, FP32 state preservation, numerical qualification, development set AP, and native score consistency are recorded separately. Currently, we are still waiting for the finalized native score results and will perform a freeze-tolerance comparison before sealing the stage results. Independent validation and candidate/statistical contracts must remain frozen in the subsequent scientific phase; Selection remains within its metadata scope, and Holdout continues to be preserved.

The original full FP16 record remains at `FAIL_FIXED_TOLERANCES`, and the previous Gate0.5 numerical record remains at `FAILED_UNCHANGED`. There are no Selection detection scores, Holdout content access, INT8, export, TensorRT, Orin, speed, or power consumption measurements.
