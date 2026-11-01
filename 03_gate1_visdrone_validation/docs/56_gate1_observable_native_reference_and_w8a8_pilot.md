# Gate1: Observable native baseline and W8A8 pilot validation



With user authorization, two new stages have been advanced: a native baseline with per-image checkpoints, and a formal epoch 30 candidate W8A8 floating-point quantization simulation. New W8A8 integration, 8-image pilot, and independent result validation are complete; the 414-image native official baseline is still running and cannot be declared passed prematurely.

# # Completed engineering integration and pilot

The new integration uses the original epoch 30 FP32 EMA, with SHA still at `4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27`. The model maintains an unfused BN structure, with 88 Conv and 81 BN layers; the 640-input Conv MAC totals 3,160,115,200, with G1 accounting for 0.1031739602404368. Activation uses symmetric per-tensor, and weight quantization uses symmetric per-output-channel 8-bit quantization/dequantization; actual convolution operations remain in FP32. This is a QDQ simulation, not hardware INT8.

Synthetic integration passes 28 tests: strict consistency of the full output tree when QDQ is disabled, independent storage of model parameters between the original and packaged models, no alteration of parameters, batch normalization, buffers, or inputs. The pilot reads 64 fixed calibration images (without reading calibration labels) and 8 debug images in a fixed order after freezing the contract, executing four configurations: FP32, q_all, q_only:G1, and r_only:G1. It completes 120 forward passes and saves 32 original arrays. The 8 FP32 arrays match exactly with existing official candidate arrays; QDQ disable and state isolation checks pass. An additional 34 tests are passed during the preparation phase.

Independent audit reads saved arrays without re-running models or calibration: full verification of 268,800 candidate entries and 3,763,200 numerical values across 32 arrays; independent reconstruction of CPU NMS shows exact consistency in all detection boxes, scores, classes, anchor identities, and ordering; target matching and aggregation are fully consistent across 32 arrays; the 8 historical FP32 arrays are strictly consistent again. The size and SHA of 101 output list members remain unchanged before and after audit. Evidence of QDQ disable and state invariance comes from frozen source code assertions, runtime logs, and sequential progress—no repeated tensor states are stored, so it is not described as independent tensor state reconstruction.

The results for 8 images are descriptive only: among 1194 valid targets, FP32 achieved 442 detections, full QDQ achieved 450 detections, G1-only QDQ achieved 444 detections, and G1 recovery achieved 444 detections. Full QDQ introduced 20 additional missed detections compared to FP32, of which 8 were recovered via G1 recovery, while 6 new missed detections were generated, resulting in a net recovery of 2. The number is small and the targets belong to debug cases; therefore, validity, statistical significance, or superiority over other recovery strategies cannot be claimed based on this data, and no official AP is computed or reported.

The next step is to extend to the existing 109 debug images and three fixed 64-image calibration seeds, following the technical validation conditions of the pilot. Seed or candidate selection must not be based on recovery ratios. A new freeze of members, submission order, original post-processing, tolerance, and forward pass budget must be established prior to proceeding.

# # Newly introduced native attempts and execution boundaries

A native attempt of 1800 seconds with full retention is completed. A new observable packaging version v1 detects inconsistency between the MAT and standard decimal JSON transmission at the last digit on the first image, and stops with the current state preserved. Diagnosis shows that 72 entries in the prediction matrix differ by no more than 1.1368683772161603e-13; transmission discrepancies are not interpreted as official score inconsistencies, and zero tolerance is not relaxed.

Version v2 switches to IEEE754 hexadecimal transmission and freezes the new contract. Bit-level verification of 828 GT/prediction matrices from 414 images passes, with no change in logical members, geometry, row order, or values; the official scoring source remains the existing fixed version. Three probe members, pre-defined with ignore region metadata, complete 9 zero-tolerance comparisons of filtered matrices, showing strict consistency between MAT reload and hexadecimal round-trip. The probes only validate packaging and estimated cost, not replacing full data scoring.

In the full-scale phase, a bounded budget of 5248 seconds for pre-filtering and 7200 seconds for aggregation is pre-frozen, with no automatic retry. Native execution performs per-image ignore filtering and saves MAT/JSON checkpoints; after completing all 414 images, 1242 filtered matrices are first compared, followed by freezing the aggregation contract and calling a single original 414-image aggregation score. The final seven FP32 metrics still require `atol=1e-10` percentage points and `rtol=1e-12`. No chunked averaging, Python pre-filtering, or NMS subset substitution is used.

Additional read-only audit independently verifies 1,742,984 double-precision values across 414 members and 828 input matrices, bit-by-bit consistent; 29 base contracts remain unchanged, with no semantic blocking issues discovered. The audit also records low-probability race windows before timeout cleanup—unverified PID creation time, absence of per-image progress within aggregation, and the fact that MAT is reloaded natively in Octave rather than parsed by third-party Python—none of which can be described as verified full 414-image mAP.

This report is sealed at timestamp `2026-10-24T02:40:45.790061+00:00`. The latest runtime status is `RUNNING_NATIVE`, phase `full_prefilter`, with `135` images completed out of 414; actual subsequent progress is determined by live status, corresponding process, checkpoints, and final state logs. This report is only a time-stamped snapshot of progress, not a complete native result.

# # Preservation and Acceptance Boundaries

The engineering integration, pilot output, native new contract binding, and previous seal phase members have been re-verified; the old report, failures, original weights, and `phase_checkpoint_manifest.json` reference documents are preserved as-is. The original formal training and three-path development mAP conclusions are carried forward by reports 53–55; FP16 historical failures and `FAIL_FIXED_TOLERANCES`, `FAILED_UNCHANGED` remain unchanged.

The full candidate box threshold remains in place; QDQ output differences from FP32 are only described in this round and cannot be considered as passing floating-point qualification for shape, finite values, or NMS consistency. Selection uses only existing metadata; holdout continues to be sealed. Independent scientific validation, real INT8/TensorRT/Orin validation have not passed.

A half-hour check of `gate1` has been conducted for the recovery of actual work performed; normal operation does not trigger routine notifications, but alerts are issued for critical completions, failures, or actions requiring handling. The current autonomous operation ends and the instance is deleted to avoid idle polling.

Evidence for this report is found in `../artifacts/gate1/20261024_followup_progress_v1`; the new status is `current_execution_state_v17.json`. All previous versions are preserved.
