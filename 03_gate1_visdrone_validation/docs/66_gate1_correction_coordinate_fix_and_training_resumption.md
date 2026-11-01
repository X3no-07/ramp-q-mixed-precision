# Gate1 Correction Training: Original Image Coordinate Bridging and Continuation



Recorded on 2026-10-26, Beijing time. This report documents launch conditions and does not prove training completion or successful results.

# # Known Issues

The first pilot stopped at 13:45:58. At that time, it had completed 128 new reference reconstructions, 64 images of range calibration, and 128 development evaluations across four untrained branches; training for the three arms had not yet been initiated.

The root cause was that the supervision sample construction passed scaled floating-point ignore boxes to the official original-image grid-matching program. This program requires the publisher to provide integer ignore regions on the original image, thus rejecting processing. This is a coordinate interface error, and cannot be resolved by rounding the boxes, relaxing scoring rules, or modifying the original annotations.

The original attempt, failure logs, frozen code, and completed outputs are preserved in `../artifacts/gate1/20261026_intervention_training_pilot_v1`. The official 30-epoch training and earlier audit conclusions are unaffected by this error.

# # Fix and Validation

The new version transforms predicted bounding boxes back to original-image coordinates and matches them against the already-bound original annotations. The positive example indices still correspond to the existing training labels; per-image verification of category, order, and transformed geometric positions is performed. The 0.002-pixel absolute tolerance in coordinate mapping is used solely for rounding checks of text labels and geometric transformations and does not alter the historical thresholds for full candidate box values.

- 10 test cases passed, including floating-point ignore box regression, CUDA forward equivalence, gradient updates, and output reconstruction.
- All 512 pre-selected training images passed the coordinate bridging validation between original annotations and training labels.
- Reuse of completed references, calibrations, four baseline branches, and the teacher array from the first image is preserved; these completed phases are not re-run.
- New contract SHA256: `9a9d4e7435a8bb9f15a7598a60b0de0da2945210204333b1b44940b7666c9fb0`

# # This Continuation Run

2026-10-26 15:11:59 Launching a new trial, launcher PID 31728, actual working process PID 7424. PID only records the identity at the time; upon recovery, the process creation time, parent-child relationship, command line, and live file must be re-verified.

The current directory is `../artifacts/gate1/20261026_intervention_training_pilot_v2`, with real-time status `attempt_01/live_status.json`. A new budget of 6014 image forward passes is allocated, with three arms each consisting of 512 images, 3 epochs, and 96 optimizer updates; the final state of each group is independently reconstructed 3 times, separately counted. The first version executes a total of 708 forward passes, of which 2 fail due to errors and are not saved; all consumed resources are retained as recorded.

The three arms remain standard QAT, training on matched budget hard examples, and precision-intervention correction training. The scope, training members, random seeds, optimizers, sequence, number of training epochs, and final candidate rules are all inherited from the first version, with no parameter search or optimal epoch selection introduced. The original weights and historical scores of the three arms are unchanged and not rewritten.

After completion, we verify the confirmation responses, status per epoch, member and source hashes, auxiliary sample budget, saved arrays, development metrics, and the full candidate box threshold. All outputs must clearly distinguish between consistency in saving, training completion, numerical thresholds, and development performance.

# # Explanation of Boundaries

This is a single-seed development pilot for an additional training round, not pure PTQ, real INT8 deployment, or independent scientific validation. The 128 training_val images are solely for development observation and do not involve any images or annotations from Selection, nor are they exposed to a Holdout set. The standard hard example group serves as a key control: only the relative changes of the correction group compared to this group can indicate whether the intervention provides additional value. Currently, we cannot claim a reduction in false positives or any publishable contribution based on this result.

Cloud GPUs may shorten subsequent training, but this round still includes CPU-based annotation matching, scoring, and file validation. Migration cannot resolve data interface issues. No cloud instances were rented, and no data or models were uploaded.
