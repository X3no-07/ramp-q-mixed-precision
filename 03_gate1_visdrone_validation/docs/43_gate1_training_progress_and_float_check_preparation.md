# Gate1 Training Progress and Floating-Point Check Preparation



As of 2026-10-22T14:00:36+08:00, the first-round candidate training has completed **20/30 epochs**, and the task continues to run. Actual batch=4, AMP is disabled, and no automatic recovery events have been observed. This progress completes training audit and floating-point checking tool preparation. Training is not yet complete, and neither the floating-point baseline nor the Gate1 scientific conclusion has been accepted.

[Current Status Snapshot](../../docs/DATA_AND_ARTIFACTS.md), [Real-time Training Logs](../../docs/DATA_AND_ARTIFACTS.md). Status files are time-stamped snapshots; future progress is based on real-time logs and acknowledgments.

# # Post-Training Checks Are Ready

Two new tools have passed **26 synthetic tests** and 25 sub-cases; no intermediate weights were loaded for evaluation in this round.

| Tool | Check Content | Current Boundary |
|---|---|---|
| Full Training Audit | 30 consecutive epochs with continuous records, per-epoch state checks, parameters, data copies, checkpoint SHA, best epoch, and tie-breaking selection | The full training has not been executed; only receipt of training completion is accepted. |
| floating-point qualification | Same final best weights, 109 fixed debug images, unified preprocessing, category and bounding box reconstruction, FP32/FP16 original outputs, and alignment with project-level objectives. | 20 synthetic test cases have passed; real-time inference has not yet been performed. |

Audit is conducted based on frozen source code, saved callbacks, and CSV evidence for the best epoch; the final checkpoint epoch after optimizer removal can be −1, and this field cannot independently recover the selection epoch. [Audit Preparation Completed](../../docs/DATA_AND_ARTIFACTS.md) records the scope and entry point of the audit; [Floating-Point Check Entry](../../docs/DATA_AND_ARTIFACTS.md) records the execution method of the floating-point check; [Source Code Verification](../../docs/DATA_AND_ARTIFACTS.md) retains actual source code snapshots, SHA hashes, and verification evidence.

Qualification uses the 109 images already involved in engineering debugging, rather than an independent confirmation set. Both models retain their unfused structure; only the computational dtype changes. FP32_reference is obtained by converting saved FP16 EMA weights back to float and cannot recover values rounded during training. The predefined box tolerances are rtol=0.005 and atol=0.5; score tolerances are rtol=0.02 and atol=0.002. Failures are retained without relaxing thresholds. Postprocessing fixes candidate confidence at 0.001, NMS IoU0.7, and max_det1000, then applies ignore rules and same-target matching using physical annotation-line identity.

These preparations and subsequent debug statistics do not equate to official VisDrone AP, Orin deployment validation, or performance conclusions. The original Gate0.5 numerical guardrail remains **FAILED_UNCHANGED**.

# # Framework Behavior Verified in This Round

The default training request max_det=300, initially set by the frozen framework code, is automatically increased at initialization to **902** based on the maximum number of object-level targets in the training and validation images; the actual effective parameters have been preserved. This adjustment is derived from the data and fixed source code, with no re-selection of hyperparameters. The training loader removes one fully duplicated positive label from each of three training images; the original annotations and prepared label files remain unchanged. The final audit will independently recalculate the original training labels 227425 and the loader-de-duplicated labels 227422, as well as the validation labels 28007.

[Effective Parameter Review](../../docs/DATA_AND_ARTIFACTS.md) retains source code identity, differences, and log excerpts. The built-in training metrics still use an unknown ignored-weight proxy; the subsequent check's 1000-detection limit is separately fixed and cannot implicitly inherit training parameters.

# # Next Execution Sequence

Complete 30 rounds of training → independently complete audit → floating-point qualification check on 109 debug images → separately review baseline availability range. Only after freezing quantization, candidate, budget, and statistics contracts will the Selection module begin scanning. The final confirmation set remains sealed.

This tool and documentation use a new path; [Previous Split and Training Report](42_gate1_split_freeze_and_initial_float_training.md) and existing deliverables are retained. Ongoing training logs are not included in the static list of archived items; the current receipt and tail log copies can be reviewed to verify the current state.
