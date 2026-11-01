# Gate1: Native baseline timeout and stage closeout



**An additional 414 training_val FP32 native official baseline tasks timed out after 1800 seconds, and real-data native consistency has not been completed. The 30-round formal training, FP32 state preservation verification, three-path floating-point inference, and Python development set AP all maintain their original conclusions.** No retraining, model inference, scoring, or native task was rerun, and no changes were made to the source code, contracts, data, old weights, or tolerances.

# # Actual outcome of the native task

The native Octave process started at 2026-10-24 02:54:48 Beijing time. At 03:24:48, the outer console recorded `subprocess.TimeoutExpired` and `1800.0 seconds`. Inspection confirmed that the process had terminated, neither `oracle_results.json/.mat` existed, and the native console contained 0 bytes. The launcher receipt recorded `temporary_alias_removed=true`, and the actual R: alias was also absent.

The launcher only writes a completion status on the normal return path. The timeout exception is cleaned up by `finally` and then thrown outward, so `RUNNING_NATIVE_OFFICIAL_CODE` in the response is an obsolete field. The original response and v15 snapshot are preserved in their original form, corrected by this report and v16 for the new status; it is not valid to use this outdated field to report the running status, nor can the process disappearance be used to report success.

[Outer control console](../../docs/DATA_AND_ARTIFACTS.md), [original launcher response](../../docs/DATA_AND_ARTIFACTS.md), [current process and alias on-site](../../docs/DATA_AND_ARTIFACTS.md).

The full comparison originally required element-wise zero-tolerance consistency across 1242 filtered matrices for 414 images, and seven FP32 metrics satisfying `atol=1e-10` percentage points and `rtol=1e-12`. Due to the absence of native outputs, the matrix comparison count is 0 and the metric comparison count is 0, and the comparison script that requires successful native results has not been executed. The timeout proves that the execution budget was not completed, and thus cannot be used to determine whether the Python results are consistent or inconsistent with the official native results.

# # Post-timeout source and result verification

| Verification scope | Result |
|---|---:|
| Original stored list members | 684/684 size and SHA match |
| Saved inference list members | 640/640 consistent |
| Octave runtime members | 2890/2890 consistent |
| Official scoring source file | 11/11 consistent |
| Previously audited FP32 capture receipt bound to audit | 30/30 consistent |
| Per-epoch FP32 saved list members | 150/150 consistent |
| Native input image members, dimensions, and order | 414/414 consistent |
| Native input raw GT rows | Line 29081 matches the original training_val annotations value by value. |
| Native input FP32 prediction rows and order | 188792 row-wise predictions match the stored predictions exactly |

Three Python scoring input indices have been re-verified against the same set of saved predictions and ground truth identities. A total of 3747 distinct binding files have been cross-checked, and an additional 414 authorized training_val original annotations have been read and verified. Model outputs, numerical statistics, or AP values have not been recomputed; previous independent CPU/CUDA reconstructions and 109 image array/NMS recalculation evidences are carried forward from the original audit and archived files.

Final FP32 state SHA remains `4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27`. The original `phase_checkpoint_manifest.json` and all referenced files were not modified.

[Read-only preservation and verification of results](../../docs/DATA_AND_ARTIFACTS.md), [this stage's machine decision](../../docs/DATA_AND_ARTIFACTS.md).

# # Performance clues and still unknown aspects

Source inspection shows that the original `dropObjectsInIgr.m` invokes `createIntImg.m` on images with category=0 ignored regions, which uses nested pixel loops. Out of 414 images, 234 trigger this branch; if fully executed, the integral image covers 292,261,200 pixels, with the inner loop accumulating 291,735,334 iterations. This is a statically derived workload based on fixed dimensions and original code, supporting subsequent cost estimation for execution.

This run lacks per-image progress tracking or intermediate checkpoints, and the empty console cannot identify which image or function is causing the timeout. Previous CPU progress indicates computation was still ongoing at that time, so this timeout cannot be considered a fully localized deadlock, nor can we confirm that the pixel loop is the sole performance bottleneck.

# # Retained development outcomes and acceptance boundaries

| Pathway | Original full bounding box threshold | training_val AP50–95(%) |
|---|---|---:|
| FP32 | Comparison reference | 12.440710037775458 |
| Full FP16 | 109 debug images and all 414 training_val images failed | 12.38260103959871 |
| Main FP32, Detection Tower FP16, Decoding FP32 | All 523 images passed | 12.438217665845206 |

Detailed evidence remains in [Report 53](53_gate1_formal_training_completion_and_fp32_state_audit.md) and [Report 54](54_gate1_formal_candidate_float_checks_and_development_ap.md). The description 'native still running' in Report 54 refers to a historical state at the time of sealing and is now superseded by this report. The Python image passed with a 32-configuration native comparison, which remains valid, but does not fully establish native consistency with actual data in this run. The training_val scores are only for development validation; descriptive target-level diagnosis cannot be equated with INT8 experiment results.

Gate1 independent scientific acceptance, and the INT8/TensorRT/Orin board-level acceptance of this candidate remain incomplete. Selection continues only with existing metadata; Holdout remains sealed; no images, labels, or detection results were read in this run. Historical `FAIL_FIXED_TOLERANCES` and `FAILED_UNCHANGED` are preserved as-is.

# # Conditions required for continuation

The next native attempt must first establish a new bounded duration or availably native environment, and then freeze the call wrapper with per-image progress and intermediate checkpoints in a new directory. The official functions, 414 members, the same FP32 submission, all 1242 matrices, and seven metrics must be preserved within native tolerance; partial images or Python pre-filtering results cannot substitute for a complete native baseline. This round does not involve blind retries, nor does it expand the original 1800-second budget.

[Next execution draft](../../docs/DATA_AND_ARTIFACTS.md) for `DRAFT_NOT_FROZEN_NOT_EXECUTED`. The independent scientific phase must separately freeze data access, candidate, and statistical decision contracts. The device connection status of this candidate was not checked in this round, and historical connection information cannot be used to claim current deployability.

The monitored target has ended; further polling will not yield new results and the half-hourly check timer has been deleted `gate1`. The failure site preservation and source/result verification for this phase have been completed; remaining native execution dependencies are subject to new budgeting or environmental decisions.

[Full delivery verification](../../docs/DATA_AND_ARTIFACTS.md), [v16 snapshot](../../docs/DATA_AND_ARTIFACTS.md).
