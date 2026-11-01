# Gate1: Dimensions, Thresholds, and Dependency Diagnostics



**Diagnostic and independent revalidation of the saved results from the 546 development set have been completed. In all three sub-sets and three threshold configurations, small-object recovery is positive, while stable false positives and false negatives are simultaneously observed.** This is an interpretive analysis added after reviewing the development results and does not represent new holdout results, pre-registered scientific validation, or threshold selection.

# # What Was Done

No retraining, model inference, calibration, NMS, or AP computation was performed, and original images, original labels, and Holdout were not read. The primary threshold 0.25 reuses existing matching states; thresholds 0.20 and 0.30 invoke the source-fixed ignore-aware project matcher only on saved GT and predictions, totaling 10920 calls. F is matched once per image at each secondary threshold and reused by all three seeds. All seeds, configurations, and thresholds are retained.

The sole attempt was run at Beijing Time 2026-10-25 23:29:25 to 23:33:37, returning 0. The new contract has a time limit of 3600 seconds, an output cap of 300 MB, and a minimum available space of 2 GB; these parameters apply only to the computation of the current saved results, while the original model's task retention line of 20 GB remains unchanged. The source binding includes 11466 saved input files, which were cross-checked before and after execution, with 0.25 individual entries reconstructed from the previous summary.

During preparation, it was discovered that the v2 script was mistakenly renamed an old input contract to contract_v2.json. The original error is preserved and the source file remains unchanged. A v3 correction was created, passed 13 pure tests, and then frozen. This is a preparation-time correction and does not involve re-running the model or any diagnostic attempts.

# # Target Dimensions and Primary Threshold Results

First, full-image matching competition is maintained, followed by stratification based on original, uncropped width × height according to stable GT identity. Small area < 1024, medium area [1024, 9216), large area ≥ 9216; these are project-defined stratifications and do not constitute official AP_S calculations.

| GT dimensions | Number of valid targets |
|---|---:|
| small | 26358 |
| medium | 11092 |
| large | 1068 |

Net recovery is calculated only within the FP32 original detection range for 'recovery minus stable false positives'. Changes in R-Q full detection must include changes in targets not originally detected in FP32; these two components cannot be mixed.

| seed | Additional misses on small objects (A) | recovery | Stable By-Pass Loss | net recovery | same-target bidirectional agreement | R-Q full detection on small objects | Stability retention rate on small objects |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026102201 | 589 | 360 | 224 | 136 | 142 | +105 | 96.383597% |
| 2026102202 | 643 | 348 | 230 | 118 | 121 | +108 | 96.254072% |
| 2026102203 | 575 | 352 | 199 | 153 | 144 | +122 | 96.794459% |

Stability retention rate on small objects under the main threshold is approximately 96.25%–96.79%, not lossless recovery. There is currently no independent acceptance contract, and thus the proposed thresholds from the prior draft cannot be used to justify scientific PASS or retroactive validation.

# # Results across all three thresholds and false positive cost

False positives lack corresponding ground truth identities, and are therefore listed separately as global FP and FP stratified by predicted bounding box area. Predicted size false positives are not treated as ground truth size mismatches. The FP changes in the table are based on full-size projects, not the official AP curve false positive metrics.

| seed | confidence | Net recovery on small objects under constraint | R-Q full detection on small objects | R-Q false positive on full-size objects |
|---|---:|---:|---:|---:|
| 2026102201 | 0.20 | 132 | +100 | +366 |
| 2026102201 | 0.25 | 136 | +105 | +206 |
| 2026102201 | 0.30 | 125 | +93 | +154 |
| 2026102202 | 0.20 | 166 | +153 | +433 |
| 2026102202 | 0.25 | 118 | +108 | +317 |
| 2026102202 | 0.30 | 113 | +110 | +250 |
| 2026102203 | 0.20 | 164 | +102 | +650 |
| 2026102203 | 0.25 | 153 | +122 | +460 |
| 2026102203 | 0.30 | 158 | +121 | +278 |

All thresholds are retained; the secondary results are not used to replace 0.25. IOU priority matching may experience reordering during threshold changes, and it cannot be assumed that each target state evolves monotonically. The AP calculation is not recalculated in this study; the full 10 AP results from report 62 and their adverse gaps are preserved at their original values.

# # Component resampling is solely for development diagnostics

75 risk-prone proxy components, fixed PCG64 seed 20261025 with 10000 multinomial pairing resampling iterations; the same weight matrix shares all seeds, thresholds, and ground truth dimensions. Pre-fixed 2.5% and 97.5% percentiles, with null assigned for zero denominators and the valid/invalid counts preserved. Three calibration seeds use the same target set without increasing independent sample size.

| seed | 0.25 per thousand effective small object R-Q detection changes | Exploratory proxy component percentile range | One net recovery per thousand small objects |
|---|---:|---:|---:|
| 2026102201 | 3.983610 | [0.520361, 7.553710] | 5.159724 |
| 2026102202 | 4.097428 | [1.955890, 6.626266] | 4.476819 |
| 2026102203 | 4.628576 | [2.222410, 7.023840] | 5.804689 |

Even if the resampling ranges of these proxies do not cross zero, overall significance or scientific acceptance cannot be claimed: the components are not independently validated real-world scenarios, and the analysis is post-hoc, conducted after observing development outcomes. The recovery rate/limitation net_recovery_rate uses A as the denominator, maintains a stable retention rate denoted by S, and separately reports the global hit effect per thousand eligible GTs along with the net recovery; conditional ratios are not used to substitute overall changes.

# # Independent verification and cross-check

Individually verify 38518 GT area mappings, 4914 per-image counting records, 180 event index arrays, and 2700 component dimension units; reconstruct 60 primary threshold event arrays from the previous state, and achieve consistency via integer reduction of 180 interval units.

Revalidation does not invoke the secondary matcher again; it passes verification based on area, event, count, component, shared weights, and interval preservation results, without claiming a second matching implementation. The original 8400 candidate box numerical records are retained in full; all nine QDQ branches remain 0/546 images passed, and neither size subsets nor threshold subsets can replace the original criteria.

# # Next Specific Preparation

Fix the independent data estimation for G1, separating it from the claim that 'model selection outperforms random or traditional methods'. Currently, a limited Holdout audit is proposed: only stream-check the outer byte count and SHA of the sealed archive's container, and the ZIP central directory, collecting file names, counts, sizes, CRCs, flags, and file name pairings—without opening ZIP members, without decompressing, without viewing images, without parsing labels, without inference, or scoring. This step still requires a clear decision on access to the sealed package and has not yet been executed; the SHA of image and label members, annotation quality, real-world leakage, and scientific sample sufficiency remain unknown, and CRC cannot substitute for SHA.

Directory inventory entry completed. 20 pure-memory ZIP test entries performed, with a single run duration of 300 seconds, up to 100000 entries, directory metadata of 64 MiB, and output of 100 MB. A read-only historical SEAL-recorded container of 311251787 bytes is bound; no other archives shall be searched. The new candidate output directory is `[OMITTED_LOCAL_PATH]`.

A new directory on D: has been written to with a 1 MiB self-generated file and read back with hash verification, at the time approximately 293 GiB of available space was present; no migration or deletion of original C: evidence has occurred. This probe only proves successful read/write at that time; future actual tasks still require re-verification of permissions and space. The C: drive currently holds about 6 GiB, and no new model task is launched under the original 20 GiB retention constraint.

All independently executable save diagnostics and preparations have been closed out; scheduled checks are maintained for deletion. The Holdout remains sealed; this iteration has neither independent scientific validation nor real-world INT8, TensorRT, or Orin performance validation.

# # Verifiable Files

- [Diagnosis Contract](../../docs/DATA_AND_ARTIFACTS.md)
- [Diagnosis Result](../../docs/DATA_AND_ARTIFACTS.md)
- [Independent Review Result](../../docs/DATA_AND_ARTIFACTS.md)
- [Pending Authorization Directory Audit Contract](../../docs/DATA_AND_ARTIFACTS.md)
- [Clarification of Independent Estimation Range](../../docs/DATA_AND_ARTIFACTS.md)
- [Delivery Verification](../../docs/DATA_AND_ARTIFACTS.md)
