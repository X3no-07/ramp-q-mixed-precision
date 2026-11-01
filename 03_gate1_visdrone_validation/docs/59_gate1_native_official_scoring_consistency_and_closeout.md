Gate1: Native Official Scoring Consistency and Current Stage Closeout



The native official scoring comparison of the 30th FP32 candidate on 414 training_val images has been verified, and the independent audit of results has also passed. All 1,242 native filtering matrices are consistent with zero tolerance; seven AP/AR metrics meet the original contract, with the maximum deviation of `1.0658141036401503e-13` percentage points. The current effort is to complete the native consistency check on the real development set for this FP32 submission, which is not the Gate1 independent scientific acceptance, and does not cover native scoring for FP16, mixed-precision, or W8A8 models.

This report is a newly added report 59. The previous reports, statuses, and lists are preserved in their original form; the current stage has been completed, and the actual deletion of `gate1` has been performed within half an hour. The final delivery is traced back to the v20 snapshot, with source preservation, validation, and the reason for termination recorded.

# # Fixed Objects, Metrics, and Execution Time

The candidate budget remains at 1, with the fixed YOLO11n 30th FP32 EMA, SHA of `4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27`. The saved 414 training_val ground truth and FP32 submission along with dictionary-ordered members are reused; no re-inference is performed, and no weight selection, configuration, or member selection is based on scores.

Each image compares `filtered_gt`, `filtered_predictions`, and `prepared_gt` matrices, totaling 1,242, element-wise `atol=0,rtol=0`. Metrics use `atol=1e-10` percentage points and `rtol=1e-12`, with no tolerance allowance; the native aggregation maintains the full 414 images and the fixed submission order.

All times below are Beijing time (UTC+8) on 2026 / 10 / 24; durations follow the saved launcher receipts:

| Stage | Start | End | Actual Duration | Scheduled Upper Limit |
|---|---|---|---:|---:|
| 3-Image Member Packaging / Cost Pilot | 10:21:48 | 10:22:08 | 20.610 seconds | 300 seconds |
| Full 414-Image Native Pre-Filtering | 10:23:15 | 11:09:13 | 2,757.500 seconds, approximately 45 minutes and 57.5 seconds | 5,248 seconds |
| Full 414-Image Native Aggregation | 11:09:20 | 11:25:46 | 985.781 seconds, approximately 16 minutes and 25.8 seconds | 7,200 seconds |
| Comparison results written | — | 11:25:47 | — | — |
| Independent audit record saved | — | 11:27:54 | — | — |

Full two-stage execution returns code 0. The acknowledgment record process stops, temporary R: aliases are cleaned up; independent audit performs on-site inspection and finds neither the original identity process nor the original mapping present. The last intermediate progress entry may still display the state from the previous stage, but the final status is determined jointly by the completion acknowledgment, complete output, comparison results, and independent audit.

# # Seven metrics precisely matched

Units are percentages; differences are in basis points. The table uses float64 values parsed by Python. Differences in the last digit of the native JSON text are not explained as new metrics.

| Metric | Python saved | Native official code | Absolute difference (basis points) |
|---|---:|---:|---:|
| AP50–95 | 12.440710037775458 | 12.440710037775439 | 1.9539925233402755e-14 |
| AP50 | 23.78255578747402 | 23.782555787473914 | 1.0658141036401503e-13 |
| AP75 | 11.46749325614797 | 11.467493256147959 | 1.0658141036401503e-14 |
| AR1 | 1.0434657596849932 | 1.0434657596849932 | 0 |
| AR10 | 5.889819061163743 | 5.889819061163744 | 1.7763568394002505e-15 |
| AR100 | 18.12260571894477 | 18.122605718944765 | 3.552713678800501e-15 |
| AR500 | 21.546394836035162 | 21.546394836035162 | 0 |

[Native comparison result](../../docs/DATA_AND_ARTIFACTS.md) and [Independent audit](../../docs/DATA_AND_ARTIFACTS.md) both pass. training_val remains in the training, development, and validation role; these scores cannot be interpreted as independent test set, selection, or holdout scientific validation.

# # Input, official source code, and numerical transmission

The official toolkit is fixed at commit `005445782213e20cb91bc50a597db3dd949e749a`. The native versions `dropObjectsInIgr`, `createIntImg`, `evalRes`, `compOas`, `calcAccuracy`, and `VOCap` are invoked. Runtime versions of Octave 11.3.0 and image 2.20.0 are fixed at `mean2`. A frozen official source and runtime list are used. Independent audit performs SHA verification on 2,890 runtime members and 4,184 source/result bindings before and after execution.

The original fixed input file SHA is `45c5f8626d1b57288cf1b7962e23ee7228c00b5be613db4b07c697e320780eb0`; the v2 IEEE754 hexadecimal column-major transmission file SHA is `eb4578a2dca4fe03a4c17cd3851902fc350678b20b095a991bc8afb8c991c1ff`. Of the 828 input GT/DT matrices, each bit matches the original logical values, with no change in members, dimensions, or order. v2 modifies only the numerical transmission packaging for exact round-trip, without altering the official scoring function, model outputs, or decision tolerances.

Complete sources, stage inputs, and execution budget are referenced in [base contract](../../docs/DATA_AND_ARTIFACTS.md), [complete pre-filtering contract](../../docs/DATA_AND_ARTIFACTS.md), and [aggregation contract](../../docs/DATA_AND_ARTIFACTS.md). Native outputs, per-image MAT files, exact hexadecimal JSON checkpoints, and all failure records are preserved.

# # Accurate range of independent audit pass

The independent audit status is `PASS_INDEPENDENT_SAVED_REAL_414_FP32_NATIVE_MATRIX_AND_SEVEN_METRIC_RECHECK_ONLY`. It verifies the complete sequence of 414 images, final stage states, sources, and saved files. It then re-plays the fixed Python pre-filtering from the original frozen input save raw lines, saves and compares 1,242 three-matrix sets. For the seven metrics, only the Python and native saved values are re-read and re-validated under the original tolerance—**no recalculation of AP, no re-execution of Octave or model, and no reading of original images or labels**.

Evidence of exact MAT content re-reading comes from the native worker's MAT/hexadecimal JSON round-trip. The independent Python audit checks the MAT file hash and JSON value, without using an independent MAT parser or independently decoding the final metrics MAT. The source code and completion acknowledgment support a single complete and ordered `calcAccuracy` aggregation, but no additional execution tracer proves the number of calls. These limitations are not waived due to this pass.

This native consistency applies only to fixed FP32 submissions. FP16, main FP32/detection tower FP16/decoding FP32, and the three W8A8 intervention branches do not automatically gain their own native AP comparisons as a result.

# # Historical failures and per-component status

The original 02:54:48–03:24:48 attempt of 1800 seconds is preserved as a timeout. Subsequently, the v1 checkpoint attempt failed at the first real image due to zero-tolerance MAT and standard JSON numerical round-trip, and is recorded as `FAILED_NATIVE_CHECKPOINT_MAT_JSON_ZERO_TOLERANCE_ROUNDTRIP`; it did not complete the 414-image run. Diagnostics revealed 72 elements in the filtering predictions had trailing transmission discrepancies, with the largest difference approximately `1.1368683772161603e-13`; this is not a justification for relaxing filtering matrix tolerances. v2 passed under a precisely defined transmission contract, and the old attempt and console logs were not rewritten or converted to PASS.

| item-by-item | current verified facts | does not represent |
|---|---|---|
| formal training | 30 epochs, 33,990 batches, 2,557 optimizer and EMA updates completed | scientific acceptance or deployment benefits |
| saving and reconstruction | 30 epochs of FP32 checkpoint list and final CPU/CUDA synthesis reconstruction audit passed | model scientific generalization established |
| floating-point baseline threshold | FP32 body / FP16 detection towers / FP32 decoding passes on 109 debug+414 training_val images, totaling 523; full FP16 fails on all 523 images | full FP16 pass or optimal precision split |
| fixed FP32 development set scoring | 414 images, 1,242 matrices, seven metrics native consistency and independent saved artifact audit passed | native scoring or independent AP testing on other branches |
| W8A8 development simulation | 109 debug × 3 seed fixed experiments without fused Conv QDQ and saved verification completed; all nine quantization branches failed on all 109/109 images, violating the original full candidate box tolerance | original numerical threshold passed, TensorRT integer execution or mode selection superiority |
| independent scientific acceptance | not yet passed | can be substituted by development results or native scoring consistency |
| Current formal candidate validation for INT8/TensorRT/Orin under real-world conditions | not yet completed | Old RUOD-R device results may be cited as reference |

Quantization differences for W8A8 are fully preserved across all 8400 candidate boxes, without substitution by confidence thresholds or NMS subsets. Net recovery across three seeds is +56/+36/+43, with 69/82/71 stable targets suffering side damage and an increase in false positives after recovery; only 6 risk-isolation components are present in 109 images, and component contributions are highly concentrated in the maximum 81 images. This result only supports development simulation descriptions and does not translate into advantages of random/standard methods, independent generalization, real INT8, or speed conclusions. Original `FAIL_FIXED_TOLERANCES` and old Gate0.5 `FAILED_UNCHANGED` are retained.

# # Contract details for the next phase remain to be clarified

The next phase template retains `DRAFT_REQUIRES_EXPLICIT_ACCESS_AND_PARAMETER_DECISIONS`: 11 primary decision fields remain unfilled, and 9 supplementary items related to duplicate members and dependency policies are still null. The template lacks a loader or executor and does not generate new real candidates; it cannot be opened via self-filled boolean authorization fields. No data is made accessible through this template.

Selection is currently restricted to METADATA_ONLY. Among the existing metadata, 548 members contain only 547 distinct image SHA records and 75 risk-isolation components; no re-validation of content has been performed in this round, and it cannot be claimed that the current file has been confirmed to be identical. No automatic deduplication, reweighting, or modification of prior members has occurred. A new independent assessment must freeze rules such as duplicate member handling, candidate/counter budget, FP/AP degradation limits, scenario event counts, uncertainty, and multiple comparisons prior to reading new results.

Holdout remains SEALED. The current split lacks member metadata, with unknown member counts, IDs, SHAs, and component isolation; historical validation sizes cannot be used to compensate, and no directory enumeration or sealed package enumeration is performed. Access to Selection/Holdout content, the next matrix, and device-phase operations all require separate authorization and frozen contracts.

No new candidates, model inference, or original image/label reading were added in this phase; scoring and auditing use already-frozen saved rows. Native control, saved artifact audit, and metadata/decision condition organization have been fully completed autonomously; remaining work depends on new research decisions and access agreements. Native work, independent saved audit, and preparation materials have all been completed. The original identity process and temporary mappings are absent; the scheduled half-hour check `gate1` has been deleted. No ongoing tasks can produce new results through polling.

[Final machine decision](../../docs/DATA_AND_ARTIFACTS.md), [Complete delivery verification](../../docs/DATA_AND_ARTIFACTS.md), [Scheduled check deletion receipt](../../docs/DATA_AND_ARTIFACTS.md), and [v20 snapshot status](../../docs/DATA_AND_ARTIFACTS.md).
