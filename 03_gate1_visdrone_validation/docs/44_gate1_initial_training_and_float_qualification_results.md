# Gate1 First Training and Floating-Point Qualification Check Results



This round's **30/30 candidate training rounds have been completed**; training records and weight audits have passed; FP32/FP16 inference on the 109 fixed debug images and target matching have been completed. Fixed full-bounding-box numerical checks resulted in **FAIL_FIXED_TOLERANCES**, thus the floating-point baseline has not been accepted, and Gate1 scientific acceptance has not passed. Independent recalculation only confirms consistency of saved results.

This result continues from the training snapshot in Report 43. [Current Completion Snapshot](../../docs/DATA_AND_ARTIFACTS.md); [Item-by-Item Acceptance Conclusion](../../docs/DATA_AND_ARTIFACTS.md). The original Gate0.5 conclusion is in [Report 39](../../02_gate05_orin_deployment/docs/39_gate05_complete_150_image_acceptance_results.md); the original full-model numerical check remains **FAILED_UNCHANGED**, and this check does not replace it.

# # Completed Training

Using a frozen training contract v3, 4534 training images, 414 training_val, 30 epochs, 640 inputs, batch=4, AMP disabled; took 141.04 minutes, no recovery or restart recorded. best from epoch 29, last from epoch 30; the framework's additional validation callback on best is marked as 31, and cannot be counted as epoch 31 training.

| File | Source Round | SHA256 |
|---|---:|---|
| best.pt | 29 | `cddd14b3ebba294e4c3200bfb50fc1db1a9459c425d8e51d583c0881d71cfcfa` |
| last.pt | 30 | `120e04905d024b78272bde21f55ed1801ed48a5acd7f744892317e1c28c7e49e` |

The final best-in-class validation proxy mAP50=0.23975,mAP50–95=0.13452, used solely for training monitoring and checkpoint selection. The standard YOLO loss does not fully suppress negative samples in ignore regions, and these numbers are **not official VisDrone AP**. Training annotations 227425 entries, deduplicated by loader to 227422 entries, with 3 duplicates already audited; the framework will default max_det from 300 to 902 based on the actual maximum object count of training_val. Both behaviors are supported by frozen source code and logs.

[Training Audit Completed](../../docs/DATA_AND_ARTIFACTS.md) verification of 30 consecutive epochs, checkpoint preservation, best epoch, 4948 training/validation copies with identity labels, 499 state tensors with finite bounds. Saved are FP16 EMA weights; conversion to FP32 **cannot recover the complete FP32 EMA** prior to saving**.

# # 109 images of numerical consistency checks

Two branches load the same best checkpoint independently, without fusing convolutions and BN layers, using a unified 640 square preprocessing and shared normalized inputs, and compare the NMS-previous `[1,14,8400]` raw outputs. FP32 refers to the aforementioned saved FP16 EMA `.float()`; tensorrt or board-end performance is not involved in this study.

| Check | Result |
|---|---|
| Output shape, finite values, positive width and height, score range | 109/109 passed |
| Bounding box tolerance `absolute_difference ≤ 0.5 + 0.005×abs(FP32)` | **109/109 images failed** |
| Bounding box element exceeds tolerance | **10849 / 3662400(0.2962%)** |
| Score tolerance `absolute_difference ≤ 0.002 + 0.02×abs(FP32)` | **0 / 9156000 element divergence** |
| Maximum absolute bounding box difference | **10.439514 pixels**, in the 640-input coordinate system |
| Independent result preservation and re-computation | PASS_SAVED_OUTPUT_CONSISTENCY_ONLY, numerical failure retained |

[Fixed numerical check on original results](../../docs/DATA_AND_ARTIFACTS.md); [Independent re-computation](../../docs/DATA_AND_ARTIFACTS.md). Re-processed 454 archived entries and re-calculated numerical checks, target matching, and partition counting. Reused a frozen target matcher, not an independent official evaluation oracle, and no model or NMS execution was repeated.

# # Target Matching Results

The fixed primary thresholds are conf=0.25 and IoU=0.5, covering 9059 valid GT targets of all valid sizes, **rather than a small-object-only metric**. The 109 images were previously used for engineering debugging and are not independent Selection/Holdout data.

| Branch | hits | missed detections | False positives (false positives at object level) | Target recall rate |
|---|---:|---:|---:|---:|
| FP32 | 3146 | 5913 | 1631 | 34.7279% |
| FP16 | 3143 | 5916 | 1636 | 34.6948% |

Common hits 3136, FP32 unique to one branch 10, FP16 unique to the other branch 7, common misses 5906. The two branches show similar hit rates, **the original output numerical validation failure is not canceled**. [Comprehensive fixed-threshold matching summary](../../docs/DATA_AND_ARTIFACTS.md).

# # Error Diagnosis and Subsequent Entry Points

Only saved raw data were analyzed; neither models nor images were loaded again. Among 10849 out-of-tolerance boxes, 5264 width and 5502 height elements fail, compared with 66/17 center x/y elements. Of these boxes, 10841 come from candidate positions whose maximum class scores are ≤0.001 in both branches. At threshold 0.25, none of the box elements at the 4931 NMS positions retained by each branch exceeds tolerance. This is a **post hoc description and cannot justify filtering candidate positions or changing acceptance criteria**. The low-threshold NMS outputs still contain 4 out-of-tolerance positions on the FP32 side and 5 on the FP16 side.

An independent counterfactual analysis only converts the saved FP32 decoded outputs to FP16, and all 109 images fall within the original fixed tolerance, with a maximum pixel difference of 0.25. This indicates that the FP16 storage rounding alone is insufficient to explain the actual errors; the contributions from input conversion, upstream network accumulation, DFL integration, or decoding remain undifferentiated. No obvious inconsistencies were found in pixel-level differences, fusion processes, or NMS contamination of raw data. However, intermediate logits are still missing to establish the specific root cause.

[Error Distribution](../../docs/DATA_AND_ARTIFACTS.md); [Independent Counterfactual Diagnosis](../../docs/DATA_AND_ARTIFACTS.md); [Training and Result Verification](../../docs/DATA_AND_ARTIFACTS.md).

The next step is to conduct precision probes on a small number of fixed debug images, independently freezing them: save the regression logits from the detection head, and switch between DFL and decoding computation precision. First, reproduce the current original outputs, then determine whether the errors occur before or during decoding. The historical results for 109 images and tolerance remain unchanged. This diagnostic does not involve quantization modules or touch the Selection module. Official evaluation semantic constraints and reference weight precision limits still need to be addressed before scientific baseline acceptance and freezing of the quantization research budget can proceed.

[Minimal Control Proposal](../../docs/DATA_AND_ARTIFACTS.md) has not been executed; [Dimension Range Supplement](../../docs/DATA_AND_ARTIFACTS.md) requires recording actual `head.reg_max` in the next phase, and cannot simply adopt the source code's default 16-bin/64-channel configuration as the actual measured result of this independent test.

548 Selection images were not evaluated for detection scoring or module ranking; the Holdout ZIP file has not been listed, unpacked, read, or inferred. No new quantization matrix was conducted. This delivery supports three separate conclusions: 'training candidate completion', 'engineering output reproducibility', and 'strict numerical qualification failure'.
