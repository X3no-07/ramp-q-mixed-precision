# Gate 0 Selection: G2 Bidirectional Isolation Summary



> Current conclusion: **G2 bidirectional evidence is not established, and the module is currently not worthy of retention FP32**. This is a selection-level elimination judgment, not the final gate conclusion.

# # Intervention Definition

- `q_only:G2`: Only G2 uses simulated W8A8; all other modules remain in FP32.
- `r_only:G2`:The entire model uses simulated W8A8, with only G2 restored to FP32.
- In the Selection phase, FP32 originally detected 1571 targets, while full W8A8 introduced 119 additional missed detections.

# # Overall Results

| Metric | G2 results |
|---|---:|
| False negatives caused by `q_only:G2` | 9 |
| Among the above false negatives, those are newly introduced in full W8A8 quantization | 8 |
| Additional false negatives recovered by `r_only:G2` in fully W8A8 models | 16 |
| Recovery rate | 13.45% |
| Criterion satisfied: G2 quantization causes loss, and G2 restoration recovers targets | 1 |
| Bidirectional same-target ratio (1/119) | 0.84% |
| G2 quantization causes loss, but restoration fails to recover G2 | 7 |
| G2 restoration recovers targets, but standalone G2 quantization does not cause loss | 15 |
| Collateral false negatives introduced by restoring G2 | 19 |
| Stable target retention rate | 98.69% |
| Net recovery (16-19) | -3 |
| G2 convolution MAC proportion proxy | 12.65% |

# # Judgment on Tentative Thresholds

| Preliminary threshold | G2 | Selection criterion |
|---|---:|---|
| Recovery rate at least 15% and at least 20 instances | 13.45%, 16 instances | FAIL |
| stable target retention rate at least 98% | 98.69% | PASS |
| Net recovery is positive | -3 | FAIL |
| Recovery module MAC proportion does not exceed 25% | 12.65% | PASS |

# # Correct Explanation

G2 only has 1 targets simultaneously satisfying both intervention conditions, indicating that the disappearance caused by G2 quantization and the recovery achieved via G2 recovery are fundamentally not the same set of targets. Although recovery of G2 occasionally recovers 16 targets, it simultaneously causes the loss of 19 originally stable targets, resulting in a net negative gain. Therefore, the overall recovery_count cannot be interpreted as reliable causal evidence for G2.

# # Evidence Boundaries and Next Steps

- These results are derived from PyTorch Q/DQ simulations of W8A8 quantization and do not reflect actual TensorRT INT8 or Jetson measurements.
- Currently only Selection is used; holdout is not employed for inference or module selection.
- G2 is temporarily marked as non-candidate, but the original full results are preserved without deletion or re-running of the selection process.
- The next step is to run only `q_only:G3`.
