# Gate 0 Selection: G6 Bidirectional Isolation Summary



> Current conclusion: **Bidirectional evidence for G6 is not established; the candidate is provisionally rejected.** Restoring G6 recovers as many targets as it causes collateral misses, leaving zero net benefit.

# # Intervention Definition

- `q_only:G6`: Only G6 uses simulated W8A8; all other modules remain in FP32.
- `r_only:G6`:The entire model uses simulated W8A8, with only G6 restored to FP32.
- In the Selection phase, FP32 originally detected 1571 targets, while full W8A8 introduced 119 additional missed detections.

# # Overall Results

| Metric | G6 results |
|---|---:|
| False negatives caused by `q_only:G6` | 3 |
| Among the above false negatives, those are newly introduced in full W8A8 quantization | 3 |
| Additional false negatives recovered by `r_only:G6` in fully W8A8 models | 3 |
| Recovery rate | 2.52% |
| Simultaneously satisfies the criterion that 'G6 quantization causes loss, and G6 restoration recovers' | 0 |
| Bidirectional object-level ratio | 0.00% |
| G6 quantization causes loss, but G6 restoration fails to recover | 3 |
| G6 restoration recovers, but standalone G6 quantization does not cause loss | 3 |
| Collateral false negatives introduced by restoring G6 | 3 |
| Stable target retention rate | 99.79% |
| Net recovery (3-3) | 0 |
| G6 convolution MAC proportion proxy | 12.86% |

# # Judgment on Tentative Thresholds

| Preliminary threshold | G6 | Selection criterion |
|---|---:|---|
| Recovery rate at least 15% and at least 20 instances | 2.52%, 3 instances | FAIL |
| stable target retention rate at least 98% | 99.79% | PASS |
| Net recovery is positive | 0 | FAIL |
| Recovery module MAC proportion does not exceed 25% | 12.86% | PASS |

# # Correct Explanation

G6 recovered 3 targets, while causing 3 stable target misses, and the bidirectional same-target count is 0.The targets involved in the recovery and destruction sides are not the same batch, thus no reliable evidence of the G6 module level is established.

# # Evidence Boundaries and Next Steps

- These results are derived from PyTorch Q/DQ simulations of W8A8 quantization and do not reflect actual TensorRT INT8 or Jetson measurements.
- Currently only Selection is used; holdout is not employed for inference or module selection.
- G6 is temporarily disqualified and all original results are retained.
- The next step is to run only `q_only:G7`.
