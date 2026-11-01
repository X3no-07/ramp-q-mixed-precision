# Gate 0 Selection: Summary of Bidirectional Isolation for G8



> Current conclusion: **The bidirectional evidence for G8 is not established; this module is provisionally eliminated.** Recovery volume, net benefit, and same-target agreement are insufficient.

# # Intervention Definition

- `q_only:G8`: Only G8 uses simulated W8A8, all other modules remain in FP32.
- `r_only:G8`: The entire model uses simulated W8A8, with only G8 restored to FP32.
- In the Selection phase, FP32 originally detected 1571 targets, while full W8A8 introduced 119 additional missed detections.

# # Overall Results

| Metric | G8 results |
|---|---:|
| False negatives caused by `q_only:G8` | 10 |
| Among the above false negatives, those are newly introduced in full W8A8 quantization | 6 |
| Additional misses under full W8A8 recovered by `r_only:G8` | 7 |
| Recovery rate | 5.88% |
| Simultaneously satisfying the criterion that 'quantizing G8 causes loss and recovering G8 recovers' | 0 |
| Bidirectional object-level ratio | 0.00% |
| Quantizing G8 causes loss, but recovering G8 does not recover | 6 |
| Recovering G8 can recover targets, but standalone quantization of G8 does not cause loss | 7 |
| False negatives introduced by recovering G8 | 3 |
| Stable target retention rate | 99.79% |
| Net recovery (7-3) | +4 |
| G8 convolution MAC ratio proxy | 9.25% |

# # Judgment on Tentative Thresholds

| Preliminary threshold | G8 | Selection criterion |
|---|---:|---|
| Recovery rate at least 15% and at least 20 instances | 5.88%, 7 targets | FAIL |
| stable target retention rate at least 98% | 99.79% | PASS |
| Net recovery is positive | +4 | Passed but very weak |
| Recovery module MAC proportion does not exceed 25% | 9.25% | PASS |

# # Correct Explanation

G8 only recovered 7 targets, and no target satisfied conditions on both intervention sides. Of the 6 full W8A8 missed targets caused by standalone G8 quantization, none were recovered during G8 restoration. Therefore, reliable module-level evidence for G8 is not established.

# # Evidence Boundaries and Next Steps

- These results are derived from PyTorch Q/DQ simulations of W8A8 quantization and do not reflect actual TensorRT INT8 or Jetson measurements.
- Currently only Selection is used; holdout is not employed for inference or module selection.
- G8 temporarily eliminated, all original results retained.
- All single-module bidirectional experiments for G1-G8 have been completed; the next step is to generate a unified ranking and pre-register up to two dual-module combinations before running the combination experiments.
