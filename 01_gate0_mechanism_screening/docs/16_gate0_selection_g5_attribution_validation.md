# Gate 0 Selection: G5 Bidirectional Isolation Summary



> Current conclusion: **G5 shows a small positive gain, but fails to meet the candidate threshold and is temporarily listed as low priority.** Its destructive side ranks second, but the recovery side and bidirectional same-target evidence are weak.

# # Intervention Definition

- `q_only:G5`: Only G5 uses simulated W8A8; all other modules remain in FP32.
- `r_only:G5`: The entire model uses simulated W8A8; only G5 recovers to FP32.
- In the Selection phase, FP32 originally detected 1571 targets, while full W8A8 introduced 119 additional missed detections.

# # Overall Results

| Metric | G5 results |
|---|---:|
| False negatives caused by `q_only:G5` | 21 |
| Among the above false negatives, those are newly introduced in full W8A8 quantization | 19 |
| Additional false negatives in full W8A8 introduced by `r_only:G5` recovery | 16 |
| Recovery rate | 13.45% |
| Satisfies the criterion that 'G5 quantization causes loss, and G5 recovery restores it' | 2 |
| Bidirectional object-level ratio (2/119) | 1.68% |
| G5 quantization causes loss, but G5 recovery does not restore it | 17 |
| G5 recovery can restore, but standalone quantization of G5 does not cause loss | 14 |
| Collateral false negatives caused by G5 recovery | 6 |
| Stable target retention rate | 99.59% |
| Net recovery (16-6) | +10 |
| G5 convolution MAC proportion proxy | 12.03% |

# # Judgment on Tentative Thresholds

| Preliminary threshold | G5 | Selection criterion |
|---|---:|---|
| Recovery rate at least 15% and at least 20 instances | 13.45%, 16 instances | FAIL |
| stable target retention rate at least 98% | 99.59% | PASS |
| Net recovery is positive | +10 | PASS |
| Recovery module MAC proportion does not exceed 25% | 12.03% | PASS |

# # Correct Explanation

The G5-quantized destructive signal ranks second after G1, but recovery of G5 only recovers 16 targets, and only 2 targets satisfy conditions on both sides of intervention. It may participate in module interactions, but cannot currently be considered a strong causal module on its own. Compared to G2 and G4, G5 has fewer collateral misses and a positive net gain, thus it is retained as a low-priority combination candidate.

# # Evidence Boundaries and Next Steps

- These results are derived from PyTorch Q/DQ simulations of W8A8 quantization and do not reflect actual TensorRT INT8 or Jetson measurements.
- Currently only Selection is used; holdout is not employed for inference or module selection.
- G5 is not currently a strong candidate for standalone module, but can be considered for future dual-module combinations with G1.
- The next step is to run only `q_only:G6`.
