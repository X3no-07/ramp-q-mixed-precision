# Gate 0 Selection: G4 Bidirectional Isolation Summary



> Current conclusion: **The bidirectional evidence for G4 is not established; this module is currently not a candidate worth retaining in FP32.**

# # Intervention Definition

- `q_only:G4`: Only G4 uses simulated W8A8; all other modules remain in FP32.
- `r_only:G4`: The entire model uses simulated W8A8, with only G4 recovering to FP32.
- In the Selection phase, FP32 originally detected 1571 targets, while full W8A8 introduced 119 additional missed detections.

# # Overall Results

| Metric | G4 results |
|---|---:|
| Misses caused by `q_only:G4` | 7 |
| Among the above false negatives, those are newly introduced in full W8A8 quantization | 6 |
| Additional misses in full W8A8 caused by `r_only:G4` recovery | 12 |
| Recovery rate | 10.08% |
| Simultaneously satisfies the goal of 'G4 quantization causes loss, G4 recovery restores it' | 0 |
| Bidirectional object-level ratio | 0.00% |
| G4 quantization causes loss, but G4 recovery does not restore it | 6 |
| G4 recovery can recover targets, but standalone G4 quantization does not cause loss | 12 |
| Collateral misses caused by G4 recovery | 10 |
| Stable target retention rate | 99.31% |
| Net recovery (12-10) | +2 |
| G4 convolution MAC proportion proxy | 13.29% |

# # Judgment on Tentative Thresholds

| Preliminary threshold | G4 | Selection criterion |
|---|---:|---|
| Recovery rate at least 15% and at least 20 instances | 10.08%, 12 cases | FAIL |
| stable target retention rate at least 98% | 99.31% | PASS |
| Net recovery is positive | +2 | Passed but very weak |
| Recovery module MAC proportion does not exceed 25% | 13.29% | PASS |

# # Correct Explanation

No G4 target satisfies both intervention sides. The 6 full-W8A8 misses lost under G4-only quantization do not reappear when G4 is restored; the 12 targets recovered by restoring G4 were not lost under G4-only quantization. These 12 aggregate recoveries therefore do not establish stable module-level causal evidence for G4.

# # Evidence Boundaries and Next Steps

- These results are derived from PyTorch Q/DQ simulations of W8A8 quantization and do not reflect actual TensorRT INT8 or Jetson measurements.
- Currently only Selection is used; holdout is not employed for inference or module selection.
- G4 is temporarily marked as non-candidate and all original results are retained.
- The next step is to run only `q_only:G5`.
