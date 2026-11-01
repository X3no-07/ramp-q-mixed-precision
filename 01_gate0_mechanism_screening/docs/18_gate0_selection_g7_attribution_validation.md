# Gate 0 Selection: G7 Bidirectional Isolation Summary



> Current conclusion: **Bidirectional evidence for G7 is not established; temporarily eliminated.** Although G7 is a small-object branch in P3, the experiment does not support prioritizing its restoration to FP32.

# # Intervention Definition

- `q_only:G7`: Only G7 uses simulated W8A8, all other modules remain in FP32.
- `r_only:G7`: The entire model uses simulated W8A8, with only G7 restored to FP32.
- In the Selection phase, FP32 originally detected 1571 targets, while full W8A8 introduced 119 additional missed detections.

# # Overall Results

| Metric | G7 results |
|---|---:|
| False negatives caused by `q_only:G7` | 16 |
| Among the above false negatives, those are newly introduced in full W8A8 quantization | 13 |
| Additional false negatives recovered by `r_only:G7` in fully W8A8 models | 10 |
| Recovery rate | 8.40% |
| Simultaneously satisfies the criterion that 'G7 quantization causes loss, and G7 restoration recovers' | 0 |
| Bidirectional object-level ratio | 0.00% |
| G7 quantization causes loss, but G7 restoration fails to recover | 13 |
| G7 recovery can recover targets, but standalone quantization of G7 does not cause any loss | 10 |
| False negatives introduced by recovering G7 | 6 |
| Stable target retention rate | 99.59% |
| Net recovery (10-6) | +4 |
| G7 convolution MAC proportion proxy | 17.78% |

# # Judgment on Tentative Thresholds

| Preliminary threshold | G7 | Selection criterion |
|---|---:|---|
| Recovery rate at least 15% and at least 20 instances | 8.40%, 10 examples | FAIL |
| stable target retention rate at least 98% | 99.59% | PASS |
| Net recovery is positive | +4 | Passed but very weak |
| Recovery module MAC proportion does not exceed 25% | 17.78% | PASS |

# # Correct Explanation

G7 is structurally most directly related to small objects among the P3 detection branches, but this structural intuition is not supported by the current isolation experiment. The 13 fully W8A8 false negatives caused by standalone quantization of G7 do not recover at all when G7 is restored; the 10 recovered targets from restoring G7 are not among the targets that were originally lost on the damaged side. Therefore, the claim that 'it is a small object branch' cannot serve as a sufficient justification for retaining FP32.

# # Evidence Boundaries and Next Steps

- These results are derived from PyTorch Q/DQ simulations of W8A8 quantization and do not reflect actual TensorRT INT8 or Jetson measurements.
- Currently only Selection is used; holdout is not employed for inference or module selection.
- G7 is temporarily eliminated, and all original results are retained.
- The next step is to run only `q_only:G8`, completing the final damaged-side scan of the last module.
