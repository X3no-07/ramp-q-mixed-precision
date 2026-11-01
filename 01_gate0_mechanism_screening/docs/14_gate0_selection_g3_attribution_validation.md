# Gate 0 Selection: G3 bidirectional isolation summary



> Current conclusion: **G3 is marginal but fails, provisionally listed as low-priority candidate**. It has a slight net gain, but lacks sufficient evidence for recovery_count quantization and bidirectional same-target consistency.

# # Intervention Definition

- `q_only:G3`: Only G3 uses simulated W8A8, all other modules remain in FP32.
- `r_only:G3`: The entire model uses simulated W8A8, with only G3 restored to FP32.
- In the Selection phase, FP32 originally detected 1571 targets, while full W8A8 introduced 119 additional missed detections.

# # Overall Results

| Metric | G3 results |
|---|---:|
| False negatives caused by `q_only:G3` | 11 |
| Among the above false negatives, those are newly introduced in full W8A8 quantization | 9 |
| Additional false negatives recovered by `r_only:G3` in fully W8A8 models | 19 |
| Recovery rate | 15.97% |
| Criterion satisfied: G3 quantization causes loss, and G3 restoration recovers targets | 2 |
| Bidirectional object-level ratio (2/119) | 1.68% |
| G3 quantization causes loss, but G3 recovery does not restore it | 7 |
| G3 recovery can recover targets, but standalone G3 quantization does not cause loss | 17 |
| Collateral misses caused by G3 recovery | 12 |
| Stable target retention rate | 99.17% |
| Net recovery (19-12) | +7 |
| G3 convolution MAC proportion proxy | 11.82% |

# # Judgment on Tentative Thresholds

| Preliminary threshold | G3 | Selection criterion |
|---|---:|---|
| Recovery rate at least 15% and at least 20 instances | 15.97%, 19 cases | not passed (few 1) |
| stable target retention rate at least 98% | 99.17% | PASS |
| Net recovery is positive | +7 | PASS |
| Recovery module MAC proportion does not exceed 25% | 11.82% | PASS |

# # Correct Explanation

G3 recovery shows a slight positive gain, but only 2 targets simultaneously satisfy both intervention conditions, indicating that the disappearance caused by G3 quantization and the recovery achieved via G3 are fundamentally not the same set of targets. The current evidence more resembles module interaction or changes near detection thresholds, and is still insufficient to support a stable G3 module-level mechanism.

# # Evidence Boundaries and Next Steps

- These results are derived from PyTorch Q/DQ simulations of W8A8 quantization and do not reflect actual TensorRT INT8 or Jetson measurements.
- Currently only Selection is used; holdout is not employed for inference or module selection.
- G3 does not enter current strong candidates but is retained as a low-priority module to be ranked uniformly after all G1-G8 modules are completed.
- The next step is to run only `q_only:G4`.
