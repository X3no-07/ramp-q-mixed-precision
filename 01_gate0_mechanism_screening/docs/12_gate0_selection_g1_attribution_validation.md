# Gate 0 Selection: G1 Bidirectional Isolation Summary



> Current conclusion: **A strong bidirectional signal is observed in G1, but it is only candidate evidence from the Selection set and does not justify opening the Holdout set prematurely, nor does it allow concluding the G2-G8 comparison.**

# # Intervention Definition

- `q_only:G1`: Only G1 uses simulated W8A8; all other modules remain FP32, to test whether G1 alone is sufficient to cause object disappearance.
- `r_only:G1`: Full model uses simulated W8A8; only G1 recovers to FP32, to test whether G1 can recover objects.
- Matching is restricted to the 1571 targets originally detected by FP32 in Selection; full W8A8 introduces 119 additional misses.

# # Overall Results

| Metric | G1 results |
|---|---:|
| `q_only:G1` caused false negatives | 100 |
| Among the above false negatives, those are newly introduced in full W8A8 quantization | 88 |
| `r_only:G1` recovered newly introduced false negatives in full W8A8 quantization | 94 |
| Satisfy the criterion of 'quantization loss in G1 and recovery restoration in G1' | 69 |
| Bidirectional same-object ratio (69/119) | 57.98% |
| G1 quantization causes loss, but G1 recovery fails to restore it | 19 |
| G1 recovery can restore, but standalone G1 quantization does not cause loss | 25 |
| False negatives caused by G1 recovery | 13 |
| Stable target retention rate | 99.10% |
| Net recovery (94-13) | 81 |
| G1 convolution MAC proportion proxy | 10.32% |

# # Judgment on Tentative Thresholds

| Preliminary threshold | G1 | Selection criterion |
|---|---:|---|
| Recovery rate at least 15% and at least 20 instances | 78.99%, 94 instances | PASS |
| stable target retention rate at least 98% | 99.10% | PASS |
| Net recovery is positive | +81 | PASS |
| Recovery module MAC proportion does not exceed 25% | 10.32% | PASS |

# # Correct Explanation

The 69 targets provide the most direct bidirectional consistency evidence: they disappear when G1 is quantized and reappear when G1 is restored from full W8A8. This indicates that G1 is likely a critical sensitive region.

However, the results are not perfectly one-to-one: 19 targets only show failure on the quantization side, and 25 targets only show success on the restoration side. This suggests that interactions among modules, threshold jittering, or one-to-one matching competition still exist within the network, and not all 119 additional misses can be attributed to G1.

# # Evidence Boundaries and Next Steps

- These results are derived from PyTorch Q/DQ simulations of W8A8 quantization and do not reflect actual TensorRT INT8 or Jetson measurements.
- Currently only Selection is used; holdout is not employed for inference or module selection.
- G1 results cannot be directly used to open holdout; it is mandatory to scan G2-G8 under the same rules before determining whether G1 is truly ranked first.
- The next step is to run only `q_only:G2`, without running other groups simultaneously.
