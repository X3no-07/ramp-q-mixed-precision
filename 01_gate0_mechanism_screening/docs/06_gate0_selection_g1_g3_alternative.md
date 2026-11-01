# Gate 0 Selection: G1+G3 Bidirectional Combination Summary



> Current Conclusion: **G1+G3 fails the preregistration incremental threshold and cannot replace G1 as a single module, nor enter the holdout set.**

# # Overall Results

| Metric | G1 single module | G1+G3 | increment |
|---|---:|---:|---:|
| Missed detections in destructive side overlap with full W8A8 missed detections | 88 | 88 | 0 |
| recovery_count | 94 | 90 | -4 |
| Recovery rate | 78.99% | 75.63% | -3.36 percentage points |
| Collateral misses | 13 | 9 | -4 |
| Stable target retention rate | 99.10% | 99.38% | +0.28 percentage points |
| Net recovery | +81 | +81 | 0 |
| same-target bidirectional agreement | 69 | 67 | -2 |
| MAC share proxy | 10.32% | 22.14% | +11.82 percentage points |

In G1+G3, 88 targets on the destruction side are newly missed under full W8A8, 90 targets are recovered on the recovery side, of which 67 are the same batch of targets. Additionally, 21 targets are only valid on the destruction side, and 23 targets are only valid on the recovery side.

# # Preregistered Threshold Verification

| frozen threshold | G1+G3 results | Decision |
|---|---:|---|
| recovery_count at least 99 | 90 | FAIL |
| net recovery at least +86 | +81 | FAIL |
| stable target retention rate at least 98% | 99.38% | PASS |
| bidirectional object-level consistency at least 69 | 67 | FAIL |
| MAC share no more than 25% | 22.14% | PASS |

# # Correct Explanation

Additional recovery of G3 reduces 4 collateral misses, but also fails to recover 4 full W8A8 newly missed detections, resulting in no net recovery improvement beyond G1, and a decline in bidirectional consistency. Meanwhile, the computation proxy FP32 increases by 11.82 percentage points. Thus, G1+G3 lacks sufficient incremental value.

# # Evidence Boundaries

- This result comes from Selection and PyTorch Q/DQ simulation of W8A8, not from TensorRT INT8 or Jetson real-world measurements.
- The holdout set is not used for inference or candidate selection in this combination.
