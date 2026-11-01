# Gate 0 Selection: Bidirectional Summary of G1+G5



> Current Conclusion: **G1+G5 passes all preregistered incremental thresholds and is eligible for inclusion in the candidate pool.** This is only a Selection-level conclusion and requires the completion of the preregistered combination G1+G3 before a unique holdout candidate can be finalized.

# # Intervention Definition

- `q_only:G1+G5`: Only G1 and G5 employ simulated W8A8 quantization; all other modules remain in FP32.
- `r_only:G1+G5`: The entire model uses simulated W8A8 quantization, with only G1 and G5 recovering to FP32 precision.
- In the Selection phase, FP32 originally detected 1571 targets, while full W8A8 introduced 119 additional missed detections.

# # Overall Results

| Metric | G1 single module | G1+G5 | increment |
|---|---:|---:|---:|
| Missed detections in destructive side overlap with full W8A8 missed detections | 88 | 100 | +12 |
| recovery_count | 94 | 100 | +6 |
| Recovery rate | 78.99% | 84.03% | +5.04 percentage points |
| Collateral misses | 13 | 8 | -5 |
| Stable target retention rate | 99.10% | 99.45% | +0.35 percentage points |
| Net recovery | +81 | +92 | +11 |
| same-target bidirectional agreement | 69 | 83 | +14 |
| MAC share proxy | 10.32% | 22.35% | +12.03 percentage points |

In G1+G5, 100 targets are newly missed on the destruction side within the full W8A8 set, and 100 targets are recovered on the restoration side, of which 83 are the same batch of targets. Additionally, 17 targets are only valid on the destruction side, and 17 targets are only valid on the restoration side.

# # Preregistered Threshold Verification

| frozen threshold | G1+G5 results | Decision |
|---|---:|---|
| recovery_count at least 99 | 100 | PASS |
| net recovery at least +86 | +92 | PASS |
| stable target retention rate at least 98% | 99.45% | PASS |
| bidirectional object-level consistency at least 69 | 83 | PASS |
| MAC share no more than 25% | 22.35% | PASS |

# # Correct Explanation

Compared to only recovering G1, additional recovery of G5 increases 6 recovered targets, while reducing 5 collateral misses, resulting in a net recovery improvement of 11; bidirectional consistent targets increase by 14. The cost is that the computation proxy FP32 rises from 10.32% to 22.35%. Therefore, G1+G5 is an effective precision–cost candidate on the Selection criterion, but cannot yet be declared the final solution.

# # Evidence Boundaries and Next Steps

- These results are derived from PyTorch Q/DQ simulations of W8A8 quantization and do not reflect actual TensorRT INT8 or Jetson measurements.
- Only Selection is used; the holdout set is not employed for inference or candidate selection.
- The full run of the backup combinations `q_only:G1+G3` and `r_only:G1+G3` must be completed according to preregistration, and cannot be prematurely terminated due to the main combination passing.
