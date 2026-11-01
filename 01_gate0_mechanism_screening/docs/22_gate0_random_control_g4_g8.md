# Gate 0 Selection: Bidirectional summary of G4+G8 randomized control trials



> Result: restoring G4+G8 recovers 13 additional misses introduced by full W8A8, but causes 15 collateral misses, for a net recovery of -2. This is control 1 among the three frozen random controls and cannot determine the final outcome alone.

# # Bidirectional Results

| Metric | Numerical |
|---|---:|
| FP32 MAC ratio proxy | 22.54% |
| Total missed targets on the disrupted side | 15 |
| Overlap with full W8A8 misses | 13/119(10.92%) |
| Recovery side recovery | 13/119(10.92%) |
| Collateral misses | 15 |
| Net recovery | -2(-1.68%) |
| same-target bidirectional agreement | 1/119(0.84%) |
| Stable-target retention | 98.97% |

# # Comparison with Current Candidate Reference

- RAMP-Q candidate `G1+G5`: net recovery +92, bidirectional same-target 83, MAC ratio 22.35%.
- This randomized control trial `G4+G8`: net recovery -2, bidirectional same-target 1, MAC share 22.54%.
- This only describes the first control trial. The preregistered primary criterion uses the average net_recovery_rate across three randomized control trials; we cannot currently declare RAMP-Q passed.

# # Next Steps

Run the destructive side of the second randomized control trial in freeze order: `q_only:G3+G8`.

# # Evidence Boundaries

This result comes from Selection and PyTorch Q/DQ simulation of W8A8, not from TensorRT or Jetson real-world testing. Bidirectional same-target means that a labeled target disappeared on the disrupted side and reappeared on the recovery side, but does not equate to proving the unique causal pathway of underlying numerical propagation.
