# Gate 0 Selection: Bidirectional Summary of G2+G8 Randomized Control



> Result: restoring G2+G8 recovers 17 additional misses introduced by full W8A8, but causes 16 collateral misses, for a net recovery of +1. This is control 3 among the three frozen random controls and cannot determine the final outcome alone.

# # Bidirectional Results

| Metric | Numerical |
|---|---:|
| FP32 MAC ratio proxy | 21.91% |
| Total missed targets on the disrupted side | 15 |
| Overlap with full W8A8 misses | 11/119(9.24%) |
| Recovery side recovery | 17/119(14.29%) |
| Collateral misses | 16 |
| Net recovery | +1(0.84%) |
| same-target bidirectional agreement | 1/119(0.84%) |
| Stable-target retention | 98.90% |

# # Comparison with Current Candidate Reference

- RAMP-Q candidate `G1+G5`: net recovery +92, bidirectional same-target 83, MAC ratio 22.35%.
- Current randomized control `G2+G8`: net recovery +1, bidirectional same-target 1, MAC ratio 21.91%.
- This describes only control 3. The preregistered primary criterion uses the mean net_recovery_rate of the three random controls; it is premature to declare that RAMP-Q passes.

# # Next Steps

All three randomized controls have been completed; the next step is to consolidate results uniformly, with no new combinations to be added.

# # Evidence Boundaries

This result comes from Selection and PyTorch Q/DQ simulation of W8A8, not from TensorRT or Jetson real-world testing. Bidirectional same-target means that a labeled target disappeared on the disrupted side and reappeared on the recovery side, but does not equate to proving the unique causal pathway of underlying numerical propagation.
