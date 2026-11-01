# Gate 0 Selection: Bidirectional Summary of G3+G8 Randomized Control



> Result: restoring G3+G8 recovers 19 additional misses introduced by full W8A8, but causes 12 collateral misses, for a net recovery of +7. This is control 2 among the three frozen random controls and cannot determine the final outcome alone.

# # Bidirectional Results

| Metric | Numerical |
|---|---:|
| FP32 MAC ratio proxy | 21.08% |
| Total missed targets on the disrupted side | 15 |
| Overlap with full W8A8 misses | 11/119(9.24%) |
| Recovery side recovery | 19/119(15.97%) |
| Collateral misses | 12 |
| Net recovery | +7(5.88%) |
| same-target bidirectional agreement | 2/119(1.68%) |
| Stable-target retention | 99.17% |

# # Comparison with Current Candidate Reference

- RAMP-Q candidate `G1+G5`: net recovery +92, bidirectional same-target 83, MAC ratio 22.35%.
- This randomized control trial `G3+G8`: net recovery +7, bidirectional same-target 2, MAC share 21.08%.
- This describes only control 2. The preregistered primary criterion uses the mean net_recovery_rate of the three random controls; it is premature to declare that RAMP-Q passes.

# # Next Steps

Run the destructive side of the next randomized control trial in freeze order: `q_only:G2+G8`.

# # Evidence Boundaries

This result comes from Selection and PyTorch Q/DQ simulation of W8A8, not from TensorRT or Jetson real-world testing. Bidirectional same-target means that a labeled target disappeared on the disrupted side and reappeared on the recovery side, but does not equate to proving the unique causal pathway of underlying numerical propagation.
