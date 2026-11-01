# Gate 0 Budgeted Random Control Final Judgment



> Judgment: **Random control PASS; overall innovation Gate remains PARTIAL.** `G1+G5` significantly outperforms three fixed random combinations, but the labeled overall sensitivity baseline also selects `G1+G5`, and the unlabeled baseline `G1+G2` has not yet completed detection-level q/r validation.

# # 1. Independent Recalculation from Original Per-Target Records

This report independently re-reads Selection's FP32, full W8A8, and per-target predictions for each combination CSV, and independently constructs new missed detections, recoveries, collateral misses, and bidirectional same-target sets, without directly copying statistical values from the summary of any single group.

| Combination | MAC Share | recovery | Collateral misses | Net Recovery (Rate) | Bidirectional Same-Target (Rate) | Stable-target retention |
|---|---:|---:|---:|---:|---:|---:|
| G1+G5 | 22.35% | 100 | 8 | +92(77.31%) | 83(69.75%) | 99.45% |
| G4+G8 | 22.54% | 13 | 15 | -2(-1.68%) | 1(0.84%) | 98.97% |
| G3+G8 | 21.08% | 19 | 12 | +7(5.88%) | 2(1.68%) | 99.17% |
| G2+G8 | 21.91% | 17 | 16 | +1(0.84%) | 1(0.84%) | 98.90% |

The average net recovery across three random combinations is +2.00, with an average net_recovery_rate of 1.68%; the average bidirectional same-target count is 1.33, and the average bidirectional same-target rate is 1.12%.

# # 2. Pre-registration criterion verification

| criterion | Actual Value | Requirement | Result |
|---|---:|---:|---|
| Candidate net_recovery_rate relative to random mean | 77.31% | At least 2.02% (1.20 times the random mean) | PASS |
| Candidate stability retention rate | 99.45% | At least 98% | PASS |
| Candidate bidirectional same-object rate | 69.75% | Not below 1.12% above the random mean | PASS |
| Candidate number of random combinations exceeded | 3/3 | At least 2/3 | PASS |

All four criteria are met, thus the random control confirms **PASS**. The candidate net_recovery_rate is approximately 46.0 times the random mean. This multiplicative factor is large, but the random mean itself is close to zero. The absolute difference should also be reported in the paper, not only the multiplicative factor.

# # 3. Why the method cannot yet be declared innovative

- Random control: passed, indicating the results are not simply obtained by randomly selecting two modules within the same budget.
- Label-based overall sensitivity baseline: fully collides with the candidate, selecting `G1+G5`, thus RAMP-Q has not yet demonstrated an independent advantage over this traditional rule.
- Unlabeled output error baseline: selects `G1+G2`, but this combination has not yet completed detection-level destruction/recovery experiments, and thus cannot determine which selection is actually better.

Therefore, the most accurate current status is `PARTIAL_RANDOM_PASS_BASELINE_PENDING`, not a complete GO, nor STOP.

# # 4. Next steps

Run the destruction arm of the unlabeled baseline: `q_only:G1+G2`; then run the corresponding `r_only:G1+G2`; finally compare it against `G1+G5` at the same budget level in the detection stage.

# # Evidence Boundaries

This conclusion only applies to RUOD-R comparing Selection with PyTorch and Q/DQ simulating W8A8. The random control uses a fixed seed and budget pool from post-hoc analysis with unknown results; it is not a pre-registered, frozen baseline from the project's inception. There is currently no evidence of TensorRT, Jetson, delay, energy consumption, VisDrone, or SOTA.
