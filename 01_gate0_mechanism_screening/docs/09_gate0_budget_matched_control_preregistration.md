# Gate 0 Pre-registration Against Budget Control



> Status: **FROZEN**. This document only freezes the control, does not run new inference. The most critical warning is: the ordinary Selection also ranks G1+G5 in terms of overall accuracy sensitivity, thus it is currently inappropriate to claim that the module selection algorithm of RAMP-Q possesses independent innovation.

# # 1. Why a control is now mandatory

Existing experiments support the idea that certain modules may selectively destroy targets, which can be recovered by a recovery module. However, it has not yet answered whether target-level methods outperform random module selection or standard sensitivity ranking. Without a budget-controlled comparison, the paper remains a feasibility observation and cannot serve as a new selection algorithm.

# # 2. Candidate and Budget Window

- Locked candidates: `G1+G5`, with FP32 MAC share proxy at 22.35%.
- Hard ceiling: 25%.
- Budget tolerance: ±2 percentage points on either side of the candidate within the same budget.
- The random pool only allows combinations of two modules, neither of which has been previously run, and both of which satisfy the budget window.
- Candidates `G1+G5` and `G1+G3` that have already been reviewed are excluded from the random pool.

# # 3. Qualified but Not Executed Random Pool

| Combination | MAC Share |
|---|---:|
| G1+G2 | 22.97% |
| G1+G4 | 23.60% |
| G1+G6 | 23.18% |
| G2+G8 | 21.91% |
| G3+G5 | 23.85% |
| G3+G8 | 21.08% |
| G4+G8 | 22.54% |
| G5+G8 | 21.28% |
| G6+G8 | 22.11% |

A deterministic shuffle is performed on the above-sorted pool using a fixed seed `20260910`, with the top three frozen:

| Order | Random Control | MAC Share | Destruction Side | Recovery Side |
|---:|---|---:|---|---|
| 1 | G4+G8 | 22.54% | `q_only:G4+G8` | `r_only:G4+G8` |
| 2 | G3+G8 | 21.08% | `q_only:G3+G8` | `r_only:G3+G8` |
| 3 | G2+G8 | 21.91% | `q_only:G2+G8` | `r_only:G2+G8` |

No combination should be changed due to a poor or overly strong first result.

# # 4. Primary Random Control Criterion

The primary dataset remains Selection, as the candidate holdout has already been seen and cannot be considered a completely untouched blind test set for these new controls.

- Primary metric: `net_recovery_rate = (recovery_count - collateral_miss_count) / all_W8A8_additional_miss_count`.
- Pass criterion: the G1+G5 net_recovery_rate must be at least 1.20 times the mean of the 3 random controls.
If the random mean is less than or equal to 0, the candidate must be positive and at least exceed 0.05.
- Guardrail one: The candidate stability retention rate must be at least 98%.
- Guardrail two: The candidate bidirectional object-level agreement rate must not be lower than the mean of the random control groups.
- Guardrail three: The candidate net_recovery_rate must exceed 3 randomly selected controls in at least 2 cases.

Risk-weighted recovery is only for exploratory analysis. The reason is that small/low-contrast weights were not frozen before observing per-module results; promoting them as the primary metric would constitute post-hoc selection.

# # 5. Traditional labeled-accuracy sensitivity baseline

Define: sort in descending order by the total number of FP32-targeted detections in Selection that become missed due to each `q_only:Gj`, with ties broken by module index; then greedily add one by one within the 25% MAC limit.

| Rank | Module | Overall additional misses | MAC Share |
|---:|---|---:|---:|
| 1 | G1 | 100 | 10.32% |
| 2 | G5 | 21 | 12.03% |
| 3 | G7 | 16 | 17.78% |
| 4 | G3 | 11 | 11.82% |
| 5 | G8 | 10 | 9.25% |
| 6 | G2 | 9 | 12.65% |
| 7 | G4 | 7 | 13.29% |
| 8 | G6 | 3 | 12.86% |

This rule selects: `G1+G5`, accounting for 22.35% of MAC. It is identical to the RAMP-Q candidate, denoted as **COLLISION**. This is not a RAMP-Q win, but rather indicates that the object-level rule currently does not produce selection results distinct from conventional labeled-accuracy sensitivity sorting.

# # 6. Unlabeled output reconstruction baseline

Before running any random combinations, complete the following baseline using the pre-frozen 64 calibration images:

1. Run FP32 and the 8 `q_only:Gj` configurations separately, and read the raw detection outputs before confidence filtering and NMS.
2. Compute the normalized mean squared error: `sum((Q-F)^2) / (sum(F^2)+1e-12)` per image.
3. Each module uses the average across the 64 images as its primary score; normalized cosine distance is used only for diagnostic purposes.
4. Within the same budget pool, select the combination with the highest sum of the two primary scores; in case of ties, prefer the one with lower MAC, then by module ID.
5. After writing down the scores, rankings, and selected combination, new random combination detection results are allowed to be viewed.

This is a baseline based on output error without target labels, which tests whether target-level recovery information provides additional selection value.

# # 7. Frozen Execution Order

1. Compute and freeze the unlabeled output reconstruction ranking.
2. `q_only:G4+G8`.
3. `r_only:G4+G8`.
4. `q_only:G3+G8`.
5. `r_only:G3+G8`.
6. `q_only:G2+G8`.
7. `r_only:G2+G8`.
8. Do not change the combination or threshold; unify the aggregation.

# # 8. Current Conclusions

- Mechanism feasibility: Still `CONDITIONAL GO`.
- Algorithm innovation: `NOVELTY COLLISION RISK`. The conventional labeled-accuracy sensitivity baseline has already collided with the candidate.
- If the unlabeled baseline also selects the same or equivalent combination, and the random controls fail to demonstrate clear superiority, the paper's main line should shift to 'quantization failure mechanisms and small-object reliability', and the novel mixed-precision search algorithm should no longer be the primary focus.

# # Evidence Boundaries

The preregistration is later than the candidate and its holdout results; random combinations are generated from a fixed seed and fixed budget pool, although they do not depend on results that have not yet been produced, this still does not constitute a fully frozen preregistration from the very beginning of the experiment. Currently, there is only evidence from PyTorch Q/DQ simulation of W8A8 on RUOD-R, with no evidence from TensorRT, Jetson, latency, power consumption, VisDrone, or SOTA.
