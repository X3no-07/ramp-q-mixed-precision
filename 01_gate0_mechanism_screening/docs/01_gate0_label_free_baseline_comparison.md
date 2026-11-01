# Gate 0 Unlabeled Baseline Detection-Level Comparison and Closeout Decision



> Decision: **Overall direction GO; mainline unchanged; selection model superiority not established.** G1+G5 descriptively outperforms the unlabeled baseline G1+G2 on Selection, but the advantage is minor, derived from the same data used for candidate selection, and the labeled sensitivity rule collides completely with G1+G5; therefore, Gate 0 cannot be written as the selection algorithm has already won.

# # 1. Same Budget Detection-Level Results

| Scheme | MAC Share | recovery | Collateral misses | Net Recovery (Rate) | Bidirectional Same-Target (Rate) | Stable-target retention |
|---|---:|---:|---:|---:|---:|---:|
| RAMP-Q Candidate G1+G5 | 22.35% | 100 | 8 | +92(77.31%) | 83(69.75%) | 99.45% |
| Unlabeled Output Error Baseline G1+G2 | 22.97% | 93 | 6 | +87(73.11%) | 73(61.34%) | 99.59% |

G1+G5 recovers 7 more targets than G1+G2, with 2 more collateral misses, resulting in a net recovery of 5 more; bidirectional same-target increases by 10, but the FP32 MAC share drops by 0.62 percentage points. The advantage of G1+G2 lies in fewer 2 collateral misses and a higher stable retention rate of 0.14 percentage points.

# # 2. Same-Target Pairwise Comparison

| Paired Items | G1+G5 Exclusive | G1+G2 Exclusive | Exploratory Two-Sided Sign Test |
|---|---:|---:|---:|
| Recovered Targets | 16 | 9 | p=0.230 |
| same-target bidirectional agreement | 20 | 10 | p=0.099 |
| Better Net Utility per Target | 19 | 14 | p=0.487 |

Per-target net utility is defined as: for all W8A8 missed detections, if they are recovered, count as +1; if originally stable and misclassified, count as -1; otherwise, count as 0. G1+G5 performs better on 19 targets, G1+G2 performs better on 14 targets, and there are 1538 targets with identical performance.

These p values are only supplementary diagnostics: the comparison rule was not pre-registered initially, and the candidate was selected from the very same Selection; therefore, even if the p value is smaller, selection bias cannot be eliminated. This result can only be described as a minor descriptive advantage and cannot be considered an independent, significant victory.

# # 3. Gate 0 Final Explanation

- Core Mechanism: GO. Module quantization systematically affects specific targets; recovery of G1 can be independently reproduced in a holdout setting, demonstrating large-scale recovery and high stability retention.
- Randomized Control under Same Budget: PASS. G1+G5 significantly outperforms three frozen random combinations.
- Unlabeled Output Error Baseline: G1+G5 has a minor descriptive advantage, but this is not an independently confirmed result.
- Labeled Overall Sensitivity Baseline: COLLISION. It also selects G1+G5; currently, there is no modal result difference that can serve as algorithmic novelty.
- Superiority of Selection Algorithm: NOT ESTABLISHED. Gate 0 is insufficient to support 'a new selection algorithm outperforms traditional methods'; however, the main line of research on target-level attribution and recovery remains unchanged.
- Mechanism and Reliability Main Line: GO. Further research can explore which small targets are sensitive to module quantization, why such sensitivity arises, and whether a generalizable reliability metric can be formed.

# # 4. Suggested paper positioning

The paper should not position 'automatically selecting G1+G5' as the sole core innovation. A more stable main thread should be: target-level quantization failure attribution, bidirectional validation of damage and recovery, small-object reliability metrics, and risk control under a given computational budget. Mixed-precision selection should be retained as an application outcome or secondary contribution.

# # 5. Next Steps

Close out RUOD-R Gate 0. No additional combinations will be pursued in the current Selection. Before entering VisDrone Gate 1, freeze the candidate generation rules, traditional labeled sensitivity, unlabeled output error, random controls, and independent validation splits. Only after completing these freezes without reviewing validation results can the selection algorithm be re-evaluated for any genuine added value.

# # Evidence Boundaries

Currently, this remains a simulation of RUOD-R under PyTorch Q/DQ on W8A8. There is no evidence of TensorRT INT8, Jetson latency/power consumption, VisDrone reproducibility, or SOTA comparisons. This report supports maintaining the target-level attribution and reliable recovery main threads and moving into the next Gate, which does not equate to the completion of method validation.
