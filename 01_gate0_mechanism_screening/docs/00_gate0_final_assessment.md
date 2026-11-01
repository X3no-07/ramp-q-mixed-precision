# RAMP-Q Gate 0: Final Audit and Decision

> **Final decision: GO for the research direction; superiority of module selection is unproven.** G1's module-level disruption–recovery signal reproduced on an independent Holdout, and G1+G5 passed comparison with three frozen budget-matched random combinations. However, the labeled aggregate-sensitivity baseline selected the same G1+G5 combination, and the label-free G1+G2 baseline achieved similar results. The research question remains object-level quantization failure attribution and reliable recovery. The evidence does not establish that a new module-selection algorithm outperforms conventional methods.

This report describes the Gate 0 decision at its original stage. Later deployment and VisDrone results are recorded separately; see the [current status](../../docs/STATUS.md).

## 1. Data and separation

| Item | Selection | Holdout |
|---|---:|---:|
| Images | 350 | 150 |
| FP32-hit targets | 1571 | 722 |
| Additional misses under full W8A8 | 119 | 46 |
| Image-set overlap | 0 | 0 |

- Selection was used for the G1–G8 scan, combination selection, and threshold definition.
- Holdout ran only the bidirectional G1+G5 and G1 controls allowed by the candidate-lock record.
- No G1+G3 switch or additional module combination was introduced after viewing Holdout results.

## 2. Reproduction of the core result

| Configuration | Split | Recovered targets (rate) | Collateral misses | Net recovery | Same-target bidirectional agreement | Stable-target retention | MAC share |
|---|---|---:|---:|---:|---:|---:|---:|
| G1 | Selection | 94/119 (78.99%) | 13 | +81 | 69/119 (57.98%) | 99.10% | 10.32% |
| G1 | Holdout | 40/46 (86.96%) | 4 | +36 | 25/46 (54.35%) | 99.41% | 10.32% |
| G1+G5 | Selection | 100/119 (84.03%) | 8 | +92 | 83/119 (69.75%) | 99.45% | 22.35% |
| G1+G5 | Holdout | 43/46 (93.48%) | 5 | +38 | 29/46 (63.04%) | 99.26% | 22.35% |

G1 exhibits high recovery, positive net recovery, high stable-target retention, and substantial same-target bidirectional agreement on both splits. This reproduces the core feasibility signal: interventions on a small number of modules can explain and recover a substantial fraction of the additional misses in simulated W8A8.

## 3. Incremental value of G5

| G1+G5 relative to G1 | Selection change | Holdout change |
|---|---:|---:|
| Recovered targets | +6 | +3 |
| Collateral misses | -5 | +1 |
| Net recovery | +11 | +2 |
| Same-target bidirectional agreement | +14 | +4 |
| Stable-target retention | +0.35 percentage points | -0.15 percentage points |
| FP32 MAC share | +12.03 percentage points | +12.03 percentage points |

G5 retains a positive incremental direction on Holdout, so there is no complete reversal. However, it recovers only 3 more targets and adds only 2 net recoveries while increasing FP32 MAC share by 12.03 percentage points. The evidence does not establish a better accuracy–compute Pareto trade-off for G1+G5 than for G1.

## 4. Preregistered Holdout criteria

| Criterion | G1+G5 Holdout | Decision |
|---|---:|---|
| Recovery rate at least 15% | 93.48% | PASS |
| Stable-target retention at least 98% | 99.26% | PASS |
| Positive net recovery | +38 | PASS |
| At least 1 same-target bidirectional match | 29 | PASS |
| Recovery, net recovery, and bidirectional metrics must not all be worse than G1 | +3, +2, +4 | PASS |
| Not wholly dominated by one image or one class | Maximum shares 11.63% / 27.91% | PASS |

G1+G5 passes its own Holdout criteria. Those criteria establish directional consistency and reproducibility; they do not establish a sufficiently large engineering benefit over G1.

## 5. Gate-level findings

| Question | Status | Evidence |
|---|---|---|
| Are there enough additional-miss examples? | PASS | 165 on the complete screen, 119 on Selection, 46 on Holdout |
| Can sensitivity be localized to a few modules? | PASS | G1 is substantially more sensitive than other modules |
| Do disruption and recovery involve the same targets? | PASS | G1 has 69/25 bidirectional targets on Selection/Holdout |
| Does recovery preserve stable targets? | PASS | G1 and G1+G5 both exceed 98% |
| Does the signal reproduce on independent Holdout? | PASS | G1 and G1+G5 agree in direction and have positive net recovery |
| Does G5 provide sufficient incremental efficiency? | INCONCLUSIVE | Only 3 additional recoveries and 2 net recoveries on Holdout for 12.03 percentage points more MAC share |
| Is the candidate better than three frozen budget-matched random combinations? | PASS | Candidate net recovery rate 77.31%, random mean 1.68%; all four preregistered criteria pass |
| Does it differ from labeled aggregate-sensitivity ranking? | COLLISION | The conventional rule also selects G1+G5; independent module-selection novelty is unproven |
| Does it outperform the label-free output-error baseline? | NOT CONFIRMED | Net recovery +92 for G1+G5 versus +87 for G1+G2; paired net utility 19 versus 14, exploratory p=0.487, with Selection bias |
| Is real TensorRT/Jetson performance established? | NOT TESTED | This stage uses PyTorch Q/DQ W8A8 simulation |

## 6. Decision and limits

### Supported conclusions

For the tested YOLO11n, RUOD-R data, and simulated W8A8 setup, early-backbone group G1 exhibits stable, reproducible module-level sensitivity associated with additional quantization misses. Restoring G1 recovers most of those misses while retaining more than 99% of previously stable targets.

On Selection, G1+G5 substantially exceeds the three fixed budget-matched random combinations. Against the label-free G1+G2 baseline, it obtains 5 more net recoveries and 10 more same-target bidirectional matches at a slightly smaller MAC share. That difference remains descriptive.

### Unsupported conclusions

- G1+G5 is not established as the optimal mixed-precision configuration.
- Its advantage over these three fixed random combinations on this Selection split does not establish general superiority over random selection.
- Target-risk-guided selection is not shown to outperform conventional labeled sensitivity ranking, because both choose the same combination.
- Significant superiority over the label-free output-error baseline is not established. G1+G2 completed bidirectional validation with similar results; the paired same-target net-utility comparison is 19 versus 14, the exploratory two-sided sign test gives p=0.487, and G1+G5 was selected on the same Selection split.
- This report does not establish transfer to VisDrone, TensorRT, or Jetson.
- Simulated W8A8 results are not real INT8 latency, energy, or deployment gains.

### Direction and stopping conditions

- **GO: mechanism and reliability.** Continue studying why particular targets disappear under particular module quantization, same-target bidirectional disruption–recovery consistency, and reliability measures for weak small objects.
- **Module-selection superiority remains unproven.** The labeled rule collides with RAMP-Q's selection and the label-free baseline is close. Mixed precision remains an attribution intervention and deployment candidate, without a superiority claim over conventional selection.
- **STOP condition:** if the directional signal does not reproduce on later VisDrone or real TensorRT INT8 checks, stop generalizing the current mechanism conclusion.

## 7. Next stage

RUOD-R Gate 0 is closed without additional combinations on the current Selection or Holdout. The original next step is a small Orin/TensorRT Gate 0.5 deployment-reality check, followed by VisDrone Gate 1. Before inspecting Gate 1 outcomes, freeze candidate generation, labeled-sensitivity and label-free-output-error baselines, random controls, and independent data splits.

The research direction remains object-level quantization failure attribution and reliable recovery, while testing under a clean design whether object-level information adds module-selection value.

## Evidence boundary

This Gate establishes feasibility only for PyTorch Q/DQ simulated W8A8 on RUOD-R. It motivates the next stage; it is neither the paper's final experiment nor a state-of-the-art result.
