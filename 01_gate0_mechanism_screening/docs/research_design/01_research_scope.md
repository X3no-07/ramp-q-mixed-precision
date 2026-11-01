# 01 Direction: Attribution and Reliable Recovery of Object-Level Quantization Failure



> Project name: RAMP-Q<br>
> Status snapshot: 2026-09-11<br>
> Current decision: **Overall direction GO; research mainline unchanged; superiority of selection model not established.**

# # 1. A one-sentence summary

This paper investigates:

> Why does INT8 make certain individual small objects disappear, which network modules are associated with those losses, and can restoring the precision of the relevant modules recover the same objects while minimizing harm to other targets?

mixed precision is not an independent objective, but rather:

1. An experimental means of checking whether module attribution identifies the relevant modules;
# # 2. Has the Mainline Changed?

# # 2. Has the main line actually changed?

No change. From the beginning to now it has always been:

```text
Identify specific targets detectable by FP32 but missed by INT8
→ Conduct quantization disruption experiments per module
→ Conduct precision recovery experiments per module
→ Check whether the same target disappears and reappears
→ Statistically analyze collateral misses and high-precision costs
→ Validate on real edge devices
```

Gate 0 does not change the problem statement, but it alters the strength of the claims we can make:

- We can claim: object-level quantization failure attribution and bidirectional recovery framework;
- Supported claim: same-target bidirectional agreement is a stricter criterion than aggregate accuracy loss or recovery count alone;
- We can present mixed-precision configurations as validation and deployment outcomes;
- We currently cannot claim: the new selection algorithm significantly outperforms traditional sensitivity-based methods.

Consolidated as:

> **Direction GO; object-level attribution and reliable recovery remains the main thread; superiority of selection strategy remains unproven.**

# # 3. Suggested Paper Title

Current Chinese title:

> **Object-level Quantization Failure Attribution and Reliable Recovery for Edge Small Object Detection**

Current English title:

> **Target-Level Attribution and Reliable Recovery of Quantization-Induced Misses in Edge Small-Object Detection**

If Gate 1 demonstrates, under clean design, that object-level rules outperform traditional selection methods, it can be upgraded to:

> **RAMP-Q: Target-Risk-Aware Mixed-Precision Quantization for Reliable Small-Object Detection**

Prior to that, do not present 'Risk-Aware Selector' as an established primary contribution.

# # 4. Core Concepts

## # 4.1 Quantization-Induced Additional Misses

The same real target must satisfy:

```text
Detected correctly by FP32
Missed by full W8A8
```

Targets not detected by FP32 originally cannot be attributed to quantization.

## # 4.2 Destruction and Recovery Sides

- `q_only:Gj`: Quantize only `Gj` and keep the rest in FP32; check whether quantizing this module is sufficient to cause the target to disappear;
- `r_only:Gj`: Start from full W8A8 and recover only `Gj`; check whether recovering this module is sufficient to bring the target back.

## # 4.3 Bidirectional Same-Target Requirement

A truly interpretable target must simultaneously belong to:

```text
q_only:Gj destruction set
∩ new misses in full W8A8 set
∩ r_only:Gj recovery set
```

For example, the G4+G8 destructive side overlaps with the full W8A8 on 13 targets, and the recovery side also recovers 13 targets, but only 1 corresponds to the same individual target. Therefore, equal totals between the two sides do not guarantee reliable attribution.

## # 4.4 Reliable Recovery Metrics

- Rescue: newly missed detections recovered;
- Collateral Miss: stable targets incorrectly affected during recovery configuration;
- Net Rescue: recovery_count minus collateral_miss_count;
- Stable Target Retention: retention rate of original stable targets;
- Bidirectional Same-Target Rate: whether destruction and recovery point to the same target;
- Cost: MAC, real latency, and energy consumption of preserving high-precision modules.

# # 5. Gate 0 Final Evidence

## # 5.1 Data and Split

| Item | Screen | Selection | Holdout |
|---|---:|---:|---:|
| Image | 500 | 350 | 150 |
| True Target | 2,785 | 1,920 | 865 |
| FP32 Hit | 2,293 | 1,571 | 722 |
| New Misses in Full W8A8 | 165 | 119 | 46 |

Selection and Holdout are split by full images with no overlap, and the random seed is `20260910`.

## # 5.2 G1 Core Signal

| Data | recovery | Collateral misses | Net recovery | same-target bidirectional agreement | Stable-target retention | MAC Proxy |
|---|---:|---:|---:|---:|---:|---:|
| Selection | 94/119 | 13 | +81 | 69/119 | 99.10% | 10.32% |
| Holdout | 40/46 | 4 | +36 | 25/46 | 99.41% | 10.32% |

Supported conclusions:

> Under the current RUOD-R, YOLO11n, and simulated W8A8 conditions, there is a stable and reproducible intervention relationship between the early backbone G1 and quantization failure of numerous specific object-level targets; restoring G1 enables many identical targets to reappear while maintaining over 99% of the original stable targets.

This is not a proof that 'G1 is the sole physical root cause'. Inter-layer nonlinear interactions, thresholds, NMS, and matching competition may still contribute to the observed results.

## # 5.3 G1+G5 Candidate

| Data | recovery | Collateral misses | Net recovery | same-target bidirectional agreement | Stable-target retention | MAC Proxy |
|---|---:|---:|---:|---:|---:|---:|
| Selection | 100/119 | 8 | +92 | 83/119 | 99.45% | 22.35% |
| Holdout | 43/46 | 5 | +38 | 29/46 | 99.26% | 22.35% |

G1+G5 increases only 3 recoveries and 2 net recoveries in Holdout relative to G1, while consuming an additional 12.03 percentage points of FP32 MAC. Therefore, G1+G5 is not a confirmed engineering Pareto-optimal solution, and G1 alone must be evaluated on real devices.

## # 5.4 Randomized Control with Same Budget

| Scheme | MAC Proxy | net_recovery_rate | same-target bidirectional agreement |
|---|---:|---:|---:|
| G1+G5 | 22.35% | 77.31% | 83 |
| G4+G8 | 22.54% | -1.68% | 1 |
| G3+G8 | 21.08% | 5.88% | 2 |
| G2+G8 | 21.91% | 0.84% | 1 |

G1+G5 passes the preregistered criterion against the three fixed random controls. This indicates that the result is not obtained by arbitrary selection of two modules under the same budget. However, there are only 3 random controls, frozen after the candidate and its Holdout results were available but before the random-control results were seen. The paper must disclose this chronology.

## # 5.5 Traditional Model Selection Baseline

The labeled overall sensitivity method also selects G1+G5, resulting in a selection collision. Therefore, the fact that 'G1+G5 was selected' cannot be considered a novel advantage over traditional methods.

Unlabeled NMS-before-output reconstruction error selects G1+G2:

| Scheme | MAC Proxy | Net recovery | same-target bidirectional agreement | Collateral misses |
|---|---:|---:|---:|---:|
| G1+G5 | 22.35% | +92 | 83 | 8 |
| G1+G2 | 22.97% | +87 | 73 | 6 |

G1+G5 has a minor descriptive advantage, but its per-target net utility shows only 19 wins and 14 losses, with an exploratory two-sided sign test `p=0.487`, and selection bias is present. Therefore, it cannot be claimed to significantly outperform the unlabeled baseline.

# # 6. Gate 0 Determination

| Hypothesis or Target | Status | Explanation |
|---|---|---|
| Quantization-induced selective misses exist | PASS | 165 additional misses out of 500 images |
| A small number of modules are correlated with specific target failures | PASS | G1 strong signal with Holdout reproducibility |
| Recovery of relevant modules can recover the same target | PASS | G1 Selection/Holdout has 69/25 bidirectional targets |
| Object-level rules outperform random module selection | Local PASS | Only passes on three fixed Selection random combinations |
| Object-level rules outperform traditional module selection | Not verified | Label collisions exist; label baseline is close |
| Real TensorRT/Orin still holds | Not validated | Currently only PyTorch Q/DQ |
| Non-submerged VisDrone still holds | Not validated | Currently only RUOD-R |

Overall decision:

> **The research direction is feasible and should continue. The truly supported aspects are object-level failure attribution and reliable recovery; the superiority of the new selection algorithm requires re-evaluation in Gate 1.**

# # 7. How to write the paper's contributions

Main contribution candidates:

1. **Object-level quantization failure perspective.** Instead of merely reporting average mAP drop, it tracks specific events where individual targets go from FP32 hit to INT8 miss.
2. **Bidirectional module intervention.** Simultaneously applies local quantization damage and local recovery within full INT8, requiring correspondence of the same target in both directions.
3. **Reliable recovery evaluation.** Simultaneously compute recovery, collateral misses, net recovery, stable object retention rate, bidirectional consistency, and precision cost.
4. **Cross-domain and real-world deployment evidence.** Only after VisDrone and Orin are completed can this contribution be formally recognized.

Secondary contribution candidates:

5. **Mixed-precision deployment.** Recover a small number of modules based on attribution results, demonstrating target reliability and cost-efficiency trade-offs.

Temporarily cannot write:

- "A novel module selection algorithm significantly outperforming traditional methods";
- "G1+G5 is the globally optimal mixed-precision configuration";
- "First study on INT8 small-object detection misses";
- "Already achieved TensorRT/Jetson acceleration";
- "RUOD-R results prove effectiveness on all small-object datasets".

# # 8. Still missing a true 'why'

Gate 0 primarily addresses "which modules are related to the disappearance and recovery of specific targets"; however, it has not yet provided a complete explanation of the underlying numerical mechanisms. The next phase should investigate:

1. Whether dynamic range activation is dominated by a small number of large values;
2. Whether small-object features are drowned by the INT8 quantization step size;
3. Whether quantization noise amplifies continuously along early high-resolution features;
4. How the confidence score for correct classes and margins against competing classes evolve;
5. The relative contributions of localization IoU degradation, confidence threshold drop, and NMS competition;
6. Whether G1 recovery improves target-region signal, not just global output error.

If the analysis stops at heatmap or correlation alone, it cannot claim to explain causal mechanisms. At minimum, evidence such as frozen metrics, target-region versus background controls, intervention-based disruption-recovery experiments, and cross-dataset reproducibility must be provided.

# # 9. When to deploy Orin

Now proceed to a small-scale Gate 0.5, rather than waiting until the full VisDrone dataset is completed before conducting the first device deployment.

## # 9.1 Gate 0.5 Objective

Early validation of whether PyTorch Q/DQ observations can be transferred to real TensorRT. If real INT8 does not introduce similar additional misses, or if TensorRT fusion fails to control the precision of target modules, the strategy must be adjusted immediately.

## # 9.2 Minimum Configuration

Build under the same Orin, JetPack, TensorRT, power mode, clock speed, thermal conditions, batch=1, and input size:

1. FP16;
2. Full INT8;
3. G1 remains in FP16, all others in INT8;
4. G1+G5 remain in FP16, all others in INT8;
5. G1+G2 remain in FP16, all others in INT8.

It is essential to inspect the engine inspector or equivalent layer information to confirm that precision constraints are actually enforced. A successful engine build does not guarantee that layers operate as intended.

## # 9.3 Gate 0.5 Pass Criteria

- Full INT8 introduces sufficient additional misses relative to FP16 for analysis;
- G1 recovery must recover a batch of identical targets, not just improve overall detection counts;
- The stability of target retention remains high;
- Mixed precision is not achieved due to extensive implicit fallback to FP16;
- Latency does not fully degrade to FP16 levels.

Gate 0.5 is solely for real-world validation; complete P50/P95, power, and energy per frame metrics are deferred to Gate 2.

# # 10. Gate 1: Clean Validation on VisDrone

RUOD-R Selection has already been used for scanning, combination selection, and comparison. No additional experiments should be added to package minor advantages as confirmed results. VisDrone Gate 1 must freeze all rules prior to reviewing validation outcomes.

Minimum workflow:

1. Fix YOLO11n, input size, training epochs, and random seed;
2. Establish FP32, FP16, and full W8A8 baseline models;
3. Split the dataset into Selection and final Holdout using whole-image partitioning;
4. Apply bidirectional interventions across G1–G8 or equivalent architectures on the Selection set;
5. Freeze object-level attribution and recovery scoring;
6. Freeze the overall labeled sensitivity and unlabeled output error baselines;
7. Freeze random combinations under the same FP16 MAC budget;
8. Only unlock the Holdout after all rules are fully locked;
9. Report all results, including cases where both object-level and traditional methods select the same combination;

Gate 1 primary success criteria:

- Sufficient `FP32-hit → W8A8-miss` small objects exist in VisDrone;
- At least one module exhibits bidirectional same-object signals reproducible on an independent Holdout;
- The mechanism is not dominated by a single class, a small number of images, or a single confidence threshold;
- Small, medium, and large object layers show distinct performance differences for weak small objects;
- Object-level metrics show additional explanatory power compared to overall mAP alone;
- To upgrade the contribution of the selection algorithm, it must outperform traditional baselines on an independent Holdout;

The last criterion fails, but the first five pass, the paper may still proceed with analysis of object-level quantization reliability and mechanism;

# # 11. Gate 2: Full Orin Deployment Validation

After candidate schemes, metrics, and measurement protocols are frozen, conduct:

- Real-world TensorRT FP16, full INT8, and mixed-precision accuracy evaluations;
- batch=1 end-to-end and pure model latency;
- P50/P95, throughput, and memory usage;
- average power and energy per frame under fixed power consumption mode;
- overhead of transitioning from FP16 to INT8;
- precision-speed-energy comparison across G1, G1+G5, and G1+G2;

If mixed precision is not faster than FP16, or if implicit fallback results in most layers operating in FP16, then a Pareto advantage for deployment cannot be claimed.

# # 12. Minimum Paper Experiment Matrix

Algorithms and detection baselines: FP32, FP16, full W8A8 PTQ, reproducible QAT/LSQ-based detection quantization, same budget random selection, labeled overall sensitivity, unlabeled output/feature reconstruction error, target-level bidirectional attribution and recovery schemes.

Detection metrics: mAP50-95, mAP50, Precision, Recall, AP_S or small object recall, rate of additional missed detections due to quantization, Rescue, Collateral Miss, Net Rescue, Bidirectional Same-Target Rate, Stable Target Retention, and stratification by area, contrast, crowding, category, and failure modes.

Device metrics: end-to-end and pure model latency, P50/P95, throughput, memory usage, average power, energy per frame, per-layer actual precision, and conversion nodes.

# # 13. Major Risks and Stop Conditions

| Risk | Current status | Handling |
|---|---|---|
| Only jittering near confidence threshold | High | Conduct freeze sensitivity analysis at 0.20/0.25/0.30 confidence levels, split by confidence, localization, and NMS failure |
| Simulated quantization cannot be migrated to TensorRT | High | Immediately perform Gate 0.5 |
| Only valid on underwater data | High | VisDrone serves as the primary non-underwater benchmark |
| Only the G1/YOLO11n mechanism is effective | Medium to high | After Gate 1, supplement with YOLO11s or a second lightweight detector |
| Select the same modules using traditional methods | Has already occurred | Do not treat the selected module result as the primary innovation |
| The gain from G1+G5 does not justify the cost | Existing risks are present | Measure G1 separately and recover it simultaneously on Orin |
| The boundary transition in mixed precision cancels out the speed gain | Unknown | Record the precision at the engine layer and at each conversion node |

Cease or scale back the claims:

- TensorRT's real INT8 cannot reproduce object-level additional misses or G1 recovery;
- VisDrone cannot reproduce any stable bidirectional module signal;
- Results are entirely determined by a single threshold or a few images;
- The so-called mixed precision actually results in a wide-range fallback to FP16;
- The recovery scheme is either comparable to or slower than FP16 and shows no significant reliability improvement.

# # 14. Recommended Execution Order This Week

1. Freeze the Orin model version, JetPack/TensorRT version, and five engine configurations at Gate 0.5
2. Confirm whether TensorRT supports and actually enforces the required local precision constraints;
3. Conduct authenticity checks for FP16, full INT8, G1, G1+G5, and G1+G2 using a small number of RUOD-R runs;
4. Establish a baseline using VisDrone data and YOLO11n, but do not activate the final Holdout split;
5. Freeze the pre-registration of Gate 1 after Gate 0.5 passes;
6. After Gate 1 begins, no return to RUOD-R Selection for searching new combinations;

# # 15. File Boundaries

Directory Path:

```text
[LOCAL_PROJECT]/06_legacy_direction
```

Save the paper's positioning, research questions, contribution boundaries, gate pathways, and decision rationale;

Engineering Directory:

```text
[LOCAL_PROJECT]\legacy_engineering
```

Save code, configurations, data lists, logs, prediction CSVs, machine-readable results, and experimental reports;

# # 16. Evidence Sources and Boundaries

This direction document is updated based on the final evidence from engineering;

1. `00_gate0_final_assessment.md`;
2. `04_gate0_holdout_g1_g5_validation.md`;
3. `02_gate0_random_control_decision.md`;
4. `03_gate0_label_free_output_error_baseline.md`;
5. `01_gate0_label_free_baseline_comparison.md`;
6. `artifacts/tables/gate0_final_decision.json`;
7. `artifacts/tables/gate0_label_free_detection_comparison.json`.

Conference submission deadlines, state-of-the-art results, and related work information retain the records from the original proposal as of 2026-09-09; no online refresh has been performed. All Related Work sections and submissions must be re-searched and verified before formal writing and submission;

Numerical conclusions are valid only for RUOD-R, the current YOLO11n weights, and PyTorch Q/DQ simulation of W8A8. Unless new evidence is provided by Gate 0.5, Gate 1, or Gate 2, extrapolation to TensorRT, Jetson, VisDrone, other detectors, or real power consumption is not permitted.
