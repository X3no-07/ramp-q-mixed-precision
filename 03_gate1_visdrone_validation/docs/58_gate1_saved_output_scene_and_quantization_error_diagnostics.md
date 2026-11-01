# Gate1: Scenario of Saved Results and Quantization Error Diagnosis



**The net recovery across the three seeds remains positive, though contributions are concentrated and show both positive and negative inter-component values; high-score candidate boxes still exhibit numerical outliers.** This round only analyzes saved results, without model forward passes, re-calibration, reading new images or annotations, re-target matching, or AP computation. The original numerical thresholds and the nine quantization branches each pass `0/109` images, with conclusions unchanged.

Formal training, saved consistency, floating-point qualification, and independent scientific validation remain separate. Current results provide development leads for further research but do not support conclusions regarding real INT8 deployment or overall method superiority.

# # 1. Component Contributions and Leave-One Description

Using the existing 109 images, 6 risk-isolated components, three fixed calibration seeds, and four configuration-preserved target states, all members are covered without selecting favorable components. The isolated components do not equate to independently verified real-world scenarios; the three seeds share the same batch of targets.

The defined net recovery follows Report 57: within the set of targets already detected in the original FP32 run, the additional misses are calculated as recovery_count minus the original stable targets' side losses. This does not represent the total change in target detection across all targets.

| calibration seed | Number of components with positive, negative, and zero net effects | Total positive contribution P | Total negative contribution N | Net total sum T | Maximum positive contribution M | M/P | M/T | Limited net recovery excluding the component with maximum positive contribution |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026102201 | 4/2/0 | 60 | -4 | +56 | 38 | 63.33% | 67.86% | +18 |
| 2026102202 | 3/2/1 | 38 | -2 | +36 | 31 | 81.58% | 86.11% | +5 |
| 2026102203 | 5/1/0 | 44 | -1 | +43 | 28 | 63.64% | 65.12% | +15 |

The maximum positive contribution from the three seeds all comes from the `train/9999990_00000_d_0000001` component. This component includes 81/109 images and 5144/9059 valid targets. Its large contribution cannot be directly interpreted as greater per-image effectiveness or sensitivity. The denominator of M/P is the sum of all positive contributions; the denominator of M/T also includes negative contributions offsetting; these two ratios cannot be mixed or treated as independent evidence proportions.

The remaining restricted net recovery is positive for all 6 leave-one-component-out cases. However, removing the largest component under the second seed leaves only +5, showing that a positive aggregate does not imply consistent effects across components. This is a descriptive deletion analysis of fixed saved counts; it involves no retraining or inference and is not cross-validation, a significance test, or independent replication. No component was removed and no experiment added on this basis.

All components' F/Q/D/R detection rates, false positives, recoveries, side losses, and bidirectional target counts are fully preserved. The relative increase in FP32 false positives in Report 57 (209/168/315) is maintained, and reporting only partial recoveries is insufficient.

# # 2. Three sub-calibration ranges

Read the previously saved 88 Conv calibration ranges and compute, layer by layer, the minimum absmax, maximum absmax, and max/min ratio for the three seeds. No recalibration or re-reading of calibration images is performed, and thus there is no zero denominator.

| Fixed grouping ranges | Conv count | Median of max/min ratio | Maximum value of max/min ratio |
|---|---:|---:|---:|
| G1 | 6 | 1.119184 | 1.241475 |
| Other groupings | 82 | 1.107023 | 1.506890 |

The maximum ratio occurs in G8's `model.23.cv3.1.2`. Range differences are not limited to G1; these dimensionless descriptions cannot prove which layer causes misses or recoveries, nor serve as evidence for sensitivity, correlation significance, or calibration strategy superiority. All 88 layers and 8 groups are retained without selection based on results.

# # 3. Secondary Hierarchical Classification by Candidate Box Error

Each candidate box is always stratified according to the same original FP32 output, based on the ten-class maximum score, without shifting due to changes in quantization scores. The intervals are left-closed and right-open, with the final layer including 1; six layers cover all candidate boxes and are not subsets used for re-verification.

| Original FP32 Maximum Class Score | Number of Candidate Boxes in Figure 109 | Number of Coordinate Elements per Box | Number of Class Score Elements |
|---|---:|---:|---:|
| [0,0.001) | 822516 | 3290064 | 8225160 |
| [0.001,0.01) | 36819 | 147276 | 368190 |
| [0.01,0.1) | 24606 | 98424 | 246060 |
| [0.1,0.25) | 10433 | 41732 | 104330 |
| [0.25,0.5) | 8599 | 34396 | 85990 |
| [0.5,1] | 12627 | 50508 | 126270 |
| All | 915600 | 3662400 | 9156000 |

The lowest-scoring layer accounts for 89.8336% of all candidate boxes. Across three seeds and three quantization configurations in nine branches, it contributes 94.1128%–97.5953% of the box coordinate violations, but only 0.0769%–0.3757% of class score violations. The distributions of these two types of errors are different, and thus the low-scoring boxes cannot be attributed as the source of score-related issues.

Even when only examining the highest original FP32 score layer[0.5,1], the nine branches still exhibit separate violations in bounding boxes and scores: the number of bounding box violations ranges from 631 to 10324, and the number of score violations ranges from 8391 to 14866. Therefore, overall numerical failures cannot be attributed solely to irrelevant changes in low-score boxes. The high-score layer here serves only as evidential support and does not replace the original complete threshold.

The original limits remain `0.5+0.005×|FP32_reference|` for boxes and `0.002+0.02×|FP32_reference|` for scores, covering all 8400 candidates and all 14 components. Exact aggregation over six layers reconstructs the original element counts, violation counts, maximum absolute errors, and maximum error/tolerance ratios for all nine branches from the 981 per-image records. Every branch still passes on `0/109` images. No confidence-filtered or NMS subset was used to change the decision.

Complete per-layer means, medians, 90th/95th/99th percentiles, and maximum errors are shown in the machine results; percentiles are computed by combining all scalar absolute errors within the layer using NumPy linear interpolation, and empty layers are not padded with zeros. This is a secondary description defined after result preservation, not a confirmation experiment for unseen results.

# # 4. Execution and Artifact Verification

Both diagnostic assessments first freeze the read-only diagnostic contract and source hash, then proceed with the current round of description computation. Component diagnosis passes 12 synthetic tests, verifying 19 frozen sources, 1308 preserved count files, and 18 delivery members; error layering passes 7 boundary and reconstruction tests, with identity consistency observed across 1090 original arrays before and after reading, and ultimately 22 delivery validations are successfully passed.

The current forward pass, calibration, new candidate generation, image/annotation content loading, new target matching, and AP calculation are all 0. The old contract, training source, weights, report, checklist, and historical `FAIL_FIXED_TOLERANCES`, `FAILED_UNCHANGED` are retained as-is. Selection is limited to existing metadata, and Holdout continues to be sealed.

The independently run FP32 native official baseline for Figure 414 remains governed by the actual live state, process identity, complete checkpoint, and final state receipt; it cannot be declared complete merely because these diagnostic preservation steps have ended. A standalone final artifact validator has been prepared, which will be invoked upon successful native full-scale execution to verify 1242 filtered matrices, seven metrics, and alias cleanup; no additional model inference or official scoring will be performed.

# # 5. Evidence Entry Points

- [Primary Interpretation of Component and Calibration Scope](../../docs/DATA_AND_ARTIFACTS.md) and [Complete Component, False Positives, Leave-One-Out, and 88-Layer Results](../../docs/DATA_AND_ARTIFACTS.md).
- [Complete Machine Results of Error Layering](../../docs/DATA_AND_ARTIFACTS.md) and [Distribution Explanation](../../docs/DATA_AND_ARTIFACTS.md).
- [Component Delivery Verification](../../docs/DATA_AND_ARTIFACTS.md) and [Final Error Layering Delivery Verification](../../docs/DATA_AND_ARTIFACTS.md).

The current round of new delivery, machine decisions, and runtime observation are recorded in `../artifacts/gate1/20261024_saved_diagnostics_delivery_v1`; the new state snapshot is v19. The current evidence does not alter the boundaries of independent scientific acceptance, true INT8/TensorRT, or Orin board-end validation, which remain uncompleted.
