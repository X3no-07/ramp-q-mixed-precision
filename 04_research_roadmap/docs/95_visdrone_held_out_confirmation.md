# VisDrone held-out confirmation

Report 95, November 8, 2026.

Across 1,610 VisDrone test-dev images, the G1 precision intervention recovered more targets, after subtracting collateral losses, than calibration redraws and the fixed module controls. The intervention tag was also associated with later changes in detection at the fixed QAT epoch 10 checkpoint, after matching on the prespecified covariates. Report 90's correction experiment remains a separate negative training result.

## Protocol

The run evaluated 19 configurations on all 1,610 images and 74,707 annotated targets. It made 30,590 model forwards with zero training updates. The images had no exact byte overlap with previous image roles. Scene independence and cross-role near-duplicate exclusion remain unverified. The model, calibration draws, detection thresholds, endpoints, and analysis rules were fixed before running inference on these images.

The measurement uses the project's matching rule at confidence 0.25 and IoU 0.50. For each calibration, the denominator is the set of targets hit by floating-point inference and missed by quantized inference. Net recovery subtracts losses among targets hit by both of those baselines from the number of baseline misses recovered by the intervention. Intervals use 2,000 paired bootstrap resamples of whole images.

The cloud run used one RTX 5090 with driver 580.105.08, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.142, NumPy 2.2.6, OpenCV 5.0.0.93, and Pillow 12.3.0. All 35 qualification forwards on existing development images matched the frozen reference arrays elementwise. The initial launch stopped before model execution because of CRLF shell line endings. Converting the launcher to LF in memory resolved the error. All 33 bound inputs were verified against the original manifest before the successful run. The completed analysis replayed all 30,590 saved image/configuration records and matched their saved scores, assignments, and false-positive counts.

## Recovery results

| Calibration draw | Baseline misses in scope | Recovered | Collateral losses | Net recovery | Net recovery / misses |
|---|---:|---:|---:|---:|---:|
| 0 | 1,479 | 936 | 467 | 469 | 31.71% |
| 1 | 1,687 | 996 | 479 | 517 | 30.65% |
| 2 | 1,484 | 984 | 430 | 554 | 37.33% |

The mean net recovery rate across the three draws was **33.23%** (95% CI [30.62%, 35.80%]). The corresponding mean over all six ordered calibration redraw comparisons was **−0.56%**. The difference was **+33.79 percentage points** (95% CI [31.19, 36.48]).

On the same target pool for draw 0, G1 had 469 net recoveries, compared with 31, 58, and 27 for G8, G3, and G2. The difference against the mean control was **+29.10 points** (95% CI [25.03, 32.89]). G1+G5 had 513 net recoveries, compared with 72, 84, and 58 for G2+G8, G3+G8, and G4+G8, a difference of **+29.86 points** (95% CI [25.55, 34.30]).

On this pool, MSE calibration alone had 582 net recoveries, and MSE calibration with G1 had 720. Only 103 targets were recovered in all three absmax draws, and pairwise Jaccard similarity ranged from 0.159 to 0.164. These comparisons cover a fixed set of modules and calibrations. They measure target-level recovery in the specified pools.

## QAT epoch 10 association

The tag identifies a target whose hit state differs between MSE-quantized inference and the corresponding G1 intervention. The fixed rule selects at most eight such targets per image. The primary comparison matches tag-positive targets to tag-negative controls on starting hit state, class, size, candidate availability, score margin, and floating-point versus intervention score disagreement. Control rates are weighted to the stratum composition of the matched tagged targets.

| Starting state and outcome | Matched / eligible tagged targets | Tagged rate | Weighted control rate | Difference (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| Hit, then missed | 698 / 768 | 42.84% | 25.41% | +17.42 | [12.98, 21.87] |
| Missed, then hit | 686 / 770 | 45.92% | 31.34% | +14.58 | [10.89, 19.79] |
| Either change, descriptive | 1,384 / 1,538 | 44.36% | 28.35% | +16.01 | [12.98, 19.79] |

All 21 prespecified epoch 10 comparisons had intervals above zero, including additional IoU stratification, within-image matching, and the filename-prefix bootstrap. The strict within-image comparison matched 290 of 1,538 tagged targets, so it applies to a narrower subset. Epochs 3 and 13 are correlated checkpoints of the same trajectory. The intervals are unadjusted for multiple comparisons.

The held-out results support an association with both subsequent losses and gains under the fixed model and matching protocol. Training gains, real INT8 deployment performance, performance across training seeds, and independent RUOD-R confirmation remain open.

The [result index](../../results/visdrone_testdev_20261108/README.md) links the numerical tables, the analysis contract, and the verification records.
