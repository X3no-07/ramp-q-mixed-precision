# Gate 0 Holdout: Summary of Bidirectional Validation between G1 and G5



> Current Conclusion: **G1 and G5 pass their respective pre-registration thresholds on an independent holdout, producing consistent bidirectional signals. Final gate judgment still requires completion of a G1 single-module holdout control.**

# # Data Scope

- Holdout images: 150 images.
- Original FP32 detections: 722 targets.
- Additional missed detections in full W8A8: 46 targets.
- Configuration locked to G1+G5 before viewing holdout module results.

# # Bidirectional Results

| Metric | Selection | Holdout |
|---|---:|---:|
| Missed detections in destructive side overlap with full W8A8 missed detections | 100/119(84.03%) | 31/46(67.39%) |
| Recovery side recovery | 100/119(84.03%) | 43/46(93.48%) |
| Collateral misses | 8 | 5 |
| Stable target retention rate | 99.45% | 99.26% |
| Net recovery | +92 | +38 |
| same-target bidirectional agreement | 83/119(69.75%) | 29/46(63.04%) |

Among the holdout, 31 of the full W8A8 newly missed targets disappear when quantized independently under G1+G5; 43 targets are recovered upon recovery of G1+G5, of which 29 belong to the same batch of bidirectionally consistent targets. Additionally, 2 targets are only valid on the destruction side, and 14 targets are only valid on the recovery side.

# # Candidate Self-Thresholds

| Holdout freeze threshold | Result | Decision |
|---|---:|---|
| Recovery rate is not less than 15% | 93.48% | PASS |
| The stability of object-level retention rate is not less than 98% | 99.26% | PASS |
| Net recovery is positive | +38 | PASS |
| at least 1 bidirectional same-object targets | 29 | PASS |
| Not dominated by a single image or a single category | See below | PASS |

# # Concentration Check

- 43 recovered targets distributed across 25 images and 7 categories.
- Maximum contribution from a single image is 5 recovered targets, accounting for 11.63%.
- The largest category is category 0, with 12 targets, accounting for 27.91%.
- 29 bidirectional same-target pairs distributed across 19 images and 7 categories; maximum per image is 10.34%, maximum per category is 31.03%.

The results are not dominated by any single image or category. However, the holdout set exhibits only 46 additional missed detections, and the proportion still shows sampling variability; thus, 93.48% cannot be regarded as a universally valid recovery rate.

# # Current Decision Boundary

- G1+G5 show positive net recovery, high stability in target retention, and a large number of bidirectional consistent samples at the object level, thus supporting the claim that 'the candidate exhibits reproducible signals'.
- The incremental advantage of G5 over G1 in the holdout set remains unverified; further execution of the preregistered `q_only:G1` and `r_only:G1` comparison is required.
- The results herein use PyTorch Q/DQ to simulate W8A8 quantization, and do not represent TensorRT INT8 or Jetson performance, speed, or power consumption evidence.
