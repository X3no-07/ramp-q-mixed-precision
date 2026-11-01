# Gate1 Frozen Scorer: Stability, False Positive Composition, and Additional CPU Overhead



2026-10-26.

**The aggregated improvement has been re-verified, but the net false positive reduction is concentrated within a prefix-group subset, and is still insufficient to claim robustness. The additional CPU cost incurred by feature computation in the current implementation remains significant.**

The model, scorer, thresholds, and data members are all kept frozen. This round is a post-hoc diagnosis and does not involve any new model inference, parameter tuning, full AP evaluation, or threshold candidates. The diagnosis does not modify the previously established pass criteria, nor does it treat the removal of any group as a new method.

# # Concentration of Aggregate Gains

| Scope | Net Hit Change | Net False Positive Change | Number of Prefix Groups | Number of Cases Still Showing Joint Improvement After Removal of Any Single Group |
|---|---:|---:|---:|---:|
| Added 286 Images | +53 | -14 | 53 | 52/53 |
| New Prefix 47 Images | +11 | +35 | 21 | 0/21 |
| Merged 414 Images | +87 | -29 | 58 | 57/58 |

Among all 414 images, the 34 images under prefix 9999975 contributed a net reduction of 78 false positives and a net increase of 1 hit. The remaining 380 images showed a net increase of 86 hits and a net increase of 49 false positives. Within the added 286 images, 25 images share the same prefix and contributed a net reduction of 61 false positives; after removal, the remaining 261 images showed a net increase of 50 hits and a net increase of 47 false positives.

Therefore, the claim of 'overall false positive reduction of 29' does not imply consistent improvement across all scenarios. All original groups are retained, especially no group that would undermine the conclusion is removed, and no group that would favor the conclusion is deleted. The filename prefix is merely an existing proxy, not a validated scenario identity; this exhaustive removal check is not an independent replication, confidence interval, or statistical significance test.

In the new prefix 47 images, removing any one of the 21 prefix groups still fails to meet the joint improvement condition. The false positive issue cannot be resolved by removing the worst-performing group.

# # Origin of False Positive Changes

The following represents the net change in false positive geometry within a fixed candidate pool: first, same-class IoU ≥ 0.5; then, other-class IoU ≥ 0.5; followed by maximum IoU in [0.1, 0.5); finally, low-overlap residuals. This is a post-hoc description of preserved false positive annotations, not a causal proof or an official error classification; low overlap cannot be directly labeled as pure background.

| Scope | Same-class Overlap | Cross-class Overlap | Low IoU with Nearby Objects | Low Overlap Residual | Total |
|---|---:|---:|---:|---:|---:|
| Added 286 Images | -141 | +26 | +61 | +40 | -14 |
| New Prefix 47 Images | -16 | +11 | +30 | +10 | +35 |
| Merged 414 Images | -206 | +32 | +99 | +46 | -29 |

Across all 414 images, false positives from same-class overlap decreased by 206, but false positives from other-class overlap increased by 32, localization neighborhood increased by 99, and low-overlap residuals increased by 46, resulting in a net reduction of 29 false positives. The scorer does not simultaneously improve all errors. For the new prefix 47 images, the changes are −16, +11, +30, and +10, respectively, resulting in a net increase of 35 false positives.

| Scope | Newly Appeared FP Anchors | Anchors No Longer Classified as FP | Newly Detected Target IDs | Loss hits target ID |
|---|---:|---:|---:|---:|
| Added 286 Images | 391 | 405 | 118 | 65 |
| New Prefix 47 Images | 100 | 65 | 26 | 15 |
| Merged 414 Images | 580 | 609 | 179 | 92 |

FP identity is matched by the same image and raw_anchor_index; target identity follows the original annotation physical row. These two identities must not be conflated. Across all 414 images, there are a net gain of 87 hits, consisting of 179 newly added hits and 92 lost hits. The net gain cannot be interpreted as preserving all original hits.

# # Actual CPU time spent by current post-processing

On the same i9-14900HX computer, for the saved output of existing 286 images, each image is preheated 1 times and fixed evaluated 3 times, resulting in a total of 858 samples. These samples cover the candidate count with the minimum of 2, median of 436, and maximum of 1000. In this round, only the current code is tested, without modifying the implementation or comparing multiple optimized versions.

| Operation | Median (ms) | P95(ms) | Mean (ms) | Maximum (ms) |
|---|---:|---:|---:|---:|
| Only linear scoring is computed | 0.079150 | 0.175830 | 0.089757 | 0.323300 |
| 22-dimensional features + scoring + stable ranking | 13.953900 | 68.678275 | 20.711539 | 91.019900 |

Timing does not include file/annotation reading, image decoding, detector execution, original NMS, CUDA output transfer, and final serialization. This is not end-to-end speed, power consumption, or Jetson latency. The 23 linear parameters are minimal, but neighbor feature computation requires pairwise overlap calculations among bounding boxes. The main cost in the current implementation lies in the feature stage, and thus the additional computation cannot be considered free.

Future engineering optimizations could verify output equivalence; even if successful, they can only demonstrate reduced cost of the current implementation and cannot replace evidence of originality or independent generalization of the detection method.

# # Precision and paper contribution boundaries

With the frozen scorer, the AP across all 414 images is 11.940007%, while the historical FP32 same-set AP is 12.440710%, resulting in a difference of -0.500703 percentage points. This historical baseline reuses previously saved results without re-running old scorers; a significant FP32 precision gap remains.

[IoU-Net](https://arxiv.org/abs/1807.11590) has already leveraged localization confidence to improve candidate selection; [GFLV2](https://arxiv.org/abs/2011.12885) has utilized statistical edge distribution to estimate localization quality and improve ranking; [DetPTQ](https://arxiv.org/abs/2304.09785) has studied quantization setting selection relevant to detection tasks. These works indicate that quality scoring and task-related quantization are not novel contributions here. This round does not claim to reproduce their full methods, nor does it substitute measured values from the paper with its own empirical results.

The marginal improvement from a conventional scorer cannot be rebranded as an intervention contribution in a failed control. Current evidence lacks a stable mechanism that outperforms strong baselines, as well as actual data not involved in development and real INT8/board-level costs. Therefore, the paper method is not marked as successful, and Gate1 is not passed.

# # Saved and Reviewed

Receipt of one-off workflow process completion with actual exit codes 0 complete. 9 pre-checks passed; independent audit verified 6 outputs, 583 reused bindings, and all 15603 floating-point geometric records; recalculated results for all leave-one-out prefix groups across three ranges, 858 timing samples, and nanosecond-to-millisecond aggregations; no re-execution of timing, training, or model inference.

After timing completion, each baseline path was cross-checked for consistency of features, scores, and rankings against existing cache. All 414 images' per-target gains and losses, each group's removal results, FP additions/removals by identity and type, and all 858 original nanosecond samples were preserved. The timing reflects the conditions of this process and hardware, but does not guarantee equivalence on other machines.

Original raw values, contracts, thresholds, results, and reports are preserved as-is; Selection load is not read, Holdout is sealed, no cloud resources, no new timing tasks. Evidence directory: `../artifacts/gate1/20261026_scorer_robustness_cost_v1`.
