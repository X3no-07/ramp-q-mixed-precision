# Gate1 Scoring Module Feature Computation: Strict Equivalence with CPU Optimization



2026-10-26.

**A fixed 128-row tiling implementation has passed bitwise consistency checks across 592 saved outputs. In paired measurements, the CPU median execution time for features + scoring + stable ranking decreased from 13.647 ms to 3.762 ms, resulting in a median ratio of 3.63.**

This result represents current engineering optimization within the implementation. The detection box, class, score, ranking, threshold, and existing detection accuracy remain unchanged; it does not address the growth of false positives in new prefix 47 images or insufficient paper contributions.

# # Unique Implementation and Consistency

The original implementation constructed a full pairwise intersection array per image. The new implementation uses a fixed 128-row tile, separately computing the intersection of width and height, preserving the same float64 computation order and identical index rules, while reducing temporary arrays. Only the tile size and one code implementation are registered—no faster configuration is searched.

All 592 members include 286 newly developed images, 128 original development images, and 178 calibration images. Each image's 22-dimensional feature vector matches bit-for-bit with the original implementation and original cache, with bit-for-bit equal scoring from the same frozen 23-parameter model and strictly identical stable ranking indices. No error tolerance was relaxed, no new images or annotations were read, and no new detector inference was introduced.

The threshold remains at 0.17473852372359241, and the model SHA value is preserved. The new function is provided only as a new engineering entry point; historical frozen scripts, contracts, and weights are unchanged. Future experiments using this implementation require binding its identity in a new contract and cannot override the original training source.

# # On-machine Paired CPU Timing

| Implementation | Median (ms) | P95ms | Mean (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| Original complete intersection array | 13.646700 | 71.098360 | 21.209864 | 93.313900 |
| Fixed 128-row tiling | 3.762050 | 19.441110 | 5.747852 | 24.100000 |

286 sets of identical saved outputs, each undergoing 1 preheating iterations per image, followed by 3 paired measurements, resulting in a total of 858 samples per realization; measurements are alternated between new and old implementations based on image index and repetition index. Among 858 pairs, optimization is performed on 851 pairs. The median ratio of pairwise execution times is 3.468, and this ratio is compared against the median execution times of the two overall populations, with both medians preserved.

Timing only includes 22-dimensional features, frozen CPU scoring, and stable Python index-based ranking. File I/O, image decoding, detector inference, NMS, CUDA output transfer, annotation reading, and final serialization are excluded. The test machine is an i9-14900HX; this is not a Jetson, power, or end-to-end detection pipeline performance metric. The CPU segment ratio cannot be interpreted as an end-to-end acceleration factor.

# # Original False Positive and Robustness Issues Remain Unchanged

The original 414-image set showed a net increase of 87 false positives and a net decrease of 29 false negatives under a fixed threshold. However, one filename prefix group contributed a net reduction of 78 false positives; removing the rest of that group still resulted in a net increase of 49 false positives. The new prefix 47 images showed an increase of 35 false positives. This engineering optimization is strictly equivalent to the original output, and all these facts remain unchanged—no changes to the implementation have been used to 'correct' the statistical results.

The complete 414-image AP of the standard scorer remains at 11.940007%, with the historical FP32 value at 12.440710%, resulting in a gap of approximately 0.501 percentage points. Quality scoring and localization quality estimation have been previously studied; this round does not demonstrate a new detection mechanism, independent generalization, or original paper contribution.

# # Testing and Independent Verification

Final 11 pre-checks passed, including 0/1/4/10/127/128/129/257/1000 candidate boundary, tie-handling, frozen score reproducibility, and input validation. During the initial pre-check, a test incorrectly treated the first tied box containing an extra random box as having no higher-scoring neighbor; this test has since been modified to isolate 4 cases of tied boxes, with the original failing JUnit retained, and the bit-wise equality test for features already passed prior to the fix.

The main process observed an exit code of 0 and received a completion acknowledgment. A saved-artifact audit independently replayed all 592 new and old feature, score, and ranking outputs, re-verifying each input identity, 858 original nanosecond timestamps, alternating measurement order, coverage, and millisecond summaries; no duplicated baseline timing, detector inference, or model training was detected.

This is an engineering improvement that is reusable and cannot be considered a complete research success. All previous FAILs and 47 false positive detections are preserved. The Selection load is not read, the Holdout is sealed, and no cloud resources or new scheduling tasks are involved.

New implementation: [gate1_scorer_exact_features.py](../scripts/gate1_scorer_exact_features.py). Evidence directory: `../artifacts/gate1/20261026_scorer_exact_CPU_optimization_v1`. Robustness diagnosis is referenced in Report 77.
