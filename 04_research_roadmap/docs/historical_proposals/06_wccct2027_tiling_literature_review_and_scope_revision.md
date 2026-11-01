# Tiling Literature Review and Topic Correction



Date: 2026-10-26. Nature: Literature judgment and topic correction, not new experimental results. No training, calibration, detection, or scoring was run in this review.

# # Current Conclusion

Common difficulties in tiling have matured solutions or well-defined research pathways; these should be adopted and compared first. The judgment that 'precision-resolution gain routing' should be the current primary recommendation is withdrawn; Document 05 is retained as a candidate design, but its method effectiveness, necessity, and unique contribution have not been established. It is not valid to conclude a research gap merely based on the combination of existing modules or the absence of a fully matching paper in the literature search.

This review checks the methods, experiments, and limitations of the paper; all the following results are reported by the authors and have not been reproduced in this project. The search scope is limited and does not represent a complete coverage of all literature.

# # What answers does the paper already provide?

## # 1. Fragmentation of object-level targets and loss of large object information: Overlapping tiling and whole-image branches are already established

[SAHI, ICIP 2022, Sections 3 and 4](https://arxiv.org/pdf/2202.06934) employs overlapping tiling, optional whole-image inference, and result fusion. Tile-based fine-tuning preserves the original image while maintaining scale consistency. The ablation study in the original paper shows that overlapping and whole-image branches have inconsistent effects on targets of different sizes and may even increase false positives. Therefore, 'adding overlapping' and 'including whole-image inference' should be considered existing solutions and cannot be presented as novel innovations.

Metric boundaries: The original paper uses AP50 and maxDets=500 in relevant tables; direct comparison with our official AP50–95 cannot be made.

## # 2. Too many background blocks and redundant feature extraction: Existing feature reuse and adaptive tiling are already available

[ESOD, Sections II-B, III-B, and III-C](https://arxiv.org/html/2407.16424v2) discusses redundant computation in additional region networks and points out that low-resolution pre-screening may filter out small objects. Its method reuses early-stage features from the detector to estimate object regions and performs adaptive tiling. The paper also compares greedy sequential tiling with parallelizable simplified tiling: the latter reduces scheduling overhead but still risks truncating large objects.

This indicates that both 'reducing background computation' and 'tiling based on object distribution' are already implemented with specific methods; the trade-offs discussed in the paper cannot be directly recharacterized as unsolved problems.

## # 3. Limited to only a few tile blocks: Existing near-neighbor methods already address selective block selection

[DenseScout, preprint 2026, Sections 3 and 5](https://arxiv.org/html/2604.25300v1) uses a lightweight, dense response-based selector to choose a limited number of tiles and incorporates data movement, inference latency, and effective coverage within time constraints into edge system evaluation. It also presents final detection experiments—this should not be misrepresented as merely covering regions. Its selector does not jointly optimize numerical precision, but this distinction alone is insufficient to prove the contribution of our joint approach.

The performance results in the preprint are reported by the authors and are not the Orin validation results of this project.

## # 4. Adding an extra filter does not always pay off: Clear counterexamples exist

[ROI-Gated SAHI, preprint 2026, Table I and Section IV-E](https://arxiv.org/html/2608.23923v1) reports on 128 COCO128 images: full SAHI achieves AP50 of 0.7569 and an average delay of 263.73 ms; static filtering yields 0.6602 and 298.24 ms; adaptive fallback results in 0.7305 and 258.46 ms. In this experiment, regions filtered out do not enter subsequent fine-grained detection.

This is evidence under specific models, datasets, and implementations and cannot be generalized to all filtering schemes. The three-case examples provided in the paper are insufficient to replace the overall results across 128 images.

# # Specific impact on the current research topic

The following statements are insufficient to support new contributions:

- 'Tiling is slow, so we first select regions.'
- 'Tiling has boundary issues, so we add overlapping and whole-image inference.'
- 'Limiting the number of tiles on edge devices.'
- 'Integrating tiling, quantization, and the selector together.'

All of these must be further grounded in a specific mechanism, condition, or constraint that has not been addressed by existing near-neighbor methods. Current literature evidence fails to establish such a difference, nor does it prove a predictable complementarity between our precision recovery and tiling.

The research sequence should be adjusted to: first organize the difficulties already addressed by existing methods and their remaining limitations, then clearly define a falsifiable difference hypothesis with a source, and finally conduct minimal validation. It is not acceptable to first commit to a claim such as 'increased detection and reduced false positives under the same computational cost' before seeking supporting evidence.

The work reviewed here has eliminated overbroad claims of novelty, provided ready-made solution pathways, and corrected the primary recommendation. It does not validate the effectiveness of alternative approaches, nor does it replace existing engineering implementations, independent scientific evaluations, or device acceptance procedures.
