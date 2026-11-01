# Gate1: Freezing the Scorer — Prefix Stability, False Positive Composition, and CPU Overhead Diagnostics



2026-10-26. Under the user's authorization to continue, this stage reads archived results to check the earlier false-positive increase on 47 images, the stability of the overall benefit, and the current scorer code's additional CPU cost.

# # Fixed Scope

The detector, scorer's 23 parameters and 22 input features, threshold `0.17473852372359241`, and all data members remain frozen. New detector image forwards total 0, new model fits 0, new threshold candidates 0, and complete AP rescoring 0. Diagnostics are not used to discard a prefix, change acceptance criteria, or select another threshold.

From the saved 414 training_val project match results, we compute the net increase/decrease in false positives and net increase/loss of correctly detected targets, grouped by original anchor identity. We then perform four geometric diagnostics using the original annotations: same-class IoU ≥ 0.5, cross-class IoU ≥ 0.5, maximum IoU in [0.1, 0.5), and low overlap residuals. Grouping is post-hoc description, not official error types or causal explanations; low overlap does not imply all cases are pure background.

For the 286 new images, their 47-subset of new prefixes, and the merged 414-image set, we group by existing filename prefix proxies. We exhaustively remove each prefix group and report the net change in detections and false positives across the remaining data. This method only checks whether the overall performance gain is concentrated — **it is not a new independent experiment, nor does it establish statistical confidence intervals or new pass thresholds**. Prefixes are not validated independent scenario identities.

# # Overhead Measurement

For the new 286 image with saved R original outputs, perform once per image without timing a preheating phase, and conduct fixed 3 measurements:

- Only the CPU time for the frozen linear scorer is measured.
- The total CPU time for extracting 22 model-visible features, scores, and stable ranking is measured.

Timing excludes file reading, annotation reading, image decoding, detector execution, original NMS, CUDA output transfer, and final result serialization. After timing, we verify strict consistency between extracted features and scores to ensure the measured implementation is the validated one. We report the median, P95, maximum, minimum, and average milliseconds across all 858 samples, along with the number of candidates.

This measurement does not constitute evidence for Jetson, INT8, power consumption, or end-to-end speed. Neighbor features include pairwise overlap computations among bounding boxes; the 23 scorer parameters do not imply that the entire feature computation involves only 23 operations. This round does not optimize the implementation or select a faster code version.

# # Paper Boundary Check

[IoU-Net](https://arxiv.org/abs/1807.11590) has studied using localization confidence to improve candidate selection; [GFLV2](https://arxiv.org/abs/2011.12885) has studied using box distribution statistics for lightweight localization quality scoring; [DetPTQ](https://arxiv.org/abs/2304.09785) has studied parameter selection for detection-specific quantization. Therefore, 're-scoring boxes' and 'using detection-specific settings to select quantization' are not novel contributions of this project and cannot be rebranded as precision intervention contributions that have not been proven.

The full 414-developed AP of the new scorer will be listed alongside the existing FP32_reference. The old reference scorer is not re-run. The saved results and this false positive geometric analysis do not replace the original full candidate-level qualification; the previous FAIL status remains valid.

# # Budget and Evidence

9 items passed pre-check, covering equivalence across prefix group overall removal, aggregation conservation, revenue concentration counterexamples, geometric grouping boundaries, millisecond units, and native implementation outputs.

Directory: `../artifacts/gate1/20261026_scorer_robustness_cost_v1`.

Contract SHA256: `cf337daf79d4f5ac8cbc4f386fc657cf0d4e51a83d4d75bbeb63922421bbae03`.

Original data, code, contract, and all prior reports remain unaltered. Independent test sets, selection, and holdout loads are not released; no cloud resources or scheduled tasks are created. The launcher observes actual exit codes upon single execution and does not retry. Result reports and saved artifact audits are documented in newly generated files.
