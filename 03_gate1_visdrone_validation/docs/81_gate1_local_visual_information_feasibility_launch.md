# Gate1 Local Visual Information Feasibility Diagnosis Initiated



2026-10-26.

The previous three-group/feature ablation study did not demonstrate stable superiority over the original scorer. This round verifies a new information question: retaining the original 22-dimensional score and bounding box features, and further incorporating local visual evidence from P3 derived from a frozen forward pass of the R module, whether it can provide additional true/false bounding box ranking information beyond the prefix fitting of the scorer.

Only one candidate is registered: a 64-dimensional center feature and a 64-dimensional center-minus-neighborhood-mean feature, concatenated with the original 22-dimensional features, resulting in a total of 150 dimensions plus a bias. The 8 surrounding sampling points are fixed at 0.75 times the width and height of the bounding box, sampled via bilinear interpolation; no ground truth, prefix, or source identity is provided. This is not a new method proposal, nor a full reproduction of IoU-Net.

Reuse the original 512 scorer-fitting images and 213716 existing soft-quality labels, with image-equal loss, fitting-image-only normalization, zero-residual initialization,100 CPU float64 Adam updates, lr0.03, and L2 coefficient 0.0005. The detector remains frozen throughout. No candidates, feature layouts, hyperparameters, or seeds are added.

The diagnosis uses only 178 original scorer-fitted prefix images that do not overlap with the prefix calibration set. These images were previously used for calibration and belong to the original detector training data; this is an information feasibility check, not an independent test. The official IoU0.5/cap500 curve under the same R@0.25 false positive budget and the per-image equally weighted soft quality BCE are compared. A diagnostic signal with continued value is only recorded if both the original 22-dimensional scorer is strictly improved in TP retention and the BCE is strictly reduced; curve thresholds are only descriptive and not frozen for new deployment thresholds.

Historical intermediate features were not saved, requiring 690 new image forwards solely to extract new P3 information. For every image, all 8400 raw candidate outputs must match historical R bitwise. Intermediate feature maps and sampled arrays are saved for independent CPU recomputation. A deployment implementation could share the same backbone computation, but that possibility is not treated as a measured speed benefit here.

Images, labels, and results for the 414 development images are not read, no new complete AP is run, and neither Selection nor Holdout payloads are read. Execution begins after 6 preflight checks, with no automatic retry after failure. All negative results are retained. Positive results would still require mechanism controls, frozen development validation, and independent scientific evidence; they would not immediately establish a successful paper.

Localization quality and ranking have prior work: [IoU-Net](https://arxiv.org/abs/1807.11590), [GFLV2](https://arxiv.org/abs/2011.12885). This round only uses a local visual linear probe to assess feasibility and does not claim originality, real INT8 performance, or end-to-end deployment validity based on this alone.

Evidence Directory: `../artifacts/gate1/20261026_local_visual_information_probe_v1`.
