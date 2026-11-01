# Paper Reconstruction: Candidate Approaches, Research Value, and Elimination Criteria



Date: 2026-10-26. Nature: Re-identification of research topics and research design, not a proven new method or experimental result.

User has explicitly stated: stronger innovation is prioritized; the timeline can be extended, with the detection foundation reused but the core contribution changed, with appropriate discarding of previous results; Orin is permitted to be used. High quality is the research standard, without pre-committing to performance outcomes, acceptance, or evaluation by any conference.

# # 1. What is being changed here

The goal of 'preserving the fixed G1 recovery conclusion' is no longer pursued. Instead, we first identify a well-justified problem, then determine the simplest yet effective solution; all existing weight and conclusion results are now excluded from the new paper's main line.

Current priorities:

1. **Immediate priority validation: the interaction between tiling view changes and real INT8 quantization.** The candidate approach is to reconstruct the task across different tiling cuts for the same target, rather than adding a runtime router. Standard tiling calibration must first serve as a strong baseline.
2. **Permitted replacement of the core alternative: the first detection and persistent missed detections of small objects in moving video.** How to balance detection of already-identified targets with newly uncovered regions within the given computational budget. A new continuous video protocol is required, and static image experiments cannot be used as evidence for video scenarios.
3. **Explanation and tooling line: whether target failures across backends are predictable, and whether recovery prioritization can be transferred.** This will initially serve as a diagnostic for the first two items; only when reproducible and broadly meaningful patterns are discovered should an independent paper be considered.

This is an order of investment, not a ranking of effectiveness. If any of these lines is simply resolved by a strong baseline, that line is terminated—no extension through metric changes or selective presentation of favorable examples is allowed.

# # 2. Which existing results are retained and which are discarded

The latest pointer and v27 SHA were read and checked alongside reports 62 and 63. Formal training, saved-artifact audits, and development evaluation are complete. On 546 images, fixed G1 restoration increases AP50–95 over full QDQ by approximately 0.0044, 0.1140, and 0.1515 percentage points while increasing project-matcher false positives by 206, 317, and 460. The recovery configurations under all three seeds remain below FP32. These results do not establish module-selection superiority.

| Asset | Usage in the new research |
|---|---|
| Target identity tracking, original image coordinate reconstruction, ignore handling, scoring entry | Reusable, but applicability must be validated in the new task |
| Fixed G1 and previous QDQ results | Retained as preliminary observations or negative results; no requirement that the new method inherit these results |
| Final 30 epochs of weight quantization | Can serve as development probes; if there is overlap between training data and new video evaluation, it should be discarded and retrained using a fresh split |
| Report and hash list already written | Evidence preservation; quantity is not a measure of paper innovation |
| Original full-candidate numerical threshold | FAIL cases must be retained as-is; new task metrics must not be packaged as passing the old threshold |
| Content of holdout members that have not been opened | Continue to be sealed; not to be used for this round's topic selection or hyperparameter tuning |

New papers must clearly distinguish between numerical equivalence, task utility, and device efficiency. Weight quantization does not need to claim that all original outputs are approximately equivalent to FP32; however, any claim of transfer equivalence must be supported by corresponding evidence. Success cannot be retrofitted by relaxing previous contractual terms.

Current machine is an RTX 5060 Laptop with approximately 8 GiB of GPU memory. The user confirms Orin is available, but the board was not reconnected this time. Version, temperature, power mode, and backend runtime status must be verified before the experiment begins. Current C-drive has about 62.5 GiB free, and D-drive has about 281.2 GiB free; the old low-disk snapshot is not to be reused.

# # 3. Screening approaches from different perspectives

| Perspective | Specific research questions | Related prior work / known risks | Decision |
|---|---|---|---|
| Interaction between tiling and quantization | After proper calibration, does quantization-induced damage for the same target still significantly vary with the cropped context? Can offline tasks be used to reconstruct and reduce this additional instability? | Tiling calibration has existing engineering practices; Reg-PTQ and InlierQ have task-related calibration methods already | Prioritize small-scale refutation experiments |
| Video and computation scheduling | During computational savings, are small objects that remain undetected being ignored over the long term? How should follow-up inspections be scheduled under mobile camera scenarios? | AdaPyramid, MaskVD, and Minimum Delay have already addressed some of these issues | Main alternative; requires complete nearest-neighbor reproduction |
| Task loss and distillation | Can compressed models preserve correctly detected targets and reduce false positive competition? | Positive-Congruent Training, HQOD, and ScaleKD already incorporate relevant ideas | Serves as a strong control; simple weighted loss is not sufficient to establish a standalone claim |
| Probability and risk control | Can the proportion of missed targets be controlled when coverage is reduced? | Conformal OD, TriCCOT, and Conformal Adaptation already have direct nearest-neighbor counterparts | Postponed; the mere existence of a theorem does not justify a standalone study |
| Cross-backend mechanisms | Can the 'sensitive modules' in simulation predict task loss in TensorRT? | Quantization differential testing and selective quantization tools already exist; however, individual version failures lack sufficient generalization | Diagnostic lines; independent publication is only justified if generalization requirements are met |
| Underwater enhancement and quantization | Does enhancement alter the benefits and costs of low-precision detection? | Enhancement has been shown to potentially harm detection performance; improved visual quality does not necessarily imply improved detection | A reserve issue; the third line should not be activated simultaneously |
| Data and evaluation | Do small object annotations and leakage across adjacent video frames alter the ranking of methods? | A new benchmark requires systematic cleaning, independent annotation, and a publicly available protocol | Prioritize research quality over compiling a checklist into a new dataset |
| Network architecture | Are new attention modules, small object heads, upsampling, or super-resolution worth adding? | Large numbers of neighbors lead to high training and attribution costs, and combined modules do not naturally constitute meaningful contributions | Not currently prioritized |

# # 4. Priority Validation A: Tiling View Changes and INT8

## # Research Question

Tiling preserves more details from the original image. We need to verify: after completing reasonable tiling calibration, does quantization still introduce additional classification, localization, or candidate competition errors for the same object across different cropped positions, contexts, and scales? The sensitivity of FP32 itself to cropping must be isolated and accounted for separately, and cannot be entirely attributed to INT8.

Existing engineering documentation explicitly requires tiling models to be calibrated on runtime tile blocks and to include full-image samples during runtime full-image branches. Therefore, 'switching to tiling calibration' cannot be considered an innovation. [EdgeFirst calibration documentation, Tiled models](https://doc.edgefirst.ai/test/models/conversion/calibration/)

## # Minimal but Discriminative Experiment

1. Fix geometric tiling and final fusion, without using ground truth to determine where to crop during inference. Compare two input paths: full-image and fixed-overlap tiling. Perform accuracy comparisons within the same tile and preprocessing setup. Objects at cropping boundaries remain in full evaluation, with additional diagnostic stratification for fully visible objects.
2. Accuracy must include FP32, a floating-point baseline from the deployment backend, and actual INT8 performance. Use a fused, exportable quantized model; un-fused Conv layers with floating-point QDQ can only serve as historical controls.
3. Calibration must compare full-image calibration, standard calibration matching runtime views, and task-relevant calibration under the same view setup. Additionally, include methods with more calibration data to distinguish whether improvements are merely due to increased data coverage. Report budgets independently by original image count, input tensor count, and offline computation cost.
4. Initial probing is limited to the already open development scope. If a stable mechanism is established, expand to two detectors with different architectures and two data domains. Default detector candidates are YOLO11n and YOLOX-s; the latter has official ONNX/TensorRT deployment access, but native compatibility has not been verified. [YOLOX author repository](https://github.com/Megvii-BaseDetection/YOLOX)
5. Match the same original image ground truth identity, record newly introduced missed detections, stable object side effects, global false positives, and official AP; retain all objects, not just those easily detected by FP32, to assess the primary effect. Object-level conditional events serve as the mechanism explanation.

## # Where does the mechanism originate if valid?

The candidate mechanism is **offline cross-tiling task reconstruction**: during training and calibration, pair different views of the same object, leveraging the floating-point outputs from each view to optimize task error and cross-view instability in the quantized model. At deployment, retain a single executable INT8 engine, without relying on concurrent floating-point teacher models or additional decision networks.

This is merely a method proposal. It must confront three counterarguments: standard multi-view calibration, conventional reconstruction using the same data and training volume, and existing task-relevant quantization losses. If a pairing loss simply feeds more views into the model or derives its effect from increased training volume, it lacks independent contribution.

Nearest neighbors: [Reg-PTQ, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html) has studied regression-specific PTQ; [InlierQ, ICLR2026](https://openreview.net/pdf?id=GN9otzf5o6) focuses on task-irrelevant activations and calibration; [HQOD](https://arxiv.org/html/2408.02561v1) has explicitly discussed the misalignment between classification and localization under quantization and NMS degradation. These existing issues cannot be rebranded as new discoveries.

## # Elimination Criteria

- Standard calibration across matched views has already eliminated the majority of the gap; additional methods show no stable improvement.
- The phenomenon only appears in old simulators, weak calibration, or on individual weights, and cannot be reproduced in actual INT8.
- In complete cropping experiments, the gap is primarily explained by scale/context damage that also exists in FP32.
- The benefit is only observed with a single seed, a single threshold, or a small number of images, and does not support overall AP or false positive cost under full evaluation.
- Simulation-based methods are effective but cannot be derived under the same rules, or their actual device overhead offsets the gains.

# # 5. Primary Alternative B: Discovery of New Objects in Moving Video

## # Plain Language Description

To save computational resources, the system always focuses on objects already detected in the previous frame. Small objects that have never been detected may remain unobserved, causing their regions to never receive high-resolution inspection. This work investigates how to allocate the budget between 'continuing to monitor previously seen objects' and 'inspecting previously unseen regions', and dynamically adjusts this allocation when camera motion occurs, occlusions are resolved, or new areas enter the frame.

Its distinction from static image block selection lies in the observation history over continuous time and the cost of missing newly appearing objects. Its relationship with prior mixed-precision work is primarily based on reusing detection, object tracking evaluations, and Orin experience; quantization in this context can serve merely as a deployment tool.

## # Existing Work Must Be Directly Compared

- [Minimum Delay Object Detection, ICCV2019 author version](https://arxiv.org/abs/1908.11092) treats the time from object appearance to detection confirmation as the target and employs sequential detection theory; it cannot be claimed to have first introduced the concept of delay discovery.
- [AdaPyramid, TMC2024, Section IV-E](https://nju-cn.github.io/files/AdaPyramid.pdf) introduces a new mechanism for new object detection, which uses offline labeled regions from fixed surveillance cameras. This specific assumption can serve as a reference point for comparison in mobile scenarios, but cannot be generalized to claim that all prior methods only support fixed cameras.
- [MaskVD, WACV2025 author version](https://arxiv.org/html/2407.12067v1) supports both fixed and mobile cameras, combining static priors from the training set, detection regions from the previous frame, and periodic full-frame refreshes. A simple periodic scan cannot be considered a novel method.
- [DenseScout, 2026 preprint](https://arxiv.org/html/2604.25300v1) is a key comparison point for single-frame budget allocation and has existing detection and device evaluation results.

## # Candidate Mechanisms and Methods

Candidate methods maintain the last time each visible region was thoroughly observed, propagating or invalidating this history after camera motion. Newly entered regions, long-uninspected areas, and regions with unstable detections are included in the computation of budget allocation. The scheduler uses real device costs to avoid directing all budget toward already-detected objects. A first version should focus on interpretable rule-based or constraint-optimized solutions, without immediately resorting to reinforcement learning.

Potential new contributions should lie in: correct updating of observation history in moving scenarios, scheduling of revisits under limited budgets, and the resulting improvement in timely detection of new objects. A heuristic based solely on 'age score plus confidence' is insufficient to support current quality goals if it does not outperform simple cycling or periodic full-frame scanning.

Region revisit intervals should only be discussed under clearly defined conditions; obtaining a scan opportunity does not guarantee detection of an object. No performance guarantees are made for arbitrary objects, arbitrary motion, or arbitrary budgets.

## # Data and Evaluation

- Prioritize the official VisDrone video/MOT dataset (https://github.com/VisDrone/VisDrone-Dataset) and the XS-VID author's dataset and evaluation (https://gjhhust.github.io/XS-VID/). This review only verifies the release notes; no new data is downloaded or read. Continuous trajectories, frame rate/time stamps, and access conditions must be confirmed before execution.
- Split training, development, and testing by original video or scene. The static training images from old VisDrone may originate from related videos and cannot be directly used to test on independent video sequences; if necessary, retrain from a shared pretraining checkpoint.
- Compare full-frame/full-tiling scanning, fixed polling, frame-extended search, motion-triggered detection, single-frame heatmap-based region selection, as well as applicable MaskVD/AdaPyramid ideas or original implementations. Any modifications or porting must be disclosed; the ported version cannot be treated as a complete reimplementation of the original method.
- Primary metrics include the time distribution of first valid detection, detection rate within a fixed period, duration of sustained missed detections, and standard detection AP and false positives. Undetected trajectories must not be excluded from delay statistics; truncations due to leaving the field of view or video end must be reported separately.
- Statistics should be performed by video or real-world scene resampling; adjacent frames must not be treated as independent samples. The 75 risk agent components in static datasets cannot be directly equated to 75 independent videos.
- Real-time experiments proceed with input time progression, including preprocessing, motion estimation, selection, data transfer, detection, and fusion. Frame accumulation or loss must be counted in detection time; offline slow-motion playback cannot be used to simulate real-time operation.

## # Elimination Criteria

Standard full-image or polling-based approaches have already achieved comparable performance under the same hardware budget; the main gains come from reduced false positive requirements, manual entry annotations, or future-frame information; there is no reliable new target identity or time baseline; motion compensation overhead exceeds savings; new region failure handling lacks stability. In such cases, further stacking of complex modules is not justified.

# # 6. Other Directions Why Are They Not the Main Focus

**Protection of existing correct object-level targets/distillation:** [Positive-Congruent Training, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html) Existing ideas on reducing negative flips have been proposed; [ScaleKD, CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html) Small-object scale distillation has already been implemented. These should help construct strong baselines; giving greater weight to small objects and distilling from correctly identified targets does not automatically constitute novelty.

**Risk-Controlled Tiling:** [Conformal Object Detection, 2023](https://proceedings.mlr.press/v204/andeol23a.html), [TriCCOT, 2026 version](https://arxiv.org/abs/2609.08659), and [Adaptive Computation with Conformal Certification, preprint 2026-10](https://arxiv.org/html/2610.19189v1) have already explored coverage and risk control with sparse computation. Conditions such as object coverage, exchangeability, and monotonicity must be individually established; region coverage cannot directly replace the performance of NMS-based detection. This round does not aim to reframe the problem by simply adding a conformal module.

**Systematic Failure Analysis:** It is possible to study why quantization sensitivity conclusions fail across domains or backends, but this requires multiple models, data domains, reasonable quantization methods, and reproducible mechanisms. A single TensorRT build error, a single drop in AP, or a large volume of audit logs is insufficient to constitute a high-quality research contribution. [SeQTO author repository](https://github.com/luludak/SeQTO) also shows that selective quantization and differential testing already exist as tool-based approaches.

**Underwater Enhancement and INT8:** The original underwater background can be reused, but the phenomenon of 'detection performance degrades after enhancement' has been studied, e.g., [WACV2025 Workshop paper](https://openaccess.thecvf.com/content/WACV2025W/MaCVi/papers/Lucas_Underwater_Image_Enhancement_and_Object_Detection_Are_Poor_Object_Detection_WACVW_2025_paper.pdf). Only by demonstrating a specific interaction between enhancement and quantization—outperforming enhancement-re-calibration, original-image detection, and joint training—can a new contribution be established.

**Data and Benchmark:** Existing annotation conflicts should be disclosed, but new benchmarks require independent annotation quality, representative task design, and publicly reproducible experiments. A simple directory inventory cannot serve as a dataset paper.

# # 7. Next Steps to Avoid Further Cost Sink

## # First Research Cycle: The Goal Is to Make Trade-offs, Not to Produce Positive Results

Set a 10-day period as the initial review cycle; no commitment is made to completing a paper within this period.

1. Complete the nearest-neighbor matrices for A and B: changes in problems, supervision, inputs, models, deployment cost, metrics, data splits, source code availability, and our actual differences. Record gaps where key full texts or implementations are unavailable; do not treat abstracts as sufficient for complete reproduction.
2. A first establishes a reasonable tiling calibration and a real INT8 minimal probe; B first confirms the continuous video data protocol, existing new-object mechanisms, and a simple scanning baseline. At most, two limited probes are entered, without simultaneously developing all candidate approaches.
3. Run experiments with each party freezing members, parameters, models, calibration and training budgets, metrics, stopping rules, and output directories. Original data and contracts remain unchanged; new experiments do not invoke the sealed holdout set.
4. Review probes in the order: 'existence of an unresolved real-world problem → distinct from existing methods → simple baseline fails to resolve it → has practical device-level value'. Only one probe passes should be selected for focused investment; if none pass, output a refutation conclusion and re-choose the topic, without selecting a comparatively less effective one to falsely claim success.

## # Minimum Evidence Standards for the Formal Paper

- A clearly defined problem, a core method, and a mechanism explaining its effective conditions. Contribution cannot be substituted by module count.
- Comparison with the nearest neighbors and a strong baseline under the same dataset, training scale, and device budget; must present full accuracy, latency, and false positive cost metrics, not just favorable performance points.
- At least two detector architectures with structural differences, and two independent evaluation domains; if the video system path is ultimately selected, ensure multi-sequence, multi-motion/density conditions, and provide applicable scope. Quantity alone does not substitute for data independence.
- Report all pre-defined random seeds and uncertainty estimates based on real, independent units; the experimental design must be fixed prior to formal testing, and test results must not be retroactively tuned.
- Measure end-to-end latency distribution, memory usage, and power consumption under the same Orin platform, with a clearly specified software version and runtime mode. Model FLOPs, simulated QDQ, or a single peak FPS value cannot substitute real measurements.
- Conduct ablation studies on key components of the method; clearly document failure scenarios, invalid conditions, and cases where simpler methods perform better.
- Existing failure records and adverse results are preserved in their original form. New evaluation criteria are defined prior to execution and do not retroactively rejudge past Gate decisions.

# # 8. Current Deliverables and Pending Tasks

This round has completed re-validation of prior results, near-neighbor retrieval across multiple research angles, A/B method drafts, strong baselines, and elimination criteria. Citations prioritize full-authority papers, official conference pages, and engineering documentation; preprints and engineering practices are separately annotated. Lack of discovery of identical prior settings does not constitute proof of novelty.

This round did not involve training, calibration, new model inference, board-level configuration changes, downloading new datasets, opening a holdout set, or submitting a paper. No scheduled checks were initiated. The next specific task is to prepare fixed-budget research probes, rather than continuing to append validation reports for old G1 conclusions.
