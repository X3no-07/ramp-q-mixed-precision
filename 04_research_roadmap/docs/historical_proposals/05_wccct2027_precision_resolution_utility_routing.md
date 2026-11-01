# WCCCT 2027 New Proposal: Routing Benefits of Precision and Resolution Compensation



Date: 2026-10-26. Nature: research_design and topic suggestion; this is a proposed method that has not yet been executed, and is not an experimental result, a frozen execution contract, or a submitted paper. The previous engineering status remains at v27; this document does not alter the conclusions of prior experiments or the scope of archived access.

> Correction on 2026-10-26: After reviewing tiling literature, this document withdraws its previous judgment of this method as the current primary recommendation. The original candidate design and associated reasoning are retained for reference; however, the validity, necessity, and unique contribution of this approach have not yet been established, and no experiments should be initiated based on this. The latest justification is found in the document \"06_wccct2027_tiling_literature_review_and_scope_revision.md\" in the same directory.

**We recommend submitting to Track 2, framing the research question as: under limited edge computing budgets, how can we predict which recovery strategy is appropriate for a given region—increasing numerical precision, preserving more original image details, combining both, or avoiding additional computation?**

Project Title: **A Benefit-Driven Precision–Resolution Routing for Edge Small-Object Detection**.

English Project Title: **Benefit-Aware Precision–Resolution Routing for Edge Small-Object Detection**.

This proposal offers significantly greater methodological innovation compared to simply continuing the fixed recovery of G1, but currently only has design rationale and no performance evidence. It is also incorrect to claim novelty based solely on the absence of existing papers with identical titles. The most critical pending validation question is whether the two recovery actions truly exhibit predictable complementarity.

# # 1. Which Track to Choose and Why

| Option | CFP Scope and Suitable Paper Focus | Current Judgment |
|---|---|---|
| Track 1 | Computer Vision, TinyML, deep learning on resource-constrained devices, edge-time real detection; suitable for detection loss, quantization error mechanisms, or target reliability algorithms | Can accommodate the current direction; if only perception algorithm evidence is achieved, the writing focus should be appropriately narrowed |
| Track 2 | Edge computing, lightweight models and inference, resource management, energy optimization; suitable for selecting recovery actions and controlling actual computational overhead | **Primary recommendation. The core of this new proposal is the allocation of budgeted regions and computational configurations** |

Both allow monocular vision work; the name of Track 1 does not imply a requirement for multimodal integration, and Track 2 does not require cloud-based human involvement. This matching judgment comes from the official CFP of the day (https://wccct.org/cfp.html). Choosing Track 2 is based on contribution focus, not because it is perceived as easier to get accepted.

The [official homepage](https://www.wccct.org/index.html) and [submission guidelines](https://www.wccct.org/sub.html) currently list: full submission deadline on 2026-11-15, conference dates from 2027-04-16 to 18; English papers of 4 to 10 pages, with a standard registration covering 5 pages. Approximately 20 days from the current suggested date, a three-day decision should be made early to determine whether the direction is worth continuing.

Regarding IEEE objectives: the homepage currently lists IEEE Xplore links for past conferences from 2020 to 2026, and the 2027 publication section states 'conference proceedings'. However, no official IEEE conference entry for 2027 has been verified, and the IEEE domain search portion is restricted in access. Therefore, past inclusion cannot be taken as confirmation of publication for the current conference; authors must verify the current conference arrangements before submission or payment. No contact with the conference or submission materials has been made.

# # 2. How Existing Work Supports the New Proposal

The latest pointers, v27, report 62 and 63 were read and verified on the same day:

- 30 epochs of training and the final FP32 state are already recorded, enabling reuse of existing models, data splits, and evaluation interfaces.
- Among the 546 development images, restoring relative AP50–95 gains with fixed G1 under full QDQ are only about **0.0044, 0.1140, 0.1515 percentage points**; the number of false positives simultaneously increases by **206, 317, 460**.
- Under the main threshold, the net recovery for small objects is limited to **136, 118, 153**, but this is not an independent acceptance criterion, nor does it represent overall AP improvement across all targets.
- All nine Q/D/R branches still have **0 out of 546 images passing the original full candidate box numerical threshold**. Floating-point convolution operations under QDQ do not equate to actual INT8 engine execution.
- The holdout set has only completed container and directory inventory; image and annotation content remains sealed.

These phenomena support the proposition that 'precision recovery brings both gains and costs'—yet they do not prove which regions require what corrective measures, nor do they prove that these gains can be predicted at runtime.

Existing research can be transformed into three components: target-level identity tracking for supervising recovery gains; side losses and false positive statistics for constraining performance; and existing models and scorers for controlled comparisons. The original numerical failures are retained, and the new method cannot rely on threshold adjustments or switching to a subset of NMS to fill in previous PASS results.

# # 3. How Existing Papers Have Progressed

The following are based solely on the original paper texts, author pages, and official websites; retrieval-based abstracts are not considered as evidence of methods, and no local reproductions of these papers have been performed. Preprints and formal publications are separately annotated.

| Nearby work | Overlapping aspects with the proposed scheme | Clear distinctions or comparisons needed |
|---|---|---|
| [Uzkent et al., WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/papers/Uzkent_Efficient_Object_Detection_in_Large_Images_Using_Deep_Reinforcement_Learning_WACV_2020_paper.pdf) | The work already learns selection regions based on recall gains and costs at high and low resolutions; the original paper does not include precision in its reward function | The concept of 'gains from additional computation' is not novel; this proposal must demonstrate the added value of the interaction between numerical precision and spatial sampling, as well as the constraints provided by false positives and side losses |
| [SAHI,ICIP 2022](https://arxiv.org/abs/2202.06934) | General tiling inference and tiling fine-tuning | An increase in AP after simple tiling alone does not constitute a new method; a comparison against tiling under the same budget is required |
| [ESOD, 2024 paper/author version](https://arxiv.org/abs/2407.16424) | Reusing the backbone for feature-level object search and tiling reduces the overhead of high-resolution detection | It is incorrect to claim first-in-class local computation; it is a key neighbor of sparse high-resolution approaches |
| [DenseScout, 2026 preprint](https://arxiv.org/html/2604.25300v1) | Learn low-budget region selection with end-to-end latency consideration; report both coverage and detection closure | This approach must learn the actual net gain of different recovery actions relative to baseline detection; it cannot be misrepresented as only reporting coverage without final detection experiments |
| [MCAQ-YOLO, 2025 preprint v2](https://arxiv.org/html/2511.12976v2) | Morphological complexity drives spatial width allocation and curriculum QAT | Spatial mixed precision has been studied; the distinguishing aspect of this work is precision-sampling dual intervention supervision and action selection, not simply 'different regions with different precision' |
| [CQTR, 2026-09 preprint](https://arxiv.org/html/2609.06581v1) | Freeze scale interventions on the detector, candidate reliability correction, and flow-level routing | Counterfactual scale recovery has been studied; this work proposes per-image, per-region precision-scale action gain selection, and requires annotated development data—cannot claim label-free operation |
| [HQOD, 2024 author paper](https://arxiv.org/abs/2408.02561); [Reg-PTQ, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html) | Classification-localization misalignment, NMS degradation, and dedicated quantization and filtering losses have all been studied | Separate packaging of 'protecting localization', 'suppressing false positives', and 'designing quantization loss' does not constitute novelty |
| [InlierQ,ICLR 2026](https://openreview.net/pdf?id=GN9otzf5o6) | Reduce task-irrelevant activations affecting PTQ by leveraging saliency and inlier probability | A reasonable PTQ baseline must be established first; it is invalid to claim universal necessity based solely on a fragile max-calibration scheme |
| [TOD, 2021 author paper](https://arxiv.org/abs/2105.08668) | The detection network has been selected based on video object size, motion, and computational capability. | Dynamic edge scheduling and model selection based on cost are not novel concepts. |

**The retrieval conclusion identifies a specific combination gap worthy of validation, rather than a proven novelty.** Among the nearby previously read content, there is no instance of the exact setting described below—"same cropping precision × resolution intervention, net gain supervision, and budget-constrained recovery action selection"—being fully present. Further investigation during implementation is required; it is therefore inappropriate to claim 'world-first'.

# # 4. Method Design: Clear recovery actions within the same region

## # 4.1 Offline establishment of four controlled outcomes

Fixed detector weights, identical cropped region, contextual boundaries, coordinate restoration, and final fusion method are maintained. On the development dataset, four forward passes are performed for each region:

|  | Base input resolution r0 | Higher input resolution r1 |
|---|---|---|
| Lower numerical precision Q | Q,r0 | Q,r1 |
| Higher numerical precision H | H,r0 | H,r1 |

For example, initial trials can be conducted with cropping inputs of 320 and 640. Here, 'higher resolution' refers to preserving more sampling details from the original image, without using generative super-resolution or relying on interpolation to restore information that did not originally exist.

Each region uses the same core area and fixed contextual boundaries. All four branches employ identical cropping coordinates. Sampling is unchanged when precision is altered; weights are unchanged when sampling is altered. All calibration seeds are generated using the same set of development calibration inputs, and these seeds are shared across branches to avoid introducing the effects of reselected calibration images into the precision evaluation.

H uses an existing FP32 reference in the pilot. **It is not valid to directly treat full FP16 as a reliable reference: previous full FP16 attempts have failed.** If FP16 acceleration of H is considered later, it must be validated as a new candidate separately; the success of body32/head16/decode32 cannot be extrapolated to full FP16 success.

To evaluate actual detection performance, the results from each branch are restored to original image coordinates and fused with the corresponding full-image baseline output according to a fixed rule, followed by a complete match against the original image's ground truth (GT). The practice of first cropping competing targets before computing 'recovery' is strictly prohibited.

Let the final detection outcomes for the same GT across the four branches be denoted as z00, z10, z01, and z11, where the first digit represents precision and the second digit represents resolution. Record:

- Precision intervention effect: dP = z10 − z00.
- Resolution intervention effect: dR = z01 − z00.
- Interaction effect between precision and resolution: dPR = z11 − z10 − z01 + z00.

All three effects may be negative. All state combinations must be preserved, not just the positive cases of recovery. These represent differences in intervention effects along a specific processing chain and cannot be used to assert the unique physical failure cause of a target.

Particularly note: The Q, r0 cropping branch may differ from the original full-image Q due to contextual changes. A separate control case—'full-image Q without recovery'—and another—'Q with r0 cropping'—must be retained. The effect of cropping cannot be attributed to precision gains.

## # 4.2 Learning the Benefits and Drawbacks of Each Action

At runtime, perform a low-precision full-image detection first, then compute minimal available features for fixed regional grids: low-precision raw candidate scores, class intervals, candidate density, estimated bounding box sizes, and inexpensive original image texture/contrast statistics.

All grid regions are considered, including those without final detection boxes; otherwise, the most critical completely missed regions would be excluded. GT, FP32 baseline outputs, and four-branch differences are used solely for offline supervision generation and are never used as online decision inputs.

First, use a small regressor or a two-layer MLP to output three quantities for each action:

1. The number of small objects that were initially missed but correctly detected after recovery (Gs).
2. The number of objects correctly detected initially but lost after recovery (L) across different sizes.
3. The change in false positives relative to the baseline output after recovery (dFP), which can be negative.

Development annotations are used, so the method is not referred to as unsupervised or label-free. There are already 546 images with observed development results; new training and cross-validation remain solely for development purposes.

A transparent prototype utility can be adopted:

**U = Gs − αL − βdFP**.

Gs represents the additional small-object detections relative to the baseline Q, not restricted to the range of FP32 original detections; it is a distinct estimate from the previously reported net recovery in old reports. The prototype can start with α=β=1 and simultaneously report the original counts for each item. Weights are chosen by research design and do not represent safety costs; weighted U cannot replace AP, fixed false positive rate recall, or a complete side-effect report.

It is necessary to treat the following two routing strategies as equivalent capacity baselines: one using net action benefits for supervision; the other using precision/sampling intervention information for multi-task supervision. **If the latter does not provide additional gains, the 2×2 diagnostic can only serve as an interpretive contribution and cannot be misrepresented as a novel routing algorithm.** Linear rewriting of the four outputs is also not an innovation.

## # 4.3 Selecting Actual Actions Within Budget

The action pool for each region includes: no recovery, Q,r0 low-cost recheck, H,r0 precision recovery, Q,r1 detail recovery, and H,r1 joint recovery. The pilot phase selects at most two regions, without introducing large-scale reinforcement learning search.

The scheduler selects actions with positive predicted benefits, costs within budget, and side-effect predictions within limits. Formally, this can be written as:

max Σ Uhat(j,a), subject to at most one action per region and the sum of costs (j,a) plus baseline and scheduling overhead not exceeding B.

Cost must include cropping, preprocessing, data transfer, additional forward passes, fusion, and synchronization. Simple greedy or small-scale enumeration can serve as solvers; backpack optimization itself is not an innovation in this paper.

The above summation is only an approximate scheduling. After merging multiple regions, NMS and overlapping bounding box interactions occur, **the sum of per-region benefits cannot be directly added as the final result**. The pilot phase limits to at most two regions, and the true joint utility of all allowable combinations can be computed using saved candidate outputs. The gap between the approximate scheduling and the GT-optimal choice that is not deployable can be evaluated.

Edge implementations prioritize using standard fixed-configuration engine switching, such as a set of Q/H × two input size engines, without implementing arbitrary pixel bitwidth within a single convolution. The cost includes redundant forward passes, engine residency in GPU memory, and switching overhead, which must be measured empirically; it is not assumed that this approach is faster than full-image high-precision detection.

# # 5. Which Components Have Potential as Paper Contributions

This forms a coherent research thread, and it is recommended to propose at most three contribution candidates:

| Candidate contributions | Required evidence | How to handle when not established |
|---|---|---|
| Intervention analysis of numerical precision and spatial sampling within the same region, identifying complementary recovery and interactions | Four-branch results at the same object level; context control via cropping; different development components and three sub-arms; proof that no single action dominates almost everywhere | If uniform scaling can resolve the issue, abandon the claim of joint precision routing |
| Action gain routing supervised jointly by newly achieved detections, side damage, and false positives | Same-budget performance outperforms random, low-confidence, density-based selection, and resolution-only routing; independent ablation under intervention supervision | If only the oracle is valid and observable feature prediction is ineffective, halt learning of routing direction |
| Edge recovery strategy based on actual cost | Real INT8 and high-precision engines; end-to-end AP/recall—latency curves, P95, memory, and power consumption; significantly outperforms simple static configurations in trade-offs | If additional forward pass gains are incurred, it does not constitute effective edge acceleration; absence of device-level evidence means no claim of real-time system implementation |

Separate elements such as 'tiling', 'mixed precision', 'net gain formula', 'three sub-arms evaluation', and 'upgrading to Orin' do not count as novel method contributions. The paper's validity hinges on whether the combination produces results that cannot be explained by neighboring methods or are beyond the reach of simple baselines.

# # 6. Necessary Controls and Limited Pilot

Formal comparisons must include: original full-image FP32 / qualified high-precision reference, reasonable PTQ full Q, original fixed G1, same-budget uniform or random tiling, confidence/density selection, resolution-only selection, precision-only selection, and the new joint routing. Fixed action pool, candidate regions, detection weights, and post-processing must be kept constant before comparing routing strategies to prevent multiple conditions from being altered simultaneously.

SAHI-style same-budget tiling and a related learning selection baseline are important external controls. If only the scoring idea is transplanted, it should be labeled as a local adaptation baseline and not claimed as a full reproduction of DenseScout/ESOD. If runnable code is unavailable, it must be disclosed and cannot be skipped to directly claim SOTA after a simple baseline.

To avoid effectiveness only on suboptimal quantization recipes, initial comparisons must include standard calibration methods such as max, MSE, or percentile. Any calibration method that can reliably implement in the target backend must become a strong baseline. If reasonable calibration eliminates the value of precision recovery, the precision branch should not be retained merely for innovation.

**The first recommendation is to conduct only a three-day stop-loss development pilot, without immediately scaling to full training. The following is a suggested budget, which is not yet authorized for execution or frozen under contract.**

- From the 546 open development members, pre-extract 96 images based on existing component metadata; do not select images based on recovery performance. Do not read the holdout set.
- For each image, fix a 2×2 core region, resulting in four patches, each with identical surrounding context; crop to 320/640 as the initial candidate input. Record context and boundary truncation; do not repeatedly adjust cropping based on boundary test results.
- Retain the existing three calibration seeds, each with 64 calibration images, using only the training-side calibration source. For each seed, perform a unified range estimation using full-image inputs with four blocks of two resolutions: 64 × (1+4 × 2) = 576 calibration forward passes per seed; across three seeds, this totals 1728 calibration forward passes.
- The upper bound for candidate forward passes is 96 × 3 × (1+4 × 4) = 4896, plus 96 full-image FP32 reference passes; in total, this results in at most **6720 real-image forward passes**. High-precision outputs that are strictly identical across seeds can be reused to reduce computation. Synthetic construction forward passes are separately recorded.
- Do not retrain the detection backbone. Only fit a simple gain predictor with fixed group cross-validation; all records from the same image, component, and across the three seeds must be within the same fold to avoid target-level leakage. Existing components serve only as risk proxies and cannot be claimed as representative real-world scenarios.
- Multi-action fusion uses saved outputs, with a maximum of two regions selected per image; each image per seed has 1+4×4+6×16 = 113 allowed combinations, enabling the computation of a GT oracle as an upper bound for diagnosis. The oracle reads ground truth, not a deployable algorithm.
- The technical entry point must be bound in a new directory with its source, inputs, and budget. Existing archived scripts, weights, contracts, and original numerical thresholds remain unchanged.

The pilot addresses only three questions: Whether there is complementary space for precision or detail recovery; whether cheap features can predict that space; and whether the net recovery—after deducting false positives, collateral misses, and actual costs—still justifies further expansion. Without these three, no additional modules should be extended beyond.

The proposed development retention criteria should be established prior to actual runtime, not determined post hoc based on results.

1. Use the best unified action and same-budget single-resolution strategy as a reference; if joint routing does not yield meaningful additional gains according to the joint oracle, terminate joint routing.
2. If a deployable predictor fails to achieve consistent performance across folds with respect to confidence, density, or randomization, terminate the main method—performance on training folds cannot be used as a substitute.
3. Under pre-defined false positive and latency budgets, there must be visible practical gains. For example, a minimum practical improvement of at least 1 percentage point in small-object recall can be proposed as a discussion benchmark; this is only a proposed business standard, not a statistical conclusion or a continuation of the old gate criterion. Official numerical values must remain frozen before runtime.
4. Success must not be judged by a single seed, a single threshold, a single screenshot, or a more lenient numerical threshold. All seeds and pre-defined budget points must be retained.
5. A clear rationale for continuation, simplification, or termination must be provided by the end of the first round. When failing, negative results must be preserved and no further scheme modifications should be attempted on the same test data to claim independent validation.

# # 7. Evidence Required to Progress from Pilot to Paper

If the pilot is successful, the learning router operates in training and development roles; all features, weights, action pools, and budget points are frozen before being applied to unseen test data. The existing 1610 holdout file name pairs do not equate to verified content quality or scenario independence; new content pre-checks and subsequent model evaluations still require explicit authorization under the original boundary requirements.

The final table must report official AP, small-object recall under the original main threshold, recall at fixed false positive levels, per-object additional misses and losses, false positives, and the usage ratio of each action. The old small-object area layer is not official AP_S and cannot be renamed directly; if a new AP_S is implemented, its scoring definition must be clearly stated.

Real-device-side recordings must include mean and P95 latency, throughput, peak memory, appropriate per-image energy consumption, and action-switching costs. Power modes, inputs, and timing boundaries are frozen, and the behavior of the worst-case complex image is included. Latency budget statistics are only empirical constraints and cannot be claimed as hard real-time guarantees.

The old QDQ has only simulation evidence and failed under full FP16 thresholds—both are technical risks that must be addressed on the new path. Conceptual exploration in the pilot cannot be elevated to 'INT8 is accepted'. If the real backend or established numerical qualification fails, the claim of a validated edge system under this proposal cannot be made, and no rewriting of old contracts should be used to circumvent this requirement.

The paper may first present a clean comparison on a single dataset; a second detector or a second dataset is used to assess generalization. Old RUOD results cannot be directly used as cross-domain validation for the new routing scheme. If time is insufficient, generalization statements should be narrowed, and no synthetic data should be fabricated.

# # 8. Recommended Timeline and Writing Structure

The current target timeline under the official website's deadline is as follows. It depends on the pilot's validity, access authorization, and availability of the runtime environment—this is not a guaranteed completion date.

| Time | Reviewable outputs |
|---|---|
| 10-26 to 10-28 | Four-action complementarity, simple gain prediction, and minimal backend feasibility; making continue/stop decisions |
| 10-29 to 11-03 | Budget-constrained routing comparison, necessary ablation studies, final model freezing, and evaluation design |
| 11-04 to 11-07 | Independent data validation upon authorization, end-to-end measurements on real devices |
| 11-08 to 11-12 | English manuscript, figures, comparison with recent work, supervisor review |
| 11-13 to 11-15 | Revisions, formatting checks, and submission buffering; submission remains explicitly authorized by the user |

Suggest organizing the evidence density around five pages: problem and neighbors; intervention definition and routing method; experimental setup; main table/cost curves/ablation; limitations and conclusions. Whether exceeding five pages is determined by actual evidence needs, not by stacking modules to appear more innovative.

If the pilot shows only resolution is useful, then the precision dimension is not established; if recovery value exists but is not predictable, it may be retained as a mechanism study option, but cannot be guaranteed to be sufficient for submission. If device cost cannot converge, do not package the method as already running in real time. All three stopping criteria are designed to make early research judgments.

# # 9. Trade-offs with Other New Approaches

| Approach | Advantages | Reasons not adopted as primary approach |
|---|---|---|
| Precision-resolution recovery gain routing in this paper | Reuses existing object-level analysis to generate new region/action decisions, directly aligning with Track 2 | Need to validate action complementarity, prediction feasibility, and actual overhead; success remains uncertain |
| Track 1: QAT for protecting the boundaries of candidate ranking, localization, and NMS decision-making | More aligned with existing quantization models; can be attempted with short-term fine-tuning | HQOD and Reg-PTQ have already covered a large portion of relevant ideas; adding a few more loss terms is insufficient to establish novelty |
| Track 1/2: high-precision keyframe recovery guided by temporal consistency | Videos may provide additional information, enabling stronger systemic questions to be proposed | The current main experiment is static image-based; introducing video data, associations, and caching would significantly expand the scope, making it unsuitable to prioritize within 20 days |

It is recommended to first allocate computational resources to the minimal falsification experiment of the primary approach, retaining previous results as motivation and baseline. Do not launch all three research lines simultaneously.

# # 10. Evidence and Scope of Completion

This work completes the conference scope verification, related work review, clear direction recommendations, and a feasible research design. No new model forward pass, training, calibration, AP evaluation, holdout member access, device testing, or scheduled tasks are included.

Cited project documents:

- [Engineering Status v27](../../../docs/DATA_AND_ARTIFACTS.md), SHA256 `cec5681d778f113820f67ca8be624dfd907ef0c8a3a49c45c9ef70b40b604fde`
- [Report 62](../../../03_gate1_visdrone_validation/docs/62_gate1_selection546_fixed_g1_validation_and_ap.md), SHA256 `32785e6d62c4279eebd9e31cbafc175bad50804985f119ee8452dc32aa48bf7d`
- [Report 63](../../../03_gate1_visdrone_validation/docs/63_gate1_saved_output_size_threshold_diagnostics_and_acceptance_preparation.md), SHA256 `d1177d88dce277f9edf3b31fc6757602cd322a7a5fedebe1e850433fbdaae2f3`

This paper does not treat the 2026-09 snapshot from the direction directory as the current experimental state, nor does it cover those snapshots. The proposed methods remain to be implemented and validated; completion of the scheme does not equate to completion of the paper, nor does it imply acceptance.
