# Candidate Main Line: Leveraging Module Precision Intervention to Train Targeted Detection Correction



Date: 2026-10-26. Status: Research hypothesis and minimal validation design; no new inference, calibration, or training has been initiated. No evidence has been provided that the method outperforms existing work. This paper offers a concrete suggestion to the user regarding 'finding better ideas on existing foundations', without modifying old reports, contracts, or the scope of sealed data.

# # Core Recommendation

Retain the existing G1/G5 module recovery idea and further leverage module precision switching as a tool to generate training supervision: for the same training image and same weights, compare results between full quantization and localized precision recovery, using real annotations to distinguish between 'recovered targets' and 'introduced errors', and train separately for each scenario. Deployment remains limited to a single fixed-precision detector configuration.

The candidate paper question is: **Does the information introduced by module precision intervention, compared to conventional difficult-sample training, better preserve true detections and reduce additional false positives under the same inference configuration?**

This is not 'adding a loss term will automatically work'. Only when the intervention provides independent gains over a competitive baseline is it worthy of being developed into a paper contribution.

# # Existing Facts and Motivation

The current latest execution pointer remains at the engineering directory's `artifacts/gate1/current_execution_state_v27.json`; Report 62 documents 546 development images and 3 calibration seeds from a fixed G1 experiment.

| Change in G1 recovery relative to full QDQ | Seed 01 | Seed 02 | Seed 03 |
|---|---:|---:|---:|
| Net change in project match hits | +186 | +220 | +226 |
| Net change in project match false positives | +206 | +317 | +460 |
| Official Python AP50–95 change / percentage points | +0.0043713076 | +0.1140277514 | +0.1515142961 |

Hit and false positive counts are based on conf≥0.25 and same-category IoU≥0.5 project matches; these are not the counts from the official AP curve. The net change in false positives does not equal the total number of newly identified false positives. The three calibration seeds share the same targets and are not three independent test sets.

This indicates that local recovery alters both real targets and false predictions, but it cannot yet determine whether false positives arise from background, localization, classification, or duplicate boxes, nor can it determine whether training can separate these two aspects. The VisDrone dataset has not yet completed G5 or G1+G5 experiments, and the previous RUOD-R results cannot be substituted.

G1 refers to an early backbone; G5 refers to top-down feature fusion. It is not valid to pre-interpret them separately as 'specifically recovering targets' and 'specifically removing false positives'. The current W8A8 is a floating-point QDQ simulation of un-fused Conv layers, not the actual INT8 inference results on the target device.

Source: [Engineering Report 62](../../../03_gate1_visdrone_validation/docs/62_gate1_selection546_fixed_g1_validation_and_ap.md)

# # How the Method is Conducted

## # 1. Establishing Change Records via Controlled Precision Switching

On training members, fix weights, inputs, and post-processing. Compare Q (full quantization), R1 (G1 recovery), and subsequently approved R5 and R15. Maintain all conditions identical except for the specified precision interventions. Use annotations and target identity tracking to establish four categories of records:

- Real targets missed by Q but detected by R: the benefit from recovery.
- Newly generated false predictions by R: the side effect of recovery; further distinguish between background, misclassification, localization errors, and duplicate boxes.
- Targets detected by Q but missed by R: damage caused by recovery.
- Targets correctly detected by both Q and R: the capability maintained during training.

Missed detections are tracked by GT identity and cannot be sampled solely from output boxes. Cross-configuration boxes are not considered the same target based on array indices. Changes in NMS candidate competition and assignment must be recorded separately to avoid treating duplicate boxes as independent background counterexamples. Ignore regions, un-scored annotations, and ambiguous areas are not used as background supervision.

These are records of intervention effects under fixed experimental conditions and do not constitute claims about causal mechanisms in the real world. FP32 predictions can assist in diagnosis, but truthfulness and category assignments remain determined by valid annotations.

## # 2. Directed correction fine-tuning on a fixed recovery configuration

The new student model starts from the same weights as the fixed recovery configuration, maintaining normal detection training, but adds targeted training only on the above-mentioned change records:

- Recovered and compromised true targets: strengthen correct classification and localization supervision to avoid losing them during filtering of false positives.
- Confirmed newly emerged background false positives: suppress the corresponding class response.
- Misclassification or localization errors: correct the category or bounding box separately; do not treat all such cases as 'no target' for penalty.
- Previously stable and correctly performing targets: retain normal supervision and verify any new damage introduced by fine-tuning.

First, use clear and simple supervision to validate whether the intervention information is useful. If attempting teacher soft supervision, only pass on annotation-verified content, without requiring the student to mimic the teacher's full predictions. YOLO-style confidence scores use the same protocol as the original detection head, and do not arbitrarily replace multi-class Sigmoid with Softmax.

This is a labeling- and extra-training-dependent research path and should no longer be referred to as pure PTQ. Historical inference QDQ code has not automatically evolved into a trainable QAT implementation; a new trainer is required with correct pseudo-quantized gradients and a deployment-consistent graph, and must be saved as a new version. Original weights are not overwritten.

## # 3. Deploy only one model

Q/R comparisons and change records are used solely for offline training. Deployment uses a trained fixed mixed-precision model, without requiring concurrent execution of Q, R, or a full FP32 teacher, nor adding a runtime true/false discrimination network.

Under the same candidate precision configuration and operator graph, the additional overhead is primarily incurred during training; this does not guarantee identical actual latency, nor does it eliminate the inherent cost of G1/G5 recovery. Real INT8/FP16 splits, fusion, and Orin latency must be independently validated.

# # Cross-comparison with existing methods

As of this retrieval, it is not acceptable to present 'separating foreground and background', 'weighting by prediction differences', or 'reducing new errors' as standalone novel contributions. The following are all required to be rigorously compared with nearby works:

| Work | Already covered content | Requirements for this candidate |
|---|---|---|
| [DeFeat,CVPR2021](https://arxiv.org/abs/2103.14475) | Distillation that separates foreground and background; background knowledge can help reduce background false positives | Adding only a background loss does not constitute innovation |
| [PDF-Distil,BMVC2021](https://www.bmva-archive.org.uk/bmvc/2021/conference/papers/paper_0259.html) | Constructing distillation weights using the prediction differences between teacher and student | Must validate that the beneficial or harmful changes from controlled module precision switching are more useful than general prediction differences |
| [Positive-Congruent Training,CVPR2021](https://arxiv.org/abs/2011.09161) | Reduce negative flipping of previously correct samples that become incorrect after model improvement | "Protecting originally correct samples" is an existing idea; it requires specific incremental evidence within detection tasks |
| [HQOD,ICME2024](https://arxiv.org/abs/2408.02561) | Coordinate classification quality and localization quality during quantization training | If the phenomenon primarily stems from misalignment between the two, existing mitigation strategies should first be examined |
| [CrossKD](https://arxiv.org/abs/2306.11369) | Mitigate label-teacher supervision conflicts by cross-head distillation | When a teacher branch is present, existing conflict-resolution mechanisms cannot be ignored |
| [DetPTQ](https://arxiv.org/abs/2304.09785) | Select local reconstruction metrics tailored to detection tasks | Stronger quantization baselines are required to exclude cases where improvements merely compensate for simple calibration defects |

The potential incremental contribution of this candidate lies solely in: **whether the direction and source of local precision intervention within the same model are more effective than general difficulty, confidence, or prediction differences in correcting errors.** This has not been proven uniquely or effectively in literature reviews or existing results. Avoid using terms such as 'first', 'solved', or claims guaranteeing high acceptance.

# # Minimal Experiment and Explicit Elimination Criteria

First validate the training signal's value using the existing G1 configuration; G5 and G1+G5 can be tested with limited combinations as planned, but success in combination should not serve as the sole entry point for this hypothesis. Strengthen the foundational calibration first to avoid comparisons with simple absmax baselines only.

Under the same fixed precision configuration, initial weights, training members, training steps, and inference pipeline, compare:

1. Module recovery without fine-tuning
2. Standard QAT fine-tuning
3. Standard difficult-sample training, using the same positive-negative sample budget and coverage across categories, scales, and confidence levels
4. Precision intervention guidance for error correction as proposed in this candidate; differ from item 3 only in sample source or weight basis

Perform identical budget PDF-Distil or HQOD adaptations on the most competitive existing methods. Record separately the additional teacher forward passes, intervention scans, and training costs. Do not equate identical training steps with equal overall R&D costs. Include a control with randomized intervention information to check whether the gains stem solely from training more difficult samples

Prioritize official-reported AP, small-object AP, and recall-vs-false-positive curves within a pre-defined range; operating points are determined solely on the development set. The goal is to recover more targets at equivalent false positive rates or achieve lower false alarms at equivalent recall under similar operational costs. Do not arbitrarily select favorable thresholds on the test set for each method. Report all seeds, intervals, and error types, not just recovery_count metrics

The existing 546 development plot has been repeatedly used in research selection and cannot be claimed as an independent validation set; new training supervision must originate solely from the training portion. Data is isolated by scene and risk component; subsequent independent data access will follow original boundaries. Holdout content remains sealed and is not accessed in this study. Training random seeds and historical calibration seeds must be clearly separated

If this candidate fails to stably outperform standard difficult-sample training or existing quantization remedies under equivalent budgets, or if gains appear only in a single seed or specific threshold, discontinue it as the primary contribution. If foundational calibration has already mitigated most damage, do not persist with the module recovery line. If only QDQ is effective and the actual backend performance does not hold, do not claim the method is viable for edge deployment

The current phase completes the candidate pathways and validation design. The historical full-candidate numerical FAIL, formal training completion, saved-artifact audit, development AP, and device acceptance are maintained as originally determined.
