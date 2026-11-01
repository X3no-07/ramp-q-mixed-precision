# Research Direction Review: Supervision Decoupling and Minimal Validation

October 29, 2026. **This is a research proposal, not an experimental result.** It is based on the terminal decision in report 90, the current training code, and the literature search recorded for this proposal. No new training, Holdout access, or device-performance measurement was performed.

The priority hypothesis is that quantization perturbations may change training supervision through dynamic label assignment. Two quantized forward passes of the same network would exchange assignment objects to isolate this feedback. The code establishes a possible path, but neither a harmful effect nor the effectiveness of cross-assigned supervision has been demonstrated. The proposed direction is new to this project; the limited search does not establish worldwide novelty.

## 1. Starting point established by report 90

| Method | AP50–95 (%) | Detections at confidence 0.25 | False positives at confidence 0.25 |
|---|---:|---:|---:|
| Continued standard QAT | 11.985899444 | 10552 | 7728 |
| Ordinary hard-example training with gradient protection | 11.918156643 | 10486 | 7494 |
| Intervention-based sampling with gradient protection | 11.877794522 | 10443 | 7494 |

The intervention arm lost 0.108104922 AP percentage points against standard QAT and detected 43 fewer targets than the equally protected hard-example arm at the same false-positive count. This experiment does not support an additional benefit from precision-intervention-based sampling. Fewer false positives, a successful artifact audit, or satisfied gradient constraints cannot be recast as method success.

The result covers one seed, 414 repeatedly used development images, and floating-point QDQ. It closes this correction branch rather than rejecting all quantization approaches. A new hypothesis must change the actual intervention. Retuning old sampling thresholds, adding correction epochs, or renaming the method does not provide new evidence.

Source: [report 90](../../03_gate1_visdrone_validation/docs/90_gate1_guarded_correction_terminal_review_and_closeout.md). Report 87's low-score-candidate analysis can motivate a question, but cannot replace the terminal result or establish the proposed mechanism.

## 2. Related work and overlap to avoid

This table summarizes only the aspects relevant to the proposed research question. Numbers from papers using different datasets, models, and bit widths are not ranked directly.

| Work | Relevant prior contribution | Implication for this project |
|---|---|---|
| [MLFQ, 2026-09-12](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70476) | Object-scale-dependent spatial weights for local feature reconstruction; reported benefits are mainly at W4A4/W3A3, with smaller and inconsistent W8A8/W6A6 differences | Small-object foreground weighting and background downweighting alone are insufficient; low-bit gains cannot be used to predict Orin INT8 gains |
| [InlierQ, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e538b85e58030aabb69945990b7f4139-Abstract-Conference.html) | Gradient-related saliency and EM distinguish useful responses from outliers for few-shot, label-free PTQ | Task-relevant activation selection has close precedents; comparisons with labeled methods must disclose the supervision difference |
| [HQOD, 2024](https://arxiv.org/abs/2408.02561) | Coordination of classification confidence and localization quality in quantized detection | Classification/localization consistency or an extra coordination loss is not a broad novelty claim |
| [LAD / CoLAD, WACV 2022](https://arxiv.org/abs/2108.10520) | Teacher-provided dynamic assignments and networks that exchange teacher roles | FP32 teacher assignment or two branches teaching each other is not new by itself |
| [Quantized YOLO oscillations, WACV 2024](https://arxiv.org/abs/2311.05109) | STE-training oscillations, EMA, and short QC correction | Quantization-training instability already has prior work; this project's fixed-scale code does not establish oscillation of learnable scales |
| [Stable-DINO, ICCV 2023](https://arxiv.org/abs/2304.04742) | Matching stability in detection | Stable matching is not itself a new problem; quantify the additional supervision changes caused by precision and their consequences |
| [Rank Mimicking, AAAI 2022](https://arxiv.org/abs/2112.04840) | Distillation of detection-candidate rankings | A ranking alternative must go beyond copying teacher rankings |
| [DetPTQ, 2023](https://arxiv.org/abs/2304.09785) | Task-loss-guided local reconstruction metrics for detection PTQ | Task-aware calibration alone is not a sufficient contribution |

The search also identified semantic/localization-decoupled assignment and open-vocabulary detector quantization. The paper should therefore target a mechanism that controlled experiments can test, rather than claim to be the first to consider small objects, supervision, or stability. This is a selected-neighbor list, not an exhaustive novelty proof.

## 3. Primary direction: supervision decoupling under quantization perturbations

### 3.1 Verified implementation path

The current training loss passes predictions from the current forward pass to `TaskAlignedAssigner`:

```python
assigner(
    pred_scores.detach().sigmoid(),
    pred_bboxes.detach() * stride_tensor,
    anchor_points * stride_tensor,
    gt_labels, gt_bboxes, mask_gt,
)
```

Source: [current training loss](../../03_gate1_visdrone_validation/scripts/gate1_ignore_trainer.py#L195). `detach()` prevents gradients from flowing through the assigner, but does not prevent numerical perturbations from changing its output.

The pinned Ultralytics implementation uses top-k assignment, a classification exponent of 0.5, and an overlap exponent of 6.0. The overlap is CIoU clipped to nonnegative values. Geometric filtering, top-k selection, competition between ground-truth objects, and soft-target normalization all contribute to supervision. The corresponding external dependency source is not distributed in this snapshot; see [artifact scope](../../docs/DATA_AND_ARTIFACTS.md).

Two qualifications matter:

1. If classification confidences shrink uniformly while candidate identities and assignments remain fixed, and epsilon effects are ignored, the common scale factor may cancel during assignment-score normalization. Lower confidence does not necessarily imply weaker supervision.
2. If a ground-truth box contains no geometrically valid candidate point, cross-quantization supervision cannot create one. Treat that structural case separately rather than attribute it to quantization.

The required measurements are which candidates become positive, which GT each candidate belongs to, the resulting soft-target weights, and their gradient effects when only numerical precision changes at fixed weights and inputs. The code establishes a possible path; these quantities have not yet been measured for the proposed study.

### 3.2 Falsifiable hypothesis

Quantization can both perturb predictions and alter the supervision generated from those predictions. The next optimization step then uses that changed supervision.

The hypothesis is that the second feedback amplifies the first error for some small objects and harms QAT. **Assignment disagreement is not necessarily harmful, and floating-point assignment is not ground truth for optimal assignment.** A controlled supervision replacement must establish a material effect.

This differs from report 90: that method used precision interventions to select additional correction samples. The proposed method changes the source of supervision in the native detection loss and does not first select objects worth rescuing from intervention gains.

### 3.3 First experiment: a 2×2 prediction/supervision swap

At fixed latent floating-point weights θ and input x, obtain an unquantized forward F and a quantized forward Q. Generate complete GT-based assignment objects A_F and A_Q: foreground masks, GT identities, target boxes, soft class targets, and normalization rules. Stop gradients through all assignment objects.

| Prediction | Supervision A_F from floating-point predictions | Supervision A_Q from quantized predictions |
|---|---|---|
| Floating-point F | L(F,A_F) | L(F,A_Q) |
| Quantized Q | L(Q,A_F) | L(Q,A_Q) |

For the corresponding parameter gradient g, the following is an algebraic identity:

```text
g(Q,A_Q) - g(F,A_F)
  = [g(Q,A_F) - g(F,A_F)]
  + [g(Q,A_Q) - g(Q,A_F)]
```

The second term changes only supervision on the same quantized forward and measures the gradient change along that path. The decomposition does not prove that the effect is harmful or that changing assignment improves AP.

Swap the complete supervision object, not just `fg_mask` while retaining incompatible GT boxes, ignore masks, or normalization denominators. F is the same-weight forward with QDQ disabled, not a separately trained stronger teacher. Different-weight teachers belong in subsequent baselines.

### 3.4 Minimal method: two quantized forwards with cross-assigned supervision

With the same network weights, training image, and bit width, draw two quantization configurations ξ₁ and ξ₂. Obtain predictions p₁ and p₂ and their assignments A₁ and A₂.

```text
Self-assigned:  L_self  = 0.5 × [L(p₁,A₁) + L(p₂,A₂)]
Cross-assigned: L_cross = 0.5 × [L(p₁,A₂) + L(p₂,A₁)]
```

Both arms use identical images, configuration-sampling sequences, two quantized forwards, and backward propagation through both prediction branches. Only the assignment pairing changes. Measure elapsed time and memory; matching forward counts does not establish exactly equal total computation.

The aim is to prevent one quantization perturbation from determining its own candidate supervision in isolation. The other assignment is not assumed to be more correct; cross-assignment may propagate errors when both assignments are wrong. It directly tests whether correlation between prediction and supervision perturbations is harmful.

Suggested configuration definition:

- Keep deployment bit widths, quantization locations, operator forms, and deterministic rounding rules fixed.
- Repeatedly sample equal-sized calibration subsets from the training pool to build a small, preregistered activation-scale configuration bank. Specify the base deployment configuration q₀ in advance.
- Draw configurations independently for the two branches while holding weights, augmentation, and other randomness fixed. Identical draws are allowed and reduce naturally to self-supervision.
- Do not describe arbitrary Gaussian noise or stochastic rounding as measured TensorRT noise. Calibration-configuration variation is the explicitly defined training perturbation here.
- Deploy one network, one forward pass, and fixed q₀. Training branches do not enter the inference graph; latency and actual INT8 execution still require device verification.

For ξ₁=ξ₂ with deterministic forwards, A₁=A₂ and L_cross equals L_self exactly. This is an implementation invariant, not a performance guarantee. Gains requiring unrealistic perturbations do not explain the original INT8 setting.

The first configuration should not add sample miners, correction heads, gradient projection, or downstream scorers. That keeps supervision pairing as the single intervention.

### 3.5 Potential contribution and rejection cases

A defensible contribution would combine reproducible mechanism evidence separating prediction error from supervision changes; same-weight, same-bit-width cross-assignment with independent quantization configurations; and repeatable task gains over matched self-supervision, standard QAT, and applicable prior methods, followed by fixed-quantizer deployment validation.

Two branches teaching each other, stable labels, or allocating extra positives to small objects cannot be claimed as standalone innovations.

Retain the following counterevidence:

- If perturbation training with self-supervision gives the same gain, attribute it to ordinary perturbation training rather than cross-assignment.
- If an LAD-style floating-point-teacher adaptation performs equally well with less total compute, an advantage for the new algorithm is unproven.
- If assignments change substantially but cross-assignment lowers AP, the mechanism exists but the intervention direction fails.
- Improved assignment consistency without detection gains is insufficient; internal metrics do not replace task metrics.
- W4A4-only gains may motivate a low-bit study, but do not establish the current Orin INT8 objective.
- If strong standard QAT or EMA/QC removes the gap, weaknesses of the old fixed-scale training setup are insufficient motivation for a new-method claim.

In expectation, cross-assignment is related to averaging supervision over quantization configurations. It should not be presented as unrelated to assignment smoothing or teacher ensembles; distinguish the computation and statistical objective. Also, `0.5*(L_self+L_cross)` is an interpolation, not a wholly independent strong baseline.

## 4. Alternative A: quantization-residual calibration across crop views

This refines an unvalidated route already proposed in [historical proposal 07](historical_proposals/07_paper_reframing_candidates_and_rejection_criteria_20261026.md); it is not newly introduced here.

The same GT may appear at different scales, context, and boundaries in two crops. Compute each view's own floating-point result first, then isolate the additional quantization discrepancy rather than attribute the floating-point model's crop sensitivity to INT8.

Let T_v be the crop coordinate transform and D_g normalize coordinate errors by GT width and height:

```text
e_g(v) = D_g [T_v⁻¹ b_Q(g,v) - T_v⁻¹ b_F(g,v)]
L_pair = mean_g || e_g(v₁) - e_g(v₂) ||₁
```

Use floating-point predictions and training GT to define candidate correspondence in each view; read the same candidate positions from the quantized branch. Matching only surviving high-score quantized boxes would exclude missed targets from the loss. Different views may select different candidates, provided they correspond to the same GT.

Combine L_pair with absolute-error or task losses: it can be zero when both views have the same incorrect displacement. Do not compare boxes across views without coordinate alignment.

Every method must see identical crops and forward counts. Compare standard multiview calibration, applicable DetPTQ/Reg-PTQ, an object-scale-weighted MLFQ adaptation, and the added same-GT residual pairing. Include a negative control that shuffles GT pairings while preserving class/scale distributions to test dependence on instance correspondence.

The possible contribution is calibrating the structure of cross-view quantization residuals rather than only single-view reconstruction error. Stop if ordinary multiview calibration closes the gap or residual differences do not cause actual misses/merging errors. For a tiled application, report latency including cropping, transfer, and merging rather than tile inference alone.

## 5. Alternative B: calibration at candidate-competition boundaries

The same numerical error can have different consequences far from a ranking/NMS boundary and at a decision-flipping boundary. A possible objective would target competition events that change the final detected-object set.

- Use training GT to distinguish duplicate candidates for one object from candidates for two real objects.
- Record classification-ranking margins, box-overlap margins relative to the NMS threshold, and prefilter top-k boundaries.
- Penalize quantization-induced erroneous suppression, missed distinct objects, or added false positives. Do not indiscriminately copy all floating-point rankings and errors.
- Modify quantization parameters during training/calibration while retaining single-model postprocessing at deployment.

A fixed-direction margin loss may protect a verified relative candidate relation. Competition between different GT objects also requires handling changes in the box-overlap graph; score-order protection alone does not preserve NMS output.

**This alternative has the greatest novelty risk.** HQOD and Rank Mimicking are close precedents. Evidence must show additional benefit from GT-identity-aware separation of useful deduplication and harmful suppression, including quantization-induced competition-graph changes. Error ranges from a finite calibration sample are not strict robustness certificates. Reject the route if ordinary ranking distillation or threshold adjustment explains the gain.

## 6. Bounded minimal validation sequence

This is a proposed plan, not a running job. The report-90 branch remains closed; the new route does not inherit a “continue until passing” rule.

### Step 1: mechanism measurements only

Preregister 128 images from the existing training pool, stratified by GT scale and crowding rather than selected from old-method successes or failures. Initially measure the existing standard-QAT endpoint and FP32 starting point separately. Within each checkpoint, compare same-weight F/Q; the two checkpoints themselves are not a precision control.

Collect:

1. Per-GT valid geometric candidate counts, positive-set changes, GT-assignment changes, and zero-positive rates.
2. Soft-target weights, normalization denominators, positive/negative classification gradients, and box-regression gradients.
3. Gradient differences under the 2×2 supervision swap on a fixed 32-image subset.
4. Repeated-forward numerical noise at a fixed configuration. F/Q differences are precision differences, not evidence of oscillation across training steps.

Stop the primary hypothesis if invalid candidate geometry dominates, precision barely changes supervision, or supervision changes have no identifiable harmful effect. Internal differences alone do not justify indefinite work; the next step must test task gains.

### Step 2: minimal training controls with a fixed budget

| Arm | Purpose |
|---|---|
| Strong standard QAT | Establish a baseline with reasonable calibration, optimization, and schedule; the old fixed-scale result remains historical context |
| Two configurations with self-assignment | Control additional forwards, perturbations, and data exposure |
| Two configurations with cross-assignment | Change only assignment pairing |
| Floating-point-teacher assignment adaptation | Test whether an ordinary LAD adaptation suffices; count the teacher's complete overhead |

Self- and cross-assigned arms can directly match their principal forward/backward budgets. Teacher, EMA/QC, HQOD, and other strong baselines must account for all training and calibration costs. Compare performance–compute curves; do not ignore teacher work or extra epochs when declaring equal budgets. Verify each adaptation rather than substitute its name for an implementation.

The first fixed schedule determines whether expansion is justified; it does not demonstrate convergence. Primary metrics are AP_small and overall AP, accompanied by PR curves and TP/FP at fixed operating points, stratified by object scale. For a fixed-FP budget, select thresholds on the designated development set and apply them unchanged later. Searching thresholds on the evaluation set is an oracle description only.

Lock the practical gain threshold, acceptable overall-AP change, and training budget before execution. Do not change the primary metric, add seeds, or select a favorable epoch after viewing curves.

### Step 3: expand only after gains in Step 2

- Use multiple training seeds, a second suitable detector, and a second domain. Define small-object scales and report across-seed variation.
- Validate fixed-q₀ inference, export and inspect actual INT8 execution, and measure end-to-end latency and memory. Floating-point QDQ success is not TensorRT/Orin success.
- The repeatedly used 414 images and existing Selection set cannot serve as fresh independent evidence. Define the independent-data protocol first and respect the established Holdout purpose.
- Choose deployment bit widths supported by the device; W4A4/W3A3 simulations do not establish Orin acceleration.

## 7. Research-topic decision

| Priority | Direction | Rationale | Main uncertainty |
|---|---|---|---|
| 1 | Supervision decoupling under quantization perturbations | A clear code intervention, a single-variable self/cross control, and single-model inference after training | Supervision changes may be harmless; LAD or ordinary perturbation training may suffice |
| 2 | Cross-crop quantization-residual calibration | Relevant to tiling and separates crop sensitivity from added quantization error | Ordinary multiview calibration may suffice; this refines proposal 07 |
| 3 | Candidate-competition boundary calibration | Directly connected to final target retention | Substantial overlap with ranking distillation and HQOD requires prior-method exclusion |

Working title: **Small-Object Detector Training Under Coupled Quantization Perturbations and Dynamic Label Assignment**. If the intervention is effective, a method-specific title could be **Cross-Assigned Supervision for Quantized Small-Object Detection**. Do not use result claims such as “solves,” “improves,” or “first” before experiments.

The next useful action is the fixed-training-sample supervision-swap measurement. It can test whether a new mechanism merits investment without retuning the correction branch that failed in report 90.
