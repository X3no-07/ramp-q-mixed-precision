# Gate1 Research Pathway Review and Baseline Priority Plan



2026-10-26.

**We recommend pausing automatic scorer addition and recovery modules, and instead allocating the next computational budget to strengthen trustworthy baseline models. Only after establishing a solid baseline should we decide whether the original mechanism warrants retention as the paper's main contribution. The current results are insufficient to support a high-quality paper contribution and do not justify the claim that the entire quantization direction is fundamentally infeasible.**

# # Newly Identified Evidence Gaps

Formal FP32 training did complete 30 epochs and 33990 batches; the training-completion and saved-state-consistency conclusions are unchanged. The original per-batch scalar loss averages 12.851972 over epochs 21–25 and 12.254106 over epochs 26–30, a relative change of-4.652%. The logs record deferred AP validation; no per-epoch validation AP curve supports a convergence decision in this round.

The training loss continues to decrease, so it cannot be claimed that failure is necessarily due to non-convergence. Similarly, it cannot be claimed that the baseline has been sufficiently trained. The batch-wise averages include the original final batch size and existing loss scaling, and are only described within the same training log. They cannot be converted into AP or ranked against external models.

The current ordinary QAT comparison only covers 512 images, 3 epochs, 96 optimizer updates, and a single training seed. It is sufficient as a control for a pilot experiment under the same budget, but it does not justify claiming that ordinary QAT or current task-aware quantization methods have been sufficiently trained. The past three correction arms and this pilot were all under the same budget and are not re-evaluated as successful or invalid by this review.

# # Existing Full 414-Image AP Gap

| Existing branches | AP50–95(%) |
|---|---:|
| Official FP32 | 12.440710038 |
| MSE Range R | 11.869287523 |
| Standard QAT Pilot | 11.921909146 |
| Original Baseline Scorer | 11.940006994 |

The old scorer improved relative to R by +0.070719471 percentage points, and relative to the standard QAT pilot by +0.018097848 percentage points, still falling short of FP32 by -0.500703044 percentage points. Combined with Report No. 77, the reduction in false positives relies on a small subset of prefix groups; thus, a minor aggregate AP difference cannot be taken as evidence of a stable method advantage.

The three new groups/feature candidates in Report 80 do not satisfy all range and leave-a-group conditions. One local visual candidate in Report 82 on calibration training tiling 178 retains 56 fewer true positives (TPs) compared to the previous scorer under the FP budget, although BCE slightly decreases, and the joint criterion is not met. A total of 7 scorer candidates are retained; these results do not support continuing with intuitive module addition and acceptance.

# # Proceed in this order next

1. **First, fix a trustworthy baseline protocol.** Agree in a separate new directory on sufficient training budget, control implementation, optimization settings, data roles, and actual target backend; retain the original 30-round candidates, without covering continuation training or rewriting contracts. Complete the qualification and training sufficiency evidence for conventional QAT and published task-aware quantization baselines. Do not treat the current unmerged Conv floating-point QDQ as a genuine INT8 backend, nor assume that adding more training rounds will necessarily improve AP.
2. **Then, evaluate whether the original mechanism provides additional value.** Only compare the original precision allocation or mitigation strategy with conventional and task-aware baselines when data, training, computational budget, and post-processing criteria are comparable. Require prior evidence for AP, fixed endpoint detection and false positive rates, and additional overheads; do not select the best threshold, prefix, seed, or module combination from development results. Independent validation requires clearly open, independent data ranges; holdout sets remain sealed.
3. **Decide retention or replacement based on results.** If a strong baseline already covers current gains, the original method becomes a control and negative result, and its main contribution is withdrawn; if stable, reproducible, and practically valuable quantization degradation remains, then formulate a new, distinguishable research hypothesis around the identified mechanism. The new hypothesis must be explicitly differentiated from prior work and cannot treat local validation, standard distillation, or task consistency as novel.

# # Constraints from prior work

[IoU-Net](https://arxiv.org/abs/1807.11590) has learned localization quality, [GFLV2](https://arxiv.org/abs/2011.12885) studies lightweight quality ranking; standard quality scoring is not a unique contribution of this project. [HQOD](https://arxiv.org/abs/2408.02561) investigates inconsistency between classification and localization quality under quantization and proposes task-related losses and IoU losses; the claim that 'quantization causes classification-localization inconsistency' is not a new discovery. [DetPTQ](https://arxiv.org/abs/2304.09785) has already used detection task losses to guide layer-wise quantization parameter selection. Their published results cannot be directly considered as performance outcomes of the local YOLO11n implementation; applicability and implementation validation are required in the next phase.

# # Actual Completion and Unexecuted Actions in This Round

This round only reads 33,990 existing training batch records and aggregated results, with no new training, image inference, AP, candidates, or thresholds. Loss statistics, source identities, and machine conclusions from the 30 rounds are preserved. Full training of the new baseline, local reproduction of HQOD/DetPTQ, genuine INT8 implementation, and new independent scientific validation have not been executed. This report does not constitute a commitment to publication in any journal and does not imply that the paper has been saved or recovered.

Read together with report 82, the clearest next action is to strengthen the experimental evidence before choosing a method, rather than continuing to seek a success narrative around the 7 scorers.
