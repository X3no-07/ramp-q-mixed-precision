# Gate1: Precision Intervention Correction Training Pilot Has Been Launched



2026-10-26,Beijing Time 13:39 initiated. The user has agreed to execute a small validation of "using changes before and after module recovery to guide error correction training." This is a new limited development experiment and does not rewrite formal training, previous numerical FAILs, or prior AP conclusions.

Current experiment directory: [Precision Intervention Training Pilot](../../docs/DATA_AND_ARTIFACTS.md). Launching does not imply training has been completed; final status shall be determined by `attempt_01/live_status.json`, actual progress, and the final completion or failure receipt.

# # Pre-launch Checks Completed

All 8 tests have passed, including: strict consistency between quantization forward pass and original QDQ arithmetic, consistent behavior between straight-through and truncated gradients, accurate gradient acquisition and update of quantized weights, region exclusion and positive sample protection, matching count of difficult samples, and strict equivalence between outputs saved after one actual CUDA training run and independently reconstructed outputs.

The FP32 inputs and full original candidate outputs from the 128 development images in this round have been individually verified against historical SHA values. Historical records only preserved the SHA of the training_val original array, not the array itself. Therefore, to enable full candidate box comparison for new training candidates, the 128 reference sets have been reconstructed. Historical prediction results are directly reused, without re-running the full 414-image training phase.

# # Frozen Experimental Scope

| Item | This round's setup |
|---|---|
| Starting point | Formal 30th round FP32 EMA, SHA: 4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27 |
| Module | Fixed G1 restoration with floating-point W8A8 fake quantization for the remaining Conv layers; unfused BN is frozen, and the fixed DFL projection is not trained |
| training members | From the original 4530 training images, select 512 images according to a fixed hashing rule, without selecting based on prediction performance |
| Development validation | Select 128 images from the original 414 training_val images based on a fixed hashing rule; training and validation image SHA values do not overlap |
| calibration | The existing calibration seed 1 with 64 images; reuse maximum-value ranges and add fixed histogram-MSE range calibration without reading calibration labels |
| Three training arms | Standard QAT; difficult sample augmentation; precision intervention-guided error correction |
| Fairness | Same initial weights, fixed G1 configuration, members, and order; the last two arms match additional positive and negative supervision by category/scale or category/confidence interval |
| Training budget | Each arm runs 3 epochs, batch size 4, gradient accumulation 4, and 96 optimizer updates; only one training random seed |
| optimizer | SGD with learning rate 0.0001, momentum0.9, and weight_decay0.0001; AMP and image augmentation disabled; use the fixed final epoch 3 rather than selecting the best epoch |
| Evaluation | Evaluate on 128 images with development AP, fixed 0.25 threshold for target counting, pre-defined 0.2/0.25/0.3 thresholds for diagnosis, and original full candidate box value thresholds |
| Forward pass budget | The main experiment uses 6720 image passes; separately record 3 saved-state reconstruction forwards for each of 3 candidates; no automatic retry after failure |

All three arms use a pre-fixed MSE range calibration, not adapting the calibration method based on this round's development AP. Both Q/R under maximum range calibration and Q/R under MSE range calibration are retained as controls.

# # What is specifically being tested in this round

Compare Q and R on training images, track recovery and damage of real targets, and screen for newly introduced background false positives after recovery. Background counterexamples must avoid all valid targets and ignored regions; misclassification, localization errors, and duplicate boxes cannot be simply suppressed as 'no target'.

The two arms with additional supervision use the same loss formulation. The key difference lies in the selection criterion for supervision: the standard difficult sample arm selects based on difficulty, while the intervention arm selects based on changes induced by precision switching under fixed weights. Whether this latter approach truly provides additional benefit remains to be confirmed by results.

The deployment vision remains a single fixed mixed-precision model. The training and testing here use floating-point QDQ, not actual TensorRT INT8, and no board-level speed improvement is reported.

# # Result Boundaries

This round is a pilot with a single seed and limited training epochs. The 128 training_val images are part of the development dataset and cannot serve as independent scientific validation; no Selection or Holdout images, labels, or detection results are read. If performance is unstable or fails to exceed the baseline on difficult samples, such findings must be recorded truthfully, without relying on threshold tuning, seed selection, or rewriting previous thresholds to sustain conclusions.

Frozen contract SHA256: `027ba8b666062979780d75d30ccdf5e0178e2ecd0b508c5f2aaf1f3f7bc4f577`

This report only records the initiation facts; subsequent results will be documented in a new report. The entry code, test scripts, contract, and source bindings from this round are preserved.
