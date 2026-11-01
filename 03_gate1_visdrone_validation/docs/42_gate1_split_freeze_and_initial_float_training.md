# Gate1: Split Freezing and First Round of Floating-Point Training



Date: 2026-10-22. This document records the data isolation, evaluation implementation, and candidate training for Gate1; it does not constitute scientific acceptance at Gate1. **A new attempt has already entered the 1st/30th training round, with batch=4 and AMP disabled; the full training rounds have not yet been completed.** The current run status and inspection time are available in [Current Run Snapshot](../../docs/DATA_AND_ARTIFACTS.md).

# # Completed Preparation

A new split plan has been generated outside the original preparation directory, with the original images, annotations, and delivered content remaining unchanged. Of the 409 images in the original training-validation set, 407 come from a single conservative group, and thus cannot be used to claim coverage of diverse scenarios. The new plan reassigns images into complete groups without considering model results, expanding the internal validation set to 58 groups.

| Purpose | Image | Conservative Group | Description |
|---|---:|---:|---|
| training | 4534 | 92 | 227425 effective positive labels, 10 classes present |
| Internal validation during training | 414 | 58 | 28007 valid positive labels, all 10 classes represented, with 41 images in the largest group |
| Calibration pool | 689 | 24 | Three pre-fixed lists of 64 images each, calibration not yet executed |
| Engineering debugging | 109 | 6 | Maintained independently, not included in this training/validation |
| Selection | 548 | 75 | Official validation set, no model inference, module selection, or scoring has been performed yet |
| Isolation | 725 | 27 | Includes annotation quality issues, associations with Selection, and risks of newly identified near-duplicates |

Grouping is defined by filename prefixes, exact content duplication, and reviewed risk edges; it cannot prove true camera or scene identity. The three calibration lists are mutually exclusive by image ID and content SHA, covering 17, 17, and 19 groups respectively, with no more than 8 images per group within each seed; the three seeds still share the same calibration pool scenes, forming a single dataset rather than three independent datasets.

Contract splitting(../../docs/DATA_AND_ARTIFACTS.md), label inventory(../../docs/DATA_AND_ARTIFACTS.md), and independent verification(../../docs/DATA_AND_ARTIFACTS.md) retain detailed lists and identities. Independent verification passes 7103 assertions, revalidating the byte count and SHA values of the original prepared directory of 21077 files; no splitting generator is introduced to replace independent checks.

# # Near-duplicate screening and conservative isolation

Only the 7019 images in the original train/val sets were read. Candidates with dHash or pHash Hamming distance at most 4 were screened. Independent exhaustive verification of all 24629671 cached-hash pairs confirmed that the 37 candidates, including 5 spanning roles, prefixes, or groups, match the tool output. This validates hash retrieval within the stated definition; it does not prove the absence of other near-duplicates or shared scenes.

Visual review reveals that `train/9999937_00000_d_0000015` and the original isolated set `train/9999950_00000_d_1007223` have nearly identical viewpoints and vehicle positions; both hash distances are 0, and the RGB thumbnail RMSE is approximately 0.00140. Conservatively merge the associated groups, adding 228 original training images to the isolated set. The other two candidate visual scenes with similar dHash values are significantly different and are not merged based on this. The known exact duplicates and prior-prefix isolation relationships are retained using the original handling.

Evidence found in [Method and Runtime Statistics](../../docs/DATA_AND_ARTIFACTS.md), [Visual Review Decisions](../../docs/DATA_AND_ARTIFACTS.md), [Contact Graph](../../docs/DATA_AND_ARTIFACTS.md), and [Exhaustive Review](../../docs/DATA_AND_ARTIFACTS.md). The test-dev ZIP contents were not opened or listed in this process.

# # Scope of evaluation implementation

Official evaluation code is fixed to [VisDrone toolkit commit 005445782213e20cb91bc50a597db3dd949e749a](https://github.com/VisDrone/VisDrone2018-DET-toolkit/tree/005445782213e20cb91bc50a597db3dd949e749a). Implemented pixel union filtering for category=0, absorption of overlapping prediction areas with score=0 within the same class, and target-level matching with project-specific determinism; original annotation physical row IDs are preserved. The project's primary thresholds are confidence ≥ 0.25 and one-to-one IoU ≥ 0.50 within the same class.

37 synthetic tests passed in actual training environments. Fixed behaviors in the source code—pixel coordinates, prediction truncation order, category aggregation, and recall denominator—cannot be replaced by standard YOLO/COCO metrics. **The full official AP was not implemented, and no MATLAB/Octave runtime validation was obtained.** Target-level counts and subsequent false positive statistics should be explicitly named as project-specific metrics.

[Source Code Review](../../docs/DATA_AND_ARTIFACTS.md), [Machine-readable Evaluation Policy](../../docs/DATA_AND_ARTIFACTS.md), and [Actual Environment Testing](../../docs/DATA_AND_ARTIFACTS.md) document boundaries. Standard YOLO positive labels omit ignored items, but training loss does not suppress negative samples in these regions; this limitation remains in place.

# # First Training Contract and Saved Failures

This experiment uses the official YOLO11n COCO initial weights from the repository, with SHA `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`. Input size is fixed at 640, batch=4, workers=0, AMP disabled, 30 epochs, SGD optimizer, lr0=0.01, momentum=0.937, weight decay=0.0005, seed=20261022, deterministic mode, patience=0, and mosaic is disabled in the last 10 epochs. All other practical default parameters and framework source code are fixed and saved during execution.

Checkpoints are selected based solely on the built-in mAP50–95 metric evaluated on the internal validation set during training. In case of ties, the later epoch is retained. This metric does not account for ignored detections and is only used as a training monitoring proxy and weight selection criterion, not as an official VisDrone AP. The 30-epoch limit is the initial candidate budget and does not imply sufficient convergence has been proven. AMP is disabled during training, but the framework saves FP16 EMA checkpoints. When establishing FP32/FP16 reference points in subsequent phases, this serialization fact must be disclosed.

The contract v1 was not used for training and was instead checked and replaced by v2. The v2 corresponding to `float_training_v1` failed in the first batch due to a new limited compatibility check between the framework and the loss dictionary, completing 0 training epochs; all logs, code, contracts, and failure evidence are retained. This is a runner compatibility failure and cannot be used to infer model quality or hardware faults. The fix added recursive checks for dictionaries/lists/tensors and regression cases, with 13 test items and 3 sub-cases passing. **The effective contract is v3, the new attempt is `float_training_v2`, and training is performed from original official weights without restoring from the failed checkpoint.**

Each batch's loss is monitored; before each checkpoint save, the live and EMA states are checked, and it is verified that the EMA converted to FP16 remains finite, preventing silent re-synchronization or acceptance of results after a mid-run interruption. If a batch-level change or abnormal value is detected, training is halted and the failure is preserved. No automatic hyperparameter re-tuning or restart occurs within the same contract. The additional callback for final best weights validation is recorded separately from the 30 training epochs.

Evidence is available at [Effective Training Contract](../../docs/DATA_AND_ARTIFACTS.md), [Contract Description](../../docs/DATA_AND_ARTIFACTS.md), [Regression Tests](../../docs/DATA_AND_ARTIFACTS.md), [Failure Review](../../docs/DATA_AND_ARTIFACTS.md). [Current Attempt Record](../../docs/DATA_AND_ARTIFACTS.md) and [Training Logs](../../docs/DATA_AND_ARTIFACTS.md) are dynamic runtime files and are not part of the static deliverable inventory; the current progress is based on time-stamped snapshots.

# # Subsequent Sequence

Complete the candidate training and freeze the best/last checkpoint identities first. Then check preprocessing, classes, box-coordinate restoration, valid-target hits, and FP32/FP16 numerical qualification. The official AP gap requires evidence from the comparison run or must remain explicitly incomplete.

Afterward, fix the quantization semantics, candidate/budget, post-processing, statistics, and research acceptance contracts before scanning modules on Selection. The final confirmed set is only released after both candidate and analytical locks are established. Currently, there are no quantization recovery results, no conclusion on superior candidate selection, and no Gate1 scientific PASS.

The original numerical guardrail for Gate0.5 remains `FAILED_UNCHANGED`; this determination is unchanged in this round. [Previous Delivery Integrity Review](../../docs/DATA_AND_ARTIFACTS.md) confirms that the 45 files from the original Gate0.5 and the 32 files from the prior Gate1 preparation delivery are consistent. The previous 40/41 reports and `current_preparation_state.json` are preserved as historical snapshots. This round adopts the new 42 report and status files.
