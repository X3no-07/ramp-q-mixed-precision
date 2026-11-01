# Gate1: Validation of Synthetic Training Pipeline and FP32 Saving Integration



Using 4 newly generated training images, 2 training_val images, and a randomly initialized YOLO11n with 10 classes and 2,591,790 parameters, one CPU training epoch completed through the actual `DetectionTrainer.train()` call. The final attempt_03 performed 2 SGD steps and 2 EMA updates. Ignore regions entered the actual TAL classification loss; the live FP32 EMA captured before saving can be rebuilt from its architecture with byte-identical CPU synthetic outputs. This validates only the synthetic engineering pipeline, not a scientific baseline or on-device acceptance.

# # Actual Execution and Verification

| Item | Result |
|---|---|
| Execution Scope | CPU, single-threaded, workers=0, batch=nbs=2, AMP disabled, 1 epoch |
| New Data | 4 train images and 2 training_val images; explicit membership lists and three-file byte/SHA checks; image IDs and image SHAs are disjoint across splits |
| Actual Updates | 2 SGD and EMA steps; the first trainable parameter in each of the two steps changes by 432 elements |
| Total Parameter Changes During the Run | 238 parameter tensors and 2,209,385 elements were recorded during execution |
| Two Positive Anchor Batches | Anchors 9 and 10; full classification rows of positive samples were preserved |
| Two Background-Ignored Anchor Batches | Anchors 67 and 98; BCE classification masking was actually triggered |
| FP32 Capture | epoch 1, EMA updates=2, before native half serialization; the source was already FP32 and was not upcast from half |
| Architecture reconstruction | Restricted state-only loading, strict state_dict, full YOLO11n rebuild; CPU parsing of synthetic inputs and outputs with exact matching |
| Three-component testing | Dataset 45, criterion/Trainer 51, FP32 hook 23; total of 119 items passed |
| AP | `SMOKE_NO_TASK_AP`, task AP not computed |

[Final training receipt](../../docs/DATA_AND_ARTIFACTS.md); [Real runtime events](../../docs/DATA_AND_ARTIFACTS.md); [Pre-frozen contract](../../docs/DATA_AND_ARTIFACTS.md); [FP32 capture receipt](../../docs/DATA_AND_ARTIFACTS.md).

# # Integration Content and Fixes

Each Dataset sample is re-validated against manifest, image, positive labels, and ignore sidecar byte counts and SHA; labels and ignore boxes share the actual loader's dual-axis ratio, integer padding, and same flip parameters, then output standard detection fields and ragged ignore regions. Only explicit restricted chains are supported; no complex augmentations are executed. score=0, category0, and category11 sources are separately preserved; invalid, empty area, or fully outside-canvas regions reject entire samples, without silent deletion, nor converting low-quality positive boxes into ignore.

The real criterion maintains the currently fixed source code TAL, box, DFL, denominator, weights, and batch multiplier; only modifies background BCE for unassigned anchors whose centers fall within ignore regions. For empty ignore cases with/without GT and class weights, loss, model parameters, and input gradients match the original framework element-by-element. The real TAL empty GT branch returns floating-point zero fg_mask; at seams, only strict helper conversion to bool is applied; original assignment tuple and box/DFL paths remain unchanged. Local logit gradients for full anchor rows are preserved, but this does not imply that shared model parameter total gradients remain unchanged; synthetic BCE reduction is not a precision gain.

The FP32 hook first captures the live EMA before calling the standard framework's save interface; the standard framework still performs deep copy followed by half-precision serialization. EMA update count and state remain unchanged before and after capture; isolated random number generation is constructed during architecture rebuild; testing covers live state/cache/RNG not being altered. Only inference state is saved, not optimizer state or full training state recovery; no proof exists that the main thread or non-reentrant locks prevent any external writer; the current calling protocol explicitly prohibits concurrent EMA writes.

The first attempt_01 functional check has passed, but source review identified that disabling sync/loggers alone is insufficient to prevent framework registration of platform callbacks. v2/attempt_02 disables integrated registration at process creation before Trainer construction, verifies all initial callbacks originate from the framework base, then adds local recording of callbacks. No modifications are made to framework files in the environment; this is a process path constraint, not claiming OS-level network sandboxing, and there is no evidence to infer that attempt_01 occurred external communication or data upload.

Independent forward audit revealed that during Trainer construction, select_device(cpu) changed the requested 1 thread to 8; attempt_01/02 receipts hard-coded threads=1, which is inaccurate. Independent process reconstruction with 1 thread fails to achieve exactness; with 8 threads, strict exactness is achieved. No modification or relaxation of exactness checks is made; original erroneous receipts and diagnostics are retained. The final solution uses v3/attempt_03: after construction, thread count is fixed to 1; at training start, each optimizer step, before saving, and final receipt, actual assertions of thread count are made. Backend flags are recorded at each step, but not claimed to be fully logged over time. Upon Trainer return, unset_deterministic is called, thus independent reconstruction verifies the actual flags at the save entry point, without relying on flags after training return. All three runs share the same synthetic fixture, not independent data evidence.

# # Independent Audit and Boundaries

Source reviews were conducted separately on Dataset, criterion, Trainer, and FP32 hook; a re-review was also performed on the corrected runner. The audit verified frozen inputs and outputs, actual events, two-step updates, mask records, and save timing; independent loading uses only the current FP32 state and reconstructs a CPU model based on the saved architecture to validate parsing of synthetic inputs and outputs. Standard best/last models only perform byte-level hashing, without deserialization. No original 30-round weights, real images, annotations, selection detection inputs, or holdout content were read.

Initial and final models only store per-tensor hash descriptions and runtime change counts; independent audit can verify tensor hash variations and count limits/sums, but cannot recompute the element-wise differences of 2,209,385 elements from un-saved initial and final tensors; TAL counts are derived from source code and event/receipt cross-validation, and no independent re-run of training assignment is performed.

The current Trainer permits only synthetic-only, single-epoch CPU execution. Separation across manifests is demonstrated by the explicit preflight in this round. An arbitrary custom dataset_factory remains a trusted extension, so universal leakage prevention cannot be claimed; the default reviewed Dataset is used here. Compatibility of the existing 4534 real training images with this invalid-ignore handling has not been checked, and no new real-data quarantine or formal training contract has been generated. CUDA training and synchronized CUDA capture require separate qualification.

[Component source review](../../docs/DATA_AND_ARTIFACTS.md); [FP32 source review](../../docs/DATA_AND_ARTIFACTS.md); [Actual result review](../../docs/DATA_AND_ARTIFACTS.md); [Independent state-only reconstruction audit](../../docs/DATA_AND_ARTIFACTS.md).

# # Current Status and Next Steps

The original 30-round training candidates, 109-image hybrid prototype, and all previous archived results remain unchanged. The original full FP16 guardrail remains `FAIL_FIXED_TOLERANCES`, and the old Gate0.5 remains `FAILED_UNCHANGED`; research Gate1, real-data AP, export/TensorRT/Orin, and speed/power efficiency have not been passed based on these.

Next steps require freezing real data members and illegal ignore/quarantine policies, qualifying verification for CUDA training and live FP32 capture, followed by formal training and acceptance conducted in accordance with independent validation, submission order, and candidate budget contracts. The precision loss from the missing half-precision EMA cannot be recovered.

[This decision round](../../docs/DATA_AND_ARTIFACTS.md); [Status snapshot v11](../../docs/DATA_AND_ARTIFACTS.md).
