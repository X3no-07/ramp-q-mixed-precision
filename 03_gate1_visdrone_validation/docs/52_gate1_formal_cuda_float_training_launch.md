# Gate1: Official CUDA Floating-Point Training Initiation



This round, authorized by the user 'Train Now!', initiates a fixed-budget ignore-aware-precision floating-point training candidate. The new training set consists of 4530 images, training_val 414 images; YOLO11n, 30 epochs, 640 pixels, batch=4, nbs=64, AMP disabled, workers=0, SGD. The candidate pre-sets the EMA from the 30-th live epoch of FP32, without selecting based on training agent AP or framework best files.

Startup time: 2026-10-23 23:43 (Beijing Time). Startup records and runtime status are separately saved; the current epoch, batch, loss, optimizer/EMA update count, and failure status are taken from [real-time progress](../../docs/DATA_AND_ARTIFACTS.md) and [machine status](../../docs/DATA_AND_ARTIFACTS.md). Background execution still depends on the computer remaining powered on, and the training process will request the system to remain awake during runtime.

**First round actually completed: 1133 training batches, 423 optimizer/EMA updates, the first live FP32 EMA successfully submitted, training has continued into the 2th round.** The first two parameter updates during true warmup have been recorded; all batches in the first round retain the full anchor classification row, with background anchor suppression actually triggered. CPU and CUDA input reconstruction during saving are precisely consistent; another independent CPU process, constrained to load the first round FP32 state and strictly reconstruct the architecture, produces output exactly matching the saved reference. These are evidences of actual training and saving chains, not task AP or scientific acceptance.

[First-round independent verification](../../docs/DATA_AND_ARTIFACTS.md); [Startup status snapshot v13](../../docs/DATA_AND_ARTIFACTS.md). The snapshot is generated at the time of progress; subsequent rounds are based on real-time progress.

# # Data Disposition and Fixed Configuration

Based on [Previous Real Data Pre-check](51_gate1_real_training_data_compatibility_preflight.md), four images directly incompatible with the new training members are excluded: three images with identical frame and duplicate positive labels, and one image with zero height in a publisher-ignore region. The original images, original labels, shared sidecars, and previous splits are preserved; no label deduplication or individual ignore region deletion is performed.

Image-level exclusion is explicitly decided for this new contract. The issue arises from the source annotations of these four images; other group members remain in their original roles and do not migrate to training_val, Selection, or other roles. After re-verifying image SHA, prefix proxy, and the cross-role separation of existing refined_component, there is no overlap between 4530 training images and 414 training_val members. The previous candidate scheme of isolating 662 images from the full group was not applied.

The training reads new independent file copies and per-image sidecars, each member is bound to the image, positive labels, sidecar size, and SHA. Each sampling still verifies member and file identity. Actual augmentation is limited to explicit loader scaling, centering LetterBox, and horizontal flip (probability 0.5); no Mosaic, MixUp, perspective, or color augmentation is performed. The original dtype, member, and complex augmentation constraints are not relaxed in the previous module; this round uses a new adapter and new contract.

| Configuration | frozen value |
|---|---|
| model/category | YOLO11n / VisDrone 10-class |
| initialization | Official YOLO11n release weights with fixed SHA; new 10-class head initialized with fixed random seed |
| budget | single candidate, 30 epochs, seed=2026102303, no hyperparameter search |
| input/batch | 640 / batch=4 / nbs=64, gradient accumulation in framework |
| optimizer | SGD,lr0=0.01,lrf=0.01,momentum=0.937,weight_decay=0.0005 |
| warmup | 3 warmup rounds requested; formal training at 1133 batches per round corresponding to 3399 warmup iterations |
| precision | FP32 state and computation path, AMP disabled, TF32 disabled |
| backend | CUDA:0, deterministic algorithms mode enabled, cuDNN benchmark disabled, CPU single-threaded |
| candidate rule | Fixed FP32 EMA at the 30th epoch; task AP to be separately verified |

The storage precision of the original initialization weights does not equal the restored FP32 EMA from prior training. In this round, the reference is taken from the actual FP32 training state. The previous 30-epoch training candidate remains unchanged, and the two runs cannot be treated as a controlled experiment involving only one factor—precision change.

[Member Disposition Contract](../../docs/DATA_AND_ARTIFACTS.md); [Formal Training Contract](../../docs/DATA_AND_ARTIFACTS.md); [Startup Record](../../docs/DATA_AND_ARTIFACTS.md).

# # CUDA Validation Before Startup

Original 119 dataset items / loss trainer / CPU capture regression and new adapter passed 10 test cases; additional 2 test cases passed for preprocessing calls and single-warmup boundary, totaling 131 test cases.

The first GPU synthesis check used 8 images, batch=4, nbs=64, per round. With a fixed single-round framework retaining the conventional training phase, the actual warmup iterations were 0, accumulating 16 batches; no optimizer or EMA updates occurred in the two batches, and the entry point was rejected as required. This failed attempt is fully retained per the original contract and did not enter formal training.

Subsequently, 128 newly synthesized training images and 2 synthesized training_val images were pre-frozen and run for 32 batches under the same 640/batch=4/nbs=64/AMP-off initialization and weight settings, resulting in 2 actual SGD and 2 EMA updates. The true TAL loss simultaneously triggered positive anchor and background-ignored anchor, with the positive anchor classification preserved in full. Before saving, CUDA synchronization was performed, and the live FP32 EMA was captured; the state-only serialization, architecture strict reconstruction, and CPU/CUDA parsing of synthesized inputs and outputs were all precisely consistent; the live state, update count, and Torch CPU/GPU RNG remained unchanged.

An independent new process was launched to reload only the current FP32 state and reconstruct the model, with CPU and CUDA outputs matching the saved reference exactly. The validation scope is this single round, 32 batches, and zero warmup iterations—thus, it cannot be claimed to cover all paths of the formal 30-round training. The actual warmup and real sample execution are still governed by runtime events and per-round checkpoint verification records.

[CUDA synthesis execution result](../../docs/DATA_AND_ARTIFACTS.md); [Independent process state recalculation](../../docs/DATA_AND_ARTIFACTS.md)

# # Saving, Progress, and Acceptance Boundaries

Before each round, a live FP32 EMA is saved separately, and the source state, update count, implementation origin, backend settings, and CPU/CUDA reconstructed outputs are verified. The FP32_reference is saved round-by-round at `formal_attempt_01/fp32_ema/epoch_XXXX/state/reference_state.pt`, with success determined by the final submitted directory and acknowledgment; .pending directories that fail are not considered successful outputs.

The framework's standard best/last checkpoints are still saved as compatible outputs, but their half-precision EMA and placeholder fitness are not used for scientific candidate selection and are not recognized as fully validated recovery training mechanisms. Task AP was not computed during training; training_val in this round is a bound training-validation member and does not serve as an unannounced selection set. Subsequent evaluation must follow an independent evaluation contract before scientific baseline qualification is determined.

The process writes event-by-event, refreshing readable progress every 25 batches; after each round, the completed epoch count is updated. Any change in frozen batch size, OOM-triggered batch reduction, non-finite loss or state, source change, precision or backend drift causes this attempt to fail and retain the current state, without automatic restart or configuration change.

This round did not include Selection inference, Holdout content access, INT8, export, TensorRT, or Orin performance measurement. The original full FP16 guardrail remains at `FAIL_FIXED_TOLERANCES`, and the old Gate0.5 numerical record remains at `FAILED_UNCHANGED`; training initiation, successful saving, or final training completion does not equate to Gate1 scientific acceptance.
