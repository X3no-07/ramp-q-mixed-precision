# Gate1 Training Pre-Components and Official Native Scoring Validation



**The native reference output of the official scoring source code has been obtained: 20 synthetic test cases successfully executed on GNU Octave 11.3.0, with 0 failures and independent output audit passed.** The two pre-components—ignoring negative loss and FP32 save format—have also been completed. However, they have not yet been integrated into actual training, and the Gate1 scientific baseline remains unverified.

[Itemized Conclusion](../../docs/DATA_AND_ARTIFACTS.md); [Status Snapshot v9](../../docs/DATA_AND_ARTIFACTS.md); [Previous Round 109-Image Mixed Prototype Report](47_gate1_independent_mixed_precision_inference_prototype.md).

# # Official Native Reference Has Been Acquired

This round, the Octave 11.3.0 portable package was obtained from official released assets and verified against release SHA-256; no PGP signature trust chain was claimed to be validated. After spatial inspection, only 2890 command-line essential files and 640420084 bytes were extracted, with no full installation, execution of post-install, or modification of the system PATH/ registry. mean2 used the same archived original image-2.20.0 function; the original official 11 source code and the old synthetic runner were preserved with byte-level identity.

Inputs are always the frozen 20 synthetic matrices: 11 groups of AP/AR aggregation, 4 groups of pixel-level filtering, and 5 single-image matching. The execution paths of six official functions are individually verified against the original files; the actual program calls the original dropObjectsInIgr, evalRes, calcAccuracy, and their dependencies; no alternative functions or hand-coded expected AP answers are used. MAT and JSON files, runtime versions, dependency lists, and console logs have all been preserved.

Independent audit verification of 2890 runtime members, 1535 isolated MAT file read dependencies, 20 use case sequence/state, empty matrix dimensions, matching matrix, and sequence consistency across seven metrics. 39 auditors passed composite testing. A03/A04 empty evaluation categories retain native NaN values in MAT files; JSON nulls cannot be interpreted as 0 points. This establishes a reference based on fixed Octave version execution of official source code, and does not prove equivalence across all MATLAB versions or real-world data.

[Native execution receipt](../../docs/DATA_AND_ARTIFACTS.md); [Native MAT results](../../docs/DATA_AND_ARTIFACTS.md); [Independent output audit](../../docs/DATA_AND_ARTIFACTS.md).

# # Python semantic comparison

First freeze all test cases, fields, row order, and zero absolute/relative tolerances, then compare the existing Python ignore filters with official-style single-image matching: the 20-case test cases' 22 synthesized image filtering/score reversal matrix, as well as the 5-group matching × 10-category GT/DT matrix; **all 166 intermediate matrices are strictly consistent**. The adapter passes 27 synthesized test cases, and both existing Python37 regression cases pass.

This round does not invoke or modify any project-defined IOU-priority matching logic. The matrix consistency only covers these fixed synthetic inputs and does not constitute a proof for all inputs. The complete Python AP/AR aggregation has not yet been implemented, and there is no comparison between Python and native AP values. Details such as category-by-image weighting, input order truncation, score=0 denominator, cross-image sorting, and VOC precision envelope aggregation remain to be fully implemented and verified.

[Pre-frozen comparison conditions](../../docs/DATA_AND_ARTIFACTS.md); [Actual local comparison results](../../docs/DATA_AND_ARTIFACTS.md).

# # Two training-preface components

**Ignore negative sample classification loss primitive: 55 CPU synthetic tests pass.** For enhanced images, when an anchor center falls within an ignore box and is not assigned as positive, the entire classification BCE row is masked; all class items at positive positions are preserved, and the target_scores summation denominator remains unchanged. When no ignore exists, values and gradients are bitwise consistent; gradients of logits for ignored negatives are zeroed out, while gradients for other logits remain unchanged under fixed inputs. Gradient aggregation across multiple anchor positions in shared model parameters still changes, and thus cannot be used to claim parameter update invariance or improved training performance.

Anchor center, half-open boundaries, class-agnostic are explicit engineering policies, not official AP rules. The component requires ignore geometry to be synchronized with all augmentations; Mosaic and sidecar batching, assigner/criterion/trainer have not yet been integrated, and thus cannot be considered a complete ignore-aware training solution. The current loss/tal source code supports gap detection, but the prior training contract did not separately fix these two files, making it impossible to reverse-engineer their historical exact identities; the fixed dataset/augmentation/tasks are consistent with the current snapshot.

**FP32_reference state preservation format: 10 CPU synthetic tests pass.** The original FP32 values, which cannot be precisely represented by FP16, are preserved bit-by-bit after saving. The component rejects automatic promotion from half-precision sources. It does not recognize prior calls that first half-then float, nor does it validate the declared training source; the precision lost in previous best/last half-precision EMA is not recoverable. There is no live EMA atomic capture, architecture reconstruction, or real-time training hook. The current implementation is not a scientifically accepted reference, nor a checkpoint with optimizer state recovery.

[Loss component and source review](../../docs/DATA_AND_ARTIFACTS.md); [FP32 format validation](../../docs/DATA_AND_ARTIFACTS.md); [Independent review of both components](../../docs/DATA_AND_ARTIFACTS.md).

Initial temporary directory ACLs, MAT test configurations, and download timeout/incomplete attempts are preserved in original logs; final testing uses a clearly defined engineering temporary directory, fully archived after resume transfer and verified via official SHA. Failed attempts are not deleted or counted as passes.

# # Pending acceptance tasks

The next steps are to integrate and test augmentation geometry, batching, training loss, and to capture live FP32 EMA before saving with atomic capture and source verification. Then, complete AP/AR aggregation comparison and independent validation contracts must be added. This round does not include model inference or retraining, nor does it involve reading real images/data annotations, Selection detection results, or Holdout content. No export, INT8, TensorRT, Orin, speed, or power consumption measurements have been performed.

109 synthetic image mixed prototype results have been maintained as passed; the original full FP16 threshold remains at `FAIL_FIXED_TOLERANCES`, and the previous Gate0.5 numerical record remains at `FAILED_UNCHANGED`. Successful synthetic execution of the official source code and passage of the two components do not substitute for scientific baseline, real dataset AP, or board-end deployment acceptance.
