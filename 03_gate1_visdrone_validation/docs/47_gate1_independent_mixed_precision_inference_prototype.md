# Gate1 Mixed Precision Independent Inference Prototype Validation



**The prototype with FP32 main stem, FP16 detection tower, and FP32 decoding has been successfully executed; all 109/109 images meet the original fixed numerical thresholds, and independent array recalculations have passed.** The logits and decoding outputs for each image are byte-by-byte consistent with the previously diagnosed cross-path endpoints of the detection heads. This validation confirms an independently computable engineering path; the original full FP16 failure records remain intact, and Gate1 scientific acceptance has not yet been achieved.

[Round-wise sub-conclusion](../../docs/DATA_AND_ARTIFACTS.md); [Status snapshot v8](../../docs/DATA_AND_ARTIFACTS.md); [Prior five-boundary diagnostic report 46](46_gate1_upstream_precision_boundary_and_mitigation.md).

# # What has actually been accomplished

The new prototype performs a single CPU deserialization of the frozen best weights per attempt, and executes a new forward pass from image to output for each image. Modules 0 to 22, including the backbone and subsequent feature fusion, use FP32 for parameters, floating-point buffers, and observed top-level outputs; the two convolution towers (cv2/cv3) in Detect use FP16 for parameters, floating-point buffers, and logits; DFL, anchor/stride geometry, and sigmoid observation tensors are FP32. No fusion is called, and no claim is made about the internal accumulation precision used in CUDA convolutions or BN kernels.

Inputs come from re-reading the same batch of 109 frozen engineering debug images, not from saved features. Before execution, only the integrity hash of historical files is checked; after each new forward pass, the old numerical arrays are loaded for comparison. The newly normalized input, three-scale FP32 features, and their FP16 conversion all precisely match the frozen reference, with no change in model state.

The source code and precision agreement are first frozen, all CPU tests conducted by 12 runner instances and 18 auditor instances pass; independent recalculation is performed on a pre-defined set of 3 pilot images and completed, then extended to the full original sequence of 109 images. The extension condition is that identity, implementation, and recalculation remain consistent, with no favorable numerical counting used to select images. The 3 pilot images are included within 109 images, and cannot be summed to form 112 independent samples.

[Frozen precision contract](../../docs/DATA_AND_ARTIFACTS.md); [Prototype source code](../scripts/probe_gate1_body32_head16.py); [Full execution receipt](../../docs/DATA_AND_ARTIFACTS.md)

# # Actual Results with Fixed Thresholds

Compare all 8400 candidate locations' 4 box scores and 10 class scores: box tolerance maintained at `0.5 + 0.005×abs(FP32_reference)`, score tolerance maintained at `0.002 + 0.02×abs(FP32_reference)`. No candidate filtering based on confidence, and no threshold relaxation.

| Check | Passed /109 | Bounding box element exceeds tolerance | Scoring element exceedance | Maximum absolute bounding box difference |
|---|---:|---:|---:|---:|
| Native full FP32 vs. full FP16 (historical original check) | 0 | 10849 | 0 | 10.439514 px |
| New direct-read image mixed prototype vs. historical FP32_reference | 109 | 0 | 0 | 0.263733 px |

The new prototype checked 3662400 bounding box elements and 9156000 score elements; the maximum absolute score difference was 0.004091561. The maximum element-wise "error/tolerance" ratios were 0.177952 for bounding boxes and 0.480744 for scores.

Engineering inspection is separately named `GATE1_BODY_FP32_HEAD_TOWERS_FP16_DECODE_FP32_DEBUG_NUMERIC_CHECK`. The previous round with only mainstem FP32, posterior fusion, and detection tower FP16 still had 6 images and 7 box elements exceeding tolerance; in this prototype, the module after stage 22 is retained in FP32, representing the low-error endpoint of that module, and no correction or re-evaluation of the path at module 10 is performed.

An independent auditor verified the identity of saved arrays, inputs, reference features and transformation statistics, endpoint byte counts, precision observation reports, fixed tolerances, and summaries. A NumPy FP64 approximation is used for plausibility checking of decoding, but it is not treated as CUDA byte-by-byte equivalence. It does not re-execute convolutions, BN layers, or image decoding. Newly computed features are only recorded via hash and statistics for deduplication; the independent audit compares against the saved reference features; actual fresh computations originate from the reviewed runner.

[Full prototype results](../../docs/DATA_AND_ARTIFACTS.md); [Independent saved output recalculation](../../docs/DATA_AND_ARTIFACTS.md).

# # Scope of Conclusion and Next Thresholds

This round has implemented known low-error cross-point endpoints as direct-read image inference modules, enabling consistency and numerical qualification for 109 debug reuse cases. It reproduces existing endpoints without adding independent generalization evidence, nor does it prove the minimal number of FP32 layers or optimal precision configuration. The validation runner still relies on historical references for acceptance checks and is not yet a standalone deployable offline package.

Frozen weights were originally saved in FP16 EMA; promoting the primary computation type does not restore the original FP32 training state. The complete ignore-negative semantics of training loss, independent data contracts, and scientific references remain unresolved. Official AP has not been implemented or verified; limited position search has not found available native MATLAB/Octave runtimes, and the synthetic fixture has not yet completed native oracle comparison. Selection remains only metadata, and Holdout remains an opaque sealed package.

This round does not introduce any new NMS, target matching, or task AP, nor does it include INT8, export, TensorRT, Orin inference, speed, memory, or power consumption measurements. The next steps should first complete scientific references, loss semantics, and official evaluation definitions, then pre-freeze independent validation and candidate budgets; if advancing this prototype, a separate backend/board qualification contract should be established. The original full FP16 check remains `FAIL_FIXED_TOLERANCES`, and the previous Gate0.5 numerical record remains `FAILED_UNCHANGED`.
