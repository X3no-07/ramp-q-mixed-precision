# Gate1 AP aggregation native comparison and ignore geometry validation



**Complete Python AP/AR aggregation has been implemented and passed against native reference using fixed 32 synthetic cases: 220/220 intermediate matrices are zero-tolerance consistent, 161 final metrics meet the pre-frozen double-precision tolerance, and 28 NaN patterns are completely identical.** The restricted geometry for ignore regions and batching components have also been completed, with 85 CPU synthetic tests passing. The real training pipeline and live FP32_reference saving have not yet been integrated, and thus Gate1 scientific acceptance has not been achieved.

[Itemized conclusions](../../docs/DATA_AND_ARTIFACTS.md); [Status snapshot v10](../../docs/DATA_AND_ARTIFACTS.md); [Previous report 48](48_gate1_training_prerequisites_and_native_official_scoring.md).

# # Complete aggregation and native reference

The new module corresponds segment-by-segment to the official commit's calcAccuracy, VOCap, evalRes, compOas, with no modifications to the original source code. The original behavior is followed: input categories are first truncated in full category submission order, then filtered by category and sorted by score; score=0 ignored GT is retained in the recall denominator, ignored detections are neither TP nor FP, categories are weighted by frequency of appearance across images, AP uses maxDets=500 and integrates the VOC precision envelope. No alternative COCO-style metric has been adopted, and no NMS is executed.

Implements a declared finite numerical matrix domain with stable ordering both within and across graphs. Native IoU thresholds of 0.5:0.05:0.95, comprising ten values, have undergone pure constant probing, matching Python formulas bit-for-bit in float64 hexadecimal representation. This implementation is fixed to a specific Octave version and is not generalizable to all MATLAB versions. The source code and 54 synthetic tests are first frozen, then independently evaluated by a comparator against native reference answers. The comparator passes 45 negative example tests.

The original 20 group reused the native outputs preserved from the previous round; in this round, the 12 group was pre-fixed with fixed pressure inputs, and the native execution was performed using the official functions unmodified. 12 / 12 succeeded, while 0 cases resulted in errors. The pressure inputs included more than 500 predictions, reversed input order, cross-category truncation, identical splits and equal IoU, score = 0 repeated absorption, uneven category distribution in the category map, empty images, category11, overlapping ignored regions, out-of-bound coordinates, and score recall endpoints. No cases were selected based on runtime results, and no hand-written AP answers were used to substitute native outputs.

[Complete aggregation source code](../scripts/gate1_official_ap.py); [Frozen implementation and 54 test cases](../../docs/DATA_AND_ARTIFACTS.md); [New native execution reports](../../docs/DATA_AND_ARTIFACTS.md).

# # Actual comparison results

Matrix dimensions, row order, and all elements are compared using zero absolute or relative tolerance. The final seven AP/AR metrics are float64 scalars in percentage units, pre-fixed at `abs(Python-native) <= 1e-10 + 1e-12×abs(native)`, allowing only rounding differences between double-precision reduction paths; NaN positions must be identical, and infinite values are rejected. This scalar tolerance is separate from the original model's FP32/FP16 numerical guardrails, which remain unchanged.

| Reference set | Exact matrix count | Final metric count | NaN metric count | Maximum finite difference (in basis points) |
|---|---:|---:|---:|---:|
| Preserved reference from the previous 20 groups | 166 | 77 | 14 | 0 |
| Newly added 12 pressure cases | 54 | 84 | 14 | 7.10543e-15 |

A total of 32 groups and 40 synthesized images, of which 23 groups generate seven final metrics. All 133 finite metrics passed, with a maximum deviation of 7.105427357601002e-15 percentage points; 28 NaN values are preserved as-is and cannot be interpreted as 0 points. The previous 20 group overlaps with the prior localized inspection and thus cannot be considered as new independent data evidence.

Independent cross-checking verified the original official 11 source code files, 2890 runtime files, and MAT-read dependent 1535 real files (wheel has 1656 entries, including directories), startup and output identities, MAT/JSON serialization, empty matrices, and NaN values. The native wrapper only exports designated intermediate matrices and the final seven metrics; the Python internal 10×10 AP and 10×10×4 AR tables are not independently output or directly validated in the native version, and no claim is made that each individual cell has been directly verified.

[Independent comparison summary](../../docs/DATA_AND_ARTIFACTS.md); [Per-case and per-metric results](../../docs/DATA_AND_ARTIFACTS.md); [Comparison policy](../../docs/DATA_AND_ARTIFACTS.md).

# # Ignored geometric and batch components

The new components support only explicitly constrained chains: the two-axis scaling ratios from original image to loader output, the scaling ratio and integer padding from the same LetterBox sampling, and the same horizontal/vertical flip decision; finally, images and sidecar identities and orders are concatenated into ragged regions and integrated into the synthetic chain test for prior classification loss primitives. No random parameters were sampled, and no residual ignored regions were removed based on bounding box area or aspect ratio rules.

85 CPU tests include 10 instances of installed LetterBox comparisons, 4 instances of RandomFlip comparisons, 3 instances of a synthetic array replacing imread in the full loader chain, as well as boundary and positive-priority checks for batching two dtypes into the loss. No real images or data annotations are loaded. The old augmentation probabilities are retained at 0, which may still trigger paths such as RandomPerspective; thus, the constrained chain is a new contract proposal, not the old 30 epoch training that has already been fixed.

scores=0, category0, and category11 are separately preserved; positive boxes subject to quality exclusion are not silently converted to publisher-ignore. Instances with zero area, completely outside the canvas, invalid sources, or precision transformation leading to loss of positive area are rejected at the sample level, requiring separate quarantine and data contracts. The compatibility of all 4534 training images has not been verified. Structural checks on identity strings and SHA fields do not confirm actual file hashes or split ownership, and future loaders will still need to validate these.

Verification of real Dataset, Format, Trainer, criterion, and disabled augmentation runs has not yet been integrated. Passing of components does not equate to complete ignore-aware training completion, nor does it prove any training benefit.

[Geometric component description](../../docs/DATA_AND_ARTIFACTS.md); [85-test execution report](../../docs/DATA_AND_ARTIFACTS.md); [Independent source audit](../../docs/DATA_AND_ARTIFACTS.md)

# # Current Acceptance Boundary

A complete aggregation implementation and native consistency evidence for synthetic use cases have been established. However, consistency across arbitrary inputs, across versions, or with real-world data has not been proven. The next step is to freeze and integrate the real training pipeline, illegal ignore handling, live FP32 EMA atomic saving, and architecture reconstruction, followed by qualification checks based on independent data, submission order, and candidate budget contracts. The precision loss from the previous FP16 EMA cannot be recovered.

No retraining or execution of detection models was performed in this round. There were no Selection detection scores, no access to Holdout data, no export, INT8 quantization, TensorRT, Orin platform, speed, or power consumption measurements. The existing 109-image mixed prototype record is retained. The original full FP16 guardrail remains `FAIL_FIXED_TOLERANCES`, and the old Gate0.5 record remains `FAILED_UNCHANGED`.
