# Gate1 Precision Error Stratification Diagnosis and Evaluation Preparation



This round has completed two sets of **3-pilot → full 109-debug**real precision comparisons, and independently verified the saved arrays. A new finding is clear:**Raising only the decoding to FP32 does not resolve the errors; differences in feature sources prior to the detection head can reproduce the errors**. When the feature source is fixed and only the head computation precision is switched, both sets of comparisons remain within original tolerance. When the head precision is fixed and the feature source is changed, errors are reproduced. The original numerical thresholds and failure results remain unchanged—**Gate1 scientific acceptance has not been passed**.

[Itemized Conclusions and Evidence Identity](../../docs/DATA_AND_ARTIFACTS.md); [New Status Snapshot](../../docs/DATA_AND_ARTIFACTS.md); 30 training rounds with unchanged weights, continuing from [Report 44](44_gate1_initial_training_and_float_qualification_results.md).

# # Execution Scope and Reproducibility Constraints

Continuing with the 29th round best (SHA `cddd14b3ebba294e4c3200bfb50fc1db1a9459c425d8e51d583c0881d71cfcfa`), two independently loaded unfused models, fixed 640 input size, and original 109 debug images. The pilot images are two post-hoc selected high-error examples and one fixed normal control. Expansion to 109 images depends solely on precise reproducibility constraints, not on favorable outcomes. The protocol is frozen before any new intervention is executed.

Both full comparisons reproduce the original FP32/FP16 raw bytes and normalized inputs image-by-image. Native DFL and geometric decoding, as well as native detection head logits reconstruction, are fully consistent. 499 saved state values are equal when promoted to FP32; the head sublayers maintain evaluation mode, intervention-unchanged states, and feature consistency. Measured `head.reg_max=16`, regression logits are `[1,64,8400]`, three-scale head inputs are `[1,64,80,80]`, `[1,128,40,40]`, and `[1,256,20,20]`. This round completes the actual runtime recording of DFL dimensions, previously only expected in source code.

Check that the asymmetric fixed threshold is still applied to all candidate locations: bounding box `0.5 + 0.005×abs(reference)`, score `0.002 + 0.02×abs(reference)`; coordinates are in 640 input pixels. The reference uses saved FP16 EMA weights boosted to FP32, and does not recover the full pre-saved FP32 training EMA. Decoding and head comparison programs pass respectively 15 and 11 CPU synthesis tests; two independent recalculators each pass 22 and 15 synthesis tests; synthesis tests and real 109 image evidence are separately saved.

# # First Layer: Network Logits Source × Decoding Precision

| Comparison: Reference → Comparison | Number of Error Images | Bounding box element exceeds tolerance | Maximum Absolute Bounding Box Difference (pixels) |
|---|---:|---:|---:|
| Original FP32 Full Model → Original FP16 Full Model | 109/109 | 10849 | 10.439514 |
| Original FP32 → FP16 Network Logits, FP32 Decoding | 109/109 | 8992 | 10.375580 |
| Same as FP32 source logits, decode 32 → 16 | 0/109 | 0 | 0.767090 |
| Same as FP16 source logits, decode 32 → 16 | 0/109 | 0 | 0.748291 |

All 4 scoring elements are within tolerance. The source logits decoding 32/16 'within threshold' does not imply bit-wise equality or that decoding has no impact. Both sources remain FP32 decoded with still 109 image-level deviations and 8992 bounding box element deviations, indicating that pre-decoding differences cannot be eliminated solely by improving decoding precision.

The original outlier set and the post-FP32 decoding intervention set share 7808 elements; the original set has 3041 unique elements, and the intervention introduces 1184 new outliers. Thus, `10849−8992` can only be interpreted as a net change in count and cannot be regarded as the error proportion caused by decoding or as an independently additive causal contribution. The three predefined worst-case logits, probabilities, and DFL distance chains are available in [saved array tracking](../../docs/DATA_AND_ARTIFACTS.md).

[Complete decoding comparison](../../docs/DATA_AND_ARTIFACTS.md); [Independent recalculation of decoding arrays](../../docs/DATA_AND_ARTIFACTS.md)

# # Layer Two: Feature Source × Detection Head Precision

Capture three-scale head inputs from two native forward passes, cross-identify feature sources with detection head precision, and **uniformly decode all intervention outputs to FP32**. The receiving end explicitly converts feature dtype across precision levels; the head includes its convolutions, BN layers, and activations, and cannot be partitioned by individual operators. The two `reference_fp32_vs_*` names are merely aliases of existing comparisons and do not add independent evidential weight.

| Comparison: Reference → Comparison | Number of Error Images | Bounding box element exceeds tolerance | Maximum Absolute Bounding Box Difference (pixels) | Scoring element exceedance |
|---|---:|---:|---:|---:|
| Fix FP32 source features, head 32 → 16 | 0/109 | 0 | 0.263733 | 0 |
| Fix FP16 source features, head 32 → 16 | 0/109 | 0 | 0.238525 | 0 |
| Fix FP32 head, feature source 32 → 16 | 109/109 | 8980 | 10.370834 | 0 |
| Fix FP16 head, feature source 32 → 16 | 109/109 | 8989 | 10.350204 | 0 |

This supports narrowing the investigation scope to the network computations and input conversions prior to the detection head; it still cannot identify a specific component such as the backbone, neck, a particular BN layer, or a specific convolution or input rounding as the sole cause, nor can it sum the two intervention metrics. A single intervention showing within tolerance on 109 images is merely a diagnostic result from reused debugging, and **does not represent a new deployed model, a formal floating-point baseline, or a scientifically validated method**.

[Complete head comparison](../../docs/DATA_AND_ARTIFACTS.md); [Independent recalculation of head arrays](../../docs/DATA_AND_ARTIFACTS.md). The auditor verifies array identity, fixed thresholds, native and historical byte counts, and decoding stages; non-linear operations are only checked via NumPy approximation, without independently rerunning convolutions or BN layers. The head state remains unchanged based on assertions from a previously reviewed runner.

# # Official evaluation preparation has also been advanced

Verified 11 official source code files at a fixed local commit `005445782213e20cb91bc50a597db3dd949e749a`; all 37 existing synthetic semantic regressions passed. Added 20 new purely synthetic inputs (11 aggregated, 4 ignoring regions, 5 matching) with identity-bound MATLAB/Octave entry points; future calls will directly invoke unmodified official functions to obtain outputs, without manually filling so-called official expected metrics.

No native runtime found on the local PATH and common directories; actual execution result is **NOT_EXECUTED_NO_NATIVE_RUNTIME_ON_PATH**. Thus, the official oracle has not been run, and the complete Python API has not been implemented, rendering API equivalence unverified. The prepared entry points do not read real images or real annotations or Holdout data; the runtime installation and download were not performed in this round.

[Official evaluation readiness statement](../../docs/DATA_AND_ARTIFACTS.md); [Actual execution result](../../docs/DATA_AND_ARTIFACTS.md)

# # Retained boundaries and next steps

The original Gate1 strict check remains **FAIL_FIXED_TOLERANCES**, and the original Gate0.5 is **FAILED_UNCHANGED**. The 109 images are reused engineering debug results, without new NMS or target scoring; 548 Selection images were not detected or ranked, Holdout ZIP files lack members, unpacking, image reading, or inference. No new quantization matrices, TensorRT/Orin precision, or performance conclusions have been established.

The next step is to separately freeze a small number of upstream module boundary probes to further narrow the input/neck/stem scope; it is not possible to directly select precision modules or perform infinite scanning based on post-hoc results from this round. Parallel evaluation work must first execute the frozen synthetic cases using the native runtime, then verify the complete AP path. The reference weight precision, ignore loss, and data independence constraints must still be addressed before scientific baseline acceptance and selection study budget freezing can be finalized.
