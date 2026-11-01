# Gate1 Upstream Boundary Precision Comparison and Mitigation Path



This round completed a comparison of five boundaries and two cross-directional paths: **3 pilot cases → all 109 images**, along with saved array independent recalculation. The cross-path of stem FP32 and posterior FP16 still has 6 out of 109 images exceeding the threshold, and the fixed numerical threshold has not yet been resolved. This investigation has further narrowed the scope to input and stem sensitivity, **but a unique layer or optimal split has not been identified, and Gate1 scientific acceptance remains unmet**.

[Itemized Conclusion](../../docs/DATA_AND_ARTIFACTS.md); [Status Snapshot v7](../../docs/DATA_AND_ARTIFACTS.md); Previous round reference [Report 45](45_gate1_precision_error_diagnosis_and_evaluation_preparation.md). Training weights and original value failure records remain unchanged.

# # Actual Execution and Verification in This Round

First, verify the real checkpoint on CPU: 24 top-level modules, Detect receiver modules at 16/19/22; pre-frozen five split points at −1/4/6/10/22. For each split point, extract all cross-boundary DAG dependencies, preserving skip connections; for example, the boundary after module 10 must carry outputs from modules 4, 6, and 10 simultaneously, not just replace module 10's tensor.

The same frozen best weights, two independently loaded non-fused models, original preprocessing, and identical runtime control conditions were used. The raw outputs per image are byte-by-byte consistent with the previous round. The complete manual replay of 23 prefix modules and final logits, along with two posterior re-replays at the same precision, are precisely reproduced against native results on the runner. The state dictionary and captured features remain unchanged before and after. All 18 runner synthetic tests and 21 independent auditor synthetic tests passed.

Independent recalculation verified the saved native/historical outputs, prefix boundary tensors, receiver type conversion hashes, endpoint aliases, and all 12 aggregated metrics and fixed thresholds. It does not re-run convolutions or BN layers; the per-layer replay hashes and state dictionaries not separately saved remain unchanged as asserted by the reviewed runner. The two native aggregations and repeated control checks should not be treated as additional independent experiments.

[Complete Boundary Comparison](../../docs/DATA_AND_ARTIFACTS.md); [Independent Array Recalculation](../../docs/DATA_AND_ARTIFACTS.md); [True Topology Record](../../docs/DATA_AND_ARTIFACTS.md).

# # Results of Fixed Thresholds Across All 109 Images

Each cross-output is decoded using FP32 and compared against the native FP32 model across all candidate positions; the box threshold remains at `0.5 + 0.005×abs(FP32_reference)`, and the score threshold remains at `0.002 + 0.02×abs(FP32_reference)`. The table below lists results by direction, and cannot be reduced into causal contributions of any individual layer. The 'front' refers to the split point and before, and 'post' refers to after the split point; boundary values are converted to the dtype received by the posterior segment.

| Split Boundary | All Cross-Module Cases | Front 32 / Back 16 Out-of-Threshold Images | Bounding box element exceeds tolerance | Front 16 / Back 32 Out-of-Threshold Images | Bounding box element exceeds tolerance |
|---|---|---:|---:|---:|---:|
| Input Conversion (−1) | [-1] | 109/109 | 8992 | 75/109 | 274 |
| After Module 4 | [4] | 88/109 | 450 | 109/109 | 6193 |
| After Module 6 | [4, 6] | 79/109 | 340 | 109/109 | 7944 |
| Stem End (After Module 10) | [4, 6, 10] | 6/109 | 7 | 109/109 | 8694 |
| Feature Fusion End (After Module 22) | [16, 19, 22] | 0/109 | 0 | 109/109 | 8980 |

Native full-precision output still has FP16 images, 109 / 109 image(s), 10849 box element deviations; both native networks, after unified FP32 decoding, still have 109 / 109 images, 8992 box element deviations. The input boundary 'before 32 / after 16' exactly matches the previous existing control; module 22 boundary reproduces the previous two head crossover paths. These are internal validation endpoints and do not add independent evidence.

Separately normalizing the input and rounding it to FP16, then converting it back to FP32 before feeding it into a full FP32 network results in **75/109 images and 274 bounding box elements exceeding tolerance**, with a maximum bounding box difference of 5.055847 pixels. This indicates that input rounding alone can affect strict raw checks and cannot attribute all discrepancies solely to half-precision convolutions or BN layers.

The backbone32/rear16 path has a maximum box difference of **3.344391 pixels** and 0 out-of-tolerance score elements. It still has 6 failing images and 7 failing box elements, so rear feature-fusion computation or boundary conversion cannot be fully excluded. Among the limited split points tested, `[22]` is the front32/rear16 split that falls within the original tolerances. This does not establish any split as minimal, optimal, or sufficient for actual task accuracy.

![Fixed thresholds comparison across boundaries](../../docs/DATA_AND_ARTIFACTS.md)

The left chart uses a symmetric logarithmic axis with zero preserved, while the right chart directly displays the number of images with tolerance violations; each bar represents a different cross-path, not a summable error decomposition. The chart is derived solely from the recalculated 109 debug images.

# # What can currently be said and what cannot

What can be said: There exists a reproducible low-error boundary replay path, extending the FP32 front-end significantly alleviates strict numerical checks in the tested combinations; subsequent investigations should focus on inputs and the main stem, using small-scale preregistered comparisons to distinguish early versus late impacts. It cannot be asserted that any single operator is the sole cause, nor can the count differences in modules 4, 6, and 10 be treated as independent, additive contributions.

The cross-dependencies preserved in the native prefix features described above are **not yet independently implemented and verified inference programs**. This round does not include new NMS, per-object scoring, or official AP, nor any deployment, performance, or power consumption testing. It cannot be claimed that half-precision operation has passed, that INT8 is effective, that the model is optimal, or that Orin/TensorRT has been validated.

The original Gate1 full FP16 numerical checks remain **FAIL_FIXED_TOLERANCES**; the original Gate0.5 remains **FAILED_UNCHANGED**. The reference remains the FP16 EMA boosted to FP32; the full FP32 training EMA cannot be restored. The 109 images are reused engineering debug results, with Selection not detected or ranked, Holdout ZIP not listed, unpacked, read, or inferred.

# # Additional environment details and next steps

Additional limited search conducted in the existing MATLAB/Octave runtime environment: no available runtime found in standard registrations or listed installation directories; some registrations refuse read access, which does not allow us to assert that no runtime is installed on the system. No software was downloaded or installed; the official 20 synthetic test cases have not been natively executed; AP equivalence has not been certified. [Search scope and retained errors](../../docs/DATA_AND_ARTIFACTS.md)

If narrowing the operator scope continues, only a small number of input or main stem boundary points, or fixed local input controls, should be frozen; if advancing low-error paths, an independent inference program must be separately implemented and the outputs, target matching, and engineering execution must be re-verified before any discussion of deployment. Scientific baselines require official evaluation confirmation, reference precision, and review of ignore loss and data independence; method ranking can only proceed after the Selection study budget and statistical plan are frozen, and Holdout remains sealed.
