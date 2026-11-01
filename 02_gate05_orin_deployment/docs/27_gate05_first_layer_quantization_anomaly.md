# Gate 0.5: First-layer Quantization Anomaly Localization



On 2026-10-19, following six frozen engine engineering checks, **a numerical anomaly in the first layer has been confirmed in small graphs on Orin / TensorRT 8.5.2.2, involving Clip→Q/DQ operations. Input routing around the fixed input has been validated, and a candidate first-layer fix preserving semantic truncation has been identified. The full detector remains uncorrected, and Gate 0.5 has not yet been passed**.

Original failures, quantization parameters, weights, full model, detection thresholds, and numerical guardrails are all preserved. No new full detector runs, 150-sample evaluation, re-calibration, or performance testing have been performed.

# # Direct Observations

Using the same float32 preprocessing array from the original `000020.jpg`, the first activation quantizer originally applies floating-point Clip[-1,+1]→QuantizeLinear→DequantizeLinear, with scale `0.007874015718698502` and zero 0. The actual input range is approximately [0.117647,1], so this Clip is bytewise identical to its input for this array.

The original graph outputs from CPU ORT and the paired graph without this Clip are element-wise identical; the correct quantized values have 110 distinct values, ranging from 15 to 127.

However, the board-level original graph transforms all 1,228,800 input values into a single dequantized constant `0.007874015718698502`. After removing this identity Clip, the board-level output matches the CPU output element-wise. The original inspector shows FP32→INT8 Reformat, followed by an INT8 PointWise Clip bounded by ±1 real values. This numerical behavior aligns with the concept of 'original floating-point boundaries being used to define integer code domains'; it is not inferred solely from layer names or formats.

The same result is observed in the first-layer convolution. The following are based on the same fixed input, original weights, original scale/zero, original Conv and SiLU layers; the two outputs represent activation before and after SiLU.

| Board-level small graph | Absolute maximum relative difference before activation vs. CPU | Absolute maximum relative difference after SiLU vs. CPU |
|---|---:|---:|
| Floating-point graph, FP32, TF32 disabled | 0, element-wise identical | 0.0000104904 |
| Floating-point graph, FP16, TF32 disabled | 0.0310437 | 0.0307062 |
| Original QDQ, enable FP16 | 40.1015298 | 36.6749301 |
| Original QDQ, disable FP16 | 40.1015298 | 36.6749301 |
| Retrain paired original QDQ Clip | 40.1015298 | 36.6749301 |
| Pair only removes the first layer identity Clip | 0.00000762939 | 0.0000143051 |

Bitwise consistency between two QDQ variants with FP16 enabled and disabled. Retraining the original Clip also matches the bitwise output of the initial erroneous run. Each small image is executed twice, producing bitwise identical outputs, all finite values.

The actual convolution weights and inputs are INT8, outputs are FP32, using the same `ampere_fp32_icudnn_int8x4_128x32_relu_small_nn_v1` tactic; no floating-point convolution is used to restore numerical values. Weights, biases, scales, zeros, and Conv/SiLU nodes are all preserved. The clip version without the node only removes that node and two unused boundary constants, and reconnects the ActivationQ input to `images`; reversing these modifications fully restores the original ModelProto in bytes.

We also attempted to preserve the original ONNX and request the Clip's computation and output to be FP32 via `precisionConstraints=obey`. This request did not prevent the actual engine from using the above INT8 Clip path, and the output remained the same erroneous constant. Thus, this configuration cannot be considered a validated fix.

# # Numerical Mechanism Reproduction

Independent CPU analysis first generates correct activation integer codes with frozen parameters, then incorrectly clips them to [-1, +1]. Since all these codes exceed 1, all codes become 1.

Integer multiply-accumulation with the original per-channel quantized weights, followed by the FP32 scale product and a single-rounding multiply-add model for the bias, reproduces **all 1,638,400 pre-activation convolution outputs bit-exactly against the raw native array, with maximum difference 0**. Ordinary dequantized floating-point convolution differs by approximately 1.19e-6. The correct CPU QDQ output differs from the anomalous native output by 40.10.

This explains the isolated anomaly at the first layer. It does not imply reading GPU-packed scales or machine instructions, nor does it prove the sole root cause of all anomalies across five full engines. The original full q_only_G1 engine also exhibits the same first two Reformat / INT8 Clip records, but the full engine subsequently uses fused SWISH and INT8 output; the additional activation pre-output tap in this round alters this fusion.

CPU analysis and predicted array: `../artifacts/gate05/20261019_firstconv_cpu_hypothesis_check`. The analysis script, original predictions, and the SHA hashes of 10 source files are preserved; subsequently, the native original array SHA was cross-verified with the full back-transmitted receipt.

# # Original Evidence and Independent Verification

Four wheels collectively constructed and executed 9 isolated small tiles. All calculations were performed independently from the saved original bytes: file listings, hash chains of images/inputs/expected outputs/scripts/engine, name bindings, actual convolution formats, boundedness of original arrays, duplicate outputs, and numerical discrepancies reported in the documentation. Exact receipt validation was completed for 188 original files.

| Original package | Number of files | ZIP SHA256 |
|---|---:|---|
| First layer precision comparisons | 53 | `ac725c4340eecd62674d4429384fb5816839a7b8590adea9c6a6ea937144bc12` |
| Activation Clip / no_clip pairing | 46 | `18523854ddbcd93c133ba0078295bcb2f9b3195823944e6a52a18eadeee90774` |
| Original image Clip with FP32 constraint attempt | 42 | `dd55328128403920867394b0e1fbc3c09b0d09eee96afc3e8dd44cb8feca27ff` |
| First layer convolution Clip / no_clip pairing | 47 | `65fb05c95375e39c3e6558f373b582721b258dff90dac38a7d7eeb09096094d9` |

The original directory sequence is as follows:

- `../artifacts/gate05/20261019_firstconv_numeric_native_evidence`
- `../artifacts/gate05/20261019_first_activation_boundary_native_evidence`
- `../artifacts/gate05/20261019_activation_fp32_clip_constraint_native_evidence`
- `../artifacts/gate05/20261019_firstconv_clip_intervention_native_evidence`

Independent root checks are `../artifacts/gate05/20261019_isolated_numeric_root_audit/all_four_probe_audit.json`, SHA256 `70146f1211db4017fb8b7a54f0c88a36f17e3bb3c12855881dcc338a0dcabb25`. Minor FP64 reduction statistics at the board end show slight discrepancies with the local NumPy/libm, which are recorded separately as rounding differences in statistical computation; the maximum difference, original arrays, bitexact matching, and finite value states are strictly verified. No adjustments to any model numerical guardrails are made based on these differences.

All build sequences use the original temperature, memory, kernel log, and time monitoring. The four rounds of monitoring lasted 350.35, 71.31, 63.27, and 260.02 seconds respectively; the highest temperature reached 51.781°C, with minimum available memory approximately 1837 MiB, and no new kernel security labels are introduced. All build, inference, and monitoring processes for this task eventually exit, with fans active, jtop active and NRestarts=0, and about 5 GiB of available memory. The startup number remains consistently `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. The board clock is still behind the client clock; original timestamps and startup numbers are preserved separately, and no timestamp manipulation is performed to align them.

# # Preservation of Truncated Semantic Fix Candidates

Clip removal was not extended to all activations. Instead, a paired rewrite replaced floating-point Clip→Q→DQ with Q→integer-code Clip[-127,+127]→DQ while retaining the original scale/zero. For each of the 88 frozen activation scales, 1813 finite synthetic values were constructed, covering positive and negative values, saturation, integer/half-integer thresholds, and adjacent float32 ULPs. The two graphs agree elementwise under ORT_DISABLE_ALL, with a maximum difference of 0. This is a test of those boundaries, not an exhaustive proof over all float32 values or the complete model. The record is `../artifacts/gate05/20261019_all_code_clip_scale_equivalence/analysis.json`.

When submitted directly to local TensorRT 8.5.2.2, the INT8 bound constants were not correctly interpreted as ±127. The inspector reported bounds of approximately 1.8e-43, and every output of the small graph became 0. An ONNX graph rewrite alone is therefore not yet a usable fix. This failed version and its original outputs are retained.

Then, using a valid, identical ONNX graph, the named Clip tensor's TensorRT activation alpha/beta parameters were explicitly set back to the original INT8 constant values of -127 and +127; all other parameters remain unchanged. This is a clearly documented backend import correction, with no modifications made to the source ONNX graph.

- For the 1811 positive and negative synthetic samples of the first scale, the output range is restored to [-1, +1]; 1801 samples match the CPU results, 10 samples near thresholds differ by one quantization step, with a maximum absolute difference of approximately 0.0078740716. No relaxation of the criterion or labeling of full numerical pass is made.
- The same rewrite and import correction are applied to the first layer Conv+SiLU of a fixed image; the maximum difference before activation is 7.62939e-6, and after activation is 1.43051e-5. Both the main and auxiliary original arrays match bit-for-bit with the prior no_clip fixed input reference. The actual convolution weights and inputs remain INT8.
- The v1 first-layer code_clip package was an unuploaded snapshot; v2 cleaned up inherited CPU references and the 'all node bytes unchanged' field before actually uploading and executing. The original weights, biases, scales/zero, Conv/SiLU, and output declarations are all preserved.

The three additional rounds introduced 4 small graphs; including the initial diagnostic rounds, there are 13 native small graphs. Exact receipts were verified for all 295 original files across seven rounds. The independent recomputation is `../artifacts/gate05/20261019_isolated_numeric_root_audit/all_seven_probe_audit.json`, SHA256 `31a2827f98b8c297443a01f5dbe21ba1a5f0aafc4dab94c8a0d62c622d4b003e`. The earlier audit of four rounds and 9 small graphs remains unchanged.

| New original package | Number of files | ZIP SHA256 |
|---|---:|---|
| Positive and negative synthetic boundaries: original Clip vs. direct code_clip | 39 | `e000771e8905ed0745fe4a152b732757df8054ae61cfb36b80f9067224c2f3a2` |
| Positive and negative boundaries: code_clip with import correction | 35 | `3aa3bffc1be331af7cb7bacc0116ff811a06806d4a47f5837028f501530e966f` |
| First layer convolution: code_clip with import correction | 33 | `f2bc91d19641791eea27b85a67244f0c2fa37ad8d5f5e3e44eadcffd03a5a9b4` |

The corresponding local directories are `20261019_code_clip_signed_native_evidence/`, `20261019_code_clip_importer_corrected_native_evidence/`, `20261019_firstconv_code_clip_corrected_native_evidence/`, all located in `../artifacts/gate05`. The corrected first-layer input package is `20261019_firstconv_code_clip_probe_v2.zip`. Sample-wise boundary differences are saved in `signed_operator_remaining_differences.json`; the failed version does not cover this.

# # How to Validate the Full Graph Next

Currently, small graph fix candidates exist. Before advancing to the full q_only_G1, the following must still be completed:

1. Reverify the full graph rewrite and import correction list, addressing any remaining backend rounding boundary discrepancies; the current operator boundary tests cannot be used as a pass criterion for full graph migration.
2. Validate the preservation of original SWISH and INT8 output in the fused prefix; in this round, the first layer has an additional activation tap before the fusion, which will be removed.
3. Then, the full q_only_G1 of the same fixed image is checked against the original output and original threshold detection to determine whether other configurations need to be rebuilt. The original full PyTorch/ORT numerical guardrail failure still requires individual resolution.

A native numerical fault point currently identified can corrupt input information, along with a local workaround; a complete model repair, true object recovery, dataset precision pass, or performance qualification cannot be declared. The previously established issue of strict integer arithmetic in 7 deep convolutions remains unresolved.
