# Gate 0.5: Prefix fusion with full candidate validation



On 2026-10-19, building upon [First Layer Clip/QDQ Localization](27_gate05_first_layer_quantization_anomaly.md). The complete q_only_G1 repair candidate has been successfully built on Orin, passed existing actual precision audits, and recovered two predictions on the fixed image `000020.jpg`. This validates the improvement of native execution on this image; however, the original numerical guardrail still fails, and Gate 0.5 has not passed. The original frozen engine, weights, calibration, and detection thresholds are all retained.

# # Prefix Fusion

Extract 11 nodes from the original q_only_G1 graph: first activation Clip/Q/DQ, first weight Q/DQ, first Conv, Sigmoid/Mul, second layer input Clip/Q/DQ. The input is the same preprocessed array as in `000020.jpg`, with the only output being `rampq_model_1_conv_a_dq`, FLOAT[1,16,320,320]. The second Conv is not included in the graph, and no additional output taps are added before or after activation.

The second activation scale is `0.7481081485748291`, with the original floating-point Clip at ±95.00973510742188. The CPU ORT_DISABLE_ALL outputs of the original graph and the code_clip-modified graph are bit-by-bit identical. All selected non-modified nodes, DQ/scale/zero, weights, and biases retain their original bytes.

Both on-device engines are actual INT8 input, weight, and output convolutions, with the SiLU five-step Tanh-point operation expressions fused into the same convolution. Both use the identical named integer tactic:

`sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_vect_c_32_tilesize256x32x64_stage4_warpsize4x1x1_g1_tensor16x8x32_t1r3s3_linkable`

The actual activation field is GENERIC, corresponding to SiLU; it does not reproduce bit-by-bit the original complete engine's dedicated SWISH tactic, thus only providing a numerical performance assessment for this controlled prefix fusion.

| Prefix | Relative CPU maximum absolute difference | RMS difference |
|---|---:|---:|
| Original floating-point Clip→Q/DQ | 36.6572983265 | 1.3701924555 |
| code_clip plus attribute import correction | 0.7481081486 | 0.0015463345 |

Among the 1,638,400 outputs of the repaired version, only 7 points differ by one second-layer quantization step; all other elements are element-wise identical. Full coordinates and values are preserved, with no new numerical pass criterion established. Both engines execute twice, producing bit-identical outputs across all elements, and all outputs are finite.

Original package `../artifacts/gate05/20261019_fused_prefix_native_evidence.zip`, 2,123,956 bytes, SHA256 `fc1ca5dbf68ad6214803d93ed716c2aa6d3ee62421a71509ac6f463ff1063a46`. The receipt of 54 original files has been verified. Independent original arrays, actual layers, and control checks are stored at `../artifacts/gate05/20261019_fused_prefix_root_audit.json`, SHA256 `316c47a9376e63dcec6e6dd04b8f32c1138c833673cb257500e82022d2c185b6`.

This round of monitoring ended normally in 187.74 seconds. After the original monitor ended, the supplementary guard independent check of the private group was completely empty, with no termination signal issued; it is an exit validation and cannot be traced to claim full coverage of the prior build.

# # Full Candidate vs CPU Comparison

The candidate only converts six floating-point Clip→Q operations in G1 to Q→INT8 Clip[-127,+127]. A total of 385 nodes and 211 initializers are involved, with the total count unchanged. Only 12 Clip/Q nodes and 12 boundary initializers are modified; 88 Convs, all DQs, scale/zero, weight Q/DQ, bias, and external output declarations are preserved in their original bytes. The original module_map, failed preflight, and diagnostic_scope are copied exactly. The full candidate is saved separately and does not overwrite the original q_only_G1.

Input package `../artifacts/gate05/20261019_corrected_qonly_candidate.zip`, 9,954,391 bytes, SHA256 `a1369a6bf301156cc658d3b6df34a62b679abd42904cb3222c046d46c6e6fdf9`. Model SHA256 `3f7d481fdf5268bea3d66c2176827dffaf3bd0628f5a2c412d48e3b6c53e9356`, spec SHA256 `aeddbe15cdf0509da74d2b4d7e714d6b29e74bdc4d0e30c3e2380d3204e841dc`.

The full original image and the full candidate set each undergo a single CPU execution of ORT_DISABLE_ALL, with the main output `[1,14,8400]` and PE auxiliary output `[1,128,20,20]` being bit-wise identical, and the differing elements being 0. The original array and the source hashes before and after are `../artifacts/gate05/20261019_corrected_qonly_cpu_equivalence`, and the analysis is performed on SHA256 `9146d33bb019bc83a34812e4977a70607f1af7ae5c72304e1e98d005dd2ea3da`.

These consistency checks validate the correctness of the rewrite on this CPU input, not the full numerical guardrail pass of the original PyTorch/ORT; the latter remains at `FAILED_UNCHANGED`. No dataset runs are performed.

# # Full native build and actual precision: Completed

New directory `/home/nvidia/rampq/gate05_corrected_qonly_20261019/gate05_corrected_qonly_candidate_20261019`, launch number `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. Only one 1800-second budget build is conducted; the Python Builder takes 1260.392 seconds, monitor takes 1263.99 seconds, and returns normally at `BUILD_SCRIPT_OK`.

After parsing, the Python Builder explicitly restores six code_clip alpha/beta values to -127/+127; INT8, FP16, prefer, TF32 disabled, 1024 MiB workspace, and avgTiming=1 are maintained as requested. The candidate engine is 7,004,606 bytes, with 316 actual layers, SHA256 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`.

The new source auditor strictly supports DQ←INT8Clip[-127,127]←Q, reusing the original actual format, integer tactic naming, and fusion attribution standards. Zero errors are detected in the new graph source; all six quantization sources from the original FloatClip are rejected. Independent re-execution of the source image and actual format audit produces output bitwise identical to the original audit, SHA256 `63776e092be4a09c0610890dcad6bf1149a645925028f7498fbf6100dcf1d7ff`.

All 88 source convolutions are successfully mapped; 6 G1 convolutions meet the INT8 input, weight, and integer tactic thresholds; the remaining 82 convolutions meet the FP16 thresholds for inputs, weights, and outputs, with status `REQUESTED_CONV_FORMATS_VERIFIED` and `ALL_REQUESTED_INT8_TACTICS_VERIFIED`. The outputs of the 6 quantized convolutions are actually 3 INT8, 2 FP32, and 1 FP16, and cannot be stated as all six outputs being INT8.

The actual activation of the first convolution in the full engine is `SWISH`, using `ampere_first_layer_filter3x3_imma_fwd_swish`; all inputs, weights, and outputs are INT8. Therefore, the full engine validation includes a dedicated SWISH path; the GENERIC restriction in the prefix engine is still retained as previously described, and the two tactics cannot be mixed in writing.

The three external bindings are all FLOAT/LINEAR: input `[1,3,640,640]`, main output `[1,14,8400]`, PE auxiliary output `[1,128,20,20]`. Passing the actual layer format does not imply numerical equivalence of the full output with PyTorch/ORT.

# # Fixed single-image execution: Prediction restored

After passing the actual format audit, the unchanged generic single-image script is used to execute `000020.jpg`; no switch is enabled to bypass strict integer auditing. The original NMS is retained at `conf=0.25 / IoU=0.7 / max_det=1000 / nc=10`. Both main and auxiliary outputs are finite values.

| Prediction category | confidence | Original image xyxy |
|---|---:|---|
| Holothurian | 0.6566590667 | [3.5625, 2.25, 248.25, 293.25] |
| Scallop | 0.3747900724 | [168.046875, 198.375, 245.578125, 286.125] |

The CPU recalculates NMS with original parameters and matches the saved float32 detections on the board end byte-for-byte. The local post-processing Ultralytics is version 8.4.142, while the board-end version is 8.4.50; here, the consistency of the saved array results is confirmed, without assuming software version equivalence. `000020.txt` contains only one holothurian annotation; the holothurian prediction has an IoU of 0.8108697094, matching under the existing rule of conf≥0.25 and same-class IoU≥0.50 for a singleton; the additional scallop prediction has no corresponding GT and thus cannot be considered a second correct target. The original q_only_G1, when recalculated with the same NMS, yields zero predictions, with the highest score across all classes at 0.0031358302. The new result confirms that the current candidate improves the native execution on this fixed image.

The maximum absolute difference of xywh in the new native output relative to the original CPU ORT main output is 5.9234314, and the maximum absolute difference of score is 0.0047020614; the maximum absolute difference of PE auxiliary output is 0.0803725123. These are only descriptive and do not introduce new tolerances or alter the original guardrails. The CPU recalculated NMS, GT matching, and original array differences are saved at `../artifacts/gate05/20261019_corrected_qonly_native/fixed_image_recheck`.

The original output SHA256 is `d4549faaf0f0d041c7c540856193b2584e94a8356e3dac11221733722c1b154a`; the original visualization and single-image report are preserved at `post_smoke_evidence/fixed_image_smoke/`. The first call is solely for diagnostic logging and does not include warm-up, repeated timing, or performance acceptance; it cannot be converted into a FPS conclusion.

The common_preparation hash of the new candidate differs from the old FP16 reference; the generic script has marked this comparison as `descriptive_only` and `shared_source_graph_hash_control_verified=false`. Even with close predictions, no non-quantized floating-point or fusion control has been established, and thus it cannot serve as causal proof for target-level module recovery.

# # Original evidence and board-end exit

Construct the original package `../artifacts/gate05/20261019_corrected_qonly_native/build_evidence.zip`, 47 files, SHA256 `78bb564761b48abac43e1aaa3bb3b9a0756b95f54c2acad3b5c0ca4043ddaaa1`.

Complete build and single-image package `../artifacts/gate05/20261019_corrected_qonly_native/post_smoke_evidence.zip`, 16,012,175 bytes, 70 original files, SHA256 `5eac1543e5151708585e57923dd82410c9b21aac81cb5cfd1137e057e35ed205`. Exact receipt, ZIP CRC, file bytes, and SHA have been verified. All 301 read-only checks passed in independent final review; supplementary source audit also passed, with no change in file size before and after the original tree. All derived reports are outside the original tree.

Build monitor PID 21274, private group 21279, guard 21284; single-image monitor 22461, private group 22466, guard 22473. Both-stage monitors returned normally, both guards `COMPLETE`, final group empty and uninterfered. Archive `post_smoke_end_board_status.json` only checks build PID; single-image exit has separate stage monitor/guard evidence. Real-time verification confirms all six PIDs exited successfully, no residual tasks, fan service active, jtop active with NRestarts=0, approximately 5056 MiB available memory, startup sequence unchanged.

Board time differs from client time; evidence is separately preserved, with no clock modification. 15W mode 2 query still shows non-administrator EMC permission prompts, no changes to power configuration. Final real-time record kept in `../artifacts/gate05/20261019_corrected_qonly_native/final_board_observation_after_corrected_qonly.json`.

# # Current Boundary and Next Step

This round completed the closed loop of 'candidate fix → full build → actual precision audit → fixed single image → evidence recovery'. Gate 0.5 remains at `INCOMPLETE`.

1. The original full PyTorch/ORT numerical guardrail remains at `FAILED_UNCHANGED`. This CPU original image/rewritten image bit-level consistency does not replace the guardrail.
2. Only one existing Selection image diagnostic execution was performed; no 150 evaluation images were run, and no new object-level damage-recovery relationship was validated.
3. Seven deep convolution integer evidence gaps remain in the original q_all and three recovery configurations; they cannot be eliminated simply because the q_only_G1 six-integer tactic passed.
4. The next step is to first resolve the comparability of the full reference and the original numerical guardrail, review the conditions under which candidate fixes can be extended to other frozen configurations, and then freeze the dataset acceptance rules. This test figure cannot replace independent evaluation, and no post-hoc modification of acceptance criteria is allowed.
