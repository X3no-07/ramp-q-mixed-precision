# Gate 0.5: Dual Convolutional Context for Localization



2026-10-20,follows [single convolution prefix three-arm comparison](31_gate05_prefix_clip_and_precision_constraint_controls.md). In the previous single Conv context, NONE/OBEY does not alter the output and does not reproduce the complete corrected q_all format filtering error. In this round, new dual Conv contexts are extracted from actual failure images, with the original complete numerical guardrail retained `FAILED_UNCHANGED`, and the Gate 0.5 remains `INCOMPLETE`.

**Result: Both configurations were successfully constructed and executed. The raw outputs of NONE/OBEY are byte-for-byte identical. For both actual convolutions, inputs, weights, and outputs are all INT8. The second Conv indeed incorporates the third Clip, and the full graph error has not been reproduced. Among the 819,200 elements compared to CPU, there are exactly 18 discrepancies involving a single integer digit; no complete model, dataset, or performance pass conclusion has been established.**

# # Frozen Scope and Preparation

This round retains only the corrected code_clip structure, comparing NONE/OBEY. The input is the original fixed image's `images` `[1,3,640,640]`, including `/model.0/conv/Conv`, `/model.1/conv/Conv`, and the original SiLU, two sets of weight QDQ, and three activation Q→Clip→DQ layers. The only output is `rampq_model_2_cv1_conv_a_dq` `[1,32,160,160]`, excluding the third Conv.

The rationale for selecting this boundary is based on the preserved complete failure build log: the third Clip is fused into the second Conv and SiLU, while the first two Clips remain as independent per-point layers. Thus, it represents a newly introduced specific fusion context; however, it cannot be assumed that failures appear monotonically with prefix length, nor can the addition of new nodes be directly attributed as the root cause.

Freeze protocol: `../artifacts/gate05/20261020_twoconv_context_protocol/protocol.json`, SHA256 `6ba4cae53e1ca115c2b49bd7fdac91d80c15ef1cc1f854c57f01cd09ae185fa0`. Source image SHA256 `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee`.

Each group is allowed at most one build, with a 300-second timeout; the outer monitor has a 900-second guardrail. Only successful engines perform two fixed-input executions, preserving actual layer information, all parsing constraints, configuration flags, and raw outputs; layer precision and output type constraints are not set or cleared. Ordinary compilation errors are retained and the process proceeds to the next frozen case. If timeouts, resource exhaustion, signals, or input drift occur, subsequent tasks are halted and the private group is cleaned up.

Preparation confirmed 19 nodes and 20 initializers. All selected nodes and constants match the serialized bytes in their respective complete source graphs. The original and corrected slices each ran once on CPU; their `[1,32,160,160]` outputs are byte-identical, with a maximum difference of 0. The raw CPU array SHA256 is `40d65c0d2a887cd192f59e4c6797827012a7515239572ed1fed73d1912282782`. No complete-model CPU inference was run, and the original guardrail was unchanged.

The corrected slice SHA256 is `5e6dfc0294a97b65810816c1ba80ef679b6d55851b03724be4c8de3e32c6c6fb`; the original-structure slice SHA256 is `d3ff05c1f7e0da4ebb51fbf8ca921c0d151d73519edf3b3f85fa1b4a3c7a5bb1`. The final sealed ZIP is `../artifacts/gate05/20261020_twoconv_context_probe.zip`, containing 10,669,204 bytes, SHA256 `599190d18de4f57bb80911ead9e436661a3a86daa4117817044079d71c25d525`; spec SHA256 `e9b48c0319f68e361977d093b25df8ceacb5f1827f8a413e5e7107f751af2a5c`, manifest SHA256 `c809fcc573074108ff2d1f5d199ab105c2afdfe9810fed472659cd4abbb3959e`.

An independent pre-launch review of 487 items passed, with no blockers; report `../artifacts/gate05/20261020_twoconv_context_review/pre_native_review.json` SHA256 `6308333eba0c6b7c2f3753943e7ec8eb78b1546462fde1a63d1e89b34494f382`. The initial local audit failed when attempting to decode the Chinese source code using the default GBK encoding; after explicitly switching to UTF-8, the review was completed. No re-run of CPU inference or modification of source evidence was performed. The transport startup script was supplemented with guard identity, status verification, and cleanup of unconfirmed private groups; the sealed computation package remained unchanged.

# # Native Results

The new directory is `/home/nvidia/rampq/gate05_twoconv_context_20261020/gate05_twoconv_context_probe_20261020`. Monitor PID 31158, private group 31163, guard PID 31169; the two builder PIDs are 31176 and 31222. Each group performs one build and two fixed-input executions, all outputs are finite, and raw bytes are consistently repeated.

| case | Builder API seconds | Actual dual Convs | CPU element count difference | CPU maximum absolute difference |
|---|---:|---|---:|---:|
| code_clip_none | 69.188 | Both inputs, weights, and outputs are INT8, each with a named integer tactic | 18 / 819,200 | 1.5619525909423828 |
| code_clip_obey | 69.375 | Same as above | 18 / 819,200 | 1.5619525909423828 |

Runner total runtime: 149.414 seconds, outer monitor total runtime: 155.63 seconds, returning `BUILD_SCRIPT_OK`. Guard final state is `COMPLETE`, no intervention, private group empty, no timeout or resource exhaustion. These are diagnostic runtimes, not performance inference measurements.

The original arrays with the same topology NONE/OBEY have identical SHA256 values `f524e075db8746be34c4282c10eb6f50f8d1483b881f47cac6fbd1ca4391c93a`, byte-by-byte consistent. The engine SHA256 values are respectively `8bcada5ac6a1132dc7f1951b1d2f6160a98ff5ba05338a4a641483fcada87fa9` and `d68aa8d00de580ca4c57fc56e3141172f91178ed8f1452035a0632c408365802`; identical outputs are not required between the two serialized engine files.

# # Actual Fusion and Constraints

Each engine contains exactly two actual Conv layers, with source names uniquely mapping to different inspector rows. The weight counts are 432 and 4608, respectively, with input, weights, and outputs all in INT8, and external bindings set to FLOAT/LINEAR.

- The activation of the first Conv is a dedicated `SWISH`, with tactic `ampere_first_layer_filter3x3_imma_fwd_swish`.
The activation of the second Conv is `GENERIC`, with tactic `sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_vect_c_32_tilesize256x32x64_stage4_warpsize4x1x1_g1_tensor16x8x32_t1r3s3_linkable`. The actual layer name includes a third occurrence of `rampq_model_2_cv1_conv_Clip`, and 7 pointwise operations include truncation at Tanh-SiLU and Min/Max.

Therefore, the fusion of the third Clip is supported by actual engine evidence in this round; however, the `[GpuLayer]` log in the complete failure graph remains only an optimization plan during failed construction and cannot be considered as evidence that the full engine has actually run.

Compared to the previous single Conv prefix, the first Conv in this round has changed from GENERIC to a dedicated SWISH. This indicates that adding graph context may alter the tactic or fusion of existing layers, and thus the differences between the two prefixes cannot be entirely attributed to the addition of new nodes. It is also incorrect to assume that failure occurs monotonically with increasing length.

Both parsed networks have 31 layers, each with 5 explicit INT8 precision constraints and 5 explicit INT8 output type constraints, with no query errors. The Clip alpha/beta correction constraints remain unchanged at three locations. The source graphs and per-layer constraints for NONE/OBEY are identical; the only difference in actual configuration is the OBEY switch. Settings such as INT8 + FP16 and TF32 disabled, PREFER disabled, workspace set to 1024 MiB, avgTiming=1 are identical. No layer precision/output type setter or reset is applied.

# # Numerical Differences

The results relative to a frozen CPU reference are identical: 18 elements differ, with an average absolute difference of `3.4320240083616226e-05`, RMS of `0.0073216512178381965`, and a maximum absolute difference of `1.5619525909423828`.

A supplementary review restores all CPU/native values to integer digits and performs FLOAT32 dequantization using the frozen FLOAT32 scale `1.5619523525238037`. Each original byte is precisely reconstructed. Among the 18 differing codes, 10 have native−CPU differences of −1, and 8 have +1; the remaining 819,182 codes are identical. Thus, it can be stated that 'there are 18 instances differing by exactly one integer digit'. Due to rounding during FLOAT32 dequantization, the maximum floating-point difference does not strictly equal the scale, and the two values must not be conflated.

These are descriptive differences; no new tolerances are introduced, and the original full PyTorch/ORT failure guardrails are not modified. This round does not include detection heads or GT matching, so local output proximity to CPU cannot be interpreted as a pass in dataset precision or object-level recovery.

# # Evidence, Final State, and Boundaries

The original archive `../artifacts/gate05/20261020_twoconv_context_native/native_evidence.zip` is 11,640,467 bytes, with SHA256 `6f9ed4fd0f3b21617eb48938cdf79910155a4d18bd29e49c55b168a88226594c`. All 89 receipts have been verified against exact file lists, sizes, SHA256, and ZIP CRC; metadata in the receipts themselves is also validated. The receipt SHA256 is `29433da0ac90fc65f58f3a709a4a2704f0455291bd55dbb6d2b812b4a0aece16`. Final SSH rechecks confirm all matches; the original tree has not been modified.

The root audit `root_native_audit.json` has a SHA256 of `3b5c0589211613898970cd1d82ee8d8fc661c405975a8d3b7f7e76c6dce5ea8a`. Independent review passes 367 items, with SHA256 `independent_native_review.json`; supplementary integer-digit and five-process final state review has SHA256 `428075bbbddc2974aea157cabef5fef1cbcd89cade0a836f0ca4de00f1dfaa8b` for `independent_code_grid_and_exit_review.json` and `d5dea87adf728041aae325da2dc905eddb05dd5ad2b60069df40046c1339ecd8` respectively.

Final observation timestamp is 2026-10-20 11:38 on the client side, while the board side remains at 2026-10-17 17:49, both recorded and uncorrected. The launch number remains `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. All five recorded PIDs are absent, and there are no other build tasks in this round. Fans and jtop are active, NRestarts=0, and available memory is 5003 MiB. The 15W/mode 2 query still carries non-administrator EMC read permission prompts, indicating incomplete privilege validation, and thus cannot be used to conclude that the previous disconnection cause has been resolved.

The previously complete q_only_G1 engine remains with SHA256 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`; the corrected q_all failure directory still contains no engine. The original model, weights, calibration scales, full failure records, and the q_only_G1 successful run are all preserved.

This round's conclusion is limited to: the formatting error in the newly added double Conv context has not been fully reproduced, and enabling OBEY in this context is insufficient to trigger it. We cannot declare the complete model constructable, nor can we rule out interactions between OBEY and other graph contexts or full-graph optimizations. We also cannot directly attribute the error to 'a node after the third Conv'. Gate 0.5 remains at `INCOMPLETE`.

# # Next Static Candidate

The candidate has been archived read-only: `../artifacts/gate05/20261020_twoconv_context_review/next_context_static_options.json`, SHA256 `41aaa27de8b5f275cf3632933e3d1609c03a0028b2d6906f1d8e1f4022ee6378`. The candidate has not yet been extracted or executed:

| Candidate Context | Static Dependency Size | Distinction and Limitations |
|---|---|---|
| DFL Softmax→QClipDQ→1×1 Conv→Reshape | 9 nodes / 7 initializers | Internal input `[1,16,4,8400]`, shape different from image prefix |
| PE reshape→depthwise128→Add→projection | 15／16 | Two inputs, two outputs; original auxiliary output constraints must be preserved |
| model.4 Split → residual → Concat → Conv → next QDQ | 31／28 | Three Convs, including branches, quantized output clone, and residual context |

Prioritize evaluating the minimal reproduction cost of DFL or PE, but must first freeze the source and semantic value of internal inputs. Existing image NPY files cannot be directly fed into internal contracts. The final internal error in the complete failure log has no node name, and thus the root cause cannot be attributed to 'a later log entry'. This round did not initiate these candidates, full model retries, recovery matrices, or evaluation on 150 images.

Stage checkpoint: `../artifacts/gate05/20261020_twoconv_context_native/checkpoint.json`; current status: `../artifacts/gate05/current_execution_state.json`.
