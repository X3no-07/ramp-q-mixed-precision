# Gate 0.5 Current Executable Work Record



Client date: 2026-10-19. The task scope is to complete the current executable engineering checks and to stop when further protocol or research planning is required. **All six frozen configurations have completed real hardware builds, actual format checks, and fixed single-image execution; Gate 0.5 has not passed and is currently halted at the stage of numerical migration and acceptance protocol planning.**

# # Final Conclusion of This Round

| Frozen Configuration | Actual INT8 / FP16 Convolution | Named integer tactic / requested INT8 count | Fixed single-image prediction count (conf=0.25) |
|---|---:|---:|---:|
| FP16 baseline | 0 / 88 | Not applicable | 2 |
| q_all | 88 / 0 | 81 / 88 | 0 |
| Only recover G1 | 82 / 6 | 75 / 82 | 0 |
| Recover G1+G5 | 74 / 14 | 67 / 74 | 0 |
| Recover G1+G2 | 77 / 11 | 70 / 77 | 0 |
| Only quantize G1 | 6 / 82 | 6 / 6 | 0 |

All engines have been genuinely executed, with both main and PE auxiliary outputs being finite values. q_all and the three recovery configurations each have 7 deep convolutions with unexposed tactic names; the format completeness and strict integer computation evidence are separately preserved. All quantization configurations show zero detections on this image, and thus cannot treat build or finite output as a pass in detection accuracy. Final re-execution of the FP16 baseline under the same startup environment still yields two predictions, with the raw output matching bit-by-bit the previous FP16 baseline output.

The frozen engineering matrix for executable configurations has been completed. The next steps require planning for numerical anomaly localization and comparable references, non-quantized floating-point and fused control, and the acceptance criteria prior to dataset evaluation. No 150-image dataset evaluation has been run, no original threshold, calibration range, or numerical tolerance adjustments have been made, and no real-machine recovery, bidirectional attribution, or performance conclusions have been established. Below are retained evidences of each failure, timeout, and final completion.

# # Newly Obtained Real-Hardware Evidence

## # Second Time-Limited Build of Complete q_all

On the same PE auxiliary output graph as [Record 25](25_gate05_full_graph_fusion_workaround_audit.md), the average timing count for TensorRT's computation scheme selection is changed from the default 8 to 1. The graph SHA256 remains `2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d`, with model, calibration range, grouping, and original numerical tolerance unchanged.

The monitoring was terminated after 1802.27 seconds due to a build timeout limit of 1800 seconds. The full engine was not generated and the complete INT8 model was not executed. The highest temperature reached was 53.031°C, the lowest available memory was 2293.53 MiB, the maximum swap usage was 0.25 MiB, and new kernel events occurred: 0. No trtexec process was found after termination, and the startup instance remained unchanged at `c4d8fcff-a824-488b-b004-6ad71eeb5d95`.

Conclusion: **Timeout, insufficient evidence**. The state of `latest_build.json` at `BUILD_RUNNING` represents residual data after interruption and cannot be used to infer that the task was still running. 25 sets of original board-side files have been returned and verified against SHA256 and byte counts. The evidence directory is `../artifacts/gate05/20261019_int8_avg1`, and the original compressed package SHA256 is: `bcecb122de52919636358c4a8a9fb01f90fdb4c6cdab0a89c8ce7dda56d4b1ea`.

## # FP16 Precision and Native Constraints with Auxiliary Binding Tests

Extracted the G4 PE depthwise convolution from the same source model, followed by a Sigmoid+Mul layer with identical principles, preserving both auxiliary and main output paths. Only the native build and constraint adoption are inspected; no inference is performed.

TensorRT 8.5.2.2 successfully built and deserialized within approximately 71.72 seconds: convolution weights in Half precision, inputs and outputs in FP16; three external bindings were verified by name and shape, all of which are FLOAT. `--precisionConstraints=obey`, the specified convolution precision, and the output type of the external FLOAT output can be simultaneously adopted.

This resolves a specific engineering question: when restoring a convolution to FP16, it is not necessary to change the auxiliary external output to HALF. It only demonstrates the constraint adoption on this small subgraph and cannot replace a full model-level layer-by-layer verification. 17 original files have been validated. Evidence directory: `../artifacts/gate05/20261019_fp16_constraint_probe`, compressed package SHA256: `c5ab8a9d83c026f32dd63d8ac2c0dfcd7d0070c88d342f52873c59420bd1afdc`.

# # Further Isolation of Numerical Differences

The numerical guardrail between the original PyTorch Q/DQ and the ONNX Runtime Q/DQ complete models remains failed. Additional diagnostics were conducted using only the original Selection image `000020.jpg`, with consistent hash values before and after execution of the source file:

| Diagnostic Intervention | Max box difference | Max score difference | Elements with differing quantized input values in Conv |
|---|---:|---:|---:|
| Original repeated comparison | 96.570465 | 0.0729592 | 3,325,427 |
| Alignment only on the output of the first convolution layer | 86.511719 | 0.0786303 | 3,283,078 |
| Layer-by-layer alignment of all 88 convolution outputs | 0 | approximately 1.31e-7 | 1 |

A residual score gap remaining after aligning all convolution outputs is observed, triggered by a floating-point difference of approximately `2.38e-7` in the SiLU / Sigmoid+Mul layer crossing a rounding boundary. This provides causal isolation evidence for the phenomenon of 'small floating-point differences being repeatedly amplified through quantization bins'. No specific errors in weights, scales, zeros, or quantization formulas were found, though all conversion issues are not entirely ruled out.

The above alignment is an active replacement of intermediate tensors for diagnostic purposes and **cannot serve as evidence that a model runs independently and passes**. The original tolerance has not been relaxed. Complete evidence is available at `../artifacts/gate05/20261019_numerical_diagnostic`.

# # Implemented Frozen Configuration Engineering

Six graphs share the same main output `output0 [1,14,8400]` and the same G4 PE auxiliary output, with original weights and non-quantized arithmetic nodes remaining consistent. Only the Q/DQ and input Clip operations for corresponding convolutions are removed according to the predefined configuration.

| Configuration | INT8 convolution | Request for FP16 convolution | First native attempt (obey / original q_all) |
|---|---:|---:|---|
| FP16 baseline | 0 | 88 | Complete build, 88 FP16 convolution formats verified, single image execution passed |
| q_all | 88 | 0 | Attempted, time-limited termination, no engine generated |
| Only recover G1 | 82 | 6 | Builder internal format error after approximately 16.50 seconds, no engine generated |
| Recover G1+G5 | 74 | 14 | Same error triggered after approximately 15.42 seconds, no engine generated |
| Recover G1+G2 | 77 | 11 | Same error triggered after approximately 16.08 seconds, no engine generated |
| Only quantize G1 | 6 | 82 | Same error triggered after approximately 10.05 seconds, no engine generated |

The image file, manifest, ZIP, and Python 3.8 syntax have been checked. Independent verification confirmed the complete ONNX structure, quantization grouping, and freeze scope for six models. The image file's declared precision does not match the actual precision adopted.

The complete FP16 reference output structure was monitored and completed building and deserialization in approximately 563.23 seconds, with engine SHA256 of `92ce7a8f487cbaf473a281b94daa953120870fa2ad992807d60390dffd1b72fb`. All 88 source convolution weights, inputs, and outputs are in FP16 format, with all external bindings set to FLOAT linear. Real single-image execution was then performed, yielding finite outputs for both main and auxiliary outputs; one prediction for abalone and one for scallop, with the unique abalone match successfully aligned in the ground truth label. The comparison with the prior FP16 output is descriptive only and cannot be attributed to any specific quantization factor.

The 27 build files and 32 inference files were cross-checked against the responses. The latter's ZIP SHA256 is `7a9c2dccc95ec429a4d3505e02c91c6af4fbfb0de68d2830a24ec4af6975b599`. The incomplete package left after the first inference result transmission is preserved as `remote_evidence.partial1.zip`; a complete retransmission was required to validate both size and hash. The device remained on the same boot after disconnection, so the incomplete download cannot be considered valid evidence.

Four mixed-precision configurations were each attempted once, all triggering the same internal assertion: `optimizer.cpp::filterQDQFormats::4421`, `!n->candidateRequirements.empty()`. No engine was generated, and no real single-image execution occurred. This is neither timeout nor a monitored-safe termination. The queue and four configurations collectively verified 88 original files, with the ZIP SHA256 of `046beddbff262c066c6c01930bb8202588f0629ee53f20f4cc9eb1a0f3c8defc`. Further small-image testing is required to distinguish the impact of Q/DQ boundaries versus precision constraints; the error cannot be directly attributed to the failure of the recovery mechanism.

## # Reproduction of minimal constraints at the G1→G2 boundary

The last G1 convolution, its Sigmoid+Mul activation, G2 input Clip/Q/DQ, quantized weights, and G2 convolution were precisely extracted from the original `r_only_G1`. The serialized bytes of 9 arithmetic nodes and 10 initializers are preserved exactly. The input is declared as an alternative upstream `[1,48,160,160]` tensor, with no actual input data executed.

| Same small-image build configuration | Result | Duration |
|---|---|---:|
| FP16 computation and output, obey | Reproduced the same internal format error | 6.125 seconds |
| FP16 computation only, obey | Reproduced the same internal format error | 6.126 seconds |
| FP16 computation and output, prefer | Built and deserialized successfully | 67.151 seconds |
| Only global FP16 enabled, no per-layer requests | Built and deserialized successfully | 65.182 seconds |

The actual inspector for the latter two cases confirmed: G1 convolutions use Half-precision weights, FP16 inputs and outputs; G2 convolutions use INT8 weights, INT8 inputs with integer tactics. The external FLOAT linear binding also conforms to the declaration. This indicates that the current small graph exists with a valid construction path satisfying the original precision requirements, and cannot be generalized from the four full-model obey errors to hardware incompatibility with FP16/INT8 mixed precision.

The next full-model attempt will use `prefer` request mode, **with actual per-layer compliance unchanged**; builder degradation does not imply audit degradation. Only the full 88-layer model achieving its requested format requirements can proceed with single-image execution. The original obey package, results, graph, quantization range, and numerical tolerance are retained. 19 minimal reproducible original files have been verified, with the compressed package SHA256 being `397bb1d7a311f93b0fc4266bfe872a07bc2f0ceba990ddb8d5f095362d81b12a`, and directory `../artifacts/gate05/20261019_mixed_constraint_probe/native_evidence`. These small engines did not perform inference.

The build tool checks all 88 convolution identities after importing the TensorRT network, then explicitly requests FP16 computation and output for restored nodes; the original package uses obey, while the second version uses prefer, with consistent audit standards. The external binding explicitly requests FLOAT linear. Native audit distinguishes between INT8 weight/input formats and integer tactics, and verifies fused members, weight counts, and shared activation sources. The single-image tool allocates buffers for all external outputs, validates by name, and checks for boundedness.

## # Full-model prefer retry and extended budget preparation

Four frozen mixed configurations have been completed once with prefer retry in original order. Only the precision request mode is changed to prefer; graph bytes, scale, zero-point, group, and actual precision acceptance criteria remain unchanged.

| Configuration | Guardian duration | Result | Maximum temperature | Minimum available memory |
|---|---:|---|---:|---:|
| r_only_G1 | 1802.48 seconds | Reached 1800-second budget, no engine generated | 54.343°C | 1538.77 MiB |
| r_only_G1_G5 | 1802.56 seconds | Reached 1800-second budget, no engine generated | 54.687°C | 1298.55 MiB |
| r_only_G1_G2 | 1802.52 seconds | Reached 1800-second budget, no engine generated | 54.156°C | 1004.11 MiB |
| q_only_G1 | 1402.37 seconds | Build, actual format verification, and fixed single-image execution completed | 53.843°C | 715.38 MiB |

All four monitoring arms showed no new kernel security events; the maximum swap usage was 9.75 MiB. The three recovery configurations are time-limited and not yet completed, and thus cannot be stated as impossible to achieve under precision requirements. `q_only_G1`'s engine SHA256 is `d727cd809926b2a016a6e18e25999701caf2beb31c24725f5019bbd894a0cf44`; all 88 source convolutions have been mapped, with 6 G1 convolutions featuring INT8 weights, INT8 activation inputs, and a named integer tactic; the remaining 82 convolutions use FP16 weights, inputs, and outputs. The counts of dual-convolution fusion with identical source inputs and weights across three arms have also been verified, and independent recalculations yield consistent results.

Subsequently, real execution was performed on the original `000020.jpg`. Both outputs are finite. No detections were observed under frozen conf=0.25; two predictions exist in the FP16 reference. The sole sea cucumber in the original single-image annotation matches in the FP16 reference but fails in q_only_G1. This observation cannot serve as validation of dataset precision or causal attribution to G1. The first call timing is not a performance benchmark.

| Return evidence | Number of original files | ZIP SHA256 |
|---|---:|---|
| Build of r_only_G1 | 26 | `3a4f66f085fd70c6bcc99c12eb15ba06e2647ce80c6d6ba90bc1271b91e39724` |
| Build of r_only_G1_G5 | 26 | `1a1b7e11acb8f0275e5e25177de744dd83f6886d91be71cb5e27e7b0a8720ab3` |
| Build of r_only_G1_G2 | 26 | `8af781c0cc895c99d386f4037a6cfdeca0531c51cf24e0da99700e727f438cda` |
| Build of q_only_G1 | 28 | `e6edfe5d9cfd0bb85e812888c45bd4dc9896a1d37ef3e4ca9de34f08492a27d0` |
| Queue and its original records under supervision | 43 | `45e478d38486839dde0658ead34173db27839149addab4b7f09a3dc65848e3d5` |
| Audit and full recovery of q_only_G1 after single-image execution | 33 | `cf4fc0f2e0ab092939d034492dcb4b830674069dd0b69ef7cbaff405d978e44a` |

Each package is first verified against the board-end recorded ZIP size and SHA256, then against the receipt's exact file set, file sizes, and SHA256. Only after this verification is local precision auditing conducted. The file count refers to the number of files within each individual compressed package; files may be duplicated across packages. Directory is `../artifacts/gate05/20261019_precision_native/prefer`; independent summary is `prefer_matrix_summary.json`; local checkpoint status is complete, but Gate 0.5 has not passed.

The client's last continuous observation of the queue was at 09:09. Afterward, the tool call did not return for an extended period, and the USB network interface disappeared during connection reestablishment. Upon recovery, a new startup number `692c7544-b8bd-415b-939c-b11cdbd9f7f0` was observed. The final states of the four items of the original queue were written into the original startup `c4d8fcff-a824-488b-b004-6ad71eeb5d95` record. The build and single-image execution of q_only_G1 occurred in these two startups, and thus cannot be claimed to have run continuously across restarts. The current logs are insufficient to determine the cause of the restart. The board-end wall clock is clearly incorrect; its original value is retained, with events distinguished by client date, startup number, and monotonic elapsed time.

## # Fixed single-image difference check for q_only_G1

Additionally, a single Selection image is run once with PyTorch CPU `q_only:G1` and once with exact ONNX ORT CPU (with graph optimization disabled). No recalibration, no modification of the model or threshold, no dataset execution.

| Saved outputs | Frozen prediction count at conf=0.25 | Highest category score |
|---|---:|---:|
| PyTorch q_only:G1,CPU | 2 | 0.657063 |
| ORT q_only_G1,CPU | 2 | 0.657379 |
| TensorRT q_only_G1 on board-end | 0 | 0.00313583 |
| TensorRT FP16 control with a shared output structure | 2 | 0.667147 |

The maximum absolute difference in scores between PyTorch and ORT is approximately 0.000491917; the maximum absolute difference between ORT and TensorRT q_only_G1 is approximately 0.657377. These discrepancies exist in the raw outputs before NMS, not after adjusting the NMS threshold. The 6 G1 quantization scales, Q/DQ weights, and 88 original convolution weights and biases are consistent. The SHA256 of 86 source files remains unchanged before and after.

Both the computer and board-end reconstruct the pre-processing input as recorded. The tensor raw byte SHA256 is `b10ad22bef235bf2d2763672da55b43baca61db7c4640db6c3803e3390e722b6`. This is a CPU reconstruction check and does not capture the prior actual CUDA input buffers. The execution script outputs are indexed by name, with each binding allocated separately and synchronized before and after execution. Source code inspection did not reveal any obvious indexing or synchronization errors.

The above comparison still shows differences between the CPU-quantized FP32 portion and the board-end's remaining 82 convolution FP16, fused operations, and backend implementations, which cannot yet be uniquely attributed. No new native numerical pass thresholds have been introduced, and the original q_all full model migration guardrail remains failed. Evidence directory is `../artifacts/gate05/20261019_qonly_fixed_image_diagnostic`, CPU raw manifest SHA256 is `e761c146a87cda48772a468624d9cba5a92fdd2931e90b3ee02221e0bd5a7c14`; board-end CPU reconstruction records and their supplementary manifest are separately saved.

## # A Full q_all Extension Budget Attempt

The previous two full q_all builds reached the 1800-second limit without triggering temperature, memory, or kernel event protections. A third attempt with a 7200-second budget was initiated after queue and qualified single-image checks completed, using the original monitoring program's `--max-seconds 7200`, PID 3635, and launch number `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. The package retains the original v4 image, build program, monitoring program, and all other build parameters in their raw byte form, with input ZIP SHA256 of `c77f1617a2279610382b2147463c6d3676aa1334d87fcc15c0de4f316ea11ac3`.

The monitoring timer was completed in **3279.35 seconds** (approximately 55 minutes) for build and deserialization. The highest temperature reached was 53.625°C, the lowest available memory was 1867.22 MiB, and swap usage was 0. No new kernel events were observed; upon completion, it was confirmed that monitor, build, and all trtexec processes had terminated. The complete engine SHA256 is `e5c9cae835bddc60bfe744d8d5af8c41c112133911936d8a85e8eddf58ebf9e5`. Therefore, the previous 30 minutes of halt can only be attributed to budget constraints, not to the inevitable failure of constructing this graph.

Layer-by-layer inspection confirmed that all 88 convolutional layers have INT8-form activation inputs and weights. Of these, 81 have named integer tactics, while 7 CUDA DepthwiseConvolution layers do not expose tactic names. **Strict integer computation evidence remains unverified**, and no substitution has been made using TacticValue or weight format to replace this evidence. The previous audit tool fixedly read the first input; the original tool has been archived and readdressed using native ConvXAsActInputIdx. Illegal or unknown indices are now treated as failures. All 88 indices in this engine are marked with native -1 for the primary input. The conclusion of 88/81/7 remains consistent before and after the fix, with no relaxation of format, fusion, or integer computation standards.

After explicitly enabling `--diagnostic-formats-only`, the same fixed Selection graph was executed. Both the main output and PE auxiliary outputs yielded finite values, yet remained zero detections at a fixed confidence level of 0.25; the only annotated sea cucumber target failed to match. The FP16 counterpart successfully matched this target. The original full model's PyTorch/ORT numerical guardrails failed as before, and no dataset passed or recovery conclusion was established.

The first Nsight attempt halted during the parameter validation phase of the legacy board-side script, without executing any images; the legacy script has been preserved. A second attempt, synchronized with the existing NVTX script, completed image execution, but the NVTX trigger failed to generate a trace, and it was not recorded as a successful capture. A third attempt using full-process capture successfully replayed the same fixed graph, retrieving the original `.nsys-rep` and SQLite. The raw outputs from the two actual executions were bitwise identical, with the highest category score approximately 0.000102222. This repeated check is not a performance measurement.

A read-only analysis of the saved SQLite database confirmed that all 7 depthwise convolutions passed full NVTX layer names and layer JSONs consistent with the inspector, along with thread-specific CUDA API intervals and unique correlationId, precisely linking each to its corresponding runtime kernel. The PE corresponds to `depthwiseConvINT8GenericKernelF`, while the others correspond to DSConvIMMA templates or `depthwiseConvINT8SpecializedKernel`. These function names confirm runtime identity but do not serve as proof of arithmetic execution, and no missing TacticName has been supplemented. The 88 INT8-formatted layers, 81 named integer tactics, and 7 kernel runtime identities are preserved. Strict audit remains at `PRECISION_REVIEW_REQUIRED`. All three attempts and replay outputs are retained; the read-only analysis was performed at `../artifacts/gate05/20261019_qall_kernel_diagnostic`, with analysis JSON SHA256 of `6cab40a4da8cbc90329fcf12feea10771360bbed2c48a588abeec5d99a1bb634`.

The 31 original files generated after build and the 53 original files generated after single-image and trace processing were cross-verified. The former has ZIP SHA256 of `13097d40d66b11a17ce8d1a448899c4a0ea975616a495b1b5d21cf1cd4d2b59c`, and the latter has `6a4adc656ff9aa8c107881a8708f58edf478b0c6822d93464053a697ed5f0d74`. The process and full recovery directory are `../artifacts/gate05/20261019_qall_extended`. The actual two-hour budget is preserved in the original launch, monitor request, and status records.

## # Results of Extended Budgets for Three Recovery Configurations

The full q_all build succeeded after approximately 55 minutes. The three recovery configurations had previously only run for 30 minutes, so each was subjected to a 7200-second budget attempt. Only the external launch and budget revision records were updated; all original bundle, graph, builder, monitor, spec, and actual precision standards were preserved. The 1800-second field from the spec is inherited and recorded alongside the external 7200-second monitoring coverage, and it cannot be falsely claimed that the original spec has been modified. A completely native directory is used with strict serial execution; ordinary failures or budget stops proceed to the next item, while hardware security events or changes in launch number halt the entire queue.

The queue completed at launch number `692c7544-b8bd-415b-939c-b11cdbd9f7f0`, with original PID 5740 and output directory `/home/nvidia/rampq/gate05_restore_extended_queue_20261019`. The original 33 bundle files are byte-for-byte identical. External plans, budget coverage, and queue scripts are all preserved. The input ZIP SHA256 is `04b38b71546de563f47de6271364e6410ba0166c51b13fddbf807150535d457a`. The local v1/v2 versions are only preparation snapshots that were not executed; the actual upload and board-side check-only execution was performed on v3.

| Configuration | Guardian duration | Maximum temperature | Minimum available memory | Maximum swap usage |
|---|---:|---:|---:|---:|
| Only recover G1 | 3393.67 seconds | 54.375°C | 1120.86 MiB | 0.75 MiB |
| Recover G1+G5 | 4062.20 seconds | 55.656°C | 746.75 MiB | 10.25 MiB |
| Recover G1+G2 | 3267.34 seconds | 55.125°C | 1090.24 MiB | 10.25 MiB |

All three supervisions ended at `BUILD_SCRIPT_OK`, with no new kernel security events. The queue records the exit of each item's owned descendants and only proceeds to the next item after the global trtexec process is empty; the final independent confirmation also verified that all task processes were empty. The extended budget successfully completed the build without altering the graph or actual precision requirements.

| Configuration | engine SHA256 |
|---|---|
| Only recover G1 | `112cbe6f6a0edceaf0e05f82a4217951663498177935e5fcacf7863a0d1fdbce` |
| Recover G1+G5 | `57ebf75a03428eb9878b9ea9f6060130403f5154c750049795cb59f471d91784` |
| Recover G1+G2 | `8054e6242c63eae17dfaf473944426c77ad3a79c25000bb093476fe31556c40e` |

All three engines pass the layer-wise format verification as requested, with 75/67/70 named integer tactics respectively, leaving 7 deep convolution tactics without names. Independent re-computation confirms consistency across the original graph, weights, groupings, fusion operations, and actual mappings. Subsequently, a diagnostic option with complete format but strictly incomplete evidence is explicitly enabled, and each executes the original `000020.jpg`, while full-process Nsight profiling is performed; both outputs are finite, yet show zero detections at conf=0.25. The highest category scores are approximately 0.000105888, 0.000145435, and 0.000096500, respectively. Across all three configurations, 21 unnamed layers pass full NVTX/registered layer JSON, thread API interval, and unique correlationId-based kernel association; no TacticName is inferred or claimed to have proven integer arithmetic instructions. Read-only analysis is conducted in `../artifacts/gate05/20261019_restore_saved_execution_diagnostic`, with JSON SHA256 equal to `0e30bbd92e9c7c4720cfd8f71d88beb77868d1bd04175e219d111beac30c09c2`. The saved annotations match shift from sea cucumber matching under FP16 comparison to non-matching; this is limited to a single fixed image description and cannot be used to declare dataset recovery failure, success, or causality.

| Return evidence | Number of original files | ZIP SHA256 |
|---|---:|---|
| Only G1 build recovery is performed | 28 | `779e3226962bf067d0031de751848d01d097b1dd92574da43c886ec3e0bac2b9` |
| Recovery of G1+G5 build | 28 | `fa9de4a7cf617ea4a2c33aa0ed2719a31a4a4c073b802feb7f62f102e341c2d7` |
| Recovery of G1+G2 build | 28 | `ab9d94ad9a3f23cf15d936c13ed029ce717eeacb238a7dd99b58420589ab7611` |
| Complete queue and external plan execution | 41 | `e30bea5ede7864f676389b273cfa7026c806c3966dc9e0f6a9d909eb7708814e` |
| Only G1 single image and trace recovery followed by full recovery | 40 | `534b44bed27037ef76067474aed73caba36386e3b1f607231d58434d8621456f` |
| Recovery of G1+G5 single image and trace followed by full recovery | 40 | `032f31d621e3160c4b2ef376cdbfea2c395996c212ef2dd60355e320c73f7d19` |
| Recovery of G1+G2 single image and trace followed by full recovery | 40 | `a8852a5b9bc1df7c829f9a56a8f3e42ed1a3330127bcd844754639ba31f9f4db` |
| Final FP16 comparison run | 32 | `d7c5b8d2cd3c9407d342018a736f56e9d66f57c82067163de14f8d7824458506` |

All packages are verified against board-end ZIP size and hash, and against exact receipt list and file hashes. The saved queue plan copy differs from the original external plan by only 189 CRLF/LF line endings, with identical JSON values and types; the request retains the original plan SHA. The aggregation tool separately distinguishes original bytes from reserialized copies; actual budget, count, path, type, and identity changes are still rejected. Pre-fix tool versions, derived error checkpoints, and documentation have been archived, and the original receipt remains unaltered.

USB network card disappearance was encountered again in this round; after re-plugging, the startup number remained unchanged, thus cannot be described as a restart. All builds and final single-image tasks have completed. The current startup re-run with FP16 comparison shows two predictions: the main output matches the previous comparison bit-by-bit, indicating that the current GPU execution and post-processing pipeline can properly handle the FP16 model. However, internal numerical issues within the quantization engine cannot be ruled out.

Evidence directory is `../artifacts/gate05/20261019_restore_extended`, `checkpoint.json` is `EXTENDED_RESTORE_LOCAL_CHECKPOINT_COMPLETE`. Independent checks, source files, and format validations have been preserved; Gate 0.5 remains unestablished.

## # Additional checks on logs and floating-point formats

The logs of the two completed quantization builds do not contain explicit records of dynamic range variation or missing scale; it is explicitly documented that no calibrator is used in precision mode, thus the title `Calibration: Dynamic` cannot be misinterpreted as evidence of random calibration. Attention MatMul and Softmax operations show FP16 input and output in q_all, q_only_G1, and FP16 comparisons. Visible floating-point formats differ in some decoding and final Sigmoid stages, indicating floating-point and fusion interference in comparisons; these records are insufficient to uniquely explain zero detections. Read-only diagnostics are saved in `../artifacts/gate05/20261019_native_range_dtype_diagnostic`, and no modifications to the graph, parameters, or conclusions are made based on this.

The fixed image annotation contains only one sea cucumber target; historical FP32 and FP16 comparisons both match this target. No additional scallop predictions appear in this single-image annotation, thus the phrase 'two detections' cannot be interpreted as 'two correctly annotated targets'. This round does not allow inference about the completeness of the entire dataset annotation.

Image and package: `../artifacts/gate05/20261019_precision_candidates`, `20261019_precision_runtime_bundles/`, `20261019_restore_extended_input_v3/`. Tools and independent audits are preserved in corresponding artifacts subdirectories; actual original results include `20261019_precision_native/`, `20261019_qall_extended/`, and `20261019_restore_extended/`.

# # Protocol stopping point

Native checks currently only address build validity, constraint adoption, fixed single-image execution, and format facts, without defining thresholds for dataset success. Before initiating evaluation on 150 images, the following must be determined:

1. How to handle failure of full model numerical migration; whether to establish an aligned reference implementation and independent object-level migration acceptance criteria.
2. Control mechanisms for non-quantized floating-point precision and fusion across engines; the common output declaration alone is insufficient to eliminate these factors.
3. Before reviewing dataset results, freeze the PASS / FAIL / INCONCLUSIVE criteria.
4. The quantized engine produces clearly abnormal numerical outputs on the fixed image. Select the next diagnostic route and define how evidence of strict integer computation in the seven depthwise convolutions will be accepted. A successful build, INT8 tensor formats, or kernel names cannot substitute for numerical correctness or missing evidence.

Original failures and all time-limited stop records are retained. Engineering checks have been frozen; no fixed dataset accuracy, recovery, bidirectional attribution, or performance conclusions are currently established. Further dataset trials, backend changes, graph structure modifications, or redefinition of transfer criteria are part of the next-phase planning and are not executed in this round.

Final board-level checks confirm fans are active, jtop is active, NRestarts=0, and 15W mode remains at 2; approximately 5 GiB of available memory is present; all processes for this task and single-graph runs have terminated. Privileged EMC parameter reads are still subject to permission restrictions and no system modifications were made. The original final state and file hashes are preserved in `../artifacts/gate05/20261019_final_board`.

Temporary SSH authorization has been revoked: only one public key associated with this task was removed, all other lines are preserved byte-for-byte and backed up; the authorized_keys file permissions remain 0600. A reconnection using the same key, BatchMode, strict host key verification, and USB source address resulted in authentication rejection, confirming that the temporary key is no longer valid for login. The revocation record and verification are stored within the final state directory. User passwords were not saved to scripts or logs.

The independent phase audit for this round is conducted in `../artifacts/gate05/20261019_stage_completion_audit`, verifying the completeness of six configurations, original failures, and uncompleted research items. This round completes the authorized current engineering phase, **not Gate 0.5 acceptance**.
