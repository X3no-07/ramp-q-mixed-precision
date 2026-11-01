# Gate 0.5: Localization of DFL Weight Quantization Branch



In this round, only the explicit WeightQ/WeightDQ of DFL is bypassed; original probability inputs, activation Q→Clip→DQ, source weight bytes, and four-dimensional output boundaries are retained. Both NONE and OBEY candidate groups are successfully built and executed twice, with consistent output bytes; actual Conv inputs, weights, and outputs are all in FP16. Therefore, this is a local control result of the weight quantization branch and its parsing/formatting context—**not a W8A8 repair or Gate pass**.

Different from the six-node graph in Report 36, this candidate removes the mathematical operations of weight quantization and must use its own new CPU reference. The original full PyTorch/ORT guardrail remains at `FAILED_UNCHANGED`; the failure of the complete q_all OBEY in the historical full corrected q_all is not covered; Gate 0.5 remains at `INCOMPLETE`. A separate control for the complete q_all NONE build has been independently frozen, and its result does not belong to this report.

# # Source Freezing and Unique Change

The complete corrected q_all source image SHA256 is `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee`. The parent six-node graph SHA256 is `805abb78ed4e4d5abc8041b2916dadc83370a14ef1776fcc0a03f163c16bcf81`, and the parent completion checkpoint SHA256 is `c5e4a11a37d32a6de9df22c42463de0a6a58ad4b92c97b64f4c318f71dd9947e`.

The new graph only removes `rampq_model_23_dfl_conv_WeightQ` and `rampq_model_23_dfl_conv_WeightDQ`, and reconnects `/model.23/dfl/conv/Conv`'s `input[1]` from `rampq_model_23_dfl_conv_w_dq` to its original initializer `model.23.dfl.conv.weight`. After restoring this input, the serialized byte size of the Conv sequence matches that of the source node; the three preserved activation nodes, five in-use initializers, and I/O declaration all retain their original byte sizes. The original floating-point weight `[1,16,1,1]` maintains its value within 0 to 15. The old weight scale/zero is retained solely as evidence of the source and is no longer connected to the candidate graph, nor is it used for activation or output scaling.

The candidate consists of 4 nodes, 5 initializers, and 1 Conv. The input remains the existing CPU probability `/model.23/dfl/Softmax_output_0`, FLOAT `[1,16,4,8400]`; it reuses the NPY SHA256 `5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520` without regeneration. The output remains `/model.23/dfl/conv/Conv_output_0`, FLOAT `[1,1,4,8400]`, representing DFL distance and not decoded into detection boxes.

Protocol `../artifacts/gate05/20261021_dfl_weight_bypass_protocol/protocol.json` is frozen before preparation and execution, with SHA256 `9b2ea26ff12ecd5a4b5a699b09c511a3ea6c2187125d037117bb85f57b7b1170`. Two cases are `weight_bypass_none` and `weight_bypass_obey`, each built at most once, for 300 seconds, with a batch duration of 900 seconds, successfully executed twice. Both cases are set to INT8+FP16, without TF32/PREFER, 1024 MiB workspace, avgTiming=1, DETAILED, external FLOAT/LINEAR; only the Clip property is corrected to ±127, without setting or resetting layer precision or output type.

# # New candidate CPU reference

The new four-node graph CPU is executed once only, using CPUExecutionProvider, ORT_DISABLE_ALL, and 4 threads; no re-execution of Softmax, upstream 72Conv, old DFL, or the full detector is performed. All 33,600 float32 elements in the new output are finite. The new expected NPZ contains only `weight_bypass_output`, derived from the actual output of this candidate run, not from the old six-node reference.

| File | SHA256 |
|---|---|
| New four-node ONNX | `020dad7f70c04f495dfe1b6e62e63e925ff6468a6cde0c157b7122a38274eb41` |
| New CPU actual NPY | `808e4a1a17e2f9fc622d858a3dba8390e45749a72afb0ac6a8fd3d8ac413cf73` |
| New expected NPZ | `dff4725f1dbbb99979723b47267973f9cde5bc390db30131ad5d39361d958526` |

The new candidate CPU and the saved old six-node W8A8 CPU output differ in 33,592 out of 33,600 elements, with a maximum absolute difference of `0.05133557319641113` and an average absolute difference of `0.019477537332845497`. This is a direct record of the semantic alteration caused by bypass weight quantization; the protocol does not require byte-level equality between old and new, nor does it adopt the old reference or introduce any tolerance.

73 source identities are maintained before and after preparation; the new native tool preserves identity throughout packaging. The package `../artifacts/gate05/20261021_dfl_weight_bypass_probe.zip` is 11,664,531 bytes, SHA256 `a86af92bdce3b72408c84f2236fe6f495baa5366d20574e97f045f1cfd1eb13b`; spec SHA256 `df7de4468cec5cb0672f89c61e63ce26decbb1d6bc52a521e453e02aad9c5b4f`, manifest SHA256 `af7e0bb5db03cd228d05df912cf4298f19ad2715a476f7c1757200fbaf1b8d3a`.

301 independent static reviews are passed prior to startup, with SHA256 `../artifacts/gate05/20261021_dfl_weight_bypass_review/pre_native_review.json` `4f314e7457fa3a4ad572323988db23f875baebe38573d815b1b9f43e4b41c623`. Pure standard library input checks pass on the board, with no execution runtime imported.

# # Native build, actual format, and original output

Two groups are each built once, resulting in two engines and four fixed-input executions, with no change in settings or retry. Builder API and monitor times are recorded only for diagnostic purposes and do not reflect inference performance.

| case | Builder API seconds | Engine bytes | Same-group two outputs | Actual Conv input/weights/output |
|---|---:|---:|---|---|
| weight_bypass_none | 55.092813 | 32,247 | Byte-consistent, all finite | FP16／Half×16／FP16 |
| weight_bypass_obey | 54.787531 | 30,475 | Byte-consistent, all finite | FP16／Half×16／FP16 |

The NONE engine SHA256 is `3259a455db211ea42d7c8a92b801369bf1727dd99f922c1ded2a002bddf75dfd`, and the OBEY engine SHA256 is `6efe9963f608ee3bc3b7f57b8c9a88c650188434ecc4a8e0093ddc18fa6144e2`. Both groups uniquely map to `/model.23/dfl/conv/Conv`, with kernel/stride of `[1,1]`, Groups=1, OutMaps=1, no bias, Activation=NONE.

NONE is `CaskConvolution`, with internal inputs and outputs in two-wide FP16, and tactic is `ampere_fp16x2_hcudnn_fp16x2_128x32_relu_interior_nn_v1`. OBEY is `CudnnConvolution`, with internal inputs and outputs in linear FP16, inspector is not provided `TacticName`, and `TacticValue=0x0000000000000038` is retained. This is a field absence, not a build or inference failure, and no tactic name is synthesized. The two external bindings remain FLOAT/LINEAR; external FLOAT outputs cannot be interpreted as internal Conv operations computed in FP32.

Both analytical networks consist of 6 layers, with an actual retention of 1 DQ precision INT8 and 1 Q output type INT8, no query errors; Clip corrections before and after remain consistent, and the requests are identical across both cases. The 2 + 2 requests from the parent six-node graph naturally decrease upon removal of weight Q/DQ nodes, without manual constraint clearing. The actual flags differ only in the OBEY switch. The INT8 builder flag, explicit activation Q/DQ, or source FP32 initializer cannot substitute for actual operator precision evidence; the actual Conv in this candidate is FP16, failing to meet the original W8A8 computation requirements.

Relative to the CPU reference of this candidate, the saved array validation is as follows. All values are descriptive and no new tolerance is defined.

| Comparison | Different elements / 33,600 | Maximum absolute difference | Mean absolute difference |
|---|---:|---:|---:|
| NONE vs New Candidate CPU | 33,594 | `0.010239601135253906` | `0.0007279089507868603` |
| OBEY vs New Candidate CPU | 33,594 | `0.00620269775390625` | `0.0004970305783319331` |
| NONE vs OBEY | 8,452 | `0.0078125` | `0.0005338795979817709` |

Runner took 123.352584 seconds, monitor took 129.53 seconds. `BUILD_SCRIPT_OK` indicates protocol-compliant batch completion; both groups succeeded independently verified by their respective engines, build results, and raw arrays. Guard status is COMPLETE, no intervention, private group is empty, no resource termination or timeout occurred.

# # Evidence Integrity and Reproducibility Scope

The original archive `../artifacts/gate05/20261021_dfl_weight_bypass_native/native_evidence.zip` is 12,003,589 bytes, SHA256 `5353ea2f980c5b9e0a50f63cb947c737551e5bd52978970f2f8ab282a0444c0c`. The receipt lists 93 files, plus the receipt itself; archive CRC, exact file set, size, and SHA256 are all verified. Receipt SHA256 is `b490e6ed56b928d430e43076da2f6a2d53d84b3204f0483fd147246a7a019875`.

Root audit `root_native_audit.json` SHA256 `20a386e6b359608356fbd6ccf61b4d11bba1fedb56e89471ccdbd571c9a2995d`. The reproduced evidence file is `independent_native_review.json`, with 519 checks passed, SHA256 `9bf0cf635c69bbf5a4d08045e645d24cc1a121cc65a56ab5ec2e81a4d2494baf`. This reproduction was performed by the same agent from the current preparer/runner/builder, already disclosed in the record: it performs character-by-character validation and re-computation of saved evidence, not independent review by an external author or final arbitration. Key conclusions still require validation by the root task or other reviewers. The reproduction did not import ORT/torch/TensorRT, did not introduce new CPU/native inference or SSH, and the original tree was not modified.

The root task also obtained the latest on-board observation: client time 2026-10-21 02:03:54, on-board time 2026-10-18 08:14:50, no time synchronization; the launch number remains `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. The PIDs of monitor, runner, guard, and two builders (40168/40173/40179/40186/40222) are all absent, indicating no task processes; the 93 receipt files and archive still match. Fans and jtop are active, NRestarts=0, available memory is 5006 MiB. Non-administrator power queries show 15W/mode 2, and there is still an EMC read permission error, thus cannot be considered a complete privilege power validation.

This observation also confirms that the engine SHA256 of corrected q_only_G1 is `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`, and the engine SHA256 of the parent six-node NONE is `961e821b6d2965078edc8209feb2b41929012a6522df027b71a2f83834056836`; the OBEY node still lacks an engine, and the historical corrected q_all remains without an engine at observation time. The final state is saved in this round `final_board_observation.json`, separated from the subsequent complete graph control record.

# # Conclusion and Subsequent Control of the Complete Graph

The OBEY node in the parent six-node graph fails under identical local probability inputs; the new candidate, with explicit Q/DQ weight removal, succeeds in both groups. This change supports further investigation of the weight quantization branch and its associated parser and format selection context, but simultaneously alters the mathematical semantics, inherent request set, and actual computation path—thus cannot be uniquely attributed to any single request or node. In particular, the actual Conv is now FP16, and the build success cannot be considered a repair of the original W8A8 complete model.

No automatic extension of the next local pair is performed. The independent NONE control of the unchanged corrected q_all has been frozen at `../artifacts/gate05/20261021_qall_none_protocol/protocol.json`, SHA256 `572120502a6db021945c63cdd781d81e2a4b1771f5d9a13c3f58d0da7278eac1`: the original source bundle's OBEY specification and contract bytes are preserved, and a single NONE build is now described by the outer control; existing complete graph CPU equivalence evidence is reused, with new CPU value at 0. This protocol requires actual precision validation after build, followed by freezing a fixed image inference protocol—no automatic inference or dataset access is allowed. This report does not prematurely declare the complete graph control compatible or successful.

SASS instructions or kernel strings within static libraries may be used for future planning, but without execution association or actual path validation, they do not constitute runtime arithmetic pass for this model. The original complete guardrail, previous OBEY failure, seven strict INT8 tactic gaps, and Gate `INCOMPLETE` are all retained.
