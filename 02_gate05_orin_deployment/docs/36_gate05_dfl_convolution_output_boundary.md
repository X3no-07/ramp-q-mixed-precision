# Gate 0.5: Boundary Localization of DFL Convolution Output



This round's check: the 8-node graph without Softmax from the previous round still triggers `filterQDQFormats::4421` under OBEY. Now only the terminal Constant and Reshape are removed, with quantization and convolution preserved; we observe whether errors still appear at the Conv output boundary.

Six-node comparison completed: NONE construction and two successful executions; OBEY still encounters the same `filterQDQFormats::4421` / `buildSerializedNetwork::751` error during construction, with no engine and no inference. It is evident that explicit Softmax and terminal Constant/Reshape are not necessary conditions for this local error; the root cause of the full model remains unestablished. Original complete PyTorch/ORT guardrails remain at `FAILED_UNCHANGED`; complete corrected q_all still has no engine; Gate 0.5 remains at `INCOMPLETE`.

# # Frozen Scope

- Precise source: complete corrected q_all ONNX SHA256 `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee`.
- Source and reference inherit completion checkpoint from Report 35, SHA256 `142eb219ccbeea141237cd1a89ecdb2c47c70b99d35159d9912ba8cea767c53f`; no modification to existing packages, failure records, scales, or weights.
- New graph: 6 nodes, 7 initializers, 1 source Conv. Remove `/model.23/dfl/Constant_1` and `/model.23/dfl/Reshape_1`; all other nodes and initializers retain original serialized bytes.
- Input remains the existing CPU probability `/model.23/dfl/Softmax_output_0`, FLOAT `[1,16,4,8400]`; NPY SHA256 `5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520`, not regenerated.
- New output is `/model.23/dfl/conv/Conv_output_0`, FLOAT `[1,1,4,8400]`, representing DFL distance with single-channel dimension, not decoded into bounding boxes.
- Only one new graph CPU execution is added. First validate dtype, shape, and 33,600 elements between the old `[1,4,8400]` and new `[1,1,4,8400]`; then compare C-order raw array bytes. The new four-dimensional expected is derived from the old array via pure reshape and preserved, with file identity recorded; no requirement for entire NPY file identity when shape headers differ.
- `code_clip_none` and `code_clip_obey` each built once, each limited to 300 seconds, batch timeout capped at 900 seconds. Two successful engine executions; failure results in no inference; ordinary compilation failures proceed per protocol; batches are terminated by timeout, resource exhaustion, external interruption, or input mismatch.
- INT8 + FP16, no TF32/PREFER, 1024 MiB workspace, avgTiming=1, DETAILED, external FLOAT/LINEAR. Only clip alpha/beta to ±127 as per existing protocol; no layer precision or output type setting or reset.

Protocol is frozen before new graph preparation and CPU/native execution: `../artifacts/gate05/20261020_dfl_conv_output_protocol/protocol.json`, SHA256 `8f6a55f640915b84d7300d1595ebb895092ccf0374e40a27b6aaf812f406e09b`.

# # Execution Record

Unique CPU validation passed: CPUExecutionProvider, ORT_DISABLE_ALL, 4 threads; only executed the new six-node graph once, without generating new probabilities or rerunning Softmax/72Conv/old DFL/complete graph. The new four-dimensional and old three-dimensional output raw array bytes are identical, with 0 differing elements; raw SHA256 `3e4280d6a97b12d4e0c33eeb1e0a462ab3d457cef8eca9ce2ac5554816f45436`. Old NPY SHA256 `fd4bcdcee24cd53651fdd7a508423c802f3b432c759647204c7b748f22abd50d`, new NPY is `d2f4327fcc792b864e420f1b1554a5b7b52b88fd27a8f01f6c07a0159c3e723b`; file shape header differs, not a numerical change.

New graph SHA256 `805abb78ed4e4d5abc8041b2916dadc83370a14ef1776fcc0a03f163c16bcf81`. Expected NPZ SHA256 for the new four-dimensional output is `711fb41fa18bef712bd069c88a552e1bc918d1d1034b6f6d32d802cbd769d0c6`, containing only `code_clip_output`. Preparing 72 source files before and after, and preserving the two native tool versions accordingly.

Package `../artifacts/gate05/20261020_dfl_conv_output_probe.zip` is 11,908,335 bytes, SHA256 `78008952bd6355d4a197913c669c3a7b18ff1e80b119d88789d19f53baed4ba7`; spec SHA256 `9e9d31b55e4358d39a08a7869aab77582f094f1d6152599c727196e2b77c3897`, manifest SHA256 `104137a621b3ea5c65a019555a1d92fc8c876fe0bf9293b5ab7cfe2e6fe71f77`.

Independent pre-launch review of 369 items passed entirely, no blockers; SHA256 `../artifacts/gate05/20261020_dfl_conv_output_review/pre_native_review.json` `a3fa1038648c560045bd79d77a5dac027d16ceb1c04bdd1ce7e486d578f97747`. Pure standard library input check passed for native runner; no runtime imported.

Temporary SSH key expired during waiting period, then restored via interactive password login; only installed USB-originated IP restriction and a two-hour valid task key. After restoring access, all source identities were re-verified before uniquely launching the two native builds for this round; no re-execution of CPU or re-packaging was performed.

Pre-launch SSH observation: still launch number `692c7544-b8bd-415b-939c-b11cdbd9f7f0`, no lingering build processes; fans and jtop active, NRestarts=0, approximately 5005 MiB of available memory. Non-administrator queries under 15W/mode 2 still show EMC read permission prompts, not considered a full privilege power check. Client and board clocks were recorded, no clock synchronization performed.

# # Native Results and Output Boundary Comparison

New directory is `/home/nvidia/rampq/gate05_dfl_conv_output_20261020/gate05_dfl_conv_output_probe_20261020`. Monitor PID 38997, private group 39002, guard PID 39008; two builder PIDs are 39015 and 39049. Each case built only once, with no setting changes or retries.

| case | Builder API seconds | Result |
|---|---:|---|
| code_clip_none | 50.380678 | Engine succeeded, 27,583 bytes; same probability input executed twice, output is finite and byte-consistent across runs |
| code_clip_obey | 1.523681 | Assertion 4421 and error 751 observed during build phase; no engine, no raw, no inference |

Runner total time: 61.965597 seconds, monitor total time: 67.28 seconds. `BUILD_SCRIPT_OK` only indicates protocol-compliant batch completion and saving of ordinary compilation failures, **not that both builds passed**. Guard status is COMPLETE, no intervention, private group empty, no timeout or resource monitoring termination. These build and monitoring durations do not represent inference performance.

The parsed boundary network has 11 layers, while the old eight-node boundary has 12 layers. Named Q/DQ requests remain unchanged: 2 DQ precision INT8, 2 Q output type INT8, no query errors; Clip corrections before and after are preserved. Actual configuration is consistent except for the OBEY switch; layer precision or output type was not set or reset.

NONE unique actual Conv corresponds to `/model.23/dfl/conv/Conv`. Input `Four wide channel vectorized row major Int8 format`, 16 Int8 weights, output `Row major linear FP32`, activation NONE, no bias; tactic remains `ampere_fp32_icudnn_int8x4_128x32_relu_interior_nn_v1`. External probability input and four-dimensional distance output are both FLOAT/LINEAR.

NONE engine SHA256 `961e821b6d2965078edc8209feb2b41929012a6522df027b71a2f83834056836`, output array raw SHA256 `f503e64fd4f24ffde83dd60e62be2dc7e832b54d311c5f564eabebb07a831d84`. Relative to the current four-dimensional CPU reference, 11,779 out of 33,600 elements differ, maximum absolute difference `1.9073486328125e-06`, average absolute difference `7.795968226024083e-08`, RMS `1.664742957045647e-07`; only for descriptive purposes, no new tolerance established.

Old eight-node NONE output `[1,4,8400]` and new six-node NONE output `[1,1,4,8400]` are identical in kernel shape, dtype, and element count; C-order array bytes are identical with 0 differing elements, maximum difference 0. Both rounds reused the same probability file, weights, scales, Q/DQ requests, and general settings. Both OBEY outputs are absent; raw comparison is `UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED`.

Different from the previous round, the native external input files in this round are indeed identical between old and new; the change lies only in the output boundaries. This still does not imply equivalence of all internal tensors or that the full graph has been repaired. A change in output contract may affect the selection of native format, and cannot be uniquely attributed to a specific node based on a single local comparison.

Root audit `root_native_audit.json` SHA256 `cc6ff251cfe20b339c5d68ef9fd3d46d0de4f0f848ff33131673ceb7f6f0133e`; evidence comparison `context_comparison.json` SHA256 `c02da4ed3cb4a37c8664243616eb744b53e431b94f052c25ad683ee69a8b9e1d`; no re-execution of the old engine was performed.

# # Original Evidence and Final Board State

Original archive `../artifacts/gate05/20261020_dfl_conv_output_native/native_evidence.zip` is 12,152,044 bytes, SHA256 `07e583b3aac44cebd4008a6c0f1fda83000a9a71c4e940ae1c6bb63f3baa63cd`. The exact receipt lists 89 files, plus the receipt metadata itself; file list, size, SHA256, and ZIP CRC are all verified. Receipt SHA256 `7aaa362c975ceaa3c3fb3d26e6c2362a6107c65c77f7b54081b47420b4051350`. A limited-speed download succeeded, followed by SSH re-check of the 89 files and archive on the board, which still matches.

Final client time is 2026-10-21 01:04, board time is 2026-10-18 07:15, with no time synchronization; the startup number remains unchanged. monitor, runner, guard, and two builders with five PIDs are all absent; no build tasks are present. Fans and jtop are active, NRestarts=0; 4994 MiB of available memory; non-administrator EMC query restrictions are preserved. `final_board_observation.json` SHA256 `87dd6299f84a39fa281fd9eb8b15ed8ada47fbbd60243969fdc3082bb6bea209`.

The eight-node NONE engine and earlier corrected DFL NONE engine from the previous round are maintained; both still show no engine in OBEY. The complete corrected q_only_G1 engine SHA256 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476` remains unchanged; the complete corrected q_all engine still shows no engine. All original numerical failures, original full guardrails, and historical strict INT8 gaps are preserved.

Independent review of 623 pieces of evidence passed entirely; report `../artifacts/gate05/20261020_dfl_conv_output_native/independent_native_review.json` SHA256 `f648aed6ad95e8f6d8d10c35b59cdebee6dd42a48f895792f600e34e021b18c4`. Coverage includes original files, graph source, actual format/constraints/flags, original outputs, and final board state; no new CPU or native inference or SSH was found during the review.

# # Next-step local control recommendation

In subsequent steps, weight Q/DQ can be bypassed separately: freeze only the input probability, maintain activation Q→Clip→DQ, output contract, and original FP32 weight bytes, and reconnect Conv weight inputs back to the source initializer. This allows distinguishing whether the explicit weight quantization branch was necessary in this local failure. The new graph alters mathematical semantics and parser request context; a new freezing protocol must be established, a separate CPU reference must be saved, and the old six-node W8A8 reference or claims of numerical equivalence cannot be reused; an FP32 initializer does not imply that the engine's actual weights are computed in FP32.

Only the static suggestion `../artifacts/gate05/20261020_dfl_conv_output_native/next_control_plan.json` is preserved, SHA256 `124615758af8e5a45a074effbbc335a3fbba4de2e37352252a2d58fcba8d5a61`. No candidate graph was generated in this round, no new CPU or native execution was performed; no new output Q/DQ, scaling, or calibration was introduced.

The checkpoint and checklist for this round are saved at `../artifacts/gate05/20261020_dfl_conv_output_native`; the checkpoint is `checkpoint.json`. Completion here refers only to bounded diagnosis and evidence closeout in this round; Gate 0.5 remains at `INCOMPLETE`.

# # Conclusion Boundary

While removing Constant and Reshape nodes, external output shapes and native format selection, as well as fusion boundaries, are also altered. Even if the results change, the change cannot be uniquely attributed to a single removed node, nor can it be claimed that the full model has been repaired. Actual Conv inputs, weights, outputs, and tactics are separately recorded; there is no prior requirement for INT8 output.

CPU probability only reflects the frozen input at a local level and does not represent validation of the upstream distribution on native. No full graph retry, parser reset, old graph re-run, dataset evaluation, object-level causal recovery, precision or performance pass has been conducted.
