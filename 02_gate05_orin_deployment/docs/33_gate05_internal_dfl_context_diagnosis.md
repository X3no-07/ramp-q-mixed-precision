# Gate 0.5: Internal Context Localization within DFL



2026-10-20, continuing from [Double Conv Context](32_gate05_two_convolution_context_diagnosis.md). Both image prefixes previously failed to reproduce the corrected q_all formatting failure. This round selects a structurally different DFL: Softmax → QClipDQ → 1×1 Conv → Reshape, comparing the same corrected topology between NONE and OBEY.

**Result: NONE successfully constructs and executes; OBEY triggers the same `filterQDQFormats::4421` / `candidateRequirements` assertion as the complete corrected q_all within approximately 1.54 seconds, without engine or inference. This is the first time a similar failure has been reproduced in a 9-node DFL slice, and it isolates the switch-related differences in this context. The specific failure node and unique cause in the full model remain unproven. Gate 0.5 is not passed.**

# # Frozen Scope and Internal Input

The DFL slice contains 9 nodes, 7 initializers, one `/model.23/dfl/conv/Conv`, with input `/model.23/dfl/Transpose_output_0` FLOAT `[1,16,4,8400]`, and output `/model.23/dfl/Reshape_1_output_0` FLOAT `[1,4,8400]`. The output is a DFL distance tensor, not a detection box. There is no output QDQ here, and thus we cannot pre-require the actual Conv output to be INT8.

Freezing protocol: `../artifacts/gate05/20261020_dfl_context_protocol/protocol.json`, SHA256 `b2723549e6acee51c86aeba3b72c8451601ba4add6088f46f7443536a1ab74ee`. The actual failure source image SHA256 is `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee`.

The internal input cannot be directly used from image NPY. In this round, we precisely extract `images` from upstream dependencies of the DFL Transpose in the corrected ONNX graph, which includes 638 nodes, 576 initializers, and 72 Convs. This is executed once on CPU using the preprocessed NPY of the original fixed image to generate a logits fixture. This runs a significant portion of the backbone and regression head upstream, but does not produce full detector outputs. Both the original and modified DFL tail versions are run once on CPU, sharing this fixture, requiring byte-level consistency in local outputs. This does not prove equivalence between the original and corrected upstream, nor does it represent the upstream input distribution in TensorRT.

Each native case is attempted exactly once, with a build upper limit of 300 seconds and a director guardrail of 900 seconds; after success, the same internal input is executed twice. All parser constraints and real configuration flags are retained, with only one Clip alpha/beta adjusted to ±127. Layer precision and output type are neither set nor cleared. Actual Conv inputs, weights, outputs, and tactic are recorded as per inspector logs, not substituted by request flags.

The upstream subgraph SHA256 is `df9e2ea0d03168f17f770bf0689a4a75c90f607df75ff98962e4eaa92764c018`. Only one CPU upstream generation fixture is actually executed, NPY SHA256 is `cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab`, the original array SHA256 is `799c5f7875edd7f9a3e46c8beba833b69b56baf04f76f060015dab2132cbb52a`; range from −20.61652374267578 to 17.923843383789062, shape and boundedness are passed.

Original and modified DFL tail are executed once on CPU, with output `[1,4,8400]` byte-for-byte identical and maximum difference of 0, original array SHA256 `3e4280d6a97b12d4e0c33eeb1e0a462ab3d457cef8eca9ce2ac5554816f45436`. CPU uses `CPUExecutionProvider`, `ORT_DISABLE_ALL`, and 4 threads. The corrected tail graph SHA256 is `a8a30c71a20d6b6dd3d031d23ce8dc8672cc62abe6c3eb8725b324fd2e600092`; byte identity of all selected source nodes, initializers, and image NPY before/after is preserved.

Sealed package `../artifacts/gate05/20261020_dfl_context_probe.zip`, 21,224,122 bytes, SHA256 `05b9b50e4ddc294151121044facef5e1711ac368c793ca5232c1c1a557c42b9f`; spec SHA256 `934e075aa58dde1829b600d3d011f5ad51c4f1da1af78f7791944f492781b14d`, manifest SHA256 `3836e82ed53afb14fe16d3b9431864c0c4a45fd0d3418c7c06a4f6fcd46ed1d0`.

The independent pre-launch review passed 2,845 checks with no blockers; report `../artifacts/gate05/20261020_dfl_context_review/pre_native_review.json`, SHA256 `c4ff2428c1ef6b291538d1fbb188502e2fa298ae977ab826b4ab5c8332a36ea4`. After repeating standard-library-only input validation on the device and checking that no build task remained and the boot ID was unchanged, execution started in the new directory `/home/nvidia/rampq/gate05_dfl_context_20261020/gate05_dfl_context_probe_20261020`. Monitor PID 32201, private process group 32206, guard PID 32212; guard status WATCHING.

# # Native Results

Each case is strictly attempted once, with no retry of failed cases or change of settings.

| case | Builder API seconds | Build Results | Fixed Input Execution |
|---|---:|---|---|
| code_clip_none | 52.211651 | Success, engine 31,119 bytes | Twice, output is bounded and byte-for-byte identical |
| code_clip_obey | 1.538202 | Returns an empty serialized result, internal TensorRT assertion | No engine generated, no inference, no raw output |

Runner took 63.993636 seconds, outer monitor took 69.3 seconds, final state `BUILD_SCRIPT_OK`. This state indicates that the script saved a standard build failure and completed the batch according to the frozen protocol, **not that both engines passed**. The guard ultimately reached COMPLETE, with no intervention, private group empty; no timeout or resource guard stop. The above durations are not inference performance measurements.

OBEY log line 14 matches the complete failure log line 275 as follows:

```text
[optimizer.cpp::filterQDQFormats::4421]
Assertion !n->candidateRequirements.empty() failed.
All of the candidates were removed, which points to the node being incorrectly marked as an int8 node.
```

Then appears an empty engine assertion of `buildSerializedNetwork::751`. The complete log does not specify a node name; the small-scale reproduction cannot reverse-verify that a named node in the full graph is the failure point.

# # Actual Precision and Numerical Values

The NONE engine has exactly one actual Conv, uniquely corresponding to `/model.23/dfl/conv/Conv`, with 16 weight elements, no bias, and NONE activation. The actual format is:

- Input: `Four wide channel vectorized row major Int8 format`
- Weights: `Int8`
- Output: `Row major linear FP32`
- Naming tactic: `ampere_fp32_icudnn_int8x4_128x32_relu_interior_nn_v1`.

Inputs, weights, and the integer naming tactic have been individually verified. Outputs are recorded as FP32. This aligns with the current protocol that does not predefine INT8 outputs; no QDQ is present, and it should not be rewritten as 'all inputs and outputs are INT8'. External inputs and outputs remain FLOAT/LINEAR with shapes consistent with the original agreement.

NONE produces two identical raw byte outputs. Engine SHA256 is `a0b23d8fce8a25bbf6ca5de9bf638dabbc9be11db23e56ad16cb2efc3ce2a33e`. Relative freeze of CPU DFL outputs:

| Description metric | Value |
|---|---:|
| Number of differing elements | 11,779 / 33,600 |
| Maximum absolute difference | 1.9073486328125e-06 |
| Mean absolute difference | 7.795968226024083e-08 |
| RMS | 1.664742957045647e-07 |

This is only descriptive numerical difference across the same internal input. No new tolerance threshold or original guardrail has been defined. OBEY produces no output. The value comparison between NONE and OBEY across groups is correctly recorded as `UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED`, and cannot be filled with zero difference or identical outputs.

# # Actual Configuration and Parser Constraints

Two groups, internal fixture, clip ±127 correction, and parser constraints are identical. In the actual configuration flags, only `OBEY_PRECISION_CONSTRAINTS` changes; INT8 + FP16 is enabled, TF32/PREREFER is disabled, workspace is 1024 MiB, avgTiming=1, DETAILED, and external FLOAT/LINEAR remain unchanged.

Each arm parses the network 14 layers, retaining 2 explicit INT8 precision constraints (ActivationDQ, WeightDQ) and 2 explicit INT8 output type constraints (ActivationQ, WeightQ), with no errors in query. Clip imports attribute correction shows identical per-layer constraints before and after correction, with no layer precision\/output setter or reset. It is incorrect to write 'the script did not set constraints' as 'the parser has no constraints'.

An additional read-only check was performed on the parser metadata of the historical full graph: the setting flags, set types, and output tensor names/shapes/dtypes for these four DFL Q/DQ layers match the semantic meaning of the current small graph. The JSON schema of the two recorders differs; a direct dictionary comparison was first performed to generate a local audit assertion. After preserving this record, a re-verification was conducted via explicit field mapping, without running CPU/native or modifying original evidence. Corresponding `full_parser_constraint_correspondence.json` SHA256 `9a3dcc476ee59b3bad0040da1744969f7ce41c0e2bf095cd709535b8033f48d8`. This only strengthens the correspondence of constraint sources and does not prove which constraint caused the compilation error.

This run was performed once in a fixed sequence. It demonstrates that under the recorded conditions, the DFL correction context results in a successful NONE run and a similar failure in OBEY, but has not yet isolated the effects of Softmax adjacency, code_clip topology, import property correction, or individual Q/DQ constraints.

# # Original Evidence and Final Board State

Archived as `artifacts/gate05/20261020_dfl_context_native/native_evidence.zip`, 21,481,431 bytes, SHA256 `a53750401ccdbb7c008c14e75cadec80a586a05f2baa32f5e985abdff527bd24`. The receipt lists 88 files, plus the receipt metadata itself; size, SHA256, and ZIP CRC have all been verified; receipt SHA256 `fe71bbf62229b969bcfc8331b06921991803847668daddb43fb831465aec25e2`. A final SSH verification confirms that all 88 files and the archive match on the board end.

Root audit `root_native_audit.json` SHA256 `507c1902886fac7490927f0f97a754a114c0a6d3637948a0aada05ed2d3247f4`; independent saved evidence review passed 309 items, `independent_native_review.json` SHA256 `18c4dbb38ea7f4f01f9906bad8d347fe198f301f7999598baf341e1bef213b46`. No re-inference or alteration of the original tree occurred in any review.

The first SCP download encountered a connection reset during both experiment and archive completion, resulting in an incomplete 21,430,272-byte file and a failure record, which were saved separately. After confirming the board-end startup was identical and the archive hash remained unchanged, a single limited-speed re-download was performed, which ultimately passed with precise validation. No re-constructed or failed experiment was run, and no incomplete ZIP was used as evidence. The specific cause of this transmission interruption remains undetermined.

The final client observation time was 2026-10-20 12:18, while the board-end timestamp remained 2026-10-17 18:29, with both recorded independently and without time synchronization. The startup number remains `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. The PIDs of monitor, runner, guard, and two builders (32201, 32206, 32212, 32219, 32300) are all absent, indicating no current build task. Fans and jtop are active, NRestarts=0, and available memory is 4989 MiB. A 15W/mode 2 query still shows non-administrator EMC read permission prompts, and no full privilege check pass conclusion is drawn.

The original complete q_only_G1 engine SHA256 remains `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`; the complete corrected q_all failure directory still lacks an engine. The original failure, weights, calibration scales, CPU/PyTorch guardrails, and prior successful runs are all preserved.

# # Conclusion and Next Control

The complete model's identical compilation assertions have now been reduced to a 9-node DFL tiling, and the OBEY switch-related differences are observed within the same corrected graph. This is a local compilation reproducibility; it does not establish the sole root cause of the full graph, does not prove that disabling the OBEY switch in the full model can successfully build, nor has it completed real-device object-level destruction/recovery, dataset, or performance validation. The original complete PyTorch/ORT numerical guardrail remains `FAILED_UNCHANGED`, and Gate 0.5 remains `INCOMPLETE`.

The next step prioritizes reusing the previously archived FLOAT Clip→Q→DQ DFL tail graph, leveraging the current fixture and CPU reference, while freezing the original structure's NONE and OBEY control groups, forming a local topology×switch comparison with the current results. The original structure does not undergo code_clip import attribute correction; all other settings, parser constraints, and actual precision audits are preserved; it still treats the code_clip topology and declared import correction as combined factors and cannot directly distinguish between them. The original tail graph SHA256 is `b60be2c2d196ed1fb4e962446f374d9e53d066476172cfa1db415194b2306d90`, and no re-execution of the upstream 72 Conv or re-selection of inputs is required.

This plan is only proposed and not yet frozen; see `next_control_plan.json`, SHA256 `5b0469faa1f055b2e3e148c1302c65824a04f31b062abd48fbf109b70040aa7d`. If Softmax is later removed, a correctly frozen Softmax output fixture must be established; the current logits cannot be directly fed into the new tiling. There is currently no reset parser constraint, no full-graph NONE/PREFER retry, no PE jump, no recovery matrix, and no evaluation of the 150-graph.

Stage checkpoint: `artifacts/gate05/20261020_dfl_context_native/checkpoint.json`; current status: `artifacts/gate05/current_execution_state.json`.
