Gate 0.5: Validation of INT8 Candidate Repair Candidates



2026-10-19,follows [complete reference divergence localization](29_gate05_full_reference_numerical_divergence.md). The current q_all candidate has passed static and full CPU rewrite equivalence checks, and the Clip attribute at board 88 has been corrected; however, TensorRT encountered an internal format filtering error after approximately 13 seconds, failing to generate the engine. The failure has been fully recovered and independently verified, without rerunning the full model. The original complete PyTorch/ORT numerical guardrail remains unmet; the previous q_only_G1 success is not overridden, and no further recovery matrices or dataset evaluations have been initiated.

# # Current Reference Protocol

`../artifacts/gate05/20261019_reference_contract/reference_contract.json` was frozen prior to native build, SHA256 `87fcbf04e88fd311784736cf77cddfdd82fc2608ed3de19882467e58707fb430`. 17 items passed independent method review, with review report SHA256 `8c8e46719852b18f5181edb36c3edb5be888d2063491b3076d9d8aaf6e23092d`.

The protocol retains four distinct roles: the original Gate 0 PyTorch Q/DQ serves as a historical simulation reference; the original ONNX CPU acts as a graph rewrite control; the hybrid from the previous round using unified Conv/SiLU is solely for discrepancy localization; the current TensorRT serves as the new diagnostic engine. These components cannot substitute for one another. The protocol does not modify the pass condition of the original Gate, nor does it reframe the convergence result of the hybrid as a pass for the original PyTorch/ORT.

The same fixed image, frozen weights, calibration, and original NMS are used in this round. Before native execution, static difference ranges must be accurate, and CPU outputs must be byte-by-byte identical across primary and auxiliary runs, with zero audit errors, input kernel validation, and independent code review passed. After native execution, the actual format, named integer tactics, and fusion thresholds used remain unchanged and unmodified.

If the actual 88-layer format, mapping, and fusion are fully intact, but evidence for named integer tactics is incomplete, only the existing explicit `--diagnostic-formats-only` branch is allowed to execute a fixed graph, preserving the strict integer evidence as incomplete. Gaps in the new engine must be individually re-verified and cannot replicate the previous seven-layer results, nor can they exempt them.

# # Candidate and Original Byte Verification

The parent graph is the original q_all plus PE auxiliary output graph, SHA256 `2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d`. The new graph has 795 nodes and 703 initializers; it modifies only 176 Clip/Q nodes and 176 Clip bound initializers. The 88 Conv layers, all DQ, scale/zero, weights, biases, other nodes, and external input/output declarations are preserved in original byte form.

The original topology `FLOAT Clip→Q→DQ` is changed to `Q→INT8 Clip[-127,+127]→DQ`. The native builder must explicitly correct the alpha/beta values of the 88 Clip nodes to -127/+127 during ONNX parsing, before construction, and must save the measured attributes before and after the modification.

The parent spec's precisionConstraints are `obey`, and the new candidate retains it; 88 INT8 and 0 recovery requests for FP16, thus no explicit half-precision layer constraint is imposed. The workspace is 1024 MiB with avgTiming=1 maintained. The new per-inference budget is explicitly set to 7200 seconds, while the original parent's 1800 seconds and prior stop recording are preserved; this is because the old q_all extended build actually required 3276.426 seconds. The previous extended build used trtexec without explicit precisionConstraints and thus cannot be considered a compilation baseline for this round.

The initial preparation halted due to overly strict static assertions on floating-point division, with 22 instances of `float32((scale*127)/scale)` not correctly equaling 127. All original bounds and `float32(scale*±127)` byte-level consistency are preserved, and the quantized integer values are accurately ±127. The preparer has thus revised the new static checks without altering any tolerances in the original full model. The failure records and individual values are fully retained.

| File | SHA256 |
|---|---|
| Candidate ZIP, 10,443,020 bytes | `f7b3b24ea613265e6712b4edc265556e760de61645958dbbce159f96070b6de3` |
| model.onnx | `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee` |
| spec.json | `f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb` |
| semantic_rewrite_audit.json | `842eff15adfedd75735815cbe4040d17a645653e3b62511149afacbf8f923be9` |
| manifest.json | `f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f` |

The 21 members of the local ZIP match the directory byte-by-byte, and 19 payload files in the manifest have passed SHA checks. The candidate package has been uploaded to the new board directory and re-verified for manifest integrity, without overriding the old graph or old engine.

# # Full CPU Original and Rewritten Graphs: Passed Equivalence Checks in This Round

The same fixed `000020.jpg` frozen preprocessing array is used. Execution is performed on the complete original ONNX and rewritten graph using CPUExecutionProvider, ORT_DISABLE_ALL, four threads, and float32.

| Output | Shape | Differing Elements | Original float32 bytes |
|---|---|---:|---|
| main output output0 | [1,14,8400] | 0 | exactly the same |
| PE auxiliary output | [1,128,20,20] | 0 | exactly the same |

The main output from the original image is byte-for-byte identical to the initial ORT output without PE tap. The SHA256 of the pre-processing array is `24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94`, and the SHA256 of the current main and auxiliary array package is `1fe8d81720679ad47fb6f6d7d47b3019dc2cbd7fccdc72ab6683f8a5d2ce705c`. Evidence is stored at `../artifacts/gate05/20261019_corrected_qall_cpu_equivalence`.

Here, the CPU equivalence on this input is rewritten via tiling modification. The original complete numerical guardrail remains PyTorch/ORT and must not be renamed or replaced. `FAILED_UNCHANGED`

# # Board-side Preparation and Current State

SSH has been restored. The temporary key is still restricted to the USB address of the computer and expires in two hours, with no stored login password. The board boot sequence number remains `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. No pending tasks were left during this round of preparation. The fan is active, jtop is active, and NRestarts=0. Available memory is approximately 5047 MiB. Board time and client time are separately recorded, with no clock adjustment, software upgrade, or change to the 15W mode. The non-administrator EMC permissions prompt is retained.

New native root directory: `/home/nvidia/rampq/gate05_corrected_qall_20261019/gate05_corrected_qall_candidate_20261019`. Out of 304 independent pre-start checks, all passed. After re-verification of inputs, tools, and CPU equivalence on the board, the single build for this round was initiated.

# # Native Build Result: Format Filtering Failed, No Engine Generated

monitor PID 23569, private build group 23574, supplementary guard PID 23575. The monitor normally observes the child process returning exit code 1, status `BUILD_SCRIPT_FAILED`, taking 13.07 seconds; the builder recorded a runtime of 10.5340614 seconds. It did not exceed the 7200-second timeout, and no extended budget run was performed.

The original error was:

```text
optimizer.cpp::filterQDQFormats::4421
Assertion !n->candidateRequirements.empty() failed.
All of the candidates were removed, which points to the node being incorrectly marked as an int8 node.
```

Subsequently, `buildSerializedNetwork::751` returned an empty engine, and Python stopped at `assert serialized is not None`. The log does not specify the exact source node that triggered this assertion. The strict tactic gap in the original seven deep convolutions cannot be directly attributed as the root cause of this failure, nor can it be asserted that OBEY was obeyed or that a specific Clip caused the failure.

The 88 Clip attribute records before and after parsing and correction, along with the build request, have been saved. Independent static re-examination confirms that all corrected alpha/beta values are -127/+127. The build request was indeed for INT8, FP16, OBEY, with PREFER and TF32 disabled. The note of '0 explicit layer precision constraints' only refers to the script not calling any layer precision or output type setter; there was no query of whether the parser itself introduced constraints. Therefore, it cannot be asserted that OBEY had no impact. The source image audit remains error-free, but these are parsing and input evidence, not actual engine precision passes.

The optimization log lists 57 Clips as independent POINTWISE plan layers and 31 Clips as fused into Conv plan layers. However, no engine was generated, so these logs cannot be considered evidence of actual execution precision or failure node localization.

No `model.engine`, no actual precision audit, no single-image execution. The existing explicit `--diagnostic-formats-only` branch is not currently usable and was not activated. The object-level recovery mechanism was not tested in this round, and the compilation failure cannot be interpreted as a mechanism non-reproducibility.

# # Failure Evidence and Final Board State

Failure package `artifacts/gate05/20261019_corrected_qall_native/build_failed_evidence.zip`, 10,532,288 bytes, 52 original files, SHA256 `027433baa5b1e18fafb72cf64b70e58e1bd2aa05b295fd8f539ae8317e1c9a99`. The receipt, ZIP CRC, byte count, and SHA of each file have been verified; receipt SHA256 `a1d7e456d7b005a16db01f199d32535e159799d3b60fa05b4881277c8f47fdd9`.

The highest temperature among 7 resource samples is 50.343°C, the lowest available memory is 4070.605 MiB, and swap space usage remains unchanged. The monitor reason is null, with no records of timeout, low memory, overheating, or USB kernel events triggered; kernel.log contains only the header of the log. The accurate conclusion is that the recorded direct failure is a TensorRT compilation error. These samples cannot rule out all unrecorded transient hardware issues.

Guard is `COMPLETE`, no intervention, final private group is empty. Final real-time verification: all three PIDs have disappeared, no residual build or inference tasks remain, the boot sequence number has not changed, the fan is active, jtop is active, NRestarts=0, available memory is approximately 5046 MiB, and no engine files exist. Record stored at `artifacts/gate05/20261019_corrected_qall_native/final_board_observation.json`.

Root audit `root_failed_build_audit.json` re-verified the source, 88 corrections, failure phase, and exit. Independent failure review passed 252 items, report `failure_independent_review.json`, SHA256 `bae8fad62b264e579b89859497ff53a2b82967fddc0b5aeb02761ea6944f2480`; the byte count before and after the original tree remains unchanged. All new review reports are outside the original tree.

# # Next Diagnostic Criterion

In a standalone, frozen parse-only metadata inspection, no Builder configuration was created, no build or inference API was invoked; only parsing of the same graph, constraint reading, and 88 Clip boundary corrections were performed in temporary memory. All 1275 queries to parsing layers succeeded:

| Parser State | Count and Source |
|---|---|
| precision_is_set=true | 176 Dequantize layers are INT8, plus 1 Shape layer is INT32 |
| output_type_is_set=true | 176 Quantize outputs are INT8, plus one corresponding Shape output is INT32 |

176 corresponds to 88 activation quantizers and 88 weight quantizers. Before and after correction, each of these two constraints is 177, and the per-item metadata remains unchanged. Thus, it is indeed incorrect to interpret "0 script setters" as implying no constraints on the network parsing; however, one cannot solely rely on this metadata to conclude that OBEY is the failure cause, nor can parsed types be taken as equivalent to the actual engine format.

Supplementary monitoring took 9.06 seconds, with no guard intervention and an empty private group. All 26 original files have been recovered and verified, with a ZIP size of 122,747 bytes and SHA256 `a567b474be0e563448ce96b7b5cc0f1e9bfff1036461b9512efe9b8c1ba32d78`; the files are stored at `artifacts/gate05/20261019_corrected_qall_native/parser_metadata_evidence/`, separated from the original full-model failure package.

Independent supplementary review passed 859 items, reporting SHA256 `parser_metadata_independent_review.json` and `bb78f349eb8784fe1374f34e777d8948e2eefb1fabd8858f0d18fea1c3ac55aa`; both tasks exited successfully; the final real-time record is at `final_board_observation_after_parser.json`. The board-end SHA of the previously successful q_only_G1 engine remains `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`, unchanged and not covered by this run.

The single full-model attempt under this protocol has concluded. The next step is to compare the original floating-point CLIP with the corrected integer CLIP, and within the same parsing context of 0 manual layer precision settings, evaluate the impact of enabling versus not enabling OBEY/PREFER. First, fix the parsing API, inputs, weights, and observed outputs to prevent simultaneous changes across multiple factors.

The actually successful fusion prefix consists of 11 nodes: one initial Conv + SiLU + second activation Q/DQ output, without the second Conv. The statement in the independent builder review suggesting a 'successful dual-Conv code_clip prefix' is overly strong; the original report is retained, and `next_probe_plan.json` is explicitly corrected and re-counted from the original ONNX. Any dual-Conv extension must be validated as a new slice, and cannot inherit the previous prefix success status.

Specific subsequent slices must be defined based on this log and the selected graph boundaries, not merely by the name of the last printed layer indicating a failure node. `next_probe_plan.json` is a non-executed plan; no new 2×2 engine build was run in this round.

If a small slice fails to reproduce the full error, it should be recorded as insufficient for root cause identification; success in such a slice cannot be used to claim full model constructability. If reproduction occurs, the original numerical semantics must be preserved and actual execution format must be verified before a new full-model attempt is reviewed. No hidden condition adjustments were made or retried in this round.

# # Not Yet Proven

No new complete numerical migration PASS, strict full INT8 PASS, dataset accuracy, object-level destruction-recovery, or performance conclusions have been established. The new common_preparation uses the audit identity from this rewrite; the previous FP16 control serves only as historical reference and descriptive comparison, not as a baseline for floating-point quantization or fusion comparison. The current diagnostic and failure evidence have been closed out; q_all native results are failure or deployment validation incomplete; Gate 0.5 remains at `INCOMPLETE`.
