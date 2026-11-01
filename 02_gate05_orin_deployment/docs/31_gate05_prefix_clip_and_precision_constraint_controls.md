# Gate 0.5: Prefix Clip and Precision Constraint Comparison



2026-10-20, following [Full INT8 Fix Candidate Failure](30_gate05_all_int8_corrected_candidate_validation.md). This round uses the previously executed single-convolution prefix to differentiate Clip structure and OBEY flag effects on local construction and numerical behavior, without re-running the full model.

**Result: All four arms successfully built and executed twice; under the same Clip structure, the raw outputs with and without OBEY are byte-for-byte identical. The format error in full q_all has not been reproduced on this single Conv context, thus cannot be attributed to OBEY. The rewritten structure significantly reduces numerical deviation on this fixed input, yet still has 7 elements differing by one quantization step; Gate 0.5 remains unpassed.**

# # Frozen Scope

The prefix includes 11 nodes and 12 initializers: the first activation Clip/Q/DQ, weight Q/DQ, the initial Conv, SiLU, and the second activation Clip/Q/DQ. The only output is `rampq_model_1_conv_a_dq`, with shape `[1,16,320,320]`. The second Conv is not present in the graph and thus cannot be referred to as a dual-convolution prefix.

Inputs come from the original fixed `000020.jpg`; reuse original images, modified images, preprocessed NPY files, and saved CPU references, without re-calibration, re-training, or image selection. The original full PyTorch/ORT guardrails are preserved `FAILED_UNCHANGED`; original full INT8 failure and only G1 quantization success records are retained.

Protocol `../artifacts/gate05/20261020_prefix_flag_protocol/protocol.json` is frozen before native execution, SHA256 `c7fdcada89c685ed52f8f65d8318854982d58f52f16cfce44f3129cdd17a307d`.

| Order | Clip structure | OBEY |
|---|---|---|
| original_none | Original floating-point Clip → Q → DQ | Disabled |
| original_obey | Original floating-point Clip → Q → DQ | Enabled |
| code_clip_none | Q → integer Clip → DQ, with boundary correction | Disabled |
| code_clip_obey | Q → integer Clip → DQ, with boundary correction | Enabled |

All other settings remain the same: TensorRT 8.5 Python Builder, INT8 + FP16, TF32 and PREFER disabled, 1024 MiB workspace, avgTiming=1, detailed inspector, external FLOAT/LINEAR. No layer precision or output type setters are used, and the parser's automatic constraints are not reset. For each case, layer constraints before and after correction are saved along with actual configuration flags. The number of script setters cannot be taken as the total number of network constraint applications.

Each case attempts only once, with a 300-second build budget; the total monitoring time across four cases is 1500 seconds. After saving compilation errors, the next frozen case is attempted; if a task times out, exceeds resource limits, or is externally interrupted, subsequent tasks are halted and private process groups are cleaned. Only successful engines are validated, actual layer information is saved, and two identical inputs are executed. Failed engines are not executed.

# # Pre-validation

The SHA256 of two original prefix images are respectively:

- original:`22b1065c01792492d658596c1069aee074e054f32648fa90bf87cc39a6d11a33`.
- code_clip:`d2b8ca8105f8a05fabb7aac66c6fbab1a2558ac35bdd5ea1736ce8e0052465de`.

The SHA256 of the original preprocessed NPY is `24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94`. Independent reading validation confirms each image consists of a single Conv layer, and the saved CPU expected outputs are byte-for-byte consistent; no CPU inference is re-executed for this run.

The current Windows USB network interface is 'Ethernet 3', with address still `[DEVICE_HOST]`; USB SSH has been restored in real time, and the board startup number remains `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. During preparation, fans are active, jtop is active, and NRestarts=0; available memory is approximately 4936 MiB, with no pending build tasks. No system, power, or clock settings have been changed.

# # Current Run Results

The final sealed package uses `../artifacts/gate05/20261020_prefix_flag_probe_v2.zip`, 1,071,567 bytes, SHA256 `7a6a1d88a50f9db72ac48aeed3a0c2923068031f956393b5fb5c4997f32fcc96`; spec SHA256 `66b1cf24fb5667fa92fe31c6e8ca483d63eef4495b6c4cc53a120f8ec7e28056`, manifest SHA256 `aa1959c7f4b4569f5129533fc51b3f669bb659f51d351438e0a6840547feee19`.

Evidence files indicating abnormal interruptions discovered during independent inspection of the first version package require correction; the original package is preserved and not deployed. Version v2 uses a separate `runner_abort.json`, without overwriting the failed child original; the hash relationships between the two versions and the source code are saved in `20261020_prefix_flag_bundle_revision_lineage.json`, with no modifications to the frozen images or protocol.

245 pre-validation items passed independently; report `../artifacts/gate05/20261020_prefix_flag_review/pre_native_review.json` SHA256 `bade9ebf0401c0b208bf1b2d9f871915bf41482bf1f42a287e6af959c0b0d10d`. After re-verifying the package, monitoring tools, and current processes on the board, four cases are launched in a new directory `/home/nvidia/rampq/gate05_prefix_flag_20261020/gate05_prefix_flag_probe_20261020`.

monitor PID 30071, private group 30076, guard PID 30078. Each group attempts once and successfully builds and executes twice. Builder API and Director monitoring times are listed below—these are diagnostic times, not inference performance measurements.

| case | Builder API seconds | Build and two executions | Relative freeze CPU reference per element count / 1,638,400 | Maximum absolute difference |
|---|---:|---|---:|---:|
| original_none | 103.033 | Success, repeated output byte-by-byte identical | 769,777 | 36.65729832649231 |
| original_obey | 102.935 | Success, repeated output byte-by-byte identical | 769,777 | 36.65729832649231 |
| code_clip_none | 68.616 | Success, repeated output byte-by-byte identical | 7 | 0.7481081485748291 |
| code_clip_obey | 68.935 | Success, repeated output byte-by-byte identical | 7 | 0.7481081485748291 |

Runner took 362.849 seconds, outer monitor took 368.42 seconds, returned `BUILD_SCRIPT_OK`. No timeout, resource termination, or guard intervention; private group ultimately empty. All seven recorded processes—four builders, one runner, one monitor, and one guard—exited successfully. All results are finite.

The raw outputs of both original and rewritten structures have SHA256 hashes of `27b5aa8940d2dd83edb62d270e427f82c0f8dfdd9387ab2638b6790e2b76ca6c` and `15a6b790c26da42840120bde810c10c03e74a7bf19bb685c6552c967ebfbb2db` respectively. This comparison is based on contiguous raw byte arrays, not whether serialized engines are identical. The outputs of both structures also match byte-for-byte with previously saved prefix raw outputs of the same structure.

The frozen CPU references for the two structures were already byte-identical. The maximum of the 7 native discrepancies after rewriting equals the second activation-boundary scale, `0.7481081485748291`. Native and CPU execution therefore cannot be described as fully identical, and no new passing tolerance was introduced on this basis.

# # Actual Precision, Fusion, and Parsing Constraints

All four inspector instances reported exactly one actual convolution, with INT8 inputs, weights, and outputs; 432 weight counts, 16 output channels, group=1, corresponding to the source Conv. External inputs and outputs remain FLOAT/LINEAR.

All four groups used the same named integer tactic:

```text
sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_vect_c_32_tilesize256x32x64_stage4_warpsize4x1x1_g1_tensor16x8x32_t1r3s3_linkable
```

Activated as `GENERIC`, the per-point expression contains 5 operations, with Tanh used to express SiLU; this differs from the dedicated `SWISH` description of the first Conv in the complete q_only_G1 repair engine, and cannot be substituted by previous fusion conclusions from the full engine, as the current actual recording must be used.

Each group of parsing networks consists of 18 layers, featuring 3 explicit INT8 precision constraints and 3 explicit INT8 output type constraints, with no unsupported or erroneous queries. After clip attribute correction, per-layer constraints remain unchanged. For the same topology none/obey, the graph hash and parsing constraints are identical, with only `OBEY_PRECISION_CONSTRAINTS` configuration flags differing; PREFER is always disabled. The three parsing constraints are not introduced by script setters: the script contains no layer precision, output type setters, or reset operations.

# # Original Evidence and Final State

Original archive is `artifacts/gate05/20261020_prefix_flag_native/native_evidence.zip`, 2,219,829 bytes, SHA256 `a66da15dad97b1a807376b8281fa19457d749ae036ef6cad88b67113171cb8b6`. The receipt lists 107 original files, plus the receipt metadata within the archive; file list, size, SHA256, and ZIP CRC are all verified. Receipt SHA256 is `344e92d945d4d8697f3f76c056e5941a5e0dc900688d650766710a3a221734e3`. Final confirmation via SSH verifies that the 107 files on the board still match, and the archive hash remains unchanged.

Independent native review passed 530 items, see `independent_native_review.json`, SHA256 `6e136e642346a90c0a210b098f8072791b26ab5a25863b70241fadedd5875205`. Root audit `root_native_audit.json` has SHA256 `1351ee0ad9caa40888e716452dd0a4f767793693f9d8f14867a7a26134a45492`; actual format, constraints, raw arrays, and evidence before and after are simultaneously verified.

The first two local failures of the root audit and corresponding source code are stored in `root_audit_attempt1/` and `root_audit_attempt2/`, not native failures or re-runs. The first missed field was the finite element count; after correction, the second revealed a cross-structure RMS discrepancy between Windows and board of only `2.220446049250313e-16` (between `1.3701950732015873` and `1.370195073201587`). All raw bytes, finiteness, element count, and maximum difference are consistent. The final audit saves both ends' mean/RMS and their difference side-by-side as descriptive floating-point reductions, without using their last-byte consistency as a new precision criterion; original numerical guardrails and original evidence remain unchanged.

At final board observation, the client timestamp was 2026-10-20, while the board clock remained 2026-10-17—both recorded separately, with no time synchronization. The launch number remained `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. Fans active, jtop active, NRestarts=0, and 5018 MiB of available memory; no running tasks in this round. These are observations from this run and do not prove that prior disconnections have been resolved. Power consumption query shows 15W / mode 2; non-administrator access to EMC permissions is still indicated, and it was not written as a full privilege check passed.

The full q_only_G1 engine remains SHA256 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`; the corrected q_all directory still lacks an engine. This round did not cover these cases, nor did it attempt a full model retry.

# # Explanation of Boundaries

All four cases succeeded, indicating that the failure of the full q_all engine was not reproduced in this prefix context. Enabling OBEY alone in this single Conv context is insufficient to trigger the error. It is therefore not valid to conclude that the full model can be built, nor can we rule out interactions between OBEY and other layers, branches, or fusion contexts.

Numerical comparisons only describe raw arrays, repeated consistency, actual tactics, and fusion differences; no new tolerance thresholds are introduced, and they do not replace the original full guardrails. No head detection, GT matching, dataset, target recovery, or performance conclusions are detected. Gate 0.5 remains `INCOMPLETE`.

Next, we will confirm and freeze a new minimal graph context slice from the actual failure case of corrected q_all, first verifying dependencies, shapes, and CPU semantics, then perform a bounded native validation. Any new Conv, branch, or residual layer should be recorded as a new context; errors are not guaranteed to appear monotonically with prefix length, and the failure at the first added node cannot be directly identified as the root cause. Currently, no full graph switch to NONE/PREFER is performed, parsing constraints are not cleared, full model re-run or matrix recovery is not initiated, and no evaluation on 150 graphs is entered.

This closed-loop cycle is seen in `artifacts/gate05/20261020_prefix_flag_native/checkpoint.json`, and the current queue status is seen in `artifacts/gate05/current_execution_state.json`.
