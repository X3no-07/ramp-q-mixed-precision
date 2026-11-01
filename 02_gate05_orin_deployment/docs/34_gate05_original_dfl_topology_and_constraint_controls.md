# Gate 0.5: Original Topology and Constraints Comparison



2026-10-20, continuing from [DFL Internal Context Localization](33_gate05_internal_dfl_context_diagnosis.md). In previous DFL small-graph corrections, NONE successfully built, and OBEY reproduced the `filterQDQFormats::4421` assertion of the full graph. This round supplements the original FLOAT Clip→Q→DQ topology for the NONE and OBEY groups, checking whether this local failure also occurs in the original structure.

**Result: The original structure also successfully builds NONE and reproduces the same assertion under OBEY. The local OBEY failure does not require the code_clip modification to appear; simultaneously, the original NONE output exhibits severe numerical deviations, while the corrected NONE shows minimal error. Compilation compatibility and numerical anomalies must be treated separately; 'being able to build' cannot be equated with 'correct results'. The full model remains uncorrected, and Gate 0.5 has not passed.**

# # Frozen Scope and Preparation

Frozen protocol `../artifacts/gate05/20261020_dfl_original_protocol/protocol.json`, SHA256 `81200cd4900a35f0533c45c7542f6dacc353203e8338a09b47cd24465cc0f6ed`. The original tail graph SHA256 is `b60be2c2d196ed1fb4e962446f374d9e53d066476172cfa1db415194b2306d90`, derived from the original q_all graph `2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d`, consisting of 9 nodes, 7 initializers, and one DFL Conv.

Continuing to use the previously authenticated internal logits fixture, NPY SHA256 `cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab`, FLOAT `[1,16,4,8400]`; using the same fixture's original and modified CPU outputs, NPZ SHA256 `fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511`. No re-execution of the upstream 72 Conv or local CPU is performed in this round; no new inputs are selected. The output remains the DFL distance from `[1,4,8400]`, not detection boxes.

The original structure does not perform code_clip import attribute correction; Clip alpha/beta are read-only and preserved without change; parser existing precision/output type constraints are retained. Each group is built once, with a 300-second time limit and a 900-second supervisor guard; only successful engines are executed twice; individual Conv inputs, weights, outputs, and tactics are checked, without assuming INT8 output. Other settings including INT8+FP16, disabled TF32/PREFER, workspace 1024 MiB, avgTiming=1, FLOAT/LINEAR are consistent with the previous stage.

Static reuse checks passed: the nodes, initializers, and I/O declarations of the original tail graph correspond exactly to the complete original source graph. The fixture, CPU NPZ, and raw bytes of both saved CPU outputs are unchanged. Additional CPU/ORT/Torch/upstream executions all total 0. The sealed archive `../artifacts/gate05/20261020_dfl_original_probe.zip` contains 11,545,669 bytes, with SHA256 `6ea5c52c4697339692d3d5e4efac01a46d033e34855a3ed8f60e4aa1d1a22709`; the spec SHA256 is `99d7b3179d9c540c06331a2135de9d645b2421722edb20ff5268e53c7b56f1ae`, and the manifest SHA256 is `aa8aa071beb59e40e48f3c0022fcf187f430285b1651df200d7c554623f1d064`.

Pre-launch independent review passed 350 items with no blockers; report `../artifacts/gate05/20261020_dfl_original_review/pre_native_review.json` SHA256 `62158ccefba1f732f88904bce8cf92c24f0992ec9dc706988770dfc78482a5ed`. After board-side repackage verification and pure standard library input checks, launched in a new directory `/home/nvidia/rampq/gate05_dfl_original_20261020/gate05_dfl_original_probe_20261020`. Monitor PID 33046, private group 33051, guard PID 33057 (WATCHING).

# # Results of the original structure in this round

The original structure was built twice, with no re-run of CPU or failed cases.

| case | Builder API seconds | Result | inference |
|---|---:|---|---|
| original_none | 52.320608 | Build succeeded, engine size 31,054 bytes | Two finite outputs with identical raw bytes |
| original_obey | 1.508866 | `filterQDQFormats::4421` / `!n->candidateRequirements.empty()`, followed by `buildSerializedNetwork::751` | No engine, no raw, no execution |

Runner total time: 64.027084 seconds, monitor total time: 69.3 seconds; `BUILD_SCRIPT_OK` indicates batch completion as per protocol, **not implying both builds passed**. Guard ended in COMPLETE state with no intervention, private group empty, no timeout or resource termination. This duration is solely for diagnostic build recording and does not reflect inference performance.

The actual original Clip alpha/beta in both groups was `−0.9999907612800598 / +0.9999907612800598`, identical before and after, explicit correction set to false. The code has no alpha/beta setter. Each group parsed the network as 14 layers, with 2 DQ precision INT8 constraints and 2 Q output type INT8 constraints, no query errors; layer precision or output type settings were neither set nor cleared. The parser request remained unchanged before and after OBEY was set.

The only actual DFL Conv in original NONE: input `Four wide channel vectorized row major Int8 format`, weight INT8 (16), output `Row major linear FP32`, no bias, activation NONE, named tactic `ampere_fp32_icudnn_int8x4_128x32_relu_interior_nn_v1`. It matches the previous round's correction of NONE in input, weight, output format, and tactic name; **same format and tactic does not imply numerical equivalence**.

Original NONE engine SHA256 `583aaa40a0a5d90e87cf00fd8be7dd08dfe90eed72d5a678e369c4f0cfb8fa3e`, raw array SHA256 `11be9724a648a8a04535876fa1e79e7ef1468c4eb463866313fca0b58f5247dd`. Output range from 0 to 0.9448731541633606, with 33,566 out of 33,600 elements differing from the frozen CPU reference: maximum absolute difference `11.957853466272354`, mean absolute difference `2.3782942463764005`, RMS `2.8428724297345527`. Therefore, the original NONE build success does not represent numerical correction.

# # Local four-cell comparison with the preceding round

The four-grid shares the same internal fixture, CPU reference, DFL weights, and scales. The saved CPU outputs for original and modified runs are byte-for-byte identical. The actual configurations in two stages are consistent except for OBEY; explicit Q/DQ precision requests have the same semantic meaning; the original and modified structures are composed with the declared import correction as a combined factor.

| Structure | OBEY | Build | Relative CPU maximum absolute difference |
|---|---|---|---:|
| Original FLOAT Clip → Q → DQ | Disabled | Successful | 11.957853466272354 |
| Original FLOAT Clip → Q → DQ | Enabled | 4421 assertions of the same type, no engine | Not comparable, no output |
| Corrected Q → INT8 Clip → DQ | Disabled | Successful | 1.9073486328125e-06 |
| Corrected Q → INT8 Clip → DQ | Enabled | 4421 assertions of the same type, no engine | Not comparable, no output |

The two NONE outputs differ by 33,566 elements, with a maximum absolute difference of `11.957853466272354`. All raw cross-group comparisons involving failed OBEY are `UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED`, and cannot be filled with zero differences or identical outputs.

The logs for both original and modified OBEY clearly include the same source location and candidateRequirements assertions, matching the historical error categories in the corrected q_all. This evidence does not support the explanation that the local error was solely introduced by code_clip modifications, but it fails to isolate specific subfactors such as Softmax adjacency, individual Q/DQ requests, tensor shapes, or FP32 outputs. It also does not establish the complete model's failure node or a unique root cause.

Cross-stage saved array audits are `topology_flag_comparison.json`, SHA256 `88fcc89eaec465dc1ed5493834898ba76c16925d0a1364c785482848e01453bb`. This is four sequential observations under the same board startup and fixed input, without repeated random trials; no tolerance was set for smaller errors, and the original numerical guardrails were not modified.

This round forms a cross-stage local topology × OBEY comparison with the existing corrected DFL group, preserving original failure and success records. The code_clip topology and its declared import attributes remain combined factors, not treated as independent factors; each grid undergoes only one fixed-order observation, and no extrapolation to random repeated trials or a unique root cause across the full graph is made.

# # Original Evidence and Final State

The original archive `../artifacts/gate05/20261020_dfl_original_native/native_evidence.zip` is 11,644,748 bytes, SHA256 `9f027ada5086843e4b967a6ab4f5a4e1d5f23bc904e30b9913e8e619db7dd96d`. The receipt lists 84 original files plus the receipt metadata itself; all entries, sizes, SHA256, and ZIP CRC pass, and the receipt SHA256 is `faa8a17c485a6551439027608457258e767e28184bee4ba6595f019af2162571`. The limited download succeeded once in this round, and the final SSH verification confirmed that all 84 board-side files and the archive match.

Root audit `root_native_audit.json` SHA256 `91a4f2b67125cf2868a2353b44e0bacbef53e82be5e3c613a613e5df8ba7ed6c`; 470 independent reviews passed, `independent_native_review.json` SHA256 `3cda350ad9140c2a9b442f853868e8d527f269b20889f61ec55e36942f7ef7c0`; supplementary failure signatures and final runtime state reviews `independent_failure_signature_and_runtime_review.json` SHA256 `5f0f63da900ff6e9d6041c12cff92cbd0c6b2a0ccfe598ee4e9fb2bc0bf1ce35`. Neither the new nor the old raw trees were modified; no CPU, engine, or SSH execution was performed during the audit.

The 35 monitoring samples recorded a maximum temperature of 50.312°C and a minimum available memory of approximately 2263.852 MiB; no kernel failure markers were logged. This is merely a resource record from this round and cannot be extrapolated to long-term board stability or the resolution of prior disconnections.

The final client recorded at 2026-10-20 12:37, while the board-side timestamp remains at 2026-10-17 18:48, both uncorrected for time. The startup number remains `692c7544-b8bd-415b-939c-b11cdbd9f7f0`; the monitor/runner/guard and two builders (PIDs 33046, 33051, 33057, 33064, 33096) are all absent, with no build tasks. Fans and jtop are active, NRestarts=0, available memory is 5001 MiB; a non-administrator read warning for EMC permissions persists under 15W/mode 2, indicating incomplete privilege validation.

The previous round's corrected DFL NONE engine remains `a0b23d8fce8a25bbf6ca5de9bf638dabbc9be11db23e56ad16cb2efc3ce2a33e`, with no engine present in its OBEY. The prior fully corrected q_only_G1 engine remains `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`; the fully corrected q_all still has no engine. All parent-stage input packages, raw data, checkpoints, original images, weights, and scales are preserved.

# # Conclusion and Next Steps

The local four-grid analysis now distinguishes two types of issues: both DFL structures exhibit the same compilation failure in OBEY; in the executable NONE branch, only the declared correction combination significantly reduces the numerical deviation on this fixed input. We cannot therefore conclude that the full graph OBEY has been fully fixed. We also cannot skip the original complete PyTorch/ORT guardrails, actual precision audits, floating-point and fusion comparisons, or independent dataset criteria. Gate 0.5 remains at `INCOMPLETE`, with no full-graph retry, dataset evaluation, real-world object-level causality, or performance pass conclusions.

The next step is to retain the numerically superior correction of DFL, and to freeze the 8-node slice following Softmax—QDQ→Conv→Reshape—with correct probability inputs. Then compare NONE versus OBEY, and examine the role of explicit Softmax adjacency context. It is necessary to generate new probability fixtures from previously saved logits via the original exact Softmax; the new local CPU output must match the existing DFL CPU output byte-for-byte; no re-execution of the upstream 72 Conv layers is required. This new slice, input, and native configuration have not yet been executed, and it is not permissible to directly feed logits into a version stripped of Softmax, without clearing parser constraints, nor to automatically generalize the result to the full graph or recover the matrix.

The static source and pending frozen steps for the 8-node / 7-initializer slice are found at `next_control_plan.json`, SHA256 `de9e44fcd058cd6d855174af6934f657e5c312ee5284f8463322c0c029160be2`. The plan itself does not equate to freezing or execution; changing input boundaries alters the optimization context, and future results cannot independently prove that Softmax is the sole cause of the full failure.

Stage checkpoint: `../artifacts/gate05/20261020_dfl_original_native/checkpoint.json`; current status: `../artifacts/gate05/current_execution_state.json`.
