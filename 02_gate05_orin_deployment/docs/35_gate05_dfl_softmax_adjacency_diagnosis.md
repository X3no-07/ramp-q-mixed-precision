# Gate 0.5: Locating Softmax Adjacency in DFL



2026-10-20, continuing from [original DFL topology and constraint comparison](34_gate05_original_dfl_topology_and_constraint_controls.md). Previously, both original and modified 9-node DFL topologies were successfully reproduced under OBEY `filterQDQFormats::4421`, and NONE configurations could be built; however, only the explicitly declared correction combinations significantly reduced numerical bias for fixed inputs. This round retains the corrected quantizer, with a cut made at the original Softmax output, comparing the new 8-node slice NONE/OBEY.

Result: After removing the Softmax, NONE still succeeds, and OBEY reproduces the same assertion 4421. The explicit Softmax operator is not a necessary condition for this local failure. The final DFL output of the new NONE matches exactly in byte-level with the original 9-node corrected NONE's raw bytes; this does not prove that the Softmax/probability components are identical internally, nor does it fully repair the entire model. Gate 0.5 remains unpassed.

# # Frozen Scope and CPU Combination Comparison

Freezing protocol: `../artifacts/gate05/20261020_dfl_softmax_cut_protocol/protocol.json`, SHA256 `bbe50e6485a72df194a13684895b4cc29a1db6ceca8ef098d05a6b5e56c56055`. Source image is the actual failure case corrected q_all, SHA256 `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee`.

Starting from the saved logits (NPY SHA256 `cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab`), passing through the original exact Softmax node, the CPU performs only one pass to generate a new probability fixture. This node has axis=1; its serialized SHA256 is `b3b4c7e2d24556983e61ae3a731e6fc82625f83a8b8c671b0e98596c7f258adc`. Subsequently, the 8-node slice is executed once on CPU, and the output must match exactly in byte-for-byte with the existing corrected DFL CPU raw array in order to be approved. This round does not re-execute the upstream 72 Conv layers or the old native versions.

The new input `/model.23/dfl/Softmax_output_0` is a FLOAT `[1,16,4,8400]` probability, and logits or images cannot be used as substitutes. The new slice contains 8 nodes, 7 initializers, and one `/model.23/dfl/conv/Conv`, with an output that remains a FLOAT `[1,4,8400]` DFL distance, not a detection box.

Two native cases, each run once for 300 seconds, with an outer total of 900 seconds. All parser precision and output type requests are retained; only the alpha/beta parameters in the original code_clip are corrected to ±127. Two successful engine executions with identical probability inputs are required; failure to infer is not considered a pass. Actual inputs, weights, outputs, and tactics are all audited, without pre-assuming INT8 output.

The preparation phase was previously interrupted; upon resumption, it was confirmed that the fresh package directory, CPU failure records, and board-side current round directory did not exist. The frozen protocol was reused along with a newly written script, and the initial execution of the agreed 2 CPU runs was performed. No prior inference was rerun, and no new input selection was made. After renewing temporary SSH access, the board remained at the same boot number with no pending tasks.

The CPU combination comparison has passed: one pass through the original Softmax to generate probabilities, one pass through the 8-node graph to produce output, matching exactly in byte-for-byte with the prior DFL CPU NPY `fd4bcdcee24cd53651fdd7a508423c802f3b432c759647204c7b748f22abd50d`. Probability NPY SHA256 `5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520`, original array SHA256 `d4c4b21e1d1a2e025d5b3eabaade7e99f3093c0cf7511795f9477a4ffd069ae7`, range `2.4000963110581318e-17` to `0.9994057416915894`. The maximum deviation of the axis=1 probability sum from 1 is approximately `2.0233e-7`, described only, with no newly introduced output tolerance thresholds established.

New slice SHA256 `1f8662a55675152d57d248b16c2eebec08c4d54b5b6ebc4841f0c177d6ac2298`; single-node Softmax-generated graph SHA256 `09bd1b2903cfa5208a54a15fb85c8f1c7074b4f655d9880b49b30bc7c70a1c06`. The nodes and initializers maintain their source serialization identity; CPU uses `CPUExecutionProvider`, `ORT_DISABLE_ALL`, and 4 threads.

Sealed package `../artifacts/gate05/20261020_dfl_softmax_cut_probe.zip` is 13,574,095 bytes, SHA256 `aeff82c52ff3b2ec1d83b4b06c21f014b30699a1e428a0c5fb37bcd3f6cb25dd`; spec SHA256 `45f284588d37e607478ce79e443d46a49b870ef0d65614c5ce0867130f50316b`, manifest SHA256 `6c456b5a08d34a9a3fb87d7fced5474f2c349d632076bfc0fb5eb8dd793ddde1`.

Independent review of 392 items prior to startup revealed no blockers, `pre_native_review.json` SHA256 `7f31020bf9ba1ce2b66a61cf96dfb59ba6b4094b75a496b156d238e9dcf4cd86`. The new directory is `/home/nvidia/rampq/gate05_dfl_softmax_cut_20261020/gate05_dfl_softmax_cut_probe_20261020`; monitor PID 37496, private group 37501, guard PID 37507.

Cutting out the Softmax also alters external input boundaries, native fusion context, and uses CPU-generated probabilities—thus, the observed output changes cannot be used to prove that the Softmax is the sole erroneous node in the full graph. The original full guardrail `FAILED_UNCHANGED`, the fully corrected q_all without engine execution, and Gate 0.5 `INCOMPLETE` remain unchanged; no full-graph retry, parser reset, matrix recovery, dataset, or performance pass conclusion has been established.

# # Native Results and Actual Precision

Two cases were built once each, with no changes to settings or retries.

| case | Builder API seconds | Build and Execution |
|---|---:|---|
| code_clip_none | 52.228163 | Engine successfully built, 29,765 bytes; two executions with identical probabilistic inputs produced finite outputs with byte-level consistency |
| code_clip_obey | 1.529954 | `filterQDQFormats::4421` / `!n->candidateRequirements.empty()`, followed by `buildSerializedNetwork::751`; no engine, no raw, no inference |

Runner took 64.168915 seconds, monitor took 69.31 seconds; the final state `BUILD_SCRIPT_OK` only indicates the protocol-prescribed saving of a standard build failure and completion of the batch — **it does not mean both engine builds passed**. guard status was COMPLETE, no intervention, private group was empty, no timeout or resource monitoring termination. The above durations are not inference performance.

The NONE engine has only one actual Conv layer, uniquely corresponding to source `/model.23/dfl/conv/Conv`, with 16 weight elements, no bias, and NONE activation. Input is `Four wide channel vectorized row major Int8 format`, weights are INT8, output is `Row major linear FP32`; tactic is `ampere_fp32_icudnn_int8x4_128x32_relu_interior_nn_v1`. External probabilistic input and DFL output distance are both FLOAT/LINEAR.

All parsed networks in this round consist of 12 layers; the previous 9-node graph had 14 layers. The four explicitly requested Q/DQ nodes remain unchanged: 2 DQ precision INT8, 2 Q output type INT8, no errors in queries; code_clip import correction preserves the request before and after. Clip alpha/beta are only adjusted to ±127 under the freeze protocol, without any layer precision or output type setter or reset. In actual configuration, only the OBEY switch differs; all other settings — INT8 + FP16, no TF32 or PREFER, 1024 MiB, avgTiming=1, DETAILED — remain consistent.

NONE engine SHA256 is `43d429867fdbd013b91dcac08e598610f1dad3c98b8b565c2dd7ba0d3ca2d514`, raw output array SHA256 is `f503e64fd4f24ffde83dd60e62be2dc7e832b54d311c5f564eabebb07a831d84`. Compared to a frozen CPU reference, there are 11,779/33,600 elements differing, maximum absolute difference `1.9073486328125e-06`, average absolute difference `7.795968226024083e-08`, RMS `1.664742957045647e-07`. These are merely descriptive metrics and no new tolerance thresholds were introduced.

# # Comparison with Context Including Softmax

| Corrected DFL Context | NONE | OBEY |
|---|---|---|
| Old 9-node graph, Softmax inside the graph, input logits | Success: actual INT8 input and weights, FP32 output | 4421 assertions of the same type, no engine |
| New 8-node graph, external CPU probability | Success: same format and tactic as above | 4421 assertions of the same type, no engine |

The final DFL outputs for two NONE cases are byte-for-byte identical, with 0 differing elements and a maximum difference of 0. Note that their external inputs are not identical: the original graph receives logits, with Softmax applied within the TensorRT graph; the new graph receives CPU probabilities. Identical final distance cannot be used to infer that all internal probabilities or quantization codes are the same, nor can it be generalized to other images. Two OBEY cases have no outputs; all raw comparisons involving them are marked as `UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED`.

Original weights, scales, CPU output reference, and named Q/DQ requests remain unchanged; native general configuration is consistent except for the OBEY setting. The conclusion of comparison is limited: even without an explicit Softmax operator, similar errors can still occur, and thus the previous localized failures cannot be fully attributed to the necessity of having Softmax. It also cannot prove that Softmax has no impact in other contexts, nor identify specific failure nodes within q_all.

Cross-phase evidence preservation comparison is `context_comparison.json`, SHA256 `e5cea499a0874a10644ba4909ea44c33b0a1e96183bbc3ca8b4ca9d35c9cf380`. Root audit `root_native_audit.json` SHA256 `0878cacc91734567c3f701fcda3800fa64505863ab1911bc9519afc208538f05`; both are array and log audits, with no re-execution of the old engine.

# # Original Evidence and Final Board State

Archive `../artifacts/gate05/20261020_dfl_softmax_cut_native/native_evidence.zip` is 13,823,424 bytes, SHA256 `1b531ae8079a7563432e9702071ca8d384695f2a2624d80ab3da7798fa7215cc`. The receipt lists 89 files, plus the receipt metadata itself; file list, size, SHA256, ZIP CRC are all verified, and the receipt SHA256 is `5538c67b45e6ba020bc095b4ad4eca8920e0cb8fdaf60b67d90a90e4b37d44b8`. Limited download succeeded once; final SSH re-check of board-side files against the archive matches perfectly.

Final client timestamp is 2026-10-20 22:00, board-side timestamp is 2026-10-18 04:10, both recorded without time synchronization. The launch number remains `692c7544-b8bd-415b-939c-b11cdbd9f7f0`. The PIDs of monitor, runner, guard, and two builders (37496, 37501, 37507, 37514, 37550) are all absent, indicating no build tasks. Fan and jtop are active, NRestarts=0, available memory is 4992 MiB; a 15W/mode 2 query still shows non-administrator EMC read permission prompts, indicating incomplete privilege validation.

Original correction DFL NONE engine `a0b23d8fce8a25bbf6ca5de9bf638dabbc9be11db23e56ad16cb2efc3ce2a33e` retained; OBEY still has no engine. Complete q_only_G1 engine `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476` retained; complete corrected q_all still has no engine. Both original structure numerical failures and full original integrity guardrails are preserved.

Independent evidence review of 597 items all passed; report `../artifacts/gate05/20261020_dfl_softmax_cut_native/independent_native_review.json` SHA256 `222617e5f6ec15c0dabb6f2e1ce697cd989359dfcf2d5914469c78f11a7326ad`. Revalidation covers 89 original response documents, image sources, parser requests, actual precision, output bytes, new and old contexts, and final states across five tasks; no new CPU/_native inference or SSH introduced. Entry and this report have undergone read-only consistency validation.

# # Next local control

Subsequent priority is to trim the terminal Constant and Reshape from the current corrected graph, retaining only the quantized and convolution output QDQ subgraph and reusing probability inputs, while freezing the NONE/OBEY comparison. This plan has not yet been executed and requires precise validation of source, shape, and CPU output byte correspondence; removing terminal nodes alters the external output contract, and thus new results cannot be directly attributed as the sole root cause of any node.

Static suggestion is 6 nodes, 7 initializers; external output changed to source Conv's FLOAT `[1,1,4,8400]`. A future new cut CPU validation should continuously compare the raw byte array of the original array with the old `[1,4,8400]` reference; shape mismatch means entire NPY file byte comparison cannot substitute array-level comparison. Static plan `../artifacts/gate05/20261020_dfl_softmax_cut_native/next_control_plan.json` SHA256 `14d7710f373fd56a8d68ec15219e234c96a9325938e25ad703696b879160dfc8`, not executed in this round.

Gate 0.5 remains at `INCOMPLETE`. This round has no full graph retry, parser constraint reset, matrix recovery, independent dataset, real-world object-level causal or performance pass conclusion. Stage evidence, checklist, and completion checkpoints are saved in `../artifacts/gate05/20261020_dfl_softmax_cut_native`, checkpoint at `checkpoint.json`; completion here refers only to bounded diagnostic closeout in this round.
