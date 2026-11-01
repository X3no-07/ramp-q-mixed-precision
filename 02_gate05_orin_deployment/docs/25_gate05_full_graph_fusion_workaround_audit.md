# Gate 0.5: Fusion bypass check for full INT8 graph



Client start date: 2026-10-18, end date: 2026-10-19. This round only checks one `q_all` configuration. **This round was stopped after a 30-minute timeout, constituting a timeout and insufficient evidence; no full engine was generated, and Gate 0.5 was not completed.** The board was still at 2026-10-16 when the build started, with startup ID `c4d8fcff-a824-488b-b004-6ad71eeb5d95` and monotonic clock timing used.

# # Changes in this round

Continuing from [previous report](24_gate05_orin_int8_build_and_transfer_audit.md), the weights, 88 convolution groupings, frozen calibration range, and numerical tolerance are retained, with only the G4 `/model.10/m/m.0/attn/pe/conv/Conv_output_0` exposed additionally. Its shape is `[1,128,20,20]`, with the purpose of enabling separate construction of depthwise convolution and subsequent residual addition. The primary output remains `output0 [1,14,8400]`.

All arithmetic nodes and initializers are identical to the original quantized graph on a byte-for-byte basis; the candidate ONNX SHA256 is `2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d`. The build package `diagnostic_bundle_v3_pe_tap.zip` has SHA256 `0aab1979390e0fab363be242d6e80d728e7699574112f734826c52d49b6b44e2`, and all 13 input files have been verified for size and hash on the board.

The builder validates the complete input-output set, shapes, and dtypes by name; the execution script allocates buffers for each output and separately verifies whether the primary and auxiliary outputs are bounded. No new bindings are ignored.

The local ONNX Runtime is run without optimization, and on the same Selection image `000020.jpg`, the candidate primary output is checked against the saved original ONNX Q/DQ output element-by-element, with a maximum difference of 0. The auxiliary output shapes are correct and all are bounded. This only indicates that the modifications to the graph output in this round have not altered the primary output of the CPU execution; the full model numerical consistency between original PyTorch Q/DQ and ONNX Q/DQ remains failed.

# # On-board build

Build directory: `/home/nvidia/rampq/gate05_int8_diagnostic_20261018_v3_pe_tap/results/build_20261016T111054_305529Z`.

Monitoring directory: `/home/nvidia/rampq/gate05_int8_monitor3_pe_tap_20261018_c4d8fcff`. Same as the previous round: 30-minute timeout, 85°C threshold, persistent low available memory, and USB/OOM error-triggered termination conditions. Only monitor the current round's independent process group.

Build logs show that the previously failing G4 PE convolution is now isolated as a separate `CONVOLUTION`, and residual addition is handled as a separate `POINTWISE`. This confirms that the output modification indeed prevented fusion at this location, but this log is not the final engine and does not prove that the entire model can be built.

At monitoring time 1802.31 seconds, the current process group was terminated due to `Build supervision time limit reached`. The original `Could not find any implementation` was not reproduced prior to termination, and a complete engine was not generated. This is recorded as **`FULL_QDQ_PE_TAP_BUILD_TIMEOUT_INCONCLUSIVE`**. The timeout cannot be reclassified as a compatibility issue resolved, nor can `KeyboardInterrupt` be treated as a spontaneous builder error; it is an exit triggered by the monitor's termination signal.

The original builder's `latest_build.json` remains stuck at `BUILD_RUNNING`, representing an outdated state after interruption. The monitoring log and the process exit status take precedence over any ongoing build. There is currently no active build in progress.

| Item | Result |
|---|---:|
| Resource sampling | 898 entries |
| Maximum temperature | 54.656°C |
| Minimum available memory | 2377.35 MiB |
| Maximum swap usage | 0.25 MiB |
| New kernel events | 0 |
| Complete engine / per-layer precision check / single-image execution | All remain incomplete |

The boot session remained unchanged after completion. The fan and jtop were active, the restart count was 0, and available memory returned to approximately 5 GB. The sizes and SHA256 digests of 21 original device files were checked against their receipts. The archive hash is `3fb9ea67c8a2cd194d08c9ea3d92c4daf648ffaad1e4c4c44f3f7b8534b92b9b`. There is no existing evidence of overheating, memory exhaustion, or a USB fault in this run.

# # Interpretation Limitations

The original strict precision pass criteria remain: all 88 source convolutions map uniquely, with INT8 weights and inputs, and possess identifiable integer tactics. Additional reports are provided for INT8 weight and input format coverage; when `TacticName` is missing, the integer tactic remains unconfirmed. Complete format evidence allows explicit selection of fixed single-image diagnostic execution, but this does not constitute a pass of strict precision audit.

Multi-source convolution fusion requires verification of complete members, same group, native weight count equal to source total weights, identical original activation sources, and byte-for-byte identical quantization parameters. When such verification is not completed, further review is retained; shared individual weights cannot be automatically considered proof of precision for multiple convolutions. The audit script has been validated using an existing FP16 engine as a counterexample, and confirms that missing images, incorrect weight counts, or identical inputs with different quantization scales will be rejected. No complete INT8 engine is available for this audit in this round.

Additional output changes affect TensorRT fusion and floating-point precision selection. Comparison with the original FP16 graph is purely descriptive. Future per-object attribution requires control over output structure and floating-point operator precision across engines. No 150-image dataset evaluation, recovery configuration, or performance testing has been run; Gate 0.5 remains incomplete.

Current round files: `../artifacts/gate05/20261018_int8_pe_tap`. All original failure cases, minimal reproducibility, and predefined numerical tolerances from the previous round are preserved.

# # Next retry is prepared but not yet executed

Native board end `trtexec --help` explicitly specifies: `--avgTiming` controls the number of times the average timing is taken when selecting the computation scheme, defaulting to 8. The next input package sets it to 1, retaining the same graph, weights, calibration range, quantization grouping, numerical tolerance, 1 GiB workspace, FP16/INT8 switch, and a 30-minute monitoring cap.

Retry package: `../artifacts/gate05/20261018_int8_pe_tap/diagnostic_bundle_v4_pe_tap_avg1.zip`, SHA256 is `86ac14e2f2e170e55ed62c2476ee215195f3ec58a5229bd0b7c8b8a0c6981436`. This package only completes local packaging, manifest verification, and Python 3.8 syntax check; it is not uploaded or launched. Whether it can shorten build time has not been validated. The computation scheme selection may change, and it remains solely for hardware executability diagnostics, not as a performance or quantization causality comparison result.
