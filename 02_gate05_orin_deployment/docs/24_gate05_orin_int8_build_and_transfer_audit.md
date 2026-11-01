# Gate 0.5: Orin INT8 Construction and Migration Verification



Client date: 2026-10-18. Current conclusion: **The complete INT8 graph fails to build on TensorRT 8.5.2.2 and no executable engine has been generated; the local numerical migration guardrail has also not passed. Gate 0.5 remains incomplete.**

# # Scope of this round

This round only checks `q_all`: all 88 convolutions in G1–G8, with explicit INT8 Q/DQ applied to weights and input activations. Other network operators are allowed to use FP16/FP32 and cannot be described as having all operations in the entire network being INT8. This round does not run evaluation on a fixed set of 150 images, nor does it execute the recovery configurations G1/G1+G5/G1+G2.

Retain the model from Gate 0, the frozen range derived from 64 training calibration images, and the 88-layer grouping. No re-calibration is performed, and no images, modules, or ranges are changed based on this round's results. The quantized graph originates from the same FP32 ONNX used in the previous FP16 run, with input `[1,3,640,640]` and output `[1,14,8400]`.

Quantization settings:

- Activations are symmetric, per-tensor, zero-point 0, scale=`max(frozen_absmax/127,1e-12)`.
- Activations are clipped to `[-127*scale,127*scale]` before entering the INT8 QuantizeLinear operation, preserving the narrow-range semantic of the original simulation.
- Convolution weights are symmetric, per-output channel, axis=0, scale=`max(per_channel_absmax,1e-12)/127`, zero-point 0.
- No additional quantization is applied to convolution outputs; subsequent convolution inputs are quantized according to their own frozen ranges.
- The explicit Q/DQ input quantization positions are referenced to [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html) and [NVIDIA Maintainer's Notes on Convolution Input/Output Precision](https://github.com/NVIDIA/TensorRT/issues/2383). The actual adoption of INT8 kernels remains determined layer-by-layer by engine evidence.

# # Local Numerical Check: Not Passed

The single-image numerical tolerance was saved prior to running for `../artifacts/gate05/20261018_int8/input_v1/protocol.json`. For the first image in the existing Selection, `000020.jpg`, a comparison was conducted between PyTorch FP32 Q/DQ simulation and ONNX Runtime CPU Q/DQ execution:

| Item | Result |
|---|---|
| All weights and biases matched against the source model | Exactly consistent |
| Independent samples of INT8 rounding and narrow-range saturation | PASS |
| Maximum absolute difference of all original bounding box outputs | 96.570465 (at 640 input coordinates) |
| Maximum absolute difference across all original class scores | 0.072959 |
| Original bounding box numerical guardrail | Failed: 14,204 out of 33,600 elements exceed the predefined combination tolerance |
| Passed after modifying the original tolerance | None |

These maximum differences are traversed across all 8,400 output positions, and do not imply that all final detection boxes have been shifted by that distance.

Further isolation checks are preserved in `local_divergence_diagnostic.json`, `isolated_quantizer_audit.json`:

1. 88 quantized weight tensors are exactly identical between PyTorch and ONNX Runtime.
2. Using the same input tensors, 22,835,200 elements of the 88 activation quantizers are fully consistent.
3. The output of the first convolutional layer already exhibits a maximum approximately `1.62125e-5` floating-point backend discrepancy; the input to the second layer has 3 elements showing quantization grid discrepancies, with such discrepancies propagating layer by layer. This record supports further investigation into the coupling between backend floating-point computations and quantization rounding, and cannot be considered as fully explaining all subsequent discrepancies.
4. ONNX graph structure audit confirms: all original non-convolutional nodes and initializers remain unchanged; convolutions are only reconnected with two quantized inputs, adding 440 quantization/clipping nodes, and freezing scale and narrow-range boundary accuracy.

Descriptive NMS results on the same images still show detections for sea cucumbers and abalone, but the confidence score for sea cucumbers increases from approximately 0.614 in PyTorch simulation to approximately 0.673 in ONNX simulation; identical detection counts do not substitute for the original numerical guardrail. No adjustment to calibration range or alternative images was made based on this observation.

# # Why Continue with Real-Device Diagnostics

Under the condition of preserving full model numerical failure, we further freeze `diagnostic_scope.json`: This response only addresses whether the explicit graph can be built on real TensorRT 8.5.2.2, whether the 88 convolutions use actual INT8 kernels, and whether a single graph can execute successfully. It does not reclassify local numerical failures as passed, nor does it directly provide conclusions on migration validity, recovery mechanisms, or accuracy.

This diagnostic does not alter the quantized graph or calibration range. Local quantization semantic validation and full model numerical checks are two distinct pieces of evidence, both retained.

# # Real-Device Build Records

The first real-device build failed after approximately 6.63 seconds, indicating `Int8 precision has been set ... but int8 is not configured in the builder`. This is due to missing build parameters. The original log is saved in `build_attempt1/`, `monitor_attempt1/`.

On the second attempt, only the missing `--int8` was added to the build command, while `--fp16 --noTF32 --memPoolSize=workspace:1024 --profilingVerbosity=detailed --buildOnly` was retained. The ONNX graph hash remains unchanged, and the second input package preserves `build_revision.json`. The TensorRT log confirms that the explicit precision mode does not invoke the calibrator. The configuration logic for the `--int8` switch is documented in [TensorRT 8.5 Official Sample Source Code](https://github.com/NVIDIA/TensorRT/blob/release/8.5/samples/common/sampleEngines.cpp).

On-device build directory: `/home/nvidia/rampq/gate05_int8_diagnostic_20261018_v2/results/build_20261016T100429_415185Z`

Construct a monitoring directory: `/home/nvidia/rampq/gate05_int8_monitor2_20261018_c4d8fcff`. Monitoring of temperature, available memory, kernel exceptions, and a 30-minute time limit is performed with best-effort; it does not guarantee automatic recovery after kernel crash.

The board clock remains 2026-10-16, while the client date is 2026-10-18. The current boot sequence number is `c4d8fcff-a824-488b-b004-6ad71eeb5d95`, with monotonic timing used for duration measurement.

The second build terminated with an error after 1605.26 seconds (approximately 26 minutes and 45 seconds) of monitoring, falling short of the 30-minute stop threshold. The actual error message was:

```text
Could not find any implementation for node
model.10.m.0.attn.pe.conv.weight + rampq_model_10_m_0_attn_pe_conv_WeightQ
+ /model.10/m/m.0/attn/pe/conv/Conv + /model.10/m/m.0/attn/Add.
```

This node belongs to G4 and involves deep convolution and residual addition fusion in the attention PE branch: input `[1,128,20,20]`, weights `[128,1,3,3]`, group=128, stride=1, padding=1. The error cannot be generalized to 'all INT8 convolutions are unsupported', nor can the layer be restored to FP16 while still being considered part of the original full INT8 configuration.

During the build process, the highest recorded temperature from 800 resource samples was 54.531°C, the minimum available memory was approximately 2467.87 MiB, and the maximum used swap was about 0.25 MiB. No new events were captured in the kernel monitoring logs; SSH commands remain executable after build completion. This is a clearly defined builder error under the same boot sequence, and current logs do not indicate overheating, memory exhaustion, USB disconnection, or board reboot.

The original 22 files from both builds have been returned and verified against board-end receipts for file size and SHA256 hash, stored in `build_attempt1/`, `monitor_attempt1/`, `build_attempt2/`, `monitor_attempt2/`; the receipt is `remote_receipt.json`. The original compressed package hash is `1af9410aa777c9866871785d73c601df30506ffa6f0240a89f31c57fa170ca86`.

# # Minimal Reproduction

To distinguish between single-convolution and fusion-related issues, identical weights, biases, quantization ranges, and a 128-channel convolution are extracted, and three separate build diagnostics are conducted, each initially capped at 90 seconds:

| Minimal graph | Result | Duration |
|---|---|---:|
| Single deep convolution | Successfully built and deserialized | 53.103 seconds |
| Deep convolution plus residual addition | Reproduced the same fusion error as in the full model | 53.001 seconds |
| Deep convolution plus residual addition with additional exposure of convolution output | Reached initial time limit; evidence insufficient | 90.256 seconds |
| The above scheme with additional output exposure, with only one resource limit retry added | Successfully built and deserialized; 180-second timeout limit | 145.510 seconds |

90-second timeout record preserved without being marked as passed. The 180-second retry only modifies the allowed build time, without altering the graph, weights, calibration range, or numerical tolerance. The original result is stored at `pe_repro_results/`, the retry result at `pe_tap_retry180/`, and the resource budget explanation at `pe_tap_retry180_request.json`.

In the smallest viable graph, the engine inspector shows `CudaDepthwiseConvolution` using INT8 weights, vectorized INT8 inputs, and FP32 outputs; residual addition is implemented as independent PointWise layers, no longer included in convolution fusion names. This convolution layer does not provide `TacticName`, so only visible weight and input formats are reported here, without further claims about Tensor Core instruction types or performance gains. Two small graphs completed only build and deserialization, with no data inference execution.

This controlled reproduction demonstrates that a single convolution can be built independently, but connecting it with residual addition triggers failure; exposing the convolution output allows the node to be built as separate operators. This conclusion applies only to the minimal graph for this node and cannot yet be generalized to guarantee successful construction of the full model. The 14 additional original files returned in the minimal reproduction have been verified against `pe_repro_remote_receipt.json`, and a total of 36 board-side files have been cross-checked in this round.

# # Candidate full graph prepared

On the failed full quantization graph, `pe_tap_full_graph_candidate/model_q_all_pe_tap.onnx` adds the G4 convolution output as a second graph output and supplements the shape metadata. All arithmetic nodes and initializers remain byte-for-byte identical; quantization positions, ranges, groupings, and primary output declarations remain consistent; structural checks have passed. The candidate SHA256 is `2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d`.

This full candidate has not yet been built or executed on the board. The additional output affects fusion and floating-point precision selection, thus the next round must adapt the additional bindings and re-examine the actual layer precision for all layers. Subsequent per-object comparisons should control the graph output structure and non-quantized floating-point precision across all engines. Node-level build success cannot be directly elevated to the full INT8 model being fixed, nor can it erase the original numerical migration failure guardrails.

# # Result files and next steps

All evidence from this round is located at `artifacts/gate05/20261018_int8/`; `checkpoint.json` preserves the current machine state, `audit_summary.json` is a machine-readable summary of verified original files. `input_v1/` retains the original failure record, and `diagnostic_bundle_v1/` and `diagnostic_bundle_v2/` store the two build inputs; the first failure result must not be deleted.

If a full engine is generated, first use `scripts/audit_gate05_int8_precision.py` to audit the original Conv identity, INT8 weights, INT8 input format, integer tactic, and group coverage. This auditor has already used an existing FP16 engine as a counterexample, and all 88 FP16 convolutions are not classified as INT8. It currently conservatively flags layers missing integer tactic names for further review; subsequent processing must be based on full inspector records, and unconfirmed items must not be ignored to achieve a pass.

Only after actual precision auditing should the same Selection image execution check be performed using `scripts/jetson_trt_int8_smoke.py`. The subsequent full Gate 0.5 still requires freezing the dataset criterion, checking the three recovery engines, completing per-object matching on the same image set, and explaining local numerical migration differences. Formal latency, FPS, power consumption, and bidirectional attribution remain unsupported by evidence in this round.
