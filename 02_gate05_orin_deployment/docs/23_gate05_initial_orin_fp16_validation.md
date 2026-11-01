# Gate 0.5: Orin FP16 First-round Check



Client date: 2026-10-18. Conclusion: **FP16 build, real single-image execution, and convolutional precision verification passed; the entire Gate 0.5 has not yet been completed.**

# # What has been validated in this round

| Check | Actual results |
|---|---|
| Board-side build | TensorRT 8.5.2.2 build succeeded and is serializable, taking 804.3 seconds |
| Engine | 7,607,688 bytes, 192 engine layers |
| Group mapping | All 88 original convolutions were mapped to 85 fused convolution layers |
| Convolution precision | The weights of all 85 fused convolution layers are Half, and their input/output tensor formats are FP16 |
| Actual execution | Executed `execute_v2` on Orin GPU, returning successfully with all outputs finite |
| Single-image comparison | Compared to the same board PyTorch FP32 with each output 2 detections of the same category, the two bounding boxes have IoU values of 0.999446 and 0.998508 |
| Confidence comparison | Absolute differences in confidence scores are 0.003055 and 0.000747 respectively |
| File return | 16 original evidence files match in size and SHA256 hash with the board-side versions, including the complete engine |

Original data available at `../artifacts/gate05/20261018_fp16`. `audit_summary.json` is a machine-readable summary, and `precision_audit.json` contains per-convolution mapping and precision evidence.

# # Fixed conditions and scope of evidence

- Device: Jetson Orin NX 8GB; Ubuntu 20.04.6; Jetson Linux 35.3.1; Python 3.8.10; PyTorch 2.0.0+nv23.05.
- Input: batch 1, RGB, 640×640, letterbox; output is `[1,14,8400]`.
- Build: `--fp16 --noTF32 --memPoolSize=workspace:1024 --profilingVerbosity=detailed --buildOnly`.
- Power settings: requested 15W mode 2; non-admin user `nvpmodel -q` returns 15W, and reports no access to EMC parameters. The full privilege query was not rerun, and this limitation is not hidden.
- Image: the first image with existing selection is `000020.jpg`, no re-selection performed.
- Single-image NMS: confidence 0.25, IoU 0.7, max_det 1000; these are pipeline check settings, not the complete set of settings used in full dataset diagnosis.
- Pre-run guardrail: number of detections is equal and non-zero across both sides, one-to-one box IoU ≥0.90, absolute confidence difference ≤0.05. Passed this check.

The engine's input/output and partial detection head decoding remain in FP32; 20 layers contain FP32 input or output tensors; this is not equivalent to 'all operations in the entire network are FP16'. The FP16 format of convolution tensors and weights does not imply that accumulators necessarily use FP16.

3 mixed-precision convolution fusion occurs at G3,G4,G6, with no cross-group fusion. The original convolution counts per group are G1=6, G2=5, G3=10, G4=19, G5=8, G6=15, G7=8, G8=17. The mapping from FP16 is available for subsequent verification, but each INT8/ mixed-precision engine must be independently re-validated and cannot reuse the current fusion results.

# # Meaning of the single-image validation

TensorRT and PyTorch's raw outputs before NMS were preserved, and detection boxes were recalculated using independent NumPy coordinate transformations. The maximum coordinate difference from the saved original image coordinates is less than 0.001 pixels, confirming no missing letterbox padding or scaling issues.

This image has only one sea cucumber annotation. The IoU between the TensorRT-detected sea cucumber box and the annotation is approximately 0.7914; another detection is for scallop, with no corresponding annotation. Therefore, 'two detections match PyTorch' does not imply 'both detections are correct', and does not constitute evidence of dataset accuracy.

The maximum difference in box values across all positions before NMS is 19.7130, and the maximum difference in category scores is 0.005245; this check passed the pre-defined final detection consistency guardrail, and no claim is made about element-wise equality of all raw outputs.

The `first_call_ms_not_benchmark=1310.87` in the log is a single, unwarmed invocation, **and cannot be converted into formal FPS, average latency, P50/P95, or end-to-end performance**.

# # Build monitoring and connection status

Sampling occurred every approximately 2 seconds during build, with 401 records: maximum observed temperature from the temperature sensor is 52.125°C, minimum available memory is approximately 1629.91 MiB, and maximum swap usage is 0. No new kernel events were captured in the monitoring logs during the build phase. These are sampling observations during the build stage, not peak inference resource usage or power consumption tests.

A `Connection reset` occurred during return transfer after building and inference. On 2026-10-18, the user confirmed that power and data cables had remained connected without being unplugged. After SSH reconnection, the boot ID was still `c4d8fcff-a824-488b-b004-6ad71eeb5d95`, and the task had terminated normally. This does not support attributing the interruption to a board freeze or reboot; the cause of intermittent SSH disconnections remains unidentified. The first engine transfer was incomplete; retransmission completed and the correct hash was verified.

During subsequent checks of the same boot, jtop shows active status, NRestarts=0, and fan service is active. Startup-related RPC control and fan speed overflow messages were already displayed; this report does not consider these as root causes of disconnection, nor does it claim that system logs are completely free of warnings.

The board clock still shows 2026-10-16, while the client shows 2026-10-18; this record uses the client date, boot number, and monotonic timestamp to resolve ambiguity, and no system time modification was made without authorization.

# # Revisitable files

| File | Purpose |
|---|---|
| `engine_build/build_request.json` | Actual build parameters |
| `engine_build/build_result.json` | Build and deserialize the original results; its 'non-inference' state refers to the state at the end of the build script execution |
| `engine_build/model_fp16.engine` | Complete engine built on the board |
| `engine_build/engine_layers.json` | Original engine inspector results |
| `monitor/` | Build console logs, resource sampling, status monitoring, and kernel supervision |
| `smoke/smoke_result.json` | Real-world execution on a single image with reference comparison |
| `smoke/raw_outputs.npz` | Original outputs from two implementations |
| `smoke/prediction.jpg` | Detection visualization |
| `remote_receipt.json` | Original file size and SHA256 checksums on the board |
| `precision_audit.json` | 88 original convolution layers matched one-to-one with actual engine layers |
| `audit_summary.json`,`checkpoint.json` | Audit conclusion and current progress |

All of the above files are located at `../artifacts/gate05/20261018_fp16`. Offline review entry is `../scripts/audit_gate05_fp16.py`, requiring NumPy and Pillow, not TensorRT, and no re-execution of board-side tasks.

Model weight SHA256: `62aa0728fba4cf9443a154d7db0b1b771a7373c1d302c6c24e3f2a9ac98061af`

ONNX SHA256:`d8ac451f2e526d4ec7c56727fa990a88cb612974e8206cf5821a25ce5e6f6f7d`.

engine SHA256:`64d91db3063ac28c1552a1ba6443ae28ca45d82d988f5cd375b39fc8cc76b85c`.

# # Pending Gate 0.5 Work

The construction, precision verification, and single-image execution for the first precision configuration have been completed. 64 original calibration images, 150 original holdout images, and 865 targets have been prepared for `../artifacts/gate05/data_v1`; the overlap between calibration and evaluation images is 0; evaluation of this batch of images has not been performed in this round. The old holdout set is solely used for deployment migration diagnosis and cannot be repackaged as an independent validation set for a new method.

Subsequent steps will be completed in sequence:

1. Freeze the criteria for passes, failures, and insufficient evidence prior to reviewing dataset results; maintain fixed model, calibration list, image list, and target matching criteria.
2. Build and verify the full INT8 convolution baseline; confirm actual INT8 kernels and G1–G8 mappings, not merely checking build switches.
3. Construct recovery configurations separately for G1, G1+G5, and G1+G2 while maintaining FP16, and report results individually.
4. Perform per-object matching across five engines on the same fixed image set, and statistically record the newly introduced missed detections, recovered targets (same-object recovery), and collateral misses relative to the native FP16 baseline, and separately evaluate the FP32→FP16 transition.

The above five configurations only validate recovery. If claiming true bidirectional attribution in TensorRT, additional `q_only:G1` destructive-side experiments are still required. No conclusions are drawn in this round regarding INT8 authenticity, mixed-precision recovery, formal latency, or power consumption.
