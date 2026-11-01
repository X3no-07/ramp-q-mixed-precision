# Gate 0 Unlabeled Output Reconstruction Error Baseline



> Results: The method of label-free output error selection identified `G1+G2` across all budgeted combinations, and selected `G1+G2` from the preregistered unexecuted random pool. This step did not read any labels or run or read the detection results of the three random combinations.

# # 1. Frozen Conditions

- Calibration images: 64 images, SHA-256 list: `02caf7d8888b72a8ac884eb95aeb76ae9c9dd984baf3b1714a379faf2eaa7069`
- Model SHA-256: `62aa0728fba4cf9443a154d7db0b1b771a7373c1d302c6c24e3f2a9ac98061af`
- Device: `cuda:0`
- Configuration: FP32 is compared against `q_only:G1` to `q_only:G8` before NMS, in terms of detection outputs.
Labels were not read; no random control results that had not yet been generated were accessed.

# # 2. Module-level Output Reconstruction Error Ranking

| Rank | Module | Average normalized MSE | Average cosine distance | MAC Share |
|---:|---|---:|---:|---:|
| 1 | G1 | 6.14538958e-03 | 3.06651546e-03 | 10.32% |
| 2 | G2 | 8.53121508e-04 | 4.26381813e-04 | 12.65% |
| 3 | G3 | 2.50677416e-04 | 1.25109746e-04 | 11.82% |
| 4 | G4 | 2.07560319e-04 | 1.03728813e-04 | 13.29% |
| 5 | G6 | 1.04126629e-04 | 5.19912693e-05 | 12.86% |
| 6 | G5 | 9.22223590e-05 | 4.60696833e-05 | 12.03% |
| 7 | G8 | 2.79029744e-05 | 1.36902453e-05 | 9.25% |
| 8 | G7 | 1.08904386e-05 | 5.41330305e-06 | 17.78% |

The primary ranking uses only the average normalized MSE. Cosine distance is used only for diagnostic purposes and does not affect the ranking. A higher error indicates that the module is more sensitive to quantization according to conventional output reconstruction methods.

# # 3. Budgeted Combination Selection

- All budget-satisfying two-module combinations: selects `G1+G2`, with a total score of 6.99851109e-03 and MAC contribution of 22.97%
- Only pre-registered without running random pool: selected `G1+G2`, score sum is 6.99851109e-03, MAC contribution is 22.97%.
- Collision between RAMP-Q candidate `G1+G5` and the current selection: no.

The term 'same-budget pool' in the preregistered set may be narrowly interpreted as the unexecuted random pool. To avoid selective reporting, both narrow and full combination diagnostics—including candidates—are publicly disclosed here. Subsequent analyses must not retain only the more favorable result.

# # 4. Interpretation

The output reconstruction error only answers the question of 'which module, when quantized, causes the largest change in the model's original output'—it does not directly address whether any real targets are missed or recovered. It serves as a traditional label-free proxy baseline and is not a conclusion about detection performance.

# # 5. Next Steps

The first random control's destructive side is executed under the frozen order: `q_only:G4+G8`. This result does not permit reconfiguration of the random combinations.

# # Evidence Boundaries

Results are derived from 64 calibration images of RUOD-R and PyTorch Q/DQ simulation of W8A8. Target labels are not used, and the images shared for calibration are not an independent test set; these results do not represent performance on TensorRT, Jetson, latency, or power consumption.
