# Gate 1: Terminal Review and Closeout of Guarded Correction

October 28, 2026, Beijing time. The original supervision workflow completed at 16:14; this review verified the terminal state and delivery.

**All three training arms, scoring jobs, and saved-artifact audits completed. Intervention-guided correction did not outperform standard QAT or ordinary hard-example training under the same protection rule. Both predefined joint criteria failed, and automatic extension of this correction branch was stopped.**

| Method | AP50–95 (%) | Detections at confidence 0.25 | False positives at confidence 0.25 |
|---|---:|---:|---:|
| Continued standard QAT | 11.985899444 | 10552 | 7728 |
| Ordinary hard-example training with gradient protection | 11.918156643 | 10486 | 7494 |
| Intervention-based sampling with gradient protection | 11.877794522 | 10443 | 7494 |

Relative to standard QAT, the intervention arm lost 0.108104922 AP percentage points, detected 109 fewer targets, and produced 234 fewer false positives. Relative to the hard-example arm with the same protection rule, it lost 0.040362121 AP percentage points and detected 43 fewer targets at the same 7494 false positives. These results do not support an additional benefit from precision-intervention-based sampling. Passing the saved-artifact audit does not change that method decision.

All 5862 delivery members were rechecked for SHA256 and byte size, with zero mismatches. Verification covered 9 per-epoch states, shared initialization, complete sample order, 3 epochs/3399 batches/852 steps per arm, 42012 training/evaluation forward passes, and 9 additional reconstructions. The independent audit had verified 1704 protection windows, 1242 all-candidate raw/NMS records, 1242 prediction/target matches, 2484 AP matrices at zero tolerance, and 384 replays of the existing 128-image, 8400-candidate thresholds. Before/after hashes for 5832 output members and identity checks for 14832 source-data files were bound to the delivery records.

Training workers, the audit, and supervision were observed to exit with code 0; all three AP jobs also exited with code 0. No corresponding running process was found at review time. This review checked saved identities and terminal states read-only; it did not rerun training, inference, AP, or the independent array audit. Report 89, the machine decision, contracts, and delivery manifests remain unchanged. A pending-supervision state in the original decision describes its historical audit time; independent terminal and delivery records establish final completion.

CUDA saved-state reconstruction was exact. The training receipt did not certify CPU/CUDA output equivalence. The original FP32 reference contains only 128 images; the other 286 images were not inferred again in this experiment, so full 414-image numerical equivalence cannot be claimed.

The evidence is limited to one seed, the repeatedly used 414-image `training_val` development set, and floating-point QDQ. Three continuation epochs do not demonstrate full convergence. The protection rule is not a performance guarantee for each target or for historical SGD updates. Protected arms require more backward computation, so equal training FLOPs are not claimed. This candidate has not obtained real INT8, TensorRT/Orin, or independent scientific acceptance. No additional Selection/Holdout access occurred, and earlier failures remain recorded.

The agreed training, scoring, saved-artifact audit, and closeout were completed. The `gate1` 50-minute scheduled check was removed using the application tool. Further work requires a revised research question, suitable strong baselines, and a fixed budget. Independent-data access and device conditions must be addressed separately. Seeds, epochs, thresholds, modules, and datasets will not be extended automatically in pursuit of a passing result. This conclusion applies to the tested correction method and does not reject quantization research as a whole.

Local evidence identifiers:

- `gate1/20261028_guarded_training_closeout_v1`
- `terminal_delivery_validation.json`
- `automation_stop_receipt.json`

Complete local evidence is outside this source-and-report edition; see [artifact availability](../../docs/DATA_AND_ARTIFACTS.md).
