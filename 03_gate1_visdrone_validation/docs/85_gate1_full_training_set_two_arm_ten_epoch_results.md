# Gate1 Results from Two Groups of 10 Training Rounds on the Full Dataset



2026-10-27.

This round involves the existing 4530 training images, two arms each with 10 epochs, and a single seed experiment with identical initialization, sequence, precision, and optimization budget. The candidate fixed epoch is set at 10, without selecting epochs or thresholds based on development AP.

| Branch | AP50–95(%) | Fixed detection threshold of 0.25 | Fixed false detection threshold of 0.25 |
|---|---:|---:|---:|
| ordinary_qat | 12.059032337 | 10596 | 7747 |
| native_intervention_correction | 11.918005380 | 10010 | 6194 |
| ordinary_qat_frozen_scorer | 12.106640226 | 9449 | 4674 |

Compared with standard QAT at the same precision, object-alignment correction changes AP by-0.141026957 percentage points, detections by-586, and false positives by-1553. The predefined joint condition of increased AP, no fewer detections, and no more false positives is not satisfied.

The third branch directly applies the old conventional scorer to the new conventional QAT output without retraining. Training members overlap and cross-detection migration exists, serving as a descriptive control only—this cannot be used to claim the contribution of the new scorer.

Independent saved-artifact checks passed:20 per-epoch FP32 records,828 all-candidate raw/NMS records,1242 prediction/target-match records, and 2484 AP input matrices were checked with zero tolerance. All output-member hashes are unchanged before and after audit. The three AP jobs terminated normally, and the seven aggregate metrics exactly match the saved AP/AR arrays. The audit repeated neither model inference nor complete AP computation.

The original numerical gate is rechecked over all 8400 candidates only for the 128 development images with existing FP32 arrays, without substituting confidence-filtered or NMS subsets. Old FP32 arrays for the other 286 images were not retained, and no new FP32 run fills that gap here; complete numerical equivalence over 414 images is therefore unqualified. The trained models differ from original FP32 outputs; saved-artifact consistency PASS cannot substitute for numerical qualification.

The current ordinary QAT and correction group both use G1-bit floating-point, with other Conv layers using W8A8-bit floating-point QDQ. Expanding the budget to ten epochs strengthened the pilot evidence, but did not demonstrate training convergence, nor did it reproduce HQOD/DetPTQ, validate true INT8, Orin, or independent scientific validation. 414 images are repeatedly used from the training_val development set; Selection/Holdout has not been read, and all historical failures are retained.

Automatic candidate expansion is halted in this phase. The next step will be to evaluate whether expanding conventional and published methods is warranted based on this fixed result; independent data and actual backend validation will require separate evidence.
