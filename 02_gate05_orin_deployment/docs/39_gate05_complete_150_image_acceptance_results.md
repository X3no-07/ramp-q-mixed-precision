# Gate 0.5: Complete 150-Image Deployment Acceptance Results

Recorded at 2026-10-21T18:31:50.428540+08:00. Acceptance testing of all six deployment configurations completed. The preregistered **same-target G1 recovery existence criterion passed**. The original full-model PyTorch/ORT numerical guardrail remains **FAILED_UNCHANGED**; numerical equivalence across backends has not passed.

## Primary result

The fixed set contains 150 images and 865 annotated targets. The on-device FP16 baseline hits **722** targets; full INT8 hits **687**. Matching by target identity identifies **44** targets hit by FP16 but missed by full INT8, and **678** stable targets hit by both.

The predefined primary configuration `r_only_G1` recovers **40 of the 44 targets (90.91%)**, while introducing **3 collateral misses among the 678 stable targets (0.4425%)**. Net recovery on this defined set is **+37**, and stable-target retention is **99.5575%**. Four additional misses remain unrecovered.

G1's total hit count is **721**, an increase of **34** over full INT8's 687. This differs from net recovery +37: full INT8 also hits 9 targets outside the FP16-hit set, while G1 hits 6 there, so the total-hit change is `40-3+(6-9)=34`. Do not describe +37 as 37 more total detections.

These totals include all object sizes. The 44 additional misses and 40 recoveries cannot all be called small objects. Hit counts are not mAP.

## Six configurations and predefined secondary comparisons

“Full INT8” denotes W8A8 requests for 88 source convolutions, not INT8 execution of every operator. Non-convolution floating-point paths and FLOAT output boundaries are retained and audited separately.

| Configuration | Audited source convolutions: INT8 / FP16 | On-device hits / 865 | ORT hits / 865 | Targets with different on-device/ORT hit states |
|---|---:|---:|---:|---:|
| `fp16_control` | 0 / 88 | 722 | 722 | 0 |
| `q_all` | 88 / 0 | 687 | 689 | 26 |
| `r_only_G1` | 82 / 6 | 721 | 724 | 17 |
| `r_only_G1_G5` | 74 / 14 | 724 | 725 | 19 |
| `r_only_G1_G2` | 77 / 11 | 717 | 721 | 8 |
| `q_only_G1` | 6 / 82 | 699 | 698 | 1 |

| Recovery configuration | Recovered additional misses / 44 | Still missed | Collateral misses / 678 stable targets | Net recovery | New hits outside FP16-hit set |
|---|---:|---:|---:|---:|---:|
| `r_only_G1` | 40 | 4 | 3 | +37 | 6 |
| `r_only_G1_G5` | 41 | 3 | 3 | +38 | 8 |
| `r_only_G1_G2` | 39 | 5 | 3 | +36 | 3 |

G1 is the predefined primary comparison. G1+G5 and G1+G2 are frozen secondary comparisons; their results were not used to replace the primary configuration. The counts do not establish that G1+G5 outperforms G1 at equal compute cost. No equal-budget superiority evaluation of latency, energy, or resource use was performed.

The auxiliary `q_only_G1` loses 35 FP16-hit targets. Of these, 26 also satisfy “missed by full INT8 and recovered by G1,” representing 26/44 (59.09%) of the additional misses. This is a limited same-target bidirectional pattern. The auxiliary engine uses the previously disclosed **PREFER mode with 82 explicit FP16 compute/output requests**, whereas the other principal members use NONE. It is not a pure G1 causal control under identical build constraints and does not replace the primary comparison.

## Execution and matching evidence

- Each of the six on-device configurations executed all 150 images once: **900 executions**. Each corresponding complete ONNX graph executed all 150 images in CPU ORT: another **900 executions**. All were full-graph executions, with 0 warmups, 0 additional inference calls, and 0 automatic retries.
- All 1,800 primary and PE outputs are finite. The 1,218 receipt-listed files inside the six on-device ZIP archives were recovered and verified. Under the same boot, all 30 task-related PIDs and the serial-queue PID were absent, all six wrapper process groups were empty, and supervision did not intervene.
- Layer-inspection JSON from the actual deserialized engines exactly matches the corresponding qualified-build inspection content. Environment identity, before/after loaded-library SHAs, maps device/inode information, source graphs, and engine identities were authenticated. Actual source-convolution precision/legal-fusion audits remain separate from the single-image runtime-kernel/static signed-INT8 arithmetic supplements. Missing historical tactic names were not fabricated.
- All configurations share the same preprocessed NPY inputs and CPU postprocessing environment. All **1,800 NMS runs** were saved normally, without warnings or timeouts. Candidate NMS uses confidence=0.001, IoU=0.7, max_det=1000. Target matching uses confidence≥0.25, the same class, IoU≥0.5, and one-to-one assignment, retaining the original reverse `(IoU, confidence, GT_index, detection_index)` priority order.
- An independent implementation rematched saved detections and verified **10,380 target-hit bits**, complete IoUs, confidences, and detection IDs. All 865 target rows, recovery/collateral-miss/net counts, per-class descriptions, and final criteria agree exactly. This review neither called the original evaluator's matching function nor reran a model or NMS.

## Numerical and scientific limitations

The following descriptive maximum absolute differences compare the corresponding saved full ORT and on-device arrays. No new tolerances are introduced, and the differences are not declared acceptable. Network coordinates are the four original coordinate channels; class scores are the other ten channels. PE is reported separately. The complete local comparison also contains per-image statistics, means, RMS values, byte differences, and identities.

| Configuration | Maximum coordinate difference | Maximum class-score difference | Maximum PE difference |
|---|---:|---:|---:|
| `fp16_control` | 103.833435 | 0.0621052682 | 0.148441553 |
| `q_all` | 459.044632 | 0.773249328 | 0.957398504 |
| `r_only_G1` | 365.509705 | 0.726792783 | 1.16247985 |
| `r_only_G1_G5` | 334.083405 | 0.827403843 | 1.17829591 |
| `r_only_G1_G2` | 476.781525 | 0.689458936 | 1.36956376 |
| `q_only_G1` | 32.5007629 | 0.0486412346 | 0.24067235 |

Despite raw-array differences, FP16 hit states agree with ORT on this set. The quantized configurations differ on 26/17/19/8/1 target-hit states, so the corresponding backends cannot be called equivalent. The original PyTorch/ORT preflight files and tolerances were not changed. The new complete ONNX reference is a separate deployment-semantic reference, not a relabeling of the original failure as a pass.

In the actual compilation, the FP16 baseline uses FP16 for DFL Softmax/decoding, while the quantized configurations use FP32; some SiLU/residual floating-point partitions and implementations also differ. These differences were disclosed and frozen before the 150-image outputs were inspected. The supported claim is therefore **same-target disappearance/recovery existence in the selected real deployment configurations**, not pure G1 arithmetic causality, a unique explanation of all differences, overall accuracy superiority, or Tensor Core performance.

The new NONE `q_only_G1` has an FP32 PE convolution and only 81 restored source convolutions reaching FP16. It remains disqualified, and its original failure is preserved. The older same-graph 6-INT8/82-FP16 engine was selected as a limited auxiliary member using its actual precision qualification in advance, not using the 150-image results.

The fixed set was used previously for diagnostics and is not a fresh independent Holdout. File identities show no overlap between the 64 calibration images and the 150-image set. This experiment does not add a new dataset, multi-seed significance, size-stratified superiority, or performance acceptance. Device and host clocks differ; evidence was associated using SHAs, boot ID, process identity, and a unique attempt marker without changing the device clock.

## Evidence identifiers and review entry points

The complete records are local-only; these links document their scope rather than distribute them:

- [Frozen acceptance contract](../../docs/DATA_AND_ARTIFACTS.md), SHA `759506e78383fa262c3cb74735c24c57844d706fdb95bf95bce168d9b143f500`.
- [Raw-output index and engineering qualification](../../docs/DATA_AND_ARTIFACTS.md).
- [Unified detection matrix](../../docs/DATA_AND_ARTIFACTS.md), SHA `cca862159ea2410626317caafe0367e59cf19724abfcd3a67ef11db49b56872b`.
- [Target-result summary](../../docs/DATA_AND_ARTIFACTS.md), SHA `a2fcc229ec551471e879b1b452c91392be1e703d79ec641eb4ec417adb701f12`.
- [Complete 865-target matrix](../../docs/DATA_AND_ARTIFACTS.md), SHA `a9c15745d837fea5524d1b6d3962227f06bad6a8345ab3fab41329f39fbbd8b4`.
- [Independent recomputation record](../../docs/DATA_AND_ARTIFACTS.md), SHA `503fa2bbdd376369d90ce2a9f020cace636032cf06427ad288da56fcde04f653`.
- [Original numerical comparison, final machine state, scope decision, and delivery manifest](../../docs/DATA_AND_ARTIFACTS.md).

Earlier research-direction documents are preserved under their frozen SHAs in local `protected_document_snapshots/`. Reports 23–38 and their failure evidence remain recorded.

Subsequent work requires a separately frozen protocol: retest on data unused for diagnostics with multiple calibration seeds, investigate the original numerical guardrail and backend differences separately, and measure real latency and energy only after candidate locking. This experiment does not automatically start the next phase or repeat the 900 on-device executions.
