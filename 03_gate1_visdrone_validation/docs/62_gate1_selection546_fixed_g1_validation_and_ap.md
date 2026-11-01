# Gate1: Fixed G1 Development Validation and AP for Selection546



**The fixed G1 development matrix for the 546 members, 10 official Python APs, and independent saved results have been fully completed. The original 548 attempt remains halted and sealed due to original annotation conflicts on the same images.**

Shared FP32 AP50–95 is 12.7011429722%. Among the three existing calibration seeds, the AP50–95 improvement relative to full QDQ for restoring G1 is sequentially +0.0043713076, +0.1140277514, and +0.1515142961 percentage points. All seeds, configurations, numerical deviations, and object-level costs are retained below; no seed is selected as optimal.

The R for three seeds remains below FP32 and is only G1 quantization D's AP50–95; relative project matching false positives increase by 206, 317, and 460 respectively. All nine Q/D/R instances pass the original complete candidate box threshold of 0/546 images, and the saved-artifact audit passes without altering floating-point qualification.

# # Execution and Scope

Real attempts were conducted at Beijing Time 2026-10-24 20:08:32 to 20:39:07, with the launcher returning 0 and a total completion acknowledgment; saved result audit and delivery were continued from 2026-10-25. Fixed YOLO11n during the 30th round of FP32 EMA, with SHA256 being `4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27`. Forward passes on real images totaled 10675: 109 debug images strictly pre-verified from the old set, 546 shared new images from FP32, and each of the three seeds underwent 3340 forward passes (each containing 64 images solely for image calibration). The stride-based forward pass constructed separately.

Four configurations are set as F=FP32, Q=full Conv W8A8 QDQ, D=only G1 QDQ, R=full QDQ with recovery of G1 to FP32. Input 640, batch1, AMP disabled, CUDA determinism enabled, and TF32 disabled; 88 Conv, 81 BN not fused, actual convolution operations are FP32. Each of the three sub-configurations is recalibrated once, with the same seed Q/D/R shared across the range, and calibration label access is 0.

Only the original members `val/0000022_00000_d_0000004` and `val/0000023_00000_d_0000008` are isolated; all other 546 items retain their full fields and original order. The exclusion is based on actual annotation conflicts on 43/40 rows of the same image identity, without using prediction or AP to filter members. No original files were deleted, moved, or re-labeled; no single annotation was selected as the ground truth. This is a clearly defined new 546 development subset, and cannot be considered a successful reproduction of the original 548 evaluation.

The 546 members have 546 unique image SHA values, distributed across 75 risk-isolated components and 76 prefix proxies. These components are not proven independent real-world scenarios. Active role identities are preserved; the intersection of 24 inactive quarantined components and prefixes is still disclosed, and no independent scenario is fabricated based on this.

# # All Development AP

The AP/AR units in the table are percentages; the difference between the two configurations is expressed in percentage points. Shared F is scored only once, with each of the nine quantization instances scored once, totaling 10 official Python image aggregations. The saved predictions and original ground truth with eight fields are retained, including publisher score, category0, ignore, and submission order.

| Submission | AP50–95 | AP50 | AP75 | AR1 | AR10 | AR100 | AR500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 12.7011429722 | 24.0604062845 | 11.7044553786 | 1.0265052495 | 5.8781179680 | 18.2331231416 | 21.8508296501 |
| 2026102201/q_all | 12.1405272056 | 23.5003491345 | 11.0237306868 | 1.0944888906 | 5.8523523865 | 17.5526911121 | 21.3185304166 |
| 2026102201/q_only_G1 | 12.6660458255 | 23.9700097077 | 11.6254707751 | 1.0638844852 | 5.9004784969 | 18.2381951958 | 21.8638400376 |
| 2026102201/r_only_G1 | 12.1448985131 | 23.5495958546 | 10.9608851770 | 1.0969985341 | 5.7227027114 | 17.5117864366 | 21.3392266942 |
| 2026102202/q_all | 12.1426844331 | 23.5074143300 | 11.0125465472 | 1.0050985284 | 5.7666627165 | 17.5943561262 | 21.3070296106 |
| 2026102202/q_only_G1 | 12.6276113605 | 23.9405826750 | 11.6403211004 | 1.0624068445 | 5.8615659567 | 18.1591588856 | 21.6960094073 |
| 2026102202/r_only_G1 | 12.2567121845 | 23.7423530956 | 11.1251357094 | 1.0453913328 | 5.7324081862 | 17.6718739757 | 21.4762064124 |
| 2026102203/q_all | 12.0906493052 | 23.4260856861 | 10.9119789578 | 1.0541623643 | 5.7917945580 | 17.6282210957 | 21.5093610843 |
| 2026102203/q_only_G1 | 12.5989387206 | 23.9052213620 | 11.5293038680 | 1.0517009047 | 5.7854507330 | 18.0845522071 | 21.7264722007 |
| 2026102203/r_only_G1 | 12.2421636013 | 23.6608443027 | 11.0750496926 | 1.0897209442 | 5.8089253694 | 17.7434971053 | 21.5472744911 |

Previously, a FP32 native baseline on 414 images was validated only for fixed submissions; this round of ten Selection submissions does not include their respective Octave original baselines. Independent audit preservation of input, order, matrix structure, score inversion, source, and seven metric domains is maintained, without re-computing AP.

# # Object-Level Recovery and Cost

Project-level matching uses a fixed confidence ≥0.25 and same-category IoU ≥0.5, with ignore handling. The false positive count here comes from project-level matching, not derived from official AP curves. A hit in F but not in Q; recovery is the intersection of A and R hits; stable by-pass loss occurs when F and Q both hit but R fails; net recovery is defined as recovery minus stable by-pass loss. This is separated from the change in total hits between R and Q.

| seed | Effective Targets | A | recovery | Stable By-Pass Loss | net recovery | same-target bidirectional agreement | Total Hits between R and Q | False Positives between R and Q | False Positives between R and F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026102201 | 38518 | 884 | 566 | 330 | 236 | 232 | +186 | +206 | +1032 |
| 2026102202 | 38518 | 960 | 571 | 322 | 249 | 226 | +220 | +317 | +623 |
| 2026102203 | 38518 | 856 | 548 | 282 | 266 | 246 | +226 | +460 | +1536 |

| seed | Configuration | hits | false positives |
|---|---|---:|---:|
| 2026102201 | fp32 | 15032 | 10459 |
| 2026102201 | q_all | 14822 | 11285 |
| 2026102201 | q_only:G1 | 14813 | 10140 |
| 2026102201 | r_only:G1 | 15008 | 11491 |
| 2026102202 | fp32 | 15032 | 10459 |
| 2026102202 | q_all | 14624 | 10765 |
| 2026102202 | q_only:G1 | 14842 | 10180 |
| 2026102202 | r_only:G1 | 14844 | 11082 |
| 2026102203 | fp32 | 15032 | 10459 |
| 2026102203 | q_all | 14842 | 11535 |
| 2026102203 | q_only:G1 | 14832 | 10098 |
| 2026102203 | r_only:G1 | 15068 | 11995 |

75 components' per-group event records, hit/false positive rates, pre-fixed maximum shares, HHI, concentration levels across the first 1/3/5 groups, and the intersection of three sub-object-level event sets are all preserved in the machine summary. The same target appears repeatedly across the three calibration ranges without increasing independent sample size; component concentration is not used as a basis for selecting new candidates or seeds.

# # Original Candidate Box Numerical Thresholds

Each image fully intact 8400 with all candidate boxes retained, box tolerance `0.5+0.005×|F|` and score tolerance `0.002+0.02×|F|`. The F value of Selection is this initial generation reference, with no fabricated historical Selection reference; 109 debug images have previously stored arrays that are strictly equal between old and new. Consistency, recovery, and AP cannot substitute for original floating-point qualification.

| seed | Configuration | Number of Passed Images | Bounding box element exceeds tolerance | Score Element Deviation | Maximum Absolute Box Error in Pixels | Maximum Absolute Score Deviation |
|---|---|---:|---:|---:|---:|---:|
| 2026102201 | fp32 | 546/546 | 0 | 0 | 0.0000000000 | 0.0000000000 |
| 2026102201 | q_all | 0/546 | 9085570 | 549554 | 519.4683685303 | 0.8795857877 |
| 2026102201 | q_only:G1 | 0/546 | 7405846 | 399667 | 486.6114959717 | 0.8204062749 |
| 2026102201 | r_only:G1 | 0/546 | 8949308 | 479080 | 482.8195800781 | 0.6326627880 |
| 2026102202 | fp32 | 546/546 | 0 | 0 | 0.0000000000 | 0.0000000000 |
| 2026102202 | q_all | 0/546 | 9705904 | 546741 | 520.9848785400 | 0.8359379359 |
| 2026102202 | q_only:G1 | 0/546 | 8218053 | 391816 | 452.5705871582 | 0.7538055405 |
| 2026102202 | r_only:G1 | 0/546 | 8611407 | 479570 | 471.3612213135 | 0.6666393429 |
| 2026102203 | fp32 | 546/546 | 0 | 0 | 0.0000000000 | 0.0000000000 |
| 2026102203 | q_all | 0/546 | 9170310 | 541377 | 467.7140808105 | 0.7781235129 |
| 2026102203 | q_only:G1 | 0/546 | 7075608 | 390096 | 372.2428131104 | 0.7224353030 |
| 2026102203 | r_only:G1 | 0/546 | 8053589 | 470019 | 412.4452819824 | 0.5973686576 |

# # Independent Re-verification and Preservation

Saved artifact replay 5460 instances of raw data, a total of 45, 864, 000 candidate boxes; 6552 instances of complete FP64 difference and 6552 instances of NMS/ coordinate reconstruction/target records are consistent. 109 new debug raw files match the previously sealed reference strictly. The checklist 26, 946 members show consistent hash values; 84 source bindings are preserved, and 230 runtime Python source code files are maintained according to the audit record.

NMS uses an independent CPU-based full-candidate processing pipeline but shares a frozen torchvision kernel. Target matching is re-executed using a fixed-source project implementation. State checks are based on the saved tensor descriptions and replay summaries from the original execution, without claiming independent reconstruction of tensors that were not separately saved. Audit verification relies on preserved GT and content pre-check evidence, and does not re-read original Selection images or labels.

Final state verification confirms the completion acknowledgment and original process identity; the actual runner has terminated. The original 548 closeout group has 187 members, and the new 546 execution has 22 members prior to execution, with consistent candidate hash values. The original 548 failure, old report, contract, weights, and historical `FAIL_FIXED_TOLERANCES`, `FAILED_UNCHANGED` are preserved.

# # Route Location

The internal fixed G1 development validation within Gate1 has been completed. Independent scientific validation, as well as the advantages over random or traditional controls under the same budget, have not been achieved in this phase. The holdout set remains sealed. The current QDQ simulation is not equivalent to actual INT8/TensorRT/Orin deployment; speed, power, and energy per frame have not been measured in this stage.

The main remaining challenges are the independent confirmation of Gate1 and the full Orin deployment in Gate2. The numerical guardrail established in Gate0.5 remains unchanged. Dependencies on cross-backend equivalence require additional validation and do not necessitate the creation of a new Gate number. Future independent data access, candidate/control matrices, or device execution will require separate freeze scopes and corresponding data/device evidence. No additional candidates, seeds, module scans, or holdout opening are automatically added in this phase. Previously suspended periodic checks remain inactive, and there are currently no pending polling tasks for training, inference, or AP runs.

# # Machine Evidence

- [Contract Execution](../../docs/DATA_AND_ARTIFACTS.md)
- [Completion Acknowledgment](../../docs/DATA_AND_ARTIFACTS.md)
- [Independent Saved Artifact Audit](../../docs/DATA_AND_ARTIFACTS.md)
- [List Identity Bridging](../../docs/DATA_AND_ARTIFACTS.md)
- [Complete Seed Summary](../../docs/DATA_AND_ARTIFACTS.md)
- [Full Delivery Verification](../../docs/DATA_AND_ARTIFACTS.md)
- [v25 State Snapshot](../../docs/DATA_AND_ARTIFACTS.md)
