# Next Gate: Independent validation of object-level quantization failure and recovery using VisDrone



Date: 2026-10-22. Document type: Next-stage research and acceptance plan; status is **DRAFT_NOT_FROZEN_NOT_EXECUTED**. This document completes the route assessment and scheme design, but does not indicate that the VisDrone experiment has been run, nor does it constitute a frozen execution contract.

# # 1. Conclusion: The next stage is Gate1, which first establishes independent validation conditions

**Following the original plan, the next stage is Gate1: a clean cross-dataset validation using VisDrone.** It must answer: whether the phenomenon observed on underwater data—where quantization causes partial disappearance of object-level targets, and partial recovery through high-precision module restoration—can hold in a different scenario on data not involved in the selection process.

Gate0.5 has completed this round of real-world deployment evaluation, and the predetermined G1 object-level recovery existence criterion has been met; however, the original full PyTorch/ORT numerical guardrail has not yet passed. Therefore, progress toward Gate1's data, model baseline, and preregistration can be advanced, but this advancement cannot be stated as 'full numerical migration has passed'. Conclusions involving cross-backend equivalence or pure module arithmetic causality still require new evidence.

| Stage | The question to be answered | Current status |
|---|---|---|
| Gate0: RUOD-R simulation screening | Does object-level quantization lead to missed detections and recoverable signals? | Completed; supports continued research, but the superiority of model selection is not established |
| Gate0.5: Early authenticity check on Orin | Can the same object-level recovery be achieved under the selected real-world deployment configuration? | The current evaluation is complete, with the primary criterion passed; numerical equivalence not established |
| **Gate1: Independent Validation on VisDrone** | Does the phenomenon remain stable when switching to non-underwater scenarios? Does object-level information enhance model selection value? | Next phase; this report is a proposal, experiments have not yet begun |
| Gate2: Comprehensive Orin Deployment Evaluation | Can the selected solution simultaneously achieve detection quality, speed, and energy efficiency? | System measurements to be scheduled after Gate1 |

The numbering follows [the next step from the original Gate0 conclusion](../../../01_gate0_mechanism_screening/docs/00_gate0_final_assessment.md) and [Section 10–11 of the original research direction](../../../01_gate0_mechanism_screening/docs/research_design/01_research_scope.md). The following segments represent the workflow within Gate1.

# # 2. Why it is worth continuing, and what remains unestablished

The most valuable empirical result currently is: with a fixed set of 150 images and 865 annotated objects, the board-end FP16 model detects 722 objects, while the full INT8 model detects 687 objects; after per-object alignment, an additional 44 missed detections are identified.

| Metric | Only recover G1 board-end results | Implication |
|---|---:|---|
| Recover additional missed detections | 40 / 44,90.91% | Most of the same object-level detections are recovered |
| Stable detections come with additional misses | 3 / 678 | Recovery is not without cost |
| Retention rate of stable detections | 99.5575% | Most original stable detections are retained |
| Net recovery of targets within FP16 hit set | +37 | 40 recoveries minus 3 collateral misses |
| Total change in hits | 687 → 721,+34 | Also affected by changes in FP16 originally missed targets |

These results and ranges are detailed in [Gate0.5 Final Report](../../../02_gate05_orin_deployment/docs/39_gate05_complete_150_image_acceptance_results.md) and [Final Machine Judgment](../../../docs/DATA_AND_ARTIFACTS.md). At the time of writing this report, 45 files in the final list were verified, with identical sizes and SHA-256 hashes; this check confirms that the referenced delivery artifacts have not changed and does not constitute new experiments.

Entry into Gate1 is justified because the recovery phenomenon is sufficiently clear and merits validation with new data; the need for new validation is equally clear:

- The 150 images used in prior diagnosis cannot be reused as an unseen out-of-distribution set for the new method; 44 and 40 are total counts across all sizes, and not all can be considered small objects.
- There remains a discrepancy between the on-device quantization configuration and the corresponding ORT target hit status; the original PyTorch/ORT guardrail remains `FAILED_UNCHANGED`.
- The compiled non-quantized floating-point path and fusion strategy are not identical; auxiliary `q_only_G1` and the build constraints of the main configuration also differ. Current support for deployment-level recovery is insufficient to establish causal validity of pure G1 arithmetic.
- The labeled traditional sensitivity method in Gate0 previously selected the same combination; the unlabeled output error baseline is also close. Current evidence does not prove that the new selection rule outperforms the traditional method. See [Original Control Pre-registration](../../../01_gate0_mechanism_screening/docs/09_gate0_budget_matched_control_preregistration.md).

# # 3. Gate1 Requires Separate Validation of Three Issues

**First, can the G1 assumption be reproduced across scenarios?** G1 originates from existing RUOD-R evidence and can be treated as an external prior candidate before reviewing VisDrone results. If G1 fails, it must not be replaced with another module after seeing the results and then claimed as 'successful migration' of G1.

**Second, can the object-level module signal be reproduced on independent data?** It is allowed to scan G1–G8 in the VisDrone Selection (development selection set) according to pre-defined rules to select candidates, and then validate them on the sealed Holdout (final confirmation set). The same target must satisfy both destruction and recovery relationships. This issue is reported separately from the fixed G1 migration.

**Third, does object-level selection outperform traditional baselines?** Compared under the same high-precision computational budget with labeled overall sensitivity, unlabeled output error, and fixed random control. If the same configuration is ultimately selected, record 'selection collision'; the same engine and same result cannot be interpreted as algorithmic superiority. When mechanism is reproduced but selection does not dominate, the research remains focused on object-level quantization reliability.

# # 4. First, Prepare Data, Models, and Numerical Boundaries Correctly

## # 4.1 Using the Image Detection Task in VisDrone

Use VisDrone-DET, not the video tracking task. The official provides training, validation, and test-dev datasets for image detection, with publicly available test-dev annotations and non-public test-challenge annotations. This proposal recommends using **test-dev as the final sealed confirmation set** for this project; public availability does not imply permission to use it for early-stage selection. [Official Data Entry](https://github.com/VisDrone/VisDrone-Dataset)

Here, 'across datasets' refers to the reproduction of the phenomenon in a new data domain and corresponding detection model, **without directly treating the RUOD-R weights for sea cucumbers, scallops, etc., as the baseline for VisDrone detection**. Prioritize using the YOLO11n architecture to establish a correctly categorized floating-point model for VisDrone; re-verify the G1–G8 source layer mappings, convolution counts, and MACs. If using an existing VisDrone checkpoint, verify its training and weight selection data; weights that include test-dev data cannot be directly applied to this project's independent validation. If switching architectures, a structural correspondence must be established, and the old model's 88 convolutions or budget percentages cannot be simply copied.

## # 4.2 Data Usage Must Be Separated Before Viewing Final Results

| Data portion | Recommended usage | Isolation requirement |
|---|---|---|
| The training portion within the official train split | Train or fine-tune floating-point models | Split by image or scene identity, as follows |
| The reserved training-validation portion within train | Early stopping and selection of floating-point checkpoints | Do not use the final confirmation set for weight selection |
| The reserved calibration portion within train | Estimate quantization ranges | Do not read labels to participate in label-free ranking; must not overlap with Selection or Holdout sets |
| The reserved engineering debugging portion within train | Label conversion, export, and numerical diagnostics | Allow debugging, but prohibit packaging into a confirmation set |
| Official val set | Selection: module scanning, candidate evaluation, and baseline comparison | Clearly designated as development data; results are not used for conclusive validation |
| Official test-dev split | Final Holdout set | Execution and scoring only after model, candidate, calibration seeds, rules, and analysis code are locked |

The actual list, count, and split seed will be specified in the contract after data inventory. Check file SHA, duplicates, and near-duplicates; group images that belong to the same scene or sequence; do not assume independence based solely on different filenames. The final set may undergo isolation completeness and reproducibility checks, but no early reading of detection results, drawing of error cases, or label-based tuning is allowed. If source checks reveal it has been used in candidate selection, disclosure is required and a separate confirmation set must be reserved.

VisDrone annotations include ignored regions and `others`; they cannot be simply discarded and then all predictions within those regions counted as false positives. The original annotations, class mapping, and ignore rules must be preserved, and object-level counting and detection evaluation must use a consistent effective object definition. [Official annotations and ignore rules](https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/README.md)

## # 4.3 Floating-point baseline first passes, then analyzes quantization

First complete label mapping, input preprocessing, bounding box reconstruction, effective target matching, and detection evaluation for FP32 and FP16 to confirm the model actually detects targets in new data. The main input is recommended to use 640×640, batch=1, to minimize variables; if engineering debugging indicates a need to change resolution, the resolution must be finalized before the Selection scan and the budget recalculated accordingly. It is not acceptable to adjust the size after reviewing the Holdout set.

It is recommended to use **3 predefined calibration seeds, each with 64 images**, as the initial scale scheme for the same floating-point weights. All quantization configurations within each seed share the calibration images, ranges, and base weights, with only the declared module precision being varied. The three calibration seeds assess calibration sensitivity, not three independent datasets, and do not prove cross-training-seed stability. Repetition of training seeds or the second model belongs to subsequent extensions and cannot be prematurely stated as already covered.

## # 4.4 Parallel management of numerical and scientific validation, without mixing judgments

Gate1 can first conduct simulation mechanism studies within the same backend and explicitly fixed Q/DQ semantics. This branch introduces false positives relative to FP32 and only supports simulation conclusions; it does not require reclassifying existing failures as passes. The recovery precision for the simulation branch is FP32, while the native deployment branch is FP16. Corresponding unquantized modules in `q_only` are also operated at their respective floating-point precision in each branch, and the two intervention types are separately tracked.

If executing the native TensorRT branch of VisDrone, relative to native FP16 definition, new false negatives are introduced, and FP32→FP16 changes are separately listed. A re-examination of each candidate layer's actual precision, non-quantized paths, and fusion differences is required. Reviews from the RUOD-R engine cannot replace evidence from new weights, new graphs, or new engines.

Any claim of 'PyTorch→ORT→TensorRT numerical equivalence' must be supported by a complete graph, clearly defined inputs, and pre-specified tolerances. The original numerical guardrails `rtol=0.005, atol=0.5` and scores `rtol=0.02, atol=0.002` are preserved and cannot be rescued by widening tolerances. If only native recovery experiments are conducted with compilation differences, it must be explicitly stated that the conclusion remains at the deployment configuration level; it cannot replace rigorous arithmetic causality validation. When a failure is diagnosed, only the dependent branch is paused, while data preparation and independent simulation validation can still proceed.

# # 5. Experiments to be conducted on Selection

## # 5.1 Basic matrix and bidirectional interventions

| Configuration | Purpose |
|---|---|
| FP32, FP16, full W8A8 | Establish floating-point and quantization baselines, separating the effects of FP16 and INT8 |
| `q_only:G1` to `q_only:G8` | Quantize only this module and check which originally detected targets disappear |
| `r_only:G1` to `r_only:G8` | Fully quantize the module and recover it, checking whether the same missed targets reappear |
| Pre-locked combinations and controls | Evaluate the actual performance of combinations, not by simply summing up individual module gains |

The minimum scan volume is 3 baseline configurations plus 16 single-module interventions per calibration seed, totaling 19 configurations; the floating-point baseline can be reused when weights, inputs, and environments are identical and identity verified. This is a planned scope, not the actual number of executions. Final Holdout evaluation only assesses the locked candidates, necessary destructive cases, and baselines—no re-scanning to select winners is allowed.

## # 5.2 Traditional baseline and budget

Retain two baseline categories from Gate0: the labeled baseline is ranked by the total additional misses caused by selection using `q_only`; the unlabeled baseline is ranked by the normalized mean squared error of NMS-processed outputs on the calibration curve, with cosine distance provided only for descriptive purposes. This directly evaluates whether 'object-level recovery information' adds value beyond existing proxies.

First define single-module and two-module budget tiers, both charged by high-precision MAC share in the current VisDrone model. Fix the exact caps, budget windows, candidate pool, tie-breaking rules, and random seeds before reading Selection results. The previous two-module design's 25% cap and ±2-percentage-point window may inform the design, but must be recomputed for the new model rather than copied. Use3 fixed random controls per budget tier where possible; if the eligible pool is too small, report the entire pool instead of expanding the budget until a favorable combination is sampled.

The object-level candidate generation rules must be pre-defined as code and formulas: primary sorting uses the net recovery rate of additional misses, with stable target retention rate as a numerical guardrail, bidirectional same-object rate as a predefined tie-breaker, followed by MAC and module index to break ties. Risk-weighted and low-contrast weighting analyses remain exploratory and are not promoted to primary metrics upon seeing results. If combinations are generated based on single-module scores, their bidirectional behavior must be validated on Selection before a candidate is selected under the frozen rules; interactions between combinations may render scores non-additive.

MAC is a budget proxy, not actual latency or power consumption. Any comparison of 'same speed' or 'same energy' is deferred to Gate2's real-world measurements.

# # 6. How to determine that a target has truly returned

Each annotated target uses a stable ID composed of 'image identity plus original annotation physical row number'. Fixed category matching, one-to-one IoU ≥ 0.50, and main confidence ≥ 0.25 are adopted, following existing deterministic matching tie rules. Thresholds of 0.20 and 0.30 are reserved for sensitivity analysis and do not replace the primary thresholds. Both original predictions and post-processed results are preserved to distinguish changes in scores, localization, and NMS suppression.

Define separately for each backend, calibration seed, and dataset:

- `F`: targets detected by the floating-point baseline in this branch; `Q`: targets detected by full W8A8 quantization.
- `A = F \ Q`: additional misses introduced by quantization; `S = F ∩ Q`: stably detected targets.
- `R`: the set of targets detected under a recovery configuration; `D`: the set of targets detected under the corresponding quantized-only configuration.
- recovery_count `|A ∩ R|`;collateral_miss_count `|S \ R|; the net recovery count is the difference between the two.
- Recovery rate `|A ∩ R| / |A|`;net recovery rate `(recovery_count−collateral_miss_count) / |A|`.
- Stable retention rate `1−|S \ R|/|S|`;bidirectional same-object count `|A ∩ R ∩ (F \ D)|`.

When the denominator is zero, mark it as 'not applicable / insufficient evidence', without setting it to 100% or treating it as a pass. Also report total detections, false positives, precision, recall, and detection AP; configurations that recover misses but introduce a large number of false positives cannot be declared overall better solely based on recovery rate.

Small objects must be defined separately. It is recommended that this study adopts the original annotation area to define tiers: areas less than 32² pixels are classified as small, 32² to less than 96² as medium, and the rest as large. The area ratio to the original image and the scaled dimensions are also recorded. This is the tiering standard adopted in this study and must not be conflated with VisDrone's official AP_S definition, nor should all recovered targets be labeled as small objects.

The official VisDrone AP aggregates IoU from 0.50 to 0.95 with its own maximum detection count setting; the official interface uses AP with maxDets=500. The object-level analysis in this project may adopt confidence=0.001, NMS IoU=0.7, and max_det=1000, but must be named separately from the official AP and validated in interface testing for truncation and ignore handling. Custom statistics must not be misrepresented as official scores. [Official evaluation interface](https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/evalDET.m)

Uncertainty should be assessed using paired bootstrap across image or scene groups, preserving all targets within the same image and all configurations during resampling. Results from the three calibration seeds are listed separately, and correlation within the same image group is preserved during aggregation—do not treat the same target appearing three times as three independent samples. Sample size, minimum detectable gain, and interval schemes are estimated and frozen on the development dataset, and cannot be replaced by a fixed 150-image sample size without a proper statistical design.

# # 7. Acceptance criteria and stopping conditions

The following represents the **proposed evaluation framework**, not a frozen threshold, nor the results already achieved by VisDrone. The 15% recovery and 98% stability retention are continuations from the previous Gate0, and cannot be considered sufficient criteria for the new task simply because the old data was easily attainable. The formal contract must additionally address minimum effective event counts, scenario group numbers, sample size/intervals, allowable degradation in false positives or mAP, and multiple comparison adjustments; without these, the Holdout will not be opened.

| item-by-item | It is recommended to require | Record accordingly when not satisfied |
|---|---|---|
| Engineering entry point | Data and model identity, category/ignore handling, matching tests, declared precision, and reference semantics are verifiable | `NOT_READY`; first complete the corresponding stage, and do not treat tool failures as mechanism failures |
| Fixed G1 migration | Validate G1 on the new Holdout according to preregistration rules; 3 calibration seeds report | If failed, retain G1 without reimplementation, and do not replace the primary candidate |
| Independent mechanism reproducibility | Lock in a candidate with recovery rate ≥15%, stability retention ≥98%, net recovery positive, and positive net recovery direction for small objects, with bidirectional evidence at the same target level | Record `FAIL` when data volume is sufficient but direction fails; record `INCONCLUSIVE` when event counts are too low or intervals are too wide |
| Robustness | Net recovery direction is consistent across the three calibration seeds; no conclusion is fully dependent on a single threshold in the 0.20/0.30 analysis; dominant behavior is checked by scenario and category | When dependent on a small number of scenarios, single category, or single threshold, narrow the claims and refrain from asserting a stable, generalizable mechanism |
| Model superiority | Independent Holdout-based paired comparison under the same budget supports the intended practical gains and uncertainty requirements, while satisfying false positive and mAP guardrails | Same selection records `COLLISION`; unclear differences record `NOT_ESTABLISHED` |
| Numerical equivalence or pure arithmetic causality | Each complete numerical contract and comparability control are established | Single-column unverified; no mechanism replication or substitution is required |

The requirement that 'at least one module passes' must refer to Selection being made and then confirmed in the Holdout set. It cannot simply scan the 8 modules in the Holdout and report only the best one. When both the fixed G1 assumption and the adaptive candidate assumption are tested, prior declaration of hierarchy, precedence, or multiple comparison correction must be made. Neither result alone should be considered jointly passed.

If the overall mechanism holds but G1 does not, investigate changes in domain-related modules; if the mechanism holds but there is no selection advantage, continue reliability analysis without packaging a new selection algorithm. If no stable bidirectional signal is observed on sufficient clean data, the generalization of 'a universal small-object quantization mechanism' should be halted, and the RUOD-R scope conclusion should be retained. If additional misses are minimal, this reflects insufficient event evidence under current settings, not an intentional calibration-induced phenomenon.

# # 8. Implementation Order, Deliverables, and Current Readiness

| Order | Specific Work | Completed Evidence |
|---|---|---|
| 1 | Audit the VisDrone dataset, categories, ignored annotations, duplicates, scene groups, and existing weights | Data sources and file listings, split checks, weight origins; final dataset access boundaries |
| 2 | Establish qualified floating-point models and engineering debugging sets, and verify G1–G8 against input and evaluation criteria | Floating-point baseline report, source layer mapping, conversion/matching checks, engineering and numerical issue log |
| 3 | Freeze research questions, seeds, budget, candidate generation, baselines, metrics, and acceptance contracts | Pre-registration document and code with timestamps and SHA hashes; key fields without missing values |
| 4 | Conduct single-module scanning only in the Selection phase, with combination confirmation and control selection | Full configuration results, including failures, collisions, and cases not reaching budget; candidate locking records |
| 5 | Release the Holdout set, evaluate it in one go according to the locked matrix, preserve original outputs and independently recompute | Per-target matrix, case-level examples, stratification by size, category, threshold, uncertainty, and sub-conclusions |
| 6 | Determine paper candidates based on Gate1 conclusions, then freeze Gate2 performance protocols | Candidate identity, deployment qualification, measurement plan; speed and energy conclusions are generated only after these are established |

The formal Gate1 deliverable must include at least: pre-registration plan, dataset and model identity list, calibration and split checklist, source module mapping, locked candidate and baseline records, numerical and actual precision review, per-target results, detection AP, false positives, and stratified results, independent re-computation records, and sub-conclusion acceptance reports. Any unproven sub-components must be explicitly left blank or marked as 'not completed'; no single PASS should cover all items.

The current local audit scope covers `data/prepared`, `data/raw`, `data/downloads`, and `runs` of the 07 engineering and 05 detection engineering components. The observed data are primarily from RUOD, with no identified VisDrone-named data or corresponding training directories. **This only indicates the absence of confirmed Gate1 resources in the reviewed path, not the absolute absence on the entire machine.** No data connection, data download, model training, or new inference runs were performed in this round.

Therefore, the next step is to **establish a floating-point baseline for VisDrone data and freeze the final confirmed dataset**. The on-device team is responsible for data preparation, baseline training/validation, and scheme freezing; the Orin platform will be used for native validation under clear engineering contracts and candidate selection, and will subsequently handle Gate2 performance measurement. Training hardware, storage, and initial power consumption have not yet been measured, and thus this report does not provide any unverified timeline commitments.

# # 9. Explanation of References and Historical Records

This report defines the 'current status' based on the [No. 39 Final Result](../../../02_gate05_orin_deployment/docs/39_gate05_complete_150_image_acceptance_results.md), [Final Judgment JSON](../../../docs/DATA_AND_ARTIFACTS.md), and [Final List](../../../docs/DATA_AND_ARTIFACTS.md) as of 2026-10-21. Previous READMEs, research status documents, and direction texts still retain their original 'not yet validated' sections, which should be understood in the context of their respective dates and cannot override the latest results; these documents have not been modified or overwritten, and evidence has been preserved accordingly.

The route follows [Original Direction Text](../../../01_gate0_mechanism_screening/docs/research_design/01_research_scope.md) and [Gate0 Final Conclusion](../../../01_gate0_mechanism_screening/docs/00_gate0_final_assessment.md). The original numerical failure basis is [Original Full Model Pre-check](../../../docs/DATA_AND_ARTIFACTS.md); the definitions of the two control groups are further detailed in [Same Budget Control Pre-registration](../../../01_gate0_mechanism_screening/docs/09_gate0_budget_matched_control_preregistration.md) and [Label-Free Output Error Baseline](../../../01_gate0_mechanism_screening/docs/03_gate0_label_free_output_error_baseline.md). These prior experiments do not constitute VisDrone results.

The official VisDrone website and evaluation code were verified on 2026-10-22, and links are provided adjacent to relevant discussions. Formal execution requires fixing the actual downloaded version and the evaluation code commit, and cannot rely solely on dynamically changing website links to ensure reproducibility.
