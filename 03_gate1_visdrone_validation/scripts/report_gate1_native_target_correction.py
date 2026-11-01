"""Write the Chinese result report using audited saved outcomes only."""
from audit_gate1_native_target_correction import N,P,OUT,A


def main():
    decision=P.read(OUT/"machine_decision.json")
    audit=P.read(OUT/"saved_delivery_audit.json")
    diagnostics=P.read(OUT/"saved_result_diagnostics.json")
    receipt=P.read(A/"completion_receipt.json")
    by_label={m["branch"]:m for m in decision["metrics"]}
    treatment="native_intervention_correction"
    delta=decision["differences"][treatment]
    base=delta["mse_r"]
    if base["descriptive_joint_AP_up_hits_not_down_FP_not_up"]:
        conclusion='A relative improvement in joint performance is observed on the current development subset compared to the pre-training baseline, but independent scientific validation or methodological contribution is not yet established.'
    elif base["false_positives"]<0 and base["hits"]<0:
        conclusion='The new version reduces false positives while decreasing true object detection; it fails to achieve the joint objective of preserving more targets and reducing false positives.'
    else:
        conclusion='The new version fails to meet the required joint improvement target; the full comparison must be retained, and local metric changes should not be packaged as methodological success.'
    hard=delta["native_hard_correction"]
    if hard["descriptive_joint_AP_up_hits_not_down_FP_not_up"]:
        extra='A relatively matched set of typical hard examples; a descriptive phenomenon of joint improvement is observed in the intervention group in this round; however, a single seed and reused development data cannot prove general additional value.'
    else:
        extra='For the group of typical hard examples with matched intervention, no joint advantage in AP, detection, or false positive rates is observed. This round cannot claim that intervention information provides additional contribution based on this result.'
    names={"fp32_reference":'Original FP32 reference',"legacy_q":'Original max-value calibration Q',"legacy_r":'Original max-value calibration recovery G1',
           "mse_q":'MSE calibration Q',"mse_r":'MSE calibration recovery G1, shared starting point',"ordinary_qat":'Existing standard QAT',
           "hard_example":'Previous version standard hard examples',"intervention":'Previous version intervention correction',
           "localization_only":'New: only improves localization',"native_hard_correction":'New: native object-level standard hard example correction',
           "native_intervention_correction":'New: native object-level intervention correction'}
    lines=['# Gate1 Native Training Objective Alignment Correction: Completion Results', "", '2026-10-26, Beijing time.', "",
           "**"+conclusion+"**", "", extra, "",
           '## Fixed Development Results', "",
           "The following are fixed 128 images from the training_val subset, which cannot be conflated with the original 414-image total set. AP uses the official Python image with native consistency validated against real-world data; the new submission does not include re-running the native official program. Target counting is based on the project matcher's results at a fixed confidence level of 0.25 and IoU0.5, and is independent of the AP internal count.", "",
           '| Branch | AP50–95(%) | Detected true targets | false positives |', "|---|---:|---:|---:|"]
    for label,m in by_label.items():
        counts=m["project_counts_conf025"]
        lines.append(f"| {names[label]} | {m['metrics_percent']['AP_all']:.6f} | {counts['hits']} | {counts['false_positive_detections']} |")
    lines += ["", 'Each new arm trained on 512 images for 3 epochs and 96 optimizer updates, using the fixed final epoch 3 without selecting the best epoch or seed. Initialization, training members, batch order, optimizer, and range calibration are identical. Matching the number of additional supervision items does not establish identical measured FLOPs or wall-clock time.', "",
              '## Difference between the Correction Intervention Group and the Primary Control', "",
              '| Comparison Object | AP Change (in percentage points) | Detection Change | False Positive Change |', "|---|---:|---:|---:|"]
    for label in ("mse_r","ordinary_qat","intervention","localization_only","native_hard_correction","fp32_reference"):
        d=delta[label]
        lines.append(f"| {names[label]} | {d['AP50_95_percentage_points']:+.6f} | {d['hits']:+d} | {d['false_positives']:+d} |")
    lines += ["", 'All three pre-specified thresholds are retained and cannot be selectively chosen as the final winner:', "",
              '| threshold | Baseline: Detection / False Positive | New Ordinary-Hard Examples: Detection / False Positive | New Intervention: Detection / False Positive |', "|---|---:|---:|---:|"]
    for threshold in ("0.2","0.25","0.3"):
        cells=[]
        for label in ("mse_r","native_hard_correction",treatment):
            ct=by_label[label]["threshold_curve"][threshold]
            cells.append(f"{ct['hits']} / {ct['false_positive_detections']}")
        lines.append(f"| {threshold} | "+" | ".join(cells)+" |")
    lines += ["", '## What Was Changed in This Round', "",
              "The original version used an additional positive example classification with IoU-based supervision, which potentially conflicted with the detector's native TAL soft-target strategy. The new version removes this extra positive classification and instead only adds IoU localization loss for selected ground truths; the additional classification term for erroneous boxes retains the current TAL target from the same iteration, penalizing only the parts with scores above that target.", "",
              'The active region introduces an extra gradient on logits that is non-zero only when p−t≥0, and zero when below the target, meaning the target does not participate in gradient computation. Testing confirms this does not directly boost detection scores; shared weights are still influenced by other losses, so this cannot guarantee that every output decreases after training. This mathematical property is not a novel scientific contribution.', "",
              'The three new arms are localization ablation, ordinary hard example correction, and intervention correction. The two correction groups are matched by category, confidence level, and error geometric type; each image contains at most 8 error channels. The native ignore mask and positive sample priority are retained; localization errors, misclassification, or boxes currently assigned as foreground are not arbitrarily labeled as pure background.', "",
              f"Fixed 601 intervention GTs; total additional erroneous channels{audit['error_channels_selected']}items, distributed as:", "",
              '| Training Error Type | Count |', "|---|---:|"]
    type_names={"same_class_overlap":'same-category overlap',"other_class_overlap":'overlap with other categories',"near_object_low_IoU":'near-object low IoU',"low_overlap_remainder":'low overlap residual terms'}
    for k in N.KINDS:lines.append(f"| {type_names[k]} | {audit['error_types'].get(k,0)} |")
    lines += ["", f"GT overlap selected by intervention/hard examples{audit['hard_intervention_positive_overlap']}items, error channel overlap{audit['hard_intervention_error_overlap']}items. The sample pool, type, and quantity are fixed before training, and all three arms' results are retained.", "",
              '## Actual Supervision Usage Records', "",
              'Count the cumulative occurrences of GT or channel across batches over three epochs, not the total number of distinct targets. Whether erroneous channels are active or are assigned positive labels by TAL varies with the model, and the pre-matched count cannot be equated to identical effective gradients at each step.', "",
              '| Group | Number of Actual Assigned Localization GTs | Number of Retained Erroneous Channels | Number of Active Penalty Events | Number of Ignored Channels | Number of Erroneous Anchors Currently in Foreground |', "|---|---:|---:|---:|---:|---:|"]
    for arm in N.ARMS:
        u=audit["auxiliary_usage_totals"][arm]
        lines.append(f"| {names[arm]} | {u['positive_objects']} | {u['selected_error_channels_kept']} | {u['active_error_channels']} | {u['ignored_error_channels']} | {u['error_anchors_currently_foreground']} |")
    lines += ["", '## Post-hoc Diagnosis of the Same Target and False Positives in Geometry', "",
              f"Specific targets recovered under original MSE with G1 recovery{diagnostics['initial_G1']['G1_rescued']}items. The table below continues to track the same target ID; it is descriptive results and does not replace overall AP or independent acceptance.", "",
              '| Group | Original G1 Recovered Target | Relative Startpoint New Hits | Relative Startpoint Lost Hits |', "|---|---:|---:|---:|"]
    for label in ("ordinary_qat","intervention",*N.ARMS):
        t=diagnostics["target_summary"][label]
        lines.append(f"| {names[label]} | {t['kept_initial_G1_rescues']} | {t['new_hits_vs_initial_R']} | {t['lost_hits_vs_initial_R']} |")
    lines += ["", 'The following follows the previously defined post-hoc geometric grouping: first, same-class IoU ≥ 0.5; then, other classes with IoU ≥ 0.5; next, maximum IoU in [0.1, 0.5); finally, the remainder with IoU < 0.1. Grouping is based only on existing false positive annotations, without modifying matches or AP. This grouping is not an official error classification or causal explanation; low overlap remainder does not imply all are purely background.', "",
              '| Branch | Same-class Overlap | Cross-class Overlap | Localization Neighborhood | Low Overlap Residual |', "|---|---:|---:|---:|---:|"]
    for label in ("mse_r","ordinary_qat","intervention",*N.ARMS):
        t=diagnostics["FP_geometry_counts"][label]
        lines.append(f"| {names[label]} | "+" | ".join(str(t[k]) for k in N.KINDS)+" |")
    lines += ["", '## Completion, Saving, and Numerical Qualification', "",
              f"Final completion acknowledgment is complete, actual process has exited, stderr is 0 bytes; main process duration is{receipt['elapsed_seconds']:.1f}seconds. The process exits after bounded waiting ends, the exit code is not directly captured, thus exit0 is not claimed, on-site record is null. A total of 4992 new image forward passes are completed within budget, with 9 additional reconstructed instances independently saved; the forward pass of the new teacher model is 0. Reference, calibration, baseline, and old model inference are all reused, and the matching of saved teacher arrays is constructed for re-computation of new supervision and separately saved.", "",
              f"10 pre-checks passed; audit re-verified{audit['manifest_members_verified']}new saved members,{audit['reused_bindings_verified']}reused bindings and{audit['authorized_input_members_rehashed']}authorized input members. 9 rounds of model floating-point states are all FP32, quantization range and BN are frozen, three groups of real weight updates are performed; the final state is strictly equal when independently reconstructed via CUDA. For the 11 branches, seven AP/AR aggregations are recalculated from saved arrays, and three new groups' full 8400 candidate thresholds are recalculated from saved raw data, without any model forward pass or full AP matching re-execution.", "",
              '| New Group | Full Candidate Threshold Passed Image Count | Maximum absolute difference between CPU and CUDA |', "|---|---:|---:|"]
    for arm in N.ARMS:
        g=audit["new_arms_guard"][arm]
        lines.append(f"| {names[arm]} | {g['passed']} / 128 | {decision['CPU_CUDA_max_abs_by_arm'][arm]:.6f} |")
    lines += ["", 'Pass for integrity preservation and CUDA reconstruction equality does not imply full candidate numerical qualification; numerical consistency across CPU/CUDA is unverified. The original numerical threshold remains unchanged, and neither confidence scores nor NMS subsets are used as substitutes. A formal 30-epoch training process, with real-world evaluation on 414 images at the native level, maintains consistency with prior FAIL cases and the first version of interface failures, all of which are preserved as-is.', "",
              '## Research Judgment and Next Steps', "", conclusion, "", extra, "",
              'If proceeding further, it is essential first to verify whether improvements in localization and candidate ranking can genuinely preserve targets under the same false positive rate. Reducing scores or extending training duration alone cannot establish this. No automatic seed, epoch, or threshold search will be conducted in this round, nor will cloud costs be incurred.', "",
              'Quality-aware scoring and its coordination with classification and localization have prior work, such as [GFL](https://arxiv.org/abs/2006.04388); task imbalance in quantization training has also been discussed in [HQOD](https://arxiv.org/abs/2408.02561). This round only draws upon the problem awareness and does not fully reproduce these methods; thus, this approach itself is not claimed as a novel contribution.', "",
              'This round remains a floating-point W8A8 QDQ development with labeled, non-fused Conv layers, not real INT8 deployment. Only one training seed is used, and these development images were previously used for diagnosis and route adjustments. Therefore, they cannot be considered an unseen test set or independently validated for scientific acceptance. The filename prefix proxy is not a validated independent scenario. Selection load is not read, holdout remains sealed; no new timing checks or cloud resources are deployed.', "",
              'Evidence directory: `artifacts/gate1/20261026_native_target_correction_pilot_v1`. Primary files include contract, completion acknowledgment, actual exit observation, `saved_delivery_audit.json`, `machine_decision.json`, `saved_result_diagnostics.json`, and final `delivery_validation.json`. The latest entry is `artifacts/gate1/latest_execution_state.json`.']
    path=P.ROOT/"docs/69_gate1_native_objective_aligned_correction_results.md"
    content="\n".join(lines)+"\n"
    if path.exists():
        P.require(path.read_text(encoding="utf-8")==content,"Existing report differs; never overwrite")
    else:
        with path.open("x",encoding="utf-8",newline="\n") as f:f.write(content)
    print(P.json.dumps({"report":P.identity(path),"conclusion":conclusion,"intervention_value":extra},ensure_ascii=True))


if __name__=="__main__":main()
