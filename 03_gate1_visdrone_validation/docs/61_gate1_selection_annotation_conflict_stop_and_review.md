# Gate1: Selection Annotation Conflict Shutdown and Independent Re-verification



**The original 548-member Selection attempted to halt at the content pre-check stage, without generating any Selection inference or AP. All 109 debug new adaptations with FP32 outputs have been fully element-wise and bit-by-bit consistent with the original sealed arrays.** The formal 30-epoch training, the FP32 candidate at epoch 30, the native consistency of 414 images, and existing debug quantization diagnostics retain their original conclusions.

# # Actual Execution

After the user responded 'continue' to the specific access issue in report 60, a new human access acknowledgment was recorded under fixed four configurations, three calibration seeds, and in the original order of the 548-member Selection, and the new source code and machine contract were frozen. There are 47 pure test items for member thresholds; 14 test methods and 22 sub-tests for the executor, all of which passed independent code review. The contract SHA256 is `0ca8d06382f4f92a33fbb2c8786d1080322a2f46eaab3b4fecfdc8d3ae16f420`.

The only attempt was initiated at 18:28:36 Beijing Time on 2026-10-24 and terminated at 18:29:06 with exit code 1, ending in final state `FAILED_PRESERVED_NO_RETRY`. A strict manual verification of 109 debug images was completed before reading the Selection images and original annotations to perform content identity, structural integrity, and duplicate annotation checks. The source code was first traversed in full for all 548 members, followed by duplicate group inspection; thus, the original content was indeed accessed within authorized scope, and it cannot be stated that 'Selection was completely unaccessed'.

Forward pass on real-world images performed only 109 debug iterations; the Selection model forward pass, three sub-recalibrations, and AP scoring all achieved 0. `shared_FP32`, three seed runs, `scores`, inference, and completion receipts are all absent. This conclusion is jointly supported by a fixed call order, 109 progress entries, exception stacks, final state receipts, and missing downstream artifacts, and is not based on system-level instruction-by-instruction tracing.

# # Conflicting Content

| Item | Member A | Member B |
|---|---|---|
| ID | `val/0000022_00000_d_0000004` | `val/0000023_00000_d_0000008` |
| Original non-empty annotation lines | 43 | 40 |
| score=1 and records for categories 1–10 | 41 | 40 |
| category=0 ignored regions | 1 | 0 |
| score=0 original records | 2 | 0 |

The original execution preserved evidence shows identical image SHA256: `0bea9d13456e6aa7a92ad1af21a89960eb66916855ed8a2c3d0371b1d7d37bd0`; both original annotations' SHA values match the frozen records. Subsequent independent audits verified the preserved evidence and metadata, with no further access to the original Selection images or labels.

Ignoring line order, the two complete eight-field annotation sets share only 1 identical record, with 42 and 39 unique records respectively; exact geometry also matches for only 1 record. The conflict cannot be explained by line reordering alone and involves geometry, category, and ignore semantics. Comparison by physical line position yields 43 differences, but a shared line position does not identify the same real target. Per-column difference counts therefore cannot be interpreted as changes to paired targets. The saved material cannot determine which annotation set is correct; neither set was selected, merged, or corrected.

# # Independent Re-verification and Preservation

`20261024_selection_content_failure_audit_v1/audit_result.json` is in state `PASS_SAVED_FAILURE_EVIDENCE_AND_109_DEBUG_REPLAY_AUDIT_SELECTION_ATTEMPT_REMAINS_FAILED`. It passes through verification based on failed evidence and debug outputs; the original Selection attempt remains in a shutdown state.

Independent recomputation of 109 new debug raw records, totaling 915600 candidate boxes and 12818400 elements, matches the old references with zero numerical tolerance and identical bit patterns. All physical lines and differences in duplicate groups were recomputed. Hashes of the 121 final members and 883 distinct bound files spanning sources, candidates, old deliveries, and old stages are unchanged before and after audit; all 684 members of the old stage manifest remain intact. The four recorded launcher, worker, and Python-shim PIDs are gone. Creation times and command lines were checked, and no unrelated process was terminated.

The complete 548 image saved-artifact auditor only completed preparation and 36 pure tests, with status `PREPARED_NOT_EXECUTED_ASSUMES_SUCCESSFUL_FUTURE_548_SCOPE`; its actual audit invocation was 0, and preparation pass cannot be used to substitute for the current Selection audit pass.

# # Future Scope

Specific recommendations prepared: temporarily isolate the above two items, while the remaining original members retain their original order 546 and are assigned to separate development contracts; membership selection is based solely on content identity and original label conflicts, without using prediction scores or AP values. It is proposed to fix the original candidate from round 30, four configurations, and three sub-sets; perform forward passes on real images `109 + 546 + 3×(64 + 6×546) = 10675` times, with a unique raw selection of 5460 samples, logical arrays, and full-element differences of 6552 samples; save and submit AP evaluations 10 times.

After the user responded again with 'continue' following the conflict notification and the proposal to isolate two items and establish a separate 546-image contract, a new, limited human-driven decision has been recorded. A new directory has been prepared, reviewed, and frozen for the 546-image contract. The failed 548-image contract, report 60, original member list, and failure site are left unmodified. This report is delivered only for the 548-image shutdown and revalidation; the actual progress of the 546-image work must be based on the new contract and acknowledgment, and cannot claim that the original 548 evaluation has been successfully reproduced.

The Selection process remains solely as development evidence; the Holdout set continues to be sealed. The original complete candidate box threshold, `FAIL_FIXED_TOLERANCES`, `FAILED_UNCHANGED`, is preserved. The real INT8, TensorRT, Orin, native scientific acceptance, and advantages over random/standard controls have not been demonstrated in this stage. The previous scheduled check has been deleted, and the current 30-second attempt failed to reconstruct the idle monitoring.

# # Checkable Files

- `../artifacts/gate1/20261024_selection_fixed_G1_execution_v1/attempt_terminal_manifest.json`: Complete list of the original attempt site.
- `../artifacts/gate1/20261024_selection_content_failure_audit_v1/audit_result.json`: Independent evidence verification and recheck.
- `../artifacts/gate1/20261024_selection_duplicate_resolution_proposal_v1`: Conflict analysis and new 546 scope recommendation.
- `../artifacts/gate1/current_execution_state_v23.json`: Snapshot of the original 548 attempt closeout; a new snapshot is established for the subsequent scope.
