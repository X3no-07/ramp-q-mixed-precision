# Current status and next steps

**Research snapshot: October 29, 2026. English edition: November 2, 2026.** The latest completed experimental branch is defined by report 90 and the Gate 1 v54 state record.

## Latest completed experiment

| Method | AP50–95 (%) | Detections at confidence 0.25 | False positives at confidence 0.25 |
|---|---:|---:|---:|
| Continued standard QAT | 11.985899444 | 10552 | 7728 |
| Standard hard-example training with gradient protection | 11.918156643 | 10486 | 7494 |
| Intervention-based sampling with gradient protection | 11.877794522 | 10443 | 7494 |

All three arms completed training, scoring, and saved-artifact audits. Intervention-guided correction did not outperform standard QAT or hard-example training under the same protection rule. Both predefined joint criteria failed, and this correction branch was closed.

The evidence covers one training seed, the repeatedly used 414-image development set, and floating-point QDQ. It does not establish real INT8 performance or independent scientific acceptance for this candidate.

## Proposed next direction

Report 91 asks whether quantization perturbations alter dynamic label assignment and thereby change the supervision used for training. The proposed controlled experiment swaps assignment objects between forward passes. The code establishes a possible feedback path, but its harmfulness and the effectiveness of cross-assigned supervision remain unproven.

- [Report 90: terminal review and closeout](../03_gate1_visdrone_validation/docs/90_gate1_guarded_correction_terminal_review_and_closeout.md)
- [Report 91: supervision decoupling and minimal validation](../04_research_roadmap/docs/91_research_direction_review_supervision_decoupling_and_minimal_validation.md)
- [Selected machine-readable result summary](../results/gate1_v54_summary.json)

Earlier statements such as “not yet tested” describe the dates of their respective reports. They do not override this status summary or the latest terminal decision.
