# Gate 0 Selection: Single-Module Ranking and Pre-Registration of Dual-Module Combination



> Conclusion: G1 is the only strong candidate that passes all single-module thresholds. The combination experiment was pre-locked to G1+G5 and G1+G3 prior to result inspection; the holdout remains sealed.

# # Unified Ranking

| Rank | Module | Status | Destruction side overlaps with full INT8 | recovery_count (rate) | Collateral misses | Net recovery | same-target bidirectional agreement | Stable-target retention | MAC Share |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | G1 | Strong candidate | 88 | 94(78.99%) | 13 | +81 | 69 | 99.10% | 10.32% |
| 2 | G5 | Low priority | 19 | 16(13.45%) | 6 | +10 | 2 | 99.59% | 12.03% |
| 3 | G3 | Low priority | 9 | 19(15.97%) | 12 | +7 | 2 | 99.17% | 11.82% |
| 4 | G7 | Eliminated | 13 | 10(8.40%) | 6 | +4 | 0 | 99.59% | 17.78% |
| 5 | G8 | Eliminated | 6 | 7(5.88%) | 3 | +4 | 0 | 99.79% | 9.25% |
| 6 | G4 | Eliminated | 6 | 12(10.08%) | 10 | +2 | 0 | 99.31% | 13.29% |
| 7 | G6 | Eliminated | 3 | 3(2.52%) | 3 | +0 | 0 | 99.79% | 12.86% |
| 8 | G2 | Eliminated | 8 | 16(13.45%) | 19 | -3 | 1 | 98.69% | 12.65% |

Ranking criteria are: status (Strong Candidate, Low Priority, Eliminated), net recovery, bidirectional same-target count, recovery_count, and lower MAC share. This rule applies only to the Selection module.

# # Single-Module Conclusions

- G1 is the only strong candidate: recovered 94 targets, net recovery +81, 69 bidirectional matches at the same object level, and accounts for 10.32% of MAC.
- G5 and G3 show only minor positive gains and limited bidirectional evidence; they are retained as sources for combination candidates.
- G2, G4, G6, G7, and G8 fail to meet net gain or bidirectional consistency requirements and are not included in the combination pool.

# # Frozen Two-Module Combinations

1. Main combination: `G1+G5`, accounting for 22.35% of MAC proxy. Run `q_only:G1+G5` and `r_only:G1+G5`.
2. Backup combination: `G1+G3`, accounting for 22.14% of MAC proxy. Run `q_only:G1+G3` and `r_only:G1+G3`.

Both combinations must be executed and fully reported. The second combination cannot be selected based on the first, and no new combinations may be introduced and then selected based on the best result.

# # Incremental Threshold for Replacing G1 with a Two-Module Combination

A combination is only allowed to replace the G1 single-module solution if it simultaneously satisfies the following conditions:

- recovery_count is at least 99, i.e., at least 5 more than G1's 94;
- net recovery is at least 86, i.e., at least 5 more than G1's +81;
- At least 98% stability of target retention;
- At least 69 bidirectional same-target matches, not worse than G1;
- The combined MAC contribution of the two modules does not exceed 25%;

If neither combination passes, lock G1 as the holdout candidate; if both pass, compare net recovery, bidirectional object-level matches, and MAC share in sequence.

# # Execution Order

1. `q_only:G1+G5`
2. `r_only:G1+G5`
3. `q_only:G1+G3`
4. `r_only:G1+G3`
5. Aggregate and lock in a single candidate; only after this step is the unique candidate allowed to run on holdout;

# # Evidence Boundaries

- All current results come from Selection and PyTorch Q/DQ simulation of W8A8, not from TensorRT INT8 or Jetson real-world testing;
- The holdout was not used for module-level inference intervention, ranking, or combination selection;
- Currently, we can only say that G1 shows strong signals in Selection, not that it has been independently validated or reached the conclusion stated in the paper;
