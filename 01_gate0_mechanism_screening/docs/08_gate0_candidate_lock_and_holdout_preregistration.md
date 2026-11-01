# Gate 0 Selection: Candidate Locking and Holdout Pre-Registration



> Selection Decision: **Lock `G1+G5` as the sole holdout candidate.** G1 serves as a necessary control, and G1+G3 is excluded from the holdout candidate comparison.

# # Candidate Comparison

| Configuration | recovery_count | Recovery rate | Collateral misses | Net recovery | same-target bidirectional agreement | Stable-target retention | MAC Share | Incremental threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G1 | 94 | 78.99% | 13 | +81 | 69 | 99.10% | 10.32% | Single-module baseline |
| G1+G5 | 100 | 84.03% | 8 | +92 | 83 | 99.45% | 22.35% | All passed |
| G1+G3 | 90 | 75.63% | 9 | +81 | 67 | 99.38% | 22.14% | 3 criteria failed |

# # Rationale for Locking

G1+G5 achieves 6 more recoveries than G1, reduces 5 collateral misses, improves net recovery by 11, increases bidirectional object-level matches by 14, and maintains MAC share at 22.35% without exceeding the 25% threshold. In contrast, G1+G3 shows no improvement in net recovery, and actually decreases both net recovery and bidirectional object-level matches by recovery_count.

Therefore, according to the pre-registration rules, the unique candidate is locked as:

```text
r_only:G1+G5
```

# # Holdout Execution Pre-Registration

Holdout contains 150 images, 722 instances of FP32 object-level detections and 46 full W8A8 additional missed detections. In the next stage, only the following modules are allowed to intervene:

1. `r_only:G1+G5`: the unique candidate, used for primary independent validation;
2. `q_only:G1+G5`: the corresponding destructive side, used to verify bidirectional same-target consistency;
3. `r_only:G1`: a necessary low-cost single-module control, used to assess whether G5's incremental gain is reproducible on holdout;
4. `q_only:G1`: the corresponding single-module destructive-side control;

No changes are allowed after reviewing the holdout results, including switching to G1+G3 or introducing any new module combinations.

# # Holdout Decision Rules

Since the holdout contains only 46 full W8A8 additional missed detections, the primary report should present the original counts and proportions, without mechanically applying the Selection criterion of 'at least 20 instances'.

G1+G5 is eligible to proceed only if it satisfies the following conditions:

- Recovery rate is at least 15%;
- Stability of target retention is at least 98%;
- Net recovery is positive;
- At least 1 pairwise consistent samples of the same individual target appear on both the corrupted and recovered sides;
- Compared to G1's single-module performance, the recovery_count, net recovery, and bidirectional same-target metrics cannot all be worse;
- The results cannot be dominated by a single image or a single category.

If the direction is opposite to Selection, net recovery is not positive, stable retention rate is below 98%, or there are no bidirectional consistent samples, then Gate 0 is transitioned to Pivot/Stop, and the Selection result is not considered reproducible.

# # Evidence Boundaries

- The current candidate selection is based solely on Selection.
- The existing results use PyTorch Q/DQ simulation for W8A8, and do not represent TensorRT INT8 performance, power consumption, or Jetson deployment outcomes.
- Passing a holdout only indicates that the core mechanism is worthy of advancing to the next stage; it does not mean that the paper's hypothesis has been finally established.
