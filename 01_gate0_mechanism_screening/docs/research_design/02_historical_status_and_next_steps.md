# 02 Current Conclusions and Next Steps



> Update Date: 2026-09-11

# # Shortest Conclusion

The research line has not changed:

> Identify which specific object-level targets disappear under INT8 quantization, locate the corresponding modules, restore those modules, and verify whether the same individual target can be recovered.

Current status:

- Object-level failure attribution and recovery: **feasible, proceed**;
- G1 signal mechanism: **already reproduced in RUOD-R independent holdout**;
- G1+G5 performance across three random budget combinations: **passed**;
- Novel selection algorithm outperforms traditional methods: **not established**;
- Real TensorRT/Orin deployment: **not tested**;
- VisDrone generalization: **not tested**.

# # Why not change the problem

Traditional methods also select G1+G5, but they only reject the claim that 'selecting G1+G5 alone is sufficient to be novel', without rejecting the following core lines:

1. Track FP32 hits to INT8 misses per object;
2. Perform destructive and recovery-side analysis per module;
3. Require bidirectional correspondence for the same object;
4. Deduct collateral misses caused by recovery;
5. Deploy recovery modules in real mixed-precision settings.

# # Next Steps

First conduct a small-scale authenticity check on Orin Gate 0.5, then proceed to VisDrone Gate 1 clean validation.

Orin minimum configuration: FP16, full INT8, G1 recovery, G1+G5 recovery, G1+G2 recovery.

Only when real TensorRT can reproduce the sequence 'disappearance of the same object — recovery via relevant modules — reappearance of the same object' does the current mechanism gain deployment significance.

Detailed roadmap see `01_research_scope.md`, experimental facts and file entry points are available in the engineering directory's `README.md`.
