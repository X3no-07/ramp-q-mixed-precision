# q_all 深度卷积的已保存运行内核诊断

7 个原生深度卷积中，7 个可通过完整 NVTX 层名、同线程 CUDA API 区间和唯一 correlationId 精确关联到运行内核。

这是内核身份记录，不是整数算术指令证明。实际输入/权重 INT8 格式、命名 tactic、运行内核身份和算术指令证据分开记录。

原严格结论保持：88 个卷积具备 INT8 输入/权重格式，81 个具备命名整数 tactic，7 个未暴露 tactic；严格审计仍为 PRECISION_REVIEW_REQUIRED。

没有重新运行引擎、访问板子或修改源 SQLite/报告。
