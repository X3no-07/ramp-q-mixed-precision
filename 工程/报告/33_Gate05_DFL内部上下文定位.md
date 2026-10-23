> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate 0.5：DFL 内部上下文定位

2026-10-20，接续[双卷积上下文](32_Gate05_双卷积上下文定位.md)。此前两个图像前缀均未复现 corrected q_all 的格式筛选失败。本轮选择结构不同的 DFL：Softmax→QClipDQ→1×1 Conv→Reshape，比较相同修正拓扑的 NONE／OBEY。

**结果：NONE 构建并执行成功，OBEY 在约 1.54 秒触发与完整 corrected q_all 相同的 `filterQDQFormats::4421`／`candidateRequirements` 断言，没有引擎和推理。首次在 9 节点的 DFL 切片中复现同类失败，并分离出此上下文的开关相关差异。完整模型的具体失败节点和唯一原因仍未证明，Gate 0.5 未通过。**

## 冻结范围与内部输入

DFL 切片包含 9 个节点、7 个初始化器，一个 `/model.23/dfl/conv/Conv`，输入为 `/model.23/dfl/Transpose_output_0` FLOAT `[1,16,4,8400]`，输出为 `/model.23/dfl/Reshape_1_output_0` FLOAT `[1,4,8400]`。输出是 DFL 距离张量，不是检测框。该处没有输出 QDQ，不能预先要求实际 Conv 输出为 INT8。

冻结协议：`artifacts/gate05/20261020_dfl_context_protocol/protocol.json`，SHA256 `b2723549e6acee51c86aeba3b72c8451601ba4add6088f46f7443536a1ab74ee`。实际失败源图 SHA256 为 `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee`。

内部输入不能直接使用图片 NPY。本轮从修正 ONNX 图精确抽取 `images` 至 DFL Transpose 的上游依赖闭包，包含 638 节点、576 初始化器、72 个 Conv，以原固定图片的预处理 NPY 在 CPU 运行一次生成 logits fixture。这会执行较大的 backbone／回归头上游部分，不输出完整检测器结果。两份 DFL 尾部原图／改写图各 CPU 一次，共用该 fixture，要求局部输出字节一致。它不证明原版／修正版上游相同，也不代表 TensorRT 上游输入分布。

每个 native case 只尝试一次，构建上限 300 秒、总监护 900 秒；成功后执行相同内部输入两次。保留全部 parser 约束和真实配置标志，只校正一处 Clip alpha/beta 至 ±127，不设置或清空层精度／输出类型。实际 Conv 输入、权重、输出和 tactic 分别按 inspector 记录，不以请求标志代替真实精度。

上游子图 SHA256 为 `df9e2ea0d03168f17f770bf0689a4a75c90f607df75ff98962e4eaa92764c018`。实际只运行一次 CPU 上游生成 fixture，NPY SHA256 为 `cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab`，原始数组 SHA256 为 `799c5f7875edd7f9a3e46c8beba833b69b56baf04f76f060015dab2132cbb52a`；范围 −20.61652374267578 至 17.923843383789062，形状与有限性通过。

原／改 DFL 尾部各 CPU 一次，输出 `[1,4,8400]` 逐字节相同、最大差为 0，原始数组 SHA256 为 `3e4280d6a97b12d4e0c33eeb1e0a462ab3d457cef8eca9ce2ac5554816f45436`。CPU 使用 `CPUExecutionProvider`、`ORT_DISABLE_ALL`、4 线程。修正尾图 SHA256 为 `a8a30c71a20d6b6dd3d031d23ce8dc8672cc62abe6c3eb8725b324fd2e600092`；所有所选源节点、初始化器及图像 NPY 的 before/after 字节身份保持。

密封包 `artifacts/gate05/20261020_dfl_context_probe.zip`，21,224,122 bytes，SHA256 `05b9b50e4ddc294151121044facef5e1711ac368c793ca5232c1c1a557c42b9f`；spec SHA256 `934e075aa58dde1829b600d3d011f5ad51c4f1da1af78f7791944f492781b14d`，manifest SHA256 `3836e82ed53afb14fe16d3b9431864c0c4a45fd0d3418c7c06a4f6fcd46ed1d0`。

启动前独立审查 2,845 项通过，无 blocker，报告 `artifacts/gate05/20261020_dfl_context_review/pre_native_review.json` SHA256 `c4ff2428c1ef6b291538d1fbb188502e2fa298ae977ab826b4ab5c8332a36ea4`。板端再次完成纯标准库输入校验，并核验无遗留构建任务、同一启动编号后，已在新目录 `/home/nvidia/rampq/gate05_dfl_context_20261020/gate05_dfl_context_probe_20261020` 启动。monitor PID 32201、私有组 32206、guard PID 32212，guard 状态 WATCHING。

## Native 结果

每个 case 严格尝试一次，没有重跑失败 case 或更改设置。

| case | Builder API 秒 | 构建结果 | 固定输入执行 |
|---|---:|---|---|
| code_clip_none | 52.211651 | 成功，engine 31,119 bytes | 两次，输出有限且逐字节相同 |
| code_clip_obey | 1.538202 | 返回空序列化结果，TensorRT 内部断言 | 未生成 engine，未推理，无 raw 输出 |

runner 共 63.993636 秒，外层 monitor 共 69.3 秒，终态 `BUILD_SCRIPT_OK`。这个状态表示脚本按冻结协议保存了一个普通构建失败并完成批次，**不是两个引擎均通过**。guard 最终 COMPLETE、未干预、私有组为空；没有超时或资源监护停止。以上用时不是推理性能测量。

OBEY 日志第 14 行与完整失败日志第 275 行同为：

```text
[optimizer.cpp::filterQDQFormats::4421]
Assertion !n->candidateRequirements.empty() failed.
All of the candidates were removed, which points to the node being incorrectly marked as an int8 node.
```

随后出现 `buildSerializedNetwork::751` 的空引擎断言。完整日志没有具体节点名；小图复现不能反向证明完整图中的某个命名节点就是失败点。

## 实际精度与数值

NONE 引擎恰有一个实际 Conv，唯一对应 `/model.23/dfl/conv/Conv`，权重数 16，无 bias、激活 NONE。实际格式为：

- 输入：`Four wide channel vectorized row major Int8 format`。
- 权重：`Int8`。
- 输出：`Row major linear FP32`。
- 命名 tactic：`ampere_fp32_icudnn_int8x4_128x32_relu_interior_nn_v1`。

输入、权重和命名整数 tactic 已单独核验，输出如实记录为 FP32。这符合本轮不预设 INT8 输出的协议；没有输出 QDQ，不应把它改写成“所有输入输出均 INT8”。外部输入输出仍为 FLOAT/LINEAR，形状与原契约一致。

NONE 两次输出原始字节完全一致，engine SHA256 为 `a0b23d8fce8a25bbf6ca5de9bf638dabbc9be11db23e56ad16cb2efc3ce2a33e`。相对冻结 CPU DFL 输出：

| 描述指标 | 值 |
|---|---:|
| 不同元素数 | 11,779 / 33,600 |
| 最大绝对差 | 1.9073486328125e-06 |
| 平均绝对差 | 7.795968226024083e-08 |
| RMS | 1.664742957045647e-07 |

这只是同一内部输入上的描述性数值差异，没有定义新通过容差或改判原护栏。OBEY 无输出，NONE/OBEY 跨组数值比较正确记为 `UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED`，不能填成零差或相同输出。

## 实际配置和解析器约束

两组图、内部 fixture、Clip ±127 校正、解析器约束相同。实际配置标志中，只有 `OBEY_PRECISION_CONSTRAINTS` 改变；INT8＋FP16 开启、TF32／PREFER 关闭、workspace 1024 MiB、avgTiming=1、DETAILED 和外部 FLOAT/LINEAR 均保持。

每组解析网络 14 层，保留 2 个显式 INT8 precision 约束（ActivationDQ、WeightDQ）和 2 个显式 INT8 output type 约束（ActivationQ、WeightQ），查询无错误。Clip 导入属性校正前后逐层约束相同，没有层 precision／output setter 或 reset。不能把“脚本没设置约束”写成“解析器没有约束”。

额外只读核对了历史完整图的 parser 元数据：这四个 DFL Q/DQ 层的设置标志、已设置类型和输出张量名／shape／dtype，与本轮小图语义相同。两个记录器的 JSON schema 不同，首次直接比字典产生本地审计断言；保留该记录后，按显式字段映射重新核对通过，未运行 CPU/native 或改原证据。对应 `full_parser_constraint_correspondence.json` SHA256 `9a3dcc476ee59b3bad0040da1744969f7ce41c0e2bf095cd709535b8033f48d8`。这只加强约束来源的对应，不证明哪个约束导致了编译错误。

本次按固定顺序各运行一次。它证明此 DFL 修正上下文在记录的条件下呈现 NONE 成功／OBEY 同类失败，尚未分别隔离 Softmax 邻接、code_clip 拓扑、导入属性修正或单个 Q/DQ 约束的作用。

## 原始证据与板端终态

归档为 `artifacts/gate05/20261020_dfl_context_native/native_evidence.zip`，21,481,431 bytes，SHA256 `a53750401ccdbb7c008c14e75cadec80a586a05f2baa32f5e985abdff527bd24`。回执列出 88 份文件，另加回执元数据自身，清单、大小、SHA256 和 ZIP CRC 均已核验；回执 SHA256 `fe71bbf62229b969bcfc8331b06921991803847668daddb43fb831465aec25e2`。最后 SSH 再次核对，板端 88 份文件和归档仍全部匹配。

根审计 `root_native_audit.json` SHA256 `507c1902886fac7490927f0f97a754a114c0a6d3637948a0aada05ed2d3247f4`；独立保存证据审查 309 项通过，`independent_native_review.json` SHA256 `18c4dbb38ea7f4f01f9906bad8d347fe198f301f7999598baf341e1bef213b46`。审查均没有重新推理或改变原始树。

第一次 SCP 下载在实验和归档均完成后遇到 connection reset，21,430,272-byte 不完整文件及失败记录另存。重新确认板端同次启动、归档哈希未变后，仅限速重新下载一次，最终精确核验通过；没有重新构建或运行失败实验，也没有用不完整 ZIP 作为证据。该次传输中断的具体原因未定位。

最终客户端观察时间为 2026-10-20 12:18，板端仍为 2026-10-17 18:29，二者分别记录，未校时。启动编号仍为 `692c7544-b8bd-415b-939c-b11cdbd9f7f0`，monitor／runner／guard／两个 builder 共五个 PID（32201、32206、32212、32219、32300）均不存在，无本轮构建任务。风扇、jtop active，NRestarts=0，可用内存 4989 MiB。15W/mode 2 查询仍有非管理员 EMC 读取权限提示，不作完整特权检查通过结论。

原完整 q_only_G1 引擎 SHA256 仍为 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`，完整 corrected q_all 失败目录仍无引擎；原失败、权重、标定尺度、CPU／PyTorch 护栏及先前成功均保留。

## 结论与下一控制

现在已将完整模型的同类编译断言缩到 9 节点 DFL 切片，并在同一修正图中观察到 OBEY 开关相关差异。这是一个局部编译复现；它没有证明完整图的唯一根因，没有证明关闭完整模型 OBEY 能构建成功，更没有完成实机目标级破坏／恢复、数据集或性能验证。原完整 PyTorch/ORT 护栏保持 `FAILED_UNCHANGED`，Gate 0.5 仍为 `INCOMPLETE`。

下一步优先用已封存的原 FLOAT Clip→Q→DQ DFL 尾图，复用本轮 fixture 和 CPU 参考，另冻结原结构 NONE／OBEY 两组控制，与当前结果构成局部拓扑×开关对照。原结构不做 code_clip 导入属性校正，其他设置、parser 约束及实际精度审计保持；它仍把 code_clip 拓扑与声明的导入校正视作组合因素，不能直接区分两者。原尾图 SHA256 为 `b60be2c2d196ed1fb4e962446f374d9e53d066476172cfa1db415194b2306d90`，不需重跑 72 Conv 上游或重新选输入。

该计划仅提出、尚未冻结执行，见 `next_control_plan.json`，SHA256 `5b0469faa1f055b2e3e148c1302c65824a04f31b062abd48fbf109b70040aa7d`。若以后剥离 Softmax，必须另冻结正确的 Softmax 输出 fixture，不能直接将当前 logits 喂给新切片。当前没有 reset parser 约束、完整图 NONE/PREFER 重试、PE 跳转、恢复矩阵或 150 图评估。

阶段检查点：`artifacts/gate05/20261020_dfl_context_native/checkpoint.json`；当前状态：`artifacts/gate05/current_execution_state.json`。
