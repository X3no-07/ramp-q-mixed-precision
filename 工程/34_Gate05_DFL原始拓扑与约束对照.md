# Gate 0.5：DFL 原始拓扑与约束对照

2026-10-20，接续[DFL 内部上下文定位](33_Gate05_DFL内部上下文定位.md)。此前修正 DFL 小图中，NONE 构建成功，OBEY 复现完整图的 `filterQDQFormats::4421` 断言。本轮补原 FLOAT Clip→Q→DQ 拓扑的 NONE／OBEY 两组，检查该局部失败是否也发生在原结构。

**结果：原结构同样 NONE 构建成功、OBEY 复现同一断言。局部 OBEY 失败不需要 code_clip 这组改动才能出现；同时，原 NONE 的输出仍有严重数值偏差，而修正 NONE 的误差很小。编译兼容与数值异常必须分开处理，不能把“能构建”写成“结果正确”。完整模型仍未修复，Gate 0.5 未通过。**

## 冻结范围与准备

冻结协议 `artifacts/gate05/20261020_dfl_original_protocol/protocol.json` SHA256 `81200cd4900a35f0533c45c7542f6dacc353203e8338a09b47cd24465cc0f6ed`。原始尾图 SHA256 为 `b60be2c2d196ed1fb4e962446f374d9e53d066476172cfa1db415194b2306d90`，来自原 q_all 图 `2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d`，9 节点、7 初始化器、一个 DFL Conv。

沿用先前认证的内部 logits fixture，NPY SHA256 `cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab`，FLOAT `[1,16,4,8400]`；沿用同 fixture 的原／改 CPU 输出，NPZ SHA256 `fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511`。本轮不重新执行 72 Conv 上游或局部 CPU，不选择新输入。输出仍是 `[1,4,8400]` 的 DFL 距离，不是检测框。

原结构不做 code_clip 的导入属性校正，Clip alpha/beta 只读取并保存，前后保持原样；保留解析器已有 precision／output type 约束。每组一次构建，限时 300 秒、总监护 900 秒，成功引擎才执行两次；实际 Conv 输入、权重、输出和 tactic 分项检查，不预设 INT8 输出。其他 INT8＋FP16、关闭 TF32／PREFER、workspace 1024 MiB、avgTiming=1、FLOAT/LINEAR 设置与前阶段一致。

静态复用检查通过：原尾图的节点、初始化器与 I/O 声明均精确对应原完整源图；fixture、CPU NPZ 和两份已保存 CPU 输出原始字节未变，新增 CPU／ORT／Torch／上游执行均为 0。密封包 `artifacts/gate05/20261020_dfl_original_probe.zip` 为 11,545,669 bytes，SHA256 `6ea5c52c4697339692d3d5e4efac01a46d033e34855a3ed8f60e4aa1d1a22709`；spec SHA256 `99d7b3179d9c540c06331a2135de9d645b2421722edb20ff5268e53c7b56f1ae`，manifest SHA256 `aa8aa071beb59e40e48f3c0022fcf187f430285b1651df200d7c554623f1d064`。

启动前独立审查 350 项通过，无 blocker，报告 `artifacts/gate05/20261020_dfl_original_review/pre_native_review.json` SHA256 `62158ccefba1f732f88904bce8cf92c24f0992ec9dc706988770dfc78482a5ed`。板端再核包和纯标准库输入检查后，在新目录 `/home/nvidia/rampq/gate05_dfl_original_20261020/gate05_dfl_original_probe_20261020` 启动。monitor PID 33046、私有组 33051、guard PID 33057（WATCHING）。

## 本轮原结构结果

原结构两组各一次构建，没有重新运行 CPU 或失败 case。

| case | Builder API 秒 | 结果 | 推理 |
|---|---:|---|---|
| original_none | 52.320608 | 构建成功，engine 31,054 bytes | 两次有限输出，原始字节完全一致 |
| original_obey | 1.508866 | `filterQDQFormats::4421`／`!n->candidateRequirements.empty()`，随后 `buildSerializedNetwork::751` | 无 engine、无 raw、未执行 |

runner 总计 64.027084 秒，monitor 总计 69.3 秒；`BUILD_SCRIPT_OK` 表示批次按协议结束，**不表示两组构建都通过**。guard 最终 COMPLETE、无干预、私有组为空，没有超时或资源停止。用时仅为诊断构建记录，不是推理性能。

实际原 Clip alpha/beta 在两组均为 `−0.9999907612800598 / +0.9999907612800598`，前后完全相同、explicit correction=false。代码没有 alpha/beta setter。每组解析网络均为 14 层，2 个 DQ precision INT8 约束、2 个 Q output type INT8 约束，无查询错误；未设置或清空层精度／输出类型。设置 OBEY 前后保留同一组 parser 请求。

原 NONE 的唯一实际 DFL Conv：输入 `Four wide channel vectorized row major Int8 format`，权重 Int8（16 个），输出 `Row major linear FP32`，无 bias、激活 NONE，命名 tactic 为 `ampere_fp32_icudnn_int8x4_128x32_relu_interior_nn_v1`。它与前轮修正 NONE 的输入／权重／输出格式和 tactic 名相同；**相同格式与 tactic 不等于数值相同**。

原 NONE engine SHA256 `583aaa40a0a5d90e87cf00fd8be7dd08dfe90eed72d5a678e369c4f0cfb8fa3e`，raw 数组 SHA256 `11be9724a648a8a04535876fa1e79e7ef1468c4eb463866313fca0b58f5247dd`。输出范围 0 至 0.9448731541633606，相对同一冻结 CPU 参考有 33,566/33,600 个元素不同：最大绝对差 `11.957853466272354`，平均绝对差 `2.3782942463764005`，RMS `2.8428724297345527`。因此原 NONE 的构建成功不是数值修复。

## 与前轮构成的局部四格

四格共享同一内部 fixture、CPU 参考、DFL 权重和尺度。原／改图的已保存 CPU 输出逐字节相同。两阶段实际配置除 OBEY 外一致，Q/DQ 显式精度请求语义相同；原／改结构与声明的导入校正为组合因素。

| 结构 | OBEY | 构建 | 相对 CPU 最大绝对差 |
|---|---|---|---:|
| 原 FLOAT Clip→Q→DQ | 关闭 | 成功 | 11.957853466272354 |
| 原 FLOAT Clip→Q→DQ | 开启 | 同类 4421 断言，无引擎 | 不可比较，无输出 |
| 修正 Q→INT8 Clip→DQ | 关闭 | 成功 | 1.9073486328125e-06 |
| 修正 Q→INT8 Clip→DQ | 开启 | 同类 4421 断言，无引擎 | 不可比较，无输出 |

两个 NONE 输出彼此有 33,566 个元素不同，最大绝对差 `11.957853466272354`。所有涉及失败 OBEY 的 raw 跨组比较均为 `UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED`，不能填成零差或相同输出。

原／改 OBEY 的日志均明确包含同一源位置和 candidateRequirements 断言，与历史完整 corrected q_all 的错误类别相同。这里的证据不再支持“该局部错误只能由 code_clip 新改写引入”的解释，但没有分离 Softmax 邻接、单个 Q/DQ 请求、张量形状或输出 FP32 等具体子因素，也不能证明完整模型的失败节点或唯一原因。

跨阶段保存数组审计为 `topology_flag_comparison.json`，SHA256 `88fcc89eaec465dc1ed5493834898ba76c16925d0a1364c785482848e01453bb`。这是同次板端启动、固定输入下的四次顺序观察，没有重复随机试验；没有为较小误差另设容差，也没有修改原护栏。

本轮与既有修正 DFL 两组构成跨阶段局部拓扑×OBEY 对照，保持原始失败和成功记录。code_clip 拓扑与其声明的导入属性校正仍是组合因素，不把它们当作两个独立因素；每格仅一次固定顺序观察，也不外推为随机重复试验或完整图唯一根因。

## 原始证据与终态

原始归档 `artifacts/gate05/20261020_dfl_original_native/native_evidence.zip` 为 11,644,748 bytes，SHA256 `9f027ada5086843e4b967a6ab4f5a4e1d5f23bc904e30b9913e8e619db7dd96d`。回执列出 84 份原始文件，另加回执元数据本身；清单、大小、SHA256、ZIP CRC 均通过，回执 SHA256 `faa8a17c485a6551439027608457258e767e28184bee4ba6595f019af2162571`。本轮限速下载一次成功，最终 SSH 再次核验全部 84 份板端文件和归档仍匹配。

根审计 `root_native_audit.json` SHA256 `91a4f2b67125cf2868a2353b44e0bacbef53e82be5e3c613a613e5df8ba7ed6c`；独立审查 470 项通过，`independent_native_review.json` SHA256 `3cda350ad9140c2a9b442f853868e8d527f269b20889f61ec55e36942f7ef7c0`；补充失败签名与运行终态审查 `independent_failure_signature_and_runtime_review.json` SHA256 `5f0f63da900ff6e9d6041c12cff92cbd0c6b2a0ccfe598ee4e9fb2bc0bf1ce35`。新、旧 raw 树均未修改，审计没有执行 CPU、engine 或 SSH。

35 次监护采样最高温度 50.312°C，最低可用内存约 2263.852 MiB，没有记录到监护内核失败标记。这只是本轮资源记录，不能外推为板子长期稳定或此前断连已解决。

最终客户端 2026-10-20 12:37、板端仍为 2026-10-17 18:48，分别记录、未校时。启动编号仍为 `692c7544-b8bd-415b-939c-b11cdbd9f7f0`，monitor／runner／guard／两个 builder 共五个 PID（33046、33051、33057、33064、33096）均不存在，无构建任务。风扇、jtop active，NRestarts=0，可用内存 5001 MiB；15W/mode 2 仍有非管理员读取 EMC 权限提示，不是完整特权检查通过。

前轮修正 DFL NONE engine 仍为 `a0b23d8fce8a25bbf6ca5de9bf638dabbc9be11db23e56ad16cb2efc3ce2a33e`，其 OBEY 仍无 engine。此前完整 q_only_G1 engine 仍为 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476`；完整 corrected q_all 仍无 engine。父阶段输入包、raw、检查点、原图、权重和尺度均保留。

## 结论和下一步

局部四格现已区分两类问题：两种 DFL 结构的 OBEY 均出现同类编译失败；在能执行的 NONE 分支，只有声明的修正组合显著减小了此固定输入上的数值偏差。不能据此宣称关闭全图 OBEY 已修复完整模型，也不能跳过原完整 PyTorch/ORT 护栏、实际精度审计、浮点／融合对照和独立数据集判据。Gate 0.5 仍为 `INCOMPLETE`，没有完整图重试、数据集、实机目标级因果或性能通过结论。

下一步优先保留数值较好的修正 DFL，另冻结 Softmax 之后的 8 节点 QDQ→Conv→Reshape 切片与正确概率输入，再比较 NONE／OBEY，检查显式 Softmax 邻接上下文的作用。必须从已保存 logits 经原精确 Softmax 生成新概率 fixture，新的局部 CPU 输出应与已有 DFL CPU 距离输出逐字节一致；不需重新运行 72 Conv 上游。该新切片、输入和 native 都尚未执行，不能直接将 logits 喂入裁去 Softmax 的图，不清空 parser 约束，也不自动推广到完整图或恢复矩阵。

该 8 节点／7 初始化器切片的静态来源和待冻结步骤见 `next_control_plan.json`，SHA256 `de9e44fcd058cd6d855174af6934f657e5c312ee5284f8463322c0c029160be2`。计划本身不等于冻结或执行；改变输入边界也会改变优化上下文，未来结果仍不能单独证明 Softmax 是完整失败的唯一原因。

阶段检查点：`artifacts/gate05/20261020_dfl_original_native/checkpoint.json`；当前状态：`artifacts/gate05/current_execution_state.json`。
