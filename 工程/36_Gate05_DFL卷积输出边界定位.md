# Gate 0.5：DFL 卷积输出边界定位

本轮检查：上一轮不含 Softmax 的 8 节点图仍在 OBEY 下触发 `filterQDQFormats::4421`。现在只移除末端 Constant 和 Reshape，保留量化与卷积，观察错误在 Conv 输出边界是否仍然出现。

六节点对照已完成：NONE 构建和两次执行成功，OBEY 仍在构建阶段出现同一 `filterQDQFormats::4421`／`buildSerializedNetwork::751` 错误，无引擎、无推理。说明显式 Softmax、末端 Constant／Reshape 均不是此局部错误出现的必要条件；仍不能认定完整模型的唯一根因。原完整 PyTorch／ORT 护栏仍为 `FAILED_UNCHANGED`，完整 corrected q_all 仍无引擎，Gate 0.5 为 `INCOMPLETE`。

## 冻结范围

- 精确来源：完整 corrected q_all ONNX SHA256 `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee`。
- 来源和参考承接报告 35 的完成检查点，SHA256 `142eb219ccbeea141237cd1a89ecdb2c47c70b99d35159d9912ba8cea767c53f`；不修改已有封包、失败记录、尺度或权重。
- 新图 6 节点、7 个 initializer、1 个源 Conv。移除 `/model.23/dfl/Constant_1` 和 `/model.23/dfl/Reshape_1`，其余节点和 initializer 保持源序列化字节。
- 输入仍是已有 CPU 概率 `/model.23/dfl/Softmax_output_0`，FLOAT `[1,16,4,8400]`；NPY SHA256 `5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520`，不重新生成。
- 新输出是 `/model.23/dfl/conv/Conv_output_0`，FLOAT `[1,1,4,8400]`，表示带单通道维度的 DFL 距离，未解码为框。
- 只新增一次新图 CPU 执行。先核旧 `[1,4,8400]` 和新 `[1,1,4,8400]` 的 dtype、形状和 33,600 个元素，再比较 C-order 原始数组字节。新四维 expected 由旧数组纯 reshape 保存，另记文件身份；不要求形状头不同的整个 NPY 文件相同。
- `code_clip_none`、`code_clip_obey` 各构建一次、各限 300 秒，批次上限 900 秒。成功引擎执行两次，失败不推理；普通编译失败按协议继续，超时、资源停止、外部中断或输入失配中止批次。
- INT8＋FP16、无 TF32／PREFER、1024 MiB workspace、avgTiming=1、DETAILED、外部 FLOAT／LINEAR。仅按既有协议修正导入 Clip alpha/beta 为 ±127，不设置或 reset 层 precision／output type。

协议在新图准备与 CPU／native 执行前冻结：`artifacts/gate05/20261020_dfl_conv_output_protocol/protocol.json`，SHA256 `8f6a55f640915b84d7300d1595ebb895092ccf0374e40a27b6aaf812f406e09b`。

## 执行记录

唯一 CPU 校验已通过：CPUExecutionProvider、ORT_DISABLE_ALL、4 threads，仅执行新六节点图一次，未生成新概率或重跑 Softmax／72Conv／旧 DFL／完整图。新四维与旧三维输出原始数组字节完全相同，0 个元素不同，raw SHA256 `3e4280d6a97b12d4e0c33eeb1e0a462ab3d457cef8eca9ce2ac5554816f45436`。旧 NPY SHA256 `fd4bcdcee24cd53651fdd7a508423c802f3b432c759647204c7b748f22abd50d`，新 NPY 为 `d2f4327fcc792b864e420f1b1554a5b7b52b88fd27a8f01f6c07a0159c3e723b`；文件形状头不同，不是数值变化。

新图 SHA256 `805abb78ed4e4d5abc8041b2916dadc83370a14ef1776fcc0a03f163c16bcf81`。新四维 expected NPZ SHA256 `711fb41fa18bef712bd069c88a552e1bc918d1d1034b6f6d32d802cbd769d0c6`，只包含 `code_clip_output`。准备前后 72 份来源文件及封存前后两份 native 工具保持。

封包 `artifacts/gate05/20261020_dfl_conv_output_probe.zip` 为 11,908,335 bytes，SHA256 `78008952bd6355d4a197913c669c3a7b18ff1e80b119d88789d19f53baed4ba7`；spec SHA256 `9e9d31b55e4358d39a08a7869aab77582f094f1d6152599c727196e2b77c3897`，manifest SHA256 `104137a621b3ea5c65a019555a1d92fc8c876fe0bf9293b5ab7cfe2e6fe71f77`。

独立启动前审查 369 项全部通过，无 blocker，`artifacts/gate05/20261020_dfl_conv_output_review/pre_native_review.json` SHA256 `a3fa1038648c560045bd79d77a5dac027d16ceb1c04bdd1ce7e486d578f97747`。native runner 的纯标准库输入检查通过，未导入执行 runtime。

等待期间临时 SSH 密钥到期，随后通过交互式密码登录恢复，仅安装 USB 来源 IP 限制、两小时有效的任务密钥。恢复访问后重新核对全部来源身份，再唯一启动本轮两组 native 构建；未重新执行 CPU 或修改封包。

启动前 SSH 观察：仍是启动编号 `692c7544-b8bd-415b-939c-b11cdbd9f7f0`，无遗留构建进程；风扇和 jtop active，NRestarts=0，可用内存约 5005 MiB。15W/mode 2 的非管理员查询仍有 EMC 读取权限提示，未视为完整特权功耗检查。客户端和板端时钟分别记录，未校时。

## Native 结果与输出边界比较

新目录为 `/home/nvidia/rampq/gate05_dfl_conv_output_20261020/gate05_dfl_conv_output_probe_20261020`。monitor PID 38997、私有组 39002、guard PID 39008；两个 builder PID 为 39015、39049。各 case 仅构建一次，没有改变设置重试。

| case | Builder API 秒 | 结果 |
|---|---:|---|
| code_clip_none | 50.380678 | 引擎成功，27,583 bytes；同一概率输入执行两次，输出有限且重复字节一致 |
| code_clip_obey | 1.523681 | 构建阶段同一 4421 断言及 751 错误；无引擎、无 raw、未推理 |

runner 共 61.965597 秒，monitor 共 67.28 秒。`BUILD_SCRIPT_OK` 只表示按协议完成批次并保存普通编译失败，**不是两组构建都通过**。guard COMPLETE、无干预、私有组为空，无超时或资源监护停止。这些构建与监护用时不代表推理性能。

新边界的解析网络为 11 层，旧八节点边界为 12 层。具名 Q/DQ 请求仍相同：2 个 DQ precision INT8、2 个 Q output type INT8，无查询错误；Clip 修正前后请求保留。实际配置除 OBEY 开关外一致，未设置或 reset 层精度或输出类型。

NONE 唯一实际 Conv 对应 `/model.23/dfl/conv/Conv`。输入 `Four wide channel vectorized row major Int8 format`，16 个 Int8 权重，输出 `Row major linear FP32`，激活 NONE、无 bias；tactic 仍为 `ampere_fp32_icudnn_int8x4_128x32_relu_interior_nn_v1`。外部概率输入和四维距离输出均为 FLOAT／LINEAR。

NONE 引擎 SHA256 `961e821b6d2965078edc8209feb2b41929012a6522df027b71a2f83834056836`，输出数组 raw SHA256 `f503e64fd4f24ffde83dd60e62be2dc7e832b54d311c5f564eabebb07a831d84`。相对本轮四维 CPU 参考，11,779／33,600 个元素不同，最大绝对差 `1.9073486328125e-06`，平均绝对差 `7.795968226024083e-08`，RMS `1.664742957045647e-07`；仅作描述，没有新增通过容差。

旧八节点 NONE 输出 `[1,4,8400]` 与新六节点 NONE 输出 `[1,1,4,8400]` 在核形状、dtype、元素数后，C-order 数组字节完全相同，0 元素不同、最大差 0。两轮复用同一概率文件、权重、尺度、Q/DQ 请求和通用设置。两个 OBEY 均无输出，相关 raw 对比为 `UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED`。

与上一轮不同，本轮新旧 native 外部输入文件确实相同，变化在输出边界；这仍不能推出所有内部张量等价或完整图已修复。输出契约改变可能影响 native 格式选择，不能从一次局部比较唯一归因某节点。

根审计 `root_native_audit.json` SHA256 `cc6ff251cfe20b339c5d68ef9fd3d46d0de4f0f848ff33131673ceb7f6f0133e`，保存证据比较 `context_comparison.json` SHA256 `c02da4ed3cb4a37c8664243616eb744b53e431b94f052c25ad683ee69a8b9e1d`；均无旧引擎重跑。

## 原始证据与板端终态

原始归档 `artifacts/gate05/20261020_dfl_conv_output_native/native_evidence.zip` 为 12,152,044 bytes，SHA256 `07e583b3aac44cebd4008a6c0f1fda83000a9a71c4e940ae1c6bb63f3baa63cd`。精确回执列出 89 份文件，另加回执元数据自身，文件清单、大小、SHA256 和 ZIP CRC 均核验。回执 SHA256 `7aaa362c975ceaa3c3fb3d26e6c2362a6107c65c77f7b54081b47420b4051350`。一次限速下载成功，随后 SSH 重查板端 89 份文件及归档仍匹配。

最终客户端为 2026-10-21 01:04，板端为 2026-10-18 07:15，未校时；启动编号未变。monitor、runner、guard、两个 builder 五个 PID 均不存在，无构建任务。风扇与 jtop active、NRestarts=0，可用内存 4994 MiB，非管理员 EMC 查询限制保留。`final_board_observation.json` SHA256 `87dd6299f84a39fa281fd9eb8b15ed8ada47fbbd60243969fdc3082bb6bea209`。

上一轮八节点 NONE 引擎及更早 corrected DFL NONE 引擎保持，各自 OBEY 仍无引擎；完整 corrected q_only_G1 引擎 SHA256 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476` 保持，完整 corrected q_all 仍无引擎。原结构数值失败、原完整护栏及历史严格 INT8 缺口全部保留。

保存证据的独立审查 623 项全部通过，报告 `artifacts/gate05/20261020_dfl_conv_output_native/independent_native_review.json` SHA256 `f648aed6ad95e8f6d8d10c35b59cdebee6dd42a48f895792f600e34e021b18c4`。覆盖原始文件、图来源、实际格式／约束／flags、原始输出和板端终态；复核没有新增 CPU／native 推理或 SSH。

## 下一项局部控制建议

后续可单独旁路权重 Q/DQ：保留冻结概率、激活 Q→Clip→DQ、输出契约和源 FP32 权重原字节，只把 Conv 权重输入接回源 initializer。这样可区分显式权重量化支路在该局部失败中是否必要。新图改变了数学语义和 parser 请求上下文，必须另冻协议、保存自己的新 CPU 参考，不能套用旧六节点 W8A8 参考或称数值等价修复；FP32 initializer 也不代表引擎实际权重一定以 FP32 计算。

仅保存静态建议 `artifacts/gate05/20261020_dfl_conv_output_native/next_control_plan.json`，SHA256 `124615758af8e5a45a074effbbc335a3fbba4de2e37352252a2d58fcba8d5a61`。本轮未生成该候选图、未执行新 CPU 或 native；没有引入新输出 Q/DQ、尺度或标定。

本轮完成检查点和清单保存在 `artifacts/gate05/20261020_dfl_conv_output_native/`，检查点为 `checkpoint.json`。完成仅指这轮有界诊断与证据收尾，Gate 0.5 仍为 `INCOMPLETE`。

## 结论边界

删除 Constant／Reshape 的同时，也改变外部输出形状及 native 格式选择、融合边界。即使结果改变，也不能唯一归因于被删的某个节点，或宣称完整模型已修复。实际 Conv 输入、权重、输出、tactic 分别记录；没有输出 INT8 的先验要求。

CPU 概率仅为冻结局部输入，不代表已验证 native 上游分布。没有完整图重试、parser reset、旧图重跑、数据集评估、目标级因果恢复、精度或性能通过结论。
