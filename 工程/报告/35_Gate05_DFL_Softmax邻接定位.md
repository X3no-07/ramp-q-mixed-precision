> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate 0.5：DFL Softmax 邻接定位

2026-10-20，接续[原始 DFL 拓扑与约束对照](34_Gate05_DFL原始拓扑与约束对照.md)。此前原／改 9 节点 DFL 均在 OBEY 下复现 `filterQDQFormats::4421`，NONE 均能构建，但只有声明的修正组合显著减小固定输入的数值偏差。本轮保留修正量化器，在原 Softmax 输出处切开，比较新的 8 节点切片 NONE／OBEY。

**结果：裁去 Softmax 后，NONE 仍成功，OBEY 仍复现同一 4421 断言。显式 Softmax 算子不是这个局部失败出现的必要条件。新 NONE 的最终 DFL 输出与旧修正 9 节点 NONE 原始字节完全相同；这不证明两者内部 Softmax／概率逐项相同，也没有修复完整模型。Gate 0.5 仍未通过。**

## 冻结范围与 CPU 组合对照

冻结协议：`artifacts/gate05/20261020_dfl_softmax_cut_protocol/protocol.json`，SHA256 `bbe50e6485a72df194a13684895b4cc29a1db6ceca8ef098d05a6b5e56c56055`。源图为实际失败 corrected q_all，SHA256 `755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee`。

从已保存的 logits（NPY SHA256 `cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab`）经原精确 Softmax 节点，CPU 仅执行一次生成新的概率 fixture。该节点 axis=1，序列化 SHA256 `b3b4c7e2d24556983e61ae3a731e6fc82625f83a8b8c671b0e98596c7f258adc`。随后 8 节点切片仅 CPU 一次，输出必须与既有修正 DFL CPU 原始数组逐字节一致，才可上板。本轮不重新执行 72 Conv 上游或旧 native。

新输入 `/model.23/dfl/Softmax_output_0` 为 FLOAT `[1,16,4,8400]` 的概率，不能使用 logits 或图片替代。新切片含 8 个节点、7 个初始化器，一个 `/model.23/dfl/conv/Conv`，输出仍为 FLOAT `[1,4,8400]` 的 DFL 距离，不是检测框。

两 native case 各一次、300 秒，外层 900 秒。保留所有 parser precision／output type 请求，只校正原命名 code_clip 的 alpha/beta 至 ±127；成功引擎执行两次相同概率输入，失败不推理。实际输入、权重、输出及 tactic 分别审计，不预设输出 INT8。

准备阶段曾在执行前中断；继续时确认 fresh 包目录、CPU 失败记录和板端本轮目录均不存在，复用冻结协议与已写新脚本，才首次执行约定的 2 次 CPU。没有重做已有推理或重新选择输入。临时 SSH 访问续期后，板子仍为同一启动编号，无遗留任务。

CPU 组合对照已通过：原 Softmax 一次生成概率，8 节点图一次输出，与旧 DFL CPU NPY `fd4bcdcee24cd53651fdd7a508423c802f3b432c759647204c7b748f22abd50d` 逐字节相同。概率 NPY SHA256 `5a7c5b1c58bd4d956a042dfeab3b11133a072ce7ab8630195f51d8f38a508520`，原始数组 SHA256 `d4c4b21e1d1a2e025d5b3eabaade7e99f3093c0cf7511795f9477a4ffd069ae7`，范围 `2.4000963110581318e-17` 至 `0.9994057416915894`。axis=1 概率和偏离 1 的最大值约 `2.0233e-7`，仅描述，没有新设输出通过容差。

新切片 SHA256 `1f8662a55675152d57d248b16c2eebec08c4d54b5b6ebc4841f0c177d6ac2298`；单节点 Softmax 生成图 SHA256 `09bd1b2903cfa5208a54a15fb85c8f1c7074b4f655d9880b49b30bc7c70a1c06`。节点和初始化器保持源序列化身份，CPU 使用 `CPUExecutionProvider`、`ORT_DISABLE_ALL`、4 线程。

密封包 `artifacts/gate05/20261020_dfl_softmax_cut_probe.zip` 为 13,574,095 bytes，SHA256 `aeff82c52ff3b2ec1d83b4b06c21f014b30699a1e428a0c5fb37bcd3f6cb25dd`；spec SHA256 `45f284588d37e607478ce79e443d46a49b870ef0d65614c5ce0867130f50316b`，manifest SHA256 `6c456b5a08d34a9a3fb87d7fced5474f2c349d632076bfc0fb5eb8dd793ddde1`。

启动前独立审查 392 项通过，无 blocker，`pre_native_review.json` SHA256 `7f31020bf9ba1ce2b66a61cf96dfb59ba6b4094b75a496b156d238e9dcf4cd86`。新目录为 `/home/nvidia/rampq/gate05_dfl_softmax_cut_20261020/gate05_dfl_softmax_cut_probe_20261020`，monitor PID 37496、私有组 37501、guard PID 37507。

切开 Softmax 也改变外部输入边界、原生融合上下文并使用 CPU 概率，结果变化不能单独证明 Softmax 是完整图唯一错误节点。原完整护栏 `FAILED_UNCHANGED`、完整 corrected q_all 无引擎、Gate 0.5 `INCOMPLETE` 保持；没有完整图重试、parser reset、恢复矩阵、数据集或性能通过结论。

## Native 结果与实际精度

两 case 各构建一次，没有改变设置重试。

| case | Builder API 秒 | 构建与执行 |
|---|---:|---|
| code_clip_none | 52.228163 | 引擎成功，29,765 bytes；相同概率输入执行两次，输出有限且字节一致 |
| code_clip_obey | 1.529954 | `filterQDQFormats::4421`／`!n->candidateRequirements.empty()`，随后 `buildSerializedNetwork::751`；无引擎、无 raw、未推理 |

runner 共 64.168915 秒，monitor 共 69.31 秒，终态 `BUILD_SCRIPT_OK` 只表示按协议保存普通构建失败并完成批次，**不是两组引擎都通过**。guard COMPLETE、没有干预、私有组为空，无超时或资源监护停止。上述用时不是推理性能。

NONE 引擎只有一个实际 Conv，唯一对应源 `/model.23/dfl/conv/Conv`，权重数 16、无 bias、激活 NONE。输入为 `Four wide channel vectorized row major Int8 format`，权重 Int8，输出 `Row major linear FP32`；tactic 为 `ampere_fp32_icudnn_int8x4_128x32_relu_interior_nn_v1`。外部概率输入／DFL 距离输出均为 FLOAT/LINEAR。

本轮解析网络均为 12 层；旧 9 节点图为 14 层。四个具名 Q/DQ 的显式请求仍相同：2 个 DQ precision INT8、2 个 Q output type INT8，查询无错误；code_clip 导入修正前后请求保持。Clip alpha/beta 仅按冻结协议校正为 ±127，没有层 precision／output type setter 或 reset。实际配置中仅 OBEY 开关不同，其他 INT8＋FP16、无 TF32／PREFER、1024 MiB、avgTiming=1、DETAILED 等设置一致。

NONE 引擎 SHA256 `43d429867fdbd013b91dcac08e598610f1dad3c98b8b565c2dd7ba0d3ca2d514`，原始输出数组 SHA256 `f503e64fd4f24ffde83dd60e62be2dc7e832b54d311c5f564eabebb07a831d84`。相对冻结 CPU 参考有 11,779/33,600 个元素不同，最大绝对差 `1.9073486328125e-06`，平均绝对差 `7.795968226024083e-08`，RMS `1.664742957045647e-07`。这些仅为描述指标，没有新设通过容差。

## 与含 Softmax 的上下文比较

| 修正 DFL 上下文 | NONE | OBEY |
|---|---|---|
| 旧 9 节点，图内 Softmax，输入 logits | 成功，实际 INT8 输入／权重、FP32 输出 | 同类 4421 断言，无引擎 |
| 新 8 节点，外部 CPU 概率 | 成功，实际格式与 tactic 同上 | 同类 4421 断言，无引擎 |

两个 NONE 的最终 DFL 输出逐字节相同，0 个元素不同，最大差为 0。注意其外部输入并不相同：旧图接收 logits，Softmax 在 TensorRT 图内；新图接收 CPU 概率。最终距离相同不能倒推出所有内部概率或量化码相同，更不能推广到其他图片。两个 OBEY 无输出，所有涉及它们的 raw 对比为 `UNAVAILABLE_ONE_OR_BOTH_CASES_FAILED`。

原权重、尺度、CPU 输出参考及具名 Q/DQ 请求保持，native 通用配置除 OBEY 外一致。对照结论限于：这里没有显式 Softmax 算子时仍能出现同类错误，不能把此前局部失败完全归因于必须存在 Softmax。也不能证明其他上下文中 Softmax 没有影响，或指认完整 q_all 的具体失败节点。

跨阶段保存证据比较为 `context_comparison.json`，SHA256 `e5cea499a0874a10644ba4909ea44c33b0a1e96183bbc3ca8b4ca9d35c9cf380`。根审计 `root_native_audit.json` SHA256 `0878cacc91734567c3f701fcda3800fa64505863ab1911bc9519afc208538f05`；均为保存数组／日志审计，没有重新运行旧引擎。

## 原始证据与板端终态

归档 `artifacts/gate05/20261020_dfl_softmax_cut_native/native_evidence.zip` 为 13,823,424 bytes，SHA256 `1b531ae8079a7563432e9702071ca8d384695f2a2624d80ab3da7798fa7215cc`。回执列出 89 份文件，另加回执元数据自身；文件清单、大小、SHA256、ZIP CRC 均核验，回执 SHA256 `5538c67b45e6ba020bc095b4ad4eca8920e0cb8fdaf60b67d90a90e4b37d44b8`。限速下载一次成功，最终 SSH 重查板端全部文件与归档仍匹配。

最终客户端为 2026-10-20 22:00，板端为 2026-10-18 04:10，分别记录、未校时。启动编号仍是 `692c7544-b8bd-415b-939c-b11cdbd9f7f0`。monitor／runner／guard／两个 builder 的五个 PID（37496、37501、37507、37514、37550）均不存在，无构建任务。风扇、jtop active，NRestarts=0，可用内存 4992 MiB；15W/mode 2 查询仍有非管理员 EMC 读取权限提示，不是完整特权核验通过。

旧修正 DFL NONE 引擎 `a0b23d8fce8a25bbf6ca5de9bf638dabbc9be11db23e56ad16cb2efc3ce2a33e` 保持，旧 OBEY 仍无引擎。完整 q_only_G1 引擎 `486845ca9bcc71e9ad6b721329855a1444184e721d64bcb7eb7daae593bd1476` 保持，完整 corrected q_all 仍无引擎。已有原结构数值失败和全部原完整护栏失败均保留。

独立保存证据审查 597 项全部通过，报告 `artifacts/gate05/20261020_dfl_softmax_cut_native/independent_native_review.json` SHA256 `222617e5f6ec15c0dabb6f2e1ce697cd989359dfcf2d5914469c78f11a7326ad`。复核覆盖 89 份原始回执文件、图来源、parser 请求、实际精度、输出字节、新旧上下文及五个任务的终态；没有新增 CPU／native 推理或 SSH。入口和本报告另经只读一致性复核。

## 下一局部控制

后续优先从当前修正图再裁去末端 Constant＋Reshape，保留至 Conv 输出的 QDQ 小图，并复用概率输入，另冻结 NONE／OBEY 对照。该方案尚未执行，应先做精确来源、形状和 CPU 输出字节对应核验；删去末端也改变外部输出契约，不能把新结果直接视为某一节点的唯一根因。

静态建议为 6 节点、7 个 initializer，外部输出改为源 Conv 的 FLOAT `[1,1,4,8400]`。未来一次新 cut CPU 校验应将其连续数组原始字节与旧 `[1,4,8400]` 参考对应；形状头不同，不能用整个 NPY 文件字节比较代替数组比较。静态计划 `artifacts/gate05/20261020_dfl_softmax_cut_native/next_control_plan.json` SHA256 `14d7710f373fd56a8d68ec15219e234c96a9325938e25ad703696b879160dfc8`，本轮未执行此建议。

Gate 0.5 仍为 `INCOMPLETE`。本轮没有完整图重试、parser 约束 reset、恢复矩阵、独立数据集、实机目标级因果或性能通过结论。阶段证据、清单和完成检查点保存在 `artifacts/gate05/20261020_dfl_softmax_cut_native/`，检查点为 `checkpoint.json`；其完成仅指本轮有界诊断收尾。
