> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate1 首轮训练与浮点资格检查结果

本轮 **30/30 轮候选训练已完成**，训练记录及权重审计通过；109 张固定 debug 图的 FP32/FP16 推理与目标匹配已完成。固定全候选框数值检查为 **FAIL_FIXED_TOLERANCES**，所以 **浮点基线尚未验收，Gate1 科学验收未通过**。独立复算通过仅说明保存结果一致。

这份结果接续报告43的训练中快照。[当前完成快照](../artifacts/gate1/current_execution_state_v5.json)；[分项验收结论](../artifacts/gate1/20261022_training_qualification_delivery/decision.json)。原 Gate0.5 结论见[报告39](39_Gate05_150图完整验收结果.md)，原完整模型数值检查仍为 FAILED_UNCHANGED，本次检查不替代它。

## 已完成的训练

使用冻结的训练合同 v3，4534 张训练图、414 张 training_val，30 轮、640 输入、batch=4、AMP关闭；用时 141.04 分钟，未记录恢复或重启。best 来自第29轮，last 来自第30轮；框架最后对 best 的额外验证回调标为31，不能算作第31轮训练。

| 文件 | 来源轮次 | SHA256 |
|---|---:|---|
| best.pt | 29 | `cddd14b3ebba294e4c3200bfb50fc1db1a9459c425d8e51d583c0881d71cfcfa` |
| last.pt | 30 | `120e04905d024b78272bde21f55ed1801ed48a5acd7f744892317e1c28c7e49e` |

最终 best 的框架内验证代理 mAP50=0.23975、mAP50–95=0.13452，仅用于训练监控及 checkpoint 选择。普通 YOLO 损失没有完整屏蔽 ignore 区域的负样本，这些数字**不是官方 VisDrone AP**。训练标签227425条，loader去重后227422条，3条重复已审计；框架将默认 max_det 从300按 training_val 实际最大对象数升为902。两项行为都有冻结源码和日志依据。

[完成训练审计](../artifacts/gate1/20261022_posttraining_checks_v1/01_training_audit/audit.json)核查30轮连续记录、保存检查、最佳轮次、4948张训练/验证副本及标签身份、499个状态张量有限值。保存的是 FP16 EMA 权重；转换为 FP32 **不能恢复保存前的完整 FP32 EMA**。

## 109张 debug 的数值检查

两分支从同一 best 独立加载，均不融合卷积与BN，使用统一640方形预处理及共享归一化输入，比较 NMS 之前的 `[1,14,8400]` 原始输出。FP32 参考来自上述已保存 FP16 EMA 的 `.float()`；本次不涉及 TensorRT 或板端性能。

| 检查 | 结果 |
|---|---|
| 输出形状、有限数、正宽高、得分范围 | 109/109通过 |
| 框容差 `abs差 ≤ 0.5 + 0.005×abs(FP32)` | **109/109图失败** |
| 框元素超差 | **10849 / 3662400（0.2962%）** |
| 得分容差 `abs差 ≤ 0.002 + 0.02×abs(FP32)` | **0 / 9156000元素超差** |
| 最大框绝对差 | **10.439514像素**，在640输入坐标系中 |
| 独立保存结果复算 | PASS_SAVED_OUTPUT_CONSISTENCY_ONLY，保留数值FAIL |

[固定数值检查原始结果](../artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification/raw_dtype_guard.json)；[独立复算](../artifacts/gate1/20261022_posttraining_checks_v1/03_qualification_output_audit/recomputed_summary.json)。复算重读454个封存成员并重算数值检查、目标匹配与分区计数；复用冻结的项目匹配器，不是独立官方评测 oracle，也没有再次执行模型或NMS。

## 项目目标匹配结果

固定主阈值 conf=0.25、IoU=0.5，共9059个有效GT，统计覆盖所有有效尺寸，**不是仅小目标指标**。109张图此前用于工程调试，不是独立 Selection/Holdout。

| 分支 | 命中 | 漏检 | 项目假阳性 | 项目目标召回率 |
|---|---:|---:|---:|---:|
| FP32 | 3146 | 5913 | 1631 | 34.7279% |
| FP16 | 3143 | 5916 | 1636 | 34.6948% |

共同命中3136个，FP32独有10个，FP16独有7个，共同漏检5906个。两分支命中接近，**不抵消原始输出数值检查失败**。[全部固定阈值及匹配汇总](../artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification/matching_summary.json)。

## 误差诊断与后续入口

只分析已保存 raw 数据，没有再次加载模型或图像：10849个框超差中，宽5264、高5502，中心x/y分别66/17；10841个来自两分支最高类别分均≤0.001的候选位置。0.25阈值下两分支各保留的4931个NMS输出位置均无框元素超差；这是**事后描述，不能据此过滤候选位置或改变通过条件**。低阈值NMS输出中仍有FP32侧4个、FP16侧5个位置超差。

独立反事实只把保存的FP32 decoded输出最终转换为FP16，109图均落在原固定容差内，框最大差0.25像素。这说明最终FP16存储舍入**单独不足以解释**实际超差；尚不能区分输入转换、网络上游累积、DFL积分或框解码各自贡献。源码未发现两分支像素不一致、融合不一致或NMS污染raw的明显问题，仍缺少中间logits证明具体原因。

[误差分布](../artifacts/gate1/20261022_dtype_guard_diagnostics/raw_failure_description.json)；[独立反事实诊断](../artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review/raw_array_diagnostics.json)；[训练及结果复核](../artifacts/gate1/20261022_posttraining_final_review/debug_qualification_review.md)。

下一项工作是单独冻结少量固定debug图的精度探针：保存检测头回归logits，分别切换DFL/框解码计算精度，先重现当前原始输出，再判断误差位于解码前还是解码内。历史109图结果和容差保持不变；该诊断不选择量化模块、不触碰Selection。官方评测语义及参考权重精度限制仍需处理，之后才能验收科学基线并冻结量化研究预算。

[最小对照提案](../artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review/README.md)尚未执行；[维度范围补充](../artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review/SCOPE_ADDENDUM.md)要求下次先记录实际 `head.reg_max`，不能把源码默认16-bin/64通道写成此次独立实测。

548张Selection未做检测评分或模块排名；Holdout ZIP未列成员、解包、读图或推理。未开展新量化矩阵。此次交付支持“训练候选完成、工程输出可复算、严格数值资格失败”这三个分开的结论。
