> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate1 AP聚合原生对照与忽略几何验证

**完整Python AP/AR聚合已实现，并通过固定32组合成用例的原生对照：220/220个中间矩阵零容差一致，161项最终指标满足预先冻结的双精度容差，28处NaN图样完全一致。** 同时完成忽略区域的受限几何和拼批组件，85项CPU合成测试通过。真实训练管线和live FP32参考保存尚未接入，Gate1科学验收仍未通过。

[分项结论](../artifacts/gate1/20261023_full_ap_delivery_v1/decision.json)；[状态快照v10](../artifacts/gate1/current_execution_state_v10.json)；[前一轮报告48](48_Gate1_训练前置组件与官方原生评分验证.md)。

## 完整聚合与原生参考

新模块逐段对应固定官方commit的calcAccuracy、VOCap、evalRes、compOas，原官方源码没有改动。按原行为先截断全类别提交顺序、再筛类别及按得分排序；保留score=0忽略GT进入召回分母、忽略检测既非TP也非FP、类别按出现图频重复加权、AP只取maxDets=500及VOC精度包络积分。没有替换成另一套COCO口径，也没有执行NMS。

实现支持声明的有限数值矩阵域，图内及跨图同分使用稳定顺序。原生0.5:0.05:0.95十个IoU阈值已做纯常量探测，与Python公式逐float64十六进制一致；这只固定所用Octave版本，不能推广到所有MATLAB版本。源码与54项合成测试先冻结，再由独立比较器读取原生答案。比较器45项负例测试通过。

旧20组沿用上一轮封存的原生输出；本轮另预先固定12组压力输入，使用未修改的官方函数重新原生执行，12/12成功、0用例错误。压力输入包括超过500个预测、输入顺序反转、跨类别截断、同分与等IoU、score=0反复吸收、类别图频不均、空图、category11、重叠忽略区域、越界坐标和分数召回端点。没有按运行结果挑选用例，也没有用手写AP答案代替原生结果。

[完整聚合源代码](../scripts/gate1_official_ap.py)；[冻结实现与54项测试](../artifacts/gate1/20261023_full_ap_preparation_v1/preparation_receipt.json)；[新增原生运行回执](../artifacts/gate1/20261023_full_ap_oracle_audit_v1/native_stress_execution_v1/launcher_receipt.json)。

## 实际比对结果

矩阵维度、行顺序和全部元素使用零绝对/相对容差。最终七项AP/AR是百分数单位的float64标量，预先固定 `abs(Python-native) <= 1e-10 + 1e-12×abs(native)`，只允许不同双精度归约路径的舍入差；NaN位置必须一致，Inf拒绝。这一标量容差与原模型FP32/FP16护栏是两项不同检查，原护栏未改。

| 参考集 | 精确矩阵数 | 最终指标数 | NaN指标数 | 最大有限差（个百分点） |
|---|---:|---:|---:|---:|
| 旧20组封存参考 | 166 | 77 | 14 | 0 |
| 新增12组压力用例 | 54 | 84 | 14 | 7.10543e-15 |

合计32组、40张合成图，其中23组会产生七项最终指标。133个有限指标全部通过，最大差7.105427357601002e-15个百分点；28个NaN保持原样，不能解释成0分。旧20组与上一轮局部检查重叠，不能当作新增独立数据证据。

独立比对核对了原官方11份源码、2890个运行时文件、MAT读取依赖的1535个实文件（wheel共有1656个条目，含目录）、启动与输出身份、MAT/JSON序列化、空矩阵和NaN。原生wrapper只导出指定中间矩阵和最终七指标；Python内部10×10 AP及10×10×4 AR表没有原生独立输出对照，不宣称其每个单元已被直接验证。

[独立比对摘要](../artifacts/gate1/20261023_full_ap_oracle_audit_v1/audit_summary.json)；[逐用例及逐指标结果](../artifacts/gate1/20261023_full_ap_oracle_audit_v1/comparison_attempt_v1.json)；[比较政策](../artifacts/gate1/20261023_full_ap_oracle_audit_v1/comparison_policy.json)。

## 忽略几何与批次组件

新组件只支持显式受限链：原图到loader实际输出的双轴缩放比例、同一次LetterBox采样的比例与整数补边、同一次水平/垂直flip决定；最后按图片与sidecar身份及顺序拼成ragged区域，接入此前分类损失原语的合成链测试。未另采随机参数，也不按正框面积/长宽比规则删除残留忽略区域。

85项CPU测试包括10次已安装LetterBox对照、4次RandomFlip对照、3次以造数组替代imread的真实loader完整链，以及两种dtype拼批到loss的边界/positive优先检查。没有加载真实图片或数据标注。仅将旧增强概率设为0仍可能走RandomPerspective等路径；因此受限链是新合同提案，并不是旧30epoch训练已被修复。

score=0、category0和category11来源分别保存；质量排除的正框不静默转成publisher-ignore。零面积、完全在画布外、非法来源及精度转型使正面积丧失等情况拒绝整样本，需另定quarantine和数据合同。没有检查现有4534张训练图是否全部兼容。身份字符串和SHA字段的结构检查，也不证明实际文件哈希或split归属，未来loader仍须核验。

真实Dataset、Format、Trainer、criterion、增强禁用的运行强制检查尚未接入。组件通过不等于完整ignore-aware训练完成，也不证明训练收益。

[几何组件说明](../artifacts/gate1/20261023_ignore_geometry_preparation_v1/README.md)；[85项测试回执](../artifacts/gate1/20261023_ignore_geometry_preparation_v1/final_tests_receipt.json)；[独立源审](../artifacts/gate1/20261023_full_ap_final_review_v1/source_review.json)。

## 当前验收边界

现在已具备完整聚合实现及声明合成用例的原生一致性证据，仍未证明任意输入、跨版本或真实数据AP一致。下一步应冻结并接入真实训练链、非法ignore处置、live FP32 EMA原子保存及架构重建，再按独立数据/提交顺序/候选预算合同进行资格检查。旧FP16 EMA丢失的精度不能恢复。

本轮没有重新训练或运行检测模型，没有Selection检测评分或Holdout内容访问，没有导出、INT8、TensorRT、Orin、速度或功耗测量。既有109图混合原型通过记录保留；原全FP16护栏仍为 `FAIL_FIXED_TOLERANCES`，旧Gate0.5记录仍为 `FAILED_UNCHANGED`。
