> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate1 训练前置组件与官方原生评分验证

**本轮取得了官方评分源码的原生参考输出：20组合成用例在GNU Octave 11.3.0上实际运行成功，0个用例错误，独立输出审计通过。** 同时完成忽略负样本损失与FP32保存格式两个前置组件。它们还没有接入实际训练，Gate1科学基线仍未验收。

[分项结论](../artifacts/gate1/20261023_acceptance_prerequisites_delivery_v1/decision.json)；[状态快照v9](../artifacts/gate1/current_execution_state_v9.json)；[前一轮109图混合原型报告](47_Gate1_混合精度独立推理原型验证.md)。

## 官方原生参考已经取得

本轮从官方发布资产获取Octave 11.3.0便携包并核对发布SHA-256；没有声称验证PGP签名信任链。空间检查后，只提取2890个命令行必需文件、640420084 bytes，没有全量安装、运行post-install或修改系统PATH/注册表。mean2使用同归档原image-2.20.0函数；原官方11份源码及旧合成runner保持字节身份。

输入始终是已冻结的20组合成矩阵：11组AP/AR聚合、4组像素忽略过滤、5组单图匹配。六个官方函数的解析路径逐一与原文件核对，程序实际调用原dropObjectsInIgr、evalRes、calcAccuracy及其依赖；没有替代函数或手写预期AP答案。MAT与JSON、运行版本、依赖清单及控制台记录均已保存。

独立审计核对2890个运行时成员、1535个隔离MAT读取依赖文件、20个用例顺序/状态、空矩阵维度、匹配矩阵和七项指标的序列化一致性。39项审计器合成测试通过。A03/A04空评价类别产生的原生NaN保留在MAT中，JSON的null不能解释成0分。这取得的是固定Octave版本执行官方源码的参考，不证明所有MATLAB版本或真实数据的等价。

[原生执行回执](../artifacts/gate1/20261023_native_oracle_runtime_v1/execution_01/launcher_receipt.json)；[原生MAT结果（见证据下载索引）](../发布清单/证据下载说明.md)；[独立输出审计](../artifacts/gate1/20261023_native_oracle_comparison_preparation_v1/native_output_audit_v1.json)。

## Python局部语义对照

先冻结全部用例、字段、行顺序及零绝对/相对容差，再比较现有Python的忽略过滤与official-style单图匹配：20组用例的22张合成图过滤/score反转矩阵，以及5组匹配×10类别的GT/DT矩阵，**全部166个中间矩阵严格一致**。比较适配器27项合成测试通过，既有Python37项回归也通过。

这次没有调用或修改项目另行定义的IoU优先目标匹配器。局部矩阵一致仅覆盖这些固定合成输入，不是所有输入的证明；Python完整AP/AR聚合仍未实现，也没有比较Python与原生AP数值。类别按图重复加权、输入顺序先截断、score=0分母、跨图排序及VOC精度包络等聚合细节，仍需完整实现与对照。

[预先冻结的比较条件](../artifacts/gate1/20261023_native_python_semantics_v1/predeclared_conditions.json)；[实际局部比较结果](../artifacts/gate1/20261023_native_python_semantics_v1/comparison_result_v1.json)。

## 两个训练前置组件

**忽略负样本分类损失原语：55项CPU合成测试通过。** 对最终增强图像内、anchor中心落在忽略框且未被分配为positive的位置，屏蔽整行分类BCE；positive位置保留全部类别项，原target_scores求和分母不变。无ignore时值和梯度逐位一致；被忽略negative的logit梯度归零，固定输入下其余logit梯度不变。共享模型参数汇集多个位置的梯度仍会改变，不能据此宣称参数更新不变或训练效果改善。

anchor中心、半开边界、class-agnostic等是明确的新工程政策，不是官方AP规则。组件要求ignore几何已与全部增强同步；Mosaic等增强、sidecar拼批、assigner/criterion/trainer尚未接入，不能称完整ignore-aware训练已解决。当前loss/tal源码支持缺口判断，但旧训练合同没有分别固定这两个文件，不能倒推其历史逐字身份；已固定的dataset/augment/tasks与当前快照相符。

**FP32参考状态保存格式：10项CPU合成测试通过。** 原本FP32、不能被FP16精确表示的人工数值保存后逐位一致；组件拒绝自动提升半精度来源。它仍不能识别调用者此前先half再float，也不证明声明的训练来源；旧best/last半精度EMA丢失的精度不能恢复。尚无live EMA原子捕获、架构重建或真实训练hook，当前不是已验收科学参考，也不是含优化器的恢复checkpoint。

[损失组件与来源审查](../artifacts/gate1/20261023_ignore_loss_preparation_v1/README.md)；[FP32格式验证](../artifacts/gate1/20261023_fp32_reference_preparation_v1/README.md)；[两组件独立复核](../artifacts/gate1/20261023_acceptance_prerequisites_review_v1/source_review.json)。

初期的临时目录ACL、MAT测试配置以及下载超时/不完整尝试均保留原日志；最终测试使用明确工程临时目录，完整归档在续传后通过官方SHA。未把失败尝试删除或算成通过。

## 还需完成的验收

下一步是接入并测试增强几何/拼批/训练损失，以及保存前live FP32 EMA的原子捕获和来源核对，再补完整AP/AR聚合比对与独立验证合同。本轮没有模型推理或重新训练，没有读取真实图像/数据标注、Selection检测结果或Holdout内容；没有导出、INT8、TensorRT、Orin、速度或功耗测量。

已有109图混合原型数值结果保持通过；原全FP16门槛仍为 `FAIL_FIXED_TOLERANCES`，旧Gate0.5数值记录仍为 `FAILED_UNCHANGED`。官方源码合成运行成功与两个组件测试通过，都不替代科学基线、真实数据集AP或板端部署验收。
