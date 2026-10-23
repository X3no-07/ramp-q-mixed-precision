# Gate1：划分冻结与首轮浮点训练

日期：2026-10-22。本文记录 Gate1 的数据隔离、评价实现和首轮训练候选；不构成 Gate1 科学验收通过。**新尝试已实际进入第 1/30 轮训练，batch=4、AMP 关闭，尚未完成全部轮次。**运行状态与检查时间见[本轮状态快照](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/current_execution_state_v2.json)。

## 已完成的准备

在原准备目录之外生成新的划分计划，原图像、标注和原交付保持不变。旧训练验证集的 409 图中有 407 图来自同一保守分组，不能据此声称覆盖多种场景。新计划在不看模型结果的条件下按完整分组重新分配，训练内验证扩展为 58 个分组。

| 用途 | 图像 | 保守分组 | 说明 |
|---|---:|---:|---|
| 训练 | 4534 | 92 | 227425 个有效正标签，10 类均存在 |
| 训练内验证 | 414 | 58 | 28007 个有效正标签，10 类均存在，最大组 41 图 |
| 校准池 | 689 | 24 | 三份预先固定的 64 图名单，尚未执行校准 |
| 工程调试 | 109 | 6 | 保持独立，不进入本次训练/验证 |
| Selection | 548 | 75 | 官方 val，尚未做模型推理、选模块或评分 |
| 隔离 | 725 | 27 | 包括标注质量问题、与 Selection 关联及新增近重复风险 |

“分组”由文件名前缀、精确内容重复和审查后的风险边构成，不能证明真实相机或场景身份。三份校准名单按图像 ID 和内容 SHA 互斥，分别覆盖 17、17、19 组，每个种子中每组最多 8 图；三个种子仍共享校准池场景，不是三个独立数据集。

[划分合同](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/split_plan_v1/split_plan.json)、[标签清点](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/split_train_val_label_inventory.json)与[独立复核](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/split_plan_v1_independent_audit/verification.json)保留详细名单与身份。独立复核通过 7103 项断言，重新核验原准备目录 21077 个文件的字节数及 SHA；没有导入划分生成器来代替独立检查。

## 近重复筛查与保守隔离

只读取原 train/val 的 7019 张图像；对 dHash/pHash 任一 Hamming 距离不超过 4 的候选做筛查。独立穷举复核全部 24629671 对缓存哈希，确认 37 个候选及其中 5 个跨角色、前缀或分组候选与工具输出一致。这证明定义内的哈希检索实现正确，不能证明不存在其他近重复或共同场景。

视觉复核发现 `train/9999937_00000_d_0000015` 与原隔离集 `train/9999950_00000_d_1007223` 的视角和车辆位置几乎相同；两种哈希距离均为 0，RGB 缩略图 RMSE 约 0.00140。保守合并关联组，新增隔离 228 张原训练图。另两对 dHash 相近的候选视觉场景明显不同，没有据此合组；已知精确重复与同前缀隔离关系沿用原有处理。

证据见[方法与运行统计](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/near_duplicate_audit/run_summary.json)、[视觉审查决定](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/near_risk_review_v1.json)、[联系图](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/near_duplicate_audit/contact_sheet_01.jpg)和[穷举复核](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/near_duplicate_exhaustive_pair_check.json)。本次未打开或列举 test-dev ZIP 内的成员。

## 评价实现的可用范围

官方评价源码固定至 [VisDrone toolkit commit 005445782213e20cb91bc50a597db3dd949e749a](https://github.com/VisDrone/VisDrone2018-DET-toolkit/tree/005445782213e20cb91bc50a597db3dd949e749a)。已实现 category=0 的像素并集过滤、同类 score=0 的预测面积交叠吸收，以及项目确定性同目标匹配；保留原标注物理行 ID。项目主阈值为 confidence≥0.25、同类一对一 IoU≥0.50。

37 项合成测试在实际训练环境中通过。固定源码中的像素坐标、预测截断顺序、类别汇总和召回分母均有特殊行为，不能以普通 YOLO/COCO 指标替代。**本次没有实现完整官方 AP，也没有获得 MATLAB/Octave 运行对照验证。**目标级计数和后续误检统计应明确命名为项目指标。

[源码复核说明](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/evaluation_sources/OFFICIAL_SOURCE_REVIEW.md)、[机器可读评价政策](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/evaluation_sources/evaluation_policy.json)和[实际环境测试](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/evaluation_existing_venv_tests.xml)记录边界。普通 YOLO 正标签省略 ignored 项，但训练损失不会屏蔽这些区域的负样本；这一限制仍然存在。

## 首轮训练合同和已保存的失败

本次采用 YOLO11n 官方 COCO 初始权重，SHA 为 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。固定输入 640、batch=4、workers=0、AMP 关闭、30 epochs、SGD、lr0=0.01、momentum=0.937、weight decay=0.0005、seed=20261022、deterministic、patience=0，最后 10 轮关闭 mosaic。其他实际默认参数及框架源码一并固定，执行时保存全部有效参数。

checkpoint 只按训练内验证集的内置 mAP50–95 选择，同分保存较晚轮次。该指标不了解 ignored，仅作为训练监控和选权重代理，不能称为官方 VisDrone AP。30 轮是首轮候选预算，不代表已证明充分收敛。训练关闭 AMP，但框架 checkpoint 保存 FP16 EMA；后续建立 FP32/FP16 参照时必须披露这个序列化事实。

合同 v1 尚未用于训练即被检查后的 v2 取代。v2 对应的 `float_training_v1` 在第一个批次因新增有限性检查不兼容框架损失字典而失败，完成训练轮数为 0；全部日志、代码、合同和失败凭证保留。这是 runner 兼容失败，不能据此推断模型质量或硬件故障。修复添加了字典/列表/张量递归检查和回归案例，13 项测试及 3 个子案例通过。**生效合同为 v3，新尝试为 `float_training_v2`，从原始官方权重重新训练，没有从失败 checkpoint 恢复。**

每个批次检查观察到的损失；每次保存前检查 live/EMA 全部状态及 EMA 转 FP16 后是否有限，阻止框架静默重同步或夹断后接受结果。发生 batch 改动或异常数值即中止，保留失败，不在同一合同内自动调参重启。最终最佳权重验证的额外回调与 30 个训练轮次分开记录。

证据见[生效训练合同](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/training_contract_v3/contract.json)、[合同说明](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/training_contract_v3/TRAINING_CONTRACT.md)、[回归测试](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/float_contract_tests_v4.xml)、[失败审查](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/failed_training_v1_review.json)。[当前尝试记录](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/float_training_v2/attempt_receipt.json)和[训练日志](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/float_training_launch_v2/stdout.log)是持续变化的运行文件，不属于静态交付清单的封存成员；当前进度以带时间的状态快照为准。

## 后续顺序

先完成本次候选训练并固定 best/last 的身份，再检查预处理、类别、框还原、有效目标命中和 FP32/FP16 数值资格。官方 AP 缺口仍需取得对照运行证据或明确保留为未完成分项。

之后固定量化语义、候选/预算、后处理、统计和研究验收合同，才在 Selection 上扫描模块。最终确认集只在候选与分析锁定后解封。当前没有量化恢复结果、选模优越性结论或 Gate1 科学 PASS。

Gate0.5 的原数值护栏仍为 `FAILED_UNCHANGED`；本轮准备工作不改变这一判定。[旧交付完整性复核](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/prior_delivery_integrity_check.json)确认原 Gate0.5 的 45 文件与前一轮 Gate1 准备交付的 32 文件均保持一致。旧 40/41 号报告和 `current_preparation_state.json` 保留为当时快照，本轮采用新的 42 号报告及状态文件。
