# Gate1：合成训练管线与FP32保存接入验证

已用新生成的4张训练图、2张training_val图和随机初始化YOLO11n（10类、2,591,790参数），通过实际 `DetectionTrainer.train()` 完成一轮CPU训练。最终采用attempt_03：2次SGD、2次EMA更新；忽略区域已进入真实TAL分类损失，保存前的live FP32 EMA可以按架构重新构建，CPU合成输入输出逐字节一致。该结果只验收这条合成工程链，不是科研基线或板端验收。

## 实际运行与检查

| 项目 | 结果 |
|---|---|
| 运行范围 | CPU、单线程、workers=0、batch=nbs=2、AMP关闭，1 epoch |
| 新数据 | 4张train、2张training_val；显式成员清单、三文件字节/SHA核验；图片ID和图片SHA跨split无交集 |
| 实际更新 | 2次SGD及EMA；两步首个可训练参数各432个元素变化 |
| 整轮参数变化 | 运行时记录238个参数张量、2,209,385个元素变化 |
| 两批正anchor | 9、10，正样本分类整行保留 |
| 两批忽略背景anchor | 67、98，分类BCE屏蔽实际触发 |
| FP32捕获 | 第1轮、EMA updates=2、原生half序列化前；源已是FP32，未从half升回 |
| 架构重建 | 受限state-only加载、strict state_dict、全YOLO11n新建；CPU解析合成输入输出exact |
| 三组件测试 | Dataset 45、criterion/Trainer 51、FP32 hook 23，合计119项通过 |
| AP | `SMOKE_NO_TASK_AP`，未计算任务AP |

[最终训练回执](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261023_training_integration_delivery_v1/attempt_03/training_receipt.json)；[真实运行事件](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261023_training_integration_delivery_v1/attempt_03/events.jsonl)；[预先冻结合同](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261023_training_integration_delivery_v1/training_contract_v3.json)；[FP32捕获回执](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261023_training_integration_delivery_v1/attempt_03/fp32_capture/epoch_0001/capture_receipt.json)。

## 接入内容及修复

Dataset每次取样重新核验manifest、图片、正标签、ignore sidecar的实际字节与SHA；标签和忽略框共用实际loader双轴比例、整数补边及同次flip参数，再输出标准检测字段和ragged ignore区域。只支持显式受限链，不执行复杂增强。score=0、category0、category11来源分开保留；非法、空面积或完全在画布外区域拒绝整样本，不静默删除，也不把质量排除的正框转成ignore。

真实criterion保持当前已固定源码的TAL、box、DFL、分母、权重和batch倍数，只修改未分配且中心落入ignore区域的背景BCE。空ignore的四组有/无GT和classweights测试中，loss、模型全部参数梯度及输入梯度与原框架逐元素相同。真实TAL空GT分支返回浮点零fg_mask，接缝处仅为严格helper转bool，原assignment tuple及box/DFL路径不变。正anchor整行的局部logit梯度保留，不代表共享模型参数的总梯度不变；合成BCE下降也不是精度收益。

FP32 hook在实际训练器保存入口先取live EMA，再调用普通框架保存；普通框架仍会deepcopy后half序列化。EMA更新数和状态在捕获前后不变，架构新建隔离构造随机数，测试也覆盖live状态/cache/RNG不被改变。只保存推理状态，不保存优化器或完整恢复训练状态；主线程与非重入锁不能证明不存在任意外部写者，当前调用协议明确禁止并发EMA写入。

首个attempt_01的功能检查已通过，但源审查发现仅关闭sync/loggers不足以阻止框架注册平台回调。v2/attempt_02在创建Trainer前于本进程禁用集成注册，核对初始回调全部来自框架base后才加入本地记录回调。未修改环境中的框架文件；这是进程路径约束，未声称OS网络沙箱，也没有证据反推attempt_01发生了外联或上传。

独立forward审计随后发现，Trainer构造时的select_device(cpu)把请求的1线程改成8；attempt_01/02回执硬写threads=1不准确。独立进程用1线程重建不exact，用8线程严格exact。未修改或放宽exact检查，保留原错误回执及诊断，最终改用v3/attempt_03：构造后重新固定1线程，训练开始、每次optimizer step、保存前和最终回执都实际断言线程数，开始/保存前/结束后记录backendflags。逐step仅断言线程，不声称其全部backendflags逐步留档。Trainer返回时会unset_deterministic，因此独立重建复现保存入口的实际flags，不用训练返回后flags替代。三次运行共用同一合成fixture，不是三组独立数据证据。

## 独立审计与边界

源审分别检查Dataset/criterion/Trainer和FP32 hook，另对修正版runner复审。结果审计复核冻结输入、输出清单、实际事件、两步更新、mask记录和保存时序；独立加载仅本轮新FP32 state，按保存的架构重新构建CPU模型并核对解析合成输入输出。普通best/last只做字节哈希，未反序列化。没有读取原30轮权重、真实图片或标注、Selection检测输入、Holdout内容。

初末模型仅保存逐张量hash描述和运行时变化计数，独立审计能复核张量hash变化及计数上限/合计，不能从未保存的初末原始张量重算2,209,385逐元素变化数；TAL计数也属于源码与事件/回执交叉核对，没有独立重跑训练assignment。

当前Trainer只允许synthetic-only、单轮CPU。跨manifest分离由本轮显式preflight证明，任意自定义dataset_factory仍是受信任扩展，不能宣称通用防泄漏；当前使用默认已审Dataset。现有4534张真实训练图是否兼容此非法ignore处置尚未检查，也没有生成新的真实数据quarantine或正式训练合同。CUDA训练及CUDA同步捕获另须资格检查。

[组件源审](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261023_training_integration_review_v1/source_review.json)；[FP32源审](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261023_training_integration_review_v1/fp32_source_review.json)；[实际结果审查](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261023_training_integration_review_v1/result_review_v3.json)；[独立state-only重建审计](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261023_training_integration_review_v1/synthetic_output_audit_v3/audit_result.json)。

## 当前状态与下一步

原30轮训练候选、109图混合原型及所有旧封存结果保持原样。原全FP16护栏仍为 `FAIL_FIXED_TOLERANCES`，旧Gate0.5仍为 `FAILED_UNCHANGED`；科研Gate1、独立真实数据AP、导出/TensorRT/Orin及速度功耗均未据此通过。

下一步须冻结真实数据成员与非法ignore/quarantine政策，资格验证目标CUDA训练及live FP32捕获，再按独立验证、提交顺序和候选预算合同开展正式训练与验收。旧half EMA丢失的精度不能恢复。

[本轮决策](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261023_training_integration_delivery_v1/decision.json)；[状态快照v11](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/current_execution_state_v11.json)。
