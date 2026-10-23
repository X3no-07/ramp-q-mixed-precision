# Gate1 浮点资格检查准备

状态：**脚本和合成检查已准备；真实模型资格检查尚未执行。** 本次没有加载训练中途权重，没有读取图像或原始标注正文，没有运行 GPU 推理。

[执行协议](qualification_protocol.json)固定使用新划分中的109张 debug。只有30轮完整训练最终回执、冻结 training_contract_v3、best/last 权重来源轮次及SHA全部通过验证后，脚本才复制并加载最终 best.pt。debug 不参与重新选择权重，Selection 与 testdev 不可用作本次推理输入。

两份模型均从同一保存权重独立加载，明确保留未融合结构；实际断言88个 Conv2d、保留 BatchNorm 和检测头输出格式。参考路是保存的 FP16 EMA 转为 FP32 计算，不能恢复原始 FP32 训练权重。两路共享固定640×640预处理，FP16输入仅由规范化FP32输入转换而来。

全部8400个位置、全部10类分数参加事前固定检查：框 `rtol=0.005, atol=0.5`，分数 `rtol=0.02, atol=0.002`。超差保留失败，不放宽阈值；有限性、shape、类别argmax、坐标有效性另行记录。旧 Gate0.5 护栏保持 **FAILED_UNCHANGED**。

后处理统一对两路raw的CPU FP32副本执行：候选置信度严格大于0.001，NMS IoU0.7，max_det1000；记录上限饱和，NMS超时警告立即失败。框先恢复未裁剪原图坐标，再由冻结的忽略区评价模块处理。预测类别0..9明确映射至发布者1..10；GT保留原始物理行身份。主置信度0.25、敏感性0.2/0.3、目标IoU0.5，均为项目全尺寸目标匹配，**不是官方VisDrone AP或小目标专属指标**。

真实执行会保存原始输出、逐图逐目标匹配、归一化输入hash、实际源码/环境、固定阈值护栏结果。输出有限、target hit或dtype护栏通过均不会自动把 `accepted_float_baseline_ready` 改为true；科学基线资格仍需单独审查，也不代表Orin/TensorRT验证或性能结论。

[准备回执](preparation_receipt.json)及[交付清单](preparation_manifest.json)绑定脚本、测试、协议与源码快照。最终20项合成检查记录于[测试输出](synthetic_tests_v5_console.txt)。首轮有一项因Windows沙箱临时目录清理权限而报错，已改成不创建临时目录的路径拒绝测试；旧失败输出保留。

真实执行入口（由父任务核验最终训练状态后决定执行，不能提前运行）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
& '..\05_水下小目标检测_工程\检测工程\.venv\Scripts\python.exe' scripts/qualify_gate1_float_baseline.py `
  --final-receipt artifacts/gate1/20261022_split_refinement/float_training_v2/attempt_receipt.json `
  --contract-dir artifacts/gate1/20261022_split_refinement/training_contract_v3 `
  --output artifacts/gate1/20261022_float_qualification_v1
```

输出目录必须不存在。失败不自动重试，也不改训练、权重、原数据或既有冻结文件。
