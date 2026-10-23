# 首个 activation Clip/Q/DQ 边界探针

只从冻结 q_only_G1 图抽取第一 Conv 输入前的3个原节点 Clip、ActivationQ、ActivationDQ。
配对 no_clip 图仅移除 Clip 及其未用上下界常量，并把原 ActivationQ 的 input[0]
重连为原 `images`；该改线的原/新序列化SHA在spec逐项记录。DQ、scale、zero和其余
initializer保留原字节。原完整候选没有修改。

相同输入 `preprocessed_input.npy` 为原000020固定图预处理数组的原字节复制，
FLOAT[1,3,640,640]。两图只有一个FLOAT输出，保持原DQ tensor名字，
不暴露INT8 Q作为外部输出。详细按名字的接口和CPU NPZ键在spec中。

此输入范围为[0,1]，原Clip上下界严格为[-1,+1]，已验证Clip前后整个数组原字节相同。
两CPU小图使用CPUExecutionProvider和ORT_DISABLE_ALL。先保存原始float32输出，
再逐元素原字节核对两图一致，以及与round-to-nearest-even后乘原scale的明确CPU期望一致。
原INT8 codes也保存在npy，仅作CPU期望，不是图的外部输出。

本实验不重跑完整检测器、不连接板子、不用CUDA，不校准、不改阈值、不做150图评估。
去Clip只是这一个固定输入上的配对数值定位，不能写成完整模型修复、迁移通过、
strict原生INT8算术、准确率或性能结论。原完整数值失败仍保留。
