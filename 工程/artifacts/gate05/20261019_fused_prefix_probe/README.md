# 首 Conv / SiLU / 第二层 activation QDQ 融合前缀探针

从冻结 exact q_only_G1 图抽取 11 个节点，截止原第二 Conv 输入 ActivationDQ；
不含第二 Conv。原图 images FLOAT[1,3,640,640] 为唯一输入，原 tensor
`rampq_model_1_conv_a_dq` FLOAT[1,16,320,320] 为唯一输出，没有额外前后 tap。
original 逐个原样复制所选源节点与 initializer 的序列化字节。

code_clip 只把两处 Float Clip -> Q 改为 Q -> INT8 Clip[-127,127]，DQ 不动。
保留 Clip/Q 名字，Q 输出换为唯一新 tensor，Clip 输出接回原 Q tensor；
四个 Clip bound initializer 保留名字并改为 INT8 -127/+127。所有显式改线、
常量原值/新值和原/新序列化 SHA 记录在 spec.json。DQ、scale、zero、权重、
bias、Conv 和原 Sigmoid/Mul SiLU 节点逐字节不变。已核对两处原边界等于
float32(scale * -127/+127)，原第二 scale 和范围也在 spec 中单列。

固定 000020 输入直接按原 NPY 字节复制。两图各用一次 CPUExecutionProvider、
ORT_DISABLE_ALL，保留原始输出后核对原字节完全相同；输出期望 NPZ 的按名字
映射在 spec 中。CPU 结果不证明 TensorRT 融合行为。

后续 native runner 必须在 ONNX 解析后把 code_clip 两个原命名 Clip activation
的 alpha/beta 显式核对并修正为 -127/+127，同时保留解析前后数值证据。
本准备器只在本地 CPU 执行，不连接 Orin、不运行 CUDA、不跑完整检测器/数据集，
不改变原完整候选、标定、权重和阈值。原完整 PyTorch/ORT 护栏 FAILED_UNCHANGED。
