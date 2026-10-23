# 两 Conv、三处激活边界的精确上下文探针

从先前失败的修正 q_all 抽取19节点、20 initializer的依赖闭包。
输入 images FLOAT[1,3,640,640]，唯一输出为原第三 Conv 输入
rampq_model_2_cv1_conv_a_dq FLOAT[1,32,160,160]；包含前两Conv与原SiLU，
不含第三Conv，不添加中间输出tap。全部所选节点及常量保留各自源图序列化字节。

matching原始 q_all 精确slice仅用于新的CPU语义对照；两slice各以固定000020
输入运行一次CPUExecutionProvider/ORT_DISABLE_ALL，必须输出字节完全一致。
native仅构建code_clip拓扑，比较none/obey两个case；每case300秒一次尝试，
batch900秒，成功后固定输入重复执行两次。只修正三处原生命名Clip alpha/beta
到-127/+127，不设置或清除解析器layer precision/output type约束。

本探针增加第二Conv与第三激活边界，是日志提供依据的局部上下文检查；
不能假定失败随前缀长度单调，不能将新增节点认定为原因。旧完整护栏保持
FAILED_UNCHANGED，先前完整q_all构建失败保留；不重新构建完整模型，
不进行数据集评估，不构成Gate 0.5通过或精度/性能证明。
