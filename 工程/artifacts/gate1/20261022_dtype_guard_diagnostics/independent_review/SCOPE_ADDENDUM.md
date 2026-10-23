# DFL 维度观察范围补充

README 中的 16-bin DFL / `[1,64,8400]` box logits 来自 YOLO11n 默认构造路径的源码预期，不是本次重新加载 checkpoint 后的独立实测；本次没有加载模型，既有 `model_provenance.json` 也没有单列 `head.reg_max`。

下一次最小诊断必须先记录实际 `head.reg_max`，并按 `[1,4*reg_max,8400]` 验证保存的 box logits。若实际值不是 16，应在新协议和输出中记录实际值，不可强行重排成 64 通道。此补充不影响已保存 decoded raw 的 `[1,14,8400]` 实测、全局最坏元素排序、最终 FP16 storage cast 反事实或原 guard 失败结论。
