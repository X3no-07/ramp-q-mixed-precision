# Gate1 首轮训练路径已跑通

检查时间：2026-10-22T04:33:05.923537+00:00。已完成第 1/30 轮训练与训练内验证，当前任务继续运行；全部 30 轮尚未完成。实际 batch=4，AMP 关闭，训练 4534 图、训练内验证 414 图。

第一轮 best/last 已保存，两个文件内容相同；保存前 live/EMA 各 499 个状态张量通过检查，其中 418 个 EMA 浮点张量转 FP16 后仍有限。另在 CPU 上重新读取封存副本，499 个状态张量全部有限，类别数为 10。未观察到自动恢复事件。

这证明训练、验证和保存路径可运行，不证明训练充分收敛、官方 VisDrone AP、浮点数值资格或 Gate1 科学验收。未做 Selection 模型推理，未读取封存测试集。首个尝试的脚本兼容失败记录和旧合同仍保留，新尝试从官方初始权重开始。

[本次核验 JSON](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/first_epoch_verified_snapshot_v2/verification.json)；[最新状态快照](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/current_execution_state_v3.json)；[数据划分与合同报告](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/42_Gate1_划分冻结与首轮浮点训练.md)；[持续变化的训练日志](C:/Users/lenovo/Desktop/视觉边缘研究/07_边缘小目标检测混合精度量化_工程/artifacts/gate1/20261022_split_refinement/float_training_launch_v2/stdout.log)。
