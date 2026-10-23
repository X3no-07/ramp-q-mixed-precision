# 09｜目标级量化失效归因与恢复：下一阶段工程

版本：2026-10-19。配套研究文稿在 [08 方向目录](../08_边缘小目标检测混合精度量化_方向/README.md)。

本工程承接 07 已完成的六配置构建和单图执行，为数值迁移诊断提供明确的起点。**Gate 0.5 仍为 INCOMPLETE，原数值护栏失败不变。** 本次新增的是离线工具、可核验的证据摘取和已有代码快照，未启动新板端实验。

## 目录与用途

```text
09_边缘小目标检测混合精度量化_工程/
  run.ps1                       Windows 统一入口
  src/stage2.py                 核验、历史状态、保存数组对比、方案读取
  config/baseline.json          六配置文件入口、数组键、固定图/NMS 协议
  config/source_manifest.json   98 份继承文件的来源、大小、SHA256
  config/next_protocol.json     下一阶段方案草案；未决容差显式为 null
  evidence/07/                 历史报告、六配置主/辅助输出、CPU 对照等
  legacy/07/                   原 src/scripts/tests/config/splits 代码快照
  tests/test_stage2.py          新工具的失败处理测试
  outputs/                     本次已生成的离线结果
  docs/                        操作说明、代码索引和边界
  requirements-offline.txt      compare 所需 NumPy 版本
```

`src/stage2.py` 和所需证据随本目录一起保存，四个新入口可以独立使用。`legacy/07` 是保留原字节的历史代码，部分入口依赖原 07 的 artifacts、05 的模型和数据，因此不是一个可直接全量重跑的独立部署包。原模型权重、engine、完整 trace、数据集和虚拟环境仍在原位置。

## 直接运行

在 **Windows PowerShell** 中执行：

```powershell
Set-Location 'C:\Users\lenovo\Desktop\视觉边缘研究\09_边缘小目标检测混合精度量化_工程'
.\run.ps1 -Task verify
.\run.ps1 -Task status
```

前者应显示 `VERIFIED` 和 `checked_files: 98`；后者应显示历史 `gate05_status: INCOMPLETE` 和六配置结果。它们仅需 Python 3.8+ 标准库，不需要 CUDA、板子或网络。

读取方案：

```powershell
.\run.ps1 -Task plan
```

复算保存数组需要 NumPy。本机已验证的环境调用方式见 [运行说明](docs/01_运行说明.md)。这一步不运行 PyTorch、ORT 或 TensorRT 推理，也不重新进行 NMS；预测数来自原保存报告，数组统计来自当前重新读取的原始输出。

## 已保存结果

- [baseline_integrity.json](outputs/baseline_integrity.json)：98 份副本及其原 07 来源逐一核验，全部一致。
- [baseline_status.json](outputs/baseline_status.json)：从原最终收口记录读取的六配置状态。
- [saved_output_comparison.json](outputs/saved_output_comparison.json)：原始数组的有限值、形状、类别分数和描述性差异。
- [工程检查报告](docs/03_工程检查报告.md)：本次执行范围及检查结果。

FP16 固定图保存预测数为 2，五个量化配置均为 0；FP16 前后主、辅助数组逐位一致。所有结果只针对保存的 `000020.jpg`，不是新的数据集准确率或实机性能评估。

## 下一步入口

先阅读 [08 的研究方案](../08_边缘小目标检测混合精度量化_方向/03_下一阶段研究方案.md)，再决定实际设备输入捕获和浮点/融合对照。`next_protocol.json` 中 `DRAFT_NOT_FROZEN_NOT_EXECUTED` 说明后续协议仍是草案；它不是自动运行队列，也不会将旧失败改判。

本目录没有 SSH 凭据或自动连接功能；历史任务的临时授权已在原阶段结束时撤销。本次文件生成不检查板子是否在线。
