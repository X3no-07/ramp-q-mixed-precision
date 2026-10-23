> 发布展示副本：仅调整链接；原始字节保存在工程目录同名文件。被排除成员及Release结果以发布清单为准。

# Gate1：数据已准备，GPU 训练预检通过

日期：2026-10-22。本轮完成 VisDrone 数据准备和训练流程检查，状态为 **PREPARATION_AND_ENGINEERING_PREFLIGHT_PASSED**。正式浮点基线尚未训练完成，Gate1 的量化机制与选模验收尚未开始。

## 本轮完成了什么

已获取 VisDrone-DET 的训练集与开发集，保留原始压缩包、图像和原标注；最终 test-dev 压缩包单独封存，未列举其内部成员、解压、查看标注或运行推理。数据来源沿用 [VisDrone 官方任务入口](https://github.com/VisDrone/VisDrone-Dataset)，下载镜像为本地 Ultralytics 8.4.142 数据配置所引用的 [Ultralytics 固定发布资源](https://github.com/ultralytics/assets/releases/tag/v0.0.0)。

| 数据部分 | 原始图像数量 | 当前用途 |
|---|---:|---|
| 官方 train | 6,471 | 按组预留训练验证、校准、调试，其余用于后续训练 |
| 官方 val | 548 | Selection；本轮仅清点和转换，没有检测结果或模块选型 |
| test-dev | 本轮未检查包内数量 | 保持原始 ZIP 封存，仅核对压缩包字节数与 SHA |

两个已准备数据集共保留 **7,019 张图、393,719 行非空原标注**。另存的 YOLO 标签仅为训练便利格式，类别 1–10 映射至 0–9；忽略区域、`others`、原始物理行身份和裁剪记录保存在 sidecar。训练框按原图边界裁剪，原图和原标注字节没有修改。

独立核验工具没有调用准备脚本的转换、分组函数，重新对照了原 ZIP、交付清单内的 **21,077 个文件**、7,019 张图的尺寸/图像/原标注、YOLO 标签和两个标注 sidecar，并重建前缀与内容 SHA 的连接组件。8 项核验通过；其结论限定为 **PASS_PREPARATION_FIDELITY_ONLY**，不代表泛化独立性、官方 AP 或 Gate1 科学结论通过。

## 当前划分是可复查的准备版本

| 角色 | 图片数 | 连接组数 | 说明 |
|---|---:|---:|---|
| train | 4,885 | 155 | 后续浮点训练候选数据 |
| training_val | 409 | 2 | 训练内验证；覆盖不足，正式训练前需改进 |
| debug | 109 | 6 | 工程调试；本轮只取其中 64 图训练检查 |
| calibration_pool | 571 | 18 | 与训练、调试、Selection 隔离；尚未校准 |
| selection | 548 | 75 | 完整官方 val，尚未检测或评分 |
| quarantine | 497 | 27 | 跨 split 关联、重复冲突或标注质量原因隔离 |

以固定种子 `20261022` 产生整组划分提案；连接组由文件名首段前缀与精确图像 SHA 共同建立。它们仍是场景代理，不能视作已经核实的真实拍摄场景。

三份校准名单使用种子 `2026102201/2026102202/2026102203`，各 64 图，全部来自校准池，内容 SHA 两两不交叠。此次没有读取这些图进行校准或模型推理。

检查发现 3 组精确重复内容记录，以及 24 条跨 train/val 前缀或内容关联记录。关联训练组件被隔离；官方 val 保留。精确重复与前缀隔离已核验，近重复图像和真实场景关联仍待检查。

## 原始标注中的异常已有明确记录

全量扫描发现训练集有两条本来就标为忽略区域的零高度框，以及一条 `score=1、category=4` 的零高度框；开发集没有非正尺寸框。

原忽略框完整保留，不生成训练目标。正类异常框为 `train/9999985_00000_d_0000020:11`，原行 `611,158,4,0,1,4,0,0`。该行保留原 score、类别、原文和物理行 ID，另标 `excluded_for_preparation_quality=true`，没有冒充发布者本来就忽略的标注。

该异常图所在整个关联组件的 **79 张训练图**均进入 quarantine；这 79 张包含在上述 497 张中，未进入训练、训练验证、调试或校准。开发集若出现同类异常仍会严格拒绝处理。相关代码已通过 **28 项测试**，首次准备失败记录与当时源脚本也保留。

## 电脑已经能完成新模型的训练流程

本机是 NVIDIA RTX 5060 Laptop GPU，8,151 MiB 显存；环境为 Python 3.12.10、PyTorch 2.10.0+cu128、Ultralytics 8.4.142。CUDA 运算和 10 类架构检查通过，重新核对了 88 个卷积的 G1–G8 映射。

从 [Ultralytics v8.3.0 官方资源](https://github.com/ultralytics/assets/releases/tag/v8.3.0)取得通用 YOLO11n 权重，与已有本地通用权重 SHA 完全一致：`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。它是通用初始化，未使用 RUOD-R 检测权重充当 VisDrone 基线。

本轮训练预检设置如下：

- 从 debug 名单取 64 图训练，从 training_val 取 32 图验证，均来自官方 train。
- 训练 1 个 epoch，640×640、实际 batch=4、workers=0、AMP 关闭、seed=20261022。
- 所选 96 张原图及标签逐条核对来源角色与 SHA，再复制到独立预检目录。缓存也写入该副本目录；预检后原始 96 组图像/标签 SHA 未变，准备目录没有新增 cache。
- 完成训练、轮末验证和框架默认的最终 best 权重验证，总流程约 **33.66 秒**。保存了 best/last 权重，10 类名称、88 个卷积和模型参数有限性检查通过。

该时间包含这次小规模检查的开销，不能据此推定完整训练工期。日志中的检测分数只用于工程诊断；该模型仅训练一轮，不能作为合格浮点基线，更不能写成 VisDrone 官方 AP、量化恢复或选模优越性证据。框架内置的 OOM/NaN 恢复政策已在运行记录中披露，外层没有自动重启训练。

预检首次导入时产生了配置目录写入警告。后续入口已改为先创建本轮专用配置目录，并通过独立的配置路径检查；原运行源码、控制台和模型结果保留，没有为这项设置修复重跑训练。

## 正式基线前还需要处理的事项

**当前最明确的设计问题是训练内验证集场景覆盖过窄：409 张图只有 2 个连接组，其中一个占 407 张。**这次小检查只能确认数据与训练流程可执行；正式训练前应重新设计覆盖更多组的训练验证划分，记录新版本，保留当前提案。

接下来按顺序完成：

1. 改进训练验证集的组覆盖，并核对真实场景身份和近重复图像；保持校准、Selection 与最终确认数据的边界。
2. 固定忽略区域的检测评价实现。普通 YOLO 训练仍可能将忽略区域作为负样本背景，内置验证也不完整实现 VisDrone 忽略规则；sidecar 存在不能代替评价实现。
3. 冻结正式浮点训练的名单、初始化、全部参数、训练预算与 checkpoint 选择规则，再训练可供量化研究使用的模型。
4. 按[Gate1 研究方案（见证据下载索引）](../发布清单/证据下载说明.md)完成数值、模块与研究验收合同，之后再进行 Selection 干预和最终 Holdout 确认。

当前没有 Gate1 的模块筛选、量化指标、独立确认或性能结论。Gate0.5 的历史结论仍以 [39 号报告](39_Gate05_150图完整验收结果.md)为准，原完整数值护栏失败保留。

## 数据与复核入口

- [准备结果与原始内容身份](../artifacts/gate1/20261022_preparation/prepared_v2/preparation_summary.json)、[逐图清单（见证据下载索引）](../发布清单/证据下载说明.md)、[划分提案](../artifacts/gate1/20261022_preparation/prepared_v2/split_proposal.json)、[全文件清单（见证据下载索引）](../发布清单/证据下载说明.md)。
- [独立准备审核](../artifacts/gate1/20261022_preparation/prepared_v2_independent_audit/audit.json)、[测试记录](../artifacts/gate1/20261022_preparation/preparation_tests_v3_console.txt)。
- [训练预检记录](../artifacts/gate1/20261022_preparation/training_preflight_v1/preflight_receipt.json)、[本次实际运行源码](../artifacts/gate1/20261022_preparation/training_preflight_v1/runner_source.py)、[后续完整性复核](../artifacts/gate1/20261022_preparation/preparation_and_preflight_review.json)。
- [最终测试集封存记录（见证据下载索引）](../发布清单/证据下载说明.md)、[官方通用权重身份](../artifacts/gate1/20261022_preparation/sources/yolo11n_official_weight_receipt.json)。
- 工程入口：[数据准备](../scripts/prepare_gate1_visdrone.py)、[独立核验](../scripts/verify_gate1_prepared_data.py)、[训练预检](../scripts/run_gate1_training_preflight.py)。这些入口拒绝覆盖已有输出；新尝试须使用新目录。

下载包匹配发布资源 ID 与字节数，本地 SHA 记录其身份；发布 API 没有提供这些资源的 digest，不能将本地计算的 SHA 宣称为发布者提供的校验值。test-dev 的封存属于执行流程约束，不是加密或操作系统读访问隔离。
