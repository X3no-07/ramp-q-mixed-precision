# Gate1 官方评价器离线就绪审计

日期：2026-10-23。本轮完成了源码身份复核、37 项既有合成回归，以及 20 个完全合成的官方运行输入和可执行入口。**官方运行尚未执行，完整 AP 等价性未认证，Gate1 没有因此通过。**

本地保留的官方源码来自 VisDrone/VisDrone2018-DET-toolkit，commit 为 `005445782213e20cb91bc50a597db3dd949e749a`。11 个文件的 SHA-256 与既有下载清单逐一相符。所有旧源码、测试和文档保持原样。详细身份与缺口见 [readiness.json](readiness.json)。

## 已验证和缺口

现有 `scripts/gate1_ignore_aware_evaluation.py` 只有 category=0 像素过滤、项目逐目标匹配和未认证的单图官方风格匹配；它没有全数据集 AP/AR 聚合。37 项测试本次重新通过，只证明声明的 Python 行为。

官方 `calcAccuracy.m` 的实际聚合需要保留输入顺序先 cap、每类和跨图按分数排序、IoU=0.50:0.05:0.95、AP maxDets=500、AR maxDets=1/10/100/500，以及 `VOCap` 的精度单调包络积分。类别列表按含该类的图像重复追加，召回分母为 `numel(gtMatch)`（包括幸存 score=0 GT）。这些源码细节不能默认为 COCO AP 或常规去重类别宏平均，也不能默默修正后仍称为同一官方口径。

本次 PATH 的 `matlab`、`octave`、`octave-cli`，以及三个常见安装目录中均未发现运行时；不是对整台电脑所有目录的穷尽搜索。实际启动记录为 [NOT_EXECUTED_NO_NATIVE_RUNTIME_ON_PATH](execution_01/launcher_receipt.json)。未安装或下载软件。

## 离线执行入口

[synthetic_fixtures.json](synthetic_fixtures.json) 含 20 个输入，分为 11 个 AP/AR 聚合、4 个像素忽略过滤、5 个单图匹配用例。覆盖空 GT/预测、score=0、类别出现频次、输入顺序 cap、精度包络、非网格召回端点、50% 忽略边界、区域并集、图像边界/舍入、原忽略框零高、同分/等 IoU、重复吸收与图像外坐标。没有手写预期指标答案；未来应由固定官方源码产生 oracle 输出。

在已有 MATLAB/Octave 出现在 PATH 且原生 `mean2`、JSON 支持可用后，从工程根目录运行：

```powershell
& 'C:\Users\lenovo\Desktop\视觉边缘研究\05_水下小目标检测_工程\检测工程\.venv\Scripts\python.exe' -B 'artifacts/gate1/20261023_official_evaluator_readiness/launch_synthetic_oracle.py' --output 'artifacts/gate1/20261023_official_evaluator_readiness/execution_02'
```

输出目录必须是本就绪包内一个新子目录；已有结果不会覆盖。启动器先检查源码和合成输入的冻结 SHA，再调用 [run_synthetic_oracle.m](run_synthetic_oracle.m)。原官方 `.m` 文件不改动，也不替换 `mean2`；缺依赖会保留失败，不能假装运行成功。入口只把合成矩阵送入 `dropObjectsInIgr`、`evalRes` 和 `calcAccuracy`，从不调用带数据路径或 `imread` 的 `evalDET`、`saveAnnoRes`。

成功时保存原生 `.mat` 与 JSON、中间过滤/匹配矩阵及七项 AP/AR 输出，另存运行版本和函数解析路径。MAT 保留空矩阵形状与 NaN，JSON 可能将非有限值转为 null，因此不能仅用 JSON 判断空输入语义。个别用例出错也会写入输出，然后以失败结束。

## 后续验收

原生输出产生后，才能比较过滤矩阵、每类匹配状态、排序/截断、空值处理和最终聚合。现阶段没有完整 Python AP 实现，因而即便将来原生合成用例全部可执行，也只是取得参考结果；仍须另行完成实现和逐项比较。项目 IoU 优先并裁框的逐目标指标仍须单独命名。

本轮未读真实图像/注释、Selection 检测输出或 Holdout ZIP，未运行模型，未改动浮点数值阈值，也未替代既有训练监控、数值资格检查或研究验收。
