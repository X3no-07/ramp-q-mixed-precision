# Gate1 忽略规则：固定源码与实现边界

核对日期：2026-10-22。官方仓库为 [VisDrone/VisDrone2018-DET-toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)，实际固定 commit 为 `005445782213e20cb91bc50a597db3dd949e749a`。`official_toolkit/` 保留本次获得的完整源码，逐文件工作副本 SHA-256 和 Git blob 身份见 `source_manifest.json`。本轮仅获取和阅读源码，未执行其默认指向 test-challenge 的 `evalDET.m`，未读 test-dev 数据、真实模型输出或运行检测评分。

## 1. 实际调用链

`evalDET.m` → `saveAnnoRes.m` → `dropObjectsInIgr.m` / `createIntImg.m` → `calcAccuracy.m` → `evalRes.m` / `compOas.m` / `VOCap.m`。另外 `findImageList.m` 决定图像遍历顺序，`displayImage.m` 是显示分支，不负责指标。源码中的名字或注释不能代替下面的实际行为。

## 2. category=0 的过滤不是普通连续 IoF

`dropObjectsInIgr.m:5–16` 只将 GT 的 category=0 作为全类忽略区域；score=0 或 category=11 本身不创建该像素图。所有忽略区域先逐元素取 `max(1, xywh)`，然后按一基坐标的包含端点区间写入像素并集。原始区域的零宽/零高会被提升为 1；源码区域赋值要求可用于 MATLAB 索引的整数坐标。

对非 category=0 GT 和每个预测，`dropObjectsInIgr.m:19–48` 将 xywh 按 MATLAB 的半数远离零规则舍入，再逐元素限制至少为 1。查询起点进一步限制在图像范围内，终点限制到右/下图像边界；积分图差分实际覆盖 `(x, x+w] × (y, y+h]` 的像素。分母仍是舍入后的原始 `w*h`，不是裁到图像内的面积。覆盖率 **≥0.5** 的 GT 和预测均被删除，重叠忽略区域不会重复计数。返回的幸存框仍是原始坐标，并没有被上述查询坐标替换。

这与“连续矩形交集面积除以预测面积≥0.5”在边界、零尺寸、舍入和图像外框等情况下不同。Python 实现显式保留该差异，并以独立小像素枚举测试积分图查询。

## 3. score=0、others 和属性

`saveAnnoRes.m:14–15` 将幸存 GT 的 score=0 变为 ignore 标志 1，将原 score=1 变为 0。`calcAccuracy.m:8–32` 只评价类别 1–10，因此 category=11 GT 和预测不计入这些类别；**源码没有把 others 框变成吸收所有类别预测的忽略区域**。

`evalRes.m:63–105` 优先匹配有效 GT；未得到有效匹配的预测可以匹配同类 score=0 GT，重叠分母改用预测面积，一个此类忽略 GT 可以吸收多个预测。截断和遮挡字段没有参与过滤或匹配，不能自行以遮挡较重等理由删除有效目标。项目严格拒绝本应有效但尺寸非正的 GT，要求先按数据准备政策隔离，而不是在评价中悄悄修复；这属于额外输入质量约束，不冒称 MATLAB 对所有异常输入的行为。

## 4. 匹配及 AP 的不同层次

官方 `evalRes.m` 按预测分数降序处理，GT 有效项排在 ignore 项之前；有效 GT 一对一，IoU 达到阈值即可匹配，等重叠时循环后遇到的有效 GT 会覆盖前一个选择。忽略匹配使用交集面积 / 预测面积。框面积与交集采用连续 `x+w`、`y+h`，无 `+1` 面积项；这一层不把框裁到图像边界。不同 MATLAB 版本的同分排序行为尚未由运行 oracle 检验。

项目既有同目标分析是另一算法：全局按 `(IoU, confidence, GT原物理行号, 预测原输入序号)` 降序贪心，一对一分配。新 `match_project_targets` 保留该口径，先做上述 category=0 过滤，再把幸存 GT/预测裁到 `[0,W]×[0,H]`，以 confidence≥0.25、IoU≥0.50 匹配有效目标；仅对未匹配预测做同类 score=0 的 IoF 吸收。两种算法分别命名、分别测试，不能把项目逐目标统计称为官方 AP。

固定源码的 AP 实现还有必须保留或明确另设口径的细节：

- `calcAccuracy.m:21–31` 先取每图提交文件原顺序的前 `[1,10,100,500]` 个预测，再分类别；不是先为全图按分数重排后截断。后续类别内部和跨图的排序才按分数进行。预测提交顺序必须冻结。
- AP 使用 maxDets=500、IoU=0.50:0.05:0.95；AR 另外报告 1/10/100/500。源码没有 NMS，本项目的 NMS 与候选截断须在上游另行固定。
- `VOCap.m` 用精度单调包络在召回率变化处积分，不是 COCO 的 101 个召回点采样。
- `calcAccuracy.m:11–15,50–56` 在每个含该类别 GT 的图像上向 `evalClass` 再追加一次类别号，未做 unique。因此最终平均可能按各类出现图像次数重复计权；不能默认为去重类别后的宏平均。
- `calcAccuracy.m:33,39` 的召回分母是所有当前类别 GT 匹配标志的个数，包括幸存 score=0 GT；不能默认为有效非忽略 GT 数。是否影响具体数据需另行核对，当前没有读取真实检测结果。

为避免未经核验地复现这些细节，本次**没有实现或宣称完整官方 AP**。内置 YOLO `mAP50–95` 继续仅用于 train 派生验证集的训练监控和选权重，普通 YOLO 对忽略区域负样本的限制仍然保留。

## 5. 本次技术验证与尚未证明项

本地实现包括原始 CSV/物理行 ID 解析、category=0 像素过滤、项目匹配及单图官方风格匹配。37 项合成测试通过；覆盖 50% 边界、并集、图像边缘、半数舍入、原忽略零高、同类 score=0 IoF、多预测吸收、others、截断/遮挡、裁剪差异、同分并列、一对一、输入顺序 cap、错误输入与空输入。

PATH 及本次检查的 `C:/Program Files/MATLAB`、`C:/Program Files/GNU Octave`、`C:/Program Files/Octave` 中均未找到可运行 MATLAB/Octave。这不等于证明整台机器绝无其他安装；本轮没有可用 oracle。**合成测试证明声明的 Python 规则，不证明官方端到端 AP 等价**。正式研究冻结前仍需官方运行 oracle 固定样例或明确将官方 AP 留作未完成分项；不能用测试通过覆盖这一缺口。

本实现也不证明数据的近重复/真实场景独立性，不运行 NMS，不校验模型数值，不输出 Gate1 科学结论。机器可读政策见 `evaluation_policy.json`。
