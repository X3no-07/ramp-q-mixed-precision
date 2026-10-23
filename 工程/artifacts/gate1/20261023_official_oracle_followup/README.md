# Gate1 既有官方评测运行环境补充搜索

日期：2026-10-23。**本次有限搜索没有发现可以直接运行的 MATLAB / Octave；官方合成 oracle 仍未执行。** 这不是对整台电脑没有安装运行时的证明。

本次在原有 PATH 与三个常见目录之外，继续检查了四处标准 App Paths、四处卸载登记、三处 MathWorks 注册表目录，以及 C 盘根目录、两个 Program Files、当前用户 Local Programs、Scoop、Chocolatey 和常见 tools / Software / Apps 安装目录。只读目录名和安装登记，没有扫描用户文档、遍历其他磁盘、修改注册表、安装或下载软件。详情见 [runtime_search_receipt.json](runtime_search_receipt.json)。

部分 App Paths 和卸载登记子项拒绝读取；收据逐根保留了错误。因此本次结论仅限可读取的记录及列明目录，不能排除受保护登记、其他磁盘或非标准路径下的安装。

旧 [官方评价器准备包](../20261023_official_evaluator_readiness/README.md) 的 12 个冻结文件已逐一复核，字节数与 SHA-256 全部一致，详见 [frozen_readiness_identity_check.json](frozen_readiness_identity_check.json)。没有往已封存目录新增执行输出，也没有修改原启动器、官方源码、20 个合成输入或原阈值。

原启动器限制输出在旧准备包内部；发现可用运行时后，应在新的执行目录另建包装器，先验证源文件身份，再调用冻结的 `run_synthetic_oracle.m`。该函数自身支持外部新输出目录。运行时还须具备原生 `mean2` 与 JSON 支持；不能替换缺失函数后冒称取得官方结果。

本次没有读取真实图像、注释、Selection 检测结果或 Holdout ZIP，没有运行模型。没有生成或认证完整官方 AP；Gate1、原数值失败及既有研究结论均不因此改变。
