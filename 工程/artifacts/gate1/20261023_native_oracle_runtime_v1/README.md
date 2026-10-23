# 原生合成评价器运行时准备

本目录为 2026-10-23 的新准备目录。旧 `20261023_official_evaluator_readiness`、`20261023_official_oracle_followup` 和官方 `.m` 文件保持封存。固定 portable 包已下载、官方发布 SHA 核验通过，获准的 CLI 子集已提取，原生评价器的 20 组合成输入已实际执行成功。独立结构与身份审计由另一程序另行记录；这里不声称完整 Python AP 等价。

选择 [GNU Octave 官方下载页](https://octave.org/download.html)列出的 Windows-64 推荐版 11.3.0。采用官方项目 [GitHub release 资产](https://github.com/gnu-octave/octave/releases/expanded_assets/release-11-3-0)中的 `octave-11.3.0-w64.7z`，HEAD 长度为 481,127,074 bytes。其发布 SHA-256 为 `fd3cf0e885467a15211b8ceded42800432a5462481324d49a896ef257e05d1a0`。归档和 detached signature 将分别比对该官方 HTTPS 页面公布的 SHA；本机没有找到 GPG，不能宣称已验证 PGP 签名或密钥信任链。GNU mirror 的一次最多 1 MiB 测速在 TLS 握手时超时，未改换来源、未另存第二份完整归档。

Octave 的 [JSON 官方文档](https://docs.octave.org/latest/JSON-data-encoding_002fdecoding.html)提供 `jsondecode`、`jsonencode`，同时说明 JSON 会丢失部分矩阵形状和 NaN 表达，因此 `.mat` 原生输出仍须保留。[`mean2` 函数参考](https://octave.sourceforge.io/image/function/mean2.html)表明它属于 image 包；计划从同一官方 Windows 归档解出原函数并明确解析路径，不编造、替换或翻译 `mean2`。[Windows 官方说明](https://wiki.octave.org/Octave_for_Microsoft_Windows)提及预装包、手动加载和 post-install 步骤；此处会先阅读脚本，而不自动运行它。

空间约束由主任务实测：下载前 C 盘可用 4,560,896,000 bytes，展开后必须保留至少 1.5 GiB。`inspect_archive.py` 已调用系统 bsdtar 成功只读列出 63,302 个成员：59,532 个普通文件未压缩总量为 2,797,722,179 bytes；不计划全量解压。保守 CLI 子集为 2,890 个文件、640,420,084 bytes，包括 CLI、DLL、核心 `.m`/`.oct`、原 image 包与许可证。提取前一次 WinAPI 实测可用空间为 4,269,973,504 bytes；还会在提取动作前再次按子集总量、78,946,304 bytes 文件系统余量和 1.5 GiB reserve 强制检查。系统 bsdtar 3.8.9 的 libarchive 上游[格式文档](https://github.com/libarchive/libarchive/wiki/LibarchiveFormats)列有 7-Zip 支持，本归档的成员读取也已实际成功。

GNU Octave 采用 [GPL-3.0-or-later](https://octave.org/copyright/)，image 包官方目录也标示该许可。解压时应保留原许可与第三方依赖声明，不把 GPL 运行时并入项目模型代码或部署引擎。

`launch_native_oracle_v2.py` 已验证 2,890 个运行时文件、11 个官方源文件和冻结的 20 组合成输入，然后调用未改动的 `run_synthetic_oracle.m`。实际输出在 [execution_01/launcher_receipt.json](execution_01/launcher_receipt.json)：退出码为 0，状态为 `EXECUTED_NATIVE_SYNTHETIC_ORACLE_UNCERTIFIED`，20 个用例均无原生错误；`.mat` 和 JSON 同时保留。CLI 实际版本为 11.3.0，`mean2` 解析到归档自带 image-2.20.0 原文件，JSON 函数为内置实现；未运行 post-install，未设置额外环境变量，未安装软件。

它不运行真实模型、不读取真实图像/标签、Selection 或 Holdout，也不修改系统 PATH 或注册表。为规避中文路径风险，执行期间曾使用临时 ASCII `R:` SUBST 映射，真实目标为本工程目录；启动器保存别名关系并以 `samefile` 验证。默认权限上下文中的 `subst` 查询返回空，保留原始空文件，另存同创建权限上下文下的 `subst_query_elevated_active.txt` 证据；不篡改执行回执。独立结构审计及局部比较完成后，已移除临时映射，退出码为 0，`R:` 路径不再存在，见 `temporary_alias_cleanup.json`。

独立审计记录在相邻 `20261023_native_oracle_comparison_preparation_v1/native_output_audit_v1.json`：20 个输入的 MAT/JSON 结构、空矩阵维度与 NaN 序列化一致，六个原函数解析身份和运行时清单均通过。另一个独立阶段在 `20261023_native_python_semantics_v1/comparison_result_v1.json` 对现有 Python 过滤/匹配中间矩阵进行比较，166/166 个矩阵严格一致；这仅覆盖现有局部语义，没有完整 AP 聚合实现或等价性认证。

`source_plan.json` 保留下载前的历史计划；当前最终状态以 `preparation_final_receipt.json` 为准。`preparation_manifest.json` 直接绑定准备源代码、所有下载/错误日志、归档/签名与原生输出；解出的 2,890 个原生依赖由 `runtime_manifest.json` 逐文件传递绑定，顶层不重复复制这些条目。

即使将来取得无 case error 的原生合成输出，也只取得参考 oracle；它不会自动认证完整 Python AP 等价性、Gate1 科学验收、真实数据集 AP 或板端部署。
