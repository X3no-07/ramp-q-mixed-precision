# Gate 0 数据划分检查

> 结论：**PASS**。本次只冻结 selection/holdout 图片名单，没有运行模块隔离实验。

## 冻结规则

- 随机种子：`20260910`
- 图片比例：selection 70%，holdout 30%
- 划分单位：整张图片；同一图片中的所有真实目标只能属于一个集合。
- 算法：先按图片名排序，再用 Python `random.Random(seed).shuffle`，前 70% 进入 selection。
- selection 用于选择模块；holdout 在模块与规则锁定前不得用于排名、调参或换模块。

## 结果总览

| 指标 | selection | holdout | 合计 |
|---|---:|---:|---:|
| 图片数 | 350 | 150 | 500 |
| 真实目标数 | 1920 | 865 | 2785 |
| FP32 命中数 | 1571 | 722 | 2293 |
| W8A8 新增漏检 | 119 | 46 | 165 |
| 含新增漏检的图片数 | 76 | 26 | 102（集合间无图片重叠） |

## 新增漏检类别分布

| 类别ID | selection | holdout | 合计 |
|---:|---:|---:|---:|
| 0 | 30 | 12 | 42 |
| 1 | 21 | 4 | 25 |
| 2 | 17 | 7 | 24 |
| 3 | 16 | 9 | 25 |
| 4 | 18 | 8 | 26 |
| 5 | 8 | 5 | 13 |
| 6 | 7 | 1 | 8 |
| 7 | 2 | 0 | 2 |

## 完整性检查

- 两集合图片重叠：0（要求 0）
- 覆盖全部 screen 图片：True
- 名单内路径全部存在：True
- selection 新增漏检不少于 100：119，通过
- holdout 新增漏检不少于 40：46，通过
- 主要类别（总新增漏检不少于 5）在两边都有：通过

## 文件与复现指纹

- selection 名单：`C:\Users\lenovo\Desktop\视觉边缘研究\07_边缘小目标检测混合精度量化_工程\splits\gate0_selection_images.txt`
- selection SHA-256：`a78348cb515553aece6e40dc02e21fc6be52df2c26989337f9d9a4638677e759`
- holdout 名单：`C:\Users\lenovo\Desktop\视觉边缘研究\07_边缘小目标检测混合精度量化_工程\splits\gate0_holdout_images.txt`
- holdout SHA-256：`4061739eb2e3b1812f2daf0b4327a65cf0cb7922eb43b30fc9457db310f976c7`
- 机器可读清单：`C:\Users\lenovo\Desktop\视觉边缘研究\07_边缘小目标检测混合精度量化_工程\artifacts\manifests\gate0_split.json`
- 输入预测表 SHA-256：`93f7a651201f2fdca747a2cfe1852f32bad5dd7cc2f6683c989ebdd258796388`

## 使用边界

- 类别ID 7总共只有2个新增漏检，随机划分后可能只出现在一侧；它不满足主要类别门槛，不能单独用于类别级结论。
- 当前划分只适用于这500张screen图片及其现有FP32/W8A8结果；若输入名单、随机种子或预测文件变化，必须重新生成并获得新的SHA-256。
- 下一步只能在selection上运行 `q_only:G1`；holdout继续封存。
