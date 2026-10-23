# Gate1 浮点资格检查输出复算工具（待真实执行）

本包只准备并合成验证输出审计器。没有读取真实模型、原始数据图像/标注，没有推理，也没有接触 Selection 或 Holdout。

`audit_gate1_float_qualification.py` 只接受完整资格检查目录，输出到新目录。它核验输入全部文件清单与 SHA；验证预先冻结的 109 张 debug 身份；独立从原始 NPZ 重算 FP32/FP16 形状、dtype、有限值、几何/概率域及固定 allclose 判据；从每个保留 anchor 核查分数、类别 +1 映射和原图坐标；重新解析原始标注的物理行 ID；重算三个置信度下的逐目标命中分区、FP/FN 和汇总。

坐标检查先以 FP32 计算 `center ± size/2`，再以 FP64 撤销 padding/gain。坐标保存一致性使用固定 `rtol=1e-12, atol=1e-9`，它与原始张量 dtype guard 的容差分别报告。

执行方式（训练和资格检查均完整结束后，由主代理指定真实路径）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '..\05_水下小目标检测_工程\检测工程\.venv\Scripts\python.exe' scripts/audit_gate1_float_qualification.py --qualification '<已完成资格检查目录>' --output '<新的审计目录>'
```

16 项合成测试通过（41.125 秒），覆盖完整 109 图合成输出、被保留的 dtype FAIL、篡改汇总、未完成状态、源身份、坐标/类别、物理行以及忽略计数。第一次测试因沙箱系统临时目录拒绝访问而失败，其记录保留；改用本工程内的临时目录后通过，没有放宽判据。

`frozen_debug_identity.json` 只从已封存的 109 行元数据派生，不打开其指向的图片或标注。原库存 SHA 为 `d056e3782aabce761d268cf3875ca8981e15be591bfa8f12d3cc8a42e017e7a1`；规范化成员身份 SHA 为 `059b06216d35535d91c21b34955ba545331664b748edd4f822802cd78e927e75`，已写入审计器。

局限：忽略规则重算使用冻结项目 evaluator，因此只证明该实现的输出一致性，没有独立 MATLAB/Octave oracle。审计器不重跑 NMS，不重新解码图片，不反序列化模型，也不重新证明全体数据划分的隔离。它不计算官方 AP，不代表 Gate1 科学验收，也不修复/替代 Gate0.5 的旧 guard。即使审计一致性通过，dtype guard 的失败仍会原样保留。
