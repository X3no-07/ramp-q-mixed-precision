# 三个 prefer 恢复配置：各一次延长预算构建

只准备工程输入；本地脚本没有连接板子、构建引擎或执行图片。

顺序固定为 `r_only_G1 -> r_only_G1_G5 -> r_only_G1_G2`。
每项仅一次、最多 7200 秒，构建预算合计最多 6 小时，另有有界清理时间。

三个 bundle 内全部原文件逐字节复制，包括 `manifest.json`、`spec.json`、模型、
冻结 QDQ、精度请求、builder、monitor 和 `SHA256SUMS`。原 spec 的
`max_build_seconds=1800` 保留原值；它不是本次实际监控时限。
实际时限由外部队列调用原 monitor 时显式传入 `--max-seconds 7200` 执行。
外部 `budget_revision.json` 和逐项 launch/request 记录这个预算差异。
请不要修改原 spec 来掩盖差异，也不要直接运行 builder 绕过监控。

ZIP 的四个一级目录均解压到 `/home/nvidia/rampq/`；三个新 bundle 根目录以
`_v3_extended_7200` 结尾，输入根目录与队列输出根目录也都是新的。
目录已经存在时请先检查，不覆盖、不删除旧尝试。

全部旧 GPU 构建和单图执行结束后，先获得板子当前 boot UUID：

```bash
cat /proc/sys/kernel/random/boot_id
```

先只检查冻结文件（不会启动 GPU 工作）：

```bash
python3 /home/nvidia/rampq/gate05_restore_extended_inputs_20261019/jetson_gate05_extended_restore_queue.py --check-only
```

使用刚获得的 UUID 启动；不要照抄历史 UUID。队列在启动前、每项前和运行中核对该值：

```bash
python3 -u /home/nvidia/rampq/gate05_restore_extended_inputs_20261019/jetson_gate05_extended_restore_queue.py --expected-boot-id LIVE_BOOT_UUID
```

本地离线校验可执行：

```powershell
python "C:\Users\lenovo\Desktop\视觉边缘研究\07_边缘小目标检测混合精度量化_工程\artifacts\gate05\20261019_restore_extended_input_v2\gate05_restore_extended_inputs_20261019\jetson_gate05_extended_restore_queue.py" --check-only --local-archive-root "C:\Users\lenovo\Desktop\视觉边缘研究\07_边缘小目标检测混合精度量化_工程\artifacts\gate05\20261019_restore_extended_input_v2"
```

共享全局构建队列锁，拒绝其他 trtexec、monitor、queue 和已知 smoke/build 工作；
每项前要求风扇 active、MemAvailable 至少 3 GiB、所有温度低于 85°C。
原 monitor 的 kernel/collector/温度/低内存保护保持原字节，新队列也重复安全检查。
前项终态须核对 monitor PID、boot、root、7200 秒有效预算；确认记录的 owned
PID/start_ticks 和私有 session/PGID 后代全部退出、全局 trtexec 为空后才前进。
普通构建失败或单纯超时允许进入下一项；安全事件、boot 改变、收集器失败、
进程丢失或无法确认清理完成会停止整队。不会自动重新运行同一个配置。

构建成功不代表实际精度采用或数值迁移通过。每个引擎仍要使用原实际层格式及
INT8 tactic 审计标准，再另行执行冻结单图。原 PyTorch/ONNX 数值失败保持失败；
本包不运行数据集评估、不改变精度或精度通过标准，不建立恢复、因果或性能结论。
