# q_all：一次延长构建预算的诊断

本包保留原 `v4_pe_tap_avg1` 的模型、量化参数、构建程序及监控程序原始字节。
唯一运行设置变化是将监控构建上限从 1800 秒延长到 7200 秒（2 小时），仅尝试一次。
`--avgTiming=1`、1 GiB workspace、精度要求和原有资源保护保持原样。

请在当前 prefer 构建队列及其子进程全部结束后运行以下命令。输出目录必须尚不存在：

```bash
python3 -u /home/nvidia/rampq/gate05_qall_extended_20261019/jetson_monitored_build.py --root /home/nvidia/rampq/gate05_qall_extended_20261019 --run /home/nvidia/rampq/gate05_qall_extended_monitor_20261019 --build-script jetson_build_qdq.py --max-seconds 7200
```

`manifest.json` / `build_revision.json` 中的时间字段只是记录；真正的时间上限由命令中的
`--max-seconds 7200` 交给原监控程序执行。直接运行 `jetson_build_qdq.py` 不会执行这个监控上限。
不要删除超时记录后反复重试，也不要同时启动其他 GPU 构建。

两次原超时的 checkpoint 和监控状态文件路径、SHA256、资源结果保存在 `build_revision.json`。
原文件保留在客户端工程目录。本包没有把既有完整模型数值一致性失败改为通过。
构建成功后仍需另行检查实际层精度和真实单图执行；本包不运行图片推理或数据集评估。
