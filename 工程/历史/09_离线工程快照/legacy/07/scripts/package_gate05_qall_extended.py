"""Freeze one longer supervised attempt of the unchanged q_all avgTiming=1 graph."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = 'gate05_qall_extended_20261019'
BOARD_ROOT = '/home/nvidia/rampq/' + ARCHIVE_ROOT
BOARD_MONITOR = '/home/nvidia/rampq/gate05_qall_extended_monitor_20261019'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def prior_timeout(directory):
    checkpoint_path = directory / 'checkpoint.json'
    monitor_path = directory / 'monitor/status.json'
    checkpoint, monitor = read(checkpoint_path), read(monitor_path)
    assert checkpoint['status'] == 'FULL_QDQ_PE_TAP_BUILD_TIMEOUT_INCONCLUSIVE'
    assert checkpoint['full_model_engine_built'] is False and checkpoint['full_model_single_image_executed'] is False
    assert checkpoint['kernel_events'] == [] and checkpoint['local_pytorch_onnx_parity'] == 'FAILED_UNCHANGED'
    assert monitor['status'] == 'STOPPED_BY_MONITOR' and monitor['reason'] == 'Build supervision time limit reached'
    assert monitor['max_seconds'] == 1800 and monitor['boot_id'] == checkpoint['boot_id']
    return {'checkpoint': str(checkpoint_path.relative_to(ROOT)).replace('\\', '/'),
            'checkpoint_sha256': sha(checkpoint_path),
            'monitor_status': str(monitor_path.relative_to(ROOT)).replace('\\', '/'),
            'monitor_status_sha256': sha(monitor_path),
            'boot_id': monitor['boot_id'], 'monitor_status_value': monitor['status'],
            'monitor_stop_reason': monitor['reason'], 'prior_max_build_seconds': 1800,
            'resources': checkpoint['resources'], 'kernel_events': [],
            'full_model_engine_built': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/gate05/20261019_qall_extended_input')
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists(), 'Choose a fresh output; preserve previous attempts'
    source = ROOT / 'artifacts/gate05/20261018_int8_pe_tap/diagnostic_bundle_v4_pe_tap_avg1'
    source_manifest = read(source / 'manifest.json')
    assert source_manifest['schema'] == 'rampq.gate05.qdq_inputs/v2'
    assert source_manifest['configuration'] == 'q_all' and source_manifest['diagnostic_only'] is True
    assert source_manifest['kernel_selection_avg_timing'] == 1
    for name, info in source_manifest['files'].items():
        path = source / name
        assert Path(name).name == name and path.resolve().parent == source.resolve()
        assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], name
    assert read(source / 'preflight.json')['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert read(source / 'build_revision.json')['max_build_seconds'] == 1800
    for name in ('jetson_build_qdq.py', 'jetson_monitored_build.py'):
        ast.parse((source / name).read_text(encoding='utf-8'), feature_version=(3, 8))
    previous = [prior_timeout(ROOT / 'artifacts/gate05/20261018_int8_pe_tap'),
                prior_timeout(ROOT / 'artifacts/gate05/20261019_int8_avg1')]
    assert len({row['boot_id'] for row in previous}) == 1
    output.mkdir(parents=True, exist_ok=False)
    for name in source_manifest['files']:
        shutil.copy2(source / name, output / name)
    command = ['python3', '-u', BOARD_ROOT + '/jetson_monitored_build.py', '--root', BOARD_ROOT,
               '--run', BOARD_MONITOR, '--build-script', 'jetson_build_qdq.py', '--max-seconds', '7200']
    revision = {'revision': 'qall_extended_7200_20261019',
                'prepared_client_at': datetime.now().astimezone().isoformat(),
                'client_date': datetime.now().astimezone().date().isoformat(),
                'status': 'STAGED_NOT_BUILT_OR_EXECUTED', 'configuration': 'q_all',
                'source_bundle_manifest_sha256': sha(source / 'manifest.json'),
                'source_build_revision_sha256': sha(source / 'build_revision.json'),
                'source_onnx_sha256': sha(source / 'model_q_all.onnx'),
                'source_builder_sha256': sha(source / 'jetson_build_qdq.py'),
                'source_monitor_sha256': sha(source / 'jetson_monitored_build.py'),
                'change': 'One supervised build budget extended from 1800 to 7200 seconds; exact existing avgTiming=1 search retained',
                'max_build_seconds': 7200, 'maximum_attempts': 1,
                'kernel_selection_avg_timing': 1, 'workspace_mib': 1024,
                'required_monitor_command': command,
                'required_launch_order': 'Launch only after the current prefer queue and all its descendants/trtexec have ended',
                'max_seconds_metadata_alone_enforces_nothing': True,
                'actual_time_limit_enforced_by': 'Unchanged jetson_monitored_build.py invoked with explicit --max-seconds 7200',
                'original_build_command_and_builder_bytes_unchanged': True,
                'graph_weights_calibration_quantization_and_numerical_tolerances_unchanged': True,
                'output_binding_contract_unchanged': True, 'nms_and_target_matching_unchanged': True,
                'precision_and_scientific_criteria_changed': False,
                'new_heuristic_or_timing_cache_added': False,
                'preflight_status': 'LOCAL_QDQ_PREFLIGHT_FAILED',
                'prior_timeout_evidence': previous,
                'reason': 'Both prior attempts ended solely at the 1800-second observation budget, without a complete engine. '
                          'Their retained resource records show temperatures below 55 C, adequate available memory, '
                          'and no new kernel/OOM safety events. A longer bounded observation preserves this existing search.',
                'limits': 'One hardware feasibility attempt only. Resource guards remain unchanged. Engine build, '
                          'actual precision inspection and execution remain separate evidence; original numerical parity '
                          'failure remains failed. No new dataset evaluation, NMS, precision masks, accuracy criteria, '
                          'rescue claim, performance benchmark, or automatic retries.'}
    write(output / 'build_revision.json', revision)
    launch = ' '.join(command)
    readme = '''# q_all：一次延长构建预算的诊断

本包保留原 `v4_pe_tap_avg1` 的模型、量化参数、构建程序及监控程序原始字节。
唯一运行设置变化是将监控构建上限从 1800 秒延长到 7200 秒（2 小时），仅尝试一次。
`--avgTiming=1`、1 GiB workspace、精度要求和原有资源保护保持原样。

请在当前 prefer 构建队列及其子进程全部结束后运行以下命令。输出目录必须尚不存在：

```bash
''' + launch + '''
```

`manifest.json` / `build_revision.json` 中的时间字段只是记录；真正的时间上限由命令中的
`--max-seconds 7200` 交给原监控程序执行。直接运行 `jetson_build_qdq.py` 不会执行这个监控上限。
不要删除超时记录后反复重试，也不要同时启动其他 GPU 构建。

两次原超时的 checkpoint 和监控状态文件路径、SHA256、资源结果保存在 `build_revision.json`。
原文件保留在客户端工程目录。本包没有把既有完整模型数值一致性失败改为通过。
构建成功后仍需另行检查实际层精度和真实单图执行；本包不运行图片推理或数据集评估。
'''
    (output / 'README.md').write_text(readme, encoding='utf-8')
    unchanged = [name for name in source_manifest['files'] if name != 'build_revision.json']
    assert all((output / name).read_bytes() == (source / name).read_bytes() for name in unchanged)
    manifest = copy.deepcopy(source_manifest)
    manifest.update(max_build_seconds=7200, maximum_attempts=1,
                    required_monitor_command=command,
                    budget_metadata_requires_explicit_monitor_argument=True)
    manifest['files'] = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                         for path in sorted(output.iterdir()) if path.is_file()}
    write(output / 'manifest.json', manifest)
    (output / 'SHA256SUMS').write_text(''.join(info['sha256'] + '  ' + name + '\n'
                                           for name, info in manifest['files'].items()), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, ARCHIVE_ROOT + '/' + path.name)
    print(json.dumps({'status': 'ONE_EXTENDED_QALL_ATTEMPT_PREPARED_NOT_RUN',
                      'output': str(output), 'archive': str(archive_path), 'archive_root': ARCHIVE_ROOT,
                      'archive_sha256': sha(archive_path), 'manifest_sha256': sha(output / 'manifest.json'),
                      'original_files_byte_identical_except_build_revision': len(unchanged),
                      'source_graph_sha256': sha(output / 'model_q_all.onnx'),
                      'max_build_seconds': 7200, 'required_monitor_command': command,
                      'inference_run': False}, indent=2))


if __name__ == '__main__':
    main()
