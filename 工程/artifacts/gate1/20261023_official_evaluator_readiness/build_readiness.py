"""Prepare and seal this source-only readiness package; no real dataset access."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = ROOT / 'artifacts/gate1/20261022_split_refinement/evaluation_sources'


def identity(path, base=ROOT):
    data = path.read_bytes()
    return {'path': path.relative_to(base).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    source_manifest = json.loads((SOURCES / 'source_manifest.json').read_text(encoding='utf-8'))
    official = []
    for row in source_manifest['files']:
        actual = identity(SOURCES / row['path'])
        assert actual['sha256'] == row['sha256'] and actual['bytes'] == row['bytes']
        official.append(actual)
    if not args.final:
        fixture = json.loads((HERE / 'synthetic_fixtures.json').read_text(encoding='utf-8'))
        write(HERE / 'fixture_contract.json', {
            'schema': 'rampq.gate1.synthetic_official_oracle_preparation/v1',
            'client_date': '2026-10-23',
            'status': 'PREPARED_NOT_NATIVE_ORACLE_VERIFIED',
            'official_source_commit': source_manifest['commit'],
            'data_role': 'synthetic', 'fixture_count': len(fixture['cases']),
            'kinds': {kind: sum(case['kind'] == kind for case in fixture['cases']) for kind in ('aggregate', 'region', 'match')},
            'bound_local_files': [identity(HERE / name, HERE) for name in (
                'prepare_synthetic_fixtures.py', 'synthetic_fixtures.json',
                'run_synthetic_oracle.m', 'launch_synthetic_oracle.py')],
            'oracle_output_expectation': 'No handwritten expected metric values. Native source outputs are the future reference; empty dimensions and NaN are preserved in MAT.',
            'source_functions_called': ['dropObjectsInIgr', 'createIntImg', 'evalRes', 'compOas', 'calcAccuracy', 'VOCap'],
            'upstream_source_modified': False,
            'source_entrypoints_never_called': ['evalDET', 'saveAnnoRes', 'displayImage', 'findImageList'],
            'native_dependencies': ['Existing MATLAB/Octave executable on PATH', 'jsondecode/jsonencode', 'mean2 available in the native session'],
            'no_install_download_or_real_data_access': True,
            'official_AP_equivalence_certified': False,
        })
        return
    regression = ET.parse(HERE / 'synthetic_regression.xml').getroot()
    suite = regression.find('testsuite') if regression.tag != 'testsuite' else regression
    assert suite is not None
    regression_counts = {name: int(suite.attrib[name]) for name in ('tests', 'failures', 'errors', 'skipped')}
    assert regression_counts == {'tests': 37, 'failures': 0, 'errors': 0, 'skipped': 0}
    receipt = json.loads((HERE / 'execution_01/launcher_receipt.json').read_text(encoding='utf-8'))
    assert receipt['status'] == 'NOT_EXECUTED_NO_NATIVE_RUNTIME_ON_PATH'
    assert receipt['official_oracle_executed'] is False
    policy = json.loads((SOURCES / 'evaluation_policy.json').read_text(encoding='utf-8'))
    for rel, prior in policy['implementation'].items():
        assert identity(ROOT / rel)['sha256'] == prior['sha256']
    gaps = [
        {'id': 'native_oracle', 'status': 'MISSING_RUNTIME_IN_CHECKED_LOCATIONS',
         'requirement': 'Execute pinned original MATLAB functions on the prepared synthetic cases; record runtime version, resolved function paths, native MAT/JSON outputs, and errors. Runtime search was limited to PATH and three conventional roots.'},
        {'id': 'full_python_ap', 'status': 'NOT_IMPLEMENTED',
         'requirement': 'Current Python module implements region filtering and single-image matching only. A separate immutable full AP/AR implementation or direct official runtime evaluation path is required.'},
        {'id': 'aggregation_semantics', 'status': 'SOURCE_REVIEWED_NOT_ORACLE_VERIFIED',
         'requirement': 'Ten IoU thresholds 0.50:0.05:0.95; AP maxDets=500; AR caps 1/10/100/500; VOC precision-envelope integral; duplicate evalClass image-frequency weighting; recall denominator numel(gtMatch) including surviving score=0 GT; empty evalClass and empty detections.'},
        {'id': 'adapter_order', 'status': 'RESEARCH_CONTRACT_NOT_FROZEN_FOR_OFFICIAL_AP',
         'requirement': 'Freeze submitted prediction order, prefilter/cap/class/sort sequence, score ties and image order; official evaluator performs no NMS. Existing project matching clips boxes and uses IoU-first; official-style matching keeps raw boxes and uses score-first.'},
        {'id': 'cross_version_ties', 'status': 'UNVERIFIED',
         'requirement': 'Native sort tie behavior must be observed with the exact chosen runtime and compared at intermediate evalRes matrices; Python stable sorting alone is not proof.'},
        {'id': 'oracle_to_port_comparison', 'status': 'NOT_RUN',
         'requirement': 'Compare native prefilter matrices, per-class matcher states, final seven AP/AR metrics, and NaN/empty states. Preserve source quirks or explicitly name a different metric; do not silently repair official code.'},
        {'id': 'research_acceptance', 'status': 'NOT_CERTIFIED',
         'requirement': 'Independent numeric qualification, real data scene-independence, loss-ignore limitations and research selection contract remain separate. Synthetic oracle preparation cannot pass Gate1 or authorize Holdout access.'},
    ]
    inspected = [SOURCES / 'source_manifest.json', SOURCES / 'OFFICIAL_SOURCE_REVIEW.md',
                 SOURCES / 'evaluation_policy.json', ROOT / 'scripts/gate1_ignore_aware_evaluation.py',
                 ROOT / 'tests/test_gate1_ignore_evaluation.py']
    readiness = {
        'schema': 'rampq.gate1.official_evaluator_readiness/v1', 'client_date': '2026-10-23',
        'status': 'OFFLINE_SYNTHETIC_ORACLE_PREPARATION_COMPLETE_NATIVE_EXECUTION_MISSING',
        'official_source_repository': source_manifest['repository'], 'official_commit': source_manifest['commit'],
        'verified_official_file_count': len(official), 'verified_official_files': official,
        'inspected_local_evidence': [identity(path) for path in inspected],
        'python_semantics_regression': regression_counts,
        'runtime_probe': identity(HERE / 'runtime_probe.json'),
        'execution_receipt': identity(HERE / 'execution_01/launcher_receipt.json'),
        'fixture_contract': identity(HERE / 'fixture_contract.json'),
        'fixture_count': 20, 'oracle_expected_outputs_handwritten': False,
        'official_oracle_executed': False, 'full_official_AP_implemented_in_python': False,
        'oracle_equivalence_verified': False, 'gate1_scientific_pass': False,
        'real_image_or_annotation_read': False, 'model_executed': False,
        'selection_output_read': False, 'holdout_zip_opened_or_listed': False,
        'download_or_install_performed': False, 'old_sources_or_documents_modified': False,
        'readiness_gaps': gaps,
        'minimal_executable_nextstep': 'Run launch_synthetic_oracle.py with a new output child after a native runtime is already available on PATH. It validates the frozen source and fixture hashes before synthetic-only execution; current attempt records unavailability.'}
    write(HERE / 'readiness.json', readiness)
    (HERE / 'README.md').write_text('''# Gate1 官方评价器离线就绪审计

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
& 'C:\\Users\\lenovo\\Desktop\\视觉边缘研究\\05_水下小目标检测_工程\\检测工程\\.venv\\Scripts\\python.exe' -B 'artifacts/gate1/20261023_official_evaluator_readiness/launch_synthetic_oracle.py' --output 'artifacts/gate1/20261023_official_evaluator_readiness/execution_02'
```

输出目录必须是本就绪包内一个新子目录；已有结果不会覆盖。启动器先检查源码和合成输入的冻结 SHA，再调用 [run_synthetic_oracle.m](run_synthetic_oracle.m)。原官方 `.m` 文件不改动，也不替换 `mean2`；缺依赖会保留失败，不能假装运行成功。入口只把合成矩阵送入 `dropObjectsInIgr`、`evalRes` 和 `calcAccuracy`，从不调用带数据路径或 `imread` 的 `evalDET`、`saveAnnoRes`。

成功时保存原生 `.mat` 与 JSON、中间过滤/匹配矩阵及七项 AP/AR 输出，另存运行版本和函数解析路径。MAT 保留空矩阵形状与 NaN，JSON 可能将非有限值转为 null，因此不能仅用 JSON 判断空输入语义。个别用例出错也会写入输出，然后以失败结束。

## 后续验收

原生输出产生后，才能比较过滤矩阵、每类匹配状态、排序/截断、空值处理和最终聚合。现阶段没有完整 Python AP 实现，因而即便将来原生合成用例全部可执行，也只是取得参考结果；仍须另行完成实现和逐项比较。项目 IoU 优先并裁框的逐目标指标仍须单独命名。

本轮未读真实图像/注释、Selection 检测输出或 Holdout ZIP，未运行模型，未改动浮点数值阈值，也未替代既有训练监控、数值资格检查或研究验收。
''', encoding='utf-8')
    files = [identity(path, HERE) for path in sorted(HERE.rglob('*')) if path.is_file() and path.name != 'manifest.json']
    write(HERE / 'manifest.json', {'schema': 'rampq.gate1.official_evaluator_readiness_manifest/v1',
                                  'client_date': '2026-10-23', 'file_count': len(files), 'files': files})
    print(json.dumps({'status': readiness['status'], 'file_count': len(files), 'official_source_files': len(official), 'synthetic_tests': regression_counts}))


if __name__ == '__main__':
    main()
