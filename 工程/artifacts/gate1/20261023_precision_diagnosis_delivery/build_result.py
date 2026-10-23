"""Publish bounded layered-precision results only after both full audits finish."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
BASE = ROOT / 'artifacts/gate1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ident(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def new(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def link(label, relative):
    return f'[{label}]({(ROOT / relative).as_posix()})'


def compare_summary(summary, audited):
    assert summary['images'] == audited['images_verified'] == 109
    for name, value in summary['comparisons'].items():
        actual = audited['comparisons'][name]
        assert actual['pass_images'] == value['diagnostic_guard_pass_images']
        assert actual['fail_images'] == value['diagnostic_guard_fail_images']
        for field in ('boxes', 'scores'):
            assert all(actual[field][key] == expected for key, expected in value[field].items())


def main():
    decode = read(BASE / '20261023_decode_precision_full_v1/summary.json')
    head = read(BASE / '20261023_head_precision_full_v1/summary.json')
    da = read(BASE / '20261023_decode_precision_full_audit_v1/recomputed_summary.json')
    ha = read(BASE / '20261023_head_precision_full_audit_v1/recomputed_summary.json')
    assert da['status'] == 'PASS_SAVED_ARRAY_CONSISTENCY_ONLY'
    assert ha['status'] == 'PASS_SAVED_HEAD_PROBE_ARRAYS_ONLY'
    compare_summary(decode, da); compare_summary(head, ha)
    assert decode['historical_bytes_all_equal'] and decode['native_reconstruction_all_exact']
    assert head['historical_bytes_all_equal'] and head['native_head_logits_all_exact']
    assert head['reference_decode_captures_all_exact'] and head['features_and_head_states_unchanged']
    assert head['reference_unified_decode_all_exact']
    assert decode['comparisons']['original_fp32_vs_original_fp16']['boxes']['violations'] == 10849
    assert decode['comparisons']['original_fp32_vs_fp16_logits_decode_fp32']['boxes']['violations'] == 8992
    assert head['comparisons']['native_heads_both_decode_fp32'] == decode['comparisons']['original_fp32_vs_fp16_logits_decode_fp32']
    readiness = read(BASE / '20261023_official_evaluator_readiness/readiness.json')
    assert not readiness['official_oracle_executed'] and not readiness['oracle_equivalence_verified']
    assert readiness['fixture_count'] == 20 and readiness['python_semantics_regression']['tests'] == 37
    trace = read(BASE / '20261023_saved_decode_trace/trace.json')
    now = datetime.now(timezone.utc).isoformat()
    conclusion = ('UPSTREAM_FEATURE_DIFFERENCES_REPRODUCED_IN_FIXED_DEBUG_CROSSES'
                  if all(head['comparisons'][n]['diagnostic_guard_fail_images'] == 0 for n in (
                      'same_fp32_features_head_fp32_vs_fp16', 'same_fp16_features_head_fp32_vs_fp16'))
                  and head['comparisons']['same_fp32_head_features_fp32_vs_fp16']['diagnostic_guard_fail_images'] > 0
                  else 'LAYERED_INTERVENTIONS_COMPLETE_REQUIRE_REVIEW')
    evidence_names = [
        '20261023_decode_probe_preparation/preparation_receipt.json',
        '20261023_decode_probe_preparation/head_preparation_receipt.json',
        '20261023_decode_precision_pilot_v1/manifest.json',
        '20261023_decode_precision_pilot_audit_v2/manifest.json',
        '20261023_decode_precision_full_v1/manifest.json',
        '20261023_decode_precision_full_audit_v1/manifest.json',
        '20261023_decode_precision_full_v1/summary.json',
        '20261023_head_precision_pilot_v1/manifest.json',
        '20261023_head_precision_pilot_audit_v1/manifest.json',
        '20261023_head_precision_full_v1/manifest.json',
        '20261023_head_precision_full_audit_v1/manifest.json',
        '20261023_head_precision_full_v1/summary.json',
        '20261023_saved_decode_trace/trace.json',
        '20261023_official_evaluator_readiness/manifest.json',
        '20261023_official_evaluator_readiness/readiness.json',
    ]
    decision = {
        'schema': 'rampq.gate1.layered_precision_diagnosis_decision/v1',
        'as_of_utc': now, 'client_date': '2026-10-23', 'status': 'COMPLETED_LAYERED_ENGINEERING_DIAGNOSIS_ONLY',
        'diagnostic_conclusion': conclusion,
        'candidate_training_epochs_complete': 30, 'checkpoint_selection_changed': False,
        'best_checkpoint_sha256': 'cddd14b3ebba294e4c3200bfb50fc1db1a9459c425d8e51d583c0881d71cfcfa',
        'decode_pilot_images': 3, 'decode_full_images': 109,
        'head_pilot_images': 3, 'head_full_images': 109,
        'image_cohort_reused_not_independent': True,
        'historical_raw_byte_reproduction_all_exact': True,
        'native_decode_and_head_logit_reconstruction_all_exact': True,
        'observed_reg_max': 16, 'observed_head_feature_shapes': [[1,64,80,80], [1,128,40,40], [1,256,20,20]],
        'decode_comparisons': da['comparisons'], 'head_comparisons': ha['comparisons'],
        'head_comparison_aliases_not_independent': {
            'reference_fp32_vs_fp32_features_fp16_head': 'same_fp32_features_head_fp32_vs_fp16',
            'reference_fp32_vs_fp16_features_fp32_head': 'same_fp32_head_features_fp32_vs_fp16'},
        'audits': {'decode': da['status'], 'head': ha['status']},
        'official_evaluator_preparation': {'status': readiness['status'], 'fixture_count': 20,
            'existing_semantics_tests_passed': 37, 'official_oracle_executed': False, 'oracle_equivalence_verified': False},
        'original_gate1_dtype_guard': 'FAIL_FIXED_TOLERANCES', 'old_gate05_numeric_guard': 'FAILED_UNCHANGED',
        'new_precision_candidate_accepted': False, 'accepted_float_baseline_ready': False, 'gate1_scientific_pass': False,
        'selection_detector_inference': False, 'holdout_opened_or_listed': False,
        'official_AP_implemented': False, 'official_oracle_verified': False,
        'new_quantization_matrix_executed': False,
        'evidence': [ident(BASE / name) for name in evidence_names],
        'limits': [
            'These two post-hoc crosses locate a broad pre-head feature-source sensitivity; they do not identify a unique layer or arithmetic operation.',
            'Inputs, backbone and neck are bundled in the captured-feature source; feature casting is explicitly part of a receiving-head precision change.',
            'Tolerance PASS means within fixed bounds, not bitwise equality, identical detections or no arithmetic effect.',
            'Counts and differences of counts are nonadditive diagnostic descriptors, not causal fractions.',
            'Saved-array auditors do not regenerate convolution/BN outputs from saved features; state immutability is a reviewed-runner assertion.',
            'Saved FP16 EMA lifted to FP32 is not recovered pre-save FP32 training EMA.',
            'No NMS, new target scoring, scientific module selection, official AP, TensorRT/Orin or performance evidence in these precision probes.',
        ]
    }
    new(OUT / 'decision.json', decision)
    state = read(BASE / 'current_execution_state_v5.json')
    state.update(schema='rampq.gate1.execution_state/v6', as_of_utc=now,
        status='COMPLETED_LAYERED_PRECISION_DIAGNOSIS_ORIGINAL_GUARD_FAILED',
        phase='PRE_HEAD_PRECISION_SENSITIVITY_DIAGNOSED_RESEARCH_ACCEPTANCE_PENDING',
        previous_state_snapshot='artifacts/gate1/current_execution_state_v5.json',
        decode_precision_probe_completed=True, decode_precision_probe_images=109,
        head_precision_probe_completed=True, head_precision_probe_images=109,
        precision_probe_history_and_native_exact=True,
        decode_saved_array_audit=da['status'], head_saved_array_audit=ha['status'],
        observed_reg_max=16, precision_diagnostic_conclusion=conclusion,
        official_evaluator_synthetic_fixtures_prepared=20, official_native_runtime_found_in_checked_locations=False,
        report='45_Gate1_精度误差分层诊断与评测准备.md',
        decision='artifacts/gate1/20261023_precision_diagnosis_delivery/decision.json',
        note='Immutable diagnostic snapshot; no precision intervention, synthetic fixture or consistency PASS replaces the historical failed guard.',
        next_steps=[
            'Freeze a bounded upstream block-boundary precision probe on debug only; retain all current failures and do not launch an unbounded layer sweep',
            'Run the fixed official-source synthetic fixtures in a native MATLAB/Octave runtime, then verify the complete evaluation path before scientific AP claims',
            'Resolve scientific reference precision/loss-ignore/data-independence limits and freeze the research budget before Selection ranking',
            'Keep Holdout opaque until all selections and analysis are frozen'])
    new(BASE / 'current_execution_state_v6.json', state)
    rows = []
    decode_labels = [
        ('original_fp32_vs_original_fp16', '原FP32全模型 → 原FP16全模型'),
        ('original_fp32_vs_fp16_logits_decode_fp32', '原FP32 → FP16网络logits、FP32解码'),
        ('same_fp32_logits_decode_fp32_vs_fp16', '同FP32来源logits，解码32 → 16'),
        ('same_fp16_logits_decode_fp32_vs_fp16', '同FP16来源logits，解码32 → 16')]
    for key, label in decode_labels:
        r = da['comparisons'][key]
        rows.append(f"| {label} | {r['fail_images']}/109 | {r['boxes']['violations']} | {r['boxes']['max_absolute_error']:.6f} |")
    head_rows = []
    head_labels = [
        ('same_fp32_features_head_fp32_vs_fp16', '固定FP32来源特征，head 32 → 16'),
        ('same_fp16_features_head_fp32_vs_fp16', '固定FP16来源特征，head 32 → 16'),
        ('same_fp32_head_features_fp32_vs_fp16', '固定FP32 head，特征来源32 → 16'),
        ('same_fp16_head_features_fp32_vs_fp16', '固定FP16 head，特征来源32 → 16')]
    for key, label in head_labels:
        r = ha['comparisons'][key]
        head_rows.append(f"| {label} | {r['fail_images']}/109 | {r['boxes']['violations']} | {r['boxes']['max_absolute_error']:.6f} | {r['scores']['violations']} |")
    head_zero = all(ha['comparisons'][key]['fail_images'] == 0 for key in ('same_fp32_features_head_fp32_vs_fp16', 'same_fp16_features_head_fp32_vs_fp16'))
    head_sentence = ('固定同一来源特征、仅切换head计算精度的两组对照都在原容差内；固定head精度、切换特征来源则重现超差。'
                     if head_zero else '检测头与特征来源对照结果见下表，不能把有超差的比较写为通过。')
    report = f'''# Gate1 精度误差分层诊断与评测准备

本轮已完成两套 **3张先导 → 全109张 debug** 的真实精度对照，并分别完成保存数组的独立复算。明确的新发现是：**只把解码升到FP32不能解决超差，进入检测头前的特征来源差异能够重现超差**。{head_sentence}原数值门槛及失败结果不变，**Gate1科学验收仍未通过**。

{link('分项结论及证据身份', 'artifacts/gate1/20261023_precision_diagnosis_delivery/decision.json')}；{link('新状态快照', 'artifacts/gate1/current_execution_state_v6.json')}。30轮训练和权重不变，接续{link('报告44', '44_Gate1_首轮训练与浮点资格检查结果.md')}。

## 运行范围与复现约束

沿用第29轮 best（SHA `cddd14b3ebba294e4c3200bfb50fc1db1a9459c425d8e51d583c0881d71cfcfa`）、两份独立加载的未融合模型、固定640输入及原109张debug。先导图为两张事后挑选的高误差样例和一张固定普通对照；扩大至109张只取决于精确复现约束，与结果有利无关。协议在各新干预执行前冻结。

两个完整对照均逐图精确重现旧FP32/FP16 raw字节及归一化输入；原生DFL/几何解码、原生检测头logits重建全部精确一致。499个保存状态值提升至FP32后相等，head子层保持eval、干预前后状态和特征不变。实测 `head.reg_max=16`，回归logits为 `[1,64,8400]`，三尺度head输入为 `[1,64,80,80]`、`[1,128,40,40]`、`[1,256,20,20]`。本轮把昨日仅源码默认预期的DFL维度补成了实际运行记录。

检查仍用全候选位置的非对称固定门槛：框 `0.5 + 0.005×abs(参考)`，得分 `0.002 + 0.02×abs(参考)`；坐标均为640输入像素。参考是保存FP16 EMA权重提升到FP32，不能恢复保存前完整FP32训练EMA。解码与head对照程序分别通过15、11项CPU合成测试，两个独立复算器分别通过22、15项合成测试；合成测试与真实109图证据分开保存。

## 第一层：网络logits来源 × 解码精度

| 对照：参考 → 比较 | 超差图数 | 框元素超差 | 最大框绝对差/像素 |
|---|---:|---:|---:|
{chr(10).join(rows)}

以上4项得分元素均无超差。同源logits解码32/16“在门槛内”不等于逐位相等或解码没有影响。两来源都经FP32解码后仍109图超差、8992个框元素超差，说明进入解码之前的差异不能由提高解码精度单独消除。

原超差集合与FP32解码干预后集合共有7808个元素；原集合独有3041个，干预后新超差1184个。因此 `10849−8992` 只能作净计数变化，不能称作解码造成的误差比例或独立可加的因果贡献。三个既定最坏位置的logits、概率和DFL距离链见{link('已保存数组追踪', 'artifacts/gate1/20261023_saved_decode_trace/trace.json')}。

{link('完整解码对照', 'artifacts/gate1/20261023_decode_precision_full_v1/summary.json')}；{link('解码数组独立复算', 'artifacts/gate1/20261023_decode_precision_full_audit_v1/recomputed_summary.json')}。

## 第二层：特征来源 × 检测头精度

从两种原生forward各捕获三尺度head输入，将来源与head计算精度交叉，**所有干预输出统一FP32解码**。跨精度接收端会显式转换特征dtype；head涵盖其卷积、BN和激活，不能据此区分其中某个算子。两项 `reference_fp32_vs_*` 名称只是已有比较的别名，不增加独立证据数量。

| 对照：参考 → 比较 | 超差图数 | 框元素超差 | 最大框绝对差/像素 | 得分元素超差 |
|---|---:|---:|---:|---:|
{chr(10).join(head_rows)}

这支持把后续排查范围收紧到**检测头之前的网络计算及输入转换**；尚不能指定主干、neck、某个BN、某层卷积或某个输入舍入是唯一原因，也不能将两类干预计数相加。某条干预在109图落入容差，只是这组复用debug上的诊断结果，**不是新部署模型、正式浮点基线或科学方法已经验收**。

{link('完整head对照', 'artifacts/gate1/20261023_head_precision_full_v1/summary.json')}；{link('head数组独立复算', 'artifacts/gate1/20261023_head_precision_full_audit_v1/recomputed_summary.json')}。审计器验证数组身份、固定门槛、原生及历史字节、各解码阶段；对非线性运算只作NumPy近似核查，没有从特征独立重跑卷积/BN。head状态不变来自已审阅runner的运行断言。

## 官方评测准备也已推进

核对本地固定commit `005445782213e20cb91bc50a597db3dd949e749a` 的11份官方源码；重跑既有37项合成语义回归全部通过。新增20项纯合成输入（11项聚合、4项忽略区域、5项匹配）及身份绑定的MATLAB/Octave入口，未来将直接调用未修改的官方函数取得输出，没有手填所谓官方预期指标。

本机PATH和已检查常见目录没有发现可用原生运行时，实际启动回执为 **NOT_EXECUTED_NO_NATIVE_RUNTIME_ON_PATH**。所以尚未运行官方oracle，完整Python AP也未实现，AP等价性未认证。准备好的入口不读取真实图像、真实标注或Holdout；本轮没有下载安装运行时。

{link('官方评测就绪说明', 'artifacts/gate1/20261023_official_evaluator_readiness/README.md')}；{link('实际未执行回执', 'artifacts/gate1/20261023_official_evaluator_readiness/execution_01/launcher_receipt.json')}。

## 保留的边界与下一步

原Gate1严格检查仍为 **FAIL_FIXED_TOLERANCES**，原Gate0.5为 **FAILED_UNCHANGED**。109张图是已复用工程debug，未做新NMS或目标评分；548张Selection未检测/排名，Holdout ZIP未列成员、解包、读图或推理。没有新量化矩阵、TensorRT/Orin精度或性能结论。

下一步可单独冻结少量上游模块边界探针，继续缩小输入/主干/neck范围；不能由本轮事后结果直接选精度模块或无限扫描。并行的评测工作需先用原生运行时执行已冻结合成用例，再核对完整AP路径。参考权重精度、ignore损失及数据独立性限制仍要处理，之后才有科学基线验收与Selection研究预算冻结。
'''
    report_path = ROOT / state['report']
    assert not report_path.exists()
    report_path.write_text(report, encoding='utf-8')
    state_path = BASE / 'current_execution_state_v6.json'
    pointer = {'schema': 'rampq.gate1.mutable_latest_state_pointer/v1', 'snapshot': state_path.relative_to(ROOT).as_posix(),
               'snapshot_sha256': sha(state_path), 'note': 'This pointer may be updated on later work. Referenced timestamped snapshots are preserved.'}
    (BASE / 'latest_execution_state.json').write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': ident(report_path), 'decision': ident(OUT / 'decision.json'), 'status': conclusion}, ensure_ascii=False))


if __name__ == '__main__':
    main()
