"""Record the completed direct-image prototype and preserve prior failures."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/gate1'
OUT = Path(__file__).parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ident(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def new(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def link(label, relative):
    return f'[{label}]({(ROOT / relative).as_posix()})'


def main():
    full = BASE / '20261023_body32_head16_full_v1'
    audit_dir = BASE / '20261023_body32_head16_full_audit_v1'
    summary, receipt, audit = (read(path) for path in (full/'summary.json', full/'receipt.json', audit_dir/'recomputed_summary.json'))
    assert receipt['status'] == 'COMPLETE_BODY32_HEAD16_DIAGNOSTIC_ONLY' and receipt['phase'] == 'COMPLETE'
    assert receipt['images_completed'] == summary['images'] == audit['images_verified'] == 109
    assert audit['status'] == 'PASS_SAVED_BODY32_HEAD16_OUTPUTS_ONLY'
    assert summary['endpoint_all_exact'] and summary['reference_features_and_inputs_all_exact'] and summary['model_state_unchanged']
    assert summary['fresh_mixed_forwards'] == 109 and summary['checkpoint_loads'] == 1
    actual, old = audit['comparison'], audit['historical_full_model_comparison_unchanged']
    assert actual['pass_images'] == summary['diagnostic_guard_pass_images']
    assert actual['fail_images'] == summary['diagnostic_guard_fail_images']
    for section in ('boxes', 'scores'):
        assert all(actual[section][key] == value for key, value in summary['comparison'][section].items())
    assert old['fail_images'] == 109 and old['boxes']['violations'] == 10849 and old['scores']['violations'] == 0
    assert summary['original_guard_replaced'] is False and summary['scientific_gate_pass'] is False
    pilot = read(BASE/'20261023_body32_head16_pilot_audit_v1/recomputed_summary.json')
    assert pilot['status'] == audit['status'] and pilot['images_verified'] == 3
    check_pass = actual['fail_images'] == 0
    prep = BASE / '20261023_body32_head16_prototype_preparation'
    contract = read(prep/'full109_protocol.json')
    now = datetime.now(timezone.utc).isoformat()
    evidence_paths = [prep/'protocol_preparation_receipt.json', prep/'pilot3_protocol.json', prep/'full109_protocol.json',
        prep/'independent_review/manifest.json', BASE/'20261023_body32_head16_pilot_v1/manifest.json',
        BASE/'20261023_body32_head16_pilot_audit_v1/manifest.json', full/'manifest.json', full/'summary.json',
        full/'model_provenance.json', audit_dir/'manifest.json', audit_dir/'recomputed_summary.json',
        OUT/'decision.json', BASE/'20261023_upstream_final_review/macro_review.json']
    decision = {'schema': 'rampq.gate1.direct_body32_head16_prototype_decision/v1',
        'at_utc': now, 'client_date': '2026-10-23', 'status': 'COMPLETED_DIRECT_IMAGE_DEBUG_PROTOTYPE_ONLY',
        'body32_head16_decode32_prototype_implemented': True, 'prototype_fixed_debug_numeric_check_pass': check_pass,
        'prototype_check_name': contract['guard_name'], 'debug_images': 109,
        'checkpoint_loads_per_attempt': 1, 'fresh_mixed_forwards': 109, 'saved_features_as_inference_inputs': False,
        'precision_contract': contract['precision_contract'], 'reference_endpoint_bytes_all_exact': True,
        'reference_features_inputs_and_casts_exact': True, 'checkpoint_selection_changed': False,
        'best_checkpoint_sha256': contract['checkpoint']['sha256'], 'comparison': actual,
        'historical_full_model_comparison_unchanged': old, 'original_gate1_dtype_guard': 'FAIL_FIXED_TOLERANCES',
        'old_gate05_numeric_guard': 'FAILED_UNCHANGED', 'old_guard_replaced': False, 'tolerances_changed': False,
        'prototype_saved_output_audit': audit['status'], 'prototype_synthetic_cpu_tests': 30,
        'head_endpoint_is_new_independent_effect_evidence': False, 'unique_layer_cause_identified': False,
        'minimal_or_optimal_partition_proven': False, 'scientific_precision_module_selected': False,
        'accepted_float_baseline_ready': False, 'gate1_scientific_pass': False,
        'new_precision_model_accepted_or_deployed': False, 'official_AP_implemented': False,
        'official_oracle_verified': False, 'selection_detector_inference': False, 'holdout_opened_or_listed': False,
        'new_NMS_or_target_scoring': False, 'quantization_matrix_executed': False,
        'latency_FPS_memory_or_energy_measured': False, 'TensorRT_or_Orin_prototype_executed': False,
        'evidence': [ident(path) for path in evidence_paths],
        'limits': [
            'Fixed reused train-derived engineering debug set only; pilot3 is included in full109, not 112 independent samples.',
            'Fresh image-to-output implementation consistency with an already diagnosed endpoint; not independent discovery or generalization evidence.',
            'Only body/head parameter, floating-buffer and observed output dtypes are asserted, not GPU kernel internal accumulation precision.',
            'The saved checkpoint was FP16 EMA. Lifting the body to FP32 does not recover original pre-save full-precision EMA.',
            'The independent auditor recomputes saved arrays and reference feature/cast statistics; it does not independently rerun convolution, BatchNorm or pixels.',
            'FP64 NumPy decoding is an approximate sanity check, separate from the unchanged FP32-reference raw tolerances.',
            'The verification runner intentionally requires historical reference identities for acceptance checks; it is not a self-contained deployment package.',
            'No NMS, task AP, sealed-data scoring, export, TensorRT, Orin, performance or energy qualification.'
        ]}
    new(OUT/'prototype_decision.json', decision)
    state = dict(read(BASE/'current_execution_state_v7.json'))
    state.update(schema='rampq.gate1.execution_state/v8', as_of_utc=now,
        status='COMPLETED_DIRECT_MIXED_PROTOTYPE_DEBUG_CHECK_ORIGINAL_GUARD_FAILED',
        phase='MIXED_DEBUG_PROTOTYPE_VERIFIED_RESEARCH_ACCEPTANCE_PENDING',
        previous_state_snapshot='artifacts/gate1/current_execution_state_v7.json',
        body32_head16_decode32_prototype_implemented=True, prototype_debug_images=109,
        prototype_fixed_debug_numeric_check_pass=check_pass, prototype_check_name=contract['guard_name'],
        prototype_saved_output_audit=audit['status'], prototype_endpoint_bytes_all_exact=True,
        prototype_fresh_mixed_forwards=109, prototype_checkpoint_loads_per_attempt=1,
        prototype_saved_features_as_inference_inputs=False, prototype_comparison=actual,
        same_backend_dtype_guard='FAIL_FIXED_TOLERANCES', accepted_float_baseline_ready=False,
        gate1_scientific_pass=False, new_precision_model_deployed=False,
        report='47_Gate1_混合精度独立推理原型验证.md',
        decision='artifacts/gate1/20261023_upstream_diagnosis_delivery/prototype_decision.json',
        next_steps=[
            'Resolve the reference precision and complete ignore-negative loss semantics before scientific baseline acceptance.',
            'Run the frozen synthetic official-evaluator fixtures with a real native MATLAB/Octave runtime and compare semantics before implementing or claiming official AP.',
            'Freeze the purpose, training/reference provenance, candidate budget and independent validation contract before any precision selection; do not optimize repeatedly on these109 debug images.',
            'If advancing this prototype, separately freeze export/backend correctness and board qualification before performance or deployment claims.',
            'Keep Selection metadata-only and Holdout opaque until all candidate, budget and statistical choices are locked.'],
        note='Fresh direct-image mixed prototype and saved-array audit completed on reused debug only. Historical full-FP16 guard and Gate0.5 guard remain failed; Gate1 scientific acceptance is pending.')
    new(BASE/'current_execution_state_v8.json', state)
    verdict = '109/109图满足原固定数值门槛' if check_pass else f"仍有{actual['fail_images']}/109图未满足原固定数值门槛"
    report = f'''# Gate1 混合精度独立推理原型验证

**主体FP32、检测塔FP16、解码FP32的直接读图原型已经实际跑完；{verdict}，独立保存数组复算通过。** 每张图的logits和解码输出均与上一轮已诊断的检测头交叉端点逐字节一致。现在验证的是一条可独立计算的工程路径，原全FP16失败记录仍然保留，Gate1科学验收尚未通过。

{link('本轮分项结论', 'artifacts/gate1/20261023_upstream_diagnosis_delivery/prototype_decision.json')}；{link('状态快照v8', 'artifacts/gate1/current_execution_state_v8.json')}；{link('前置五边界诊断报告46', '46_Gate1_上游边界精度对照与缓解路径.md')}。

## 实际完成了什么

新原型每次尝试只从CPU反序列化一次冻结best权重，每图执行一次新图像到输出的混合前向。模块0至22，包括主干和后续特征融合，参数、浮点buffer及观测到的顶层输出均为FP32；Detect的cv2/cv3两组卷积塔参数、浮点buffer和logits为FP16；DFL、锚点/步长几何和sigmoid的观测张量为FP32。没有调用融合，也不声称CUDA卷积或BN内核内部每一步采用某种累加精度。

输入来自重新读取的同一批109张冻结工程debug图片，而非保存的特征。运行前仅对历史文件字节做完整性哈希；每张图的新前向结束后，才加载旧数值数组作比较。新归一化输入、三尺度FP32特征及其FP16转换都与冻结参考精确对应，模型状态前后不变。

源代码和精度合同先冻结，12项runner、18项auditor合成CPU测试全部通过；按预先固定的3张先导图执行并完成独立复算，再扩展为原顺序全部109张。扩展条件是身份、实现和复算一致，未用有利数值计数挑图。3张先导图包含于109张，不能相加为112个独立样本。

{link('已冻结精度合同', 'artifacts/gate1/20261023_body32_head16_prototype_preparation/full109_protocol.json')}；{link('原型源代码', 'scripts/probe_gate1_body32_head16.py')}；{link('全量执行回执', 'artifacts/gate1/20261023_body32_head16_full_v1/receipt.json')}。

## 固定门槛的实际结果

比较全部8400候选位置的4个框量和10类得分：框容差保持 `0.5 + 0.005×abs(FP32参考)`，得分容差保持 `0.002 + 0.02×abs(FP32参考)`。未按置信度屏蔽候选，也未放宽门槛。

| 检查 | 通过图/109 | 框元素超差 | 得分元素超差 | 最大框绝对差 |
|---|---:|---:|---:|---:|
| 原生全FP32对全FP16（历史原检查） | {old['pass_images']} | {old['boxes']['violations']} | {old['scores']['violations']} | {old['boxes']['max_absolute_error']:.6f} px |
| 新直接读图混合原型对历史FP32参考 | {actual['pass_images']} | {actual['boxes']['violations']} | {actual['scores']['violations']} | {actual['boxes']['max_absolute_error']:.6f} px |

新原型共检查{actual['boxes']['elements']}个框元素、{actual['scores']['elements']}个得分元素；最大得分绝对差为{actual['scores']['max_absolute_error']:.9f}。最大逐元素“误差/容差”分别为框{actual['boxes']['max_error_over_tolerance']:.6f}、得分{actual['scores']['max_error_over_tolerance']:.6f}。

工程检查单独命名为 `{contract['guard_name']}`。上一轮“仅主干FP32、后段融合和检测塔FP16”仍有6张图、7个框元素超差；本原型连同后续特征融合一起保留FP32，因此对应的是模块22之后的低误差端点，并未修正或重判模块10路径。

独立auditor核对保存数组身份、输入及参考特征/转换统计、端点字节、精度观测报告、固定容差和汇总；用NumPy FP64作近似解码合理性检查，不把它当作CUDA逐字节等价。它不重新执行卷积、BN或图片解码。新特征仅以哈希和统计去重记录，独立审计对照的是已保存参考特征；实际fresh计算来自经审查runner。

{link('完整原型结果', 'artifacts/gate1/20261023_body32_head16_full_v1/summary.json')}；{link('独立保存输出复算', 'artifacts/gate1/20261023_body32_head16_full_audit_v1/recomputed_summary.json')}。

## 结论适用范围与后续门槛

这轮把已知低误差交叉端点实现成了直接读图的推理模块，支持实现一致性及109张复用debug上的数值资格。它复现的是已有端点，不增加独立泛化证据，也没有证明最少FP32层数或最优精度配置。验证runner还依赖历史参考做验收检查，尚不是可单独部署的离线包。

冻结权重原本保存为FP16 EMA，提升主体计算类型不会恢复保存前的原始FP32训练状态。训练损失的完整ignore-negative语义、独立数据合同和科学参考仍需解决。官方AP未实现或验证；有限位置搜索没有找到可用的原生MATLAB/Octave运行时，合成fixture仍未完成原生oracle比对。Selection仍仅元数据，Holdout仍为不透明封存包。

本轮没有新增NMS、目标匹配或任务AP，也没有INT8、导出、TensorRT、Orin推理、速度、显存或功耗测量。下一步应先补科学参考、损失与官方评测语义，再预先冻结独立验证及候选预算；若推进这个原型，另立后端/板端资格合同。原全FP16检查维持 `FAIL_FIXED_TOLERANCES`，旧Gate0.5数值记录维持 `FAILED_UNCHANGED`。
'''
    report_path = ROOT/'47_Gate1_混合精度独立推理原型验证.md'
    assert not report_path.exists()
    report_path.write_text(report, encoding='utf-8')
    pointer = {'schema': 'rampq.gate1.mutable_latest_state_pointer/v1',
        'snapshot': 'artifacts/gate1/current_execution_state_v8.json',
        'snapshot_sha256': sha(BASE/'current_execution_state_v8.json'),
        'note': 'This pointer may be updated on later work. Referenced timestamped snapshots are preserved.'}
    (BASE/'latest_execution_state.json').write_text(json.dumps(pointer, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'report': ident(report_path), 'decision': ident(OUT/'prototype_decision.json'),
        'state': ident(BASE/'current_execution_state_v8.json'), 'prototype_check_pass': check_pass}, ensure_ascii=False))


if __name__ == '__main__':
    main()
