"""Publish a new bounded result snapshot without rewriting frozen evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
BASE = ROOT / 'artifacts/gate1/20261022_posttraining_checks_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def new_json(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def link(label, relative):
    return f'[{label}]({(ROOT / relative).as_posix()})'


def main():
    now = datetime.now(timezone.utc).isoformat()
    receipt_path = ROOT / 'artifacts/gate1/20261022_split_refinement/float_training_v2/attempt_receipt.json'
    receipt = read(receipt_path)
    training = read(BASE / '01_training_audit/audit.json')
    qualification = read(BASE / '02_float_qualification/qualification_receipt.json')
    guard = read(BASE / '02_float_qualification/raw_dtype_guard.json')
    matching = read(BASE / '02_float_qualification/matching_summary.json')['thresholds']['0.25']
    audit = read(BASE / '03_qualification_output_audit/recomputed_summary.json')
    description = read(ROOT / 'artifacts/gate1/20261022_dtype_guard_diagnostics/raw_failure_description.json')
    independent = read(ROOT / 'artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review/raw_array_diagnostics.json')
    assert receipt['status'] == 'COMPLETE_FLOAT_TRAINING_REQUIRES_SCIENTIFIC_REVIEW'
    assert training['epochs'] == 30 and training['selected_best_epoch_zero_based'] == 28
    assert qualification['images_completed'] == 109 and qualification['raw_finite_and_shape']
    assert guard['status'] == description['guard_status_unchanged'] == audit['dtype_guard_preserved'] == 'FAIL_FIXED_TOLERANCES'
    assert len(guard['guard_failed_images']) == 109
    assert audit['status'] == 'PASS_SAVED_OUTPUT_CONSISTENCY_ONLY'
    assert description['totals']['box_violations'] == 10849 and description['totals']['score_violations'] == 0
    assert matching['paired_targets'] == audit['thresholds']['0.25']['paired_targets']
    for cp in training['checkpoints'].values():
        p = Path(cp['path'])
        assert p.stat().st_size == cp['bytes'] and sha(p) == cp['sha256']

    evidence_paths = [
        receipt_path,
        BASE / '01_training_audit/audit.json',
        BASE / '02_float_qualification/manifest.json',
        BASE / '02_float_qualification/qualification_receipt.json',
        BASE / '02_float_qualification/raw_dtype_guard.json',
        BASE / '02_float_qualification/matching_summary.json',
        BASE / '03_qualification_output_audit/manifest.json',
        BASE / '03_qualification_output_audit/recomputed_summary.json',
        ROOT / 'artifacts/gate1/20261022_dtype_guard_diagnostics/raw_failure_description.json',
        ROOT / 'artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review/raw_array_diagnostics.json',
    ]
    decision = {
        'schema': 'rampq.gate1.training_and_float_qualification_decision/v1',
        'as_of_utc': now,
        'status': 'TRAINING_COMPLETE_FLOAT_DTYPE_QUALIFICATION_FAILED',
        'training_candidate_complete': True,
        'completed_training_epochs': 30,
        'best_training_epoch_one_based': 29,
        'last_training_epoch_one_based': 30,
        'training_audit': training['status'],
        'checkpoints': training['checkpoints'],
        'debug_images': 109,
        'debug_raw_finite_shape_domain': 'PASS_ENGINEERING_VALIDITY_ONLY',
        'same_backend_dtype_guard': guard['status'],
        'guard_failed_images': 109,
        'all_anchor_box_elements': description['totals']['box_elements'],
        'all_anchor_box_violations': description['totals']['box_violations'],
        'all_anchor_score_elements': description['totals']['score_elements'],
        'all_anchor_score_violations': description['totals']['score_violations'],
        'saved_output_audit': audit['status'],
        'debug_project_matching_main_threshold': .25,
        'debug_project_matching_main': matching,
        'posthoc_diagnostics_do_not_change_guard': True,
        'accepted_float_baseline_ready': False,
        'gate1_scientific_pass': False,
        'official_AP_implemented': False,
        'official_oracle_verified': False,
        'selection_detector_inference': False,
        'holdout_opened_or_listed': False,
        'old_gate05_numeric_guard': 'FAILED_UNCHANGED',
        'new_quantization_research_started': False,
        'evidence': [identity(p) for p in evidence_paths],
        'next_bounded_step': 'Freeze a separate precision probe on fixed debug images to distinguish upstream network/logit differences from DFL and box decoding arithmetic; do not alter the historical guard or tune thresholds.',
        'limits': qualification['limits'] + [
            'Saved raw outputs cannot localize an intermediate arithmetic cause.',
            'All-anchor guard failure is retained even when main-confidence retained boxes have no elementwise box violation.',
            'No scientific method or module ranking, official AP, TensorRT/Orin accuracy or performance claim follows from this delivery.'
        ],
    }
    decision_path = OUT / 'decision.json'
    new_json(decision_path, decision)
    report_relative = '44_Gate1_首轮训练与浮点资格检查结果.md'
    state = {
        'schema': 'rampq.gate1.execution_state/v5', 'as_of_utc': now,
        'status': decision['status'], 'phase': 'COMPLETED_CANDIDATE_REQUIRES_NUMERICAL_DIAGNOSIS',
        'previous_state_snapshot': 'artifacts/gate1/current_execution_state_v4.json',
        'training_started': True, 'training_complete': True,
        'completed_training_epochs': 30, 'planned_epochs': 30,
        'training_attempt': 'artifacts/gate1/20261022_split_refinement/float_training_v2',
        'training_contract': 'artifacts/gate1/20261022_split_refinement/training_contract_v3/contract.json',
        'training_receipt': receipt_path.relative_to(ROOT).as_posix(),
        'best_training_epoch_one_based': 29, 'last_training_epoch_one_based': 30,
        'best_checkpoint_sha256': training['checkpoints']['best.pt']['sha256'],
        'last_checkpoint_sha256': training['checkpoints']['last.pt']['sha256'],
        'effective_batch': 4, 'effective_amp': False, 'recovery_events_observed': 0,
        'counts': {'train': 4534, 'training_val': 414, 'calibration_pool': 689, 'debug': 109, 'quarantine': 725, 'selection': 548},
        'training_audit': training['status'], 'float_qualification_executed': True,
        'float_qualification_images': 109, 'float_qualification_raw_valid': True,
        'same_backend_dtype_guard': guard['status'], 'saved_output_audit': audit['status'],
        'accepted_float_baseline_ready': False, 'gate1_scientific_pass': False,
        'selection_detector_inference': False, 'holdout_opened_or_listed': False,
        'official_AP_implemented': False, 'official_oracle_verified': False,
        'old_gate05_numeric_guard': 'FAILED_UNCHANGED',
        'report': report_relative, 'decision': decision_path.relative_to(ROOT).as_posix(),
        'next_steps': [
            'Run a separately frozen, bounded precision probe without changing historical outputs or tolerances',
            'Validate official evaluation semantics and address the reference checkpoint precision limits before scientific baseline acceptance',
            'Freeze quantization research, candidate, budget and statistical contracts before Selection scanning',
            'Keep Holdout opaque and sealed until selections and analysis are locked'
        ],
        'note': 'Immutable completion snapshot. Dtype FAIL and independent consistency PASS are different outcomes.'
    }
    state_path = ROOT / 'artifacts/gate1/current_execution_state_v5.json'
    new_json(state_path, state)

    max_abs = max(row['boxes']['max_absolute_error'] for row in guard['guards'])
    low = description['score_buckets']['union_score_le_0.001']['box_violations']
    metrics = receipt['diagnostic_builtin_metrics']
    report = f'''# Gate1 首轮训练与浮点资格检查结果

本轮 **30/30 轮候选训练已完成**，训练记录及权重审计通过；109 张固定 debug 图的 FP32/FP16 推理与目标匹配已完成。固定全候选框数值检查为 **FAIL_FIXED_TOLERANCES**，所以 **浮点基线尚未验收，Gate1 科学验收未通过**。独立复算通过仅说明保存结果一致。

这份结果接续报告43的训练中快照。{link('当前完成快照', 'artifacts/gate1/current_execution_state_v5.json')}；{link('分项验收结论', 'artifacts/gate1/20261022_training_qualification_delivery/decision.json')}。原 Gate0.5 结论见{link('报告39', '39_Gate05_150图完整验收结果.md')}，原完整模型数值检查仍为 FAILED_UNCHANGED，本次检查不替代它。

## 已完成的训练

使用冻结的训练合同 v3，4534 张训练图、414 张 training_val，30 轮、640 输入、batch=4、AMP关闭；用时 {receipt['elapsed_seconds'] / 60:.2f} 分钟，未记录恢复或重启。best 来自第29轮，last 来自第30轮；框架最后对 best 的额外验证回调标为31，不能算作第31轮训练。

| 文件 | 来源轮次 | SHA256 |
|---|---:|---|
| best.pt | 29 | `{training['checkpoints']['best.pt']['sha256']}` |
| last.pt | 30 | `{training['checkpoints']['last.pt']['sha256']}` |

最终 best 的框架内验证代理 mAP50={metrics['metrics/mAP50(B)']:.5f}、mAP50–95={metrics['metrics/mAP50-95(B)']:.5f}，仅用于训练监控及 checkpoint 选择。普通 YOLO 损失没有完整屏蔽 ignore 区域的负样本，这些数字**不是官方 VisDrone AP**。训练标签227425条，loader去重后227422条，3条重复已审计；框架将默认 max_det 从300按 training_val 实际最大对象数升为902。两项行为都有冻结源码和日志依据。

{link('完成训练审计', 'artifacts/gate1/20261022_posttraining_checks_v1/01_training_audit/audit.json')}核查30轮连续记录、保存检查、最佳轮次、4948张训练/验证副本及标签身份、499个状态张量有限值。保存的是 FP16 EMA 权重；转换为 FP32 **不能恢复保存前的完整 FP32 EMA**。

## 109张 debug 的数值检查

两分支从同一 best 独立加载，均不融合卷积与BN，使用统一640方形预处理及共享归一化输入，比较 NMS 之前的 `[1,14,8400]` 原始输出。FP32 参考来自上述已保存 FP16 EMA 的 `.float()`；本次不涉及 TensorRT 或板端性能。

| 检查 | 结果 |
|---|---|
| 输出形状、有限数、正宽高、得分范围 | 109/109通过 |
| 框容差 `abs差 ≤ 0.5 + 0.005×abs(FP32)` | **109/109图失败** |
| 框元素超差 | **10849 / 3662400（{100 * description['box_violation_fraction']:.4f}%）** |
| 得分容差 `abs差 ≤ 0.002 + 0.02×abs(FP32)` | **0 / 9156000元素超差** |
| 最大框绝对差 | **{max_abs:.6f}像素**，在640输入坐标系中 |
| 独立保存结果复算 | PASS_SAVED_OUTPUT_CONSISTENCY_ONLY，保留数值FAIL |

{link('固定数值检查原始结果', 'artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification/raw_dtype_guard.json')}；{link('独立复算', 'artifacts/gate1/20261022_posttraining_checks_v1/03_qualification_output_audit/recomputed_summary.json')}。复算重读454个封存成员并重算数值检查、目标匹配与分区计数；复用冻结的项目匹配器，不是独立官方评测 oracle，也没有再次执行模型或NMS。

## 项目目标匹配结果

固定主阈值 conf=0.25、IoU=0.5，共9059个有效GT，统计覆盖所有有效尺寸，**不是仅小目标指标**。109张图此前用于工程调试，不是独立 Selection/Holdout。

| 分支 | 命中 | 漏检 | 项目假阳性 | 项目目标召回率 |
|---|---:|---:|---:|---:|
| FP32 | 3146 | 5913 | 1631 | 34.7279% |
| FP16 | 3143 | 5916 | 1636 | 34.6948% |

共同命中3136个，FP32独有10个，FP16独有7个，共同漏检5906个。两分支命中接近，**不抵消原始输出数值检查失败**。{link('全部固定阈值及匹配汇总', 'artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification/matching_summary.json')}。

## 误差诊断与后续入口

只分析已保存 raw 数据，没有再次加载模型或图像：10849个框超差中，宽5264、高5502，中心x/y分别66/17；{low}个来自两分支最高类别分均≤0.001的候选位置。0.25阈值下两分支各保留的4931个NMS输出位置均无框元素超差；这是**事后描述，不能据此过滤候选位置或改变通过条件**。低阈值NMS输出中仍有FP32侧4个、FP16侧5个位置超差。

独立反事实只把保存的FP32 decoded输出最终转换为FP16，109图均落在原固定容差内，框最大差0.25像素。这说明最终FP16存储舍入**单独不足以解释**实际超差；尚不能区分输入转换、网络上游累积、DFL积分或框解码各自贡献。源码未发现两分支像素不一致、融合不一致或NMS污染raw的明显问题，仍缺少中间logits证明具体原因。

{link('误差分布', 'artifacts/gate1/20261022_dtype_guard_diagnostics/raw_failure_description.json')}；{link('独立反事实诊断', 'artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review/raw_array_diagnostics.json')}；{link('训练及结果复核', 'artifacts/gate1/20261022_posttraining_final_review/debug_qualification_review.md')}。

下一项工作是单独冻结少量固定debug图的精度探针：保存检测头回归logits，分别切换DFL/框解码计算精度，先重现当前原始输出，再判断误差位于解码前还是解码内。历史109图结果和容差保持不变；该诊断不选择量化模块、不触碰Selection。官方评测语义及参考权重精度限制仍需处理，之后才能验收科学基线并冻结量化研究预算。

{link('最小对照提案', 'artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review/README.md')}尚未执行；{link('维度范围补充', 'artifacts/gate1/20261022_dtype_guard_diagnostics/independent_review/SCOPE_ADDENDUM.md')}要求下次先记录实际 `head.reg_max`，不能把源码默认16-bin/64通道写成此次独立实测。

548张Selection未做检测评分或模块排名；Holdout ZIP未列成员、解包、读图或推理。未开展新量化矩阵。此次交付支持“训练候选完成、工程输出可复算、严格数值资格失败”这三个分开的结论。
'''
    report_path = ROOT / report_relative
    assert not report_path.exists()
    report_path.write_text(report, encoding='utf-8')
    pointer = {'schema': 'rampq.gate1.mutable_latest_state_pointer/v1',
               'snapshot': state_path.relative_to(ROOT).as_posix(), 'snapshot_sha256': sha(state_path),
               'note': 'This pointer may be updated on later work. Referenced timestamped snapshots are preserved.'}
    (ROOT / 'artifacts/gate1/latest_execution_state.json').write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': report_relative, 'state': pointer, 'decision': identity(decision_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
