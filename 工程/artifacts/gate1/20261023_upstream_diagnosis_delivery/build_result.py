"""Publish completed fixed-frontier results without changing any old guard."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/gate1'
OUT = Path(__file__).parent
CUTS = [-1,4,6,10,22]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ident(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def new(path, obj):
    assert not path.exists(),path
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def link(label, relative):
    return f'[{label}]({(ROOT / relative).as_posix()})'


def key(cut, source, suffix):
    return f'reference_fp32_vs_cut_{cut}_source_{source}_suffix_{suffix}'


def main():
    probe = BASE / '20261023_upstream_frontier_full_v1'
    summary = read(probe / 'summary.json')
    receipt = read(probe / 'receipt.json')
    audit = read(BASE / '20261023_upstream_frontier_full_audit_v1/recomputed_summary.json')
    assert receipt['status']=='COMPLETE_UPSTREAM_DIAGNOSTIC_ONLY' and receipt['images_completed']==109
    assert summary['images']==audit['images_verified']==109
    assert summary['all_native_controls_exact'] and summary['historical_bytes_all_equal']
    assert audit['status']=='PASS_SAVED_UPSTREAM_FRONTIER_ARRAYS_ONLY'
    assert summary['cuts']==CUTS and not summary['scientific_gate_pass']
    for name,row in summary['comparisons'].items():
        actual = audit['comparisons'][name]
        assert row['diagnostic_guard_fail_images']==actual['fail_images']
        assert row['diagnostic_guard_pass_images']==actual['pass_images']
        for section in ('boxes','scores'):
            assert all(actual[section][k]==v for k,v in row[section].items())
    assert audit['comparisons']['original_fp32_vs_original_fp16']['boxes']['violations']==10849
    assert audit['comparisons']['native_heads_both_decode_fp32']['boxes']['violations']==8992
    forward_pass_cuts = [cut for cut in CUTS if audit['comparisons'][key(cut,'fp32','fp16')]['fail_images']==0]
    backbone_path = audit['comparisons'][key(10,'fp32','fp16')]
    input_only = audit['comparisons'][key(-1,'fp16','fp32')]
    topology = read(BASE / '20261023_upstream_probe_preparation/topology_inspection_v1/topology.json')
    assert topology['cut_frontiers']==summary['frontiers']
    now = datetime.now(timezone.utc).isoformat()
    evidence = [
        '20261023_upstream_probe_preparation/topology_inspection_v1/topology.json',
        '20261023_upstream_probe_preparation/preparation_receipt.json',
        '20261023_upstream_probe_preparation/pilot3_protocol.json',
        '20261023_upstream_probe_preparation/full109_protocol.json',
        '20261023_upstream_frontier_preparation/independent_review/manifest.json',
        '20261023_upstream_frontier_pilot_v1/manifest.json',
        '20261023_upstream_frontier_pilot_audit_v1/manifest.json',
        '20261023_upstream_frontier_full_v1/manifest.json',
        '20261023_upstream_frontier_full_v1/summary.json',
        '20261023_upstream_frontier_full_audit_v1/manifest.json',
        '20261023_upstream_frontier_full_audit_v1/recomputed_summary.json',
        '20261023_official_oracle_followup/manifest.json',
        '20261023_official_oracle_followup/runtime_search_receipt.json',
    ]
    decision = {'schema':'rampq.gate1.upstream_frontier_diagnosis_decision/v1',
        'as_of_utc':now,'client_date':'2026-10-23','status':'COMPLETE_BOUNDED_UPSTREAM_DEBUG_DIAGNOSIS_ONLY',
        'completed_training_epochs':30,'checkpoint_selection_changed':False,
        'best_checkpoint_sha256':'cddd14b3ebba294e4c3200bfb50fc1db1a9459c425d8e51d583c0881d71cfcfa',
        'pilot_images':3,'full_debug_images':109,'macro_cuts':CUTS,'dag_frontiers':summary['frontiers'],
        'all_native_historical_manual_and_same_dtype_controls_exact':True,
        'same_dtype_controls_are_not_new_independent_evidence':True,
        'input_and_head_endpoints_include_previous_aliases':True,
        'all_crossed_outputs_decode_dtype':'fp32','comparisons':audit['comparisons'],
        'fp32_prefix_fp16_suffix_cuts_within_fixed_tolerances_on_reused_debug':forward_pass_cuts,
        'backbone_fp32_suffix_fp16_debug_guard':backbone_path,
        'input_cast_only_with_fp32_network_debug_guard':input_only,
        'saved_array_audit':audit['status'],
        'unique_causal_layer_or_operation_identified':False,
        'minimal_or_optimal_precision_partition_proven':False,
        'scientific_precision_module_selected':False,
        'new_precision_model_accepted_or_deployed':False,
        'original_gate1_dtype_guard':'FAIL_FIXED_TOLERANCES','old_gate05_numeric_guard':'FAILED_UNCHANGED',
        'accepted_float_baseline_ready':False,'gate1_scientific_pass':False,
        'selection_detector_inference':False,'holdout_opened_or_listed':False,
        'new_target_scoring':False,'new_quantization_matrix_executed':False,
        'official_AP_implemented':False,'official_oracle_verified':False,
        'official_runtime_followup':'No usable runtime found in explicitly limited readable locations; access errors retained, no native oracle execution.',
        'evidence':[ident(BASE/name) for name in evidence],
        'limits':[
            'Reused train-derived engineering debug images, not independent Selection or Holdout.',
            'A complete frontier includes every crossing DAG edge. Its intervention bundles prefix arithmetic, skip paths and receiving-dtype casts.',
            'A passing prefix/suffix counterfactual is not a standalone accepted model, optimal allocation or deployment result.',
            'Differences of violation counts are nonlinear and not additive causal contributions.',
            'Only union frontiers/manual end outputs are stored. Other per-module replay assertions/hashes and state immutability are reviewed-runner observations, not independent convolution execution.',
            'Saved FP16 EMA lifted to FP32 cannot recover original pre-save full-precision training EMA.',
            'No new NMS, target scoring, AP, TensorRT/Orin accuracy, latency, FPS, memory or power measurement.'
        ]}
    new(OUT/'decision.json',decision)
    previous = read(BASE/'current_execution_state_v6.json')
    state = dict(previous)
    state.update(schema='rampq.gate1.execution_state/v7',as_of_utc=now,
        status='COMPLETED_BOUNDED_UPSTREAM_FRONTIER_DIAGNOSIS_ORIGINAL_GUARD_FAILED',
        phase='UPSTREAM_NUMERIC_MITIGATION_OBSERVED_RESEARCH_ACCEPTANCE_PENDING',
        previous_state_snapshot='artifacts/gate1/current_execution_state_v6.json',
        upstream_frontier_probe_complete=True,upstream_frontier_probe_images=109,upstream_frontier_macro_cuts=CUTS,
        upstream_frontier_saved_array_audit=audit['status'],
        fp32_prefix_fp16_suffix_debug_tolerance_pass_cuts=forward_pass_cuts,
        backbone_fp32_suffix_fp16_debug_guard_fail_images=backbone_path['fail_images'],
        input_cast_only_fp32_network_debug_guard_fail_images=input_only['fail_images'],
        accepted_float_baseline_ready=False,gate1_scientific_pass=False,
        unique_causal_layer_identified=False,new_precision_model_deployed=False,
        official_runtime_extended_search_complete=True,official_oracle_verified=False,
        report='46_Gate1_上游边界精度对照与缓解路径.md',
        decision='artifacts/gate1/20261023_upstream_diagnosis_delivery/decision.json',
        next_steps=[
            'Use a separately frozen small refinement only if needed; investigate input/early backbone and later backbone without a full layer sweep or tolerance changes',
            'If advancing a hybrid path, separately specify a standalone implementation and engineering qualification before any deployment or performance claim',
            'Complete native official-oracle comparison and resolve scientific reference/loss/data-independence requirements before accepting a baseline',
            'Freeze candidate/budget/statistical contracts before Selection ranking, and keep Holdout opaque until all choices are locked'],
        note='Immutable frontier-diagnostic snapshot. The old full-FP16 guard still fails; favorable crossed outputs are reused-debug counterfactuals only.')
    new(BASE/'current_execution_state_v7.json',state)
    rows=[]
    labels={-1:'输入转换（−1）',4:'模块4后',6:'模块6后',10:'主干末端（模块10后）',22:'特征融合末端（模块22后）'}
    for cut in CUTS:
        a,b=(audit['comparisons'][key(cut,*pair)] for pair in [('fp32','fp16'),('fp16','fp32')])
        rows.append(f"| {labels[cut]} | {summary['frontiers'][str(cut)]} | {a['fail_images']}/109 | {a['boxes']['violations']} | {b['fail_images']}/109 | {b['boxes']['violations']} |")
    if backbone_path['fail_images']==0:
        lead='**主干保持FP32，后续特征融合与检测头采用FP16、最后统一FP32解码，这条交叉路径在全部109张复用debug图上落入原固定容差**。'
    else:
        lead=f"主干FP32、后段FP16这条交叉路径仍有{backbone_path['fail_images']}/109图超差，尚未解决固定数值门槛。"
    report=f'''# Gate1 上游边界精度对照与缓解路径

本轮完成固定五个边界、两个交叉方向的 **3张先导 → 全109张图** 对照，以及保存数组独立复算。{lead}这把排查范围进一步收紧到输入/主干精度敏感性，**尚未定位唯一层或最优划分，Gate1科学验收仍未通过**。

{link('分项结论', 'artifacts/gate1/20261023_upstream_diagnosis_delivery/decision.json')}；{link('状态快照v7', 'artifacts/gate1/current_execution_state_v7.json')}；前一轮见{link('报告45', '45_Gate1_精度误差分层诊断与评测准备.md')}。训练权重和原数值失败记录保持不变。

## 本轮实际执行与校验

先在CPU核对真实checkpoint：24个顶层模块，Detect接收模块16/19/22；预先固定五个切分点−1/4/6/10/22。对每个切分点取所有跨边界DAG依赖，保留跳连；例如模块10之后的边界必须同时携带模块4、6、10的输出，不能只换模块10一个张量。

同一份已冻结best权重、两次独立加载的未融合模型、原预处理和相同运行控制；每图原始输出与上一轮逐字节一致。完整手工重放的23个前缀模块及最终logits、每个边界两种同精度后段重放均在runner上精确复现原生结果，状态字典与捕获特征前后保持不变。18项runner合成测试、21项独立审计器合成测试全部通过。

独立复算核查保存的原生/历史输出、前缀边界张量、接收端类型转换哈希、端点别名、全部12项汇总与固定门槛。它不重跑卷积/BN；未单独保存的每层重放哈希和状态字典不变来自经审查runner的断言。两项原生汇总及重复控制不应当作额外独立实验。

{link('完整边界对照', 'artifacts/gate1/20261023_upstream_frontier_full_v1/summary.json')}；{link('独立保存数组复算', 'artifacts/gate1/20261023_upstream_frontier_full_audit_v1/recomputed_summary.json')}；{link('真实拓扑记录', 'artifacts/gate1/20261023_upstream_probe_preparation/topology_inspection_v1/topology.json')}。

## 全109张图的固定门槛结果

每个交叉输出都用FP32解码，与原生FP32模型比较全部候选位置；框门槛仍为 `0.5 + 0.005×abs(FP32参考)`，得分门槛仍为 `0.002 + 0.02×abs(FP32参考)`。下表分列不同方向，不能相减成某层的因果贡献。前段指切分点及之前，后段指切分点之后；边界值会转换为后段接收dtype。

| 切分边界 | 全部跨界模块 | 前32/后16超差图 | 框元素超差 | 前16/后32超差图 | 框元素超差 |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

原生全FP16输出仍有109/109图、10849个框元素超差；两种原生网络都统一FP32解码后仍有109/109图、8992个框元素超差。输入边界“前32/后16”正是后一个已有对照；模块22边界复现上一轮两条head交叉路径。它们是内部校验端点，不增加独立证据。

单独把归一化输入舍入为FP16、提升回FP32再交给全FP32网络，已有 **{input_only['fail_images']}/109图、{input_only['boxes']['violations']}个框元素超差**，最大框差{input_only['boxes']['max_absolute_error']:.6f}像素。这说明输入舍入本身也能影响严格raw检查，不能把全部差异归到某个半精度卷积或BN。

主干32/后段16路径的最大框差为 **{backbone_path['boxes']['max_absolute_error']:.6f}像素**，得分元素超差{backbone_path['scores']['violations']}。该路径仍有{backbone_path['fail_images']}张图、{backbone_path['boxes']['violations']}个框元素超差，说明后段特征融合计算或边界转换仍不能完全排除。在本轮有限切分点中，前32/后16落入原容差的切分点为 `{forward_pass_cuts}`；这不是证明某一点是最小、最优或足以满足真实任务精度。

![各边界的固定门槛对照]({(OUT/'frontier_comparison.png').as_posix()})

左图采用保留零值的对称对数轴，右图直接显示超差图数；各柱是不同交叉路径，不是可相加的误差分解。图表仅来自已复算的109张复用debug结果。

## 现在可说和仍不能说的内容

可说的是：存在可复现的低误差边界重放路径，延长FP32前段在所测组合中明显缓解严格数值检查；后续排查可以侧重输入及主干，继续用小范围预注册对照区分早期与后期影响。不能据此指定某个算子是唯一原因，也不能将模块4/6/10的计数差当作独立可加的贡献。

上述交叉依赖已经保存的原生前缀特征，**还不是另行实现并验收的独立推理程序**。本轮没有新NMS、逐目标评分或官方AP，也没有部署、性能或功耗测试。不能声称半精度整体已通过、INT8方案有效、模型最优或Orin/TensorRT已验证。

原Gate1全FP16数值检查仍为 **FAIL_FIXED_TOLERANCES**；原Gate0.5仍为 **FAILED_UNCHANGED**。参考仍是保存FP16 EMA提升到FP32，不能恢复保存前完整FP32训练EMA。109张图是已复用工程debug，Selection未检测/排名，Holdout ZIP未列成员、解包、读图或推理。

## 评测环境补充与下一步

另完成既有MATLAB/Octave运行环境的有限补充搜索：可读取的标准登记和列明安装目录未发现可用运行时，部分登记拒绝读取，不能据此断言全机未安装。未下载安装软件，官方20个合成用例仍未原生执行；AP等价性未认证。{link('搜索范围及保留错误', 'artifacts/gate1/20261023_official_oracle_followup/README.md')}。

下一步若继续缩小算子范围，应只冻结少量输入/主干边界或固定局部输入的对照；若推进低误差路径，应单独实现推理程序并重新检查输出、目标匹配和工程运行，再讨论部署。科学基线还需官方评测核对、参考精度及ignore损失/数据独立性审阅；Selection研究预算和统计方案冻结后才能进入方法排名，Holdout继续封存。
'''
    report_path=ROOT/state['report'];assert not report_path.exists();report_path.write_text(report,encoding='utf-8')
    state_path=BASE/'current_execution_state_v7.json'
    pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':state_path.relative_to(ROOT).as_posix(),
             'snapshot_sha256':sha(state_path),'note':'This pointer may be updated on later work. Referenced timestamped snapshots are preserved.'}
    (BASE/'latest_execution_state.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':ident(report_path),'decision':ident(OUT/'decision.json'),'pass_cuts':forward_pass_cuts},ensure_ascii=False))


if __name__=='__main__':
    main()
