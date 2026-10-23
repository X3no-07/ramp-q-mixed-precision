"""Publish the completed conditional deployment evaluation from saved evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/gate05'
MATRIX = BASE / '20261021_dataset_matrix'
FINAL = BASE / '20261021_final_acceptance'
ORDER = ['fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    contract_path = BASE / '20261021_dataset_contracts/acceptance_contract.json'
    contract = read(contract_path)
    summary_path = MATRIX / 'target_evaluation/summary.json'
    target_path = MATRIX / 'target_evaluation/target_matrix.jsonl'
    review_path = MATRIX / 'target_evaluation/independent_result_verification.json'
    summary, review = read(summary_path), read(review_path)
    require(sha(contract_path) == '759506e78383fa262c3cb74735c24c57844d706fdb95bf95bce168d9b143f500' and
            sha(summary_path) == review['summary_sha256'] and sha(target_path) == review['target_matrix_sha256'] and
            sha(MATRIX / 'saved_prediction_matrix.json') == review['matrix_sha256'], 'Final evidence identity mismatch')
    require(review['status'] == 'ALL_SAVED_MATCHING_BITS_TARGET_SETS_AND_VERDICT_INDEPENDENTLY_RECOMPUTED' and
            review['full_target_rows_exact'] and review['target_hit_bits_verified'] == 10380 and
            review['new_model_calls'] == review['new_NMS_calls'] == 0 and review['protected_sources_unchanged'] and
            review['recomputed_primary_verdict'] == summary['primary_verdict'], 'Independent result verification incomplete')
    require(sha(ROOT / 'scripts/verify_gate05_saved_target_result_independent.py') == review['verification_tool_sha256'] ==
            '12b8d6c098ff2a88e3f0a2df234500e37c0cde719da978ad36e7896931ad0941', 'Independent verifier source changed')
    original = contract['protected_history']
    require(sha(ROOT / original['protected_preflight_path']) == original['protected_preflight_sha256'] and
            sha(ROOT / 'src/gate0.py') == 'f3c4d80658b79ae685e569b37c5ea89af9e6e9dd4a63ce86aff2806a7fcc5adb', 'Historical guard or matching implementation changed')
    post = read(MATRIX / 'postprocessed_detections/completion.json')
    require(post['NMS_calls'] == 1800 and post['original_raw_files_unchanged'] and
            post['status'] == 'ALL1800_COMMON_CPU_NMS_OUTPUTS_SAVED_NO_TARGET_SCORING', 'Common postprocessing incomplete')
    numeric = read(MATRIX / 'raw_native_ORT_descriptive_comparison.json')
    require(numeric['status'] == 'SAVED_RAW_DESCRIPTIVE_COMPARISON_COMPLETE' and numeric['all_outputs_finite'] and
            numeric['new_numeric_tolerances'] is False and numeric['numeric_equivalence_PASS_claim'] is False,
            'Raw descriptive comparison incomplete or incorrectly relabelled')
    targets = [json.loads(line) for line in target_path.read_text(encoding='utf-8').splitlines()]
    require(len(targets) == 865 and len({r['target_id'] for r in targets}) == 865, 'Target universe incomplete')
    counts = {backend: {cfg: sum(row[backend][cfg]['matched'] for row in targets) for cfg in ORDER} for backend in ('native', 'ort')}
    now = datetime.now().astimezone().isoformat()
    require(not FINAL.exists() and not (ROOT / '39_Gate05_150图完整验收结果.md').exists(), 'Final deliverable already exists; no overwrite')
    FINAL.mkdir()
    preserved = FINAL / 'protected_document_snapshots'
    preserved.mkdir()
    source_snapshots = {}
    for filename in ('README.md', '研究状态与后续计划.md'):
        source = ROOT / filename
        saved = preserved / filename
        saved.write_bytes(source.read_bytes())
        source_snapshots[filename] = {'snapshot': saved.relative_to(ROOT).as_posix(), 'sha256': sha(saved)}
    require(source_snapshots['研究状态与后续计划.md']['sha256'] == contract['sources']['research_route']['sha256'],
            'Preserved original research scope citation differs from the frozen contract')
    r = summary['recovery']['r_only_G1']
    a, stable = len(summary['added_miss_target_ids']), len(summary['stable_target_ids'])
    selected_table = read(BASE / '20261021_dataset_contracts/qualification/selected_identity_table.json')
    precision = {r['name']: (r['quantized_source_convs'], r['actual_fp16_source_convs']) for r in selected_table['members']}
    rows = ['| 配置 | 已审源卷积 INT8 / FP16 | 板端命中 / 865 | ORT 命中 / 865 | 板端与 ORT 命中状态不一致 |',
            '|---|---:|---:|---:|---:|']
    for cfg in ORDER:
        rows.append(f"| `{cfg}` | {precision[cfg][0]} / {precision[cfg][1]} | {counts['native'][cfg]} | {counts['ort'][cfg]} | {summary['native_ORT_target_disagreements'][cfg]} |")
    recovery_rows = ['| 恢复配置 | 救回新增漏检 / 44 | 仍漏检 | 稳定目标附带漏检 / 678 | 净恢复计数 | FP16 外新增命中 |',
                     '|---|---:|---:|---:|---:|---:|']
    for cfg, value in summary['recovery'].items():
        recovery_rows.append(f"| `{cfg}` | {len(value['rescued_target_ids'])} | {len(value['still_missing_target_ids'])} | {len(value['collateral_target_ids'])} | +{value['net_count']} | {len(value['outside_FP16_target_ids'])} |")
    numeric_rows = ['| 配置 | 网络坐标最大绝对差 | 类别分数最大绝对差 | PE 最大绝对差 |', '|---|---:|---:|---:|']
    for value in numeric['configurations']:
        agg = value['aggregates']
        numeric_rows.append('| `' + value['configuration'] + '` | ' + ' | '.join(f"{agg[key]['max_abs_difference_finite_pairs']:.9g}" for key in
                            ('main_coordinates_4', 'class_scores_10', 'PE_output')) + ' |')
    report = f'''# Gate0.5：150 图完整部署验收结果

记录日期：{now}。本次六配置部署验收已完整完成；事前冻结的 **G1 同目标恢复存在性判据通过**。原完整 PyTorch/ORT 数值护栏仍为 **FAILED_UNCHANGED**，本次不能宣称数值迁移等价性通过。

## 主结果

固定 150 张图、865 个标注目标。板端 FP16 基线命中 **{summary['FP16_hits']}** 个，全 INT8 命中 **{summary['q_all_hits']}** 个。以逐目标身份比较，FP16 能检出而全 INT8 未检出的目标为 **{a}** 个，二者共同命中的稳定目标为 **{stable}** 个。

预定主配置 `r_only_G1` 救回这 44 个目标中的 **40 个（90.91%）**，同时使 678 个稳定目标中的 **3 个（0.4425%）** 漏检。该限定集合的净恢复计数为 **+37**，稳定目标保留率为 **99.5575%**。仍有 4 个新增漏检未救回。

G1 配置总命中数是 **721**，相对全 INT8 的 687 增加 **34**。这与净恢复 +37 的定义不同：全 INT8 在 FP16 未命中的目标中另命中 9 个，G1 配置另命中 6 个，故总命中差为 `40-3+(6-9)=34`。不能把 +37 写成总检出量增加 37。

总数包括全部尺寸目标，不能把 44 个新增漏检或 40 个恢复全部叫作“小目标”。命中数也不是 mAP。

## 六配置结果与预定次级比较

本文“全 INT8”指 88 个源卷积的 W8A8 配置，不表示每个网络算子均为 INT8。非卷积浮点路径和 FLOAT 输出边界另外保留并审查。

{chr(10).join(rows)}

{chr(10).join(recovery_rows)}

G1 是预定主比较。G1+G5、G1+G2 均为事前冻结的次级结果，没有依据它们的结果更换主配置。上述计数不能证明 G1+G5 在同等计算成本下优于 G1；本次未做延迟、能耗或资源预算相同的优越性评测。

辅助 `q_only_G1` 使 35 个 FP16 命中目标消失，其中 26 个同时满足“全 INT8 漏检、G1 恢复后返回”，占新增漏检 26/44（59.09%）。这属于有限的同目标双向模式。该辅助引擎采用预先披露的 **PREFER + 82 个显式 FP16 compute/output 请求**，其余主要成员采用 NONE；它不能作为相同构建约束的纯 G1 因果控制，也未替换主比较。

## 执行与匹配证据

- 六组板端各执行一次完整 150 图，共 **900 次**；六个相应完整 ONNX 图在 CPU ORT 各执行 150 图，共 **900 次**。均为完整图，0 预热、0 额外推理、0 自动重试。
- 所有 1,800 份主输出和 PE 输出均为有限值。板端 6 个 ZIP 的 1,218 个 receipt 内文件已回收并核验；同一 boot 下 30 个任务相关 PID 与串行队列 PID 全部不存在，6 个 wrapper 进程组物理清空，监护未干预。
- 实际反序列化引擎的层检查 JSON 与各自合格构建的层检查内容完全一致；环境、前后加载库 SHA、maps 设备号/inode、源图和引擎身份均认证。逐源卷积实际精度/合法融合审查与各自单图的运行内核/静态 signed INT8 算术补充保持分开，未补造历史缺失的 tactic 名称。
- 同一批共享预处理 NPY、同一 CPU 后处理环境；全部 **1,800 次 NMS** 正常保存，没有警告或超时。候选 NMS confidence=0.001、IoU=0.7、max_det=1000；目标匹配 confidence≥0.25、同类别、IoU≥0.5、一对一，沿用原逆序 `(IoU, confidence, GT索引, 检测索引)` 优先规则。
- 独立实现从保存检测重新匹配，验证 **10,380 个目标命中位**及完整 IoU、置信度、检测 ID；865 个目标行、恢复/附带漏检/净计数、各类别描述和最终判据均完全一致。复核没有调用原 evaluator 的匹配函数，也没有重跑模型或 NMS。

## 数值与研究边界

以下为相应完整 ORT 与板端保存数组的描述性最大绝对差，未引入新的容差，也不将其判为可接受。网络坐标是原输出 4 个坐标通道，类别分数为另外 10 个通道；PE 单列。完整报告还包含逐图、均值、RMS、字节差异及全部身份。

{chr(10).join(numeric_rows)}

FP16 虽有原始数组差异，但此集合的目标命中位与 ORT 一致；量化配置存在 26/17/19/8/1 个目标命中位差异，不能称相应后端等价。原 PyTorch/ORT 预检文件及原容差未修改。新完整 ONNX 参考是独立部署语义参考，不是把原失败改名为通过。

实际编译中 FP16 基线的 DFL Softmax/解码为 FP16，量化配置为 FP32，部分 SiLU/residual 浮点划分及实现也不同。这些差异已在读取本次 150 图输出之前披露并冻结。因此这里支持的是 **所选真实部署配置中的同目标消失/恢复存在性**，不证明纯 G1 算术因果、全部差异的唯一机制、总体精度优越性或 Tensor Core 性能。

新 NONE `q_only_G1` 的 PE Conv 实际 FP32、仅 81 个恢复源卷积达到 FP16，仍判失格，原失败保留。事前按实际精度资格选择旧的同图 6 INT8+82 FP16 引擎作为有限辅助成员，没有用 150 图结果选引擎。

固定集合此前用于诊断，不能包装成新鲜独立留出集；64 张校准图与 150 图集合的文件身份仍保持不交叠。没有在此开展新数据集、多种子显著性、尺寸分层优越性或性能验收。板端时钟与电脑时钟不同，本次以 SHA、boot ID、进程身份和唯一尝试标记关联证据，未调整板端时钟。

## 证据与复核入口

- [冻结验收合同](artifacts/gate05/20261021_dataset_contracts/acceptance_contract.json)，SHA `759506e78383fa262c3cb74735c24c57844d706fdb95bf95bce168d9b143f500`。
- [原始输出索引](artifacts/gate05/20261021_dataset_matrix/raw_family_index.json)、[工程资格](artifacts/gate05/20261021_dataset_matrix/engineering_eligibility.json)。
- [统一检测矩阵](artifacts/gate05/20261021_dataset_matrix/saved_prediction_matrix.json)，SHA `{review['matrix_sha256']}`。
- [目标结果摘要](artifacts/gate05/20261021_dataset_matrix/target_evaluation/summary.json)，SHA `{sha(summary_path)}`。
- [完整 865 目标矩阵](artifacts/gate05/20261021_dataset_matrix/target_evaluation/target_matrix.jsonl)，SHA `{sha(target_path)}`。
- [独立复算记录](artifacts/gate05/20261021_dataset_matrix/target_evaluation/independent_result_verification.json)，SHA `{sha(review_path)}`。
- [原始数值比较](artifacts/gate05/20261021_dataset_matrix/raw_native_ORT_descriptive_comparison.json)。
- [最终机器状态与范围判定](artifacts/gate05/20261021_final_acceptance/final_decision.json)、[最终文件清单](artifacts/gate05/20261021_final_acceptance/final_manifest.json)。旧研究路线原文已按冻结 SHA 保存于 `protected_document_snapshots/`，原 23–38 阶段记录与失败材料保留。

后续工作应另行冻结：用未参与诊断的数据和多个校准种子复验，并针对原数值护栏与后端差异单独推进验证；选型锁定后再测真实延迟和能耗。本轮不会自动开始下一阶段或再次运行这 900 次板端任务。
'''
    report_path = ROOT / '39_Gate05_150图完整验收结果.md'
    report_path.write_text(report, encoding='utf-8')
    decision = {'schema': 'rampq.gate05.final_completed_evaluation_decision/v1', 'completed_at': now,
        'status': 'COMPLETE_CONDITIONAL_DEPLOYMENT_EVALUATION', 'gate05_evaluation_complete': True,
        'primary_verdict': summary['primary_verdict'], 'unqualified_numerical_migration_PASS': False,
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'original_guard_path': original['protected_preflight_path'],
        'original_guard_sha256': original['protected_preflight_sha256'], 'thresholds_changed': False,
        'images': 150, 'physical_targets': 865, 'native_model_calls': 900, 'full_CPU_ORT_model_calls': 900,
        'common_CPU_NMS_calls': 1800, 'new_model_or_NMS_calls_by_finalizer': 0,
        'independently_verified_target_bits': 10380, 'hits': counts,
        'added_misses': a, 'stable_targets': stable, 'primary_G1_rescued': len(r['rescued_target_ids']),
        'primary_G1_stable_collateral': len(r['collateral_target_ids']), 'primary_G1_net_within_FP16_universe': r['net_count'],
        'primary_G1_total_hits_change_over_qall': counts['native']['r_only_G1']-counts['native']['q_all'],
        'native_ORT_target_disagreements': summary['native_ORT_target_disagreements'],
        'report': report_path.name, 'protected_document_snapshots': source_snapshots,
        'remaining_work_inside_frozen_evaluation': [], 'next_stage_started': False,
        'limits': ['Conditional actual-deployment existence; no pure G1 arithmetic causality',
                   'Original numerical guard not repaired; corresponding ORT/native differences retained',
                   'Auxiliary q_only uses predeclared different PREFER policy',
                   'Previously inspected diagnostic set, not fresh independent holdout',
                   'All-size target counts, not mAP or exclusive small-object claims',
                   'No latency, energy or general superiority validation']}
    write(FINAL / 'final_decision.json', decision)
    state_path = BASE / 'current_execution_state.json'
    (FINAL / 'execution_state_before_final_delivery.json').write_bytes(state_path.read_bytes())
    state = read(state_path)
    state.update(client_at=now, gate05_status='COMPLETE_CONDITIONAL_DEPLOYMENT_EVALUATION', active_native_job=None,
                 report=report_path.name, final_decision='20261021_final_acceptance/final_decision.json',
                 next=['A new independent validation stage requires a separate frozen plan; not started in this task'])
    state['dataset_evaluation'].update(status='COMPLETE_INDEPENDENTLY_RECOMPUTED_AND_DELIVERED', pending=[],
                 independent_target_bits_verified=10380, unqualified_numerical_migration_PASS=False,
                 final_decision='20261021_final_acceptance/final_decision.json')
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for filename in ('README.md', '研究状态与后续计划.md'):
        path = ROOT / filename
        old = path.read_text(encoding='utf-8')
        if filename == 'README.md':
            old = old.replace('> 当前阶段：RUOD-R Gate 0 已收口；Gate 0.5 DFL 已缩至 QDQ＋Conv 六节点，NONE 成功而 OBEY 仍同类断言失败；Softmax 与末端 Reshape 均不是该局部失败的必要条件，原完整护栏仍失败（2026-10-21）',
                              '> 当前阶段：Gate0.5 六配置部署验收已完成，G1 同目标恢复存在性通过；原完整数值护栏仍失败（2026-10-21）')
        first, rest = old.split('\n', 1)
        update = ('\n\n## 最新验收结果（2026-10-21）\n\n'
                  'Gate0.5 的六配置真实部署验收已完整完成，独立复算验证 10,380 个目标命中位。150 图/865 目标中，'
                  'FP16 命中 722、全 INT8 命中 687；全 INT8 相对 FP16 新增漏检 44 个。预定主配置仅恢复 G1，'
                  '救回其中 40 个，稳定目标附带漏检 3 个，限定集合净恢复 +37；总命中 721，较全 INT8 增加 34。\n\n'
                  '**通过的是所选真实部署配置的同目标恢复存在性判据。原完整 PyTorch/ORT 数值护栏保持 FAILED_UNCHANGED，'
                  '不能宣称数值迁移等价、纯 G1 算术因果、mAP/性能或总体优越性通过。**'
                  '本次统计包含全部尺寸目标，旧辅助 q_only 的不同 PREFER 约束已披露，诊断复用集合不冒充新鲜独立留出集。\n\n'
                  '完整结果、恢复与附带漏检定义、对应 ORT 差异及证据见 '
                  '[39_Gate05_150图完整验收结果](39_Gate05_150图完整验收结果.md)。'
                  '以下保留此前阶段说明；冻结引用的旧原文按原 SHA 保存于 '
                  '`artifacts/gate05/20261021_final_acceptance/protected_document_snapshots/`。\n')
        path.write_text(first + update + rest, encoding='utf-8')
    paths = [report_path, ROOT/'README.md', ROOT/'研究状态与后续计划.md', state_path,
             contract_path, BASE/'20261021_dataset_contracts/raw_contract.json', MATRIX/'raw_family_index.json',
             MATRIX/'engineering_eligibility.json', MATRIX/'raw_native_ORT_descriptive_comparison.json',
             MATRIX/'saved_prediction_matrix.json', MATRIX/'postprocessed_detections/completion.json',
             MATRIX/'postprocessed_detections/environment.json', summary_path, target_path, review_path,
             BASE/'20261021_dataset_native_downloads/final_native_observation.json',
             BASE/'20261021_gate05_completion_plan/assembler_final_independent_review.json',
             BASE/'20261021_dataset_contracts/qualification/selected_identity_table.json',
             BASE/'20261021_dataset_contracts/qualification/technical_decision.json',
             BASE/'20261021_dataset_contracts/qualification/float_fusion_comparability.json',
             ROOT/original['protected_preflight_path'], ROOT/'src/gate0.py']
    for filename in ('run_gate05_frozen_raw_dataset_v2.py', 'postprocess_gate05_frozen_raw_matrix_v2.py',
                     'evaluate_gate05_saved_target_matrix_v2.py', 'assemble_gate05_saved_raw_index.py',
                     'compare_gate05_saved_raw_reference.py', 'verify_gate05_saved_target_result_independent.py',
                     'finalize_gate05_saved_acceptance.py'):
        paths.append(ROOT/'scripts'/filename)
    paths.extend(path for path in FINAL.rglob('*') if path.is_file())
    for stream in read(MATRIX/'raw_family_index.json')['streams']:
        paths.append(MATRIX/stream['file'])
    files = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(paths))}
    write(FINAL/'final_manifest.json', {'schema':'rampq.gate05.final_evidence_delivery_manifest/v1', 'completed_at':now,
        'status':'COMPLETE_CONDITIONAL_DEPLOYMENT_EVALUATION', 'files':files,
        'scope':'Root evidence and nested raw/detection manifests authenticate full artifacts; per-array raw hashes remain in all12 raw manifests.',
        'new_model_or_NMS_calls':0, 'original_complete_model_numerical_guard':'FAILED_UNCHANGED'})
    print(json.dumps({'status':decision['status'], 'primary_verdict':summary['primary_verdict'],
                      'report':str(report_path), 'final_manifest_sha256':sha(FINAL/'final_manifest.json'),
                      'files':len(files), 'new_model_or_NMS_calls':0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
