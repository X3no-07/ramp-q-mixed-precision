"""Saved-document-only Gate0.5 acceptance gap audit; never executes a model."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    output = OUT / 'acceptance_gap_audit.json'
    readable = OUT / 'acceptance_gap_audit.md'
    assert not output.exists() and not readable.exists()
    refs = {}

    def ref(key, relative, start=None, end=None, needle=None, pointer=None):
        path = ROOT / relative
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        if needle is not None:
            hits = [index + 1 for index, text in enumerate(lines) if needle in text]
            assert hits, (relative, needle)
            start = hits[0]
        end = end or start
        refs[key] = {'path': relative, 'sha256': sha(path), 'line_start': start, 'line_end': end,
                     'excerpt': '\n'.join(lines[start - 1:end]), 'json_pointer': pointer}
        return key

    ref('fp16_original_guard', '23_Gate05_Orin_FP16首轮检查.md', 20, 40)
    ref('original_gate_remaining', '23_Gate05_Orin_FP16首轮检查.md', 78, 89)
    ref('original_local_failure', '24_Gate05_Orin_INT8构建与迁移检查.md', 19, 47)
    ref('original_kernel_proxy', '24_Gate05_Orin_INT8构建与迁移检查.md', 98, 104)
    ref('original_quant_scheme', '24_Gate05_Orin_INT8构建与迁移检查.md', 7, 17)
    proto = 'artifacts/gate05/20261018_int8/input_v1/protocol.json'
    ref('build_pass_field', proto, needle='"build_pass"', pointer='/build_pass')
    ref('execution_pass_field', proto, needle='"execution_pass"', pointer='/execution_pass')
    ref('not_a_pass_condition', proto, needle='"not_a_pass_condition"', pointer='/not_a_pass_condition')
    ref('dataset_gate_unfrozen', proto, needle='"dataset_gate_decision_rules"', pointer='/dataset_gate_decision_rules')
    ref('original_numeric_tolerances', proto, needle='"local_preflight"', pointer='/local_preflight')
    ref('six_old_configs', '26_Gate05_当前可执行工作收口.md', 7, 18)
    ref('old_kernel_trace_limit', '26_Gate05_当前可执行工作收口.md', 139, 147)
    ref('restore_kernel_trace_limit', '26_Gate05_当前可执行工作收口.md', 163, 171)
    ref('reference_decision_explicitly_open', '26_Gate05_当前可执行工作收口.md', 198, 207)
    ref('reference_failure_reproduced', '29_Gate05_完整参考数值分歧定位.md', 7, 16)
    ref('rounding_causal_evidence', '29_Gate05_完整参考数值分歧定位.md', 28, 41)
    ref('hybrid_not_independent', '29_Gate05_完整参考数值分歧定位.md', 49, 65)
    ref('new_reference_protocol_needed', '29_Gate05_完整参考数值分歧定位.md', 82, 84)
    ref('corrected_full_compiler_failure', '30_Gate05_全INT8修复候选验证.md', 54, 72)
    ref('parser_imported_constraints', '30_Gate05_全INT8修复候选验证.md', 84, 105)
    ref('current_research_acceptance', '研究状态与后续计划.md', 167, 187)
    ref('current_catalog_not_acceptance', 'README.md', 383, 407)
    ref('gate2_performance_separate', '研究状态与后续计划.md', 202, 214)
    contract = 'artifacts/gate05/20261019_reference_contract/reference_contract.json'
    ref('contract_not_gate_amendment', contract, needle='"not_a_gate_amendment"', pointer='/not_a_gate_amendment')
    ref('contract_reference_roles', contract, 7, 22, pointer='/reference_roles')
    ref('contract_actual_precision', contract, 93, 108, pointer='/native_audit_and_execution')
    ref('contract_floating_control_gap', contract, 110, 113, pointer='/control_identity')
    ref('contract_dataset_forbidden', contract, 115, 128, pointer='/dataset_gate')
    ref('named_tactic_regex_proxy', 'scripts/audit_gate05_int8_precision.py', 128, 148)
    ref('target_matching_source', 'src/gate0.py', 424, 474)
    ref('data_fixed_and_used', 'artifacts/gate05/data_v1/manifest.json', 1, 14)
    ref('corrected_qonly_status', 'artifacts/gate05/20261019_corrected_qonly_native/checkpoint.json', 1, 27)
    state_relative = 'artifacts/gate05/current_execution_state.json'
    ref('state_snapshot', state_relative, needle='"gate05_status"', pointer='/gate05_status')
    state = read(ROOT / state_relative)
    qtrace = read(ROOT / 'artifacts/gate05/20261019_qall_kernel_diagnostic/analysis.json')
    kernels = []
    for index, row in enumerate(qtrace['layers']):
        entries = [k for correlation in row['correlations'] for k in correlation['kernel_rows_with_exact_correlation_id']]
        kernels.append({'saved_analysis_pointer': '/layers/' + str(index), 'native_layer_name': row['native_layer_name'],
            'native_type': row['native_layer_type'], 'native_tactic_name': row['native_tactic_name'],
            'instruction_proven_in_saved_trace': row['integer_arithmetic_instruction_proven'],
            'launches': [{'mangled_symbol': k['mangled_name_text'], 'demangled_symbol': k['demangled_name_text'],
                          'correlation_id': k['correlationId'], 'grid': [k['gridX'], k['gridY'], k['gridZ']],
                          'block': [k['blockX'], k['blockY'], k['blockZ']]} for k in entries]})
    ref('saved_seven_exact_kernel_identities', 'artifacts/gate05/20261019_qall_kernel_diagnostic/analysis.json',
        needle='"target_depthwise_layers"', pointer='/layers')
    ref('saved_twentyone_restore_kernel_identities', 'artifacts/gate05/20261019_restore_saved_execution_diagnostic/analysis.json',
        needle='"restores"', pointer='/restores')
    minimal = [
        {'id': 'A_identity', 'requirement': 'Freeze same model/calibration/150-image list/target IDs/preprocessing and source-to-G1-G8 mapping; record same Orin/software/batch/power.',
         'original_requirement': True, 'sources': ['original_gate_remaining', 'current_research_acceptance', 'data_fixed_and_used'],
         'current': 'Mostly reusable immutable identities; candidate preparation/control family still needs one new frozen identity.', 'gap': 'Candidate-dependent common graph and floating-control identities.'},
        {'id': 'B_deployment', 'requirement': 'Five complete real engines: matched FP16, q_all, r_only_G1, r_only_G1_G5, r_only_G1_G2; each requested Conv actually mapped and precision verified, finite real execution.',
         'original_requirement': True, 'sources': ['original_gate_remaining', 'build_pass_field', 'execution_pass_field', 'current_research_acceptance'],
         'current': 'Old five built/executed but all quantized fixed-image outputs anomalous. Corrected q_only alone now healthy; corrected full q_all has no engine.',
         'gap': 'Numerically credible corrected q_all + three corrected restores + matched floating control; cannot mix corrected q_only with anomalous old engines as repair success.'},
        {'id': 'C_reference_controls', 'requirement': 'Keep historical PyTorch/ORT failure and independently establish comparable deployment reference plus floating/fusion controls before target inference.',
         'original_requirement': 'Original raw guard failed; later explicit open reference/control decision, not a requirement to force PyTorch/ORT bit equality.',
         'sources': ['original_local_failure', 'reference_decision_explicitly_open', 'hybrid_not_independent', 'contract_reference_roles', 'contract_floating_control_gap'],
         'current': 'Backend-rounding explanation strong for fixed image; hybrid is not independent. Current contract explicitly single-image-only.',
         'gap': 'A new independent end-to-end reference/control contract with no cached native tensor injection; protect old guard/tolerances/status and state new estimand.'},
        {'id': 'D_preoutcome_rules', 'requirement': 'Freeze PASS/FAIL/INCONCLUSIVE target-level rules and denominator before running the fixed 150 images.',
         'original_requirement': True, 'sources': ['original_gate_remaining', 'dataset_gate_unfrozen', 'contract_dataset_forbidden', 'current_research_acceptance'],
         'current': 'No dataset criterion frozen and no native150-image run recorded.',
         'gap': 'New protocol must state target-level event definitions, collateral loss, abstention/technical failure handling, and planned comparisons; no threshold tuning after results.'},
        {'id': 'E_target_migration', 'requirement': 'Same150 images/all865 targets: local-FP16-relative added misses, same-target rescue, collateral misses; separately FP32-to-FP16 changes; CSV/raw/cases/report.',
         'original_requirement': True, 'sources': ['original_gate_remaining', 'current_research_acceptance', 'target_matching_source'],
         'current': 'Only fixed000020 smoke/GT; no dataset target result.', 'gap': 'Run every frozen image/config exactly once per protocol, preserve all targets/negative cases and raw outputs.'},
        {'id': 'F_bidirectional_conditional', 'requirement': 'q_only_G1 is required only if claiming real TensorRT bidirectional same-target attribution; other groups need own q_only only when making those claims.',
         'original_requirement': 'Conditional extension, not necessary for five-engine recovery-only conclusion.',
         'sources': ['original_gate_remaining', 'current_research_acceptance'],
         'current': 'Corrected full q_only_G1 is available; no150-image results.', 'gap': 'If claiming G1 dual attribution, evaluate it and intersect the same target IDs, no inference from detection-count equality.'}
    ]
    extra = [
        {'condition': 'Every INT8 source must have a nonempty inspector TacticName matching i8|s8|int8|imma.',
         'classification': 'Project-added conservative proof proxy for original real-integer-kernel requirement. Cannot silently waive; direct authenticated machine-code evidence can be a separately registered equivalent evidence channel.',
         'sources': ['build_pass_field', 'original_kernel_proxy', 'named_tactic_regex_proxy', 'contract_actual_precision']},
        {'condition': 'OBEY must be enabled, no parser-request reset, one-shot per-stage timing/CPU byte proof.',
         'classification': 'Frozen diagnostic controls of particular stages, not original Gate scientific outcome. Each frozen failed attempt stays failed. New controlled compiler recipe may use NONE/PREFER if actual requested precision still passes; no hidden resets/fallback.',
         'sources': ['original_gate_remaining', 'parser_imported_constraints', 'contract_actual_precision']},
        {'condition': 'All old and new graph raw outputs or PyTorch/ORT output bits must be identical.',
         'classification': 'Exact array equality is specific to declared graph-rewrite proofs on a fixture; not general cross-backend inference identity. Original tolerance guard is binding for its original comparison and remains failed.',
         'sources': ['fp16_original_guard', 'original_numeric_tolerances', 'hybrid_not_independent', 'contract_reference_roles']},
        {'condition': 'Every Conv output INT8, all network ops INT8, Tensor Core-only arithmetic, FP16 accumulation, FPS/energy Pareto gains.',
         'classification': 'Not Gate0.5 requirements. Original quantization does not newly quantize Conv outputs; other ops may float; Gate2 covers performance. Integer MAC evidence can be SIMT integer or IMMA, not necessarily TensorCore.',
         'sources': ['original_quant_scheme', 'fp16_original_guard', 'gate2_performance_separate']},
        {'condition': 'Unique full-model failing source node or all local diagnostic experiments must be resolved before any valid corrected deployment can proceed.',
         'classification': 'Not an original acceptance requirement. Preserve unresolved root cause; a new complete implementation can qualify via static semantic proof, actual precision, credible independent reference, and target criteria. Partial slices alone cannot qualify it.',
         'sources': ['original_gate_remaining', 'reference_decision_explicitly_open', 'current_research_acceptance']}
    ]
    reference = {
        'answer': 'A separately frozen independent deployment reference is legitimate; turning the old FAILED guard into PASS or relabeling a cache-injected/hybrid run as original reference is not.',
        'basis': ['reference_decision_explicitly_open', 'rounding_causal_evidence', 'hybrid_not_independent', 'new_reference_protocol_needed', 'contract_reference_roles'],
        'minimum_new_contract': [
            'Keep original preflight path/hash/rtol/atol and FAILED_UNCHANGED. Label new contract as deployment migration verification, not retrospective Gate0 simulation parity.',
            'Independent end-to-end original/corrected ONNX CPU with immutable images/weights/scales, CPUExecutionProvider/ORT_DISABLE_ALL/threads pinned; no native outputs or recorded hidden tensors injected into reference.',
            'Require exact selected-node/initializer/quantizer-rule preservation and same-input quantizer tests as semantic proof; distinguish signed zero numerical equality from byte identity.',
            'Use own CPU output for each new quantization configuration. Weight-QDQ bypass changes mathematical model and cannot be q_all-equivalent or inherit its expected array.',
            'Establish complete unquantized/floating controls on the same common output preparation; inspect nonquantized formats and fusions. Report residual backend differences instead of assigning all change to quantization.',
            'Define native-to-reference numeric/target decision before new dataset inference. Do not set a large all-output tolerance to cover already seen96.57box error or use same detection count as correctness.',
            'For mixed-G1 causal claim, use same native FP16 baseline and same backend/control family; separately report Gate0 PyTorch FP32-to-native FP16 shifts. Do not conflate FP32-restoration simulation with FP16 deployment restoration.'
        ],
        'cannot_infer': 'The fixed-image backend explanation does not prove all images or TensorRT correct; no promise that original raw guard can be passed without changing implementation.'
    }
    sass = {
        'existing_evidence': '7 q_all plus21 restore source layers already correlate exact inspector JSON/NVTX thread/CUDA launch correlationId to full mangled kernel symbols; no instruction evidence yet.',
        'qall_engine_sha256': qtrace['engine_sha256'], 'sqlite_sha256': qtrace['sqlite_sha256'], 'seven_layers': kernels,
        'bounded_readonly_first_path': [
            'Inventory the board-installed cuobjdump/nvdisasm version, actual library paths/build IDs/SHA256 and architecture. Do not upgrade packages or assume engine.plan itself contains a readable ELF.',
            'Search only relevant TensorRT/CUDA fatbin-bearing libraries from actual installed version for the exact saved mangled symbols; cuobjdump --list-elf and --dump-elf-symbols, then --dump-sass --function <exact_symbol> <library>. Preserve full stdout/stderr/toolversion/hash, not grep fragments.',
            'If resident/JIT kernel has no matching extractable symbol, record static route inconclusive; bounded future module/cubin identity capture or NsightCompute SASS path may be needed. Do not mass dump all libraries or claim a function spelling proves arithmetic.',
            'Tie disassembly architecture+library/cubin identity+exact symbol to the engine/run kernel identity. Same spelling alone cannot distinguish multiple architectures/overloads/JIT modules.',
            'For each MAC loop, trace loaded quantized activation/weight operands into signed8bit IDP4A/IDP or IMMA signed8bit/INT32 accumulator (or demonstrably equivalent integer MAC dataflow), with scaling/conversion/output FP32 separately. Address/pointer IMAD alone is not proof.',
            'Check control flow/predicates: presence of integer instructions in an unexecuted branch is insufficient to claim executed arithmetic. If unresolved, preserve STATIC_INTEGER_PATH_FOUND versus EXECUTED_INTEGER_PATH_VERIFIED separately; optional narrowly scoped instruction profiling only if needed.',
            'Create a new external supplement referencing unchanged old precision_audit, rather than fabricating missing TacticName or overwriting old strict failure. New candidate engines need their own layer/launch/module mapping; old binary evidence can be reused only with exact identity chain.'
        ],
        'expected_terminal_states': ['VERIFIED_INTEGER_MAC_FOR_ALL_REQUESTED_SOURCES', 'FORMAT_ONLY_INCONCLUSIVE', 'ARITHMETIC_FALLBACK_OBSERVED_FAIL'],
        'not_required': 'INT8 external output, integer-only epilogue or specifically Tensor Core arithmetic.',
        'no_disassembly_or_board_command_run_in_this_audit': True
    }
    reusable = [
        {'asset': 'data_v1 manifest64 calibration/150 evaluation/865GT/no image-content overlap', 'use': 'Reuse fixed list and labels; oldHoldout is deployment migration diagnostic, not fresh method-confirmation data.', 'source': 'data_fixed_and_used'},
        {'asset': '88Conv group/weight/calibration identities, 23FP16 smoke, old common-output FP16 engine', 'use': 'Reuse provenance, matching implementation and execution tool tests. New corrected preparation needs an explicitly matched control or proof of what changed.', 'source': 'original_gate_remaining'},
        {'asset': 'Original six native engines/raw/receipts/traces', 'use': 'Reuse negative history, exact running-kernel mapping and SASS targets. Do not present zero-detection old graphs as corrected numeric validation.', 'source': 'six_old_configs'},
        {'asset': 'Corrected q_only_G1 engine486845..., strict6INT8/82FP16, fixedGTmatch', 'use': 'Can reuse exact engine for conditional dual experiment if new common/control contract admits its identity; fixedGTmatch alone not dataset pass.', 'source': 'corrected_qonly_status'},
        {'asset': 'All88 codeClip static/synthetic CPU equivalence and fullq_all fixed-image primary+aux byte proof', 'use': 'Avoid repeating already authenticated semantic preparation; add only proofs needed by new family or varied image inputs.', 'source': 'corrected_full_compiler_failure'},
        {'asset': '29 full/replay/hybrid and31–36 local failed/success contexts', 'use': 'Reuse diagnosis to choose bounded full deployment candidate; no obligation to chase every local derivative before evaluating a valid full implementation.', 'source': 'rounding_causal_evidence'}
    ]
    closure = [
        {'step': 1, 'action': 'Freeze a Gate0.5 acceptance/reference/control protocol now, with signed identities and outcome states. Separate completed engineering task from scientific PASS/FAIL/INCONCLUSIVE.',
         'completion_artifact': 'New gate_acceptance_protocol.json + independent review. Existing reference_contract remains unchanged single-image diagnostic.', 'stop': 'Do not run150 until criterion/control decision is frozen.'},
        {'step': 2, 'action': 'Parallelize static SASS identity/instruction audit of saved kernels with a corrected full-model compiler recipe. Prefer one controlled fullq_all attempt with only NONE instead of OBEY/PREFER, preserving all parser QDQ requests and Clip correction; inspect actual formats/fusions afterward.',
         'completion_artifact': 'New protocol/controlled derivative + full engine/receipt OR exact terminal compiler failure; no promise NONE repairs whole graph.',
         'stop': 'If full build fails, use the exact failure evidence to select at most a new bounded next control. No endless unconstrained local matrix or silent semantics change.'},
        {'step': 3, 'action': 'Qualify corrected fullq_all numerics on existing fixed image and independent CPU reference with same control family; inspect actual everyConv and relevant floating/fusion changes. Reuse semantic equivalence already valid.',
         'completion_artifact': 'Native raw/CPU comparison+GT diagnostics+complete actual mapping/SASS supplement; retain原failed.',
         'stop': 'Anomalous output or precision fallback is technical FAIL/INCONCLUSIVE, not evidence mechanism failed.'},
        {'step': 4, 'action': 'Extend same validated corrected preparation to three originally planned restores and matched FP16; no changing groups/calibration based on results. Reuse q_only_G1 only if compatible and claim needs dual attribution.',
         'completion_artifact': 'Five required full engines and actual precision audit/finite executable checks; optional sixthq_only.',
         'stop': 'Any missing/wrong requested precision prevents deployment-qualified PASS; preserve each terminal status.'},
        {'step': 5, 'action': 'Run locked150images/865targets across qualified engines and independent FP32/FP16 controls, applying existing sameclass one-to-one matching. Save targetCSV/raw/predictions/allnegatives and same-target cases.',
         'completion_artifact': 'Table of FP16 hits, q_all added misses, G1/G1G5/G1G2 rescue/collateral/net and conditional dual-target intersection.',
         'stop': 'No postoutcome threshold/image/module changes. OldHoldout cannot become new independent confirmation.'},
        {'step': 6, 'action': 'Issue scientific terminal outcome and engineering completion separately. PASS only if frozen meaningful target-recovery criteria and all technical prerequisites hold; technically sound nonreproduction is FAIL, absent eligible events or unresolved precision/reference is INCONCLUSIVE.',
         'completion_artifact': 'FinalGate0.5 report+machine-readable acceptance receipt+immutable evidence manifest+clean task exit.',
         'stop': 'User asks completion, not guaranteedpositive result. Do not manufacture PASS or keep a completed negative experiment open just to chase a favorable result.'}
    ]
    proposed_rules = {
        'status': 'PROPOSAL_ONLY_NOT_FROZEN_CRITERIA',
        'event_definitions': {'F': 'Targets matched by this nativeFP16 baseline', 'Q': 'Targets matched by q_all',
            'R_G1': 'Targets matched by r_only_G1', 'added_misses': 'F minus Q', 'rescued_G1': '(F minus Q) intersect R_G1',
            'collateral_G1': '(F intersect Q) minus R_G1', 'dual_G1': '(F minus Q) intersect (F minus q_only_G1 hits) intersect R_G1'},
        'logical_minimum': 'Nonempty added-miss set and nonempty same-targetG1 rescue are necessary for existential recovery reproduction. They are not sufficient for practical effect/robustness or method superiority.',
        'must_still_choose_preoutcome': 'If PASS includes netbenefit/stable-retention/effectsize/confidence interval thresholds, put explicit numbers and handling of small denominators into the new protocol BEFORE150inference. No such quantitativeGate0.5 threshold is present in original23/24; do not invent one as inherited.',
        'FAIL_vs_INCONCLUSIVE': 'All technical prerequisites met but frozen recovery criterion fails => mechanism not reproduced on this diagnostic set. No added-miss denominator/sparse eligible events or technical comparability unmet => INCONCLUSIVE as preregistered. Both are reportable completion, neither is PASS.'}
    official = [
        {'title': 'TensorRT8.6.1 Developer Guide7.4, nearest archived family to installed8.5.2.2',
         'url': 'https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html',
         'verified_via': 'Official indexed excerpts; direct archive redirects inaccessible in browser.',
         'use': 'Explicit Q/DQ controls precision representation; do not infer requirement for manually forcing all outputsINT8. Version8.6 guidance supports rationale, not a guarantee of8.5.2.2 behavior.'},
        {'title': 'CUDA11.4.4 Binary Utilities2.2/Instruction Set Reference',
         'url': 'https://docs.nvidia.com/cuda/archive/11.4.4/cuda-binary-utilities/index.html',
         'use': 'cuobjdump dump-sass/function/list-elf and nvdisasm disassembly; IDP4A/IMMA definitions. Verify actual installed tool/SM support first, no assumed upgrade.'},
        {'title': 'TensorRT10.x Working with Quantized Types, contextual numerical principle only',
         'url': 'https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html',
         'use': 'Q/DQ optimization may change floating operation order, so bit identity is generally not promised. This does not retroactively invalidate the project original tolerance failure or prove this engine correct.'}
    ]
    report = {'schema': 'rampq.gate05.acceptance_gap_audit/v1',
        'client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'READ_ONLY_ACCEPTANCE_GAPS_AND_COMPLETION_PATH_AUDITED_NOT_A_CRITERIA_AMENDMENT',
        'gate05_current_status': 'INCOMPLETE', 'audit_runs_CPU_native_or_SSH': False, 'oldcriteria_or_entry_documents_modified': False,
        'original_failure_and_tolerances_must_remain': True, 'sources': refs, 'minimum_acceptance': minimal,
        'diagnostic_additions_vs_original_requirements': extra, 'independent_reference_contract_assessment': reference,
        'strict_integer_evidence_path': sass, 'reusable_evidence': reusable, 'completion_sequence': closure,
        'candidate_dataset_outcome_rules': proposed_rules, 'official_sources': official,
        'current_state_snapshot': {'sha256_at_read': refs['state_snapshot']['sha256'], 'client_at': state.get('client_at'),
            'gate05_status': state.get('gate05_status'), 'current_stage_status': state.get('current_stage_status'),
            'no_dataset_inference_run': state.get('no_dataset_inference_run'),
            'warning': 'Historical state.goal says stop for planning; new direct user instruction continue untilGate0.5completion supersedes that old authorization text. Snapshot is not a live board check.'}}
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    def cite(key):
        r = refs[key]
        return '[' + r['path'] + ':' + str(r['line_start']) + '](' + (ROOT / r['path']).as_posix() + ':' + str(r['line_start']) + ')'
    text = f'''# Gate0.5 验收缺口与闭环审计

当前仍为 INCOMPLETE。应停止把每个局部诊断收口当成 Gate 完成；闭环目标是完整可比部署、预先冻结的逐目标评估和可审计终态。科学结果可以是 PASS、FAIL 或 INCONCLUSIVE，完成任务不保证机制复现。

## 原始最小验收

原23要求 FP16、q_all、G1恢复、G1+G5恢复、G1+G2恢复五个完整 engine，同一固定集逐目标检查新增漏检、同目标救回和附带漏检，并另查 FP32→FP16。q_only:G1 只有在声称实机双向归因时才是必需的第六个配置。数据规则须在数据集输出前冻结。{cite('original_gate_remaining')}；{cite('current_research_acceptance')}。

原24协议的 build_pass 是88源Conv映射到真实INT8加权内核，execution_pass 是固定图真实执行且有限；明确不要求INT8检测数等于FP16。它没有定义数据集通过门槛。README9.11只是证据目录，不能把文件齐全当验收。{cite('build_pass_field')}；{cite('execution_pass_field')}；{cite('not_a_pass_condition')}；{cite('current_catalog_not_acceptance')}。

## 当前真正缺口

1. 旧五个量化配置虽能运行，固定图均零检测，不能作为已修复的可信基线。只有另存 corrected q_only_G1 已恢复固定图且6INT8/82FP16完整。修正全q_all仍无engine，三个修正恢复图尚未完成。
2. 没有为新 corrected preparation 建立完整匹配的浮点/融合控制。旧FP16可复用历史来源，不能无条件宣称新家族控制已匹配。
3. 原PyTorch/ORT完整护栏失败仍保留；当前reference_contract只允许一次单图诊断，明示not_a_gate_amendment，不能自动覆盖150图验收。
4. q_all及三恢复各7层只有INT8格式和运行内核身份，缺严格整数MAC证据。未运行固定150图、未生成逐目标迁移结果，也未冻PASS/FAIL/INCONCLUSIVE。

证据：{cite('six_old_configs')}；{cite('corrected_qonly_status')}；{cite('corrected_full_compiler_failure')}；{cite('contract_floating_control_gap')}；{cite('contract_dataset_forbidden')}。

## 哪些属于附加诊断条件

“命名TacticName必须含int8/imma”等是本项目保守证据代理，不是整数计算本身的定义。不能跳过七层，但可另冻直接机器码证据通道，保留旧审计状态，不伪造TacticName。{cite('named_tactic_regex_proxy')}；{cite('contract_actual_precision')}。

OBEY、每轮小图CPU字节等价及300/900秒预算是相应冻结实验的条件，不是原Gate科学要求。新独立构建可仅改变请求方式而保留真实逐层审计，不能把旧失败改判。所有Conv输出INT8、所有算子INT8、必须TensorCore、FPS/能耗优势也不是Gate0.5要求。原图没有新增Conv输出量化；性能在Gate2。{cite('original_quant_scheme')}；{cite('gate2_performance_separate')}。

唯一全图错误节点定位也不是Gate原要求。修复方案若能通过来源/语义/完整实际精度/独立参考/目标级验收，可以诚实保留根因未穷尽；局部成功自身仍不算完整部署证据。

## 原护栏失败与独立参考可以并存

26已明确把“另设可对齐参考实现及独立目标级迁移验收”列为开放决策。29在固定图上证实微小Conv差跨舍入边界，hybrid收敛只是反事实，不能当独立参考。{cite('reference_decision_explicitly_open')}；{cite('rounding_causal_evidence')}；{cite('hybrid_not_independent')}。

正当做法是新冻部署参考协议：原guard路径/哈希/rtol/atol/FAILED_UNCHANGED全部不动；用无隐藏张量注入的完整ONNX CPU独立运行每个实际配置，权重/尺度/算子语义静态验证与跨后端数值/目标一致性分开。配置数学语义若变，例如旁路权重QDQ，必须用该候选自己的CPU参考，不能称q_all等价修复。native比较必须与同族FP16和浮点/融合控制相配；不能为覆盖已见96.57框误差而补设大容差。新协议不保证旧护栏会变PASS，也不把旧Gate0模拟参考换名。

## 七层整数MAC证据的最短路径

已有q_all7层及恢复21层的精确NVTX/inspectorJSON/同线程CUDA API/correlationId→完整mangled symbol映射，可复用而不重跑旧推理。先只读检查实装工具/库/buildID/哈希和SM，使用cuobjdump list-elf、dump-elf-symbols，再对匹配symbol dump-sass；不要假定engine.plan就是ELF。若库中无可对应cubin，记录inconclusive，才考虑新的有界module/cubin捕获。

必须把该SM/库或cubin身份/符号与运行kernel对应，追踪激活与权重的整数加载到signed8 IDP4A/IDP或IMMA/INT32累加主循环，区分FP32缩放/输出epilogue。地址计算IMAD、函数名、TacticValue或仅出现一条未执行分支的整数指令均不够。静态可达整数路径与已执行路径分开记录；若控制流不能判定，才提最小指令剖析。新的补充报告引用原audit，不覆盖它。新engine需要自己的身份链。完整七层列表和符号在JSON中。{cite('saved_seven_exact_kernel_identities')}；{cite('saved_twentyone_restore_kernel_identities')}。

官方工具说明：[CUDA11.4.4 Binary Utilities](https://docs.nvidia.com/cuda/archive/11.4.4/cuda-binary-utilities/index.html)。其支持按函数导出SASS，并区分IDP4A与IMMA整数运算；实际工具版本/架构支持仍须核实。TRT近邻归档[8.6.1开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html)解释QDQ显式表示；8.6文档不是8.5.2.2行为保证。跨后端不普遍保证逐位相同的背景见[NVIDIA量化说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)，不能据此抹去本项目原容差失败。

## 建议闭环顺序

先冻验收/参考/浮点控制和终态规则；并行做保存内核的静态SASS证据与一个有界完整corrected q_all构建控制，优先评审仅关闭OBEY/PREFER、保留QDQ parser请求/Clip校正的新全图尝试，实际精度仍严格审。它不是已经证明的全图修复。若完整数值和格式可信，再扩三恢复及匹配FP16，按声明需要复用q_only_G1，然后跑固定150图/865目标，生成逐目标CSV和终态报告。不要无限追加局部矩阵，也不能凭旁路权重的小图替代全量化模型。

数据事件沿用同类一对一匹配：F为本机FP16命中，Q为q_all命中，新增漏检F−Q，G1救回(F−Q)∩R，附带漏检(F∩Q)−R；双向另交q_only_G1漏检目标ID。原始协议没有Gate0.5净收益/稳定保持率的数字门槛，若PASS要求这些实用效果，必须先明确预注册，不能伪称继承旧规则。非空新增漏检且至少同目标救回是存在性复现的逻辑必要条件，不足以证明稳健收益或方法优越。

技术条件满足但预定复现标准失败，应报告FAIL；缺目标分母/事件太少或精度参考不成立，按预注册报INCONCLUSIVE。二者均可形成如实的阶段收口，不能写PASS。旧Holdout只作部署诊断，不能重新包装新方法独立确认集。正式FPS/功耗不在本轮验收范围。

本审计只读取文档和保存JSON；没有CPU/板端推理、SSH、改旧判据或入口文件。结构化完整证据、逐项源行和后续产物要求见[acceptance_gap_audit.json]({output.as_posix()})。
'''
    with readable.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
    print(json.dumps({'status': report['status'], 'source_references': len(refs), 'minimum_acceptance_items': len(minimal),
                      'audit_sha256': sha(output), 'readable_sha256': sha(readable)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
