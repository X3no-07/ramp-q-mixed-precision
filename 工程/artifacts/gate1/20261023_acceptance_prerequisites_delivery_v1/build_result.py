"""Publish observed synthetic prerequisites; never promote model acceptance."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/gate1'
OUT=Path(__file__).parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def identity(path):return {'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}


def new(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def link(label,relative):return f'[{label}]({(ROOT/relative).as_posix()})'


def main():
    loss=BASE/'20261023_ignore_loss_preparation_v1'
    fp=BASE/'20261023_fp32_reference_preparation_v1'
    native=BASE/'20261023_native_oracle_runtime_v1'
    comp=BASE/'20261023_native_oracle_comparison_preparation_v1'
    semantics=BASE/'20261023_native_python_semantics_v1'
    review=read(BASE/'20261023_acceptance_prerequisites_review_v1/source_review.json')
    assert review['status']=='PASS_SYNTHETIC_COMPONENT_SOURCE_METADATA_AND_SCOPE_REVIEW_ONLY' and not review['issues']
    assert read(loss/'final_tests_v2_receipt.json')['tests_passed']==55
    assert read(fp/'verification_receipt.json')['synthetic_cpu_tests']==10
    launcher=read(native/'execution_01/launcher_receipt.json')
    assert launcher['status']=='EXECUTED_NATIVE_SYNTHETIC_ORACLE_UNCERTIFIED' and launcher['returncode']==0
    oracle=read(native/'execution_01/oracle_results.json')
    assert oracle['receipt']['case_errors']==0 and oracle['receipt']['fixture_count']==20
    audit=read(comp/'native_output_audit_v1.json')
    assert audit['status']=='PASS_NATIVE_SYNTHETIC_OUTPUT_STRUCTURE_IDENTITIES_ONLY' and audit['fixture_count']==20
    assert len(audit['resolved_function_path_checks'])==6
    assert all(row['method']=='live_samefile' for row in audit['resolved_function_path_checks'])
    result=read(semantics/'comparison_result_v1.json')
    assert result['fixture_count']==20 and result['native_matching_fixture_count']==5
    assert result['matrix_checks']==166 and result['no_AP_metric_values_compared']
    equal=result['different_matrix_checks']==0
    assert result['status']==('MATCH_FIXED_SYNTHETIC_PARTIAL_INTERMEDIATES_ONLY' if equal else 'MISMATCH_FIXED_SYNTHETIC_PARTIAL_INTERMEDIATES')
    prep=read(comp/'preparation.json')
    assert prep['auditor_synthetic_regression']['tests']==39 and prep['existing_Python_semantic_regression']['tests']==37
    conditions=read(semantics/'predeclared_conditions.json')
    assert conditions['regression']['tests']==27 and conditions['absolute_tolerance']==conditions['relative_tolerance']==0
    # Require all local trees to be complete before publishing a state snapshot.
    for path in (native/'preparation_manifest.json',semantics/'manifest.json'):
        assert path.is_file(),path
    now=datetime.now(timezone.utc).isoformat()
    evidence=[loss/'manifest.json',loss/'source_gap_review.json',loss/'final_tests_v2_receipt.json',
        fp/'manifest.json',fp/'verification_receipt.json',
        BASE/'20261023_acceptance_prerequisites_review_v1/source_review.json',
        BASE/'20261023_acceptance_prerequisites_review_v1/source_review_manifest.json',
        native/'preparation_manifest.json',native/'runtime_manifest.json',native/'execution_01/launcher_receipt.json',
        native/'execution_01/oracle_results.mat',comp/'preparation_manifest.json',comp/'native_output_audit_v1.json',
        semantics/'manifest.json',semantics/'predeclared_conditions.json',semantics/'comparison_result_v1.json']
    decision={'schema':'rampq.gate1.acceptance_prerequisites_decision/v1','at_utc':now,'client_date':'2026-10-23',
        'status':'COMPLETED_SYNTHETIC_TRAINING_PRIMITIVES_AND_NATIVE_EVALUATOR_REFERENCE_ONLY',
        'ignore_negative_BCE_primitive_implemented':True,'ignore_loss_synthetic_tests':55,
        'ignore_geometry_loader_augmentation_assigner_trainer_integration_complete':False,
        'ignore_policy_is_new_engineering_choice_not_official_AP_rule':True,
        'retained_logit_gradients_preserved_at_fixed_inputs':True,'shared_model_parameter_gradients_unchanged_claimed':False,
        'historical_loss_tal_byte_identity_proven':False,'fp32_state_format_implemented':True,'fp32_format_synthetic_tests':10,
        'live_FP32_EMA_hook_integrated':False,'concurrent_atomic_FP32_capture_proven':False,
        'old_fp16_checkpoint_precision_recovered':False,'full_precision_training_origin_proven':False,
        'official_native_synthetic_oracle_executed':True,'official_native_fixture_count':20,'native_case_errors':0,
        'official_source_commit':launcher['source_commit'],'native_runtime':launcher['runtime']['version'],
        'official_native_functions_identity_checked':6,'native_output_audit':audit['status'],
        'native_output_structure_auditor_synthetic_tests':39,'existing_Python_semantics_regression_tests':37,
        'partial_comparison_adapter_synthetic_tests':27,'partial_semantics_status':result['status'],
        'partial_intermediate_matrices_checked':result['matrix_checks'],
        'partial_intermediate_matrices_equal':result['equal_matrix_checks'],
        'partial_intermediate_matrices_different':result['different_matrix_checks'],
        'python_prefilter_and_single_image_matcher_equal_on_fixed_synthetic_cases':equal,
        'AP_metric_values_compared_between_Python_and_native':False,'full_Python_AP_implemented':False,
        'official_AP_equivalence_certified':False,'real_dataset_AP_evaluated':False,
        'training_started_this_stage':False,'model_executed_this_stage':False,
        'original_gate1_dtype_guard':'FAIL_FIXED_TOLERANCES','old_gate05_numeric_guard':'FAILED_UNCHANGED',
        'accepted_float_baseline_ready':False,'gate1_scientific_pass':False,
        'selection_detector_inference':False,'holdout_opened_or_listed':False,
        'backend_export_Orin_TensorRT_or_performance_qualified':False,
        'evidence':[identity(path) for path in evidence],
        'limits':[
            'Synthetic arithmetic and source/runtime outputs only; no model, actual training, real-image annotation scoring, Selection or Holdout.',
            'The ignore primitive requires correctly transformed post-augmentation boxes; augmentation/loader/criterion/trainer integration remains absent.',
            'Retained logit derivatives do not imply unchanged shared-parameter gradients or proven training benefit.',
            'Current loss/tal snapshots were not individually pinned in the historical training contract; no retrospective byte identity is asserted.',
            'The FP32 format cannot establish training provenance, recover old half precision or atomically capture concurrent EMA updates.',
            'Octave11.3.0 executes unchanged official .m code and original image mean2, not MATLAB cross-runtime equivalence.',
            'The finite fixed partial comparison is not full AP/AR aggregation equivalence or general real-data certification.',
            'Native empty-evaluation NaNs stay NaN in MAT; JSON null does not mean zero accuracy.',
            'Neither original numerical guard nor scientific baseline, deployment or performance status is changed.'
        ]}
    new(OUT/'decision.json',decision)
    state=dict(read(BASE/'current_execution_state_v8.json'))
    state.update(schema='rampq.gate1.execution_state/v9',as_of_utc=now,
        status='COMPLETED_SYNTHETIC_PREREQUISITES_NATIVE_ORACLE_REFERENCE_RESEARCH_ACCEPTANCE_PENDING',
        phase='TRAINING_INTEGRATION_AND_FULL_AP_EQUIVALENCE_PENDING',
        previous_state_snapshot='artifacts/gate1/current_execution_state_v8.json',
        ignore_negative_loss_primitive_synthetic_tests=55,ignore_aware_training_pipeline_integrated=False,
        fp32_reference_format_synthetic_tests=10,live_fp32_reference_hook_integrated=False,
        official_native_synthetic_oracle_executed=True,official_native_synthetic_fixture_count=20,native_oracle_case_errors=0,
        official_native_runtime='GNU Octave '+launcher['runtime']['version'],
        official_native_saved_output_audit=audit['status'],
        partial_Python_native_intermediate_comparison=result['status'],partial_Python_native_matrices_checked=result['matrix_checks'],
        partial_Python_native_matrices_different=result['different_matrix_checks'],
        full_Python_AP_implemented=False,official_AP_implemented=False,official_oracle_verified=False,
        official_oracle_verified_field_scope='Full Python AP/native equivalence remains uncertified; native synthetic reference is separately recorded as executed.',
        accepted_float_baseline_ready=False,gate1_scientific_pass=False,
        report='48_Gate1_训练前置组件与官方原生评分验证.md',decision='artifacts/gate1/20261023_acceptance_prerequisites_delivery_v1/decision.json',
        next_steps=[
            'Freeze a tested geometry/augmentation/collation/criterion/trainer contract before using the ignore primitive on actual training.',
            'Integrate atomic live FP32 EMA capture with architecture reconstruction and independently bound provenance before accepting a scientific reference.',
            'Implement and compare full AP/AR aggregation against the acquired native synthetic reference, preserving upstream quirks or clearly naming another metric.',
            'Freeze candidate/budget/independent-validation contracts; keep Selection metadata-only and Holdout opaque.',
            'Separately qualify export/backend/board behavior before any deployment or performance claims.'],
        note='Native official-source synthetic reference acquired and partial comparison recorded. Training primitives are not integrated. Old dtype failures and scientific/deployment limits are preserved.')
    new(BASE/'current_execution_state_v9.json',state)
    partial_verdict=(f"全部{result['matrix_checks']}个中间矩阵严格一致" if equal else
        f"{result['matrix_checks']}个矩阵中有{result['different_matrix_checks']}个不一致；差异原样保存，未修改旧Python实现或官方输出")
    report=f'''# Gate1 训练前置组件与官方原生评分验证

**本轮取得了官方评分源码的原生参考输出：20组合成用例在GNU Octave {launcher['runtime']['version']}上实际运行成功，0个用例错误，独立输出审计通过。** 同时完成忽略负样本损失与FP32保存格式两个前置组件。它们还没有接入实际训练，Gate1科学基线仍未验收。

{link('分项结论','artifacts/gate1/20261023_acceptance_prerequisites_delivery_v1/decision.json')}；{link('状态快照v9','artifacts/gate1/current_execution_state_v9.json')}；{link('前一轮109图混合原型报告','47_Gate1_混合精度独立推理原型验证.md')}。

## 官方原生参考已经取得

本轮从官方发布资产获取Octave 11.3.0便携包并核对发布SHA-256；没有声称验证PGP签名信任链。空间检查后，只提取2890个命令行必需文件、640420084 bytes，没有全量安装、运行post-install或修改系统PATH/注册表。mean2使用同归档原image-2.20.0函数；原官方11份源码及旧合成runner保持字节身份。

输入始终是已冻结的20组合成矩阵：11组AP/AR聚合、4组像素忽略过滤、5组单图匹配。六个官方函数的解析路径逐一与原文件核对，程序实际调用原dropObjectsInIgr、evalRes、calcAccuracy及其依赖；没有替代函数或手写预期AP答案。MAT与JSON、运行版本、依赖清单及控制台记录均已保存。

独立审计核对2890个运行时成员、1535个隔离MAT读取依赖文件、20个用例顺序/状态、空矩阵维度、匹配矩阵和七项指标的序列化一致性。39项审计器合成测试通过。A03/A04空评价类别产生的原生NaN保留在MAT中，JSON的null不能解释成0分。这取得的是固定Octave版本执行官方源码的参考，不证明所有MATLAB版本或真实数据的等价。

{link('原生执行回执','artifacts/gate1/20261023_native_oracle_runtime_v1/execution_01/launcher_receipt.json')}；{link('原生MAT结果','artifacts/gate1/20261023_native_oracle_runtime_v1/execution_01/oracle_results.mat')}；{link('独立输出审计','artifacts/gate1/20261023_native_oracle_comparison_preparation_v1/native_output_audit_v1.json')}。

## Python局部语义对照

先冻结全部用例、字段、行顺序及零绝对/相对容差，再比较现有Python的忽略过滤与official-style单图匹配：20组用例的22张合成图过滤/score反转矩阵，以及5组匹配×10类别的GT/DT矩阵，**{partial_verdict}**。比较适配器27项合成测试通过，既有Python37项回归也通过。

这次没有调用或修改项目另行定义的IoU优先目标匹配器。局部矩阵一致仅覆盖这些固定合成输入，不是所有输入的证明；Python完整AP/AR聚合仍未实现，也没有比较Python与原生AP数值。类别按图重复加权、输入顺序先截断、score=0分母、跨图排序及VOC精度包络等聚合细节，仍需完整实现与对照。

{link('预先冻结的比较条件','artifacts/gate1/20261023_native_python_semantics_v1/predeclared_conditions.json')}；{link('实际局部比较结果','artifacts/gate1/20261023_native_python_semantics_v1/comparison_result_v1.json')}。

## 两个训练前置组件

**忽略负样本分类损失原语：55项CPU合成测试通过。** 对最终增强图像内、anchor中心落在忽略框且未被分配为positive的位置，屏蔽整行分类BCE；positive位置保留全部类别项，原target_scores求和分母不变。无ignore时值和梯度逐位一致；被忽略negative的logit梯度归零，固定输入下其余logit梯度不变。共享模型参数汇集多个位置的梯度仍会改变，不能据此宣称参数更新不变或训练效果改善。

anchor中心、半开边界、class-agnostic等是明确的新工程政策，不是官方AP规则。组件要求ignore几何已与全部增强同步；Mosaic等增强、sidecar拼批、assigner/criterion/trainer尚未接入，不能称完整ignore-aware训练已解决。当前loss/tal源码支持缺口判断，但旧训练合同没有分别固定这两个文件，不能倒推其历史逐字身份；已固定的dataset/augment/tasks与当前快照相符。

**FP32参考状态保存格式：10项CPU合成测试通过。** 原本FP32、不能被FP16精确表示的人工数值保存后逐位一致；组件拒绝自动提升半精度来源。它仍不能识别调用者此前先half再float，也不证明声明的训练来源；旧best/last半精度EMA丢失的精度不能恢复。尚无live EMA原子捕获、架构重建或真实训练hook，当前不是已验收科学参考，也不是含优化器的恢复checkpoint。

{link('损失组件与来源审查','artifacts/gate1/20261023_ignore_loss_preparation_v1/README.md')}；{link('FP32格式验证','artifacts/gate1/20261023_fp32_reference_preparation_v1/README.md')}；{link('两组件独立复核','artifacts/gate1/20261023_acceptance_prerequisites_review_v1/source_review.json')}。

初期的临时目录ACL、MAT测试配置以及下载超时/不完整尝试均保留原日志；最终测试使用明确工程临时目录，完整归档在续传后通过官方SHA。未把失败尝试删除或算成通过。

## 还需完成的验收

下一步是接入并测试增强几何/拼批/训练损失，以及保存前live FP32 EMA的原子捕获和来源核对，再补完整AP/AR聚合比对与独立验证合同。本轮没有模型推理或重新训练，没有读取真实图像/数据标注、Selection检测结果或Holdout内容；没有导出、INT8、TensorRT、Orin、速度或功耗测量。

已有109图混合原型数值结果保持通过；原全FP16门槛仍为 `FAIL_FIXED_TOLERANCES`，旧Gate0.5数值记录仍为 `FAILED_UNCHANGED`。官方源码合成运行成功与两个组件测试通过，都不替代科学基线、真实数据集AP或板端部署验收。
'''
    report_path=ROOT/'48_Gate1_训练前置组件与官方原生评分验证.md'
    assert not report_path.exists();report_path.write_text(report,encoding='utf-8')
    pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v9.json',
        'snapshot_sha256':sha(BASE/'current_execution_state_v9.json'),'note':'This pointer may be updated on later work. Referenced timestamped snapshots are preserved.'}
    (BASE/'latest_execution_state.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':identity(report_path),'state':identity(BASE/'current_execution_state_v9.json'),
        'decision':identity(OUT/'decision.json'),'partial_comparison_status':result['status']},ensure_ascii=False))


if __name__=='__main__':main()
