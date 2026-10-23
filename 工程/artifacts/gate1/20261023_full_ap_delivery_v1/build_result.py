"""Publish completed bounded AP/native agreement and geometry prerequisites."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/gate1'
HERE=Path(__file__).parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def ident(path):return {'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}


def new(path,value):
 assert not path.exists(),path
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def link(label,relative):return f'[{label}]({(ROOT/relative).as_posix()})'


def main():
 ap=BASE/'20261023_full_ap_preparation_v1'
 audit=BASE/'20261023_full_ap_oracle_audit_v1'
 geometry=BASE/'20261023_ignore_geometry_preparation_v1'
 review_path=BASE/'20261023_full_ap_final_review_v1/source_review.json'
 source_review=read(review_path)
 assert source_review['status'].startswith('PASS') and source_review['issues']==[]
 ap_prep=read(ap/'preparation_receipt.json')
 assert ap_prep['test_count']==54 and ap_prep['test_failures']==0
 assert ap_prep['native_metric_answers_read_before_freeze'] is False
 summary=read(audit/'audit_summary.json');result=read(audit/'comparison_attempt_v1.json')
 assert summary['status']==result['status']=='PASS_DECLARED_SYNTHETIC_FULL_AP_NATIVE_COMPARISON_ONLY'
 assert summary['case_count']==32 and summary['aggregate_case_count']==23 and summary['image_count']==40
 assert summary['matrix_count_exact']==result['matrix_count']==220
 assert summary['metric_count']==result['metric_count']==161
 assert summary['finite_metric_count']==133 and summary['matching_nan_metric_count']==28
 assert summary['auditor_negative_regression_tests_passed']==45
 assert summary['max_finite_metric_absolute_error_percentage_units']==max(row['max_finite_metric_absolute_error'] for row in result['suites'])
 assert result['reader']['dependency_verification']['verified_extracted_member_count']==1535
 native=read(audit/'native_stress_execution_v1/launcher_receipt.json')
 assert native['status']=='EXECUTED_12_NATIVE_STRESS_CASES_ONLY' and native['returncode']==0 and native['native_case_errors']==0
 geometric=read(geometry/'final_tests_receipt.json')
 assert geometric['counts']=={'tests':85,'failures':0,'errors':0,'skipped':0}
 assert read(geometry/'manifest.json')['status']=='SEALED_SYNTHETIC_PRIMITIVES_ONLY_REAL_TRAINING_NOT_INTEGRATED'
 assert (audit/'audit_delivery_manifest.json').is_file()
 evidence=[ap/'preparation_manifest.json',ap/'preparation_receipt.json',
  audit/'inputs_manifest.json',audit/'comparator_preparation_manifest.json',audit/'comparison_contract_v1.json',
  audit/'comparison_attempt_v1.json',audit/'audit_summary.json',audit/'audit_delivery_manifest.json',
  audit/'native_stress_execution_v1/manifest.json',geometry/'manifest.json',geometry/'final_tests_receipt.json',
  geometry/'source_review.json',review_path,HERE/'native_constants_probe.json',HERE/'prior_delivery_identity_check.json']
 now=datetime.now(timezone.utc).isoformat()
 decision={'schema':'rampq.gate1.full_ap_and_geometry_decision/v1','at_utc':now,'client_date':'2026-10-23',
  'status':'COMPLETED_DECLARED_SYNTHETIC_AP_NATIVE_COMPARISON_AND_GEOMETRY_PRIMITIVES_ONLY',
  'full_Python_AP_implemented':True,'source_faithful_AP_AR_mirror_implemented':True,
  'implementation_synthetic_tests':54,'independent_comparator_negative_tests':45,
  'old_native_reference_cases_reused':20,'new_native_stress_cases_executed':12,'native_stress_case_errors':0,
  'declared_comparison_case_count':32,'aggregate_case_count':23,'synthetic_image_count':40,
  'zero_tolerance_intermediate_matrices':220,'final_AP_AR_metric_count':161,'finite_metric_count':133,
  'matching_nan_metric_count':28,'max_finite_metric_error_percentage_points':summary['max_finite_metric_absolute_error_percentage_units'],
  'metric_comparison_tolerance':summary['metric_tolerance'],'matrix_comparison_tolerance':summary['matrix_tolerance'],
  'declared_synthetic_output_agreement':True,'native_runtime':'GNU Octave 11.3.0',
  'official_source_commit':'005445782213e20cb91bc50a597db3dd949e749a',
  'native_internal_AP_AR_tables_compared':False,'arbitrary_input_equivalence_certified':False,
  'real_data_AP_computed':False,'geometry_primitives_implemented':True,'geometry_CPU_synthetic_tests':85,
  'geometry_crosschecks':geometric['tests_include'],'geometry_policy_is_new_restricted_training_proposal':True,
  'geometry_loader_Dataset_Format_Trainer_criterion_integration_complete':False,
  'existing_4534_train_images_or_sidecars_checked_for_new_policy':False,
  'actual_file_hash_or_split_membership_proven_by_geometry_helper':False,
  'live_FP32_reference_capture_integrated':False,'new_training_or_model_inference_executed':False,
  'source_review_status':source_review['status'],'original_dtype_guard_modified':False,
  'original_gate1_dtype_guard':'FAIL_FIXED_TOLERANCES','old_gate05_numeric_guard':'FAILED_UNCHANGED',
  'accepted_float_baseline_ready':False,'gate1_scientific_pass':False,
  'selection_detector_inference':False,'holdout_opened_or_listed':False,
  'export_TensorRT_Orin_or_performance_qualification_executed':False,
  'evidence':[ident(path) for path in evidence],
  'limits':[
   'The old20 native references are reused and overlap previous local-semantic checks; this is not32 new independent datasets or model trials.',
   'The32 fixed synthetic cases cover declared matrix domain and chosen Octave11.3.0 stable-tie behavior, not arbitrary inputs or MATLAB-version equivalence.',
   'Only native-exported intermediate matrices and seven final metrics are compared; native internal AP10x10 and AR10x10x4 tables are not exported/independently verified.',
   'The new tiny double scalar metric tolerance does not modify either frozen model dtype guard.',
   'Restricted resize/LetterBox/flips and ragged sidecar collation are primitives, not complete real training integration.',
   'Zero/empty publisher geometry and precision-underflow rejection require a new quarantine/data contract; the existing4534 training images were not inspected.',
   'Metadata identity shape checks are not file digest verification or true split membership; those remain future loader duties.',
   'No real AP, new model training, Selection/Holdout scoring, export/backend/board or performance result is acquired.'
  ]}
 new(HERE/'decision.json',decision)
 state=dict(read(BASE/'current_execution_state_v9.json'))
 removed={key:state.pop(key) for key in ('official_AP_implemented','official_oracle_verified','official_oracle_verified_field_scope','official_native_runtime_found_in_checked_locations') if key in state}
 state.update(schema='rampq.gate1.execution_state/v10',as_of_utc=now,
  status='COMPLETED_BOUNDED_FULL_AP_NATIVE_COMPARISON_AND_IGNORE_GEOMETRY_PRIMITIVES',
  phase='REAL_TRAINING_INTEGRATION_FP32_CAPTURE_AND_INDEPENDENT_QUALIFICATION_PENDING',
  previous_state_snapshot='artifacts/gate1/current_execution_state_v9.json',
  deprecated_ambiguous_fields_from_previous_snapshot=removed,
  deprecated_field_replacement_note='Implementation, declared synthetic agreement and arbitrary-input certification now have separate explicit fields. Old checked-location runtime search remains a historical v9 observation.',
  full_Python_AP_implemented=True,source_faithful_Python_AP_mirror_implemented=True,
  declared_synthetic_AP_native_agreement=True,declared_synthetic_comparison_cases=32,
  declared_synthetic_exact_matrices=220,declared_synthetic_final_metrics=161,
  matching_nan_metrics=28,arbitrary_input_AP_equivalence_certified=False,real_dataset_AP_computed=False,
  geometry_synthetic_tests=85,restricted_ignore_geometry_and_ragged_collation_implemented=True,
  ignore_aware_training_pipeline_integrated=False,live_fp32_reference_hook_integrated=False,
  new_training_started_this_stage=False,new_model_inference_executed_this_stage=False,
  accepted_float_baseline_ready=False,gate1_scientific_pass=False,
  report='49_Gate1_AP聚合原生对照与忽略几何验证.md',decision='artifacts/gate1/20261023_full_ap_delivery_v1/decision.json',
  next_steps=[
   'Freeze and implement the restricted loader/Dataset/Format/criterion/Trainer integration and explicit invalid-ignore quarantine policy; verify synthetic batches before actual training.',
   'Integrate atomic live FP32 EMA capture with architecture reconstruction and independently bound source provenance; old half EMA precision cannot be recovered.',
   'Freeze real-data prediction-submission ordering, preprocessing and validation/data-independence contracts before computing or claiming scientific AP.',
   'Preserve candidate/budget/statistical precommitments; keep Selection metadata-only and Holdout opaque.',
   'Qualify export/backend/board behavior separately before deployment or performance claims.'],
  note='Complete AP/AR source mirror and declared synthetic native comparison now implemented. Geometry primitives are synthetic-tested but real training/reference capture remain unintegrated. Frozen old dtype failures are preserved.')
 new(BASE/'current_execution_state_v10.json',state)
 rows=[]
 for suite in result['suites']:
  rows.append(f"| {'旧20组封存参考' if suite['id']=='frozen20' else '新增12组压力用例'} | {suite['matrix_count']} | {suite['metric_count']} | {suite['metric_nan_count']} | {suite['max_finite_metric_absolute_error']:.6g} |")
 report=f'''# Gate1 AP聚合原生对照与忽略几何验证

**完整Python AP/AR聚合已实现，并通过固定32组合成用例的原生对照：220/220个中间矩阵零容差一致，161项最终指标满足预先冻结的双精度容差，28处NaN图样完全一致。** 同时完成忽略区域的受限几何和拼批组件，85项CPU合成测试通过。真实训练管线和live FP32参考保存尚未接入，Gate1科学验收仍未通过。

{link('分项结论','artifacts/gate1/20261023_full_ap_delivery_v1/decision.json')}；{link('状态快照v10','artifacts/gate1/current_execution_state_v10.json')}；{link('前一轮报告48','48_Gate1_训练前置组件与官方原生评分验证.md')}。

## 完整聚合与原生参考

新模块逐段对应固定官方commit的calcAccuracy、VOCap、evalRes、compOas，原官方源码没有改动。按原行为先截断全类别提交顺序、再筛类别及按得分排序；保留score=0忽略GT进入召回分母、忽略检测既非TP也非FP、类别按出现图频重复加权、AP只取maxDets=500及VOC精度包络积分。没有替换成另一套COCO口径，也没有执行NMS。

实现支持声明的有限数值矩阵域，图内及跨图同分使用稳定顺序。原生0.5:0.05:0.95十个IoU阈值已做纯常量探测，与Python公式逐float64十六进制一致；这只固定所用Octave版本，不能推广到所有MATLAB版本。源码与54项合成测试先冻结，再由独立比较器读取原生答案。比较器45项负例测试通过。

旧20组沿用上一轮封存的原生输出；本轮另预先固定12组压力输入，使用未修改的官方函数重新原生执行，12/12成功、0用例错误。压力输入包括超过500个预测、输入顺序反转、跨类别截断、同分与等IoU、score=0反复吸收、类别图频不均、空图、category11、重叠忽略区域、越界坐标和分数召回端点。没有按运行结果挑选用例，也没有用手写AP答案代替原生结果。

{link('完整聚合源代码','scripts/gate1_official_ap.py')}；{link('冻结实现与54项测试','artifacts/gate1/20261023_full_ap_preparation_v1/preparation_receipt.json')}；{link('新增原生运行回执','artifacts/gate1/20261023_full_ap_oracle_audit_v1/native_stress_execution_v1/launcher_receipt.json')}。

## 实际比对结果

矩阵维度、行顺序和全部元素使用零绝对/相对容差。最终七项AP/AR是百分数单位的float64标量，预先固定 `abs(Python-native) <= 1e-10 + 1e-12×abs(native)`，只允许不同双精度归约路径的舍入差；NaN位置必须一致，Inf拒绝。这一标量容差与原模型FP32/FP16护栏是两项不同检查，原护栏未改。

| 参考集 | 精确矩阵数 | 最终指标数 | NaN指标数 | 最大有限差（个百分点） |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

合计32组、40张合成图，其中23组会产生七项最终指标。133个有限指标全部通过，最大差{summary['max_finite_metric_absolute_error_percentage_units']:.16g}个百分点；28个NaN保持原样，不能解释成0分。旧20组与上一轮局部检查重叠，不能当作新增独立数据证据。

独立比对核对了原官方11份源码、2890个运行时文件、MAT读取依赖的1535个实文件（wheel共有1656个条目，含目录）、启动与输出身份、MAT/JSON序列化、空矩阵和NaN。原生wrapper只导出指定中间矩阵和最终七指标；Python内部10×10 AP及10×10×4 AR表没有原生独立输出对照，不宣称其每个单元已被直接验证。

{link('独立比对摘要','artifacts/gate1/20261023_full_ap_oracle_audit_v1/audit_summary.json')}；{link('逐用例及逐指标结果','artifacts/gate1/20261023_full_ap_oracle_audit_v1/comparison_attempt_v1.json')}；{link('比较政策','artifacts/gate1/20261023_full_ap_oracle_audit_v1/comparison_policy.json')}。

## 忽略几何与批次组件

新组件只支持显式受限链：原图到loader实际输出的双轴缩放比例、同一次LetterBox采样的比例与整数补边、同一次水平/垂直flip决定；最后按图片与sidecar身份及顺序拼成ragged区域，接入此前分类损失原语的合成链测试。未另采随机参数，也不按正框面积/长宽比规则删除残留忽略区域。

85项CPU测试包括10次已安装LetterBox对照、4次RandomFlip对照、3次以造数组替代imread的真实loader完整链，以及两种dtype拼批到loss的边界/positive优先检查。没有加载真实图片或数据标注。仅将旧增强概率设为0仍可能走RandomPerspective等路径；因此受限链是新合同提案，并不是旧30epoch训练已被修复。

score=0、category0和category11来源分别保存；质量排除的正框不静默转成publisher-ignore。零面积、完全在画布外、非法来源及精度转型使正面积丧失等情况拒绝整样本，需另定quarantine和数据合同。没有检查现有4534张训练图是否全部兼容。身份字符串和SHA字段的结构检查，也不证明实际文件哈希或split归属，未来loader仍须核验。

真实Dataset、Format、Trainer、criterion、增强禁用的运行强制检查尚未接入。组件通过不等于完整ignore-aware训练完成，也不证明训练收益。

{link('几何组件说明','artifacts/gate1/20261023_ignore_geometry_preparation_v1/README.md')}；{link('85项测试回执','artifacts/gate1/20261023_ignore_geometry_preparation_v1/final_tests_receipt.json')}；{link('独立源审','artifacts/gate1/20261023_full_ap_final_review_v1/source_review.json')}。

## 当前验收边界

现在已具备完整聚合实现及声明合成用例的原生一致性证据，仍未证明任意输入、跨版本或真实数据AP一致。下一步应冻结并接入真实训练链、非法ignore处置、live FP32 EMA原子保存及架构重建，再按独立数据/提交顺序/候选预算合同进行资格检查。旧FP16 EMA丢失的精度不能恢复。

本轮没有重新训练或运行检测模型，没有Selection检测评分或Holdout内容访问，没有导出、INT8、TensorRT、Orin、速度或功耗测量。既有109图混合原型通过记录保留；原全FP16护栏仍为 `FAIL_FIXED_TOLERANCES`，旧Gate0.5记录仍为 `FAILED_UNCHANGED`。
'''
 report_path=ROOT/'49_Gate1_AP聚合原生对照与忽略几何验证.md'
 assert not report_path.exists();report_path.write_text(report,encoding='utf-8')
 pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v10.json',
  'snapshot_sha256':sha(BASE/'current_execution_state_v10.json'),'note':'This pointer may be updated on later work. Referenced timestamped snapshots are preserved.'}
 (BASE/'latest_execution_state.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'report':ident(report_path),'state':ident(BASE/'current_execution_state_v10.json'),'decision':ident(HERE/'decision.json')},ensure_ascii=False))


if __name__=='__main__':main()
