"""Publish bounded synthetic integration results without changing older evidence."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
ATTEMPT=HERE/'attempt_03'
REVIEW=ROOT/'artifacts/gate1/20261023_training_integration_review_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
    assert not p.exists(), f'Do not overwrite immutable evidence: {p}'
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def binding(p):return {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def link(label,p):return f'[{label}]({p.resolve().as_posix()})'

receipt=read(ATTEMPT/'training_receipt.json')
capture=read(ATTEMPT/'fp32_capture/epoch_0001/capture_receipt.json')
events=[json.loads(line) for line in (ATTEMPT/'events.jsonl').read_text(encoding='utf-8').splitlines()]
capture_settings=next(v['runtime_cpu_settings'] for v in events if v['event']=='save_model_entry_before_capture')
assert receipt['status'].startswith('PASS_ACTUAL_SYNTHETIC_CPU_TRAINER')
assert receipt['ema_updates']==2 and len(receipt['optimizer_steps'])==2
assert receipt['external_integration_registration_disabled_in_process'] is True
assert receipt['threads']==1 and capture_settings['intraop_threads']==1 and capture_settings['interop_threads']==1
assert capture['reconstruction']['output_values_dtypes_shapes_exact'] is True
review_paths=[REVIEW/'source_review.json',REVIEW/'fp32_source_review.json',
             ROOT/'artifacts/gate1/20261023_training_runner_source_review_v3/runner_source_review.json',
             REVIEW/'result_review_v3.json',
             REVIEW/'synthetic_output_audit_v3/audit_result.json']
for p in review_paths:
    assert p.is_file(),f'Final review missing: {p}'
    value=read(p)
    assert str(value.get('status','')).startswith('PASS'),f'Review not passing: {p}'

components={
 'dataset':ROOT/'artifacts/gate1/20261023_ignore_dataset_integration_v1/manifest.json',
 'criterion_trainer':ROOT/'artifacts/gate1/20261023_ignore_trainer_integration_v1/manifest.json',
 'fp32_capture':ROOT/'artifacts/gate1/20261023_fp32_live_capture_v1/manifest.json'}
decision={
 'schema':'rampq.gate1.synthetic_training_integration_decision/v1',
 'as_of_utc':datetime.now(timezone.utc).isoformat(),
 'status':'COMPLETED_ACTUAL_SYNTHETIC_CPU_TRAINER_IGNORE_AND_PRE_HALF_FP32_CAPTURE_ONLY',
 'accepted_attempt':'attempt_03','accepted_attempt_manifest':binding(ATTEMPT/'manifest.json'),
 'accepted_training_receipt':binding(ATTEMPT/'training_receipt.json'),
 'contract':binding(HERE/'training_contract_v3.json'),'preparation':binding(HERE/'preparation_manifest_v3.json'),
 'component_manifests':{k:binding(p) for k,p in components.items()},
 'independent_reviews':[binding(p) for p in review_paths],
 'synthetic_component_tests':{'dataset':45,'criterion_trainer':51,'fp32_capture':23,'total':119},
 'actual_training':{'epochs':1,'train_images':4,'validation_images_built':2,'task_AP_computed':False,
    'random_initialized_model':'YOLO11n_nc10','model_parameters':receipt['model_parameters'],
    'optimizer_steps':2,'ema_updates':2,'changed_parameter_tensors':receipt['changed_parameter_tensors'],
    'changed_parameter_elements_runtime_observed':receipt['changed_parameter_elements'],
    'changed_element_count_independently_recomputed_from_saved_raw_tensors':False,
    'positive_anchors_per_batch':[v['assigned_positive_anchors'] for v in receipt['actual_criterion_batches']],
    'ignored_negative_anchors_per_batch':[v['ignored_negative_anchors'] for v in receipt['actual_criterion_batches']],
    'positive_rows_preserved':True,'device':'cpu','amp':False,'workers':0,'threads':1,
    'cpu_settings_at_capture':capture_settings,'cpu_settings_after_train_return':receipt['runtime_cpu_settings'],
    'each_optimizer_step_thread_setting_asserted':True,'each_optimizer_step_full_backend_flags_logged':False},
 'fp32_capture':{'live_EMA_already_FP32':True,'ema_updates':2,'before_native_half_serialization':True,
    'state_and_update_count_unchanged':True,'strict_architecture_reconstruction':True,
    'analytic_CPU_input_outputs_exact':True,'independent_state_only_CPU_reconstruction':True,
    'optimizer_resume_saved':False,'arbitrary_external_concurrent_writers_proven_absent':False,
    'GPU_capture_qualified':False,'scientific_reference_accepted':False},
 'preserved_attempt_01':{'functional_training_checks_passed':True,'final_execution_acceptance':False,
    'reason':'sync/logger settings did not suppress platform callback registration',
    'external_connection_or_upload_observed_or_proven':False,
    'superseded_by_process_local_registration_disable_and_base_callback_inventory_in_attempt_02_and_03':True},
 'preserved_attempt_02':{'functional_training_checks_passed':True,'final_execution_acceptance':False,
    'hardcoded_threads_1_receipt_inaccurate':True,'effective_intraop_threads_diagnosed':8,
    'independent_1_thread_output_exact':False,'independent_8_thread_output_exact':True,
    'reason':'Trainer select_device(cpu) reset thread count after the original pre-constructor setting',
    'superseded_by_observed_post_constructor_thread_enforcement_in_attempt_03':True,
    'exact_output_guard_relaxed':False},
 'execution_scope':'fixed synthetic fixture, default Dataset factory, one CPU epoch, explicit split preflight',
 'universal_split_or_custom_factory_boundary_certified':False,
 'real_dataset_training_with_ignore_completed':False,'real_dataset_AP_computed':False,
 'selection_detector_inference':False,'holdout_opened_or_listed':False,
 'old_checkpoints_modified':False,'accepted_float_baseline_ready':False,'gate1_scientific_pass':False,
 'original_gate1_dtype_guard':'FAIL_FIXED_TOLERANCES','old_gate05_numeric_guard':'FAILED_UNCHANGED',
 'next_steps':[
   'Freeze a NEW real-data contract with explicit invalid-ignore rejection/quarantine rules and separately verified train/training_val membership; do not reuse the synthetic-only runner as real-data authorization.',
   'Qualify live FP32 capture for the intended CUDA training lifecycle and preserve full-precision EMA before any half serialization; optimizer/resume is a separate contract.',
   'Bind independent validation, prediction submission order and candidate/budget/statistical commitments before scientific AP or baseline qualification.',
   'Qualify export, TensorRT, Orin and performance separately while preserving both original numerical failures.']}
write(HERE/'decision.json',decision)

old_path=ROOT/'artifacts/gate1/current_execution_state_v10.json'
state=deepcopy(read(old_path))
state.update(schema='rampq.gate1.execution_state/v11',as_of_utc=decision['as_of_utc'],
 status=decision['status'],phase='REAL_DATA_CONTRACT_CUDA_CAPTURE_AND_INDEPENDENT_QUALIFICATION_PENDING',
 previous_state_snapshot=old_path.relative_to(ROOT).as_posix(),
 report='50_Gate1_合成训练管线与FP32保存接入验证.md',
 decision=(HERE/'decision.json').relative_to(ROOT).as_posix(),next_steps=decision['next_steps'],
 new_training_started_this_stage=True,new_training_in_this_stage_scope='synthetic_only_CPU_1_epoch_4_train_images',
 new_model_inference_executed_this_stage=True,
 note='Actual one-epoch CPU synthetic Trainer integration passed; original 30-epoch candidate and both frozen numerical failures remain unchanged. No real-data ignore-aware training or scientific qualification accepted.')
for key in ('ignore_aware_training_pipeline_integrated','live_fp32_reference_hook_integrated'):
    state.setdefault('deprecated_ambiguous_fields_from_previous_snapshot',{})[key]=state.pop(key,None)
state.update(synthetic_ignore_Dataset_criterion_Trainer_integration_passed=True,
 real_dataset_ignore_aware_training_pipeline_qualified=False,
 controlled_CPU_live_FP32_capture_integrated=True,CUDA_live_FP32_capture_qualified=False,
 scientific_FP32_training_reference_accepted=False,
 synthetic_training_attempt=(ATTEMPT/'training_receipt.json').relative_to(ROOT).as_posix(),
 synthetic_training_epochs=1,synthetic_training_images=4,synthetic_training_optimizer_steps=2,
 synthetic_training_EMA_updates=2,synthetic_integration_component_tests=119,
 independent_synthetic_FP32_reconstruction_passed=True,
 old_gate05_numeric_guard='FAILED_UNCHANGED',same_backend_dtype_guard='FAIL_FIXED_TOLERANCES',
 gate1_scientific_pass=False,accepted_float_baseline_ready=False)
snapshot=ROOT/'artifacts/gate1/current_execution_state_v11.json'
write(snapshot,state)

report=f'''# Gate1：合成训练管线与FP32保存接入验证

已用新生成的4张训练图、2张training_val图和随机初始化YOLO11n（10类、{receipt['model_parameters']:,}参数），通过实际 `DetectionTrainer.train()` 完成一轮CPU训练。最终采用attempt_03：2次SGD、2次EMA更新；忽略区域已进入真实TAL分类损失，保存前的live FP32 EMA可以按架构重新构建，CPU合成输入输出逐字节一致。该结果只验收这条合成工程链，不是科研基线或板端验收。

## 实际运行与检查

| 项目 | 结果 |
|---|---|
| 运行范围 | CPU、单线程、workers=0、batch=nbs=2、AMP关闭，1 epoch |
| 新数据 | 4张train、2张training_val；显式成员清单、三文件字节/SHA核验；图片ID和图片SHA跨split无交集 |
| 实际更新 | 2次SGD及EMA；两步首个可训练参数各432个元素变化 |
| 整轮参数变化 | 运行时记录{receipt['changed_parameter_tensors']}个参数张量、{receipt['changed_parameter_elements']:,}个元素变化 |
| 两批正anchor | 9、10，正样本分类整行保留 |
| 两批忽略背景anchor | 67、98，分类BCE屏蔽实际触发 |
| FP32捕获 | 第1轮、EMA updates=2、原生half序列化前；源已是FP32，未从half升回 |
| 架构重建 | 受限state-only加载、strict state_dict、全YOLO11n新建；CPU解析合成输入输出exact |
| 三组件测试 | Dataset 45、criterion/Trainer 51、FP32 hook 23，合计119项通过 |
| AP | `SMOKE_NO_TASK_AP`，未计算任务AP |

{link('最终训练回执',ATTEMPT/'training_receipt.json')}；{link('真实运行事件',ATTEMPT/'events.jsonl')}；{link('预先冻结合同',HERE/'training_contract_v3.json')}；{link('FP32捕获回执',ATTEMPT/'fp32_capture/epoch_0001/capture_receipt.json')}。

## 接入内容及修复

Dataset每次取样重新核验manifest、图片、正标签、ignore sidecar的实际字节与SHA；标签和忽略框共用实际loader双轴比例、整数补边及同次flip参数，再输出标准检测字段和ragged ignore区域。只支持显式受限链，不执行复杂增强。score=0、category0、category11来源分开保留；非法、空面积或完全在画布外区域拒绝整样本，不静默删除，也不把质量排除的正框转成ignore。

真实criterion保持当前已固定源码的TAL、box、DFL、分母、权重和batch倍数，只修改未分配且中心落入ignore区域的背景BCE。空ignore的四组有/无GT和classweights测试中，loss、模型全部参数梯度及输入梯度与原框架逐元素相同。真实TAL空GT分支返回浮点零fg_mask，接缝处仅为严格helper转bool，原assignment tuple及box/DFL路径不变。正anchor整行的局部logit梯度保留，不代表共享模型参数的总梯度不变；合成BCE下降也不是精度收益。

FP32 hook在实际训练器保存入口先取live EMA，再调用普通框架保存；普通框架仍会deepcopy后half序列化。EMA更新数和状态在捕获前后不变，架构新建隔离构造随机数，测试也覆盖live状态/cache/RNG不被改变。只保存推理状态，不保存优化器或完整恢复训练状态；主线程与非重入锁不能证明不存在任意外部写者，当前调用协议明确禁止并发EMA写入。

首个attempt_01的功能检查已通过，但源审查发现仅关闭sync/loggers不足以阻止框架注册平台回调。v2/attempt_02在创建Trainer前于本进程禁用集成注册，核对初始回调全部来自框架base后才加入本地记录回调。未修改环境中的框架文件；这是进程路径约束，未声称OS网络沙箱，也没有证据反推attempt_01发生了外联或上传。

独立forward审计随后发现，Trainer构造时的select_device(cpu)把请求的1线程改成8；attempt_01/02回执硬写threads=1不准确。独立进程用1线程重建不exact，用8线程严格exact。未修改或放宽exact检查，保留原错误回执及诊断，最终改用v3/attempt_03：构造后重新固定1线程，训练开始、每次optimizer step、保存前和最终回执都实际断言线程数，开始/保存前/结束后记录backendflags。逐step仅断言线程，不声称其全部backendflags逐步留档。Trainer返回时会unset_deterministic，因此独立重建复现保存入口的实际flags，不用训练返回后flags替代。三次运行共用同一合成fixture，不是三组独立数据证据。

## 独立审计与边界

源审分别检查Dataset/criterion/Trainer和FP32 hook，另对修正版runner复审。结果审计复核冻结输入、输出清单、实际事件、两步更新、mask记录和保存时序；独立加载仅本轮新FP32 state，按保存的架构重新构建CPU模型并核对解析合成输入输出。普通best/last只做字节哈希，未反序列化。没有读取原30轮权重、真实图片或标注、Selection检测输入、Holdout内容。

初末模型仅保存逐张量hash描述和运行时变化计数，独立审计能复核张量hash变化及计数上限/合计，不能从未保存的初末原始张量重算{receipt['changed_parameter_elements']:,}逐元素变化数；TAL计数也属于源码与事件/回执交叉核对，没有独立重跑训练assignment。

当前Trainer只允许synthetic-only、单轮CPU。跨manifest分离由本轮显式preflight证明，任意自定义dataset_factory仍是受信任扩展，不能宣称通用防泄漏；当前使用默认已审Dataset。现有4534张真实训练图是否兼容此非法ignore处置尚未检查，也没有生成新的真实数据quarantine或正式训练合同。CUDA训练及CUDA同步捕获另须资格检查。

{link('组件源审',REVIEW/'source_review.json')}；{link('FP32源审',REVIEW/'fp32_source_review.json')}；{link('实际结果审查',REVIEW/'result_review_v3.json')}；{link('独立state-only重建审计',REVIEW/'synthetic_output_audit_v3/audit_result.json')}。

## 当前状态与下一步

原30轮训练候选、109图混合原型及所有旧封存结果保持原样。原全FP16护栏仍为 `FAIL_FIXED_TOLERANCES`，旧Gate0.5仍为 `FAILED_UNCHANGED`；科研Gate1、独立真实数据AP、导出/TensorRT/Orin及速度功耗均未据此通过。

下一步须冻结真实数据成员与非法ignore/quarantine政策，资格验证目标CUDA训练及live FP32捕获，再按独立验证、提交顺序和候选预算合同开展正式训练与验收。旧half EMA丢失的精度不能恢复。

{link('本轮决策',HERE/'decision.json')}；{link('状态快照v11',snapshot)}。
'''
report_path=ROOT/state['report']
assert not report_path.exists()
report_path.write_text(report,encoding='utf-8')
pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':snapshot.relative_to(ROOT).as_posix(),
         'snapshot_sha256':sha(snapshot),'note':'This pointer may be updated on later work. Referenced timestamped snapshots are preserved.'}
(ROOT/'artifacts/gate1/latest_execution_state.json').write_text(json.dumps(pointer,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':decision['status'],'report':report_path.name,'state_sha256':sha(snapshot),
                  'decision_sha256':sha(HERE/'decision.json')},indent=2))
