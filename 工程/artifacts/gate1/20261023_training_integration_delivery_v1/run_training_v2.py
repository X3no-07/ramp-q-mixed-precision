"""Run actual DetectionTrainer.train on six freshly generated synthetic images.

No old checkpoint, real image, Selection/Holdout, board or task AP is involved.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / 'attempt_02'
sys.path.insert(0, str(ROOT/'scripts'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def require(cond, message):
    if not cond:
        raise ValueError(message)

manifest_path = HERE/'preparation_manifest_v2.json'
manifest_sha = sha(manifest_path)
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
for rel, info in manifest['files'].items():
    p = ROOT/rel
    require(p.stat().st_size == info['bytes'] and sha(p)==info['sha256'], f'Frozen input changed: {rel}')
require(not ATTEMPT.exists(),'Attempt output must be new')
ATTEMPT.mkdir()
os.environ.update(YOLO_CONFIG_DIR=str(ATTEMPT/'settings'), YOLO_OFFLINE='true',
                  WANDB_MODE='disabled', COMET_MODE='DISABLED', PYTHONDONTWRITEBYTECODE='1')
(ATTEMPT/'settings').mkdir()
start=time.perf_counter()
events=[]

def event(name, **fields):
    entry={'sequence':len(events),'event':name,**fields}
    events.append(entry)
    with (ATTEMPT/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(entry,allow_nan=False)+'\n')

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    import ultralytics
    ultralytics.settings.update({**{k:False for k in ('sync','wandb','comet','clearml','mlflow','dvc','raytune','tensorboard')},
        'runs_dir':str(ATTEMPT/'runs'),'weights_dir':str(ATTEMPT/'unused_weights')})
    import gate1_fp32_live_capture as C
    from ultralytics.utils import callbacks as framework_callbacks
    # Process-local isolation: the framework's registration dispatcher includes
    # a platform integration whose activation is not guarded by settings.sync.
    # Replace only registration, retaining original on-disk framework sources.
    def disable_external_integration_registration(instance):
        event('external_integration_registration_disabled',disk_sources_modified=False)
    framework_callbacks.add_integration_callbacks=disable_external_integration_registration
    from gate1_ignore_trainer import RestrictedIgnoreTrainer
    contract=json.loads((HERE/'training_contract_v2.json').read_text(encoding='utf-8'))
    cfg=json.loads((HERE/'model_cfg.json').read_text(encoding='utf-8'))
    contract_sha=sha(HERE/'training_contract_v2.json')
    require(C.sources()==contract['runtime_sources'],'Runtime identity drift before training')
    initial_parameters={}
    initial_description={}
    step_receipts=[]

    class CapturingTrainer(C.FP32LiveCaptureMixin, RestrictedIgnoreTrainer):
        def optimizer_step(self):
            parameter=next(p for p in self.model.parameters() if p.requires_grad)
            before=parameter.detach().clone()
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),'Missing/invalid actual gradient')
            gradient_absmax=float(parameter.grad.detach().abs().max())
            ema_before=self.ema.updates
            super().optimizer_step()
            changed=int((before!=parameter.detach()).sum())
            record={'step':len(step_receipts)+1,'ema_before':ema_before,'ema_after':self.ema.updates,
                    'first_trainable_parameter_changed_elements':changed,'gradient_absmax':gradient_absmax,
                    'actual_framework_optimizer_step_returned':True}
            step_receipts.append(record)
            event('actual_optimizer_step_completed',**record)

        def save_model(self):
            event('save_model_entry_before_capture',ema_updates=self.ema.updates,
                  normal_last_exists=self.last.exists())
            result=super().save_model()
            event('save_model_return_after_capture_and_framework_serialization',
                  capture_receipts=len(self._fp32_live_capture.receipts),normal_last_exists=self.last.exists())
            return result

    def on_start(t):
        require(t.amp is False and t.args.amp is False and t.device.type=='cpu','Unexpected precision/device')
        require(t.accumulate==1 and len(t.train_loader)==2,'Expected exactly two optimizer opportunities')
        initial_parameters.update({n:p.detach().clone() for n,p in t.model.named_parameters()})
        initial_description.update(C.FORMAT.describe_state(C.FORMAT.freeze_state(t.model.state_dict())))
        t.configure_fp32_capture(ATTEMPT/'fp32_capture',contract_sha,[1],input_shape=(1,3,64,64),
                                 expected_source_sha256=contract['runtime_sources'])
        event('on_train_start',train_samples=len(t.train_loader.dataset),val_samples=len(t.test_loader.dataset),
              batches=len(t.train_loader),accumulate=t.accumulate,ema_updates=t.ema.updates,
              model_parameters=sum(p.numel() for p in t.model.parameters()),model_random_initialized=True)

    def on_epoch(t):
        t.train_loader.dataset.set_epoch(t.epoch)
        event('on_train_epoch_start',epoch_one_based=t.epoch+1)

    def on_batch(t):
        require(bool(torch.isfinite(t.loss).all()),'Nonfinite actual loss')
        event('on_train_batch_end',epoch_one_based=t.epoch+1,ema_updates=t.ema.updates,
              loss=[float(x) for x in t.loss.detach().flatten()],
              mask_receipt=deepcopy(t.model.criterion.last_mask_receipt))

    def on_save(t):
        require(len(t._fp32_live_capture.receipts)==1 and t.last.is_file(),'Missing captured/ordinary save')
        event('on_model_save',capture_receipt_exists=(ATTEMPT/'fp32_capture/epoch_0001/capture_receipt.json').is_file(),
              framework_half_checkpoint_sha256=sha(t.last))

    trainer=CapturingTrainer(training_contract=contract,model_cfg=cfg,fp32_capture_callback=None,
                             overrides=contract['overrides'])
    callback_inventory={k:[f.__module__+'.'+f.__qualname__ for f in funcs] for k,funcs in trainer.callbacks.items()}
    require(all(f.__module__=='ultralytics.utils.callbacks.base' for funcs in trainer.callbacks.values() for f in funcs),
            'Unexpected non-base integration callback registered')
    write(ATTEMPT/'base_callback_inventory.json',callback_inventory)
    event('base_callbacks_verified_before_custom_hooks',external_integration_callbacks=0)
    trainer.add_callback('on_train_start',on_start)
    trainer.add_callback('on_train_epoch_start',on_epoch)
    trainer.add_callback('on_train_batch_end',on_batch)
    trainer.add_callback('on_model_save',on_save)
    event('actual_DetectionTrainer_train_call_enter')
    trainer.train()
    event('actual_DetectionTrainer_train_call_return')
    changed={n:int((initial_parameters[n]!=p.detach()).sum()) for n,p in trainer.model.named_parameters()}
    require(len(step_receipts)==2 and trainer.ema.updates==2,'Expected exactly two real optimizer/EMA steps')
    require(sum(changed.values())>0,'No actual trainable parameter changed')
    require(all(v['first_trainable_parameter_changed_elements']>0 for v in step_receipts),'Each step must update a parameter')
    observations=trainer.model.criterion.observed_batches
    require(len(observations)==2,'Expected criterion execution once per actual batch')
    require(any(v['assigned_positive_anchors']>0 and v['ignored_negative_anchors']>0 for v in observations),
            'Actual criterion did not exercise both positive priority and ignored negatives')
    require(all(v['positive_rows_preserved'] for v in observations),'Positive rows lost')
    require(all(bool(torch.isfinite(v).all()) for v in trainer.model.state_dict().values() if v.is_floating_point()),'Nonfinite final state')
    require(len(trainer._fp32_live_capture.receipts)==1,'Capture count differs')
    capture=trainer._fp32_live_capture.receipts[0]
    require(capture['phase']==C.PHASE and capture['ema_updates_before']==2 and capture['ema_updates_after']==2,'Capture phase/updates differ')
    require(capture['reconstruction']['output_values_dtypes_shapes_exact'],'Reconstructed output differs')
    require(C.sources()==contract['runtime_sources'],'Runtime source identity drift after training')
    write(ATTEMPT/'initial_model_state_description.json',initial_description)
    write(ATTEMPT/'final_model_state_description.json',C.FORMAT.describe_state(C.FORMAT.freeze_state(trainer.model.state_dict())))
    write(ATTEMPT/'changed_parameters.json',changed)
    receipt={'schema':'rampq.gate1.actual_synthetic_trainer_receipt/v1',
        'status':'PASS_ACTUAL_SYNTHETIC_CPU_TRAINER_WITH_IGNORE_AND_PRE_HALF_FP32_CAPTURE_ONLY',
        'preparation_manifest_sha256':manifest_sha,'training_contract_sha256':contract_sha,
        'actual_train_call_returned':True,'epochs':1,'train_images':4,'validation_images_built':2,
        'validation':'SMOKE_NO_TASK_AP','device':'cpu','amp':False,'threads':1,
        'new_random_model':'YOLO11n_nc10','model_parameters':sum(p.numel() for p in trainer.model.parameters()),
        'optimizer_steps':step_receipts,'ema_updates':trainer.ema.updates,
        'changed_parameter_tensors':sum(v>0 for v in changed.values()),'changed_parameter_elements':sum(changed.values()),
        'actual_criterion_batches':observations,'integration_events':trainer.integration_events,
        'capture_receipt':'fp32_capture/epoch_0001/capture_receipt.json',
        'ordinary_framework_half_checkpoint_saved':True,'normal_checkpoint_never_used_as_fp32_reference':True,
        'external_integration_registration_disabled_in_process':True,
        'base_callback_inventory_verified_before_custom_hooks':True,'framework_disk_sources_modified':False,
        'old_checkpoint_read_or_recovered':False,'real_image_or_label_inputs':False,
        'selection_or_holdout_accessed':False,'real_dataset_AP_computed':False,
        'scientific_gate_pass':False,'board_or_CUDA_capture_qualified':False,
        'old_gate05_numeric_guard':'FAILED_UNCHANGED','old_gate1_dtype_guard':'FAIL_FIXED_TOLERANCES',
        'elapsed_seconds':time.perf_counter()-start}
    write(ATTEMPT/'training_receipt.json',receipt)
    members={p.relative_to(ATTEMPT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
             for p in sorted(ATTEMPT.rglob('*')) if p.is_file()}
    write(ATTEMPT/'manifest.json',{'schema':'rampq.gate1.actual_synthetic_training_outputs/v1','files':members})
    print(json.dumps({'status':receipt['status'],'optimizer_steps':2,'ema_updates':trainer.ema.updates,
                      'criterion_batches':observations,'changed_parameter_elements':sum(changed.values()),
                      'manifest_sha256':sha(ATTEMPT/'manifest.json')},indent=2))
except Exception as error:
    write(ATTEMPT/'failure_receipt.json',{'status':'FAILED_ACTUAL_SYNTHETIC_TRAINER_ATTEMPT',
          'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
          'elapsed_seconds':time.perf_counter()-start})
    raise
