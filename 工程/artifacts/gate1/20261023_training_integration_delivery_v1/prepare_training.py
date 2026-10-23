"""Create new synthetic inputs and freeze a concrete CPU Trainer invocation."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['COMET_MODE'] = 'DISABLED'
config_dir = HERE / 'preparation_settings'
config_dir.mkdir(exist_ok=False)
os.environ['YOLO_CONFIG_DIR'] = str(config_dir)

import yaml
import ultralytics
ultralytics.settings.update({k: False for k in ('sync','wandb','comet','clearml','mlflow','dvc','raytune','tensorboard')})
from gate1_ignore_dataset import build_synthetic_fixture
from gate1_ignore_geometry import RESTRICTED_OPTIONS
from gate1_fp32_live_capture import sources

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

fixture = build_synthetic_fixture(HERE / 'synthetic_fixture_v1', train_count=4, val_count=2)
split_rows = {role:json.loads(Path(fixture[prefix+'_manifest']).read_text(encoding='utf-8'))['samples']
              for role,prefix in [('train','train'),('training_val','val')]}
for field in ('image_id','image_sha256'):
    values = {role:{r['image_id'] if field=='image_id' else r['image']['sha256'] for r in rows}
              for role,rows in split_rows.items()}
    assert not values['train'].intersection(values['training_val']), 'Synthetic split identity overlap'
write(HERE/'split_identity_preflight.json', {'status':'PASS_EXPLICIT_SYNTHETIC_SPLIT_IDS_AND_IMAGE_SHA_DISJOINT',
      'train_images':4,'training_val_images':2,'identity_intersection':[], 'image_sha256_intersection':[],
      'real_dataset_membership_certified':False})
yaml_path = Path(ultralytics.__file__).parent / 'cfg/models/11/yolo11.yaml'
cfg = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
cfg.update(nc=10, scale='n')
write(HERE / 'model_cfg.json', cfg)
overrides = dict(project=str(HERE / 'attempt_01'), name='train', exist_ok=False,
                 batch=2, nbs=2, imgsz=64, seed=2026102302,
                 fliplr=1.0, flipud=1.0, lr0=0.001, lrf=1.0,
                 momentum=0.9, weight_decay=0.0, patience=0, verbose=False)
contract = {
    'schema':'rampq.gate1.synthetic_ignore_trainer_contract/v1',
    'synthetic_only':True, 'class_count':10, 'names':[str(i) for i in range(10)],
    'datasets':{role:{'manifest_path':fixture[prefix+'_manifest'], 'manifest_sha256':fixture[prefix+'_manifest_sha256']}
                for role,prefix in [('train','train'),('training_val','val')]},
    'restricted_options':RESTRICTED_OPTIONS,
    'model_cfg_sha256':sha(HERE/'model_cfg.json'),
    'model_source':{'path':str(yaml_path.resolve()),'sha256':sha(yaml_path)},
    'runtime_sources':sources(), 'overrides':overrides,
    'predeclared_acceptance':{
        'actual_train_call':True,'epochs':1,'train_images':4,'validation_images':2,
        'optimizer_steps':2,'ema_updates':2,'actual_positive_and_ignored_negative_anchors_required':True,
        'positive_rows_preserved':True,'finite_losses_and_state':True,'train_parameters_changed':True,
        'fp32_capture_before_framework_serialization':True,'fresh_cpu_state_and_output_exact':True,
        'validation':'SMOKE_NO_TASK_AP','no_original_checkpoint_or_real_image_inputs':True,
        'device':'cpu','threads':1,'external_concurrent_ema_writers_allowed':False},
    'scope':'NEW_RANDOM_YOLO11N_SYNTHETIC_CPU_TRAINER_INTEGRATION_ONLY',
    'old_gate05_numeric_guard':'FAILED_UNCHANGED','old_gate1_dtype_guard':'FAIL_FIXED_TOLERANCES',
    'scientific_gate_pass':False,
}
write(HERE/'training_contract.json', contract)
inputs = [HERE/'training_contract.json',HERE/'model_cfg.json',HERE/'prepare_training.py',HERE/'run_training.py',HERE/'split_identity_preflight.json']
inputs += [ROOT/'scripts'/name for name in ('gate1_ignore_dataset.py','gate1_ignore_geometry.py',
    'gate1_ignore_negative_loss.py','gate1_ignore_trainer.py','gate1_fp32_live_capture.py','gate1_fp32_reference_state.py')]
inputs += list((HERE/'synthetic_fixture_v1').rglob('*'))
members = {p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
           for p in sorted(inputs) if p.is_file()}
write(HERE/'preparation_manifest.json', {'schema':'rampq.gate1.synthetic_training_preparation/v1','files':members})
print(json.dumps({'status':'FROZEN_NEW_SYNTHETIC_TRAINER_INPUTS','files':len(members),
                  'training_contract_sha256':sha(HERE/'training_contract.json'),
                  'preparation_manifest_sha256':sha(HERE/'preparation_manifest.json')},indent=2))
