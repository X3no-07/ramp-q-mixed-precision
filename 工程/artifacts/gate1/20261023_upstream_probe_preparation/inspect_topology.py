"""CPU-only checkpoint topology inspection; no images or forward execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / 'topology_inspection_v1'
CHECKPOINT = ROOT / 'artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification/best.pt'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists()
    assert sha(CHECKPOINT) == 'cddd14b3ebba294e4c3200bfb50fc1db1a9459c425d8e51d583c0881d71cfcfa'
    OUT.mkdir(parents=True)
    config = OUT / 'ultralytics_config'; config.mkdir()
    os.environ['YOLO_CONFIG_DIR'] = str(config)
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    import torch
    import ultralytics
    from ultralytics.utils import SETTINGS
    SETTINGS.update({key: False for key in ('sync', 'wandb', 'comet', 'clearml', 'mlflow', 'dvc', 'raytune', 'tensorboard')})
    framework = Path(ultralytics.__file__).resolve().parent
    env = json.loads((CHECKPOINT.parent / 'environment.json').read_text(encoding='utf-8'))
    for name, item in env['framework_sources'].items():
        assert sha(framework / name) == item['sha256'], name
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    key = 'ema' if isinstance(payload.get('ema'), torch.nn.Module) else 'model'
    model = payload[key]
    assert all(v.device.type == 'cpu' for v in model.state_dict().values())
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    modules = []
    for index, module in enumerate(model.model):
        assert int(module.i) == index
        f = module.f
        f = int(f) if isinstance(f, int) else [int(x) for x in f]
        deps = [f] if isinstance(f, int) else f
        actual = [index - 1 if source == -1 else (index + source if source < -1 else source) for source in deps]
        assert all(-1 <= source < index for source in actual)
        modules.append({'index': index, 'from': f, 'dependencies': actual,
                        'type': type(module).__module__ + '.' + type(module).__name__,
                        'parameters': sum(v.numel() for v in module.parameters()),
                        'saved_for_skip': index in model.save})
    assert len(modules) == 24 and modules[-1]['dependencies'] == [16,19,22]
    cuts = [-1,4,6,10,22]
    frontiers = {str(cut): sorted({dependency for module in modules[cut+1:] for dependency in module['dependencies'] if dependency <= cut}) for cut in cuts}
    assert frontiers == {'-1': [-1], '4': [4], '6': [4,6], '10': [4,6,10], '22': [16,19,22]}
    result = {'schema': 'rampq.gate1.cpu_checkpoint_topology/v1', 'status': 'CPU_TOPOLOGY_OBSERVED_NO_FORWARD',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint_sha256': sha(CHECKPOINT),
              'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
              'script_sha256': sha(Path(__file__)), 'modules': modules, 'model_save': [int(x) for x in model.save],
              'macro_cuts': cuts, 'cut_frontiers': frontiers, 'head_reg_max': int(model.model[-1].reg_max),
              'model_stride': model.stride.tolist(), 'native_forward_executed': False,
              'gpu_execution': False, 'images_or_annotations_opened': False,
              'selection_or_holdout_accessed': False, 'saved_fp16_checkpoint_lift_recovers_original_fp32': False}
    target = OUT / 'topology.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'modules':len(modules),'cuts':cuts,'frontiers':frontiers,'head_reg_max':result['head_reg_max'],'topology_sha256':sha(target)}))


if __name__ == '__main__':
    main()
