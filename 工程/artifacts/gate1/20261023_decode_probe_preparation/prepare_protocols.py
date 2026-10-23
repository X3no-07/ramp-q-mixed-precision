"""Freeze pilot and full-debug precision probes before any new model execution."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
QUALIFICATION = ROOT / 'artifacts/gate1/20261022_posttraining_checks_v1/02_float_qualification'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, data):
    assert not path.exists(), path
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    runner = ROOT / 'scripts/probe_gate1_decode_precision.py'
    qualifier = ROOT / 'scripts/qualify_gate1_float_baseline.py'
    assert sha(qualifier) == 'b981869ad658e0619454cdbe2361c8a34bd029adf2a8bbb7f85590853c736f34'
    old_delivery = ROOT / 'artifacts/gate1/20261022_training_qualification_delivery/delivery_manifest.json'
    for name, row in read(old_delivery)['files'].items():
        path = ROOT / name
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], name
    env_path = QUALIFICATION / 'environment.json'
    environment = read(env_path)
    rows = read(QUALIFICATION / 'debug_source_manifest.json')['rows']
    ids = [row['image_id'] for row in rows]
    assert len(ids) == len(set(ids)) == 109
    assert all(row['assigned_split'] == 'debug' and row['source_split'] == 'train' for row in rows)
    pilot = ['train/0000170_01201_d_0000001', 'train/9999990_00000_d_0000024', 'train/0000046_00000_d_0000087']
    assert set(pilot) < set(ids)
    common = {
        'schema': 'rampq.gate1.decode_precision_probe_protocol/v1',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'runner_sha256': sha(runner),
        'qualifier': identity(qualifier),
        'qualification_dir': str(QUALIFICATION),
        'qualification_manifest': identity(QUALIFICATION / 'manifest.json'),
        'checkpoint': identity(QUALIFICATION / 'best.pt'),
        'runtime_environment': identity(env_path),
        'framework_source_sha256': {name: entry['sha256'] for name, entry in environment['framework_sources'].items()},
        'package_versions': environment['packages'],
        'controls': {'tf32': False, 'float32_matmul_precision': 'highest', 'autocast_enabled': False,
                     'deterministic_algorithms': True, 'cudnn_benchmark': False},
        'data_role': 'debug', 'selection_detector_inference': False, 'holdout_accessed': False,
        'historical_equality_required': True, 'native_reconstruction_required': True,
        'raw_tolerances': {'box_rtol': .005, 'box_atol': .5, 'score_rtol': .02, 'score_atol': .002},
        'old_gate05_guard': 'FAILED_UNCHANGED',
        'historical_gate1_dtype_guard': 'FAIL_FIXED_TOLERANCES',
        'head_reg_max_policy': 'Observe actual runtime reg_max and assert 4*reg_max box-logit channels; never assume 64 from construction defaults.',
        'parameter_buffer_policy': 'Two independent saved FP16 checkpoint loads; require equality after lifting state values to FP32. Record runtime anchor/stride caches and compare values.',
        'interventions': ['fp32_logits_fp32_decode', 'fp32_logits_fp16_decode', 'fp16_logits_fp32_decode', 'fp16_logits_fp16_decode'],
        'arithmetic_policy': 'Use frozen source order: softmax over bins, fixed integral convolution, anchor endpoints, xywh and stride multiplication. Explicitly record logits cast into each decode dtype.',
        'comparison_policy': 'Keep FP32 original decoded output as asymmetric reference for fixed all-anchor tolerance descriptions. Separate same-logit decode contrasts from source-logit contrasts, without assuming additive contributions.',
        'reconstruction_policy': 'Require native decode reconstruction exact torch.equal to each forward output, plus exact NumPy dtype/shape/values and normalized input SHA against historical outputs.',
        'automatic_retries': 0, 'checkpoint_selection': 'Original training_val selected best epoch29; no new checkpoint selection.',
        'scientific_gate_pass': False, 'accepted_float_baseline_ready': False,
        'scope': 'Post-hoc engineering diagnosis on reused debug images, not an acceptance override, generalization test, method selection or performance benchmark.',
        'limits': ['Saved FP16 EMA lifted to FP32 does not recover original pre-save FP32 training state.',
                   'A new diagnostic intervention never replaces the historical full-model guards.',
                   'No NMS, target matching, official AP or Selection/Holdout inference in this probe.',
                   'Input/parameter/output dtype does not prove GPU internal accumulation precision.'],
    }
    paths = []
    for stage, image_ids in [('pilot3', pilot), ('full109', ids)]:
        protocol = dict(common, stage=stage, image_ids=image_ids)
        protocol['execution_condition'] = ('First execute all3 diagnostic examples; abort on identity, historical or reconstruction failure.' if stage == 'pilot3'
                                         else 'Execute only after pilot3 completes and native reconstruction plus historical reproduction are exact for all3. Extend to all109 regardless of diagnostic allclose outcome; never select a favorable subset.')
        path = OUT / f'{stage}_protocol.json'
        write_new(path, protocol)
        paths.append(path)
    write_new(OUT / 'preparation_receipt.json', {
        'schema': 'rampq.gate1.decode_probe_preparation/v1',
        'status': 'PROTOCOLS_FROZEN_BEFORE_NEW_MODEL_EXECUTION',
        'at_utc': datetime.now(timezone.utc).isoformat(),
        'runner': identity(runner), 'protocols': [identity(p) for p in paths],
        'old_delivery_files_verified': 75, 'old_delivery_manifest': identity(old_delivery),
        'no_real_inference_in_this_preparation': True,
    })
    print(json.dumps({'protocols': [identity(p) for p in paths]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
