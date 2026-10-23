"""Freeze a small DAG-correct macro frontier study before new inference."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/gate1'
OUT = Path(__file__).parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ident(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def new(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    runner = ROOT / 'scripts/probe_gate1_upstream_frontier.py'
    spec = importlib.util.spec_from_file_location('new_upstream_protocol_only', runner)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    observed_path = OUT / 'topology_inspection_v1/topology.json'
    observed = read(observed_path)
    assert observed['status'] == 'CPU_TOPOLOGY_OBSERVED_NO_FORWARD'
    assert not observed['gpu_execution'] and not observed['images_or_annotations_opened']
    assert sha(observed_path) == 'bc7d77a7261c8f32d02f5bf7e5baff2c5f86ad8e7d4579fdbb6d66b8788059da'
    topology = {'modules': [{'i': row['index'], 'f': row['from'], 'type': row['type']} for row in observed['modules']],
                'save': observed['model_save']}
    helper.validate_topology(topology)
    assert helper.CUTS == [-1,4,6,10,22]
    assert helper.FRONTIERS == observed['cut_frontiers']
    assert {str(cut): helper.frontier_indices(topology, cut) for cut in helper.CUTS} == observed['cut_frontiers']
    inherited = read(BASE / '20261023_decode_probe_preparation/head_full109_protocol.json')
    keys = ['qualifier','qualification_dir','qualification_manifest','checkpoint','runtime_environment',
            'framework_source_sha256','package_versions','controls','data_role','selection_detector_inference',
            'holdout_accessed','historical_equality_required','native_reconstruction_required','raw_tolerances',
            'old_gate05_guard','historical_gate1_dtype_guard','head_reg_max_policy','parameter_buffer_policy',
            'automatic_retries','checkpoint_selection','scientific_gate_pass','accepted_float_baseline_ready','limits','decoder_runner']
    common = {key: inherited[key] for key in keys}
    reference = BASE / '20261023_head_precision_full_v1'
    ref_receipt = read(reference / 'receipt.json')
    assert ref_receipt['status'] == 'COMPLETE_HEAD_DIAGNOSTIC_ONLY' and ref_receipt['images_completed'] == 109
    assert ref_receipt['historical_bytes_all_equal'] and ref_receipt['native_head_logits_all_exact']
    assert read(BASE / '20261023_head_precision_full_audit_v1/recomputed_summary.json')['status'] == 'PASS_SAVED_HEAD_PROBE_ARRAYS_ONLY'
    common.update({
        'schema': helper.SCHEMA, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'runner_sha256': sha(runner),
        'head_runner': ident(ROOT / 'scripts/probe_gate1_head_precision.py'),
        'reference_head_probe_dir': str(reference), 'reference_head_probe_manifest': ident(reference / 'manifest.json'),
        'topology': topology, 'observed_topology_identity': ident(observed_path),
        'cuts': helper.CUTS, 'frontiers': helper.FRONTIERS, 'intervention': helper.INTERVENTION,
        'scope': 'Bounded post-hoc debug macro-frontier intervention, not precision-module selection or a scientific/deployment fix.',
        'native_replay_policy': 'Complete native manual graph replay and both same-dtype suffixes at every cut must exactly reproduce frozen native logits before crossed results are interpreted.',
        'cut_policy': 'Transfer every tensor on every graph edge crossing the cut; retain skip dependencies, not only the most recent output.',
        'data_storage_policy': 'Store each native union frontier once, ten crossed logit/FP32 decoded outputs, and native controls; cast frontiers and repeated controls are bound by hashes and independently reconstructed from saved inputs.',
        'comparison_policy': 'Use the unchanged all-anchor tolerances and original native FP32 decoded reference. Report all predeclared cuts and both crossed directions without selecting favorable subsets.',
        'aliases': {'cut22': 'Repeats the two prior head-cross endpoints; not independent new evidence.',
                    'input_fp32_to_suffix_fp16': 'Repeats native FP16 network followed by FP32 decode, since the receiving model casts the identical input.',
                    'same_dtype_controls': 'Repeated exact-reproduction controls, not extra independent observations.'},
        'nonadditivity': 'Different frontier interventions are nonlinear and preserve multiple skip paths. Do not subtract counts to assign a unique layer or causal percentage.',
        'limits': common['limits'] + helper.LIMITS,
    })
    pilot = ['train/0000170_01201_d_0000001','train/9999990_00000_d_0000024','train/0000046_00000_d_0000087']
    paths = []
    for stage, ids in [('pilot3',pilot),('full109',inherited['image_ids'])]:
        p = dict(common,stage=stage,image_ids=ids)
        p['execution_condition'] = ('First validate the fixed three diagnostic examples; preserve and stop on any identity/native replay failure.' if stage=='pilot3'
            else 'Extend only after pilot completes and saved-array audit confirms identity/replay/summary consistency. Execute all109 regardless of favorable or unfavorable diagnostic guards. No automatic retries or threshold changes.')
        path = OUT / f'{stage}_protocol.json'; new(path,p); paths.append(path)
    new(OUT / 'preparation_receipt.json',{'schema':'rampq.gate1.upstream_frontier_preparation/v1',
        'status':'BOUNDED_PROTOCOLS_FROZEN_BEFORE_INFERENCE','runner':ident(runner),'protocols':[ident(p) for p in paths],
        'observed_topology':ident(observed_path),'real_inference_by_this_preparation':False})
    print(json.dumps({'protocols':[ident(p) for p in paths]},ensure_ascii=False))


if __name__ == '__main__':
    main()
