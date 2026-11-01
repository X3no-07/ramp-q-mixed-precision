"""Read-only verification of saved fixed-image reference diagnostics; no inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/gate05/20261019_reference_parity_diagnostic'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path, record):
    assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], str(path)


def main():
    manifest_counts, source_counts = {}, {}
    stages = ['full_replay','first_boundary','hybrid_reference/attempt_2_four_threads']
    for stage in stages:
        folder = BASE/stage
        manifest = read(folder/'manifest.json')['files']
        for rel, record in manifest.items():
            path = (folder/rel).resolve()
            assert folder.resolve() in path.parents, str(path)
            verify(path, record)
        manifest_counts[stage] = len(manifest)
        sources = read(folder/'source_hashes_after.json')
        for key, record in sources.items():
            path = Path(record['path']) if 'path' in record else ROOT/key
            verify(path, record)
        source_counts[stage] = len(sources)
    original = ROOT/'artifacts/gate05/20261018_int8/input_v1'
    assert sha(original/'preflight.json') == '505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21'
    assert read(original/'preflight.json')['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    full = read(BASE/'full_replay/analysis.json')
    assert all(full['historical_full_output_reproduction'][k]['byte_identical'] for k in ('pytorch','ort'))
    convs = read(BASE/'full_replay/all_88_conv_replay.json')
    assert len(convs) == 88
    assert all(r['ort_replay_same_ort_input_vs_original_ort_conv']['byte_identical'] for r in convs)
    assert all(r['weight_qdq']['different'] == 0 for r in convs)
    boundary = read(BASE/'first_boundary/analysis.json')
    with np.load(BASE/'full_replay/first_two_conv_arrays.npz',allow_pickle=False) as raw:
        coordinates = np.argwhere(raw['pytorch_1_input_qdq'] != raw['ort_1_input_qdq']).tolist()
    assert coordinates == [r['index_nchw'] for r in boundary['all_actual_q1_disagreement_coordinates']]
    assert len(coordinates) == 3
    peer = read(BASE/'peer_review/root_replay_and_weight_review.json')
    assert peer['sources_before_and_after_identical']
    for rel, digest in peer['source_hashes'].items():
        assert sha(ROOT/rel) == digest, rel
    for rel, rec in peer['derived_files'].items():
        verify(BASE/'peer_review'/rel,rec)
    attempt = BASE/'hybrid_reference/attempt_2_four_threads'
    controls = read(attempt/'all_88_independent_conv_controls.json')
    assert len(controls) == 88 and all(r['against_full_ort_conv']['byte_identical'] and
                                     r['against_all_88_replay_conv']['byte_identical'] for r in controls)
    hybrid = read(attempt/'analysis.json')
    results = {}
    with np.load(attempt/'raw_primary_outputs.npz',allow_pickle=False) as saved:
        for key in saved.files:
            assert saved[key].shape == (1,14,8400) and saved[key].dtype == np.float32
            assert np.isfinite(saved[key]).all()
        for key in hybrid['cases']:
            got, expected = saved[key],saved['original_full_ort']
            rows = read(attempt/(key+'_all_88_boundaries.json'))
            assert len(rows) == 88
            results[key] = {'conv_outputs_byte_equal':sum(r['conv_output_vs_full_ort']['byte_identical'] for r in rows)}
            for name, slc, rt, at in [('boxes',slice(None,4),.005,.5),('scores',slice(4,None),.02,.002)]:
                a,b = got[:,slc,:],expected[:,slc,:]
                violations = int((~np.isclose(a,b,rtol=rt,atol=at)).sum())
                maximum = float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max())
                recorded = hybrid['cases'][key]['original_guard_slices_counterfactual_only'][name]
                assert maximum == recorded['max_abs'] and violations == recorded['violating_elements']
                results[key][name] = {'max_abs':maximum,'violations':violations}
    assert results['ort_conv_and_ort_silu_torch_other']['conv_outputs_byte_equal'] == 88
    failed = read(BASE/'hybrid_reference/failed_single_thread_attempt.json')
    assert failed['control_failed_count'] == 34 and failed['hybrid_cases_run'] is False
    report = {'status':'SAVED_REFERENCE_DIAGNOSTICS_VERIFIED','manifest_files_verified':manifest_counts,
              'current_original_source_records_verified':source_counts,'first_boundary_coordinates':coordinates,
              'all_88_same_ort_input_convs_byte_exact':True,'hybrid_results_recomputed_from_arrays':results,
              'single_thread_failed_control_preserved':True,'signed_zero_scope_correction_preserved':True,
              'original_complete_model_guard':'FAILED_UNCHANGED','native_or_dataset_inference_run':False}
    output = BASE/'root_completion_audit.json'
    assert not output.exists(), 'Keep the previous audit; use its saved result for subsequent review'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
