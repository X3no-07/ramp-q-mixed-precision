"""Join the saved-output CPU rechecks without rerunning any model or NMS."""
import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)


def main():
    nms_path, numeric_path = OUT / 'nms_gt_analysis.json', OUT / 'numeric_arrays_summary.json'
    nms, numeric = read(nms_path), read(numeric_path)
    assert nms['all_sources_unchanged'] and numeric['all_source_hashes_unchanged']
    assert nms['new_native_cpu_nms_detections_bit_exact_to_saved_smoke_float32']
    assert nms['old_native_cpu_nms_detections_bit_exact_to_saved_smoke_float32']
    assert nms['ground_truth_count'] == nms['new_native_gt_supported_detection_count'] == 1
    assert nms['scallop_ground_truth_count'] == 0
    for name, expected in nms['source_hashes_after'].items():
        path = ROOT / name
        assert sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes'], name
    for name, expected in numeric['source_hashes_after'].items():
        assert sha(Path(numeric['source_files'][name])) == expected, name
    assert sha(Path(numeric['program']['path'])) == numeric['program']['sha256']
    metric_keys = ['elements', 'max_abs', 'mean_abs', 'rmse', 'p50_abs', 'p95_abs', 'p99_abs',
                   'signed_mean', 'different_elements', 'equal_elements', 'worst']
    compact = lambda row: {key: row[key] for key in metric_keys}
    summary = {'schema': 'rampq.gate05.corrected_qonly_fixed_image_saved_raw_recheck/v1',
               'status': 'SAVED_RAW_CPU_RECHECK_COMPLETE_DESCRIPTIVE_ONLY',
               'image': '000020.jpg', 'source_image_shape_hw': nms['source_image_shape_hw'],
               'nms': nms['nms'], 'gt_match_iou_threshold_from_original_config': nms['gt_match_iou_threshold_from_original_config'],
               'new_native_cpu_nms_exactly_reproduces_saved_smoke_float32': True,
               'new_native_detections': nms['cpu_nms_detections']['new_native_saved_raw'],
               'native_unique_holothurian_gt_iou': nms['detection_ious_to_unique_gt']['new_native_saved_raw'][0]['iou_to_unique_gt_unrounded'],
               'new_native_gt_supported_detection_count': 1, 'supplied_ground_truth_count': 1,
               'supplied_scallop_ground_truth_count': 0,
               'scallop_interpretation': nms['scallop_detection_interpretation'],
               'ground_truth_matches': nms['original_gate0_one_to_one_target_matches'],
               'raw_max_scores': {name: row['global_max_score'] for name, row in nms['raw_score_statistics'].items()},
               'old_native_cpu_nms_detection_count': len(nms['cpu_nms_detections']['old_native_saved_raw']),
               'native_minus_cpu_original_arithmetic_dtype': 'float64',
               'native_minus_cpu_original_primary_all': compact(numeric['primary']['overall']),
               'native_minus_cpu_original_primary_xywh': compact(numeric['primary']['xywh']),
               'native_minus_cpu_original_primary_class_scores': compact(numeric['primary']['class_scores']),
               'native_minus_cpu_original_primary_per_channel': [dict(channel=row['channel'], name=row['name'], **compact(row))
                                                               for row in numeric['primary']['channels']],
               'native_minus_cpu_original_auxiliary_all': compact(numeric['auxiliary']['overall']),
               'auxiliary_per_channel_full_statistics_file': numeric_path.name,
               'auxiliary_channel_count': len(numeric['auxiliary']['channels']),
               'saved_cpu_original_vs_code_clip': numeric['saved_cpu_original_vs_code_clip'],
               'component_evidence': {nms_path.name: {'bytes': nms_path.stat().st_size, 'sha256': sha(nms_path)},
                                      numeric_path.name: {'bytes': numeric_path.stat().st_size, 'sha256': sha(numeric_path)}},
               'local_postprocessing_versions': {'torch': nms['torch_version'], 'ultralytics': nms['ultralytics_version']},
               'archived_native_smoke_ultralytics': nms['native_smoke_ultralytics_version'],
               'postprocessing_version_scope': 'Local CPU package versions differ from native; actual new and old saved detection arrays were nonetheless reproduced exactly as FLOAT32.',
               'all_original_source_files_still_hash_identical': True,
               'original_complete_model_pytorch_ort_guard': 'FAILED_UNCHANGED',
               'new_tolerances_defined': False, 'numerical_migration_pass_claim': False,
               'accuracy_pass_claim': False, 'causal_attribution_allowed': False,
               'performance_pass_claim': False, 'dataset_evaluation_run': False,
               'new_full_model_inference_run': False, 'new_native_execution_run': False, 'cuda_used': False,
               'limits': 'Only saved arrays from one original fixed Selection image were postprocessed. Numeric differences are descriptive with no new tolerance or parity PASS. The single supplied GT supports one holothurian match; the scallop detection has no supplied GT and cannot establish a second correct target. Original PyTorch/ORT guard remains failed. Common fusion/nonquantized floating precision controls, native full numerical migration, dataset accuracy and performance remain unestablished.'}
    write_new(OUT / 'analysis.json', summary)
    files = {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(OUT.iterdir()) if path.is_file() and path.name not in ('manifest.json', 'SHA256SUMS')}
    write_new(OUT / 'manifest.json', {'schema': 'rampq.gate05.fixed_image_saved_raw_recheck_files/v1',
                                     'files': files, 'source_raw_trees_modified': False,
                                     'partial_attempt_file_not_used': 'nms_gt_analysis.failed_serialization.partial'})
    files['manifest.json'] = {'bytes': (OUT / 'manifest.json').stat().st_size, 'sha256': sha(OUT / 'manifest.json')}
    (OUT / 'SHA256SUMS').write_text(''.join(row['sha256'] + '  ' + name + '\n' for name, row in sorted(files.items())), encoding='utf-8')
    print(json.dumps({'analysis_sha256': sha(OUT / 'analysis.json'), 'manifest_sha256': sha(OUT / 'manifest.json'),
                      'new_gt_iou': summary['native_unique_holothurian_gt_iou'],
                      'primary_score_difference': summary['native_minus_cpu_original_primary_class_scores'],
                      'auxiliary_difference': summary['native_minus_cpu_original_auxiliary_all'],
                      'files': len(files)}, indent=2))


if __name__ == '__main__':
    main()
