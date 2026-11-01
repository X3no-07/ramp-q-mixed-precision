"""Audit the collected one-configuration INT8 diagnostic, preserving parity failure."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/gate05/20261018_int8'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = read(RUN / 'remote_receipt.json')
    for relative, expected in receipt['files'].items():
        path = RUN / relative
        assert path.stat().st_size == expected['bytes'], relative
        assert sha(path) == expected['sha256'], relative
    status = read(RUN / 'monitor_attempt2/status.json')
    build = read(RUN / 'build_attempt2/build_result.json')
    if build['status'] == 'BUILD_FAILED':
        assert status['status'] == 'BUILD_SCRIPT_FAILED' and status['build_returncode'] != 0
        local = read(RUN / 'input_v1/preflight.json')
        assert local['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
        samples = [json.loads(line) for line in (RUN / 'monitor_attempt2/resources.jsonl').read_text().splitlines() if line.strip()]
        errors = [line for line in (RUN / 'build_attempt2/trtexec_build.log').read_text(errors='replace').splitlines() if '[E]' in line]
        report = {
            'schema': 'rampq.gate05.int8_audit/v1', 'client_date': '2026-10-18',
            'status': 'FULL_INT8_BUILD_FAILED_AT_G4_PE_CONV_ADD',
            'verified_remote_files': len(receipt['files']), 'engine_created': False,
            'real_int8_full_model_execution': False, 'build_seconds': status['elapsed_seconds'],
            'build_errors': errors, 'boot_id': status['boot_id'],
            'local_full_model_qdq_parity': local['status'],
            'local_max_abs_box_difference': local['max_abs_box_difference'],
            'local_max_abs_score_difference': local['max_abs_score_difference'],
            'build_resource_samples': len(samples),
            'build_max_observed_temperature_c': max(max(r['temperatures_c'].values()) for r in samples),
            'build_min_observed_available_memory_mib': min(r['MemAvailable_kib'] for r in samples) / 1024,
            'build_max_observed_swap_used_mib': max(r['SwapTotal_kib']-r['SwapFree_kib'] for r in samples) / 1024,
            'kernel_monitor_log': (RUN / 'monitor_attempt2/kernel.log').read_text(),
            'dataset_evaluation_run': False, 'benchmark_run': False, 'full_gate05_status': 'INCOMPLETE',
            'scope': 'Full-model build fails for PE depthwise Conv+Add in G4. '
                     'No full-model INT8 execution, precision coverage, rescue, accuracy, or performance result.',
        }
        repro = RUN / 'pe_repro_results/summary.json'
        if repro.exists():
            data = read(repro)
            report['minimal_repro'] = [dict(variant=row['variant'], returncode=row['returncode'],
                                            timeout=row['timeout'], seconds=row['seconds'],
                                            errors=row.get('errors', [])) for row in data['results']]
        probe_receipt = RUN / 'pe_repro_remote_receipt.json'
        if probe_receipt.exists():
            probe_files = read(probe_receipt)['files']
            for relative, expected in probe_files.items():
                path = RUN / relative
                assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], relative
            retry = read(RUN / 'pe_tap_retry180/summary.json')['results'][0]
            assert retry['returncode'] == 0 and not retry['timeout']
            conv = retry['convolutions'][0]
            assert conv['Weights']['Type'] == 'Int8'
            assert 'INT8' in conv['Inputs'][0]['Format/Datatype'].upper()
            assert '/attn/Add' not in conv['Name']
            report.update(probe_verified_remote_files=len(probe_files),
                          total_verified_remote_files=len(probe_files)+len(receipt['files']),
                          minimal_tap_workaround={'status':'NODE_BUILD_PASS_FULL_MODEL_UNBUILT',
                              'seconds':retry['seconds'],'first_90_second_trial':'TIMEOUT_INCONCLUSIVE',
                              'retry_budget_seconds':retry['max_seconds'],
                              'weights_type':conv['Weights']['Type'],
                              'input_formats':[x['Format/Datatype'] for x in conv['Inputs']],
                              'output_formats':[x['Format/Datatype'] for x in conv['Outputs']],
                              'tactic_name_available':bool(conv.get('TacticName')),
                              'scope':'INT8 weight/input storage formats and separated Conv/Add in a small engine; no full-model inference'})
        (RUN / 'audit_summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        checkpoint = read(RUN / 'checkpoint.json')
        checkpoint.update(status=report['status'], engine_created=False, real_int8_model_execution=False,
                          verified_remote_files=len(receipt['files']), build_seconds=report['build_seconds'],
                          audit_summary='audit_summary.json')
        if 'minimal_tap_workaround' in report:
            checkpoint.update(minimal_repro_status='FUSION_FAILURE_REPRODUCED_AND_NODE_TAP_WORKAROUND_BUILT',
                              total_verified_remote_files=report['total_verified_remote_files'],
                              full_graph_candidate='pe_tap_full_graph_candidate/candidate.json',
                              candidate_whole_model_build_run=False)
        (RUN / 'checkpoint.json').write_text(json.dumps(checkpoint, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
        return
    precision = read(RUN / 'build_attempt2/precision_audit.json')
    smoke = read(RUN / 'smoke/smoke_result.json')
    local = read(RUN / 'input_v1/preflight.json')
    assert status['status'] == 'BUILD_SCRIPT_OK' and status['build_returncode'] == 0
    assert build['status'] == 'BUILT_AND_DESERIALIZED' and build['configuration'] == 'q_all'
    assert sha(RUN / 'build_attempt2/model_q_all.engine') == build['engine_sha256'] == smoke['engine_sha256']
    assert sha(RUN / 'diagnostic_bundle_v2/model_q_all.onnx') == build['onnx_sha256']
    assert sha(RUN / 'build_attempt2/engine_layers.json') == precision['layer_file_sha256']
    assert precision['status'] == 'ALL_88_INT8_CONVS_VERIFIED' and precision['verified_int8_convs'] == 88
    assert local['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert build['local_preflight_status'] == local['status'] and build['diagnostic_only']
    assert smoke['status'] == 'INT8_SINGLE_IMAGE_EXECUTION_OK' and smoke['execution_success'] and smoke['finite_output']
    assert smoke['boot_id'] == status['boot_id'] == receipt['boot_id']
    samples = [json.loads(line) for line in (RUN / 'monitor_attempt2/resources.jsonl').read_text().splitlines() if line.strip()]
    report = {
        'schema': 'rampq.gate05.int8_audit/v1', 'client_date': '2026-10-18',
        'status': 'HARDWARE_INT8_DIAGNOSTIC_PASS_NUMERICAL_MIGRATION_UNRESOLVED',
        'verified_remote_files': len(receipt['files']), 'engine_sha256': build['engine_sha256'],
        'engine_bytes': build['engine_bytes'], 'engine_layers': build['engine_layers'],
        'build_seconds': status['elapsed_seconds'], 'boot_id': smoke['boot_id'],
        'verified_int8_convs': precision['verified_int8_convs'], 'groups': precision['groups'],
        'int8_detection_count': len(smoke['int8_detections']), 'fp16_detection_count': len(smoke['fp16_detections']),
        'local_full_model_qdq_parity': local['status'],
        'local_max_abs_box_difference': local['max_abs_box_difference'],
        'local_max_abs_score_difference': local['max_abs_score_difference'],
        'source_graph_and_ranges_changed_after_preflight': False,
        'build_max_observed_temperature_c': max(max(r['temperatures_c'].values()) for r in samples),
        'build_min_observed_available_memory_mib': min(r['MemAvailable_kib'] for r in samples) / 1024,
        'build_max_observed_swap_used_mib': max(r['SwapTotal_kib']-r['SwapFree_kib'] for r in samples) / 1024,
        'build_resource_samples': len(samples), 'dataset_evaluation_run': False, 'benchmark_run': False,
        'full_gate05_status': 'INCOMPLETE',
        'scope': 'All grouped Conv INT8 kernels and one fixed Selection image execute on Jetson. '
                 'Full-model PyTorch-vs-ONNX parity remains failed. Dataset accuracy, target rescue, '
                 'bidirectional attribution, formal latency, and power are unverified.',
        'clock_note': 'Client date is 2026-10-18. Board clock reports 2026-10-16; use boot ID and monotonic duration.'}
    (RUN / 'audit_summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    checkpoint = read(RUN / 'checkpoint.json')
    checkpoint.update(status=report['status'], verified_int8_convs=report['verified_int8_convs'],
                      engine_sha256=report['engine_sha256'], engine_bytes=report['engine_bytes'],
                      build_seconds=report['build_seconds'], verified_remote_files=report['verified_remote_files'],
                      audit_summary='audit_summary.json', single_image_result='smoke/smoke_result.json',
                      actual_precision_result='build_attempt2/precision_audit.json')
    (RUN / 'checkpoint.json').write_text(json.dumps(checkpoint, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
