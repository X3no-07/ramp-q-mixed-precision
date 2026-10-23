"""Freeze one already prepared graph and native precision requests; does not run it."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / 'artifacts/gate05/20261019_precision_candidates'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    import onnx
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', required=True,
                        choices=['fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles' / args.configuration
    assert not output.exists(), 'Preserve frozen bundles; choose a fresh output'
    source_manifest = json.loads((CANDIDATES / 'manifest.json').read_text(encoding='utf-8'))
    for relative, info in source_manifest['files'].items():
        path = CANDIDATES / relative
        assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], relative
    preparation = json.loads((CANDIDATES / 'preparation.json').read_text(encoding='utf-8'))
    selected = CANDIDATES / args.configuration
    audit = json.loads((selected / 'graph_audit.json').read_text(encoding='utf-8'))
    assert audit['configuration'] == args.configuration and audit['status'] == 'GRAPH_PREPARED_AND_STRUCTURALLY_AUDITED_ONLY'
    assert sha(selected / 'model.onnx') == audit['graph_sha256']
    output.mkdir(parents=True, exist_ok=False)
    for name in ('model.onnx', 'graph_audit.json'):
        shutil.copy2(selected / name, output / name)
    frozen = ROOT / 'artifacts/gate05/20261018_int8/input_v1'
    for source, name in [(frozen / 'module_map.json', 'module_map.json'), (frozen / 'preflight.json', 'preflight.json'),
                         (frozen / 'protocol.json', 'reference_protocol.json')]:
        shutil.copy2(source, output / name)
    preflight = json.loads((output / 'preflight.json').read_text(encoding='utf-8'))
    reference_protocol = json.loads((output / 'reference_protocol.json').read_text(encoding='utf-8'))
    assert preflight['status'] == preparation['original_numerical_failure_retained'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    scope = {'schema': 'rampq.gate05.precision_candidate_scope/v1',
             'status': 'DIAGNOSTIC_MATRIX_BUILD_AND_FIXED_IMAGE_ONLY',
             'prepared_client_at': datetime.now().astimezone().isoformat(),
             'configuration': args.configuration, 'failed_preflight_sha256': sha(output / 'preflight.json'),
             'original_tolerances_changed': False, 'dataset_evaluation_run': False,
             'permitted_conclusion': 'Build, deserialization, actual layer format/tactic evidence and one fixed-image execution only',
             'not_established': 'Numerical migration, target rescue, bidirectional attribution, dataset accuracy or performance',
             'common_output_structure_required': True,
             'nonquantized_float_control': 'Pending native validation and protocol decision; no causal comparison claim',
             'reason': 'Implement previously frozen five configurations and optional q_only:G1 while retaining original numerical failure'}
    write(output / 'diagnostic_scope.json', scope)
    graph = onnx.load(str(output / 'model.onnx'))
    serialized_outputs = [value.SerializeToString() for value in graph.graph.output]
    output_hash = hashlib.sha256(b''.join(len(value).to_bytes(8, 'big') + value for value in serialized_outputs)).hexdigest()
    requests = [{'module': row['module'], 'onnx_node': row['onnx_node'], 'group': row['group'],
                 'precision': 'INT8' if row['quantized'] else 'FP16',
                 'weight_input': row['original_weight_initializer'],
                 'original_activation_input': row['activation_source']} for row in audit['mapping']]
    spec = {'schema': 'rampq.gate05.precision_candidate_build_spec/v1',
            'configuration': args.configuration, 'diagnostic_only': True,
            'graph_sha256': sha(output / 'model.onnx'), 'graph_audit_sha256': sha(output / 'graph_audit.json'),
            'module_map_sha256': sha(output / 'module_map.json'), 'failed_preflight_sha256': sha(output / 'preflight.json'),
            'diagnostic_scope_sha256': sha(output / 'diagnostic_scope.json'),
            'quantized_conv_count': audit['quantized_conv_count'],
            'fp16_conv_count': audit['desired_restored_fp16_conv_count'], 'requests': requests,
            'expected_binding_contract': audit['common_expected_binding_contract'], 'primary_output_name': 'output0',
            'original_local_preflight_status': preflight['status'], 'workspace_mib': 1024, 'avg_timing': 1,
            'max_build_seconds': 1800, 'restored_conv_compute_precision': 'fp16', 'restored_conv_output_type': 'fp16',
            'precisionConstraints': 'obey', 'external_input_io_format': 'fp32:chw',
            'external_output_io_formats': ['fp32:chw', 'fp32:chw'],
            'common_preparation_sha256': sha(CANDIDATES / 'preparation.json'),
            'common_source_fp32_onnx_sha256': preparation['source_sha256']['source_fp32_onnx'],
            'common_source_weights_sha256': reference_protocol['source_weights_sha256'],
            'common_q_all_pe_tap_reference_sha256': preparation['source_sha256']['source_q_all_pe_tap_onnx'],
            'common_output_declaration_sha256': output_hash,
            'nonquantized_float_control_status': 'PENDING_NATIVE_VALIDATION_AND_PROTOCOL',
            'limits': 'FLOAT graph tensors do not enforce FP16. Native requested names, actual constraints and all I/O '
                      'must validate; failure or fallback is not a precision pass. Original full numerical failure retained.'}
    write(output / 'spec.json', spec)
    for name in ('jetson_build_precision_candidate.py', 'jetson_monitored_build.py'):
        path = ROOT / 'scripts' / name
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
        shutil.copy2(path, output / name)
    files = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in sorted(output.iterdir()) if path.is_file()}
    write(output / 'manifest.json', {'schema': 'rampq.gate05.precision_runtime_bundle/v1',
                                    'configuration': args.configuration, 'files': files})
    (output / 'SHA256SUMS').write_text(''.join(info['sha256'] + '  ' + name + '\n'
                                            for name, info in files.items()), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    archive_root = 'gate05_precision_' + args.configuration + '_20261019'
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, archive_root + '/' + path.name)
    print(json.dumps({'status': 'RUNTIME_BUNDLE_PREPARED_NOT_RUN', 'configuration': args.configuration,
                      'bundle_root': str(output), 'archive': str(archive_path), 'archive_root': archive_root,
                      'archive_sha256': sha(archive_path), 'files': len(files), 'spec_sha256': sha(output / 'spec.json')}, indent=2))


if __name__ == '__main__':
    main()
