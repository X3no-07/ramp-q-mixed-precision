"""Extract an unchanged, exact G1-FP16 -> G2-INT8 boundary for build diagnostics."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIRST = '/model.2/cv2/conv/Conv'
SECOND = '/model.3/conv/Conv'
VARIANTS = ('compute_output_obey', 'compute_only_obey', 'compute_output_prefer', 'global_fp16_only')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def message_sha(message):
    return hashlib.sha256(message.SerializeToString()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    import onnx
    from onnx import helper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/gate05/20261019_mixed_constraint_probe')
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists(), 'Choose a fresh directory; preserve existing evidence'
    source_root = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles/r_only_G1'
    manifest = json.loads((source_root / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['configuration'] == 'r_only_G1'
    for name, info in manifest['files'].items():
        path = source_root / name
        assert path.resolve().parent == source_root.resolve()
        assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], name
    source = source_root / 'model.onnx'
    model = onnx.load(str(source))
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = {value.name: value for value in list(inferred.graph.input) +
              list(inferred.graph.value_info) + list(inferred.graph.output)}
    nodes = {node.name: node for node in model.graph.node}
    constants = {value.name: value for value in model.graph.initializer}
    first, second = nodes[FIRST], nodes[SECOND]
    prefix = 'rampq_model_3_conv'
    selected_names = {FIRST, SECOND, '/model.2/cv2/act/Sigmoid', '/model.2/cv2/act/Mul'}
    selected_names.update(node.name for node in model.graph.node if node.name.startswith(prefix + '_'))
    assert len(selected_names) == 9
    selected = [copy.deepcopy(node) for node in model.graph.node if node.name in selected_names]
    needed_constants = {name for node in selected for name in node.input if name in constants}
    copied_constants = [copy.deepcopy(value) for value in model.graph.initializer if value.name in needed_constants]
    assert len(copied_constants) == 10
    assert first.input[1] == 'model.2.cv2.conv.weight'
    assert second.input[0] == prefix + '_a_dq' and second.input[1] == prefix + '_w_dq'
    external, final = first.input[0], second.output[0]
    available = {external} | needed_constants
    for node in selected:
        assert set(node.input) - {''} <= available, node.name
        available.update(node.output)
    assert final in available
    input_shape = [value.dim_value for value in values[external].type.tensor_type.shape.dim]
    output_shape = [value.dim_value for value in values[final].type.tensor_type.shape.dim]
    assert input_shape == [1, 48, 160, 160] and output_shape == [1, 64, 80, 80]
    extracted = copy.deepcopy(model)
    extracted.graph.CopyFrom(helper.make_graph(selected, 'exact_existing_G1_G2_precision_boundary',
                             [copy.deepcopy(values[external])], [copy.deepcopy(values[final])], copied_constants))
    onnx.checker.check_model(extracted, full_check=True)
    assert all(node.SerializeToString() == nodes[node.name].SerializeToString() for node in extracted.graph.node)
    assert all(value.SerializeToString() == constants[value.name].SerializeToString()
               for value in extracted.graph.initializer)
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(extracted, str(output / 'model.onnx'))
    shutil.copy2(source_root / 'preflight.json', output / 'preflight.json')
    runner = ROOT / 'scripts/jetson_mixed_constraint_probe.py'
    ast.parse(runner.read_text(encoding='utf-8'), feature_version=(3, 8))
    shutil.copy2(runner, output / runner.name)
    spec = {'schema': 'rampq.gate05.mixed_constraint_probe/v1',
            'client_prepared_at': datetime.now().astimezone().isoformat(),
            'diagnostic_only': True, 'source_configuration': 'r_only_G1',
            'source_graph_sha256': sha(source), 'source_manifest_sha256': sha(source_root / 'manifest.json'),
            'source_spec_sha256': sha(source_root / 'spec.json'), 'graph_sha256': sha(output / 'model.onnx'),
            'failed_preflight_sha256': sha(output / 'preflight.json'),
            'original_numerical_parity_status': 'LOCAL_QDQ_PREFLIGHT_FAILED',
            'source_nodes_serialized_sha256': {node.name: message_sha(node) for node in extracted.graph.node},
            'source_initializers_serialized_sha256': {value.name: message_sha(value)
                                                      for value in extracted.graph.initializer},
            'serialized_nodes_and_initializers_unchanged': True,
            'external_input': {'name': external, 'shape': input_shape, 'dtype': 'DataType.FLOAT',
                               'meaning': 'Synthetic external activation replaces the upstream network; no images used'},
            'external_output': {'name': final, 'shape': output_shape, 'dtype': 'DataType.FLOAT'},
            'source_requests': [{'onnx_node': FIRST, 'group': 'G1', 'precision': 'FP16',
                                 'weight_input': 'model.2.cv2.conv.weight'},
                                {'onnx_node': SECOND, 'group': 'G2', 'precision': 'INT8',
                                 'weight_input': 'model.3.conv.weight'}],
            'nodes': len(selected), 'initializers': len(copied_constants),
            'variants': list(VARIANTS), 'max_seconds_per_variant': 90, 'max_total_seconds': 360,
            'workspace_mib': 1024, 'avg_timing': 1,
            'auxiliary_graph_outputs_added': False,
            'variant_changes': {'compute_output_obey': 'FP16 compute and output requests, obey',
                               'compute_only_obey': 'FP16 compute request only, obey',
                               'compute_output_prefer': 'FP16 compute and output requests, prefer',
                               'global_fp16_only': 'Global --fp16; no per-layer precision/output requests'},
            'scope': 'Build/deserialization and native format evidence for an exact two-Conv slice only. '
                     'No inference, new full-model configuration, numerical migration, target rescue, '
                     'dataset evaluation, changed criteria, or performance conclusion. '
                     'A preferred or omitted constraint is not proof of FP16 adoption. '
                     'Native FP32 or INT8 output on the restored Conv is not renamed an FP16 format pass.'}
    write(output / 'spec.json', spec)
    files = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in sorted(output.iterdir()) if path.is_file()}
    write(output / 'manifest.json', {'schema': 'rampq.gate05.mixed_constraint_probe_bundle/v1', 'files': files})
    (output / 'SHA256SUMS').write_text(''.join(info['sha256'] + '  ' + name + '\n'
                                           for name, info in files.items()), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, 'gate05_mixed_constraint_probe_20261019/' + path.name)
    print(json.dumps({'status': 'EXACT_MIXED_BOUNDARY_PROBE_PREPARED_NOT_RUN',
                      'output': str(output), 'archive': str(archive_path), 'archive_sha256': sha(archive_path),
                      'input_shape': input_shape, 'output_shape': output_shape,
                      'nodes': len(selected), 'initializers': len(copied_constants)}, indent=2))


if __name__ == '__main__':
    main()
