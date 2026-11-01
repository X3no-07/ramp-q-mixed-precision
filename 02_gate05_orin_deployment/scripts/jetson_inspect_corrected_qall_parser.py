"""Inspect parsed q_all metadata only; no engine/configuration/inference API.

The frozen input root is read only. --output must name a fresh directory outside
that root. Bounds are corrected only in a temporary in-memory parsed network.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

sys.dont_write_bytecode = True


EXPECTED_GRAPH_SHA256 = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
EXPECTED_SPEC_SHA256 = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
EXPECTED_MANIFEST_SHA256 = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
EXPECTED_CONTRACT_SHA256 = '87fcbf04e88fd311784736cf77cddfdd82fc2608ed3de19882467e58707fb430'
EXPECTED_INSPECTION_CONTRACT_SHA256 = '7b0cccdde9f8549a4a0361bd2e8f76393bdc9caa7ec96dea080b9da7475788f9'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_new(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def validate_inputs(root):
    assert sha(root / 'model.onnx') == EXPECTED_GRAPH_SHA256
    assert sha(root / 'spec.json') == EXPECTED_SPEC_SHA256
    assert sha(root / 'manifest.json') == EXPECTED_MANIFEST_SHA256
    assert sha(root / 'reference_contract.json') == EXPECTED_CONTRACT_SHA256
    manifest = read(root / 'manifest.json')
    checked = {}
    for name, info in manifest['files'].items():
        path = (root / name).resolve()
        assert path.parent == root and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], name
        checked[name] = dict(info)
    spec, contract = read(root / 'spec.json'), read(root / 'reference_contract.json')
    assert spec['configuration'] == 'q_all' and spec['quantized_conv_count'] == 88 and spec['fp16_conv_count'] == 0
    assert spec['activation_quantizer_topology'] == 'Q_INT8_CODE_CLIP_DQ'
    assert spec['requires_native_clip_importer_correction'] is True
    assert len(spec['clip_names']) == len(set(spec['clip_names'])) == 88
    assert spec['native_diagnostic_reference_contract_sha256'] == EXPECTED_CONTRACT_SHA256
    assert spec['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED'
    assert contract['not_a_gate_amendment'] is True and contract['gate05_status'] == 'INCOMPLETE'
    assert contract['protected_failure']['original_complete_model_guard'] == 'FAILED_UNCHANGED'
    assert contract['protected_failure']['tolerances_changed'] is False
    return spec, checked


def query(owner, name, *args):
    """Keep unsupported properties and per-query failures explicit."""
    try:
        value = getattr(owner, name)
    except AttributeError as error:
        return {'supported': False, 'ok': False, 'error': type(error).__name__ + ': ' + str(error)}
    except Exception as error:
        return {'supported': None, 'ok': False, 'error': type(error).__name__ + ': ' + str(error)}
    try:
        value = value(*args) if callable(value) else value
        if isinstance(value, (str, bool, int)) or value is None:
            result = value
        elif isinstance(value, float):
            result = value if math.isfinite(value) else repr(value)
        else:
            result = str(value)
        return {'supported': True, 'ok': True, 'value': result}
    except Exception as error:
        return {'supported': True, 'ok': False, 'error': type(error).__name__ + ': ' + str(error)}


def tensor_info(value):
    if value is None:
        return None
    return {'name': value.name, 'shape': list(value.shape), 'dtype': str(value.dtype),
            'allowed_formats': query(value, 'allowed_formats')}


def is_true(result):
    return result.get('ok') is True and result.get('value') is True


def snapshot(layers, initial_types):
    rows = []
    for index, layer in enumerate(layers):
        precision_set = query(layer, 'precision_is_set')
        # An unset precision property's value is not meaningful. Do not read it.
        precision = (query(layer, 'precision') if is_true(precision_set) else
                     {'queried': False, 'reason': 'precision_is_set is false or unavailable'})
        outputs = []
        for output_index in range(layer.num_outputs):
            output_set = query(layer, 'output_type_is_set', output_index)
            output_type = (query(layer, 'get_output_type', output_index) if is_true(output_set) else
                           {'queried': False, 'reason': 'output_type_is_set is false or unavailable'})
            outputs.append({'index': output_index, 'tensor': tensor_info(layer.get_output(output_index)),
                            'output_type_is_set': output_set, 'get_output_type': output_type,
                            'get_output_type_is_a_set_constraint': is_true(output_set)})
        rows.append({'index': index, 'name': layer.name, 'layer_type_at_parse': initial_types[layer.name],
                     'precision_is_set': precision_set, 'precision_if_set': precision,
                     'inputs': [tensor_info(layer.get_input(i)) for i in range(layer.num_inputs)],
                     'outputs': outputs})
    precision_layers = [row['name'] for row in rows if is_true(row['precision_is_set'])]
    output_constraints = [{'name': row['name'], 'output_index': output['index'],
                           'get_output_type': output['get_output_type']}
                          for row in rows for output in row['outputs'] if is_true(output['output_type_is_set'])]
    errors = []
    for row in rows:
        for field in ('precision_is_set', 'precision_if_set'):
            data = row[field]
            if data.get('queried') is False:
                continue
            if data.get('ok') is not True:
                errors.append({'name': row['name'], 'field': field, 'query': data})
        for output in row['outputs']:
            for field in ('output_type_is_set', 'get_output_type'):
                if output[field].get('queried') is False:
                    continue
                if output[field].get('ok') is not True:
                    errors.append({'name': row['name'], 'output_index': output['index'], 'field': field,
                                   'query': output[field]})
    return {'layer_count': len(rows), 'layers_with_precision_set_count': len(precision_layers),
            'layers_with_precision_set': precision_layers, 'outputs_with_type_set_count': len(output_constraints),
            'outputs_with_type_set': output_constraints, 'metadata_query_errors': errors,
            'all_precision_and_output_constraint_queries_succeeded': not errors, 'layers': rows,
            'output_type_reading_note': 'get_output_type is queried only when output_type_is_set=true; unset values are not meaningful constraints.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspection-contract', type=Path,
                        default=Path(__file__).resolve().parent / 'parser_metadata_contract.json')
    parser.add_argument('--check-input-only', action='store_true')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    details = output
    assert root.is_dir() and output != root and root not in output.parents, 'Output must be outside the frozen root'
    assert not output.exists() and output.parent.is_dir(), 'Output must be fresh and its parent must exist'
    spec, source_hashes = validate_inputs(root)
    inspection_contract_path = args.inspection_contract.resolve()
    assert sha(inspection_contract_path) == EXPECTED_INSPECTION_CONTRACT_SHA256
    inspection_contract = read(inspection_contract_path)
    assert inspection_contract['schema'] == 'rampq.gate05.corrected_qall_parser_metadata_contract/v1'
    assert inspection_contract['attempts'] == 1 and inspection_contract['max_seconds'] == 120
    assert inspection_contract['source_model_sha256'] == EXPECTED_GRAPH_SHA256
    assert inspection_contract['source_spec_sha256'] == EXPECTED_SPEC_SHA256
    assert inspection_contract['no_config_flag_or_layer_precision_setter_allowed'] is True
    assert inspection_contract['not_a_full_model_build_retry'] is True
    assert inspection_contract['original_pytorch_ort_guard'] == 'FAILED_UNCHANGED'
    assert inspection_contract['gate05_status'] == 'INCOMPLETE'
    common = {'schema': 'rampq.gate05.corrected_qall_parser_metadata_only/v1', 'frozen_root': str(root),
              'separate_output_directory': str(output), 'tool_file': Path(__file__).name,
              'tool_sha256': sha(Path(__file__)), 'python': platform.python_version(),
              'machine': platform.machine(), 'client_or_board_utc': datetime.now(timezone.utc).isoformat(),
              'source_graph_sha256': EXPECTED_GRAPH_SHA256, 'source_spec_sha256': EXPECTED_SPEC_SHA256,
              'source_manifest_sha256': EXPECTED_MANIFEST_SHA256, 'source_contract_sha256': EXPECTED_CONTRACT_SHA256,
              'inspection_contract_sha256': EXPECTED_INSPECTION_CONTRACT_SHA256,
              'engine_build_api_called': False, 'builder_configuration_created': False, 'inference_run': False,
              'dataset_evaluation_run': False, 'precision_or_output_type_setter_calls': 0,
              'builder_config_flags_set': [], 'frozen_source_modified': False,
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE'}
    if args.check_input_only:
        print(json.dumps(dict(common, status='PARSER_INSPECTION_INPUTS_VERIFIED_NO_TENSORRT_IMPORT'), indent=2))
        return
    assert platform.system() == 'Linux' and platform.machine() == 'aarch64'
    assert str(root) == inspection_contract['source_native_root'], 'Unexpected frozen native source root'
    details.mkdir(exist_ok=False)
    write_new(details / 'request.json', dict(common, status='PARSE_METADATA_PENDING', source_manifest_entries=source_hashes))
    started = time.monotonic()
    try:
        import tensorrt as trt
        assert trt.__version__.startswith('8.5.'), trt.__version__
        common.update(tensorrt=trt.__version__, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        importer = trt.OnnxParser(network, logger)
        parse_ok = importer.parse((root / 'model.onnx').read_bytes())
        parse_report = {'parse_ok': bool(parse_ok), 'parser_error_count': importer.num_errors,
                        'parser_errors': [str(importer.get_error(i)) for i in range(importer.num_errors)]}
        write_new(details / 'parser_result.json', parse_report)
        assert parse_ok, parse_report['parser_errors']
        layers = [network.get_layer(i) for i in range(network.num_layers)]
        by_name = {layer.name: layer for layer in layers}
        assert len(by_name) == len(layers), 'Parsed names not unique'
        initial_types = {layer.name: str(layer.type) for layer in layers}
        before = snapshot(layers, initial_types)
        write_new(details / 'constraints_before_bounds_correction.json', before)
        bounds_before, bounds_after = [], []
        for name in spec['clip_names']:
            clip = by_name[name]
            assert clip.type == trt.LayerType.ACTIVATION
            clip.__class__ = trt.IActivationLayer
            assert clip.type == trt.ActivationType.CLIP
            bounds_before.append({'name': name, 'alpha': query(clip, 'alpha'), 'beta': query(clip, 'beta')})
        write_new(details / 'clip_bounds_before.json', {'count': len(bounds_before), 'layers': bounds_before})
        # The only setters in this script affect temporary activation bounds.
        for name in spec['clip_names']:
            clip = by_name[name]
            clip.__class__ = trt.IActivationLayer
            clip.alpha, clip.beta = -127., 127.
            assert clip.alpha == -127. and clip.beta == 127.
            bounds_after.append({'name': name, 'alpha': clip.alpha, 'beta': clip.beta})
        assert len(bounds_after) == 88
        write_new(details / 'clip_bounds_after.json', {'count': 88, 'bound_attribute_setter_calls': 176,
                  'precision_or_output_type_setter_calls': 0, 'layers': bounds_after})
        after = snapshot(layers, initial_types)
        write_new(details / 'constraints_after_bounds_correction.json', after)
        current_spec, after_hashes = validate_inputs(root)
        assert source_hashes == after_hashes and spec == current_spec
        write_new(details / 'source_hashes_after.json', after_hashes)
        before_constraint_values = [(row['name'], row['precision_is_set'], row['precision_if_set'],
                                     [(out['index'], out['output_type_is_set'], out['get_output_type']) for out in row['outputs']])
                                    for row in before['layers']]
        after_constraint_values = [(row['name'], row['precision_is_set'], row['precision_if_set'],
                                    [(out['index'], out['output_type_is_set'], out['get_output_type']) for out in row['outputs']])
                                   for row in after['layers']]
        good_queries = before['all_precision_and_output_constraint_queries_succeeded'] and after['all_precision_and_output_constraint_queries_succeeded']
        report = dict(common, status='PARSER_METADATA_OBSERVED' if good_queries else 'PARSER_METADATA_QUERY_REVIEW_REQUIRED',
                      elapsed_seconds=time.monotonic() - started, parsed_layer_count=len(layers), clip_correction_count=88,
                      temporary_bound_attribute_setter_calls=176, constraint_metadata_unchanged_by_bounds_correction=before_constraint_values == after_constraint_values,
                      before={'layers_with_precision_set_count': before['layers_with_precision_set_count'],
                              'outputs_with_type_set_count': before['outputs_with_type_set_count'],
                              'query_error_count': len(before['metadata_query_errors'])},
                      after={'layers_with_precision_set_count': after['layers_with_precision_set_count'],
                             'outputs_with_type_set_count': after['outputs_with_type_set_count'],
                             'query_error_count': len(after['metadata_query_errors'])},
                      limits='Parser metadata only; no engine, tactic, numerical, accuracy, performance or causal pass. A zero setter count describes this script; observed parser constraint states are reported separately.')
    except BaseException as error:
        report = dict(common, status='PARSER_METADATA_INSPECTION_FAILED', elapsed_seconds=time.monotonic() - started,
                      error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        write_new(output / 'inspection_result.json', report)
        raise
    write_new(output / 'inspection_result.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
