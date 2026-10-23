"""Inspect actual INT8 Conv coverage. Non-INT8 or unmapped layers fail closed."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    layers = json.loads(args.layers.read_text(encoding='utf-8-sig'))['Layers']
    mapping = json.loads(args.map.read_text(encoding='utf-8-sig'))['mapping']
    rows, errors = [], []
    for item in mapping:
        pattern = re.escape(item['onnx_node']) + r'(?=$|[\s,+|)])'
        # Q/DQ names include the module identity too: only match actual convolutions.
        matches = [layer for layer in layers if 'Convolution' in layer.get('LayerType', '')
                   and re.search(pattern, layer['Name'])]
        row = {'module': item['pytorch_module'], 'group': item['group'], 'onnx_node': item['onnx_node'],
               'matching_engine_layers': len(matches), 'verified_int8_kernel': False}
        if len(matches) != 1:
            errors.append('Expected one Conv match for ' + item['pytorch_module'])
        else:
            layer = matches[0]
            weight_type = layer.get('Weights', {}).get('Type', '')
            inputs = [t.get('Format/Datatype', '') for t in layer.get('Inputs', [])]
            outputs = [t.get('Format/Datatype', '') for t in layer.get('Outputs', [])]
            tactic = layer.get('TacticName', '')
            # Require quantized weights, quantized activation input, and an integer tactic.
            # Float output is valid for INT8-input weighted operations.
            weights_int8 = weight_type.lower() in ('int8', 'int8_t', 'signed int8', 'signed byte')
            activation_int8 = bool(inputs) and 'INT8' in inputs[0].upper()
            integer_tactic = bool(re.search(r'i8|s8|int8|imma', tactic, re.IGNORECASE))
            verified = weights_int8 and activation_int8 and integer_tactic
            row.update(engine_layer=layer['Name'], engine_layer_type=layer['LayerType'],
                       weights_type=weight_type, input_formats=inputs, output_formats=outputs,
                       tactic=tactic, verified_int8_kernel=verified,
                       checks={'weight_int8': weights_int8, 'activation_input_int8': activation_int8,
                               'integer_tactic': integer_tactic})
            if not verified:
                errors.append('INT8 kernel evidence incomplete for ' + item['pytorch_module'])
        rows.append(row)
    assert len(rows) == 88, 'Unexpected Conv inventory'
    mapped_names = {row['engine_layer'] for row in rows if 'engine_layer' in row}
    unmapped_convs = [layer['Name'] for layer in layers if 'Convolution' in layer.get('LayerType', '')
                     and layer['Name'] not in mapped_names]
    errors.extend('Unmapped engine convolution: ' + name for name in unmapped_convs)
    fusions = []
    for name in sorted(mapped_names):
        members = [row for row in rows if row.get('engine_layer') == name]
        if len(members) > 1:
            fusions.append({'engine_layer': name, 'modules': [row['module'] for row in members],
                            'groups': sorted({row['group'] for row in members})})
    group_counts = {}
    for row in rows:
        group = group_counts.setdefault(row['group'], {'total_convs': 0, 'verified_int8_convs': 0})
        group['total_convs'] += 1
        group['verified_int8_convs'] += int(row['verified_int8_kernel'])
    report = {'schema': 'rampq.gate05.int8_precision_audit/v1',
              'status': 'ALL_88_INT8_CONVS_VERIFIED' if not errors else 'PRECISION_REVIEW_REQUIRED',
              'layer_file_sha256': hashlib.sha256(args.layers.read_bytes()).hexdigest(),
              'engine_layers': len(layers), 'source_convs': len(rows),
              'verified_int8_convs': sum(row['verified_int8_kernel'] for row in rows),
              'groups': group_counts, 'errors': errors, 'mapping': rows,
              'unmapped_engine_convs': unmapped_convs, 'fusions': fusions,
              'scope': 'Inspector evidence of INT8 Conv kernels only, not full numerical migration or dataset accuracy'}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'mapping'}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
