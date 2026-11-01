"""Map exported Conv nodes through weight identities, not exporter scope names."""
import hashlib
import json
from pathlib import Path

import onnx


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'artifacts/jetson_trt_fp16_20261016_v1/model.onnx'
    inventory_path = root / 'artifacts/manifests/module_inventory.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8-sig'))
    by_name = {row['name']: row for row in inventory['modules']}
    graph = onnx.load(str(source)).graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    rows = []
    for node in graph.node:
        if node.op_type != 'Conv':
            continue
        weight_name = node.input[1]
        if not weight_name.endswith('.weight'):
            raise RuntimeError('Unexpected weight identity: ' + weight_name)
        module = weight_name[:-len('.weight')]
        entry = by_name[module]
        assert list(initializers[weight_name].dims) == entry['weight_shape'], module
        rows.append({'onnx_node': node.name, 'pytorch_module': module,
                     'group': entry['group'], 'macs': entry['macs'], 'weight_input': weight_name})
    assert len(rows) == inventory['conv_count'] == 88
    assert {row['pytorch_module'] for row in rows} == set(by_name)
    counts = {group: sum(row['group'] == group for row in rows) for group in inventory['groups']}
    assert all(counts[group] == inventory['groups'][group]['conv_count'] for group in counts)
    report = {'schema': 'rampq.gate05.onnx_module_map/v1',
              'onnx_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
              'mapping_method': 'Exact Conv weight initializer identity and shape',
              'mapped_conv_count': len(rows), 'groups': counts, 'mapping': rows,
              'boundary': 'Pre-fusion ONNX mapping only; TensorRT layer precision must be inspected separately.'}
    destination = root / 'artifacts/gate05/onnx_module_map.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'ONNX_GROUP_MAP_OK', 'count': len(rows), 'groups': counts}))


if __name__ == '__main__':
    main()
