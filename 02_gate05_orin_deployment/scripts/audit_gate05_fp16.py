"""Audit copied Jetson FP16 evidence; does not build or run an engine."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/gate05/20261018_fp16'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def main():
    receipt = read(RUN / 'remote_receipt.json')
    for relative, expected in receipt['files'].items():
        path = RUN / relative
        assert path.stat().st_size == expected['bytes'], relative + ': truncated transfer'
        assert sha(path) == expected['sha256'], relative + ': checksum mismatch'
    build = read(RUN / 'engine_build/build_result.json')
    smoke = read(RUN / 'smoke/smoke_result.json')
    status = read(RUN / 'monitor/status.json')
    group_map = read(ROOT / 'artifacts/gate05/onnx_module_map.json')
    assert build['status'] == 'BUILT_AND_DESERIALIZED'
    assert status['status'] == 'BUILD_SCRIPT_OK' and status['build_returncode'] == 0
    assert sha(RUN / 'engine_build/model_fp16.engine') == build['engine_sha256'] == smoke['engine_sha256']
    assert build['onnx_sha256'] == group_map['onnx_sha256']
    assert sha(ROOT / 'artifacts/jetson_trt_fp16_20261016_v1/model.onnx') == build['onnx_sha256']
    assert smoke['boot_id'] == status['boot_id'] == receipt['boot_id']
    assert smoke['status'] == 'PASS' and smoke['execution_success']
    layers = read(RUN / 'engine_build/engine_layers.json')['Layers']
    assert len(layers) == build['engine_layers']
    convs = [layer for layer in layers if 'Convolution' in layer['LayerType']]
    mapped = []
    for item in group_map['mapping']:
        pattern = re.escape(item['onnx_node']) + r'(?=$|[\s,+|)])'
        matches = [layer for layer in convs if re.search(pattern, layer['Name'])]
        assert len(matches) == 1, item['onnx_node']
        layer = matches[0]
        assert layer['Weights']['Type'] == 'Half', layer['Name']
        assert all('FP16' in tensor['Format/Datatype'] for tensor in layer['Inputs'] + layer['Outputs'])
        mapped.append(dict(item, engine_layer=layer['Name'], engine_layer_type=layer['LayerType'],
                           weights_type=layer['Weights']['Type'],
                           input_formats=[t['Format/Datatype'] for t in layer['Inputs']],
                           output_formats=[t['Format/Datatype'] for t in layer['Outputs']]))
    assert len(mapped) == 88
    assert {row['engine_layer'] for row in mapped} == {layer['Name'] for layer in convs}
    fusion = []
    for layer in convs:
        members = [row for row in mapped if row['engine_layer'] == layer['Name']]
        assert len({row['group'] for row in members}) == 1, 'Cross-group fusion needs manual review'
        if len(members) > 1:
            fusion.append({'engine_layer': layer['Name'], 'modules': [m['pytorch_module'] for m in members],
                           'group': members[0]['group']})
    fp32_io_layers = [layer['Name'] for layer in layers
                     if any('FP32' in t['Format/Datatype'] for t in layer.get('Inputs', []) + layer.get('Outputs', []))]
    precision = {
        'status': 'FP16_CONV_MAPPING_VERIFIED', 'engine_layers': len(layers),
        'source_convs_mapped': len(mapped), 'engine_conv_layers': len(convs),
        'group_conv_counts': group_map['groups'], 'same_group_fusions': fusion,
        'layers_with_fp32_tensor_io': fp32_io_layers, 'mapping': mapped,
        'boundary': 'Conv weights and tensor formats are FP16; accumulation may use FP32. '
                    'Input/output and some decoding operations are FP32. This is not an all-FP16 network.',
    }

    source = ROOT / 'artifacts/jetson_fp16_smoke_20261016_v1'
    assert sha(source / '000020.jpg') == smoke['image_sha256']
    assert sha(source / 'best.pt') == smoke['weights_sha256']
    width, height = Image.open(source / '000020.jpg').size
    gain = min(640 / width, 640 / height)
    pad_x = round((640 - round(width * gain)) / 2 - 0.1)
    pad_y = round((640 - round(height * gain)) / 2 - 0.1)
    raw = np.load(RUN / 'smoke/raw_outputs.npz', allow_pickle=False)
    coordinates = []
    for name, key in [('tensorrt', 'tensorrt_detections'), ('pytorch', 'pytorch_fp32_detections')]:
        values = raw[name][0]
        assert values.shape == (14, 8400) and np.isfinite(values).all()
        for box in smoke[key]:
            anchors = np.flatnonzero(np.abs(values[4 + int(box[5])] - box[4]) < 1e-7)
            assert len(anchors), 'Saved confidence missing from raw output'
            xywh = values[:4, anchors].T.astype(np.float64)
            xyxy = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2,
                                   xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
            xyxy -= np.array([pad_x, pad_y, pad_x, pad_y])
            xyxy /= gain
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
            errors = np.abs(xyxy - np.asarray(box[:4])).max(axis=1)
            best = int(errors.argmin())
            assert float(errors[best]) < 1e-3, 'Coordinate restoration mismatch'
            coordinates.append({'source': name, 'class_id': int(box[5]),
                                'raw_anchor': int(anchors[best]), 'max_abs_error_pixels': float(errors[best])})
    parity = []
    reference = smoke['pytorch_fp32_detections']
    actual = smoke['tensorrt_detections']
    assert len(reference) == len(actual) > 0
    available = set(range(len(reference)))
    for box in actual:
        candidates = [j for j in available if reference[j][5] == box[5]]
        assert candidates, 'Unmatched detection class'
        j = max(candidates, key=lambda k: iou(box[:4], reference[k][:4]))
        overlap = iou(box[:4], reference[j][:4])
        difference = abs(box[4] - reference[j][4])
        assert overlap >= 0.9 and difference <= 0.05
        parity.append({'class_id': int(box[5]), 'iou': overlap, 'confidence_abs_difference': difference})
        available.remove(j)
    annotation = []
    for line in (source / '000020.txt').read_text().splitlines():
        cls, cx, cy, w, h = [float(v) for v in line.split()]
        gt = [(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height]
        candidates = [b for b in actual if b[5] == cls]
        annotation.append({'class_id': int(cls), 'xyxy': gt,
                           'best_tensorrt_same_class_iou': max([iou(gt, b[:4]) for b in candidates], default=0)})
    resources = [json.loads(line) for line in (RUN / 'monitor/resources.jsonl').read_text().splitlines() if line.strip()]
    summary = {
        'schema': 'rampq.gate05.fp16_audit/v1', 'status': 'FP16_BUILD_AND_SINGLE_IMAGE_AUDIT_PASS',
        'client_date': '2026-10-18', 'boot_id': receipt['boot_id'],
        'verified_remote_file_count': len(receipt['files']), 'engine_sha256': build['engine_sha256'],
        'engine_bytes': build['engine_bytes'], 'build_seconds': status['elapsed_seconds'],
        'resource_scope': 'Build only; 2-second samples, not a deployment power/performance benchmark',
        'resource_samples': len(resources),
        'max_observed_temperature_c': max(max(r['temperatures_c'].values()) for r in resources),
        'min_observed_available_memory_mib': min(r['MemAvailable_kib'] for r in resources) / 1024,
        'max_observed_swap_used_mib': max(r['SwapTotal_kib']-r['SwapFree_kib'] for r in resources) / 1024,
        'kernel_monitor_log': (RUN / 'monitor/kernel.log').read_text(),
        'precision': {k: v for k, v in precision.items() if k != 'mapping'},
        'detection_parity': parity, 'coordinate_restore_audit': coordinates,
        'image_size': [width, height], 'annotations': annotation,
        'annotation_boundary': 'Single image has one annotated holothurian and two detections. '
                               'Agreement with PyTorch is not evidence that both detections are correct.',
        'dataset_evaluation_run': False, 'benchmark_run': False,
        'full_gate05_status': 'INCOMPLETE: INT8 and mixed-precision engines and fixed-data evaluation pending',
        'clock_note': 'Board clock remains 2026-10-16; use client date and boot ID for this evidence.',
    }
    for filename, data in [('precision_audit.json', precision), ('audit_summary.json', summary)]:
        (RUN / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('status', 'verified_remote_file_count', 'build_seconds',
                     'max_observed_temperature_c', 'min_observed_available_memory_mib', 'full_gate05_status')}, indent=2))


if __name__ == '__main__':
    main()
