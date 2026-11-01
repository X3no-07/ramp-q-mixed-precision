"""Actual single-image TensorRT execution, compared with board-local PyTorch FP32."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--smoke-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_dir = args.output / '.ultralytics'
    config_dir.mkdir()
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    manifest = json.loads((args.smoke_root / 'manifest.json').read_text())
    for name in ('best.pt', '000020.jpg'):
        assert sha(args.smoke_root / name) == manifest['files'][name]['sha256'], name
    build = json.loads(args.engine.with_name('build_result.json').read_text())
    assert build['status'] == 'BUILT_AND_DESERIALIZED'
    assert sha(args.engine) == build['engine_sha256']
    print('SMOKE_FILES_VERIFIED', flush=True)

    import cv2
    import numpy as np
    import torch
    import tensorrt as trt
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
    from torchvision.ops import box_iou

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    image = cv2.imread(str(args.smoke_root / '000020.jpg'))
    assert image is not None
    padded = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=image)
    array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    x = torch.from_numpy(array).to('cuda')
    torch.cuda.synchronize()
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    assert engine is not None
    assert engine.num_bindings == 2
    context = engine.create_execution_context()
    assert context is not None
    buffers = []
    descriptions = []
    input_count = 0
    output_index = None
    for i in range(engine.num_bindings):
        shape = tuple(engine.get_binding_shape(i))
        dtype = engine.get_binding_dtype(i)
        assert engine.get_binding_format(i) == trt.TensorFormat.LINEAR
        assert dtype in (trt.float32, trt.float16), str(dtype)
        torch_dtype = torch.float32 if dtype == trt.float32 else torch.float16
        if engine.binding_is_input(i):
            input_count += 1
            assert shape == (1, 3, 640, 640)
            buffer = x.to(dtype=torch_dtype).contiguous()
        else:
            assert shape == (1, 14, 8400)
            output_index = i
            buffer = torch.empty(shape, device='cuda', dtype=torch_dtype)
        buffers.append(buffer)
        descriptions.append({'name': engine.get_binding_name(i), 'shape': shape,
                             'dtype': str(dtype), 'input': engine.binding_is_input(i)})
    assert input_count == 1 and output_index is not None
    torch.cuda.synchronize()
    start = time.perf_counter()
    success = context.execute_v2([int(buffer.data_ptr()) for buffer in buffers])
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert success, 'TensorRT execute_v2 returned false'
    actual = buffers[output_index].float().clone()
    assert bool(torch.isfinite(actual).all()), 'Non-finite TensorRT output'
    print('TENSORRT_EXECUTION_OK', flush=True)

    reference = YOLO(str(args.smoke_root / 'best.pt'), task='detect').model.float().eval().to('cuda')
    reference.fuse()
    with torch.inference_mode():
        expected = reference(x)
        expected = expected[0] if isinstance(expected, (tuple, list)) else expected
    torch.cuda.synchronize()
    assert expected.shape == actual.shape
    assert bool(torch.isfinite(expected).all())
    settings = {'conf_thres': 0.25, 'iou_thres': 0.7, 'max_det': 1000, 'nc': 10}
    trt_boxes = non_max_suppression(actual.clone(), **settings)[0].detach().cpu()
    pt_boxes = non_max_suppression(expected.clone(), **settings)[0].detach().cpu()
    for boxes in (trt_boxes, pt_boxes):
        ops.scale_boxes((640, 640), boxes[:, :4], image.shape[:2])
    # Predeclared single-input pipeline guard, not a dataset accuracy criterion.
    guard = {'same_detection_count': len(trt_boxes) == len(pt_boxes),
             'minimum_class_matched_iou': 0.90, 'maximum_confidence_difference': 0.05}
    matched = []
    available = set(range(len(pt_boxes)))
    for row in trt_boxes:
        candidates = [j for j in sorted(available) if int(pt_boxes[j, 5]) == int(row[5])]
        if not candidates:
            continue
        overlaps = box_iou(row[:4].reshape(1, 4), pt_boxes[candidates, :4])[0]
        selected = int(torch.argmax(overlaps))
        j = candidates[selected]
        overlap = float(overlaps[selected])
        difference = abs(float(row[4]) - float(pt_boxes[j, 4]))
        matched.append({'class_id': int(row[5]), 'iou': overlap, 'confidence_abs_difference': difference})
        available.remove(j)
    parity = (len(pt_boxes) > 0 and guard['same_detection_count'] and len(matched) == len(pt_boxes)
              and all(m['iou'] >= 0.90 and m['confidence_abs_difference'] <= 0.05 for m in matched))
    delta = (actual - expected).abs()
    np.savez_compressed(str(args.output / 'raw_outputs.npz'),
                        tensorrt=actual.detach().cpu().numpy(), pytorch=expected.detach().cpu().numpy())
    report = {
        'scope': 'One fixed Selection image on Jetson; execution/parity smoke only, no dataset or performance claim',
        'status': 'PASS' if parity else 'PARITY_GUARD_FAILED', 'execution_success': bool(success),
        'engine_sha256': sha(args.engine), 'weights_sha256': sha(args.smoke_root / 'best.pt'),
        'image_sha256': sha(args.smoke_root / '000020.jpg'), 'image': '000020.jpg',
        'torch': torch.__version__, 'tensorrt': trt.__version__, 'ultralytics': ultralytics.__version__,
        'gpu': torch.cuda.get_device_name(0), 'bindings': descriptions,
        'nms': settings, 'parity_guard': guard, 'matched_detections': matched,
        'tensorrt_detections': trt_boxes.tolist(), 'pytorch_fp32_detections': pt_boxes.tolist(),
        'max_abs_box_output_difference': float(delta[:, :4].max()),
        'max_abs_score_output_difference': float(delta[:, 4:].max()),
        'first_call_ms_not_benchmark': elapsed_ms,
        'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    }
    (args.output / 'smoke_result.json').write_text(json.dumps(report, indent=2) + '\n')
    annotated = image.copy()
    for box in trt_boxes.tolist():
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        text = '{} {:.3f}'.format(reference.names[int(box[5])], box[4])
        cv2.putText(annotated, text, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    assert cv2.imwrite(str(args.output / 'prediction.jpg'), annotated)
    print(json.dumps(report, indent=2), flush=True)
    if not parity:
        raise RuntimeError('Single-image FP16 parity guard failed; inspect saved outputs')
    print('TENSORRT_FP16_SINGLE_IMAGE_OK', flush=True)


if __name__ == '__main__':
    main()
