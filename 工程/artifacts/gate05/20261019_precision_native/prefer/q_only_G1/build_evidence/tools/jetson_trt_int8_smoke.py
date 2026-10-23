"""Execute the diagnostic INT8 engine on one fixed image; no accuracy pass gate."""
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


def binding_contract(bindings):
    """Require exact named static I/O, irrespective of TensorRT binding order."""
    names = [row['name'] for row in bindings]
    assert len(names) == len(set(names)), 'Duplicate binding name'
    assert sum(row['input'] is True for row in bindings) == 1, 'Expected one image input'
    for row in bindings:
        assert isinstance(row['input'], bool)
        assert row['shape'] and all(isinstance(size, int) and size > 0 for size in row['shape'])
        assert row['dtype'] in ('DataType.FLOAT', 'DataType.HALF')
    return sorted((row['name'], row['input'], tuple(row['shape']), row['dtype']) for row in bindings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--fp16-smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-formats-only', action='store_true',
                        help='Allow fixed-image diagnostic when all 88 INT8 input/weight formats are mapped '
                             'but integer tactic evidence remains incomplete; does not pass precision review')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(args.output / '.ultralytics')
    build = json.loads((args.build / 'build_result.json').read_text())
    assert build['status'] == 'BUILT_AND_DESERIALIZED' and build['configuration'] == 'q_all'
    engine_path = args.build / 'model_q_all.engine'
    assert sha(engine_path) == build['engine_sha256']
    precision = json.loads((args.build / 'precision_audit.json').read_text())
    if args.diagnostic_formats_only:
        assert precision['all_88_int8_formats_verified'] is True
        assert precision['format_status'] == 'ALL_88_CONVS_WITH_INT8_INPUT_AND_WEIGHT_FORMATS'
        assert precision['source_convs'] == precision['convs_with_int8_input_and_weight_formats'] == 88
        assert not precision['unmapped_engine_convs']
        assert build['diagnostic_only'] is True
        assert build['local_preflight_status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
        bundle = args.build.parent.parent
        scope = json.loads((bundle / 'diagnostic_scope.json').read_text())
        assert scope['status'] == 'DIAGNOSTIC_BUILD_ONLY_FULL_PARITY_FAILED'
        assert scope['original_tolerances_changed'] is False and scope['dataset_evaluation_run'] is False
        assert scope['failed_preflight_sha256'] == sha(bundle / 'preflight.json')
        assert json.loads((bundle / 'preflight.json').read_text())['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    else:
        assert precision['status'] == 'ALL_88_INT8_CONVS_VERIFIED'
    assert precision['layer_file_sha256'] == sha(args.build / 'engine_layers.json')
    source_manifest = json.loads((args.source / 'manifest.json').read_text())
    image_path = args.source / '000020.jpg'
    assert sha(image_path) == source_manifest['files']['000020.jpg']['sha256']
    fp16 = json.loads((args.fp16_smoke / 'smoke_result.json').read_text())
    assert fp16['image_sha256'] == sha(image_path) and fp16['status'] == 'PASS'
    assert fp16['engine_sha256'] == '64d91db3063ac28c1552a1ba6443ae28ca45d82d988f5cd375b39fc8cc76b85c'
    assert sha(args.fp16_smoke / 'raw_outputs.npz') == 'f38d9b113c148198735dfd0a2577c17b93c8cd46ac1f97002a6c0e0a756cf45b'
    print('INT8_SMOKE_INPUTS_VERIFIED', flush=True)
    import cv2
    import numpy as np
    import torch
    import tensorrt as trt
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
    assert torch.cuda.is_available()
    image = cv2.imread(str(image_path))
    assert image is not None
    padded = LetterBox(new_shape=(640,640),auto=False,stride=32)(image=image)
    array = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None], dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).to('cuda')
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    buffers = []
    descriptions = []
    output_index = None
    expected = build.get('expected_binding_contract')
    assert expected, 'Build must freeze the exact named I/O contract'
    expected_primary = build['primary_output_name']
    expected_primary_row = next(row for row in expected if row['name'] == expected_primary)
    assert not expected_primary_row['input'] and expected_primary_row['shape'] == [1,14,8400]
    for i in range(engine.num_bindings):
        shape = tuple(engine.get_binding_shape(i))
        dtype = engine.get_binding_dtype(i)
        assert engine.get_binding_format(i) == trt.TensorFormat.LINEAR
        assert dtype in (trt.float32, trt.float16)
        torch_dtype = torch.float32 if dtype == trt.float32 else torch.float16
        if engine.binding_is_input(i):
            assert shape == (1,3,640,640)
            buffer = tensor.to(dtype=torch_dtype).contiguous()
        else:
            if engine.get_binding_name(i) == expected_primary:
                assert shape == (1,14,8400)
                output_index = i
            buffer = torch.empty(shape, dtype=torch_dtype, device='cuda')
        buffers.append(buffer)
        descriptions.append({'name':engine.get_binding_name(i),'shape':list(shape),'dtype':str(dtype),
                             'input':bool(engine.binding_is_input(i))})
    assert binding_contract(descriptions) == binding_contract(expected), 'Engine binding contract changed'
    assert output_index is not None
    torch.cuda.synchronize()
    started = time.perf_counter()
    success = context.execute_v2([int(buffer.data_ptr()) for buffer in buffers])
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter()-started)*1000
    assert success
    finite_outputs = {row['name']:bool(torch.isfinite(buffers[i]).all())
                      for i,row in enumerate(descriptions) if not row['input']}
    assert finite_outputs and all(finite_outputs.values()), 'Nonfinite primary or auxiliary output'
    actual = buffers[output_index].float().clone()
    assert bool(torch.isfinite(actual).all())
    settings = {'conf_thres':0.25,'iou_thres':0.7,'max_det':1000,'nc':10}
    boxes = non_max_suppression(actual.clone(),**settings)[0].detach().cpu()
    ops.scale_boxes((640,640),boxes[:,:4],image.shape[:2])
    raw = actual.detach().cpu().numpy()
    reference = np.load(str(args.fp16_smoke/'raw_outputs.npz'),allow_pickle=False)['tensorrt']
    assert raw.shape == reference.shape == (1,14,8400)
    saved = {'tensorrt_int8':raw,'tensorrt_fp16':reference}
    auxiliary_output_keys = {}
    for i,row in enumerate(descriptions):
        if not row['input'] and i != output_index:
            key = 'auxiliary_output_{}'.format(len(auxiliary_output_keys))
            auxiliary_output_keys[row['name']] = key
            saved[key] = buffers[i].float().detach().cpu().numpy()
    np.savez_compressed(str(args.output/'raw_outputs.npz'),**saved)
    graph_structure_confounded = bool(auxiliary_output_keys)
    execution_status = ('QDQ_DIAGNOSTIC_SINGLE_IMAGE_EXECUTION_OK' if args.diagnostic_formats_only else
                        'INT8_SINGLE_IMAGE_EXECUTION_OK')
    report = {'schema':'rampq.gate05.int8_smoke/v2','status':execution_status,
              'execution_success':True,'finite_output':True,'configuration':'q_all',
              'diagnostic_formats_only_opt_in':args.diagnostic_formats_only,
              'precision_audit_status':precision['status'],
              'precision_format_status':precision.get('format_status'),
              'all_outputs_finite':finite_outputs,'auxiliary_output_npz_keys':auxiliary_output_keys,
              'engine_sha256':sha(engine_path),'image_sha256':sha(image_path),
              'fp16_reference_engine_sha256':fp16['engine_sha256'],'bindings':descriptions,
              'expected_binding_contract':expected,'primary_output_name':expected_primary,
              'comparison_to_old_fp16_descriptive_only':True,
              'comparison_graph_structure_confounded':graph_structure_confounded,
              'comparison_limits':'Old FP16 uses a different output structure when auxiliary outputs are present. '
                                  'Fusion/floating precision changes prevent attributing differences solely to quantization.',
              'nms':settings,'int8_detections':boxes.tolist(),'fp16_detections':fp16['tensorrt_detections'],
              'max_abs_box_output_difference_to_fp16':float(np.abs(raw[:,:4]-reference[:,:4]).max()),
              'max_abs_score_output_difference_to_fp16':float(np.abs(raw[:,4:]-reference[:,4:]).max()),
              'first_call_ms_not_benchmark':elapsed_ms,'tensorrt':trt.__version__,'torch':torch.__version__,
              'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'board_time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'numerical_migration_status':'LOCAL_FULL_MODEL_QDQ_PARITY_FAILED_RETAINS_FAILURE',
              'scope':'Fixed Selection image execution only; no dataset accuracy, target rescue, or performance claim'}
    (args.output/'smoke_result.json').write_text(json.dumps(report,indent=2)+'\n')
    names = ['holothurian','echinus','scallop','starfish','fish','corals','diver','cuttlefish','turtle','jellyfish']
    annotated=image.copy()
    for row in boxes.tolist():
        x1,y1,x2,y2=[int(round(v)) for v in row[:4]]
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,180,255),2)
        cv2.putText(annotated,'{} {:.3f}'.format(names[int(row[5])],row[4]),(x1,max(18,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,180,255),2)
    assert cv2.imwrite(str(args.output/'prediction.jpg'),annotated)
    print(json.dumps(report,indent=2),flush=True)
    print(execution_status,flush=True)


if __name__ == '__main__':
    main()
